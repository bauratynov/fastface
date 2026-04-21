// Session 26 — End-to-end INT8 inference driver.
//
// Activations stay int8 between ops; per-tensor calibrated scales loaded from
// models/op_scales.bin. Fused post-Conv epilogue (bias + optional BN + PReLU + requant).
// S13 correctness fixes applied: Conv bias, BLOCK_START/shortcut block_buf, Flatten NHWC→NCHW.

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include "compat.h"
#include <immintrin.h>
#include <io.h>
#include <fcntl.h>

void compute_col_sums(const int8_t* B, int K, int N, int32_t* col_sums);
void pack_B_vnni(const int8_t* B, int K, int N, int8_t* Bp);
void pack_conv_weight_nhwc(const int8_t* weight, int Cout, int Cin, int Kh, int Kw, int K_padded,
                            int8_t* w_rowmajor, int8_t* w_packed, int32_t* col_sums);
void fastface_conv2d_i8_nhwc(const int8_t* input, int Cin, int H_in, int W_in,
                              int Cout, int Kh, int Kw, int stride, int pad,
                              const int8_t* weight_packed, const int32_t* col_sums,
                              int H_out, int W_out, int32_t* output, uint8_t* scratch_Au);

// S26/S36/S38 fused epilogue kernels (optional per-channel scale args)
void fused_epilogue_int8(
    const int32_t* acc, int N_pos, int Cout,
    float in_scale, const float* weight_scales, const float* bias,
    const float* bn_scale, const float* bn_offset, const float* prelu_slope,
    const int8_t* add_src, float add_scale,
    const float* add_scale_per_ch,
    const float* inv_out_per_ch,
    float out_scale, int8_t* out_i8);
void add_requant_int8(const int8_t* a, float a_scale, const int8_t* b, float b_scale,
                       int8_t* out, float out_scale, int n);
void add_requant_int8_pc(const int8_t* a, const float* a_scale_pc,
                          const int8_t* b, const float* b_scale_pc,
                          int8_t* out, const float* out_inv_scale_pc,
                          int N_pos, int C);
void add_bn_requant_int8(const int8_t* a, float a_scale,
                         const int8_t* b, float b_scale,
                         const float* bn_scale, const float* bn_offset,
                         int8_t* out, float out_scale,
                         int N_pos, int C);
void quantize_fp32_nhwc_to_int8(const float* in, int N, float scale, int8_t* out);
void quantize_fp32_nhwc_to_int8_per_channel(
    const float* in, int N_pos, int C, const float* inv_scale_per_ch, int8_t* out);

// S31: vectorized matvec int8 GEMM for final Linear(25088 -> 512)
void fastface_gemm_i8_matvec_vnni(
    const uint8_t* Au, const int8_t* W,
    const int32_t* w_row_sum, const float* W_scale, const float* bias,
    float A_scale, float* out, int N, int K);
void bn_prelu_requant_int8(
    const int8_t* in_i8, float in_scale,
    const float* in_scale_per_ch,
    const float* bn_scale, const float* bn_offset, const float* prelu_slope,
    const float* inv_out_per_ch,
    float out_scale, int8_t* out_i8, int N_pos, int C);

#define OP_CONV 1
#define OP_BN 2
#define OP_PRELU 3
#define OP_ADD 4
#define OP_GEMM 5
#define OP_FLATTEN 6
#define OP_SAVE_ID 7
#define OP_BLOCK_START 8

typedef struct {
    uint8_t type;
    uint16_t Cin, Cout, Kh, Kw, stride, pad;
    const int8_t*  conv_w;
    const float*   conv_scales;
    const float*   conv_bias;
    const float*   bn_scale;
    const float*   bn_offset;
    uint16_t bn_size;
    const float*   prelu_slope;
    uint16_t prelu_size;
    uint32_t N, K;
    const int8_t*  gemm_w;
    const float*   gemm_scales;
    const float*   gemm_bias;
} Op;
typedef struct { void* data; size_t size; uint32_t n_ops; Op* ops; uint8_t version; } FFW2;
int ffw2_load(const char* path, FFW2* out);
typedef struct { int C, H, W; } Shape;

static double now_s(void) {
    LARGE_INTEGER q, f; QueryPerformanceCounter(&q); QueryPerformanceFrequency(&f);
    return (double)q.QuadPart / (double)f.QuadPart;
}

#ifdef PROFILE_OPS
static double op_type_time[16] = {0};
static uint64_t op_type_count[16] = {0};
static const char* op_type_name[16] = {
    "0?", "CONV", "BN", "PRELU", "ADD", "GEMM", "FLATTEN", "SAVE_ID", "BLOCK_START"
};
#define PROF_START(t0)   do { (t0) = now_s(); } while (0)
#define PROF_END(t0, t)  do { op_type_time[(t) & 15] += now_s() - (t0); op_type_count[(t) & 15]++; } while (0)
#else
#define PROF_START(t0)   ((void)0)
#define PROF_END(t0, t)  ((void)0)
#endif

typedef struct {
    float input_scale;
    uint32_t n_ops;
    float* scales;  // [n_ops], 0 means "no quant target / marker op"
    // S36 phase B / S38: optional per-channel scales.
    float**  inv_out_per_ch;   // [n_ops] pointers into inv_out_flat, or NULL per op
    float**  out_scale_per_ch; // [n_ops] pointers into out_scale_flat (forward scales)
    uint32_t* inv_out_len;     // [n_ops]
    float*   inv_out_flat;     // backing storage for inv scales
    float*   out_scale_flat;   // backing storage for forward scales
    float*   input_pc_scale;   // per-channel input scale (nullptr if per-tensor only)
    float*   input_pc_inv_scale;  // pre-inverted
    uint32_t input_pc_len;
} OpScales;

// S36: Try loading OPSC2 per-channel file. If present, populates inv_out_per_ch.
// Returns 0 on success, nonzero if file missing or wrong format (non-fatal).
static int load_op_scales_v2(const char* path, OpScales* out) {
    FILE* f = fopen(path, "rb");
    if (!f) return -1;
    char magic[5];
    if (fread(magic, 1, 5, f) != 5 || memcmp(magic, "OPSC2", 5) != 0) { fclose(f); return -2; }
    uint32_t n_ops = 0;
    fread(&n_ops, sizeof(uint32_t), 1, f);
    if (n_ops != out->n_ops) {
        fprintf(stderr, "OPSC2 n_ops mismatch: %u vs expected %u\n", n_ops, out->n_ops);
        fclose(f); return -3;
    }
    uint32_t n_in = 0;
    fread(&n_in, sizeof(uint32_t), 1, f);
    out->input_pc_len = n_in;
    out->input_pc_scale = (float*)_aligned_malloc(n_in * sizeof(float), 64);
    fread(out->input_pc_scale, sizeof(float), n_in, f);
    out->input_pc_inv_scale = (float*)_aligned_malloc(n_in * sizeof(float), 64);
    for (uint32_t c = 0; c < n_in; c++) {
        out->input_pc_inv_scale[c] = 1.0f / (out->input_pc_scale[c] + 1e-9f);
    }

    out->inv_out_per_ch   = (float**)calloc(n_ops, sizeof(float*));
    out->out_scale_per_ch = (float**)calloc(n_ops, sizeof(float*));
    out->inv_out_len      = (uint32_t*)calloc(n_ops, sizeof(uint32_t));

    // First pass: read lengths + sum
    long payload_start = ftell(f);
    size_t total_floats = 0;
    for (uint32_t i = 0; i < n_ops; i++) {
        uint32_t nch = 0; fread(&nch, sizeof(uint32_t), 1, f);
        out->inv_out_len[i] = nch;
        total_floats += nch;
        fseek(f, nch * sizeof(float), SEEK_CUR);
    }
    out->inv_out_flat   = (float*)_aligned_malloc((total_floats + 8) * sizeof(float), 64);
    out->out_scale_flat = (float*)_aligned_malloc((total_floats + 8) * sizeof(float), 64);

    // Second pass: actually read + invert
    fseek(f, payload_start, SEEK_SET);
    size_t off = 0;
    for (uint32_t i = 0; i < n_ops; i++) {
        uint32_t nch = 0; fread(&nch, sizeof(uint32_t), 1, f);
        if (nch == 0) continue;
        float tmp[32768];
        if (nch > 32768) { fprintf(stderr, "OPSC2 nch too large: %u\n", nch); fclose(f); return -4; }
        fread(tmp, sizeof(float), nch, f);
        out->inv_out_per_ch[i]   = out->inv_out_flat + off;
        out->out_scale_per_ch[i] = out->out_scale_flat + off;
        for (uint32_t c = 0; c < nch; c++) {
            out->inv_out_flat[off + c]   = 1.0f / (tmp[c] + 1e-9f);
            out->out_scale_flat[off + c] = tmp[c];
        }
        off += nch;
    }
    fclose(f);
    fprintf(stderr, "OPSC2: loaded per-channel scales (total floats=%zu, input_pc_ch=%u)\n",
            total_floats, n_in);
    return 0;
}

static int load_op_scales(const char* path, OpScales* out) {
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "open %s fail\n", path); return -1; }
    char magic[4];
    if (fread(magic, 1, 4, f) != 4 || memcmp(magic, "OPSC", 4) != 0) { fclose(f); return -2; }
    fread(&out->n_ops, sizeof(uint32_t), 1, f);
    fread(&out->input_scale, sizeof(float), 1, f);
    out->scales = (float*)malloc(out->n_ops * sizeof(float));
    fread(out->scales, sizeof(float), out->n_ops, f);
    fclose(f);
    return 0;
}


int main(int argc, char** argv) {
    const char* path = (argc > 1) ? argv[1] : "models/w600k_r50_ffw2.bin";
    const char* in_path = NULL, * out_path = NULL;
    int server_mode = 0;
    int threads = -1; /* -1 = default (8) */
    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "--in") == 0 && i + 1 < argc) in_path = argv[++i];
        else if (strcmp(argv[i], "--out") == 0 && i + 1 < argc) out_path = argv[++i];
        else if (strcmp(argv[i], "--server") == 0) server_mode = 1;
        else if (strcmp(argv[i], "--threads") == 0 && i + 1 < argc) threads = atoi(argv[++i]);
    }
    if (threads > 0) omp_set_num_threads(threads);
    else if (!getenv("OMP_NUM_THREADS")) omp_set_num_threads(8);
    int validate_mode = (in_path && out_path);

    FFW2 m = {0};
    if (ffw2_load(path, &m) != 0) { fprintf(stderr, "load fail\n"); return 1; }
    fprintf(server_mode ? stderr : stdout, "FastFace INT8 (S26)  ops=%u\n", m.n_ops);

    OpScales sc = {0};
    if (load_op_scales("models/op_scales.bin", &sc) != 0) {
        fprintf(stderr, "Failed to load op_scales.bin. Run: N_CALIB=20 python export_op_scales.py\n");
        return 1;
    }
    if (sc.n_ops != m.n_ops) {
        fprintf(stderr, "op_scales count mismatch: %u vs %u FFW3 ops\n", sc.n_ops, m.n_ops);
        return 1;
    }
    // S36: optionally load per-channel OPSC2 file. If missing, driver falls back
    // to per-tensor scalar scales (current behavior).
    const char* v2_path = getenv("OPSC2_PATH");
    if (!v2_path) v2_path = "models/op_scales_v2.bin";
    int v2_ok = (load_op_scales_v2(v2_path, &sc) == 0);
    fprintf(server_mode ? stderr : stdout, "Loaded %u op scales (input_scale=%.4f%s)\n\n",
           sc.n_ops, sc.input_scale, v2_ok ? ", +per-channel" : "");

    // --- Residual-aware shape pre-pass ---
    uint8_t* is_shortcut = (uint8_t*)calloc(m.n_ops, 1);
    for (uint32_t i = 1; i < m.n_ops; i++) {
        if (m.ops[i].type == OP_CONV && m.ops[i-1].type == OP_SAVE_ID) is_shortcut[i] = 1;
    }
    Shape* shape_after = (Shape*)calloc(m.n_ops, sizeof(Shape));
    Shape* in_shape    = (Shape*)calloc(m.n_ops, sizeof(Shape));
    Shape cur = {3, 112, 112}, block_in = cur;
    size_t max_act = 0, max_im = 0, max_i32 = 0;
    for (uint32_t i = 0; i < m.n_ops; i++) {
        Op* op = &m.ops[i];
        Shape sh_in_op = is_shortcut[i] ? block_in : cur;
        in_shape[i] = sh_in_op;
        if (op->type == OP_CONV) {
            int H_out = (sh_in_op.H + 2 * op->pad - op->Kh) / op->stride + 1;
            int W_out = (sh_in_op.W + 2 * op->pad - op->Kw) / op->stride + 1;
            cur.C = op->Cout; cur.H = H_out; cur.W = W_out;
            size_t n = (size_t)cur.C * cur.H * cur.W;
            if (n > max_act) max_act = n;
            if (n > max_i32) max_i32 = n;
            int Kr = op->Cin * op->Kh * op->Kw;
            int Kp = (Kr + 3) & ~3;
            size_t im = (size_t)H_out * W_out * Kp;
            if (im > max_im) max_im = im;
        } else if (op->type == OP_GEMM) { cur.C = (int)op->N; cur.H = 1; cur.W = 1; }
        shape_after[i] = cur;
        if (op->type == OP_BLOCK_START) block_in = cur;
    }

    // Pack int8 weights
    typedef struct { int8_t* packed_w; int32_t* col_sums; } ConvPack;
    ConvPack* packs = (ConvPack*)calloc(m.n_ops, sizeof(ConvPack));
    for (uint32_t i = 0; i < m.n_ops; i++) {
        Op* op = &m.ops[i];
        if (op->type == OP_CONV) {
            int Kr = op->Cin * op->Kh * op->Kw;
            int Kp = (Kr + 3) & ~3;
            int8_t* wrow = (int8_t*)_aligned_malloc((size_t)Kp * op->Cout, 64);
            packs[i].packed_w = (int8_t*)_aligned_malloc((size_t)Kp * op->Cout, 64);
            packs[i].col_sums = (int32_t*)_aligned_malloc((size_t)op->Cout * sizeof(int32_t), 64);
            pack_conv_weight_nhwc(op->conv_w, op->Cout, op->Cin, op->Kh, op->Kw, Kp, wrow, packs[i].packed_w, packs[i].col_sums);
            _aligned_free(wrow);
        }
    }

    // S31: Precompute final-GEMM row sums and expected Au scratch.
    // For OP_GEMM (final Linear), K is padded-to-32 automatically because
    // the actual K = 512*7*7 = 25088 = 32 * 784.
    int32_t* gemm_w_row_sum = NULL;
    uint8_t* gemm_Au = NULL;
    for (uint32_t i = 0; i < m.n_ops; i++) {
        Op* op = &m.ops[i];
        if (op->type == OP_GEMM) {
            gemm_w_row_sum = (int32_t*)_aligned_malloc((size_t)op->N * sizeof(int32_t), 64);
            for (uint32_t n = 0; n < op->N; n++) {
                int32_t s = 0;
                const int8_t* r = op->gemm_w + (size_t)n * op->K;
                for (uint32_t k = 0; k < op->K; k++) s += (int32_t)r[k];
                gemm_w_row_sum[n] = s;
            }
            gemm_Au = (uint8_t*)_aligned_malloc((size_t)op->K + 64, 64);
            break;
        }
    }

    // Scratch buffers (int8 between ops, int32 for Conv output, plus im2col)
    int8_t* act_a   = (int8_t*) _aligned_malloc(max_act + 64, 64);
    int8_t* act_b   = (int8_t*) _aligned_malloc(max_act + 64, 64);
    int8_t* block_buf = (int8_t*)_aligned_malloc(max_act + 64, 64);
    int32_t* conv_out = (int32_t*)_aligned_malloc((max_i32 + 64) * sizeof(int32_t), 64);
    uint8_t* scratch_im = (uint8_t*)_aligned_malloc(max_im + 64, 64);
    int8_t* id_slots[24];
    float id_scales[24] = {0};
    for (int k = 0; k < 24; k++) id_slots[k] = (int8_t*)_aligned_malloc(max_act + 64, 64);
    float* fp32_tmp = (float*)_aligned_malloc((max_act + 64) * sizeof(float), 64);

    // Per-op input scale tracking: tracks the scale of the current A buffer.
    // current_scale[i] = output scale of op i. A after op i has scale current_scale[i].
    // Input to op 0 has scale sc.input_scale.

    // --- Input: file or random ---
    int N_INPUT = 3 * 112 * 112;
    float* fp32_input = (float*)_aligned_malloc(N_INPUT * sizeof(float), 64);
    if (validate_mode) {
        FILE* f = fopen(in_path, "rb"); if (!f) return 2;
        fread(fp32_input, sizeof(float), N_INPUT, f); fclose(f);
    } else {
        srand(42);
        for (int i = 0; i < N_INPUT; i++) fp32_input[i] = ((rand() % 2001) - 1000) / 1000.0f;
    }

    // Forward function
    float final_emb[512];

    #define RUN_ONCE() do {                                                                  \
        /* S38: per-channel initial quant when FFW4 + OPSC2 input_pc_scale exist. */         \
        if (m.version >= 4 && sc.input_pc_inv_scale) {                                       \
            quantize_fp32_nhwc_to_int8_per_channel(                                          \
                fp32_input, 112 * 112, 3, sc.input_pc_inv_scale, act_a);                      \
        } else {                                                                              \
            quantize_fp32_nhwc_to_int8(fp32_input, N_INPUT, sc.input_scale, act_a);          \
        }                                                                                     \
        int8_t* A = act_a;                                                                   \
        int8_t* BB = act_b;                                                                  \
        float A_scale = sc.input_scale;                                                      \
        const float* A_scale_pc = (m.version >= 4) ? sc.input_pc_scale : NULL;                \
        const float* block_scale_pc = A_scale_pc;                                            \
        const float* id_scales_pc[24] = {NULL};                                              \
        int save_idx = 0, add_idx = 0;                                                       \
        float block_scale = sc.input_scale;                                                  \
        memcpy(block_buf, act_a, N_INPUT);                                                   \
        for (uint32_t i = 0; i < m.n_ops; i++) {                                             \
            Op* op = &m.ops[i];                                                              \
            Shape sh_in  = in_shape[i];                                                      \
            Shape sh_out = shape_after[i];                                                   \
            int HW_in  = sh_in.H * sh_in.W;                                                  \
            int HW_out = sh_out.H * sh_out.W;                                                \
            float next_scale = sc.scales[i];                                                 \
            if (next_scale <= 0) next_scale = A_scale;                                       \
            double _prof_t0 = 0; int _prof_t = op->type;                                     \
            (void)_prof_t; (void)_prof_t0;                                                   \
            PROF_START(_prof_t0);                                                            \
            switch (op->type) {                                                              \
                case OP_CONV: {                                                              \
                    const int8_t* conv_in = is_shortcut[i] ? block_buf : A;                  \
                    float conv_in_scale = is_shortcut[i] ? block_scale : A_scale;            \
                    /* S38: FFW4 folds S_a into weights, so the runtime in_scale is 1.0. */   \
                    /* The per-channel scale info is "baked" into both w_int and a_int.  */   \
                    if (m.version >= 4) conv_in_scale = 1.0f;                                \
                    /* Look ahead: Conv+PReLU OR Conv+ADD (mutually exclusive in IResNet) */ \
                    const float* fuse_prelu = NULL;                                          \
                    const int8_t* fuse_add_src = NULL;                                       \
                    float fuse_add_scale = 0.0f;                                             \
                    const float* fuse_add_scale_pc = NULL;                                   \
                    float eff_next_scale = next_scale;                                       \
                    int skip_next = 0;                                                       \
                    int add_fused = 0;                                                       \
                    uint32_t eff_idx = i;                                                    \
                    if (i + 1 < m.n_ops && m.ops[i+1].type == OP_PRELU) {                    \
                        fuse_prelu = m.ops[i+1].prelu_slope;                                 \
                        eff_next_scale = sc.scales[i+1];                                     \
                        if (eff_next_scale <= 0) eff_next_scale = next_scale;                \
                        skip_next = 1;                                                       \
                        eff_idx = i + 1;                                                     \
                    } else if (i + 1 < m.n_ops && m.ops[i+1].type == OP_ADD) {               \
                        fuse_add_src   = id_slots[add_idx % 24];                             \
                        fuse_add_scale = id_scales[add_idx % 24];                            \
                        /* S38: per-channel shortcut scale if FFW4 */                        \
                        if (m.version >= 4) fuse_add_scale_pc = id_scales_pc[add_idx % 24];  \
                        eff_next_scale = sc.scales[i+1];                                     \
                        if (eff_next_scale <= 0) eff_next_scale = next_scale;                \
                        skip_next = 1;                                                       \
                        add_fused = 1;                                                       \
                        eff_idx = i + 1;                                                     \
                    }                                                                        \
                    const float* pc_inv = (m.version >= 4 && sc.inv_out_per_ch)              \
                                          ? sc.inv_out_per_ch[eff_idx] : NULL;                \
                    fastface_conv2d_i8_nhwc(conv_in, sh_in.C, sh_in.H, sh_in.W,              \
                        op->Cout, op->Kh, op->Kw, op->stride, op->pad,                       \
                        packs[i].packed_w, packs[i].col_sums,                                \
                        sh_out.H, sh_out.W, conv_out, scratch_im);                           \
                    fused_epilogue_int8(conv_out, HW_out, op->Cout,                          \
                        conv_in_scale, op->conv_scales, op->conv_bias,                       \
                        NULL, NULL, fuse_prelu, fuse_add_src, fuse_add_scale,                \
                        fuse_add_scale_pc, pc_inv, eff_next_scale, BB);                      \
                    { int8_t* t = A; A = BB; BB = t; }                                       \
                    A_scale = eff_next_scale;                                                \
                    if (m.version >= 4 && sc.out_scale_per_ch                                \
                        && sc.out_scale_per_ch[eff_idx]) A_scale_pc = sc.out_scale_per_ch[eff_idx]; \
                    if (skip_next) i++;  /* skip the fused PReLU or ADD */                   \
                    if (add_fused) add_idx++;                                                \
                    break;                                                                   \
                }                                                                            \
                case OP_BN: {                                                                \
                    /* S38: for every BN in FFW4, use per-channel in/out to maintain */       \
                    /* per-channel scale semantics throughout the pipeline.         */        \
                    const float* bn_inv_out = NULL;                                          \
                    if (m.version >= 4 && sc.inv_out_per_ch && sc.inv_out_per_ch[i]           \
                        && sc.inv_out_len[i] == (uint32_t)sh_in.C) {                         \
                        bn_inv_out = sc.inv_out_per_ch[i];                                   \
                    }                                                                        \
                    const float* bn_in_pc = NULL;                                            \
                    if (m.version >= 4 && sc.out_scale_per_ch && A_scale_pc) {                \
                        bn_in_pc = A_scale_pc;                                               \
                    }                                                                        \
                    bn_prelu_requant_int8(A, A_scale, bn_in_pc,                              \
                                          op->bn_scale, op->bn_offset, NULL,                  \
                                          bn_inv_out, next_scale, A, HW_in, sh_in.C);        \
                    A_scale = next_scale;                                                    \
                    if (m.version >= 4 && sc.out_scale_per_ch                                \
                        && sc.out_scale_per_ch[i]) A_scale_pc = sc.out_scale_per_ch[i];      \
                    break;                                                                   \
                }                                                                            \
                case OP_PRELU: {                                                             \
                    const float* prelu_in_pc = NULL;                                         \
                    const float* prelu_inv_out = NULL;                                       \
                    if (m.version >= 4 && sc.out_scale_per_ch && A_scale_pc) {                \
                        prelu_in_pc = A_scale_pc;                                            \
                    }                                                                        \
                    if (m.version >= 4 && sc.inv_out_per_ch && sc.inv_out_per_ch[i]) {        \
                        prelu_inv_out = sc.inv_out_per_ch[i];                                 \
                    }                                                                        \
                    bn_prelu_requant_int8(A, A_scale, prelu_in_pc, NULL, NULL, op->prelu_slope, \
                                          prelu_inv_out, next_scale, A, HW_in, sh_in.C);     \
                    A_scale = next_scale;                                                    \
                    if (m.version >= 4 && sc.out_scale_per_ch                                \
                        && sc.out_scale_per_ch[i]) A_scale_pc = sc.out_scale_per_ch[i];      \
                    break;                                                                   \
                }                                                                            \
                case OP_BLOCK_START:                                                         \
                    memcpy(block_buf, A, (size_t)sh_in.C * HW_in);                           \
                    block_scale = A_scale;                                                   \
                    block_scale_pc = A_scale_pc;                                             \
                    break;                                                                   \
                case OP_ADD: {                                                               \
                    /* S105 NOTE: in FFW4 FastFace's Conv+ADD lookahead fusion absorbs the   \
                     * ADD into the preceding Conv's fused_epilogue_int8, so this default    \
                     * pure-ADD path is dead code for v4 models. Kept for backward compat.   \
                     */                                                                       \
                    add_requant_int8(A, A_scale, id_slots[add_idx % 24], id_scales[add_idx % 24], \
                                      A, next_scale, sh_out.C * HW_out);                     \
                    A_scale = next_scale;                                                    \
                    add_idx++;                                                               \
                    break;                                                                   \
                }                                                                            \
                case OP_SAVE_ID:                                                             \
                    memcpy(id_slots[save_idx % 24], A, (size_t)sh_in.C * HW_in);             \
                    id_scales[save_idx % 24] = A_scale;                                      \
                    id_scales_pc[save_idx % 24] = A_scale_pc;                                \
                    save_idx++;                                                              \
                    break;                                                                   \
                case OP_GEMM: {                                                              \
                    /* S31: shift A into uint8 domain and call VNNI matvec */                \
                    const __m256i xor_mask = _mm256_set1_epi8((char)0x80);                   \
                    uint32_t K32 = op->K & ~31u;                                              \
                    for (uint32_t k = 0; k < K32; k += 32) {                                 \
                        __m256i a = _mm256_loadu_si256((const __m256i*)(A + k));             \
                        _mm256_storeu_si256((__m256i*)(gemm_Au + k),                         \
                                            _mm256_xor_si256(a, xor_mask));                  \
                    }                                                                        \
                    for (uint32_t k = K32; k < op->K; k++) {                                 \
                        gemm_Au[k] = (uint8_t)((int)A[k] + 128);                             \
                    }                                                                        \
                    /* S37: FFW4 has per-channel activation scale pre-folded into weights, */  \
                    /* so matvec uses A_scale=1.0 (int8 values already encode correct ref).  */  \
                    float matvec_A_scale = (m.version >= 4) ? 1.0f : A_scale;                \
                    fastface_gemm_i8_matvec_vnni(                                            \
                        gemm_Au, op->gemm_w, gemm_w_row_sum,                                 \
                        op->gemm_scales, op->gemm_bias,                                      \
                        matvec_A_scale, final_emb, (int)op->N, (int)op->K);                  \
                    break;                                                                   \
                }                                                                            \
                case OP_FLATTEN: {                                                           \
                    int Hf = sh_in.H, Wf = sh_in.W, Cf = sh_in.C;                            \
                    for (int c = 0; c < Cf; c++) {                                           \
                        for (int h = 0; h < Hf; h++) {                                       \
                            for (int w = 0; w < Wf; w++) {                                   \
                                BB[c * Hf * Wf + h * Wf + w] = A[(h * Wf + w) * Cf + c];     \
                            }                                                                \
                        }                                                                    \
                    }                                                                        \
                    { int8_t* t = A; A = BB; BB = t; }                                       \
                    break;                                                                   \
                }                                                                            \
            }                                                                                \
            PROF_END(_prof_t0, _prof_t);                                                     \
        }                                                                                    \
    } while(0)

    RUN_ONCE();
    fprintf(stderr, "First pass OK\n");
    fprintf(stderr, "final_emb[0..4] = %g %g %g %g %g\n",
            final_emb[0], final_emb[1], final_emb[2], final_emb[3], final_emb[4]);

    if (validate_mode) {
        FILE* f = fopen(out_path, "wb");
        fwrite(final_emb, sizeof(float), 512, f);
        fclose(f);
        return 0;
    }

    // S56: server mode — stream faces over stdin/stdout until EOF.
    // Input: N_INPUT = 3*112*112 fp32 NHWC per face.
    // Output: 512 fp32 embedding per face.
    if (server_mode) {
#ifdef _WIN32
        /* Set stdin/stdout to binary to avoid Windows \r\n translation */
        _setmode(_fileno(stdin),  _O_BINARY);
        _setmode(_fileno(stdout), _O_BINARY);
#endif
        fprintf(stderr, "Server mode ready. Awaiting %d-float inputs on stdin.\n", N_INPUT);
        while (1) {
            size_t got = fread(fp32_input, sizeof(float), (size_t)N_INPUT, stdin);
            if (got == 0) break;  /* EOF */
            if (got != (size_t)N_INPUT) {
                fprintf(stderr, "partial read (%zu/%d floats) — abort\n", got, N_INPUT);
                return 4;
            }
            RUN_ONCE();
            size_t wrote = fwrite(final_emb, sizeof(float), 512, stdout);
            if (wrote != 512) { fprintf(stderr, "partial write\n"); return 5; }
            fflush(stdout);
        }
        return 0;
    }

    // Bench
    const int ITER = 30;
    double best = 1e9;
    for (int trial = 0; trial < 5; trial++) {
        double t0 = now_s();
        for (int it = 0; it < ITER; it++) RUN_ONCE();
        double dt = (now_s() - t0) / ITER;
        if (dt < best) best = dt;
        printf("Trial %d: %.2f ms/inference  (%.1f face/s)\n", trial, dt * 1000, 1/dt);
    }
    printf("\n========== INT8 END-TO-END RESULT ==========\n");
    printf("Best: %.2f ms/inference  (%.1f face/s)\n", best * 1000, 1/best);
    printf("ORT baseline: ~32 ms\n");
    printf("Speedup: %.2fx\n", 32.0 / (best * 1000));
    printf("============================================\n");

#ifdef PROFILE_OPS
    printf("\n========== OP-TYPE BREAKDOWN (cumulative across all trials) ==========\n");
    double total = 0;
    for (int t = 0; t < 9; t++) total += op_type_time[t];
    for (int t = 1; t < 9; t++) {
        if (op_type_count[t] == 0) continue;
        double ms_total = op_type_time[t] * 1000.0;
        double pct = 100.0 * op_type_time[t] / total;
        double ms_per = (op_type_time[t] / op_type_count[t]) * 1000.0;
        printf("  %-14s %4llu calls  %7.1f ms total  %5.2f%%  %6.4f ms/call\n",
               op_type_name[t], (unsigned long long)op_type_count[t], ms_total, pct, ms_per);
    }
    printf("  %-14s                %7.1f ms total  100%%\n", "TOTAL", total * 1000.0);
    printf("=======================================================================\n");
#endif

    return 0;
}
