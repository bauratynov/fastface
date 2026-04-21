// Session 24 — FP32 batched driver. Based on arcface_forward_fp32.c.
// Accepts --batch N (or BATCH env). Processes N faces at once.
// Winograd batch-fused (M_gemm = N*num_tiles). Non-Winograd convs per-face loop.
// BN/PReLU/Add apply to total N*HW positions at once.

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include <windows.h>

void pack_B_fp32(const float* B, int K, int N, float* Bp);
void fastface_gemm_fp32(const float* A, const float* Bp, float* C, int M, int K, int N);
void pack_conv_weight_fp32_nhwc(const float* weight, int Cout, int Cin, int Kh, int Kw, int K_padded,
                                  float* w_rowmajor, float* w_packed);
void fastface_conv2d_fp32_nhwc(const float* input, int Cin, int H_in, int W_in,
                                int Cout, int Kh, int Kw, int stride, int pad,
                                const float* weight_packed, int H_out, int W_out,
                                float* output, float* scratch_im);
void fastface_conv2d_fp32_nhwc_batched(
    const float* input, int B, int Cin, int H_in, int W_in,
    int Cout, int Kh, int Kw, int stride, int pad,
    const float* weight_packed, int H_out, int W_out,
    float* output, float* scratch_im);
void bn_fp32_nhwc(float* x, const float* scale, const float* offset, int HW, int Cout);
void prelu_fp32_nhwc(float* x, const float* slope, int HW, int Cout);
void add_bias_nhwc(float* x, const float* bias, int HW, int Cout);
void add_fp32(const float* a, const float* b, float* out, int n);

void winograd_precompute_weights_packed(
    const float* weight, int Cout, int Cin, float* U_packed, float* scratch);
void fastface_winograd_conv_3x3_s1_p1_packed_bias_batched(
    const float* input, int B, int Cin, int H_in, int W_in,
    int Cout, const float* U_packed, const float* bias,
    float* output, float* V_wino, float* M_wino);
void fastface_winograd_conv_3x3_s1_p1_packed_bias_batched_bn(
    const float* input, int B, int Cin, int H_in, int W_in,
    int Cout, const float* U_packed, const float* bias,
    const float* bn_scale, const float* bn_offset,
    float* output, float* V_wino, float* M_wino);
void fastface_winograd_conv_3x3_s1_p1_packed_bias_batched_bn_prelu(
    const float* input, int B, int Cin, int H_in, int W_in,
    int Cout, const float* U_packed, const float* bias,
    const float* bn_scale, const float* bn_offset,
    const float* prelu_slope,
    float* output, float* V_wino, float* M_wino);

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
typedef struct { void* data; size_t size; uint32_t n_ops; Op* ops; } FFW2;
int ffw2_load(const char* path, FFW2* out);
typedef struct { int C, H, W; } Shape;

static double now_s(void) {
    LARGE_INTEGER q, f; QueryPerformanceCounter(&q); QueryPerformanceFrequency(&f);
    return (double)q.QuadPart / (double)f.QuadPart;
}

static void dequant_conv_weight(const int8_t* w_int, const float* scales,
                                 int Cout, int Cin, int Kh, int Kw, float* w_fp32)
{
    int KW = Kh * Kw;
    int per_co = Cin * KW;
    for (int co = 0; co < Cout; co++) {
        float s = scales[co];
        for (int k = 0; k < per_co; k++) w_fp32[co * per_co + k] = (float)w_int[co * per_co + k] * s;
    }
}

int main(int argc, char** argv) {
    if (!getenv("OMP_NUM_THREADS")) omp_set_num_threads(8);

    const char* path = (argc > 1) ? argv[1] : "models/w600k_r50_ffw2.bin";
    const char* in_path = NULL;
    const char* out_path = NULL;
    int B = 1;
    const char* bstr = getenv("BATCH");
    if (bstr) B = atoi(bstr);
    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "--batch") == 0 && i + 1 < argc) B = atoi(argv[++i]);
        else if (strcmp(argv[i], "--in") == 0 && i + 1 < argc) in_path = argv[++i];
        else if (strcmp(argv[i], "--out") == 0 && i + 1 < argc) out_path = argv[++i];
    }
    if (B < 1) B = 1;
    int validate_mode = (in_path && out_path);

    FFW2 m = {0};
    if (ffw2_load(path, &m) != 0) { fprintf(stderr, "load fail\n"); return 1; }

    printf("FastFace FP32 Batched (S24)  batch=%d  ops=%u\n\n", B, m.n_ops);

    // --- Residual-aware shape + shortcut pre-pass (same as per-face driver) ---
    uint8_t* is_shortcut = (uint8_t*)calloc(m.n_ops, 1);
    for (uint32_t i = 1; i < m.n_ops; i++) {
        if (m.ops[i].type == OP_CONV && m.ops[i-1].type == OP_SAVE_ID) is_shortcut[i] = 1;
    }
    Shape* shape_after = (Shape*)calloc(m.n_ops, sizeof(Shape));
    Shape* in_shape    = (Shape*)calloc(m.n_ops, sizeof(Shape));
    Shape cur = {3, 112, 112};
    Shape block_in = cur;
    size_t max_act_per_face = 0, max_im_per_face = 0;
    for (uint32_t i = 0; i < m.n_ops; i++) {
        Op* op = &m.ops[i];
        Shape sh_in_op = is_shortcut[i] ? block_in : cur;
        in_shape[i] = sh_in_op;
        if (op->type == OP_CONV) {
            int H_out = (sh_in_op.H + 2 * op->pad - op->Kh) / op->stride + 1;
            int W_out = (sh_in_op.W + 2 * op->pad - op->Kw) / op->stride + 1;
            cur.C = op->Cout; cur.H = H_out; cur.W = W_out;
            int M = H_out * W_out;
            int M_padded = (M + 3) & ~3;
            size_t n = (size_t)M_padded * cur.C;
            if (n > max_act_per_face) max_act_per_face = n;
            int Kr = op->Cin * op->Kh * op->Kw;
            int Kp = (Kr + 15) & ~15;
            size_t im = (size_t)M_padded * Kp;  // per-face im2col
            if (im > max_im_per_face) max_im_per_face = im;
        } else if (op->type == OP_GEMM) {
            cur.C = (int)op->N; cur.H = 1; cur.W = 1;
        }
        shape_after[i] = cur;
        if (op->type == OP_BLOCK_START) block_in = cur;
    }

    // --- Dequant + pack Conv weights (per-face, batch-independent) + Winograd precompute ---
    typedef struct { float* w_fp32; float* w_packed; int K_padded; float* U_wino; } ConvFP;
    ConvFP* packs = (ConvFP*)calloc(m.n_ops, sizeof(ConvFP));
    for (uint32_t i = 0; i < m.n_ops; i++) {
        Op* op = &m.ops[i];
        if (op->type == OP_CONV) {
            int Kr = op->Cin * op->Kh * op->Kw;
            int Kp = (Kr + 15) & ~15;
            packs[i].K_padded = Kp;
            size_t w_elems = (size_t)op->Cout * op->Cin * op->Kh * op->Kw;
            packs[i].w_fp32 = (float*)_aligned_malloc(w_elems * sizeof(float), 64);
            dequant_conv_weight(op->conv_w, op->conv_scales, op->Cout, op->Cin, op->Kh, op->Kw, packs[i].w_fp32);
            float* w_rowmajor = (float*)_aligned_malloc((size_t)Kp * op->Cout * sizeof(float), 64);
            packs[i].w_packed = (float*)_aligned_malloc((size_t)Kp * op->Cout * sizeof(float), 64);
            pack_conv_weight_fp32_nhwc(packs[i].w_fp32, op->Cout, op->Cin, op->Kh, op->Kw, Kp, w_rowmajor, packs[i].w_packed);
            _aligned_free(w_rowmajor);
            if (op->Kh == 3 && op->Kw == 3 && op->stride == 1 && op->pad == 1) {
                size_t u_bytes = (size_t)16 * op->Cin * op->Cout * sizeof(float);
                packs[i].U_wino = (float*)_aligned_malloc(u_bytes, 64);
                float* tmp_scratch = (float*)_aligned_malloc(u_bytes, 64);
                winograd_precompute_weights_packed(packs[i].w_fp32, op->Cout, op->Cin, packs[i].U_wino, tmp_scratch);
                _aligned_free(tmp_scratch);
            }
        }
    }

    // --- Scratch: multiply per-face buffers by B ---
    size_t max_act_total = max_act_per_face * (size_t)B;
    size_t max_im_total  = max_im_per_face * (size_t)B;  // batched im2col writes B * per-face
    // Winograd V/M: 16 * B * num_tiles_padded_per_face_to_4 * C
    size_t max_V_wino = 0, max_M_wino = 0;
    {
        Shape csh = {3, 112, 112}; Shape blk = csh;
        for (uint32_t i = 0; i < m.n_ops; i++) {
            Op* op = &m.ops[i];
            Shape in_sh = is_shortcut[i] ? blk : csh;
            if (op->type == OP_CONV) {
                int H_out = (in_sh.H + 2 * op->pad - op->Kh) / op->stride + 1;
                int W_out = (in_sh.W + 2 * op->pad - op->Kw) / op->stride + 1;
                if (op->Kh == 3 && op->Kw == 3 && op->stride == 1 && op->pad == 1) {
                    int tH = (H_out + 1) / 2, tW = (W_out + 1) / 2;
                    size_t ntil = (size_t)B * tH * tW;
                    size_t ntil_padded = (ntil + 3) & ~3ULL;
                    size_t v  = 16 * ntil_padded * op->Cin;
                    size_t me = 16 * ntil_padded * op->Cout;
                    if (v > max_V_wino) max_V_wino = v;
                    if (me > max_M_wino) max_M_wino = me;
                }
                csh.C = op->Cout; csh.H = H_out; csh.W = W_out;
            } else if (op->type == OP_GEMM) { csh.C = (int)op->N; csh.H = 1; csh.W = 1; }
            if (op->type == OP_BLOCK_START) blk = csh;
        }
    }

    float* scratch_im    = (float*)_aligned_malloc((max_im_total + 64) * sizeof(float), 64);
    float* act_a         = (float*)_aligned_malloc((max_act_total + 64) * sizeof(float), 64);
    float* act_b         = (float*)_aligned_malloc((max_act_total + 64) * sizeof(float), 64);
    float* block_buf     = (float*)_aligned_malloc((max_act_total + 64) * sizeof(float), 64);
    float* V_wino_scratch = (float*)_aligned_malloc((max_V_wino + 64) * sizeof(float), 64);
    float* M_wino_scratch = (float*)_aligned_malloc((max_M_wino + 64) * sizeof(float), 64);
    float* id_slots[24];
    for (int k = 0; k < 24; k++) id_slots[k] = (float*)_aligned_malloc((max_act_total + 64) * sizeof(float), 64);

    printf("Scratch: act=%.1f MB, im=%.1f MB, V/M=%.1f/%.1f MB, id_slots=%.1f MB\n",
           max_act_total * 4 / 1048576.0, max_im_total * 4 / 1048576.0,
           max_V_wino * 4 / 1048576.0, max_M_wino * 4 / 1048576.0,
           24 * max_act_total * 4 / 1048576.0);

    // --- Input: from file (validate) or random (bench) ---
    int N_INPUT_per_face = 3 * 112 * 112;
    float* input_batch = (float*)_aligned_malloc((size_t)B * N_INPUT_per_face * sizeof(float), 64);
    if (validate_mode) {
        FILE* f = fopen(in_path, "rb");
        if (!f) { fprintf(stderr, "open fail\n"); return 2; }
        size_t got = fread(input_batch, sizeof(float), (size_t)B * N_INPUT_per_face, f);
        fclose(f);
        if (got != (size_t)B * N_INPUT_per_face) { fprintf(stderr, "short input\n"); return 3; }
    } else {
        srand(42);
        for (int i = 0; i < B * N_INPUT_per_face; i++) input_batch[i] = ((rand() % 2001) - 1000) / 1000.0f;
    }

    // --- Forward runner: processes all B faces in one pass per op ---
    #define RUN_ONCE() do {                                                                  \
        memcpy(act_a, input_batch, (size_t)B * N_INPUT_per_face * sizeof(float));            \
        memcpy(block_buf, input_batch, (size_t)B * N_INPUT_per_face * sizeof(float));        \
        float* A = act_a;                                                                    \
        float* BB = act_b;                                                                   \
        int save_idx = 0, add_idx = 0;                                                       \
        const float* pending_bn_scale = NULL;                                                \
        const float* pending_bn_offset = NULL;                                               \
        int pending_is_block_in = 0;  /* if BN is the 1st op of residual block */            \
        for (uint32_t i = 0; i < m.n_ops; i++) {                                             \
            Op* op = &m.ops[i];                                                              \
            Shape sh_in  = in_shape[i];                                                      \
            Shape sh_out = shape_after[i];                                                   \
            int HW_in  = sh_in.H * sh_in.W;                                                  \
            int HW_out = sh_out.H * sh_out.W;                                                \
            int B_HW_in  = B * HW_in;                                                        \
            int B_HW_out = B * HW_out;                                                       \
            switch (op->type) {                                                              \
                case OP_CONV: {                                                              \
                    const float* conv_in = is_shortcut[i] ? block_buf : A;                   \
                    int bias_fused = 0;                                                      \
                    int prelu_fused = 0;                                                     \
                    const float* fuse_prelu = NULL;                                          \
                    if (i + 1 < m.n_ops && m.ops[i+1].type == OP_PRELU) {                    \
                        fuse_prelu = m.ops[i+1].prelu_slope;                                 \
                    }                                                                        \
                    if (packs[i].U_wino && !is_shortcut[i]) {                                \
                        if (pending_bn_scale) {                                               \
                            bn_fp32_nhwc(A, pending_bn_scale, pending_bn_offset, B_HW_in, sh_in.C); \
                            pending_bn_scale = NULL; pending_bn_offset = NULL;                \
                        }                                                                    \
                        fastface_winograd_conv_3x3_s1_p1_packed_bias_batched_bn_prelu(       \
                            conv_in, B, sh_in.C, sh_in.H, sh_in.W,                           \
                            op->Cout, packs[i].U_wino, op->conv_bias,                        \
                            NULL, NULL, fuse_prelu, BB,                                      \
                            V_wino_scratch, M_wino_scratch);                                 \
                        bias_fused = 1;                                                      \
                        if (fuse_prelu) prelu_fused = 1;                                     \
                    } else if (packs[i].U_wino) {                                            \
                        /* Shortcut Conv: flush pending BN first (applies to block_buf). */  \
                        if (pending_bn_scale) {                                              \
                            bn_fp32_nhwc(block_buf, pending_bn_scale, pending_bn_offset, B_HW_in, sh_in.C); \
                            pending_bn_scale = NULL; pending_bn_offset = NULL;               \
                        }                                                                    \
                        fastface_winograd_conv_3x3_s1_p1_packed_bias_batched(                \
                            conv_in, B, sh_in.C, sh_in.H, sh_in.W,                           \
                            op->Cout, packs[i].U_wino, op->conv_bias, BB,                    \
                            V_wino_scratch, M_wino_scratch);                                 \
                        bias_fused = 1;                                                      \
                    } else {                                                                 \
                        if (pending_bn_scale) {                                              \
                            bn_fp32_nhwc((float*)conv_in, pending_bn_scale, pending_bn_offset, B_HW_in, sh_in.C); \
                            pending_bn_scale = NULL; pending_bn_offset = NULL;               \
                        }                                                                    \
                        /* Per-face loop (safer; batched had precision drift) */             \
                        for (int b = 0; b < B; b++) {                                        \
                            fastface_conv2d_fp32_nhwc(                                       \
                                conv_in  + (size_t)b * sh_in.C * sh_in.H * sh_in.W,          \
                                sh_in.C, sh_in.H, sh_in.W,                                   \
                                op->Cout, op->Kh, op->Kw, op->stride, op->pad,               \
                                packs[i].w_packed,                                           \
                                sh_out.H, sh_out.W,                                          \
                                BB + (size_t)b * op->Cout * sh_out.H * sh_out.W,             \
                                scratch_im);                                                 \
                        }                                                                    \
                    }                                                                        \
                    { float* t = A; A = BB; BB = t; }                                        \
                    if (!bias_fused)                                                         \
                        add_bias_nhwc(A, op->conv_bias, B_HW_out, op->Cout);                 \
                    if (prelu_fused) i++;                                                    \
                    break;                                                                   \
                }                                                                            \
                case OP_BN:                                                                  \
                    bn_fp32_nhwc(A, op->bn_scale, op->bn_offset, B_HW_in, sh_in.C);          \
                    break;                                                                   \
                case OP_BLOCK_START:                                                         \
                    /* Any stale pending BN (should not happen here) — flush */              \
                    if (pending_bn_scale) {                                                  \
                        bn_fp32_nhwc(A, pending_bn_scale, pending_bn_offset, B_HW_in, sh_in.C); \
                        pending_bn_scale = NULL; pending_bn_offset = NULL;                   \
                    }                                                                        \
                    memcpy(block_buf, A, (size_t)sh_in.C * B_HW_in * sizeof(float));         \
                    break;                                                                   \
                case OP_PRELU:                                                               \
                    if (pending_bn_scale) {                                                  \
                        bn_fp32_nhwc(A, pending_bn_scale, pending_bn_offset, B_HW_in, sh_in.C); \
                        pending_bn_scale = NULL; pending_bn_offset = NULL;                   \
                    }                                                                        \
                    prelu_fp32_nhwc(A, op->prelu_slope, B_HW_in, sh_in.C);                   \
                    break;                                                                   \
                case OP_ADD:                                                                 \
                    if (pending_bn_scale) {                                                  \
                        bn_fp32_nhwc(A, pending_bn_scale, pending_bn_offset, B_HW_in, sh_in.C); \
                        pending_bn_scale = NULL; pending_bn_offset = NULL;                   \
                    }                                                                        \
                    add_fp32(A, id_slots[add_idx % 24], A, sh_out.C * B_HW_out);             \
                    add_idx++;                                                               \
                    break;                                                                   \
                case OP_SAVE_ID:                                                             \
                    if (pending_bn_scale) {                                                  \
                        bn_fp32_nhwc(A, pending_bn_scale, pending_bn_offset, B_HW_in, sh_in.C); \
                        pending_bn_scale = NULL; pending_bn_offset = NULL;                   \
                    }                                                                        \
                    memcpy(id_slots[save_idx % 24], A, (size_t)sh_in.C * B_HW_in * sizeof(float)); \
                    save_idx++;                                                              \
                    break;                                                                   \
                case OP_GEMM: {                                                              \
                    if (pending_bn_scale) {                                                  \
                        bn_fp32_nhwc(A, pending_bn_scale, pending_bn_offset, B_HW_in, sh_in.C); \
                        pending_bn_scale = NULL; pending_bn_offset = NULL;                   \
                    }                                                                        \
                    for (int b = 0; b < B; b++) {                                            \
                        for (uint32_t nn = 0; nn < op->N; nn++) {                            \
                            float s = op->gemm_bias[nn];                                     \
                            for (uint32_t kk = 0; kk < op->K; kk++) {                        \
                                s += A[(size_t)b * op->K + kk] * (float)op->gemm_w[nn * op->K + kk] * op->gemm_scales[nn]; \
                            }                                                                \
                            BB[(size_t)b * op->N + nn] = s;                                  \
                        }                                                                    \
                    }                                                                        \
                    { float* t = A; A = BB; BB = t; }                                        \
                    break;                                                                   \
                }                                                                            \
                case OP_FLATTEN: {                                                           \
                    if (pending_bn_scale) {                                                  \
                        bn_fp32_nhwc(A, pending_bn_scale, pending_bn_offset, B_HW_in, sh_in.C); \
                        pending_bn_scale = NULL; pending_bn_offset = NULL;                   \
                    }                                                                        \
                    int Hf = sh_in.H, Wf = sh_in.W, Cf = sh_in.C;                            \
                    for (int b = 0; b < B; b++) {                                            \
                        for (int c = 0; c < Cf; c++) {                                       \
                            for (int h = 0; h < Hf; h++) {                                   \
                                for (int w = 0; w < Wf; w++) {                               \
                                    BB[(size_t)b * Cf * Hf * Wf + c * Hf * Wf + h * Wf + w]  \
                                        = A[(size_t)b * Hf * Wf * Cf + (h * Wf + w) * Cf + c]; \
                                }                                                            \
                            }                                                                \
                        }                                                                    \
                    }                                                                        \
                    { float* t = A; A = BB; BB = t; }                                        \
                    break;                                                                   \
                }                                                                            \
            }                                                                                \
        }                                                                                    \
        final_A_ptr = A;                                                                     \
    } while(0)

    float* final_A_ptr = NULL;
    RUN_ONCE();
    fprintf(stderr, "First pass OK\n");

    if (validate_mode) {
        FILE* f = fopen(out_path, "wb");
        if (!f) { fprintf(stderr, "out fail\n"); return 4; }
        fwrite(final_A_ptr, sizeof(float), (size_t)B * 512, f);
        fclose(f);
        fprintf(stderr, "wrote %d*512 floats\n", B);
        return 0;
    }

    // Bench loop
    const int ITER = 10;
    double best = 1e9;
    for (int trial = 0; trial < 5; trial++) {
        double t0 = now_s();
        for (int it = 0; it < ITER; it++) RUN_ONCE();
        double dt = (now_s() - t0) / ITER;
        if (dt < best) best = dt;
        printf("Trial %d: %.2f ms/batch  (%.2f ms/face, %.1f face/s)\n",
               trial, dt * 1000, dt * 1000 / B, B / dt);
    }

    printf("\n========== BATCH=%d RESULT ==========\n", B);
    printf("Best: %.2f ms/batch = %.2f ms/face = %.1f face/s\n", best * 1000, best * 1000 / B, B / best);
    printf("=======================================\n");

    return 0;
}
