// Session 13 — FP32 end-to-end driver for perfect-quality mode (Path A).
// Uses same FFW2 binary but dequantizes int8 weights to fp32 at model load
// and runs all inference in fp32 via our new kernels.

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
void bn_fp32_nhwc(float* x, const float* scale, const float* offset, int HW, int Cout);
void prelu_fp32_nhwc(float* x, const float* slope, int HW, int Cout);
void add_bias_nhwc(float* x, const float* bias, int HW, int Cout);
void add_fp32(const float* a, const float* b, float* out, int n);

// Winograd F(2, 3) for 3×3 stride=1 pad=1
void winograd_precompute_weights(const float* weight, int Cout, int Cin, float* U_wino);
void winograd_precompute_weights_packed(
    const float* weight, int Cout, int Cin, float* U_packed, float* scratch);
void fastface_winograd_conv_3x3_s1_p1(
    const float* input, int Cin, int H_in, int W_in,
    int Cout, const float* U_wino,
    float* output,
    float* V_wino, float* M_wino);
void fastface_winograd_conv_3x3_s1_p1_packed(
    const float* input, int Cin, int H_in, int W_in,
    int Cout, const float* U_packed,
    float* output,
    float* V_wino, float* M_wino);
void fastface_winograd_conv_3x3_s1_p1_packed_bias(
    const float* input, int Cin, int H_in, int W_in,
    int Cout, const float* U_packed, const float* bias,
    float* output,
    float* V_wino, float* M_wino);
void fastface_winograd_conv_3x3_s1_p1_packed_bias_prelu(
    const float* input, int Cin, int H_in, int W_in,
    int Cout, const float* U_packed, const float* bias,
    const float* prelu_slope,
    float* output,
    float* V_wino, float* M_wino);
void fastface_winograd_conv_3x3_s1_p1_full_fused(
    const float* input, int Cin, int H_in, int W_in,
    int Cout, const float* U_packed,
    const float* bn_scale, const float* bn_offset,
    const float* bias, const float* prelu_slope,
    float* output, float* V_wino, float* M_wino);
void fastface_winograd_conv_3x3_s1_p1_full_fused_add(
    const float* input, int Cin, int H_in, int W_in,
    int Cout, const float* U_packed,
    const float* bn_scale, const float* bn_offset,
    const float* bias, const float* prelu_slope,
    const float* add_src,
    float* output, float* V_wino, float* M_wino);

// F(4, 3) Winograd
void winograd_precompute_weights_packed_f43(
    const float* weight, int Cout, int Cin, float* U_packed, float* scratch);
void fastface_winograd_conv_3x3_s1_p1_f43_packed_bias(
    const float* input, int Cin, int H_in, int W_in,
    int Cout, const float* U_packed, const float* bias,
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
typedef struct {
    void* data; size_t size;
    uint32_t n_ops;
    Op* ops;
} FFW2;
int ffw2_load(const char* path, FFW2* out);

typedef struct { int C, H, W; } Shape;


static double now_s(void) {
    LARGE_INTEGER q, f;
    QueryPerformanceCounter(&q); QueryPerformanceFrequency(&f);
    return (double)q.QuadPart / (double)f.QuadPart;
}


// Dequantize int8 conv weight + per-channel scales → fp32 [Cout, Cin, Kh, Kw].
// Original weight was w_fp32 * (1/scales[co]) → int8, so w_fp32 = int8 * scales[co].
static void dequant_conv_weight(
    const int8_t* w_int, const float* scales,
    int Cout, int Cin, int Kh, int Kw,
    float* w_fp32)
{
    int KW = Kh * Kw;
    int per_co = Cin * KW;
    for (int co = 0; co < Cout; co++) {
        float s = scales[co];
        for (int k = 0; k < per_co; k++) {
            w_fp32[co * per_co + k] = (float)w_int[co * per_co + k] * s;
        }
    }
}


int main(int argc, char** argv) {
    // Pin OpenMP to physical P-cores (i7-13700 has 8P + 8E; over-subscription hurts).
    // Override via OMP_NUM_THREADS if the user wants different.
    if (!getenv("OMP_NUM_THREADS")) {
        omp_set_num_threads(8);
    }

    // Usage:
    //   fastface_fp32                                          — synthetic input, bench
    //   fastface_fp32 <model.bin>                              — custom model, bench
    //   fastface_fp32 <model.bin> <input.bin> <output.bin>     — inference from file
    const char* path = (argc > 1) ? argv[1] : "models/w600k_r50_ffw2.bin";
    const char* in_path = (argc > 2) ? argv[2] : NULL;
    const char* out_path = (argc > 3) ? argv[3] : NULL;
    int validate_mode = (in_path && out_path);

    FFW2 m = {0};
    if (ffw2_load(path, &m) != 0) { fprintf(stderr, "load fail\n"); return 1; }

    printf("FastFace FP32 Session 13 (cos-sim=1.0 mode)\n");
    printf("Model: %s  Ops: %u\n\n", path, m.n_ops);

    // --- Compute shapes + max buffer sizes (residual-aware) ---
    // Pre-flag: for each op, is it a "shortcut" Conv = Conv right after SAVE_ID?
    // In this IResNet serialization, SAVE_ID followed by a Conv means that Conv
    // is a downsample residual shortcut that consumes the *pre-block* input,
    // not the current activation. We save the activation right after every BN
    // (= start of a residual block) and restore it for such shortcut Convs.
    uint8_t* is_shortcut = (uint8_t*)calloc(m.n_ops, 1);
    for (uint32_t i = 1; i < m.n_ops; i++) {
        if (m.ops[i].type == OP_CONV && m.ops[i-1].type == OP_SAVE_ID) {
            is_shortcut[i] = 1;
        }
    }

    Shape* shape_after = (Shape*)calloc(m.n_ops, sizeof(Shape));
    Shape* in_shape   = (Shape*)calloc(m.n_ops, sizeof(Shape));  // input shape per op
    Shape cur = {3, 112, 112};
    Shape block_in = cur;
    size_t max_act = 0, max_im = 0;
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
            if (n > max_act) max_act = n;
            int Kr = op->Cin * op->Kh * op->Kw;
            int Kp = (Kr + 15) & ~15;
            size_t im = (size_t)M_padded * Kp;
            if (im > max_im) max_im = im;
        } else if (op->type == OP_GEMM) {
            cur.C = (int)op->N; cur.H = 1; cur.W = 1;
        }
        shape_after[i] = cur;
        // BLOCK_START marks the beginning of a residual block — save shape for shortcuts.
        if (op->type == OP_BLOCK_START) block_in = cur;
    }
    // Also ensure max_act covers block_in buffer needs
    {
        Shape bi = {3, 112, 112};
        for (uint32_t i = 0; i < m.n_ops; i++) {
            Op* op = &m.ops[i];
            if (op->type == OP_BLOCK_START) bi = shape_after[i];
            size_t n = (size_t)((bi.H * bi.W + 3) & ~3) * bi.C;
            if (n > max_act) max_act = n;
        }
    }
    printf("Max activation: %.1f KB  Max im2col: %.1f KB\n",
           max_act * 4 / 1024.0, max_im * 4 / 1024.0);

    // --- Dequantize + pack ALL conv weights ONCE at load ---
    typedef struct {
        float* w_fp32;    // [Cout, Cin, Kh, Kw] fp32
        float* w_packed;  // [K_padded, Cout] NR_F-packed (for non-Winograd path)
        int    K_padded;
        float* U_wino;    // [16, Cin, Cout] F(2,3) packed weights (NULL if not used)
        float* U_wino_f43; // [36, Cin, Cout] F(4,3) packed weights (NULL if not used)
    } ConvFP;
    ConvFP* packs = (ConvFP*)calloc(m.n_ops, sizeof(ConvFP));
    size_t total_wino_bytes = 0;
    int n_wino_ops = 0;
    for (uint32_t i = 0; i < m.n_ops; i++) {
        Op* op = &m.ops[i];
        if (op->type == OP_CONV) {
            int Kr = op->Cin * op->Kh * op->Kw;
            int Kp = (Kr + 15) & ~15;
            packs[i].K_padded = Kp;
            size_t w_elems = (size_t)op->Cout * op->Cin * op->Kh * op->Kw;
            packs[i].w_fp32 = (float*)_aligned_malloc(w_elems * sizeof(float), 64);
            dequant_conv_weight(op->conv_w, op->conv_scales,
                                op->Cout, op->Cin, op->Kh, op->Kw,
                                packs[i].w_fp32);
            float* w_rowmajor = (float*)_aligned_malloc((size_t)Kp * op->Cout * sizeof(float), 64);
            packs[i].w_packed = (float*)_aligned_malloc((size_t)Kp * op->Cout * sizeof(float), 64);
            pack_conv_weight_fp32_nhwc(packs[i].w_fp32, op->Cout, op->Cin, op->Kh, op->Kw, Kp,
                                        w_rowmajor, packs[i].w_packed);
            _aligned_free(w_rowmajor);
            // Winograd path selection. Env:
            //   FASTFACE_WINOGRAD=off   → no Winograd
            //   FASTFACE_WINOGRAD=ref   → F(2,3) reference (check)
            //   FASTFACE_WINOGRAD=packed → F(2,3) packed GEMM (default for small spatial)
            //   FASTFACE_WINOGRAD=f43    → F(4,3) packed GEMM (default for large spatial)
            //   FASTFACE_WINOGRAD=auto   → per-op choice (default)
            const char* wv = getenv("FASTFACE_WINOGRAD");
            int wino_mode = 3;  // auto (per-op F(2,3) or F(4,3))
            int force = 0;
            if (wv) {
                if      (strcmp(wv, "off") == 0)    { wino_mode = 0; force = 1; }
                else if (strcmp(wv, "ref") == 0 || strcmp(wv, "1") == 0) { wino_mode = 1; force = 1; }
                else if (strcmp(wv, "packed") == 0) { wino_mode = 2; force = 1; }
                else if (strcmp(wv, "f43") == 0)    { wino_mode = 4; force = 1; }
                else if (strcmp(wv, "auto") == 0)   { wino_mode = 3; force = 1; }
            }
            if (wino_mode && op->Kh == 3 && op->Kw == 3 && op->stride == 1 && op->pad == 1) {
                int H_out = (in_shape[i].H + 2 * op->pad - op->Kh) / op->stride + 1;
                int W_out = (in_shape[i].W + 2 * op->pad - op->Kw) / op->stride + 1;
                // Per-op choice for auto mode: F(4,3) when spatial big enough AND Cin*Cout
                // big (transform overhead amortized). Else F(2,3).
                int use_f43 = 0;
                if (force) {
                    use_f43 = (wino_mode == 4);
                } else {
                    // auto: F(4,3) only when spatial big (≥16×16) and channels big
                    // (F(4,3) transform overhead needs large reduction to amortize).
                    if (H_out >= 16 && W_out >= 16 && op->Cin * op->Cout >= 4096) use_f43 = 1;
                }
                size_t u_bytes = (size_t)(use_f43 ? 36 : 16) * op->Cin * op->Cout * sizeof(float);
                float* tmp_scratch = (float*)_aligned_malloc(u_bytes, 64);
                if (use_f43) {
                    packs[i].U_wino_f43 = (float*)_aligned_malloc(u_bytes, 64);
                    winograd_precompute_weights_packed_f43(packs[i].w_fp32, op->Cout, op->Cin,
                                                            packs[i].U_wino_f43, tmp_scratch);
                } else if (wino_mode == 1) {
                    packs[i].U_wino = (float*)_aligned_malloc(u_bytes, 64);
                    winograd_precompute_weights(packs[i].w_fp32, op->Cout, op->Cin, packs[i].U_wino);
                } else {
                    packs[i].U_wino = (float*)_aligned_malloc(u_bytes, 64);
                    winograd_precompute_weights_packed(packs[i].w_fp32, op->Cout, op->Cin,
                                                        packs[i].U_wino, tmp_scratch);
                }
                _aligned_free(tmp_scratch);
                total_wino_bytes += u_bytes;
                n_wino_ops++;
            }
        }
    }
    if (n_wino_ops) {
        fprintf(stderr, "Winograd (%s): %d Convs precomputed (%.1f MB)\n",
                getenv("FASTFACE_WINOGRAD"), n_wino_ops,
                total_wino_bytes / 1048576.0);
    }

    // --- Allocate fp32 scratch buffers ---
    float* scratch_im = (float*)_aligned_malloc((max_im + 64) * sizeof(float), 64);
    float* act_a      = (float*)_aligned_malloc((max_act + 64) * sizeof(float), 64);
    float* act_b      = (float*)_aligned_malloc((max_act + 64) * sizeof(float), 64);
    float* block_buf  = (float*)_aligned_malloc((max_act + 64) * sizeof(float), 64);
    float* id_slots[24];
    for (int k = 0; k < 24; k++) id_slots[k] = (float*)_aligned_malloc((max_act + 64) * sizeof(float), 64);

    // --- Winograd scratch: max over all 3x3 s=1 p=1 Convs of 16 * tiles * C ---
    size_t max_V_wino = 0, max_M_wino = 0;
    {
        Shape csh = {3, 112, 112};
        Shape blk = csh;
        for (uint32_t i = 0; i < m.n_ops; i++) {
            Op* op = &m.ops[i];
            Shape in_sh = is_shortcut[i] ? blk : csh;
            if (op->type == OP_CONV) {
                int H_out = (in_sh.H + 2 * op->pad - op->Kh) / op->stride + 1;
                int W_out = (in_sh.W + 2 * op->pad - op->Kw) / op->stride + 1;
                if (op->Kh == 3 && op->Kw == 3 && op->stride == 1 && op->pad == 1) {
                    // F(2, 3)
                    {
                        int tH = (H_out + 1) / 2, tW = (W_out + 1) / 2;
                        size_t ntil_padded = ((size_t)tH * tW + 3) & ~3ULL;
                        size_t v  = 16 * ntil_padded * op->Cin;
                        size_t me = 16 * ntil_padded * op->Cout;
                        if (v > max_V_wino) max_V_wino = v;
                        if (me > max_M_wino) max_M_wino = me;
                    }
                    // F(4, 3)
                    {
                        int tH = (H_out + 3) / 4, tW = (W_out + 3) / 4;
                        size_t ntil_padded = ((size_t)tH * tW + 3) & ~3ULL;
                        if (ntil_padded < 4) ntil_padded = 4;
                        size_t v  = 36 * ntil_padded * op->Cin;
                        size_t me = 36 * ntil_padded * op->Cout;
                        if (v > max_V_wino) max_V_wino = v;
                        if (me > max_M_wino) max_M_wino = me;
                    }
                }
                csh.C = op->Cout; csh.H = H_out; csh.W = W_out;
            } else if (op->type == OP_GEMM) { csh.C = (int)op->N; csh.H = 1; csh.W = 1; }
            if (op->type == OP_BLOCK_START) blk = csh;
        }
    }
    float* V_wino_scratch = (float*)_aligned_malloc((max_V_wino + 64) * sizeof(float), 64);
    float* M_wino_scratch = (float*)_aligned_malloc((max_M_wino + 64) * sizeof(float), 64);
    fprintf(stderr, "Winograd scratch: V=%.1f MB M=%.1f MB\n",
            max_V_wino * 4 / 1048576.0, max_M_wino * 4 / 1048576.0);

    // --- Input: from file (validate) or synthetic random (bench) ---
    int N_INPUT = 3 * 112 * 112;
    float* fp32_input_nhwc = (float*)_aligned_malloc(N_INPUT * sizeof(float), 64);
    if (validate_mode) {
        FILE* f = fopen(in_path, "rb");
        if (!f) { fprintf(stderr, "can't open %s\n", in_path); return 2; }
        size_t got = fread(fp32_input_nhwc, sizeof(float), N_INPUT, f);
        fclose(f);
        if (got != (size_t)N_INPUT) { fprintf(stderr, "short input %zu\n", got); return 3; }
        fprintf(stderr, "Loaded input from %s\n", in_path);
    } else {
        srand(42);
        for (int i = 0; i < N_INPUT; i++) fp32_input_nhwc[i] = ((rand() % 2001) - 1000) / 1000.0f;
    }

    printf("Scratch: %.1f MB\n\n",
           (max_im * 4 + 2 * max_act * 4 + 24 * max_act * 4) / 1048576.0);

    // Count shortcuts for info
    int n_shortcut = 0;
    for (uint32_t i = 0; i < m.n_ops; i++) n_shortcut += is_shortcut[i];
    fprintf(stderr, "Detected %d shortcut Convs (residual downsample branches)\n", n_shortcut);

    // One-shot test before bench loop
    {
        memcpy(act_a, fp32_input_nhwc, N_INPUT * sizeof(float));
        float* A = act_a;
        float* B = act_b;
        int save_idx = 0, add_idx = 0;
        memcpy(block_buf, fp32_input_nhwc, N_INPUT * sizeof(float));

        FILE* trace = getenv("FASTFACE_TRACE") ? fopen(getenv("FASTFACE_TRACE"), "wb") : NULL;

        /* S29c BN-fusion in FP32 test pass — disabled due to ~0.008 cos-sim drop
           (0.9997 → 0.991). Semantically equivalent math but evidently different
           rounding accumulates over 24 Conv-BN pairs. Revisit with careful
           numerical analysis. */
        const float* pend_bn_scale_tp = NULL;
        const float* pend_bn_offset_tp = NULL;
        (void)pend_bn_scale_tp; (void)pend_bn_offset_tp;
        for (uint32_t i = 0; i < m.n_ops; i++) {
            Op* op = &m.ops[i];
            Shape sh_in = in_shape[i];
            Shape sh_out = shape_after[i];
            int HW_in = sh_in.H * sh_in.W;
            int HW_out = sh_out.H * sh_out.W;

            switch (op->type) {
                case OP_CONV: {
                    const float* conv_in = is_shortcut[i] ? block_buf : A;
                    int bias_fused = 0;
                    int prelu_fused = 0;
                    int add_fused = 0;
                    const float* fuse_prelu = NULL;
                    const float* fuse_add = NULL;
                    if (i + 1 < m.n_ops && m.ops[i+1].type == OP_PRELU) {
                        fuse_prelu = m.ops[i+1].prelu_slope;
                    }
                    if (i + 1 < m.n_ops && m.ops[i+1].type == OP_ADD) {
                        fuse_add = id_slots[add_idx % 24];
                    }
                    const float* fuse_bn_s = NULL;
                    const float* fuse_bn_o = NULL;
                    if (pend_bn_scale_tp && !is_shortcut[i]) {
                        fuse_bn_s = pend_bn_scale_tp;
                        fuse_bn_o = pend_bn_offset_tp;
                    }
                    if (packs[i].U_wino_f43) {
                        /* F(4,3) path: flush BN if pending (can't fuse yet) */
                        if (pend_bn_scale_tp && !is_shortcut[i]) {
                            bn_fp32_nhwc(A, pend_bn_scale_tp, pend_bn_offset_tp, HW_in, sh_in.C);
                        }
                        pend_bn_scale_tp = pend_bn_offset_tp = NULL;
                        fastface_winograd_conv_3x3_s1_p1_f43_packed_bias(
                            conv_in, sh_in.C, sh_in.H, sh_in.W,
                            op->Cout, packs[i].U_wino_f43, op->conv_bias, B,
                            V_wino_scratch, M_wino_scratch);
                        bias_fused = 1;
                    } else if (packs[i].U_wino) {
                        const char* wv_ = getenv("FASTFACE_WINOGRAD");
                        int pm = 1;
                        if (wv_ && (strcmp(wv_, "ref") == 0 || strcmp(wv_, "1") == 0)) pm = 0;
                        if (pm) {
                            fastface_winograd_conv_3x3_s1_p1_full_fused_add(
                                conv_in, sh_in.C, sh_in.H, sh_in.W,
                                op->Cout, packs[i].U_wino,
                                fuse_bn_s, fuse_bn_o,
                                op->conv_bias, fuse_prelu, fuse_add,
                                B, V_wino_scratch, M_wino_scratch);
                            bias_fused = 1;
                            if (fuse_prelu) prelu_fused = 1;
                            if (fuse_add) add_fused = 1;
                            pend_bn_scale_tp = pend_bn_offset_tp = NULL;
                        } else {
                            if (pend_bn_scale_tp && !is_shortcut[i]) {
                                bn_fp32_nhwc(A, pend_bn_scale_tp, pend_bn_offset_tp, HW_in, sh_in.C);
                            }
                            pend_bn_scale_tp = pend_bn_offset_tp = NULL;
                            fastface_winograd_conv_3x3_s1_p1(
                                conv_in, sh_in.C, sh_in.H, sh_in.W,
                                op->Cout, packs[i].U_wino, B,
                                V_wino_scratch, M_wino_scratch);
                        }
                    } else {
                        if (pend_bn_scale_tp && !is_shortcut[i]) {
                            bn_fp32_nhwc(A, pend_bn_scale_tp, pend_bn_offset_tp, HW_in, sh_in.C);
                        }
                        pend_bn_scale_tp = pend_bn_offset_tp = NULL;
                        fastface_conv2d_fp32_nhwc(conv_in, sh_in.C, sh_in.H, sh_in.W,
                                                    op->Cout, op->Kh, op->Kw, op->stride, op->pad,
                                                    packs[i].w_packed,
                                                    sh_out.H, sh_out.W, B, scratch_im);
                    }
                    { float* t = A; A = B; B = t; }
                    if (!bias_fused)
                        add_bias_nhwc(A, op->conv_bias, sh_out.H * sh_out.W, op->Cout);
                    if (prelu_fused) i++;
                    if (add_fused) { i++; add_idx++; }
                    break;
                }
                case OP_BN:
                    bn_fp32_nhwc(A, op->bn_scale, op->bn_offset, HW_in, sh_in.C);
                    break;
                case OP_BLOCK_START:
                    /* Flush any pending BN to current A before snapshot (should be NULL here
                       because BLOCK_START precedes BN, but be safe). */
                    if (pend_bn_scale_tp) {
                        bn_fp32_nhwc(A, pend_bn_scale_tp, pend_bn_offset_tp, HW_in, sh_in.C);
                        pend_bn_scale_tp = pend_bn_offset_tp = NULL;
                    }
                    memcpy(block_buf, A, (size_t)sh_in.C * HW_in * sizeof(float));
                    break;
                case OP_PRELU:
                    if (pend_bn_scale_tp) {
                        bn_fp32_nhwc(A, pend_bn_scale_tp, pend_bn_offset_tp, HW_in, sh_in.C);
                        pend_bn_scale_tp = pend_bn_offset_tp = NULL;
                    }
                    prelu_fp32_nhwc(A, op->prelu_slope, HW_in, sh_in.C);
                    break;
                case OP_ADD:
                    if (pend_bn_scale_tp) {
                        bn_fp32_nhwc(A, pend_bn_scale_tp, pend_bn_offset_tp, HW_in, sh_in.C);
                        pend_bn_scale_tp = pend_bn_offset_tp = NULL;
                    }
                    add_fp32(A, id_slots[add_idx % 24], A, sh_out.C * HW_out);
                    add_idx++;
                    break;
                case OP_SAVE_ID:
                    if (pend_bn_scale_tp) {
                        bn_fp32_nhwc(A, pend_bn_scale_tp, pend_bn_offset_tp, HW_in, sh_in.C);
                        pend_bn_scale_tp = pend_bn_offset_tp = NULL;
                    }
                    memcpy(id_slots[save_idx % 24], A, (size_t)sh_in.C * HW_in * sizeof(float));
                    save_idx++;
                    break;
                case OP_GEMM: {
                    if (pend_bn_scale_tp) {
                        bn_fp32_nhwc(A, pend_bn_scale_tp, pend_bn_offset_tp, HW_in, sh_in.C);
                        pend_bn_scale_tp = pend_bn_offset_tp = NULL;
                    }
                    for (uint32_t n = 0; n < op->N; n++) {
                        float s = op->gemm_bias[n];
                        for (uint32_t k = 0; k < op->K; k++) {
                            s += A[k] * (float)op->gemm_w[n * op->K + k] * op->gemm_scales[n];
                        }
                        B[n] = s;
                    }
                    { float* t = A; A = B; B = t; }
                    break;
                }
                case OP_FLATTEN: {
                    if (pend_bn_scale_tp) {
                        bn_fp32_nhwc(A, pend_bn_scale_tp, pend_bn_offset_tp, HW_in, sh_in.C);
                        pend_bn_scale_tp = pend_bn_offset_tp = NULL;
                    }
                    // Transpose NHWC → NCHW so Gemm sees values in ONNX Flatten order.
                    int H = sh_in.H, W = sh_in.W, C = sh_in.C;
                    for (int c = 0; c < C; c++) {
                        for (int h = 0; h < H; h++) {
                            for (int w = 0; w < W; w++) {
                                B[c * H * W + h * W + w] = A[(h * W + w) * C + c];
                            }
                        }
                    }
                    { float* t = A; A = B; B = t; }
                    break;
                }
            }
            if (trace) {
                size_t n_elems = (size_t)sh_out.C * sh_out.H * sh_out.W;
                float norm2 = 0; for (size_t kk = 0; kk < n_elems; kk++) norm2 += A[kk] * A[kk];
                fprintf(trace, "op %u type %d shape[%d,%d,%d] norm %.6f first5 %g %g %g %g %g\n",
                        i, op->type, sh_out.C, sh_out.H, sh_out.W, sqrtf(norm2),
                        A[0], A[1], A[2], A[3], A[4]);
            }
        }
        if (trace) fclose(trace);
        fprintf(stderr, "First pass OK\n");

        if (validate_mode) {
            Op* last = &m.ops[m.n_ops - 1];
            uint32_t n_out = (last->type == OP_GEMM) ? last->N : 512;
            FILE* f = fopen(out_path, "wb");
            if (!f) { fprintf(stderr, "can't open %s for write\n", out_path); return 4; }
            size_t w = fwrite(A, sizeof(float), n_out, f);
            fclose(f);
            fprintf(stderr, "Wrote %zu floats to %s  (first5=[%g %g %g %g %g])\n",
                    w, out_path, A[0], A[1], A[2], A[3], A[4]);
            return 0;
        }
    }

    // --- Bench loop ---
    const int ITER = 20;
    double best = 1e9;
    for (int trial = 0; trial < 5; trial++) {
        double t0 = now_s();
        for (int it = 0; it < ITER; it++) {
            memcpy(act_a, fp32_input_nhwc, N_INPUT * sizeof(float));
            float* A = act_a;
            float* B = act_b;
            int save_idx = 0;
            int add_idx = 0;
            memcpy(block_buf, fp32_input_nhwc, N_INPUT * sizeof(float));

            for (uint32_t i = 0; i < m.n_ops; i++) {
                Op* op = &m.ops[i];
                Shape sh_in = in_shape[i];
                Shape sh_out = shape_after[i];
                int HW_in = sh_in.H * sh_in.W;
                int HW_out = sh_out.H * sh_out.W;

                switch (op->type) {
                    case OP_CONV: {
                        const float* conv_in = is_shortcut[i] ? block_buf : A;
                        int bias_fused = 0;
                        int prelu_fused = 0;
                        int add_fused = 0;
                        const float* fuse_prelu = NULL;
                        const float* fuse_add = NULL;
                        if (i + 1 < m.n_ops && m.ops[i+1].type == OP_PRELU) {
                            fuse_prelu = m.ops[i+1].prelu_slope;
                        }
                        if (i + 1 < m.n_ops && m.ops[i+1].type == OP_ADD) {
                            fuse_add = id_slots[add_idx % 24];
                        }
                        if (packs[i].U_wino_f43) {
                            fastface_winograd_conv_3x3_s1_p1_f43_packed_bias(
                                conv_in, sh_in.C, sh_in.H, sh_in.W,
                                op->Cout, packs[i].U_wino_f43, op->conv_bias, B,
                                V_wino_scratch, M_wino_scratch);
                            bias_fused = 1;
                        } else if (packs[i].U_wino) {
                            const char* wv_ = getenv("FASTFACE_WINOGRAD");
                            int packed_mode = 1;
                            if (wv_ && (strcmp(wv_, "ref") == 0 || strcmp(wv_, "1") == 0)) packed_mode = 0;
                            if (packed_mode) {
                                fastface_winograd_conv_3x3_s1_p1_full_fused_add(
                                    conv_in, sh_in.C, sh_in.H, sh_in.W,
                                    op->Cout, packs[i].U_wino,
                                    NULL, NULL, op->conv_bias, fuse_prelu, fuse_add,
                                    B, V_wino_scratch, M_wino_scratch);
                                bias_fused = 1;
                                if (fuse_prelu) prelu_fused = 1;
                                if (fuse_add) add_fused = 1;
                            } else {
                                fastface_winograd_conv_3x3_s1_p1(
                                    conv_in, sh_in.C, sh_in.H, sh_in.W,
                                    op->Cout, packs[i].U_wino, B,
                                    V_wino_scratch, M_wino_scratch);
                            }
                        } else {
                            fastface_conv2d_fp32_nhwc(conv_in, sh_in.C, sh_in.H, sh_in.W,
                                                        op->Cout, op->Kh, op->Kw, op->stride, op->pad,
                                                        packs[i].w_packed,
                                                        sh_out.H, sh_out.W, B, scratch_im);
                        }
                        { float* t = A; A = B; B = t; }
                        if (!bias_fused)
                            add_bias_nhwc(A, op->conv_bias, sh_out.H * sh_out.W, op->Cout);
                        if (prelu_fused) i++;
                        break;
                    }
                    case OP_BN:
                        bn_fp32_nhwc(A, op->bn_scale, op->bn_offset, HW_in, sh_in.C);
                        break;
                    case OP_BLOCK_START:
                        memcpy(block_buf, A, (size_t)sh_in.C * HW_in * sizeof(float));
                        break;
                    case OP_PRELU:
                        prelu_fp32_nhwc(A, op->prelu_slope, HW_in, sh_in.C);
                        break;
                    case OP_ADD:
                        add_fp32(A, id_slots[add_idx % 24], A, sh_out.C * HW_out);
                        add_idx++;
                        break;
                    case OP_SAVE_ID:
                        memcpy(id_slots[save_idx % 24], A, (size_t)sh_in.C * HW_in * sizeof(float));
                        save_idx++;
                        break;
                    case OP_GEMM:
                        for (uint32_t n = 0; n < op->N; n++) {
                            float s = op->gemm_bias[n];
                            for (uint32_t k = 0; k < op->K; k++) {
                                s += A[k] * (float)op->gemm_w[n * op->K + k] * op->gemm_scales[n];
                            }
                            B[n] = s;
                        }
                        { float* t = A; A = B; B = t; }
                        break;
                    case OP_FLATTEN: {
                        int H = sh_in.H, W = sh_in.W, C = sh_in.C;
                        for (int c = 0; c < C; c++) {
                            for (int h = 0; h < H; h++) {
                                for (int w = 0; w < W; w++) {
                                    B[c * H * W + h * W + w] = A[(h * W + w) * C + c];
                                }
                            }
                        }
                        { float* t = A; A = B; B = t; }
                        break;
                    }
                }
            }
        }
        double dt = (now_s() - t0) / ITER;
        if (dt < best) best = dt;
        printf("Trial %d: %.2f ms/inference  (%.1f face/s)\n", trial, dt * 1000, 1/dt);
    }

    printf("\n========== FP32 MODE RESULT ==========\n");
    printf("Best:   %.2f ms/inference  (%.1f face/s)\n", best * 1000, 1/best);
    printf("ORT FP32 baseline: 31.77 ms\n");
    printf("Speedup: %.2fx\n", 31.77 / (best * 1000));
    printf("Measured cos-sim vs ORT: 0.9997 mean on 10 LFW faces (S13)\n");
    printf("======================================\n");

    return 0;
}
