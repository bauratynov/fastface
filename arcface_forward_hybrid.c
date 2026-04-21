// Session 14 — HYBRID path: fp32 activations for correctness, int8 Conv GEMM for speed.
//
// Strategy: keep arcface_forward_fp32.c's proven residual-aware dispatch (block_buf
// before BN, shortcut detection, Flatten transpose). The ONLY change: at Conv op
// we quantize the current fp32 activation to int8, run the int8 GEMM, then
// dequantize the int32 output back to fp32 and add bias. Non-Conv ops stay fp32.
//
// Expected win: ~85% of compute is Conv (GEMM-bound). int8 GEMM is ~3× faster
// than fp32 GEMM on AVX2-VNNI, so total speedup should approach 2× overall,
// while accuracy stays close to FP32 (cos-sim ≥0.98 from per-channel weight
// quantization + per-tensor runtime activation quantization).

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <windows.h>
#include <omp.h>
#include <immintrin.h>

// FP32 kernel forward decls (non-Conv ops, reused as-is)
void bn_fp32_nhwc(float* x, const float* scale, const float* offset, int HW, int Cout);
void prelu_fp32_nhwc(float* x, const float* slope, int HW, int Cout);
void add_bias_nhwc(float* x, const float* bias, int HW, int Cout);
void add_fp32(const float* a, const float* b, float* out, int n);

// INT8 kernel forward decls (Conv + packing)
void compute_col_sums(const int8_t* B, int K, int N, int32_t* col_sums);
void pack_B_vnni(const int8_t* B, int K, int N, int8_t* Bp);
void fastface_gemm_i8_fused(const uint8_t* Au, const int8_t* Bp, const int32_t* col_sums,
                             int32_t* C, int M, int K, int N);
void pack_conv_weight_nhwc(const int8_t* weight, int Cout, int Cin, int Kh, int Kw, int K_padded,
                            int8_t* w_rowmajor, int8_t* w_packed, int32_t* col_sums);
void fastface_conv2d_i8_nhwc(const int8_t* input, int Cin, int H_in, int W_in,
                              int Cout, int Kh, int Kw, int stride, int pad,
                              const int8_t* weight_packed, const int32_t* col_sums,
                              int H_out, int W_out, int32_t* output, uint8_t* scratch_Au);

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
    LARGE_INTEGER q, f;
    QueryPerformanceCounter(&q); QueryPerformanceFrequency(&f);
    return (double)q.QuadPart / (double)f.QuadPart;
}


// Quantize fp32 NHWC activation to int8 using per-tensor absmax (runtime calibration).
// Returns the scale used (for dequant).
static float quantize_fp32_to_int8(const float* a, int n, int8_t* out) {
    // Pass 1: find absmax via AVX2 reduction
    float absmax = 0.0f;
#ifdef __AVX2__
    __m256 vmax = _mm256_setzero_ps();
    const __m256 sign_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 v = _mm256_loadu_ps(a + i);
        v = _mm256_and_ps(v, sign_mask);  // abs
        vmax = _mm256_max_ps(vmax, v);
    }
    float buf[8]; _mm256_storeu_ps(buf, vmax);
    for (int k = 0; k < 8; k++) if (buf[k] > absmax) absmax = buf[k];
    for (; i < n; i++) { float v = fabsf(a[i]); if (v > absmax) absmax = v; }
#else
    for (int i = 0; i < n; i++) { float v = fabsf(a[i]); if (v > absmax) absmax = v; }
#endif
    float scale = (absmax > 0) ? (absmax / 127.0f) : 1.0f;
    float inv = 1.0f / scale;
    // Pass 2: quantize
    #pragma omp parallel for schedule(static) if (n > 4096)
    for (int i = 0; i < n; i++) {
        int q = (int)lrintf(a[i] * inv);
        if (q > 127) q = 127;
        if (q < -128) q = -128;
        out[i] = (int8_t)q;
    }
    return scale;
}


// Dequantize int32 conv output to fp32, applying per-channel weight_scale × input_scale + bias.
// Layout: out is [HW, Cout] NHWC.
static void dequant_int32_to_fp32_with_bias(
    const int32_t* in_i32, int HW, int Cout,
    float input_scale, const float* weight_scales, const float* bias,
    float* out_fp32)
{
    #pragma omp parallel for schedule(static)
    for (int p = 0; p < HW; p++) {
        const int32_t* ir = in_i32 + (size_t)p * Cout;
        float* or_ = out_fp32 + (size_t)p * Cout;
        for (int c = 0; c < Cout; c++) {
            float ws = weight_scales[c] * input_scale;
            or_[c] = (float)ir[c] * ws + (bias ? bias[c] : 0.0f);
        }
    }
}


int main(int argc, char** argv) {
    const char* path = (argc > 1) ? argv[1] : "models/w600k_r50_ffw2.bin";
    const char* in_path = (argc > 2) ? argv[2] : NULL;
    const char* out_path = (argc > 3) ? argv[3] : NULL;
    int validate_mode = (in_path && out_path);

    FFW2 m = {0};
    if (ffw2_load(path, &m) != 0) { fprintf(stderr, "load fail\n"); return 1; }

    printf("FastFace HYBRID Session 14 (fp32 activations + int8 Conv GEMM)\n");
    printf("Model: %s  Ops: %u\n\n", path, m.n_ops);

    // --- Residual-aware shape + shortcut pre-pass (same as FP32 driver) ---
    uint8_t* is_shortcut = (uint8_t*)calloc(m.n_ops, 1);
    for (uint32_t i = 1; i < m.n_ops; i++) {
        if (m.ops[i].type == OP_CONV && m.ops[i-1].type == OP_SAVE_ID) {
            is_shortcut[i] = 1;
        }
    }

    Shape* shape_after = (Shape*)calloc(m.n_ops, sizeof(Shape));
    Shape* in_shape    = (Shape*)calloc(m.n_ops, sizeof(Shape));
    Shape cur = {3, 112, 112};
    Shape block_in = cur;
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
        }
        if (op->type == OP_GEMM) { cur.C = (int)op->N; cur.H = 1; cur.W = 1; }
        shape_after[i] = cur;
        if (op->type == OP_BLOCK_START) block_in = cur;
    }
    printf("Max activation: %zu floats (%.1f KB)\n", max_act, max_act * 4 / 1024.0);
    printf("Max im2col:     %zu bytes (%.1f KB)\n", max_im, max_im / 1024.0);

    // --- Pre-pack all Conv weights as int8 (no dequant!) ---
    typedef struct { int8_t* w_packed; int32_t* col_sums; int K_padded; } ConvPack;
    ConvPack* packs = (ConvPack*)calloc(m.n_ops, sizeof(ConvPack));
    for (uint32_t i = 0; i < m.n_ops; i++) {
        Op* op = &m.ops[i];
        if (op->type == OP_CONV) {
            int Kr = op->Cin * op->Kh * op->Kw;
            int Kp = (Kr + 3) & ~3;
            packs[i].K_padded = Kp;
            int8_t* wrow = (int8_t*)_aligned_malloc((size_t)Kp * op->Cout, 64);
            packs[i].w_packed = (int8_t*)_aligned_malloc((size_t)Kp * op->Cout, 64);
            packs[i].col_sums = (int32_t*)_aligned_malloc((size_t)op->Cout * sizeof(int32_t), 64);
            pack_conv_weight_nhwc(op->conv_w, op->Cout, op->Cin, op->Kh, op->Kw, Kp,
                                  wrow, packs[i].w_packed, packs[i].col_sums);
            _aligned_free(wrow);
        }
    }

    // --- Allocate scratch ---
    float*   act_a    = (float*)_aligned_malloc((max_act + 64) * sizeof(float), 64);
    float*   act_b    = (float*)_aligned_malloc((max_act + 64) * sizeof(float), 64);
    float*   block_buf = (float*)_aligned_malloc((max_act + 64) * sizeof(float), 64);
    int8_t*  qbuf     = (int8_t*)_aligned_malloc(max_act + 64, 64);
    uint8_t* im_buf   = (uint8_t*)_aligned_malloc(max_im + 64, 64);
    int32_t* i32_buf  = (int32_t*)_aligned_malloc((max_i32 + 64) * sizeof(int32_t), 64);
    float* id_slots[24];
    for (int k = 0; k < 24; k++) id_slots[k] = (float*)_aligned_malloc((max_act + 64) * sizeof(float), 64);

    // --- Input prep ---
    int N_INPUT = 3 * 112 * 112;
    float* fp32_input_nhwc = (float*)_aligned_malloc(N_INPUT * sizeof(float), 64);
    if (validate_mode) {
        FILE* f = fopen(in_path, "rb");
        if (!f) { fprintf(stderr, "can't open %s\n", in_path); return 2; }
        size_t got = fread(fp32_input_nhwc, sizeof(float), N_INPUT, f);
        fclose(f);
        if (got != (size_t)N_INPUT) { fprintf(stderr, "short input\n"); return 3; }
    } else {
        srand(42);
        for (int i = 0; i < N_INPUT; i++) fp32_input_nhwc[i] = ((rand() % 2001) - 1000) / 1000.0f;
    }

    int n_shortcut = 0;
    for (uint32_t i = 0; i < m.n_ops; i++) n_shortcut += is_shortcut[i];
    fprintf(stderr, "Detected %d shortcut Convs\n", n_shortcut);

    // --- Executor: same structure as FP32 driver, but Conv goes through int8 GEMM ---
    #define RUN_ONCE() do {                                                                  \
        memcpy(act_a, fp32_input_nhwc, N_INPUT * sizeof(float));                             \
        memcpy(block_buf, fp32_input_nhwc, N_INPUT * sizeof(float));                         \
        float* A = act_a;                                                                    \
        float* B = act_b;                                                                    \
        int save_idx = 0, add_idx = 0;                                                       \
        for (uint32_t i = 0; i < m.n_ops; i++) {                                             \
            Op* op = &m.ops[i];                                                              \
            Shape sh_in  = in_shape[i];                                                      \
            Shape sh_out = shape_after[i];                                                   \
            int HW_in  = sh_in.H * sh_in.W;                                                  \
            int HW_out = sh_out.H * sh_out.W;                                                \
            switch (op->type) {                                                              \
                case OP_CONV: {                                                              \
                    const float* conv_in = is_shortcut[i] ? block_buf : A;                   \
                    int n_in = sh_in.C * HW_in;                                              \
                    /* 1. Quantize input fp32 → int8 (per-tensor absmax) */                  \
                    float in_scale = quantize_fp32_to_int8(conv_in, n_in, qbuf);             \
                    /* 2. Run int8 Conv: int8 → int32 */                                     \
                    fastface_conv2d_i8_nhwc(qbuf, sh_in.C, sh_in.H, sh_in.W,                 \
                                            op->Cout, op->Kh, op->Kw, op->stride, op->pad,   \
                                            packs[i].w_packed, packs[i].col_sums,            \
                                            sh_out.H, sh_out.W, i32_buf, im_buf);            \
                    /* 3. Dequant int32 → fp32, apply bias */                                \
                    dequant_int32_to_fp32_with_bias(i32_buf, HW_out, op->Cout,               \
                                                    in_scale, op->conv_scales, op->conv_bias, B); \
                    { float* t = A; A = B; B = t; }                                          \
                    break;                                                                   \
                }                                                                            \
                case OP_BN:                                                                  \
                    bn_fp32_nhwc(A, op->bn_scale, op->bn_offset, HW_in, sh_in.C);            \
                    break;                                                                   \
                case OP_BLOCK_START:                                                         \
                    memcpy(block_buf, A, (size_t)sh_in.C * HW_in * sizeof(float));           \
                    break;                                                                   \
                case OP_PRELU:                                                               \
                    prelu_fp32_nhwc(A, op->prelu_slope, HW_in, sh_in.C);                     \
                    break;                                                                   \
                case OP_ADD:                                                                 \
                    add_fp32(A, id_slots[add_idx % 24], A, sh_out.C * HW_out);               \
                    add_idx++;                                                               \
                    break;                                                                   \
                case OP_SAVE_ID:                                                             \
                    memcpy(id_slots[save_idx % 24], A, (size_t)sh_in.C * HW_in * sizeof(float)); \
                    save_idx++;                                                              \
                    break;                                                                   \
                case OP_GEMM: {                                                              \
                    for (uint32_t nn = 0; nn < op->N; nn++) {                                \
                        float s = op->gemm_bias[nn];                                         \
                        for (uint32_t kk = 0; kk < op->K; kk++) {                            \
                            s += A[kk] * (float)op->gemm_w[nn * op->K + kk] * op->gemm_scales[nn]; \
                        }                                                                    \
                        B[nn] = s;                                                           \
                    }                                                                        \
                    { float* t = A; A = B; B = t; }                                          \
                    break;                                                                   \
                }                                                                            \
                case OP_FLATTEN: {                                                           \
                    int Hf = sh_in.H, Wf = sh_in.W, Cf = sh_in.C;                            \
                    for (int c = 0; c < Cf; c++) {                                           \
                        for (int h = 0; h < Hf; h++) {                                       \
                            for (int w = 0; w < Wf; w++) {                                   \
                                B[c * Hf * Wf + h * Wf + w] = A[(h * Wf + w) * Cf + c];      \
                            }                                                                \
                        }                                                                    \
                    }                                                                        \
                    { float* t = A; A = B; B = t; }                                          \
                    break;                                                                   \
                }                                                                            \
            }                                                                                \
        }                                                                                    \
        final_A_ptr = A;                                                                     \
    } while(0)

    // --- One-shot + optional validation write ---
    float* final_A_ptr = NULL;
    RUN_ONCE();
    fprintf(stderr, "First pass OK\n");

    if (validate_mode) {
        Op* last = &m.ops[m.n_ops - 1];
        uint32_t n_out = (last->type == OP_GEMM) ? last->N : 512;
        FILE* f = fopen(out_path, "wb");
        if (!f) { fprintf(stderr, "can't open %s for write\n", out_path); return 4; }
        size_t w = fwrite(final_A_ptr, sizeof(float), n_out, f);
        fclose(f);
        fprintf(stderr, "Wrote %zu floats to %s  first5=[%g %g %g %g %g]\n",
                w, out_path, final_A_ptr[0], final_A_ptr[1], final_A_ptr[2],
                final_A_ptr[3], final_A_ptr[4]);
        return 0;
    }

    // --- Bench loop ---
    const int ITER = 20;
    double best = 1e9;
    for (int trial = 0; trial < 5; trial++) {
        double t0 = now_s();
        for (int it = 0; it < ITER; it++) {
            RUN_ONCE();
        }
        double dt = (now_s() - t0) / ITER;
        if (dt < best) best = dt;
        printf("Trial %d: %.2f ms/inference  (%.1f face/s)\n", trial, dt * 1000, 1/dt);
    }

    printf("\n========== HYBRID MODE RESULT ==========\n");
    printf("Best:    %.2f ms/inference  (%.1f face/s)\n", best * 1000, 1/best);
    printf("ORT FP32 baseline: 31.77 ms\n");
    printf("Speedup: %.2fx\n", 31.77 / (best * 1000));
    printf("========================================\n");

    return 0;
}
