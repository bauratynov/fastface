// Session 4 — Conv2d with fused im2col+pack + fused finalize + no output transpose.
// Output layout: [M, Cout] (NHWC-flat). Downstream next-conv2d takes this directly.
//
// Key changes from v1:
//   - Fused `im2col_pack_u8` — single pass reading NCHW input, writing u8-shifted NHWC-ish
//   - `fastface_gemm_i8_fused` — kernel does compensation inside microkernel epilogue
//   - No output transpose — direct write to [M, Cout]

#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
#include <immintrin.h>
#include <windows.h>

// Forward decls
void compute_col_sums(const int8_t* B, int K, int N, int32_t* col_sums);
void pack_B_vnni(const int8_t* B, int K, int N, int8_t* Bp);
void fastface_gemm_i8_fused(const uint8_t* Au, const int8_t* Bp, const int32_t* col_sums,
                             int32_t* C, int M, int K, int N);

// FUSED im2col + add-128 shift.
// Reads input NCHW int8, writes [H_out*W_out, K_padded] uint8 shifted (+128).
// Zero-padding (treated as signed zero = 0, which becomes u8 128 after shift).
static void im2col_pack_u8(
    const int8_t* input,
    int Cin, int H_in, int W_in,
    int Kh, int Kw, int stride, int pad,
    int H_out, int W_out, int K_padded,
    uint8_t* out)
{
    int K_real = Cin * Kh * Kw;
    #pragma omp parallel for schedule(static)
    for (int oh = 0; oh < H_out; oh++) {
        for (int ow = 0; ow < W_out; ow++) {
            int row = oh * W_out + ow;
            uint8_t* out_row = out + (size_t)row * K_padded;
            int idx = 0;
            for (int c = 0; c < Cin; c++) {
                for (int kh = 0; kh < Kh; kh++) {
                    for (int kw = 0; kw < Kw; kw++) {
                        int ih = oh * stride - pad + kh;
                        int iw = ow * stride - pad + kw;
                        int8_t v = 0;
                        if (ih >= 0 && ih < H_in && iw >= 0 && iw < W_in) {
                            v = input[(size_t)c * H_in * W_in + (size_t)ih * W_in + iw];
                        }
                        out_row[idx++] = (uint8_t)((int)v + 128);
                    }
                }
            }
            // Zero-padded tail (signed 0 → u8 128)
            for (; idx < K_padded; idx++) out_row[idx] = 128;
        }
    }
}

// One-shot weight prep at model load (same as before).
void pack_conv_weight_v2(
    const int8_t* weight, int Cout, int Cin, int Kh, int Kw, int K_padded,
    int8_t* weight_rowmajor, int8_t* weight_packed, int32_t* col_sums)
{
    int K_real = Cin * Kh * Kw;
    memset(weight_rowmajor, 0, (size_t)K_padded * Cout);
    for (int co = 0; co < Cout; co++) {
        for (int ci = 0; ci < Cin; ci++) {
            for (int kh = 0; kh < Kh; kh++) {
                for (int kw = 0; kw < Kw; kw++) {
                    int k = ci * Kh * Kw + kh * Kw + kw;
                    weight_rowmajor[(size_t)k * Cout + co] =
                        weight[(size_t)co * Cin * Kh * Kw + (size_t)ci * Kh * Kw + kh * Kw + kw];
                }
            }
        }
    }
    pack_B_vnni(weight_rowmajor, K_padded, Cout, weight_packed);
    compute_col_sums(weight_rowmajor, K_padded, Cout, col_sums);
    // Adjust col_sums for the fact that zero-pad rows in A contribute 128 each (since we shift input).
    // For pad rows: A_u = 128, A_signed = 0. So extra contribution = sum_over_pad_rows(128 * B[k,j]).
    // But those pad rows have B_k = 0 (we pad weight with zeros too), so extra = 0. Safe.
    // Also: im2col pad positions get u8 = 128 (from signed 0). Same compensation handles them.
}

// Full Conv2d v2: fused pipeline, output NHWC layout [M, Cout].
void fastface_conv2d_i8_v2(
    const int8_t*  input, int Cin, int H_in, int W_in,
    int Cout, int Kh, int Kw, int stride, int pad,
    const int8_t*  weight_packed, const int32_t* col_sums,
    int H_out, int W_out,
    int32_t*       output,         // [M, Cout] NHWC-flat — NO transpose back
    uint8_t*       scratch_Au)     // [M, K_padded]
{
    int K_real = Cin * Kh * Kw;
    int K_padded = (K_real + 3) & ~3;
    int M = H_out * W_out;

    im2col_pack_u8(input, Cin, H_in, W_in, Kh, Kw, stride, pad,
                   H_out, W_out, K_padded, scratch_Au);
    fastface_gemm_i8_fused(scratch_Au, weight_packed, col_sums, output, M, K_padded, Cout);
}

// ---------- Benchmark harness v2 ----------

static double now_s(void) {
    LARGE_INTEGER q, f;
    QueryPerformanceCounter(&q);
    QueryPerformanceFrequency(&f);
    return (double)q.QuadPart / (double)f.QuadPart;
}

typedef struct {
    const char* name;
    int Cin, H_in, W_in;
    int Cout, Kh, Kw, stride, pad;
} ConvShape;

static void bench_shape(const ConvShape* s) {
    int H_out = (s->H_in + 2 * s->pad - s->Kh) / s->stride + 1;
    int W_out = (s->W_in + 2 * s->pad - s->Kw) / s->stride + 1;
    int K_real = s->Cin * s->Kh * s->Kw;
    int K_padded = (K_real + 3) & ~3;
    int M = H_out * W_out;

    printf("  %s: input[%dx%dx%d]  weight[%dx%dx%dx%d] s=%d p=%d\n",
           s->name, s->Cin, s->H_in, s->W_in,
           s->Cout, s->Cin, s->Kh, s->Kw, s->stride, s->pad);
    printf("    output[%dx%dx%d]  M=%d K_padded=%d N=%d\n",
           s->Cout, H_out, W_out, M, K_padded, s->Cout);

    if (M % 4 != 0 || s->Cout % 8 != 0) {
        printf("    SKIP (alignment)\n\n");
        return;
    }

    size_t in_size = (size_t)s->Cin * s->H_in * s->W_in;
    size_t w_size = (size_t)s->Cout * s->Cin * s->Kh * s->Kw;
    int8_t*  input  = _aligned_malloc(in_size, 64);
    int8_t*  weight = _aligned_malloc(w_size, 64);
    int8_t*  w_rowmajor = _aligned_malloc((size_t)K_padded * s->Cout, 64);
    int8_t*  w_packed   = _aligned_malloc((size_t)K_padded * s->Cout, 64);
    int32_t* col_sums = _aligned_malloc((size_t)s->Cout * sizeof(int32_t), 64);
    int32_t* output = _aligned_malloc((size_t)M * s->Cout * sizeof(int32_t), 64);
    uint8_t* scratch_Au = _aligned_malloc((size_t)M * K_padded, 64);

    srand(42);
    for (size_t i = 0; i < in_size; i++) input[i] = (int8_t)((rand() & 0xFF) - 128);
    for (size_t i = 0; i < w_size; i++) weight[i] = (int8_t)((rand() & 0xFF) - 128);

    pack_conv_weight_v2(weight, s->Cout, s->Cin, s->Kh, s->Kw, K_padded,
                        w_rowmajor, w_packed, col_sums);

    for (int w = 0; w < 3; w++) {
        fastface_conv2d_i8_v2(input, s->Cin, s->H_in, s->W_in,
                              s->Cout, s->Kh, s->Kw, s->stride, s->pad,
                              w_packed, col_sums, H_out, W_out, output, scratch_Au);
    }

    const int N_ITER = 200;
    double t0 = now_s();
    for (int it = 0; it < N_ITER; it++) {
        fastface_conv2d_i8_v2(input, s->Cin, s->H_in, s->W_in,
                              s->Cout, s->Kh, s->Kw, s->stride, s->pad,
                              w_packed, col_sums, H_out, W_out, output, scratch_Au);
    }
    double t_conv = (now_s() - t0) / N_ITER;

    double gops = (2.0 * M * K_real * s->Cout) / 1e9;
    printf("    FastFace v2 conv2d: %.3f ms  (%.1f GOps/s)\n", t_conv * 1000, gops / t_conv);

    _aligned_free(input); _aligned_free(weight); _aligned_free(w_rowmajor); _aligned_free(w_packed);
    _aligned_free(col_sums); _aligned_free(output); _aligned_free(scratch_Au);
    printf("\n");
}

int main(void) {
    printf("FastFace Conv2d Session 4 (fused pipeline) benchmark\n");
    printf("i7-13700, 8 threads AVX-VNNI\n\n");

    ConvShape shapes[] = {
        {"Stem 7x7 s=2 p=3",   3, 112, 112, 64, 7, 7, 2, 3},
        {"Mid 3x3 s=2 p=1",    64, 56, 56, 128, 3, 3, 2, 1},
        {"Deep 1x1 s=1 p=0",   256, 14, 14, 256, 1, 1, 1, 0},
        {"Bottle 3x3 s=1 p=1", 128, 28, 28, 128, 3, 3, 1, 1},
    };

    for (int i = 0; i < 4; i++) bench_shape(&shapes[i]);
    return 0;
}
