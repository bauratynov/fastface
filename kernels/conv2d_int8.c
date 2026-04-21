// Session 3: Conv2d INT8 with im2col + our GEMM kernel.
// Full Conv2d pipeline: NCHW int8 input + weight [Cout, Cin, Kh, Kw] int8 →
// NCHW int8 output (via per-channel dequant + re-quant).
//
// For 1×1 conv we skip im2col and route directly to GEMM.

#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
#include <immintrin.h>
#include <windows.h>

// Forward decls from gemm_int8.c
void compute_col_sums(const int8_t* B, int K, int N, int32_t* col_sums);
void pack_A_to_u8(const int8_t* A, int M, int K, uint8_t* Au);
void pack_B_vnni(const int8_t* B, int K, int N, int8_t* Bp);
void fastface_gemm_i8(const uint8_t* Au, const int8_t* Bp, int32_t* C, int M, int K, int N);
void fastface_gemm_i8_finalize(int32_t* C, int M, int N, const int32_t* col_sums);

// im2col: input [Cin, H_in, W_in] int8 → output [H_out*W_out, Cin*Kh*Kw] int8.
// Padding with zero bytes (INT8 "zero" is 0, fine for signed int8; signed-A shift handles it).
// K_padded = next multiple of 4 ≥ Cin*Kh*Kw (for VNNI alignment).
static void im2col_nchw(
    const int8_t* input,   // [Cin, H_in, W_in]
    int Cin, int H_in, int W_in,
    int Kh, int Kw,
    int stride, int pad,
    int H_out, int W_out, int K_padded,
    int8_t* out)           // [H_out*W_out, K_padded]
{
    int K_real = Cin * Kh * Kw;
    #pragma omp parallel for schedule(static)
    for (int oh = 0; oh < H_out; oh++) {
        for (int ow = 0; ow < W_out; ow++) {
            int row = oh * W_out + ow;
            int8_t* out_row = out + (size_t)row * K_padded;
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
                        out_row[idx++] = v;
                    }
                }
            }
            // Zero-pad trailing K-positions to K_padded
            for (; idx < K_padded; idx++) out_row[idx] = 0;
        }
    }
}

// Reshape weight [Cout, Cin, Kh, Kw] int8 → [K_padded, Cout] int8 then VNNI-pack.
// Actual output is a pre-packed int8_t* ready for fastface_gemm_i8.
// Also compute col_sums once (for signed-A compensation).
void pack_conv_weight(
    const int8_t* weight,   // [Cout, Cin, Kh, Kw]
    int Cout, int Cin, int Kh, int Kw, int K_padded,
    int8_t* weight_rowmajor,  // scratch [K_padded, Cout], must be preallocated
    int8_t* weight_packed,    // output [K_padded, Cout] in VNNI panel layout
    int32_t* col_sums)        // [Cout]
{
    int K_real = Cin * Kh * Kw;
    // Build row-major [K_padded, Cout] from weight [Cout, Cin, Kh, Kw]:
    //   weight[co, ci, kh, kw] = weight_rowmajor[ci*Kh*Kw + kh*Kw + kw, co]
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
    // Zero-pad rows K_real..K_padded-1
    for (int k = K_real; k < K_padded; k++) {
        for (int co = 0; co < Cout; co++) {
            weight_rowmajor[(size_t)k * Cout + co] = 0;
        }
    }
    // VNNI-pack
    pack_B_vnni(weight_rowmajor, K_padded, Cout, weight_packed);
    // Column sums for compensation
    compute_col_sums(weight_rowmajor, K_padded, Cout, col_sums);
}

// Full Conv2d forward (int8 in, int32 out before dequant).
// Caller handles quant/dequant + BN fold as per-channel scale/bias.
void fastface_conv2d_i8(
    const int8_t*  input,        // [Cin, H_in, W_in]
    int Cin, int H_in, int W_in,
    int Cout, int Kh, int Kw, int stride, int pad,
    const int8_t*  weight_packed,// pre-packed per pack_conv_weight
    const int32_t* col_sums,     // [Cout], pre-computed
    int H_out, int W_out,
    int32_t*       output,       // [Cout, H_out, W_out] — will be HWC-flat then caller transposes
    int8_t*        scratch_im2col, // [H_out*W_out, K_padded]
    uint8_t*       scratch_Au,     // [H_out*W_out, K_padded]
    int32_t*       scratch_gemm)   // [H_out*W_out, Cout]
{
    int K_real = Cin * Kh * Kw;
    int K_padded = (K_real + 3) & ~3;
    int M = H_out * W_out;

    if (Kh == 1 && Kw == 1 && stride == 1 && pad == 0) {
        // 1×1 conv: input layout [Cin, H*W] → treat as GEMM directly.
        // Reshape view: interpret input as [Cin, H*W], then transpose to [H*W, Cin] = im2col-trivial.
        // For simplicity, still go through im2col path (it's just a memcopy for 1×1/s1/p0).
    }

    im2col_nchw(input, Cin, H_in, W_in, Kh, Kw, stride, pad,
                H_out, W_out, K_padded, scratch_im2col);
    pack_A_to_u8(scratch_im2col, M, K_padded, scratch_Au);
    fastface_gemm_i8(scratch_Au, weight_packed, scratch_gemm, M, K_padded, Cout);
    fastface_gemm_i8_finalize(scratch_gemm, M, Cout, col_sums);

    // Transpose [H_out*W_out, Cout] → [Cout, H_out*W_out] (back to NCHW-like layout).
    #pragma omp parallel for schedule(static)
    for (int co = 0; co < Cout; co++) {
        for (int idx = 0; idx < M; idx++) {
            output[(size_t)co * M + idx] = scratch_gemm[(size_t)idx * Cout + co];
        }
    }
}

// ---------- Benchmark harness ----------

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

    // M must be div by MR=4. For our shapes (56*56, 28*28, 14*14) it is.
    // Cout must be div by NR=8. 64, 128, 256 all ✓.

    printf("  %s: input[%dx%dx%d]  weight[%dx%dx%dx%d] s=%d p=%d\n",
           s->name, s->Cin, s->H_in, s->W_in,
           s->Cout, s->Cin, s->Kh, s->Kw, s->stride, s->pad);
    printf("    output[%dx%dx%d]  M=%d K_real=%d K_padded=%d N=%d\n",
           s->Cout, H_out, W_out, M, K_real, K_padded, s->Cout);

    if (M % 4 != 0 || s->Cout % 8 != 0) {
        printf("    SKIP: M%%4=%d or Cout%%8=%d not zero\n", M%4, s->Cout%8);
        return;
    }

    // Allocate
    size_t in_size = (size_t)s->Cin * s->H_in * s->W_in;
    size_t w_size = (size_t)s->Cout * s->Cin * s->Kh * s->Kw;
    int8_t*  input  = _aligned_malloc(in_size, 64);
    int8_t*  weight = _aligned_malloc(w_size, 64);
    int8_t*  w_rowmajor = _aligned_malloc((size_t)K_padded * s->Cout, 64);
    int8_t*  w_packed   = _aligned_malloc((size_t)K_padded * s->Cout, 64);
    int32_t* col_sums = _aligned_malloc((size_t)s->Cout * sizeof(int32_t), 64);
    int32_t* output = _aligned_malloc((size_t)s->Cout * M * sizeof(int32_t), 64);
    int8_t*  scratch_im2col = _aligned_malloc((size_t)M * K_padded, 64);
    uint8_t* scratch_Au = _aligned_malloc((size_t)M * K_padded, 64);
    int32_t* scratch_gemm = _aligned_malloc((size_t)M * s->Cout * sizeof(int32_t), 64);

    srand(42);
    for (size_t i = 0; i < in_size; i++) input[i] = (int8_t)((rand() & 0xFF) - 128);
    for (size_t i = 0; i < w_size; i++) weight[i] = (int8_t)((rand() & 0xFF) - 128);

    // Pre-pack weight (once per model load)
    pack_conv_weight(weight, s->Cout, s->Cin, s->Kh, s->Kw, K_padded,
                     w_rowmajor, w_packed, col_sums);

    // Warm-up
    for (int w = 0; w < 3; w++) {
        fastface_conv2d_i8(input, s->Cin, s->H_in, s->W_in,
                           s->Cout, s->Kh, s->Kw, s->stride, s->pad,
                           w_packed, col_sums, H_out, W_out, output,
                           scratch_im2col, scratch_Au, scratch_gemm);
    }

    const int N_ITER = 100;
    double t0 = now_s();
    for (int it = 0; it < N_ITER; it++) {
        fastface_conv2d_i8(input, s->Cin, s->H_in, s->W_in,
                           s->Cout, s->Kh, s->Kw, s->stride, s->pad,
                           w_packed, col_sums, H_out, W_out, output,
                           scratch_im2col, scratch_Au, scratch_gemm);
    }
    double t_conv = (now_s() - t0) / N_ITER;

    double gmacs = (double)M * K_real * s->Cout / 1e9;
    double gops_s = (2 * gmacs) / t_conv;
    printf("    FastFace conv2d: %.3f ms  (%.1f GOps/s)\n", t_conv * 1000, gops_s);

    _aligned_free(input); _aligned_free(weight); _aligned_free(w_rowmajor); _aligned_free(w_packed);
    _aligned_free(col_sums); _aligned_free(output);
    _aligned_free(scratch_im2col); _aligned_free(scratch_Au); _aligned_free(scratch_gemm);
    printf("\n");
}

int main(int argc, char** argv) {
    printf("FastFace Conv2d Session 3 benchmark\n");
    printf("i7-13700, 8 threads AVX-VNNI\n\n");

    ConvShape shapes[] = {
        // Stem 7x7 stride=2 would need M = 56*56 = 3136 (div 4), Cout=64 (div 8) ✓
        // But K_real = 3*7*7=147 → padded to 148 (not div 4). Actually (147+3)&~3 = 148. OK.
        {"Stem 7x7 s=2 p=3",   3, 112, 112, 64, 7, 7, 2, 3},
        // Mid 3x3: 56→28, 64→128 channels
        {"Mid 3x3 s=2 p=1",    64, 56, 56, 128, 3, 3, 2, 1},
        // Deep 1x1: 14x14, 256→256 (pointwise)
        {"Deep 1x1 s=1 p=0",   256, 14, 14, 256, 1, 1, 1, 0},
        // Also: bottleneck 3x3 stride=1 (common pattern)
        {"Bottle 3x3 s=1 p=1", 128, 28, 28, 128, 3, 3, 1, 1},
    };

    int n_shapes = sizeof(shapes) / sizeof(shapes[0]);
    for (int i = 0; i < n_shapes; i++) {
        bench_shape(&shapes[i]);
    }

    return 0;
}
