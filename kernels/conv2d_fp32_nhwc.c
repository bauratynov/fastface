// Session 13 — FP32 NHWC Conv2d for perfect-quality mode.
// Reads NHWC fp32 input, writes NHWC fp32 output (consistent with INT8 path).

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <immintrin.h>
#include <math.h>

void pack_B_fp32(const float* B, int K, int N, float* Bp);
void fastface_gemm_fp32(const float* A, const float* Bp, float* C, int M, int K, int N);


// NHWC im2col for FP32 — no quantization shift, just spatial unfolding.
void im2col_fp32_nhwc(
    const float* input, int Cin, int H_in, int W_in,
    int Kh, int Kw, int stride, int pad,
    int H_out, int W_out, int K_padded, float* out)
{
    int K_real = Cin * Kh * Kw;
    #pragma omp parallel for schedule(static)
    for (int oh = 0; oh < H_out; oh++) {
        for (int ow = 0; ow < W_out; ow++) {
            int row = oh * W_out + ow;
            float* out_row = out + (size_t)row * K_padded;
            int idx = 0;
            for (int kh = 0; kh < Kh; kh++) {
                for (int kw = 0; kw < Kw; kw++) {
                    int ih = oh * stride - pad + kh;
                    int iw = ow * stride - pad + kw;
                    if (ih >= 0 && ih < H_in && iw >= 0 && iw < W_in) {
                        const float* src = input + ((size_t)ih * W_in + iw) * Cin;
                        memcpy(out_row + idx, src, (size_t)Cin * sizeof(float));
                        idx += Cin;
                    } else {
                        memset(out_row + idx, 0, (size_t)Cin * sizeof(float));
                        idx += Cin;
                    }
                }
            }
            for (; idx < K_padded; idx++) out_row[idx] = 0;
        }
    }
}


// Pack conv weight [Cout, Cin, Kh, Kw] NCHW → NHWC-compatible [K_padded, Cout] then NR_F-panel pack.
void pack_conv_weight_fp32_nhwc(
    const float* weight, int Cout, int Cin, int Kh, int Kw, int K_padded,
    float* w_rowmajor, float* w_packed)
{
    memset(w_rowmajor, 0, (size_t)K_padded * Cout * sizeof(float));
    for (int co = 0; co < Cout; co++) {
        for (int kh = 0; kh < Kh; kh++) {
            for (int kw = 0; kw < Kw; kw++) {
                for (int ci = 0; ci < Cin; ci++) {
                    int k = (kh * Kw + kw) * Cin + ci;
                    w_rowmajor[(size_t)k * Cout + co] =
                        weight[((size_t)co * Cin + ci) * Kh * Kw + kh * Kw + kw];
                }
            }
        }
    }
    pack_B_fp32(w_rowmajor, K_padded, Cout, w_packed);
}


// Full NHWC Conv2d FP32: [H_in, W_in, Cin] → [H_out*W_out, Cout].
void fastface_conv2d_fp32_nhwc(
    const float* input, int Cin, int H_in, int W_in,
    int Cout, int Kh, int Kw, int stride, int pad,
    const float* weight_packed,
    int H_out, int W_out,
    float* output, float* scratch_im)
{
    int K_real = Cin * Kh * Kw;
    int K_padded = (K_real + 15) & ~15;
    int M = H_out * W_out;
    int M_padded = (M + 3) & ~3;

    im2col_fp32_nhwc(input, Cin, H_in, W_in, Kh, Kw, stride, pad,
                     H_out, W_out, K_padded, scratch_im);
    if (M_padded > M) {
        memset(scratch_im + (size_t)M * K_padded, 0,
               (size_t)(M_padded - M) * K_padded * sizeof(float));
    }
    fastface_gemm_fp32(scratch_im, weight_packed, output, M_padded, K_padded, Cout);
}

// Batched variant: process B faces at once. Input: [B, H_in, W_in, Cin].
// Output: [B, H_out*W_out, Cout]. Internally builds one big im2col with
// M = B*HW_out rows, then a single GEMM — weight B matrix stays hot across batches.
void fastface_conv2d_fp32_nhwc_batched(
    const float* input, int B, int Cin, int H_in, int W_in,
    int Cout, int Kh, int Kw, int stride, int pad,
    const float* weight_packed,
    int H_out, int W_out,
    float* output, float* scratch_im)
{
    int K_real = Cin * Kh * Kw;
    int K_padded = (K_real + 15) & ~15;
    int M_per_face = H_out * W_out;
    int M_total = B * M_per_face;
    int M_padded = (M_total + 3) & ~3;

    // Batched im2col: each face's tiles written consecutively in scratch_im.
    #pragma omp parallel for schedule(static)
    for (int b = 0; b < B; b++) {
        const float* face_in = input + (size_t)b * H_in * W_in * Cin;
        float* face_im = scratch_im + (size_t)b * M_per_face * K_padded;
        im2col_fp32_nhwc(face_in, Cin, H_in, W_in, Kh, Kw, stride, pad,
                         H_out, W_out, K_padded, face_im);
    }
    if (M_padded > M_total) {
        memset(scratch_im + (size_t)M_total * K_padded, 0,
               (size_t)(M_padded - M_total) * K_padded * sizeof(float));
    }
    // Single GEMM over all B*M rows — weight re-used across batches.
    fastface_gemm_fp32(scratch_im, weight_packed, output, M_padded, K_padded, Cout);
}


// --- Activation ops (FP32, in-place) ---

// BN: per-channel scale + offset (pre-computed from gamma/beta/mean/var).
// Input shape [HW, Cout] NHWC.
void bn_fp32_nhwc(float* x, const float* scale, const float* offset, int HW, int Cout) {
    #pragma omp parallel for schedule(static)
    for (int p = 0; p < HW; p++) {
        float* row = x + (size_t)p * Cout;
        for (int c = 0; c < Cout; c++) {
            row[c] = row[c] * scale[c] + offset[c];
        }
    }
}

// PReLU per-channel: x = (x >= 0) ? x : x * slope[c].
// Input shape [HW, Cout] NHWC.
void prelu_fp32_nhwc(float* x, const float* slope, int HW, int Cout) {
    #pragma omp parallel for schedule(static)
    for (int p = 0; p < HW; p++) {
        float* row = x + (size_t)p * Cout;
        int c = 0;
#ifdef __AVX2__
        for (; c + 8 <= Cout; c += 8) {
            __m256 vx = _mm256_loadu_ps(row + c);
            __m256 vs = _mm256_loadu_ps(slope + c);
            __m256 vneg = _mm256_mul_ps(vx, vs);
            __m256 mask = _mm256_cmp_ps(vx, _mm256_setzero_ps(), _CMP_LT_OS);
            _mm256_storeu_ps(row + c, _mm256_blendv_ps(vx, vneg, mask));
        }
#endif
        for (; c < Cout; c++) {
            if (row[c] < 0) row[c] *= slope[c];
        }
    }
}

// Add per-channel bias to NHWC tensor [HW, Cout].
void add_bias_nhwc(float* x, const float* bias, int HW, int Cout) {
    if (!bias) return;
    #pragma omp parallel for schedule(static)
    for (int p = 0; p < HW; p++) {
        float* row = x + (size_t)p * Cout;
        int c = 0;
#ifdef __AVX2__
        for (; c + 8 <= Cout; c += 8) {
            _mm256_storeu_ps(row + c, _mm256_add_ps(_mm256_loadu_ps(row + c), _mm256_loadu_ps(bias + c)));
        }
#endif
        for (; c < Cout; c++) row[c] += bias[c];
    }
}

// Add: elementwise fp32 tensor add.
void add_fp32(const float* a, const float* b, float* out, int n) {
    int i = 0;
#ifdef __AVX2__
    for (; i + 8 <= n; i += 8) {
        _mm256_storeu_ps(out + i, _mm256_add_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i)));
    }
#endif
    for (; i < n; i++) out[i] = a[i] + b[i];
}
