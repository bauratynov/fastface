// Session 7 — IResNet-50 activation kernels.
//
// Tensor layout convention for this engine: [Cout][HW] row-major.
// (Equivalent to NCHW with N=1, flattened spatial.)
//
// All three kernels work per-channel (outer loop over Cout) parallelized via OMP.

#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include <immintrin.h>
#include <windows.h>


// bn_prelu_quant_fused:
// For each output channel `c` (Cout total):
//   for each spatial pixel `p` in [0, HW):
//     x_fp = (float)in[c * HW + p] * conv_scale[c]
//     x_fp = x_fp * bn_scale[c] + bn_offset[c]
//     if x_fp < 0: x_fp *= prelu_slope[c]
//     q = round(x_fp / out_scale), clamp [-128, 127]
//     out[c * HW + p] = (int8_t)q
//
// Combines 4 ops (dequant + BN + PRelu + requant) in one SIMD pass, memory bound at L2/L3.
void bn_prelu_quant_fused(
    const int32_t* __restrict__ in,
    const float*   __restrict__ conv_scale,
    const float*   __restrict__ bn_scale,
    const float*   __restrict__ bn_offset,
    const float*   __restrict__ prelu_slope,
    float out_scale,
    int8_t* __restrict__ out,
    int Cout, int HW)
{
    const float inv_out_scale = 1.0f / out_scale;
    #pragma omp parallel for schedule(static)
    for (int c = 0; c < Cout; c++) {
        // Per-channel combined scale (conv_scale * bn_scale)
        const float per_c_scale = conv_scale[c] * bn_scale[c];
        const float per_c_offset = bn_offset[c];
        const float per_c_slope = prelu_slope[c];

        const int32_t* row_in = in + (size_t)c * HW;
        int8_t* row_out = out + (size_t)c * HW;

        int p = 0;
#ifdef __AVX2__
        const __m256 vscale = _mm256_set1_ps(per_c_scale);
        const __m256 voffset = _mm256_set1_ps(per_c_offset);
        const __m256 vslope = _mm256_set1_ps(per_c_slope);
        const __m256 vinv_out = _mm256_set1_ps(inv_out_scale);
        const __m256 vzero = _mm256_setzero_ps();
        const __m256 vpos_max = _mm256_set1_ps(127.0f);
        const __m256 vneg_max = _mm256_set1_ps(-128.0f);

        for (; p + 8 <= HW; p += 8) {
            __m256i vi = _mm256_loadu_si256((const __m256i*)(row_in + p));
            __m256 vx = _mm256_cvtepi32_ps(vi);
            vx = _mm256_fmadd_ps(vx, vscale, voffset);
            // PRelu: where x<0, multiply by slope
            __m256 neg_part = _mm256_mul_ps(vx, vslope);
            __m256 mask = _mm256_cmp_ps(vx, vzero, _CMP_LT_OS);
            vx = _mm256_blendv_ps(vx, neg_part, mask);
            // Re-quantize
            vx = _mm256_mul_ps(vx, vinv_out);
            // Round + clamp
            vx = _mm256_min_ps(_mm256_max_ps(vx, vneg_max), vpos_max);
            __m256i vq = _mm256_cvtps_epi32(vx);  // rounds to nearest (default)
            // Pack to int8: 8 × int32 → 8 × int8 (write byte per lane)
            // Use manual pack since we only have 8 elements
            int32_t tmp[8];
            _mm256_storeu_si256((__m256i*)tmp, vq);
            for (int k = 0; k < 8; k++) row_out[p + k] = (int8_t)tmp[k];
        }
#endif
        // Scalar tail
        for (; p < HW; p++) {
            float x = (float)row_in[p] * per_c_scale + per_c_offset;
            if (x < 0) x *= per_c_slope;
            x *= inv_out_scale;
            int32_t q = (int32_t)lrintf(x);
            if (q > 127) q = 127;
            if (q < -128) q = -128;
            row_out[p] = (int8_t)q;
        }
    }
}


// add_residual_int32: elementwise add of two int32 tensors.
// Used for skip connections in IResNet residual blocks.
void add_residual_int32(
    const int32_t* __restrict__ a,
    const int32_t* __restrict__ b,
    int32_t* __restrict__ out,
    int n)
{
    int i = 0;
#ifdef __AVX2__
    for (; i + 8 <= n; i += 8) {
        __m256i va = _mm256_loadu_si256((const __m256i*)(a + i));
        __m256i vb = _mm256_loadu_si256((const __m256i*)(b + i));
        _mm256_storeu_si256((__m256i*)(out + i), _mm256_add_epi32(va, vb));
    }
#endif
    for (; i < n; i++) out[i] = a[i] + b[i];
}


// prelu_quant_int32_to_int8: no-BN PRelu + quantize.
// Used after the stem conv (which has no BN preceding it in pre-activation IResNet).
void prelu_quant_int32_to_int8(
    const int32_t* __restrict__ in,
    const float*   __restrict__ conv_scale,
    const float*   __restrict__ prelu_slope,
    float out_scale,
    int8_t* __restrict__ out,
    int Cout, int HW)
{
    const float inv_out_scale = 1.0f / out_scale;
    #pragma omp parallel for schedule(static)
    for (int c = 0; c < Cout; c++) {
        const float per_c_scale = conv_scale[c];
        const float per_c_slope = prelu_slope[c];
        const int32_t* row_in = in + (size_t)c * HW;
        int8_t* row_out = out + (size_t)c * HW;

        int p = 0;
#ifdef __AVX2__
        const __m256 vscale = _mm256_set1_ps(per_c_scale);
        const __m256 vslope = _mm256_set1_ps(per_c_slope);
        const __m256 vinv_out = _mm256_set1_ps(inv_out_scale);
        const __m256 vzero = _mm256_setzero_ps();
        const __m256 vpos = _mm256_set1_ps(127.0f);
        const __m256 vneg = _mm256_set1_ps(-128.0f);

        for (; p + 8 <= HW; p += 8) {
            __m256i vi = _mm256_loadu_si256((const __m256i*)(row_in + p));
            __m256 vx = _mm256_mul_ps(_mm256_cvtepi32_ps(vi), vscale);
            __m256 neg = _mm256_mul_ps(vx, vslope);
            __m256 mask = _mm256_cmp_ps(vx, vzero, _CMP_LT_OS);
            vx = _mm256_blendv_ps(vx, neg, mask);
            vx = _mm256_mul_ps(vx, vinv_out);
            vx = _mm256_min_ps(_mm256_max_ps(vx, vneg), vpos);
            __m256i vq = _mm256_cvtps_epi32(vx);
            int32_t tmp[8];
            _mm256_storeu_si256((__m256i*)tmp, vq);
            for (int k = 0; k < 8; k++) row_out[p + k] = (int8_t)tmp[k];
        }
#endif
        for (; p < HW; p++) {
            float x = (float)row_in[p] * per_c_scale;
            if (x < 0) x *= per_c_slope;
            x *= inv_out_scale;
            int32_t q = (int32_t)lrintf(x);
            if (q > 127) q = 127;
            if (q < -128) q = -128;
            row_out[p] = (int8_t)q;
        }
    }
}


// ==================== Micro-bench ====================

static double now_s(void) {
    LARGE_INTEGER q, f;
    QueryPerformanceCounter(&q);
    QueryPerformanceFrequency(&f);
    return (double)q.QuadPart / (double)f.QuadPart;
}

int main(void) {
    printf("FastFace Session 7 activation kernels benchmark\n");
    printf("i7-13700, 8 threads AVX2\n\n");

    // Representative dimensions: mid-layer 28x28 128ch (matches Mid 3x3 Conv2d output).
    const int Cout = 128;
    const int HW = 28 * 28;
    const int N_TOTAL = Cout * HW;

    int32_t* in32  = _aligned_malloc(N_TOTAL * sizeof(int32_t), 64);
    int32_t* b32   = _aligned_malloc(N_TOTAL * sizeof(int32_t), 64);
    int32_t* sum32 = _aligned_malloc(N_TOTAL * sizeof(int32_t), 64);
    int8_t*  out8  = _aligned_malloc(N_TOTAL, 64);

    float* conv_scale = _aligned_malloc(Cout * sizeof(float), 64);
    float* bn_scale   = _aligned_malloc(Cout * sizeof(float), 64);
    float* bn_offset  = _aligned_malloc(Cout * sizeof(float), 64);
    float* prelu_slope = _aligned_malloc(Cout * sizeof(float), 64);

    srand(42);
    for (int i = 0; i < N_TOTAL; i++) {
        in32[i] = (rand() % 2001) - 1000;
        b32[i]  = (rand() % 2001) - 1000;
    }
    for (int c = 0; c < Cout; c++) {
        conv_scale[c] = 0.01f + 0.001f * (rand() % 100);
        bn_scale[c]   = 0.9f + 0.01f * (rand() % 20);
        bn_offset[c]  = 0.1f * ((rand() % 20) - 10);
        prelu_slope[c] = 0.25f;  // typical PReLU init
    }

    const int ITER = 2000;

    // --- bn_prelu_quant_fused ---
    for (int w = 0; w < 10; w++)
        bn_prelu_quant_fused(in32, conv_scale, bn_scale, bn_offset, prelu_slope, 0.05f, out8, Cout, HW);
    double t0 = now_s();
    for (int i = 0; i < ITER; i++)
        bn_prelu_quant_fused(in32, conv_scale, bn_scale, bn_offset, prelu_slope, 0.05f, out8, Cout, HW);
    double t_bn = (now_s() - t0) / ITER;
    printf("bn_prelu_quant_fused (Cout=%d HW=%d): %.4f ms  (%.2f GB/s read)\n",
           Cout, HW, t_bn * 1000, (double)N_TOTAL * 4 / t_bn / 1e9);

    // --- prelu_quant_int32_to_int8 ---
    for (int w = 0; w < 10; w++)
        prelu_quant_int32_to_int8(in32, conv_scale, prelu_slope, 0.05f, out8, Cout, HW);
    t0 = now_s();
    for (int i = 0; i < ITER; i++)
        prelu_quant_int32_to_int8(in32, conv_scale, prelu_slope, 0.05f, out8, Cout, HW);
    double t_pr = (now_s() - t0) / ITER;
    printf("prelu_quant_int32_to_int8     (Cout=%d HW=%d): %.4f ms  (%.2f GB/s read)\n",
           Cout, HW, t_pr * 1000, (double)N_TOTAL * 4 / t_pr / 1e9);

    // --- add_residual_int32 ---
    for (int w = 0; w < 10; w++)
        add_residual_int32(in32, b32, sum32, N_TOTAL);
    t0 = now_s();
    for (int i = 0; i < ITER; i++)
        add_residual_int32(in32, b32, sum32, N_TOTAL);
    double t_add = (now_s() - t0) / ITER;
    printf("add_residual_int32            (n=%d):         %.4f ms  (%.2f GB/s r+w)\n",
           N_TOTAL, t_add * 1000, (double)N_TOTAL * 12 / t_add / 1e9);

    // Time budget per layer (typical IResNet block):
    // 1 BN+PReLU+quant + 2 Conv + 1 Add = ~1 activation pass + 2 convs + 1 add
    // For the whole model with 25 PReLU and 24 Adds, activations contribute:
    //   25 * t_bn + 24 * t_add ≈ per inference activation overhead
    double activation_per_inf_ms = (25 * t_bn + 24 * t_add) * 1000;
    printf("\nEstimated total activation overhead per inference: %.3f ms\n", activation_per_inf_ms);
    printf("(vs ORT profile ~5%% of 32 ms = 1.6 ms — we should match or beat)\n");

    _aligned_free(in32); _aligned_free(b32); _aligned_free(sum32); _aligned_free(out8);
    _aligned_free(conv_scale); _aligned_free(bn_scale);
    _aligned_free(bn_offset); _aligned_free(prelu_slope);
    return 0;
}
