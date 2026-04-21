// S31 — Vectorized int8 matvec for the final Linear(25088 -> 512).
//
// Replaces the scalar triple-cast loop in arcface_forward_int8.c that was
// eating 24% of total INT8 inference time. Uses AVX-VNNI dpbusd with the
// uint8/int8 signed trick: Au = A XOR 0x80 (shifts into uint8 domain), then
//   dot_signed = dpbusd(Au, W) - 128 * sum(W_row)
// Row sums of W are precomputed once at load time in the caller.
//
// Shape: M=1, K=25088 (multiple of 32), N=512.

#include <stdint.h>
#include <immintrin.h>
#include <omp.h>

static inline int32_t hsum_epi32_avx2(__m256i v) {
    __m128i lo = _mm256_castsi256_si128(v);
    __m128i hi = _mm256_extracti128_si256(v, 1);
    __m128i s  = _mm_add_epi32(lo, hi);
    __m128i sh = _mm_shuffle_epi32(s, _MM_SHUFFLE(2, 3, 0, 1));
    s = _mm_add_epi32(s, sh);
    sh = _mm_shuffle_epi32(s, _MM_SHUFFLE(1, 0, 3, 2));
    s = _mm_add_epi32(s, sh);
    return _mm_cvtsi128_si32(s);
}

// Au: activation already shifted into uint8 domain (A XOR 0x80) — caller does this once
// W: row-major [N, K] int8 weight
// w_row_sum: precomputed sum of W[n,:] per row (int32, computed once at load time)
// W_scale: per-row dequant scale (fp32[N])
// bias: per-row bias (fp32[N])
// A_scale: per-tensor activation scale
// out: fp32[N]
void fastface_gemm_i8_matvec_vnni(
    const uint8_t* __restrict__ Au,
    const int8_t*  __restrict__ W,
    const int32_t* __restrict__ w_row_sum,
    const float*   __restrict__ W_scale,
    const float*   __restrict__ bias,
    float A_scale,
    float* __restrict__ out,
    int N, int K)
{
    #pragma omp parallel for schedule(static)
    for (int n = 0; n < N; n++) {
        __m256i acc = _mm256_setzero_si256();
        const int8_t* Wn = W + (size_t)n * K;
        int k = 0;
        // Unroll x2 to hide dpbusd latency (ymm chain has throughput 1/cycle on Alder/Raptor).
        __m256i acc2 = _mm256_setzero_si256();
        for (; k + 64 <= K; k += 64) {
            __m256i au0 = _mm256_loadu_si256((const __m256i*)(Au + k));
            __m256i w0  = _mm256_loadu_si256((const __m256i*)(Wn + k));
            __m256i au1 = _mm256_loadu_si256((const __m256i*)(Au + k + 32));
            __m256i w1  = _mm256_loadu_si256((const __m256i*)(Wn + k + 32));
#ifdef __AVXVNNI__
            acc  = _mm256_dpbusd_epi32(acc,  au0, w0);
            acc2 = _mm256_dpbusd_epi32(acc2, au1, w1);
#else
            const __m256i ones16 = _mm256_set1_epi16(1);
            __m256i p0 = _mm256_maddubs_epi16(au0, w0);
            __m256i p1 = _mm256_maddubs_epi16(au1, w1);
            acc  = _mm256_add_epi32(acc,  _mm256_madd_epi16(p0, ones16));
            acc2 = _mm256_add_epi32(acc2, _mm256_madd_epi16(p1, ones16));
#endif
        }
        for (; k + 32 <= K; k += 32) {
            __m256i au0 = _mm256_loadu_si256((const __m256i*)(Au + k));
            __m256i w0  = _mm256_loadu_si256((const __m256i*)(Wn + k));
#ifdef __AVXVNNI__
            acc = _mm256_dpbusd_epi32(acc, au0, w0);
#else
            const __m256i ones16 = _mm256_set1_epi16(1);
            __m256i p = _mm256_maddubs_epi16(au0, w0);
            acc = _mm256_add_epi32(acc, _mm256_madd_epi16(p, ones16));
#endif
        }
        acc = _mm256_add_epi32(acc, acc2);
        int32_t signed_dot = hsum_epi32_avx2(acc) - 128 * w_row_sum[n];
        out[n] = bias[n] + (float)signed_dot * A_scale * W_scale[n];
    }
}
