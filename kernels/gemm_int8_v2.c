// Session 4 — fused GEMM kernel:
// - Microkernel stores C with compensation already subtracted (no finalize pass)
// - Tile 4×8 (same as S2) for correctness
// - col_sums passed into kernel, subtracted during store

#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <immintrin.h>

#define MR 4
#define NR 8

// (reused from S2, unchanged)
void compute_col_sums(const int8_t* B, int K, int N, int32_t* col_sums) {
    for (int j = 0; j < N; j++) {
        int32_t s = 0;
        for (int k = 0; k < K; k++) s += (int32_t)B[k * N + j];
        col_sums[j] = s;
    }
}

void pack_B_vnni(const int8_t* B, int K, int N, int8_t* Bp) {
    int nt = N / NR;
    int K4 = K / 4;
    for (int t = 0; t < nt; t++) {
        for (int k4 = 0; k4 < K4; k4++) {
            for (int j = 0; j < NR; j++) {
                for (int p = 0; p < 4; p++) {
                    Bp[t * K4 * NR * 4 + k4 * NR * 4 + j * 4 + p] =
                        B[(k4 * 4 + p) * N + t * NR + j];
                }
            }
        }
    }
}

// Fused microkernel: compute C_tile = Au*B - 128*col_sums_tile
// col_sums_tile is a YMM (8 int32 lanes) — pre-loaded by caller for the tile's 8 output cols.
static inline void microkernel_4x8_fused(
    const uint8_t* __restrict__ Au,
    const int8_t*  __restrict__ Bp,
    int K,
    int lda,
    int32_t* __restrict__ C,
    int ldc,
    __m256i col_sums_128)  // already multiplied by 128
{
    __m256i c0 = _mm256_setzero_si256();
    __m256i c1 = _mm256_setzero_si256();
    __m256i c2 = _mm256_setzero_si256();
    __m256i c3 = _mm256_setzero_si256();

    int K4 = K / 4;
    for (int k4 = 0; k4 < K4; k4++) {
        const __m256i b = _mm256_loadu_si256((const __m256i*)(Bp + k4 * 32));
        const __m256i a0 = _mm256_set1_epi32(*(const int*)(Au + 0 * lda + k4 * 4));
        const __m256i a1 = _mm256_set1_epi32(*(const int*)(Au + 1 * lda + k4 * 4));
        const __m256i a2 = _mm256_set1_epi32(*(const int*)(Au + 2 * lda + k4 * 4));
        const __m256i a3 = _mm256_set1_epi32(*(const int*)(Au + 3 * lda + k4 * 4));
#ifdef __AVXVNNI__
        c0 = _mm256_dpbusd_epi32(c0, a0, b);
        c1 = _mm256_dpbusd_epi32(c1, a1, b);
        c2 = _mm256_dpbusd_epi32(c2, a2, b);
        c3 = _mm256_dpbusd_epi32(c3, a3, b);
#else
        const __m256i ones16 = _mm256_set1_epi16(1);
        __m256i p;
        p = _mm256_maddubs_epi16(a0, b); c0 = _mm256_add_epi32(c0, _mm256_madd_epi16(p, ones16));
        p = _mm256_maddubs_epi16(a1, b); c1 = _mm256_add_epi32(c1, _mm256_madd_epi16(p, ones16));
        p = _mm256_maddubs_epi16(a2, b); c2 = _mm256_add_epi32(c2, _mm256_madd_epi16(p, ones16));
        p = _mm256_maddubs_epi16(a3, b); c3 = _mm256_add_epi32(c3, _mm256_madd_epi16(p, ones16));
#endif
    }

    // Fused epilogue: subtract 128*col_sums, store
    _mm256_storeu_si256((__m256i*)(C + 0 * ldc), _mm256_sub_epi32(c0, col_sums_128));
    _mm256_storeu_si256((__m256i*)(C + 1 * ldc), _mm256_sub_epi32(c1, col_sums_128));
    _mm256_storeu_si256((__m256i*)(C + 2 * ldc), _mm256_sub_epi32(c2, col_sums_128));
    _mm256_storeu_si256((__m256i*)(C + 3 * ldc), _mm256_sub_epi32(c3, col_sums_128));
}

// Public fused GEMM (no separate finalize needed).
void fastface_gemm_i8_fused(
    const uint8_t* Au,
    const int8_t*  Bp,
    const int32_t* col_sums,  // [N]
    int32_t*       C,
    int M, int K, int N)
{
    int K4 = K / 4;
    int nt = N / NR;
    // Pre-shift col_sums by <<7 (multiply by 128) so we can subtract directly with epi32 sub.
    // Load once per column tile.
    #pragma omp parallel for schedule(static) collapse(2)
    for (int i = 0; i < M; i += MR) {
        for (int t = 0; t < nt; t++) {
            __m256i cs = _mm256_loadu_si256((const __m256i*)(col_sums + t * NR));
            __m256i cs128 = _mm256_slli_epi32(cs, 7);  // multiply by 128 via left-shift
            microkernel_4x8_fused(
                Au + i * K,
                Bp + t * K4 * NR * 4,
                K, K,
                C + i * N + t * NR, N,
                cs128);
        }
    }
}
