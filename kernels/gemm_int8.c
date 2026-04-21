// FastFace INT8 GEMM — AVX-VNNI/AVX2 kernel for ArcFace feasibility study (Session 2 v2).
//
// Fix for v2: B packed with K-4-chunked layout so that each dpbusd lane holds
// 4 consecutive k-values at same column. With A broadcast 4 bytes, the 8 output
// lanes cover 8 output columns, each accumulating 4 k-contributions per instr.
//
// Packed B layout (for one N-tile of NR=8 cols):
//   for each k4 in 0..K/4:
//     for each j in 0..NR-1:
//       for p in 0..3:
//         Bp[k4 * NR*4 + j*4 + p] = B[(k4*4+p) * N + tile_col_base + j]

#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <immintrin.h>

#define MR 4   // rows per register tile
#define NR 8   // cols per register tile (1 YMM of int32 output per row)

// Compute column sums of B for the unsigned-A compensation trick.
void compute_col_sums(const int8_t* B, int K, int N, int32_t* col_sums) {
    for (int j = 0; j < N; j++) {
        int32_t s = 0;
        for (int k = 0; k < K; k++) s += (int32_t)B[k * N + j];
        col_sums[j] = s;
    }
}

// Pack A: unsigned shift A + 128. Row-major, same layout.
void pack_A_to_u8(const int8_t* A, int M, int K, uint8_t* Au) {
    for (int i = 0; i < M; i++) {
        for (int k = 0; k < K; k++) {
            Au[i * K + k] = (uint8_t)((int)A[i * K + k] + 128);
        }
    }
}

// Pack B for VNNI: K-4-chunked × N-tile NR=8.
// Bp is [n_tile_count][K/4][NR][4].
void pack_B_vnni(const int8_t* B, int K, int N, int8_t* Bp) {
    int nt = N / NR;
    int K4 = K / 4;  // assume K divisible by 4
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

// Inner 4×8 micro-kernel.
// Au: pointer to row 0 of MR rows of A (uint8, K cols, row stride = lda).
// Bp: pointer to tile's K4*NR*4 packed bytes (32 bytes per k4).
static inline void microkernel_4x8(
    const uint8_t* __restrict__ Au,
    const int8_t*  __restrict__ Bp,
    int K,
    int lda,
    int32_t* __restrict__ C,
    int ldc)
{
    __m256i c0 = _mm256_setzero_si256();
    __m256i c1 = _mm256_setzero_si256();
    __m256i c2 = _mm256_setzero_si256();
    __m256i c3 = _mm256_setzero_si256();

    int K4 = K / 4;
    for (int k4 = 0; k4 < K4; k4++) {
        const __m256i b = _mm256_loadu_si256((const __m256i*)(Bp + k4 * 32));
        // Load 4-byte A chunks for 4 rows and broadcast
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

    _mm256_storeu_si256((__m256i*)(C + 0 * ldc), c0);
    _mm256_storeu_si256((__m256i*)(C + 1 * ldc), c1);
    _mm256_storeu_si256((__m256i*)(C + 2 * ldc), c2);
    _mm256_storeu_si256((__m256i*)(C + 3 * ldc), c3);
}

// Public: GEMM int8 × int8 → int32 (A pre-shifted to u8, B pre-VNNI-packed).
// M % MR == 0 and N % NR == 0 required. K % 4 == 0 required.
void fastface_gemm_i8(
    const uint8_t* Au,
    const int8_t*  Bp,
    int32_t*       C,
    int M, int K, int N)
{
    int K4 = K / 4;
    int nt = N / NR;
    #pragma omp parallel for schedule(static) collapse(2)
    for (int i = 0; i < M; i += MR) {
        for (int t = 0; t < nt; t++) {
            microkernel_4x8(Au + i * K,
                            Bp + t * K4 * NR * 4,
                            K, K,
                            C + i * N + t * NR, N);
        }
    }
}

// Compensation pass: C_signed[i,j] = C_unsigned[i,j] - 128 * col_sum[j].
void fastface_gemm_i8_finalize(int32_t* C, int M, int N, const int32_t* col_sums) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            C[i * N + j] -= 128 * col_sums[j];
        }
    }
}

// Scalar reference for correctness checks.
void fastface_gemm_i8_ref(
    const int8_t* A, const int8_t* B, int32_t* C, int M, int K, int N)
{
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            int32_t s = 0;
            for (int k = 0; k < K; k++) {
                s += (int32_t)A[i * K + k] * (int32_t)B[k * N + j];
            }
            C[i * N + j] = s;
        }
    }
}
