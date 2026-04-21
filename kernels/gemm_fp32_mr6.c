// Session 18 — FP32 GEMM with MR=6, NR=16 micro-kernel (BLIS/OpenBLAS standard for AVX2).
//
// Register budget on AVX2 (16 YMM × 8 fp32 lanes):
//   12 regs — C accumulators (6 rows × 2 YMM of 8 fp32 = 6×16 tile)
//    2 regs — B panel loads (b0, b1 for the current k column)
//    1 reg  — A broadcast (broadcast A[row, k] into 8 lanes)
//   — matches 75% accumulator utilization vs 50% for MR=4
//
// B packing is identical to gemm_fp32.c (NR=16 panels, pack_B_fp32 shared).

#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <immintrin.h>

#define MR6 6
#define NR6 16

// Micro-kernel: 6 rows × 16 cols, AVX2+FMA.
static inline void microkernel_6x16_fp32(
    const float* __restrict__ A,   // [MR6 × lda]
    const float* __restrict__ Bp,  // [K × NR6] packed panel
    int K, int lda,
    float* __restrict__ C,         // [MR6 × ldc]
    int ldc)
{
    __m256 c00 = _mm256_setzero_ps(), c01 = _mm256_setzero_ps();
    __m256 c10 = _mm256_setzero_ps(), c11 = _mm256_setzero_ps();
    __m256 c20 = _mm256_setzero_ps(), c21 = _mm256_setzero_ps();
    __m256 c30 = _mm256_setzero_ps(), c31 = _mm256_setzero_ps();
    __m256 c40 = _mm256_setzero_ps(), c41 = _mm256_setzero_ps();
    __m256 c50 = _mm256_setzero_ps(), c51 = _mm256_setzero_ps();

    for (int k = 0; k < K; k++) {
        const __m256 b0 = _mm256_loadu_ps(Bp + k * NR6);
        const __m256 b1 = _mm256_loadu_ps(Bp + k * NR6 + 8);
        __m256 a;
        a = _mm256_broadcast_ss(A + 0 * lda + k);
        c00 = _mm256_fmadd_ps(a, b0, c00);  c01 = _mm256_fmadd_ps(a, b1, c01);
        a = _mm256_broadcast_ss(A + 1 * lda + k);
        c10 = _mm256_fmadd_ps(a, b0, c10);  c11 = _mm256_fmadd_ps(a, b1, c11);
        a = _mm256_broadcast_ss(A + 2 * lda + k);
        c20 = _mm256_fmadd_ps(a, b0, c20);  c21 = _mm256_fmadd_ps(a, b1, c21);
        a = _mm256_broadcast_ss(A + 3 * lda + k);
        c30 = _mm256_fmadd_ps(a, b0, c30);  c31 = _mm256_fmadd_ps(a, b1, c31);
        a = _mm256_broadcast_ss(A + 4 * lda + k);
        c40 = _mm256_fmadd_ps(a, b0, c40);  c41 = _mm256_fmadd_ps(a, b1, c41);
        a = _mm256_broadcast_ss(A + 5 * lda + k);
        c50 = _mm256_fmadd_ps(a, b0, c50);  c51 = _mm256_fmadd_ps(a, b1, c51);
    }
    _mm256_storeu_ps(C + 0 * ldc + 0, c00); _mm256_storeu_ps(C + 0 * ldc + 8, c01);
    _mm256_storeu_ps(C + 1 * ldc + 0, c10); _mm256_storeu_ps(C + 1 * ldc + 8, c11);
    _mm256_storeu_ps(C + 2 * ldc + 0, c20); _mm256_storeu_ps(C + 2 * ldc + 8, c21);
    _mm256_storeu_ps(C + 3 * ldc + 0, c30); _mm256_storeu_ps(C + 3 * ldc + 8, c31);
    _mm256_storeu_ps(C + 4 * ldc + 0, c40); _mm256_storeu_ps(C + 4 * ldc + 8, c41);
    _mm256_storeu_ps(C + 5 * ldc + 0, c50); _mm256_storeu_ps(C + 5 * ldc + 8, c51);
}

// MR=4 fallback for M tails when M % 6 != 0 (up to 5 remaining rows).
// Uses the same math as MR=6 but with 4 accumulator rows to handle sizes 4 and 5.
static inline void microkernel_4x16_tail(
    const float* __restrict__ A, const float* __restrict__ Bp,
    int K, int lda, float* __restrict__ C, int ldc, int m_rows)
{
    __m256 c00 = _mm256_setzero_ps(), c01 = _mm256_setzero_ps();
    __m256 c10 = _mm256_setzero_ps(), c11 = _mm256_setzero_ps();
    __m256 c20 = _mm256_setzero_ps(), c21 = _mm256_setzero_ps();
    __m256 c30 = _mm256_setzero_ps(), c31 = _mm256_setzero_ps();
    __m256 c40 = _mm256_setzero_ps(), c41 = _mm256_setzero_ps();
    for (int k = 0; k < K; k++) {
        __m256 b0 = _mm256_loadu_ps(Bp + k * NR6);
        __m256 b1 = _mm256_loadu_ps(Bp + k * NR6 + 8);
        if (m_rows > 0) { __m256 a = _mm256_broadcast_ss(A + 0*lda + k); c00 = _mm256_fmadd_ps(a, b0, c00); c01 = _mm256_fmadd_ps(a, b1, c01); }
        if (m_rows > 1) { __m256 a = _mm256_broadcast_ss(A + 1*lda + k); c10 = _mm256_fmadd_ps(a, b0, c10); c11 = _mm256_fmadd_ps(a, b1, c11); }
        if (m_rows > 2) { __m256 a = _mm256_broadcast_ss(A + 2*lda + k); c20 = _mm256_fmadd_ps(a, b0, c20); c21 = _mm256_fmadd_ps(a, b1, c21); }
        if (m_rows > 3) { __m256 a = _mm256_broadcast_ss(A + 3*lda + k); c30 = _mm256_fmadd_ps(a, b0, c30); c31 = _mm256_fmadd_ps(a, b1, c31); }
        if (m_rows > 4) { __m256 a = _mm256_broadcast_ss(A + 4*lda + k); c40 = _mm256_fmadd_ps(a, b0, c40); c41 = _mm256_fmadd_ps(a, b1, c41); }
    }
    if (m_rows > 0) { _mm256_storeu_ps(C + 0*ldc, c00); _mm256_storeu_ps(C + 0*ldc + 8, c01); }
    if (m_rows > 1) { _mm256_storeu_ps(C + 1*ldc, c10); _mm256_storeu_ps(C + 1*ldc + 8, c11); }
    if (m_rows > 2) { _mm256_storeu_ps(C + 2*ldc, c20); _mm256_storeu_ps(C + 2*ldc + 8, c21); }
    if (m_rows > 3) { _mm256_storeu_ps(C + 3*ldc, c30); _mm256_storeu_ps(C + 3*ldc + 8, c31); }
    if (m_rows > 4) { _mm256_storeu_ps(C + 4*ldc, c40); _mm256_storeu_ps(C + 4*ldc + 8, c41); }
}

// Public GEMM: A [M×K] row-major × Bp (pack_B_fp32 panels, NR=16) → C [M×N].
// N must be multiple of NR6 (=16). M arbitrary — handles 6-row full tiles + tail.
void fastface_gemm_fp32_mr6(const float* A, const float* Bp, float* C,
                             int M, int K, int N)
{
    int nt = N / NR6;
    int M6 = (M / MR6) * MR6;  // number of full 6-row bands
    int tail = M - M6;

    #pragma omp parallel for schedule(static) collapse(2)
    for (int i = 0; i < M6; i += MR6) {
        for (int t = 0; t < nt; t++) {
            microkernel_6x16_fp32(
                A + i * K,
                Bp + t * K * NR6,
                K, K,
                C + i * N + t * NR6, N);
        }
    }
    if (tail > 0) {
        #pragma omp parallel for schedule(static)
        for (int t = 0; t < nt; t++) {
            microkernel_4x16_tail(
                A + M6 * K,
                Bp + t * K * NR6,
                K, K,
                C + M6 * N + t * NR6, N,
                tail);
        }
    }
}
