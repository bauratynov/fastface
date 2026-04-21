// Session 13 — FP32 GEMM kernel for Path A (perfect-quality mode).
//
// Layout: row-major A [M×K] fp32, row-major B [K×N] fp32, out C [M×N] fp32.
// Tile: 4 rows × 16 cols (2 YMM of 8 fp32 each per row) × 4 accumulators per row tile.
// Uses AVX2 FMA for inner loop.

#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <immintrin.h>


#define MR_F 4
#define NR_F 16  // 2 YMM lanes of 8 fp32 each

// Pack B into NR_F-wide column panels (row-major inside a panel).
void pack_B_fp32(const float* B, int K, int N, float* Bp) {
    int nt = N / NR_F;
    for (int t = 0; t < nt; t++) {
        for (int k = 0; k < K; k++) {
            for (int j = 0; j < NR_F; j++) {
                Bp[t * K * NR_F + k * NR_F + j] = B[k * N + t * NR_F + j];
            }
        }
    }
}

// Micro-kernel: 4 rows × 16 cols, FMA-accumulated.
static inline void microkernel_4x16_fp32(
    const float* __restrict__ A,  // [MR_F × lda]
    const float* __restrict__ Bp, // [K × NR_F] packed
    int K, int lda,
    float* __restrict__ C,        // [MR_F × ldc]
    int ldc)
{
    __m256 c00 = _mm256_setzero_ps(), c01 = _mm256_setzero_ps();
    __m256 c10 = _mm256_setzero_ps(), c11 = _mm256_setzero_ps();
    __m256 c20 = _mm256_setzero_ps(), c21 = _mm256_setzero_ps();
    __m256 c30 = _mm256_setzero_ps(), c31 = _mm256_setzero_ps();

    for (int k = 0; k < K; k++) {
        const __m256 b0 = _mm256_loadu_ps(Bp + k * NR_F);
        const __m256 b1 = _mm256_loadu_ps(Bp + k * NR_F + 8);
        const __m256 a0 = _mm256_broadcast_ss(A + 0 * lda + k);
        const __m256 a1 = _mm256_broadcast_ss(A + 1 * lda + k);
        const __m256 a2 = _mm256_broadcast_ss(A + 2 * lda + k);
        const __m256 a3 = _mm256_broadcast_ss(A + 3 * lda + k);
        c00 = _mm256_fmadd_ps(a0, b0, c00);  c01 = _mm256_fmadd_ps(a0, b1, c01);
        c10 = _mm256_fmadd_ps(a1, b0, c10);  c11 = _mm256_fmadd_ps(a1, b1, c11);
        c20 = _mm256_fmadd_ps(a2, b0, c20);  c21 = _mm256_fmadd_ps(a2, b1, c21);
        c30 = _mm256_fmadd_ps(a3, b0, c30);  c31 = _mm256_fmadd_ps(a3, b1, c31);
    }

    _mm256_storeu_ps(C + 0 * ldc + 0, c00);  _mm256_storeu_ps(C + 0 * ldc + 8, c01);
    _mm256_storeu_ps(C + 1 * ldc + 0, c10);  _mm256_storeu_ps(C + 1 * ldc + 8, c11);
    _mm256_storeu_ps(C + 2 * ldc + 0, c20);  _mm256_storeu_ps(C + 2 * ldc + 8, c21);
    _mm256_storeu_ps(C + 3 * ldc + 0, c30);  _mm256_storeu_ps(C + 3 * ldc + 8, c31);
}

// Public GEMM: A [M×K] row-major × Bp (pre-packed NR_F panels) → C [M×N].
// M must be multiple of MR_F=4, N of NR_F=16.
void fastface_gemm_fp32(const float* A, const float* Bp, float* C,
                         int M, int K, int N)
{
    int nt = N / NR_F;
    #pragma omp parallel for schedule(static) collapse(2)
    for (int i = 0; i < M; i += MR_F) {
        for (int t = 0; t < nt; t++) {
            microkernel_4x16_fp32(
                A + i * K,
                Bp + t * K * NR_F,
                K, K,
                C + i * N + t * NR_F, N);
        }
    }
}

// GEMM with tail-pad: if N not divisible by NR_F, pad B with zeros internally.
// For convenience we expose this as the main entry.
void fastface_gemm_fp32_safe(const float* A, const float* B, float* C,
                              int M, int K, int N)
{
    int N_padded = (N + NR_F - 1) & ~(NR_F - 1);
    int M_padded = (M + MR_F - 1) & ~(MR_F - 1);
    // Quick path
    if (M % MR_F == 0 && N % NR_F == 0) {
        // Pack B with a small stack allocation or heap if large
        float* Bp = (float*)malloc((size_t)K * N_padded * sizeof(float));
        for (int k = 0; k < K; k++) {
            for (int j = 0; j < N; j++) Bp[k * N + j] = B[k * N + j];
        }
        // Pack properly
        int nt = N / NR_F;
        float* Bp2 = (float*)malloc((size_t)K * N * sizeof(float));
        for (int t = 0; t < nt; t++) {
            for (int k = 0; k < K; k++) {
                for (int j = 0; j < NR_F; j++) {
                    Bp2[t * K * NR_F + k * NR_F + j] = B[k * N + t * NR_F + j];
                }
            }
        }
        fastface_gemm_fp32(A, Bp2, C, M, K, N);
        free(Bp); free(Bp2);
        return;
    }
    // Slow path for non-aligned — fallback
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float s = 0;
            for (int k = 0; k < K; k++) s += A[i * K + k] * B[k * N + j];
            C[i * N + j] = s;
        }
    }
}
