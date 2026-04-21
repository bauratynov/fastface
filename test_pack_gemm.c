// Sanity test: pack_B_fp32 + fastface_gemm_fp32 vs naive matmul
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

void pack_B_fp32(const float* B, int K, int N, float* Bp);
void fastface_gemm_fp32(const float* A, const float* Bp, float* C, int M, int K, int N);

int main(void) {
    int M = 8, K = 32, N = 16;
    float* A = (float*)_aligned_malloc(M * K * sizeof(float), 64);
    float* B = (float*)_aligned_malloc(K * N * sizeof(float), 64);
    float* Bp = (float*)_aligned_malloc(K * N * sizeof(float), 64);
    float* C_ref = (float*)calloc(M * N, sizeof(float));
    float* C_our = (float*)_aligned_malloc(M * N * sizeof(float), 64);

    srand(1);
    for (int i = 0; i < M * K; i++) A[i] = (rand() % 200 - 100) / 100.0f;
    for (int i = 0; i < K * N; i++) B[i] = (rand() % 200 - 100) / 100.0f;
    memset(C_our, 0, M * N * sizeof(float));

    // Naive reference: C[m, n] = Σ_k A[m, k] * B[k, n]
    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++) {
            float s = 0;
            for (int k = 0; k < K; k++) s += A[m * K + k] * B[k * N + n];
            C_ref[m * N + n] = s;
        }

    pack_B_fp32(B, K, N, Bp);
    fastface_gemm_fp32(A, Bp, C_our, M, K, N);

    float maxdiff = 0;
    for (int i = 0; i < M * N; i++) {
        float d = fabsf(C_ref[i] - C_our[i]);
        if (d > maxdiff) maxdiff = d;
    }
    printf("M=%d K=%d N=%d  maxdiff=%g  C_ref[0..4]=[%.4f %.4f %.4f %.4f %.4f]\n",
            M, K, N, maxdiff, C_ref[0], C_ref[1], C_ref[2], C_ref[3], C_ref[4]);
    printf("                            C_our[0..4]=[%.4f %.4f %.4f %.4f %.4f]\n",
            C_our[0], C_our[1], C_our[2], C_our[3], C_our[4]);
    return 0;
}
