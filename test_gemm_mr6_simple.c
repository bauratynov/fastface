#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

void pack_B_fp32(const float* B, int K, int N, float* Bp);
void fastface_gemm_fp32(const float* A, const float* Bp, float* C, int M, int K, int N);
void fastface_gemm_fp32_mr6(const float* A, const float* Bp, float* C, int M, int K, int N);

int main(void) {
    int M = 12, K = 32, N = 16;
    printf("alloc...\n"); fflush(stdout);
    float* A  = (float*)_aligned_malloc((size_t)M * K * sizeof(float), 64);
    float* B  = (float*)_aligned_malloc((size_t)K * N * sizeof(float), 64);
    float* Bp = (float*)_aligned_malloc((size_t)K * N * sizeof(float), 64);
    float* C  = (float*)_aligned_malloc((size_t)M * N * sizeof(float), 64);
    printf("fill...\n"); fflush(stdout);
    srand(1);
    for (int i = 0; i < M*K; i++) A[i] = (rand()%200 - 100) / 100.0f;
    for (int i = 0; i < K*N; i++) B[i] = (rand()%200 - 100) / 100.0f;
    memset(C, 0, M * N * sizeof(float));
    printf("pack...\n"); fflush(stdout);
    pack_B_fp32(B, K, N, Bp);
    printf("gemm...\n"); fflush(stdout);
    fastface_gemm_fp32_mr6(A, Bp, C, M, K, N);
    printf("C[0]=%.4f C[1]=%.4f C[10]=%.4f C[63]=%.4f done\n",
            C[0], C[1], C[10], C[M*N - 1]);
    _aligned_free(A); _aligned_free(B); _aligned_free(Bp); _aligned_free(C);
    return 0;
}
