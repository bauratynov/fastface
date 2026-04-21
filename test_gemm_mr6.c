// Sanity test: fastface_gemm_fp32_mr6 vs naive matmul, and vs fastface_gemm_fp32 (MR=4).
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <windows.h>

void pack_B_fp32(const float* B, int K, int N, float* Bp);
void fastface_gemm_fp32(const float* A, const float* Bp, float* C, int M, int K, int N);
void fastface_gemm_fp32_mr6(const float* A, const float* Bp, float* C, int M, int K, int N);

static double now_s(void) {
    LARGE_INTEGER q, f; QueryPerformanceCounter(&q); QueryPerformanceFrequency(&f);
    return (double)q.QuadPart / (double)f.QuadPart;
}

static int test_case(int M, int K, int N) {
    float* A  = (float*)_aligned_malloc((size_t)M * K * sizeof(float), 64);
    float* B  = (float*)_aligned_malloc((size_t)K * N * sizeof(float), 64);
    float* Bp = (float*)_aligned_malloc((size_t)K * N * sizeof(float), 64);
    float* C_ref = (float*)_aligned_malloc((size_t)M * N * sizeof(float), 64);
    float* C_mr4 = (float*)_aligned_malloc((size_t)M * N * sizeof(float), 64);
    float* C_mr6 = (float*)_aligned_malloc((size_t)M * N * sizeof(float), 64);
    memset(C_ref, 0, (size_t)M * N * sizeof(float));

    srand(12345 + M + K + N);
    for (int i = 0; i < M * K; i++) A[i] = (rand() % 200 - 100) / 100.0f;
    for (int i = 0; i < K * N; i++) B[i] = (rand() % 200 - 100) / 100.0f;
    memset(C_mr4, 0, M * N * sizeof(float));
    memset(C_mr6, 0, M * N * sizeof(float));

    // Naive reference
    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++) {
            float s = 0;
            for (int k = 0; k < K; k++) s += A[m*K + k] * B[k*N + n];
            C_ref[m*N + n] = s;
        }

    pack_B_fp32(B, K, N, Bp);
    memset(C_mr4, 0, M*N*sizeof(float));
    memset(C_mr6, 0, M*N*sizeof(float));
    int can_mr4 = (M % 4 == 0);
    if (can_mr4) fastface_gemm_fp32(A, Bp, C_mr4, M, K, N);
    fastface_gemm_fp32_mr6(A, Bp, C_mr6, M, K, N);

    float md4 = 0, md6 = 0;
    for (int i = 0; i < M * N; i++) {
        float d6 = fabsf(C_ref[i] - C_mr6[i]);
        if (d6 > md6) md6 = d6;
        if (can_mr4) {
            float d4 = fabsf(C_ref[i] - C_mr4[i]);
            if (d4 > md4) md4 = d4;
        }
    }

    // Timing (MR=6 only; MR=4 only when M%4==0)
    double best4 = 0, best6 = 1e9;
    for (int t = 0; t < 3; t++) {
        double t0 = now_s();
        for (int it = 0; it < 20; it++) fastface_gemm_fp32_mr6(A, Bp, C_mr6, M, K, N);
        double dt = (now_s() - t0) / 20;
        if (dt < best6) best6 = dt;
    }
    if (can_mr4) {
        best4 = 1e9;
        for (int t = 0; t < 3; t++) {
            double t0 = now_s();
            for (int it = 0; it < 20; it++) fastface_gemm_fp32(A, Bp, C_mr4, M, K, N);
            double dt = (now_s() - t0) / 20;
            if (dt < best4) best4 = dt;
        }
    }
    printf("M=%4d K=%4d N=%4d  mr4: %s (err %.2e)  mr6: %.3f ms (err %.2e)  %s\n",
           M, K, N,
           can_mr4 ? "avail" : "n/a",
           md4, best6 * 1000, md6,
           can_mr4 ? (best4 < best6 ? "MR4 faster" : "MR6 faster") : "");
    if (can_mr4) printf("    mr4 %.3f ms  mr6 %.3f ms  mr6/mr4 = %.2fx\n",
                        best4 * 1000, best6 * 1000, best4 / best6);
    fflush(stdout);

    _aligned_free(A); _aligned_free(B); _aligned_free(Bp);
    _aligned_free(C_ref); _aligned_free(C_mr4); _aligned_free(C_mr6);
    return (md4 > 1e-3 || md6 > 1e-3) ? 1 : 0;
}

int main(void) {
    int bad = 0;
    // Shapes from real Winograd GEMMs in our model
    bad += test_case(   4,  512,  512);
    bad += test_case(  16,  512,  512);
    bad += test_case(  52,  256,  256);
    bad += test_case( 196,  128,  256);
    bad += test_case( 784,   64,  128);
    bad += test_case(3136,   64,   64);
    bad += test_case(3136,    3,   64);
    // Odd-M cases (only mr6 handles these)
    bad += test_case(  10,  128,  128);
    bad += test_case(  25,  128,  128);
    bad += test_case(  49,  128,  128);
    printf("==== %d FAIL ====\n", bad);
    printf("==== %d FAILURES ====\n", bad);
    return bad;
}
