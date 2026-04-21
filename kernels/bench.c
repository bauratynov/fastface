// Benchmark harness v2: correct VNNI packing.
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <windows.h>

void compute_col_sums(const int8_t* B, int K, int N, int32_t* col_sums);
void pack_A_to_u8(const int8_t* A, int M, int K, uint8_t* Au);
void pack_B_vnni(const int8_t* B, int K, int N, int8_t* Bp);
void fastface_gemm_i8(const uint8_t* Au, const int8_t* Bp, int32_t* C, int M, int K, int N);
void fastface_gemm_i8_finalize(int32_t* C, int M, int N, const int32_t* col_sums);
void fastface_gemm_i8_ref(const int8_t* A, const int8_t* B, int32_t* C, int M, int K, int N);

static double now_s(void) {
    LARGE_INTEGER q, f;
    QueryPerformanceCounter(&q);
    QueryPerformanceFrequency(&f);
    return (double)q.QuadPart / (double)f.QuadPart;
}

int main(int argc, char** argv) {
    int M = 784, K = 576, N = 128;
    if (argc == 4) { M = atoi(argv[1]); K = atoi(argv[2]); N = atoi(argv[3]); }

    printf("FastFace INT8 GEMM benchmark v2\n");
    printf("M=%d K=%d N=%d  (arcface mid-layer: 28x28 spatial, 3x3x64 -> 128)\n\n", M, K, N);

    int8_t*  A  = (int8_t*)_aligned_malloc((size_t)M * K, 64);
    int8_t*  B  = (int8_t*)_aligned_malloc((size_t)K * N, 64);
    uint8_t* Au = (uint8_t*)_aligned_malloc((size_t)M * K, 64);
    int8_t*  Bp = (int8_t*)_aligned_malloc((size_t)K * N, 64);
    int32_t* C  = (int32_t*)_aligned_malloc((size_t)M * N * sizeof(int32_t), 64);
    int32_t* col_sums = (int32_t*)_aligned_malloc((size_t)N * sizeof(int32_t), 64);

    srand(42);
    for (int i = 0; i < M * K; i++) A[i] = (int8_t)((rand() & 0xFF) - 128);
    for (int i = 0; i < K * N; i++) B[i] = (int8_t)((rand() & 0xFF) - 128);

    pack_A_to_u8(A, M, K, Au);
    pack_B_vnni(B, K, N, Bp);
    compute_col_sums(B, K, N, col_sums);

    // Correctness on small sub-problem
    printf("--- Correctness (M=%d, full K, full N) ---\n", 16);
    int Msub = 16;
    int32_t* Csub = (int32_t*)malloc((size_t)Msub * N * sizeof(int32_t));
    int32_t* Cref = (int32_t*)malloc((size_t)Msub * N * sizeof(int32_t));
    fastface_gemm_i8(Au, Bp, Csub, Msub, K, N);
    fastface_gemm_i8_finalize(Csub, Msub, N, col_sums);
    fastface_gemm_i8_ref(A, B, Cref, Msub, K, N);
    int mismatches = 0;
    int32_t max_diff = 0;
    for (int i = 0; i < Msub * N; i++) {
        int32_t d = Csub[i] > Cref[i] ? Csub[i] - Cref[i] : Cref[i] - Csub[i];
        if (d > max_diff) max_diff = d;
        if (d != 0) mismatches++;
    }
    printf("Mismatches: %d / %d, max_diff=%d  %s\n\n",
           mismatches, Msub * N, max_diff,
           (mismatches == 0) ? "BIT-EXACT" : "FAIL");
    free(Csub); free(Cref);

    if (mismatches != 0) {
        printf("Correctness failed — skipping perf bench\n");
        return 1;
    }

    // Warm-up
    for (int i = 0; i < 3; i++) {
        fastface_gemm_i8(Au, Bp, C, M, K, N);
    }

    const int N_ITER = 200;
    double t0 = now_s();
    for (int it = 0; it < N_ITER; it++) {
        fastface_gemm_i8(Au, Bp, C, M, K, N);
    }
    double t_avg = (now_s() - t0) / N_ITER;

    double gops = (2.0 * (double)M * K * N) / 1e9;
    double gops_s = gops / t_avg;

    printf("--- Performance ---\n");
    printf("GEMM avg:     %.3f ms  (%.2f GOps/s)\n", t_avg * 1000, gops_s);
    printf("\n--- Context ---\n");
    printf("i7-13700 AVX-VNNI theoretical peak (8 P-cores @ 4.2 GHz): ~1075 GOps/s\n");
    printf("Realistic sustained (memory+compute mixed): ~250-500 GOps/s\n");
    printf("Percent of theoretical peak: %.1f%%\n", 100.0 * gops_s / 1075.0);

    _aligned_free(A); _aligned_free(B); _aligned_free(Au); _aligned_free(Bp);
    _aligned_free(C); _aligned_free(col_sums);
    return 0;
}
