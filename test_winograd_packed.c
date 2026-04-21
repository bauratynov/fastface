// Debug: compare reference Winograd vs packed Winograd on a single Conv.
// Feed identical fp32 weights + input, check outputs match (max abs diff).
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>

void winograd_precompute_weights(const float* weight, int Cout, int Cin, float* U_wino);
void winograd_precompute_weights_packed(const float* weight, int Cout, int Cin,
                                         float* U_packed, float* scratch);
void fastface_winograd_conv_3x3_s1_p1(
    const float* input, int Cin, int H_in, int W_in,
    int Cout, const float* U_wino,
    float* output, float* V_wino, float* M_wino);
void fastface_winograd_conv_3x3_s1_p1_packed(
    const float* input, int Cin, int H_in, int W_in,
    int Cout, const float* U_packed,
    float* output, float* V_wino, float* M_wino);

static int test_case(int Cin, int Cout, int H, int W) {
    int tH = (H + 1) / 2, tW = (W + 1) / 2;
    int ntile_pad = ((tH * tW) + 3) & ~3;
    if (ntile_pad < 4) ntile_pad = 4;
    float* in_data = (float*)_aligned_malloc(H * W * Cin * sizeof(float), 64);
    float* weights = (float*)_aligned_malloc(Cout * Cin * 9 * sizeof(float), 64);
    float* U_ref   = (float*)_aligned_malloc(16 * Cin * Cout * sizeof(float), 64);
    float* U_pack  = (float*)_aligned_malloc(16 * Cin * Cout * sizeof(float), 64);
    float* tmp     = (float*)_aligned_malloc(16 * Cin * Cout * sizeof(float), 64);
    float* out_ref = (float*)_aligned_malloc(H * W * Cout * sizeof(float), 64);
    float* out_pk  = (float*)_aligned_malloc(H * W * Cout * sizeof(float), 64);
    float* V_ref   = (float*)_aligned_malloc(16 * ntile_pad * Cin * sizeof(float), 64);
    float* M_ref   = (float*)_aligned_malloc(16 * ntile_pad * Cout * sizeof(float), 64);
    float* V_pk    = (float*)_aligned_malloc(16 * ntile_pad * Cin * sizeof(float), 64);
    float* M_pk    = (float*)_aligned_malloc(16 * ntile_pad * Cout * sizeof(float), 64);

    srand(1);
    for (int i = 0; i < H * W * Cin; i++) in_data[i] = (rand() % 200 - 100) / 100.0f;
    for (int i = 0; i < Cout * Cin * 9; i++) weights[i] = (rand() % 200 - 100) / 100.0f;

    winograd_precompute_weights(weights, Cout, Cin, U_ref);
    winograd_precompute_weights_packed(weights, Cout, Cin, U_pack, tmp);

    // Compare U matrices (before GEMM). Reference U_ref is [16, Cin, Cout] row-major.
    // U_pack is the packed form: same logical content but reshuffled by pack_B_fp32.
    // We can't compare directly — just run through both kernels.

    memset(out_ref, 0, H * W * Cout * sizeof(float));
    memset(out_pk,  0, H * W * Cout * sizeof(float));

    fastface_winograd_conv_3x3_s1_p1(in_data, Cin, H, W, Cout, U_ref, out_ref, V_ref, M_ref);
    fastface_winograd_conv_3x3_s1_p1_packed(in_data, Cin, H, W, Cout, U_pack, out_pk, V_pk, M_pk);

    float maxdiff = 0; int maxpos = -1;
    for (int i = 0; i < H * W * Cout; i++) {
        float d = fabsf(out_ref[i] - out_pk[i]);
        if (d > maxdiff) { maxdiff = d; maxpos = i; }
    }
    printf("H=%d W=%d Cin=%d Cout=%d ntile_pad=%d  maxdiff = %g  at pos %d\n",
            H, W, Cin, Cout, ntile_pad, maxdiff, maxpos);

    _aligned_free(in_data); _aligned_free(weights);
    _aligned_free(U_ref); _aligned_free(U_pack); _aligned_free(tmp);
    _aligned_free(out_ref); _aligned_free(out_pk);
    _aligned_free(V_ref); _aligned_free(M_ref);
    _aligned_free(V_pk);  _aligned_free(M_pk);
    return maxdiff > 1e-3 ? 1 : 0;
}

int main(void) {
    int bad = 0;
    bad += test_case(16, 32, 8, 8);     // even H/W, small
    bad += test_case(3, 64, 112, 112);  // stem-like: Cin=3, H=W=112
    bad += test_case(64, 64, 56, 56);
    bad += test_case(64, 128, 56, 56);  // Cin != Cout
    bad += test_case(128, 128, 28, 28);
    bad += test_case(256, 256, 14, 14); // 14x14 ntile=49 (not div by 4)
    bad += test_case(256, 512, 14, 14);
    bad += test_case(512, 512, 7, 7);   // odd H/W (stage 5-ish)
    bad += test_case(512, 512, 4, 4);   // small spatial
    bad += test_case(512, 512, 2, 2);   // tiny, ntile=1 → pad to 4
    printf("==== %d FAIL ====\n", bad);
    return bad;
}
