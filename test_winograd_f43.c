// Correctness test: F(4,3) Winograd vs naive 3×3 conv.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

void winograd_precompute_weights_packed_f43(
    const float* weight, int Cout, int Cin, float* U_packed, float* scratch);
void fastface_winograd_conv_3x3_s1_p1_f43_packed_bias(
    const float* input, int Cin, int H_in, int W_in,
    int Cout, const float* U_packed, const float* bias,
    float* output, float* V_wino, float* M_wino);

// Naive 3×3 stride=1 pad=1 conv for ground truth.
static void naive_conv(
    const float* input, int Cin, int H, int W,
    int Cout, const float* weight, const float* bias,
    float* output)
{
    for (int oh = 0; oh < H; oh++) {
        for (int ow = 0; ow < W; ow++) {
            for (int co = 0; co < Cout; co++) {
                float s = bias ? bias[co] : 0.0f;
                for (int ci = 0; ci < Cin; ci++) {
                    for (int kh = 0; kh < 3; kh++) {
                        for (int kw = 0; kw < 3; kw++) {
                            int ih = oh + kh - 1;
                            int iw = ow + kw - 1;
                            if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                                float x = input[((size_t)ih * W + iw) * Cin + ci];
                                float w = weight[((size_t)co * Cin + ci) * 9 + kh * 3 + kw];
                                s += x * w;
                            }
                        }
                    }
                }
                output[((size_t)oh * W + ow) * Cout + co] = s;
            }
        }
    }
}

static int test(int Cin, int Cout, int H, int W) {
    int tH = (H + 3) / 4, tW = (W + 3) / 4;
    int ntile_p = (tH * tW + 3) & ~3;
    if (ntile_p < 4) ntile_p = 4;

    float* in_data = (float*)_aligned_malloc((size_t)H*W*Cin*sizeof(float), 64);
    float* w       = (float*)_aligned_malloc((size_t)Cout*Cin*9*sizeof(float), 64);
    float* bias    = (float*)_aligned_malloc((size_t)Cout*sizeof(float), 64);
    float* out_ref = (float*)_aligned_malloc((size_t)H*W*Cout*sizeof(float), 64);
    float* out_w   = (float*)_aligned_malloc((size_t)H*W*Cout*sizeof(float), 64);
    size_t u_bytes = (size_t)36*Cin*Cout*sizeof(float);
    float* U_pk    = (float*)_aligned_malloc(u_bytes, 64);
    float* tmp     = (float*)_aligned_malloc(u_bytes, 64);
    float* V_pk    = (float*)_aligned_malloc((size_t)36*ntile_p*Cin*sizeof(float), 64);
    float* M_pk    = (float*)_aligned_malloc((size_t)36*ntile_p*Cout*sizeof(float), 64);

    srand(12345 + Cin + Cout + H + W);
    for (int i = 0; i < H*W*Cin; i++) in_data[i] = (rand()%200 - 100) / 100.0f;
    for (int i = 0; i < Cout*Cin*9; i++) w[i] = (rand()%200 - 100) / 100.0f;
    for (int i = 0; i < Cout; i++) bias[i] = (rand()%200 - 100) / 100.0f;

    memset(out_ref, 0, H*W*Cout*sizeof(float));
    memset(out_w,   0, H*W*Cout*sizeof(float));
    naive_conv(in_data, Cin, H, W, Cout, w, bias, out_ref);
    winograd_precompute_weights_packed_f43(w, Cout, Cin, U_pk, tmp);
    fastface_winograd_conv_3x3_s1_p1_f43_packed_bias(
        in_data, Cin, H, W, Cout, U_pk, bias, out_w, V_pk, M_pk);

    float maxdiff = 0, rel_diff = 0;
    for (int i = 0; i < H*W*Cout; i++) {
        float d = fabsf(out_ref[i] - out_w[i]);
        if (d > maxdiff) maxdiff = d;
    }
    float max_abs_ref = 0;
    for (int i = 0; i < H*W*Cout; i++) if (fabsf(out_ref[i]) > max_abs_ref) max_abs_ref = fabsf(out_ref[i]);
    rel_diff = maxdiff / (max_abs_ref + 1e-9f);
    printf("H=%d W=%d Cin=%d Cout=%d  ntile_p=%d  maxdiff=%.2e  rel=%.2e  max|ref|=%.2e\n",
            H, W, Cin, Cout, ntile_p, maxdiff, rel_diff, max_abs_ref);
    fflush(stdout);

    _aligned_free(in_data); _aligned_free(w); _aligned_free(bias);
    _aligned_free(out_ref); _aligned_free(out_w);
    _aligned_free(U_pk); _aligned_free(tmp); _aligned_free(V_pk); _aligned_free(M_pk);
    return rel_diff > 1e-3f ? 1 : 0;
}

int main(void) {
    int bad = 0;
    bad += test(16, 32, 8, 8);       // small even
    bad += test(64, 64, 56, 56);     // large even
    bad += test(64, 128, 56, 56);
    bad += test(128, 128, 28, 28);
    bad += test(256, 256, 14, 14);   // 14 % 4 = 2 → tiles = ceil(14/4)=4, ntile=16
    bad += test(256, 512, 14, 14);
    bad += test(512, 512, 7, 7);     // 7 % 4 = 3 → tiles=2, ntile=4
    bad += test(512, 512, 4, 4);
    bad += test(512, 512, 2, 2);     // tiny
    bad += test(3, 64, 112, 112);    // stem
    printf("==== %d FAIL ====\n", bad);
    return bad;
}
