// S45 microbench: batched conv vs sequential for one shape.
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <windows.h>

void pack_conv_weight_nhwc(
    const int8_t* weight, int Cout, int Cin, int Kh, int Kw, int K_padded,
    int8_t* w_rowmajor, int8_t* w_packed, int32_t* col_sums);
void fastface_conv2d_i8_nhwc(
    const int8_t* input, int Cin, int H_in, int W_in,
    int Cout, int Kh, int Kw, int stride, int pad,
    const int8_t* weight_packed, const int32_t* col_sums,
    int H_out, int W_out, int32_t* output, uint8_t* scratch_Au);
void fastface_conv2d_i8_nhwc_batched(
    const int8_t* input, int B, int Cin, int H_in, int W_in,
    int Cout, int Kh, int Kw, int stride, int pad,
    const int8_t* weight_packed, const int32_t* col_sums,
    int H_out, int W_out, int32_t* output, uint8_t* scratch_Au);

static double now_s(void) {
    LARGE_INTEGER q, f; QueryPerformanceCounter(&q); QueryPerformanceFrequency(&f);
    return (double)q.QuadPart / (double)f.QuadPart;
}

int main(int argc, char** argv) {
    int B    = argc > 1 ? atoi(argv[1]) : 8;
    int H    = argc > 2 ? atoi(argv[2]) : 14;
    int W    = argc > 3 ? atoi(argv[3]) : 14;
    int Cin  = argc > 4 ? atoi(argv[4]) : 512;
    int Cout = argc > 5 ? atoi(argv[5]) : 512;
    int Kh   = argc > 6 ? atoi(argv[6]) : 3;
    int Kw   = argc > 7 ? atoi(argv[7]) : 3;
    int stride = 1, pad = (Kh == 3 ? 1 : 0);
    int H_out = (H + 2*pad - Kh)/stride + 1;
    int W_out = (W + 2*pad - Kw)/stride + 1;
    int K_real = Cin*Kh*Kw;
    int K_padded = (K_real + 3) & ~3;

    size_t in_sz = (size_t)B * H * W * Cin;
    size_t out_sz = (size_t)B * H_out * W_out * Cout;
    size_t w_sz = (size_t)Cout * Cin * Kh * Kw;
    size_t scratch_sz = (size_t)B * H_out * W_out * K_padded;

    int8_t* input = _aligned_malloc(in_sz, 64);
    int8_t* weight = _aligned_malloc(w_sz, 64);
    int8_t* w_row = _aligned_malloc((size_t)K_padded * Cout, 64);
    int8_t* w_pack = _aligned_malloc((size_t)K_padded * Cout, 64);
    int32_t* col_sums = _aligned_malloc((size_t)Cout * sizeof(int32_t), 64);
    int32_t* out1 = _aligned_malloc(out_sz * sizeof(int32_t), 64);
    int32_t* out2 = _aligned_malloc(out_sz * sizeof(int32_t), 64);
    uint8_t* scratch = _aligned_malloc(scratch_sz, 64);

    srand(42);
    for (size_t i = 0; i < in_sz; i++) input[i] = (int8_t)((rand() % 255) - 127);
    for (size_t i = 0; i < w_sz; i++) weight[i] = (int8_t)((rand() % 255) - 127);
    pack_conv_weight_nhwc(weight, Cout, Cin, Kh, Kw, K_padded, w_row, w_pack, col_sums);

    printf("Shape: B=%d H=%d W=%d Cin=%d Cout=%d %dx%d s=%d p=%d  H_out=%d W_out=%d K_padded=%d\n",
           B, H, W, Cin, Cout, Kh, Kw, stride, pad, H_out, W_out, K_padded);
    printf("Input:  %.1f MB  Scratch: %.1f MB  Output: %.1f MB  Weights: %.1f MB\n",
           in_sz/1024./1024., scratch_sz/1024./1024.,
           out_sz*4/1024./1024., w_sz/1024./1024.);

    const int ITERS = 50;
    // Warmup
    for (int it = 0; it < 3; it++) {
        for (int b = 0; b < B; b++) {
            fastface_conv2d_i8_nhwc(input + (size_t)b*Cin*H*W, Cin, H, W,
                Cout, Kh, Kw, stride, pad, w_pack, col_sums, H_out, W_out,
                out1 + (size_t)b*Cout*H_out*W_out, scratch);
        }
    }

    double best_seq = 1e9;
    for (int trial = 0; trial < 5; trial++) {
        double t0 = now_s();
        for (int it = 0; it < ITERS; it++) {
            for (int b = 0; b < B; b++) {
                fastface_conv2d_i8_nhwc(input + (size_t)b*Cin*H*W, Cin, H, W,
                    Cout, Kh, Kw, stride, pad, w_pack, col_sums, H_out, W_out,
                    out1 + (size_t)b*Cout*H_out*W_out, scratch);
            }
        }
        double dt = (now_s() - t0) / ITERS * 1000.0;
        if (dt < best_seq) best_seq = dt;
    }

    // Warmup batched
    for (int it = 0; it < 3; it++) {
        fastface_conv2d_i8_nhwc_batched(input, B, Cin, H, W,
            Cout, Kh, Kw, stride, pad, w_pack, col_sums, H_out, W_out,
            out2, scratch);
    }
    double best_batched = 1e9;
    for (int trial = 0; trial < 5; trial++) {
        double t0 = now_s();
        for (int it = 0; it < ITERS; it++) {
            fastface_conv2d_i8_nhwc_batched(input, B, Cin, H, W,
                Cout, Kh, Kw, stride, pad, w_pack, col_sums, H_out, W_out,
                out2, scratch);
        }
        double dt = (now_s() - t0) / ITERS * 1000.0;
        if (dt < best_batched) best_batched = dt;
    }

    // Correctness check
    int diffs = 0;
    for (size_t i = 0; i < out_sz; i++) {
        if (out1[i] != out2[i]) {
            if (diffs < 5) printf("  diff[%zu]: %d vs %d\n", i, out1[i], out2[i]);
            diffs++;
        }
    }
    printf("Correctness: %s (diffs=%d / %zu)\n",
           diffs == 0 ? "PASS" : "FAIL", diffs, out_sz);

    printf("Sequential (B loop) : %.3f ms total = %.3f ms/face\n", best_seq, best_seq / B);
    printf("Batched GEMM        : %.3f ms total = %.3f ms/face\n", best_batched, best_batched / B);
    printf("Speedup batched     : %.2fx\n", best_seq / best_batched);

    return 0;
}
