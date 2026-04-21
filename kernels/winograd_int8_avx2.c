// S35 phase 2 — INT8 Winograd F(2,3) with AVX2 inner GEMM.
//
// Layout:
//   U[16 * Cout * Cin] int16 — pre-transformed weights (G' g G'^T), done offline.
//     order: [i*4+j, oc, ci] with ci innermost (contiguous for SIMD reductions).
//   V[16 * n_tiles * Cin] int16 — input transform output, ci innermost.
//   M[16 * n_tiles * Cout] int32 — per-(i,j) GEMM result.
//
// Phase 2 scope: correctness bit-exact vs scalar ref + bench on 56x56x128x128.

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <omp.h>
#include <immintrin.h>

// Forward decl from ref.c — we reuse the weight transform since it's offline cost.
void winograd_int8_transform_weights_ref(
    const int8_t* g, int Cout, int Cin, int16_t* u);

// Input transform: NHWC [H, W, Cin] int8 -> V[16, n_tiles_h * n_tiles_w, Cin] int16
// Tiles are 4x4 overlapping by 2 (stride-2 sliding over padded input).
// Output row count: n_tiles_h = H/2, n_tiles_w = W/2 (stride 1 pad 1, H and W even).
static void winograd_int8_input_transform_avx2(
    const int8_t* input,
    int Cin, int H, int W,
    int16_t* V)
{
    int n_th = H / 2, n_tw = W / 2;

    #pragma omp parallel for schedule(static) collapse(2)
    for (int th = 0; th < n_th; th++) {
        for (int tw = 0; tw < n_tw; tw++) {
            int tile = th * n_tw + tw;
            int ih0 = th * 2 - 1, iw0 = tw * 2 - 1;

            // Load 4x4 tile at granularity of 16 int16 per channel chunk.
            // To keep code simple, handle each ci as a block of 16 via loops.
            int ci = 0;
            for (; ci + 16 <= Cin; ci += 16) {
                __m256i d[4][4];
                for (int ti = 0; ti < 4; ti++) {
                    for (int tj = 0; tj < 4; tj++) {
                        int ih = ih0 + ti, iw = iw0 + tj;
                        if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                            __m128i i8 = _mm_loadu_si128(
                                (const __m128i*)(input + ((size_t)ih * W + iw) * Cin + ci));
                            d[ti][tj] = _mm256_cvtepi8_epi16(i8);
                        } else {
                            d[ti][tj] = _mm256_setzero_si256();
                        }
                    }
                }
                // B^T d B transform. First B^T from left (columns):
                __m256i t0[4][4], v[4][4];
                for (int j = 0; j < 4; j++) {
                    t0[0][j] = _mm256_sub_epi16(d[0][j], d[2][j]);
                    t0[1][j] = _mm256_add_epi16(d[1][j], d[2][j]);
                    t0[2][j] = _mm256_sub_epi16(d[2][j], d[1][j]);
                    t0[3][j] = _mm256_sub_epi16(d[1][j], d[3][j]);
                }
                // Then B from right (rows):
                for (int i = 0; i < 4; i++) {
                    v[i][0] = _mm256_sub_epi16(t0[i][0], t0[i][2]);
                    v[i][1] = _mm256_add_epi16(t0[i][1], t0[i][2]);
                    v[i][2] = _mm256_sub_epi16(t0[i][2], t0[i][1]);
                    v[i][3] = _mm256_sub_epi16(t0[i][1], t0[i][3]);
                }
                // Store V[i*4+j, tile, ci..ci+16]
                for (int i = 0; i < 4; i++) {
                    for (int j = 0; j < 4; j++) {
                        int16_t* dst = V + ((size_t)(i * 4 + j) * (n_th * n_tw) + tile) * Cin + ci;
                        _mm256_storeu_si256((__m256i*)dst, v[i][j]);
                    }
                }
            }
            // Scalar tail for remaining ci
            for (; ci < Cin; ci++) {
                int16_t d[4][4];
                for (int ti = 0; ti < 4; ti++) {
                    for (int tj = 0; tj < 4; tj++) {
                        int ih = ih0 + ti, iw = iw0 + tj;
                        if (ih >= 0 && ih < H && iw >= 0 && iw < W)
                            d[ti][tj] = (int16_t)input[((size_t)ih * W + iw) * Cin + ci];
                        else
                            d[ti][tj] = 0;
                    }
                }
                int16_t t0[4][4], v[4][4];
                for (int j = 0; j < 4; j++) {
                    t0[0][j] = d[0][j] - d[2][j];
                    t0[1][j] = d[1][j] + d[2][j];
                    t0[2][j] = d[2][j] - d[1][j];
                    t0[3][j] = d[1][j] - d[3][j];
                }
                for (int i = 0; i < 4; i++) {
                    v[i][0] = t0[i][0] - t0[i][2];
                    v[i][1] = t0[i][1] + t0[i][2];
                    v[i][2] = t0[i][2] - t0[i][1];
                    v[i][3] = t0[i][1] - t0[i][3];
                }
                for (int i = 0; i < 4; i++) {
                    for (int j = 0; j < 4; j++) {
                        V[((size_t)(i * 4 + j) * (n_th * n_tw) + tile) * Cin + ci] = v[i][j];
                    }
                }
            }
        }
    }
}

// 16 GEMMs: for each (i,j) in 0..16, compute M[ij, tile, oc] = sum_ci V[ij, tile, ci] * U[ij, oc, ci]
// Inner reduction uses _mm256_madd_epi16 (int16*int16 pair-sum -> int32).
static void winograd_int8_gemm_avx2(
    const int16_t* V,     // [16, n_tiles, Cin]
    const int16_t* U,     // [16, Cout, Cin] — packed such that [ij, oc, ci] is Cin-innermost
    int Cin, int n_tiles, int Cout,
    int32_t* M)           // [16, n_tiles, Cout]
{
    #pragma omp parallel for schedule(static) collapse(2)
    for (int ij = 0; ij < 16; ij++) {
        for (int tile = 0; tile < n_tiles; tile++) {
            const int16_t* v_row = V + ((size_t)ij * n_tiles + tile) * Cin;
            for (int oc = 0; oc < Cout; oc++) {
                const int16_t* u_row = U + ((size_t)ij * Cout + oc) * Cin;
                __m256i acc = _mm256_setzero_si256();
                int ci = 0;
                for (; ci + 16 <= Cin; ci += 16) {
                    __m256i v = _mm256_loadu_si256((const __m256i*)(v_row + ci));
                    __m256i u = _mm256_loadu_si256((const __m256i*)(u_row + ci));
                    acc = _mm256_add_epi32(acc, _mm256_madd_epi16(v, u));
                }
                // Horizontal sum of 8 int32 lanes
                __m128i lo = _mm256_castsi256_si128(acc);
                __m128i hi = _mm256_extracti128_si256(acc, 1);
                __m128i s = _mm_add_epi32(lo, hi);
                s = _mm_add_epi32(s, _mm_shuffle_epi32(s, _MM_SHUFFLE(2, 3, 0, 1)));
                s = _mm_add_epi32(s, _mm_shuffle_epi32(s, _MM_SHUFFLE(1, 0, 3, 2)));
                int32_t sum = _mm_cvtsi128_si32(s);
                // Tail
                for (; ci < Cin; ci++) sum += (int32_t)v_row[ci] * (int32_t)u_row[ci];
                M[((size_t)ij * n_tiles + tile) * Cout + oc] = sum;
            }
        }
    }
}

// Output transform: M[16, n_tiles, Cout] -> output[H, W, Cout] as int32 (NHWC).
// For each tile, apply A^T M A to get 2x2 int32 tile. Output is 4*Y (G' = 2G scale).
static void winograd_int8_output_transform(
    const int32_t* M,
    int n_tiles, int Cout,
    int H_out, int W_out,
    int32_t* output)
{
    int n_th = H_out / 2, n_tw = W_out / 2;

    #pragma omp parallel for schedule(static) collapse(2)
    for (int th = 0; th < n_th; th++) {
        for (int tw = 0; tw < n_tw; tw++) {
            int tile = th * n_tw + tw;
            int oh = th * 2, ow = tw * 2;
            for (int oc = 0; oc < Cout; oc++) {
                int32_t Mtile[4][4];
                for (int i = 0; i < 4; i++)
                    for (int j = 0; j < 4; j++)
                        Mtile[i][j] = M[((size_t)(i * 4 + j) * n_tiles + tile) * Cout + oc];
                // A^T M
                int32_t t[2][4];
                for (int j = 0; j < 4; j++) {
                    t[0][j] = Mtile[0][j] + Mtile[1][j] + Mtile[2][j];
                    t[1][j] = Mtile[1][j] - Mtile[2][j] - Mtile[3][j];
                }
                // (A^T M) A
                int32_t y[2][2];
                for (int i = 0; i < 2; i++) {
                    y[i][0] = t[i][0] + t[i][1] + t[i][2];
                    y[i][1] = t[i][1] - t[i][2] - t[i][3];
                }
                output[((size_t)(oh + 0) * W_out + (ow + 0)) * Cout + oc] = y[0][0];
                output[((size_t)(oh + 0) * W_out + (ow + 1)) * Cout + oc] = y[0][1];
                output[((size_t)(oh + 1) * W_out + (ow + 0)) * Cout + oc] = y[1][0];
                output[((size_t)(oh + 1) * W_out + (ow + 1)) * Cout + oc] = y[1][1];
            }
        }
    }
}

// Public entry: AVX2 Winograd F(2,3) conv. Stride 1 pad 1. H and W even.
void fastface_conv2d_i8_winograd_f23_avx2(
    const int8_t* input, int Cin, int H, int W,
    int Cout,
    const int16_t* U_wino,  // [16, Cout, Cin]
    int16_t* V_scratch,     // [16, n_tiles, Cin]
    int32_t* M_scratch,     // [16, n_tiles, Cout]
    int32_t* output)
{
    int n_th = H / 2, n_tw = W / 2;
    int n_tiles = n_th * n_tw;

    winograd_int8_input_transform_avx2(input, Cin, H, W, V_scratch);
    winograd_int8_gemm_avx2(V_scratch, U_wino, Cin, n_tiles, Cout, M_scratch);
    winograd_int8_output_transform(M_scratch, n_tiles, Cout, H, W, output);
}

#ifdef WINOGRAD_I8_AVX2_TEST
// Forward decls from ref impl
extern void winograd_int8_transform_weights_ref(const int8_t* g, int Cout, int Cin, int16_t* u);
extern void fastface_conv2d_i8_winograd_f23_ref(
    const int8_t* input, int Cin, int H, int W, int Cout,
    const int16_t* u_weights, int32_t* output);

#include <windows.h>
static double now_s(void) {
    LARGE_INTEGER q, f; QueryPerformanceCounter(&q); QueryPerformanceFrequency(&f);
    return (double)q.QuadPart / (double)f.QuadPart;
}

// Repack U from reference layout [16, Cout, Cin] stored as [i*4+j, Cout, Cin]
// The ref stores it in [i*4+j, Cout*Cin] which is the same flat layout: u_flat[ij*Cout*Cin + oc*Cin + ci].
// So no repack needed — layouts match.

// Declarations for the VNNI im2col path we bench against.
extern void pack_conv_weight_nhwc(
    const int8_t* weight, int Cout, int Cin, int Kh, int Kw, int K_padded,
    int8_t* w_rowmajor, int8_t* w_packed, int32_t* col_sums);
extern void fastface_conv2d_i8_nhwc(
    const int8_t* input, int Cin, int H_in, int W_in,
    int Cout, int Kh, int Kw, int stride, int pad,
    const int8_t* weight_packed, const int32_t* col_sums,
    int H_out, int W_out, int32_t* output, uint8_t* scratch_Au);

int main(void) {
    const int H = 56, W = 56, Cin = 128, Cout = 128;
    size_t n_in   = (size_t)H * W * Cin;
    size_t n_w    = (size_t)Cout * Cin * 9;
    size_t n_out  = (size_t)H * W * Cout;
    int n_tiles = (H / 2) * (W / 2);

    int8_t*  input  = (int8_t*) _aligned_malloc(n_in,  64);
    int8_t*  weight = (int8_t*) _aligned_malloc(n_w,   64);
    int16_t* U      = (int16_t*)_aligned_malloc((size_t)16 * Cout * Cin * sizeof(int16_t), 64);
    int16_t* V      = (int16_t*)_aligned_malloc((size_t)16 * n_tiles * Cin * sizeof(int16_t), 64);
    int32_t* M      = (int32_t*)_aligned_malloc((size_t)16 * n_tiles * Cout * sizeof(int32_t), 64);
    int32_t* out_avx2 = (int32_t*)_aligned_malloc(n_out * sizeof(int32_t), 64);
    int32_t* out_ref  = (int32_t*)_aligned_malloc(n_out * sizeof(int32_t), 64);
    int32_t* out_vnni = (int32_t*)_aligned_malloc(n_out * sizeof(int32_t), 64);

    srand(42);
    for (size_t i = 0; i < n_in; i++) input[i]  = (int8_t)((rand() % 255) - 127);
    for (size_t i = 0; i < n_w;  i++) weight[i] = (int8_t)((rand() % 255) - 127);

    // Transform weights for Winograd
    winograd_int8_transform_weights_ref(weight, Cout, Cin, U);

    // 1. Correctness check: AVX2 vs scalar reference
    fastface_conv2d_i8_winograd_f23_ref(input, Cin, H, W, Cout, U, out_ref);
    fastface_conv2d_i8_winograd_f23_avx2(input, Cin, H, W, Cout, U, V, M, out_avx2);

    int32_t max_err = 0; size_t n_bad = 0;
    for (size_t i = 0; i < n_out; i++) {
        int32_t e = out_avx2[i] - out_ref[i];
        if (e < 0) e = -e;
        if (e > max_err) max_err = e;
        if (e != 0) n_bad++;
    }
    printf("AVX2 vs scalar ref correctness (56x56x128x128):\n");
    printf("  non-exact: %zu / %zu\n  max err: %d\n",
           n_bad, n_out, max_err);
    if (max_err == 0)
        printf("PASS bit-exact.\n\n");
    else {
        printf("FAIL — debug AVX2 impl before benching.\n");
        return 1;
    }

    // 2. Bench AVX2 Winograd
    const int BENCH_ITERS = 100;
    double wino_best = 1e9;
    for (int trial = 0; trial < 5; trial++) {
        double t0 = now_s();
        for (int k = 0; k < BENCH_ITERS; k++) {
            fastface_conv2d_i8_winograd_f23_avx2(input, Cin, H, W, Cout, U, V, M, out_avx2);
        }
        double dt = (now_s() - t0) / BENCH_ITERS * 1000.0;
        if (dt < wino_best) wino_best = dt;
    }

    // 3. Bench VNNI im2col for comparison (same shape 3x3 stride 1 pad 1)
    int Kr = Cin * 3 * 3;
    int Kp = (Kr + 3) & ~3;
    int8_t* w_row   = (int8_t*)_aligned_malloc((size_t)Kp * Cout, 64);
    int8_t* w_pack  = (int8_t*)_aligned_malloc((size_t)Kp * Cout, 64);
    int32_t* col_s  = (int32_t*)_aligned_malloc((size_t)Cout * sizeof(int32_t), 64);
    uint8_t* scr_im = (uint8_t*)_aligned_malloc((size_t)H * W * Kp + 64, 64);
    pack_conv_weight_nhwc(weight, Cout, Cin, 3, 3, Kp, w_row, w_pack, col_s);

    double vnni_best = 1e9;
    for (int trial = 0; trial < 5; trial++) {
        double t0 = now_s();
        for (int k = 0; k < BENCH_ITERS; k++) {
            fastface_conv2d_i8_nhwc(input, Cin, H, W, Cout, 3, 3, 1, 1,
                                    w_pack, col_s, H, W, out_vnni, scr_im);
        }
        double dt = (now_s() - t0) / BENCH_ITERS * 1000.0;
        if (dt < vnni_best) vnni_best = dt;
    }

    printf("Bench per-conv (56x56x128x128 3x3 stride 1 pad 1, 100 iter x 5 trials, 8 threads):\n");
    printf("  Winograd AVX2: %.3f ms\n", wino_best);
    printf("  VNNI im2col:   %.3f ms\n", vnni_best);
    printf("  Winograd/VNNI speedup: %.2fx\n", vnni_best / wino_best);

    if (wino_best < vnni_best * 0.77) // >=1.3x
        printf("VERDICT: proceed to phase 3 integration.\n");
    else if (wino_best < vnni_best * 0.95)
        printf("VERDICT: marginal. Further optimization needed before integration.\n");
    else
        printf("VERDICT: ABORT moonshot. Pivot to per-channel cos-sim instead.\n");

    _aligned_free(input); _aligned_free(weight);
    _aligned_free(U); _aligned_free(V); _aligned_free(M);
    _aligned_free(out_avx2); _aligned_free(out_ref); _aligned_free(out_vnni);
    _aligned_free(w_row); _aligned_free(w_pack); _aligned_free(col_s); _aligned_free(scr_im);
    return 0;
}
#endif
