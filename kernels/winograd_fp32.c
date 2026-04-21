// Session 16 — Winograd F(2, 3) convolution for FP32 3×3 stride=1 pad=1.
//
// F(2, 3) computes a 2x2 output tile from a 4x4 input tile using a 3x3 filter,
// with 16 multiplies vs 36 (direct) = 2.25× fewer multiplies.
// Matrices (standard F(2,3)):
//   G = [[1, 0, 0], [1/2, 1/2, 1/2], [1/2, -1/2, 1/2], [0, 0, 1]]          4x3
//   Bt = [[1, 0, -1, 0], [0, 1, 1, 0], [0, -1, 1, 0], [0, 1, 0, -1]]        4x4
//   At = [[1, 1, 1, 0], [0, 1, -1, -1]]                                     2x4
// Algorithm for one output tile:
//   U = G * g * G^T      shape 4x4  (weight domain, per (co, ci))
//   V = Bt * d * Bt^T    shape 4x4  (input domain, per (tile, ci))
//   M = U ⊙ V            shape 4x4  (elementwise, then sum over ci)
//   Y = At * M * At^T    shape 2x2  (output tile)
//
// Layout conventions for batched ops:
//   weight_wino: [16, Cin, Cout]   — 16 flattened xi*4+nu, with Cin/Cout matrix per (xi,nu)
//   input_wino:  [16, num_tiles, Cin]
//   M:           [16, num_tiles, Cout]
//   -> For each of 16 (xi, nu) positions, do one GEMM: [num_tiles, Cin] × [Cin, Cout] = [num_tiles, Cout].
//
// This reference path is correctness-first; optimization passes can come later.

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <immintrin.h>
#include <math.h>

// FP32 packed GEMM (from kernels/gemm_fp32.c)
void pack_B_fp32(const float* B, int K, int N, float* Bp);
void fastface_gemm_fp32(const float* A, const float* Bp, float* C, int M, int K, int N);

#define WMR 4
#define WNR 16

// Single-tile micro-kernel, same math as gemm_fp32.c but inlined for batched use.
static inline void wino_micro_4x16(
    const float* __restrict__ A, const float* __restrict__ Bp,
    int K, int lda, float* __restrict__ C, int ldc)
{
    __m256 c00 = _mm256_setzero_ps(), c01 = _mm256_setzero_ps();
    __m256 c10 = _mm256_setzero_ps(), c11 = _mm256_setzero_ps();
    __m256 c20 = _mm256_setzero_ps(), c21 = _mm256_setzero_ps();
    __m256 c30 = _mm256_setzero_ps(), c31 = _mm256_setzero_ps();
    for (int k = 0; k < K; k++) {
        __m256 b0 = _mm256_loadu_ps(Bp + k * WNR);
        __m256 b1 = _mm256_loadu_ps(Bp + k * WNR + 8);
        __m256 a0 = _mm256_broadcast_ss(A + 0 * lda + k);
        __m256 a1 = _mm256_broadcast_ss(A + 1 * lda + k);
        __m256 a2 = _mm256_broadcast_ss(A + 2 * lda + k);
        __m256 a3 = _mm256_broadcast_ss(A + 3 * lda + k);
        c00 = _mm256_fmadd_ps(a0, b0, c00);  c01 = _mm256_fmadd_ps(a0, b1, c01);
        c10 = _mm256_fmadd_ps(a1, b0, c10);  c11 = _mm256_fmadd_ps(a1, b1, c11);
        c20 = _mm256_fmadd_ps(a2, b0, c20);  c21 = _mm256_fmadd_ps(a2, b1, c21);
        c30 = _mm256_fmadd_ps(a3, b0, c30);  c31 = _mm256_fmadd_ps(a3, b1, c31);
    }
    _mm256_storeu_ps(C + 0 * ldc + 0, c00); _mm256_storeu_ps(C + 0 * ldc + 8, c01);
    _mm256_storeu_ps(C + 1 * ldc + 0, c10); _mm256_storeu_ps(C + 1 * ldc + 8, c11);
    _mm256_storeu_ps(C + 2 * ldc + 0, c20); _mm256_storeu_ps(C + 2 * ldc + 8, c21);
    _mm256_storeu_ps(C + 3 * ldc + 0, c30); _mm256_storeu_ps(C + 3 * ldc + 8, c31);
}

// Batched Winograd GEMM: 16 independent [M, K] × [K, N] = [M, N] matmuls
// covered by ONE OpenMP parallel region to avoid fork/join overhead.
// V: 16 slabs of [M, K] row-major; U: 16 slabs pre-packed with pack_B_fp32; M_out: 16 slabs of [M, N].
static inline void wino_batched_gemm16(
    const float* V, const float* U, float* Mout, int M, int K, int N)
{
    int nt = N / WNR;
    #pragma omp parallel for schedule(static) collapse(3)
    for (int xn = 0; xn < 16; xn++) {
        for (int i = 0; i < M; i += WMR) {
            for (int t = 0; t < nt; t++) {
                const float* V_xi = V + (size_t)xn * M * K;
                const float* U_xi = U + (size_t)xn * K * N;
                float* M_xi = Mout + (size_t)xn * M * N;
                wino_micro_4x16(
                    V_xi + (size_t)i * K,
                    U_xi + (size_t)t * K * WNR,
                    K, K,
                    M_xi + (size_t)i * N + (size_t)t * WNR, N);
            }
        }
    }
}


// ----- Transform helpers -----

// Weight transform: U[4][4] = G * g * G^T where g is 3x3.
static void weight_transform_g3_to_u4(const float* g, float* U) {
    // Gg: 4x3, intermediate
    float Gg[4][3];
    for (int j = 0; j < 3; j++) {
        float g0 = g[0*3 + j], g1 = g[1*3 + j], g2 = g[2*3 + j];
        Gg[0][j] = g0;
        Gg[1][j] = 0.5f * (g0 + g1 + g2);
        Gg[2][j] = 0.5f * (g0 - g1 + g2);
        Gg[3][j] = g2;
    }
    for (int i = 0; i < 4; i++) {
        float r0 = Gg[i][0], r1 = Gg[i][1], r2 = Gg[i][2];
        U[i*4 + 0] = r0;
        U[i*4 + 1] = 0.5f * (r0 + r1 + r2);
        U[i*4 + 2] = 0.5f * (r0 - r1 + r2);
        U[i*4 + 3] = r2;
    }
}

// Input transform: V[4][4] = Bt * d * Bt^T where d is 4x4.
static inline void input_transform_d4_to_v4(const float* d, float* V) {
    // Btd: 4x4
    float Btd[4][4];
    for (int j = 0; j < 4; j++) {
        float d0 = d[0*4 + j], d1 = d[1*4 + j], d2 = d[2*4 + j], d3 = d[3*4 + j];
        Btd[0][j] = d0 - d2;
        Btd[1][j] = d1 + d2;
        Btd[2][j] = -d1 + d2;
        Btd[3][j] = d1 - d3;
    }
    // V = Btd * Bt^T (same transform applied to columns)
    for (int i = 0; i < 4; i++) {
        float r0 = Btd[i][0], r1 = Btd[i][1], r2 = Btd[i][2], r3 = Btd[i][3];
        V[i*4 + 0] = r0 - r2;
        V[i*4 + 1] = r1 + r2;
        V[i*4 + 2] = -r1 + r2;
        V[i*4 + 3] = r1 - r3;
    }
}

// Output transform: Y[2][2] = At * M * At^T where M is 4x4.
static inline void output_transform_m4_to_y2(const float* M, float* Y) {
    // AtM: 2x4
    float AtM[2][4];
    for (int j = 0; j < 4; j++) {
        float m0 = M[0*4 + j], m1 = M[1*4 + j], m2 = M[2*4 + j], m3 = M[3*4 + j];
        AtM[0][j] = m0 + m1 + m2;
        AtM[1][j] = m1 - m2 - m3;
    }
    // Y = AtM * At^T (same transform applied to columns)
    for (int i = 0; i < 2; i++) {
        float r0 = AtM[i][0], r1 = AtM[i][1], r2 = AtM[i][2], r3 = AtM[i][3];
        Y[i*2 + 0] = r0 + r1 + r2;
        Y[i*2 + 1] = r1 - r2 - r3;
    }
}


// Precompute Winograd weights for all (co, ci) pairs and pack for packed-GEMM.
// Input weight:    [Cout, Cin, 3, 3]  (NCHW weight layout as in FFW3)
// Output U_packed: [16, Cin, Cout] pre-packed via pack_B_fp32 with K=Cin, N=Cout.
//   (so for each (xi, nu) position we have a packed-B block ready for fastface_gemm_fp32)
//
// Caller must provide `scratch` of size at least Cin*Cout floats — used as temp
// row-major before packing.
void winograd_precompute_weights_packed(
    const float* weight, int Cout, int Cin, float* U_packed, float* scratch)
{
    // Phase 1: compute raw U[16] per (co, ci), scatter into scratch_row[xi*nu][ci, co].
    // We use U_packed itself as scratch for the row-major form, then overwrite each
    // (xi, nu) slab in-place via pack_B_fp32. Simpler: use explicit `scratch`.
    // Layout of scratch: [16, Cin, Cout] row-major (same as old U_wino).
    #pragma omp parallel for schedule(static)
    for (int co = 0; co < Cout; co++) {
        float U[16];
        for (int ci = 0; ci < Cin; ci++) {
            const float* g = weight + ((size_t)co * Cin + ci) * 9;
            weight_transform_g3_to_u4(g, U);
            for (int xn = 0; xn < 16; xn++) {
                scratch[(size_t)xn * Cin * Cout + ci * Cout + co] = U[xn];
            }
        }
    }
    // Phase 2: pack each of 16 [Cin, Cout] matrices into NR-panel layout.
    #pragma omp parallel for schedule(static)
    for (int xn = 0; xn < 16; xn++) {
        const float* Brow = scratch  + (size_t)xn * Cin * Cout;
        float*       Bp   = U_packed + (size_t)xn * Cin * Cout;
        pack_B_fp32(Brow, Cin, Cout, Bp);
    }
}

// Kept for API compatibility (referenced by older builds).
void winograd_precompute_weights(const float* weight, int Cout, int Cin, float* U_wino) {
    #pragma omp parallel for schedule(static)
    for (int co = 0; co < Cout; co++) {
        float U[16];
        for (int ci = 0; ci < Cin; ci++) {
            const float* g = weight + ((size_t)co * Cin + ci) * 9;
            weight_transform_g3_to_u4(g, U);
            for (int xn = 0; xn < 16; xn++) {
                U_wino[(size_t)xn * Cin * Cout + ci * Cout + co] = U[xn];
            }
        }
    }
}


// Reference Winograd Conv2d FP32 NHWC for 3×3 stride=1 pad=1.
// Assumes H_out == H_in, W_out == W_in (from s=1, p=1).
// If H_out or W_out is odd, we pad internally to even and trim.
//
// input:  [H_in, W_in, Cin]  NHWC fp32
// U_wino: precomputed weight transform [16, Cin, Cout]
// output: [H_out, W_out, Cout] NHWC fp32
// Packed-GEMM Winograd path. Layout changes from reference:
//   V_wino is organized as [16, num_tiles_padded, Cin] where num_tiles_padded is
//   (num_tiles + 3) & ~3. Padded rows are zero so GEMM produces zero contributions.
//   U_packed is [16 * Cin * Cout] pre-packed via pack_B_fp32 at load time.
//   M_wino is [16, num_tiles_padded, Cout].
// Each of the 16 (xi, nu) GEMMs is a call to fastface_gemm_fp32(V_xi, U_p_xi, M_xi,
//   num_tiles_padded, Cin, Cout).
// Variant that fuses per-output-channel bias add into the output transform.
// `bias` may be NULL (treated as zeros).
void fastface_winograd_conv_3x3_s1_p1_packed_bias(
    const float* input, int Cin, int H_in, int W_in,
    int Cout, const float* U_packed, const float* bias,
    float* output,
    float* V_wino, float* M_wino);

void fastface_winograd_conv_3x3_s1_p1_packed(
    const float* input, int Cin, int H_in, int W_in,
    int Cout, const float* U_packed,
    float* output,
    float* V_wino,   // scratch, >= 16 * num_tiles_padded * Cin
    float* M_wino)   // scratch, >= 16 * num_tiles_padded * Cout
{
    fastface_winograd_conv_3x3_s1_p1_packed_bias(input, Cin, H_in, W_in, Cout,
                                                    U_packed, NULL, output, V_wino, M_wino);
}

// Batched + BN-fused variant: applies per-channel BN (scale, offset) to the
// input tensor inline during Winograd input transform, removing a full memory pass.
// Optional per-channel prelu_slope applies PReLU in output transform.
// If bn_scale/bn_offset/prelu_slope are NULL, each is skipped.
// Input layout: [B, H_in, W_in, Cin]. Output: [B, H_out, W_out, Cout].
void fastface_winograd_conv_3x3_s1_p1_packed_bias_batched_bn_prelu(
    const float* input, int B, int Cin, int H_in, int W_in,
    int Cout, const float* U_packed, const float* bias,
    const float* bn_scale, const float* bn_offset,
    const float* prelu_slope,
    float* output,
    float* V_wino, float* M_wino);

void fastface_winograd_conv_3x3_s1_p1_packed_bias_batched_bn(
    const float* input, int B, int Cin, int H_in, int W_in,
    int Cout, const float* U_packed, const float* bias,
    const float* bn_scale, const float* bn_offset,
    float* output,
    float* V_wino, float* M_wino)
{
    fastface_winograd_conv_3x3_s1_p1_packed_bias_batched_bn_prelu(
        input, B, Cin, H_in, W_in, Cout, U_packed, bias,
        bn_scale, bn_offset, NULL, output, V_wino, M_wino);
}

void fastface_winograd_conv_3x3_s1_p1_packed_bias_batched(
    const float* input, int B, int Cin, int H_in, int W_in,
    int Cout, const float* U_packed, const float* bias,
    float* output,
    float* V_wino, float* M_wino)
{
    fastface_winograd_conv_3x3_s1_p1_packed_bias_batched_bn_prelu(
        input, B, Cin, H_in, W_in, Cout, U_packed, bias,
        NULL, NULL, NULL, output, V_wino, M_wino);
}

void fastface_winograd_conv_3x3_s1_p1_packed_bias_batched_bn_prelu(
    const float* input, int B, int Cin, int H_in, int W_in,
    int Cout, const float* U_packed, const float* bias,
    const float* bn_scale, const float* bn_offset,
    const float* prelu_slope,
    float* output,
    float* V_wino, float* M_wino)
{
    int H_out = H_in, W_out = W_in;
    int tile_H = (H_out + 1) / 2;
    int tile_W = (W_out + 1) / 2;
    int tiles_per_face = tile_H * tile_W;
    int num_tiles = B * tiles_per_face;
    int num_tiles_padded = (num_tiles + 3) & ~3;

    if (num_tiles_padded > num_tiles) {
        int pad_rows = num_tiles_padded - num_tiles;
        for (int xn = 0; xn < 16; xn++) {
            memset(V_wino + (size_t)xn * num_tiles_padded * Cin
                   + (size_t)num_tiles * Cin, 0,
                   (size_t)pad_rows * Cin * sizeof(float));
        }
    }

    // Input transform with batch dim outermost
    #pragma omp parallel for schedule(static) collapse(3)
    for (int b = 0; b < B; b++) {
        for (int th = 0; th < tile_H; th++) {
            for (int tw = 0; tw < tile_W; tw++) {
                int tidx = b * tiles_per_face + th * tile_W + tw;
                const float* batch_input = input + (size_t)b * H_in * W_in * Cin;
                int ci = 0;
#ifdef __AVX2__
                for (; ci + 8 <= Cin; ci += 8) {
                    __m256 d00, d01, d02, d03, d10, d11, d12, d13;
                    __m256 d20, d21, d22, d23, d30, d31, d32, d33;
                    __m256 vs, vo;
                    if (bn_scale) {
                        vs = _mm256_loadu_ps(bn_scale + ci);
                        vo = _mm256_loadu_ps(bn_offset + ci);
                    } else {
                        vs = _mm256_set1_ps(1.0f); vo = _mm256_setzero_ps();
                    }
                    #define LOAD_D(yy, xx, var) do {                                         \
                        int ih = 2 * th - 1 + (yy);                                          \
                        int iw = 2 * tw - 1 + (xx);                                          \
                        if (ih >= 0 && ih < H_in && iw >= 0 && iw < W_in) {                  \
                            __m256 _raw = _mm256_loadu_ps(batch_input + ((size_t)ih * W_in + iw) * Cin + ci); \
                            var = bn_scale ? _mm256_fmadd_ps(_raw, vs, vo) : _raw;            \
                        } else {                                                             \
                            var = _mm256_setzero_ps();                                       \
                        }                                                                    \
                    } while(0)
                    LOAD_D(0,0,d00); LOAD_D(0,1,d01); LOAD_D(0,2,d02); LOAD_D(0,3,d03);
                    LOAD_D(1,0,d10); LOAD_D(1,1,d11); LOAD_D(1,2,d12); LOAD_D(1,3,d13);
                    LOAD_D(2,0,d20); LOAD_D(2,1,d21); LOAD_D(2,2,d22); LOAD_D(2,3,d23);
                    LOAD_D(3,0,d30); LOAD_D(3,1,d31); LOAD_D(3,2,d32); LOAD_D(3,3,d33);
                    #undef LOAD_D
                    __m256 bt00 = _mm256_sub_ps(d00, d20);
                    __m256 bt01 = _mm256_sub_ps(d01, d21);
                    __m256 bt02 = _mm256_sub_ps(d02, d22);
                    __m256 bt03 = _mm256_sub_ps(d03, d23);
                    __m256 bt10 = _mm256_add_ps(d10, d20);
                    __m256 bt11 = _mm256_add_ps(d11, d21);
                    __m256 bt12 = _mm256_add_ps(d12, d22);
                    __m256 bt13 = _mm256_add_ps(d13, d23);
                    __m256 bt20 = _mm256_sub_ps(d20, d10);
                    __m256 bt21 = _mm256_sub_ps(d21, d11);
                    __m256 bt22 = _mm256_sub_ps(d22, d12);
                    __m256 bt23 = _mm256_sub_ps(d23, d13);
                    __m256 bt30 = _mm256_sub_ps(d10, d30);
                    __m256 bt31 = _mm256_sub_ps(d11, d31);
                    __m256 bt32 = _mm256_sub_ps(d12, d32);
                    __m256 bt33 = _mm256_sub_ps(d13, d33);
                    __m256 v00 = _mm256_sub_ps(bt00, bt02);
                    __m256 v01 = _mm256_add_ps(bt01, bt02);
                    __m256 v02 = _mm256_sub_ps(bt02, bt01);
                    __m256 v03 = _mm256_sub_ps(bt01, bt03);
                    __m256 v10 = _mm256_sub_ps(bt10, bt12);
                    __m256 v11 = _mm256_add_ps(bt11, bt12);
                    __m256 v12 = _mm256_sub_ps(bt12, bt11);
                    __m256 v13 = _mm256_sub_ps(bt11, bt13);
                    __m256 v20 = _mm256_sub_ps(bt20, bt22);
                    __m256 v21 = _mm256_add_ps(bt21, bt22);
                    __m256 v22 = _mm256_sub_ps(bt22, bt21);
                    __m256 v23 = _mm256_sub_ps(bt21, bt23);
                    __m256 v30 = _mm256_sub_ps(bt30, bt32);
                    __m256 v31 = _mm256_add_ps(bt31, bt32);
                    __m256 v32 = _mm256_sub_ps(bt32, bt31);
                    __m256 v33 = _mm256_sub_ps(bt31, bt33);
                    size_t slab = (size_t)num_tiles_padded * Cin;
                    float* base = V_wino + (size_t)tidx * Cin + ci;
                    _mm256_storeu_ps(base + 0  * slab, v00);
                    _mm256_storeu_ps(base + 1  * slab, v01);
                    _mm256_storeu_ps(base + 2  * slab, v02);
                    _mm256_storeu_ps(base + 3  * slab, v03);
                    _mm256_storeu_ps(base + 4  * slab, v10);
                    _mm256_storeu_ps(base + 5  * slab, v11);
                    _mm256_storeu_ps(base + 6  * slab, v12);
                    _mm256_storeu_ps(base + 7  * slab, v13);
                    _mm256_storeu_ps(base + 8  * slab, v20);
                    _mm256_storeu_ps(base + 9  * slab, v21);
                    _mm256_storeu_ps(base + 10 * slab, v22);
                    _mm256_storeu_ps(base + 11 * slab, v23);
                    _mm256_storeu_ps(base + 12 * slab, v30);
                    _mm256_storeu_ps(base + 13 * slab, v31);
                    _mm256_storeu_ps(base + 14 * slab, v32);
                    _mm256_storeu_ps(base + 15 * slab, v33);
                }
#endif
                for (; ci < Cin; ci++) {
                    float bs = bn_scale ? bn_scale[ci] : 1.0f;
                    float bo = bn_offset ? bn_offset[ci] : 0.0f;
                    float d[16];
                    for (int y = 0; y < 4; y++) {
                        for (int x = 0; x < 4; x++) {
                            int ih = 2 * th - 1 + y;
                            int iw = 2 * tw - 1 + x;
                            if (ih >= 0 && ih < H_in && iw >= 0 && iw < W_in) {
                                float raw = batch_input[((size_t)ih * W_in + iw) * Cin + ci];
                                d[y*4 + x] = bn_scale ? (raw * bs + bo) : raw;
                            } else {
                                d[y*4 + x] = 0.0f;
                            }
                        }
                    }
                    float V[16];
                    input_transform_d4_to_v4(d, V);
                    for (int xn = 0; xn < 16; xn++) {
                        V_wino[(size_t)xn * num_tiles_padded * Cin
                             + (size_t)tidx * Cin + ci] = V[xn];
                    }
                }
            }
        }
    }

    // Batched GEMM: M grows to B * tiles, weight reused B× across batches.
    wino_batched_gemm16(V_wino, U_packed, M_wino, num_tiles_padded, Cin, Cout);

    // Output transform with batch dim
    #pragma omp parallel for schedule(static) collapse(3)
    for (int b = 0; b < B; b++) {
        for (int th = 0; th < tile_H; th++) {
            for (int tw = 0; tw < tile_W; tw++) {
                int tidx = b * tiles_per_face + th * tile_W + tw;
                float* batch_output = output + (size_t)b * H_out * W_out * Cout;
                int co = 0;
                size_t slab = (size_t)num_tiles_padded * Cout;
                const float* base = M_wino + (size_t)tidx * Cout;
#ifdef __AVX2__
                for (; co + 8 <= Cout; co += 8) {
                    __m256 m00 = _mm256_loadu_ps(base + 0  * slab + co);
                    __m256 m01 = _mm256_loadu_ps(base + 1  * slab + co);
                    __m256 m02 = _mm256_loadu_ps(base + 2  * slab + co);
                    __m256 m03 = _mm256_loadu_ps(base + 3  * slab + co);
                    __m256 m10 = _mm256_loadu_ps(base + 4  * slab + co);
                    __m256 m11 = _mm256_loadu_ps(base + 5  * slab + co);
                    __m256 m12 = _mm256_loadu_ps(base + 6  * slab + co);
                    __m256 m13 = _mm256_loadu_ps(base + 7  * slab + co);
                    __m256 m20 = _mm256_loadu_ps(base + 8  * slab + co);
                    __m256 m21 = _mm256_loadu_ps(base + 9  * slab + co);
                    __m256 m22 = _mm256_loadu_ps(base + 10 * slab + co);
                    __m256 m23 = _mm256_loadu_ps(base + 11 * slab + co);
                    __m256 m30 = _mm256_loadu_ps(base + 12 * slab + co);
                    __m256 m31 = _mm256_loadu_ps(base + 13 * slab + co);
                    __m256 m32 = _mm256_loadu_ps(base + 14 * slab + co);
                    __m256 m33 = _mm256_loadu_ps(base + 15 * slab + co);
                    __m256 atm00 = _mm256_add_ps(_mm256_add_ps(m00, m10), m20);
                    __m256 atm01 = _mm256_add_ps(_mm256_add_ps(m01, m11), m21);
                    __m256 atm02 = _mm256_add_ps(_mm256_add_ps(m02, m12), m22);
                    __m256 atm03 = _mm256_add_ps(_mm256_add_ps(m03, m13), m23);
                    __m256 atm10 = _mm256_sub_ps(_mm256_sub_ps(m10, m20), m30);
                    __m256 atm11 = _mm256_sub_ps(_mm256_sub_ps(m11, m21), m31);
                    __m256 atm12 = _mm256_sub_ps(_mm256_sub_ps(m12, m22), m32);
                    __m256 atm13 = _mm256_sub_ps(_mm256_sub_ps(m13, m23), m33);
                    __m256 y00 = _mm256_add_ps(_mm256_add_ps(atm00, atm01), atm02);
                    __m256 y01 = _mm256_sub_ps(_mm256_sub_ps(atm01, atm02), atm03);
                    __m256 y10 = _mm256_add_ps(_mm256_add_ps(atm10, atm11), atm12);
                    __m256 y11 = _mm256_sub_ps(_mm256_sub_ps(atm11, atm12), atm13);
                    if (bias) {
                        __m256 vb = _mm256_loadu_ps(bias + co);
                        y00 = _mm256_add_ps(y00, vb);
                        y01 = _mm256_add_ps(y01, vb);
                        y10 = _mm256_add_ps(y10, vb);
                        y11 = _mm256_add_ps(y11, vb);
                    }
                    if (prelu_slope) {
                        __m256 sl = _mm256_loadu_ps(prelu_slope + co);
                        __m256 zero = _mm256_setzero_ps();
                        __m256 mask, neg;
                        mask = _mm256_cmp_ps(y00, zero, _CMP_LT_OS); neg = _mm256_mul_ps(y00, sl); y00 = _mm256_blendv_ps(y00, neg, mask);
                        mask = _mm256_cmp_ps(y01, zero, _CMP_LT_OS); neg = _mm256_mul_ps(y01, sl); y01 = _mm256_blendv_ps(y01, neg, mask);
                        mask = _mm256_cmp_ps(y10, zero, _CMP_LT_OS); neg = _mm256_mul_ps(y10, sl); y10 = _mm256_blendv_ps(y10, neg, mask);
                        mask = _mm256_cmp_ps(y11, zero, _CMP_LT_OS); neg = _mm256_mul_ps(y11, sl); y11 = _mm256_blendv_ps(y11, neg, mask);
                    }
                    int oh0 = 2 * th, ow0 = 2 * tw;
                    if (oh0 < H_out && ow0 < W_out)
                        _mm256_storeu_ps(batch_output + ((size_t)oh0 * W_out + ow0) * Cout + co, y00);
                    if (oh0 < H_out && ow0 + 1 < W_out)
                        _mm256_storeu_ps(batch_output + ((size_t)oh0 * W_out + ow0 + 1) * Cout + co, y01);
                    if (oh0 + 1 < H_out && ow0 < W_out)
                        _mm256_storeu_ps(batch_output + ((size_t)(oh0 + 1) * W_out + ow0) * Cout + co, y10);
                    if (oh0 + 1 < H_out && ow0 + 1 < W_out)
                        _mm256_storeu_ps(batch_output + ((size_t)(oh0 + 1) * W_out + ow0 + 1) * Cout + co, y11);
                }
#endif
                for (; co < Cout; co++) {
                    float M[16];
                    for (int xn = 0; xn < 16; xn++) {
                        M[xn] = M_wino[(size_t)xn * slab + (size_t)tidx * Cout + co];
                    }
                    float Y[4];
                    output_transform_m4_to_y2(M, Y);
                    float bb = bias ? bias[co] : 0.0f;
                    float sl = prelu_slope ? prelu_slope[co] : 1.0f;
                    for (int y = 0; y < 2; y++) {
                        int oh = 2 * th + y;
                        if (oh >= H_out) continue;
                        for (int x = 0; x < 2; x++) {
                            int ow = 2 * tw + x;
                            if (ow >= W_out) continue;
                            float v = Y[y*2 + x] + bb;
                            if (prelu_slope && v < 0) v *= sl;
                            batch_output[((size_t)oh * W_out + ow) * Cout + co] = v;
                        }
                    }
                }
            }
        }
    }
}


// Extended variant with optional per-position residual add (for Conv+ADD fusion).
void fastface_winograd_conv_3x3_s1_p1_full_fused_add(
    const float* input, int Cin, int H_in, int W_in,
    int Cout, const float* U_packed,
    const float* bn_scale, const float* bn_offset,
    const float* bias, const float* prelu_slope,
    const float* add_src,  // optional: per-(hw,co) values added before store
    float* output,
    float* V_wino, float* M_wino);

void fastface_winograd_conv_3x3_s1_p1_full_fused(
    const float* input, int Cin, int H_in, int W_in,
    int Cout, const float* U_packed,
    const float* bn_scale, const float* bn_offset,
    const float* bias, const float* prelu_slope,
    float* output,
    float* V_wino, float* M_wino)
{
    fastface_winograd_conv_3x3_s1_p1_full_fused_add(
        input, Cin, H_in, W_in, Cout, U_packed,
        bn_scale, bn_offset, bias, prelu_slope, NULL,
        output, V_wino, M_wino);
}

void fastface_winograd_conv_3x3_s1_p1_full_fused_add(
    const float* input, int Cin, int H_in, int W_in,
    int Cout, const float* U_packed,
    const float* bn_scale, const float* bn_offset,
    const float* bias, const float* prelu_slope,
    const float* add_src,
    float* output,
    float* V_wino, float* M_wino)
{
    int H_out = H_in;
    int W_out = W_in;
    int tile_H = (H_out + 1) / 2;
    int tile_W = (W_out + 1) / 2;
    int num_tiles = tile_H * tile_W;
    int num_tiles_padded = (num_tiles + 3) & ~3;

    if (num_tiles_padded > num_tiles) {
        int pad_rows = num_tiles_padded - num_tiles;
        for (int xn = 0; xn < 16; xn++) {
            memset(V_wino + (size_t)xn * num_tiles_padded * Cin
                   + (size_t)num_tiles * Cin, 0,
                   (size_t)pad_rows * Cin * sizeof(float));
        }
    }

    // Input transform with optional BN fusion inline.
    #pragma omp parallel for schedule(static) collapse(2)
    for (int th = 0; th < tile_H; th++) {
        for (int tw = 0; tw < tile_W; tw++) {
            int tidx = th * tile_W + tw;
            int ci = 0;
#ifdef __AVX2__
            for (; ci + 8 <= Cin; ci += 8) {
                __m256 vs, vo;
                if (bn_scale) {
                    vs = _mm256_loadu_ps(bn_scale + ci);
                    vo = _mm256_loadu_ps(bn_offset + ci);
                } else { vs = _mm256_set1_ps(1.0f); vo = _mm256_setzero_ps(); }
                __m256 d00,d01,d02,d03,d10,d11,d12,d13,d20,d21,d22,d23,d30,d31,d32,d33;
                #define LOADD(yy, xx, var) do {                                               \
                    int ih = 2*th - 1 + (yy); int iw = 2*tw - 1 + (xx);                       \
                    if (ih >= 0 && ih < H_in && iw >= 0 && iw < W_in) {                       \
                        __m256 raw = _mm256_loadu_ps(input + ((size_t)ih*W_in + iw)*Cin + ci); \
                        var = bn_scale ? _mm256_fmadd_ps(raw, vs, vo) : raw;                  \
                    } else { var = _mm256_setzero_ps(); }                                     \
                } while(0)
                LOADD(0,0,d00); LOADD(0,1,d01); LOADD(0,2,d02); LOADD(0,3,d03);
                LOADD(1,0,d10); LOADD(1,1,d11); LOADD(1,2,d12); LOADD(1,3,d13);
                LOADD(2,0,d20); LOADD(2,1,d21); LOADD(2,2,d22); LOADD(2,3,d23);
                LOADD(3,0,d30); LOADD(3,1,d31); LOADD(3,2,d32); LOADD(3,3,d33);
                #undef LOADD
                __m256 bt00=_mm256_sub_ps(d00,d20), bt01=_mm256_sub_ps(d01,d21);
                __m256 bt02=_mm256_sub_ps(d02,d22), bt03=_mm256_sub_ps(d03,d23);
                __m256 bt10=_mm256_add_ps(d10,d20), bt11=_mm256_add_ps(d11,d21);
                __m256 bt12=_mm256_add_ps(d12,d22), bt13=_mm256_add_ps(d13,d23);
                __m256 bt20=_mm256_sub_ps(d20,d10), bt21=_mm256_sub_ps(d21,d11);
                __m256 bt22=_mm256_sub_ps(d22,d12), bt23=_mm256_sub_ps(d23,d13);
                __m256 bt30=_mm256_sub_ps(d10,d30), bt31=_mm256_sub_ps(d11,d31);
                __m256 bt32=_mm256_sub_ps(d12,d32), bt33=_mm256_sub_ps(d13,d33);
                __m256 v00=_mm256_sub_ps(bt00,bt02), v01=_mm256_add_ps(bt01,bt02);
                __m256 v02=_mm256_sub_ps(bt02,bt01), v03=_mm256_sub_ps(bt01,bt03);
                __m256 v10=_mm256_sub_ps(bt10,bt12), v11=_mm256_add_ps(bt11,bt12);
                __m256 v12=_mm256_sub_ps(bt12,bt11), v13=_mm256_sub_ps(bt11,bt13);
                __m256 v20=_mm256_sub_ps(bt20,bt22), v21=_mm256_add_ps(bt21,bt22);
                __m256 v22=_mm256_sub_ps(bt22,bt21), v23=_mm256_sub_ps(bt21,bt23);
                __m256 v30=_mm256_sub_ps(bt30,bt32), v31=_mm256_add_ps(bt31,bt32);
                __m256 v32=_mm256_sub_ps(bt32,bt31), v33=_mm256_sub_ps(bt31,bt33);
                size_t slab = (size_t)num_tiles_padded * Cin;
                float* base = V_wino + (size_t)tidx * Cin + ci;
                _mm256_storeu_ps(base + 0*slab, v00); _mm256_storeu_ps(base + 1*slab, v01);
                _mm256_storeu_ps(base + 2*slab, v02); _mm256_storeu_ps(base + 3*slab, v03);
                _mm256_storeu_ps(base + 4*slab, v10); _mm256_storeu_ps(base + 5*slab, v11);
                _mm256_storeu_ps(base + 6*slab, v12); _mm256_storeu_ps(base + 7*slab, v13);
                _mm256_storeu_ps(base + 8*slab, v20); _mm256_storeu_ps(base + 9*slab, v21);
                _mm256_storeu_ps(base + 10*slab, v22); _mm256_storeu_ps(base + 11*slab, v23);
                _mm256_storeu_ps(base + 12*slab, v30); _mm256_storeu_ps(base + 13*slab, v31);
                _mm256_storeu_ps(base + 14*slab, v32); _mm256_storeu_ps(base + 15*slab, v33);
            }
#endif
            for (; ci < Cin; ci++) {
                float bs = bn_scale ? bn_scale[ci] : 1.0f;
                float bo = bn_scale ? bn_offset[ci] : 0.0f;
                float d[16];
                for (int y = 0; y < 4; y++) for (int x = 0; x < 4; x++) {
                    int ih = 2*th - 1 + y, iw = 2*tw - 1 + x;
                    if (ih >= 0 && ih < H_in && iw >= 0 && iw < W_in) {
                        float raw = input[((size_t)ih*W_in + iw)*Cin + ci];
                        d[y*4+x] = bn_scale ? (raw * bs + bo) : raw;
                    } else d[y*4+x] = 0.0f;
                }
                float V[16]; input_transform_d4_to_v4(d, V);
                for (int xn = 0; xn < 16; xn++)
                    V_wino[(size_t)xn * num_tiles_padded * Cin + (size_t)tidx * Cin + ci] = V[xn];
            }
        }
    }

    wino_batched_gemm16(V_wino, U_packed, M_wino, num_tiles_padded, Cin, Cout);

    // Output transform + bias + PReLU (same code as existing).
    #pragma omp parallel for schedule(static) collapse(2)
    for (int th = 0; th < tile_H; th++) {
        for (int tw = 0; tw < tile_W; tw++) {
            int tidx = th * tile_W + tw;
            int co = 0;
            size_t slab = (size_t)num_tiles_padded * Cout;
            const float* base = M_wino + (size_t)tidx * Cout;
#ifdef __AVX2__
            for (; co + 8 <= Cout; co += 8) {
                __m256 m00 = _mm256_loadu_ps(base + 0*slab + co);
                __m256 m01 = _mm256_loadu_ps(base + 1*slab + co);
                __m256 m02 = _mm256_loadu_ps(base + 2*slab + co);
                __m256 m03 = _mm256_loadu_ps(base + 3*slab + co);
                __m256 m10 = _mm256_loadu_ps(base + 4*slab + co);
                __m256 m11 = _mm256_loadu_ps(base + 5*slab + co);
                __m256 m12 = _mm256_loadu_ps(base + 6*slab + co);
                __m256 m13 = _mm256_loadu_ps(base + 7*slab + co);
                __m256 m20 = _mm256_loadu_ps(base + 8*slab + co);
                __m256 m21 = _mm256_loadu_ps(base + 9*slab + co);
                __m256 m22 = _mm256_loadu_ps(base + 10*slab + co);
                __m256 m23 = _mm256_loadu_ps(base + 11*slab + co);
                __m256 m30 = _mm256_loadu_ps(base + 12*slab + co);
                __m256 m31 = _mm256_loadu_ps(base + 13*slab + co);
                __m256 m32 = _mm256_loadu_ps(base + 14*slab + co);
                __m256 m33 = _mm256_loadu_ps(base + 15*slab + co);
                __m256 a00=_mm256_add_ps(_mm256_add_ps(m00,m10),m20);
                __m256 a01=_mm256_add_ps(_mm256_add_ps(m01,m11),m21);
                __m256 a02=_mm256_add_ps(_mm256_add_ps(m02,m12),m22);
                __m256 a03=_mm256_add_ps(_mm256_add_ps(m03,m13),m23);
                __m256 a10=_mm256_sub_ps(_mm256_sub_ps(m10,m20),m30);
                __m256 a11=_mm256_sub_ps(_mm256_sub_ps(m11,m21),m31);
                __m256 a12=_mm256_sub_ps(_mm256_sub_ps(m12,m22),m32);
                __m256 a13=_mm256_sub_ps(_mm256_sub_ps(m13,m23),m33);
                __m256 y00=_mm256_add_ps(_mm256_add_ps(a00,a01),a02);
                __m256 y01=_mm256_sub_ps(_mm256_sub_ps(a01,a02),a03);
                __m256 y10=_mm256_add_ps(_mm256_add_ps(a10,a11),a12);
                __m256 y11=_mm256_sub_ps(_mm256_sub_ps(a11,a12),a13);
                if (bias) {
                    __m256 vb = _mm256_loadu_ps(bias + co);
                    y00 = _mm256_add_ps(y00, vb); y01 = _mm256_add_ps(y01, vb);
                    y10 = _mm256_add_ps(y10, vb); y11 = _mm256_add_ps(y11, vb);
                }
                if (prelu_slope) {
                    __m256 sl = _mm256_loadu_ps(prelu_slope + co);
                    __m256 zero = _mm256_setzero_ps(); __m256 mask, neg;
                    mask=_mm256_cmp_ps(y00,zero,_CMP_LT_OS); neg=_mm256_mul_ps(y00,sl); y00=_mm256_blendv_ps(y00,neg,mask);
                    mask=_mm256_cmp_ps(y01,zero,_CMP_LT_OS); neg=_mm256_mul_ps(y01,sl); y01=_mm256_blendv_ps(y01,neg,mask);
                    mask=_mm256_cmp_ps(y10,zero,_CMP_LT_OS); neg=_mm256_mul_ps(y10,sl); y10=_mm256_blendv_ps(y10,neg,mask);
                    mask=_mm256_cmp_ps(y11,zero,_CMP_LT_OS); neg=_mm256_mul_ps(y11,sl); y11=_mm256_blendv_ps(y11,neg,mask);
                }
                int oh0 = 2*th, ow0 = 2*tw;
                if (oh0 < H_out && ow0 < W_out) {
                    float* dst = output + ((size_t)oh0 * W_out + ow0) * Cout + co;
                    if (add_src) y00 = _mm256_add_ps(y00, _mm256_loadu_ps(add_src + ((size_t)oh0 * W_out + ow0) * Cout + co));
                    _mm256_storeu_ps(dst, y00);
                }
                if (oh0 < H_out && ow0 + 1 < W_out) {
                    float* dst = output + ((size_t)oh0 * W_out + ow0 + 1) * Cout + co;
                    if (add_src) y01 = _mm256_add_ps(y01, _mm256_loadu_ps(add_src + ((size_t)oh0 * W_out + ow0 + 1) * Cout + co));
                    _mm256_storeu_ps(dst, y01);
                }
                if (oh0 + 1 < H_out && ow0 < W_out) {
                    float* dst = output + ((size_t)(oh0 + 1) * W_out + ow0) * Cout + co;
                    if (add_src) y10 = _mm256_add_ps(y10, _mm256_loadu_ps(add_src + ((size_t)(oh0 + 1) * W_out + ow0) * Cout + co));
                    _mm256_storeu_ps(dst, y10);
                }
                if (oh0 + 1 < H_out && ow0 + 1 < W_out) {
                    float* dst = output + ((size_t)(oh0 + 1) * W_out + ow0 + 1) * Cout + co;
                    if (add_src) y11 = _mm256_add_ps(y11, _mm256_loadu_ps(add_src + ((size_t)(oh0 + 1) * W_out + ow0 + 1) * Cout + co));
                    _mm256_storeu_ps(dst, y11);
                }
            }
#endif
            for (; co < Cout; co++) {
                float M[16];
                for (int xn = 0; xn < 16; xn++) M[xn] = M_wino[(size_t)xn * slab + (size_t)tidx * Cout + co];
                float Y[4]; output_transform_m4_to_y2(M, Y);
                float b = bias ? bias[co] : 0.0f;
                float sl = prelu_slope ? prelu_slope[co] : 1.0f;
                for (int y = 0; y < 2; y++) {
                    int oh = 2*th + y; if (oh >= H_out) continue;
                    for (int x = 0; x < 2; x++) {
                        int ow = 2*tw + x; if (ow >= W_out) continue;
                        float v = Y[y*2+x] + b;
                        if (prelu_slope && v < 0) v *= sl;
                        if (add_src) v += add_src[((size_t)oh * W_out + ow) * Cout + co];
                        output[((size_t)oh * W_out + ow) * Cout + co] = v;
                    }
                }
            }
        }
    }
}


// Thin wrappers — dispatch to _full_fused with appropriate NULLs.
void fastface_winograd_conv_3x3_s1_p1_packed_bias(
    const float* input, int Cin, int H_in, int W_in,
    int Cout, const float* U_packed, const float* bias,
    float* output, float* V_wino, float* M_wino)
{
    fastface_winograd_conv_3x3_s1_p1_full_fused(
        input, Cin, H_in, W_in, Cout, U_packed,
        NULL, NULL, bias, NULL, output, V_wino, M_wino);
}

void fastface_winograd_conv_3x3_s1_p1_packed_bias_prelu(
    const float* input, int Cin, int H_in, int W_in,
    int Cout, const float* U_packed, const float* bias,
    const float* prelu_slope,
    float* output, float* V_wino, float* M_wino)
{
    fastface_winograd_conv_3x3_s1_p1_full_fused(
        input, Cin, H_in, W_in, Cout, U_packed,
        NULL, NULL, bias, prelu_slope, output, V_wino, M_wino);
}

void fastface_winograd_conv_3x3_s1_p1_packed_bias_OLD(
    const float* input, int Cin, int H_in, int W_in,
    int Cout, const float* U_packed, const float* bias,
    const float* prelu_slope,
    float* output,
    float* V_wino,
    float* M_wino)
{
    int H_out = H_in;
    int W_out = W_in;
    int tile_H = (H_out + 1) / 2;
    int tile_W = (W_out + 1) / 2;
    int num_tiles = tile_H * tile_W;
    int num_tiles_padded = (num_tiles + 3) & ~3;

    // Zero only the padded (tail) rows of each V slab — real rows are overwritten.
    // (Previously zeroed the entire buffer, which was 12+ MB per Conv and dominated.)
    if (num_tiles_padded > num_tiles) {
        int pad_rows = num_tiles_padded - num_tiles;
        for (int xn = 0; xn < 16; xn++) {
            memset(V_wino + (size_t)xn * num_tiles_padded * Cin
                   + (size_t)num_tiles * Cin, 0,
                   (size_t)pad_rows * Cin * sizeof(float));
        }
    }

    // --- Input transform: per-tile 4×4 → V[16, t, ci] with t in [0, num_tiles).
    // SIMD along ci: for each tile, process 8 channels at a time since NHWC
    // stores channels contiguously. Each `d[y*4+x]` holds 8 channel lanes.
    #pragma omp parallel for schedule(static) collapse(2)
    for (int th = 0; th < tile_H; th++) {
        for (int tw = 0; tw < tile_W; tw++) {
            int tidx = th * tile_W + tw;
            int ci = 0;
#ifdef __AVX2__
            for (; ci + 8 <= Cin; ci += 8) {
                __m256 d00, d01, d02, d03, d10, d11, d12, d13;
                __m256 d20, d21, d22, d23, d30, d31, d32, d33;
                #define LOAD_D(yy, xx, var) do {                                         \
                    int ih = 2 * th - 1 + (yy);                                          \
                    int iw = 2 * tw - 1 + (xx);                                          \
                    if (ih >= 0 && ih < H_in && iw >= 0 && iw < W_in)                    \
                        var = _mm256_loadu_ps(input + ((size_t)ih * W_in + iw) * Cin + ci); \
                    else                                                                 \
                        var = _mm256_setzero_ps();                                       \
                } while(0)
                LOAD_D(0,0,d00); LOAD_D(0,1,d01); LOAD_D(0,2,d02); LOAD_D(0,3,d03);
                LOAD_D(1,0,d10); LOAD_D(1,1,d11); LOAD_D(1,2,d12); LOAD_D(1,3,d13);
                LOAD_D(2,0,d20); LOAD_D(2,1,d21); LOAD_D(2,2,d22); LOAD_D(2,3,d23);
                LOAD_D(3,0,d30); LOAD_D(3,1,d31); LOAD_D(3,2,d32); LOAD_D(3,3,d33);
                #undef LOAD_D
                // Row transform: Btd[i][j] per col j
                __m256 bt00 = _mm256_sub_ps(d00, d20);
                __m256 bt01 = _mm256_sub_ps(d01, d21);
                __m256 bt02 = _mm256_sub_ps(d02, d22);
                __m256 bt03 = _mm256_sub_ps(d03, d23);
                __m256 bt10 = _mm256_add_ps(d10, d20);
                __m256 bt11 = _mm256_add_ps(d11, d21);
                __m256 bt12 = _mm256_add_ps(d12, d22);
                __m256 bt13 = _mm256_add_ps(d13, d23);
                __m256 bt20 = _mm256_sub_ps(d20, d10);
                __m256 bt21 = _mm256_sub_ps(d21, d11);
                __m256 bt22 = _mm256_sub_ps(d22, d12);
                __m256 bt23 = _mm256_sub_ps(d23, d13);
                __m256 bt30 = _mm256_sub_ps(d10, d30);
                __m256 bt31 = _mm256_sub_ps(d11, d31);
                __m256 bt32 = _mm256_sub_ps(d12, d32);
                __m256 bt33 = _mm256_sub_ps(d13, d33);
                // Column transform: V[i][j] = Btd[i][:] * Bt^T row j
                __m256 v00 = _mm256_sub_ps(bt00, bt02);
                __m256 v01 = _mm256_add_ps(bt01, bt02);
                __m256 v02 = _mm256_sub_ps(bt02, bt01);
                __m256 v03 = _mm256_sub_ps(bt01, bt03);
                __m256 v10 = _mm256_sub_ps(bt10, bt12);
                __m256 v11 = _mm256_add_ps(bt11, bt12);
                __m256 v12 = _mm256_sub_ps(bt12, bt11);
                __m256 v13 = _mm256_sub_ps(bt11, bt13);
                __m256 v20 = _mm256_sub_ps(bt20, bt22);
                __m256 v21 = _mm256_add_ps(bt21, bt22);
                __m256 v22 = _mm256_sub_ps(bt22, bt21);
                __m256 v23 = _mm256_sub_ps(bt21, bt23);
                __m256 v30 = _mm256_sub_ps(bt30, bt32);
                __m256 v31 = _mm256_add_ps(bt31, bt32);
                __m256 v32 = _mm256_sub_ps(bt32, bt31);
                __m256 v33 = _mm256_sub_ps(bt31, bt33);
                // Scatter: V_wino[xn][tidx][ci..ci+7]
                size_t slab = (size_t)num_tiles_padded * Cin;
                float* base = V_wino + (size_t)tidx * Cin + ci;
                _mm256_storeu_ps(base + 0  * slab, v00);
                _mm256_storeu_ps(base + 1  * slab, v01);
                _mm256_storeu_ps(base + 2  * slab, v02);
                _mm256_storeu_ps(base + 3  * slab, v03);
                _mm256_storeu_ps(base + 4  * slab, v10);
                _mm256_storeu_ps(base + 5  * slab, v11);
                _mm256_storeu_ps(base + 6  * slab, v12);
                _mm256_storeu_ps(base + 7  * slab, v13);
                _mm256_storeu_ps(base + 8  * slab, v20);
                _mm256_storeu_ps(base + 9  * slab, v21);
                _mm256_storeu_ps(base + 10 * slab, v22);
                _mm256_storeu_ps(base + 11 * slab, v23);
                _mm256_storeu_ps(base + 12 * slab, v30);
                _mm256_storeu_ps(base + 13 * slab, v31);
                _mm256_storeu_ps(base + 14 * slab, v32);
                _mm256_storeu_ps(base + 15 * slab, v33);
            }
#endif
            // Scalar tail
            for (; ci < Cin; ci++) {
                float d[16];
                for (int y = 0; y < 4; y++) {
                    for (int x = 0; x < 4; x++) {
                        int ih = 2 * th - 1 + y;
                        int iw = 2 * tw - 1 + x;
                        d[y*4 + x] = (ih >= 0 && ih < H_in && iw >= 0 && iw < W_in)
                            ? input[((size_t)ih * W_in + iw) * Cin + ci]
                            : 0.0f;
                    }
                }
                float V[16];
                input_transform_d4_to_v4(d, V);
                for (int xn = 0; xn < 16; xn++) {
                    V_wino[(size_t)xn * num_tiles_padded * Cin
                         + (size_t)tidx * Cin + ci] = V[xn];
                }
            }
        }
    }

    // --- Batched GEMM: 16 packed matmuls in a single OpenMP parallel region ---
    wino_batched_gemm16(V_wino, U_packed, M_wino, num_tiles_padded, Cin, Cout);

    // --- Output transform: per-tile SIMD over co (NHWC-contiguous).
    // Process 8 output channels at once via AVX2. Fuses per-channel bias add.
    #pragma omp parallel for schedule(static) collapse(2)
    for (int th = 0; th < tile_H; th++) {
        for (int tw = 0; tw < tile_W; tw++) {
            int tidx = th * tile_W + tw;
            int co = 0;
            size_t slab = (size_t)num_tiles_padded * Cout;
            const float* base = M_wino + (size_t)tidx * Cout;
#ifdef __AVX2__
            for (; co + 8 <= Cout; co += 8) {
                __m256 m00 = _mm256_loadu_ps(base + 0  * slab + co);
                __m256 m01 = _mm256_loadu_ps(base + 1  * slab + co);
                __m256 m02 = _mm256_loadu_ps(base + 2  * slab + co);
                __m256 m03 = _mm256_loadu_ps(base + 3  * slab + co);
                __m256 m10 = _mm256_loadu_ps(base + 4  * slab + co);
                __m256 m11 = _mm256_loadu_ps(base + 5  * slab + co);
                __m256 m12 = _mm256_loadu_ps(base + 6  * slab + co);
                __m256 m13 = _mm256_loadu_ps(base + 7  * slab + co);
                __m256 m20 = _mm256_loadu_ps(base + 8  * slab + co);
                __m256 m21 = _mm256_loadu_ps(base + 9  * slab + co);
                __m256 m22 = _mm256_loadu_ps(base + 10 * slab + co);
                __m256 m23 = _mm256_loadu_ps(base + 11 * slab + co);
                __m256 m30 = _mm256_loadu_ps(base + 12 * slab + co);
                __m256 m31 = _mm256_loadu_ps(base + 13 * slab + co);
                __m256 m32 = _mm256_loadu_ps(base + 14 * slab + co);
                __m256 m33 = _mm256_loadu_ps(base + 15 * slab + co);
                // Row transform: AtM[i][j] = At_i . M[:,j]
                __m256 atm00 = _mm256_add_ps(_mm256_add_ps(m00, m10), m20);
                __m256 atm01 = _mm256_add_ps(_mm256_add_ps(m01, m11), m21);
                __m256 atm02 = _mm256_add_ps(_mm256_add_ps(m02, m12), m22);
                __m256 atm03 = _mm256_add_ps(_mm256_add_ps(m03, m13), m23);
                __m256 atm10 = _mm256_sub_ps(_mm256_sub_ps(m10, m20), m30);
                __m256 atm11 = _mm256_sub_ps(_mm256_sub_ps(m11, m21), m31);
                __m256 atm12 = _mm256_sub_ps(_mm256_sub_ps(m12, m22), m32);
                __m256 atm13 = _mm256_sub_ps(_mm256_sub_ps(m13, m23), m33);
                // Column transform: Y[i][j] = AtM[i][:] . At^T col j
                __m256 y00 = _mm256_add_ps(_mm256_add_ps(atm00, atm01), atm02);
                __m256 y01 = _mm256_sub_ps(_mm256_sub_ps(atm01, atm02), atm03);
                __m256 y10 = _mm256_add_ps(_mm256_add_ps(atm10, atm11), atm12);
                __m256 y11 = _mm256_sub_ps(_mm256_sub_ps(atm11, atm12), atm13);
                if (bias) {
                    __m256 vb = _mm256_loadu_ps(bias + co);
                    y00 = _mm256_add_ps(y00, vb);
                    y01 = _mm256_add_ps(y01, vb);
                    y10 = _mm256_add_ps(y10, vb);
                    y11 = _mm256_add_ps(y11, vb);
                }
                if (prelu_slope) {
                    __m256 sl = _mm256_loadu_ps(prelu_slope + co);
                    __m256 zero = _mm256_setzero_ps();
                    __m256 mask, neg;
                    mask = _mm256_cmp_ps(y00, zero, _CMP_LT_OS); neg = _mm256_mul_ps(y00, sl);
                    y00 = _mm256_blendv_ps(y00, neg, mask);
                    mask = _mm256_cmp_ps(y01, zero, _CMP_LT_OS); neg = _mm256_mul_ps(y01, sl);
                    y01 = _mm256_blendv_ps(y01, neg, mask);
                    mask = _mm256_cmp_ps(y10, zero, _CMP_LT_OS); neg = _mm256_mul_ps(y10, sl);
                    y10 = _mm256_blendv_ps(y10, neg, mask);
                    mask = _mm256_cmp_ps(y11, zero, _CMP_LT_OS); neg = _mm256_mul_ps(y11, sl);
                    y11 = _mm256_blendv_ps(y11, neg, mask);
                }
                // Scatter to output[oh, ow, co..co+7] for 2×2 tile
                int oh0 = 2 * th, ow0 = 2 * tw;
                if (oh0 < H_out && ow0 < W_out)
                    _mm256_storeu_ps(output + ((size_t)oh0 * W_out + ow0) * Cout + co, y00);
                if (oh0 < H_out && ow0 + 1 < W_out)
                    _mm256_storeu_ps(output + ((size_t)oh0 * W_out + ow0 + 1) * Cout + co, y01);
                if (oh0 + 1 < H_out && ow0 < W_out)
                    _mm256_storeu_ps(output + ((size_t)(oh0 + 1) * W_out + ow0) * Cout + co, y10);
                if (oh0 + 1 < H_out && ow0 + 1 < W_out)
                    _mm256_storeu_ps(output + ((size_t)(oh0 + 1) * W_out + ow0 + 1) * Cout + co, y11);
            }
#endif
            for (; co < Cout; co++) {
                float M[16];
                for (int xn = 0; xn < 16; xn++) {
                    M[xn] = M_wino[(size_t)xn * slab + (size_t)tidx * Cout + co];
                }
                float Y[4];
                output_transform_m4_to_y2(M, Y);
                float b = bias ? bias[co] : 0.0f;
                float sl = prelu_slope ? prelu_slope[co] : 1.0f;
                for (int y = 0; y < 2; y++) {
                    int oh = 2 * th + y;
                    if (oh >= H_out) continue;
                    for (int x = 0; x < 2; x++) {
                        int ow = 2 * tw + x;
                        if (ow >= W_out) continue;
                        float v = Y[y*2 + x] + b;
                        if (prelu_slope && v < 0) v *= sl;
                        output[((size_t)oh * W_out + ow) * Cout + co] = v;
                    }
                }
            }
        }
    }
}


// Reference (naive-GEMM) fallback — retained for A/B testing.
void fastface_winograd_conv_3x3_s1_p1(
    const float* input, int Cin, int H_in, int W_in,
    int Cout, const float* U_wino,
    float* output,
    float* V_wino,   // scratch, at least 16 * tile_H * tile_W * Cin
    float* M_wino)   // scratch, at least 16 * tile_H * tile_W * Cout
{
    int H_out = H_in;
    int W_out = W_in;
    int tile_H = (H_out + 1) / 2;
    int tile_W = (W_out + 1) / 2;
    int num_tiles = tile_H * tile_W;

    // --- Input transform: for each tile, for each ci, compute V[4,4] ---
    #pragma omp parallel for schedule(static) collapse(2)
    for (int th = 0; th < tile_H; th++) {
        for (int tw = 0; tw < tile_W; tw++) {
            int tidx = th * tile_W + tw;
            // Each tile covers output rows [2*th, 2*th+1] × cols [2*tw, 2*tw+1].
            // Input tile covers rows [2*th - 1, 2*th + 2] (4 rows) with pad=1.
            for (int ci = 0; ci < Cin; ci++) {
                float d[16];
                for (int y = 0; y < 4; y++) {
                    for (int x = 0; x < 4; x++) {
                        int ih = 2 * th - 1 + y;
                        int iw = 2 * tw - 1 + x;
                        if (ih >= 0 && ih < H_in && iw >= 0 && iw < W_in) {
                            d[y*4 + x] = input[((size_t)ih * W_in + iw) * Cin + ci];
                        } else {
                            d[y*4 + x] = 0.0f;
                        }
                    }
                }
                float V[16];
                input_transform_d4_to_v4(d, V);
                for (int xn = 0; xn < 16; xn++) {
                    V_wino[(size_t)xn * num_tiles * Cin + (size_t)tidx * Cin + ci] = V[xn];
                }
            }
        }
    }

    // --- Batched GEMM: for each of 16 (xi, nu), [num_tiles, Cin] × [Cin, Cout] = [num_tiles, Cout].
    // Inner loop written for auto-vectorization: outer-product accumulation.
    //   M[t, co] = Σ_ci V[t, ci] * U[ci, co]
    // With ci in the outer position and co as the contiguous axis, the compiler
    // vectorizes the inner `for co` as AVX2 FMA over U[ci, :].
    #pragma omp parallel for schedule(static) collapse(2)
    for (int xn = 0; xn < 16; xn++) {
        for (int t = 0; t < num_tiles; t++) {
            const float* V_row = V_wino + (size_t)xn * num_tiles * Cin + (size_t)t * Cin;
            const float* U_mat = U_wino + (size_t)xn * Cin * Cout;
            float* M_row = M_wino + (size_t)xn * num_tiles * Cout + (size_t)t * Cout;
            // Initialize output row from first ci
            float v0 = V_row[0];
            const float* U_row0 = U_mat + 0 * Cout;
            int co = 0;
#ifdef __AVX2__
            __m256 vb0 = _mm256_set1_ps(v0);
            for (; co + 8 <= Cout; co += 8) {
                _mm256_storeu_ps(M_row + co,
                    _mm256_mul_ps(vb0, _mm256_loadu_ps(U_row0 + co)));
            }
#endif
            for (; co < Cout; co++) M_row[co] = v0 * U_row0[co];
            // Accumulate remaining ci
            for (int ci = 1; ci < Cin; ci++) {
                float v = V_row[ci];
                const float* U_row = U_mat + (size_t)ci * Cout;
                co = 0;
#ifdef __AVX2__
                __m256 vb = _mm256_set1_ps(v);
                for (; co + 8 <= Cout; co += 8) {
                    __m256 acc = _mm256_loadu_ps(M_row + co);
                    acc = _mm256_fmadd_ps(vb, _mm256_loadu_ps(U_row + co), acc);
                    _mm256_storeu_ps(M_row + co, acc);
                }
#endif
                for (; co < Cout; co++) M_row[co] += v * U_row[co];
            }
        }
    }

    // --- Output transform: for each tile, for each co, compute Y[2,2] and scatter ---
    #pragma omp parallel for schedule(static) collapse(2)
    for (int th = 0; th < tile_H; th++) {
        for (int tw = 0; tw < tile_W; tw++) {
            int tidx = th * tile_W + tw;
            for (int co = 0; co < Cout; co++) {
                float M[16];
                for (int xn = 0; xn < 16; xn++) {
                    M[xn] = M_wino[(size_t)xn * num_tiles * Cout + (size_t)tidx * Cout + co];
                }
                float Y[4];
                output_transform_m4_to_y2(M, Y);
                // Scatter to output — skip positions beyond H_out/W_out (internal padding trim)
                for (int y = 0; y < 2; y++) {
                    int oh = 2 * th + y;
                    if (oh >= H_out) continue;
                    for (int x = 0; x < 2; x++) {
                        int ow = 2 * tw + x;
                        if (ow >= W_out) continue;
                        output[((size_t)oh * W_out + ow) * Cout + co] = Y[y*2 + x];
                    }
                }
            }
        }
    }

}
