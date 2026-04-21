// Session 19 — Winograd F(4, 3): 6×6 input tile → 4×4 output tile using 3×3 kernel.
//
// Standard matrices from Lavin & Gray 2016 (arxiv 1509.09308), interpolation points (0,1,-1,2,-2):
//
// BT = [[ 4,  0, -5,  0,  1,  0],
//       [ 0, -4, -4,  1,  1,  0],
//       [ 0,  4, -4, -1,  1,  0],
//       [ 0, -2, -1,  2,  1,  0],
//       [ 0,  2, -1, -2,  1,  0],
//       [ 0,  4,  0, -5,  0,  1]]
//
// G  = [[ 1/4,    0,      0   ],
//       [-1/6,  -1/6,   -1/6  ],
//       [-1/6,   1/6,   -1/6  ],
//       [ 1/24,  1/12,   1/6  ],
//       [ 1/24, -1/12,   1/6  ],
//       [  0,    0,       1   ]]
//
// AT = [[1, 1,  1, 1,  1, 0],
//       [0, 1, -1, 2, -2, 0],
//       [0, 1,  1, 4,  4, 0],
//       [0, 1, -1, 8, -8, 1]]
//
// Complexity per 4×4 output tile (one Cin → Cout):
//   36 ewise multiplies (vs 144 for direct 3×3 conv on 4×4 output) = 4× savings.
// Transforms add ~1200 flops of overhead per (tile, channel) but this is amortized
// across many channels in the batched GEMM phase.
//
// Numerical stability is WORSE than F(2,3) but still OK in fp32 (Barbara 2020).
// Do NOT combine with int8 (Mori 2024).

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <immintrin.h>
#include <math.h>

void pack_B_fp32(const float* B, int K, int N, float* Bp);
void fastface_gemm_fp32(const float* A, const float* Bp, float* C, int M, int K, int N);

// --- Scalar transforms ---

// Weight transform: U[6][6] = G * g * G^T, g is 3×3.
static void weight_transform_g3_to_u6(const float* g, float* U)
{
    // G * g: 6×3
    float Gg[6][3];
    for (int j = 0; j < 3; j++) {
        float g0 = g[0*3 + j], g1 = g[1*3 + j], g2 = g[2*3 + j];
        Gg[0][j] =  0.25f * g0;
        Gg[1][j] = -(g0 + g1 + g2) / 6.0f;
        Gg[2][j] = -(g0 - g1 + g2) / 6.0f;
        Gg[3][j] =  (g0 / 24.0f) + (g1 / 12.0f) + (g2 / 6.0f);
        Gg[4][j] =  (g0 / 24.0f) - (g1 / 12.0f) + (g2 / 6.0f);
        Gg[5][j] =  g2;
    }
    // (Gg) * G^T: 6×6
    for (int i = 0; i < 6; i++) {
        float r0 = Gg[i][0], r1 = Gg[i][1], r2 = Gg[i][2];
        U[i*6 + 0] =  0.25f * r0;
        U[i*6 + 1] = -(r0 + r1 + r2) / 6.0f;
        U[i*6 + 2] = -(r0 - r1 + r2) / 6.0f;
        U[i*6 + 3] =  (r0 / 24.0f) + (r1 / 12.0f) + (r2 / 6.0f);
        U[i*6 + 4] =  (r0 / 24.0f) - (r1 / 12.0f) + (r2 / 6.0f);
        U[i*6 + 5] =  r2;
    }
}

// Input transform: V[6][6] = BT * d * BT^T, d is 6×6.
// Row application: BT * d (row dot col).
//   row0 =  4*d0 - 5*d2 + d4
//   row1 = -4*d1 - 4*d2 + d3 + d4
//   row2 =  4*d1 - 4*d2 - d3 + d4
//   row3 = -2*d1 - d2 + 2*d3 + d4
//   row4 =  2*d1 - d2 - 2*d3 + d4
//   row5 =  4*d1 - 5*d3 + d5
static inline void input_transform_d6_to_v6(const float* d, float* V)
{
    float Bd[6][6];
    // BT * d (rows)
    for (int j = 0; j < 6; j++) {
        float d0 = d[0*6+j], d1 = d[1*6+j], d2 = d[2*6+j], d3 = d[3*6+j], d4 = d[4*6+j], d5 = d[5*6+j];
        Bd[0][j] =  4*d0 - 5*d2 + d4;
        Bd[1][j] = -4*d1 - 4*d2 + d3 + d4;
        Bd[2][j] =  4*d1 - 4*d2 - d3 + d4;
        Bd[3][j] = -2*d1 -   d2 + 2*d3 + d4;
        Bd[4][j] =  2*d1 -   d2 - 2*d3 + d4;
        Bd[5][j] =  4*d1 - 5*d3 + d5;
    }
    // Bd * BT^T (cols): same formulas but applied column-wise.
    for (int i = 0; i < 6; i++) {
        float r0 = Bd[i][0], r1 = Bd[i][1], r2 = Bd[i][2], r3 = Bd[i][3], r4 = Bd[i][4], r5 = Bd[i][5];
        V[i*6 + 0] =  4*r0 - 5*r2 + r4;
        V[i*6 + 1] = -4*r1 - 4*r2 + r3 + r4;
        V[i*6 + 2] =  4*r1 - 4*r2 - r3 + r4;
        V[i*6 + 3] = -2*r1 -   r2 + 2*r3 + r4;
        V[i*6 + 4] =  2*r1 -   r2 - 2*r3 + r4;
        V[i*6 + 5] =  4*r1 - 5*r3 + r5;
    }
}

// Output transform: Y[4][4] = AT * M * AT^T, M is 6×6.
// Row application (AT row dot M col):
//   row0 = m0 + m1 + m2 + m3 + m4
//   row1 =      m1 - m2 + 2*m3 - 2*m4
//   row2 =      m1 + m2 + 4*m3 + 4*m4
//   row3 =      m1 - m2 + 8*m3 - 8*m4 + m5
static inline void output_transform_m6_to_y4(const float* M, float* Y)
{
    float AM[4][6];
    for (int j = 0; j < 6; j++) {
        float m0 = M[0*6+j], m1 = M[1*6+j], m2 = M[2*6+j], m3 = M[3*6+j], m4 = M[4*6+j], m5 = M[5*6+j];
        AM[0][j] = m0 + m1 + m2 + m3 + m4;
        AM[1][j] =      m1 - m2 + 2*m3 - 2*m4;
        AM[2][j] =      m1 + m2 + 4*m3 + 4*m4;
        AM[3][j] =      m1 - m2 + 8*m3 - 8*m4 + m5;
    }
    for (int i = 0; i < 4; i++) {
        float r0 = AM[i][0], r1 = AM[i][1], r2 = AM[i][2], r3 = AM[i][3], r4 = AM[i][4], r5 = AM[i][5];
        Y[i*4 + 0] = r0 + r1 + r2 + r3 + r4;
        Y[i*4 + 1] =      r1 - r2 + 2*r3 - 2*r4;
        Y[i*4 + 2] =      r1 + r2 + 4*r3 + 4*r4;
        Y[i*4 + 3] =      r1 - r2 + 8*r3 - 8*r4 + r5;
    }
}


// --- Packed weight precompute: 36 slabs × [Cin, Cout] packed via pack_B_fp32 ---
void winograd_precompute_weights_packed_f43(
    const float* weight, int Cout, int Cin, float* U_packed, float* scratch)
{
    #pragma omp parallel for schedule(static)
    for (int co = 0; co < Cout; co++) {
        float U[36];
        for (int ci = 0; ci < Cin; ci++) {
            const float* g = weight + ((size_t)co * Cin + ci) * 9;
            weight_transform_g3_to_u6(g, U);
            for (int xn = 0; xn < 36; xn++) {
                scratch[(size_t)xn * Cin * Cout + ci * Cout + co] = U[xn];
            }
        }
    }
    #pragma omp parallel for schedule(static)
    for (int xn = 0; xn < 36; xn++) {
        const float* Brow = scratch  + (size_t)xn * Cin * Cout;
        float*       Bp   = U_packed + (size_t)xn * Cin * Cout;
        pack_B_fp32(Brow, Cin, Cout, Bp);
    }
}


// --- Batched 36-matmul GEMM (MR=4, NR=16; uses inline microkernel) ---
#define F43_WMR 4
#define F43_WNR 16

static inline void f43_micro_4x16(
    const float* __restrict__ A, const float* __restrict__ Bp,
    int K, int lda, float* __restrict__ C, int ldc)
{
    __m256 c00 = _mm256_setzero_ps(), c01 = _mm256_setzero_ps();
    __m256 c10 = _mm256_setzero_ps(), c11 = _mm256_setzero_ps();
    __m256 c20 = _mm256_setzero_ps(), c21 = _mm256_setzero_ps();
    __m256 c30 = _mm256_setzero_ps(), c31 = _mm256_setzero_ps();
    for (int k = 0; k < K; k++) {
        __m256 b0 = _mm256_loadu_ps(Bp + k * F43_WNR);
        __m256 b1 = _mm256_loadu_ps(Bp + k * F43_WNR + 8);
        __m256 a0 = _mm256_broadcast_ss(A + 0 * lda + k);
        __m256 a1 = _mm256_broadcast_ss(A + 1 * lda + k);
        __m256 a2 = _mm256_broadcast_ss(A + 2 * lda + k);
        __m256 a3 = _mm256_broadcast_ss(A + 3 * lda + k);
        c00 = _mm256_fmadd_ps(a0, b0, c00); c01 = _mm256_fmadd_ps(a0, b1, c01);
        c10 = _mm256_fmadd_ps(a1, b0, c10); c11 = _mm256_fmadd_ps(a1, b1, c11);
        c20 = _mm256_fmadd_ps(a2, b0, c20); c21 = _mm256_fmadd_ps(a2, b1, c21);
        c30 = _mm256_fmadd_ps(a3, b0, c30); c31 = _mm256_fmadd_ps(a3, b1, c31);
    }
    _mm256_storeu_ps(C + 0*ldc,     c00); _mm256_storeu_ps(C + 0*ldc + 8, c01);
    _mm256_storeu_ps(C + 1*ldc,     c10); _mm256_storeu_ps(C + 1*ldc + 8, c11);
    _mm256_storeu_ps(C + 2*ldc,     c20); _mm256_storeu_ps(C + 2*ldc + 8, c21);
    _mm256_storeu_ps(C + 3*ldc,     c30); _mm256_storeu_ps(C + 3*ldc + 8, c31);
}

static inline void f43_batched_gemm36(
    const float* V, const float* U, float* Mout, int M, int K, int N)
{
    int nt = N / F43_WNR;
    #pragma omp parallel for schedule(static) collapse(3)
    for (int xn = 0; xn < 36; xn++) {
        for (int i = 0; i < M; i += F43_WMR) {
            for (int t = 0; t < nt; t++) {
                const float* V_xi = V + (size_t)xn * M * K;
                const float* U_xi = U + (size_t)xn * K * N;
                float* M_xi = Mout + (size_t)xn * M * N;
                f43_micro_4x16(
                    V_xi + (size_t)i * K,
                    U_xi + (size_t)t * K * F43_WNR,
                    K, K,
                    M_xi + (size_t)i * N + (size_t)t * F43_WNR, N);
            }
        }
    }
}


// --- End-to-end Winograd F(4, 3) Conv ---
// Requires H_out, W_out divisible by 4 (caller pads input to nearest 4 if needed).
// Output: H_out × W_out × Cout NHWC, bias optionally fused per-channel.
void fastface_winograd_conv_3x3_s1_p1_f43_packed_bias(
    const float* input, int Cin, int H_in, int W_in,
    int Cout, const float* U_packed, const float* bias,
    float* output,
    float* V_wino,   // scratch, >= 36 * num_tiles_padded * Cin
    float* M_wino)   // scratch, >= 36 * num_tiles_padded * Cout
{
    int H_out = H_in, W_out = W_in;  // s=1, p=1
    int tile_H = (H_out + 3) / 4;    // ceil(H_out/4)
    int tile_W = (W_out + 3) / 4;
    int num_tiles = tile_H * tile_W;
    int num_tiles_padded = (num_tiles + F43_WMR - 1) & ~(F43_WMR - 1);
    if (num_tiles_padded < F43_WMR) num_tiles_padded = F43_WMR;

    // Zero tail padding rows of V per slab
    if (num_tiles_padded > num_tiles) {
        int pad_rows = num_tiles_padded - num_tiles;
        for (int xn = 0; xn < 36; xn++) {
            memset(V_wino + (size_t)xn * num_tiles_padded * Cin
                   + (size_t)num_tiles * Cin, 0,
                   (size_t)pad_rows * Cin * sizeof(float));
        }
    }

    // --- Input transform: 6×6 tile, stride 4 (tile centers at 4*th, 4*tw) ---
    // Input region of tile (th, tw) spans rows [4*th - 1, 4*th + 4] (6 rows).
    #pragma omp parallel for schedule(static) collapse(2)
    for (int th = 0; th < tile_H; th++) {
        for (int tw = 0; tw < tile_W; tw++) {
            int tidx = th * tile_W + tw;
            for (int ci = 0; ci < Cin; ci++) {
                float d[36];
                for (int y = 0; y < 6; y++) {
                    for (int x = 0; x < 6; x++) {
                        int ih = 4 * th - 1 + y;
                        int iw = 4 * tw - 1 + x;
                        d[y*6 + x] = (ih >= 0 && ih < H_in && iw >= 0 && iw < W_in)
                            ? input[((size_t)ih * W_in + iw) * Cin + ci]
                            : 0.0f;
                    }
                }
                float V[36];
                input_transform_d6_to_v6(d, V);
                for (int xn = 0; xn < 36; xn++) {
                    V_wino[(size_t)xn * num_tiles_padded * Cin
                         + (size_t)tidx * Cin + ci] = V[xn];
                }
            }
        }
    }

    // --- 36 batched GEMMs ---
    f43_batched_gemm36(V_wino, U_packed, M_wino, num_tiles_padded, Cin, Cout);

    // --- Output transform: 4×4 tile + bias ---
    #pragma omp parallel for schedule(static) collapse(2)
    for (int th = 0; th < tile_H; th++) {
        for (int tw = 0; tw < tile_W; tw++) {
            int tidx = th * tile_W + tw;
            for (int co = 0; co < Cout; co++) {
                float M[36];
                for (int xn = 0; xn < 36; xn++) {
                    M[xn] = M_wino[(size_t)xn * num_tiles_padded * Cout
                                 + (size_t)tidx * Cout + co];
                }
                float Y[16];
                output_transform_m6_to_y4(M, Y);
                float b = bias ? bias[co] : 0.0f;
                for (int y = 0; y < 4; y++) {
                    int oh = 4 * th + y;
                    if (oh >= H_out) continue;
                    for (int x = 0; x < 4; x++) {
                        int ow = 4 * tw + x;
                        if (ow >= W_out) continue;
                        output[((size_t)oh * W_out + ow) * Cout + co] = Y[y*4 + x] + b;
                    }
                }
            }
        }
    }
}
