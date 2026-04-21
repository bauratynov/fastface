// Session 9 — NHWC Conv2d kernel.
//
// Input layout [H, W, C] (row-major, channel-minor) — same as GEMM output from previous conv.
// Output layout [H_out, W_out, Cout] — same so next conv can read directly.
//
// im2col becomes SIMPLER with NHWC: each receptive-field position reads Cin values
// contiguously (channel-minor), so inner loop is a plain memcpy of Cin bytes per (kh, kw).

#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <omp.h>
#include <immintrin.h>

void compute_col_sums(const int8_t* B, int K, int N, int32_t* col_sums);
void pack_B_vnni(const int8_t* B, int K, int N, int8_t* Bp);
void fastface_gemm_i8_fused(const uint8_t* Au, const int8_t* Bp, const int32_t* col_sums,
                             int32_t* C, int M, int K, int N);


// NHWC im2col + unsigned shift (+128) fused.
// Input: [H_in, W_in, Cin] int8
// Output: [H_out*W_out, K_padded] uint8 where K_real = Kh*Kw*Cin (and padded to multiple of 4 with 128=u8(0_signed))
//
// Key property: for each output spatial position (oh, ow), the K_real bytes
// laid out as: for kh in 0..Kh: for kw in 0..Kw: for c in 0..Cin: input[ih, iw, c]
// i.e., channel is INNERMOST — matches how Cin values are stored contiguously in NHWC.
void im2col_nhwc_u8(
    const int8_t* input,
    int Cin, int H_in, int W_in,
    int Kh, int Kw, int stride, int pad,
    int H_out, int W_out, int K_padded,
    uint8_t* out)
{
    int K_real = Cin * Kh * Kw;
    #pragma omp parallel for schedule(static)
    for (int oh = 0; oh < H_out; oh++) {
        for (int ow = 0; ow < W_out; ow++) {
            int row = oh * W_out + ow;
            uint8_t* out_row = out + (size_t)row * K_padded;
            int idx = 0;
            for (int kh = 0; kh < Kh; kh++) {
                for (int kw = 0; kw < Kw; kw++) {
                    int ih = oh * stride - pad + kh;
                    int iw = ow * stride - pad + kw;
                    if (ih >= 0 && ih < H_in && iw >= 0 && iw < W_in) {
                        // Channel-minor contiguous copy for this (ih, iw) position.
                        const int8_t* src = input + ((size_t)ih * W_in + iw) * Cin;
                        // Shift-by-128 to u8 while copying
                        int c = 0;
#ifdef __AVX2__
                        const __m256i bias = _mm256_set1_epi8((char)128);
                        for (; c + 32 <= Cin; c += 32) {
                            __m256i v = _mm256_loadu_si256((const __m256i*)(src + c));
                            _mm256_storeu_si256((__m256i*)(out_row + idx), _mm256_add_epi8(v, bias));
                            idx += 32;
                        }
#endif
                        for (; c < Cin; c++) {
                            out_row[idx++] = (uint8_t)((int)src[c] + 128);
                        }
                    } else {
                        // Zero pad (signed 0 → u8 128)
                        memset(out_row + idx, 128, Cin);
                        idx += Cin;
                    }
                }
            }
            // K tail padding
            for (; idx < K_padded; idx++) out_row[idx] = 128;
        }
    }
}


// Pack weight: input [Cout, Cin, Kh, Kw] NCHW → packed NR-panel layout K_padded × Cout.
// Same as v2 pack_conv_weight but input layout reshaped for our VNNI-pack.
void pack_conv_weight_nhwc(
    const int8_t* weight, int Cout, int Cin, int Kh, int Kw, int K_padded,
    int8_t* w_rowmajor, int8_t* w_packed, int32_t* col_sums)
{
    // Target layout: w_rowmajor[k, co] where k iterates (kh, kw, ci) — channel-innermost
    // to match NHWC im2col order. So k = (kh*Kw + kw) * Cin + ci.
    memset(w_rowmajor, 0, (size_t)K_padded * Cout);
    for (int co = 0; co < Cout; co++) {
        for (int kh = 0; kh < Kh; kh++) {
            for (int kw = 0; kw < Kw; kw++) {
                for (int ci = 0; ci < Cin; ci++) {
                    int k = (kh * Kw + kw) * Cin + ci;
                    // Input weight stored as [Cout, Cin, Kh, Kw]
                    w_rowmajor[(size_t)k * Cout + co] =
                        weight[((size_t)co * Cin + ci) * Kh * Kw + kh * Kw + kw];
                }
            }
        }
    }
    pack_B_vnni(w_rowmajor, K_padded, Cout, w_packed);
    compute_col_sums(w_rowmajor, K_padded, Cout, col_sums);
}


// NHWC Conv2d: input [H_in, W_in, Cin] int8 → output [H_out*W_out, Cout] int32.
// Output is naturally NHWC-flat since GEMM writes [M, N] = [spatial, channels].
// NO transpose needed. Next op reads this same layout.
// S44: fast path for 1x1 conv stride=1 pad=0. im2col is just a shifted copy,
// so we XOR 0x80 directly on input into scratch_Au and call GEMM — saves
// the 9x im2col expansion overhead that a general im2col does.
static inline void shift_i8_to_u8_avx2(const int8_t* src, uint8_t* dst, size_t n) {
#ifdef __AVX2__
    const __m256i bias = _mm256_set1_epi8((char)128);
    size_t i = 0;
    for (; i + 32 <= n; i += 32) {
        __m256i v = _mm256_loadu_si256((const __m256i*)(src + i));
        _mm256_storeu_si256((__m256i*)(dst + i), _mm256_add_epi8(v, bias));
    }
    for (; i < n; i++) dst[i] = (uint8_t)((int)src[i] + 128);
#else
    for (size_t i = 0; i < n; i++) dst[i] = (uint8_t)((int)src[i] + 128);
#endif
}

// S44: 1x1 stride=1 pad=0 direct path. Needs Cin multiple of 4 for the VNNI GEMM.
static void fastface_conv2d_i8_nhwc_1x1(
    const int8_t* input, int Cin, int H, int W,
    int Cout,
    const int8_t* weight_packed, const int32_t* col_sums,
    int32_t* output,
    uint8_t* scratch_Au)
{
    int M = H * W;
    int K_real = Cin;
    int K_padded = (K_real + 3) & ~3;
    if (K_padded == K_real) {
        // No padding needed: shift in-place into scratch_Au
        shift_i8_to_u8_avx2(input, scratch_Au, (size_t)M * Cin);
    } else {
        // Per-row pad with 128 (= unsigned 0 zero-point)
        for (int row = 0; row < M; row++) {
            shift_i8_to_u8_avx2(input + (size_t)row * Cin,
                                scratch_Au + (size_t)row * K_padded, Cin);
            for (int k = K_real; k < K_padded; k++) {
                scratch_Au[(size_t)row * K_padded + k] = 128;
            }
        }
    }
    fastface_gemm_i8_fused(scratch_Au, weight_packed, col_sums, output, M, K_padded, Cout);
}

void fastface_conv2d_i8_nhwc(
    const int8_t* input, int Cin, int H_in, int W_in,
    int Cout, int Kh, int Kw, int stride, int pad,
    const int8_t* weight_packed, const int32_t* col_sums,
    int H_out, int W_out,
    int32_t* output,
    uint8_t* scratch_Au)
{
    // S44 fast path for 1x1 s=1 pad=0 conv
    if (Kh == 1 && Kw == 1 && stride == 1 && pad == 0) {
        fastface_conv2d_i8_nhwc_1x1(input, Cin, H_in, W_in, Cout,
                                    weight_packed, col_sums, output, scratch_Au);
        return;
    }
    int K_real = Cin * Kh * Kw;
    int K_padded = (K_real + 3) & ~3;
    int M = H_out * W_out;

    im2col_nhwc_u8(input, Cin, H_in, W_in, Kh, Kw, stride, pad,
                   H_out, W_out, K_padded, scratch_Au);
    fastface_gemm_i8_fused(scratch_Au, weight_packed, col_sums, output, M, K_padded, Cout);
}

// S45: batched conv. Packs B faces' im2col outputs sequentially into scratch_Au,
// then calls a single large GEMM with M = B * H_out * W_out. Amortizes VNNI
// weight loads across all B faces.
//   input: [B, H_in, W_in, Cin] int8 contiguous
//   output: [B, H_out, W_out, Cout] int32 contiguous
//   scratch_Au: at least B * H_out * W_out * K_padded bytes
void fastface_conv2d_i8_nhwc_batched(
    const int8_t* input, int B, int Cin, int H_in, int W_in,
    int Cout, int Kh, int Kw, int stride, int pad,
    const int8_t* weight_packed, const int32_t* col_sums,
    int H_out, int W_out,
    int32_t* output,
    uint8_t* scratch_Au)
{
    int K_real = Cin * Kh * Kw;
    int K_padded = (K_real + 3) & ~3;
    int M_face = H_out * W_out;
    size_t input_stride  = (size_t)Cin  * H_in  * W_in;
    size_t scratch_stride = (size_t)M_face * K_padded;

    if (Kh == 1 && Kw == 1 && stride == 1 && pad == 0) {
        // Batched 1x1 fast path
        for (int b = 0; b < B; b++) {
            const int8_t* inb = input + b * input_stride;
            uint8_t* scrb = scratch_Au + b * scratch_stride;
            if (K_padded == K_real) {
                shift_i8_to_u8_avx2(inb, scrb, (size_t)M_face * Cin);
            } else {
                for (int row = 0; row < M_face; row++) {
                    shift_i8_to_u8_avx2(inb + (size_t)row * Cin,
                                        scrb + (size_t)row * K_padded, Cin);
                    for (int k = K_real; k < K_padded; k++)
                        scrb[(size_t)row * K_padded + k] = 128;
                }
            }
        }
    } else {
        // General im2col for B faces
        for (int b = 0; b < B; b++) {
            im2col_nhwc_u8(input + b * input_stride, Cin, H_in, W_in,
                           Kh, Kw, stride, pad, H_out, W_out, K_padded,
                           scratch_Au + b * scratch_stride);
        }
    }
    // One large GEMM: M = B * M_face. Output interleaved per-batch.
    fastface_gemm_i8_fused(scratch_Au, weight_packed, col_sums, output,
                           B * M_face, K_padded, Cout);
}


// Final Gemm: flattened int8 input [1, K=2048] × weight [N=512, K=2048] → fp32 output [512].
// M=1, so GEMM does a single row-dot. We use our int8 GEMM which is overkill for M=1 but works.
void fastface_fc_i8(
    const int8_t* input_flat,    // [K]
    const int8_t* weight_packed, // N/NR × K × NR
    const int32_t* col_sums,     // [N]
    int32_t* output_i32,         // [N] int32
    const float* scales,         // [N] per-row fp32 scale
    const float* bias,           // [N]
    float* output_fp32,          // [N] final embedding
    int K, int N)
{
    // Pad M to 4 (our GEMM microkernel needs MR=4). Use 1 real row + 3 zero-padded.
    static uint8_t Au_padded[4 * 4096];  // max K*MR
    int K_padded = (K + 3) & ~3;
    memset(Au_padded, 128, sizeof(Au_padded));  // u8(0_signed) = 128
    for (int k = 0; k < K; k++) Au_padded[k] = (uint8_t)((int)input_flat[k] + 128);

    int32_t C_padded[4 * 512];  // max N
    fastface_gemm_i8_fused(Au_padded, weight_packed, col_sums, C_padded, 4, K_padded, N);

    // Take row 0 only, dequant + add bias
    for (int n = 0; n < N; n++) {
        output_i32[n] = C_padded[n];
        output_fp32[n] = (float)C_padded[n] * scales[n] + bias[n];
    }
}
