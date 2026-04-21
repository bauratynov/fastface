// Session 26 — INT8 post-Conv fused epilogue kernels.
//
// After int8 Conv: int32 accumulator [N_pos, Cout] → int8 [N_pos, Cout] using
// per-output-channel dequant scale + optional bias + optional BN + optional PReLU,
// then requantize with calibrated next-layer scale.
//
// Math per (position p, channel co):
//   fp = acc[p, co] * (in_scale * weight_scale[co])
//   fp += bias[co]              (if bias != NULL)
//   fp = fp * bn_scale[co] + bn_offset[co]   (if bn_scale != NULL)
//   fp = (fp >= 0) ? fp : fp * prelu_slope[co]   (if slope != NULL)
//   q  = clip(lrintf(fp / out_scale), -128, 127)
//   out[p, co] = q

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include <immintrin.h>


// S36/S38: inv_out_per_ch optional per-channel inverse-scale for requant.
// S38: add_scale_per_ch optional per-channel shortcut scale.
void fused_epilogue_int8(
    const int32_t* acc, int N_pos, int Cout,
    float in_scale,
    const float* weight_scales,
    const float* bias,
    const float* bn_scale,
    const float* bn_offset,
    const float* prelu_slope,
    const int8_t* add_src, float add_scale,
    const float* add_scale_per_ch,  // S38: optional per-channel shortcut scale
    const float* inv_out_per_ch,    // S36: optional per-channel inv_out
    float out_scale,
    int8_t* out_i8)
{
    float inv_out_scalar = 1.0f / (out_scale + 1e-9f);

    #pragma omp parallel for schedule(static)
    for (int p = 0; p < N_pos; p++) {
        const int32_t* acc_row = acc + (size_t)p * Cout;
        const int8_t*  add_row = add_src ? add_src + (size_t)p * Cout : NULL;
        int8_t* out_row = out_i8 + (size_t)p * Cout;
        int co = 0;
#ifdef __AVX2__
        __m256 v_inscale = _mm256_set1_ps(in_scale);
        __m256 v_addscale = _mm256_set1_ps(add_scale);
        __m256 v_invout_scalar = _mm256_set1_ps(inv_out_scalar);
        __m256 v_half    = _mm256_set1_ps(0.5f);
        for (; co + 8 <= Cout; co += 8) {
            __m256i vi32 = _mm256_loadu_si256((const __m256i*)(acc_row + co));
            __m256 vfp   = _mm256_cvtepi32_ps(vi32);
            __m256 vws   = _mm256_loadu_ps(weight_scales + co);
            vfp = _mm256_mul_ps(_mm256_mul_ps(vfp, v_inscale), vws);
            if (bias) vfp = _mm256_add_ps(vfp, _mm256_loadu_ps(bias + co));
            if (add_row) {
                __m128i ai8 = _mm_loadl_epi64((const __m128i*)(add_row + co));
                __m256 v_as_eff = add_scale_per_ch
                    ? _mm256_loadu_ps(add_scale_per_ch + co)
                    : v_addscale;
                __m256 af  = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(ai8)), v_as_eff);
                vfp = _mm256_add_ps(vfp, af);
            }
            if (bn_scale) {
                __m256 bs = _mm256_loadu_ps(bn_scale + co);
                __m256 bo = _mm256_loadu_ps(bn_offset + co);
                vfp = _mm256_fmadd_ps(vfp, bs, bo);
            }
            if (prelu_slope) {
                __m256 sl = _mm256_loadu_ps(prelu_slope + co);
                __m256 neg = _mm256_mul_ps(vfp, sl);
                __m256 mask = _mm256_cmp_ps(vfp, _mm256_setzero_ps(), _CMP_LT_OS);
                vfp = _mm256_blendv_ps(vfp, neg, mask);
            }
            __m256 v_invout = inv_out_per_ch
                ? _mm256_loadu_ps(inv_out_per_ch + co)
                : v_invout_scalar;
            vfp = _mm256_mul_ps(vfp, v_invout);
            // Round-to-nearest: add +/- 0.5 then truncate
            __m256 sign_bit = _mm256_and_ps(vfp, _mm256_castsi256_ps(_mm256_set1_epi32(0x80000000)));
            __m256 rounder  = _mm256_or_ps(v_half, sign_bit);
            __m256i vi     = _mm256_cvttps_epi32(_mm256_add_ps(vfp, rounder));
            // Clamp to int8 range
            __m256i vclo = _mm256_max_epi32(vi, _mm256_set1_epi32(-128));
            __m256i vchi = _mm256_min_epi32(vclo, _mm256_set1_epi32(127));
            // Pack 8 int32 -> 8 int8 (via int16 intermediate)
            __m128i lo = _mm256_castsi256_si128(vchi);
            __m128i hi = _mm256_extracti128_si256(vchi, 1);
            __m128i i16 = _mm_packs_epi32(lo, hi);
            __m128i i8  = _mm_packs_epi16(i16, i16);
            _mm_storel_epi64((__m128i*)(out_row + co), i8);
        }
#endif
        for (; co < Cout; co++) {
            float fp = (float)acc_row[co] * (in_scale * weight_scales[co]);
            if (bias) fp += bias[co];
            if (add_row) {
                float as_eff = add_scale_per_ch ? add_scale_per_ch[co] : add_scale;
                fp += (float)add_row[co] * as_eff;
            }
            if (bn_scale) fp = fp * bn_scale[co] + bn_offset[co];
            if (prelu_slope && fp < 0) fp *= prelu_slope[co];
            float inv_out = inv_out_per_ch ? inv_out_per_ch[co] : inv_out_scalar;
            int q = (int)lrintf(fp * inv_out);
            if (q > 127) q = 127;
            if (q < -128) q = -128;
            out_row[co] = (int8_t)q;
        }
    }
}


// Dequant + add + requant for int8 residual ADD.
void add_requant_int8(
    const int8_t* a, float a_scale,
    const int8_t* b, float b_scale,
    int8_t* out, float out_scale,
    int n)
{
    float inv_out = 1.0f / (out_scale + 1e-9f);
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++) {
        float fp = (float)a[i] * a_scale + (float)b[i] * b_scale;
        int q = (int)lrintf(fp * inv_out);
        if (q > 127) q = 127;
        if (q < -128) q = -128;
        out[i] = (int8_t)q;
    }
}

/* S105: per-channel variant. Closes the sim-binary gap on pure OP_ADD
 * (BN->Add path) where scalar scales were a lossy approximation of the
 * per-channel scales already tracked but not used.
 * NHWC layout: channel is innermost.
 */
void add_requant_int8_pc(
    const int8_t* a, const float* a_scale_pc,
    const int8_t* b, const float* b_scale_pc,
    int8_t* out, const float* out_inv_scale_pc,
    int N_pos, int C)
{
    #pragma omp parallel for schedule(static)
    for (int p = 0; p < N_pos; p++) {
        const int8_t* ar = a + (size_t)p * C;
        const int8_t* br = b + (size_t)p * C;
        int8_t* orow     = out + (size_t)p * C;
        for (int c = 0; c < C; c++) {
            float fp = (float)ar[c] * a_scale_pc[c] + (float)br[c] * b_scale_pc[c];
            int q = (int)lrintf(fp * out_inv_scale_pc[c]);
            if (q > 127) q = 127;
            if (q < -128) q = -128;
            orow[c] = (int8_t)q;
        }
    }
}

// Fused ADD + per-channel BN + requant for the Add→BN→next-Conv pattern.
// Saves one full memory pass by applying BN inline during requant.
// N_pos is spatial positions count, C is channels (innermost in NHWC).
void add_bn_requant_int8(
    const int8_t* a, float a_scale,
    const int8_t* b, float b_scale,
    const float* bn_scale, const float* bn_offset,
    int8_t* out, float out_scale,
    int N_pos, int C)
{
    float inv_out = 1.0f / (out_scale + 1e-9f);
    #pragma omp parallel for schedule(static)
    for (int p = 0; p < N_pos; p++) {
        const int8_t* ar = a + (size_t)p * C;
        const int8_t* br = b + (size_t)p * C;
        int8_t* orow = out + (size_t)p * C;
        int c = 0;
#ifdef __AVX2__
        __m256 v_as = _mm256_set1_ps(a_scale);
        __m256 v_bs = _mm256_set1_ps(b_scale);
        __m256 v_inv = _mm256_set1_ps(inv_out);
        __m256 v_half = _mm256_set1_ps(0.5f);
        for (; c + 8 <= C; c += 8) {
            __m128i ai8 = _mm_loadl_epi64((const __m128i*)(ar + c));
            __m128i bi8 = _mm_loadl_epi64((const __m128i*)(br + c));
            __m256 af = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(ai8)), v_as);
            __m256 bf = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(bi8)), v_bs);
            __m256 sum = _mm256_add_ps(af, bf);
            if (bn_scale) {
                __m256 bs = _mm256_loadu_ps(bn_scale + c);
                __m256 bo = _mm256_loadu_ps(bn_offset + c);
                sum = _mm256_fmadd_ps(sum, bs, bo);
            }
            sum = _mm256_mul_ps(sum, v_inv);
            __m256 sign_bit = _mm256_and_ps(sum, _mm256_castsi256_ps(_mm256_set1_epi32(0x80000000)));
            __m256 rounder = _mm256_or_ps(v_half, sign_bit);
            __m256i vi = _mm256_cvttps_epi32(_mm256_add_ps(sum, rounder));
            __m256i vclo = _mm256_max_epi32(vi, _mm256_set1_epi32(-128));
            __m256i vchi = _mm256_min_epi32(vclo, _mm256_set1_epi32(127));
            __m128i lo = _mm256_castsi256_si128(vchi);
            __m128i hi = _mm256_extracti128_si256(vchi, 1);
            __m128i i16 = _mm_packs_epi32(lo, hi);
            __m128i i8  = _mm_packs_epi16(i16, i16);
            _mm_storel_epi64((__m128i*)(orow + c), i8);
        }
#endif
        for (; c < C; c++) {
            float fp = (float)ar[c] * a_scale + (float)br[c] * b_scale;
            if (bn_scale) fp = fp * bn_scale[c] + bn_offset[c];
            int q = (int)lrintf(fp * inv_out);
            if (q > 127) q = 127;
            if (q < -128) q = -128;
            orow[c] = (int8_t)q;
        }
    }
}


// Quantize fp32 → int8 NHWC with given scale (no runtime absmax; scale precomputed).
void quantize_fp32_nhwc_to_int8(const float* in, int N, float scale, int8_t* out) {
    float inv = 1.0f / (scale + 1e-9f);
    #pragma omp parallel for schedule(static) if (N > 4096)
    for (int i = 0; i < N; i++) {
        int q = (int)lrintf(in[i] * inv);
        if (q > 127) q = 127;
        if (q < -128) q = -128;
        out[i] = (int8_t)q;
    }
}

// S38: per-channel quantize for NHWC tensors. N_pos = H*W, C = channels.
// inv_scale_per_ch[c] = 1 / S_c[c]. Memory: NHWC means channels are innermost.
void quantize_fp32_nhwc_to_int8_per_channel(
    const float* in, int N_pos, int C,
    const float* inv_scale_per_ch,
    int8_t* out)
{
    #pragma omp parallel for schedule(static) if (N_pos > 64)
    for (int p = 0; p < N_pos; p++) {
        const float* ir = in + (size_t)p * C;
        int8_t* or_ = out + (size_t)p * C;
        for (int c = 0; c < C; c++) {
            int q = (int)lrintf(ir[c] * inv_scale_per_ch[c]);
            if (q > 127) q = 127;
            if (q < -128) q = -128;
            or_[c] = (int8_t)q;
        }
    }
}


// Standalone int8 BN + PReLU with requant (for ops not following Conv).
// S37: inv_out_per_ch optional — when non-NULL overrides scalar out_scale.
// S38: in_scale_per_ch optional — when non-NULL overrides scalar in_scale.
void bn_prelu_requant_int8(
    const int8_t* in_i8, float in_scale,
    const float* in_scale_per_ch,
    const float* bn_scale, const float* bn_offset,
    const float* prelu_slope,
    const float* inv_out_per_ch,
    float out_scale,
    int8_t* out_i8, int N_pos, int C)
{
    float inv_out_scalar = 1.0f / (out_scale + 1e-9f);
    #pragma omp parallel for schedule(static)
    for (int p = 0; p < N_pos; p++) {
        const int8_t* in_row = in_i8 + (size_t)p * C;
        int8_t* out_row = out_i8 + (size_t)p * C;
        int c = 0;
#ifdef __AVX2__
        __m256 v_inscale_scalar = _mm256_set1_ps(in_scale);
        __m256 v_invout_scalar = _mm256_set1_ps(inv_out_scalar);
        __m256 v_half    = _mm256_set1_ps(0.5f);
        for (; c + 8 <= C; c += 8) {
            __m128i i8 = _mm_loadl_epi64((const __m128i*)(in_row + c));
            __m256i i32 = _mm256_cvtepi8_epi32(i8);
            __m256 v_inscale = in_scale_per_ch
                ? _mm256_loadu_ps(in_scale_per_ch + c)
                : v_inscale_scalar;
            __m256 vfp = _mm256_mul_ps(_mm256_cvtepi32_ps(i32), v_inscale);
            if (bn_scale) {
                __m256 bs = _mm256_loadu_ps(bn_scale + c);
                __m256 bo = _mm256_loadu_ps(bn_offset + c);
                vfp = _mm256_fmadd_ps(vfp, bs, bo);
            }
            if (prelu_slope) {
                __m256 sl = _mm256_loadu_ps(prelu_slope + c);
                __m256 neg = _mm256_mul_ps(vfp, sl);
                __m256 mask = _mm256_cmp_ps(vfp, _mm256_setzero_ps(), _CMP_LT_OS);
                vfp = _mm256_blendv_ps(vfp, neg, mask);
            }
            __m256 v_invout = inv_out_per_ch
                ? _mm256_loadu_ps(inv_out_per_ch + c)
                : v_invout_scalar;
            vfp = _mm256_mul_ps(vfp, v_invout);
            __m256 sign_bit = _mm256_and_ps(vfp, _mm256_castsi256_ps(_mm256_set1_epi32(0x80000000)));
            __m256 rounder  = _mm256_or_ps(v_half, sign_bit);
            __m256i vi = _mm256_cvttps_epi32(_mm256_add_ps(vfp, rounder));
            __m256i vclo = _mm256_max_epi32(vi, _mm256_set1_epi32(-128));
            __m256i vchi = _mm256_min_epi32(vclo, _mm256_set1_epi32(127));
            __m128i lo = _mm256_castsi256_si128(vchi);
            __m128i hi = _mm256_extracti128_si256(vchi, 1);
            __m128i i16 = _mm_packs_epi32(lo, hi);
            __m128i out8  = _mm_packs_epi16(i16, i16);
            _mm_storel_epi64((__m128i*)(out_row + c), out8);
        }
#endif
        for (; c < C; c++) {
            float is_eff = in_scale_per_ch ? in_scale_per_ch[c] : in_scale;
            float fp = (float)in_row[c] * is_eff;
            if (bn_scale) fp = fp * bn_scale[c] + bn_offset[c];
            if (prelu_slope && fp < 0) fp *= prelu_slope[c];
            float inv_out = inv_out_per_ch ? inv_out_per_ch[c] : inv_out_scalar;
            int q = (int)lrintf(fp * inv_out);
            if (q > 127) q = 127;
            if (q < -128) q = -128;
            out_row[c] = (int8_t)q;
        }
    }
}
