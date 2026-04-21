# FastFace — ARM NEON porting plan (S55 reconnaissance)

**Goal:** port the ship-path INT8 driver (arcface_forward_int8.c + kernels/) from
x86 AVX2/AVX-VNNI to ARM64 NEON so FastFace runs on Apple Silicon, Snapdragon,
RK3588, Ambarella CV72S, etc.

## Current state (S55)

- x86 AVX-VNNI build delivers 13.27 ms/face b=1, 2.375× ORT, cos-sim=FP32 on LFW.
- No ARM cross-compiler in local environment. Requires separate setup step.
- No ARM hardware for timing validation. Bringup can be done via QEMU user-mode
  emulation for correctness, then handed to ARM device for real benchmarks.

## Required toolchain

Pick ONE of:

- **Linux cross**: `apt install gcc-aarch64-linux-gnu g++-aarch64-linux-gnu`.
  Build with `aarch64-linux-gnu-gcc -march=armv8.2-a+dotprod`.
- **Apple Silicon native**: install Xcode on a Mac, build locally with
  `clang -arch arm64 -mcpu=apple-m1`. Easiest path if Mac is available.
- **Docker ARM**: `docker run --platform linux/arm64 -it arm64v8/ubuntu`
  with QEMU. Slow but correctness-only.

## Instruction mapping (concrete per kernel)

### `kernels/conv2d_nhwc.c` (im2col + 1x1 fast path) — 8 intrinsics, LOW effort

| x86 intrinsic | count | NEON equivalent | notes |
|---|---:|---|---|
| `_mm256_loadu_si256` | 2 | `vld1q_u8` + extra load | 256b → two 128b loads |
| `_mm256_storeu_si256` | 2 | `vst1q_u8` + extra store | same |
| `_mm256_add_epi8` | 1 | `vaddq_s8` | direct |
| `_mm256_set1_epi8` | 1 | `vdupq_n_s8` | direct |

Effort: ~30 min. Trivially portable — just 8-bit add/broadcast.

### `kernels/gemm_int8_v2.c` (VNNI GEMM) — 36 intrinsics, MEDIUM effort

| x86 intrinsic | count | NEON equivalent | notes |
|---|---:|---|---|
| `_mm256_dpbusd_epi32` (`#ifdef __AVXVNNI__`) | 4 | `vdotq_s32` (`+dotprod`) | NEON has `sdot`/`udot` — signed-vs-unsigned-handled differently |
| `_mm256_maddubs_epi16` (fallback) | 4 | emulate via `vmull_s8` + `vpadalq_s16` | no direct equiv |
| `_mm256_madd_epi16` | 4 | `vpaddlq_s32(vmull_s16)` | int16×int16 pair add |
| `_mm256_set1_epi32`, `_mm256_setzero_si256` | 6 | `vdupq_n_s32`, `vdupq_n_s32(0)` | direct |
| `_mm256_slli_epi32` | 1 | `vshlq_n_s32` | direct |
| `_mm256_sub_epi32` | 4 | `vsubq_s32` | direct |
| `_mm256_storeu_si256` | 4 | `vst1q_s32` (×2) | split |
| `_mm256_loadu_si256` | 4 | `vld1q_s8`/`vld1q_s32` | split |
| others | 5 | various | — |

Critical note: `_mm256_dpbusd_epi32` takes (u8, i8) → i32. NEON `vdotq_s32`
takes (i8, i8). Signed trick: we already do `Au = A XOR 0x80` so `Au` is
uint8, but NEON `udotq_u32` would accumulate to u32. For signed result,
use `vdotq_s32` directly on the int8 (skip the XOR) and adjust col_sums
compensation. Or stick with the maddubs fallback path (no NEON dotprod needed).

Effort: ~2-3 sessions. Dotprod variant requires care.

### `kernels/int8_epilogue.c` (fused dequant/bias/BN/PReLU/ADD/requant) — 107 intrinsics, HIGH effort

This is the largest kernel. Intrinsics are mostly fp32 ops:

| x86 intrinsic | count | NEON equivalent |
|---|---:|---|
| `_mm256_loadu_ps`, `_mm256_storeu_ps` | 14+ | `vld1q_f32`×2, `vst1q_f32`×2 (half width) |
| `_mm256_mul_ps`, `_mm256_add_ps` | 17 | `vmulq_f32`, `vaddq_f32` |
| `_mm256_fmadd_ps` | ~5 | `vfmaq_f32` |
| `_mm256_set1_ps` | 11 | `vdupq_n_f32` |
| `_mm256_cmp_ps(LT)` | ~2 | `vcltq_f32` |
| `_mm256_blendv_ps` | ~2 | `vbslq_f32` |
| `_mm256_cvtepi32_ps` | 5 | `vcvtq_f32_s32` |
| `_mm256_cvttps_epi32` | ~3 | `vcvtq_s32_f32` |
| `_mm256_cvtepi8_epi32` | 4 | `vmovl_s8` + `vmovl_s16` |
| `_mm_packs_epi16`, `_mm_packs_epi32` | ~4 | `vqmovn_s16`, `vqmovn_s32` |
| `_mm_loadl_epi64`, `_mm_storel_epi64` | 6 | `vld1_s8`, `vst1_s8` |

Effort: ~1-2 sessions. Mechanical translation, lots of lines but each
op has a clean NEON equivalent.

### `kernels/gemm_int8_matvec.c` (final Linear) — 31 intrinsics, MEDIUM effort

Same VNNI dpbusd pattern as `gemm_int8_v2.c` but M=1. Uses:
- `_mm256_dpbusd_epi32` or `_mm256_maddubs_epi16` + `_mm256_madd_epi16` fallback
- XOR-shift on input (128 bytes per call)
- Horizontal sum `hsum_epi32_avx2`

NEON mapping similar to gemm_int8_v2. Horizontal sum with `vaddvq_s32`
(single instruction on ARMv8, cleaner than AVX2 hsum dance).

Effort: ~1 session.

## Total estimated effort

~5-8 focused sessions for a correct NEON port. First milestone: passing
regression test under QEMU emulation. Second milestone: timed on real
Apple Silicon / RK3588.

## Expected performance on ARM

- Apple M-series has NEON + `+dotprod` + even Apple `amx` undocumented
  matrix coprocessor. Expected ~10-15 ms/face (similar to i7 class, maybe
  faster for small matmuls).
- RK3588 / Snapdragon 8cx: NEON only, no AVX-VNNI equivalent. Expected
  ~25-35 ms/face (SDOT helps, but clock/cache is weaker).
- Ambarella CV72S: has NPU; NEON path is fallback. Would want to wire into
  CVflow for peak perf, but NEON baseline is a useful reference.

## Order of attack

1. Fork `kernels/conv2d_nhwc_neon.c` and do the 8-intrinsic 1x1 path first.
   Test standalone.
2. Port `gemm_int8_v2_neon.c` using `sdot`. Test with standalone microbench.
3. Port `int8_epilogue_neon.c` — mechanical.
4. Port `gemm_int8_matvec_neon.c` — similar to gemm_int8_v2.
5. Build `fastface_int8.exe` (aarch64) — link.
6. Run under QEMU vs golden input → should pass regression test.
7. Hand off to ARM device for real timing.

## What this S55 sprint produced

- This map.
- `tests/run_regression.py` (S54) is the bar any ARM port must clear
  before being accepted. Golden input bytes and FP32/INT8 references
  are bit-level, not platform-specific — they work on ARM as oracle.

## NOT done this sprint

- Actual code port (multi-session work).
- QEMU bringup.
- ARM-specific calibration tuning (initial test should use same FFW4).
