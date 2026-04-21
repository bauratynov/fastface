# S35 phase 2 — INT8 Winograd AVX2: NEGATIVE RESULT, moonshot aborted

Date: 2026-04-20
Predecessor: S34 phase 1 proved int-exact math.
Outcome: AVX2 Winograd is 0.66x the VNNI im2col path on the target shape.
Decision: abort moonshot, pivot to per-channel cos-sim fix (S36).

## Implementation

`kernels/winograd_int8_avx2.c`:
- Input transform: 16 ci per SIMD iteration via `_mm256_cvtepi8_epi16` + {+/-} arithmetic.
- 16 separate GEMMs via `_mm256_madd_epi16` reduction over Cin.
- Output transform: scalar per-tile, OpenMP parallel.

Correctness: **bit-exact vs scalar reference on 56x56x128x128** (0 / 401,408
non-exact). Integer-exact math proven in S34 carries over.

## Benchmark

Shape: 56x56x128x128 3x3 stride 1 pad 1 (mid-IResNet), 8 threads, 5 trials x 100 iter:

| path | ms/conv |
|---|---:|
| VNNI im2col (current) | **0.513** |
| Winograd AVX2 (this) | 0.775 |

**Winograd is 0.66x the speed (51% SLOWER).**

## Root cause analysis

1. **Throughput gap between instructions:**
   - VNNI `vpdpbusd_epi32`: 32 int8 multiplications + 8 int32 accumulations per cycle per core.
   - Winograd inner is `vpmaddwd` (`_mm256_madd_epi16`): 16 int16 multiplications + 8 int32 pair-sums per cycle per core.
   - VNNI does 2x the multiplications per cycle.

2. **Memory bandwidth doubled:**
   - VNNI reads int8 weights + int8 activations.
   - Winograd reads int16 pre-transformed weights + int16 post-transform activations.
   - At a fixed L2/L3 bandwidth, Winograd halves the effective throughput.

3. **Winograd's arithmetic saving (2.25x) is defeated:**
   - Expected naive: 2.25x fewer FLOPs -> 2.25x faster.
   - Real: 2.25x fewer FLOPs / 2.0x throughput / 2.0x memory = 0.56x expected. Measured 0.66x (slightly better due to pre-transform caching).

## Conclusion

The standard narrative for Winograd speedup assumes the reduced-arithmetic kernel
runs at the SAME throughput as direct convolution. On AVX2 with AVX-VNNI available,
direct INT8 convolution has a dominant throughput advantage that Winograd's int16
kernel can't match.

This result would likely flip with:
- **AVX-512 VNNI**: `vpdpbusd` zmm version does 64 int8 ops/cycle; still same 2x gap
  over `vpmaddwd` zmm (32 int16/cycle). So same conclusion on AVX-512.
- **AVX-VNNI-INT16** (newer): `vpdpwssd` does int16 dot products at 32 ops/cycle.
  Would close the throughput gap. i7-13700 does not have it. Tiger Lake+ servers do.
- **ARM NEON** without VNNI-equivalent: direct int8 uses `sdot` or scalar — Winograd
  should help here. Worth revisiting on ARM.

For i7-13700 x86 AVX-VNNI, **direct im2col+VNNI is the right approach for 3x3 convs**.

## Moonshot decision

Aborting Winograd path per phase 2 gate criteria ("<1.3x speedup => abort"). Cost:
1 session of scalar ref (S34) + 1 session of SIMD (S35) = ~2 sessions. Value: clear
negative result documented in KB; future researchers avoid repeating this dead end.

## Pivot: S36 = per-channel activation quant for cos-sim 0.954 -> 0.986+

With speed moonshot closed, the next-highest-value work is the cos-sim fix that
unblocks production shipping. S21 Python simulation proved per-channel activation
quant reaches 0.986 (vs our current 0.954). Implementation plan:

1. Update `export_op_scales.py` -> v2 emits per-channel scales per op.
2. Update `prepare_weights_v2.py` -> v3 folds per-input-channel activation scales into
   Conv weights before per-OC quantization.
3. Bump FFW3 -> FFW4 with per-channel op_scales structure.
4. Extend `fused_epilogue_int8` + `add_requant_int8` + `bn_prelu_requant_int8` +
   `quantize_fp32_nhwc_to_int8` to take per-channel out_scale/in_scale.
5. Validate end-to-end cos-sim on LFW test set (target mean >=0.98).

Speed impact: expected ~0 ms change (per-channel is same memory pattern, just
different load inside inner loop).

## Artifact: kernels/winograd_int8_avx2.c and winograd_int8_ref.c

Both files are retained in repo as research artifacts. Reference impl is proven
integer-exact and documents the Winograd F(2,3) + G'=2G approach; useful for
future ARM port or when AVX-VNNI-INT16 becomes available.

## Rollback

No production files were modified by S35. The new files are standalone and
built only under `-DWINOGRAD_I8_AVX2_TEST` guard.
