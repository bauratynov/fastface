# S34 phase 1 — INT8 Winograd F(2,3) scalar reference

Date: 2026-04-20
Baseline: S32 INT8 13.36 ms, 2.355x ORT, cos-sim 0.9543.
S33 profile: CONV = 90.7% of remaining time; 3x3 convs are the biggest slice.
Moonshot target: INT8 Winograd F(2,3) -> ~10 ms / 3.15x ORT.

## Math

F(2,3) Winograd computes a 3x3 conv over a 4x4 input tile to produce a
2x2 output tile via:

```
Y = A^T [ (G g G^T) (*) (B^T d B) ] A
```

where (*) is elementwise mult summed over input channels.

Standard F(2,3) matrices:
```
B^T = [[1, 0, -1, 0], [0, 1, 1, 0], [0, -1, 1, 0], [0, 1, 0, -1]]
G   = [[1, 0, 0], [1/2, 1/2, 1/2], [1/2, -1/2, 1/2], [0, 0, 1]]
A^T = [[1, 1, 1, 0], [0, 1, -1, -1]]
```

To keep arithmetic in integers we use `G' = 2G`:
```
G' = [[2, 0, 0], [1, 1, 1], [1, -1, 1], [0, 0, 2]]
```
Then `(G' g G'^T) = 4 (G g G^T)`, so the Winograd output is 4x the correct
conv result. We fold `1/4` into the dequant scale factor downstream.

## Integer accumulator sizing

- **Input transform** `v = B^T d B`. B^T rows use coefficients in {-1, 0, 1},
  so |v| <= 4 * 127 = 508. Fits int16.
- **Weight transform** `u = G' g G'^T`. G' entries max |2|, realistic bound
  |u| <= 1016. Fits int16.
- **Elementwise product** `v * u <= 508 * 1016 = 516K`. Fits int32.
- **Sum over Cin=512** gives |M[i,j]| <= 516K * 512 = 264M. Fits int32 (max 2.1B).
- **Output transform** `A^T M A`. A coefficients in {-1, 0, 1}, sums of 4 int32
  products — still int32-safe (factor < 16).

## Reference impl (scalar)

File `kernels/winograd_int8_ref.c` provides:
- `winograd_int8_transform_weights_ref` — offline: int8 3x3 weights -> int16 4x4 Winograd grid.
- `fastface_conv2d_i8_winograd_f23_ref` — scalar F(2,3) Winograd conv.
- Test driver under `#ifdef WINOGRAD_I8_REF_TEST`.

Direct vs Winograd is integer-exact for integer math (expected Y_wino = 4 * Y_direct).

## Correctness sweep

Stressed with full int8 range [-127, 127] random inputs and weights, stride 1 pad 1.

| shape (H×W×Cin×Cout) | elements | non-exact | max abs err |
|---|---:|---:|---:|
| 8×8×16×8 (tiny) | 512 | 0 | 0 |
| 56×56×128×128 (mid IResNet) | 401,408 | 0 | 0 |
| 28×28×256×256 (deep IResNet) | 200,704 | 0 | 0 |
| 14×14×512×512 (deepest, max Cin) | 100,352 | 0 | 0 |

**PASS: zero error on every tested shape.** Math confirmed across full IResNet-100 range.

Max observed |M[i,j]| value at Cin=512: ~1M. Headroom vs int32 max: ~2000x.
No risk of overflow even with fully saturated activations.

## Phase 2 plan (S35)

Need AVX2 vectorization to make it faster than the existing im2col+GEMM path.

Key steps:
1. **Input transform**: process 8 ci at once per (tile, i, j) position. Entries of B^T are {-1, 0, 1}: just add/sub/negate — 4-5 additions per 4x4 tile.
2. **Batch the tiles**: process n_tiles x Cin int16 input grid for each of 16 (i,j) positions. This becomes 16 separate GEMMs of shape [n_tiles, Cin] x [Cin, Cout].
3. **Per-(i,j) GEMM**: use `_mm256_madd_epi16(v_i16, u_i16)` which gives int32 pair-sums in one instruction. Same throughput as VNNI `dpbusd_epi32` (8 int32 ops/cycle), but operates on int16 instead of int8.
4. **Output transform**: A^T entries in {-1, 0, 1}, 4 adds per 4x4 tile.

Expected perf vs baseline im2col+VNNI:
- FLOPs: 16/9 = 1.78x more data, BUT each elementwise GEMM is only N_tiles x Cin x Cout
  vs 9 x N_tiles x Cin x Cout for direct. Net: 16/(9*9) = 0.20x the FLOPs. Basically
  same as FP32 Winograd theory — 2.25x speedup.
- Memory: weights are int16 so 2x bandwidth per weight byte. Net memory: 16/9 * 2 = 3.5x more.
  Could be a bottleneck for memory-bound shapes (deep IResNet with large Cin).
- Realistic expectation: 1.5-1.8x speedup on the 3x3 convs, 1.2-1.4x on total CONV time.

Target: 12.84 ms CONV -> ~9-10 ms CONV -> total 10.5-11.5 ms / 2.7-3.0x ORT.

## Risk analysis

1. **Memory bandwidth for int16 weights** could dominate for deep convs. If it
   does, we can try switching weights to int8 in Winograd domain (needs rescaling
   via 2x divide — loses 1 bit of precision per transform). Acceptable fallback.
2. **Overhead of 16 small GEMMs** vs 1 large GEMM might hurt small convs (stem 112x112).
   Use Winograd only for mid-deep convs (Cin >= 64 typically). Stem stays on im2col.
3. **Output transform cost**: the 4 additions per tile may amount to non-trivial
   time if not well vectorized. AVX2 can do 8 tiles at once across Cout.

## Scope of S35 (next session)

1. Write AVX2 input transform (`winograd_int8_input_transform_avx2`).
2. Write 16-GEMM driver using `_mm256_madd_epi16`.
3. Write AVX2 output transform.
4. Validate correctness: Winograd output matches scalar reference within 1 LSB.
5. Bench vs im2col+VNNI baseline on one 56x56x128x128 shape. Decide whether to
   integrate into the main path.

If phase 2 bench shows <1.3x speedup on 56x56 shape, downgrade expectation and
abort Winograd moonshot. If >=1.5x, integrate in phase 3.
