# S33 — Re-profile INT8 post-S32

Date: 2026-04-20
Baseline: S32 INT8 13.36 ms median, 2.355x ORT, cos-sim 0.9543.
Goal: with S31 (matvec) and S32 (ADD fusion) both landed, find new biggest bar.

## Result

Running fastface_int8_prof.exe (151 inferences, 2137.8 ms total):

| op type | calls | total ms | % | ms/call | delta vs S30 |
|---|---:|---:|---:|---:|---|
| **CONV** | 8003 | 1939.3 | **90.7** | 0.242 | -87 ms (was 2026) |
| BN | 3926 | 135.1 | 6.3 | 0.034 | -17 ms (was 152) |
| GEMM | 151 | 32.6 | 1.5 | 0.216 | **-1061 ms** (was 1094) ← S31 |
| ADD | 0 | 0 | 0 | — | **-1220 ms** FUSED ← S32 |
| SAVE_ID | 3624 | 16.1 | 0.8 | 0.004 | ~= |
| BLOCK_START | 3624 | 13.5 | 0.6 | 0.004 | ~= |
| FLATTEN | 151 | 1.2 | 0.06 | 0.008 | ~= |

Per inference (14.16 ms raw / 13.36 ms P-core-pinned stable):
- CONV 12.84 ms
- BN 0.90 ms
- GEMM 0.22 ms (was 7.25 ms)
- ADD 0 (was 8.1 ms)
- rest 0.2 ms

## Interpretation

S31 + S32 collapsed their target ops:
- GEMM went from 24.2% → 1.5% (33x smaller per-call).
- ADD went from 27.0% → 0% (fully absorbed into Conv epilogue).

**CONV now dominates at 90.7%.** Further speed gains must come from inside the Conv kernel itself.

## Remaining conv breakdown (estimated)

53 convs per inference, 12.84 ms total. Split by kernel shape:
- ~24 are 3x3 stride-1 pad-1 (Winograd-eligible)
- ~29 are 1x1 (many) + a few stride-2 3x3 + stem 7x7

Conv kernel = im2col + VNNI GEMM. VNNI GEMM already ~4 TOPs effective
(rough estimate). Remaining bandwidth slack:
- im2col writes K_padded * H_out * W_out uint8 per conv. 3x3 im2col expands
  memory by 9x the input size.
- No Winograd path = all 3x3 go through im2col+GEMM at full FLOP count.

## S34+ candidates (ROI-ranked by potential speedup)

1. **INT8 Winograd F(2,3) for 3x3 stride-1 pad-1** — moonshot.
   Math reduces 3x3 FLOPs by 2.25x. If all 24 Winograd-eligible convs
   halve their work: expected -3 to -4 ms = 10 ms / 3.15x ORT.
   Risk: INT8 Winograd requires int16/int32 intermediate accumulators
   because the Winograd transforms involve non-integer values (1/2, -1/2).
   Multi-session (2-3 sessions estimated).

2. **Direct 1x1 Conv (skip im2col memcpy)** — small win.
   For Kh=Kw=1 stride=1 pad=0, im2col is just "XOR 0x80 memcpy". If we
   feed the VNNI GEMM directly on shifted input, save one memory pass.
   Estimated savings: 0.3-0.5 ms.

3. **Fuse BN into preceding Conv epilogue where pattern permits.**
   BN is 0.90 ms, 30 ops. Not all BN directly follow Conv (some precede
   the next block's Conv and get folded in prepare_weights); residual
   BN ops are standalone. Estimated savings: 0.3-0.5 ms.

4. **Per-channel INT8 activation quant (weight-fold variant)** — NOT speed.
   Ships cos-sim 0.954 -> 0.986 (S21 proven). Necessary for biometric
   production use but zero speed impact.

5. **Batched INT8** — port FP32 batching to INT8 path. Could match or
   beat FP32 batched 22.40 ms/face at B=8.

## Takeaway

Two sprints (S31+S32) delivered 15.5 ms of saving. Linear pickings are done.
Next 1 ms needs kernel-level work: either Winograd (moonshot, multi-session)
or incremental im2col/BN improvements (safe, 0.3-0.5 ms each).

Decision for S34: kick off INT8 Winograd project. Phase 1 = scalar reference
impl of F(2,3) with int16 intermediate. Validate correctness on a small
tile before replacing the main path.
