# S107 -- Weight-fold rounding IS the sim-binary gap source

**Date:** 2026-04-21
**Status:** POSITIVE. Gap source identified and quantified.

## Test

Modify the Python sim to quantize `w_folded = w_orig * S_a[ci]` per-OC
(matching binary), then divide by S_a[ci] after dequant to recover an
effective `w_recovered`. Run Conv with `w_recovered` on per-channel-quanted
fp32 input. If this matches the binary's mathematical operation, sim mean
should drop from 0.99623 toward the binary's 0.99326.

## Result (Python sim, 200 calib WITH_PRINCESS, 100 test faces seed 7777)

| sim variant | mean | median | min | >=0.99 |
|---|---:|---:|---:|---:|
| UNFOLDED weight quant (sim default, S98-S106) | 0.99623 | 0.99663 | 0.98525 | 99/100 |
| **FOLDED weight quant (binary-match)**        | **0.99208** | 0.99232 | 0.98048 | 89/100 |
| Reference: C binary on same faces             | 0.99326 | 0.99360 | 0.98338 | 92/100 |

Delta folded vs unfolded sim: **-0.00415**
Delta folded sim vs binary:   **-0.00118**

## Interpretation

Weight-fold rounding IS a major error source (~0.004 mean cos-sim worth).
The binary sits BETWEEN the two sim variants because:

| source          | unfolded sim | folded sim | binary |
|---|---|---|---|
| weight quant error | low (unfolded) | moderate (folded) | moderate (folded) |
| accumulation       | fp32 (rounding) | fp32 (rounding)   | **int32 (exact)** |

Binary's int32 accumulation saves 0.001-0.002 mean cos-sim vs fp32
accumulation in the sim. That's why the binary isn't as bad as the
folded-sim. But it is still worse than unfolded-sim because folding
introduces weight rounding error that unfolded avoids.

The gap between sim (0.99623) and binary (0.99326) is now fully
explained as the net of:
  +weight_fold_error (-0.004) +int32_accum_saving (+0.001) = -0.003

## Implication -- how to close the gap

We cannot eliminate the weight-fold: the Conv kernel needs a
per-OC-uniform dequant after int32 accumulation, and per-ci activation
scale must be factored in somehow. The fold is the right architecture.

What CAN be done:

1. **Cross-layer equalization (DFQ/ZeroQ style).** Rescale activations
   before a Conv such that S_a[ci] is more uniform across ci, reducing
   the max(|w_folded|) inflation. Absorbed into the previous BN's gamma.
   Potential +0.001 to +0.003 binary cos-sim.

2. **AdaRound on folded weights.** Learn per-element rounding (+1 or 0)
   that minimizes reconstruction loss under the fold quantization.
   Typical +0.001 to +0.005. Complex to implement.

3. **INT16 for the first few layers.** Higher-precision weight quant on
   stem Conv + first block. Costs 2x memory for those layers (~1% of
   total), might save ~0.002 mean cos-sim. Simple to try.

4. **Mixed-precision calibration.** For the specific layers where S_a
   variance is largest, keep unfolded (and use per-ci i32-multiply hack
   at runtime -- slower). Partial fix.

## Next

- **S108**: cross-layer equalization experiment (DFQ-style rescaling of
  BN gamma to make downstream Conv's S_a more uniform).
- Commit S106-S107 as the diagnostic pair.

## Log file

`sprint_work/s107_fold.log`

## Artefact

`quick_s107_fold_rounding.py`
