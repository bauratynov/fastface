# S106 -- Rounding convention is NOT the sim-binary gap source

**Date:** 2026-04-21
**Status:** Negative. HATZ rounding moves sim mean by -0.00001, not -0.003.

## Test

Replaced `torch.round` (banker's, half-to-even) in the sim with
round-half-away-from-zero (to match the C binary's
`add(vfp, copysign(0.5, vfp)) + truncate` pattern). Applied to both
`fake_quant_per_channel_sym` (activations) and `fake_quant_weight_per_oc`
(weights).

## Result (Python sim, 200 calib WITH_PRINCESS, 100 test faces seed 7777)

| rounding | mean | median | min | >=0.99 |
|---|---:|---:|---:|---:|
| banker (numpy default) | 0.99623 | 0.99663 | 0.98525 | 99/100 |
| HATZ (binary-match)   | 0.99622 | 0.99659 | 0.98526 | 99/100 |

**Delta: -0.00001 mean.** Rounding convention is statistically invisible
at this resolution. Ties are too rare in real activation/weight
distributions to matter.

## Revised hypothesis for S107

The remaining candidate is the **weight-fold rounding difference**:

- **Sim** quantizes `w_orig` per-OC:  scale = max(|w_orig|) / 127
- **Binary** quantizes `w_folded = w_orig * S_a[ci]` per-OC:
  scale = max(|w_orig * S_a[ci]|) / 127

When S_a varies across input channels, these are *different*
quantizations. Recovering w_orig from the binary's int8 weight gives
per-ci-non-uniform rounding noise: large for channels with small S_a,
small for channels with large S_a. The sim's rounding noise is uniform
per-OC.

S107: modify the sim to fake-quant FOLDED weights (`w_orig * S_a[ci]`
per-OC) and divide back by S_a[ci] when computing the Conv output. If
sim mean drops to ~0.993 matching the binary, this IS the gap source.

If S107 also shows no shift, the gap is deeper -- likely in accumulation
order or the int32 quantization's clip-and-round behaviour differing
between torch.F.conv2d (via fp32) and the binary's int8 SIMD path.

## Log file

`sprint_work/s106_rounding.log`

## Artefact

`quick_s106_rounding_match.py`
