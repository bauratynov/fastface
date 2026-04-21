# S101 -- Weight percentile vs absmax: NEGATIVE

**Date:** 2026-04-21
**Status:** Rejected -- weight absmax (current) is correct. Rules out a plausible PTQ lever.

## Hypothesis

Per-output-channel weight absmax is wasteful if each OC has one or two
extreme-magnitude weights. Clipping those at p=99.99 / p=99.9 could tighten
the INT8 scale and give more precision to the body of the weight distribution.

## Result (Python sim, N_CALIB=200 WITH_PRINCESS, 100 faces x 2 seeds)

| weight_pct | seed 7777 mean | seed 11111 mean | mom    | delta vs absmax |
|-----------:|---------------:|----------------:|-------:|----------------:|
| 100.0 (absmax, baseline) | 0.99623 | 0.99654 | 0.99638 | 0.00000 |
| 99.99 | 0.99602 | 0.99623 | 0.99613 | **-0.00026** |
| 99.9  | 0.95982 | 0.95917 | 0.95949 | **-0.03689** |

Clipping at p=99.99 already loses 0.00026 mean; clipping at p=99.9 is
catastrophic (-0.037). The top 0.1% of weights in each OC are NOT outliers
to throw away -- they are the signal that dominates the layer's output.

## Why absmax is the correct default for weights

1. Conv weights in trained networks are near-Gaussian, not heavy-tailed.
   Typical kurtosis 3-4; nothing to clip.
2. The gradient signal during training pushes high-magnitude weights
   precisely because they carry the most information. Quantising with a
   wider range so they survive is worth the precision cost on small
   weights.
3. Per-OC granularity (already enabled) handles the case where different
   channels have very different magnitude ranges. That is where all the
   weight-calibration win comes from; percentile adds nothing.

## Implication

- Do not revisit weight-side percentile calibration in later sprints.
- Weight-side optimisation, if any, will come from learned rounding
  (AdaRound) or structured pruning, not clipping.

## Log file

`sprint_work/s101_weight_pct.log`

## Artefact

`quick_s101_weight_pct.py`
