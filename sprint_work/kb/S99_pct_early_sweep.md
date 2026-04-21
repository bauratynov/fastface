# S99 -- PCT_EARLY sweep with cross-seed validation

**Date:** 2026-04-21
**Status:** PCT_EARLY=99.0 wins, stable across seeds. Ready for LFW 10-fold validation in S100.

## Setup

Fixed: N_CALIB=200 (WITH_PRINCESS), PCT_LATE=99.9, SEED_CALIB=1.
Swept: PCT_EARLY in {98.0, 99.0, 99.5, 99.7, 99.9}.
Cross-seed test: two independent 100-face batches, seeds 7777 and 11111.

## Results (Python per-channel sim, 100 test faces each seed)

```
PCT_EARLY  | seed 7777: mean  min  >=99 | seed 11111: mean  min  >=99 | mean-of-means
-----------+----------------------------+------------------------------+---------------
    98.0   |    0.99638 0.98648  99/100 |    0.99666 0.98954  99/100  |   0.99652
    99.0   |    0.99638 0.98864  99/100 |    0.99666 0.99059 100/100  |   0.99652
    99.5   |    0.99629 0.98753  99/100 |    0.99659 0.99115 100/100  |   0.99644
    99.7   |    0.99629 0.98465  99/100 |    0.99650 0.99034 100/100  |   0.99639
    99.9   |    0.99623 0.98525  99/100 |    0.99654 0.99033 100/100  |   0.99638   <- baseline
```

**Winner: PCT_EARLY=99.0**

- Best mean-of-means: 0.99652 (tied with 98.0, but 99.0 has better min on seed 11111)
- +0.00014 mean vs uniform p=99.9 baseline, stable across both seeds
- Min improvement on Princess-class outliers: +0.00339 (seed 7777) / +0.00026 (seed 11111)
- Flat tail (99.5/99.7/99.9 converge) suggests the win comes from tightening
  **early** layers, not from any second-order effect.

## Interpretation

The sim sees a small but real improvement when early layers get p=99.0 instead
of p=99.9. Seed 7777 improvement is concentrated on the Princess outlier (min
0.985 -> 0.989). Seed 11111 doesn't have a comparable outlier, so the win is
distributed (all 100 samples improve by ~0.0001 average).

The 98.0 vs 99.0 tie suggests the early-half activations already have very
clean tails (dense distributions from normalised [-1,1] pixel input and the
low-level Conv features). Clipping further at p=98.0 doesn't hurt but doesn't
help vs p=99.0.

## Next step

**S100** -- regenerate `models/op_scales_v2.bin` and `models/w600k_r50_ffw4.bin`
with `DEPTH_EARLY_PCT=99.0 PERCENTILE=99.9 N_CALIB=200 WITH_PRINCESS=1`, then run
`bench_lfw_full.py` 10-fold. If INT8 accuracy moves above 99.650% (v1.1.0),
adopt as v1.2.0. If within noise of 99.650%, keep v1.1.0 (no reason to change
production calibration for a sim-only improvement).

## Log file

`sprint_work/s99_pct_early_sweep.log`

## Artefact

`quick_s99_pct_early_sweep.py` (untracked)
