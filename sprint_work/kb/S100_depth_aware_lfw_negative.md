# S100 -- Depth-aware calibration LFW 10-fold: NEGATIVE, reverting

**Date:** 2026-04-21
**Status:** Rejected. Python sim win did NOT translate to LFW protocol metric.

## Setup

Regenerated `models/op_scales_v2.bin` and `models/w600k_r50_ffw4.bin`
with the S99 winner config:

```
N_CALIB=200  PERCENTILE=99.9  WITH_PRINCESS=1  DEPTH_EARLY_PCT=99.0  SEED=1
```

Ran `bench_lfw_full.py` (10-fold, 6000 pairs, best-threshold-per-fold protocol).

## Result

| version | calibration | LFW 10-fold accuracy | vs FP32 |
|---|---|---:|---:|
| v1.1.0 (S95) | p99.9 uniform + Princess   | **99.650 +/- 0.229 %** | +0.017 pp (INT8 wins) |
| S100         | p99.9 late + p99.0 early   | 99.633 +/- 0.221 %     | +0.000 pp (tied)      |
| ORT FP32     | -                          | 99.633 +/- 0.221 %     | -                     |

Depth-aware calibration **dropped accuracy by 0.017 pp** (1 pair of 6000).
The mean cos-sim vs ORT improved in sim (+0.00015 on 100 faces), but this
did not translate to improved pair separation on the LFW protocol.

## Why the sim was misleading

- Mean cos-sim vs FP32 is a *fidelity* metric: how close is my embedding
  to the FP32 reference?
- LFW protocol is a *separation* metric: how well do same-identity pairs
  cluster tighter than cross-identity pairs, under a learned per-fold
  threshold?
- Improving fidelity does not guarantee improving separation. Tightening
  early-layer scales at p=99.0 made embeddings slightly closer to FP32
  on average, but the shifts were correlated within identity clusters
  (systematic rather than random), so pair separation neutralised.

## Lesson

- The mean-cos-sim proxy is directionally useful when gains are large
  (e.g. S85 fold +0.006, S91 outlier fix +0.005), but at deltas below
  ~0.001 it is no longer trustworthy as a predictor of LFW outcomes.
- Future sprint candidates with sub-0.0002 sim gains on 100 faces should
  be rejected without spending the LFW validation budget, OR validated
  on a two-fold mini-LFW (600 pairs) before committing to full 10-fold.

## Actions

- Restored `models/op_scales_v2.bin` and `models/w600k_r50_ffw4.bin` from
  `.v1_1_0` backups.
- Regression test confirms bit-exact match with v1.1.0 golden output.
- Leaving `DEPTH_EARLY_PCT` env var wiring in place (no cost) but not
  setting it in the `calibrate` Makefile target.

## Log files

- `sprint_work/s100_export_opsc2.log`
- `sprint_work/s100_prepare_ffw4.log`
- `sprint_work/s100_lfw_10fold.log`

## Next

- **S101+:** switch sprint focus from PTQ hyper-param tuning (diminishing
  returns below noise floor) to bigger levers: AdaRound (learned rounding
  per weight), INT16 for the final Gemm, or QAT fine-tune.
- The S97 flip experiment was also negative -- we have now confirmed the
  PTQ ceiling at ~0.993 mean cos-sim and LFW 99.65% is a real structural
  limit, not a calibration-sweep away.
