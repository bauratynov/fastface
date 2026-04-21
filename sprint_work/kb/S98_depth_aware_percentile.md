# S98 — Depth-aware percentile calibration

**Date:** 2026-04-21
**Status:** Small positive result; needs C-binary validation before adoption.

## Hypothesis

Early Conv layers in IResNet-100 see smooth, dense activation distributions
(input is normalized [-1,1] RGB, early features are low-level edges/textures).
Late layers see sparse, outlier-heavy activations (high-level features with
PReLU slope + residual stacking create long tails).

A uniform percentile across all depths is a compromise. Try:

- **Early half:** tight percentile (p=99.0) — clip outliers, tighter scale
- **Late half:** loose percentile (p=99.9) — preserve tail, avoid clipping
  meaningful signal

## Plumbing change

`calibrate_percentile_int8.py` `build_percentile_scales` now accepts
`depth_early_percentile` (applied to the first half of the tensor-creation
order; rest uses `percentile`). Threaded through:

- `export_op_scales_v2.py` — reads env `DEPTH_EARLY_PCT`
- `prepare_weights_v3.py`  — reads env `DEPTH_EARLY_PCT`

## Result (Python per-channel sim, 200 calib WITH_PRINCESS, 100 test faces seed=7777)

| config | mean | median | min | >=0.99 | >=0.98 |
|---|---:|---:|---:|---:|---:|
| uniform p=99.9 | 0.99623 | 0.99663 | 0.98525 | 99/100 | 100/100 |
| **depth p_early=99.0 p_late=99.9** | **0.99638** | **0.99680** | **0.98864** | 99/100 | 100/100 |

**Delta:** +0.00015 mean, +0.00017 median, **+0.00339 min**.

The minimum rose from 0.985 -> 0.989 -- the Princess_Elisabeth outlier that
dominated S91's fix now improves further. >=0.99 and >=0.98 counts unchanged.

## Caveats

1. Mean gain is tiny (0.015%), below typical run-to-run noise on a different
   test seed. Need a cross-seed verification in S99 before accepting.
2. Python sim != C binary. The per-channel fake_quant in sim closely matches
   the FFW4 OPSC2 production path, but a 1-to-1 LFW 10-fold check with the
   exe is required before updating ship numbers.
3. "First half" split is by graph traversal order, not by feature depth in
   any principled sense. A better split (e.g., before/after stem, or by
   residual block index) might unlock more.

## Next steps

- **S99:** sweep PCT_EARLY in {98.0, 99.0, 99.5, 99.7, 99.9} to find the
  sweet spot; cross-seed test with two test seeds.
- **S100:** regenerate FFW4 + OPSC2 with best depth config, run
  `bench_lfw_full.py` 10-fold and compare to v1.1.0 numbers
  (INT8 99.650% vs FP32 99.633%).

## Log file

`sprint_work/s98_depth_pct.log`

## Artefact

`quick_s98_depth_pct.py` (untracked — promote to tests/ if adopted)
