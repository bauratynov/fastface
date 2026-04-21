# S104 -- v1.1.0 LFW 10-fold reproduction confirmed

**Date:** 2026-04-21
**Status:** Passes. Restored artefacts = original v1.1.0.

## Setup

After S100 reverted the depth-aware calibration via .v1_1_0 backup files,
run `bench_lfw_full.py --seed 42 --lfw-dir data/lfw` to confirm we are
at the genuine v1.1.0 numbers (not a silent regression).

## Result

| path | LFW 10-fold accuracy |
|---|---:|
| ORT FP32 | 99.633 +/- 0.221 % |
| FastFace INT8 | **99.650 +/- 0.229 %** |
| gap | **-0.017 pp (INT8 wins)** |

Identical to the v1.1.0 ship number. Artefact restoration was clean.

## Per-fold comparison vs S100 (depth-aware)

| fold | v1.1.0 INT8 | S100 INT8 | delta |
|---:|---:|---:|---:|
|  1 | 99.83 | 99.67 | **+0.17** |
|  2 | 99.50 | 99.50 | 0     |
|  3 | 99.17 | 99.17 | 0     |
|  4 | 99.50 | 99.50 | 0     |
|  5 | 99.67 | 99.67 | 0     |
|  6 | 99.83 | 99.83 | 0     |
|  7 | 99.50 | 99.50 | 0     |
|  8 | 100.00 | 100.00 | 0    |
|  9 | 99.67 | 99.67 | 0     |
| 10 | 99.83 | 99.83 | 0     |

The S100 regression was concentrated on fold 1 (lost 1 pair out of 600).
Nine folds were identical -- supports the reading that the total delta of
-0.017 pp came from a single pair crossing the decision boundary, not a
systematic accuracy loss.

This confirms the S102 screening rule: sub-0.005 sim-delta experiments
can easily swing LFW by +/-0.02 pp, which is within fold-level pair
boundary noise.

## Log file

`sprint_work/s104_lfw_v1_1_0_repro.log`

## Next

Pivot to binary-gap analysis or AdaRound. 27 sprints remaining before
post-S131 polish mandate.
