# S32 — Fuse ADD into INT8 Conv epilogue

Date: 2026-04-20
Baseline: S31 INT8 21.06 ms median, 1.50x ORT, cos-sim 0.9507.
S30 profile: OP_ADD = 27% of runtime (8.1 ms/inf, 24 ops).

## Implementation

Extended `fused_epilogue_int8` signature with two optional params:
`const int8_t* add_src, float add_scale`. When `add_src` is non-NULL, the
kernel adds `add_src[p, co] * add_scale` to the fp accumulator between
bias and BN/PReLU — in the same pass as dequant and requant, so no extra
memory sweep.

AVX2 inner loop: loads 8 int8 from `add_row`, sign-extends to int32,
converts to fp32, multiplies by broadcast `add_scale`, adds to `vfp`.

Driver look-ahead in `arcface_forward_int8.c` `OP_CONV` case: detects
`Conv followed by ADD` pattern (mutually exclusive with `Conv followed
by PReLU` since IResNet Conv2 has no PReLU before ADD), grabs shortcut
pointer + scale from `id_slots[add_idx]`, passes to epilogue, skips the
ADD op, increments add_idx.

## Correctness

cos-sim vs FP32 reference: **0.9543** (up from S31's 0.9507).

The improvement comes from eliminating 24 intermediate requants between
Conv and ADD. Each requant rounds to int8, losing precision; keeping the
fp32 sum through the full epilogue preserves more signal before the
final requant.

## Stable bench (20 interleaved runs, 3s cooldown, P-core HIGH)

| metric | INT8 S32 | ORT FP32 |
|---|---:|---:|
| min | 13.17 ms | 30.98 ms |
| **median** | **13.36 ms** | 31.46 ms |
| mean | 13.35 ms | 31.46 ms |
| max | 13.45 ms | 31.90 ms |

**Speedup: 2.355x median, 2.352x best-to-best, 2.356x mean.**
**INT8 won 20/20 runs. Spread 0.28 ms (tightest measurement in project history).**

## Gain decomposition

| session | median ms | delta ms | vs ORT |
|---|---:|---:|---:|
| S26 baseline | 28.85 | — | 1.11x |
| S30 profile (no change) | 28.85 | 0 | 1.11x |
| S31 VNNI matvec | 21.06 | -7.79 | 1.50x |
| **S32 ADD fusion** | **13.36** | **-7.70** | **2.355x** |

Two sprints delivered 15.5 ms of saving (from 28.85 -> 13.36), turning
the INT8 path from modestly-above-ORT into the project's flagship result.

## Scoreboard

| path | median | vs ORT | cos-sim | notes |
|---|---:|---:|---:|---|
| **INT8 b=1 (S32)** | **13.36 ms** | **2.355x** | 0.954 | flagship b=1 |
| FP32 b=1 (S29d) | 25.93 ms | 1.21x | 0.9997 | ship-quality b=1 |
| FP32 B=8 | 22.40 /face | 1.59x | 0.9997 | ship-quality batch |
| ORT FP32 | 31.46 ms | 1.00x | 1.0 | baseline |

## What's next

The INT8 path has now eaten its two biggest time sinks. Remaining budget
(S30 profile, scaled pro-rata): ~10.4 ms for Conv, ~1 ms for BN/rest.

1. **Per-channel INT8 weight scales** — correctness fix. cos-sim 0.954 -> 0.986+.
   Not a speed win; needed to ship as biometric-grade.
2. **INT8 Winograd for 3x3 conv** — potential 2-4 ms saving. High risk.
3. **Batched INT8** — port S24 batching to int8 path. Could match or beat
   the 22 ms/face of FP32 batched.
4. **Direct-conv for 1x1** — skip im2col entirely for the many 1x1 convs
   (lots of them in IResNet).

## Rollback

Will tag `s32-int8-add-fusion` after commit. Extension to epilogue signature
is backward-compatible in intent — callers that don't pass add_src get
identical old behavior.
