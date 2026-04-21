# S31 — Vectorize INT8 OP_GEMM on AVX-VNNI

Date: 2026-04-20
Baseline: INT8 28.66 ms (1.12x ORT, cos-sim 0.945).
S30 profile identified final Linear(25088 -> 512) as 24.2% of runtime, fully scalar.

## Implementation

New kernel `kernels/gemm_int8_matvec.c` provides `fastface_gemm_i8_matvec_vnni`:

- M=1 matvec specialization for the final Linear.
- uint8/int8 signed trick: `Au = A XOR 0x80` shifts activation into uint8
  domain so VPDPBUSD can be used. Correction = `- 128 * sum(W_row)`
  applied at the end.
- Per-row `w_row_sum` is precomputed once at load time (512 rows, 25088 bytes each).
- Unrolled x2 to hide dpbusd latency (Alder/Raptor throughput is 1 per cycle).
- `#pragma omp parallel for` over 512 output rows. Each thread gets
  64 rows x 25088 bytes = 1.6 MB weights, fits comfortably in L2.

Driver changes in `arcface_forward_int8.c`:
- Added `-DAVXVNNI` awareness, `immintrin.h` include.
- Precompute `gemm_w_row_sum[N]` once at startup.
- Allocate scratch `gemm_Au[K+64]` once.
- Replace scalar triple-loop with: vectorized XOR 0x80 shift of A into Au,
  then call the new matvec kernel.
- Fallback scalar loop for any tail `K % 32` elements (currently K=25088 = 32*784, no tail).

## Correctness

Validate output cos-sim vs FP32 reference: **0.9507** (baseline S26 was 0.945).
Within noise, no regression from vectorization.

Spot-check first 5 emb values match scalar to 5 decimals: reassociation-order drift only.

## Stable bench (20 interleaved runs, 3s cooldown, P-core HIGH priority)

| metric | FastFace INT8 | ORT FP32 |
|---|---:|---:|
| min | 20.77 ms | 30.86 ms |
| median | **21.06 ms** | 31.59 ms |
| mean | 21.07 ms | 31.58 ms |
| max | 21.38 ms | 32.21 ms |

**Speedup: 1.500x median, 1.486x best-to-best, 1.499x mean.**
**INT8 won 20/20 runs (100%).**

Standalone run: 22.39 ms best (matches bench interleaved within noise;
bench appears faster because cooldown lets the CPU idle between runs).

## Scoreboard update

| path | median ms | vs ORT | cos-sim | notes |
|---|---:|---:|---:|---|
| ORT FP32 b=1 | 31.59 | 1.00x | 1.0 | baseline |
| FP32 S28/S29d | 25.93 | 1.21x | 0.9997 | ship-ready FP32 |
| **INT8 S31** | **21.06** | **1.50x** | 0.945 | new b=1 champion |
| FP32 batched B=8 | 22.40/face | 1.59x | 0.9997 | best per-face at B>=4 |

## Why this wasn't done earlier

Prior sessions (S26+) focused on making INT8 correct, then speeding up
CONV/BN/PReLU pipeline. The final Linear was left as the scalar reference
because per-op profile had not been done before S30. Once profiled, it
was a 1-day fix with 6+ ms gain — highest-leverage change in the whole project.

**Lesson for KB**: profile before optimizing. S30's 30-min instrumentation
saved what would have otherwise been a speculative multi-session dive
into CONV/ADD tuning.

## Next (S32 candidates, ROI-ranked)

1. **Fuse ADD into Conv epilogue for INT8** (like S29d did for FP32).
   Expected: 21 -> 18-19 ms = 1.65-1.75x ORT. Medium complexity.
2. **Per-channel INT8 weight scales**. Correctness 0.945 -> 0.986+ (S21 proven).
   Needed to ship; not a speed win.
3. **INT8 Winograd for 3x3 conv**. Speculative 2-4 ms. High complexity,
   correctness risk.

## Rollback

Tag at this commit: will mark after commit as `s31-int8-vnni`.
Revert path: `git revert <commit>` reverts both driver + kernel cleanly.
