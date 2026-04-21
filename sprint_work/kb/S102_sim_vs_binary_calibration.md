# S102 -- Python sim vs C binary calibration

**Date:** 2026-04-21
**Status:** Critical methodology finding. Sets the noise floor for future PTQ sprints.

## Setup

Measure how the per-channel Python fake-quant sim tracks the production
fastface_int8.exe. Both run on the same 100 LFW faces (seed 7777) with
the v1.1.0 calibration recipe (N_CALIB=200, PERCENTILE=99.9, WITH_PRINCESS=1).
Cos-sim is computed vs ORT FP32 in both cases.

## Result

| path | mean | median | min | >=0.99 |
|---|---:|---:|---:|---:|
| Python per-channel sim | 0.99623 | 0.99663 | 0.98525 | 99/100  |
| C binary (v1.1.0)      | 0.99326 | 0.99360 | 0.98338 | 92/100  |

**Per-face delta (binary - sim):**

- mean delta: **-0.00296**
- std: 0.00175
- min: -0.00976   (one face where binary is 1% worse than sim)
- max: +0.00096   (no face is meaningfully better in binary)
- |delta| median: 0.00284,  |delta| p95: 0.00615

## Interpretation

The C binary is systematically ~0.003 below the sim on mean cos-sim, with
per-face noise of ~0.002 std. The sim's "99/100 >=0.99" shrinks to "92/100"
on the binary because 7 faces cross the 0.99 threshold downward in the
0.003 gap.

Likely sources of the gap:

1. **Accumulation order.** Winograd F(2,3) and GEMM i8->i32 accumulate in
   different orders than PyTorch's F.conv2d -> fake_quant. Finite-precision
   INT32 sum is not associative; re-ordering changes low bits.
2. **Per-channel activation requant.** The binary scales i32 -> i8 via per-OC
   scale+round (OPSC2), then feeds into the next i8->i32 Conv. The sim
   fake-quants at the fp32 output via per-OC scale then un-dequants -- same
   algebra but different rounding boundaries.
3. **Add quantization.** The sim allows the Add output to pass as fp32 then
   fake-quant. The binary uses per-channel activation requant on Add output
   into INT8 before the next Conv. Residual paths stack these.
4. **Final Gemm requant.** S94 oracle showed Gemm FP32 barely moves cos-sim
   (+0.00001 in its sim), so this is probably not the bottleneck -- but
   combined with stem-Conv+residual accumulation, the errors stack.

## Implication -- sprint screening rule

A sprint's sim delta must exceed **|delta| p95 = 0.006** -- or at minimum
the |delta| std-of-delta = 0.002 -- to be worth validating on the C binary.

**Retrospective check:**

- S98 sim delta: +0.00015 (well below noise) --> S100 LFW regressed (consistent)
- S99 sim delta: +0.00014 (well below noise) --> S100 LFW regressed (consistent)

This is consistent with the sim-binary gap being the correct explanation.

## Actionable next-level moonshots

Only sprints with plausible >0.005 sim delta:

- **SmoothQuant / per-input-channel activation smoothing** (preceded by BN+PReLU). Potential +0.003 to +0.01.
- **AdaRound / learned weight rounding**. Typical +0.001 to +0.005 in literature.
- **INT16 mixed precision on identified hot layer** (S82 found no hot spot, so unlikely).
- **Shrink the sim-binary gap directly**: diagnose which binary operation
  contributes most to the 0.003 gap. If the final Gemm's activation requant
  is half of it, a targeted fix could pull the binary up to ~0.995 mean
  without any calibration change.

## Log file

`sprint_work/s102_sim_vs_binary.log`

## Artefact

`quick_s102_sim_vs_binary.py`
