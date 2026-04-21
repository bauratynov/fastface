# S108 -- S_a variance profile (DFQ leverage assessment)

**Date:** 2026-04-21
**Status:** Informative. Variance is huge but dominated by near-zero channels,
limiting DFQ leverage.

## Setup

For each Conv/Gemm input tensor, compute per-channel activation scale S_a[ci]
(p99.9 percentile / 127 on 200 calib faces WITH_PRINCESS). Measure
max/min, coefficient of variation, max/median across ci.

## Result

- Total Conv/Gemm layers profiled: 54
- Layers with max(S_a)/min(S_a) > 10:  43/54
- Layers with max(S_a)/min(S_a) > 100: 39/54
- Typical CV (std/mean) of S_a: 0.3-0.6

Top-5 most-skewed Conv inputs:

| rank | op   | #ci  | max/min | cv    | max/med  |
|-----:|:-----|-----:|--------:|------:|---------:|
|  1   | Conv | 64   | inf     | 1.173 | 984205   |
|  2   | Conv | 64   | 5.6e35  | 1.863 | 887841   |
|  3   | Gemm | 25088| 1.3e27  | 0.441 | 3.25     |
|  4   | Conv | 256  | 2.4e26  | 0.511 | 4.65     |
|  5   | Conv | 256  | 2.1e25  | 0.578 | 6.04     |

## Interpretation

The astronomical max/min ratios are misleading -- they come from calibration's
`np.where(pc_absmax > 0, pc_absmax, 1e-6)` floor for dead channels (channels
with ~zero activation magnitude after p99.9). Ratios in the 1e25-1e35 range
mean "real channels vs. near-zero floor," not "real channels vs. real channels".

The meaningful metric is **max/median**:

- Ranks 1-2: max/med ≈ 1M -- bimodal. A few live channels vastly larger
  than the rest.
- Rank 3 (final Gemm): max/med = 3.25 -- actually moderate.
- Most layers: max/med 2-10 -- moderate per-channel variation.

And CV < 1 for most layers means std <= mean: distribution is not
pathologically multi-modal. The variance is real but bounded.

## DFQ leverage assessment

Cross-layer equalization's purpose is to balance per-ci magnitudes so the
per-OC max(|w_folded|) isn't dominated by a single ci. Leverage depends on
max/median, not max/min (the near-zero tail is already absorbed by the
p99.9 floor and doesn't inflate the max).

For 40/54 layers with max/med < 10, DFQ has low leverage (<0.001 cos-sim
gain expected). For 14/54 layers with max/med > 10, DFQ might yield
+0.001 to +0.003, but also:

- DFQ's rescale is a LINEAR transformation. If it's mathematically
  preserving, it doesn't change max(|w_folded|) -- only rearranges the
  per-ci distribution. Per my working-out in S108 notes: the fold IS
  invariant to this rescaling (since w_folded = (W/r) * (S_a*r) = W*S_a).
- DFQ helps UNFOLDED weight quant where it balances the per-OC-max.
  With fold, the benefit disappears.

**Conclusion**: DFQ is unlikely to reduce the sim-binary gap because
the gap source is the fold itself, not the pre-fold weight imbalance.

## Pivoting S109

Instead of DFQ, try:
1. **INT16 for the first 2-3 Convs** (ranks 1-2 highest CV). Marginal cost
   (~1% memory), tests whether stem precision is the dominant bottleneck.
2. **Dead-channel pruning in fold**: for ci with S_a effectively zero,
   force w_folded[co,ci,...] = 0 to reduce max(|w_folded|) inflation.
   But already observed: those channels are in the 1e-6 floor and DON'T
   inflate the max (they're tiny).
3. **Per-[O,I]-channel weight quant** for the worst 2-3 layers. Runtime
   incompatibility unless kernel supports it.

## Log file

`sprint_work/s108_sa_variance.log`

## Artefact

`quick_s108_sa_variance_profile.py`
