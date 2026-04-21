# S103 -- SmoothQuant alpha sweep: NEGATIVE

**Date:** 2026-04-21
**Status:** Rejected. SmoothQuant migrates per-input-channel activation range
into weights -- but IResNet does not have the transformer-style per-channel
activation outliers this trick is designed for.

## Setup

Applied SmoothQuant only to Convs/Gemm whose input is from BN or PReLU
(not Add, since Add output has no per-channel affine param we can fold into).
49 of ~50 smoothable Convs got factors.

For each smoothable op, s_c = max(|X_c|)^alpha / max(|W_:,c|)^(1-alpha)
with clamp to [1e-3, 1e3]. Activation is divided by s, weight multiplied
by s, per-input-channel.

Sim used the v1.1.0 activation scales rescaled by 1/s for smoothed inputs.

## Result (Python sim, 100 faces seed 7777)

| alpha | mean | median | min | >=0.99 |
|------:|-----:|-------:|----:|-------:|
| 0.00 (no-op baseline) | 0.99623 | 0.99663 | 0.98525 | 99/100 |
| 0.25  | 0.07878 | 0.07331 | -0.05536 | 0/100 |
| 0.50  | 0.97722 | 0.98002 | 0.90931 | 2/100 |
| 0.75  | 0.99375 | 0.99448 | 0.97342 | 93/100 |

alpha=0.25 catastrophe: the weight^0.75 term drives s_c to the clamp edge
for input channels with tiny max weight, amplifying those activations 1000x
and clobbering the quantization range.

alpha=0.5 (canonical paper setting): -0.019 from baseline.
alpha=0.75: -0.00248 (below noise floor but wrong direction).

## Why SmoothQuant does not help here

1. SmoothQuant was designed for LLMs, where self-attention produces
   per-channel activation outliers 10-100x the typical magnitude (specifically
   Layer-Norm gamma blow-up). IResNet has no such pattern.
2. IResNet uses PReLU (learned per-channel negative slope) after every BN.
   PReLU already provides a per-channel rescaling, so the activation
   distributions entering each Conv are already reasonably balanced.
3. Our per-channel activation INT8 already captures any remaining per-channel
   variance. Adding an extra affine migration into the weights just perturbs
   the weight distribution (which was clean) at the cost of barely improving
   activation quant.

## Retrospective on remaining PTQ levers

- [x] Per-channel activation percentile sweep (S98-S100) -- negative on LFW
- [x] Weight percentile clipping (S101) -- negative
- [x] SmoothQuant (S103) -- negative
- [x] Ensemble calibration (S83) -- negative
- [x] Flip augmentation in calib (S97) -- negative
- [x] Smart outlier inclusion (S91 Princess) -- POSITIVE, already shipped
- [x] Trailing BN-after-Gemm fold (S85) -- POSITIVE, already shipped

The PTQ hyper-parameter space for this network is exhausted. Remaining
levers are:
- AdaRound -- high effort, typical +0.1-0.5 pp LFW in literature
- QAT fine-tune -- very high effort, needs backprop infrastructure
- **Shrink the sim-binary gap** -- S102 found the 0.003 gap; if half of it
  comes from one binary op (e.g. Add requant), fixing that translates DIRECTLY
  to binary cos-sim with zero calibration change.

Pivot for S104+: diagnose which binary operation contributes most to the
0.003 sim-binary gap. This is higher expected value than AdaRound for the
same or less effort.

## Log file

`sprint_work/s103_smoothquant.log`

## Artefact

`quick_s103_smoothquant_sim.py`
