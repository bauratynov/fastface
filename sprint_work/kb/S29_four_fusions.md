# S29 — Four fusion attempts

Date: 2026-04-20
Baseline: S28 median 26.20 ms, 1.21x ORT, cos-sim 0.9997.

## Summary table

| Attempt | What | Median ms | vs ORT | cos-sim | Verdict |
|---|---|---:|---:|---:|---|
| S29a | PReLU fusion in batched Winograd output (B=8) | 22.40 /face | 1.59x | 0.999694 | WIN |
| S29b | INT8 BN fused into ADD requant | n/a | n/a | n/a | BLOCKED — semantics |
| S29c | FP32 BN fused into Winograd input transform | n/a | n/a | 0.991 | NEGATIVE — FMA drift |
| S29d | FP32 ADD fused into Conv epilogue (b=1) | 25.93 | 1.214x | 0.999702 | WIN (tiny) |

## S29a — PReLU in batched driver (WIN)

Extend S28's Winograd-output PReLU fusion to the batched driver. Look-ahead
checks `m.ops[i+1].type == OP_PRELU`, passes `prelu_slope` to
`fastface_winograd_conv_3x3_s1_p1_batched_bn_prelu`, skips the next op.
AVX2 blendv applies leaky activation after bias in the 4x4 output tile.

Measured B=8: 22.40 ms/face (previously 26.xx). Speedup 1.59x over ORT at B=8
(which does not batch-scale on CPU for w600k_r50).

## S29b — INT8 BN→ADD requant (BLOCKED)

Goal: fuse the BN immediately following an ADD into the requant step,
saving one pass over the int8 activation tensor.

Kernel was implemented (`add_bn_requant_int8` in `kernels/int8_epilogue.c`):
AVX2 vectorized, loads two int8 inputs, dequants both, sums, applies per-channel
BN scale/offset, requants with output scale.

Blocker: in the IResNet-100 graph the ADD is always followed by
`OP_BLOCK_START` (residual snapshot), not directly by `OP_BN`. Fusing
BN-after-ADD would poison the `block_buf` snapshot with BN-applied values,
breaking the pre-BN shortcut semantics used inside each residual block.

Driver look-ahead reverted. Kernel retained for future use in a graph
where BN directly follows ADD (unlikely without graph surgery).

## S29c — FP32 BN into Winograd input (NEGATIVE)

Goal: apply per-channel BN to the Winograd input transform, removing a full
pass over the feature map before each Conv.

Math is semantically equivalent (BN is affine, interior transform is linear,
padding is 0 → BN offset never multiplies a padded cell). Implemented and
verified compile.

Failure mode: accumulated FMA rounding over ~24 fused BN+Conv pairs drops
end-to-end cos-sim from 0.9997 to 0.991. That's below our 0.995 correctness
floor. Driver changes reverted; kernel (`_full_fused`) retained as scaffolding
for S29d and future work.

Lesson: fusion that is algebraically correct can still fail on accumulated
precision. Need per-layer cos-sim diff to pinpoint the culprit layer before
trying again — likely an early-stem layer where channel magnitudes are high.

## S29d — FP32 ADD into Conv epilogue (WIN)

Goal: merge the residual ADD into the Winograd output transform of the
second conv in each block, so one store instead of two.

Implementation: `fastface_winograd_conv_3x3_s1_p1_full_fused_add` takes an
optional `add_src` pointer. After bias+PReLU, the 8-wide output vector is
added to the corresponding `add_src` cell, then stored. Driver look-ahead
picks up `OP_ADD` after `OP_CONV_3x3_s1_p1` and passes the shortcut slot.

Correctness: cos-sim 0.999702 (matches S28 baseline to 4 decimals).

Stable bench (20 interleaved runs, 3 s cooldown, P-core affinity, HIGH priority):

- FastFace: min 25.78, median 25.93, mean 25.98, max 26.30 ms
- ORT:      min 30.77, median 31.49, mean 31.62, max 32.92 ms
- Speedup: best 1.193x, median 1.214x, mean 1.217x
- FastFace won 20/20 runs (100%)

Gain over S28 baseline: 26.20 → 25.93 ms, ~1%. Small but measurable and
consistent (20/20 wins at tight variance).

## Takeaways for future fusion work

1. Batched path has more headroom than b=1 (S29a delivered 1.59x for free).
2. Graph-level fusions must respect residual snapshot invariants — check
   BLOCK_START boundaries before merging across ADD.
3. Linear-algebra equivalence is necessary but not sufficient — verify
   cos-sim end-to-end after every fusion, not just layer-local sanity.
4. Cheap epilogue merges (bias, activation, add) are the lowest-risk wins.
   BN fusion into input/body needs precision analysis first.
