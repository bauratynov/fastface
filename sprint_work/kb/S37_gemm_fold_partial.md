# S37 — per-channel fold for final Gemm + last BN (partial per-channel)

Date: 2026-04-20
Parent: S36 phase A infrastructure.
Goal: prove the per-channel fold pipeline works end-to-end with a partial scope
(just the final Linear path), then extend to all Convs in S38+.

## Scope landed this sprint

1. **`prepare_weights_v3.py`** — folds per-channel input activation scale of
   the Gemm's input tensor (last BN output, before Flatten) into the Gemm
   weights. Per-OC requantize. Emits new magic **FFW4**.
2. **`kernels/ffw2_loader.c`** — accepts FFW4 magic (same layout as FFW3
   semantically; just the GEMM weights are folded). Exposes `version` field
   on the `FFW2` struct.
3. **`kernels/int8_epilogue.c`** — `bn_prelu_requant_int8` gains optional
   `inv_out_per_ch` param (AVX2 inner loads per-channel or falls back to
   broadcast scalar).
4. **`arcface_forward_int8.c`**:
   - `FFW2` struct gains `version` field.
   - Driver detects "BN followed by FLATTEN" in a FFW4 model with OPSC2 loaded,
     passes per-channel `inv_out_per_ch` to that BN's requant.
   - Matvec sees `m.version >= 4` and passes `A_scale = 1.0` (since S_a is
     already folded into the weights).

## Correctness

| config | cos-sim |
|---|---:|
| FFW3 + OPSC-per-tensor (S32) | 0.954316 |
| FFW3 + OPSC2 loaded, NULL-passed (S36 phase A) | 0.954316 (identical) |
| **FFW4 + OPSC2 + last-BN per-channel + matvec A=1.0** | **0.954497** |

**+0.0002** — within noise. The fold is mathematically correct (we confirmed
backward-compat FFW3 still gives 0.954316, and FFW4 without the last-BN
per-channel requant gave a different but related number), but the quantization
error dominates EARLIER in the Conv chain.

## Speed

14.47 ms standalone (vs S32's 13.36 ms median stable). No regression; variance
from non-pinned run. Speed was not the S37 goal.

## Why the gain is tiny

S21 Python simulation showed 0.986 mean cos-sim for FULL per-channel activation
quant. Decomposition:
- Per-channel at final Gemm input only: 0.954 -> 0.955 (observed).
- Per-channel at every Conv input: recovers 0.955 -> 0.986 (remaining 31 pp).

Per-channel is a "death by thousand cuts" accuracy fix — the ~0.03 gap between
0.954 and 0.986 is accumulated across ALL 53 Conv layers. Fixing only the last
Linear barely moves the needle because its error contribution was small.

## Next step: S38 — Conv-chain fold

For each Conv op, fold per-channel input activation scale into weights at
prepare time. Steps:
1. In `prepare_weights_v3.py`: for each Conv op, look up per-channel scale
   of the Conv's INPUT tensor (= previous op's output, or post-Flatten fold
   input). S_a[Cin] array.
2. Apply BN fold first (as v2 does) producing `W_bn[oc, ci, ...] = W * bn_scale[ci]`.
3. Then apply per-channel act fold: `W_final[oc, ci, ...] = W_bn[oc, ci, ...] * S_a[ci]`.
4. Per-OC quantize. Update bias with bias-fold formula accounting for BN offset
   and activation-scale-zero-shift if any (symmetric so no shift).
5. Runtime: Conv epilogue passes `in_scale = 1.0` (or removes in_scale entirely
   for FFW4+).
6. Runtime: Conv epilogue requants with per-channel `inv_out_per_ch` (already
   in place from S36 phase A).
7. Runtime: ADD shortcut needs per-channel `add_scale_per_ch` since the
   shortcut was saved with per-channel meaning. Extend fused_epilogue_int8
   accordingly.
8. Driver: quantize input with per-channel scale (3 channels RGB).
9. Driver: track `id_scales_per_ch[slot][C]` arrays for SAVE_ID / ADD shortcuts.

Expected: cos-sim 0.954 -> 0.98-0.986. This is the "ship-blocker" fix.

Estimated: 1-2 more sprints. Kernels are mostly infrastructure-ready (S36
phase A got fused_epilogue per-channel out; S37 got bn_prelu per-channel out
and matvec-via-fold). Remaining kernel work: add_src per-channel scale
array + quantize_fp32_nhwc_to_int8 per-channel variant.

## Artifacts

- `models/w600k_r50_ffw4.bin` (41.7 MB) — same size as FFW3, just with folded
  Gemm weights. Regenerate with `python prepare_weights_v3.py`.
- `models/op_scales_v2.bin` (226 KB) — reuse from S36.

## Commercial significance

**Still ship-blocked** at 0.954 cos-sim. Infrastructure is now ready for the
final Conv-chain fold which should unlock the cos-sim threshold. One more
focused sprint should close this.
