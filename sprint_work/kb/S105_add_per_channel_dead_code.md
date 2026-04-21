# S105 -- Per-channel Add requant: dead code, gap source misidentified

**Date:** 2026-04-21
**Status:** Neutral. Added `add_requant_int8_pc` kernel (now unused); pure OP_ADD path never executes for FFW4 due to Conv+ADD lookahead fusion. Gap source identified for S106.

## What I tried

Hypothesised that `add_requant_int8` (scalar dequant/requant for residual
ADD) was a key gap source since the binary tracks per-channel scales for
both inputs (A_scale_pc, id_scales_pc) but passes only scalar scales to
this kernel. Added `add_requant_int8_pc` and modified the pure OP_ADD
case to use per-channel when FFW4+OPSC2 available.

## Debug finding

Bit-exact regression and 100-face cos-sim UNCHANGED after the patch.
OP_TRACE instrumentation of the runtime op loop showed:

```
type CONV:        53 instances
type BN:          25 instances
type GEMM:         1 instance
type FLATTEN:      1 instance
type SAVE_ID:     24 instances
type BLOCK_START: 24 instances
type ADD:          0  (!)
type PRELU:        0  (!)
```

The FFW4 file has 177 ops total including 24 OP_ADD and 25 OP_PRELU, but
**Conv+PReLU and Conv+ADD lookahead fusion** at line 371/377 swallows both
PRELU and ADD into the preceding Conv's `fused_epilogue_int8` call. The
pure OP_ADD and OP_PRELU cases are dead code for FFW4 v4 models.

FFW4 op sequence (first 11 ops):
```
op[0] = CONV   (stem)
op[1] = PRELU  -> fused into op[0] via lookahead
op[2] = BLOCK_START
op[3] = BN     (pre-block BN; not fused)
op[4] = CONV
op[5] = PRELU  -> fused into op[4]
op[6] = CONV
op[7] = SAVE_ID
op[8] = CONV
op[9] = ADD    -> fused into op[8] via Conv+ADD lookahead
op[10] = SAVE_ID
```

## Where the gap actually lives

Inspection of `fused_epilogue_int8` confirms it DOES use per-channel
scales when OPSC2 is present:

- `inv_out_per_ch` (per-channel output scale) -- applied on requant
- `add_scale_per_ch` (per-channel shortcut scale) -- applied to residual add

The ONLY scalar path left is `in_scale` (Conv input dequant), but line 365
forces `conv_in_scale = 1.0f` for FFW4 because per-input-channel activation
scale is **pre-folded into Conv weights** at prepare_weights_v3.py step
(S38 fold).

So per-channel scale is correctly carried through. The 0.003 sim-binary
gap must therefore come from a different source. The next hypothesis is
**rounding convention**:

- Python sim uses `torch.round` = banker's rounding (half-to-even).
- Binary uses `add(vfp, copysign(0.5, vfp))` + truncate = round-half-away-from-zero.

These differ on exact ties (0.5 -> 0 vs 1), and for typical fp32 values
the noise stacks across ~130 layers. S106 will test by replicating the
binary rounding convention in the Python sim and checking whether sim
mean drops from 0.99623 toward 0.99326.

## Actions

- Reverted OP_ADD case to original scalar-only implementation.
- Left `add_requant_int8_pc` kernel in place (unused, ~30 lines) for any
  future non-FFW4 path that might benefit.
- Removed all debug `fprintf` and trace instrumentation.
- Regression bit-exact pass confirmed.

## Lesson

Before writing 100 lines of C to fix a suspected gap source, add a quick
runtime instrumentation print to confirm the suspect code path even
executes. Would have caught "dead code" in 30 seconds.

## Log files

- `sprint_work/s105_sim_vs_binary.log` (pre-revert, unchanged binary output)
- `sprint_work/s105_build.log`
