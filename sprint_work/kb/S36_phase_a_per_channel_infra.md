# S36 phase A — per-channel INT8 infrastructure (kernel + loader)

Date: 2026-04-20
Parent: commercialization roadmap from BZ ident'd S36 as ship-blocker for all 3 sales angles.
Target: cos-sim 0.954 -> 0.986+ via per-channel activation quant (S21 research proven).

## What landed this sprint

1. `export_op_scales_v2.py` — reuses S21 per-channel calibration logic, walks
   the ONNX graph in FFW3 op order, and emits **OPSC2** format:
   ```
   'OPSC2' u32 n_ops  u32 n_input_ch  f32[n_input_ch]  (per-op: u32 nch, f32[nch])
   ```
   Marker ops (BLOCK_START, SAVE_ID, FLATTEN) emit nch=0.
   Run: `N_CALIB=20 python export_op_scales_v2.py`
   Output: `models/op_scales_v2.bin` (226 KB, 178 ops, 130 with per-channel data).

2. `fused_epilogue_int8` kernel extended with optional `const float* inv_out_per_ch`
   parameter. When NULL, falls back to scalar `out_scale` (current behavior).
   When non-NULL, loads per-channel inv_out via `_mm256_loadu_ps` in the AVX2
   inner loop.

3. `arcface_forward_int8.c` driver: `load_op_scales_v2()` tries to load
   `models/op_scales_v2.bin` after the per-tensor `op_scales.bin`. If found,
   precomputes and stores `inv_out_per_ch[n_ops]` arrays (pre-inverted for hot path).
   Env var `OPSC2_PATH` overrides.

## Backward-compat validation

With OPSC2 loaded but Conv path still passing `inv_out_per_ch=NULL`:
- cos-sim: **0.954316** (exactly matches S32 baseline — no change)
- speed: 14 ms standalone (~13.36 ms stable median)

Confirms the infrastructure change is semantics-preserving.

## Why Phase B (activation) didn't land in this sprint

Attempted to wire the per-channel `inv_out_per_ch` into the Conv epilogue.
Result: cos-sim collapsed to **-0.056** (model output junk).

Root cause: **INCOHERENT pipeline**. Changing only Conv's OUTPUT scale to
per-channel produces int8 activations with per-channel meaning, but consumers
still use SCALAR scales:

1. `fused_epilogue_int8` Conv input (`in_scale`) is SCALAR — the next Conv reads
   this int8 and dequants with a single scalar, losing the per-channel information
   that was just encoded.
2. ADD shortcut (`add_scale`) inside fused epilogue is SCALAR — shortcut tensors
   saved with per-channel scale get rescaled incorrectly.
3. `quantize_fp32_nhwc_to_int8` for the initial input is SCALAR.
4. Final `fastface_gemm_i8_matvec_vnni` takes scalar `A_scale`.

All consumers must upgrade together.

Also hit secondary bug: `float tmp[2048]` in OPSC2 loader was too small for
`nch=25088` (flattened-before-Gemm tensor); bumped to `float tmp[32768]`.

## Phase B plan (full per-channel activation)

Coherent pipeline upgrade required. Estimated 2-3 more sprints:

1. **Kernels** — add per-channel in_scale variants for:
   - `fused_epilogue_int8` — optional `const float* in_scale_per_ch`. Need this
     for the Conv's ACTIVATION input, not weight_scales.
   - `quantize_fp32_nhwc_to_int8` — per-channel output scale (for initial input).
   - `add_requant_int8` — per-channel a_scale, b_scale, out_scale (if standalone
     ADD ever fires, though S32 fusion eliminates all in current flow).
   - `bn_prelu_requant_int8` — per-channel in_scale, out_scale.

2. **Matvec** — the final `fastface_gemm_i8_matvec_vnni` needs per-channel
   input scale. Approach: pre-multiply `W[oc, k] * A_scale[k_channel]` at load
   time where k_channel = k / (H_last * W_last). Works because per-channel
   acts are fixed at load.

3. **Driver** — replace scalar scale tracking (`A_scale`, `block_scale`) with
   per-channel array tracking. For each op, the effective output has a
   per-channel scale array.

4. **Weight pre-fold for Conv** — for Conv ops where input has per-channel
   scale, we can either:
   - (a) Pass per-channel `in_scale_per_ch` and have the kernel fold inline
     (extra memory load per inner iteration).
   - (b) Pre-fold into weights at load time: `W_eff[oc, ci, ...] = W[oc, ci, ...] * S_a[ci]`
     then re-quantize per-OC. Saves runtime cost but changes model data path.

   (a) is simpler for phase B; (b) is the optimization if phase B shows
   correctness but too-slow. Current speed impact unclear.

## Expected outcome after phase B

Per S21 simulation: cos-sim 0.954 -> 0.986 mean on LFW 10-test set. Good news:
S21 proved this is reachable with the same weights we already have. No need
to retrain, recalibrate, or fine-tune. Pure C-port work.

Speed impact: expected ~0 ms. The hot-loop AVX2 paths change from
`_mm256_set1_ps(scalar)` to `_mm256_loadu_ps(per_ch + co)` — same throughput,
just a different source operand. Memory per op: 64-512 floats extra per
Cout-length scale, <1 MB total footprint.

## Commercial significance

All three commercial angles (NDAA turnstile, Trassir OEM, SoC reference) gate
on cos-sim >= 0.98 for biometric use. **Phase B landing = unlocks commercialization.**
Per KB roadmap, prerequisite for any meaningful customer conversation.
