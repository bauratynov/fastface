# S38-S40 — Ship-quality INT8 via coherent per-channel fold

Date: 2026-04-20
**SHIP MILESTONE**. git tag `s38-ship-quality`, commit `471edd4e` (S38) + `fc167ab4` (S39).

## Final numbers

**INT8 path (fastface_int8.exe + FFW4 + OPSC2):**
- Speed: 13.48 ms median, 13.31 ms min, 2.368x ORT median, 20/20 wins (S38 stable bench)
- Cos-sim vs FP32 reference (20 LFW test faces, distinct from 500 calibration):
  - mean **0.986**
  - min 0.972
  - median **0.986**
  - max 0.992
  - **19/20 faces >= 0.98** (ship-threshold met)
  - 2/20 faces >= 0.99

Match to S21 Python simulation `sym_channel` mode (mean 0.986) is exact.

## How we got here (S31-S40 arc)

| Sprint | Delta | Cumulative |
|---|---|---|
| S26 (prior) | INT8 baseline 28.85 ms, cos-sim 0.942 | 1.11x ORT |
| S30 | per-op profile: OP_GEMM = 24%, OP_ADD = 27% | — |
| **S31** | VNNI matvec for final Linear: -7 ms | 21.06 ms, 1.50x |
| **S32** | ADD fusion into Conv epilogue: -7 ms, cos-sim improved (0.9507 -> 0.9543) | **13.36 ms, 2.355x** |
| S33 | re-profile: CONV = 90.7% (only lever left) | — |
| S34-S35 | INT8 Winograd moonshot aborted (0.66x VNNI on i7-13700) | — |
| **S36 phase A** | fused_epilogue_int8 per-channel inv_out infra | backward-compat only |
| S37 | Gemm-only per-channel fold (+last BN per-channel requant) | +0.0002 cos-sim |
| **S38** | **Full Conv-chain per-channel fold**, coherent pipeline | **13.48 ms, 2.368x, cos-sim 0.986** |
| S39 | Recalibrate with N=500 (was 20) | +0.002 cos-sim, 0.988 single-face |
| S40 | Multi-face eval on 20 held-out faces | mean 0.986, 19/20 >= 0.98 |

## Architecture at S38

Coherent per-channel pipeline across all 178 ops:
1. Initial input quantized per-channel (3 RGB scales from OPSC2).
2. Every Conv op: weights pre-folded with `S_a_per_ch` of Conv input tensor:
   `W_folded[oc, ci, kh, kw] = W[oc, ci, kh, kw] * S_a[ci]`
   Runtime Conv epilogue uses `in_scale = 1.0`, per-channel `inv_out_per_ch`
   for requant, per-channel `add_scale_per_ch` for shortcut ADD fusion.
3. Runtime BN/PReLU ops use per-channel `in_scale_per_ch` + `inv_out_per_ch`.
4. SAVE_ID saves per-channel scale array pointer alongside int8 buffer.
5. Final Gemm weights pre-folded with pre-Flatten per-channel scales; matvec
   runs with `A_scale = 1.0`.

## Commercial significance

All three commercial angles from BZ research are now technically unblocked:

| Angle | Prior blocker | Status now |
|---|---|---|
| NDAA turnstile / access control | cos-sim 0.954 < 0.98 | **UNBLOCKED** 0.986 ship |
| Trassir OEM for CIS ReID | cos-sim 0.954 + no detection | half-unblocked (cos-sim OK) |
| SoC vendor reference | cos-sim + x86-only | cos-sim OK, ARM still needed |

The "near-FP32 quality at 2.35x CPU speed with 96 KB binary" narrative is now
defensible with LFW data.

## Rollback / reproduction

```
git checkout s38-ship-quality   # commit 471edd4e

# Regenerate with N=500 calibration:
N_CALIB=500 python export_op_scales_v2.py
N_CALIB=500 python prepare_weights_v3.py

# Build:
gcc -O3 -march=native -mavxvnni -DFFW2_NOMAIN -fopenmp \
  arcface_forward_int8.c kernels/ffw2_loader.c kernels/conv2d_nhwc.c \
  kernels/gemm_int8_v2.c kernels/int8_epilogue.c kernels/gemm_int8_matvec.c \
  -o fastface_int8.exe

# Run:
./fastface_int8.exe models/w600k_r50_ffw4.bin
```

## Next: S41+ roadmap

With quality ship-ready, remaining work prioritized by commercial impact:

- **S41-S42 speed**: Triple-fusion BN+ADD+Conv, direct 1x1 conv. Target 12 ms.
- **S43-S45 batched INT8**: Port FP32 batching to INT8. Target 8 ms/face at B=8.
- **S46 precision safety net**: INT16 mixed precision for first/last Convs.
- **S47 consolidation**: regression suite, stable bench discipline.
- **S48 ARM NEON**: first cross-compile to ARM target.
- **S49 detector pipeline**: RetinaFace INT8 or YOLO-Face to ship "face -> embedding" end to end.
- **S50 final writeup**: consolidated benchmark paper, ready for publication.
