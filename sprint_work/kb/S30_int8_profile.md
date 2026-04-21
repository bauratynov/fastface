# S30 — INT8 per-op-type profile

Date: 2026-04-20
Baseline: fastface_int8.exe runs 28.66 ms/inference, cos-sim 0.945, 1.12x ORT.
Question: why is INT8 slower than FP32 (25.93 ms) when i7-13700 has AVX-VNNI
and the INT8 GEMM already uses VPDPBUSD?

## Instrumentation

Added `PROFILE_OPS` compile-time flag to `arcface_forward_int8.c`. Wraps each
op dispatch with `QueryPerformanceCounter` and accumulates by op type.

Guard on `main` in `kernels/ffw2_loader.c` (`#ifndef FFW2_NOMAIN`) so the
loader file can be linked into other drivers without colliding mains.

Build:
```
gcc -O3 -march=native -mavx2 -mfma -mavxvnni -DPROFILE_OPS -DFFW2_NOMAIN -fopenmp \
  arcface_forward_int8.c kernels/ffw2_loader.c kernels/conv2d_nhwc.c \
  kernels/gemm_int8_v2.c kernels/int8_epilogue.c -o fastface_int8_prof.exe
```

## Result (5 trials x 30 iter + 1 warmup = 151 inferences, total 4525 ms)

| op type | calls | total ms | % | ms/call |
|---|---:|---:|---:|---:|
| CONV | 8003 | 2026.3 | **44.8** | 0.253 |
| ADD | 3624 | 1220.7 | **27.0** | 0.337 |
| GEMM | 151 | 1094.2 | **24.2** | **7.25** |
| BN | 3926 | 151.9 | 3.4 | 0.039 |
| SAVE_ID | 3624 | 16.8 | 0.4 | 0.005 |
| BLOCK_START | 3624 | 14.0 | 0.3 | 0.004 |
| FLATTEN | 151 | 1.1 | 0.0 | 0.007 |

Per-inference: **CONV 13.4 ms + ADD 8.1 ms + GEMM 7.25 ms + rest 1.2 ms = 29.95 ms**.

## Findings

### Finding 1 — OP_GEMM is fully scalar

Final Linear(25088 -> 512) runs 512 x 25088 = 12.8M multiplies per inference
in a naive triple-nested C loop that does int8 -> fp32 cast per element:

```c
for (nn = 0; nn < op->N; nn++) {
    float s = op->gemm_bias[nn];
    for (kk = 0; kk < op->K; kk++) {
        s += (float)A[kk] * A_scale * (float)op->gemm_w[...] * op->gemm_scales[nn];
    }
    final_emb[nn] = s;
}
```

At 7.25 ms it is single-handedly the reason INT8 is slower than FP32 at b=1.
Expected after VNNI vectorization: ~0.5-1.0 ms. Gain: **~6.5 ms/inference**.

### Finding 2 — OP_ADD is bandwidth bound

24 ADD ops per inference, total 8.1 ms. That's 0.34 ms per call at i8
granularity over modest tensor sizes (max 128x28x28 = 100KB). Already
AVX2-vectorized in `add_requant_int8`. The cost here is the memory pass
itself — each add reads 2 tensors, writes 1.

Realistic optimization: fuse ADD into Conv epilogue (like S29d did for FP32).
Gain: ~2-4 ms if we can remove 12-24 memory passes.

### Finding 3 — CONV is well-tuned

44.8% / 53 conv ops per inference = 0.25 ms/call average. CONV uses
im2col + VPDPBUSD GEMM. Largest is the stem 7x7 over 112x112, smallest
are late 1x1 ops.

No Winograd path for INT8 — could potentially speed up the 3x3 convs
further (24 of the 53 ops), but INT8 Winograd requires careful i16/i32
intermediate handling. Moonshot tier.

## S31 priority order (ROI-ranked)

1. **Vectorize OP_GEMM (VNNI int8) — 24% of runtime** — expected 28.66 -> 22 ms
   (1.4x ORT). This is the highest-leverage single change.
2. **Fuse ADD into Conv epilogue** — expected 22 -> 19-20 ms (1.6x ORT).
3. **INT8 Winograd for 3x3 Conv2d** — moonshot, speculative 5-8 ms.
4. **Per-channel scales** — correctness fix (0.945 -> 0.986), not speed.

## Rollback

Profiling guard is purely compile-time (`-DPROFILE_OPS`). Production binary
`fastface_int8.exe` is unaffected. The guard added around `ffw2_loader.c`
main is backward-compatible (`FFW2_NOMAIN` default-off).
