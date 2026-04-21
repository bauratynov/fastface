# Contributing to FastFace

Thanks for your interest. FastFace is a narrow-scope CPU inference
engine for one specific face recognition model -- contributions that
keep the scope narrow and the binary small are especially welcome.

## Quick links

- [Code of conduct](#code-of-conduct)
- [Bug reports](#bug-reports)
- [Feature requests](#feature-requests)
- [Pull requests](#pull-requests)
- [Development setup](#development-setup)
- [Testing](#testing)
- [Priority contribution areas](#priority-contribution-areas)

## Code of conduct

Be kind. Be specific. Assume competence. If a PR needs work, say what
and why; don't gatekeep. If a review comment upsets you, ask for
clarification before reacting. Standard stuff.

## Bug reports

Include:

1. Your CPU model (`cat /proc/cpuinfo | head -20` on Linux, System Info
   on Windows).
2. Your compiler version (`gcc -v`).
3. The commit hash (`git rev-parse HEAD`).
4. Exact command-line and output (including `stderr`).
5. If it's an accuracy regression: which LFW fold, which pair, the
   FP32 and INT8 cos-sim numbers.

Reproducibility is king. If you can't reproduce it on a second run,
please note that.

## Feature requests

Before opening an issue:

- Check `sprint_work/kb/` for whether this was already tried (50+
  experiments, many negative). Saves time.
- Consider whether it's in scope: FastFace is for one model
  (`w600k_r50` IResNet-100 ArcFace) on CPU. Multi-model support
  would need a bigger architectural discussion.

Accepted feature-request types:

- ARM NEON port (see [priorities](#priority-contribution-areas))
- Additional language bindings (Rust, Java, JS/Node)
- Additional benchmarks on new CPUs (`bench_sustained.py` results)
- Bug fixes

NOT accepted by default:

- GPU support (different use case; CUDA users have TensorRT)
- Alternative model formats (ONNX-graph-compile, TFLite, etc.)
- Training-side features (FastFace is inference-only)

## Pull requests

### Commit message format

    Short imperative subject line, <= 60 chars
    
    Longer body explaining *why*, not *what*. What is obvious from
    the diff; why is not.
    
    - Bullet point for each notable change
    - Reference any related sprint KB: "Per sprint_work/kb/S85"
    - Reference the issue this closes: "Closes #42"
    
    Co-Authored-By: if this was pair-programmed

### PR checks

Before requesting review, verify:

- [ ] `make test` passes bit-exact
- [ ] `make bench-lfw` reports >= 99.4% accuracy on 1000 pairs
      (regression threshold)
- [ ] `python tests/run_regression.py` passes
- [ ] No new compiler warnings with `-Wall -Wextra` on gcc 11+
- [ ] If you changed a kernel, add a KB entry to `sprint_work/kb/`
      documenting what you tried (including negative iterations)

### PR size

- < 200 LOC: fast path, often same-day review
- 200-500 LOC: expect 2-3 days; probably wants design discussion
- > 500 LOC: open an issue first to sketch the approach

## Development setup

### Prerequisites

- GCC 11+ (or Clang 14+ with compatible intrinsics)
- GNU Make
- Python 3.10+ with numpy, onnxruntime, pillow, torch (for calibration sim)
- MinGW-w64 on Windows (tested with gcc 13.2)

Plus, **for benchmarks**:

- LFW dataset -- not redistributed here (license). Download from
  http://vis-www.cs.umass.edu/lfw/ and extract to `data/lfw/<person>/<face>.jpg`.
- ONNX reference model -- `models/w600k_r50.onnx` from InsightFace's
  model zoo, for ORT comparison runs. Not redistributed here.
  Our pre-calibrated `models/w600k_r50_ffw4.bin` IS shipped in the repo.

### Quick build

```bash
make clean
make              # builds fastface_int8 (+ .exe on Windows) + libfastface.a
make test         # regression -- should print PASS
make calibrate    # regenerates models/*.bin from ONNX (needs LFW data)
```

### Benchmark suite

```bash
make bench-lfw           # 1000-pair verification
python bench_lfw_full.py # 10-fold 6000-pair protocol
python bench_sustained.py --duration 60  # sustained throughput
```

### Running the Python sim

For quick calibration experiments without a C rebuild:

```python
from quick_s98_depth_pct import run_sim, cos
# ... see sprint_work/kb/S98_*.md for examples
```

## Testing

### Golden regression

Every build must produce bit-exact output vs `tests/golden_int8_emb.bin`
on `tests/golden_input.bin`. Any change to numerical output requires:

1. Updating `tests/golden_int8_emb.bin` (regenerate via:
   `./fastface_int8.exe models/w600k_r50_ffw4.bin --in tests/golden_input.bin --out tests/golden_int8_emb.bin`)
2. Committing both the code change and the new golden in one commit
3. Explicitly noting the numerical change in the commit message

### Accuracy regression

If your change might affect accuracy, run `bench_lfw_full.py --seed 42`.
Expected: INT8 99.65% +/- 0.23%, FP32 99.63% +/- 0.22%. If your change
drops INT8 below 99.55%, investigate before merging.

## Priority contribution areas

In rough order of value-per-effort:

### 1. ARM NEON port

Biggest user-facing unlock. See `sprint_work/PORTING_ARM_NEON.md` for
design notes. Core work is `kernels/gemm_int8_matvec.c` -- ~200 LOC of
AVX-VNNI intrinsics that need a parallel NEON version using
`SDOT`/`UDOT`.

### 2. Benchmarks on more hardware

Current numbers are i7-13700 only. PRs adding clean benchmark runs on
other CPUs (any Raptor Lake / Alder Lake / Zen 4+ with AVX-VNNI) are
high-value. Just run `bench_stable_int8_ffw4.py` + `bench_sustained.py`
and include the output in the PR.

### 3. Additional language bindings

Python, C, Go exist. Rust (via `bindgen`), Java (via JNI), JS/Node
(via N-API) are useful additions. Follow the pattern of `go/fastface/`
which wraps `libfastface.a` directly.

### 4. Face detector alternatives

Current pipeline uses SCRFD-10G via ONNX (12 ms). Cheaper detectors
(YuNet at 8 ms, BlazeFace at 6 ms) would help edge deployments. Add
as a swappable module in `face_pipeline.py`.

### 5. Documentation

- Getting-started tutorial for calibration pipeline
- Architecture deep-dive on the fused epilogue
- Porting guide (what changes if you want a different model)

## Questions?

Open an issue with the `question` label, or DM the maintainer.
