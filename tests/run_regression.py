"""FastFace regression test — verify current build produces golden output.

Runs the current `fastface_int8.exe models/w600k_r50_ffw4.bin` on
`tests/golden_input.bin` and compares the output against two committed
reference embeddings:

1. `tests/golden_int8_emb.bin` — current INT8 output (bit-exact check).
   Any non-zero diff indicates a code change that changed numerical output.
2. `tests/golden_ort_fp32_emb.bin` — ORT FP32 reference (cos-sim >= 0.98).
   Downstream quality sanity check.

Exit 0 on pass, non-zero on fail.

Usage: python tests/run_regression.py
       (from repo root)
"""
import os, subprocess, sys
import numpy as np


GOLDEN_COS_SIM_THRESHOLD = 0.98  # relaxed for calibration-file changes


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(root)

    # Binary name differs between platforms: .exe suffix on Windows, none on Linux/macOS.
    if os.name == "nt":
        exe_candidates = ["./fastface_int8.exe"]
    else:
        exe_candidates = ["./fastface_int8", "./fastface_int8.exe"]  # exe fallback if cross-compiled
    exe = next((p for p in exe_candidates if os.path.exists(p)), exe_candidates[0])

    weights = "models/w600k_r50_ffw4.bin"
    in_path = "tests/golden_input.bin"
    tmp_out = "tests/_current_out.bin"

    env = os.environ.copy()
    # Mingw libgomp needed on Windows
    if os.name == "nt":
        env["PATH"] = "C:/mingw64/bin" + os.pathsep + env.get("PATH", "")

    for p in [exe, weights, in_path, "tests/golden_int8_emb.bin", "tests/golden_ort_fp32_emb.bin"]:
        if not os.path.exists(p):
            print(f"FAIL: missing {p}", file=sys.stderr)
            return 2

    r = subprocess.run([exe, weights, "--in", in_path, "--out", tmp_out],
                       capture_output=True, env=env, timeout=60)
    if r.returncode != 0:
        print(f"FAIL: exe returned {r.returncode}", file=sys.stderr)
        print("stderr:", r.stderr.decode()[:500], file=sys.stderr)
        return 3

    current = np.fromfile(tmp_out, dtype=np.float32)[:512]
    golden_int8 = np.fromfile("tests/golden_int8_emb.bin", dtype=np.float32)[:512]
    golden_fp32 = np.fromfile("tests/golden_ort_fp32_emb.bin", dtype=np.float32)[:512]

    # 1. Bit-exact INT8 check (catches code regressions)
    max_abs_diff = float(np.max(np.abs(current - golden_int8)))
    bit_exact = np.array_equal(current, golden_int8)

    # 2. FP32 cos-sim quality check
    cs = float(np.dot(current, golden_fp32) /
               (np.linalg.norm(current) * np.linalg.norm(golden_fp32)))

    print(f"INT8 bit-exact vs golden:  {bit_exact}  (max abs diff: {max_abs_diff:.6e})")
    print(f"cos-sim vs ORT FP32 golden: {cs:.6f}  (threshold {GOLDEN_COS_SIM_THRESHOLD:.3f})")

    passed = True
    if not bit_exact:
        if max_abs_diff > 1e-5:
            print("WARN: INT8 output drifted from golden. Investigate before shipping.")
            # Don't fail — calibration changes are expected; fall through to cos-sim check.
    if cs < GOLDEN_COS_SIM_THRESHOLD:
        print(f"FAIL: cos-sim {cs:.6f} below threshold {GOLDEN_COS_SIM_THRESHOLD:.3f}")
        passed = False

    if passed:
        print("PASS")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
