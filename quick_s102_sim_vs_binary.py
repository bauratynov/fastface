"""S102 -- calibrate: Python sim vs actual C binary on 100 LFW faces.

Run real fastface_int8.exe (v1.1.0 artefacts restored after S100) on the
same 100 test faces we've been using (seed 7777). Compare cos-sim vs ORT.
Also run the Python per-channel sim on the same faces.

Goal: quantify the gap. If sim tracks binary within 0.001 on mean cos-sim,
future sim experiments are trustworthy for screening. If larger gap, we
need to validate on the binary directly before committing.
"""
import os, sys, time, subprocess, numpy as np
sys.path.insert(0, '.')
import torch
import onnxruntime as ort
from extract_onnx import parse_model
from calibrate_per_channel_int8 import load_lfw_batch
from calibrate_percentile_int8 import collect_per_image_absmax, build_percentile_scales
from calibrate_include_princess import load_lfw_with_princess
from quick_s98_depth_pct import run_sim, cos


def main():
    n_calib   = int(os.environ.get("N_CALIB", "200"))
    n_test    = int(os.environ.get("N_TEST", "100"))
    pct_late  = float(os.environ.get("PCT_LATE", "99.9"))
    seed_test = int(os.environ.get("SEED_TEST", "7777"))

    g = parse_model("models/w600k_r50.onnx")

    print("1) load calib (200 WITH_PRINCESS)...", flush=True)
    calib_inputs, _ = load_lfw_with_princess("data/lfw", n_calib, seed=1)

    print(f"2) load test ({n_test} seed={seed_test})...", flush=True)
    test_inputs, _ = load_lfw_batch("data/lfw", n_test, seed=seed_test)

    print("3) ORT FP32 refs...", flush=True)
    sess = ort.InferenceSession("models/w600k_r50.onnx",
                                 providers=["CPUExecutionProvider"])
    ort_embs = [sess.run(None, {sess.get_inputs()[0].name: x})[0].flatten()
                for x in test_inputs]

    print("4) Python per-channel sim on test...", flush=True)
    t0 = time.perf_counter()
    per_image = collect_per_image_absmax(g, calib_inputs)
    print(f"   calib done in {time.perf_counter()-t0:.1f}s")
    act_scales = build_percentile_scales(per_image, percentile=pct_late)
    sim_sims = np.asarray([cos(fp, run_sim(g, x, act_scales))
                            for x, fp in zip(test_inputs, ort_embs)])
    print(f"   [Python sim]  mean {sim_sims.mean():.5f}  median {np.median(sim_sims):.5f}  "
          f"min {sim_sims.min():.5f}  >=0.99 {int((sim_sims>=0.99).sum())}/100")

    print("\n5) C binary (v1.1.0 restored) on same faces...", flush=True)
    env = os.environ.copy()
    for p in ("C:/mingw64/bin", "C:\\mingw64\\bin"):
        if os.path.isdir(p):
            env["PATH"] = p + os.pathsep + env.get("PATH", "")
            break
    _suffix = ".exe" if os.name == "nt" else ""
    _exe = next((p for p in (f"./fastface_int8{_suffix}", "./fastface_int8.exe", "./fastface_int8")
                 if os.path.exists(p)), f"./fastface_int8{_suffix}")
    proc = subprocess.Popen(
        [_exe, "models/w600k_r50_ffw4.bin", "--server"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        env=env, bufsize=0)

    bin_sims = []
    # binary expects NHWC fp32; our test_inputs are NCHW; transpose
    for x, fp in zip(test_inputs, ort_embs):
        nhwc = np.transpose(x[0], (1, 2, 0)).astype(np.float32)
        proc.stdin.write(nhwc.tobytes()); proc.stdin.flush()
        emb = np.frombuffer(proc.stdout.read(512 * 4), dtype=np.float32).copy()
        bin_sims.append(cos(fp, emb))
    proc.stdin.close(); proc.wait(timeout=5)

    bin_sims = np.asarray(bin_sims)
    print(f"   [C binary ]  mean {bin_sims.mean():.5f}  median {np.median(bin_sims):.5f}  "
          f"min {bin_sims.min():.5f}  >=0.99 {int((bin_sims>=0.99).sum())}/100")

    print("\n6) sim vs binary delta per face:")
    delta = bin_sims - sim_sims
    print(f"   mean delta {delta.mean():+.5f}  std {delta.std():.5f}  "
          f"min delta {delta.min():+.5f}  max delta {delta.max():+.5f}")
    print(f"   |delta| median {np.median(np.abs(delta)):.5f}  "
          f"|delta| p95 {np.percentile(np.abs(delta),95):.5f}")


if __name__ == "__main__":
    main()
