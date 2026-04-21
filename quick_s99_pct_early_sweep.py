"""S99 -- sweep PCT_EARLY with cross-seed test validation.

Reuses the S98 sim path. Sweeps PCT_EARLY over {98.0, 99.0, 99.5, 99.7, 99.9}
with PCT_LATE=99.9, evaluating each config on TWO independent 100-face test
batches (seeds 7777 and 11111) to separate signal from noise.

Goal: find the PCT_EARLY sweet spot and confirm it is stable across seeds.
"""
import os, sys, time, numpy as np
sys.path.insert(0, '.')
import torch
import onnxruntime as ort
from extract_onnx import parse_model
from calibrate_per_channel_int8 import load_lfw_batch
from calibrate_percentile_int8 import collect_per_image_absmax, build_percentile_scales
from calibrate_include_princess import load_lfw_with_princess

from quick_s98_depth_pct import run_sim, cos


def evaluate(g, test_inputs, ort_embs, scales):
    sims = []
    for x, fp in zip(test_inputs, ort_embs):
        sims.append(cos(fp, run_sim(g, x, scales)))
    return np.asarray(sims)


def main():
    n_calib    = int(os.environ.get("N_CALIB", "200"))
    n_test     = int(os.environ.get("N_TEST",  "100"))
    pct_late   = float(os.environ.get("PCT_LATE", "99.9"))
    seed_calib = int(os.environ.get("SEED_CALIB", "1"))

    test_seeds = [7777, 11111]
    early_grid = [98.0, 99.0, 99.5, 99.7, 99.9]

    print(f"S99 PCT_EARLY sweep  N_CALIB={n_calib} N_TEST={n_test}  "
          f"PCT_LATE={pct_late}  test_seeds={test_seeds}")

    g = parse_model("models/w600k_r50.onnx")

    print(f"\n1) loading calib faces ({n_calib}, WITH_PRINCESS)...", flush=True)
    calib_inputs, _ = load_lfw_with_princess("data/lfw", n_calib, seed=seed_calib)

    print(f"2) loading test faces x {len(test_seeds)}...", flush=True)
    test_sets = {}
    for ts in test_seeds:
        test_sets[ts], _ = load_lfw_batch("data/lfw", n_test, seed=ts)

    print(f"3) ORT FP32 ground truth for each test set...", flush=True)
    sess = ort.InferenceSession("models/w600k_r50.onnx", providers=["CPUExecutionProvider"])
    inp_name = sess.get_inputs()[0].name
    ort_embs = {
        ts: [sess.run(None, {inp_name: x})[0].flatten() for x in test_sets[ts]]
        for ts in test_seeds
    }

    print(f"4) collect per-image absmax on calib set...", flush=True)
    t0 = time.perf_counter()
    per_image = collect_per_image_absmax(g, calib_inputs)
    print(f"   done in {time.perf_counter()-t0:.1f}s ({len(per_image)} tensors)")

    print(f"\n5) sweep PCT_EARLY (PCT_LATE={pct_late}):")
    header = f"  {'PCT_EARLY':>9} | "
    for ts in test_seeds:
        header += f"seed={ts}: mean  min   >=99 | "
    header += "  mean-of-means"
    print(header)
    print("  " + "-" * (len(header) - 2))

    results = []
    for e in early_grid:
        dep_val = None if e == pct_late else e
        scales = build_percentile_scales(per_image, percentile=pct_late,
                                          depth_early_percentile=dep_val)
        row_means = []
        row = f"  {e:>9.1f} | "
        for ts in test_seeds:
            t0 = time.perf_counter()
            sims = evaluate(g, test_sets[ts], ort_embs[ts], scales)
            dt = time.perf_counter() - t0
            row += f"       {sims.mean():.5f} {sims.min():.5f} {int((sims>=0.99).sum()):>3}/100 | "
            row_means.append(sims.mean())
        mean_of_means = float(np.mean(row_means))
        row += f"  {mean_of_means:.5f}"
        print(row, flush=True)
        results.append((e, mean_of_means, row_means))

    print("\nRanking (by mean-of-means):")
    for e, mom, means in sorted(results, key=lambda r: -r[1]):
        delta_s1 = means[0] - results[-1][2][0]
        delta_s2 = means[1] - results[-1][2][1]
        print(f"  PCT_EARLY={e:>5.1f}  mom={mom:.5f}  "
              f"seed1 delta vs PCT_EARLY=99.9: {delta_s1:+.5f}  "
              f"seed2 delta: {delta_s2:+.5f}")


if __name__ == "__main__":
    main()
