"""S108 -- profile S_a[ci] variance per Conv input.

The S107 gap source is weight-fold rounding. The error on recovered
W[co,ci,...] is amplified for small S_a[ci]. If S_a[ci] is roughly
uniform across ci, the fold rounding is already well-balanced and
cross-layer equalization (DFQ) cannot help.

Profile for every Conv/Gemm in the model:
  S_a_max / S_a_min
  coefficient of variation (std / mean)
  max / median ratio

Rank Convs by S_a variance -- the high-variance ones are where DFQ
has leverage.
"""
import os, sys, numpy as np
sys.path.insert(0, '.')
from extract_onnx import parse_model
from calibrate_per_channel_int8 import load_lfw_batch
from calibrate_percentile_int8 import collect_per_image_absmax, build_percentile_scales
from calibrate_include_princess import load_lfw_with_princess


def main():
    n_calib = int(os.environ.get("N_CALIB", "200"))
    pct_late = float(os.environ.get("PCT_LATE", "99.9"))

    g = parse_model("models/w600k_r50.onnx")
    print("1) load calib...", flush=True)
    calib_inputs, _ = load_lfw_with_princess("data/lfw", n_calib, seed=1)

    print("2) per-image absmax...", flush=True)
    per_image = collect_per_image_absmax(g, calib_inputs)
    act_scales = build_percentile_scales(per_image, percentile=pct_late)

    # For each Conv/Gemm, look at the input tensor's per-ci scale
    print("\n3) per-Conv S_a[ci] variance profile:\n")
    rows = []
    for ni, node in enumerate(g["nodes"]):
        if node["op_type"] not in ("Conv", "Gemm"):
            continue
        in_name = node["inputs"][0]
        S_a = act_scales.get(in_name)
        if S_a is None or S_a.size < 2:
            continue
        s = S_a.astype(np.float32)
        s_nonzero = s[s > 0]
        if s_nonzero.size < 2:
            continue
        max_min = float(s_nonzero.max() / s_nonzero.min())
        cv = float(s_nonzero.std() / s_nonzero.mean())
        max_med = float(s_nonzero.max() / np.median(s_nonzero))
        rows.append((ni, node["op_type"], in_name, len(s), max_min, cv, max_med))

    # Sort by max/min ratio descending
    rows.sort(key=lambda r: -r[4])
    print(f"  {'rank':>4} {'op':>4} {'#ci':>6} {'max/min':>10} {'cv':>7} {'max/med':>8}  input_name")
    print(f"  {'-'*4} {'-'*4} {'-'*6} {'-'*10} {'-'*7} {'-'*8}  ----------")
    for i, (ni, ot, nm, nci, mm, cv, mmd) in enumerate(rows[:20]):
        op_short = ot[:4]
        print(f"  {i+1:>4} {op_short:>4} {nci:>6} {mm:>10.2f} {cv:>7.3f} {mmd:>8.2f}  {nm}")

    print(f"\n  total Conv/Gemm with S_a data: {len(rows)}")
    mm_vals = np.asarray([r[4] for r in rows])
    print(f"  S_a max/min distribution:  median {np.median(mm_vals):.2f}  "
          f"p75 {np.percentile(mm_vals, 75):.2f}  p95 {np.percentile(mm_vals, 95):.2f}  "
          f"p99 {np.percentile(mm_vals, 99):.2f}")

    # If median max/min is close to 1 (say <3), S_a is roughly uniform, DFQ low leverage.
    # If median is >10 or many layers have >20, DFQ has clear leverage.
    high_var = sum(1 for r in rows if r[4] > 10)
    very_high = sum(1 for r in rows if r[4] > 100)
    print(f"\n  # layers with max/min > 10: {high_var}/{len(rows)}")
    print(f"  # layers with max/min > 100: {very_high}/{len(rows)}")
    if very_high > 5:
        print("  >>> DFQ has clear leverage, worth implementing.")
    elif high_var > 10:
        print("  >>> DFQ has moderate leverage, implementation may yield +0.001..+0.003 mean cos-sim.")
    else:
        print("  >>> S_a already relatively uniform, DFQ unlikely to help meaningfully.")


if __name__ == "__main__":
    main()
