"""S84 — measure per-Conv mean activation shift between FP32 and INT8-sim.

For each Conv op:
  fp32_mean[oc] = mean over N calibration images and spatial of FP32 output
  int8_mean[oc] = mean over N calibration images and spatial of INT8 output
  shift[oc] = fp32_mean[oc] - int8_mean[oc]

If shifts are consistent (low variance across calibration samples) and
have meaningful magnitude, bias correction may help. If they're noise-sized,
it won't.
"""
import sys, os, time, numpy as np
sys.path.insert(0, '.')
import torch
import torch.nn.functional as F
from extract_onnx import parse_model
from calibrate_per_channel_int8 import (
    collect_ranges, load_lfw_batch, fake_quant_per_channel_sym, fake_quant_weight_per_oc,
)


def run_tracked(g, input_chw, scales, track_ops):
    """Walk graph with INT8 quant; record FP32 (pre-quant) and INT8 (post-quant)
    outputs of the named ops. Returns dict tensor_name -> (fp32, int8) numpy."""
    init = {t["name"]: t for t in g["initializers"]}
    nodes = g["nodes"]
    tensors = {}
    tracked = {}

    def qapply(t, name):
        if name not in scales: return t
        return fake_quant_per_channel_sym(t, scales[name])

    tensors["input.1"] = qapply(torch.from_numpy(input_chw).float(), "input.1")
    for ni, node in enumerate(nodes):
        op = node["op_type"]; outs = node["outputs"]
        if op == "Conv":
            x = tensors[node["inputs"][0]]
            w_raw = torch.from_numpy(init[node["inputs"][1]]["numpy"].astype(np.float32))
            w_q = fake_quant_weight_per_oc(w_raw)
            bias = None
            if len(node["inputs"]) > 2 and node["inputs"][2]:
                bt = init.get(node["inputs"][2])
                if bt and bt.get("numpy") is not None:
                    bias = torch.from_numpy(bt["numpy"].astype(np.float32))
            attrs = {a["name"]: a for a in node["attrs"]}
            y = F.conv2d(x, w_q, bias=bias, stride=attrs["strides"]["ints"][0], padding=attrs["pads"]["ints"][0])
            # Track this Conv if its output is in track_ops
            if outs[0] in track_ops:
                fp32_out = y.detach().numpy().copy()
                int8_out = qapply(y, outs[0]).detach().numpy().copy()
                tracked[outs[0]] = (fp32_out, int8_out)
        elif op == "BatchNormalization":
            x = tensors[node["inputs"][0]]
            gamma = torch.from_numpy(init[node["inputs"][1]]["numpy"].astype(np.float32))
            beta  = torch.from_numpy(init[node["inputs"][2]]["numpy"].astype(np.float32))
            mean  = torch.from_numpy(init[node["inputs"][3]]["numpy"].astype(np.float32))
            var   = torch.from_numpy(init[node["inputs"][4]]["numpy"].astype(np.float32))
            eps = [a for a in node["attrs"] if a["name"] == "epsilon"][0]["f"]
            y = F.batch_norm(x, mean, var, gamma, beta, training=False, eps=eps)
        elif op == "PRelu":
            x = tensors[node["inputs"][0]]
            slope = torch.from_numpy(init[node["inputs"][1]]["numpy"].astype(np.float32))
            if slope.ndim == 1: slope = slope.view(1, -1, 1, 1)
            elif slope.ndim > 1 and slope.numel() == slope.shape[0]: slope = slope.view(1, -1, 1, 1)
            y = torch.where(x >= 0, x, x * slope)
        elif op == "Add":
            y = tensors[node["inputs"][0]] + tensors[node["inputs"][1]]
        elif op == "Flatten":
            y = tensors[node["inputs"][0]].flatten(1)
        elif op == "Gemm":
            x = tensors[node["inputs"][0]]
            w_raw = torch.from_numpy(init[node["inputs"][1]]["numpy"].astype(np.float32))
            w_q = fake_quant_weight_per_oc(w_raw)
            b = torch.from_numpy(init[node["inputs"][2]]["numpy"].astype(np.float32)) if len(node["inputs"]) > 2 else None
            attrs = {a["name"]: a for a in node["attrs"]}
            transB = attrs.get("transB", {"i": 0})["i"]
            B = w_q.T if transB else w_q
            if x.shape[-1] != B.shape[0] and x.shape[-1] == B.shape[-1]: B = B.T
            y = x @ B
            if b is not None: y = y + b
        else:
            continue
        if op != "Flatten":
            y = qapply(y, outs[0])
        tensors[outs[0]] = y
    return tracked


def main():
    N = 20
    inputs, _ = load_lfw_batch("data/lfw", N, seed=1)
    g = parse_model("models/w600k_r50.onnx")
    ranges = collect_ranges(g, inputs)
    scales = {name: (np.maximum(np.abs(mn), np.abs(mx)).astype(np.float32) / 127.0 + 1e-9)
              for name, (mn, mx) in ranges.items()}

    # Track outputs of all Convs
    conv_outs = [n["outputs"][0] for n in g["nodes"] if n["op_type"] == "Conv"]
    print(f"Tracking {len(conv_outs)} Conv outputs on {N} calibration images...")
    # Accumulate per-Conv per-channel mean across images
    sums_fp32 = {}; sums_int8 = {}; counts = {}
    for idx, x in enumerate(inputs):
        tracked = run_tracked(g, x, scales, set(conv_outs))
        for name, (fp32, int8) in tracked.items():
            # Mean over batch+spatial dims to get per-channel means
            fp32_pc = fp32.mean(axis=(0, 2, 3))
            int8_pc = int8.mean(axis=(0, 2, 3))
            if name not in sums_fp32:
                sums_fp32[name] = fp32_pc.copy(); sums_int8[name] = int8_pc.copy()
                counts[name] = 1
            else:
                sums_fp32[name] += fp32_pc; sums_int8[name] += int8_pc
                counts[name] += 1
        if (idx+1) % 10 == 0: print(f"  {idx+1}/{N}", flush=True)

    # Compute per-Conv per-channel bias shift (fp32 - int8)
    print(f"\n=== Per-Conv bias shift statistics ===")
    print(f"{'tensor':<10s}  {'Cout':>5s}  {'|shift|_mean':>13s}  {'|shift|_max':>13s}  {'relative':>10s}")
    all_shifts = []
    for name in conv_outs[:50]:  # first 50 for brevity
        if name not in sums_fp32: continue
        fp32_mean = sums_fp32[name] / counts[name]
        int8_mean = sums_int8[name] / counts[name]
        shift = fp32_mean - int8_mean
        abs_shift = np.abs(shift)
        # Normalize by fp32 scale to see if shift is a meaningful fraction
        fp32_scale = np.abs(fp32_mean).mean() + 1e-9
        rel = abs_shift.mean() / fp32_scale
        all_shifts.append((name, abs_shift.mean(), abs_shift.max(), rel, shift.size))
    # Sort by relative shift
    all_shifts.sort(key=lambda x: -x[3])
    print("Top 20 Convs by relative bias shift:")
    for name, mean_abs, max_abs, rel, n in all_shifts[:20]:
        print(f"  {name:<10s}  {n:>5d}  {mean_abs:>13.5f}  {max_abs:>13.5f}  {rel:>9.2%}")
    print("\nBottom 5:")
    for name, mean_abs, max_abs, rel, n in all_shifts[-5:]:
        print(f"  {name:<10s}  {n:>5d}  {mean_abs:>13.5f}  {max_abs:>13.5f}  {rel:>9.2%}")


if __name__ == "__main__":
    main()
