"""S82 — per-layer sensitivity analysis.

For each quantizable op, simulate "what if this one op stayed FP32 while
everything else is INT8?" and measure cos-sim improvement. Rank layers by
recovered cos-sim delta. Top layers are the ones most hurting accuracy.
"""
import os, sys, glob, random, time, numpy as np
sys.path.insert(0, '.')
import torch
import torch.nn.functional as F
import onnxruntime as ort
from PIL import Image
from extract_onnx import parse_model
from calibrate_per_channel_int8 import (
    collect_ranges, load_lfw_batch, fake_quant_per_channel_sym, fake_quant_weight_per_oc,
)


def load_one_face(path):
    img = Image.open(path).convert("RGB")
    w, h = img.size; s = 150
    img = img.crop(((w-s)//2, max(0,(h-s)//2-10), (w-s)//2+s, max(0,(h-s)//2-10)+s)).resize((112,112), Image.BILINEAR)
    arr = (np.asarray(img, dtype=np.float32) - 127.5) / 127.5
    return np.transpose(arr, (2,0,1))[None].copy()


def run_with_skip(g, input_chw, scales, skip_op_idx):
    """Walk the graph with per-channel INT8 quant, but skip the quantize step
    on the op at `skip_op_idx` (0-indexed within Conv/BN/PRelu/Add/Gemm ops).
    Returns final embedding."""
    init = {t["name"]: t for t in g["initializers"]}
    nodes = g["nodes"]
    tensors = {}

    def qapply(t, name):
        if name not in scales: return t
        s = scales[name]
        return fake_quant_per_channel_sym(t, s)

    quant_idx = -1

    tensors["input.1"] = qapply(torch.from_numpy(input_chw).float(), "input.1")
    for ni, node in enumerate(nodes):
        op = node["op_type"]; outs = node["outputs"]
        if op == "Conv":
            quant_idx += 1
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
        elif op == "BatchNormalization":
            quant_idx += 1
            x = tensors[node["inputs"][0]]
            gamma = torch.from_numpy(init[node["inputs"][1]]["numpy"].astype(np.float32))
            beta  = torch.from_numpy(init[node["inputs"][2]]["numpy"].astype(np.float32))
            mean  = torch.from_numpy(init[node["inputs"][3]]["numpy"].astype(np.float32))
            var   = torch.from_numpy(init[node["inputs"][4]]["numpy"].astype(np.float32))
            eps = [a for a in node["attrs"] if a["name"] == "epsilon"][0]["f"]
            y = F.batch_norm(x, mean, var, gamma, beta, training=False, eps=eps)
        elif op == "PRelu":
            quant_idx += 1
            x = tensors[node["inputs"][0]]
            slope = torch.from_numpy(init[node["inputs"][1]]["numpy"].astype(np.float32))
            if slope.ndim == 1: slope = slope.view(1, -1, 1, 1)
            elif slope.ndim > 1 and slope.numel() == slope.shape[0]: slope = slope.view(1, -1, 1, 1)
            y = torch.where(x >= 0, x, x * slope)
        elif op == "Add":
            quant_idx += 1
            y = tensors[node["inputs"][0]] + tensors[node["inputs"][1]]
        elif op == "Flatten":
            y = tensors[node["inputs"][0]].flatten(1)
        elif op == "Gemm":
            quant_idx += 1
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
        # Apply output quant UNLESS this is the skipped op
        if op != "Flatten" and quant_idx != skip_op_idx:
            y = qapply(y, outs[0])
        tensors[outs[0]] = y
    return tensors[nodes[-1]["outputs"][0]].numpy().flatten()


def cos_sim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def main():
    n_calib = 100
    calib_inputs, _ = load_lfw_batch("data/lfw", n_calib, seed=1)
    test_inputs, _  = load_lfw_batch("data/lfw", 5, seed=999)  # small for speed
    g = parse_model("models/w600k_r50.onnx")

    print(f"Calibrating on {n_calib} faces...", flush=True)
    t0 = time.perf_counter()
    ranges = collect_ranges(g, calib_inputs)
    scales = {}
    for name, (tmin, tmax) in ranges.items():
        absmax = np.maximum(np.abs(tmin), np.abs(tmax)).astype(np.float32)
        absmax = np.where(absmax > 0, absmax, 1e-6)
        scales[name] = (absmax / 127.0).astype(np.float32)
    print(f"  done in {time.perf_counter()-t0:.1f}s")

    # Count quantizable ops
    n_q_ops = sum(1 for n in g["nodes"] if n["op_type"] in ("Conv", "BatchNormalization", "PRelu", "Add", "Gemm"))
    print(f"n_quantizable_ops = {n_q_ops}")

    sess = ort.InferenceSession("models/w600k_r50.onnx", providers=["CPUExecutionProvider"])
    # FP32 reference
    fp32_embs = [sess.run(None, {sess.get_inputs()[0].name: x})[0].flatten() for x in test_inputs]

    # Baseline INT8 (skip_op_idx = -1, nothing skipped)
    print("Baseline cos-sim with full INT8...", flush=True)
    base = [cos_sim(run_with_skip(g, x, scales, -1), f) for x, f in zip(test_inputs, fp32_embs)]
    base_mean = np.mean(base)
    print(f"  baseline mean cos-sim: {base_mean:.6f}")

    # For each quantizable op, skip and measure
    print(f"\nTrying each of {n_q_ops} quantizable ops as FP32-skip:")
    print(f"{'op_idx':>6s}  {'op_type':<22s}  {'mean_cs':>10s}  {'delta':>8s}")
    deltas = []
    node_info = []
    qi = -1
    for ni, n in enumerate(g["nodes"]):
        if n["op_type"] not in ("Conv", "BatchNormalization", "PRelu", "Add", "Gemm"):
            continue
        qi += 1
        node_info.append((qi, n["op_type"], n["outputs"][0]))
    # Subsample for speed (too slow to test all 131 ops)
    target_ops = range(0, n_q_ops, max(1, n_q_ops // 30))  # ~30 samples
    for qi in target_ops:
        skipped = [cos_sim(run_with_skip(g, x, scales, qi), f) for x, f in zip(test_inputs, fp32_embs)]
        mean_cs = np.mean(skipped)
        delta = mean_cs - base_mean
        deltas.append((qi, mean_cs, delta, node_info[qi][1], node_info[qi][2]))
        print(f"  {qi:>6d}  {node_info[qi][1]:<22s}  {mean_cs:>9.6f}  {delta:>+.6f}")

    print(f"\n=== TOP 5 most sensitive ops (highest gain if kept FP32) ===")
    for qi, cs, dlt, opt, tensor_name in sorted(deltas, key=lambda x: -x[2])[:5]:
        print(f"  op_idx={qi:3d}  {opt:<22s}  tensor={tensor_name:<6s}  cos-sim gain {dlt:+.6f}")


if __name__ == "__main__":
    main()
