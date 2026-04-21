"""Session 11 — simulate INT8 quantization in torch reference to measure quant loss.

Approach (fake-quant / QAT-style):
- At each activation boundary, apply quant-dequant round-trip:
    q = round(x / scale), clamp [-128, 127]
    x' = q * scale
- Scale is computed from calibration (absmax over a few calibration inputs)
- Weights: per-output-channel absmax quantization (as we do in prepare_weights_v2)

Measure cos-sim(torch_int8_sim, ORT) — tells us the BEST cos-sim our C binary can achieve
assuming we correctly implement INT8 arithmetic.
"""
import sys
sys.path.insert(0, '.')
from extract_onnx import parse_model
import numpy as np
import torch
import torch.nn.functional as F
import onnxruntime as ort
import time


def cos_sim(a, b):
    a = np.asarray(a).flatten().astype(np.float32)
    b = np.asarray(b).flatten().astype(np.float32)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def fake_quant_per_tensor(x, scale):
    """Quant-dequant round-trip per-tensor symmetric."""
    q = torch.round(x / scale).clamp(-128, 127)
    return q * scale


def fake_quant_per_channel_weight(w, axis=0):
    """Per-output-channel quant-dequant for 4D conv weight [Cout, Cin, Kh, Kw]."""
    dims = [d for d in range(w.dim()) if d != axis]
    # absmax along non-axis dims
    absmax = w.abs().amax(dim=dims, keepdim=True)
    scale = absmax / 127.0
    scale_safe = torch.where(scale > 0, scale, torch.ones_like(scale))
    q = torch.round(w / scale_safe).clamp(-128, 127)
    return q * scale_safe


def compute_calibration_scales(input_tensors, verbose=False):
    """Run torch reference on multiple inputs, compute per-activation-boundary scales.
    Returns dict {output_name: scale (fp32 scalar)}."""
    g = parse_model("models/w600k_r50.onnx")
    init = {t["name"]: t for t in g["initializers"]}
    nodes = g["nodes"]

    # Collect absmax per tensor output name across all calibration inputs
    absmax = {}

    for idx, inp in enumerate(input_tensors):
        tensors = {}
        tensors["input.1"] = torch.from_numpy(inp).float()

        for ni, node in enumerate(nodes):
            op = node["op_type"]; outs = node["outputs"]
            if op == "Conv":
                x = tensors[node["inputs"][0]]
                w = torch.from_numpy(init[node["inputs"][1]]["numpy"].astype(np.float32))
                attrs = {a["name"]: a for a in node["attrs"]}
                stride = attrs["strides"]["ints"][0]
                pad = attrs["pads"]["ints"][0]
                y = F.conv2d(x, w, stride=stride, padding=pad)
            elif op == "BatchNormalization":
                x = tensors[node["inputs"][0]]
                gamma = torch.from_numpy(init[node["inputs"][1]]["numpy"].astype(np.float32))
                beta = torch.from_numpy(init[node["inputs"][2]]["numpy"].astype(np.float32))
                mean = torch.from_numpy(init[node["inputs"][3]]["numpy"].astype(np.float32))
                var = torch.from_numpy(init[node["inputs"][4]]["numpy"].astype(np.float32))
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
                w = torch.from_numpy(init[node["inputs"][1]]["numpy"].astype(np.float32))
                b = None
                if len(node["inputs"]) > 2:
                    bt = init.get(node["inputs"][2])
                    if bt and bt.get("numpy") is not None:
                        b = torch.from_numpy(bt["numpy"].astype(np.float32))
                attrs = {a["name"]: a for a in node["attrs"]}
                transB = attrs.get("transB", {"i": 0})["i"]
                B = w.T if transB else w
                if x.dim() == 2 and B.dim() == 2 and x.shape[-1] != B.shape[0]:
                    if x.shape[-1] == B.shape[-1]: B = B.T
                y = x @ B
                if b is not None: y = y + b
            else:
                continue

            tensors[outs[0]] = y
            # Record absmax
            am = float(y.abs().max().item())
            if outs[0] not in absmax:
                absmax[outs[0]] = am
            else:
                absmax[outs[0]] = max(absmax[outs[0]], am)

    # Convert absmax → scale = absmax / 127
    scales = {name: (am / 127.0 if am > 0 else 1.0) for name, am in absmax.items()}
    if verbose:
        print(f"Calibrated {len(scales)} tensor scales")
    return scales


def run_int8_simulation(input_chw, calib_scales):
    """Run forward with quant-dequant at each activation boundary + per-channel weight quant."""
    g = parse_model("models/w600k_r50.onnx")
    init = {t["name"]: t for t in g["initializers"]}
    nodes = g["nodes"]

    tensors = {}
    # Input: quantize using the input scale (computed from this input's absmax)
    x0 = torch.from_numpy(input_chw).float()
    input_scale = float(x0.abs().max().item()) / 127.0
    tensors["input.1"] = fake_quant_per_tensor(x0, input_scale)

    for ni, node in enumerate(nodes):
        op = node["op_type"]; outs = node["outputs"]

        if op == "Conv":
            x = tensors[node["inputs"][0]]
            w = torch.from_numpy(init[node["inputs"][1]]["numpy"].astype(np.float32))
            w_q = fake_quant_per_channel_weight(w, axis=0)  # per-Cout channel
            attrs = {a["name"]: a for a in node["attrs"]}
            stride = attrs["strides"]["ints"][0]
            pad = attrs["pads"]["ints"][0]
            y = F.conv2d(x, w_q, stride=stride, padding=pad)

        elif op == "BatchNormalization":
            x = tensors[node["inputs"][0]]
            gamma = torch.from_numpy(init[node["inputs"][1]]["numpy"].astype(np.float32))
            beta = torch.from_numpy(init[node["inputs"][2]]["numpy"].astype(np.float32))
            mean = torch.from_numpy(init[node["inputs"][3]]["numpy"].astype(np.float32))
            var = torch.from_numpy(init[node["inputs"][4]]["numpy"].astype(np.float32))
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
            w = torch.from_numpy(init[node["inputs"][1]]["numpy"].astype(np.float32))
            w_q = fake_quant_per_channel_weight(w, axis=0)
            b = None
            if len(node["inputs"]) > 2:
                bt = init.get(node["inputs"][2])
                if bt and bt.get("numpy") is not None:
                    b = torch.from_numpy(bt["numpy"].astype(np.float32))
            attrs = {a["name"]: a for a in node["attrs"]}
            transB = attrs.get("transB", {"i": 0})["i"]
            B = w_q.T if transB else w_q
            if x.dim() == 2 and B.dim() == 2 and x.shape[-1] != B.shape[0]:
                if x.shape[-1] == B.shape[-1]: B = B.T
            y = x @ B
            if b is not None: y = y + b

        # Apply quant-dequant at activation boundary (except the final Gemm output — keep as fp32)
        if op != "Gemm" and outs[0] in calib_scales:
            s = calib_scales[outs[0]]
            y = fake_quant_per_tensor(y, s)

        tensors[outs[0]] = y

    return tensors[nodes[-1]["outputs"][0]].numpy()


def main():
    np.random.seed(42)
    inp = np.random.randn(1, 3, 112, 112).astype(np.float32)

    # ORT ground truth
    sess = ort.InferenceSession("models/w600k_r50.onnx", providers=["CPUExecutionProvider"])
    inp_name = sess.get_inputs()[0].name
    ort_out = sess.run(None, {inp_name: inp})[0].flatten()

    # Calibrate on 5 inputs
    print("Calibrating activation scales on 5 inputs...")
    calib_inputs = [np.random.randn(1, 3, 112, 112).astype(np.float32) for _ in range(5)]
    t0 = time.perf_counter()
    scales = compute_calibration_scales(calib_inputs, verbose=True)
    print(f"Calibration done in {time.perf_counter()-t0:.1f}s")

    # INT8 sim on test input
    print("\nRunning INT8 simulation...")
    t0 = time.perf_counter()
    int8_out = run_int8_simulation(inp, scales).flatten()
    print(f"INT8 sim done in {time.perf_counter()-t0:.1f}s")
    print(f"  norm={np.linalg.norm(int8_out):.4f}  first5={int8_out[:5]}")
    print(f"  ORT first5={ort_out[:5]}")

    sim = cos_sim(ort_out, int8_out)
    print(f"\n==== Cosine similarity ORT vs INT8-sim: {sim:.6f} ====")

    if sim >= 0.95:
        print("EXCELLENT — INT8 loss tolerable, production-ready")
    elif sim >= 0.90:
        print("GOOD — acceptable INT8 loss for real-world nearest-neighbor")
    elif sim >= 0.80:
        print("OK for MVP — calibration could improve")
    elif sim >= 0.7:
        print("MARGINAL — needs better calibration or per-channel activation quant")
    else:
        print("POOR — investigate overflow or scale errors")


if __name__ == "__main__":
    main()
