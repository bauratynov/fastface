"""S94 oracle: full INT8 sim EXCEPT final Gemm stays FP32.
Measures theoretical upper-bound on cos-sim if we could eliminate Gemm
quantization error (e.g., via INT16 matvec).
"""
import sys, numpy as np, time, glob, random
sys.path.insert(0, '.')
import torch
import torch.nn.functional as F
import onnxruntime as ort
from PIL import Image
from extract_onnx import parse_model
from calibrate_per_channel_int8 import (
    collect_ranges, load_lfw_batch, fake_quant_per_channel_sym, fake_quant_weight_per_oc,
)


def load_face(path):
    img = Image.open(path).convert("RGB")
    w, h = img.size; s = 150
    img = img.crop(((w-s)//2, max(0,(h-s)//2-10), (w-s)//2+s, max(0,(h-s)//2-10)+s)).resize((112,112), Image.BILINEAR)
    arr = (np.asarray(img, dtype=np.float32) - 127.5) / 127.5
    return np.transpose(arr, (2,0,1))[None].copy()


def run_sim(g, input_chw, scales, skip_gemm_quant=False):
    init = {t["name"]: t for t in g["initializers"]}
    nodes = g["nodes"]
    tensors = {}

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
            # Oracle: use FP32 weight instead of per-OC quantized
            w_q = w_raw if skip_gemm_quant else fake_quant_weight_per_oc(w_raw)
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
    return tensors[nodes[-1]["outputs"][0]].numpy().flatten()


def main():
    n_test = 20
    calib_inputs, _ = load_lfw_batch("data/lfw", 100, seed=1)
    test_inputs, _  = load_lfw_batch("data/lfw", n_test, seed=7777)
    g = parse_model("models/w600k_r50.onnx")

    print(f"Calibrating...", flush=True)
    ranges = collect_ranges(g, calib_inputs)
    scales = {name: np.maximum(np.abs(mn), np.abs(mx)).astype(np.float32) / 127.0 + 1e-9
              for name, (mn, mx) in ranges.items()}

    sess = ort.InferenceSession("models/w600k_r50.onnx", providers=["CPUExecutionProvider"])
    fp32 = [sess.run(None, {sess.get_inputs()[0].name: x})[0].flatten() for x in test_inputs]

    # Full INT8
    int8_all = [run_sim(g, x, scales, skip_gemm_quant=False) for x in test_inputs]
    # Oracle: all INT8 EXCEPT Gemm = FP32 weights
    int8_fp32gemm = [run_sim(g, x, scales, skip_gemm_quant=True) for x in test_inputs]

    def cs(a, b): return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
    cs_all = [cs(f, i) for f, i in zip(fp32, int8_all)]
    cs_oracle = [cs(f, i) for f, i in zip(fp32, int8_fp32gemm)]
    print(f"\n=== Oracle test on {n_test} faces ===")
    print(f"  all INT8:      mean {np.mean(cs_all):.5f}  min {np.min(cs_all):.5f}")
    print(f"  Gemm FP32:     mean {np.mean(cs_oracle):.5f}  min {np.min(cs_oracle):.5f}")
    print(f"  delta (oracle - all): {np.mean(cs_oracle)-np.mean(cs_all):+.5f}")


if __name__ == "__main__":
    main()
