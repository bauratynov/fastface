"""Test: runtime per-channel absmax activation quant (no calibration).
Hypothesis: runtime adapts to each input's distribution exactly → higher cos-sim.
"""
import sys, os, glob, random
sys.path.insert(0, '.')
from extract_onnx import parse_model
import numpy as np
import torch
import torch.nn.functional as F
import onnxruntime as ort
from PIL import Image


def load_face(path):
    img = Image.open(path).convert("RGB")
    w, h = img.size; s = 150
    left = (w - s) // 2; top = max(0, (h - s) // 2 - 10)
    img = img.crop((left, top, left + s, top + s)).resize((112, 112), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32)
    arr = (arr - 127.5) / 127.5
    return np.transpose(arr, (2, 0, 1))[None].copy()


def cos_sim(a, b):
    a, b = a.flatten(), b.flatten()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def runtime_per_channel_quant(x):
    """Per-channel symmetric fake-quant, runtime absmax."""
    if x.ndim == 4:
        axes = (0, 2, 3); view = (1, -1, 1, 1)
    elif x.ndim == 2:
        axes = (0,); view = (1, -1)
    else:
        return x
    absmax = x.abs().amax(dim=axes, keepdim=True) + 1e-9
    scale = absmax / 127.0
    q = torch.round(x / scale).clamp(-127, 127)
    return q * scale


def fake_quant_weight_per_oc(w):
    w_np = w.numpy()
    axes = tuple(range(1, w_np.ndim))
    absmax = np.max(np.abs(w_np), axis=axes, keepdims=True) + 1e-9
    scale = absmax / 127.0
    q = np.round(w_np / scale).clip(-127, 127)
    return torch.from_numpy((q * scale).astype(np.float32))


def run_sim(g, input_chw):
    init = {t["name"]: t for t in g["initializers"]}
    nodes = g["nodes"]
    tensors = {"input.1": runtime_per_channel_quant(torch.from_numpy(input_chw).float())}
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
            y = runtime_per_channel_quant(y)
        tensors[outs[0]] = y
    return tensors[nodes[-1]["outputs"][0]].numpy().flatten()


def main():
    g = parse_model("models/w600k_r50.onnx")
    paths = sorted(glob.glob('data/lfw/**/*.jpg', recursive=True))
    random.seed(42); random.shuffle(paths)
    sess = ort.InferenceSession('models/w600k_r50.onnx', providers=['CPUExecutionProvider'])
    cs = []
    for p in paths[:10]:
        x = load_face(p)
        ort_out = sess.run(None, {sess.get_inputs()[0].name: x})[0].flatten()
        int8_out = run_sim(g, x)
        c = cos_sim(ort_out, int8_out)
        cs.append(c)
        print(f"  {os.path.basename(p):<30} cos-sim = {c:.4f}")
    import statistics
    print(f"\nmean={statistics.mean(cs):.6f}  min={min(cs):.6f}  max={max(cs):.6f}")


if __name__ == "__main__":
    main()
