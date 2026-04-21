"""S42 — percentile-based per-channel calibration.

Instead of absmax across all images (outlier-dominated), we track per-image
per-channel absmax and take the PERCENTILE across images. Env PERCENTILE
(default 99) controls it.

Exposes collect_per_image_absmax(g, inputs) -> dict name -> array of per-image
per-channel absmax shape [n_images, n_channels].
"""
import sys, os, glob, random, time
sys.path.insert(0, '.')
from extract_onnx import parse_model
import numpy as np
import torch
import torch.nn.functional as F


def load_face(path):
    from PIL import Image
    img = Image.open(path).convert("RGB")
    w, h = img.size; s = 150
    left = (w - s) // 2; top = max(0, (h - s) // 2 - 10)
    img = img.crop((left, top, left + s, top + s)).resize((112, 112), Image.BILINEAR)
    arr = (np.asarray(img, dtype=np.float32) - 127.5) / 127.5
    return np.transpose(arr, (2, 0, 1))[None].copy()


def load_lfw_batch(lfw_dir, n, seed):
    paths = sorted(glob.glob(os.path.join(lfw_dir, "**", "*.jpg"), recursive=True))
    random.seed(seed); random.shuffle(paths)
    return [load_face(p) for p in paths[:n]], paths[:n]


def collect_per_image_absmax(g, inputs):
    """For each tensor, track per-image per-channel absmax."""
    init = {t["name"]: t for t in g["initializers"]}
    nodes = g["nodes"]
    result = {}  # name -> list of per-image abs arrays

    def track(name, t):
        arr = t.detach().numpy()
        if arr.ndim == 4:
            absmax = np.max(np.abs(arr), axis=(0, 2, 3)).astype(np.float32)
        elif arr.ndim == 2:
            absmax = np.max(np.abs(arr), axis=0).astype(np.float32)
        else:
            absmax = np.array([float(np.abs(arr).max())], dtype=np.float32)
        if name not in result:
            result[name] = []
        result[name].append(absmax)

    for idx, inp in enumerate(inputs):
        tensors = {"input.1": torch.from_numpy(inp).float()}
        track("input.1", tensors["input.1"])
        for ni, node in enumerate(nodes):
            op = node["op_type"]; outs = node["outputs"]
            if op == "Conv":
                x = tensors[node["inputs"][0]]
                w = torch.from_numpy(init[node["inputs"][1]]["numpy"].astype(np.float32))
                bias = None
                if len(node["inputs"]) > 2 and node["inputs"][2]:
                    bt = init.get(node["inputs"][2])
                    if bt and bt.get("numpy") is not None:
                        bias = torch.from_numpy(bt["numpy"].astype(np.float32))
                attrs = {a["name"]: a for a in node["attrs"]}
                y = F.conv2d(x, w, bias=bias, stride=attrs["strides"]["ints"][0], padding=attrs["pads"]["ints"][0])
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
                w = torch.from_numpy(init[node["inputs"][1]]["numpy"].astype(np.float32))
                b = torch.from_numpy(init[node["inputs"][2]]["numpy"].astype(np.float32)) if len(node["inputs"]) > 2 else None
                attrs = {a["name"]: a for a in node["attrs"]}
                transB = attrs.get("transB", {"i": 0})["i"]
                B = w.T if transB else w
                if x.shape[-1] != B.shape[0] and x.shape[-1] == B.shape[-1]: B = B.T
                y = x @ B
                if b is not None: y = y + b
            else:
                continue
            tensors[outs[0]] = y
            track(outs[0], y)
        if (idx + 1) % 50 == 0:
            print(f"  processed {idx+1}/{len(inputs)}", flush=True)
    # stack each tensor's list into array [n_images, n_channels]
    return {name: np.stack(v, axis=0) for name, v in result.items()}


def build_percentile_scales(per_image_absmax, percentile=99.0, depth_early_percentile=None):
    """Return dict name -> per-channel scale using `percentile` over images.
    If depth_early_percentile is given, tensors in the first half of the graph
    use that percentile instead."""
    names = list(per_image_absmax.keys())
    half = len(names) // 2
    scales = {}
    for i, name in enumerate(names):
        arr = per_image_absmax[name]
        p = depth_early_percentile if (depth_early_percentile is not None and i < half) else percentile
        pc_absmax = np.percentile(arr, p, axis=0).astype(np.float32)
        pc_absmax = np.where(pc_absmax > 0, pc_absmax, 1e-6).astype(np.float32)
        scales[name] = pc_absmax / 127.0
    return scales
