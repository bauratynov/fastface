"""Session 12 — calibrate INT8 activation scales using real LFW faces,
then measure cos-sim of INT8 simulation vs ORT on held-out faces.
"""
import sys
sys.path.insert(0, '.')
from extract_onnx import parse_model
from validate_int8_simulated import (fake_quant_per_tensor, fake_quant_per_channel_weight, cos_sim)
import numpy as np
import torch
import torch.nn.functional as F
import onnxruntime as ort
from PIL import Image
import os, glob, random, time


def load_lfw_preprocessed(lfw_dir, n_images, seed=0):
    """Load n_images from LFW, resize to 112×112, normalize to ArcFace input format.
    Returns [N, 3, 112, 112] float32 in ArcFace-expected range."""
    # Find all .jpg in LFW
    all_paths = sorted(glob.glob(os.path.join(lfw_dir, "**", "*.jpg"), recursive=True))
    random.seed(seed)
    random.shuffle(all_paths)
    paths = all_paths[:n_images]

    batch = np.zeros((n_images, 3, 112, 112), dtype=np.float32)
    for i, p in enumerate(paths):
        img = Image.open(p).convert("RGB")
        # LFW is 250×250, face approximately centered. Center-crop ~150×150 + resize to 112×112
        w, h = img.size
        s = 150
        left = (w - s) // 2
        top = (h - s) // 2 - 10  # face is slightly above center in LFW
        img_crop = img.crop((left, max(0, top), left + s, max(0, top) + s))
        img_112 = img_crop.resize((112, 112), Image.BILINEAR)
        arr = np.asarray(img_112, dtype=np.float32)  # [112, 112, 3] RGB [0, 255]
        # ArcFace preprocessing: normalize to [-1, 1]: (pixel - 127.5) / 127.5
        arr = (arr - 127.5) / 127.5
        # NHWC → NCHW
        arr = np.transpose(arr, (2, 0, 1))
        batch[i] = arr
    return batch, paths


def compute_activation_scales_via_ort_hooks(onnx_path, inputs):
    """Capture per-layer activation tensors by running ORT with intermediate outputs exposed.
    Workaround since we can't modify the ONNX graph: we rebuild layer-by-layer in torch
    using the same weights, matching ORT exactly (S10 verified cos-sim=1.0)."""
    g = parse_model(onnx_path)
    init = {t["name"]: t for t in g["initializers"]}
    nodes = g["nodes"]

    # absmax per tensor output name, max across all inputs
    absmax = {}

    for idx in range(inputs.shape[0]):
        inp = inputs[idx:idx+1]
        tensors = {"input.1": torch.from_numpy(inp).float()}
        for node in nodes:
            op = node["op_type"]; outs = node["outputs"]
            if op == "Conv":
                x = tensors[node["inputs"][0]]
                w = torch.from_numpy(init[node["inputs"][1]]["numpy"].astype(np.float32))
                attrs = {a["name"]: a for a in node["attrs"]}
                y = F.conv2d(x, w, stride=attrs["strides"]["ints"][0], padding=attrs["pads"]["ints"][0])
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
            am = float(y.abs().max().item())
            if outs[0] not in absmax:
                absmax[outs[0]] = am
            else:
                absmax[outs[0]] = max(absmax[outs[0]], am)
        if (idx + 1) % 50 == 0:
            print(f"  processed {idx+1}/{inputs.shape[0]}", flush=True)

    scales = {name: (am / 127.0 if am > 0 else 1.0) for name, am in absmax.items()}
    return scales


def run_int8_with_scales(input_chw, scales):
    """INT8 simulation with calibrated scales. Same as validate_int8_simulated but scales passed in."""
    g = parse_model("models/w600k_r50.onnx")
    init = {t["name"]: t for t in g["initializers"]}
    nodes = g["nodes"]
    tensors = {}
    x0 = torch.from_numpy(input_chw).float()
    tensors["input.1"] = fake_quant_per_tensor(x0, float(x0.abs().max()) / 127.0)

    for ni, node in enumerate(nodes):
        op = node["op_type"]; outs = node["outputs"]
        if op == "Conv":
            x = tensors[node["inputs"][0]]
            w = torch.from_numpy(init[node["inputs"][1]]["numpy"].astype(np.float32))
            w_q = fake_quant_per_channel_weight(w, axis=0)
            attrs = {a["name"]: a for a in node["attrs"]}
            y = F.conv2d(x, w_q, stride=attrs["strides"]["ints"][0], padding=attrs["pads"]["ints"][0])
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
            b = torch.from_numpy(init[node["inputs"][2]]["numpy"].astype(np.float32)) if len(node["inputs"]) > 2 else None
            attrs = {a["name"]: a for a in node["attrs"]}
            transB = attrs.get("transB", {"i": 0})["i"]
            B = w_q.T if transB else w_q
            if x.shape[-1] != B.shape[0] and x.shape[-1] == B.shape[-1]: B = B.T
            y = x @ B
            if b is not None: y = y + b
        if op != "Gemm" and outs[0] in scales:
            y = fake_quant_per_tensor(y, scales[outs[0]])
        tensors[outs[0]] = y

    return tensors[nodes[-1]["outputs"][0]].numpy()


def main():
    LFW_DIR = "data/lfw"
    N_CALIB = 500
    N_TEST = 20

    print(f"Loading {N_CALIB + N_TEST} LFW faces...")
    all_faces, paths = load_lfw_preprocessed(LFW_DIR, N_CALIB + N_TEST, seed=0)
    calib = all_faces[:N_CALIB]
    test = all_faces[N_CALIB:]
    print(f"  Calibration: {calib.shape}")
    print(f"  Test (held-out): {test.shape}")

    print(f"\nCalibrating activation scales on {N_CALIB} LFW faces...")
    t0 = time.perf_counter()
    scales = compute_activation_scales_via_ort_hooks("models/w600k_r50.onnx", calib)
    t_calib = time.perf_counter() - t0
    print(f"Calibration done: {len(scales)} tensor scales, {t_calib:.1f}s")

    # ORT reference on held-out
    sess = ort.InferenceSession("models/w600k_r50.onnx", providers=["CPUExecutionProvider"])
    inp_name = sess.get_inputs()[0].name

    print(f"\nEvaluating on {N_TEST} held-out faces...")
    cos_sims = []
    for i in range(N_TEST):
        x = test[i:i+1]
        ort_out = sess.run(None, {inp_name: x})[0].flatten()
        int8_out = run_int8_with_scales(x, scales).flatten()
        sim = cos_sim(ort_out, int8_out)
        cos_sims.append(sim)
        if i < 5:
            print(f"  Face {i}: cos-sim = {sim:.4f}")

    mean_sim = np.mean(cos_sims)
    min_sim = np.min(cos_sims)
    max_sim = np.max(cos_sims)
    print(f"\n==== Cosine similarity ORT vs INT8 (calibrated on LFW) ====")
    print(f"Mean: {mean_sim:.4f}")
    print(f"Min:  {min_sim:.4f}")
    print(f"Max:  {max_sim:.4f}")
    print(f"N test faces: {N_TEST}")

    if mean_sim >= 0.95:
        print("\nEXCELLENT — production-ready INT8 quality")
    elif mean_sim >= 0.85:
        print("\nGOOD — MVP acceptable for real-world face matching")
    elif mean_sim >= 0.7:
        print("\nOK for nearest-neighbor use cases")
    else:
        print("\nSTILL POOR — investigate per-layer")

    # Save scales for C binary to use
    import struct
    with open("models/lfw_calib_scales.bin", "wb") as f:
        f.write(b"SCAL")
        f.write(struct.pack("<I", len(scales)))
        for name, s in scales.items():
            name_b = name.encode()
            f.write(struct.pack("<H", len(name_b)))
            f.write(name_b)
            f.write(struct.pack("<f", s))
    print(f"\nSaved {len(scales)} scales to models/lfw_calib_scales.bin")


if __name__ == "__main__":
    main()
