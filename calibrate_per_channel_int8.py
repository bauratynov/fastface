"""Session 21 — per-channel INT8 activation quantization, symmetric and asymmetric.

Hypothesis: per-tensor INT8 fails because PReLU creates per-channel
asymmetric distributions that a single scalar scale can't capture.
Per-channel asymmetric quant (scale and zero_point per channel) gives
each channel its own dynamic range.

Test in Python simulation; if cos-sim ≥ 0.95 we port to C.
"""
import sys, os, glob, random, time, argparse
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


def load_lfw_batch(lfw_dir, n, seed):
    paths = sorted(glob.glob(os.path.join(lfw_dir, "**", "*.jpg"), recursive=True))
    random.seed(seed); random.shuffle(paths)
    return [load_face(p) for p in paths[:n]], paths[:n]


# ---------- Collect per-channel min/max over calibration set ----------
def collect_ranges(g, inputs):
    """For each tensor, track per-channel min and max across calibration inputs.
    Returns dict name → (min_array, max_array). For 4D tensors, C axis is 1."""
    init = {t["name"]: t for t in g["initializers"]}
    nodes = g["nodes"]
    ranges = {}  # name -> (min, max) numpy

    def track(name, t):
        arr = t.detach().numpy()
        if arr.ndim == 4:
            axes = (0, 2, 3)
            tmin = arr.min(axis=axes)
            tmax = arr.max(axis=axes)
        else:
            axes = tuple(i for i in range(arr.ndim) if i != arr.ndim - 1)
            if len(axes) == 0:
                tmin = arr.min(); tmax = arr.max()
                tmin = np.array([tmin]); tmax = np.array([tmax])
            else:
                tmin = arr.min(axis=axes)
                tmax = arr.max(axis=axes)
        if name in ranges:
            prev_min, prev_max = ranges[name]
            ranges[name] = (np.minimum(prev_min, tmin), np.maximum(prev_max, tmax))
        else:
            ranges[name] = (tmin.copy(), tmax.copy())

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
        if (idx + 1) % 10 == 0:
            print(f"  processed {idx+1}/{len(inputs)}", flush=True)
    return ranges


def fake_quant_per_channel_sym(x, scale_c):
    """Per-channel symmetric. scale_c is [C] array broadcasting on axis 1 for 4D, last for 2D."""
    if x.ndim == 4:
        s = torch.tensor(scale_c).view(1, -1, 1, 1)
    elif x.ndim == 2:
        s = torch.tensor(scale_c).view(1, -1)
    else:
        s = torch.tensor(scale_c)
    q = torch.round(x / (s + 1e-9)).clamp(-127, 127)
    return q * s


def fake_quant_per_channel_asym(x, scale_c, zero_c):
    """Per-channel asymmetric. scale_c, zero_c per channel."""
    if x.ndim == 4:
        s = torch.tensor(scale_c).view(1, -1, 1, 1)
        z = torch.tensor(zero_c).view(1, -1, 1, 1)
    elif x.ndim == 2:
        s = torch.tensor(scale_c).view(1, -1)
        z = torch.tensor(zero_c).view(1, -1)
    else:
        s = torch.tensor(scale_c); z = torch.tensor(zero_c)
    # Asymmetric: q = clip(round((x - z) / s), 0, 255), dequant = q * s + z
    q = torch.round((x - z) / (s + 1e-9)).clamp(-128, 127)
    return q * s + z


def fake_quant_weight_per_oc(w):
    w_np = w.numpy()
    axes = tuple(range(1, w_np.ndim))
    absmax = np.max(np.abs(w_np), axis=axes, keepdims=True) + 1e-9
    scale = absmax / 127.0
    q = np.round(w_np / scale).clip(-127, 127)
    return torch.from_numpy((q * scale).astype(np.float32))


def compute_scales(ranges, mode="asym"):
    """mode: 'sym_channel' (absmax per channel), 'asym_channel' (zero_point per channel),
    'sym_tensor' (absmax across tensor), 'asym_tensor' (min/max across tensor)."""
    scales = {}
    zeros = {}
    for name, (tmin, tmax) in ranges.items():
        if mode == "sym_channel":
            absmax = np.maximum(np.abs(tmin), np.abs(tmax))
            scales[name] = (absmax / 127.0).astype(np.float32)
            zeros[name] = np.zeros_like(scales[name])
        elif mode == "asym_channel":
            # Asymmetric: midpoint as zero_point, range/255 as scale
            zp = ((tmin + tmax) * 0.5).astype(np.float32)
            rng = ((tmax - tmin) / 255.0).astype(np.float32)
            scales[name] = rng
            zeros[name] = zp
        elif mode == "sym_tensor":
            am = float(max(abs(tmin.min()), abs(tmax.max())))
            scales[name] = np.full(tmin.shape, am / 127.0, dtype=np.float32)
            zeros[name] = np.zeros_like(scales[name])
    return scales, zeros


def run_int8_simulation(g, input_chw, scales, zeros, mode="asym_channel"):
    """Walk graph applying per-channel fake-quant at activation boundaries."""
    init = {t["name"]: t for t in g["initializers"]}
    nodes = g["nodes"]
    tensors = {}

    def qapply(t, name):
        if name not in scales: return t
        s = scales[name]; z = zeros[name]
        # If scales is 1D length 1 and tensor is 4D → broadcast across channel anyway
        if mode == "asym_channel":
            return fake_quant_per_channel_asym(t, s, z)
        else:
            return fake_quant_per_channel_sym(t, s)

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

    return tensors[nodes[-1]["outputs"][0]].numpy().flatten()


def cos_sim(a, b):
    a, b = a.flatten(), b.flatten()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["sym_channel", "asym_channel", "sym_tensor"], default="asym_channel")
    ap.add_argument("--n_calib", type=int, default=20)
    ap.add_argument("--n_test", type=int, default=10)
    args = ap.parse_args()

    print(f"=== mode={args.mode}  n_calib={args.n_calib}  n_test={args.n_test} ===", flush=True)

    calib_inputs, _ = load_lfw_batch("data/lfw", args.n_calib, seed=1)
    test_inputs,  _ = load_lfw_batch("data/lfw", args.n_test,  seed=999)

    g = parse_model("models/w600k_r50.onnx")

    print(f"Collecting per-channel ranges on {args.n_calib} faces...", flush=True)
    t0 = time.perf_counter()
    ranges = collect_ranges(g, calib_inputs)
    print(f"  done in {time.perf_counter()-t0:.1f}s ({len(ranges)} tensors)", flush=True)

    scales, zeros = compute_scales(ranges, mode=args.mode)

    sess = ort.InferenceSession("models/w600k_r50.onnx", providers=["CPUExecutionProvider"])
    inp_name = sess.get_inputs()[0].name

    print(f"\nSimulating INT8 ({args.mode}) on {args.n_test} test faces...", flush=True)
    cs = []
    for i, x in enumerate(test_inputs):
        ort_out = sess.run(None, {inp_name: x})[0].flatten()
        int8_out = run_int8_simulation(g, x, scales, zeros, mode=args.mode)
        c = cos_sim(ort_out, int8_out)
        cs.append(c)
        if i < 5: print(f"  face {i}: cos-sim = {c:.4f}")

    print(f"\n=== RESULT: {args.mode} ===")
    print(f"mean cos-sim = {np.mean(cs):.6f}")
    print(f"min / max    = {np.min(cs):.6f} / {np.max(cs):.6f}")

    if np.mean(cs) >= 0.99:
        verdict = "EXCELLENT — near-ideal, direct C port viable"
    elif np.mean(cs) >= 0.95:
        verdict = "SHIP — production INT8 quality"
    elif np.mean(cs) >= 0.85:
        verdict = "GOOD — MVP"
    elif np.mean(cs) >= 0.70:
        verdict = "OK — NN search only"
    else:
        verdict = "FAIL — this mode insufficient"
    print(f"verdict: {verdict}")


if __name__ == "__main__":
    main()
