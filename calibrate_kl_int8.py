"""Session 20 — KL-divergence INT8 calibration (Migacz 2017 algorithm).

Pipeline:
 1. Collect per-tensor activation histograms (2048 bins) over 500 LFW faces.
 2. For each tensor, search threshold T in [128, 2048):
      - Clip histogram to [-T, T], sum tail into boundary bins.
      - Quantize to 128 int8 bins; expand back to T-bin reference scale.
      - Compute KL(reference || quantized).
    Pick T* = argmin KL. scale_tensor = T* × bin_width / 127.
 3. Quantize per-channel weights (absmax per output channel).
 4. Run simulated int8 forward: at each op boundary, fake-quant activations
    with the calibrated scale.
 5. Measure cos-sim vs ORT on held-out 20 faces.

If cos-sim ≥ 0.95 → proceed to C implementation. Else: refine the calibration.
"""
import sys, os, glob, random, time
sys.path.insert(0, '.')
from extract_onnx import parse_model
import numpy as np
import torch
import torch.nn.functional as F
import onnxruntime as ort
from PIL import Image


# ---------- Data loading ----------
def load_face(path):
    img = Image.open(path).convert("RGB")
    w, h = img.size; s = 150
    left = (w - s) // 2; top = max(0, (h - s) // 2 - 10)
    img = img.crop((left, top, left + s, top + s)).resize((112, 112), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32)
    arr = (arr - 127.5) / 127.5
    return np.transpose(arr, (2, 0, 1))[None].copy()  # [1, 3, H, W]


def load_lfw_batch(lfw_dir, n, seed):
    paths = sorted(glob.glob(os.path.join(lfw_dir, "**", "*.jpg"), recursive=True))
    random.seed(seed); random.shuffle(paths)
    return [load_face(p) for p in paths[:n]], paths[:n]


# ---------- Histogram + KL search ----------
BINS = 2048
QBINS = 128  # int8 bins per side; symmetric total range is [-QBINS, QBINS)

def update_histogram(hist_state, tensor):
    """Accumulate absolute-value histogram of `tensor` in [0, absmax).
    Returns updated (bins, absmax) tuple."""
    bins, cur_max = hist_state
    arr = np.abs(tensor.flatten().astype(np.float32))
    tmax = float(arr.max())
    if tmax == 0:
        return bins, cur_max
    if cur_max == 0 or tmax > cur_max:
        # Scale existing bins down: new_bin_width = tmax / BINS.
        new_bins = np.zeros(BINS, dtype=np.float64)
        if cur_max > 0 and bins.sum() > 0:
            # Rebin the existing histogram.
            old_edges = np.linspace(0, cur_max, BINS + 1)
            for i in range(BINS):
                if bins[i] == 0: continue
                center = 0.5 * (old_edges[i] + old_edges[i + 1])
                new_idx = min(int(center / tmax * BINS), BINS - 1)
                new_bins[new_idx] += bins[i]
        bins = new_bins
        cur_max = tmax
    # Add new tensor to histogram
    idx = np.minimum((arr / cur_max * BINS).astype(np.int32), BINS - 1)
    np.add.at(bins, idx, 1)
    return bins, cur_max


def kl_optimal_threshold(hist, absmax):
    """Migacz 2017: find T in [QBINS, BINS) minimizing KL(P || Q)."""
    total = hist.sum()
    if total == 0 or absmax == 0:
        return absmax
    bin_width = absmax / BINS
    best_T, best_kl = BINS, float('inf')

    for T in range(QBINS, BINS + 1):
        # Reference distribution P: clip to [0, T), put tail mass in the T-1 bin.
        P = hist[:T].copy()
        P[T - 1] += hist[T:].sum()
        p_sum = P.sum()
        if p_sum == 0: continue
        P = P / p_sum

        # Quantized distribution Q: group T bins into QBINS groups, then expand back.
        group_size = T / QBINS
        Q_expanded = np.zeros(T, dtype=np.float64)
        for qi in range(QBINS):
            lo = int(np.floor(qi * group_size))
            hi = int(np.floor((qi + 1) * group_size))
            if qi == QBINS - 1: hi = T
            block = hist[lo:hi]
            mass = block.sum()
            nonzero = (block > 0).sum()
            if nonzero == 0: continue
            avg = mass / nonzero
            Q_expanded[lo:hi] = np.where(block > 0, avg, 0.0)
        # Add tail mass from hist[T:] into last expanded block only if Q has nonzero there
        q_sum = Q_expanded.sum()
        if q_sum == 0: continue
        Q_expanded = Q_expanded / q_sum

        # KL only over nonzero P entries; smooth Q by eps.
        eps = 1e-9
        mask = P > 0
        kl = float(np.sum(P[mask] * np.log((P[mask] + eps) / (Q_expanded[mask] + eps))))
        if kl < best_kl:
            best_kl, best_T = kl, T

    return best_T * bin_width


# ---------- Torch reconstruction with hooks for histogram ----------
def build_tensors_and_collect(g, inputs):
    """Run torch-by-name forward collecting per-tensor activation histograms
    across all `inputs`. Returns dict {tensor_name: (hist, absmax)}."""
    init = {t["name"]: t for t in g["initializers"]}
    nodes = g["nodes"]
    hists = {}  # name -> (hist_array, absmax)

    def track(name, t):
        arr = t.detach().numpy()
        hists[name] = update_histogram(hists.get(name, (np.zeros(BINS, dtype=np.float64), 0.0)), arr)

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
    return hists


# ---------- Simulated int8 forward ----------
def fake_quant(x, scale):
    """Per-tensor symmetric fake quantization: q = clamp(round(x/scale), -127, 127), then dequantize."""
    if scale <= 0:
        return x
    q = torch.round(x / scale).clamp(-127, 127)
    return q * scale


def fake_quant_weight_per_oc(w):
    """Per-output-channel weight quant (Conv: axis 0, Gemm: axis 0)."""
    w_np = w.numpy()
    axes = tuple(range(1, w_np.ndim))
    absmax = np.max(np.abs(w_np), axis=axes, keepdims=True) + 1e-9
    scale = absmax / 127.0
    q = np.round(w_np / scale).clip(-127, 127)
    return torch.from_numpy((q * scale).astype(np.float32))


def run_int8_simulation(g, input_chw, scales):
    """Walk the graph in fp32 but apply fake-quant at each activation boundary
    using `scales` dict (tensor_name -> scale)."""
    init = {t["name"]: t for t in g["initializers"]}
    nodes = g["nodes"]
    tensors = {}
    tensors["input.1"] = fake_quant(torch.from_numpy(input_chw).float(), scales.get("input.1", 0))

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

        # Apply activation fake-quant using calibrated scale
        s = scales.get(outs[0], 0)
        if s > 0 and op != "Flatten":  # Flatten just reshapes
            y = fake_quant(y, s)
        tensors[outs[0]] = y

    last_out = nodes[-1]["outputs"][0]
    return tensors[last_out].numpy().flatten()


def cos_sim(a, b):
    a, b = a.flatten(), b.flatten()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def main():
    N_CALIB = int(os.environ.get("N_CALIB", "50"))  # small for quick test; bump to 500 for real
    N_TEST = 10

    print(f"=== Loading {N_CALIB} calibration + {N_TEST} test LFW faces ===", flush=True)
    calib_inputs, _ = load_lfw_batch("data/lfw", N_CALIB, seed=1)
    test_inputs,  _ = load_lfw_batch("data/lfw", N_TEST, seed=999)

    print(f"=== Collecting activation histograms ({N_CALIB} faces) ===", flush=True)
    g = parse_model("models/w600k_r50.onnx")
    t0 = time.perf_counter()
    hists = build_tensors_and_collect(g, calib_inputs)
    print(f"  histograms collected in {time.perf_counter()-t0:.1f}s  ({len(hists)} tensors)", flush=True)

    print("=== Computing KL-optimal thresholds ===", flush=True)
    t0 = time.perf_counter()
    scales = {}
    for name, (h, am) in hists.items():
        scales[name] = kl_optimal_threshold(h, am)
    print(f"  KL search done in {time.perf_counter()-t0:.1f}s", flush=True)

    # Diagnostics
    for k in list(hists.keys())[:5] + list(hists.keys())[-5:]:
        h, am = hists[k]
        print(f"  {k:<12}  absmax={am:8.3f}  scale={scales[k]:8.4f}  ratio={scales[k]/am if am>0 else 0:.3f}")

    print("\n=== Simulating INT8 on 10 test faces ===", flush=True)
    sess = ort.InferenceSession("models/w600k_r50.onnx", providers=["CPUExecutionProvider"])
    inp_name = sess.get_inputs()[0].name
    cs = []
    for i, x in enumerate(test_inputs):
        ort_out = sess.run(None, {inp_name: x})[0].flatten()
        int8_out = run_int8_simulation(g, x, scales)
        c = cos_sim(ort_out, int8_out)
        cs.append(c)
        print(f"  face {i}: cos-sim = {c:.4f}")

    print(f"\n=== RESULT ===")
    print(f"mean cos-sim (KL-calibrated INT8 sim):  {np.mean(cs):.6f}")
    print(f"min / max:                              {np.min(cs):.6f} / {np.max(cs):.6f}")

    if np.mean(cs) >= 0.95:
        print("EXCELLENT — ship-ready INT8 quality, proceed to C implementation.")
    elif np.mean(cs) >= 0.85:
        print("GOOD — production MVP quality.")
    elif np.mean(cs) >= 0.70:
        print("OK — useful for NN-search but not identity verification.")
    else:
        print("NEEDS WORK — naive per-tensor KL isn't enough; try per-channel activation quant, mixed precision, or QAT.")

    # Save scales for future use
    import struct
    with open("models/kl_scales.bin", "wb") as f:
        f.write(b"KLSC")
        f.write(struct.pack("<I", len(scales)))
        for name, s in scales.items():
            nb = name.encode()
            f.write(struct.pack("<H", len(nb)))
            f.write(nb)
            f.write(struct.pack("<f", s))
    print(f"\nSaved {len(scales)} scales to models/kl_scales.bin")


if __name__ == "__main__":
    main()
