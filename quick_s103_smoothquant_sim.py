"""S103 -- SmoothQuant sim (alpha-grid).

For each Conv that takes a BN- or PReLU-produced activation as input
(i.e. NOT directly after an Add, where we cannot absorb the smoothing
scale into an affine parameter), compute per-input-channel smoothing:

    s_c = max(|X_c|)^alpha / max(|W_:,c|)^(1-alpha),  alpha in [0, 1]

then quantize X/s and W*s. Mathematically the Conv output is unchanged,
but X/s has a tighter per-channel range (favours activation quant) while
W*s has slightly more outlier-y per-input-channel range (weights are
already per-OC quantized so this hurts less).

We apply this INSIDE the Python per-channel sim only. If the sim delta
vs baseline exceeds |delta|_p95 = 0.006 from S102, it is worth plumbing
into production (fold scale into the preceding BN's gamma/beta).

Sweeps alpha in {0.0 (no-op), 0.25, 0.5 (canonical), 0.75}.
"""
import os, sys, time, numpy as np
sys.path.insert(0, '.')
import torch
import torch.nn.functional as F
import onnxruntime as ort
from extract_onnx import parse_model
from calibrate_per_channel_int8 import load_lfw_batch, fake_quant_per_channel_sym, fake_quant_weight_per_oc
from calibrate_percentile_int8 import collect_per_image_absmax, build_percentile_scales
from calibrate_include_princess import load_lfw_with_princess


def compute_per_in_channel_abs(g, calib_inputs):
    """For each Conv input tensor, collect per-input-channel absmax over calib set.
    Returns dict tensor_name -> array [C_in] of absmax.
    """
    init = {t["name"]: t for t in g["initializers"]}
    nodes = g["nodes"]
    # which tensor names are Conv inputs
    conv_inputs = set()
    for node in nodes:
        if node["op_type"] in ("Conv", "Gemm"):
            conv_inputs.add(node["inputs"][0])
    acc = {name: None for name in conv_inputs}
    count = {name: 0 for name in conv_inputs}

    for idx, inp in enumerate(calib_inputs):
        tensors = {"input.1": torch.from_numpy(inp).float()}
        for node in nodes:
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
            if outs[0] in acc:
                if y.ndim == 4:
                    a = y.detach().abs().amax(dim=(0,2,3)).numpy()
                elif y.ndim == 2:
                    a = y.detach().abs().amax(dim=0).numpy()
                else:
                    a = np.array([float(y.detach().abs().max())])
                acc[outs[0]] = a if acc[outs[0]] is None else np.maximum(acc[outs[0]], a)
                count[outs[0]] += 1
        if (idx+1) % 50 == 0:
            print(f"  collected per-ic absmax {idx+1}/{len(calib_inputs)}", flush=True)
    return acc


def build_smooth_factors(g, per_ic_absmax, alpha):
    """For each Conv whose input is from BN or PReLU, compute per-input-channel
    smoothing factor s_c. Returns dict node_idx -> np.array [C_in]."""
    init = {t["name"]: t for t in g["initializers"]}
    nodes = g["nodes"]
    producer = {o: i for i, n in enumerate(nodes) for o in n["outputs"]}

    factors = {}
    for ni, node in enumerate(nodes):
        if node["op_type"] not in ("Conv", "Gemm"):
            continue
        in_name = node["inputs"][0]
        pi = producer.get(in_name, -1)
        if pi < 0:
            # graph input -- skip (alpha=0 for stem)
            continue
        prod_op = nodes[pi]["op_type"]
        if prod_op not in ("BatchNormalization", "PRelu"):
            continue  # can't absorb scale into Add output
        act_absmax = per_ic_absmax.get(in_name, None)
        if act_absmax is None:
            continue
        # weight per-input-channel absmax
        w = init[node["inputs"][1]]["numpy"].astype(np.float32)
        if node["op_type"] == "Conv":
            # w shape [Cout, Cin, kH, kW]; per Cin
            w_abs = np.max(np.abs(w), axis=(0, 2, 3))
        else:  # Gemm
            # w shape [N, K] or [K, N] depending on transB; we need per K (input features)
            # If transB=1, weight is (N, K) and input K matches last dim
            attrs = {a["name"]: a for a in node["attrs"]}
            transB = attrs.get("transB", {"i": 0})["i"]
            if transB:
                w_abs = np.max(np.abs(w), axis=0)  # over N -> leaves K
            else:
                w_abs = np.max(np.abs(w), axis=1)  # over N -> leaves K
        # guard
        act_a = act_absmax.clip(min=1e-9)
        w_a = w_abs.clip(min=1e-9)
        s = (act_a ** alpha) / (w_a ** (1.0 - alpha))
        # clamp s to avoid extreme rescaling
        s = np.clip(s, 1e-3, 1e3).astype(np.float32)
        factors[ni] = s
    return factors


def run_sim_smooth(g, input_chw, act_scales, smooth_factors):
    init = {t["name"]: t for t in g["initializers"]}
    nodes = g["nodes"]
    tensors = {}

    def qapply(t, name):
        if name not in act_scales: return t
        return fake_quant_per_channel_sym(t, act_scales[name])

    tensors["input.1"] = qapply(torch.from_numpy(input_chw).float(), "input.1")
    for ni, node in enumerate(nodes):
        op = node["op_type"]; outs = node["outputs"]
        if op == "Conv":
            x = tensors[node["inputs"][0]]
            w_raw = torch.from_numpy(init[node["inputs"][1]]["numpy"].astype(np.float32))
            s = smooth_factors.get(ni)
            if s is not None:
                s_t = torch.from_numpy(s)
                # scale activation by 1/s per-input-channel
                x = x / s_t.view(1, -1, 1, 1)
                # scale weight by s per-input-channel
                w_raw = w_raw * s_t.view(1, -1, 1, 1)
            w_q = fake_quant_weight_per_oc(w_raw)
            bias = None
            if len(node["inputs"]) > 2 and node["inputs"][2]:
                bt = init.get(node["inputs"][2])
                if bt and bt.get("numpy") is not None:
                    bias = torch.from_numpy(bt["numpy"].astype(np.float32))
            attrs = {a["name"]: a for a in node["attrs"]}
            y = F.conv2d(x, w_q, bias=bias, stride=attrs["strides"]["ints"][0],
                         padding=attrs["pads"]["ints"][0])
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
            s = smooth_factors.get(ni)
            if s is not None:
                s_t = torch.from_numpy(s)
                # Gemm: scale last dim of x by 1/s, scale weight input-axis by s
                x = x / s_t.view(1, -1)
                attrs = {a["name"]: a for a in node["attrs"]}
                transB = attrs.get("transB", {"i": 0})["i"]
                if transB:
                    w_raw = w_raw * s_t.view(1, -1)  # (N, K) * (1, K)
                else:
                    w_raw = w_raw * s_t.view(-1, 1)  # (K, N) * (K, 1)
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


def cos(a, b): return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def main():
    n_calib   = int(os.environ.get("N_CALIB", "200"))
    n_test    = int(os.environ.get("N_TEST",  "100"))
    pct_late  = float(os.environ.get("PCT_LATE", "99.9"))
    seed_test = int(os.environ.get("SEED_TEST",  "7777"))

    alpha_grid = [0.0, 0.25, 0.5, 0.75]  # 0.0 = no smoothing baseline

    g = parse_model("models/w600k_r50.onnx")
    print(f"S103 SmoothQuant alpha sweep  N_CALIB={n_calib} N_TEST={n_test}")

    print(f"\n1) calib (200 WITH_PRINCESS)...", flush=True)
    calib_inputs, _ = load_lfw_with_princess("data/lfw", n_calib, seed=1)
    print(f"2) test ({n_test} seed={seed_test})...", flush=True)
    test_inputs, _ = load_lfw_batch("data/lfw", n_test, seed=seed_test)
    print(f"3) ORT refs...", flush=True)
    sess = ort.InferenceSession("models/w600k_r50.onnx", providers=["CPUExecutionProvider"])
    ort_embs = [sess.run(None, {sess.get_inputs()[0].name: x})[0].flatten() for x in test_inputs]

    print(f"4) activation per-image absmax (for scales)...", flush=True)
    t0 = time.perf_counter()
    per_image = collect_per_image_absmax(g, calib_inputs)
    print(f"   done in {time.perf_counter()-t0:.1f}s")
    act_scales_base = build_percentile_scales(per_image, percentile=pct_late)

    print(f"5) per-input-channel absmax on calib set...", flush=True)
    t0 = time.perf_counter()
    per_ic = compute_per_in_channel_abs(g, calib_inputs)
    per_ic_ready = {k: v for k, v in per_ic.items() if v is not None}
    print(f"   done in {time.perf_counter()-t0:.1f}s  ({len(per_ic_ready)}/{len(per_ic)} tensors with stats)")

    print(f"\n6) alpha sweep (baseline alpha=0 = no smoothing):")
    for alpha in alpha_grid:
        if alpha == 0.0:
            factors = {}  # no-op
        else:
            factors = build_smooth_factors(g, per_ic_ready, alpha)
        # Post-smoothing, activation scales change. To simplify evaluation, we
        # re-use the baseline scales for alpha=0 and build fresh scales per-alpha
        # by re-running collect_per_image_absmax. That doubles runtime; instead,
        # approximate by rescaling baseline scales by 1/s for smoothed inputs.
        act_scales = dict(act_scales_base)
        # For Conv inputs that got smoothed, the new activation is old/s, so
        # per-channel scale becomes old_scale / s.
        init = {t["name"]: t for t in g["initializers"]}
        nodes = g["nodes"]
        for ni, s in factors.items():
            in_name = nodes[ni]["inputs"][0]
            if in_name in act_scales and act_scales[in_name].shape == s.shape:
                act_scales[in_name] = (act_scales[in_name] / s).astype(np.float32)

        t0 = time.perf_counter()
        sims = np.asarray([cos(fp, run_sim_smooth(g, x, act_scales, factors))
                            for x, fp in zip(test_inputs, ort_embs)])
        dt = time.perf_counter() - t0
        print(f"   alpha={alpha:.2f}  mean {sims.mean():.5f}  median {np.median(sims):.5f}  "
              f"min {sims.min():.5f}  >=0.99 {int((sims>=0.99).sum())}/100  ({dt:.1f}s)  "
              f"[{len(factors)} Convs smoothed]", flush=True)


if __name__ == "__main__":
    main()
