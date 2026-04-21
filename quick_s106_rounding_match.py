"""S106 -- rounding convention test.

Replace the sim's torch.round (banker's rounding, half-to-even) with
round-half-away-from-zero to match the C binary:

    q = sign(x) * floor(abs(x) + 0.5)

If this is the main gap source, sim mean should drop from 0.99623 toward
the binary's 0.99326 on 100 LFW faces.
"""
import os, sys, time, numpy as np
sys.path.insert(0, '.')
import torch
import torch.nn.functional as F
import onnxruntime as ort
from extract_onnx import parse_model
from calibrate_per_channel_int8 import load_lfw_batch
from calibrate_percentile_int8 import collect_per_image_absmax, build_percentile_scales
from calibrate_include_princess import load_lfw_with_princess


def round_hatz(t):
    """Round-half-away-from-zero."""
    return torch.sign(t) * torch.floor(torch.abs(t) + 0.5)


def fake_quant_per_channel_sym_hatz(x, scale_c):
    """Per-channel symmetric with HATZ rounding (binary-matched)."""
    if x.ndim == 4:
        s = torch.tensor(scale_c).view(1, -1, 1, 1)
    elif x.ndim == 2:
        s = torch.tensor(scale_c).view(1, -1)
    else:
        s = torch.tensor(scale_c)
    q = round_hatz(x / (s + 1e-9)).clamp(-127, 127)
    return q * s


def fake_quant_weight_per_oc_hatz(w):
    """Per-OC weight fake-quant with HATZ rounding."""
    w_np = w.numpy() if isinstance(w, torch.Tensor) else w
    axes = tuple(range(1, w_np.ndim))
    absmax = np.max(np.abs(w_np), axis=axes, keepdims=True) + 1e-9
    scale = absmax / 127.0
    q_fp = w_np / scale
    q = np.sign(q_fp) * np.floor(np.abs(q_fp) + 0.5)
    q = np.clip(q, -127, 127)
    return torch.from_numpy((q * scale).astype(np.float32))


def run_sim(g, input_chw, act_scales, use_hatz=False):
    qact = fake_quant_per_channel_sym_hatz if use_hatz else None
    wq   = fake_quant_weight_per_oc_hatz if use_hatz else None
    if not use_hatz:
        from calibrate_per_channel_int8 import fake_quant_per_channel_sym, fake_quant_weight_per_oc
        qact = fake_quant_per_channel_sym
        wq = fake_quant_weight_per_oc

    init = {t["name"]: t for t in g["initializers"]}
    nodes = g["nodes"]
    tensors = {}

    def qapply(t, name):
        if name not in act_scales: return t
        return qact(t, act_scales[name])

    tensors["input.1"] = qapply(torch.from_numpy(input_chw).float(), "input.1")
    for ni, node in enumerate(nodes):
        op = node["op_type"]; outs = node["outputs"]
        if op == "Conv":
            x = tensors[node["inputs"][0]]
            w_raw = torch.from_numpy(init[node["inputs"][1]]["numpy"].astype(np.float32))
            w_q = wq(w_raw)
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
            w_q = wq(w_raw)
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
    n_test    = int(os.environ.get("N_TEST", "100"))
    pct_late  = float(os.environ.get("PCT_LATE", "99.9"))
    seed_test = int(os.environ.get("SEED_TEST", "7777"))

    g = parse_model("models/w600k_r50.onnx")

    print("S106 rounding-match hypothesis test")
    print(f"  baseline sim (banker): expected mean 0.99623")
    print(f"  binary (HATZ):         expected mean 0.99326")
    print(f"  gap:                   0.00297")

    print("\n1) load calib (200 WITH_PRINCESS)...", flush=True)
    calib_inputs, _ = load_lfw_with_princess("data/lfw", n_calib, seed=1)

    print(f"2) load test ({n_test} seed={seed_test})...", flush=True)
    test_inputs, _ = load_lfw_batch("data/lfw", n_test, seed=seed_test)

    print("3) ORT FP32 refs...", flush=True)
    sess = ort.InferenceSession("models/w600k_r50.onnx", providers=["CPUExecutionProvider"])
    ort_embs = [sess.run(None, {sess.get_inputs()[0].name: x})[0].flatten() for x in test_inputs]

    print("4) collect per-image absmax...", flush=True)
    t0 = time.perf_counter()
    per_image = collect_per_image_absmax(g, calib_inputs)
    print(f"   done in {time.perf_counter()-t0:.1f}s")
    act_scales = build_percentile_scales(per_image, percentile=pct_late)

    for use_hatz in (False, True):
        t0 = time.perf_counter()
        sims = np.asarray([cos(fp, run_sim(g, x, act_scales, use_hatz=use_hatz))
                            for x, fp in zip(test_inputs, ort_embs)])
        dt = time.perf_counter() - t0
        label = "HATZ (binary-match)" if use_hatz else "banker (numpy default)"
        print(f"   [{label}]  mean {sims.mean():.5f}  median {np.median(sims):.5f}  "
              f"min {sims.min():.5f}  >=0.99 {int((sims>=0.99).sum())}/100  ({dt:.1f}s)")


if __name__ == "__main__":
    main()
