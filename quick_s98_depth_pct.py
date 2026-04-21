"""S98 — depth-aware percentile quick test (Python sim, 100 LFW faces).

Compares uniform percentile vs depth-aware percentile calibration on the
simulated per-channel INT8 pipeline. No C-binary rebuild needed.

Usage:
    python quick_s98_depth_pct.py

Env knobs (optional):
    N_CALIB (default 200)  — calibration faces
    N_TEST  (default 100)  — evaluation faces
    PCT_LATE (default 99.9)
    PCT_EARLY (default 99.0)
    SEED_CALIB (default 1)
    SEED_TEST  (default 7777)
"""
import os, sys, time, numpy as np
sys.path.insert(0, '.')
import torch
import torch.nn.functional as F
import onnxruntime as ort
from extract_onnx import parse_model
from calibrate_per_channel_int8 import (
    load_lfw_batch, fake_quant_per_channel_sym, fake_quant_weight_per_oc,
)
from calibrate_percentile_int8 import collect_per_image_absmax, build_percentile_scales
from calibrate_include_princess import load_lfw_with_princess


def run_sim(g, input_chw, scales):
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


def cos(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def eval_scales(g, test_inputs, ort_embs, scales, label):
    t0 = time.perf_counter()
    sims = []
    for x, fp in zip(test_inputs, ort_embs):
        e = run_sim(g, x, scales)
        sims.append(cos(fp, e))
    sims = np.asarray(sims)
    dt = time.perf_counter() - t0
    print(f"  [{label}]  mean {sims.mean():.5f}  median {np.median(sims):.5f}  "
          f"min {sims.min():.5f}  >=0.99 {int((sims >= 0.99).sum())}/{len(sims)}  "
          f"(>=0.98 {int((sims >= 0.98).sum())}/{len(sims)})  ({dt:.1f}s)")
    return sims


def main():
    n_calib    = int(os.environ.get("N_CALIB", "200"))
    n_test     = int(os.environ.get("N_TEST",  "100"))
    pct_late   = float(os.environ.get("PCT_LATE",  "99.9"))
    pct_early  = float(os.environ.get("PCT_EARLY", "99.0"))
    seed_calib = int(os.environ.get("SEED_CALIB", "1"))
    seed_test  = int(os.environ.get("SEED_TEST",  "7777"))

    print(f"S98 depth-aware percentile quick test")
    print(f"  N_CALIB={n_calib} N_TEST={n_test} PCT_LATE={pct_late} PCT_EARLY={pct_early}")
    print(f"  calib: WITH_PRINCESS mode")

    g = parse_model("models/w600k_r50.onnx")

    print(f"\n1) loading calib faces ({n_calib})...", flush=True)
    calib_inputs, _ = load_lfw_with_princess("data/lfw", n_calib, seed=seed_calib)

    print(f"2) loading test faces ({n_test})...", flush=True)
    test_inputs, _ = load_lfw_batch("data/lfw", n_test, seed=seed_test)

    print(f"3) ORT FP32 ground truth...", flush=True)
    sess = ort.InferenceSession("models/w600k_r50.onnx", providers=["CPUExecutionProvider"])
    inp_name = sess.get_inputs()[0].name
    ort_embs = [sess.run(None, {inp_name: x})[0].flatten() for x in test_inputs]

    print(f"4) collect per-image absmax on calib set...", flush=True)
    t0 = time.perf_counter()
    per_image = collect_per_image_absmax(g, calib_inputs)
    print(f"   done in {time.perf_counter()-t0:.1f}s ({len(per_image)} tensors)")

    print(f"\n5) evaluate scale configs on {n_test} test faces:")
    # A) uniform late percentile
    scales_uni = build_percentile_scales(per_image, percentile=pct_late)
    eval_scales(g, test_inputs, ort_embs, scales_uni, f"uniform p={pct_late}")
    # B) depth-aware
    scales_dep = build_percentile_scales(per_image, percentile=pct_late,
                                          depth_early_percentile=pct_early)
    eval_scales(g, test_inputs, ort_embs, scales_dep,
                f"depth-aware early={pct_early} late={pct_late}")


if __name__ == "__main__":
    main()
