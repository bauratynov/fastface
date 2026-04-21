"""S101 -- per-OC weight percentile (vs absmax) test.

Currently Conv/Gemm weights use per-output-channel absmax scaling. This
scales to the max abs value in each output channel's receptive field.

Test: use percentile instead (p=99.9, 99.99, 100.0=absmax) to clip
potential weight outliers within each output channel.

Python sim, 100 LFW faces, same calibration infrastructure as S98/S99.
Compares 3 weight-percentile configs while keeping activation calibration
fixed at v1.1.0 (p=99.9 + WITH_PRINCESS, N_CALIB=200).
"""
import os, sys, time, numpy as np
sys.path.insert(0, '.')
import torch
import torch.nn.functional as F
import onnxruntime as ort
from extract_onnx import parse_model
from calibrate_per_channel_int8 import load_lfw_batch, fake_quant_per_channel_sym
from calibrate_percentile_int8 import collect_per_image_absmax, build_percentile_scales
from calibrate_include_princess import load_lfw_with_princess


def fake_quant_weight_pct(w, pct=100.0):
    """Per-OC weight fake-quant with given percentile (100.0 = absmax)."""
    w_np = w.numpy() if isinstance(w, torch.Tensor) else w
    axes = tuple(range(1, w_np.ndim))
    if pct >= 100.0:
        amax = np.max(np.abs(w_np), axis=axes, keepdims=True)
    else:
        abs_w = np.abs(w_np)
        amax = np.percentile(abs_w, pct, axis=axes, keepdims=True)
    amax = amax + 1e-9
    scale = amax / 127.0
    q = np.round(w_np / scale).clip(-127, 127)
    return torch.from_numpy((q * scale).astype(np.float32))


def run_sim(g, input_chw, act_scales, weight_pct=100.0):
    init = {t["name"]: t for t in g["initializers"]}
    nodes = g["nodes"]
    tensors = {}

    def qact(t, name):
        if name not in act_scales: return t
        return fake_quant_per_channel_sym(t, act_scales[name])

    tensors["input.1"] = qact(torch.from_numpy(input_chw).float(), "input.1")
    for ni, node in enumerate(nodes):
        op = node["op_type"]; outs = node["outputs"]
        if op == "Conv":
            x = tensors[node["inputs"][0]]
            w_raw = torch.from_numpy(init[node["inputs"][1]]["numpy"].astype(np.float32))
            w_q = fake_quant_weight_pct(w_raw, pct=weight_pct)
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
            w_q = fake_quant_weight_pct(w_raw, pct=weight_pct)
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
            y = qact(y, outs[0])
        tensors[outs[0]] = y
    return tensors[nodes[-1]["outputs"][0]].numpy().flatten()


def cos(a, b): return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def main():
    n_calib    = int(os.environ.get("N_CALIB", "200"))
    n_test     = int(os.environ.get("N_TEST",  "100"))
    pct_late   = float(os.environ.get("PCT_LATE",  "99.9"))
    seed_calib = int(os.environ.get("SEED_CALIB", "1"))

    test_seeds = [7777, 11111]
    weight_grid = [100.0, 99.99, 99.9]  # absmax, then two percentiles

    print(f"S101 weight percentile sweep  N_CALIB={n_calib} N_TEST={n_test}")
    print(f"  activation: p={pct_late} WITH_PRINCESS, weight pct grid: {weight_grid}")

    g = parse_model("models/w600k_r50.onnx")

    print(f"\n1) loading calib faces ({n_calib}, WITH_PRINCESS)...", flush=True)
    calib_inputs, _ = load_lfw_with_princess("data/lfw", n_calib, seed=seed_calib)
    print(f"2) loading test faces x {len(test_seeds)}...", flush=True)
    test_sets = {ts: load_lfw_batch("data/lfw", n_test, seed=ts)[0] for ts in test_seeds}

    print(f"3) ORT FP32 ground truth...", flush=True)
    sess = ort.InferenceSession("models/w600k_r50.onnx", providers=["CPUExecutionProvider"])
    inp_name = sess.get_inputs()[0].name
    ort_embs = {ts: [sess.run(None, {inp_name: x})[0].flatten() for x in test_sets[ts]]
                for ts in test_seeds}

    print(f"4) collect per-image absmax (activations)...", flush=True)
    t0 = time.perf_counter()
    per_image = collect_per_image_absmax(g, calib_inputs)
    print(f"   done in {time.perf_counter()-t0:.1f}s")
    act_scales = build_percentile_scales(per_image, percentile=pct_late)

    print(f"\n5) weight-pct sweep:")
    print(f"  {'w_pct':>7} | {'seed 7777 mean min >=99':>34} | {'seed 11111 mean min >=99':>34} | mom")
    print(f"  " + "-" * 84)

    results = []
    for wp in weight_grid:
        row = f"  {wp:>7.2f} |"
        means = []
        for ts in test_seeds:
            t0 = time.perf_counter()
            sims = []
            for x, fp in zip(test_sets[ts], ort_embs[ts]):
                sims.append(cos(fp, run_sim(g, x, act_scales, weight_pct=wp)))
            sims = np.asarray(sims)
            row += f"  {sims.mean():.5f} {sims.min():.5f} {int((sims>=0.99).sum()):>3}/100  ({time.perf_counter()-t0:.1f}s) |"
            means.append(sims.mean())
        mom = float(np.mean(means))
        row += f"  {mom:.5f}"
        print(row, flush=True)
        results.append((wp, mom, means))

    print("\nRanking (by mean-of-means):")
    baseline_mom = [r for r in results if r[0] == 100.0][0][1]
    for wp, mom, means in sorted(results, key=lambda r: -r[1]):
        delta = mom - baseline_mom
        print(f"  weight_pct={wp:>6.2f}  mom={mom:.5f}  delta vs absmax: {delta:+.5f}")


if __name__ == "__main__":
    main()
