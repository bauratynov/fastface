"""S107 -- weight-fold rounding hypothesis.

The binary folds per-input-channel activation scale S_a[ci] into Conv
weights BEFORE int8 quantization (prepare_weights_v3.py S38). That means
the binary's quantization error on the recovered w_orig is per-ci-weighted:
small where S_a is large, large where S_a is small.

The sim's fake_quant_weight_per_oc quantizes raw w_orig, giving uniform
per-OC rounding noise.

To match binary behavior: quantize w_folded = w_orig * S_a[ci] per-OC,
then divide back by S_a[ci] after dequant. Conv with this weight on
fp32 input (pre-scaled-and-quantized by fake_quant_per_channel_sym via
S_a) should match the binary's mathematical operation.

If sim mean drops toward 0.993, THIS is the gap. If not, gap is deeper.
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


def fake_quant_weight_folded(w_orig, S_a_per_ci):
    """Quantize w * S_a[ci] per-OC, then divide back. Matches binary behavior.
    w_orig shape: [Cout, Cin, Kh, Kw]. S_a_per_ci shape: [Cin]."""
    w_np = w_orig.numpy() if isinstance(w_orig, torch.Tensor) else w_orig
    if S_a_per_ci is None:
        axes = tuple(range(1, w_np.ndim))
        absmax = np.max(np.abs(w_np), axis=axes, keepdims=True) + 1e-9
        scale = absmax / 127.0
        q = np.round(w_np / scale).clip(-127, 127)
        return torch.from_numpy((q * scale).astype(np.float32))
    # Fold: w_folded = w * S_a[ci]
    S = np.asarray(S_a_per_ci, dtype=np.float32)
    if w_np.ndim == 4:
        S_b = S.reshape(1, -1, 1, 1)
    else:
        S_b = S.reshape(1, -1)
    w_folded = w_np * S_b
    # Quantize folded per-OC
    axes = tuple(range(1, w_np.ndim))
    absmax = np.max(np.abs(w_folded), axis=axes, keepdims=True) + 1e-9
    scale_folded = absmax / 127.0
    q_folded = np.round(w_folded / scale_folded).clip(-127, 127)
    w_folded_dequant = q_folded * scale_folded
    # Unfold: divide back by S_a[ci]
    w_recovered = w_folded_dequant / (S_b + 1e-12)
    return torch.from_numpy(w_recovered.astype(np.float32))


def run_sim_foldq(g, input_chw, act_scales, fold_weights=False):
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
            in_name = node["inputs"][0]
            x = tensors[in_name]
            w_raw = torch.from_numpy(init[node["inputs"][1]]["numpy"].astype(np.float32))
            if fold_weights and in_name in act_scales:
                # S_a is the per-Cin scale matching the Conv input's per-channel act scale
                S_a = act_scales[in_name]
                w_q = fake_quant_weight_folded(w_raw, S_a)
            else:
                w_q = fake_quant_weight_folded(w_raw, None)
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
            in_name = node["inputs"][0]
            x = tensors[in_name]
            w_raw = torch.from_numpy(init[node["inputs"][1]]["numpy"].astype(np.float32))
            # Gemm: simplest -- quantize w as-is (no per-ci fold for Gemm weights)
            w_q = fake_quant_weight_folded(w_raw, None)
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
    n_calib   = int(os.environ.get("N_CALIB", "200"))
    n_test    = int(os.environ.get("N_TEST", "100"))
    pct_late  = float(os.environ.get("PCT_LATE", "99.9"))
    seed_test = int(os.environ.get("SEED_TEST", "7777"))

    g = parse_model("models/w600k_r50.onnx")

    print("S107 weight-fold rounding hypothesis test")
    print(f"  expected baseline:  mean 0.99623 (v1.1.0 sim)")
    print(f"  expected fold match: should drop toward 0.993 if fold-rounding is gap source")

    print("\n1) load calib...", flush=True)
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

    for fold in (False, True):
        t0 = time.perf_counter()
        sims = np.asarray([cos(fp, run_sim_foldq(g, x, act_scales, fold_weights=fold))
                            for x, fp in zip(test_inputs, ort_embs)])
        dt = time.perf_counter() - t0
        label = "FOLDED weight quant (binary-match)" if fold else "UNFOLDED weight quant (sim default)"
        print(f"   [{label}]  mean {sims.mean():.5f}  median {np.median(sims):.5f}  "
              f"min {sims.min():.5f}  >=0.99 {int((sims>=0.99).sum())}/100  ({dt:.1f}s)")


if __name__ == "__main__":
    main()
