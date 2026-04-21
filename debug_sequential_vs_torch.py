"""Diagnose C binary mismatch:
  (a) Run torch tensor-name reconstruction on LFW face → should match ORT (cos-sim≈1)
  (b) Run "sequential" reconstruction mimicking our C binary's block_buf heuristic
      → if this also matches ORT, our dataflow model is correct and bug is in the C code.
      If it diverges, the heuristic doesn't capture the real graph.
"""
import sys, os, glob, random, time
sys.path.insert(0, '.')
from extract_onnx import parse_model
import numpy as np
import torch
import torch.nn.functional as F
import onnxruntime as ort
from PIL import Image


def cos_sim(a, b):
    a = a.flatten(); b = b.flatten()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def load_face(lfw_dir, seed=42):
    paths = sorted(glob.glob(os.path.join(lfw_dir, "**", "*.jpg"), recursive=True))
    random.seed(seed); random.shuffle(paths)
    img = Image.open(paths[0]).convert("RGB")
    w, h = img.size; s = 150
    left = (w - s) // 2; top = max(0, (h - s) // 2 - 10)
    img = img.crop((left, top, left + s, top + s)).resize((112, 112), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32)
    arr = (arr - 127.5) / 127.5
    nchw = np.transpose(arr, (2, 0, 1))[None, ...].copy()
    return nchw


def torch_by_name(input_nchw):
    """Reference: walk graph by tensor name — ground truth."""
    g = parse_model("models/w600k_r50.onnx")
    init = {t["name"]: t for t in g["initializers"]}
    nodes = g["nodes"]
    tensors = {"input.1": torch.from_numpy(input_nchw).float()}
    for ni, node in enumerate(nodes):
        op = node["op_type"]; outs = node["outputs"]
        if op == "Conv":
            x = tensors[node["inputs"][0]]
            w = torch.from_numpy(init[node["inputs"][1]]["numpy"].astype(np.float32))
            bias = None
            if len(node["inputs"]) > 2 and node["inputs"][2]:
                bt = init.get(node["inputs"][2])
                if bt is not None and bt.get("numpy") is not None:
                    bias = torch.from_numpy(bt["numpy"].astype(np.float32))
            attrs = {a["name"]: a for a in node["attrs"]}
            y = F.conv2d(x, w, bias=bias, stride=attrs["strides"]["ints"][0], padding=attrs["pads"]["ints"][0])
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
        else: continue
        tensors[outs[0]] = y
    return tensors[nodes[-1]["outputs"][0]].numpy()


def sequential_like_c_with_trace(input_nchw, ref_tensors):
    """Mimic our C binary exactly: linear op list from prepare_weights_v2,
    save_id stack, block_buf heuristic, shortcut = Conv-after-SAVE_ID."""
    g = parse_model("models/w600k_r50.onnx")
    init = {t["name"]: t for t in g["initializers"]}
    nodes = g["nodes"]

    # Mirror prepare_weights_v2.py's op sequence + identity marking
    producer = {}
    for i, n in enumerate(nodes):
        for o in n["outputs"]: producer[o] = i
    identity_sources = set()
    for i, n in enumerate(nodes):
        if n["op_type"] != "Add": continue
        in0, in1 = n["inputs"][:2]
        p0, p1 = producer.get(in0, -1), producer.get(in1, -1)
        if p0 < 0 and p1 < 0: continue
        if p0 != -1 and (p1 == -1 or p0 < p1): identity_sources.add(p0)
        else: identity_sources.add(p1)

    # Build linearized op list exactly like prepare_weights_v2
    ops = []  # list of ('type', node_ref or None)
    for ni, node in enumerate(nodes):
        t = node["op_type"]
        if t in ("Conv","BatchNormalization","PRelu","Add","Gemm","Flatten"):
            ops.append((t, ni))
        if ni in identity_sources:
            ops.append(("SAVE_ID", ni))

    # Detect shortcut: Conv preceded by SAVE_ID
    is_shortcut = [False] * len(ops)
    for i in range(1, len(ops)):
        if ops[i][0] == "Conv" and ops[i-1][0] == "SAVE_ID":
            is_shortcut[i] = True

    # Execute
    A = torch.from_numpy(input_nchw).float()
    block_buf = A.clone()
    id_stack = []  # FIFO; save appends, add pops front
    first_bad = None
    for i, (t, ni) in enumerate(ops):
        node = nodes[ni] if ni is not None else None
        if t == "Conv":
            w = torch.from_numpy(init[node["inputs"][1]]["numpy"].astype(np.float32))
            bias = None
            if len(node["inputs"]) > 2 and node["inputs"][2]:
                bt = init.get(node["inputs"][2])
                if bt is not None and bt.get("numpy") is not None:
                    bias = torch.from_numpy(bt["numpy"].astype(np.float32))
            attrs = {a["name"]: a for a in node["attrs"]}
            s = attrs["strides"]["ints"][0]; p = attrs["pads"]["ints"][0]
            inp = block_buf if is_shortcut[i] else A
            A = F.conv2d(inp, w, bias=bias, stride=s, padding=p)
        elif t == "BatchNormalization":
            # Save block input BEFORE BN — residual shortcuts consume pre-BN tensor
            block_buf = A.clone()
            gamma = torch.from_numpy(init[node["inputs"][1]]["numpy"].astype(np.float32))
            beta = torch.from_numpy(init[node["inputs"][2]]["numpy"].astype(np.float32))
            mean = torch.from_numpy(init[node["inputs"][3]]["numpy"].astype(np.float32))
            var = torch.from_numpy(init[node["inputs"][4]]["numpy"].astype(np.float32))
            eps = [a for a in node["attrs"] if a["name"] == "epsilon"][0]["f"]
            A = F.batch_norm(A, mean, var, gamma, beta, training=False, eps=eps)
        elif t == "PRelu":
            slope = torch.from_numpy(init[node["inputs"][1]]["numpy"].astype(np.float32))
            if slope.ndim == 1: slope = slope.view(1, -1, 1, 1)
            elif slope.ndim > 1 and slope.numel() == slope.shape[0]: slope = slope.view(1, -1, 1, 1)
            A = torch.where(A >= 0, A, A * slope)
        elif t == "Add":
            sv = id_stack.pop(0)
            A = A + sv
        elif t == "SAVE_ID":
            id_stack.append(A.clone())
        elif t == "Flatten":
            A = A.flatten(1)
        elif t == "Gemm":
            w = torch.from_numpy(init[node["inputs"][1]]["numpy"].astype(np.float32))
            b = torch.from_numpy(init[node["inputs"][2]]["numpy"].astype(np.float32)) if len(node["inputs"]) > 2 else None
            attrs = {a["name"]: a for a in node["attrs"]}
            transB = attrs.get("transB", {"i": 0})["i"]
            B = w.T if transB else w
            if A.shape[-1] != B.shape[0] and A.shape[-1] == B.shape[-1]: B = B.T
            A = A @ B
            if b is not None: A = A + b

        # Compare to reference at this op's output node
        if ni is not None and t != "SAVE_ID":
            out_name = nodes[ni]["outputs"][0]
            if out_name in ref_tensors:
                ref = ref_tensors[out_name]
                if A.shape == ref.shape:
                    diff = (A - ref).abs().max().item()
                    c = float((A.flatten().dot(ref.flatten()) /
                               (A.flatten().norm() * ref.flatten().norm() + 1e-9)).item())
                    if c < 0.999 and first_bad is None:
                        first_bad = (i, t, ni, out_name, c, diff, list(A.shape))
                else:
                    if first_bad is None:
                        first_bad = (i, t, ni, out_name, -1, -1, list(A.shape), list(ref.shape))
    return A.numpy(), is_shortcut, ops, first_bad


def torch_by_name_with_tensors(input_nchw):
    g = parse_model("models/w600k_r50.onnx")
    init = {t["name"]: t for t in g["initializers"]}
    nodes = g["nodes"]
    tensors = {"input.1": torch.from_numpy(input_nchw).float()}
    for ni, node in enumerate(nodes):
        op = node["op_type"]; outs = node["outputs"]
        if op == "Conv":
            x = tensors[node["inputs"][0]]
            w = torch.from_numpy(init[node["inputs"][1]]["numpy"].astype(np.float32))
            bias = None
            if len(node["inputs"]) > 2 and node["inputs"][2]:
                bt = init.get(node["inputs"][2])
                if bt is not None and bt.get("numpy") is not None:
                    bias = torch.from_numpy(bt["numpy"].astype(np.float32))
            attrs = {a["name"]: a for a in node["attrs"]}
            y = F.conv2d(x, w, bias=bias, stride=attrs["strides"]["ints"][0], padding=attrs["pads"]["ints"][0])
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
        else: continue
        tensors[outs[0]] = y
    return tensors


def main():
    nchw = load_face("data/lfw", seed=42)
    sess = ort.InferenceSession("models/w600k_r50.onnx", providers=["CPUExecutionProvider"])
    inp_name = sess.get_inputs()[0].name
    ort_out = sess.run(None, {inp_name: nchw})[0].flatten()
    print(f"ORT:                   norm={np.linalg.norm(ort_out):.4f}")

    ref_tensors = torch_by_name_with_tensors(nchw)
    ref_out = list(ref_tensors.values())[-1].numpy().flatten()
    print(f"Torch-by-name:         norm={np.linalg.norm(ref_out):.4f}  cos-vs-ORT={cos_sim(ort_out, ref_out):.6f}")

    seq_out, is_shortcut, ops, first_bad = sequential_like_c_with_trace(nchw, ref_tensors)
    seq_out = seq_out.flatten()
    print(f"Sequential (C-style):  norm={np.linalg.norm(seq_out):.4f}  cos-vs-ORT={cos_sim(ort_out, seq_out):.6f}")
    print(f"  shortcuts detected:  {sum(is_shortcut)}")
    print(f"  total ops:           {len(ops)}")
    print(f"  first divergence:    {first_bad}")


if __name__ == "__main__":
    main()
