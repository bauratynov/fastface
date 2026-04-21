"""Session 10 — fp32 reference forward using torch (fast) to validate graph extraction.
If cos-sim(our_torch_reconstruction, ORT) ≈ 1.0 → graph is correctly understood.
Then next step (future session): add INT8 quantization and check cos-sim degradation.
"""
import sys
sys.path.insert(0, '.')
from extract_onnx import parse_model
import numpy as np
import torch
import torch.nn.functional as F
import onnxruntime as ort
import time


def cos_sim(a, b):
    a = np.asarray(a).flatten().astype(np.float32)
    b = np.asarray(b).flatten().astype(np.float32)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def run_torch_reconstruction(input_nchw):
    """Walk the graph, execute each op via torch (FP32). Should match ORT exactly."""
    g = parse_model("models/w600k_r50.onnx")
    init = {t["name"]: t for t in g["initializers"]}
    nodes = g["nodes"]

    # Input tensor by name
    tensors = {}
    input_name = "input.1"
    tensors[input_name] = torch.from_numpy(input_nchw).float()

    for ni, node in enumerate(nodes):
        op = node["op_type"]
        outs = node["outputs"]

        if op == "Conv":
            x = tensors[node["inputs"][0]]
            w = torch.from_numpy(init[node["inputs"][1]]["numpy"].astype(np.float32))
            attrs = {a["name"]: a for a in node["attrs"]}
            stride = attrs["strides"]["ints"][0]
            pad = attrs["pads"]["ints"][0]
            bias = None
            if len(node["inputs"]) > 2 and node["inputs"][2]:
                bt = init.get(node["inputs"][2])
                if bt and bt.get("numpy") is not None:
                    bias = torch.from_numpy(bt["numpy"].astype(np.float32))
            y = F.conv2d(x, w, bias=bias, stride=stride, padding=pad)
            tensors[outs[0]] = y

        elif op == "BatchNormalization":
            x = tensors[node["inputs"][0]]
            gamma = torch.from_numpy(init[node["inputs"][1]]["numpy"].astype(np.float32))
            beta = torch.from_numpy(init[node["inputs"][2]]["numpy"].astype(np.float32))
            mean = torch.from_numpy(init[node["inputs"][3]]["numpy"].astype(np.float32))
            var = torch.from_numpy(init[node["inputs"][4]]["numpy"].astype(np.float32))
            eps = [a for a in node["attrs"] if a["name"] == "epsilon"][0]["f"]
            y = F.batch_norm(x, mean, var, gamma, beta, training=False, eps=eps)
            tensors[outs[0]] = y

        elif op == "PRelu":
            x = tensors[node["inputs"][0]]
            slope = torch.from_numpy(init[node["inputs"][1]]["numpy"].astype(np.float32))
            # ONNX PRelu: slope shape may be [C] or [C,1,1]; broadcast across [N,C,H,W]
            if slope.ndim == 1:
                slope = slope.view(1, -1, 1, 1)
            elif slope.ndim > 1 and slope.numel() == slope.shape[0]:
                slope = slope.view(1, -1, 1, 1)
            y = torch.where(x >= 0, x, x * slope)
            tensors[outs[0]] = y

        elif op == "Add":
            a = tensors[node["inputs"][0]]
            b = tensors[node["inputs"][1]]
            tensors[outs[0]] = a + b

        elif op == "Flatten":
            x = tensors[node["inputs"][0]]
            attrs = {a["name"]: a for a in node["attrs"]}
            axis = attrs.get("axis", {"i": 1})["i"]
            if axis == 1:
                y = x.flatten(1)  # [N, -1]
            else:
                y = x.flatten(axis)
            tensors[outs[0]] = y

        elif op == "Gemm":
            x = tensors[node["inputs"][0]]
            w = torch.from_numpy(init[node["inputs"][1]]["numpy"].astype(np.float32))
            b = None
            if len(node["inputs"]) > 2:
                bt = init.get(node["inputs"][2])
                if bt and bt.get("numpy") is not None:
                    b = torch.from_numpy(bt["numpy"].astype(np.float32))
            attrs = {a["name"]: a for a in node["attrs"]}
            transA = attrs.get("transA", {"i": 0})["i"]
            transB = attrs.get("transB", {"i": 0})["i"]
            alpha = attrs.get("alpha", {"f": 1.0})["f"]
            beta_v = attrs.get("beta", {"f": 1.0})["f"]
            # Default ONNX Gemm: Y = alpha * A * B + beta * C
            # Ensure correct shape; some models omit transB but weight is [N, K] so need transpose
            A = x.T if transA else x
            B = w.T if transB else w
            # Auto-fix if shapes don't match (weight might be [N, K] without transB set)
            if A.dim() == 2 and B.dim() == 2 and A.shape[-1] != B.shape[0]:
                if A.shape[-1] == B.shape[-1]:
                    B = B.T
            y = alpha * (A @ B)
            if b is not None:
                y = y + beta_v * b
            tensors[outs[0]] = y

    last_out = nodes[-1]["outputs"][0]
    return tensors[last_out].numpy()


def main():
    np.random.seed(42)
    inp = np.random.randn(1, 3, 112, 112).astype(np.float32)

    # ORT
    sess = ort.InferenceSession("models/w600k_r50.onnx", providers=["CPUExecutionProvider"])
    inp_name = sess.get_inputs()[0].name
    t0 = time.perf_counter()
    ort_out = sess.run(None, {inp_name: inp})[0].flatten()
    t_ort = time.perf_counter() - t0
    print(f"ORT:    output[512], norm={np.linalg.norm(ort_out):.4f}  time={t_ort*1000:.1f}ms")
    print(f"  first5={ort_out[:5]}")

    # Torch reconstruction
    print("\nRunning torch reconstruction...")
    t0 = time.perf_counter()
    ref_out = run_torch_reconstruction(inp).flatten()
    t_ref = time.perf_counter() - t0
    print(f"Torch reconstruction: norm={np.linalg.norm(ref_out):.4f}  time={t_ref*1000:.0f}ms")
    print(f"  first5={ref_out[:5]}")

    sim = cos_sim(ort_out, ref_out)
    print(f"\nCosine similarity: {sim:.6f}")
    if sim > 0.9999:
        print("PERFECT MATCH: graph topology + weights are correctly extracted")
    elif sim > 0.99:
        print("Very close: numerical precision diff, safe")
    elif sim > 0.9:
        print("Close but something mild mismatches — investigate")
    else:
        print("BROKEN: graph reconstruction does not match ORT")


if __name__ == "__main__":
    main()
