"""Compare C binary's per-op output trace against torch-by-name reference tensors."""
import sys; sys.path.insert(0, '.')
from debug_sequential_vs_torch import torch_by_name_with_tensors, load_face
from extract_onnx import parse_model
import numpy as np


def main():
    nchw = load_face("data/lfw", seed=42)
    tensors = torch_by_name_with_tensors(nchw)
    g = parse_model("models/w600k_r50.onnx")
    nodes = g["nodes"]

    # Build op index → node index mapping matching prepare_weights_v2
    producer = {o: i for i, n in enumerate(nodes) for o in n["outputs"]}
    identity_sources = set()
    for i, n in enumerate(nodes):
        if n["op_type"] != "Add": continue
        in0, in1 = n["inputs"][:2]
        p0, p1 = producer.get(in0, -1), producer.get(in1, -1)
        if p0 < 0 and p1 < 0: continue
        if p0 != -1 and (p1 == -1 or p0 < p1): identity_sources.add(p0)
        else: identity_sources.add(p1)

    op_node = []
    for ni, node in enumerate(nodes):
        t = node["op_type"]
        if t in ("Conv","BatchNormalization","PRelu","Add","Gemm","Flatten"):
            op_node.append(ni)
        if ni in identity_sources:
            op_node.append(-1)  # SAVE_ID, no corresponding node

    # Parse C trace
    c_ops = []
    with open("c_trace.txt") as f:
        for line in f:
            parts = line.strip().split()
            if not parts: continue
            idx = int(parts[1]); op_type = int(parts[3])
            shape = parts[4]  # "shape[C,H,W]"
            norm_c = float(parts[6])
            first5 = [float(p) for p in parts[8:13]]
            c_ops.append((idx, op_type, shape, norm_c, first5))

    print(f"{'idx':>4} {'node':>4} {'type':<6} {'name':<6} "
          f"{'py_norm':>10} {'c_norm':>10} {'pct':>7} {'cos':>8}")

    for idx, op_type, shape_str, norm_c, first5_c in c_ops:
        ni = op_node[idx]
        if ni < 0:
            continue  # SAVE_ID has no reference node
        out_name = nodes[ni]["outputs"][0]
        if out_name not in tensors:
            continue
        t = tensors[out_name]
        t_np = t.numpy()
        norm_py = float(np.linalg.norm(t_np))
        # For Conv/BN/PRelu/Add on 4D tensor, convert to NHWC for first-5 comparison
        if t_np.ndim == 4:
            # first5 = [N=0,H=0,W=0, c=0..4]
            first5_py = t_np[0, :5, 0, 0].flatten()
            # Cos-sim: full tensor values (layout doesn't matter for cos-sim)
            cos = float(np.dot(t_np.flatten(), t_np.flatten()) > 0)  # placeholder
            # actual cos-sim comparison requires C full tensor — skip, only norm
        else:
            first5_py = t_np.flatten()[:5]
        pct = 100 * (norm_c - norm_py) / (norm_py + 1e-9)
        node_type = nodes[ni]["op_type"][:6]
        flag = "<<< BIG DIFF" if abs(pct) > 5 else ""
        print(f"{idx:>4} {ni:>4} {node_type:<6} {out_name:<6} "
              f"{norm_py:>10.4f} {norm_c:>10.4f} {pct:>6.2f}% "
              f"  py_first5={first5_py.round(4).tolist()} c_first5={[round(x, 4) for x in first5_c]} {flag}")


if __name__ == "__main__":
    main()
