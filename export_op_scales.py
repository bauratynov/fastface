"""Export per-op int8 output scales (per-tensor = max over per-channel absmax).

Format:
  magic 'OPSC'
  u32 n_ops
  f32[n_ops] scales   (0.0 = not applicable / ignored)
"""
import sys, os, struct, random, time
sys.path.insert(0, '.')
from extract_onnx import parse_model
from calibrate_per_channel_int8 import collect_ranges, load_lfw_batch
import numpy as np


def main():
    n_calib = int(os.environ.get("N_CALIB", "20"))
    calib_inputs, _ = load_lfw_batch("data/lfw", n_calib, seed=1)
    print(f"Calibrating on {n_calib} faces...", flush=True)
    g = parse_model("models/w600k_r50.onnx")
    t0 = time.perf_counter()
    ranges = collect_ranges(g, calib_inputs)
    print(f"  done in {time.perf_counter()-t0:.1f}s")

    # Per-tensor scale = max over per-channel absmax / 127.
    # We also expose per-channel later if needed.
    per_tensor_scale = {}
    for name, (tmin, tmax) in ranges.items():
        absmax = float(max(abs(tmin.min()), abs(tmax.max())))
        per_tensor_scale[name] = absmax / 127.0

    # Walk ONNX nodes in order. Each node has outputs[0] tensor name.
    # FFW3 op sequence follows ONNX node order; SAVE_ID/BLOCK_START interleaved.
    # We must recreate the same op list the prepare_weights_v2 script created.
    nodes = g["nodes"]

    # Reproduce identity_sources heuristic
    producer = {o: i for i, n in enumerate(nodes) for o in n["outputs"]}
    identity_sources = set()
    for i, n in enumerate(nodes):
        if n["op_type"] != "Add": continue
        in0, in1 = n["inputs"][:2]
        p0, p1 = producer.get(in0, -1), producer.get(in1, -1)
        if p0 < 0 and p1 < 0: continue
        if p0 != -1 and (p1 == -1 or p0 < p1): identity_sources.add(p0)
        else: identity_sources.add(p1)

    # S85: same skip for trailing BN-after-Gemm as prepare_weights_v3
    skip_nodes = set()
    for i, n in enumerate(nodes):
        if n["op_type"] == "Gemm" and i + 1 < len(nodes) \
           and nodes[i+1]["op_type"] == "BatchNormalization" \
           and nodes[i+1]["inputs"][0] == n["outputs"][0]:
            skip_nodes.add(i + 1)

    # Walk like prepare_weights_v2 and assign scales
    op_scales = []
    input_scale = per_tensor_scale.get("input.1", 1.0)
    pending_bn_tensor = None  # name of BN output
    for ni, node in enumerate(nodes):
        if ni in skip_nodes: continue
        op = node["op_type"]
        if op == "Conv":
            # Check if preceding was BN (BLOCK_START + OP_BN legacy emit — for first Conv of block)
            if pending_bn_tensor:
                op_scales.append(0.0)  # BLOCK_START (no output tensor)
                op_scales.append(0.0)  # OP_BN (legacy, since we can't fold here: for now it's a no-op tag)
                pending_bn_tensor = None
            # OP_CONV's output tensor is node["outputs"][0]
            op_scales.append(per_tensor_scale.get(node["outputs"][0], 0.0))
        elif op == "BatchNormalization":
            # Pending — emitted as BLOCK_START + OP_BN in prepare_weights_v2, or folded?
            # In current prepare_weights, BN-before-Conv emits OP_BLOCK_START + OP_BN (legacy).
            # BN not-before-Conv emits only OP_BN.
            # Here we simulate: just stash for the next Conv to flush.
            pending_bn_tensor = node["outputs"][0]
        elif op == "PRelu":
            if pending_bn_tensor:
                # Flush pending BN as legacy OP_BN (no following Conv)
                op_scales.append(per_tensor_scale.get(pending_bn_tensor, 0.0))
                pending_bn_tensor = None
            op_scales.append(per_tensor_scale.get(node["outputs"][0], 0.0))
        elif op == "Add":
            if pending_bn_tensor:
                op_scales.append(per_tensor_scale.get(pending_bn_tensor, 0.0))
                pending_bn_tensor = None
            op_scales.append(per_tensor_scale.get(node["outputs"][0], 0.0))
        elif op == "Flatten":
            if pending_bn_tensor:
                op_scales.append(per_tensor_scale.get(pending_bn_tensor, 0.0))
                pending_bn_tensor = None
            op_scales.append(per_tensor_scale.get(node["outputs"][0], 0.0))
        elif op == "Gemm":
            if pending_bn_tensor:
                op_scales.append(per_tensor_scale.get(pending_bn_tensor, 0.0))
                pending_bn_tensor = None
            op_scales.append(per_tensor_scale.get(node["outputs"][0], 0.0))
        if ni in identity_sources:
            op_scales.append(0.0)  # SAVE_IDENTITY marker

    # Flush trailing BN
    if pending_bn_tensor:
        op_scales.append(per_tensor_scale.get(pending_bn_tensor, 0.0))
        pending_bn_tensor = None

    n_ops = len(op_scales)
    print(f"n_ops derived: {n_ops}")

    with open("models/op_scales.bin", "wb") as f:
        f.write(b"OPSC")
        f.write(struct.pack("<I", n_ops))
        f.write(struct.pack("<f", input_scale))  # input scale goes first
        for s in op_scales:
            f.write(struct.pack("<f", s))
    print(f"Wrote op scales (n={n_ops}) to models/op_scales.bin  input_scale={input_scale:.4f}")

    # Diagnostic: print first few scales
    for i, s in enumerate(op_scales[:20]):
        print(f"  op[{i}] scale = {s:.4f}")


if __name__ == "__main__":
    main()
