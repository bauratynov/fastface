"""S36 phase A — Export PER-CHANNEL op output scales in op-order.

Difference vs export_op_scales.py: emits a f32[Cout] array per op (not a scalar).
Enables per-channel activation requantization in the C driver, which S21
Python simulation showed lifts cos-sim 0.965 -> 0.986 for symmetric per-channel.

Format (OPSC2):
  magic  'OPSC2'
  u32 n_ops
  u32 n_input_ch, f32[n_input_ch] input_scales
  for each op:
    u32 n_channels (0 for marker ops like BLOCK_START / SAVE_ID / FLATTEN)
    f32[n_channels] per-channel activation scale
"""
import sys, os, struct, time
sys.path.insert(0, '.')
from extract_onnx import parse_model
from calibrate_per_channel_int8 import collect_ranges, load_lfw_batch
from calibrate_percentile_int8 import collect_per_image_absmax, build_percentile_scales
import numpy as np


def load_pcs8_scales(path):
    """Load a PCS8 per-channel scales file into {tensor_name: np.float32[nch]}."""
    with open(path, 'rb') as f:
        data = f.read()
    assert data[:4] == b'PCS8'
    n = struct.unpack_from('<I', data, 4)[0]
    off = 8
    out = {}
    for _ in range(n):
        nlen = struct.unpack_from('<H', data, off)[0]; off += 2
        name = data[off:off+nlen].decode(); off += nlen
        nch = struct.unpack_from('<I', data, off)[0]; off += 4
        arr = np.frombuffer(data[off:off+nch*4], dtype=np.float32).copy()
        off += nch*4
        out[name] = arr
    return out


def main():
    n_calib = int(os.environ.get("N_CALIB", "20"))
    percentile = float(os.environ.get("PERCENTILE", "0"))  # 0 = disabled = absmax
    scales_path = os.environ.get("SCALES_PATH", "")
    if scales_path:
        print(f"Loading pre-computed scales from {scales_path}...")
        per_channel_scale = load_pcs8_scales(scales_path)
        g = parse_model("models/w600k_r50.onnx")
        # Skip calibration entirely, jump to graph walk
    elif os.environ.get("DIVERSE", "0") == "1":
        from calibrate_diverse import load_diverse_batch
        calib_inputs, _ = load_diverse_batch("data/lfw", n_calib, seed=int(os.environ.get("SEED", "1")))
    elif os.environ.get("WITH_PRINCESS", "0") == "1":
        from calibrate_include_princess import load_lfw_with_princess
        calib_inputs, _ = load_lfw_with_princess("data/lfw", n_calib, seed=int(os.environ.get("SEED", "1")))
    elif os.environ.get("SMART", "0") == "1":
        from calibrate_smart import load_lfw_smart
        calib_inputs, _ = load_lfw_smart("data/lfw", n_calib, seed=int(os.environ.get("SEED", "1")))
    elif os.environ.get("FLIP", "0") == "1":
        from calibrate_flip import load_lfw_flip
        calib_inputs, _ = load_lfw_flip("data/lfw", n_calib, seed=int(os.environ.get("SEED", "1")))
    else:
        calib_inputs, _ = load_lfw_batch("data/lfw", n_calib, seed=int(os.environ.get("SEED", "1")))
    clip_mult = float(os.environ.get("CLIP_MULT", "1.0"))
    if scales_path:
        # Pre-loaded scales — g already set above, skip calibration
        print(f"  loaded {len(per_channel_scale)} tensor scales")
        t0 = time.perf_counter()
    else:
        print(f"Calibrating on {n_calib} faces... (percentile={percentile or 'absmax'})", flush=True)
        g = parse_model("models/w600k_r50.onnx")
        t0 = time.perf_counter()

    if scales_path:
        pass  # already loaded above
    elif percentile > 0 and percentile < 100:
        # S42: percentile-based per-channel calibration over per-image absmax
        per_image = collect_per_image_absmax(g, calib_inputs)
        print(f"  collected per-image absmax in {time.perf_counter()-t0:.1f}s ({len(per_image)} tensors)")
        dep = os.environ.get("DEPTH_EARLY_PCT", "")
        dep_val = float(dep) if dep else None
        if dep_val is not None:
            print(f"  depth-aware: early-half percentile={dep_val}, late-half={percentile}")
        per_channel_scale = build_percentile_scales(per_image, percentile=percentile,
                                                    depth_early_percentile=dep_val)
        if clip_mult < 1.0:
            for k in per_channel_scale:
                per_channel_scale[k] = per_channel_scale[k] * clip_mult
    else:
        ranges = collect_ranges(g, calib_inputs)
        print(f"  done in {time.perf_counter()-t0:.1f}s ({len(ranges)} tensors)")
        per_channel_scale = {}
        for name, (tmin, tmax) in ranges.items():
            absmax = np.maximum(np.abs(tmin), np.abs(tmax)).astype(np.float32)
            if clip_mult < 1.0:
                absmax = absmax * clip_mult
            absmax = np.where(absmax > 0, absmax, 1e-6).astype(np.float32)
            per_channel_scale[name] = (absmax / 127.0).astype(np.float32)

    # Walk the ONNX graph, reproducing the FFW4 op sequence (same heuristic
    # as prepare_weights_v3.py, including the S85 trailing-BN-after-Gemm fold).
    nodes = g["nodes"]
    producer = {o: i for i, n in enumerate(nodes) for o in n["outputs"]}
    identity_sources = set()
    for i, n in enumerate(nodes):
        if n["op_type"] != "Add": continue
        in0, in1 = n["inputs"][:2]
        p0, p1 = producer.get(in0, -1), producer.get(in1, -1)
        if p0 < 0 and p1 < 0: continue
        if p0 != -1 and (p1 == -1 or p0 < p1): identity_sources.add(p0)
        else: identity_sources.add(p1)

    # S85: same trailing-BN-after-Gemm skip as prepare_weights_v3.py
    skip_nodes = set()
    for i, n in enumerate(nodes):
        if n["op_type"] == "Gemm" and i + 1 < len(nodes) \
           and nodes[i+1]["op_type"] == "BatchNormalization" \
           and nodes[i+1]["inputs"][0] == n["outputs"][0]:
            skip_nodes.add(i + 1)

    # op-order scales: list of numpy arrays (or None for marker ops).
    op_scales = []
    pending_bn = None
    for ni, node in enumerate(nodes):
        if ni in skip_nodes: continue
        op = node["op_type"]
        if op == "Conv":
            if pending_bn:
                op_scales.append(None)  # BLOCK_START marker
                op_scales.append(per_channel_scale.get(pending_bn, None))  # legacy OP_BN emit
                pending_bn = None
            op_scales.append(per_channel_scale.get(node["outputs"][0], None))
        elif op == "BatchNormalization":
            pending_bn = node["outputs"][0]
        elif op in ("PRelu", "Add", "Flatten", "Gemm"):
            if pending_bn:
                op_scales.append(per_channel_scale.get(pending_bn, None))
                pending_bn = None
            op_scales.append(per_channel_scale.get(node["outputs"][0], None))
        if ni in identity_sources:
            op_scales.append(None)  # SAVE_IDENTITY marker (no output tensor)

    if pending_bn:
        op_scales.append(per_channel_scale.get(pending_bn, None))
        pending_bn = None

    n_ops = len(op_scales)
    n_emitted = sum(1 for s in op_scales if s is not None)
    print(f"n_ops derived: {n_ops}  (with scale: {n_emitted}, markers: {n_ops-n_emitted})")

    input_scale = per_channel_scale.get("input.1", np.array([1.0, 1.0, 1.0], dtype=np.float32))

    out_path = "models/op_scales_v2.bin"
    with open(out_path, "wb") as f:
        f.write(b"OPSC2")
        f.write(struct.pack("<I", n_ops))
        f.write(struct.pack("<I", len(input_scale)))
        f.write(input_scale.astype(np.float32).tobytes())
        for s in op_scales:
            if s is None:
                f.write(struct.pack("<I", 0))
            else:
                f.write(struct.pack("<I", len(s)))
                f.write(s.astype(np.float32).tobytes())
    sz = os.path.getsize(out_path)
    print(f"Wrote per-channel op scales (n_ops={n_ops}) to {out_path}  size={sz} bytes")

    # Diagnostic print
    print("\nFirst 15 ops:")
    for i, s in enumerate(op_scales[:15]):
        if s is None:
            print(f"  op[{i}]: <marker>")
        else:
            print(f"  op[{i}]: Cout={len(s)}  scale[0]={s[0]:.5f}  max={s.max():.5f}  min>0={s[s>0].min() if (s>0).any() else 0:.5f}")


if __name__ == "__main__":
    main()
