"""S37 — FFW4 format: final Gemm weights pre-folded with per-channel activation scale.

Rationale:
  The Gemm's input `A[k]` (k in 0..25088) is Flatten(last_BN_output) with
  shape-before-flatten [1, 512, 7, 7]. Per S21 calibration, this tensor has a
  per-channel scale S_c[c] (c in 0..512). After our NCHW flatten, index
  k = c*49 + h*7 + w, so effective S_a[k] = S_c[k/49].

  Pre-fold: W_eff[oc, k] = W[oc, k] * S_a[k], then per-OC quantize.
  At runtime: matvec computes acc = dot(a_int, w_int_folded), output = acc * new_scale + bias.
  No A_scale needed (= 1.0) because S_a is already in w_int.

  Producer of the int8 Gemm input (the last BN before Flatten) must output int8
  values quantized with per-channel scale S_c. That's handled by the driver +
  per-channel bn_prelu_requant_int8 (S37 runtime side).

Otherwise identical to prepare_weights_v2. Format magic FFW4.
"""
import sys, struct, os
sys.path.insert(0, '.')
from extract_onnx import parse_model
from calibrate_per_channel_int8 import collect_ranges, load_lfw_batch
from calibrate_percentile_int8 import collect_per_image_absmax, build_percentile_scales
import numpy as np


OP_CONV = 1
OP_BN = 2
OP_PRELU = 3
OP_ADD = 4
OP_GEMM = 5
OP_FLATTEN = 6
OP_SAVE_IDENTITY = 7
OP_BLOCK_START = 8


def quantize_per_channel_int8(w_fp32):
    co = w_fp32.shape[0]
    absmax = np.abs(w_fp32.reshape(co, -1)).max(axis=1)
    scales = absmax / 127.0
    scales_safe = np.where(scales > 0, scales, 1.0)
    w_int = np.round(w_fp32 / scales_safe.reshape((-1,) + (1,)*(w_fp32.ndim-1))).astype(np.int8)
    return w_int, scales.astype(np.float32)


def main():
    g = parse_model("models/w600k_r50.onnx")
    init_by_name = {t["name"]: t for t in g["initializers"]}
    nodes = g["nodes"]

    # Collect per-channel activation ranges for weight fold
    n_calib = int(os.environ.get("N_CALIB", "20"))
    percentile = float(os.environ.get("PERCENTILE", "0"))
    scales_path = os.environ.get("SCALES_PATH", "")

    pc_scales_override = None
    if scales_path:
        # Load PCS8 pre-computed scales
        import struct as _struct
        with open(scales_path, 'rb') as f:
            data = f.read()
        assert data[:4] == b'PCS8'
        n_tensors = _struct.unpack_from('<I', data, 4)[0]
        off = 8
        pc_scales_override = {}
        for _ in range(n_tensors):
            nlen = _struct.unpack_from('<H', data, off)[0]; off += 2
            name = data[off:off+nlen].decode(); off += nlen
            nch = _struct.unpack_from('<I', data, off)[0]; off += 4
            arr = np.frombuffer(data[off:off+nch*4], dtype=np.float32).copy()
            off += nch*4
            pc_scales_override[name] = arr
        print(f"Loaded {len(pc_scales_override)} pre-computed scales from {scales_path}")

    if pc_scales_override is None:
        seed = int(os.environ.get("SEED", "1"))
        if os.environ.get("DIVERSE", "0") == "1":
            from calibrate_diverse import load_diverse_batch
            calib_inputs, _ = load_diverse_batch("data/lfw", n_calib, seed=seed)
        elif os.environ.get("WITH_PRINCESS", "0") == "1":
            from calibrate_include_princess import load_lfw_with_princess
            calib_inputs, _ = load_lfw_with_princess("data/lfw", n_calib, seed=seed)
        elif os.environ.get("SMART", "0") == "1":
            from calibrate_smart import load_lfw_smart
            calib_inputs, _ = load_lfw_smart("data/lfw", n_calib, seed=seed)
        elif os.environ.get("FLIP", "0") == "1":
            from calibrate_flip import load_lfw_flip
            calib_inputs, _ = load_lfw_flip("data/lfw", n_calib, seed=seed)
        else:
            calib_inputs, _ = load_lfw_batch("data/lfw", n_calib, seed=seed)
    clip_mult = float(os.environ.get("CLIP_MULT", "1.0"))
    if pc_scales_override is not None:
        def get_pc_scale(tensor_name):
            s = pc_scales_override.get(tensor_name)
            if s is None: return None
            if clip_mult < 1.0: s = s * clip_mult
            return s.astype(np.float32)
    elif percentile > 0 and percentile < 100:
        print(f"Collecting per-image absmax on {n_calib} faces (percentile={percentile})...", flush=True)
        per_image = collect_per_image_absmax(g, calib_inputs)
        dep = os.environ.get("DEPTH_EARLY_PCT", "")
        dep_val = float(dep) if dep else None
        if dep_val is not None:
            print(f"  depth-aware: early-half percentile={dep_val}, late-half={percentile}")
        pc_scales_all = build_percentile_scales(per_image, percentile=percentile,
                                                 depth_early_percentile=dep_val)
        def get_pc_scale(tensor_name):
            s = pc_scales_all.get(tensor_name)
            if s is None: return None
            if clip_mult < 1.0: s = s * clip_mult
            return s.astype(np.float32)
    else:
        print(f"Collecting per-channel ranges on {n_calib} faces (absmax)...", flush=True)
        ranges = collect_ranges(g, calib_inputs)
        def get_pc_scale(tensor_name):
            if tensor_name not in ranges:
                return None
            tmin, tmax = ranges[tensor_name]
            absmax = np.maximum(np.abs(tmin), np.abs(tmax)).astype(np.float32)
            if clip_mult < 1.0:
                absmax = absmax * clip_mult
            absmax = np.where(absmax > 0, absmax, 1e-6).astype(np.float32)
            return (absmax / 127.0).astype(np.float32)

    # Find Flatten node; its input tensor's per-channel scale is what we fold
    flatten_input = None
    for n in nodes:
        if n["op_type"] == "Flatten":
            flatten_input = n["inputs"][0]
            break
    assert flatten_input is not None, "no Flatten node"
    S_c = get_pc_scale(flatten_input)
    print(f"Flatten input tensor = {flatten_input!r}, per-channel scale shape = {S_c.shape}")

    # Build producer map + identity sources (same as v2)
    producer = {}
    for i, n in enumerate(nodes):
        for o in n["outputs"]:
            producer[o] = i
    identity_sources = set()
    for ni, n in enumerate(nodes):
        if n["op_type"] != "Add": continue
        in0, in1 = n["inputs"][:2]
        p0, p1 = producer.get(in0, -1), producer.get(in1, -1)
        if p0 < 0 and p1 < 0: continue
        if p0 != -1 and (p1 == -1 or p0 < p1): identity_sources.add(p0)
        else: identity_sources.add(p1)

    ops = []
    pending_bn = None

    def flush_legacy_bn():
        nonlocal pending_bn
        if pending_bn is not None:
            s, o = pending_bn
            ops.append({"op": OP_BN, "scale": s, "offset": o})
            pending_bn = None

    # S85: detect trailing-BN-after-Gemm pattern so we fold it into Gemm instead
    # of emitting as a runtime OP_BN. Without this fold, the runtime op acts on
    # the wrong tensor (int8 buffer after Gemm, not final_emb fp32).
    gemm_bn_fold = None  # (gamma, beta) per-output-channel if trailing BN present
    for i, n in enumerate(nodes):
        if n["op_type"] != "Gemm": continue
        # Check if the next op operates on this Gemm's output and is BN
        if i + 1 < len(nodes) and nodes[i+1]["op_type"] == "BatchNormalization" \
           and nodes[i+1]["inputs"][0] == n["outputs"][0]:
            bn = nodes[i+1]
            gamma = init_by_name[bn["inputs"][1]]["numpy"].astype(np.float32)
            beta  = init_by_name[bn["inputs"][2]]["numpy"].astype(np.float32)
            mean  = init_by_name[bn["inputs"][3]]["numpy"].astype(np.float32)
            var   = init_by_name[bn["inputs"][4]]["numpy"].astype(np.float32)
            eps = 1e-5
            for a in bn["attrs"]:
                if a["name"] == "epsilon": eps = a["f"]
            scale = gamma / np.sqrt(var + eps)
            offset = beta - mean * scale
            gemm_bn_fold = (scale.astype(np.float32), offset.astype(np.float32))
            # Disable this BN by marking its output name for skip in the walk
            bn["_s85_folded"] = True
            print(f"S85: folding trailing BN (#{i+1}) into Gemm (#{i}): "
                  f"output tensors {n['outputs'][0]} -> {bn['outputs'][0]}")

    for ni, node in enumerate(nodes):
        if node.get("_s85_folded"): continue
        op = node["op_type"]
        if op == "Conv":
            wname = node["inputs"][1]
            w = init_by_name[wname]["numpy"].astype(np.float32)
            Cout, Cin, Kh, Kw = w.shape
            attrs = {a["name"]: a for a in node["attrs"]}
            stride = attrs["strides"]["ints"][0] if "strides" in attrs else 1
            pads = attrs["pads"]["ints"] if "pads" in attrs else [0, 0, 0, 0]
            pad = pads[0] if pads else 0
            if len(node["inputs"]) > 2:
                bname = node["inputs"][2]
                btensor = init_by_name.get(bname)
                bias = btensor["numpy"].astype(np.float32) if btensor else np.zeros(Cout, np.float32)
            else:
                bias = np.zeros(Cout, np.float32)
            if pending_bn is not None:
                bn_scale, bn_offset = pending_bn
                ops.append({"op": OP_BLOCK_START})
                ops.append({"op": OP_BN, "scale": bn_scale, "offset": bn_offset})
                pending_bn = None
            # S38: fold per-channel input activation scale S_a[ci] into weights.
            # Runtime Conv input is the ONNX Conv input tensor -- either BN output
            # (if preceded by BN emitted as runtime OP_BN) or prev op output.
            conv_input_name = node["inputs"][0]
            S_a = get_pc_scale(conv_input_name)
            if S_a is not None and S_a.shape == (Cin,):
                # W_final[oc, ci, kh, kw] = W[oc, ci, kh, kw] * S_a[ci]
                w_folded = w * S_a.reshape(1, Cin, 1, 1)
            else:
                # Fallback: no per-channel data. Keep unfolded so scale[oc] matches.
                w_folded = w
            w_int, scales = quantize_per_channel_int8(w_folded)
            ops.append({"op": OP_CONV, "w_int8": w_int, "scales": scales,
                        "bias": bias.astype(np.float32),
                        "Cin": Cin, "Cout": Cout, "Kh": Kh, "Kw": Kw,
                        "stride": stride, "pad": pad})
        elif op == "BatchNormalization":
            flush_legacy_bn()
            gamma = init_by_name[node["inputs"][1]]["numpy"].astype(np.float32)
            beta  = init_by_name[node["inputs"][2]]["numpy"].astype(np.float32)
            mean  = init_by_name[node["inputs"][3]]["numpy"].astype(np.float32)
            var   = init_by_name[node["inputs"][4]]["numpy"].astype(np.float32)
            eps_attr = [a for a in node["attrs"] if a["name"] == "epsilon"]
            eps = eps_attr[0]["f"] if eps_attr else 1e-5
            scale = gamma / np.sqrt(var + eps)
            offset = beta - mean * scale
            pending_bn = (scale.astype(np.float32), offset.astype(np.float32))
        elif op == "PRelu":
            flush_legacy_bn()
            slope_t = init_by_name[node["inputs"][1]]["numpy"].astype(np.float32).flatten()
            ops.append({"op": OP_PRELU, "slope": slope_t})
        elif op == "Add":
            flush_legacy_bn()
            ops.append({"op": OP_ADD})
        elif op == "Gemm":
            flush_legacy_bn()
            w = init_by_name[node["inputs"][1]]["numpy"].astype(np.float32)
            b = init_by_name[node["inputs"][2]]["numpy"].astype(np.float32) if len(node["inputs"]) > 2 else np.zeros(w.shape[0], np.float32)
            N, K = w.shape
            # S37 FFW4: fold per-channel activation scale S_c into Gemm weights.
            # Flatten emits NCHW-order (our C flatten): k = c*49 + h*7 + w, so c_k = k // 49.
            # Spatial size inferred from K and len(S_c).
            assert len(S_c) > 0
            spatial = K // len(S_c)
            assert K == spatial * len(S_c), f"K={K} not divisible by Cout_flat={len(S_c)}"
            # S_a_flat[k] = S_c[k // spatial]  (c outer, h*w inner)
            S_a_flat = np.repeat(S_c, spatial).astype(np.float32)  # c-outer broadcast
            assert S_a_flat.shape == (K,)
            w_eff = w * S_a_flat[None, :]   # [N, K]
            b_eff = b.copy()
            # S85: fold trailing BN (y = gamma*Gemm + beta) into Gemm weights+bias
            if gemm_bn_fold is not None:
                gamma, beta = gemm_bn_fold
                # z[oc] = gamma[oc] * (W_eff[oc,:] @ A + b[oc]) + beta[oc]
                #      = (gamma[oc] * W_eff[oc,:]) @ A + (gamma[oc]*b[oc] + beta[oc])
                w_eff = w_eff * gamma[:, None]
                b_eff = b_eff * gamma + beta
                print(f"  S85 folded trailing BN into Gemm (gamma range {gamma.min():.3f}..{gamma.max():.3f})")
            w_int, scales_new = quantize_per_channel_int8(w_eff.reshape(N, K, 1, 1))
            ops.append({"op": OP_GEMM, "w_int8": w_int.reshape(N, K), "scales": scales_new,
                        "bias": b_eff.astype(np.float32), "N": N, "K": K,
                        "_ffw4_folded": True})
            print(f"  Gemm folded: N={N} K={K} spatial={spatial}  max(S_c)={S_c.max():.4f}  "
                  f"old_scale_max={np.abs(w).reshape(N,-1).max(axis=1).max()/127:.6f}  "
                  f"new_scale_max={scales_new.max():.6f}")
        elif op == "Flatten":
            flush_legacy_bn()
            ops.append({"op": OP_FLATTEN})
        if ni in identity_sources:
            flush_legacy_bn()
            ops.append({"op": OP_SAVE_IDENTITY})

    flush_legacy_bn()

    out_path = "models/w600k_r50_ffw4.bin"
    with open(out_path, "wb") as f:
        f.write(b"FFW4")
        f.write(struct.pack("<I", len(ops)))
        for op in ops:
            f.write(struct.pack("<B", op["op"]))
            t = op["op"]
            if t == OP_CONV:
                f.write(struct.pack("<HHHHHH",
                                    op["Cin"], op["Cout"], op["Kh"], op["Kw"],
                                    op["stride"], op["pad"]))
                f.write(op["w_int8"].tobytes())
                f.write(op["scales"].tobytes())
                f.write(op["bias"].tobytes())
            elif t == OP_BN:
                f.write(struct.pack("<H", op["scale"].size))
                f.write(op["scale"].tobytes())
                f.write(op["offset"].tobytes())
            elif t == OP_PRELU:
                f.write(struct.pack("<H", op["slope"].size))
                f.write(op["slope"].tobytes())
            elif t in (OP_ADD, OP_BLOCK_START, OP_FLATTEN, OP_SAVE_IDENTITY):
                pass
            elif t == OP_GEMM:
                f.write(struct.pack("<II", op["N"], op["K"]))
                f.write(op["w_int8"].tobytes())
                f.write(op["scales"].tobytes())
                f.write(op["bias"].tobytes())

    sz = os.path.getsize(out_path) / 1024 / 1024
    print(f"\nSerialized to {out_path} ({sz:.1f} MB, magic FFW4)")


if __name__ == "__main__":
    main()
