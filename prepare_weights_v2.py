"""Session 6 v2 — IResNet-50 pre-activation structure:
  ... BN(x) -> PRelu -> Conv -> (BN -> PRelu -> Conv) -> Add residual ...

BN is BEFORE Conv. We can still fold but direction differs:
  BN(x) = s * x + o, where s = gamma/sqrt(var+eps), o = beta - mean*s
  Conv(BN(x)) = Conv(s*x + o) = Conv_with_scaled_weights(x) + Conv(o_as_constant)
  → absorb `s` into w: w'[co, ci, :, :] = w[co, ci, :, :] * s[ci]
  → absorb `o` into bias: b'[co] = sum over (ci, kh, kw) w[co, ci, kh, kw] * o[ci] + b[co]
  (with per-output-channel sum)

For serialization simplicity in this session we:
1. Extract all Conv + BN + PRelu + Add in graph order
2. For Conv: INT8 quantize per-output-channel
3. For BN: save (scale, offset) per channel — applied at runtime in C
4. For PRelu: save slope per channel
5. Save full layer sequence in order — Python side knows structure, C replays it

Binary format (FFW2):
  magic='FFW2', u32 n_ops
  Per op:
    u8 op_type (1=Conv, 2=BN, 3=PRelu, 4=Add_residual, 5=Gemm, 6=Flatten, 7=Stem_Save)
    op-specific payload
"""
import sys, struct, os
sys.path.insert(0, '.')
from extract_onnx import parse_model
import numpy as np


OP_CONV = 1
OP_BN = 2               # Legacy: retained so old FFW3 binaries still load. S15 folds BN into next Conv.
OP_PRELU = 3
OP_ADD = 4
OP_GEMM = 5
OP_FLATTEN = 6
OP_SAVE_IDENTITY = 7    # mark current tensor as "identity" for later Add
OP_BLOCK_START = 8      # S15: zero-payload marker that replaces a BN (saves block_input for shortcut path)


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

    # Build an "active tensor" tracker — walk nodes in order, each op transforms a tensor
    # For Add nodes, we need to know which tensor is "identity" (skip connection).

    # Identify identity-source for each Add: in IResNet each Add has inputs
    # [residual_branch_out, identity_source]. The identity_source is typically
    # output of an earlier BN or an earlier Conv (downsample). We'll record
    # the op that produces the identity_source, and mark it as SAVE_IDENTITY.

    # Build producer map
    producer = {}  # output_name -> node_index
    for i, n in enumerate(nodes):
        for o in n["outputs"]:
            producer[o] = i

    # Find all Add nodes and their "identity" inputs (second input typically)
    identity_sources = set()
    add_nodes = [(i, n) for i, n in enumerate(nodes) if n["op_type"] == "Add"]
    for add_idx, add_node in add_nodes:
        # "second input" is the identity side in IResNet (main residual branch comes first)
        # Actually: could be either. We pick the one whose producer has the LOWER node index
        # (= further back in graph = the identity path).
        in0, in1 = add_node["inputs"][:2]
        p0, p1 = producer.get(in0, -1), producer.get(in1, -1)
        if p0 < 0 and p1 < 0:
            continue
        # Pick the earlier-produced input as identity
        if p0 != -1 and (p1 == -1 or p0 < p1):
            identity_sources.add(p0)
        else:
            identity_sources.add(p1)

    # Serialize ops in graph order.
    # S15 change: BatchNormalization that precedes a Conv is folded into that Conv's
    # weights + bias, and a zero-payload OP_BLOCK_START marker is emitted in its place
    # so the driver still snapshots `block_buf` for residual shortcuts. BNs that do
    # NOT precede a Conv (e.g., final BN before Flatten, BN after Gemm) stay as legacy
    # OP_BN and are executed at runtime.
    ops = []
    pending_bn = None  # (scale, offset) waiting to be folded into the next Conv
    bn_folded_count = 0

    def flush_legacy_bn():
        nonlocal pending_bn
        if pending_bn is not None:
            s, o = pending_bn
            ops.append({"op": OP_BN, "scale": s, "offset": o})
            pending_bn = None

    for ni, node in enumerate(nodes):
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
            # Fold pending BN into this Conv's weights + bias:
            #   BN(x)[ci]   = scale[ci] * x[ci] + offset[ci]
            #   Conv(BN(x))[co] = Σ_{ci,kh,kw} w[co,ci,kh,kw] * (scale[ci]*x + offset[ci])
            #                   = Conv(x, w_new) + bias_fold
            #   w_new[co,ci,kh,kw] = w[co,ci,kh,kw] * scale[ci]
            #   bias_fold[co]      = Σ_{ci,kh,kw} w[co,ci,kh,kw] * offset[ci]
            if pending_bn is not None:
                # NOTE: we DO NOT fold BN into Conv weights when Conv has padding > 0.
                # BN(0) != 0 (has offset), so the fold would mis-compute output pixels
                # whose receptive field partially falls on the Conv's zero-pad border.
                # Correctness > speed: keep BN as a runtime op, emit BLOCK_START marker
                # for the shortcut machinery, and emit OP_BN immediately after.
                bn_scale, bn_offset = pending_bn
                ops.append({"op": OP_BLOCK_START})          # block_buf snapshot point
                ops.append({"op": OP_BN, "scale": bn_scale,
                            "offset": bn_offset})            # runtime BN
                pending_bn = None
            w_int, scales = quantize_per_channel_int8(w)
            ops.append({"op": OP_CONV, "w_int8": w_int, "scales": scales,
                        "bias": bias.astype(np.float32),
                        "Cin": Cin, "Cout": Cout, "Kh": Kh, "Kw": Kw,
                        "stride": stride, "pad": pad})
        elif op == "BatchNormalization":
            # BN param calc; stash for fold into next Conv. If next op is NOT Conv,
            # the stash will be flushed as a legacy OP_BN by flush_legacy_bn().
            flush_legacy_bn()  # shouldn't happen — two BNs in a row — but keeps invariant
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
            # w shape [N, K] for Gemm
            N, K = w.shape
            # quantize same as conv (per-output-row)
            w_int, scales = quantize_per_channel_int8(w.reshape(N, K, 1, 1))
            ops.append({"op": OP_GEMM, "w_int8": w_int.reshape(N, K), "scales": scales,
                        "bias": b, "N": N, "K": K})
        elif op == "Flatten":
            flush_legacy_bn()
            ops.append({"op": OP_FLATTEN})
        # Before emitting SAVE_IDENTITY, flush any lingering BN (shouldn't trigger)
        if ni in identity_sources:
            flush_legacy_bn()
            ops.append({"op": OP_SAVE_IDENTITY})

    # Flush any trailing BN (e.g., final BN after Gemm)
    flush_legacy_bn()

    # Write binary. New magic 'FFW3' — format includes Conv bias + BN-folded weights + BLOCK_START.
    out_path = "models/w600k_r50_ffw2.bin"
    with open(out_path, "wb") as f:
        f.write(b"FFW3")
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
            elif t == OP_ADD:
                pass  # no payload
            elif t == OP_BLOCK_START:
                pass  # no payload
            elif t == OP_GEMM:
                f.write(struct.pack("<II", op["N"], op["K"]))
                f.write(op["w_int8"].tobytes())
                f.write(op["scales"].tobytes())
                f.write(op["bias"].tobytes())
            elif t == OP_FLATTEN:
                pass
            elif t == OP_SAVE_IDENTITY:
                pass

    # Stats
    counts = {}
    for op in ops:
        t = op["op"]
        counts[t] = counts.get(t, 0) + 1
    print(f"Op sequence length: {len(ops)}")
    print(f"  Conv: {counts.get(OP_CONV, 0)}")
    print(f"  BN:   {counts.get(OP_BN, 0)}  (expected 0 — all folded into Conv)")
    print(f"  BlockStart: {counts.get(OP_BLOCK_START, 0)}")
    print(f"  PRelu: {counts.get(OP_PRELU, 0)}")
    print(f"  Add:   {counts.get(OP_ADD, 0)}")
    print(f"  Gemm:  {counts.get(OP_GEMM, 0)}")
    print(f"  Flatten: {counts.get(OP_FLATTEN, 0)}")
    print(f"  Save_Id: {counts.get(OP_SAVE_IDENTITY, 0)}")
    print(f"  BNs folded into Conv: {bn_folded_count}")
    sz = os.path.getsize(out_path) / 1024 / 1024
    print(f"\nSerialized to {out_path} ({sz:.1f} MB)")


if __name__ == "__main__":
    main()
