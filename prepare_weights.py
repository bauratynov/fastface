"""Fold BN into Conv weights + INT8 quantize + serialize for C engine.

Output layout: models/w600k_r50_int8.bin
  Header: u32 magic='FFW1', u32 n_layers
  For each layer, packed:
    u8 op_type (1=Conv, 2=PRelu, 3=Gemm, 4=Flatten, 5=Add_ref, 6=Stem_in)
    For Conv: u16 Cin, Cout, Kh, Kw, stride, pad  (little-endian)
               u8 has_bn  (1 if BN-folded bias included)
               int8 weight[Cout*Cin*Kh*Kw]
               f32 scale[Cout]    (one per output channel for dequant)
               f32 bias[Cout]
    For PRelu: u16 Cout
               f32 slope[Cout]
    For Gemm: u16 M, N, K
               int8 weight[N*K]
               f32 scale[N]
               f32 bias[N]

For feasibility Session 6 we skip the full serialization and just:
- parse + extract all conv weights + BN params
- perform offline BN fold
- INT8 quantize per-channel
- report: how many bytes, how many layers, final memory footprint
"""
import sys
sys.path.insert(0, '.')
from extract_onnx import parse_model
import numpy as np


def fold_bn_into_conv(conv_w, conv_b, gamma, beta, mean, var, eps=1e-5):
    """
    Fold BN (gamma*(x-mean)/sqrt(var+eps) + beta) into preceding Conv.
    New: w_fold[co] = w[co] * gamma[co] / sqrt(var[co] + eps)
         b_fold[co] = (b[co] - mean[co]) * gamma[co] / sqrt(var[co] + eps) + beta[co]
    """
    s = gamma / np.sqrt(var + eps)     # [Cout]
    new_w = conv_w * s.reshape(-1, 1, 1, 1)
    if conv_b is None:
        conv_b = np.zeros_like(mean)
    new_b = (conv_b - mean) * s + beta
    return new_w.astype(np.float32), new_b.astype(np.float32)


def quantize_per_channel_int8(w_fp32):
    """Symmetric per-output-channel INT8. Returns int8 weight + fp32 scales."""
    co = w_fp32.shape[0]
    absmax = np.abs(w_fp32.reshape(co, -1)).max(axis=1)
    scales = absmax / 127.0
    scales_safe = np.where(scales > 0, scales, 1.0)
    w_int = np.round(w_fp32 / scales_safe.reshape(-1, 1, 1, 1)).astype(np.int8)
    return w_int, scales.astype(np.float32)


def main():
    g = parse_model("models/w600k_r50.onnx")
    init_by_name = {t["name"]: t for t in g["initializers"]}

    # Build ordered list of Conv+BN pairs. In IResNet-50 structure,
    # each Conv is typically followed by a BN on the same output name.
    # We walk nodes in topological order and pair up.
    nodes = g["nodes"]

    # Strategy: for each Conv, find the NEXT BN in node order that consumes Conv's output.
    node_by_first_input = {}
    for i, n in enumerate(nodes):
        if n["inputs"]:
            key = n["inputs"][0]
            node_by_first_input.setdefault(key, []).append((i, n))

    conv_data = []  # list of dicts: {weight_fp32 (BN-folded), bias_fp32, scales_int8, stride, pad, Cin, Cout, Kh, Kw}
    bn_absorbed_count = 0
    total_params_fp32 = 0
    total_bytes_int8 = 0

    for ni, node in enumerate(nodes):
        if node["op_type"] != "Conv":
            continue
        wname = node["inputs"][1]
        w_tensor = init_by_name[wname]
        w = w_tensor["numpy"]  # [Cout, Cin, Kh, Kw]
        Cout, Cin, Kh, Kw = w.shape

        # Conv bias (rare in IResNet because BN follows)
        b = None
        if len(node["inputs"]) > 2 and node["inputs"][2]:
            b_t = init_by_name.get(node["inputs"][2])
            if b_t and b_t.get("numpy") is not None:
                b = b_t["numpy"]

        # Attributes
        attrs = {a["name"]: a for a in node["attrs"]}
        stride = attrs["strides"]["ints"][0] if "strides" in attrs else 1
        pads = attrs["pads"]["ints"] if "pads" in attrs else [0, 0, 0, 0]
        pad = pads[0] if pads else 0

        # Look for following BN that takes conv output
        conv_out_name = node["outputs"][0]
        followers = node_by_first_input.get(conv_out_name, [])
        bn_node = None
        for fi, fn in followers:
            if fn["op_type"] == "BatchNormalization":
                bn_node = fn
                break

        if bn_node is not None:
            gamma = init_by_name[bn_node["inputs"][1]]["numpy"]
            beta = init_by_name[bn_node["inputs"][2]]["numpy"]
            mean = init_by_name[bn_node["inputs"][3]]["numpy"]
            var = init_by_name[bn_node["inputs"][4]]["numpy"]
            eps_attr = [a for a in bn_node["attrs"] if a["name"] == "epsilon"]
            eps = eps_attr[0]["f"] if eps_attr else 1e-5
            w_folded, b_folded = fold_bn_into_conv(w, b, gamma, beta, mean, var, eps)
            bn_absorbed_count += 1
        else:
            w_folded = w.astype(np.float32)
            b_folded = b.astype(np.float32) if b is not None else np.zeros(Cout, dtype=np.float32)

        w_int, scales = quantize_per_channel_int8(w_folded)

        conv_data.append({
            "name": node["name"],
            "w_fp32_folded": w_folded,
            "b_fp32_folded": b_folded,
            "w_int8": w_int,
            "scales": scales,
            "stride": stride, "pad": pad,
            "Cin": Cin, "Cout": Cout, "Kh": Kh, "Kw": Kw,
        })
        total_params_fp32 += w_folded.size
        total_bytes_int8 += w_int.nbytes + scales.nbytes + b_folded.nbytes

    print(f"Total Conv layers: {len(conv_data)}")
    print(f"  BN-folded: {bn_absorbed_count}")
    print(f"  Raw (no BN): {len(conv_data) - bn_absorbed_count}")
    print(f"\nWeight sizes:")
    print(f"  FP32 total: {total_params_fp32 * 4 / 1024 / 1024:.1f} MB")
    print(f"  INT8 + scales + bias: {total_bytes_int8 / 1024 / 1024:.1f} MB")
    print(f"  Compression ratio: {total_params_fp32 * 4 / total_bytes_int8:.2f}x")

    print(f"\nFirst 10 Conv layers after BN-fold:")
    for i, cd in enumerate(conv_data[:10]):
        w = cd["w_int8"]
        rng_orig = (cd["w_fp32_folded"].min(), cd["w_fp32_folded"].max())
        print(f"  L{i}: {cd['Cout']}x{cd['Cin']}x{cd['Kh']}x{cd['Kw']} s={cd['stride']} p={cd['pad']} "
              f"scales[{cd['scales'].min():.4f}..{cd['scales'].max():.4f}] range={rng_orig}")

    # Serialize to binary
    out_path = "models/w600k_r50_int8.bin"
    with open(out_path, "wb") as f:
        import struct
        # Header: magic + n_layers
        f.write(b"FFW1")
        f.write(struct.pack("<I", len(conv_data)))

        # Also write PRelu layer params (separate pass — walk PRelu nodes in order)
        prelu_data = []
        for node in nodes:
            if node["op_type"] == "PRelu":
                slope_t = init_by_name[node["inputs"][1]]["numpy"]
                prelu_data.append(slope_t.astype(np.float32).flatten())
        print(f"\nPRelu layers: {len(prelu_data)}")

        # Write Conv layers (sequential, in graph order)
        for cd in conv_data:
            f.write(struct.pack("<HHHHHH",
                                cd["Cin"], cd["Cout"], cd["Kh"], cd["Kw"],
                                cd["stride"], cd["pad"]))
            f.write(cd["w_int8"].tobytes())
            f.write(cd["scales"].tobytes())
            f.write(cd["b_fp32_folded"].tobytes())

        # Number of PRelu layers + their slopes
        f.write(struct.pack("<I", len(prelu_data)))
        for s in prelu_data:
            f.write(struct.pack("<I", s.size))
            f.write(s.tobytes())

    import os
    sz = os.path.getsize(out_path) / 1024 / 1024
    print(f"\nSerialized to {out_path} ({sz:.1f} MB)")


if __name__ == "__main__":
    main()
