"""Minimal ONNX protobuf parser — extract graph structure + weight tensors.
Only parses what we need: NodeProto, TensorProto, AttributeProto.
Pure Python, no deps beyond stdlib + numpy.

ONNX proto3 wire format:
  tag = (field_num << 3) | wire_type
  wire_type 0 = varint, 2 = length-delimited (strings/submessages), 5 = fixed32
"""
import struct
import numpy as np


def read_varint(buf, pos):
    v = 0
    shift = 0
    while True:
        b = buf[pos]; pos += 1
        v |= (b & 0x7F) << shift
        if b < 0x80: return v, pos
        shift += 7


def read_tag(buf, pos):
    tag, pos = read_varint(buf, pos)
    return (tag >> 3), (tag & 7), pos


def read_bytes(buf, pos):
    n, pos = read_varint(buf, pos)
    return bytes(buf[pos:pos+n]), pos + n


def skip_field(buf, pos, wire):
    if wire == 0: _, pos = read_varint(buf, pos); return pos
    if wire == 1: return pos + 8
    if wire == 2: n, pos = read_varint(buf, pos); return pos + n
    if wire == 5: return pos + 4
    raise ValueError(f"unknown wire {wire}")


# ---- Parsers ----

def parse_attribute(buf):
    """Returns dict {name, type, ints, floats, i, f, s}."""
    a = {"name": "", "type": 0, "ints": [], "floats": [], "i": 0, "f": 0.0, "s": b""}
    pos = 0
    while pos < len(buf):
        fn, wire, pos = read_tag(buf, pos)
        if fn == 1 and wire == 2:  # name
            n, pos = read_varint(buf, pos); a["name"] = buf[pos:pos+n].decode(); pos += n
        elif fn == 2 and wire == 5:  # f (fixed32 float, per ONNX proto)
            a["f"] = struct.unpack_from("<f", buf, pos)[0]; pos += 4
        elif fn == 3 and wire == 0:  # i (varint int64, per ONNX proto)
            v, pos = read_varint(buf, pos)
            if v >= (1 << 63): v -= (1 << 64)
            a["i"] = v
        elif fn == 4 and wire == 2:  # s
            n, pos = read_varint(buf, pos); a["s"] = bytes(buf[pos:pos+n]); pos += n
        elif fn == 7 and wire == 2:  # floats (packed)
            n, pos = read_varint(buf, pos)
            end = pos + n
            while pos < end:
                a["floats"].append(struct.unpack_from("<f", buf, pos)[0]); pos += 4
        elif fn == 8 and wire == 2:  # ints (packed)
            n, pos = read_varint(buf, pos)
            end = pos + n
            while pos < end:
                v, pos = read_varint(buf, pos)
                if v >= (1 << 63): v -= (1 << 64)
                a["ints"].append(v)
        elif fn == 8 and wire == 0:  # single int repeated
            v, pos = read_varint(buf, pos)
            if v >= (1 << 63): v -= (1 << 64)
            a["ints"].append(v)
        elif fn == 20 and wire == 0:  # type (AttributeType enum)
            a["type"], pos = read_varint(buf, pos)
        else:
            pos = skip_field(buf, pos, wire)
    return a


def parse_node(buf):
    """Returns dict {inputs, outputs, name, op_type, attrs}."""
    n = {"inputs": [], "outputs": [], "name": "", "op_type": "", "attrs": []}
    pos = 0
    while pos < len(buf):
        fn, wire, pos = read_tag(buf, pos)
        if fn == 1 and wire == 2:  # input
            sbuf, pos = read_bytes(buf, pos); n["inputs"].append(sbuf.decode())
        elif fn == 2 and wire == 2:  # output
            sbuf, pos = read_bytes(buf, pos); n["outputs"].append(sbuf.decode())
        elif fn == 3 and wire == 2:  # name
            sbuf, pos = read_bytes(buf, pos); n["name"] = sbuf.decode()
        elif fn == 4 and wire == 2:  # op_type
            sbuf, pos = read_bytes(buf, pos); n["op_type"] = sbuf.decode()
        elif fn == 5 and wire == 2:  # attribute
            sbuf, pos = read_bytes(buf, pos); n["attrs"].append(parse_attribute(sbuf))
        else:
            pos = skip_field(buf, pos, wire)
    return n


def parse_tensor(buf):
    """Returns dict with dims, data_type, name, numpy_data."""
    t = {"dims": [], "data_type": 0, "name": "", "raw_data": b"", "float_data": [], "int32_data": [], "int64_data": []}
    pos = 0
    while pos < len(buf):
        fn, wire, pos = read_tag(buf, pos)
        if fn == 1 and wire == 0:  # dims single
            v, pos = read_varint(buf, pos)
            if v >= (1 << 63): v -= (1 << 64)
            t["dims"].append(v)
        elif fn == 1 and wire == 2:  # dims packed
            n, pos = read_varint(buf, pos); end = pos + n
            while pos < end:
                v, pos = read_varint(buf, pos)
                if v >= (1 << 63): v -= (1 << 64)
                t["dims"].append(v)
        elif fn == 2 and wire == 0:  # data_type
            t["data_type"], pos = read_varint(buf, pos)
        elif fn == 8 and wire == 2:  # name
            sbuf, pos = read_bytes(buf, pos); t["name"] = sbuf.decode()
        elif fn == 9 and wire == 2:  # raw_data
            n, pos = read_varint(buf, pos); t["raw_data"] = bytes(buf[pos:pos+n]); pos += n
        elif fn == 4 and wire == 2:  # float_data packed
            n, pos = read_varint(buf, pos); end = pos + n
            while pos < end:
                t["float_data"].append(struct.unpack_from("<f", buf, pos)[0]); pos += 4
        else:
            pos = skip_field(buf, pos, wire)

    # Materialize as numpy
    DTYPE_MAP = {1: np.float32, 2: np.uint8, 3: np.int8, 4: np.uint16, 5: np.int16,
                 6: np.int32, 7: np.int64, 9: np.bool_, 10: np.float16, 11: np.float64}
    dt = DTYPE_MAP.get(t["data_type"])
    if t["raw_data"] and dt is not None:
        t["numpy"] = np.frombuffer(t["raw_data"], dtype=dt).copy().reshape(t["dims"]) if t["dims"] else np.frombuffer(t["raw_data"], dtype=dt).copy()
    elif t["float_data"]:
        t["numpy"] = np.array(t["float_data"], dtype=np.float32).reshape(t["dims"]) if t["dims"] else np.array(t["float_data"], dtype=np.float32)
    else:
        t["numpy"] = None
    return t


def parse_graph(buf):
    """Returns dict with nodes, initializers, inputs, outputs."""
    g = {"nodes": [], "initializers": [], "name": ""}
    pos = 0
    while pos < len(buf):
        fn, wire, pos = read_tag(buf, pos)
        if fn == 1 and wire == 2:  # node
            sbuf, pos = read_bytes(buf, pos); g["nodes"].append(parse_node(sbuf))
        elif fn == 2 and wire == 2:  # name
            sbuf, pos = read_bytes(buf, pos); g["name"] = sbuf.decode()
        elif fn == 5 and wire == 2:  # initializer
            sbuf, pos = read_bytes(buf, pos); g["initializers"].append(parse_tensor(sbuf))
        else:
            pos = skip_field(buf, pos, wire)
    return g


def parse_model(path):
    """ModelProto.graph is field 7."""
    with open(path, "rb") as f:
        buf = memoryview(f.read())
    g = None
    pos = 0
    while pos < len(buf):
        fn, wire, pos = read_tag(buf, pos)
        if fn == 7 and wire == 2:  # graph
            sbuf, pos = read_bytes(buf, pos)
            g = parse_graph(sbuf)
        else:
            pos = skip_field(buf, pos, wire)
    return g


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "models/w600k_r50.onnx"
    print(f"Parsing {path}...")
    g = parse_model(path)

    print(f"Graph: {g['name']}")
    print(f"Nodes: {len(g['nodes'])}")
    print(f"Initializers: {len(g['initializers'])}")

    # Op histogram
    op_counts = {}
    for n in g["nodes"]:
        op_counts[n["op_type"]] = op_counts.get(n["op_type"], 0) + 1
    print(f"\nOp histogram:")
    for op, c in sorted(op_counts.items(), key=lambda x: -x[1]):
        print(f"  {op}: {c}")

    # List Conv layers with their weight shapes
    print(f"\nConv layers:")
    init_by_name = {t["name"]: t for t in g["initializers"]}
    convs = [n for n in g["nodes"] if n["op_type"] == "Conv"]
    for i, n in enumerate(convs[:15]):
        wname = n["inputs"][1]
        w = init_by_name.get(wname)
        wshape = w["dims"] if w else "?"
        attrs = {a["name"]: (a["ints"] if a["ints"] else (a["i"] if a["i"] else a["f"])) for a in n["attrs"]}
        print(f"  Conv{i}: w={wshape} strides={attrs.get('strides')} pads={attrs.get('pads')}")

    print(f"\nTotal Conv layers: {len(convs)}")
    print(f"Initializer total params: {sum(np.prod(t['dims']) if t['dims'] else 0 for t in g['initializers']):,}")
