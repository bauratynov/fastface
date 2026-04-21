"""Session 5: bench REAL InsightFace w600k_r50.onnx via ONNX Runtime CPU.
Dump architecture (input/output shape, conv layers) for our C port.
This establishes the HONEST baseline we must beat.
"""
import onnxruntime as ort
import numpy as np
import time
import os

MODEL = "models/w600k_r50.onnx"
sz_mb = os.path.getsize(MODEL) / 1024 / 1024
print(f"Model: {MODEL} ({sz_mb:.1f} MB)")

# Load with CPU EP, 8 threads
so = ort.SessionOptions()
so.intra_op_num_threads = 8
so.inter_op_num_threads = 1
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
so.enable_profiling = True

sess = ort.InferenceSession(MODEL, sess_options=so, providers=["CPUExecutionProvider"])

print("\n--- Model I/O ---")
for inp in sess.get_inputs():
    print(f"  input '{inp.name}': shape={inp.shape} dtype={inp.type}")
for out in sess.get_outputs():
    print(f"  output '{out.name}': shape={out.shape} dtype={out.type}")

# Prepare input
input_name = sess.get_inputs()[0].name
input_shape = [s if isinstance(s, int) else 1 for s in sess.get_inputs()[0].shape]
# InsightFace uses [N, 3, 112, 112] typically
x = np.random.randn(*input_shape).astype(np.float32)
print(f"\nInput tensor: {x.shape} {x.dtype}")

# Warm up
for _ in range(5):
    out = sess.run(None, {input_name: x})

# Bench
N_ITER = 100
t0 = time.perf_counter()
for _ in range(N_ITER):
    out = sess.run(None, {input_name: x})
t_avg = (time.perf_counter() - t0) / N_ITER

print(f"\n--- PERFORMANCE (ORT CPU, 8 threads) ---")
print(f"Avg: {t_avg*1000:.2f} ms/inference ({1/t_avg:.1f} face/s)")

# Inspect graph via onnx if available
try:
    import onnx
    m = onnx.load(MODEL)
    g = m.graph
    conv_ops = [n for n in g.node if n.op_type == "Conv"]
    total_ops = len(g.node)
    print(f"\n--- Graph inspection ---")
    print(f"Total nodes: {total_ops}")
    op_counts = {}
    for n in g.node:
        op_counts[n.op_type] = op_counts.get(n.op_type, 0) + 1
    for op, c in sorted(op_counts.items(), key=lambda x: -x[1])[:15]:
        print(f"  {op}: {c}")
    # Output conv shapes
    print(f"\n--- First 10 Conv layers (from graph) ---")
    for i, n in enumerate(conv_ops[:10]):
        inp_names = list(n.input)
        kern_name = inp_names[1]
        for init in g.initializer:
            if init.name == kern_name:
                kshape = list(init.dims)
                print(f"  Conv{i}: kernel shape {kshape}  strides={[a.ints for a in n.attribute if a.name=='strides']}  pads={[a.ints for a in n.attribute if a.name=='pads']}")
                break
except ImportError:
    print("\n(onnx package not installed; graph inspection skipped)")

# End profiler session
prof_file = sess.end_profiling()
print(f"\nProfile saved to: {prof_file}")
