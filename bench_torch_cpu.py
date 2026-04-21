"""
Session 1 baseline: measure torch CPU (MKL/oneDNN) on ResNet-50 ArcFace.

Why torch CPU instead of ORT: torch 2.6 uses oneDNN backend for conv2d on x86,
same as ONNX Runtime's CPU EP. The performance ceiling is essentially identical.
ORT has marginally less Python overhead per inference but the underlying kernel
compute is oneDNN in both cases.

Outputs:
1. End-to-end latency per inference
2. Per-layer breakdown via torch.profiler (identifies the hot ops)
3. Throughput at batch=1 and batch=8

This establishes the CEILING we need to beat with Rust AVX2.
"""
import sys
sys.path.insert(0, '.')
import torch
from torch.profiler import profile, record_function, ProfilerActivity
from build_arcface import ResNet50ArcFace
import time


def warmup(model, x, n=5):
    for _ in range(n):
        _ = model(x)


def bench(model, x, n=20, label=""):
    t0 = time.perf_counter()
    for _ in range(n):
        _ = model(x)
    dt = (time.perf_counter() - t0) / n
    print(f"{label}: {dt*1000:.2f} ms/iter ({1/dt:.1f} inf/s)")
    return dt


def main():
    # Force CPU, limit threads for deterministic measurement
    torch.set_num_threads(8)  # Match BinaryAI -t 8 baseline
    torch.set_num_interop_threads(1)

    print("=" * 60)
    print(f"FastFace Session 1: ResNet-50 ArcFace baseline (torch CPU)")
    print(f"torch: {torch.__version__}")
    print(f"MKLDNN enabled: {torch.backends.mkldnn.is_available()}")
    print(f"threads: {torch.get_num_threads()}")
    print("=" * 60)

    model = ResNet50ArcFace(embedding_dim=512).eval()
    model = model.to(memory_format=torch.channels_last)

    # FP32 baseline at batch=1 (typical face rec deployment)
    x1 = torch.randn(1, 3, 112, 112).to(memory_format=torch.channels_last)
    with torch.inference_mode():
        warmup(model, x1, n=10)
        print()
        print("--- FP32 batch=1 (typical face rec) ---")
        t_fp32_b1 = bench(model, x1, n=30, label="FP32 b=1")

        print()
        print("--- FP32 batch=8 (batched face rec) ---")
        x8 = torch.randn(8, 3, 112, 112).to(memory_format=torch.channels_last)
        warmup(model, x8, n=3)
        t_fp32_b8 = bench(model, x8, n=20, label="FP32 b=8")

    # Quantize to INT8 (production deployment)
    print()
    print("--- INT8 quantized batch=1 (production setting) ---")
    try:
        import torch.ao.quantization as tq
        q_model = ResNet50ArcFace(embedding_dim=512).eval()
        q_model = tq.quantize_dynamic(q_model, {torch.nn.Linear}, dtype=torch.qint8)
        q_model = q_model.eval()
        with torch.inference_mode():
            warmup(q_model, torch.randn(1, 3, 112, 112), n=5)
            t_int8_b1 = bench(q_model, torch.randn(1, 3, 112, 112), n=20, label="INT8 dynamic b=1")
    except Exception as e:
        print(f"INT8 bench skipped: {e}")
        t_int8_b1 = None

    # Per-op profiling
    print()
    print("--- Per-op breakdown (profiler) ---")
    with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        with torch.inference_mode():
            for _ in range(10):
                _ = model(x1)

    key_avg = prof.key_averages()
    total = sum(e.self_cpu_time_total for e in key_avg)
    print(f"Total self CPU: {total/1000:.1f} ms (across 10 iter)")
    print()
    print(f"{'Op name':<40} {'Self CPU %':>10} {'Self ms':>12} {'#calls':>8}")
    print("-" * 74)
    sorted_events = sorted(key_avg, key=lambda e: -e.self_cpu_time_total)
    for e in sorted_events[:15]:
        name = e.key[:38]
        pct = 100.0 * e.self_cpu_time_total / total if total > 0 else 0
        print(f"{name:<40} {pct:>10.1f} {e.self_cpu_time_total/1000:>12.2f} {e.count:>8}")

    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"FP32 b=1: {t_fp32_b1*1000:.2f} ms  ({1/t_fp32_b1:.1f} face/s)")
    print(f"FP32 b=8: {t_fp32_b8*1000:.2f} ms / batch  → {8/t_fp32_b8:.1f} face/s")
    if t_int8_b1 is not None:
        print(f"INT8 b=1: {t_int8_b1*1000:.2f} ms  ({1/t_int8_b1:.1f} face/s)")
    print()
    print("Baseline established. Rust must beat FP32 b=1 ({:.2f} ms) to win.".format(t_fp32_b1*1000))


if __name__ == "__main__":
    main()
