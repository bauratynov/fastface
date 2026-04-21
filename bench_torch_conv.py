"""Benchmark torch.nn.functional.conv2d on the same 4 shapes our C code tested."""
import torch
import torch.nn.functional as F
import time

torch.set_num_threads(8)

shapes = [
    # (name, Cin, H, W, Cout, Kh, Kw, stride, pad)
    ("Stem 7x7 s=2 p=3",   3, 112, 112, 64, 7, 7, 2, 3),
    ("Mid 3x3 s=2 p=1",    64, 56, 56, 128, 3, 3, 2, 1),
    ("Deep 1x1 s=1 p=0",   256, 14, 14, 256, 1, 1, 1, 0),
    ("Bottle 3x3 s=1 p=1", 128, 28, 28, 128, 3, 3, 1, 1),
]

# Our measured times (from conv_bench.exe)
ours = {
    "Stem 7x7 s=2 p=3":    0.287,
    "Mid 3x3 s=2 p=1":     0.281,
    "Deep 1x1 s=1 p=0":    0.203,
    "Bottle 3x3 s=1 p=1":  0.353,
}

print(f"{'Shape':<22} {'torch FP32 ms':>14} {'ours INT8 ms':>14} {'speedup':>10}")
print("-" * 64)

for name, Cin, H, W, Cout, Kh, Kw, stride, pad in shapes:
    x = torch.randn(1, Cin, H, W, dtype=torch.float32)
    w = torch.randn(Cout, Cin, Kh, Kw, dtype=torch.float32)
    x = x.to(memory_format=torch.channels_last)

    # Warmup
    with torch.inference_mode():
        for _ in range(5):
            _ = F.conv2d(x, w, stride=stride, padding=pad)

    N_ITER = 200
    with torch.inference_mode():
        t0 = time.perf_counter()
        for _ in range(N_ITER):
            _ = F.conv2d(x, w, stride=stride, padding=pad)
        t_avg = (time.perf_counter() - t0) / N_ITER

    our_ms = ours[name]
    speedup = t_avg * 1000 / our_ms
    print(f"{name:<22} {t_avg*1000:>14.3f} {our_ms:>14.3f} {speedup:>9.2f}x")
