"""Compare our INT8 GEMM vs torch FP32 matmul on same shape (proxy for oneDNN ceiling)."""
import torch
import time

M, K, N = 784, 576, 128
A = torch.randn(M, K, dtype=torch.float32)
B = torch.randn(K, N, dtype=torch.float32)

torch.set_num_threads(8)

# Warmup
for _ in range(5):
    _ = A @ B

N_ITER = 200
t0 = time.perf_counter()
for _ in range(N_ITER):
    _ = A @ B
t_fp32 = (time.perf_counter() - t0) / N_ITER

gops = (2.0 * M * K * N) / 1e9
gops_s_fp32 = gops / t_fp32

print(f"torch FP32 matmul ({M}x{K}x{N}): {t_fp32*1000:.3f} ms => {gops_s_fp32:.1f} GFLOPs/s")
print()
print(f"Our INT8 GEMM (previous bench):  0.135 ms => 853 GOps/s")
print()
print(f"INT8/FP32 GOps ratio: {853.0 / gops_s_fp32:.2f}x")
print()
print(f"Note: FP32 AVX2 peak = 1075 GFLOPs/s (8 P-cores × 32 FLOPs/cycle × 4.2 GHz)")
print(f"      INT8 VNNI peak = 1075 GOps/s (same peak on AVX2+VNNI i7-13700)")
print(f"      They share port-0 fmadd/dpbusd unit")
