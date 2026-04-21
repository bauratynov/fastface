# GPU (Vulkan) path — build & run instructions

Requires Vulkan SDK. Check if installed:
```
where glslc
where vulkaninfo
```

If not installed:
1. Download Vulkan SDK: https://vulkan.lunarg.com/sdk/home
2. Install (~1 GB, ~10 min)
3. Reboot or `source $VULKAN_SDK/setup-env.sh` / add to PATH on Windows

## Compile compute shader to SPIR-V

```
cd gpu
glslc conv3x3_s1_p1.comp -o conv3x3_s1_p1.spv
```

Generates `conv3x3_s1_p1.spv` binary that Vulkan runtime can load.

## Host-side wrapper (not yet written)

The full Vulkan host code for FastFace needs:
- Instance + device creation with a compute-capable queue
- Device memory allocation for weights (one-time upload)
- Staging buffer for input (per-inference upload ~150 KB)
- Descriptor sets binding weights + activations
- Command buffer: for each Conv → vkCmdDispatch with workgroup counts
- Memory barriers between layers
- Readback of final 512-float embedding

Estimated effort for complete pipeline:
- 1 session: host boilerplate + 1 working Conv kernel
- 2 sessions: all Conv variants (3x3 s=1 p=1, 3x3 s=2, 1x1, Gemm)
- 1 session: BN + PReLU + Add + Flatten kernels
- 1 session: memory management + optimization
- 1 session: benchmarking + comparison with ORT CUDA provider

Total: ~5 sessions of focused GPU work.

## Alternative — use ORT with GPU provider

Quickest GPU option: install `onnxruntime-gpu` (replaces CPU-only package):
```
pip install onnxruntime-gpu
```
Then in code, replace provider list:
```python
providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
sess = ort.InferenceSession('w600k_r50.onnx', providers=providers)
```

This uses ORT's own GPU kernels (CUDA 11/12 compatible) — zero FastFace GPU code needed.
Expected: **2-4 ms/face** on RTX 5060 Ti, but you pay ORT's Python dependency (~500 MB env).

The FastFace custom GPU path (Vulkan) targets:
- **Portability**: runs on NVIDIA/AMD/Intel without vendor-specific runtime
- **Smaller deployment**: ~50 MB Vulkan runtime vs 500+ MB ORT+CUDA
- **Custom kernels** tuned to our weights + Winograd

Performance expectations for custom GPU:
- b=1: ~2-3 ms/face (Vulkan overhead dominates)
- b=8: ~0.3-0.5 ms/face (GPU parallelism shines at large batches)
