"""Session 22 - Stable bench: pin FastFace to 8 P-cores, elevate priority,
run many trials with cooldown, compare distribution vs ORT on identical settings.
"""
import subprocess, time, os, sys, statistics
import numpy as np

N_RUNS = 20
COOLDOWN_S = 3.0

def run_fastface():
    """Single run of fastface_fp32 with high priority + P-core affinity."""
    # Windows: use start /affinity HEX /B /HIGH cmd to pin & prioritize.
    # i7-13700: P-core threads = logical CPUs 0-15 (P-cores + HT pairs).
    # We want pure 8 P-cores without HT siblings -> threads 0,2,4,6,8,10,12,14 = 0x5555
    # OR all 16 P-core threads = 0xFFFF. Try pure P-cores first.
    mask = "FFFF"  # lower 16 logical = all P-core threads with HT; 0-7 without HT = 0xFF
    cmd = f'cmd /c start /affinity 0x{mask} /B /WAIT /HIGH .\\fastface_fp32.exe'
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60)
    # Parse "Best:   XX.XX ms"
    for line in r.stdout.splitlines():
        if "Best:" in line:
            parts = line.split()
            ms = float(parts[1])
            return ms
    print("OUT:", r.stdout)
    print("ERR:", r.stderr)
    return None


def run_ort_warm(sess, nchw, n_iter=30):
    t0 = time.perf_counter()
    for _ in range(n_iter):
        sess.run(None, {sess.get_inputs()[0].name: nchw})
    return (time.perf_counter() - t0) / n_iter * 1000


def main():
    import onnxruntime as ort
    from PIL import Image
    import glob, random
    paths = sorted(glob.glob('data/lfw/**/*.jpg', recursive=True))
    random.seed(42); random.shuffle(paths)
    img = Image.open(paths[0]).convert("RGB")
    w, h = img.size; s = 150
    img = img.crop(((w-s)//2, max(0,(h-s)//2-10), (w-s)//2+s, max(0,(h-s)//2-10)+s)).resize((112,112), Image.BILINEAR)
    arr = (np.asarray(img, dtype=np.float32) - 127.5) / 127.5
    nchw = np.transpose(arr, (2, 0, 1))[None].copy()

    so = ort.SessionOptions()
    so.intra_op_num_threads = 8
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession("models/w600k_r50.onnx", so, providers=["CPUExecutionProvider"])
    # Warmup
    for _ in range(10): sess.run(None, {sess.get_inputs()[0].name: nchw})

    print(f"=== Running {N_RUNS} interleaved trials with {COOLDOWN_S}s cooldown ===\n", flush=True)
    fast_times = []
    ort_times = []
    for i in range(N_RUNS):
        time.sleep(COOLDOWN_S)
        ft = run_fastface()
        time.sleep(COOLDOWN_S)
        ot = run_ort_warm(sess, nchw, n_iter=50)
        fast_times.append(ft); ort_times.append(ot)
        print(f"  run {i+1:2d}: FastFace {ft:.2f}  ORT {ot:.2f}  ratio {ot/ft:.3f}")

    print(f"\n=== Statistics over {N_RUNS} runs ===")
    print(f"FastFace: min={min(fast_times):.2f}  median={statistics.median(fast_times):.2f}  mean={statistics.mean(fast_times):.2f}  max={max(fast_times):.2f}")
    print(f"ORT:      min={min(ort_times):.2f}  median={statistics.median(ort_times):.2f}  mean={statistics.mean(ort_times):.2f}  max={max(ort_times):.2f}")
    ratio_min = min(ort_times) / min(fast_times)
    ratio_med = statistics.median(ort_times) / statistics.median(fast_times)
    ratio_mean = statistics.mean(ort_times) / statistics.mean(fast_times)
    print(f"\nSpeedup (FastFace vs ORT):")
    print(f"  best-to-best:   {ratio_min:.3f}x")
    print(f"  median-median:  {ratio_med:.3f}x")
    print(f"  mean-mean:      {ratio_mean:.3f}x")

    wins = sum(1 for f, o in zip(fast_times, ort_times) if f < o)
    print(f"\nFastFace won in {wins}/{N_RUNS} runs  ({100*wins/N_RUNS:.0f}%)")


if __name__ == "__main__":
    main()
