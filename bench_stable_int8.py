"""S31 - Stable bench for INT8 path vs ORT. 20 interleaved trials,
3s cooldown, P-core affinity HIGH priority, same protocol as bench_stable.py.
"""
import subprocess, time, statistics
import numpy as np

N_RUNS = 20
COOLDOWN_S = 3.0


def run_fastface_int8():
    mask = "FFFF"
    cmd = f'cmd /c start /affinity 0x{mask} /B /WAIT /HIGH .\\fastface_int8.exe'
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60)
    for line in r.stdout.splitlines():
        if "Best:" in line:
            parts = line.split()
            return float(parts[1])
    print("OUT:", r.stdout); print("ERR:", r.stderr)
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
    for _ in range(10): sess.run(None, {sess.get_inputs()[0].name: nchw})

    print(f"=== Running {N_RUNS} interleaved INT8 trials with {COOLDOWN_S}s cooldown ===\n", flush=True)
    fast_times = []
    ort_times = []
    for i in range(N_RUNS):
        time.sleep(COOLDOWN_S)
        ft = run_fastface_int8()
        time.sleep(COOLDOWN_S)
        ot = run_ort_warm(sess, nchw, n_iter=50)
        fast_times.append(ft); ort_times.append(ot)
        print(f"  run {i+1:2d}: INT8 {ft:.2f}  ORT {ot:.2f}  ratio {ot/ft:.3f}", flush=True)

    print(f"\n=== Statistics over {N_RUNS} runs ===")
    print(f"INT8: min={min(fast_times):.2f}  median={statistics.median(fast_times):.2f}  mean={statistics.mean(fast_times):.2f}  max={max(fast_times):.2f}")
    print(f"ORT:  min={min(ort_times):.2f}  median={statistics.median(ort_times):.2f}  mean={statistics.mean(ort_times):.2f}  max={max(ort_times):.2f}")
    ratio_min = min(ort_times) / min(fast_times)
    ratio_med = statistics.median(ort_times) / statistics.median(fast_times)
    ratio_mean = statistics.mean(ort_times) / statistics.mean(fast_times)
    print(f"\nSpeedup (INT8 vs ORT):")
    print(f"  best-to-best:   {ratio_min:.3f}x")
    print(f"  median-median:  {ratio_med:.3f}x")
    print(f"  mean-mean:      {ratio_mean:.3f}x")

    wins = sum(1 for f, o in zip(fast_times, ort_times) if f < o)
    print(f"\nINT8 won in {wins}/{N_RUNS} runs  ({100*wins/N_RUNS:.0f}%)")


if __name__ == "__main__":
    main()
