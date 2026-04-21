"""S46 batched stable bench: 20 runs of fastface_int8_batched.exe with 3s cooldown."""
import subprocess, time, statistics


N_RUNS = 20
COOLDOWN_S = 3.0


def run_batched():
    mask = "5555"  # S48: pure P-cores, HT off (slight win vs FFFF)
    cmd = f'cmd /c start /affinity 0x{mask} /B /WAIT /HIGH .\\fastface_int8_batched.exe models\\w600k_r50_ffw4.bin --batch 8'
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=120)
    for line in r.stdout.splitlines():
        if "Best:" in line and "ms/face" in line:
            # "Best: 104.97 ms/batch = 13.12 ms/face  (76.2 face/s)"
            parts = line.split()
            # Find "ms/face" token and take the preceding number
            for i, tok in enumerate(parts):
                if tok == "ms/face":
                    return float(parts[i - 1])
    print("OUT:", r.stdout[:500])
    print("ERR:", r.stderr[:500])
    return None


def main():
    print(f"=== Running {N_RUNS} B=8 batched trials with {COOLDOWN_S}s cooldown ===\n", flush=True)
    times = []
    for i in range(N_RUNS):
        time.sleep(COOLDOWN_S)
        t = run_batched()
        if t is None:
            print(f"  run {i+1:2d}: FAIL")
            continue
        times.append(t)
        print(f"  run {i+1:2d}: {t:.2f} ms/face", flush=True)

    if not times:
        print("ALL FAILED")
        return
    ort_median = 31.5  # from prior stable benches
    print(f"\n=== B=8 Stable Bench (n={len(times)}) ===")
    print(f"  min    = {min(times):.2f} ms/face")
    print(f"  median = {statistics.median(times):.2f} ms/face")
    print(f"  mean   = {statistics.mean(times):.2f} ms/face")
    print(f"  max    = {max(times):.2f} ms/face")
    print(f"\nvs ORT {ort_median:.1f} ms/face:")
    print(f"  best-to-best:  {ort_median/min(times):.3f}x")
    print(f"  median-median: {ort_median/statistics.median(times):.3f}x")


if __name__ == "__main__":
    main()
