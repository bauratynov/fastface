"""S69 — sustained-load benchmark.

Push N embeddings through fastface_int8.exe --server continuously and
report per-call latency distribution. Detects thermal throttling or
startup/steady-state regressions over long runs.

Usage: python bench_sustained.py [--n 10000] [--batch 1]
"""
import argparse, os, subprocess, sys, time
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from fastface import FastFace


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--report-every", type=int, default=1000)
    args = ap.parse_args()

    # Load a golden face (any real face will do; distribution is realistic)
    inp_bytes = open(os.path.join(ROOT, "tests", "golden_input.bin"), "rb").read()
    arr = np.frombuffer(inp_bytes, dtype=np.float32).reshape(112, 112, 3).copy()

    print(f"Sustained load: {args.n} embeddings, batch={args.batch}, threads={args.threads}", flush=True)
    print(f"{'iter':>8s}  {'latency_ms':>10s}  {'window_avg_ms':>14s}  {'window_face_s':>13s}")

    lats = np.zeros(args.n, dtype=np.float64)
    with FastFace(batch=args.batch, threads=args.threads) as ff:
        # Warm up briefly
        for _ in range(5):
            if args.batch == 1:
                ff.embed(arr)
            else:
                ff.embed_batch([arr] * args.batch)

        t0_total = time.perf_counter()
        for i in range(args.n):
            t0 = time.perf_counter()
            if args.batch == 1:
                ff.embed(arr)
            else:
                ff.embed_batch([arr] * args.batch)
            lats[i] = (time.perf_counter() - t0) * 1000.0 / args.batch
            if (i + 1) % args.report_every == 0:
                window = lats[i + 1 - args.report_every : i + 1]
                win_avg = window.mean()
                print(f"{i+1:>8d}  {lats[i]:>10.2f}  {win_avg:>14.2f}  {1000.0/win_avg:>13.0f}",
                      flush=True)
        total = (time.perf_counter() - t0_total)

    total_faces = args.n * args.batch
    print()
    print(f"=== Statistics over {args.n} iterations ({total_faces} faces) ===")
    print(f"  total wall: {total:.1f} s  ({total_faces/total:.1f} face/s sustained)")
    print(f"  min:    {lats.min():.2f} ms")
    print(f"  median: {np.median(lats):.2f} ms")
    print(f"  mean:   {lats.mean():.2f} ms")
    print(f"  p95:    {np.percentile(lats, 95):.2f} ms")
    print(f"  p99:    {np.percentile(lats, 99):.2f} ms")
    print(f"  p99.9:  {np.percentile(lats, 99.9):.2f} ms")
    print(f"  max:    {lats.max():.2f} ms")
    # First-quarter vs last-quarter to detect thermal drift
    q = args.n // 4
    first = lats[:q].mean()
    last  = lats[-q:].mean()
    print(f"  first 25% mean: {first:.2f} ms")
    print(f"  last  25% mean: {last:.2f} ms  (drift: {last-first:+.2f} ms)")


if __name__ == "__main__":
    main()
