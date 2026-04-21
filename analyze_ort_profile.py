"""Analyze ORT profile JSON — what ops consume time in w600k_r50?"""
import json
import glob
import os

# Find most recent profile
profs = sorted(glob.glob("onnxruntime_profile__*.json"), key=os.path.getmtime)
if not profs:
    print("No profile file")
    raise SystemExit
prof = profs[-1]
print(f"Analyzing: {prof}")

with open(prof) as f:
    data = json.load(f)

# ORT profiler events are a list of dicts with args like { "op_name": ..., "dur": microseconds }
op_time = {}  # op_type -> total microseconds
op_count = {}
total = 0
for e in data:
    if e.get("cat") == "Node" or e.get("cat") == "node":
        op = e.get("args", {}).get("op_name", "unknown")
        dur = e.get("dur", 0)
        op_time[op] = op_time.get(op, 0) + dur
        op_count[op] = op_count.get(op, 0) + 1
        total += dur

print(f"\nTotal kernel time across all runs: {total/1000:.1f} ms")
print(f"\n{'Op':<20} {'count':>6} {'total ms':>10} {'%':>6} {'us/call':>10}")
print("-" * 54)
for op, t in sorted(op_time.items(), key=lambda x: -x[1]):
    c = op_count[op]
    pct = 100 * t / total if total else 0
    print(f"{op:<20} {c:>6} {t/1000:>10.2f} {pct:>5.1f}% {t/c:>10.1f}")
