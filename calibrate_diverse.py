"""S50 — brightness-diverse calibration set.

Scan LFW, bucket by mean pixel brightness, pick N/2 low-brightness + N/2
high-brightness to ensure outlier robustness. Used as a drop-in for
load_lfw_batch when env DIVERSE=1 is set.
"""
import os, glob, random
import numpy as np
from PIL import Image


def load_face_arr(path):
    img = Image.open(path).convert("RGB")
    w, h = img.size; s = 150
    left = (w - s) // 2; top = max(0, (h - s) // 2 - 10)
    img = img.crop((left, top, left + s, top + s)).resize((112, 112), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32)
    mean_brightness = float(arr.mean())
    arr_norm = (arr - 127.5) / 127.5
    return np.transpose(arr_norm, (2, 0, 1))[None].copy(), mean_brightness


def load_diverse_batch(lfw_dir, n_total, seed=1):
    paths = sorted(glob.glob(os.path.join(lfw_dir, "**", "*.jpg"), recursive=True))
    random.seed(seed); random.shuffle(paths)
    # Quick scan 3x n_total to bucket
    scan_n = min(3 * n_total + 100, len(paths))
    scan_paths = paths[:scan_n]
    bucketed = []  # (mean_brightness, path, arr)
    for p in scan_paths:
        arr, mb = load_face_arr(p)
        bucketed.append((mb, p, arr))
    bucketed.sort(key=lambda x: x[0])
    # Take lowest n/2 and highest n/2
    half = n_total // 2
    low = bucketed[:half]
    high = bucketed[-half:]
    mid = bucketed[len(bucketed)//2 - half//2 : len(bucketed)//2 + half//2]
    # Mix: low + mid + high
    take_low  = low[:n_total // 3]
    take_high = high[:n_total // 3]
    take_mid  = mid[:n_total - len(take_low) - len(take_high)]
    selected = take_low + take_mid + take_high
    random.shuffle(selected)
    arrs = [a for _, _, a in selected]
    picked_paths = [p for _, p, _ in selected]
    print(f"Diverse set: low_brightness avg={np.mean([x[0] for x in take_low]):.1f}, "
          f"mid={np.mean([x[0] for x in take_mid]):.1f}, "
          f"high={np.mean([x[0] for x in take_high]):.1f}", flush=True)
    return arrs, picked_paths
