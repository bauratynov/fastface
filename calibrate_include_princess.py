"""S91 — calibrate with Princess_Elisabeth_0001 explicitly included.

Patch load_lfw_batch so first face is Princess, then 199 random from seed 1.
"""
import os, glob, random
import numpy as np
from PIL import Image


def load_face(path):
    img = Image.open(path).convert("RGB")
    w, h = img.size; s = 150
    left = (w - s) // 2; top = max(0, (h - s) // 2 - 10)
    img = img.crop((left, top, left + s, top + s)).resize((112, 112), Image.BILINEAR)
    arr = (np.asarray(img, dtype=np.float32) - 127.5) / 127.5
    return np.transpose(arr, (2, 0, 1))[None].copy()


def load_lfw_with_princess(lfw_dir, n, seed):
    paths = sorted(glob.glob(os.path.join(lfw_dir, "**", "*.jpg"), recursive=True))
    # Force Princess first
    princess = [p for p in paths if "Princess_Elisabeth_0001" in p]
    others = [p for p in paths if p not in princess]
    random.seed(seed); random.shuffle(others)
    picked = princess + others[:n - len(princess)]
    return [load_face(p) for p in picked], picked
