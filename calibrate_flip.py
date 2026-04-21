"""S97 — calibration with horizontal flip augmentation."""
import os, glob, random
import numpy as np
from PIL import Image


def load_face(path, flip=False):
    img = Image.open(path).convert("RGB")
    w, h = img.size; s = 150
    left = (w - s) // 2; top = max(0, (h - s) // 2 - 10)
    img = img.crop((left, top, left + s, top + s)).resize((112, 112), Image.BILINEAR)
    if flip:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    arr = (np.asarray(img, dtype=np.float32) - 127.5) / 127.5
    return np.transpose(arr, (2, 0, 1))[None].copy()


def load_lfw_flip(lfw_dir, n, seed):
    """Return 2*n faces: n originals + n flipped."""
    paths = sorted(glob.glob(os.path.join(lfw_dir, "**", "*.jpg"), recursive=True))
    # Force Princess first if hard-inclusion active
    princess_p = [p for p in paths if "Princess_Elisabeth_0001" in p]
    others = [p for p in paths if p not in princess_p]
    random.seed(seed); random.shuffle(others)
    picked = princess_p + others[:n - len(princess_p)]
    arrs = []
    for p in picked:
        arrs.append(load_face(p, flip=False))
        arrs.append(load_face(p, flip=True))
    return arrs, picked
