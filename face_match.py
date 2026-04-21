"""face_match.py -- 60-second FastFace demo.

Compare two face images and decide if they're the same person.

    python face_match.py a.jpg b.jpg

Outputs cosine similarity and a same/different verdict using the LFW-
calibrated threshold of 0.20.

Requires: numpy, pillow, built fastface_int8.exe, models/w600k_r50_ffw4.bin.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from PIL import Image

from fastface import FastFace

# Threshold picked from S51 LFW 1000-pair best-threshold (0.206 for INT8);
# rounded down slightly for conservative verdict.
SAME_THRESHOLD = 0.20


def load_face(path):
    img = Image.open(path).convert("RGB")
    w, h = img.size
    s = min(w, h)
    left = (w - s) // 2
    top = max(0, (h - s) // 2 - 10)
    img = img.crop((left, top, left + s, top + s)).resize((112, 112), Image.BILINEAR)
    arr = (np.asarray(img, dtype=np.float32) - 127.5) / 127.5
    return arr


def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <image_a> <image_b>")
        sys.exit(1)

    a_path, b_path = sys.argv[1:3]
    arr_a = load_face(a_path)
    arr_b = load_face(b_path)

    with FastFace() as ff:
        emb_a = ff.embed(arr_a)
        emb_b = ff.embed(arr_b)
        sim = FastFace.cos_sim(emb_a, emb_b)

    verdict = "SAME" if sim >= SAME_THRESHOLD else "DIFFERENT"
    print(f"image A: {a_path}")
    print(f"image B: {b_path}")
    print(f"cosine similarity: {sim:+.4f}")
    print(f"verdict (threshold {SAME_THRESHOLD}): {verdict}")


if __name__ == "__main__":
    main()
