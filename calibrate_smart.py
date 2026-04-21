"""S92 — smart calibration: force hard faces into calibration set."""
import os, glob, random
import numpy as np
from PIL import Image


# Hard faces identified from S91 calibration as worst cos-sim vs FP32.
# These improve calibration coverage for atypical distributions.
HARD_FACES = [
    "Cassandra_Heise/Cassandra_Heise_0001.jpg",
    "Federico_Trillo/Federico_Trillo_0001.jpg",
    "Florencia_Macri/Florencia_Macri_0001.jpg",
    "Princess_Elisabeth/Princess_Elisabeth_0001.jpg",
    "Mark_Heller/Mark_Heller_0001.jpg",
    "Lara_Logan/Lara_Logan_0001.jpg",
    "Matthew_Perry/Matthew_Perry_0003.jpg",
    "Penelope_Cruz/Penelope_Cruz_0002.jpg",
]


def load_face(path):
    img = Image.open(path).convert("RGB")
    w, h = img.size; s = 150
    left = (w - s) // 2; top = max(0, (h - s) // 2 - 10)
    img = img.crop((left, top, left + s, top + s)).resize((112, 112), Image.BILINEAR)
    arr = (np.asarray(img, dtype=np.float32) - 127.5) / 127.5
    return np.transpose(arr, (2, 0, 1))[None].copy()


def load_lfw_smart(lfw_dir, n, seed):
    hard_paths = []
    for rel in HARD_FACES:
        p = os.path.join(lfw_dir, rel.replace("/", os.sep))
        if os.path.exists(p):
            hard_paths.append(p)
    all_paths = sorted(glob.glob(os.path.join(lfw_dir, "**", "*.jpg"), recursive=True))
    others = [p for p in all_paths if p not in hard_paths]
    random.seed(seed); random.shuffle(others)
    picked = hard_paths + others[:n - len(hard_paths)]
    return [load_face(p) for p in picked], picked
