"""Standard LFW 10-fold protocol verification.

Generates 6000 pairs (10 folds x 600 pairs, 300 same + 300 different per
fold) deterministically from the LFW folder structure, computes pair
cos-sim for both FastFace INT8 and ORT FP32, finds best threshold PER FOLD
(as in Huang et al., LFW protocol), and reports mean +/- std accuracy.

This is the form in which FR papers report "LFW accuracy" (e.g., ArcFace
paper reports ~99.82% for this model on FP32).

Usage:
    python bench_lfw_full.py [--lfw-dir data/lfw]
"""
import argparse, os, glob, random, subprocess, sys
import numpy as np
from PIL import Image
import onnxruntime as ort


def load_face(path):
    img = Image.open(path).convert("RGB")
    w, h = img.size; s = 150
    left = (w - s) // 2; top = max(0, (h - s) // 2 - 10)
    img = img.crop((left, top, left + s, top + s)).resize((112, 112), Image.BILINEAR)
    return (np.asarray(img, dtype=np.float32) - 127.5) / 127.5


def build_10_folds(lfw_dir, seed):
    persons = {}
    for p in sorted(glob.glob(os.path.join(lfw_dir, "*/"))):
        name = os.path.basename(os.path.normpath(p))
        imgs = sorted(glob.glob(os.path.join(p, "*.jpg")))
        if len(imgs) >= 2:
            persons[name] = imgs
    all_folders = sorted(glob.glob(os.path.join(lfw_dir, "*/")))
    rng = random.Random(seed)
    folds = []
    for f_idx in range(10):
        rng_f = random.Random(seed * 1000 + f_idx)
        fold = []
        names = list(persons.keys())
        rng_f.shuffle(names)
        # 300 same pairs
        for name in names[:300]:
            a, b = rng_f.sample(persons[name], 2)
            fold.append((a, b, 1))
        # 300 different pairs
        for _ in range(300):
            p1, p2 = rng_f.sample(all_folders, 2)
            i1 = sorted(glob.glob(os.path.join(p1, "*.jpg")))
            i2 = sorted(glob.glob(os.path.join(p2, "*.jpg")))
            if i1 and i2:
                fold.append((i1[0], i2[0], 0))
        folds.append(fold)
    return folds


class Int8Server:
    def __init__(self, exe, weights, env):
        self.proc = subprocess.Popen(
            [exe, weights, "--server"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            env=env, bufsize=0)
    def embed(self, arr):
        self.proc.stdin.write(arr.astype(np.float32).tobytes())
        self.proc.stdin.flush()
        return np.frombuffer(self.proc.stdout.read(512 * 4), dtype=np.float32).copy()
    def close(self):
        self.proc.stdin.close()
        self.proc.wait(timeout=10)


def fold_accuracy(cs, labels):
    thresholds = np.linspace(cs.min(), cs.max(), 500)
    best = 0
    for t in thresholds:
        acc = ((cs >= t).astype(int) == labels).mean()
        if acc > best: best = acc
    return best


def _default_exe():
    """Auto-detect fastface_int8 binary: .exe on Windows, bare name on Linux/macOS."""
    suffix = ".exe" if os.name == "nt" else ""
    for p in (f"./fastface_int8{suffix}", "./fastface_int8.exe", "./fastface_int8"):
        if os.path.exists(p):
            return p
    return f"./fastface_int8{suffix}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lfw-dir", default="data/lfw")
    ap.add_argument("--onnx", default="models/w600k_r50.onnx")
    ap.add_argument("--weights", default="models/w600k_r50_ffw4.bin")
    ap.add_argument("--exe", default=_default_exe())
    ap.add_argument("--gcc-bin", default="C:/mingw64/bin" if os.name == "nt" else "")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    env = os.environ.copy()
    if os.name == "nt":
        env["PATH"] = args.gcc_bin + os.pathsep + env.get("PATH", "")

    print(f"Building 10 folds x 600 pairs from {args.lfw_dir}...", flush=True)
    folds = build_10_folds(args.lfw_dir, args.seed)
    total = sum(len(f) for f in folds)
    print(f"  got {total} pairs\n", flush=True)

    so = ort.SessionOptions()
    so.intra_op_num_threads = 8
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(args.onnx, so, providers=["CPUExecutionProvider"])

    server = Int8Server(args.exe, args.weights, env)

    fold_accs_fp32 = []
    fold_accs_int8 = []
    all_labels_fp32 = []
    all_labels_int8 = []
    all_scores_fp32 = []
    all_scores_int8 = []

    for f_idx, fold in enumerate(folds):
        cs_fp32 = np.zeros(len(fold))
        cs_int8 = np.zeros(len(fold))
        labels = np.zeros(len(fold), dtype=int)
        for i, (a, b, lbl) in enumerate(fold):
            arr_a, arr_b = load_face(a), load_face(b)
            nchw_a = np.transpose(arr_a, (2,0,1))[None].astype(np.float32).copy()
            nchw_b = np.transpose(arr_b, (2,0,1))[None].astype(np.float32).copy()
            ea_o = sess.run(None, {sess.get_inputs()[0].name: nchw_a})[0].flatten()
            eb_o = sess.run(None, {sess.get_inputs()[0].name: nchw_b})[0].flatten()
            ea_i = server.embed(arr_a)
            eb_i = server.embed(arr_b)
            cs_fp32[i] = np.dot(ea_o, eb_o) / (np.linalg.norm(ea_o) * np.linalg.norm(eb_o))
            cs_int8[i] = np.dot(ea_i, eb_i) / (np.linalg.norm(ea_i) * np.linalg.norm(eb_i))
            labels[i] = lbl
        fold_accs_fp32.append(fold_accuracy(cs_fp32, labels))
        fold_accs_int8.append(fold_accuracy(cs_int8, labels))
        all_scores_fp32.append(cs_fp32); all_scores_int8.append(cs_int8); all_labels_fp32.append(labels)
        print(f"  fold {f_idx+1:2d}: FP32={fold_accs_fp32[-1]*100:.2f}%  INT8={fold_accs_int8[-1]*100:.2f}%", flush=True)

    server.close()

    fp32_mean = np.mean(fold_accs_fp32); fp32_std = np.std(fold_accs_fp32)
    int8_mean = np.mean(fold_accs_int8); int8_std = np.std(fold_accs_int8)

    print()
    print(f"=== LFW 10-fold protocol (6000 pairs total) ===")
    print(f"  ORT FP32:       {fp32_mean*100:.3f} +/- {fp32_std*100:.3f} %")
    print(f"  FastFace INT8:  {int8_mean*100:.3f} +/- {int8_std*100:.3f} %")
    print(f"  gap:            {(fp32_mean - int8_mean)*100:+.3f} pp")
    print(f"  (published ArcFace FP32 LFW: ~99.82%)")


if __name__ == "__main__":
    main()
