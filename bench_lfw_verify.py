"""FastFace LFW verification benchmark.

Compares FastFace INT8 vs ORT FP32 reference on standard face verification
task over LFW pairs. Reports best-threshold accuracy, TAR@FAR, and AUC.

Usage:
    python bench_lfw_verify.py [--n-pairs 1000] [--seed 42]

Requires:
    data/lfw/<person>/<image>.jpg  (LFW dataset)
    models/w600k_r50.onnx          (ORT reference)
    models/w600k_r50_ffw4.bin      (FastFace weights, via prepare_weights_v3.py)
    fastface_int8.exe              (built from arcface_forward_int8.c)
"""
import argparse, os, sys, glob, random, subprocess
import numpy as np
from PIL import Image
import onnxruntime as ort


def load_face(path):
    img = Image.open(path).convert("RGB")
    w, h = img.size; s = 150
    left = (w - s) // 2; top = max(0, (h - s) // 2 - 10)
    img = img.crop((left, top, left + s, top + s)).resize((112, 112), Image.BILINEAR)
    return (np.asarray(img, dtype=np.float32) - 127.5) / 127.5  # HWC [-1, 1]


def build_pairs(lfw_dir, n_each, seed):
    persons = {}
    for p in sorted(glob.glob(os.path.join(lfw_dir, "*/"))):
        name = os.path.basename(os.path.normpath(p))
        imgs = sorted(glob.glob(os.path.join(p, "*.jpg")))
        if len(imgs) >= 2:
            persons[name] = imgs
    all_folders = sorted(glob.glob(os.path.join(lfw_dir, "*/")))
    rng = random.Random(seed)
    same, diff = [], []
    names_multi = list(persons.keys())
    rng.shuffle(names_multi)
    for name in names_multi[:n_each]:
        a, b = rng.sample(persons[name], 2)
        same.append((a, b, 1))
    for _ in range(n_each):
        p1, p2 = rng.sample(all_folders, 2)
        i1 = sorted(glob.glob(os.path.join(p1, "*.jpg")))
        i2 = sorted(glob.glob(os.path.join(p2, "*.jpg")))
        if i1 and i2:
            diff.append((i1[0], i2[0], 0))
    return same + diff


def embed_ort(sess, arr):
    nchw = np.transpose(arr, (2, 0, 1))[None].astype(np.float32).copy()
    return sess.run(None, {sess.get_inputs()[0].name: nchw})[0].flatten()


class Int8Server:
    """Persistent --server subprocess; write one face of fp32 NHWC, read 512 fp32 emb."""
    def __init__(self, exe, weights, env):
        self.proc = subprocess.Popen(
            [exe, weights, "--server"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            env=env, bufsize=0)
    def embed(self, arr):
        buf = arr.astype(np.float32).tobytes()
        self.proc.stdin.write(buf)
        self.proc.stdin.flush()
        out = self.proc.stdout.read(512 * 4)
        return np.frombuffer(out, dtype=np.float32).copy()
    def close(self):
        self.proc.stdin.close()
        self.proc.wait(timeout=5)


def best_threshold_accuracy(scores, labels):
    thresholds = np.linspace(scores.min(), scores.max(), 1000)
    best, best_t = 0.0, 0.0
    for t in thresholds:
        pred = (scores >= t).astype(int)
        acc = (pred == labels).mean()
        if acc > best:
            best, best_t = acc, t
    return best, best_t


def tar_at_far(scores, labels, target_far):
    pos_scores = scores[labels == 1]
    neg_scores = scores[labels == 0]
    n_neg = len(neg_scores)
    k = max(1, int(target_far * n_neg))
    threshold = np.sort(neg_scores)[::-1][k - 1]
    tar = (pos_scores >= threshold).mean()
    actual_far = (neg_scores >= threshold).mean()
    return tar, actual_far, threshold


def auc_score(scores, labels):
    order = np.argsort(scores)[::-1]
    lbls = labels[order]
    n_pos = lbls.sum(); n_neg = len(lbls) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    tp = 0; acc = 0.0
    for l in lbls:
        if l == 1:
            tp += 1
        else:
            acc += tp / (n_pos * n_neg)
    return acc


def _default_exe():
    """Auto-detect fastface_int8 binary: .exe on Windows, bare name on Linux/macOS."""
    suffix = ".exe" if os.name == "nt" else ""
    for p in (f"./fastface_int8{suffix}", "./fastface_int8.exe", "./fastface_int8"):
        if os.path.exists(p):
            return p
    return f"./fastface_int8{suffix}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-pairs", type=int, default=1000, help="total pairs (half same, half different)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lfw-dir", default="data/lfw")
    ap.add_argument("--onnx", default="models/w600k_r50.onnx")
    ap.add_argument("--weights", default="models/w600k_r50_ffw4.bin")
    ap.add_argument("--exe", default=_default_exe())
    ap.add_argument("--gcc-bin", default="C:/mingw64/bin" if os.name == "nt" else "",
                    help="for libgomp DLL PATH (Windows/mingw only)")
    args = ap.parse_args()

    n_each = args.n_pairs // 2
    print(f"Building {n_each} same + {n_each} different pairs from {args.lfw_dir} (seed={args.seed})...", flush=True)
    pairs = build_pairs(args.lfw_dir, n_each, args.seed)
    print(f"  got {len(pairs)} total pairs\n", flush=True)

    env = os.environ.copy()
    env["PATH"] = args.gcc_bin + os.pathsep + env.get("PATH", "")

    so = ort.SessionOptions()
    so.intra_op_num_threads = 8
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(args.onnx, so, providers=["CPUExecutionProvider"])

    server = Int8Server(args.exe, args.weights, env)

    ort_scores = np.zeros(len(pairs), np.float32)
    int8_scores = np.zeros(len(pairs), np.float32)
    labels = np.zeros(len(pairs), np.int32)

    for i, (a, b, lbl) in enumerate(pairs):
        arr_a, arr_b = load_face(a), load_face(b)
        ea_o, eb_o = embed_ort(sess, arr_a), embed_ort(sess, arr_b)
        ea_i, eb_i = server.embed(arr_a), server.embed(arr_b)
        ort_scores[i]  = np.dot(ea_o, eb_o) / (np.linalg.norm(ea_o) * np.linalg.norm(eb_o))
        int8_scores[i] = np.dot(ea_i, eb_i) / (np.linalg.norm(ea_i) * np.linalg.norm(eb_i))
        labels[i] = lbl
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(pairs)}", flush=True)

    server.close()

    print()
    print(f"=== LFW verification ({len(pairs)} pairs: {n_each} same + {n_each} different) ===")
    ort_acc, ort_t  = best_threshold_accuracy(ort_scores, labels)
    int8_acc, int8_t = best_threshold_accuracy(int8_scores, labels)
    print(f"Best-threshold accuracy:")
    print(f"  ORT FP32:       {ort_acc*100:.2f}%   (threshold {ort_t:.3f})")
    print(f"  FastFace INT8:  {int8_acc*100:.2f}%   (threshold {int8_t:.3f})")
    print(f"  gap:            {(ort_acc - int8_acc)*100:+.3f} pp")
    print()

    for far in (0.01, 0.001):
        ort_tar, ort_fa, _ = tar_at_far(ort_scores, labels, far)
        int8_tar, int8_fa, _ = tar_at_far(int8_scores, labels, far)
        print(f"TAR @ FAR <= {far}:")
        print(f"  ORT FP32:       TAR={ort_tar*100:.2f}%  (actual FAR={ort_fa:.4f})")
        print(f"  FastFace INT8:  TAR={int8_tar*100:.2f}%  (actual FAR={int8_fa:.4f})")
    print()

    print(f"AUC:")
    print(f"  ORT FP32:       {auc_score(ort_scores, labels):.5f}")
    print(f"  FastFace INT8:  {auc_score(int8_scores, labels):.5f}")


if __name__ == "__main__":
    main()
