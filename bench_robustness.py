"""S59 — augmentation robustness test.

Apply several degradations (blur, noise, JPEG) to LFW pair images and
measure how LFW 1000-pair accuracy degrades for both ORT FP32 and
FastFace INT8. If INT8 tracks FP32 under degradations, INT8 is a true
drop-in replacement — not just on pristine benchmark data.
"""
import os, io, glob, random, subprocess, argparse
import numpy as np
from PIL import Image, ImageFilter
import onnxruntime as ort


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


def load_face_with_aug(path, aug):
    img = Image.open(path).convert("RGB")
    w, h = img.size; s = 150
    left = (w - s) // 2; top = max(0, (h - s) // 2 - 10)
    img = img.crop((left, top, left + s, top + s)).resize((112, 112), Image.BILINEAR)

    if aug["type"] == "blur":
        img = img.filter(ImageFilter.GaussianBlur(radius=aug["sigma"]))
    elif aug["type"] == "noise":
        arr = np.asarray(img, dtype=np.float32)
        rng = np.random.default_rng(aug["seed"])
        arr = arr + rng.normal(0, aug["stddev"], arr.shape).astype(np.float32)
        arr = np.clip(arr, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr)
    elif aug["type"] == "jpeg":
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=aug["quality"])
        buf.seek(0)
        img = Image.open(buf).convert("RGB")
    return (np.asarray(img, dtype=np.float32) - 127.5) / 127.5


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


def best_threshold_accuracy(scores, labels):
    thresholds = np.linspace(scores.min(), scores.max(), 500)
    best = 0
    for t in thresholds:
        acc = ((scores >= t).astype(int) == labels).mean()
        if acc > best: best = acc
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-pairs", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lfw-dir", default="data/lfw")
    args = ap.parse_args()

    env = os.environ.copy()
    if os.name == "nt":
        env["PATH"] = "C:/mingw64/bin" + os.pathsep + env.get("PATH", "")

    pairs = build_pairs(args.lfw_dir, args.n_pairs // 2, args.seed)
    print(f"Got {len(pairs)} pairs\n", flush=True)

    so = ort.SessionOptions(); so.intra_op_num_threads = 8
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession("models/w600k_r50.onnx", so, providers=["CPUExecutionProvider"])
    # Cross-platform binary name: .exe on Windows, bare on Linux/macOS.
    _exe_suffix = ".exe" if os.name == "nt" else ""
    _exe = next((p for p in (f"./fastface_int8{_exe_suffix}", "./fastface_int8.exe", "./fastface_int8")
                 if os.path.exists(p)), f"./fastface_int8{_exe_suffix}")
    server = Int8Server(_exe, "models/w600k_r50_ffw4.bin", env)

    augs = [
        {"name": "clean",           "type": "none"},
        {"name": "blur sigma=1.0",  "type": "blur",  "sigma": 1.0},
        {"name": "blur sigma=2.0",  "type": "blur",  "sigma": 2.0},
        {"name": "noise sd=10",     "type": "noise", "stddev": 10, "seed": 1},
        {"name": "noise sd=25",     "type": "noise", "stddev": 25, "seed": 1},
        {"name": "jpeg q=50",       "type": "jpeg",  "quality": 50},
        {"name": "jpeg q=20",       "type": "jpeg",  "quality": 20},
    ]

    print(f"{'augmentation':22s}  {'FP32 acc':>10s}  {'INT8 acc':>10s}  {'gap':>8s}")
    print(f"{'-'*22:22s}  {'-'*10:>10s}  {'-'*10:>10s}  {'-'*8:>8s}")
    for aug in augs:
        cs_fp32 = np.zeros(len(pairs))
        cs_int8 = np.zeros(len(pairs))
        labels = np.zeros(len(pairs), dtype=int)
        for i, (a, b, lbl) in enumerate(pairs):
            arr_a = load_face_with_aug(a, aug) if aug["type"] != "none" else load_face_with_aug(a, {"type":"none"})
            arr_b = load_face_with_aug(b, aug) if aug["type"] != "none" else load_face_with_aug(b, {"type":"none"})
            # ORT
            nchw_a = np.transpose(arr_a, (2,0,1))[None].astype(np.float32).copy()
            nchw_b = np.transpose(arr_b, (2,0,1))[None].astype(np.float32).copy()
            ea_o = sess.run(None, {sess.get_inputs()[0].name: nchw_a})[0].flatten()
            eb_o = sess.run(None, {sess.get_inputs()[0].name: nchw_b})[0].flatten()
            # INT8
            ea_i = server.embed(arr_a); eb_i = server.embed(arr_b)
            cs_fp32[i] = np.dot(ea_o, eb_o) / (np.linalg.norm(ea_o) * np.linalg.norm(eb_o))
            cs_int8[i] = np.dot(ea_i, eb_i) / (np.linalg.norm(ea_i) * np.linalg.norm(eb_i))
            labels[i] = lbl
        acc_fp32 = best_threshold_accuracy(cs_fp32, labels) * 100
        acc_int8 = best_threshold_accuracy(cs_int8, labels) * 100
        print(f"{aug['name']:22s}  {acc_fp32:>9.2f}%  {acc_int8:>9.2f}%  {acc_fp32-acc_int8:>+7.2f}pp")

    server.close()


if __name__ == "__main__":
    main()
