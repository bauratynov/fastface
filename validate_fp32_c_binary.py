"""Session 13 — validate fastface_fp32.exe against ORT on a real LFW face.

Workflow:
 1. Load a random LFW face, preprocess to 112x112 float32 [-1,1], NCHW.
 2. Run ORT → reference embedding.
 3. Transpose to NHWC, write to models/validate_input.bin.
 4. Invoke fastface_fp32.exe models/w600k_r50_ffw2.bin models/validate_input.bin models/validate_output.bin.
 5. Read our output, compute cos-sim vs ORT.

Expectation: cos-sim ≥ 0.9999 (FP32 vs FP32, no quantization intermediate).
"""
import sys, os, glob, random, struct, subprocess, time
import numpy as np
import onnxruntime as ort
from PIL import Image


def load_face(lfw_dir, seed=0):
    paths = sorted(glob.glob(os.path.join(lfw_dir, "**", "*.jpg"), recursive=True))
    random.seed(seed); random.shuffle(paths)
    p = paths[0]
    img = Image.open(p).convert("RGB")
    w, h = img.size
    s = 150
    left = (w - s) // 2
    top = max(0, (h - s) // 2 - 10)
    img_crop = img.crop((left, top, left + s, top + s))
    img_112 = img_crop.resize((112, 112), Image.BILINEAR)
    arr = np.asarray(img_112, dtype=np.float32)  # [H, W, C]
    arr = (arr - 127.5) / 127.5
    return arr, p


def main():
    arr_hwc, face_path = load_face("data/lfw", seed=42)
    print(f"Using face: {face_path}")
    nchw = np.transpose(arr_hwc, (2, 0, 1))[None, ...].astype(np.float32).copy()
    nhwc = arr_hwc.astype(np.float32).copy()  # [H, W, C]
    print(f"NCHW shape: {nchw.shape}  NHWC shape: {nhwc.shape}")

    # ORT reference
    sess = ort.InferenceSession("models/w600k_r50.onnx", providers=["CPUExecutionProvider"])
    inp_name = sess.get_inputs()[0].name
    t0 = time.perf_counter()
    ort_out = sess.run(None, {inp_name: nchw})[0].flatten()
    t_ort = (time.perf_counter() - t0) * 1000
    print(f"ORT:        norm={np.linalg.norm(ort_out):.4f}  first5={ort_out[:5]}  time={t_ort:.2f} ms")

    # Write NHWC input for C binary
    os.makedirs("models", exist_ok=True)
    with open("models/validate_input.bin", "wb") as f:
        f.write(nhwc.tobytes())

    # Run our binary in validate mode
    t0 = time.perf_counter()
    r = subprocess.run(
        ["./fastface_fp32.exe", "models/w600k_r50_ffw2.bin",
         "models/validate_input.bin", "models/validate_output.bin"],
        capture_output=True, text=True, timeout=60)
    t_c = (time.perf_counter() - t0) * 1000
    print(f"C binary exit={r.returncode}, time={t_c:.2f} ms (includes load)")
    if r.returncode != 0:
        print("STDOUT:", r.stdout)
        print("STDERR:", r.stderr)
        sys.exit(1)

    with open("models/validate_output.bin", "rb") as f:
        ours = np.frombuffer(f.read(), dtype=np.float32)
    print(f"Ours:       norm={np.linalg.norm(ours):.4f}  first5={ours[:5]}")

    cos = float(np.dot(ort_out, ours) / (np.linalg.norm(ort_out) * np.linalg.norm(ours) + 1e-9))
    l2 = float(np.linalg.norm(ort_out - ours))
    print(f"\n==== CORRECTNESS ====")
    print(f"Cos-sim:  {cos:.6f}")
    print(f"L2 diff:  {l2:.4f}")
    print(f"ORT norm: {np.linalg.norm(ort_out):.4f}")
    print(f"Our norm: {np.linalg.norm(ours):.4f}")

    if cos >= 0.9999:
        print("PERFECT — FP32 path is exact")
    elif cos >= 0.99:
        print("EXCELLENT — near-exact FP32 path")
    elif cos >= 0.95:
        print("GOOD — minor numerical drift")
    elif cos >= 0.8:
        print("OK — real correctness issue, investigate per-op")
    else:
        print("BROKEN — search for bugs")


if __name__ == "__main__":
    main()
