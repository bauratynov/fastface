"""S81 end-to-end face pipeline demo.

    python face_pipeline.py <image.jpg>

Detects faces via RetinaFace ONNX (models/det_10g.onnx), aligns each
crop to 112x112 using the 5-point landmarks (standard ArcFace template),
and produces a 512-dim embedding per face via FastFace INT8.

Prints one line per face: [i] bbox=(x0, y0, x1, y1) conf=0.XX  emb=[first 5 values].

This is a first proof-of-concept of the complete pipeline. The detector
still runs on ORT FP32 (~25 ms @ 640x640); porting it to FastFace's
native INT8 path is S82+ work.
"""
import os, sys, argparse
import numpy as np
from PIL import Image
import onnxruntime as ort
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fastface import FastFace


# Standard ArcFace alignment template (5 points, 112x112)
ARCFACE_TEMPLATE = np.array([
    [38.2946, 51.6963],  # left eye
    [73.5318, 51.5014],  # right eye
    [56.0252, 71.7366],  # nose
    [41.5493, 92.3655],  # left mouth
    [70.7299, 92.2041],  # right mouth
], dtype=np.float32)


def distance2bbox(points, distance, max_shape=None):
    """[cx, cy] + [l, t, r, b] -> [x0, y0, x1, y1]."""
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    return np.stack([x1, y1, x2, y2], axis=-1)


def distance2kps(points, distance):
    """[cx, cy] + [dx, dy] * 5 -> [x, y] * 5."""
    preds = []
    for i in range(0, distance.shape[1], 2):
        px = points[:, 0] + distance[:, i]
        py = points[:, 1] + distance[:, i + 1]
        preds.append(px); preds.append(py)
    return np.stack(preds, axis=-1).reshape(-1, 5, 2)


def retinaface_detect(sess, img_bgr, det_size=640, conf_thresh=0.5):
    """Return list of (bbox, landmarks, score) for faces in img."""
    h, w = img_bgr.shape[:2]
    # Letterbox to det_size x det_size
    scale = min(det_size / w, det_size / h)
    nw, nh = int(w * scale), int(h * scale)
    canvas = np.zeros((det_size, det_size, 3), dtype=np.float32)
    resized = np.asarray(
        Image.fromarray(img_bgr).resize((nw, nh), Image.BILINEAR),
        dtype=np.float32)
    canvas[:nh, :nw, :] = resized
    # Normalize: (x - 127.5) / 128.0 (RetinaFace-specific)
    canvas = (canvas - 127.5) / 128.0
    blob = canvas.transpose(2, 0, 1)[None].astype(np.float32)

    outs = sess.run(None, {sess.get_inputs()[0].name: blob})
    # 9 outputs: score, bbox, kps x 3 strides
    strides = [8, 16, 32]
    num_anchors = 2
    boxes_all, kps_all, scores_all = [], [], []
    for i, stride in enumerate(strides):
        scores = outs[i].reshape(-1)
        bbox_preds = outs[i + 3] * stride
        kps_preds = outs[i + 6] * stride
        feat_h = feat_w = det_size // stride
        anchor_centers = np.stack(
            np.mgrid[:feat_h, :feat_w][::-1], axis=-1
        ).astype(np.float32).reshape(-1, 2) * stride
        anchor_centers = np.repeat(anchor_centers, num_anchors, axis=0)

        keep = scores >= conf_thresh
        if not keep.any():
            continue
        boxes = distance2bbox(anchor_centers[keep], bbox_preds[keep])
        kps = distance2kps(anchor_centers[keep], kps_preds[keep])
        boxes_all.append(boxes / scale)
        kps_all.append(kps / scale)
        scores_all.append(scores[keep])

    if not boxes_all:
        return []
    boxes = np.concatenate(boxes_all)
    kps = np.concatenate(kps_all)
    scores = np.concatenate(scores_all)

    # NMS (simple greedy)
    keep_idx = nms(boxes, scores, iou_thresh=0.4)
    return [(boxes[i], kps[i], float(scores[i])) for i in keep_idx]


def nms(boxes, scores, iou_thresh=0.4):
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while len(order) > 0:
        i = order[0]; keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0, xx2 - xx1); h = np.maximum(0, yy2 - yy1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        inds = np.where(ovr <= iou_thresh)[0]
        order = order[inds + 1]
    return keep


def align_face(img, kps):
    """Align 5-point kps to ArcFace 112x112 template via similarity transform."""
    # Solve umeyama transform (rigid + uniform scale) between source kps and template
    src = kps.astype(np.float32)
    dst = ARCFACE_TEMPLATE
    mean_src = src.mean(0); mean_dst = dst.mean(0)
    src_c = src - mean_src; dst_c = dst - mean_dst
    H = src_c.T @ dst_c
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, d]) @ U.T
    scale = S.sum() / (src_c ** 2).sum()
    t = mean_dst - scale * (R @ mean_src)
    M = np.zeros((2, 3), dtype=np.float32)
    M[:, :2] = scale * R
    M[:, 2] = t

    # Apply affine warp (output 112x112)
    h, w = img.shape[:2]
    # Inverse for backward warping
    M_inv = np.zeros((2, 3), dtype=np.float32)
    A = M[:, :2]
    b = M[:, 2]
    A_inv = np.linalg.inv(A)
    M_inv[:, :2] = A_inv
    M_inv[:, 2] = -A_inv @ b

    aligned = np.zeros((112, 112, 3), dtype=np.float32)
    # Manual bilinear backward warp
    ys, xs = np.mgrid[:112, :112].astype(np.float32)
    src_x = M_inv[0, 0] * xs + M_inv[0, 1] * ys + M_inv[0, 2]
    src_y = M_inv[1, 0] * xs + M_inv[1, 1] * ys + M_inv[1, 2]
    ix = np.clip(src_x.astype(int), 0, w - 2)
    iy = np.clip(src_y.astype(int), 0, h - 2)
    dx = np.clip(src_x - ix, 0, 1); dy = np.clip(src_y - iy, 0, 1)
    for c in range(3):
        p00 = img[iy, ix, c]
        p10 = img[iy, ix + 1, c]
        p01 = img[iy + 1, ix, c]
        p11 = img[iy + 1, ix + 1, c]
        aligned[..., c] = (p00 * (1 - dx) * (1 - dy) + p10 * dx * (1 - dy) +
                          p01 * (1 - dx) * dy + p11 * dx * dy)
    # Normalize to [-1, 1]
    return (aligned - 127.5) / 127.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--det-size", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.5)
    args = ap.parse_args()

    img = np.asarray(Image.open(args.image).convert("RGB"))
    print(f"image shape: {img.shape}")

    det_sess = ort.InferenceSession("models/det_10g.onnx",
                                     providers=["CPUExecutionProvider"])
    print("RetinaFace loaded (det_10g.onnx)")

    faces = retinaface_detect(det_sess, img, args.det_size, args.conf)
    print(f"detected {len(faces)} faces")

    if not faces:
        return

    with FastFace() as ff:
        for i, (box, kps, score) in enumerate(faces):
            aligned = align_face(img, kps)
            emb = ff.embed(aligned)
            print(f"[{i}] bbox=({box[0]:.0f},{box[1]:.0f},{box[2]:.0f},{box[3]:.0f}) "
                  f"conf={score:.3f}  emb[:5]={emb[:5]}")


if __name__ == "__main__":
    main()
