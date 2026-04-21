"""S119 -- generate plots for the public README.

Reads the LFW 10-fold S104 log and emits:
- docs/lfw_per_fold.svg   : per-fold accuracy bar chart (INT8 vs FP32)
- docs/speed_comparison.svg : bench summary bar chart

Requires matplotlib. If matplotlib isn't available, falls back to
creating text placeholders.
"""
import os, sys, re

OUT_DIR = "docs"
os.makedirs(OUT_DIR, exist_ok=True)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    HAVE_MPL = True
except ImportError:
    HAVE_MPL = False


# Parse LFW per-fold numbers from the committed S104 log
LFW_LOG = "sprint_work/s104_lfw_v1_1_0_repro.log"
int8_per_fold = []
fp32_per_fold = []
if os.path.exists(LFW_LOG):
    with open(LFW_LOG) as f:
        for line in f:
            m = re.match(r"\s+fold\s+(\d+):\s+FP32=([\d.]+)%\s+INT8=([\d.]+)%", line)
            if m:
                fp32_per_fold.append(float(m.group(2)))
                int8_per_fold.append(float(m.group(3)))

print(f"Parsed {len(int8_per_fold)} folds from {LFW_LOG}")


def plot_lfw():
    if not HAVE_MPL or not int8_per_fold:
        with open(os.path.join(OUT_DIR, "lfw_per_fold.txt"), "w", encoding="utf-8") as f:
            f.write(f"FP32: {fp32_per_fold}\nINT8: {int8_per_fold}\n")
        print(f"  wrote text fallback to {OUT_DIR}/lfw_per_fold.txt")
        return

    folds = list(range(1, len(int8_per_fold) + 1))
    x = np.arange(len(folds))
    w = 0.38
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(x - w/2, fp32_per_fold, w, label="ORT FP32 (reference)", color="#6c757d")
    ax.bar(x + w/2, int8_per_fold, w, label="FastFace INT8", color="#28a745")
    ax.set_xticks(x)
    ax.set_xticklabels([f"fold {i}" for i in folds], rotation=0, fontsize=9)
    ax.set_ylim(98.5, 100.2)
    ax.set_ylabel("LFW verification accuracy (%)")
    ax.set_title(f"LFW 10-fold verification: INT8 99.650% +/- 0.229% vs FP32 99.633% +/- 0.221%")
    ax.axhline(np.mean(fp32_per_fold), linestyle="--", color="#6c757d", alpha=0.6,
               label=f"FP32 mean {np.mean(fp32_per_fold):.3f}%")
    ax.axhline(np.mean(int8_per_fold), linestyle="--", color="#28a745", alpha=0.6,
               label=f"INT8 mean {np.mean(int8_per_fold):.3f}%")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "lfw_per_fold.svg")
    fig.savefig(path, format="svg")
    print(f"  wrote {path}")


def plot_speed():
    if not HAVE_MPL:
        with open(os.path.join(OUT_DIR, "speed_comparison.txt"), "w", encoding="utf-8") as f:
            f.write("b=1 burst  FastFace 13.27 ms, ORT 31.46 ms\n"
                    "b=1 sustained  FastFace 20.5 ms, ORT 38 ms\n"
                    "B=8 batched  FastFace 11.09 ms, ORT 32 ms-equivalent\n")
        print(f"  wrote text fallback to {OUT_DIR}/speed_comparison.txt")
        return

    labels = ["b=1 burst", "b=1 sustained\n(4 thread, inf)", "B=8 batched"]
    fastface = [13.27, 20.5, 11.09]
    ort      = [31.46, 38.0, 32.0]  # ORT does not batch-scale

    x = np.arange(len(labels))
    w = 0.38
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x - w/2, ort,      w, label="ORT + InsightFace (reference)", color="#6c757d")
    ax.bar(x + w/2, fastface, w, label="FastFace INT8", color="#28a745")
    for i, (o, f) in enumerate(zip(ort, fastface)):
        ax.text(i - w/2, o + 0.5, f"{o:.1f}", ha="center", fontsize=9, color="#6c757d")
        ax.text(i + w/2, f + 0.5, f"{f:.2f}", ha="center", fontsize=9, color="#28a745")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("ms per face (lower is better)")
    ax.set_title("FastFace INT8 vs ORT+InsightFace on Intel i7-13700")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "speed_comparison.svg")
    fig.savefig(path, format="svg")
    print(f"  wrote {path}")


if __name__ == "__main__":
    plot_lfw()
    plot_speed()
    print("done.")
