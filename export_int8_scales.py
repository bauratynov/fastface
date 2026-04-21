"""Export per-channel activation scales for C consumption.

Runs collect_ranges from calibrate_per_channel_int8, then writes per-tensor
symmetric per-channel absmax scales to a binary file that C can mmap.

Format:
  magic 'PCS8'
  u32 num_tensors
  for each tensor:
    u16 name_len, name_bytes
    u32 channels, f32[channels] scale
"""
import sys, os, glob, random, struct, time
sys.path.insert(0, '.')
from extract_onnx import parse_model
from calibrate_per_channel_int8 import collect_ranges, load_lfw_batch
import numpy as np


def main():
    n_calib = int(os.environ.get("N_CALIB", "20"))
    calib_inputs, _ = load_lfw_batch("data/lfw", n_calib, seed=1)
    print(f"Collecting per-channel ranges on {n_calib} faces...", flush=True)
    g = parse_model("models/w600k_r50.onnx")
    t0 = time.perf_counter()
    ranges = collect_ranges(g, calib_inputs)
    print(f"  done in {time.perf_counter()-t0:.1f}s ({len(ranges)} tensors)")

    out_path = "models/pc_int8_scales.bin"
    with open(out_path, "wb") as f:
        f.write(b"PCS8")
        f.write(struct.pack("<I", len(ranges)))
        for name, (tmin, tmax) in ranges.items():
            absmax = np.maximum(np.abs(tmin), np.abs(tmax)).astype(np.float32)
            scale = absmax / 127.0
            nb = name.encode()
            f.write(struct.pack("<H", len(nb)))
            f.write(nb)
            f.write(struct.pack("<I", scale.size))
            f.write(scale.tobytes())
    print(f"Wrote {len(ranges)} per-channel scales to {out_path}")


if __name__ == "__main__":
    main()
