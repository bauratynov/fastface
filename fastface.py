"""FastFace — thin Python wrapper around the INT8 CPU face-embedding engine.

Usage:
    from fastface import FastFace
    import numpy as np
    from PIL import Image

    ff = FastFace()  # starts subprocess in --server mode
    img = Image.open("face.jpg").convert("RGB").resize((112, 112))
    arr = (np.asarray(img, dtype=np.float32) - 127.5) / 127.5  # HWC [-1, 1]
    emb = ff.embed(arr)                 # [512] fp32 embedding
    sim = ff.cos_sim(emb, other_emb)    # float
    ff.close()

Expected input: HWC [112, 112, 3] float32 in range [-1, 1].
Expected output: [512] float32 embedding (raw, not L2-normalized).

Batched mode:
    ff = FastFace(batch=8)
    embs = ff.embed_batch(arr_list)     # [B, 512]

Supports `with` context manager for automatic cleanup.
"""
import os
import subprocess
from typing import List, Optional, Sequence
import numpy as np


FACE_NELEMS = 3 * 112 * 112  # 37632


def _default_gcc_bin():
    # On Windows we link libgomp dynamically — PATH needs the mingw bin.
    if os.name == "nt":
        for p in ("C:/mingw64/bin", "C:\\mingw64\\bin"):
            if os.path.isdir(p):
                return p
    return None


class FastFace:
    def __init__(self,
                 weights: str = "models/w600k_r50_ffw4.bin",
                 exe: Optional[str] = None,
                 batch: int = 1,
                 threads: Optional[int] = None,
                 gcc_bin: Optional[str] = None,
                 cwd: Optional[str] = None):
        """Start a persistent FastFace INT8 subprocess in --server mode.

        Parameters
        ----------
        weights : path to w600k_r50_ffw4.bin
        exe     : path to fastface_int8.exe (b=1) or fastface_int8_batched.exe (B>1).
                  If None, picks the right one based on `batch`.
        batch   : 1 for the single-face driver, N>1 for the batched driver.
        gcc_bin : optional path to mingw bin with libgomp-1.dll; auto-detected on Windows.
        cwd     : working directory for the subprocess (default: parent of exe).
        """
        self.batch = int(batch)
        if exe is None:
            # Binary suffix is .exe on Windows, empty on Linux/macOS.
            suffix = ".exe" if os.name == "nt" else ""
            name = "fastface_int8_batched" if self.batch > 1 else "fastface_int8"
            # Prefer native-suffixed binary; fall back to .exe for mingw cross-compiled on Linux.
            candidates = [f"./{name}{suffix}", f"./{name}.exe", f"./{name}"]
            exe = next((p for p in candidates if os.path.exists(p)), candidates[0])
        self.exe = exe
        self.weights = weights

        env = os.environ.copy()
        gcc_bin = gcc_bin or _default_gcc_bin()
        if gcc_bin:
            env["PATH"] = gcc_bin + os.pathsep + env.get("PATH", "")

        args = [exe, weights]
        if self.batch > 1:
            args += ["--batch", str(self.batch)]
        if threads is not None:
            args += ["--threads", str(threads)]
        args += ["--server"]

        if cwd is None:
            cwd = os.path.dirname(os.path.abspath(exe)) or "."

        self.proc = subprocess.Popen(
            args, cwd=cwd, env=env,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            bufsize=0)
        # Confirm subprocess actually running
        if self.proc.poll() is not None:
            err = self.proc.stderr.read().decode(errors="replace")
            raise RuntimeError(f"FastFace subprocess failed to start: {err}")

    def embed(self, arr: np.ndarray) -> np.ndarray:
        """Embed one face image. arr is HWC float32 [-1, 1] shape (112, 112, 3)."""
        if self.batch != 1:
            raise RuntimeError("batch>1 instance; call embed_batch instead")
        if arr.dtype != np.float32 or arr.size != FACE_NELEMS:
            raise ValueError(f"arr must be float32 with {FACE_NELEMS} elements, got {arr.dtype}/{arr.size}")
        self.proc.stdin.write(arr.tobytes())
        self.proc.stdin.flush()
        raw = self.proc.stdout.read(512 * 4)
        if len(raw) != 512 * 4:
            raise RuntimeError(f"short read {len(raw)}; subprocess may have died")
        return np.frombuffer(raw, dtype=np.float32).copy()

    def embed_batch(self, arrs: Sequence[np.ndarray]) -> np.ndarray:
        """Embed B faces. arrs is a list of HWC arrays. Returns [B, 512]."""
        if len(arrs) != self.batch:
            raise ValueError(f"expected {self.batch} arrays, got {len(arrs)}")
        buf = b"".join(a.astype(np.float32).tobytes() for a in arrs)
        self.proc.stdin.write(buf)
        self.proc.stdin.flush()
        raw = self.proc.stdout.read(self.batch * 512 * 4)
        return np.frombuffer(raw, dtype=np.float32).reshape(self.batch, 512).copy()

    @staticmethod
    def cos_sim(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    def close(self):
        if self.proc is None: return
        try:
            self.proc.stdin.close()
            self.proc.wait(timeout=10)
        except Exception:
            self.proc.kill()
        self.proc = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


if __name__ == "__main__":
    # Self-test against the committed golden input/output pair.
    import sys
    root = os.path.dirname(os.path.abspath(__file__))
    os.chdir(root)
    with FastFace() as ff:
        inp = np.fromfile("tests/golden_input.bin", dtype=np.float32)
        if inp.size != FACE_NELEMS:
            print(f"unexpected golden_input size {inp.size}")
            sys.exit(1)
        arr = inp.reshape(112, 112, 3)
        emb = ff.embed(arr)
        golden = np.fromfile("tests/golden_int8_emb.bin", dtype=np.float32)[:512]
        if np.array_equal(emb, golden):
            print("SELF-TEST PASS: embedding matches golden bit-exact")
        else:
            print(f"SELF-TEST FAIL: max diff {np.max(np.abs(emb - golden)):.6e}")
            sys.exit(1)
