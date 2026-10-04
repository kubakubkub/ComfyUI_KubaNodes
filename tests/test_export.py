"""
Model free test of the director's delivery export (kubakub/director/export.py): every format writes a short
clip (a known colour patch, a moving box, sound, alpha where asked), then the file is read back: size, frame count,
pixel format, Rec.709 tags, alpha, sound and the colour error after the round trip.

Run from the ComfyUI_windows_portable folder:
    python_embeded\\python.exe ComfyUI\\custom_nodes\\ComfyUI_KubaNodes\\tests\\test_export.py
"""

import glob
import os
import shutil
import subprocess
import sys
import tempfile

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from kubakub.director import export as ex  # noqa: E402
from kubakub.director import media  # noqa: E402

failures = []


def check(name, cond, detail=""):
    print(("ok   " if cond else "FAIL ") + name + (f"  ({detail})" if detail else ""))
    if not cond:
        failures.append(name)


W, H, N, FPS, SR = 320, 180, 12, 25, 48000
COL = np.array([0.30, 0.55, 0.80], np.float32)           # a known colour patch (left half)
tmp = tempfile.mkdtemp(prefix="kuba_export_")
audio = (0.3 * np.sin(2 * np.pi * 440 * np.arange(int(N / FPS * SR)) / SR)).astype(np.float32)[None].repeat(2, 0)


def frame(k):
    f = np.zeros((H, W, 3), np.float32)
    f[:, :W // 2] = COL
    f[60:120, 170 + k * 8:210 + k * 8] = 1.0               # a moving white box
    a = np.zeros((H, W), np.float32); a[:, :W // 2] = 1; a[60:120, 170 + k * 8:210 + k * 8] = 1
    return f, a


def run(fmt, alpha=False):
    w = ex.Writer(fmt, os.path.join(tmp, fmt + ("_a" if alpha else "")), "show", W, H, FPS, alpha=alpha, audio=audio, sr=SR)
    for k in range(N):
        w.write(*frame(k))
    return w.close()


exe = media.ffmpeg_exe()
for fmt in ("prores4444", "prores422hq", "h264", "h264_444", "h265", "preview"):
    files = run(fmt, alpha=(fmt == "prores4444"))
    path = files[0]
    info = media.probe(path)
    head = subprocess.run([exe, "-hide_banner", "-i", path], capture_output=True).stderr.decode("utf-8", "replace")
    want = (W // 2, H // 2) if fmt == "preview" else (W, H)
    tagged = "bt709" in head
    dec = media.decode(path, [0, N - 1], info=info)
    f0 = media.mix(dec, [(0, 1.0)])
    patch = f0[H // 2 - 10:H // 2 + 10, 20:60, :3].reshape(-1, 3).mean(0) if fmt != "preview" else f0[H // 4 - 5:H // 4 + 5, 10:30, :3].reshape(-1, 3).mean(0)
    err = float(np.abs(patch - COL).max() * 255)
    ok = (info["w"], info["h"]) == want and info["frames"] == N and tagged and info["audio"] and err <= (2.5 if "prores" in fmt or fmt == "h264_444" else 4)
    check(f"{fmt}: {want[0]}x{want[1]}, {N} frames, Rec.709 tags, sound, colour error {err:.1f}/255",
          ok, f"{info['w']}x{info['h']} {info['frames']} frames {info['pix_fmt']} tagged={tagged} audio={info['audio']}")
    if fmt == "prores4444":
        check("prores4444 with alpha: 4:4:4 with alpha, alpha comes back", info["alpha"] and info["pix_fmt"].startswith("yuva444p1")
              and dec[0][10, 300, 3] < 1000 and dec[0][10, 20, 3] > 60000, info["pix_fmt"])
for bits in (8, 16):
    files = run(f"png{bits}", alpha=True)
    pngs = sorted(glob.glob(os.path.join(files[0], "*.png")))
    img = cv2.imread(pngs[0], cv2.IMREAD_UNCHANGED)
    maxv = 65535 if bits == 16 else 255
    c = img[H // 2, 30, [2, 1, 0]].astype(np.float32) / maxv
    check(f"png{bits}: {N} RGBA files, {bits} bit, exact colour, alpha, a .wav next to them",
          len(pngs) == N and img.shape == (H, W, 4) and img.dtype == (np.uint16 if bits == 16 else np.uint8)
          and np.abs(c - COL).max() * 255 <= 0.6 and img[10, 300, 3] == 0 and img[10, 20, 3] == maxv and any(f.endswith(".wav") for f in files),
          f"{len(pngs)} files {img.shape} {img.dtype} colour {np.round(c * 255, 1).tolist()}")
try:
    w = ex.Writer("h264", os.path.join(tmp, "bad"), "x", W, H, FPS)
    w.write(np.zeros((10, 10, 3), np.float32))
    check("a frame of the wrong size is refused", False)
except ValueError:
    check("a frame of the wrong size is refused", True)
    w.close()

shutil.rmtree(tmp, ignore_errors=True)
print()
if failures:
    print(f"{len(failures)} FAILED: {', '.join(failures)}")
    sys.exit(1)
print("all export tests passed")
