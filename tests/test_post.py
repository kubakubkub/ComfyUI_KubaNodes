"""
Model free test of the post tools (kubakub/post.py): colour match, .cube LUTs, deflicker, retime, burn in.

Run from the ComfyUI_windows_portable folder:
    python_embeded\\python.exe ComfyUI\\custom_nodes\\ComfyUI_KubaNodes\\tests\\test_post.py
"""

import os
import sys
import tempfile
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from kubakub import post as pp  # noqa: E402

failures = []


def check(name, cond, detail=""):
    print(("ok   " if cond else "FAIL ") + name + (f"  ({detail})" if detail else ""))
    if not cond:
        failures.append(name)


rng = np.random.default_rng(0)
src = (rng.random((120, 160, 3)) * 0.6 + 0.2).astype(np.float32)
M = np.array([[0.9, 0.1, 0.0], [0.05, 0.8, 0.1], [0.0, 0.1, 1.1]])
ref = np.clip(src.reshape(-1, 3) @ M.T + np.array([0.05, -0.02, 0.0]), 0, 1).reshape(src.shape).astype(np.float32)

# --- colour match
fit = pp.colour_fit(src[None], ref, "mkl")
err = float(np.abs(pp.colour_apply(src, fit) - ref).mean())
check("mkl: an affine colour change is matched", err < 0.01, f"mean error {err:.4f}")
fit2 = pp.colour_fit(src[None], ref, "mean_std")
m2 = pp.colour_apply(src, fit2)
check("mean_std: channel means match", np.allclose(m2.reshape(-1, 3).mean(0), ref.reshape(-1, 3).mean(0), atol=0.01))
check("strength 0 = unchanged", np.allclose(pp.colour_apply(src, fit, 0.0), src))

# --- LUTs
with tempfile.TemporaryDirectory() as d:
    p = pp.write_cube(os.path.join(d, "id.cube"), lambda g: g, 17)
    lut = pp.parse_cube(open(p, encoding="utf-8").read())
    check("identity .cube round trip", float(np.abs(pp.apply_lut(src, lut) - src).max()) < 1e-4)
    p2 = pp.write_cube(os.path.join(d, "match.cube"), lambda g: pp.colour_apply(g, fit), 33)
    baked = pp.apply_lut(src, pp.parse_cube(open(p2, encoding="utf-8").read()))
    check("colour match baked as .cube = the same look", float(np.abs(baked - pp.colour_apply(src, fit)).max()) < 0.01,
          f"{np.abs(baked - pp.colour_apply(src, fit)).max():.4f}")
    inv = pp.parse_cube("LUT_1D_SIZE 2\n1 1 1\n0 0 0\n")
    check("1D .cube (invert)", np.allclose(pp.apply_lut(src, inv), 1 - src, atol=1e-5))
    red = pp.parse_cube("TITLE \"x\"\nLUT_3D_SIZE 2\n" + "\n".join("1 0 0" if (i & 1) else "0 0 0" for i in range(8)))
    check(".cube order: red changes fastest", np.allclose(pp.apply_lut(np.array([[[1, 0, 0]]], np.float32), red), [1, 0, 0])
          and np.allclose(pp.apply_lut(np.array([[[0, 1, 1]]], np.float32), red), [0, 0, 0]))
    t = time.perf_counter()
    big = rng.random((1080, 1920, 3), dtype=np.float32)
    pp.apply_lut(big, pp.parse_cube(open(p2, encoding="utf-8").read()))
    check("1080p frame through a 33^3 LUT", time.perf_counter() - t < 3.0, f"{time.perf_counter() - t:.2f} s")

# --- deflicker
base = np.full((60, 80, 3), 0.5, np.float32)
jit = 1 + 0.12 * np.sin(np.arange(40) * 2.7)                      # fast brightness jumps
fade = np.linspace(0.6, 1.0, 40)                                   # a slow fade (wanted)
frames = [pp.to_srgb(pp.to_linear(base) * j * f) for j, f in zip(jit, fade)]
gains = pp.deflicker_gains(frames, window=9, grid=4)
out = [pp.apply_gain(f, g) for f, g in zip(frames, gains)]
lum_in = np.array([pp.to_linear(f).mean() for f in frames])
lum_out = np.array([pp.to_linear(f).mean() for f in out])
resid_in = np.std(lum_in / (0.214 * fade))
resid_out = np.std(lum_out / (0.214 * fade))
check("deflicker: frame to frame jumps shrink", resid_out < 0.25 * resid_in, f"{resid_in:.4f} -> {resid_out:.4f}")
check("deflicker: the slow fade survives", lum_out[-5:].mean() > 1.4 * lum_out[:5].mean(),
      f"{lum_out[:5].mean():.3f} -> {lum_out[-5:].mean():.3f}")

# --- retime
seq = []
for i in range(5):
    f = np.zeros((90, 160, 3), np.float32)
    f[30:60, 10 + 20 * i:40 + 20 * i] = 1.0
    seq.append(f)
ts = pp.retime_times(5, 0.5)
check("speed 0.5: 9 frames from 5", len(ts) == 9 and np.allclose(ts[1], 0.5))
check("exact frame count", len(pp.retime_times(5, 1.0, 13)) == 13)
check("speed 2: every second frame", np.allclose(pp.retime_times(5, 2.0), [0, 2, 4]))
mid = pp.between(seq[1], seq[2], 0.5, "flow")
xs = np.nonzero(mid[45].max(-1) > 0.5)[0]
cx = xs.mean() if len(xs) else -1
check("flow in-between: the square moved halfway", abs(cx - 54.5) < 3, f"centre x {cx:.1f} (expected 54.5)")
bl = pp.between(seq[1], seq[2], 0.5, "blend")
check("blend in-between: a cross-fade", np.isclose(bl[45, 35, 0], 0.5) and np.isclose(bl[45, 65, 0], 0.5)
      and np.isclose(bl[45, 55, 0], 1.0))

from concurrent.futures import ThreadPoolExecutor  # noqa: E402
big_seq = [np.roll(np.tile(seq[0], (4, 4, 1)), 7 * i, axis=1) for i in range(12)]
with ThreadPoolExecutor(max_workers=8) as pool:                 # the node runs frames in threads: must not crash
    res = list(pool.map(lambda i: pp.between(big_seq[i], big_seq[i + 1], 0.5, "flow"), range(11)))
check("flow in-betweens from 8 threads at once", len(res) == 11 and all(np.isfinite(r).all() for r in res))

# --- burn in
check("timecode", pp.timecode(26, 25) == "00:00:01:01" and pp.timecode(25 * 3661, 25) == "01:01:01:00")
img = np.full((360, 640, 3), 0.7, np.float32)
b = pp.burn_text(img, "show  00012  00:00:00:12", "bottom left", 0.05)
check("burn in: text box bottom left", np.abs(b[300:, :200] - img[300:, :200]).max() > 0.2
      and np.allclose(b[:100, 400:], img[:100, 400:], atol=0.005))

print("\n" + ("ALL OK" if not failures else f"{len(failures)} FAILED: {failures}"))
sys.exit(1 if failures else 0)
