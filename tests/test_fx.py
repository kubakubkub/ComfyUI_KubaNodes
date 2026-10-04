"""
Model free test for director layer effects (kubakub/director/fx.py) and their use in render.py.

Run from the ComfyUI_windows_portable folder:
    python_embeded\\python.exe ComfyUI\\custom_nodes\\ComfyUI_KubaNodes\\tests\\test_fx.py
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from kubakub.director import fx  # noqa: E402
from kubakub.director import render as rd  # noqa: E402

failures = []


def check(name, cond, detail=""):
    print(("ok   " if cond else "FAIL ") + name + (f"  ({detail})" if detail and not cond else ""))
    if not cond:
        failures.append(name)


H, W = 80, 120
rgb = np.zeros((H, W, 3), np.float32)
rgb[30:50, 50:70] = 1.0                                  # a white square on black
alpha = np.ones((H, W), np.float32)

d = fx.parse({})
check("defaults: nothing active", not fx.active(d) and fx.margin(d) == 0)
p = fx.parse({"blur": 999, "grain": -1, "glow_threshold": 5})
check("parse clamps", p["blur"] == 200 and p["grain"] == 0 and p["glow_threshold"] == 0.99)
o, a = fx.apply(rgb, alpha, fx.parse({"blur": 4}))
check("blur: soft edge, energy kept", 0.1 < o[40, 48, 0] < 0.9 and abs(o.sum() - rgb.sum()) / rgb.sum() < 0.02)
o, a = fx.apply(rgb, alpha, fx.parse({"sharpen": 1.5, "sharpen_radius": 2}))
check("sharpen: overshoot at the edge, flat areas unchanged", o[40, 51, 0] >= 0.99 and o[40, 48, 0] <= 0.01 and np.allclose(o[5, 5], 0) and np.allclose(o[40, 60], 1))
g0 = np.zeros((H, W, 3), np.float32); g0[38:42, 58:62] = 0.9
o, a = fx.apply(g0, alpha, fx.parse({"glow": 2, "glow_radius": 6, "glow_threshold": 0.5}))
check("glow: bright spot bleeds into the dark around it", o[40, 66, 0] > 0.05 and o[40, 110, 0] < 0.01)
o, a = fx.apply(np.zeros((H, W, 3), np.float32) + 0.2, alpha, fx.parse({"glow": 2, "glow_radius": 6, "glow_threshold": 0.5}))
check("glow: below the threshold nothing glows", np.allclose(o, 0.2, atol=1e-5))
grey = np.full((H, W, 3), 0.5, np.float32)
o1, _ = fx.apply(grey, alpha, fx.parse({"grain": 0.5, "grain_size": 2}), seed=7)
o2, _ = fx.apply(grey, alpha, fx.parse({"grain": 0.5, "grain_size": 2}), seed=7)
o3, _ = fx.apply(grey, alpha, fx.parse({"grain": 0.5, "grain_size": 2}), seed=8)
check("grain: noisy around the same mean, same seed = same grain, next frame differs",
      0.01 < o1.std() < 0.2 and abs(o1.mean() - 0.5) < 0.02 and np.array_equal(o1, o2) and not np.array_equal(o1, o3))

# in the render: a layer with a blur spreads past its box; a grade layer's glow works on what is below
base = np.zeros((H, W, 3), np.float32)
red = np.zeros((20, 20, 4), np.float32); red[..., 0] = 1; red[..., 3] = 1
doc = lambda *ls: {"version": 1, "layers": list(ls)}
BASE = {"id": "base", "name": "b", "kind": "base"}
A = {"id": "a", "name": "red", "kind": "image", "source": "input:0", "x": 50, "y": 30, "w": 20, "h": 20}
plain = rd.render(doc(A, BASE), base, {"input:0": red})["image"]
blurred = rd.render(doc(dict(A, fx={"blur": 3}), BASE), base, {"input:0": red})["image"]
check("render: layer blur spreads outside the box", plain[40, 47, 0] == 0 and blurred[40, 47, 0] > 0.05 and blurred[40, 60, 0] > 0.9)
G = {"id": "g", "name": "glow all", "kind": "adjust", "fx": {"glow": 2, "glow_radius": 5, "glow_threshold": 0.4}}
gl = rd.render(doc(G, A, BASE), base, {"input:0": red})["image"]
check("render: a grade layer's effects work on everything below", gl[40, 73, 0] > 0.05 and plain[40, 73, 0] == 0)
mid = np.full((20, 20, 4), 0.5, np.float32); mid[..., 3] = 1         # grain shows in the mid tones (overlay)
s1 = rd.render(doc(dict(A, w=W, h=H, x=0, y=0, fx={"grain": 0.6}), BASE), base, {"input:0": mid}, frame_seed=3)["image"]
s2 = rd.render(doc(dict(A, w=W, h=H, x=0, y=0, fx={"grain": 0.6}), BASE), base, {"input:0": mid}, frame_seed=4)["image"]
check("render: grain changes per frame", not np.array_equal(s1, s2))

print()
if failures:
    print(f"{len(failures)} FAILED: {', '.join(failures)}")
    sys.exit(1)
print("all fx tests passed")
