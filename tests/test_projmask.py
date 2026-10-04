"""
Model free test for the director's projection mask (kubakub/director/projmask.py): the three template forms,
invert, grow / shrink, feather, scaling to the delivery size, applying.

Run from the ComfyUI_windows_portable folder:
    python_embeded\\python.exe ComfyUI\\custom_nodes\\ComfyUI_KubaNodes\\tests\\test_projmask.py
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from kubakub.director import projmask as pm  # noqa: E402

failures = []


def check(name, cond, detail=""):
    print(("ok   " if cond else "FAIL ") + name + (f"  ({detail})" if detail and not cond else ""))
    if not cond:
        failures.append(name)


H, W = 60, 80
bld = np.zeros((H, W), bool)
bld[10:50, 20:60] = True                                  # the building

check("settings: none / off / no source", pm.settings({}) is None and pm.settings({"projection_mask": {"on": False, "source": "scene"}}) is None
      and pm.settings({"projection_mask": {"on": True}}) is None)
s = pm.settings({"projection_mask": {"on": True, "source": "scene", "grow": 999, "feather": -3, "view": "bad"}})
check("settings: clamped, default view", s["grow"] == 200 and s["feather"] == 0 and s["view"] == "black" and s["source"] == "scene")

overlay = np.zeros((H, W, 4), np.uint8)
overlay[..., 3] = np.where(bld, 0, 255)                   # opaque black outside, transparent building
m, how = pm.from_image(overlay)
check("overlay template: transparent = building", np.array_equal(m > 0.5, bld) and "overlay" in how, how)
cutout = np.zeros((H, W, 4), np.uint8)
cutout[..., :3] = 255
cutout[..., 3] = np.where(bld, 255, 0)                    # the building opaque white
m, how = pm.from_image(cutout)
check("cutout template: opaque = building", np.array_equal(m > 0.5, bld) and "cutout" in how, how)
white = np.where(bld[..., None], 255, 0).astype(np.uint8).repeat(3, -1)
m, how = pm.from_image(white)
check("white on black: white = building", np.array_equal(m > 0.5, bld))
opaque_all = np.dstack([white, np.full((H, W), 255, np.uint8)])
m, _ = pm.from_image(opaque_all)
check("alpha all opaque: read like no alpha", np.array_equal(m > 0.5, bld))

base = bld.astype(np.float32)
check("invert", np.array_equal(pm.refine(base, invert=True) > 0.5, ~bld))
g = pm.refine(base, grow=3)
check("grow 3 px", g[10, 17] == 1 and g[10, 16] == 0 and g[7, 30] == 1)
sh = pm.refine(base, grow=-3)
check("shrink 3 px", sh[13, 30] == 1 and sh[12, 30] == 0)
f = pm.refine(base, feather=6)
check("feather: soft edge, inside and far outside kept", 0.2 < f[30, 20] < 0.8 and f[30, 40] > 0.99 and f[30, 2] < 0.01)

big, note = pm.fit(base, 160, 120)
check("fit: scaled to the delivery size with a note", big.shape == (120, 160) and "scaled" in note)
same, note = pm.fit(base, W, H)
check("fit: same size untouched", same is base and note == "")

rgb = np.ones((H, W, 3), np.float32)
out = pm.apply(rgb, base)
check("apply: black outside, unchanged inside", out[5, 5].sum() == 0 and np.allclose(out[30, 40], 1))

print()
if failures:
    print(f"{len(failures)} FAILED: {', '.join(failures)}")
    sys.exit(1)
print("all projection mask tests passed")
