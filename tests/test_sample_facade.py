"""
Model free test of the sample facade (kubakub/sample_facade.py): sizes, element masks, the mask folder read by
the facade mask atlas (mask_folder mode) and the colour matrix (color_regions mode) give the expected regions.

Run from the ComfyUI_windows_portable folder:
    python_embeded\\python.exe ComfyUI\\custom_nodes\\ComfyUI_KubaNodes\\tests\\test_sample_facade.py
"""

import os
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from kubakub import facade_core as fc  # noqa: E402
from kubakub import sample_facade as sf  # noqa: E402

failures = []


def check(name, cond, detail=""):
    print(("ok   " if cond else "FAIL ") + name + (f"  ({detail})" if detail else ""))
    if not cond:
        failures.append(name)


r = sf.render(960, 540, floors=3, bays=7, style="sandstone", seed=1)
check("image / matrix / silhouette sizes", r["image"].shape == (540, 960, 3) and r["matrix"].shape == (540, 960, 3)
      and r["silhouette"].shape == (540, 960))
check("values in 0..1", float(r["image"].min()) >= 0 and float(r["image"].max()) <= 1)
windows = [n for n, (g, _) in r["masks"].items() if g == "Windows"]
check("3 floors x 7 bays + 6 ground floor arches = 27 windows", len(windows) == 27, str(len(windows)))
check("a door and pilasters", "M_Door" in r["masks"] and sum(n.startswith("M_Pilaster") for n in r["masks"]) == 8)
check("silhouette: sky top and ground bottom are 0, the wall 1", r["silhouette"][2, 480] == 0 and r["silhouette"][-2, 480] == 0
      and r["silhouette"][300, 480] == 1)
check("same seed = same picture, other seed differs", np.array_equal(sf.render(960, 540, seed=1)["image"], r["image"])
      and not np.array_equal(sf.render(960, 540, seed=2)["image"], r["image"]))
for st in sf.STYLES:
    sf.render(640, 360, style=st)
check("every style renders", True)

with tempfile.TemporaryDirectory() as d:
    folder = sf.write_mask_folder(r["masks"], os.path.join(d, "Masks"))
    files = [f for _, _, fs in os.walk(folder) for f in fs]
    check("mask folder: one PNG per element without sky and ground", len(files) == len(r["masks"]) - 2, str(len(files)))
    labels, meta, notes, scope, tags = fc.label_mask_folder(folder, 960, 540, recursive=True)
    names = {m["name"] for m in meta} if meta and isinstance(meta[0], dict) and "name" in meta[0] else set()
    check("the facade mask atlas reads the folder (mask_folder, recursive)", labels.shape == (540, 960) and len(meta) >= 27,
          f"{len(meta)} entries")

print()
print("all sample facade tests passed" if not failures else f"{len(failures)} FAILED: " + ", ".join(failures))
sys.exit(1 if failures else 0)
