"""
Model free test of kubakub render passes (kubakub/render_passes.py): a folder of passes named <render>_<pass>.png is
sorted into picture / depth / normal / masks, shared masks reach every render, inverted and 16-bit files read right,
and the masks become regions through the regions from masks logic.

Run from the ComfyUI_windows_portable folder:
    python_embeded\\python.exe ComfyUI\\custom_nodes\\ComfyUI_KubaNodes\\tests\\test_render_passes.py
"""

import os
import sys
import tempfile

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from kubakub import facade_core as fc  # noqa: E402
from kubakub import render_passes as rp  # noqa: E402

failures = []


def check(name, cond, detail=""):
    print(("ok   " if cond else "FAIL ") + name + (f"  ({detail})" if detail and not cond else ""))
    if not cond:
        failures.append(name)


def save(folder, name, img):
    cv2.imencode(os.path.splitext(name)[1], img)[1].tofile(os.path.join(folder, name))


W, H = 320, 200
with tempfile.TemporaryDirectory() as d:
    beauty = np.full((H, W, 3), (40, 90, 200), np.uint8)                   # BGR: an orange-ish wall
    for nm in ("facade_a", "facade_b"):
        save(d, f"{nm}_beauty.png", beauty)
    cut = np.full((H, W), 255, np.uint8)                                   # black = cut open: two holes
    cut[40:90, 30:100] = 0
    cut[110:170, 180:290] = 0
    save(d, "facade_a_cut.png", cut)
    depth16 = np.tile(np.linspace(65535, 20000, H).astype(np.uint16)[:, None], (1, W))   # far (white) at the top
    save(d, "facade_a_depth.png", depth16)
    save(d, "facade_a_normal.png", np.full((H // 2, W // 2, 3), (255, 128, 128), np.uint8))   # half size
    facade = np.zeros((H, W), np.uint8)
    facade[20:190, 10:310] = 255
    save(d, "facade_mask.png", facade)
    ids = np.zeros((H, W, 3), np.uint8)
    ids[:, : W // 2] = (0, 0, 255)
    ids[:, W // 2:] = (0, 255, 0)
    save(d, "facade_a_ids.png", ids)
    save(d, "_sheet_beauty.jpg", beauty)
    open(os.path.join(d, "facade_a.exr"), "wb").write(b"not read")
    open(os.path.join(d, "facade_a_recipe.json"), "w").write("{}")

    found = rp.discover(d)
    check("two renders in the folder", sorted(found["renders"]) == ["facade_a", "facade_b"], str(sorted(found["renders"])))
    a = found["renders"]["facade_a"]
    check("picture, depth, normal and the other passes are told apart", all(a[k] for k in ("beauty", "depth", "normal"))
          and sorted(a["masks"]) == ["cut", "ids"], str(a))
    check("a file of no render is a shared mask; sheets and EXRs are skipped", list(found["shared"]) == ["facade_mask"]
          and set(found["skipped"]) == {"_sheet_beauty.jpg", "facade_a.exr"}, f"{found['shared']} {found['skipped']}")

    r = rp.load(d, "facade_a", invert="cut")
    names = [n for n, _ in r["masks"]]
    masks = dict(r["masks"])
    check("the picture comes back in RGB", r["beauty"].shape == (H, W, 3)
          and np.allclose(r["beauty"][5, 5], np.array([200, 90, 40]) / 255, atol=1e-3), str(r["beauty"][5, 5]))
    check("masks: cut and the shared facade mask; the ID map is left out with a note", names == ["cut", "facade_mask"]
          and any("ID map" in n for n in r["notes"]), f"{names} {r['notes']}")
    check("invert = cut: the holes are inside", masks["cut"][60, 60] == 1 and masks["cut"][5, 5] == 0
          and abs(masks["cut"].mean() - (50 * 70 + 60 * 110) / (W * H)) < 1e-6)
    check("16-bit depth keeps its range", abs(r["depth"][0, 0, 0] - 1.0) < 1e-4 and abs(r["depth"][-1, 0, 0] - 20000 / 65535) < 1e-3,
          f"{r['depth'][0, 0, 0]} {r['depth'][-1, 0, 0]}")
    check("a half-size normal pass is resized, with a note", r["normal"].shape == (H, W, 3) and any("normal" in n for n in r["notes"]))
    near = rp.load(d, "facade_a", depth="near is white")["depth"][..., 0]
    check("near is white: stretched to 0..1, the far top row black", near[0, 0] < 0.01 and near[-1, 0] > 0.99, f"{near[0, 0]} {near[-1, 0]}")
    b = rp.load(d, "FACADE_B")
    check("the other render (name without case) has only the shared mask and no depth", [n for n, _ in b["masks"]] == ["facade_mask"]
          and b["depth"] is None and b["normal"] is None)
    check("no name = the first render", rp.load(d)["name"] == "facade_a")
    check("exclude leaves a mask out", [n for n, _ in rp.load(d, "facade_a", exclude="facade*")["masks"]] == ["cut"])
    try:
        rp.load(d, "nothing")
        check("an unknown render name is refused and the folder's renders are named", False)
    except ValueError as e:
        check("an unknown render name is refused and the folder's renders are named", "facade_a" in str(e), str(e))

    # the masks as regions: every hole its own region, cut out of the wall (regions from masks, split_masks = cut)
    entries = [{"source": n, "name": n, "group": None, "mask": masks[n] > 0.5} for n in names]
    labels, atlas = fc.atlas_from_masks(entries, W, H, split_masks="cut", min_region_area=0)
    got = sorted(g["name"] for g in atlas["regions"])
    check("regions: cut_01, cut_02 and the wall around them", got == ["cut_01", "cut_02", "facade_mask"], str(got))
    wall = [g for g in atlas["regions"] if g["name"] == "facade_mask"][0]
    check("the wall region has the holes cut out", wall["area"] == int(facade.astype(bool).sum()) - (50 * 70 + 60 * 110), str(wall["area"]))

with tempfile.TemporaryDirectory() as d:
    save(d, "beauty.png", np.zeros((40, 60, 3), np.uint8))
    rgba = np.zeros((40, 60, 4), np.uint8)
    rgba[10:20, 10:30, 3] = 255                                            # a mask kept in the alpha channel
    save(d, "windows.png", rgba)
    r = rp.load(d)
    check("plain names (beauty.png, windows.png) and a mask in the alpha channel", r["name"] == ""
          and [n for n, _ in r["masks"]] == ["windows"] and r["masks"][0][1].sum() == 200, f"{r['name']!r} {r['masks'][0][1].sum()}")
with tempfile.TemporaryDirectory() as d:
    try:
        rp.load(d)
        check("an empty folder says how to name the picture", False)
    except ValueError as e:
        check("an empty folder says how to name the picture", "_beauty" in str(e), str(e))

print()
print("all render passes tests passed" if not failures else f"{len(failures)} FAILED: " + ", ".join(failures))
sys.exit(1 if failures else 0)
