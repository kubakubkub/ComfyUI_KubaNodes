"""
Model free test of the built-in samples (kubakub/samples.py): what each node uses when its file or folder field is
empty. Every sample is written and then read by the logic of the node it is for.

Run from the ComfyUI_windows_portable folder:
    python_embeded\\python.exe ComfyUI\\custom_nodes\\ComfyUI_KubaNodes\\tests\\test_samples.py
"""

import os
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from kubakub import cryptomatte as cm  # noqa: E402
from kubakub import exr  # noqa: E402
from kubakub import facade_core as fc  # noqa: E402
from kubakub import idmaps as im  # noqa: E402
from kubakub import render_passes as rp  # noqa: E402
from kubakub import samples  # noqa: E402

failures = []


def check(name, cond, detail=""):
    print(("ok   " if cond else "FAIL ") + name + (f"  ({detail})" if detail and not cond else ""))
    if not cond:
        failures.append(name)


with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as root:   # the PDF reader keeps its file open
    obj = samples.model(root)
    check("3D file: an .obj with its .mtl", os.path.isfile(obj) and os.path.isfile(obj[:-4] + ".mtl"))
    stamp = os.stat(obj).st_mtime_ns
    check("... written once, the same path the second time", samples.model(root) == obj and os.stat(obj).st_mtime_ns == stamp)

    folder = samples.passes(root, 640, 360)
    r = rp.load(folder)
    names = [n for n, _ in r["masks"]]
    check("render passes: picture, depth and the masks windows / door / facade_mask", r["name"] == "sample"
          and r["beauty"].shape == (360, 640, 3) and r["depth"] is not None and names == ["door", "windows", "facade_mask"],
          f"{r['name']} {names} {r['notes']}")
    masks = dict(r["masks"])
    check("... the windows lie inside the facade mask", masks["windows"].sum() > 1000
          and ((masks["windows"] > 0.5) & ~(masks["facade_mask"] > 0.5)).sum() == 0)

    folder = samples.id_renders(root, 640, 360)
    passes, ref = im.discover(folder)
    check("ID renders: the passes elements and level, and a clay", set(passes) == {"elements", "level"} and ref is not None,
          f"{list(passes)} {ref}")
    b = im.build(folder, regions_pass="elements", tag_passes="level", split_parts="*", min_region_area=16)
    groups = {g["group_id"] for g in b["atlas"]["regions"]}
    n_win = sum(1 for g in b["atlas"]["regions"] if g["group_id"] == "windows")
    check("... read into regions: 27 windows, a door, the wall", n_win == 27 and {"windows", "door", "wall"} <= groups,
          f"{n_win} windows, groups {sorted(groups)}")
    tagged = [g for g in b["atlas"]["regions"] if g["group_id"] == "windows" and any(t.startswith("level_") for t in g.get("tags", []))]
    check("... windows carry their floor as a tag", len(tagged) == 27, str(len(tagged)))

    path = samples.cryptomatte(root, 640, 360)
    data = exr.read(path)
    lays = cm.layers(data["attrs"])
    main = cm.pick_layer(list(lays), "object")
    labels, names = cm.label_map(data["channels"], main, lays[main]["manifest"], "")
    check("cryptomatte: objects by name (window_f1_c01 ...), a material layer, the picture",
          "window_f1_c01" in names and "door" in names and len(names) > 40 and cm.pick_layer(list(lays), "material")
          and cm.beauty(data["channels"]) is not None, f"{len(names)} names {names[:4]}")
    tags = cm.tags_from(data["channels"], cm.pick_layer([n for n in lays if n != main], "material"),
                        lays[cm.pick_layer(list(lays), "material")]["manifest"], labels, len(names))
    check("... windows are glass", tags[names.index("window_f1_c01")] == ["glass"], str(tags[names.index("window_f1_c01")]))

    folder = samples.mask_folder(root, 640, 360)
    lab, meta, notes, scope, tg = fc.label_mask_folder(folder, 640, 360, recursive=True)
    check("mask folder: Groups / Windows read by the mask atlas", lab.shape == (360, 640) and len(meta) >= 27, str(len(meta)))
    check("... one folder per size", samples.mask_folder(root, 320, 180) != folder)

    try:
        import fitz  # noqa: F401
    except ImportError:
        fitz = None
        print("skip the layered PDF (PyMuPDF not installed)")
    if fitz:
        from kubakub import illustrator as il
        pdf = samples.illustrator(root)
        b = il.build(pdf, min_region_area=16)
        roles = b["roles"]
        n_win = sum(1 for g in b["atlas"]["regions"] if str(g["group_id"]).lower().startswith("windows"))
        check("layered PDF: MASK = outside, LINES = lines, the windows and the door as shapes",
              roles.get("MASK") == "outside" and roles.get("LINES") == "lines" and n_win == 27,
              f"{roles}, {n_win} window regions, {len(b['atlas']['regions'])} regions")
    check("the note names the field to paste into", "file" in samples.note("3D file", "file") and "sample" in samples.note("x", "y"))

print()
print("all samples tests passed" if not failures else f"{len(failures)} FAILED: " + ", ".join(failures))
sys.exit(1 if failures else 0)
