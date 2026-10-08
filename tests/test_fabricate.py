"""
Model free test of the fabricate nodes (nodes/fabricate): an empty 'mesh_path' runs the built-in sample model and says
so, the 'height' mask of the relief nodes, 'clearance_mm' and the 'flagged' masks of the turntable check, and a second
prow wing with another pixel size keeping its size in mm.

Run from the ComfyUI_windows_portable folder:
    python_embeded\\python.exe ComfyUI\\custom_nodes\\ComfyUI_KubaNodes\\tests\\test_fabricate.py
"""

import json
import os
import shutil
import sys
import tempfile

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(HERE))))     # ComfyUI (folder_paths)
sys.path.insert(0, os.path.dirname(HERE))
import folder_paths  # noqa: E402

TMP = tempfile.mkdtemp(prefix="kub_fabricate_test_")
folder_paths.set_temp_directory(os.path.join(TMP, "temp"))
folder_paths.set_output_directory(os.path.join(TMP, "output"))
os.makedirs(os.path.join(TMP, "output"))

sys.path.insert(0, HERE)     # embedded Python leaves the script folder off the path
import _pack  # noqa: E402,F401  (the pack as a package, see tests/_pack.py)
from kubapack.nodes.fabricate import mesh_to_field_Kub as mtf  # noqa: E402
from kubapack.nodes.fabricate import relief_forge_Kub as rf  # noqa: E402

failures = []


def check(name, cond, detail=""):
    print(("ok   " if cond else "FAIL ") + name + (f"  ({detail})" if detail else ""))
    if not cond:
        failures.append(name)


def obj_verts(path):
    wings, cur = {}, None
    verts = [[float(x) for x in ln.split()[1:4]] for ln in open(path) if ln.startswith("v ")]
    for ln in open(path):
        if ln.startswith("g "):
            cur = wings.setdefault(ln.split()[1], set())
        elif ln.startswith("f "):
            cur.update(int(t) - 1 for t in ln.split()[1:])
    V = np.array(verts)
    return {k: V[sorted(v)] for k, v in wings.items()}


try:
    # ---- the two mesh nodes: outputs stay in place, new ones at the end
    check("mesh to relief field: outputs appended at the end",
          mtf.MeshOrthoFieldKub.RETURN_NAMES == ("field", "preview", "undercut", "report", "height")
          and len(mtf.MeshOrthoFieldKub.RETURN_TYPES) == len(mtf.MeshOrthoFieldKub.OUTPUT_TOOLTIPS) == 5)
    check("turntable: outputs appended at the end",
          mtf.MeshTurntableCheckKub.RETURN_NAMES == ("overlays", "report", "passes", "flagged")
          and len(mtf.MeshTurntableCheckKub.RETURN_TYPES) == len(mtf.MeshTurntableCheckKub.OUTPUT_TOOLTIPS) == 4)
    opt = mtf.MeshTurntableCheckKub.INPUT_TYPES()["optional"]
    check("turntable: clearance_mm is optional, advanced, with a tooltip and today's value",
          opt["clearance_mm"][1]["default"] == 150.0 and opt["clearance_mm"][1]["advanced"] and opt["clearance_mm"][1]["tooltip"])
    for mod in (mtf, rf):
        for cid, cls in mod.NODE_CLASS_MAPPINGS.items():
            it = cls.INPUT_TYPES()
            missing = [k for part in ("required", "optional") for k, v in it.get(part, {}).items()
                       if not (len(v) > 1 and v[1].get("tooltip"))]
            check(f"{cid}: every input has a tooltip", not missing, ", ".join(missing))

    # ---- empty mesh_path = the sample model
    field, prev, undercut, report, height = mtf.MeshOrthoFieldKub().run("", "m", "Y", 0.0, 0.0, 128, 0.0, 0.0)
    rep = json.loads(report)
    check("empty mesh_path runs the sample and the report says so", "built-in sample" in rep.get("note", "") and "mesh_path" in rep["note"],
          rep.get("note", ""))
    check("the sample comes out at its real size (24 m wide facade)", 24000 < rep["panel_width_mm"] < 26500 and 15500 < rep["panel_height_mm"] < 17500,
          f"{rep['panel_width_mm']} x {rep['panel_height_mm']} mm")
    h = field["height_mm"]
    check("height is a mask of the field, 0..1", tuple(height.shape) == (1,) + h.shape and height.dtype == torch.float32
          and float(height.min()) == 0.0 and abs(float(height.max()) - 1.0) < 1e-6)
    check("height is the preview's picture", torch.allclose(height, prev[..., 0]))
    lo, hi = rep["height_range_mm"]
    check("the report gives the mm range of height", abs(lo - float(h.min())) < 0.06 and abs(hi - float(h.max())) < 0.06, f"{lo} .. {hi}")
    check("height * range gives the field back", np.abs(height[0].numpy() * (h.max() - h.min()) + h.min() - h).max() < 1e-2)
    rep_cm = json.loads(mtf.MeshOrthoFieldKub().run("  ", "cm", "Y", 0.0, 0.0, 128, 0.0, 0.0)[3])
    check("the sample with other units says which settings fit it", "mesh_units = m" in rep_cm["note"])
    check("IS_CHANGED works on an empty field", mtf.MeshOrthoFieldKub.IS_CHANGED("") == mtf.MeshOrthoFieldKub.IS_CHANGED(""))
    try:
        mtf.MeshOrthoFieldKub().run(os.path.join(TMP, "nope.obj"), "m", "Y", 0.0, 0.0, 128, 0.0, 0.0)
        check("a wrong path still stops with a plain message", False)
    except FileNotFoundError as e:
        check("a wrong path still stops with a plain message", "mesh_path" in str(e))

    # a pasted file has no note
    own = os.path.join(TMP, "step.obj")       # a wall 2 m wide, 2 m high with a ledge 0.2 m deep at 1 m
    with open(own, "w") as f:
        f.write("v -1 0 0\nv 1 0 0\nv 1 1 0\nv -1 1 0\nv -1 1 -0.2\nv 1 1 -0.2\nv 1 2 -0.2\nv -1 2 -0.2\n"
                "v -1 0 0.2\nv 1 0 0.2\nv 1 1 0.2\nv -1 1 0.2\n"
                "f 9 10 11 12\nf 5 6 7 8\nf 12 11 6 5\n")
    out = mtf.MeshOrthoFieldKub().run(own, "m", "Y", 0.0, 0.0, 128, 0.0, 0.0)
    check("a pasted mesh has no sample note", "note" not in json.loads(out[3]))

    # ---- turntable: flagged masks + clearance
    tt = mtf.MeshTurntableCheckKub()
    overlays, report, passes, flagged = tt.run(own, "m", "Y", 2, 0.0, 128, 0.0, 2500.0, 25.0, 80.0)
    rep = json.loads(report)
    check("turntable: one flagged mask per view at the overlay size",
          tuple(flagged.shape) == tuple(overlays.shape[:3]) and flagged.dtype == torch.float32, f"{tuple(flagged.shape)}")
    check("turntable: the ledge is flagged in the front view and the mask shows it",
          rep["per_view"][0]["footholds_found"] > 0 and float(flagged[0].sum()) >= rep["per_view"][0]["footholds_found"] and not passes)
    red = (overlays[0, ..., 0] > 0.6) & (overlays[0, ..., 1] < 0.4)
    check("turntable: the mask covers what the overlay paints", bool((flagged[0] > 0.5)[red].all()) and bool(red.any()))
    check("turntable: default clearance is the old silent value", rep["clearance_mm"] == 150.0)
    same = json.loads(tt.run(own, "m", "Y", 2, 0.0, 128, 0.0, 2500.0, 25.0, 80.0, clearance_mm=150.0)[1])
    check("turntable: clearance_mm = 150 gives the same result as before", same["per_view"] == rep["per_view"])
    # a 300 mm band looks past a slot that a 50 mm band stops inside: the slot's floor no longer counts
    deep = os.path.join(TMP, "slope.obj")     # ledge at 1 m, and 0.1 m above it the wall comes forward again
    with open(deep, "w") as f:
        f.write("v -1 0 0.2\nv 1 0 0.2\nv 1 1 0.2\nv -1 1 0.2\nv -1 1 0\nv 1 1 0\nv 1 1.1 0\nv -1 1.1 0\n"
                "v -1 1.1 0.2\nv 1 1.1 0.2\nv 1 2 0.2\nv -1 2 0.2\n"
                "f 1 2 3 4\nf 5 6 7 8\nf 9 10 11 12\nf 4 3 6 5\nf 8 7 10 9\n")
    near = json.loads(tt.run(deep, "m", "Y", 1, 0.0, 128, 0.0, 2500.0, 25.0, 80.0, clearance_mm=50.0)[1])
    far = json.loads(tt.run(deep, "m", "Y", 1, 0.0, 128, 0.0, 2500.0, 25.0, 80.0, clearance_mm=300.0)[1])
    check("turntable: clearance_mm changes what counts (a slot 100 mm high; the top of the block stays flagged)",
          near["total_footholds"] > far["total_footholds"] > 0 and near["clearance_mm"] == 50.0,
          f"{near['total_footholds']} / {far['total_footholds']}")
    s_over, s_rep, _, s_flag = tt.run("", "m", "Y", 1, 0.0, 128, 0.0, 2500.0, 25.0, 80.0)
    check("turntable: empty mesh_path runs the sample and says so", "built-in sample" in json.loads(s_rep).get("note", "")
          and tuple(s_flag.shape) == tuple(s_over.shape[:3]))

    # ---- relief nodes: height
    for cid, names in (("ReliefFieldKub", ("field", "preview", "height")), ("ReliefMouldKub", ("field", "preview", "report", "height")),
                       ("ReliefAntiPerchKub", ("field", "preview", "height"))):
        cls = rf.NODE_CLASS_MAPPINGS[cid]
        check(f"{cid}: height appended at the end", cls.RETURN_NAMES == names and cls.RETURN_TYPES[-1] == "MASK"
              and len(cls.OUTPUT_TOOLTIPS) == len(names))
    yy, xx = np.mgrid[0:60, 0:80].astype(np.float32)
    depth = torch.from_numpy(np.stack([np.clip(1 - np.hypot(xx - 40, yy - 30) / 30, 0, 1)] * 3, -1))[None]
    field, prev, height = rf.ReliefFieldKub().run(depth, 800.0, 120.0, 300.0, False, 1.0, 0.0)
    check("relief field: height = the field / max_relief_mm", tuple(height.shape) == (1, 60, 80)
          and np.abs(height[0].numpy() * 120.0 - field["height_mm"]).max() < 1e-3 and torch.allclose(height, prev[..., 0]))
    mf, mprev, mrep, mheight = rf.ReliefMouldKub().run(field, 20.0, 4.0, 1024)
    lo, hi = json.loads(mrep)["height_range_mm"]
    check("mould prep: the report gives the mm range of height",
          np.abs(mheight[0].numpy() * (hi - lo) + lo - mf["height_mm"]).max() < 0.11, f"{lo} .. {hi}")
    af, aprev, aheight = rf.ReliefAntiPerchKub().run(mf, 45.0, 400.0)
    check("anti perch: height is the treated field 0..1", tuple(aheight.shape) == (1, 60, 80) and torch.allclose(aheight, aprev[..., 0]))
    flat = rf._make_field(np.full((8, 8), 5.0, np.float32), 10.0)
    check("a flat field gives an all zero height, no NaN", float(rf._height(flat).abs().max()) == 0.0)

    # ---- prow: field_b with another pixel size keeps its size in mm
    node = rf.ReliefExportProwKub()
    a = rf._make_field(np.zeros((40, 80), np.float32), 10.0)                    # 800 x 400 mm
    b_fine = rf._make_field(np.zeros((80, 120), np.float32), 5.0)               # 600 x 400 mm at twice the resolution
    b_same = rf._make_field(np.zeros((40, 60), np.float32), 10.0)               # 600 x 400 mm
    p_fine = obj_verts(node.run(a, 180.0, "kub_test/prow", 0.0, 1, field_b=b_fine)[0])
    p_same = obj_verts(node.run(a, 180.0, "kub_test/prow", 0.0, 1, field_b=b_same)[0])
    size = lambda v: np.ptp(v, axis=0)                                           # noqa: E731
    check("prow: wing a is 800 x 400 mm", abs(size(p_fine["wing_0"])[0] - 790) < 1 and abs(size(p_fine["wing_0"])[2] - 390) < 1,
          str(size(p_fine["wing_0"])))
    check("prow: a finer field_b keeps its 600 x 400 mm (was 1200 x 800 at field_a's pixel size)",
          np.abs(size(p_fine["wing_1"]) - size(p_same["wing_1"])).max() < 1e-3 and abs(size(p_fine["wing_1"])[0] - 590) < 1,
          str(size(p_fine["wing_1"])))
    ramp = rf._make_field(np.tile(np.linspace(0, 50, 120, dtype=np.float32), (80, 1)), 5.0)
    check("prow: resampling keeps the heights in mm", abs(float(rf._to_px_mm(ramp, 10.0).max()) - 50.0) < 1.0
          and rf._to_px_mm(ramp, 10.0).shape == (40, 60) and rf._to_px_mm(ramp, 5.0) is not None
          and rf._to_px_mm(ramp, 5.0).shape == (80, 120))
    p_one = obj_verts(node.run(a, 90.0, "kub_test/prow", 0.0, 1)[0])
    check("prow: one field is still mirrored onto both wings", np.allclose(size(p_one["wing_0"]), size(p_one["wing_1"]), atol=1e-3))
finally:
    shutil.rmtree(TMP, ignore_errors=True)

print()
if failures:
    print(f"{len(failures)} FAILED: {', '.join(failures)}")
    sys.exit(1)
print("all fabricate tests passed")
