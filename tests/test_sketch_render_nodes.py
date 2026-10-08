"""
Model free test of the sketch and from-renders NODES (nodes/sketch/nodes_sketch.py, nodes/from_renders/nodes_passes.py,
nodes_cryptomatte.py): the outputs added at the end (report, closed_lines, mask), the optional matrix and scope inputs,
EXR passes and pasted paths. The logic below them is tested in test_sketch.py, test_render_passes.py, test_cryptomatte.py.

Run from the ComfyUI_windows_portable folder:
    python_embeded\\python.exe ComfyUI\\custom_nodes\\ComfyUI_KubaNodes\\tests\\test_sketch_render_nodes.py
"""

import os
import shutil
import sys
import tempfile

import cv2
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(HERE))))     # ComfyUI (folder_paths, comfy_api)
sys.path.insert(0, os.path.dirname(HERE))
import folder_paths  # noqa: E402

TMP = tempfile.mkdtemp(prefix="kkd_sketch_")
folder_paths.set_temp_directory(os.path.join(TMP, "temp"))
os.makedirs(os.path.join(TMP, "temp"), exist_ok=True)

sys.path.insert(0, HERE)     # embedded Python leaves the script folder off the path
import _pack  # noqa: E402,F401  (the pack as a package, see tests/_pack.py)

import kubapack.nodes.from_renders.nodes_cryptomatte as nc  # noqa: E402
import kubapack.nodes.from_renders.nodes_passes as npa  # noqa: E402
import kubapack.nodes.sketch.nodes_sketch as ns  # noqa: E402
from kubakub import exr, samples  # noqa: E402
from kubakub import sample_facade as sf  # noqa: E402

failures = []


def check(name, cond, detail=""):
    print(("ok   " if cond else "FAIL ") + name + (f"  ({detail})" if detail and not cond else ""))
    if not cond:
        failures.append(name)


def refused(fn, *words):
    try:
        fn()
    except ValueError as e:
        return all(w in str(e) for w in words), str(e)
    return False, "no error"


def outs(node):
    return [o.id for o in node.define_schema().outputs]


def ins(node):
    return {i.id: i for i in node.define_schema().inputs}


# ---- schemas: the old outputs keep their place, the new ones are at the end, new inputs are optional
check("scan to line outputs", outs(ns.KUBA_ScanToLine) == ["drawing", "line", "projection", "found", "corners", "report"])
check("regions from sketch outputs", outs(ns.KUBA_RegionsFromSketch) ==
      ["regions", "region_masks", "regions_json", "preview", "report", "closed_lines"])
check("line overlay outputs", outs(ns.KUBA_LineOverlay) == ["image", "mask"])
check("render passes outputs", outs(npa.KUBA_RenderPasses) == ["beauty", "depth", "normal", "masks", "names", "report"])
check("regions from cryptomatte outputs", outs(nc.KUBA_RegionsFromCryptomatte) ==
      ["regions", "region_masks", "regions_json", "preview", "render", "report"])
for node, new in ((ns.KUBA_RegionsFromSketch, ("matrix", "scope")), (npa.KUBA_RenderPasses, ("matrix",)),
                  (nc.KUBA_RegionsFromCryptomatte, ("scope",))):
    i = ins(node)
    check(f"{node.__name__}: {', '.join(new)} optional, with a tooltip",
          all(i[k].optional and i[k].tooltip for k in new), str([(k, i[k].optional) for k in new]))
for node in (ns.KUBA_ScanToLine, ns.KUBA_RegionsFromSketch, ns.KUBA_LineOverlay):
    check(f"{node.__name__}: every output has a tooltip", all(o.tooltip for o in node.define_schema().outputs))

# ---- scan to line: a report line per photo
W, H = 960, 540
photo = sf.sketch_photo(W, H, 3, 7, 0)
other = np.ascontiguousarray(photo[:, ::-1])                           # a second frame: the same photo mirrored
batch = torch.from_numpy(np.stack([photo, other]).astype(np.float32))
res = ns.KUBA_ScanToLine.execute(batch, W, H, "pencil", 0.5, 1.0).result
rows = res[5].split("\n")
check("scan to line: one report line per photo, with how the paper was found and the corners",
      len(rows) == 2 and rows[0].startswith("photo 0: ") and rows[1].startswith("photo 1: ")
      and all("paper" in r and f"-> {W}x{H}" in r and "corners " in r for r in rows), res[5])
check("scan to line: the corners output is still the first photo's", res[4] in rows[0] and res[0].shape == (2, H, W, 3)
      and res[1].shape == (2, H, W))
res0 = ns.KUBA_ScanToLine.execute(batch[:1], 0, 0, "pencil", 0.5, 1.0, corners="10,10 900,12 890,600 12,610").result
check("scan to line: given corners are named in the report", "given corners" in res0[5] and "10,10" in res0[5], res0[5])
res2 = ns.KUBA_ScanToLine.execute(torch.cat([batch[:1], batch[:1]]), 0, 0, "pencil", 0.5, 1.0).result
check("scan to line: paper's own size, both frames share it", res2[0].shape[0] == 2 and len(res2[5].split("\n")) == 2)

# ---- regions from sketch: report, closed_lines, matrix, scope
drawing, line = res[0][:1], res[1][:1]
base = ns.KUBA_RegionsFromSketch.execute(drawing, 24, 0.3, True, "keep", 100, True, line=line).result
regions, closed = base[0], base[5]
n_base = len(regions.table["regions"])
check("regions from sketch: a report (regions, colours, bridges) and the closed lines", f"{n_base} regions in" in base[4]
      and "named by colour: " in base[4] and "red 1" in base[4] and "bridged" in base[4]
      and closed.shape == (1, H, W) and closed.dtype == torch.float32, base[4])
thr = (line[0] > 0.3).numpy()
cl = closed[0].numpy() > 0.5
check("closed_lines holds the strokes and more: the bridges", 0 < cl.sum() and (cl & thr).sum() > 0.8 * thr.sum()
      and (cl & ~cv2.dilate(thr.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)).sum() > 0,
      f"{cl.sum()} {thr.sum()}")
check("no scope connected: the regions carry none", regions.scope is None)
open_ = ns.KUBA_RegionsFromSketch.execute(drawing, 0, 0.3, True, "keep", 100, True, line=line).result
check("gap_px 0: nothing bridged, fewer closed lines", "0 gap(s)" in open_[4] and open_[5].sum() < closed.sum())

scope = torch.zeros((1, H, W))
scope[:, :, : W // 2] = 1.0
sc = ns.KUBA_RegionsFromSketch.execute(drawing, 24, 0.3, True, "keep", 100, True, line=line, scope=scope).result
lab = sc[0].labels[0].numpy()
check("scope: no region outside it, fewer regions, said in the report", (lab[:, W // 2:] == -1).all() and (lab[:, : W // 2] >= 0).any()
      and len(sc[0].table["regions"]) < n_base and "scope" in sc[4] and sc[0].scope is not None
      and float(sc[0].scope.sum()) == H * (W // 2), sc[4])
drop = ns.KUBA_RegionsFromSketch.execute(drawing, 24, 0.3, True, "drop", 100, True, line=line, scope=scope).result
check("scope and outside = drop work together", (drop[0].labels[0].numpy()[:, W // 2:] == -1).all()
      and float(drop[0].scope.sum()) < H * (W // 2) and "touching the edge dropped" in drop[4], drop[4])
ok, msg = refused(lambda: ns.KUBA_RegionsFromSketch.execute(drawing, 24, 0.3, True, "keep", 100, True, line=line,
                                                           scope=torch.ones((1, 50, 60))), "scope", "60x50", f"{W}x{H}")
check("a scope of another size is refused in plain words", ok, msg)

matrix = torch.full((1, H, W, 3), 0.25)
mx = ns.KUBA_RegionsFromSketch.execute(drawing, 24, 0.3, True, "keep", 100, True, line=line, matrix=matrix).result
check("matrix of the drawing's size: the same regions, the matrix behind the preview",
      np.array_equal(mx[0].labels.numpy(), regions.labels.numpy()) and not torch.equal(mx[3], base[3])
      and "fitted" not in mx[4])
big = torch.full((1, H * 2, W * 2, 3), 0.25)
bx = ns.KUBA_RegionsFromSketch.execute(drawing, 24, 0.3, True, "keep", 400, True, line=line, matrix=big).result
check("a matrix twice the size: the drawing is fitted to it, with a note; gap_px stays in drawing pixels",
      bx[0].labels.shape == (1, H * 2, W * 2) and bx[3].shape == (1, H * 2, W * 2, 3) and bx[5].shape == (1, H * 2, W * 2)
      and f"fitted to the matrix {W * 2}x{H * 2}" in bx[4] and abs(len(bx[0].table["regions"]) - n_base) <= 0.2 * n_base,
      f"{bx[4]} / {n_base}")
ok, msg = refused(lambda: ns.KUBA_RegionsFromSketch.execute(drawing, 24, 0.3, True, "keep", 100, True, line=line,
                                                           matrix=torch.zeros((1, 512, 512, 3))), "another aspect", "512x512")
check("a matrix of another aspect is refused in plain words", ok, msg)
old = ns.KUBA_RegionsFromSketch.execute(drawing, 24, 0.3, True, "keep", 100, True).result
check("without line, matrix and scope the node runs as before", len(old) == 6 and len(old[0].table["regions"]) > 5)

# ---- line overlay: the mask of the line as laid on
img = torch.full((2, H, W, 3), 0.8)
lo = ns.KUBA_LineOverlay.execute(img, line, 1.0, "multiply", "#000000", 0, 0.0).result
check("line overlay: mask = the line, one per frame", lo[1].shape == (2, H, W) and torch.allclose(lo[1][0], line[0].clamp(0, 1))
      and lo[0].shape == (2, H, W, 3))
lg = ns.KUBA_LineOverlay.execute(img, line, 0.5, "multiply", "#000000", 4, 2.0).result
check("line overlay: the mask is grown and softened, and does not depend on amount", float(lg[1].sum()) > 1.5 * float(lo[1].sum())
      and 0.0 <= float(lg[1].min()) and float(lg[1].max()) <= 1.0
      and torch.allclose(lg[0][0, ..., 0], 0.8 * (1 - 0.5 * lg[1][0]), atol=1e-5))
half = ns.KUBA_LineOverlay.execute(torch.full((1, H // 2, W // 2, 3), 0.8), line, 1.0, "screen", "#ffffff", 0, 0.0).result
check("line overlay: the mask has the image's size", half[1].shape == (1, H // 2, W // 2))

# ---- render passes: sample, matrix, EXR
rp_s = npa.KUBA_RenderPasses.execute("").result
sh, sw = rp_s[0].shape[1:3]
check("render passes, empty folder: the sample, said in the report", "sample" in rp_s[5] and rp_s[3].shape[1:] == (sh, sw))
rp_m = npa.KUBA_RenderPasses.execute("", matrix=torch.zeros((1, sh // 2, sw // 2, 3))).result
check("render passes + matrix: picture, depth, normal and masks at the matrix size, with a note",
      rp_m[0].shape == (1, sh // 2, sw // 2, 3) and rp_m[1].shape == rp_m[0].shape and rp_m[2].shape == rp_m[0].shape
      and rp_m[3].shape[1:] == (sh // 2, sw // 2) and "fitted to the matrix" in rp_m[5] and rp_m[4] == rp_s[4], rp_m[5])
rp_x = npa.KUBA_RenderPasses.execute("", matrix=torch.zeros((1, 400, 400, 3))).result
check("the sample with a matrix of other proportions still runs, at its own size, and says so",
      rp_x[0].shape == rp_s[0].shape and "other proportions" in rp_x[5], rp_x[5])
folder = os.path.join(TMP, "exr passes")
os.makedirs(folder)
f32 = lambda v: np.full((90, 160), v, np.float32)  # noqa: E731
exr.write(os.path.join(folder, "shot_beauty.exr"), {"R": f32(0.2), "G": f32(0.2), "B": f32(0.2)})
exr.write(os.path.join(folder, "shot_depth.exr"), {"Z": np.tile(np.linspace(9, 3, 90, dtype=np.float32)[:, None], (1, 160))})
win = np.zeros((90, 160), np.float32)
win[20:40, 30:70] = 1
exr.write(os.path.join(folder, "shot_windows.exr"), {"A": win})
rx = npa.KUBA_RenderPasses.execute(f'"{folder}"').result
check("render passes: a folder of EXR passes (pasted with quotes)", rx[0].shape == (1, 90, 160, 3) and rx[4] == "windows"
      and float(rx[3].sum()) == 800 and abs(float(rx[0][0, 0, 0, 0]) - 0.4845) < 1e-3
      and float(rx[1].max()) == 1.0 and float(rx[1].min()) == 0.0 and "depth: yes" in rx[5], rx[5])
ok, msg = refused(lambda: npa.KUBA_RenderPasses.execute(folder, matrix=torch.zeros((1, 160, 160, 3))), "another aspect", "160x90")
check("render passes: your own folder with a matrix of another aspect is refused", ok, msg)
ri = ins(npa.KUBA_RenderPasses)
check("render passes: exr_layers is optional, the last input, with a tooltip", ri["exr_layers"].optional and ri["exr_layers"].tooltip
      and list(ri) == ["folder", "render", "invert", "exclude", "depth", "matrix", "exr_layers"], str(list(ri)))
one = os.path.join(TMP, "one file", "shot.exr")
os.makedirs(os.path.dirname(one))
door = np.zeros((90, 160), np.float32)
door[50:90, 100:120] = 1
exr.write(one, {"C.r": f32(0.2), "C.g": f32(0.2), "C.b": f32(0.2), "Z_cam.Z": np.tile(np.linspace(9, 3, 90, dtype=np.float32)[:, None], (1, 160)),
                "N.x": f32(0.0), "N.y": f32(1.0), "N.z": f32(0.0), "windows.Y": win, "door.Y": door,
                "diffuse.R": f32(0.8), "diffuse.G": f32(0.1), "diffuse.B": f32(0.1)})
rl = npa.KUBA_RenderPasses.execute(f'"{one}"').result
check("render passes: one multilayer EXR in 'folder': picture, depth, normal and its mask layers",
      rl[0].shape == (1, 90, 160, 3) and abs(float(rl[0][0, 0, 0, 0]) - 0.4845) < 1e-3 and float(rl[1].max()) == 1.0
      and float(rl[1].min()) == 0.0 and abs(float(rl[2][0, 0, 0, 1]) - 1.0) < 1e-6 and rl[4] == "door\nwindows"
      and rl[3].shape == (2, 90, 160) and float(rl[3][1].sum()) == 800, rl[5])
check("render passes: the report lists every layer of the file, its channels and what it was used as",
      "layers in shot.exr:" in rl[5] and "  C (r g b): picture" in rl[5] and "  Z_cam (Z): depth" in rl[5]
      and "  windows (Y): mask" in rl[5] and "  diffuse (R G B): not read" in rl[5], rl[5])
rk = npa.KUBA_RenderPasses.execute(one, exr_layers="win*\nmask = diffuse").result
check("render passes: exr_layers picks the layers (a wildcard, a forced mask)", rk[4] == "diffuse\nwindows"
      and rk[3].shape == (2, 90, 160) and "  door (Y): not read" in rk[5], rk[5])
check("render passes: the fingerprint follows the EXR file and the picked layers",
      npa.KUBA_RenderPasses.fingerprint_inputs(one) != npa.KUBA_RenderPasses.fingerprint_inputs(one, exr_layers="door")
      and str(os.stat(one).st_mtime_ns) in npa.KUBA_RenderPasses.fingerprint_inputs(f'"{one}"'))

# ---- regions from cryptomatte: scope, pasted paths
cr = nc.KUBA_RegionsFromCryptomatte.execute("", "object", "material", "", 0, True).result
ch, cw = cr[0].labels.shape[1:]
n_all = len(cr[0].table["regions"])
check("cryptomatte, empty file: the sample", "built-in sample" in cr[5] and n_all > 3 and cr[0].scope is None)
sc = torch.zeros((1, ch, cw))
sc[:, :, : cw // 2] = 1.0
cs = nc.KUBA_RegionsFromCryptomatte.execute("", "object", "material", "", 0, True, scope=sc).result
lab = cs[0].labels[0].numpy()
check("cryptomatte + scope: no region outside it, fewer objects, the regions carry the scope",
      (lab[:, cw // 2:] == -1).all() and (lab[:, : cw // 2] >= 0).any() and 0 < len(cs[0].table["regions"]) < n_all
      and cs[0].scope is not None and float(cs[0].scope.sum()) == ch * (cw // 2) and "scope" in cs[5]
      and cs[1].shape[0] == len(cs[0].table["regions"]) and all(r["area"] > 0 for r in cs[0].table["regions"]), cs[5])
ok, msg = refused(lambda: nc.KUBA_RegionsFromCryptomatte.execute("", "object", "material", "", 0, True,
                                                                scope=torch.ones((1, 30, 40))), "scope", "40x30")
check("cryptomatte: a scope of another size is refused in plain words", ok, msg)
half = torch.zeros((1, ch // 2, cw // 2, 3))
hs = torch.ones((1, ch // 2, cw // 2))
hs[:, : ch // 4] = 0
cm_ = nc.KUBA_RegionsFromCryptomatte.execute("", "object", "material", "", 0, True, matrix=half, scope=hs).result
check("cryptomatte: with a matrix the scope has the matrix's size", cm_[0].labels.shape == (1, ch // 2, cw // 2)
      and (cm_[0].labels[0, : ch // 4].numpy() == -1).all())
path = samples.cryptomatte(folder_paths.get_temp_directory())
os.environ["KKD_TEST_DIR"] = os.path.dirname(path)
for how, text in (("double quotes", f'  "{path}" '), ("single quotes", f"'{path}'"),
                  ("a %VARIABLE%", os.path.join("%KKD_TEST_DIR%", os.path.basename(path)))):
    q = nc.KUBA_RegionsFromCryptomatte.execute(text, "object", "material", "", 0, True).result
    check(f"cryptomatte: a path pasted with {how}", len(q[0].table["regions"]) == n_all and "built-in sample" not in q[5], q[5])
    check(f"cryptomatte: the fingerprint follows the file ({how})",
          nc.KUBA_RegionsFromCryptomatte.fingerprint_inputs(text).endswith(str(os.path.getmtime(path))))

shutil.rmtree(TMP, ignore_errors=True)
print()
print("all sketch / render node tests passed" if not failures else f"{len(failures)} FAILED: " + ", ".join(failures))
sys.exit(1 if failures else 0)
