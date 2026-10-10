"""
Model free test of kubakub form height (kubakub/formheight.py, nodes/scene3d/nodes_formheight.py): a flat wall as the
scene, and models with a box, a thin panel and a speck in front of it. The model is drawn on the graphics card, so
this needs the GL that comes with ComfyUI (as the shader tests do).

Run from the ComfyUI_windows_portable folder:
    python_embeded\\python.exe ComfyUI\\custom_nodes\\ComfyUI_KubaNodes\\tests\\test_formheight.py
"""

import json
import os
import shutil
import sys
import tempfile

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(HERE))))     # ComfyUI (folder_paths, comfy_api)
sys.path.insert(0, os.path.dirname(HERE))
import folder_paths  # noqa: E402

TMP = tempfile.mkdtemp(prefix="kkd_formheight_")
folder_paths.set_output_directory(os.path.join(TMP, "output"))
folder_paths.set_temp_directory(os.path.join(TMP, "temp"))

sys.path.insert(0, HERE)     # embedded Python leaves the script folder off the path
import _pack  # noqa: E402,F401  (the pack as a package, see tests/_pack.py)

import kubapack.nodes.scene3d.nodes_formheight as nf  # noqa: E402
from kubakub import formheight as fh  # noqa: E402
from kubakub.scene3d import scene_ids as si  # noqa: E402
from kubakub.scene3d import scene_view as sv  # noqa: E402

failures = []


def check(name, cond, detail=""):
    print(("ok   " if cond else "FAIL ") + name + (f"  ({detail})" if detail and not cond else ""))
    if not cond:
        failures.append(name)


W, H = 260, 140                                    # an ortho camera 13 x 7 m: 20 pixels per metre
CAM = np.eye(4)
CAM[:3, :3] = [[1, 0, 0], [0, 0, -1], [0, 1, 0]]  # looks along +y, z is up (Blender's front view)
CAM[:3, 3] = (6, -30, 3)


def make_scene(folder):
    """A wall 10 x 6 m with its front at y = 0, as blender_export.py writes a scene."""
    xs = -0.5 + (np.arange(W) + 0.5) * 13.0 / W
    zs = 6.5 - (np.arange(H) + 0.5) * 7.0 / H
    X, Z = np.meshgrid(xs, zs)
    on = (X >= 0) & (X <= 10) & (Z >= 0) & (Z <= 6)
    pos = np.where(on[..., None], np.stack([X, np.zeros_like(X), Z], -1), 0).astype(np.float32)
    nrm = np.where(on[..., None], np.array([0, -1.0, 0]), 0)
    os.makedirs(folder, exist_ok=True)
    np.save(os.path.join(folder, "faceid.npy"), on.astype(np.uint32))
    np.save(os.path.join(folder, "position.npy"), pos)
    np.save(os.path.join(folder, "normal.npy"), nrm.astype(np.float16))
    np.savez_compressed(os.path.join(folder, "mesh.npz"), vert=np.array([[0, 0, 0], [10, 0, 0], [10, 0, 6], [0, 0, 6]], np.float32))
    info = {"file": "synthetic", "width": W, "height": H,
            "camera": {"name": "cam", "type": "ORTHO", "matrix_world": CAM.tolist(),
                       "projection": [[2 / 13, 0, 0, 0], [0, 2 / 7, 0, 0], [0, 0, -2 / 99.9, -100.1 / 99.9], [0, 0, 0, 1]]}}
    json.dump(info, open(os.path.join(folder, "scene.json"), "w"))
    return on


def panel(x0, x1, z0, z1, y):
    """A flat piece facing the camera at depth y (scene world, z up) -> (vertices, faces)."""
    return np.array([[x0, y, z0], [x1, y, z0], [x1, y, z1], [x0, y, z1]], float), np.array([[0, 1, 2], [0, 2, 3]])


def join(*parts):
    v, f, at = [], [], 0
    for pv, pf in parts:
        v.append(pv)
        f.append(pf + at)
        at += len(pv)
    return np.concatenate(v), np.concatenate(f)


folder = os.path.join(TMP, "scene")
on = make_scene(folder)
scene = si.load(folder)
w = fh.wall(scene)
cam = w["cam"]
check("wall: the real wall where the scene saw it, the wall's plane beside it, 5 cm per pixel",
      np.allclose(w["depth"], 30.0, atol=1e-3) and w["seen"].sum() == on.sum() and abs(w["pixel_m"] - 0.05) < 1e-6
      and np.allclose(w["out"], 1.0, atol=1e-5), f"{w['depth'].min()} {w['depth'].max()} {w['pixel_m']}")

# the model: its own wall 0.1 m in front of the real one, a box 1.5 m out, a speck 1 m out, a form past the outline,
# and a line 10 cm thin along the wall (a cornice that sits a little off)
MODEL = join(panel(0, 10, 0, 6, -0.1), panel(4, 6, 2, 4, -1.5), panel(1, 1.2, 1, 1.2, -1.0), panel(9.5, 11, 5, 5.8, -0.8),
             panel(0, 10, 4.5, 4.6, -0.5))
try:
    depth = fh.model_depth(cam, *MODEL)
except Exception as e:  # noqa: BLE001
    print(f"skip: no graphics context here ({e})")
    shutil.rmtree(TMP, ignore_errors=True)
    sys.exit(0)
box = np.zeros((H, W), bool)
box[50:90, 90:130] = True                         # x 4..6, z 2..4 at 20 px per metre
check("depth: the box is drawn where the camera sees it, to the pixel, at its distance",
      depth.shape == (H, W) and np.allclose(depth[box], 28.5, atol=1e-3) and not np.isclose(depth[~box], 28.5, atol=1e-3).any()
      and depth[0, 0] == 0, f"{np.isclose(depth, 28.5, atol=1e-3).sum()} px of {box.sum()}")
check("depth: what is nearer hides what is behind it", abs(float(depth[70, 110]) - 28.5) < 1e-3 and abs(float(depth[20, 20]) - 29.9) < 1e-3)

h, forms = fh.height(depth, w, wall_m=0.3, smallest_form_m=0.5)
past = np.zeros((H, W), bool)
past[14:30, 200:230] = True                       # x 9.5..11, z 5..5.8
check("height: counted in metres from wall_m, 0 on the wall", np.allclose(h[box], 1.2, atol=1e-3) and h[20, 20] == 0 and forms[box].all(),
      f"{h[box].min()} {h[box].max()}")
check("height: a generated wall that is 10 cm off the real one stays black", not forms[on & ~box & ~past].any(), f"{forms.sum()} px")
check("height: a form that reaches past the building's outline is counted from the wall's plane",
      np.allclose(h[past], 0.5, atol=1e-3) and not forms[~(box | past)].any(), f"{h[past].min()} {h[past].max()}")
h0, forms0 = fh.height(depth, w, wall_m=0.3, smallest_form_m=0.0)
check("height: smallest_form_m drops the speck and the thin line, 0 keeps them", not forms[106:110, 30:34].any() and forms0[106:110, 30:34].all()
      and abs(float(h0[108, 32]) - 0.7) < 1e-3 and not forms[38:40].any() and forms0[38:40, 10:210].all())
niche = dict(w, depth=w["depth"].copy())
niche["depth"][50:90, 90:110] += 0.4                 # the real wall has a niche behind half of the box
flat, _f = fh.height(depth, niche, 0.3, 0.5)
real, _f = fh.height(depth, niche, 0.3, 0.5, flat=False)
check("counted from: the wall's plane hides the real wall's relief in the form, the real surface shows it",
      np.allclose(flat[box], 1.2, atol=1e-3) and np.allclose(real[50:90, 90:110], 1.6, atol=1e-3) and np.allclose(real[50:90, 110:130], 1.2, atol=1e-3))
deep = dict(w, depth=w["depth"].copy())
deep["depth"][100:120, 150:180] += 1.0                # a niche 1 m deep that the model's wall closes flush
check("counted from the wall's plane: a model that only fills a niche up to the wall is no form; from the real surface it is",
      not fh.height(depth, deep, 0.3, 0.5)[1][100:120, 150:180].any() and fh.height(depth, deep, 0.3, 0.5, flat=False)[1][100:120, 150:180].all())
g = fh.with_forms(depth, forms, w)
check("depth with forms: the real wall with the forms on it, near = white", abs(float(g[70, 110]) - 1.0) < 1e-6 and float(g[20, 20]) == 0.0
      and float(g[0, 0]) == 0.0 and 0.3 < float(g[20, 210]) < 0.7, f"{g[70, 110]} {g[20, 20]} {g[20, 210]}")

as_file = np.stack([MODEL[0][:, 0], MODEL[0][:, 2], -MODEL[0][:, 1]], 1)       # the same model as a .glb holds it (y up)
check("up: a y up model turned into the scene is the same model", np.allclose(fh.to_scene(as_file, True), MODEL[0])
      and np.allclose(fh.to_scene(as_file, True, 0.01), MODEL[0] * 0.01))
check("lies on: the model's wall covers the building, a model somewhere else does not",
      fh.lies_on(depth, w, 0.6)[0] > 0.9 and fh.lies_on(depth, w, 0.6)[1] == 1.0
      and fh.lies_on(fh.model_depth(cam, fh.to_scene(MODEL[0], True), MODEL[1]), w, 0.6)[0] < 0.1)

# a perspective camera: a flat wall has one depth all over, and a point lands where the camera says
P = np.array([[1.5, 0, 0, 0], [0, 1.5 * W / H, 0, 0], [0, 0, -100.1 / 99.9, -20 / 99.9], [0, 0, -1, 0]])
pcam = sv.Camera({"width": W, "height": H, "camera": {"type": "PERSP", "matrix_world": CAM.tolist(), "projection": P.tolist()}})
dp = fh.model_depth(pcam, *panel(3, 9, 1, 5, 0.0))
u, v, d = pcam.project(np.array([[3.0, 0, 5.0], [9.0, 0, 1.0]]))
inside = dp > 0
ys, xs = np.nonzero(inside)
check("perspective: a flat wall has its depth in every pixel, and its corners are where the camera puts them",
      np.allclose(dp[inside], 30.0, atol=1e-3) and abs(xs.min() - u[0]) < 1.0 and abs(ys.min() - v[0]) < 1.0
      and abs(xs.max() + 1 - u[1]) < 1.0 and abs(ys.max() + 1 - v[1]) < 1.0, f"{xs.min()} {u[0]:.1f} {ys.min()} {v[0]:.1f} {xs.max()} {u[1]:.1f}")

# ---- the node
sc = {"folder": folder, "unit_scale": 1.0}
out = nf.KUBA_FormHeight.execute(sc).result
img, mask = out[0][0].numpy(), out[1][0].numpy()
r = 0.18 * 10.0
check("node, no model: a sample form, white at its top, round, on the middle of the wall",
      "a sample form" in out[3] and "WARNING" not in out[3] and img.shape == (H, W, 3) and abs(float(img.max()) - 1.0) < 1e-6 and img[70, 110, 0] > 0.95
      and abs(mask.sum() * 0.05 ** 2 - np.pi * (r ** 2 - 0.5 ** 2)) < 0.6 and mask[5, 5] == 0, f"{mask.sum() * 0.0025:.2f} m2\n{out[3]}")
mesh = nf.io.Mesh.Type(torch.from_numpy(as_file.astype(np.float32))[None], torch.from_numpy(MODEL[1].astype(np.int64))[None])
out = nf.KUBA_FormHeight.execute(sc, model=mesh, wall_m=0.3).result
img, mask = out[0][0].numpy(), out[1][0].numpy()
check("node, a y up MESH: auto reads it the right way up and says so; the box is white, the lower form grey",
      "with y up" in out[3] and "covers 100 % of the building and lies on 9" in out[3] and np.allclose(img[box], 1.0) and abs(float(img[20, 210, 0]) - 0.5 / 1.2) < 1e-3
      and mask[box].all() and "up to 1.20 m" in out[3], out[3])
out = nf.KUBA_FormHeight.execute(sc, model=mesh, white_at_m=2.4, wall_m=0.0, smallest_form_m=0.0).result
check("node: white_at_m sets the scale", abs(float(out[0][0][70, 110, 0]) - 1.5 / 2.4) < 1e-3 and "white = 2.40 m" in out[3], out[3])
out = nf.KUBA_FormHeight.execute(sc, model=mesh, counted_from=fh.FROMS[1]).result
check("node: counted_from is passed on and named", "counted from the real surface" in out[3], out[3])
out = nf.KUBA_FormHeight.execute(sc, model=mesh, white_at_m=1.0, wall_m=0.3).result
check("node: a form above white_at_m is clipped and that is said", float(out[0].max()) == 1.0 and "clipped to white" in out[3])
away = nf.io.Mesh.Type(torch.from_numpy((as_file + np.array([40.0, 0, 0])).astype(np.float32))[None], mesh.faces)
out = nf.KUBA_FormHeight.execute(sc, model=away).result
check("node: a model that is somewhere else is said, with what to do", "WARNING: the model is not in front" in out[3] and "align 3d model" in out[3]
      and "neither with y up nor with z up" in out[3] and "it lies on the building that way" not in out[3]
      and float(out[0].max()) == 0.0, out[3])
deep = nf.io.Mesh.Type(torch.from_numpy((as_file + np.array([0.0, 0, 5.0])).astype(np.float32))[None], mesh.faces)
out = nf.KUBA_FormHeight.execute(sc, model=deep, up=fh.UPS[1]).result
check("node: a model whose wall stands 5 m off the real one is said", "covers the building but lies on only 0 %" in out[3], out[3])
# ---- the forms in the model's own colours
out = nf.KUBA_FormHeight.execute(sc, colours=True).result
c = out[4][0].numpy()
check("colours, the sample: the form in its colour on grey where there is no clay render", out[4].shape == (1, H, W, 3)
      and np.allclose(c[70, 110], nf.SAMPLE_COLOUR, atol=0.01) and np.allclose(c[5, 5], 0.5), f"{c[70, 110]} {c[5, 5]}")
check("colours off: the picture is the clay alone", np.allclose(nf.KUBA_FormHeight.execute(sc).result[4].numpy(), 0.5))
tex = torch.zeros((1, 8, 8, 3))
tex[:, :, :4, 0] = 1.0                              # the left half of the texture red, the right half blue
tex[:, :, 4:, 2] = 1.0
uvs = torch.from_numpy(np.stack([as_file[:, 0] / 10.0, 1.0 - as_file[:, 1] / 6.0], 1).astype(np.float32))[None]
worn = nf.io.Mesh.Type(mesh.vertices, mesh.faces, uvs=uvs, texture=tex)
out = nf.KUBA_FormHeight.execute(sc, model=worn, wall_m=0.3, colours=True).result
c = out[4][0].numpy()
check("colours, a textured MESH: the forms wear their texture the right way round, the wall stays the real one",
      np.allclose(c[70, 94], (1, 0, 0), atol=0.02) and np.allclose(c[70, 126], (0, 0, 1), atol=0.02) and np.allclose(c[20, 20], 0.5)
      and np.allclose(out[0].numpy(), nf.KUBA_FormHeight.execute(sc, model=worn, wall_m=0.3).result[0].numpy()), f"{c[70, 94]} {c[70, 126]} {c[20, 20]}")
top_red = torch.zeros((1, 8, 8, 3))
top_red[:, :4, :, 0] = 1.0                           # the upper half of the picture red
worn = nf.io.Mesh.Type(mesh.vertices, mesh.faces, uvs=uvs, texture=top_red)
c = nf.KUBA_FormHeight.execute(sc, model=worn, wall_m=0.3, colours=True).result[4][0].numpy()
check("colours: the top of the texture is the top of the building", c[52, 110, 0] > 0.9 and c[88, 110, 0] < 0.1, f"{c[52, 110]} {c[88, 110]}")
out = nf.KUBA_FormHeight.execute(sc, model=mesh, wall_m=0.3, colours=True).result
check("colours: a model without any is said, the picture is the clay", "carries no texture and no colours" in out[3] and np.allclose(out[4].numpy(), 0.5))
try:
    import trimesh
    p = os.path.join(TMP, "forms.glb")
    trimesh.Trimesh(vertices=MODEL[0] * 100.0, faces=MODEL[1], process=False).export(p)    # trimesh writes points as they are
    out = nf.KUBA_FormHeight.execute({"folder": folder, "unit_scale": 0.01}, model_file=p, up=fh.UPS[2], wall_m=0.3).result
    check("node, a file in centimetres: the scene's unit_scale brings it to metres", "up to 1.20 m" in out[3] and np.allclose(out[0][0].numpy()[box], 1.0), out[3])
except ImportError:
    print("skip the file part: trimesh is not installed")

shutil.rmtree(TMP, ignore_errors=True)
print("\nALL PASSED" if not failures else f"\n{len(failures)} FAILED: {failures}")
sys.exit(1 if failures else 0)
