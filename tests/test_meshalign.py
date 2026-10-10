"""
Model free test of kubakub align 3d model: the fit (kubakub/meshalign.py) on models that were moved, turned, scaled,
stretched along one axis and mirrored on purpose, the facade front fit on a whole invented building with a form in
front of its wall, and the node (nodes/scene3d/nodes_align3d.py) with files and with MESH inputs.

Run from the ComfyUI_windows_portable folder:
    python_embeded\\python.exe ComfyUI\\custom_nodes\\ComfyUI_KubaNodes\\tests\\test_meshalign.py
"""

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

TMP = tempfile.mkdtemp(prefix="kkd_align3d_")
folder_paths.set_output_directory(os.path.join(TMP, "output"))

sys.path.insert(0, HERE)     # embedded Python leaves the script folder off the path
import _pack  # noqa: E402,F401  (the pack as a package, see tests/_pack.py)

import kubapack.nodes.scene3d.nodes_align3d as na  # noqa: E402
from kubakub import meshalign as ma  # noqa: E402

failures = []


def check(name, cond, detail=""):
    print(("ok   " if cond else "FAIL ") + name + (f"  ({detail})" if detail and not cond else ""))
    if not cond:
        failures.append(name)


def raises(fn):
    try:
        fn()
    except (ValueError, ImportError) as e:
        return str(e)
    return ""


def rot(axis, deg):
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    return {"x": np.array([[1, 0, 0], [0, c, -s], [0, s, c]]), "y": np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]]),
            "z": np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])}[axis]


(TV, TF), _ = na._sample()                                    # the sample facade: 24 m wide, 16 m high, flat
T = ma.sample_surface(TV, TF, 80000, 1)                     # the template dense, as the node takes it
SIZE = float(np.linalg.norm(TV.max(0) - TV.min(0)))

pts = ma.sample_surface(TV, TF, 5000, 2)
d = np.abs(pts[:, 2] - np.clip(pts[:, 2], TV[:, 2].min(), TV[:, 2].max())).max()
check("points are taken from the surface, spread over it", len(pts) == 5000 and d < 1e-9 and pts[:, 0].min() < -10 and pts[:, 0].max() > 10)
check("a model without faces gives its vertices", len(ma.sample_surface(TV[:50], None, 5000)) == 50)

CASES = [("moved and turned 37° about the height", rot("y", 37), 1.0, (1, 1, 1), (4.0, -2.5, 7.0), "keep the size", False),
         ("lying on its back, facing away, a thirtieth of the size", rot("x", -90) @ rot("y", 180), 0.031, (1, 1, 1), (40.0, 3.0, -9.0), "one scale", False),
         ("turned on all axes and a quarter too deep", rot("z", 20) @ rot("x", 65) @ rot("y", -140), 2.4, (1, 1, 1.25), (-3.0, 12.0, 1.0), "per axis", False),
         ("a third too high", rot("y", 15), 0.5, (1, 1.3, 1), (1.0, 1.0, 1.0), "per axis", False)]
for name, R, s, stretch, shift, scale, mirror in CASES:
    lost = (TV * np.array(stretch)) @ R.T * s + np.array(shift)
    res = ma.align(ma.sample_surface(lost, TF, 20000, 3), T, scale, mirror)
    back = ma.apply(res["matrix"], lost)
    err = float(np.linalg.norm(back - TV, axis=1).mean())
    check(f"fit: {name}: every vertex back within 0.5 % of the size", err < 0.005 * SIZE and res["error"] < 0.005 * SIZE
          and res["before"] > res["error"] * 5 and res["mirrored"] == (stretch[0] < 0), f"{err:.4f} of {SIZE:.1f}, {res['scales']}")
    text = "\n".join(ma.describe(res, "m"))
    if name.startswith("turned on all axes"):
        ratio = res["scales"][2] / np.median(res["scales"])
        check("fit: the misread depth is found (a quarter too deep, within a few percent) and named in the report",
              abs(ratio - 0.8) < 0.04 and "z was 2" in text and "too long" in text and "corrected" in text, f"{ratio:.3f}\n{text}")
    if name == "a third too high":
        one = ma.align(ma.sample_surface(lost, TF, 20000, 3), T, "one scale")
        check("fit: the misread height is named; with one scale it cannot be made right, and the error shows it",
              "y was 30 % too long" in text and one["error"] > res["error"] * 1.5, f"{one['error']:.4f} vs {res['error']:.4f}\n{text}")
res = ma.align(T + np.array([0.3, 0.0, 0.0]), T, "keep the size")
check("fit, keep the size: no scale", np.allclose(res["scales"], 1.0, atol=1e-6) and res["error"] < 0.01)
flip = ma.align((TV * np.array([-1.0, 1.0, 1.0])) @ rot("y", 15).T, T, "one scale", mirror=False)
check("fit: without 'mirror' a fit is never mirrored", not flip["mirrored"] and np.linalg.det(flip["matrix"][:3, :3]) > 0)
lump = np.random.default_rng(4).random((1500, 3)) * 3.0 + np.array([8.0, 16.0, 0.0])        # a tower on one corner: no longer the same left and right
T2 = np.concatenate([T, lump])
gone = (T2 * np.array([-1.0, 1.0, 1.0])) @ rot("y", 15).T * 0.5 + 1.0
yes, no = ma.align(gone, T2, "one scale", mirror=True), ma.align(gone, T2, "one scale", mirror=False)
check("fit: a mirrored model is found with 'mirror' and fits; without it the tower stays on the wrong side",
      yes["mirrored"] and yes["error"] < 0.005 * SIZE and no["error"] > yes["error"] * 1.5, f"{yes['error']:.3f} {no['error']:.3f}")
check("fit: too few points is said", "at least 16 points" in raises(lambda: ma.align(T[:3], T)) and "at least 16 points" in raises(lambda: ma.align(T, T[:12])))
half = ma.sample_surface(TV, TF, 12000, 5)
half = half[half[:, 2] > -0.05] @ rot("y", 25).T * 0.2 + 3.0     # only the front, as image to 3D sees it
res = ma.align(half, T, "one scale")
check("fit: a model that shows only the front still lands on the front", res["error"] < 0.01 * SIZE, f"{res['error']:.3f}")

# ---- the facade front: image to 3D makes a whole building of a facade picture (back wall, sides, roof) and the forms
# of the picture stand in front of the wall. Only the front is fitted; the depth keeps the model's proportions.
LO, HI = TV.min(0), TV.max(0)
DEEP = 9.0


def building(n=60000, seed=6):
    """Points on the sample facade, on an invented box behind it and on a ball in front of it."""
    rng = np.random.default_rng(seed)
    u = rng.random((n, 3))
    back = np.stack([LO[0] + u[:, 0] * (HI[0] - LO[0]), LO[1] + u[:, 1] * (HI[1] - LO[1]), np.full(n, LO[2] - DEEP)], 1)
    side = np.stack([np.where(u[:, 0] < 0.5, LO[0], HI[0]), LO[1] + u[:, 1] * (HI[1] - LO[1]), LO[2] - u[:, 2] * DEEP], 1)
    roof = np.stack([LO[0] + u[:, 0] * (HI[0] - LO[0]), np.full(n, HI[1]), LO[2] - u[:, 2] * DEEP], 1)
    g = rng.normal(size=(n // 3, 3))
    ball = g / np.linalg.norm(g, axis=1, keepdims=True) * 2.5 + np.array([3.0, 0.55 * (LO[1] + HI[1]), HI[2] + 2.0])
    return np.concatenate([ma.sample_surface(TV, TF, n, seed), back, side[: n // 2], roof[: n // 2], ball])


B = building()
GEN = np.array([0.95, 1.0, 1.0]) * 0.033            # the generator's size, and a width read 5 % too short
check("front: a thin template with a far deeper model is seen as a facade; the sample pair is not",
      ma.looks_like_a_front(B * GEN, T) and not ma.looks_like_a_front(T * 0.03, T))
for name, R, shift in (("as it stands", np.eye(3), (0.0, 0.0, 0.0)), ("turned 30° about the height and moved", rot("y", 30), (40.0, 3.0, -12.0)),
                       ("facing the other way", rot("y", 180), (5.0, 0.0, 2.0))):
    real = T @ R.T + np.array(shift)
    res = ma.align_front(B * GEN + np.array([0.3, -0.2, 0.1]), real, "per axis")
    back = ma.apply(res["matrix"], TV * GEN + np.array([0.3, -0.2, 0.1]))
    err = float(np.linalg.norm(back - (TV @ R.T + np.array(shift)), axis=1).mean())
    check(f"front, template {name}: the facade comes back within 1 % of the size, the box behind it does not pull",
          err < 0.01 * SIZE and res["error"] < 0.01 * SIZE and abs(res["scales"][0] / res["scales"][1] - 1 / 0.95) < 0.02
          and res["front"]["behind"] > 0.5 * DEEP and res["front"]["ahead"] > 2.0, f"{err:.3f} of {SIZE:.1f}, {res['scales']}, {res['front']}")
whole = ma.align(B[::4] * GEN, T, "per axis")
text = "\n".join(ma.describe(whole, "m"))
front_text = "\n".join(ma.describe(res, "m"))
check("front: the whole-model fit on the same building says to use the facade front; the front report says what it did",
      ("Set fit to 'the facade front'" in text or whole["error"] > 2 * res["error"]) and "fitted on the facade front" in front_text
      and "of the model's front is wall" in front_text, text + "\n" + front_text)
one = ma.align_front(B * 0.033, T, "one scale")
keep = ma.align_front(B + np.array([3.0, 1.0, 0.5]), T, "keep the size")
check("front: one scale keeps the proportions, keep the size only moves",
      abs(one["scales"][0] / one["scales"][1] - 1) < 1e-6 and abs(one["scales"][0] - 1 / 0.033) < 0.02 / 0.033
      and np.allclose(keep["scales"], 1.0, atol=1e-6) and keep["error"] < 0.01 * SIZE, f"{one['scales']} {keep['scales']} {keep['error']:.3f}")
fv = np.array([[0.0, 0, 0.5], [1, 0, 0.5], [0, 1, 0.5], [0.0, 0, -5], [1, 0, -5], [0, 1, -5]])
check("front: faces behind the template's back are the ones cut", list(ma.front_faces(fv, [[0, 1, 2], [3, 4, 5]], {"normal": [0, 0, 1.0], "back": -1.0})) == [True, False])

# ---- the node
out = na.KUBA_AlignModel.execute().result
check("node, nothing connected: the sample is fitted and written", out[0].vertices.shape[0] == 1 and os.path.isfile(out[1]) and out[1].endswith("aligned_model_00001.glb")
      and "a sample is used" in out[3] and "z was 2" in out[3] and "too long" in out[3] and len(out[2].split("\n")) == 4, out[3])
check("node: the model output lies on the template", float(np.abs(out[0].vertices[0].numpy() - TV).mean()) < 0.005 * SIZE)
try:
    import trimesh
    lost = TV @ rot("y", 50).T * 0.1 + np.array([5.0, 5.0, 5.0])
    a, b = os.path.join(TMP, "gen.glb"), os.path.join(TMP, "real.obj")
    trimesh.Trimesh(vertices=lost, faces=TF, process=False).export(a)
    trimesh.Trimesh(vertices=TV, faces=TF, process=False).export(b)
    out = na.KUBA_AlignModel.execute(model_file=a, template_file=b, scale="one scale", save_name="fitted").result
    got = trimesh.load(out[1], force="mesh", process=False)
    check("node, two files: the written .glb lies on the template", out[1].endswith("fitted_00001.glb")
          and float(np.abs(np.asarray(got.bounds) - np.array([TV.min(0), TV.max(0)])).max()) < 0.01 * SIZE, str(got.bounds))
    M = np.array([[float(x) for x in row.split(",")] for row in out[2].split("\n")])
    check("node: the matrix output is the one that was applied", M.shape == (4, 4) and float(np.abs(ma.apply(M, lost) - TV).mean()) < 0.005 * SIZE)
    mesh = na.io.Mesh.Type(torch.from_numpy(lost.astype(np.float32))[None], torch.from_numpy(TF)[None])
    real = na.io.Mesh.Type(torch.from_numpy(TV.astype(np.float32))[None], torch.from_numpy(TF)[None])
    out = na.KUBA_AlignModel.execute(model=mesh, template=real, scale="one scale").result
    check("node, MESH inputs: the model comes out moved, the input is left alone", float(np.abs(out[0].vertices[0].numpy() - TV).mean()) < 0.005 * SIZE
          and torch.allclose(mesh.vertices[0], torch.from_numpy(lost.astype(np.float32))))
    check("node: a second run does not overwrite the first file", out[1].endswith("aligned_model_00002.glb"), out[1])
    # a whole invented building as a file with a texture-less box and ball: auto takes the front and cuts the back
    box = trimesh.creation.box(extents=(HI[0] - LO[0], HI[1] - LO[1], DEEP))
    box.apply_translation(((LO[0] + HI[0]) / 2, (LO[1] + HI[1]) / 2, LO[2] - DEEP / 2 - 0.05))
    ball = trimesh.creation.icosphere(subdivisions=3, radius=2.5)
    ball.apply_translation((3.0, 0.55 * (LO[1] + HI[1]), HI[2] + 2.0))
    house = trimesh.util.concatenate([trimesh.Trimesh(vertices=TV, faces=TF, process=False), box, ball])
    house.apply_scale(0.033)
    c = os.path.join(TMP, "house.glb")
    house.export(c)
    out = na.KUBA_AlignModel.execute(model_file=c, template_file=b, save_name="front").result
    got = trimesh.load(out[1], force="mesh", process=False)
    lo, hi = np.asarray(got.bounds)
    check("node, a whole building on a facade: auto fits the front and cuts the invented back away",
          "the facade front is fitted" in out[3] and "cut_the_back:" in out[3] and len(got.faces) < len(house.faces)
          and float(np.abs(lo[:2] - LO[:2]).max()) < 0.015 * SIZE and float(np.abs(hi[:2] - HI[:2]).max()) < 0.015 * SIZE
          and lo[2] > LO[2] - 1.0 and hi[2] > HI[2] + 3.0 and out[0].faces.shape[1] == len(got.faces), f"{lo} {hi}\n{out[3]}")
    out = na.KUBA_AlignModel.execute(model_file=c, template_file=b, fit="the facade front", cut_the_back=False, mirror=True).result
    got = trimesh.load(out[1], force="mesh", process=False)
    check("node, cut_the_back off: the whole building is kept behind the facade; mirror is said to be unused",
          len(got.faces) == len(house.faces) and np.asarray(got.bounds)[0][2] < LO[2] - 0.5 * DEEP and "mirror is not used" in out[3], out[3])
    hm = na.io.Mesh.Type(torch.from_numpy(np.asarray(house.vertices, np.float32))[None], torch.from_numpy(np.asarray(house.faces, np.int64))[None])
    out = na.KUBA_AlignModel.execute(model=hm, template=real, fit="the facade front").result
    check("node, MESH inputs with the facade front: the faces behind are gone from the MESH, the input is left alone",
          out[0].faces.shape[1] < hm.faces.shape[1] and hm.faces.shape[1] == len(house.faces) and out[0].vertices.shape == hm.vertices.shape)
    out = na.KUBA_AlignModel.execute(model_file=c, template_file=b, fit="the whole model").result
    check("node: 'the whole model' still does the old fit", "fitted on the facade front" not in out[3] and "cut_the_back" not in out[3])
    check("node: a missing template is said", "real model is missing" in raises(lambda: na.KUBA_AlignModel.execute(model_file=a)))
    check("node: a format it cannot read says what to export", "Export it from Blender" in raises(lambda: na.KUBA_AlignModel.execute(model_file=a, template_file=__file__)))
    check("node: a path that is no file is said", "is not a file" in raises(lambda: na.KUBA_AlignModel.execute(model_file=a, template_file=os.path.join(TMP, "none.glb"))))
except ImportError:
    print("skip the file part: trimesh is not installed")

shutil.rmtree(TMP, ignore_errors=True)
print("\nALL PASSED" if not failures else f"\n{len(failures)} FAILED: {failures}")
sys.exit(1 if failures else 0)
