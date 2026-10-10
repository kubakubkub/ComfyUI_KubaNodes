"""
Model free test of kubakub height animate (kubakub/heightanim.py, nodes/motion/nodes_height.py): the ways a height
map grows, the lit relief, and the node with a sample, a depth map with a wall grey, a field and a sound curve.

Run from the ComfyUI_windows_portable folder:
    python_embeded\\python.exe ComfyUI\\custom_nodes\\ComfyUI_KubaNodes\\tests\\test_heightanim.py
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

TMP = tempfile.mkdtemp(prefix="kkd_heightanim_")
folder_paths.set_temp_directory(os.path.join(TMP, "temp"))

sys.path.insert(0, HERE)     # embedded Python leaves the script folder off the path
import _pack  # noqa: E402,F401  (the pack as a package, see tests/_pack.py)

import kubapack.nodes.motion.nodes_height as nh  # noqa: E402
from kubakub import heightanim as ha  # noqa: E402

failures = []


def check(name, cond, detail=""):
    print(("ok   " if cond else "FAIL ") + name + (f"  ({detail})" if detail and not cond else ""))
    if not cond:
        failures.append(name)


W, H = 160, 96
yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)


def hill(cx, cy, r, top):
    return np.clip(1.0 - np.hypot(xx - cx, yy - cy) / r, 0, 1) * top


h = np.maximum(hill(40, 48, 30, 1.0), hill(120, 48, 20, 0.5)).astype(np.float32)      # a high hill left, a low one right

check("from grey: 'wall' is the grey that counts as the wall, colour pictures are read as grey",
      np.allclose(ha.from_grey(np.full((2, 2), 0.6), 0.2), 0.5) and float(ha.from_grey(np.full((2, 2), 0.1), 0.2).max()) == 0.0
      and np.allclose(ha.from_grey(np.full((2, 2, 3), 0.25)), 0.25))
check("all together: every form rises at once, 0 is the bare wall, 1 the whole height", float(ha.grown(h, 0.0).max()) == 0.0
      and np.allclose(ha.grown(h, 0.5), h * 0.5) and np.allclose(ha.grown(h, 1.0), h))
p = ha.grown(h, 0.4, "peaks first")
check("peaks first: the highest points come through the wall first", p[48, 40] > 0.39 and p[48, 120] == 0.0 and float(p.max()) <= 0.4 + 1e-6
      and np.allclose(ha.grown(h, 1.0, "peaks first"), h) and float(ha.grown(h, 0.0, "peaks first").max()) == 0.0)
check("peaks first: a value above 1 does not lift the wall", float(ha.grown(h, 1.5, "peaks first")[0, 0]) == 0.0
      and np.allclose(ha.grown(h, 1.5, "peaks first"), h))
b = ha.grown(h, 0.4, "built up in layers")
check("built up in layers: cut flat at the level reached", abs(float(b[48, 40]) - 0.4) < 1e-6 and abs(float(b[48, 120]) - 0.4) < 1e-6
      and np.allclose(b, np.minimum(h, 0.4)) and np.allclose(ha.grown(h, 1.0, "built up in layers"), h))
f = ha.ramp_over(h)
a = ha.grown(h, 0.5, "along the field", f, 0.1)
check("along the field: without a field the forms rise from left to right", f[48, 15] < f[48, 135] and a[48, 40] > 0.9 and a[48, 120] == 0.0
      and np.allclose(ha.grown(h, 1.0, "along the field", f, 0.1), h) and float(ha.grown(h, 0.0, "along the field", f, 0.1).max()) == 0.0,
      f"{a[48, 40]} {a[48, 120]}")

flat = ha.lit(np.zeros((H, W), np.float32))
left = ha.lit(h, angle=0.0)
top = ha.lit(h, angle=90.0)
check("lit: a flat wall is one even grey", flat.shape == (H, W, 3) and float(np.ptp(flat)) < 1e-6 and 0.7 < float(flat[0, 0, 0]) < 0.9)
check("lit: light from the left brightens the left slope and darkens the right one", left[48, 28, 0] > flat[0, 0, 0] + 0.05 > left[48, 52, 0] + 0.1,
      f"{left[48, 28, 0]} {flat[0, 0, 0]} {left[48, 52, 0]}")
check("lit: light from above brightens the upper slope", top[36, 40, 0] > flat[0, 0, 0] + 0.05 > top[60, 40, 0] + 0.1)
base = np.zeros((H, W, 3), np.float32)
base[..., 0] = 1.0
check("lit: a picture on the relief keeps its colour", float(ha.lit(h, base)[..., 1:].max()) == 0.0 and float(ha.lit(h, base)[48, 28, 0]) > 0.8)
check("lit: more relief = stronger light and shade", np.ptp(ha.lit(h, relief=3.0)) > np.ptp(left) + 0.05)

# ---- the node
out = nh.KUBA_HeightAnimate.execute(seconds=1.0, fps=10.0, scale=0.25, cycle=0.5).result
check("node, nothing connected: a sample grows, black at the start, the whole height at the end",
      "a sample is used" in out[6] and out[0].shape == (10, 180, 320, 3) and out[1].shape == out[0].shape and out[2].shape == (10, 180, 320)
      and float(out[0][0].max()) == 0.0 and float(out[0][-1].max()) > 0.9 and out[5] == 10 and out[3] == 10.0, out[6])
img = torch.from_numpy(np.repeat(h[None, ..., None], 3, -1))
out = nh.KUBA_HeightAnimate.execute(height=img, seconds=2.0, fps=5.0, cycle=2.0, easing="linear").result
m = out[2].numpy()
check("node: a ramp over the whole length rises evenly", np.allclose(m[5], h * 0.5, atol=1e-5) and np.allclose(out[0][5, ..., 0].numpy(), m[5])
      and float(m[-1].max()) > 0.89 and out[4]["waveform"].shape[-1] == 2 * 44100)
out = nh.KUBA_HeightAnimate.execute(height=img * 0.5 + 0.5, wall=0.5, seconds=1.0, fps=4.0, cycle=0.5, easing="linear").result
check("node: a depth map with the wall at mid grey grows only what stands in front of it; a held value repeats",
      np.allclose(out[2][-1].numpy(), h, atol=1e-5) and torch.equal(out[2][-1], out[2][-2]) and torch.equal(out[1][-1], out[1][-2]))
field = torch.from_numpy(np.tile(np.linspace(1, 0, W, dtype=np.float32), (H, 1)))[None]      # right first
out = nh.KUBA_HeightAnimate.execute(height=img, field=field, grow="along the field", seconds=1.0, fps=10.0, cycle=1.0, easing="linear").result
m = out[2].numpy()
check("node: along a field the forms rise in its order", m[5][48, 120] > 0.45 and m[5][48, 40] == 0.0)
out = nh.KUBA_HeightAnimate.execute(height=img, field=field, seconds=0.4, fps=10.0).result
check("node: a field that is not used is said", "the field is not used" in out[6])
out = nh.KUBA_HeightAnimate.execute(height=img, curve="beats", cycle=0.2, seconds=2.0, fps=20.0).result
m = out[2].numpy().max(axis=(1, 2))
check("node, beats without a sound: the built-in beat, the forms come up on every beat", "the built-in beat" in out[6] and "120" in out[6]
      and m.max() > 0.9 and (np.diff(m) > 0.2).sum() >= 3, f"{np.round(m, 2)}\n{out[6]}")
out = nh.KUBA_HeightAnimate.execute(height=img, curve="beats", cycle=0.2, fps=10.0, scale=0.5).result
check("node, beats without a sound and seconds = 0: 4 s, as without a sound curve", out[5] == 40 and out[4]["waveform"].shape[-1] == 4 * 44100, out[6])
out = nh.KUBA_HeightAnimate.execute(height=img, curve="beats", cycle=0.2, seconds=12.0, fps=5.0, scale=0.5).result
m = out[2].numpy().max(axis=(1, 2))
check("node: the built-in beat lasts as long as the clip", (np.diff(m[45:]) > 0.2).sum() >= 2, f"{np.round(m[45:], 2)}")
colour = torch.zeros((1, 48, 80, 3))
colour[..., 2] = 1.0
grey = torch.full((1, H, W, 3), 0.5)
out = nh.KUBA_HeightAnimate.execute(height=img, image=colour, wall_image=grey, seconds=1.0, fps=10.0, cycle=0.5, grow="peaks first", relief=0.0).result
first, last = out[1][0].numpy(), out[1][-1].numpy()
check("node, wall_image: the bare wall shows until a form comes out, then the form wears 'image'",
      np.allclose(first, 0.8 * 0.5, atol=1e-3) and last[48, 40, 2] > 0.7 and last[48, 40, 0] < 0.01 and np.allclose(last[2, 2], 0.8 * 0.5, atol=1e-3),
      f"{first[48, 40]} {last[48, 40]} {last[2, 2]}")
out = nh.KUBA_HeightAnimate.execute(height=img, image=colour, seconds=0.2, fps=10.0).result
check("node: a picture of another size is fitted and lit", out[1].shape == (2, H, W, 3) and float(out[1][..., :2].max()) == 0.0 and float(out[1][..., 2].max()) > 0.7)

# ---- in the director: a layer whose picture is a height map (render.py finish, heightanim.layer_look)
import json  # noqa: E402
import re  # noqa: E402
import subprocess  # noqa: E402
from kubakub.director import render as rd  # noqa: E402

rgb = np.repeat(h[..., None], 3, -1)
check("layer look: 'height' is the grown height as grey, 'relief' the node's lit relief at the canvas width",
      np.allclose(ha.layer_look(rgb, 0.5, "all together")[..., 0], h * 0.5)
      and np.allclose(ha.layer_look(rgb, 1.0, "peaks first", look="relief", angle=20.0, relief=2.0, width=W),
                      ha.lit(h, None, 20.0, 2.0, W, ha.LAYER_BLUR * W / 1920.0), atol=1e-5)
      and np.allclose(ha.layer_look(rgb, 1.0, look="shading", width=W)[0, 0], 1.0) and float(ha.layer_look(rgb, 1.0, look="shading", width=W).min()) < 0.6)
BASE = np.full((H, W, 3), 0.25, np.float32)
SRC = {"input:0": np.concatenate([rgb, np.ones((H, W, 1), np.float32)], -1)}


def doc(height, **layer):
    return {"version": 1, "layers": [{"id": "hm", "name": "height", "kind": "image", "source": "input:0", "x": 0, "y": 0, "w": W, "h": H,
                                      "height": height, **layer}, {"id": "base", "name": "base", "kind": "base"}]}


img0 = rd.render(doc({"mode": "all together", "value": 0.5, "look": "height"}), BASE, SRC, image_only=True)["image"]
check("director render: a height layer shows the height grown so far", np.allclose(img0[..., 0], h * 0.5, atol=1e-5))
plain = rd.render(doc(None), BASE, SRC, image_only=True)["image"]
check("director render: without 'height' the layer is the plain picture", np.allclose(plain[..., 0], h, atol=1e-5))
keyed = doc({"mode": "peaks first", "value": 0.0, "look": "height"}, anim={"height.value": [{"t": 0, "v": 0.0}, {"t": 2, "v": 1.0}]})
keyed["timeline"] = {"duration": 2.0, "fps": 10}
mid = rd.render(rd.animate(keyed, 1.0), BASE, SRC, image_only=True)["image"]
check("director render: 'grown' takes keyframes (peaks first, half way)", np.allclose(mid[..., 0], np.clip(h - 0.5, 0, None), atol=1e-5))
rel = rd.render(doc({"mode": "all together", "value": 1.0, "look": "relief", "angle": 0.0, "relief": 2.0}), BASE, SRC, image_only=True)["image"]
check("director render: 'relief' is lit from the side", np.allclose(rel, ha.lit(h, None, 0.0, 2.0, W, ha.LAYER_BLUR * W / 1920.0), atol=1e-5)
      and rel[48, 28, 0] > rel[0, 0, 0] > rel[48, 52, 0])
check("director document: unknown modes and looks fall back, values are kept in range",
      rd.parse(doc({"mode": "sideways", "look": "x", "value": -2, "wall": 5}))["layers"][0]["height"]
      == {"mode": "all together", "value": 0.0, "wall": 0.99, "look": "relief", "angle": 35.0, "relief": 1.0}
      and rd.parse(doc({"on": False}))["layers"][0]["height"] is None)

node = shutil.which("node")
if not node:
    print("skip the window twin: node.js not found")
else:                                               # the window's own code (web/kubakub_director.js) on the same numbers
    js = open(os.path.join(HERE, "..", "web", "kubakub_director.js"), encoding="utf-8").read()
    engine = js[js.index("// ---- height: a layer's picture as a height map"):js.index("// ---- audio: beats from a mono signal")]
    CASES = [(v, m) for v in (0.0, 0.3, 1.0, 1.4) for m in ha.GROWS[:3]]
    SHADES = [(0.0, 1.0, 1.2), (35.0, 2.5, 0.9), (140.0, 0.6, 2.2), (-90.0, 1.0, 0.1)]      # angle, relief, blur
    vec = os.path.join(TMP, "height_vectors.json")
    os.makedirs(TMP, exist_ok=True)
    json.dump({"h": h.ravel().tolist(), "w": W, "hh": H, "cases": CASES, "shades": SHADES}, open(vec, "w"))
    runner = engine + """
const V = JSON.parse(require("fs").readFileSync(process.argv[2], "utf8")), h = Float32Array.from(V.h);
console.log(JSON.stringify({ grown: V.cases.map(([v, m]) => Array.from(heightGrown(h, v, m))),
  shade: V.shades.map(([a, r, s]) => Array.from(heightShade(h, V.w, V.hh, a, r, 640, s))),
  lists: [HEIGHT_GROWS, HEIGHT_LOOKS, HEIGHT_DEF] }));
"""
    run = os.path.join(TMP, "height_twin.js")
    open(run, "w", encoding="utf-8").write(runner)
    res = subprocess.run([node, run, vec], capture_output=True, text=True)
    if res.returncode:
        check("window twin runs", False, res.stderr[-600:])
    else:
        JS = json.loads(res.stdout)
        worst = max(float(np.abs(np.array(g, np.float32) - ha.grown(h, v, m).ravel()).max()) for g, (v, m) in zip(JS["grown"], CASES))
        check(f"window twin: {len(CASES)} grown heights agree per pixel", worst < 1e-6, f"worst {worst}")
        worst = max(float(np.abs(np.array(g, np.float32).reshape(H, W) - ha.lit(h, None, a, r, 640, s)[..., 0]).max())
                    for g, (a, r, s) in zip(JS["shade"], SHADES))
        check(f"window twin: {len(SHADES)} lit reliefs agree per pixel (blur, slope, light)", worst < 2e-4, f"worst {worst}")
        check("window twin: the lists and defaults are the render's", JS["lists"][0] == list(ha.GROWS[:3]) and JS["lists"][1] == list(ha.LOOKS)
              and rd.parse(doc({}))["layers"][0]["height"] == JS["lists"][2], f"{JS['lists']}")
    check("window: the height is read, kept and drawn (the places that copy a layer know it)",
          all(js.count(s) >= n for s, n in (("height: L.height ? { ...L.height } : undefined", 2), ("height: s.height ? { ...s.height } : undefined", 1),
                                           ('"light_react", "height", "motion"]', 1), ("this.applyHeight(lc, L, R)", 1), ("if (L.height) c.height", 1))))

shutil.rmtree(TMP, ignore_errors=True)
print("\nALL PASSED" if not failures else f"\n{len(failures)} FAILED: {failures}")
sys.exit(1 if failures else 0)
