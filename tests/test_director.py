"""
Model free test for the kubakub director renderer (kubakub/director): blend modes and colour grade
as the browser does them, placement, front / back order with holes, layer masks, region clips, the
masks / regions / rules that go back into the workflow.

Run from the ComfyUI_windows_portable folder:
    python_embeded\\python.exe ComfyUI\\custom_nodes\\ComfyUI_KubaNodes\\tests\\test_director.py
"""

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PACK = os.path.dirname(HERE)
sys.path.insert(0, PACK)                           # kubakub as a package, facade_core top level
from kubakub.director import blend as bl  # noqa: E402
from kubakub.director import render as rd  # noqa: E402
from kubakub import plan as pl  # noqa: E402

failures = []


def check(name, cond, detail=""):
    print(("ok   " if cond else "FAIL ") + name + (f"  ({detail})" if detail and not cond else ""))
    if not cond:
        failures.append(name)


rng = np.random.default_rng(0)
b = rng.random((64, 3)).astype(np.float64)
s = rng.random((64, 3)).astype(np.float64)

# --- blend modes -------------------------------------------------------------------------------
check("normal returns the source", np.allclose(bl.blend("normal", b, s), s))
check("multiply / screen", np.allclose(bl.blend("multiply", b, s), b * s) and np.allclose(bl.blend("screen", b, s), 1 - (1 - b) * (1 - s)))
check("difference / exclusion", np.allclose(bl.blend("difference", b, s), abs(b - s)) and np.allclose(bl.blend("exclusion", b, s), b + s - 2 * b * s))
check("overlay = hard light with the layers swapped", np.allclose(bl.blend("overlay", b, s), bl.blend("hard light", s, b)))
# W3C soft light at known points: s = 0.5 leaves the backdrop; s = 1, b = 0.25 -> D = 0.25 -> 0.25
check("soft light (W3C) keeps the backdrop at 0.5", np.allclose(bl.blend("soft light", b, np.full_like(b, 0.5)), b))
check("soft light (W3C) point value", np.isclose(bl.blend("soft light", np.array([0.5]), np.array([1.0]))[0], np.sqrt(0.5)))
check("color dodge / burn limits", np.allclose(bl.blend("color dodge", np.zeros(3), np.ones(3)), 0) and np.allclose(bl.blend("color burn", np.ones(3), np.zeros(3)), 1))
lum = lambda c: 0.3 * c[..., 0] + 0.59 * c[..., 1] + 0.11 * c[..., 2]  # noqa: E731
check("luminosity keeps the source's luminance", np.allclose(lum(bl.blend("luminosity", b, s)), lum(s), atol=1e-6))
check("color keeps the backdrop's luminance", np.allclose(lum(bl.blend("color", b, s)), lum(b), atol=1e-6))
try:                                                # same formulas as ComfyUI core where the definitions agree
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(PACK))))
    from comfy_extras import compositor_blend as core
    same = all(np.allclose(bl.blend(ours, b, s), core.blend_pixel(theirs, b, s), atol=1e-6) for ours, theirs in
               (("multiply", "multiply"), ("screen", "screen"), ("overlay", "overlay"), ("darken", "darken"),
                ("lighten", "lighten"), ("difference", "difference"), ("exclusion", "exclusion"), ("hard light", "hard-light")))
    check("matches ComfyUI core's blend formulas (separable modes)", same)
except Exception as e:  # noqa: BLE001
    print(f"skip core comparison ({type(e).__name__}: {e})")

# --- colour grade (CSS filter chain) ---------------------------------------------------------------
img = rng.random((8, 8, 3)).astype(np.float32)
check("grade identity", np.allclose(bl.grade(img), img, atol=1e-6))
check("brightness 2 doubles and clamps", np.allclose(bl.grade(img, brightness=2), np.clip(img * 2, 0, 1)))
g0 = bl.grade(img, saturate=0)
check("saturate 0 = grey with the CSS weights", np.allclose(g0[..., 0], g0[..., 1], atol=1e-6) and
      np.allclose(g0[..., 0], img @ np.array([0.213, 0.715, 0.072]), atol=1e-5))
check("hue-rotate 360 = identity", np.allclose(bl.grade(img, hue=360), img, atol=1e-5))
check("sepia 1 on white", np.allclose(bl.grade(np.ones((1, 1, 3), np.float32), sepia=1)[0, 0], np.clip([1.351, 1.203, 0.937], 0, 1), atol=1e-3))

# --- render --------------------------------------------------------------------------------------
W, H = 120, 80
base = np.full((H, W, 3), 0.5, np.float32)
red = np.zeros((10, 20, 4), np.float32); red[..., 0] = 1; red[..., 3] = 1
blue = np.zeros((10, 10, 4), np.float32); blue[..., 2] = 1; blue[..., 3] = 1
labels = np.full((H, W), 0, np.int32)                     # region 0 = wall, region 1 = a window
labels[20:40, 70:100] = 1
table = {"regions": [{"region_id": 0, "name": "wall", "group_id": "wall", "tags": ["front"]},
                     {"region_id": 1, "name": "W_F1_C01", "group_id": "Windows", "tags": []}]}
src = {"input:0": red, "input:1": blue}


def doc(*layers):
    return {"version": 1, "canvas": [W, H], "layers": list(layers)}


BASE = {"id": "base", "name": "facade", "kind": "base"}
r = rd.render(doc({"id": "a", "name": "red box", "kind": "image", "source": "input:0", "x": 10, "y": 10, "w": 40, "h": 20}, BASE), base, src, labels, table)
im = r["image"]
check("placed layer covers its box", np.allclose(im[15:25, 15:45], [1, 0, 0], atol=1e-3))
check("outside the box the base shows", np.allclose(im[50:70, 60:110], 0.5, atol=1e-6))
r = rd.render(doc({"id": "a", "name": "rot", "kind": "image", "source": "input:0", "x": 40, "y": 30, "w": 40, "h": 20, "rotation": 90}, BASE), base, src)
ys, xs = np.nonzero(r["image"][..., 1] < 0.25)
check("rotation 90: the box stands upright", (ys.max() - ys.min()) > (xs.max() - xs.min()) * 1.5, f"{np.ptp(ys)} x {np.ptp(xs)}")
half = np.zeros((10, 20, 4), np.float32); half[:, :10, 0] = 1; half[:, 10:, 2] = 1; half[..., 3] = 1
r = rd.render(doc({"id": "a", "name": "flip", "kind": "image", "source": "input:2", "x": 0, "y": 0, "w": 40, "h": 20, "flip_h": True}, BASE), base, {"input:2": half})
check("flip_h mirrors the layer", r["image"][10, 5, 2] > 0.9 and r["image"][10, 35, 0] > 0.9)

# front / back: a layer below the base shows only through the base's holes
behind = {"id": "sky", "name": "sky", "kind": "image", "source": "input:1", "x": 0, "y": 0, "w": W, "h": H}
r = rd.render(doc(dict(BASE, holes="group:Windows"), behind), base, src, labels, table)
check("behind the facade: seen only through the hole", np.allclose(r["image"][25, 80], [0, 0, 1], atol=1e-3)
      and np.allclose(r["image"][60, 20], 0.5, atol=1e-6))
seen = {e[0]: rd.mask_full(e, W, H) for e in r["layer_masks"]}
check("layer mask = what the audience sees (the hole only)", np.allclose(seen["sky"], labels == 1))

# masks: shape, invert, brightness
A = {"id": "a", "name": "red", "kind": "image", "source": "input:0", "x": 0, "y": 0, "w": 60, "h": 40, "mask": {"by": "m"}}
M = {"id": "m", "name": "stencil", "kind": "image", "source": "input:1", "x": 20, "y": 10, "w": 20, "h": 20, "visible": False}
r = rd.render(doc(A, M, BASE), base, src)
check("mask by a hidden layer keeps only its shape", np.allclose(r["image"][20, 30], [1, 0, 0], atol=1e-3) and np.allclose(r["image"][5, 5], 0.5))
r = rd.render(doc(dict(A, mask={"by": "m", "invert": True}), M, BASE), base, src)
check("cut out: the mask punches a hole", np.allclose(r["image"][20, 30], 0.5) and np.allclose(r["image"][5, 5], [1, 0, 0], atol=1e-3))
grad = np.zeros((H, W, 4), np.float32); grad[..., :3] = np.linspace(0, 1, W)[None, :, None]; grad[..., 3] = 1
r = rd.render(doc(dict(A, w=W, h=H, mask={"by": "g", "mode": "brightness"}), {"id": "g", "name": "matte", "kind": "paint", "source": "file:g.png", "visible": False}, BASE),
              base, {"input:0": red, "file:g.png": grad})
r0 = rd.render(doc({"id": "a", "name": "stretched", "kind": "image", "source": "input:0", "x": 0, "y": 0, "w": W, "h": H}, BASE), base, src)
check("a stretched layer is opaque up to its box edge (like drawImage)", np.allclose(r0["image"][:, [0, W - 1], 0], 1, atol=1e-3)
      and np.allclose(r0["image"][[0, H - 1], :, 0], 1, atol=1e-3))
check("brightness mask fades with the matte", r["image"][40, 5, 0] < 0.55 and r["image"][40, W - 3, 0] > 0.95)
r = rd.render(doc(dict(A, mask={"by": "below"}), M, BASE), base, src)
check("clipping to the layer below", np.allclose(r["image"][20, 30], [1, 0, 0], atol=1e-3) and np.allclose(r["image"][5, 5], 0.5))

# clip to a region, grade only what is below
r = rd.render(doc(dict(A, w=W, h=H, clip="W_F1_*", mask={}), BASE), base, src, labels, table)
check("clip: only inside the matching region", np.allclose(r["image"][25, 80], [1, 0, 0], atol=1e-3) and np.allclose(r["image"][60, 20], 0.5))
G = {"id": "grade", "name": "grade", "kind": "adjust", "adjust": {"brightness": 0.5}}
r = rd.render(doc(dict(A, x=0, y=0, w=20, h=10, mask={}), G, BASE), base, src)
check("adjustment grades everything below", np.allclose(r["image"][60, 60], 0.25, atol=1e-6))
check("... but not the layer above it", np.allclose(r["image"][5, 10], [1, 0, 0], atol=1e-3))
r = rd.render(doc(dict(A, blend="multiply", mask={}), BASE), base, src)
check("blend multiply in the render", np.allclose(r["image"][20, 30], [0.5, 0, 0], atol=1e-3))
r = rd.render(doc(dict(A, opacity=0.5, mask={}), BASE), base, src)
check("opacity", np.allclose(r["image"][20, 30], [0.75, 0.25, 0.25], atol=1e-3))

# changed mask, regions, rules
K = {"id": "k", "name": "kiosk", "kind": "image", "source": "input:0", "x": 10, "y": 50, "w": 30, "h": 15, "action": "rediffuse", "prompt": "copper kiosk", "denoise": 0.6}
E = {"id": "e", "name": "ivy", "kind": "image", "source": "input:1", "x": 80, "y": 50, "w": 20, "h": 20, "action": "edges"}
P = {"id": "p", "name": "plain", "kind": "image", "source": "input:1", "x": 50, "y": 5, "w": 10, "h": 10, "action": "keep"}
r = rd.render(doc(K, E, P, BASE), base, src, labels, table, seam_px=3)
ch = r["changed"]
check("changed: the re-diffused layer is inside", ch[57, 25] == 1)
check("changed: edges-only layer has a band but no centre", ch[50, 90] == 1 and ch[60, 90] == 0)
check("changed: kept layer untouched", ch[10, 55] == 0)
names = [g["name"] for g in r["table"]["regions"]]
check("regions: placed layers become named regions, old ones stay", {"kiosk", "ivy", "plain", "wall", "W_F1_C01"} <= set(names), str(names))
kid = names.index("kiosk")
check("regions: the label map has the layer where it is seen", (r["labels"][57, 25] == kid))
check("regions: director layers are tagged", "director" in r["table"]["regions"][kid]["tags"] and "rediffuse" in r["table"]["regions"][kid]["tags"])
bb = r["table"]["regions"][kid]["bbox"]
check("regions: bbox is x, y, w, h like every regions table (1 px anti-aliased edge)", abs(bb[0] - 10) <= 1 and abs(bb[1] - 50) <= 1 and abs(bb[2] - 30) <= 2 and abs(bb[3] - 15) <= 2, str(bb))
# the sequence path: image only and cached region masks give the same picture
cache = {}
io = rd.render(doc(dict(K, clip="W_F1_*"), E, P, dict(BASE, holes="group:Windows")), base, src, labels, table, image_only=True, mask_cache=cache)
full = rd.render(doc(dict(K, clip="W_F1_*"), E, P, dict(BASE, holes="group:Windows")), base, src, labels, table)
again = rd.render(doc(dict(K, clip="W_F1_*"), E, P, dict(BASE, holes="group:Windows")), base, src, labels, table, image_only=True, mask_cache=cache)
check("image_only: same image as the full render, nothing else computed", np.array_equal(io["image"], full["image"]) and set(io) == {"image", "notes"})
check("mask cache: reused selections give the same image", np.array_equal(again["image"], full["image"]) and any(k[0] == "mask" for k in cache))
plan = pl.resolve(r["table"], r["rules"])
pk = plan["regions"][kid]
check("rules resolve in the plan: kiosk re-diffused with its prompt", pk["strategy"] == "inpaint" and abs(pk["denoise"] - 0.6) < 1e-9
      and "copper kiosk" in pk["prompt"], json.dumps(pk)[:200])
check("rules: edges-only = keep + blend on, keep = blend off",
      plan["regions"][names.index("ivy")]["strategy"] == "keep" and plan["regions"][names.index("ivy")]["blend"] is True
      and plan["regions"][names.index("plain")]["blend"] is False)

# shapes: solids, ellipses, polygons, feather, a hidden shape as a mask
SOL = {"id": "s", "name": "solid", "kind": "shape", "shape": {"type": "rect"}, "color": "#ff0000", "x": 0, "y": 0, "w": W, "h": H}
r = rd.render(doc(SOL, BASE), base, src)
check("solid: the whole canvas in its colour", np.allclose(r["image"][5, 5], [1, 0, 0]) and np.allclose(r["image"][70, 110], [1, 0, 0]))
ELL = {"id": "e", "name": "ellipse", "kind": "shape", "shape": {"type": "ellipse"}, "color": "#00ff00", "x": 20, "y": 10, "w": 80, "h": 60}
r = rd.render(doc(ELL, BASE), base, src)
check("ellipse: centre filled, box corner not", np.allclose(r["image"][40, 60], [0, 1, 0]) and np.allclose(r["image"][12, 22], 0.5))
TRI = {"id": "t", "name": "tri", "kind": "shape", "shape": {"type": "poly", "points": [[0, 1], [1, 1], [0.5, 0]]}, "color": "#0000ff",
       "x": 20, "y": 10, "w": 80, "h": 60}
r = rd.render(doc(TRI, BASE), base, src)
check("polygon: inside the triangle filled, above its sides not", np.allclose(r["image"][60, 60], [0, 0, 1]) and np.allclose(r["image"][15, 25], 0.5))
FE = dict(ELL, shape={"type": "ellipse", "feather": 6})
r = rd.render(doc(FE, BASE), base, src)
check("feather: soft edge that spreads outside the box", 0.5 < r["image"][40, 21, 1] < 0.95 and r["image"][40, 16, 1] > 0.5)
MASKED = {"id": "a", "name": "red", "kind": "image", "source": "input:0", "x": 0, "y": 0, "w": W, "h": H, "mask": {"by": "e"}}
r = rd.render(doc(MASKED, dict(ELL, visible=False), BASE), base, src)
check("hidden shape as a mask: the layer only inside the ellipse", np.allclose(r["image"][40, 60], [1, 0, 0]) and np.allclose(r["image"][12, 22], 0.5))
check("shape: short / bad polygons fall back to a rect", rd.parse(doc(dict(TRI, shape={"type": "poly", "points": [[0, 0]]}), BASE))["layers"][0]["shape"]["type"] == "rect")

# shaded by the light: the layer x (light on the clay / clay albedo)
lightrgba = np.zeros((H, W, 4), np.float32); lightrgba[..., 3] = 1
lightrgba[:, :60, :3] = 0.7                               # left half lit like the plain clay (albedo 0.7) -> factor 1
lightrgba[:, 60:, :3] = 0.35                              # right half in shadow -> factor 0.5
LT = {"id": "lt", "name": "lamps", "kind": "light", "light": {"clay": 0.7}, "visible": False}
SH = {"id": "a", "name": "red", "kind": "image", "source": "input:0", "x": 0, "y": 0, "w": W, "h": H, "light_react": {"amount": 1}}
r = rd.render(doc(SH, LT, BASE), base, dict(src, **{"light:lt": lightrgba}))
check("light react: lit side unchanged, shadow side halved", np.allclose(r["image"][40, 30], [1, 0, 0], atol=1e-5) and np.allclose(r["image"][40, 90], [0.5, 0, 0], atol=1e-5))
r = rd.render(doc(dict(SH, light_react={"amount": 0.5}), LT, BASE), base, dict(src, **{"light:lt": lightrgba}))
check("light react: amount 0.5 = halfway", np.allclose(r["image"][40, 90], [0.75, 0, 0], atol=1e-5))
r = rd.render(doc(SH, BASE), base, src)
check("light react without a light layer: a note, the layer unchanged", np.allclose(r["image"][40, 90], [1, 0, 0]) and any("react" in n for n in r["notes"]))

# review regressions: name clashes, prompts, odd region names, stable input ids
tbl2 = {"regions": [{"region_id": 0, "name": "kiosk", "group_id": "wall", "tags": []},
                    {"region_id": 1, "name": "Window Frame", "group_id": "Windows", "tags": []}]}
K2 = dict(K, prompt="copper kiosk" + chr(10) + "warm light // grain")
r = rd.render(doc(K2, BASE), base, src, labels, tbl2)
names2 = [g["name"] for g in r["table"]["regions"]]
check("a layer named like an existing region gets its own name", "kiosk_2" in names2 and "[name:kiosk_2]" in r["rules"], str(names2))
check("prompts become one line without the comment marker", "prompt = copper kiosk warm light / grain" in r["rules"], r["rules"])
p2 = pl.resolve(r["table"], r["rules"])
check("... and the rules still resolve", p2["regions"][names2.index("kiosk_2")]["strategy"] == "inpaint")
check("input regions keep their ids and names", [g["region_id"] for g in r["table"]["regions"][:2]] == [0, 1]
      and names2[:2] == ["kiosk", "Window Frame"])
r = rd.render(doc(dict(A, w=W, h=H, clip="Window Frame", mask={}), BASE), base, src, labels, tbl2)
check("clip to a region name with a space (exact match)", np.allclose(r["image"][25, 80], [1, 0, 0], atol=1e-3))
r = rd.render(doc(dict(A, w=W, h=H, clip="a]b:[c", mask={}), BASE), base, src, labels, tbl2)
check("an unparsable clip selector matches nothing instead of crashing", np.allclose(r["image"][25, 80], 0.5))

# document errors
for bad, why in (({"layers": [BASE, BASE]}, "two bases"), ({"layers": [{"id": "x", "kind": "image"}]}, "no base"),
                 ({"layers": [dict(BASE), {"id": "base", "kind": "image"}]}, "duplicate id"),
                 ({"layers": [dict(BASE, blend="glow")]}, "unknown blend")):
    try:
        rd.parse(bad)
        check(f"document error: {why}", False)
    except rd.DocumentError:
        check(f"document error: {why}", True)
check("empty document = just the base", len(rd.parse("")["layers"]) == 1)
r = rd.render(doc({"id": "x", "name": "gone", "kind": "image", "source": "input:9"}, BASE), base, src)
check("missing source is skipped with a note", any("not found" in n for n in r["notes"]) and np.allclose(r["image"], 0.5))

# light layers (the Cycles render itself needs Blender: here only the document side)
from kubakub.scene3d import scene_view as sv  # noqa: E402
LIGHT = {"id": "L9", "name": "night", "kind": "light", "blend": "multiply",
         "light": {"environment": "sunset", "lights": [{"type": "area", "x": 2, "height": 5, "distance": 4, "power": 900, "color": "#ff8000"},
                                                        {"type": "sun", "azimuth": 20, "elevation": 30, "power": 3, "on": False}],
                   "glow": [{"by": "layer:a", "color": "#00ff00", "strength": 50}, {"by": "group:Windows"}, {"by": "  "}]}}
d = rd.parse(doc(LIGHT, BASE))
check("light layer parses, keeps its rig, action keep", d["layers"][0]["kind"] == "light" and d["layers"][0]["action"] == "keep"
      and d["layers"][0]["light"]["environment"] == "sunset")
lit = np.zeros((H, W, 4), np.float32); lit[..., :3] = 0.5; lit[..., 3] = 1
r = rd.render(doc(LIGHT, BASE), base, {"light:L9": lit})
check("light layer composites like a full-canvas layer (multiply 0.5 x 0.5)", np.allclose(r["image"], 0.25, atol=1e-5))
check("a light layer is a look, not a placed object", not r["layer_masks"] and r["changed"].max() == 0)
r = rd.render(doc(LIGHT, BASE), base, {})
check("unrendered light layer: skipped with a note", np.allclose(r["image"], 0.5) and any("not rendered" in n for n in r["notes"]))
rig = sv.rig_from_doc(LIGHT["light"])
check("rig: switched-off lights dropped, colours as rgb", len(rig["lights"]) == 1 and np.allclose(rig["lights"][0]["color"], [1, 128 / 255, 0]))
check("rig: empty glow entries dropped, defaults filled", len(rig["glow"]) == 2 and rig["glow"][1]["strength"] == 20
      and rig["samples"] == 64 and rig["background"] == "black")
bad = sv.rig_from_doc({"environment": "mars", "samples": "lots", "lights": [{"type": "laser"}, {"type": "point", "x": "nan"}]})
check("rig: bad values fall back instead of failing", bad["environment"] == "night" and bad["samples"] == 64
      and len(bad["lights"]) == 1 and bad["lights"][0]["x"] == 0)
check("rig: projector off by default, casts the layers below", rig["projector"] == {"on": False, "brightness": 1.0, "mode": "light", "source": "below"}
      and rig["view"] == "AgX")
pr = sv.rig_from_doc({"projector": {"on": True, "source": "layer:a", "brightness": 2}, "view": "Standard"})["projector"]
check("rig: projector from a layer", pr["on"] and pr["source"] == "layer:a" and pr["brightness"] == 2)
check("rig: unknown projector source falls back", sv.rig_from_doc({"projector": {"source": "sky"}})["projector"]["source"] == "below")
check("projector power: brightness x distance^2 / albedo", np.isclose(sv.projector_power(2, 10, 0.5), 2 * 4 * np.pi ** 2 * 100 / 0.5)
      and sv.projector_power(1, 0, 0) > 0)
A_ = {"id": "a", "name": "red box", "kind": "image", "source": "input:0", "x": 10, "y": 10, "w": 40, "h": 20}
gd = doc(LIGHT, A_, BASE)
gm = rd.glow_masks(gd, rig, src, labels, table, W, H)
check("glow by layer = the layer's placed shape", len(gm) == 2 and gm[0][0][15:25, 15:45].min() > 0.99 and gm[0][0][50:, 60:].max() == 0
      and np.allclose(gm[0][1], [0, 1, 0]) and gm[0][2] == 50)
check("glow by region selector = the region mask", gm[1][0][25, 80] == 1 and gm[1][0][60, 20] == 0)
check("glow by a missing layer is dropped", rd.glow_masks(gd, sv.rig_from_doc({"glow": [{"by": "layer:nope"}]}), src, labels, table, W, H) == [])
clipped = rd.layer_shape(doc(dict(A_, w=W, h=H, x=0, y=0, clip="group:Windows"), BASE), "a", src, labels, table, W, H)
check("layer shape follows its clip", clipped[25, 80] == 1 and clipped[60, 20] == 0)

# timeline: same numbers as the window's tests (scratch harness test4)
keys = [{"t": 0, "v": 50, "e": "smooth"}, {"t": 2, "v": 200, "e": "smooth"}]
check("key_value smooth half way", abs(rd.key_value(keys, 1) - 125) < 1e-9)
check("key_value linear / hold", abs(rd.key_value([dict(keys[0], e="linear"), keys[1]], 0.5) - 87.5) < 1e-9
      and rd.key_value([dict(keys[0], e="hold"), keys[1]], 1.9) == 50)
check("key_value before / after / unsorted", rd.key_value(keys, -1) == 50 and rd.key_value(keys[::-1], 5) == 200)
check("key_value colours like the window", rd.key_value([{"t": 0, "v": "#000000"}, {"t": 4, "v": "#ffffff"}], 2) == "#808080")
check("key_value strings / bools step", rd.key_value([{"t": 0, "v": True}, {"t": 1, "v": False}], 0.9) is True)
ad = {"layers": [{"id": "A", "kind": "image", "x": 0, "anim": {"x": keys, "opacity": [{"t": 0, "v": 1}, {"t": 2, "v": 0, "e": "linear"}]}},
                 {"id": "Lt", "kind": "light", "light": {"lights": [{"id": "p1", "power": 100}]},
                  "anim": {"light.lights.#p1.power": [{"t": 0, "v": 100}, {"t": 2, "v": 300}], "light.lights.#gone.power": [{"t": 0, "v": 1}]}},
                 {"id": "base", "kind": "base"}], "timeline": {"duration": 2, "fps": 10}}
a1 = rd.animate(ad, 1)
check("animate writes layer and lamp values", a1["layers"][0]["x"] == 125 and a1["layers"][1]["light"]["lights"][0]["power"] == 200)
check("animate: missing lamp id ignored, input untouched", ad["layers"][0]["x"] == 0 and len(a1["layers"][1]["light"]["lights"]) == 1)
check("timeline frames", rd.timeline(ad)["frames"] == 20 and rd.timeline({})["fps"] == 25)
check("animated document still renders", rd.render(rd.animate(ad, 1), base, {})["image"].shape == base.shape)
vd_doc = doc({"id": "a", "name": "jelly", "kind": "image", "source": "input:0", "x": 10, "y": 10, "w": 40, "h": 20,
              "video": {"on": True, "prompt": "it  floats // slowly", "t_start": 1, "t_end": 3.5, "motion": 0.8}}, BASE)
rv = rd.render(vd_doc, base, src, labels, table)
check("animated layer -> animate rules", "animate = on" in rv["rules"] and "video_prompt = it floats / slowly" in rv["rules"]
      and "t_start = 1" in rv["rules"] and "t_end = 3.5" in rv["rules"] and "motion = 0.8" in rv["rules"], rv["rules"])
pv = pl.resolve(rv["table"], rv["rules"])
jl = next(e for e in pv["regions"] if e["name"] == "jelly")
check("... and the plan reads them", jl.get("animate") is True and jl.get("t_end") == 3.5 and jl.get("video_prompt") == "it floats / slowly", jl)
check("no video block: no animate", "animate" not in rd.render(doc(dict(vd_doc["layers"][0], video={}), BASE), base, src, labels, table)["rules"])
sd = rd.scale_doc({"canvas": [W, H], "layers": [{"id": "A", "x": 10, "y": 20, "w": 40, "h": 60}]}, 0.5)
check("scale_doc", sd["layers"][0]["x"] == 5 and sd["layers"][0]["h"] == 30 and sd["canvas"] == [round(W / 2), round(H / 2)])

print()
if failures:
    print(f"{len(failures)} FAILED: {', '.join(failures)}")
    sys.exit(1)
print("all director tests passed")
