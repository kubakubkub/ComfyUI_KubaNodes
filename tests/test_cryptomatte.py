"""
Model free test of the EXR reader (kubakub/exr.py) and cryptomatte decoding (kubakub/cryptomatte.py):
a synthetic cryptomatte EXR written by exr.write (NONE and ZIP), and - if Blender 4.x is installed - real Blender
cryptomatte renders (ZIP read in numpy, PIZ through Blender's OpenImageIO), made in a temp folder.

Run from the ComfyUI_windows_portable folder:
    python_embeded\\python.exe ComfyUI\\custom_nodes\\ComfyUI_KubaNodes\\tests\\test_cryptomatte.py
"""

import json
import os
import subprocess
import sys
import tempfile
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from kubakub import cryptomatte as cm, exr  # noqa: E402
from kubakub.scene3d import bridge  # noqa: E402

failures = []


def check(name, cond, detail=""):
    print(("ok   " if cond else "FAIL ") + name + (f"  ({detail})" if detail else ""))
    if not cond:
        failures.append(name)


# --- synthetic: three objects, hashes as float bit patterns, a manifest in the header
H, W = 48, 64
hashes = {"wall": 0x3F1A2B3C, "window_01": 0x40112233, "window_02": 0x41445566}
ids = np.full((H, W), hashes["wall"], np.uint32)
ids[10:20, 8:24] = hashes["window_01"]
ids[10:20, 36:52] = hashes["window_02"]
ids[40:, :] = 0                                                  # nothing rendered at the bottom
cov = np.where(ids > 0, 1.0, 0.0).astype(np.float32)
mat = np.where((ids == hashes["window_01"]) | (ids == hashes["window_02"]), 0x3E000001, 0x3D000002).astype(np.uint32)
chan = {
    "View.CryptoObject00.r": ids.view(np.float32), "View.CryptoObject00.g": cov,
    "View.CryptoObject00.b": np.zeros((H, W), np.float32), "View.CryptoObject00.a": np.zeros((H, W), np.float32),
    "View.CryptoMaterial00.r": mat.view(np.float32), "View.CryptoMaterial00.g": cov,
    "View.CryptoMaterial00.b": np.zeros((H, W), np.float32), "View.CryptoMaterial00.a": np.zeros((H, W), np.float32),
    "View.Combined.R": np.full((H, W), 0.2, np.float32), "View.Combined.G": np.full((H, W), 0.4, np.float32),
    "View.Combined.B": np.full((H, W), 0.6, np.float16),
}
attrs = {"cryptomatte/aaa0001/name": "View.CryptoObject",
         "cryptomatte/aaa0001/manifest": json.dumps({k: f"{v:08x}" for k, v in hashes.items()}),
         "cryptomatte/bbb0002/name": "View.CryptoMaterial",
         "cryptomatte/bbb0002/manifest": json.dumps({"glass": "3e000001", "stone": "3d000002"})}

with tempfile.TemporaryDirectory() as d:
    for comp in ("none", "zip"):
        p = os.path.join(d, f"s_{comp}.exr")
        exr.write(p, chan, comp, attrs)
        r = exr.read(p)
        same = all(np.array_equal(np.asarray(r["channels"][k]).view(np.uint8), np.asarray(v).view(np.uint8))
                   for k, v in chan.items())
        check(f"{comp}: every channel bit-exact after write / read", same and r["compression"] == comp)
    lays = cm.layers(r["attrs"])
    check("two cryptomatte layers found", sorted(lays) == ["View.CryptoMaterial", "View.CryptoObject"], str(list(lays)))
    check("pick layer by short name", cm.pick_layer(list(lays), "object") == "View.CryptoObject"
          and cm.pick_layer(list(lays), "MATERIAL") == "View.CryptoMaterial" and cm.pick_layer(list(lays), "asset") is None)
    lab, names = cm.label_map(r["channels"], "View.CryptoObject", lays["View.CryptoObject"]["manifest"])
    check("names from the manifest", sorted(names) == ["wall", "window_01", "window_02"], str(names))
    check("labels where the objects are", names[lab[15, 10]] == "window_01" and names[lab[15, 40]] == "window_02"
          and names[lab[30, 30]] == "wall")
    check("nothing rendered = no region", (lab[40:] == -1).all())
    lab2, names2 = cm.label_map(r["channels"], "View.CryptoObject", lays["View.CryptoObject"]["manifest"], "wall")
    check("exclude by name", names2 == sorted(names2) and "wall" not in names2 and (lab2[30, 30] == -1))
    tags = cm.tags_from(r["channels"], "View.CryptoMaterial", lays["View.CryptoMaterial"]["manifest"], lab, len(names))
    check("material names as tags", {n: t for n, t in zip(names, tags)} ==
          {"wall": ["stone"], "window_01": ["glass"], "window_02": ["glass"]}, str(tags))
    b = cm.beauty(r["channels"])
    check("the render picture", b is not None and np.allclose(b[0, 0], [0.2, 0.4, 0.6], atol=1e-3))
    unknown = dict(lays["View.CryptoObject"]["manifest"])
    unknown.pop("window_02")
    _, n3 = cm.label_map(r["channels"], "View.CryptoObject", unknown)
    check("an id missing from the manifest gets its hash as name", "id_41445566" in n3, str(n3))
    big = {"a.R": np.random.default_rng(0).random((1080, 1920), dtype=np.float32)}
    p = os.path.join(d, "big.exr")
    exr.write(p, big, "zip")
    t = time.perf_counter()
    rb = exr.read(p)
    check("1080p float channel read fast", time.perf_counter() - t < 1.5 and np.array_equal(rb["channels"]["a.R"], big["a.R"]),
          f"{time.perf_counter() - t:.2f} s")

    # --- real Blender cryptomatte (skipped without Blender)
    blender = bridge.find_blender("")
    if not blender or not os.path.isfile(blender):
        print("skip real Blender cryptomatte (no Blender 4.x found)")
    else:
        script = os.path.join(d, "mk.py")
        with open(script, "w", encoding="utf-8") as f:
            f.write(r'''
import bpy, sys, os
out = sys.argv[-1]
bpy.ops.wm.read_factory_settings(use_empty=True)
sc = bpy.context.scene
def box(name, x, z, sx, sz, y, mname):
    bpy.ops.mesh.primitive_cube_add(size=1, location=(x, y, z)); o = bpy.context.object; o.name = name
    o.scale = (sx, 0.2, sz); m = bpy.data.materials.get(mname) or bpy.data.materials.new(mname); o.data.materials.append(m)
box("wall", 0, 3, 12, 6, 0.2, "stone")
for i in range(3):
    box(f"window_{i + 1:02d}", -4 + i * 4, 3.5, 2, 2, 0, "glass")
bpy.ops.object.camera_add(location=(0, -20, 3), rotation=(1.5708, 0, 0)); sc.camera = bpy.context.object
sc.camera.data.type = "ORTHO"; sc.camera.data.ortho_scale = 13
sc.render.engine = "CYCLES"; sc.cycles.samples = 1; sc.cycles.device = "CPU"
sc.render.resolution_x, sc.render.resolution_y = 240, 135
vl = sc.view_layers[0]; vl.use_pass_cryptomatte_object = True; vl.use_pass_cryptomatte_material = True
s = sc.render.image_settings; s.file_format = "OPEN_EXR_MULTILAYER"; s.color_depth = "32"
for codec in ("ZIP", "PIZ"):
    s.exr_codec = codec; sc.render.filepath = os.path.join(out, "crypto_" + codec.lower() + ".exr")
    bpy.ops.render.render(write_still=True)
''')
        subprocess.run([blender, "-b", "--factory-startup", "--python", script, "--", d], capture_output=True, timeout=300)
        for codec, how in (("zip", "zip"), ("piz", "via Blender")):
            p = os.path.join(d, f"crypto_{codec}.exr")
            if not os.path.isfile(p):
                check(f"Blender wrote {codec}", False)
                continue
            r = exr.read(p)
            lays = cm.layers(r["attrs"])
            obj = cm.pick_layer(list(lays), "object")
            lab, names = cm.label_map(r["channels"], obj, lays[obj]["manifest"])
            H2, W2 = lab.shape
            mid = [names[lab[int(H2 * 0.42), int(W2 * (0.5 + k * 4 / 13))]] if lab[int(H2 * 0.42), int(W2 * (0.5 + k * 4 / 13))] >= 0 else None
                   for k in (-1, 0, 1)]
            check(f"Blender {codec} ({r['compression']}): the three windows by name", mid == ["window_01", "window_02", "window_03"]
                  and r["compression"] == how, str(mid))
            mt = cm.pick_layer([n for n in lays if n != obj], "material")
            tg = cm.tags_from(r["channels"], mt, lays[mt]["manifest"], lab, len(names))
            check(f"Blender {codec}: windows tagged glass", all(t == ["glass"] for n, t in zip(names, tg) if n.startswith("window")))

print("\n" + ("ALL OK" if not failures else f"{len(failures)} FAILED: {failures}"))
sys.exit(1 if failures else 0)
