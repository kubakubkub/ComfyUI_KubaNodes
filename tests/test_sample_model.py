"""
Model free test of the sample model (kubakub/sample_model.py): the OBJ it writes, its size in metres, its objects and
materials, and that every stone, frame and voussoir is a loose part of its own (what kubakub scene pieces moves).

Run from the ComfyUI_windows_portable folder:
    python_embeded\\python.exe ComfyUI\\custom_nodes\\ComfyUI_KubaNodes\\tests\\test_sample_model.py
"""

import os
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PACK = os.path.dirname(HERE)
sys.path.insert(0, PACK)
sys.path.insert(0, os.path.join(PACK, "kubakub", "scene3d"))
from kubakub import sample_model as sm  # noqa: E402
import scene_ids as si  # noqa: E402

failures = []


def check(name, cond, detail=""):
    print(("ok   " if cond else "FAIL ") + name + (f"  ({detail})" if detail else ""))
    if not cond:
        failures.append(name)


def read_obj(path):
    verts, faces, objects, mats = [], [], [], set()
    for line in open(path, encoding="utf-8"):
        t = line.split()
        if not t:
            continue
        if t[0] == "v":
            verts.append([float(x) for x in t[1:4]])
        elif t[0] == "f":
            faces.append([int(x) - 1 for x in t[1:]])
        elif t[0] == "o":
            objects.append(t[1])
        elif t[0] == "usemtl":
            mats.add(t[1])
    return np.array(verts, np.float32), faces, objects, mats


def body(path):
    return open(path, encoding="utf-8").read().split("\n", 2)[2]          # without the comment and the mtllib line


with tempfile.TemporaryDirectory() as d:
    path = os.path.join(d, "sample.obj")
    info = sm.write_obj(path, floors=3, bays=7, width_m=24.0, height_m=16.0, seed=1)
    V, faces, objects, mats = read_obj(path)
    check("the .obj and its .mtl are written", os.path.isfile(path) and os.path.isfile(path[:-4] + ".mtl"))
    check("24 m wide, 16 m high, stands on the ground", abs(info["width_m"] - 24) < 0.01 and abs(info["height_m"] - 16) < 0.05
          and abs(float(V[:, 1].min())) < 1e-4 and abs(float(V[:, 1].max()) - info["height_m"]) < 1e-3,
          f"{info['width_m']:.2f} x {info['height_m']:.2f}, y {V[:, 1].min():.3f}..{V[:, 1].max():.3f}")
    check("the wall front is z = 0 (stones stand out a few cm), the wall is half a metre thick",
          float(V[:, 2].min()) > -0.6 and float(V[:, 2].max()) < 0.6)
    check("objects named by element", {"wall_F0", "wall_F1", "wall_attic", "door", "door_leaf", "cornice_top",
                                       "pilaster_01", "window_F2_C03", "frame_F2_C03", "arch_F0_C01", "sill_F1"} <= set(objects),
          str(sorted(set(objects))[:12]))
    check("21 windows with frames, 6 arches, 8 pilasters", sum(o.startswith("frame_") for o in objects) == 21
          and sum(o.startswith("arch_") for o in objects) == 6 and sum(o.startswith("pilaster_") for o in objects) == 8)
    check("four materials", mats == set(sm.MATERIALS), str(mats))
    check("every face has 3 or more corners inside the vertex list", all(len(f) >= 3 and max(f) < len(V) for f in faces))

    lt = np.array([len(f) for f in faces], np.int32)
    mesh = {"vert": V, "loop_total": lt, "loop_vert": np.array([i for f in faces for i in f], np.int32)}
    parts = si.loose_parts(mesh, 1e-4)
    check("every piece is a loose part of its own", len(np.unique(parts)) == info["pieces"],
          f"{len(np.unique(parts))} loose parts, {info['pieces']} pieces")
    vol = np.zeros(int(parts.max()) + 1)                 # every piece is closed: outward faces = a positive volume
    for f, p in zip(faces, parts):
        for k in range(1, len(f) - 1):
            vol[p] += float(np.dot(V[f[0]], np.cross(V[f[k]], V[f[k + 1]]))) / 6.0
    check("faces point outwards (positive volume per part)", bool((vol > 0).all()), f"{int((vol <= 0).sum())} parts inside out")
    check("hundreds of stones to move", info["pieces"] > 300, str(info["pieces"]))

    sm.write_obj(os.path.join(d, "again.obj"), floors=3, bays=7, width_m=24.0, height_m=16.0, seed=1)
    check("same settings = same file", body(path) == body(os.path.join(d, "again.obj")))
    small = sm.write_obj(os.path.join(d, "small.obj"), floors=1, bays=2, width_m=8.0, height_m=7.0)
    check("other sizes build (1 floor, 2 bays)", small["pieces"] > 20 and abs(small["width_m"] - 8) < 0.01)

print()
print("all sample model tests passed" if not failures else f"{len(failures)} FAILED: " + ", ".join(failures))
sys.exit(1 if failures else 0)
