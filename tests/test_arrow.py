"""
Model free test of the drawn arrow ('vector'): the text of it, the mask field along it (kubakub/maskfields.py), what
a shader gets of it (kubakub/shadertoy.py, kubakub/shader_render.py), the two nodes that take it and the director's
shape layers. The window's twin of the field is compared per pixel in tests/test_motion.py.

Run from the ComfyUI_windows_portable folder:
    python_embeded\\python.exe ComfyUI\\custom_nodes\\ComfyUI_KubaNodes\\tests\\test_arrow.py
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

TMP = tempfile.mkdtemp(prefix="kkd_arrow_")
folder_paths.set_temp_directory(os.path.join(TMP, "temp"))

sys.path.insert(0, HERE)     # embedded Python leaves the script folder off the path
import _pack  # noqa: E402,F401  (the pack as a package, see tests/_pack.py)

import kubapack.nodes.masks.nodes_masks as nm  # noqa: E402
import kubapack.nodes.shader.nodes_shader as ns  # noqa: E402
from kubakub import maskfields as mf  # noqa: E402
from kubakub import shader_render as sr  # noqa: E402
from kubakub import shadertoy as st  # noqa: E402
from kubakub.director import render as rd  # noqa: E402

failures = []


def check(name, cond, detail=""):
    print(("ok   " if cond else "FAIL ") + name + (f"  ({detail})" if detail and not cond else ""))
    if not cond:
        failures.append(name)


def raises(fn):
    try:
        fn()
    except ValueError as e:
        return str(e)
    return ""


# ---- the text
check("text: four numbers, commas or spaces", mf.parse_vector("0.2, 0.5, 0.8 0.5") == (0.2, 0.5, 0.8, 0.5) and mf.parse_vector([0, 0, 1, 1]) == (0.0, 0.0, 1.0, 1.0))
check("text: empty and an arrow without a length are no arrow", mf.parse_vector("") is None and mf.parse_vector("  ") is None and mf.parse_vector("0.3,0.3,0.3,0.3") is None)
check("text: anything else says what it should be", "four numbers" in raises(lambda: mf.parse_vector("0.2, 0.5")) and "four numbers" in raises(lambda: mf.parse_vector("a, b, c, d")))

# ---- the field
lab = np.zeros((101, 201), np.int32)                          # one region over the whole map
f, _ = mf.field(lab, 1, "direction", "all together", angle=90.0, vector=(0.25, 0.5, 0.75, 0.5))
check("field, direction, all together: 0 at the arrow's start, 1 at its end, flat before and after",
      abs(f[50, 50]) < 1e-6 and abs(f[50, 150] - 1) < 1e-6 and abs(f[50, 100] - 0.5) < 1e-6 and f[50, 10] == 0 and f[50, 190] == 1
      and abs(f[5, 100] - 0.5) < 1e-6, str(f[50, ::25]))
f, _ = mf.field(lab, 1, "direction", "all together", vector=(0.5, 0.9, 0.5, 0.1))
check("field: an arrow pointing up runs from the bottom to the top", f[95, 100] == 0 and f[5, 100] == 1 and 0.4 < f[50, 30] < 0.6)
f, _ = mf.field(lab, 1, "direction", "all together", vector=(0.25, 0.5, 0.75, 0.5), invert=True)
check("field: invert turns it around", abs(f[50, 50] - 1) < 1e-6 and abs(f[50, 150]) < 1e-6)
f, _ = mf.field(lab, 1, "radial", "all together", centre=(0.9, 0.9), vector=(0.5, 0.5, 0.75, 0.5))
check("field, radial, all together: the start is the centre, the length how far it reaches",
      abs(f[50, 100]) < 1e-6 and abs(f[50, 150] - 1) < 1e-6 and abs(f[50, 50] - 1) < 1e-6 and abs(f[50, 125] - 0.5) < 1e-6 and f[0, 0] == 1)
two = np.full((40, 100), -1, np.int32)
two[5:35, 5:45], two[5:35, 55:95] = 0, 1
f, _ = mf.field(two, 2, "direction", "each region", angle=0.0, vector=(0.5, 0.1, 0.5, 0.9))
check("field, direction, each region: every region takes the arrow's angle", f[5, 20] == 0 and f[34, 20] == 1 and f[5, 80] == 0 and f[34, 80] == 1
      and abs(f[20, 6] - f[20, 43]) < 1e-6)
a, _ = mf.field(two, 2, "edge distance", "each region", vector=(0.1, 0.1, 0.9, 0.9))
b, _ = mf.field(two, 2, "edge distance", "each region")
c, _ = mf.field(two, 2, "direction", "all together", angle=30.0, vector=(0.4, 0.4, 0.4, 0.4))
d, _ = mf.field(two, 2, "direction", "all together", angle=30.0)
check("field: other kinds and an arrow without a length are as before", np.array_equal(a, b) and np.array_equal(c, d))

# ---- the mask field node
out = nm.KUBA_MaskField.execute(mask=torch.ones((1, 101, 201)), field="direction", per="all together", vector="0.25, 0.5, 0.75, 0.5").result
check("mask field node: the arrow is used and the report says so", abs(float(out[0][0, 50, 50])) < 1e-6 and abs(float(out[0][0, 50, 150]) - 1) < 1e-6
      and "the drawn arrow is used" in out[2], out[2])
out = nm.KUBA_MaskField.execute(mask=torch.ones((1, 40, 80)), field="noise", vector="0.25, 0.5, 0.75, 0.5").result
check("mask field node: a field that has no direction says that the arrow is not used", "not used by field = noise" in out[2], out[2])
check("mask field node: a wrong text is an error that says what to type", "four numbers" in raises(lambda: nm.KUBA_MaskField.execute(mask=torch.ones((1, 8, 8)), vector="left")))
same = nm.KUBA_MaskField.execute(mask=torch.ones((1, 40, 80)), field="direction", per="all together", angle=20.0).result[0]
old = mf.field(np.zeros((40, 80), np.int32), 1, "direction", "all together", 20.0)[0]
check("mask field node: without an arrow nothing changes", np.array_equal(same[0].numpy(), old))

# ---- what a shader gets
check("shader uniforms: y counts from the bottom", sr.vector_values((0.2, 0.9, 0.7, 0.1)) == {"kAx": 0.2, "kAy": 1.0 - 0.9, "kBx": 0.7, "kBy": 1.0 - 0.1, "kVector": 1.0, "kVecField": 1.0}
      and sr.vector_values(None) == {} and sr.vector_values((0, 0, 1, 1), as_field=False)["kVecField"] == 0.0)
t = st.translate("void mainImage(out vec4 c, in vec2 f){ c = vec4(kFrom, kDir.y, 1.0); }")
check("shader text: a shader that reads the arrow or the field follows it", t.follows and st.translate(st.preset_code("scan band")).follows
      and st.translate(st.preset_code("stripes")).follows and not st.translate(st.preset_code("grade")).follows)
try:
    PROBE = "void mainImage(out vec4 c, in vec2 f){ c = vec4(kField(kUV), kFrom.x, kDir.y * 0.5 + 0.5, 1.0); }"
    i1 = ns.KUBA_Shader.execute("rings", code=PROBE, width=200, height=100, seconds=0.04, loop=False, vector="0.25, 0.5, 0.75, 0.5").result
    px = i1[0][0]
    check("shader node: the arrow is the field, kFrom its start", abs(float(px[50, 50, 0])) < 0.02 and abs(float(px[50, 150, 0]) - 1) < 0.02
          and abs(float(px[50, 100, 0]) - 0.5) < 0.02 and abs(float(px[3, 3, 1]) - 0.25) < 0.01 and abs(float(px[3, 3, 2]) - 0.5) < 0.01
          and "the shader follows it" in i1[5], i1[5])
    up = ns.KUBA_Shader.execute("rings", code=PROBE, width=200, height=100, seconds=0.04, loop=False, vector="0.5, 0.9, 0.5, 0.1").result[0][0]
    check("shader node: an arrow drawn upwards points up for the shader, and its field runs from the bottom to the top",
          float(up[3, 3, 2]) > 0.99 and float(up[92, 100, 0]) < 0.05 and float(up[8, 100, 0]) > 0.95, str(up[3, 3]))
    none = ns.KUBA_Shader.execute("rings", code=PROBE, width=200, height=100, seconds=0.04, loop=False).result
    check("shader node: without an arrow the report has no word of one and kFrom is 0", "arrow" not in none[5]
          and float(none[0][0, 3, 3, 1]) < 0.01, none[5])
    ramp = torch.linspace(1, 0, 200)[None, None].repeat(1, 100, 1)
    own = ns.KUBA_Shader.execute("rings", code=PROBE, width=200, height=100, seconds=0.04, loop=False, field=ramp, vector="0.25, 0.5, 0.75, 0.5").result[0][0]
    check("shader node: a connected field comes first, the arrow still gives its start and direction",
          float(own[50, 20, 0]) > 0.85 and abs(float(own[3, 3, 1]) - 0.25) < 0.01)
    plain = ns.KUBA_Shader.execute("grade", width=64, height=32, seconds=0.04, vector="0.25, 0.5, 0.75, 0.5").result[5]
    check("shader node: a shader without a direction says that it does not use the arrow", "does not use it" in plain, plain)
    a0 = ns.KUBA_Shader.execute("stripes", width=120, height=120, seconds=0.04, controls="angle = 0, stripes = 4").result[0][0]
    a1 = ns.KUBA_Shader.execute("stripes", width=120, height=120, seconds=0.04, controls="angle = 0, stripes = 4", vector="0.5, 0.9, 0.5, 0.1").result[0][0]
    check("preset stripes: the arrow turns them", float((a0[:, 60] - a0[0, 60]).abs().max()) < 0.05 and float((a1[60, :] - a1[60, 0]).abs().max()) < 0.05
          and float((a1[:, 60] - a1[0, 60]).abs().max()) > 0.3)
    r0 = ns.KUBA_Shader.execute("rings", width=120, height=120, seconds=0.04, vector="0.25, 0.25, 0.5, 0.25").result[0][0]
    check("preset rings: they start at the arrow's start", float((r0[30, 30 - 12] - r0[30, 30 + 12]).abs().max()) < 0.03
          and float((r0[30 - 12, 30] - r0[30 + 12, 30]).abs().max()) < 0.03)
except Exception as e:  # noqa: BLE001
    if "graphics context" in str(e) or "GL" in type(e).__name__:
        print("skip the GPU part: no GL context here")
    else:
        raise

# ---- the director's shape layer keeps it
sh = rd._shape({"type": "rect", "shader": "scan band", "vec": [0.1, 0.2, 0.9, 0.2]})
check("director: a shader fill keeps its arrow", sh.get("vec") == [0.1, 0.2, 0.9, 0.2])
check("director: a broken or empty arrow is dropped", "vec" not in rd._shape({"type": "rect", "shader": "scan band", "vec": [0.1, "x", 0.9, 0.2]})
      and "vec" not in rd._shape({"type": "rect", "shader": "scan band", "vec": [0.4, 0.4, 0.4, 0.4]}) and "vec" not in rd._shape({"type": "rect", "vec": [0, 0, 1, 1]}))

shutil.rmtree(TMP, ignore_errors=True)
print("\nALL PASSED" if not failures else f"\n{len(failures)} FAILED: {failures}")
sys.exit(1 if failures else 0)
