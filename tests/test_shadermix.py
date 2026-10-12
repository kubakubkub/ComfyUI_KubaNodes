"""
Model free test of kubakub shader mix: the joining of two pictures (kubakub/shadermix.py), what the engine gives a
shader that is bent by another one (kBend, kMixTex), and the node (nodes/shader/nodes_mix.py). The GPU part is skipped
with a note where ComfyUI's GL context does not start.

Run from the ComfyUI_windows_portable folder:
    python_embeded\\python.exe ComfyUI\\custom_nodes\\ComfyUI_KubaNodes\\tests\\test_shadermix.py
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

TMP = tempfile.mkdtemp(prefix="kkd_mix_")
folder_paths.set_temp_directory(os.path.join(TMP, "temp"))

sys.path.insert(0, HERE)     # embedded Python leaves the script folder off the path
import _pack  # noqa: E402,F401  (the pack as a package, see tests/_pack.py)

import kubapack.nodes.shader.nodes_mix as nx  # noqa: E402
from kubakub import shader_render as sr  # noqa: E402
from kubakub import shadermix as mx  # noqa: E402
from kubakub import shadertoy as st  # noqa: E402

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


# ---- joining two pictures
a = np.zeros((4, 8, 3), np.float32)
a[:, 4:] = 1.0                                              # A: dark left, bright right
b = np.full((4, 8, 3), 0.5, np.float32)
check("join: 0 is A and 1 is B, in every way", all(np.array_equal(mx.join(a, b, 0.0, h), a) and np.array_equal(mx.join(a, b, 1.0, h), b) for h in mx.HOW))
check("join, blend: a plain fade", np.allclose(mx.join(a, b, 0.25, "blend"), a * 0.75 + b * 0.25))
half = mx.join(a, b, 0.5, "a cuts b", soft=0.05)
check("join, a cuts b: B takes A's bright places first", np.allclose(half[:, 4:], 0.5) and np.allclose(half[:, :4], 0.0))
k = [float(mx.cut(np.full((1, 1, 3), 0.5, np.float32), m, 0.1)[0, 0, 0]) for m in (0.0, 0.2, 0.5, 0.8, 1.0)]
check("cut: nothing at 0, everything at 1, growing in between", k[0] == 0 and k[-1] == 1 and all(x <= y for x, y in zip(k, k[1:])) and 0 < k[2] < 1, str(k))
check("bend: none at both ends, the most half way", mx.bend(0.0, 0.2) == 0 and abs(mx.bend(1.0, 0.2)) < 1e-9 and abs(mx.bend(0.5, 0.2) - 0.2) < 1e-9
      and mx.bend(-3, 0.2) == 0)

# ---- the engine: a shader pushed by another picture
t = st.translate("void mainImage(out vec4 c, in vec2 f){ c = vec4(kUV, f.x / iResolution.x, 1.0); }")
check("every shader carries the bend (off by default)", "kMixTex" in t.source and "kStart();" in t.source and "kBend" in st.FLOATS)
try:
    def run(static, tex=None):
        out = []
        sr.render(t.source, 64, 32, [{}], static, tex or {}, lambda i, px: out.append(px.copy()))
        return out[0]
    plain = run({})
    push = np.zeros((1, 1, 3), np.float32)
    push[..., 0], push[..., 1] = 1.0, 0.5                    # red 1 = all the way to the right, green 0.5 = not up or down
    moved = run({"kBend": 0.25}, {sr.MIX_UNIT: lambda i: ("p", push)})
    check("bend off: the shader draws where it is", abs(float(plain[5, 16, 0]) - 16.5 / 64) < 1e-3)
    check("bend on: red pushes sideways by kBend x the height, in kUV and in fragCoord, not up or down",
          abs(float(moved[5, 16, 0]) - (16.5 + 0.25 * 32) / 64) < 1e-3 and abs(float(moved[5, 16, 2]) - (16.5 + 8) / 64) < 1e-3
          and abs(float(moved[5, 16, 1]) - float(plain[5, 16, 1])) < 1e-3, str(moved[5, 16]))
    after = run({})
    check("the bend does not stick to the compiled shader", np.allclose(after, plain))

    # ---- the node
    A = "void mainImage(out vec4 c, in vec2 f){ c = vec4(1.0, 0.0, 0.0, 1.0); }"
    B = "void mainImage(out vec4 c, in vec2 f){ c = vec4(0.0, 0.0, 1.0, 1.0); }"
    ON = "void mainImage(out vec4 c, in vec2 f){ c = vec4(texture(iChannel0, kUV).rgb * vec3(0.0, 1.0, 0.0) + vec3(0.0, 0.5, 0.0), 1.0); }"
    kw = dict(code_a=A, code_b=B, width=32, height=16, seconds=0.4, fps=10.0)
    for m, want in ((0.0, (1, 0, 0)), (1.0, (0, 0, 1)), (0.25, (0.75, 0, 0.25))):
        img = nx.KUBA_ShaderMix.execute("rings", "rings", how="blend", mix=m, **kw).result[0]
        check(f"node, blend at {m}: {want}", img.shape == (4, 16, 32, 3) and torch.allclose(img[0, 3, 3], torch.tensor(want, dtype=torch.float32), atol=0.01), str(img[0, 3, 3]))
    out = nx.KUBA_ShaderMix.execute("rings", "rings", how="blend", mix=1.0, curve="ramp", cycle=0.4, **kw).result
    reds = [float(out[0][i, 0, 0, 0]) for i in range(4)]
    check("node: a ramp runs from A to B over its cycle", reds[0] > 0.99 and reds[0] > reds[1] > reds[2] > reds[3] and "mix: ramp" in out[5], str(reds))
    check("node: masks, fps, sound and count come out", out[1].shape == (4, 16, 32) and out[2] == 10.0 and out[4] == 4 and out[3]["waveform"].shape[-1] == 17640)
    on = nx.KUBA_ShaderMix.execute("rings", "rings", how="b works on a", mix=1.0, code_a=A, code_b=ON, width=32, height=16, seconds=0.1, fps=10.0).result
    check("node, b works on a: B reads A's picture", torch.allclose(on[0][0, 3, 3], torch.tensor((0.0, 0.5, 0.0)), atol=0.01), str(on[0][0, 3, 3]))
    flat = nx.KUBA_ShaderMix.execute("rings", "rings", how="b works on a", mix=0.5, **dict(kw, seconds=0.1)).result[5]
    check("node, b works on a with a B that reads no picture: the report says it is a plain fade", "plain fade" in flat, flat)
    pic = torch.zeros((1, 16, 32, 3))
    pic[..., 1] = 1.0
    onpic = nx.KUBA_ShaderMix.execute("rings", "rings", how="blend", mix=1.0, image=pic, code_a=A, code_b=ON, width=0, height=0, seconds=0.1, fps=10.0).result
    check("node: a connected image is what a shader works on, and gives the size", onpic[0].shape == (1, 16, 32, 3)
          and torch.allclose(onpic[0][0, 3, 3], torch.tensor((0.0, 1.0, 0.0)), atol=0.01), str(onpic[0][0, 3, 3]))
    UVX = "void mainImage(out vec4 c, in vec2 f){ c = vec4(vec3(kUV.x), 1.0); }"
    PUSH = "void mainImage(out vec4 c, in vec2 f){ c = vec4(1.0, 0.5, 0.0, 1.0); }"
    bent = nx.KUBA_ShaderMix.execute("rings", "rings", how="a bends b", mix=0.5, strength=0.25, code_a=PUSH, code_b=UVX, width=64, height=32, seconds=0.1, fps=10.0).result[0]
    want = 0.5 * 1.0 + 0.5 * (16.5 + 8) / 64                   # half A's red, half B drawn 8 pixels to the right
    check("node, a bends b: A pushes the places B draws at", abs(float(bent[0, 5, 16, 0]) - want) < 0.01, f"{float(bent[0, 5, 16, 0]):.3f} vs {want:.3f}")
    # ---- A and B as finished clips (two kubakub shader nodes on 'a' and 'b')
    schema = nx.KUBA_ShaderMix.define_schema()
    by = {i.id: i for i in schema.inputs}
    check("node: a, b and sound are the last inputs; the preset lists, controls and size are advanced",
          [i.id for i in schema.inputs][-3:] == ["a", "b", "sound"] and all(by[k].advanced for k in ("shader_a", "shader_b", "controls_a", "controls_b", "width", "height", "seconds", "loops"))
          and not by["how"].advanced and not by["mix"].advanced and by["how"].default == "a cuts b")
    red, blue = torch.zeros(3, 16, 32, 3), torch.zeros(2, 8, 16, 3)
    red[..., 0], blue[..., 2] = 1.0, 1.0
    two = nx.KUBA_ShaderMix.execute("rings", "rings", how="blend", mix=0.25, a=red, b=blue, width=640, height=360, seconds=9.0, fps=10.0).result
    check("node, two clips: A sets the size and the length, B is resized and plays again", two[0].shape == (3, 16, 32, 3)
          and torch.allclose(two[0][2, 3, 3], torch.tensor((0.75, 0.0, 0.25)), atol=0.01) and "resized to A's 32x16" in two[5]
          and "plays again" in two[5] and "A = the clip on 'a', B = the clip on 'b'" in two[5] and "controls of" not in two[5], two[5])
    half = torch.zeros(1, 16, 32, 3)
    half[:, :, 16:] = 1.0
    cutted = nx.KUBA_ShaderMix.execute("rings", "rings", how="a cuts b", mix=0.5, soft=0.01, a=half, b=blue[:1], fps=10.0).result[0]
    check("node, two clips, a cuts b: B shows first where A is bright", float(cutted[0, 8, 28, 2]) > 0.9 and float(cutted[0, 8, 4, 2]) < 0.1, f"{cutted[0, 8, 28]} {cutted[0, 8, 4]}")
    push = torch.zeros(1, 32, 64, 3)
    push[..., 0], push[..., 1] = 1.0, 0.5
    uvx = torch.linspace(0.5 / 64, 1.0 - 0.5 / 64, 64)[None, None, :, None].expand(1, 32, 64, 3).clone()
    bent2 = nx.KUBA_ShaderMix.execute("rings", "rings", how="a bends b", mix=0.5, strength=0.25, a=push, b=uvx, fps=10.0).result[0]
    check("node, two clips, a bends b: A pushes B's picture as it pushes a shader B", abs(float(bent2[0, 5, 16, 2]) - 0.5 * (16.5 + 8) / 64) < 0.01
          and abs(float(bent2[0, 5, 16, 0]) - want) < 0.01, f"{bent2[0, 5, 16]}")
    said = nx.KUBA_ShaderMix.execute("rings", "rings", how="b works on a", mix=0.5, a=red, b=blue, fps=10.0).result[5]
    check("node, two clips, b works on a: said to be a plain fade, with the cable that does it", "plain fade" in said and "image0" in said, said)
    mixed = nx.KUBA_ShaderMix.execute("rings", "rings", how="b works on a", mix=1.0, a=half.expand(1, 16, 32, 3) * torch.tensor((0.0, 1.0, 0.0)), code_b=ON, fps=10.0, seconds=0.1).result
    check("node, a clip on 'a' and a preset B: B works on A's picture", mixed[0].shape == (1, 16, 32, 3) and float(mixed[0][0, 8, 28, 1]) > 0.9
          and "controls of B" in mixed[5] and "controls of A" not in mixed[5], mixed[5])
    presets = nx.KUBA_ShaderMix.execute("plasma", "eight bit", width=64, height=32, seconds=0.1).result
    check("node, nothing connected: two presets mix", presets[0].shape[1:] == (32, 64, 3) and float(presets[0].max()) > 0.2
          and "A = 'plasma', B = 'eight bit'" in presets[5], presets[5])
    check("node: a shader that builds on its own last frame is named", "last frame" in raises(lambda: nx.KUBA_ShaderMix.execute("trails", "plasma", width=32, height=16, seconds=0.1)))
    check("node: a broken shader says which one", "shader B" in raises(lambda: nx.KUBA_ShaderMix.execute("plasma", "plasma", code_b="float x;", width=32, height=16, seconds=0.1)))
except Exception as e:  # noqa: BLE001
    if "graphics context" in str(e):
        print("skip the GPU part: no GL context here")
    else:
        raise

shutil.rmtree(TMP, ignore_errors=True)
print("\nALL PASSED" if not failures else f"\n{len(failures)} FAILED: {failures}")
sys.exit(1 if failures else 0)
