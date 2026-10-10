"""
Model free test of kubakub shader: the text side (kubakub/shadertoy.py: a Shadertoy or a fragment shader becomes one
GLSL ES 3.00 shader), time and sound per frame (kubakub/shader_render.py), every preset on the GPU and the node
(nodes/shader/nodes_shader.py). The GPU part is skipped with a note where ComfyUI's GL context does not start.

Run from the ComfyUI_windows_portable folder:
    python_embeded\\python.exe ComfyUI\\custom_nodes\\ComfyUI_KubaNodes\\tests\\test_shadertoy.py
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

TMP = tempfile.mkdtemp(prefix="kkd_shader_")
folder_paths.set_temp_directory(os.path.join(TMP, "temp"))

sys.path.insert(0, HERE)     # embedded Python leaves the script folder off the path
import _pack  # noqa: E402,F401  (the pack as a package, see tests/_pack.py)

import kubapack.nodes.shader.nodes_shader as ns  # noqa: E402
from kubakub import shader_render as sr  # noqa: E402
from kubakub import shadertoy as st  # noqa: E402
from kubakub import sound as so  # noqa: E402

failures = []


def check(name, cond, detail=""):
    print(("ok   " if cond else "FAIL ") + name + (f"  ({detail})" if detail and not cond else ""))
    if not cond:
        failures.append(name)


def raises(fn, *words):
    try:
        fn()
    except Exception as e:  # noqa: BLE001
        return all(w in str(e) for w in words), str(e)
    return False, "no error"


TOY = """// a comment that names iChannel3 and mainSound must not count
uniform float speed;   // 2 0 10  how fast
uniform float glow;    // 0.5
uniform float plain;
void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = fragCoord / iResolution.xy;
    vec3 c = texture2D(iChannel1, uv).rgb;
    fragColor = vec4(uv * (0.5 + 0.5 * sin(iTime * speed)), glow + kLow + plain, 1.0) + vec4(c, 0.0);
}
"""
GLSL = """#version 300 es
precision highp float;
uniform sampler2D u_image0;
uniform vec2 u_resolution;
uniform float amount; // 0.25 0 1
in vec2 v_texCoord;
layout(location = 0) out vec4 fragColor0;
void main() {
    fragColor0 = vec4(v_texCoord, amount, 0.5);
}
"""

# ---- text
t = st.translate(TOY)
check("a Shadertoy is recognised", t.kind == "shadertoy" and t.source.startswith("#version 300 es") and "void main()" in t.source)
check("controls: default, min, max from the comment",
      [(c.name, c.default, c.lo, c.hi) for c in t.controls] == [("speed", 2.0, 0.0, 10.0), ("glow", 0.5, 0.0, 1.0), ("plain", 0.0, 0.0, 1.0)],
      str(t.controls))
check("channels and names it reads (comments do not count)", t.channels == [1] and t.uses == {"kLow"}, f"{t.channels} {t.uses}")
check("texture2D becomes texture", "texture2D" not in t.source and "texture(iChannel1" in t.source)
check("line numbers are those of the pasted code", "#line 1\n// a comment" in t.source)
g = st.translate(GLSL)
check("a plain fragment shader keeps its main", g.kind == "glsl" and g.source.count("void main()") == 1
      and g.source.count("#version") == 1 and "out vec4 fragColor0" not in g.source and g.channels == [] and
      [c.name for c in g.controls] == ["amount"])
old = st.translate("varying vec2 v_texCoord;\nout vec4 colour;\nvoid main(){ colour = vec4(1.0); gl_FragColor.a = 0.5; }")
check("own output name and gl_FragColor map to the output", "#define colour kOut" in old.source and "kOut.a" in old.source)
vals, notes = st.parse_values("speed = 4, GLOW: .25\nnope = 1\nrubbish", t.controls)
check("controls text: values, case, what does not fit is said",
      vals == {"speed": 4.0, "glow": 0.25, "plain": 0.0} and len(notes) == 2 and "nope" in notes[0], f"{vals} {notes}")
check("empty shader", raises(lambda: st.translate("  // nothing\n"), "empty")[0])
check("neither mainImage nor main", raises(lambda: st.translate("float f() { return 1.0; }"), "mainImage", "Ctrl+A")[0])
check("a link is not code", raises(lambda: st.translate("https://www.shadertoy.com/view/XslGRr"), "link")[0])
two = TOY + "\nvoid mainImage(out vec4 a, in vec2 b) { a = vec4(0.0); }"
check("buffer tabs are named", raises(lambda: st.translate(two), "Buffer A-D")[0])
check("cubemap is named", raises(lambda: st.translate("uniform samplerCube c;\n" + TOY), "cubemap")[0])
check("sound shader is named", raises(lambda: st.translate(TOY + "\nvec2 mainSound(int s, float t){return vec2(0.0);}\nvoid mainSound(){}"), "mainSound")[0])
check("a compile log becomes short lines", st.explain("ERROR: 0:7: 'foo' : undeclared identifier\nERROR: 0:9: bad") ==
      "line 7: 'foo' : undeclared identifier\nline 9: bad")
names = list(st.presets())
check("presets are found, the first ones in their old order", len(names) >= 30 and names[:8] == ["mapping grid", "scan band", "warped noise",
      "caustics", "voronoi cells", "rings", "chromatic split", "glass"] and len(set(names)) == len(names), str(names))
check("every preset has an about line and translates", all(st.about(st.preset_code(n)) and st.translate(st.preset_code(n)).kind == "shadertoy" for n in names))

# ---- time and sound
ft = sr.frame_times(50, 25.0, 2.0, 1.0, extra=5)
check("frame times: clock, loop phase, the tail goes on", len(ft) == 55 and ft[0]["iTime"] == 1.0 and abs(ft[25]["iTime"] - 3.0) < 1e-9
      and ft[25]["kLoop"] == 0.5 and ft[50]["kLoop"] == 0.0 and ft[52]["iFrame"] == 52)
check("loops: the phase runs that often", sr.frame_times(40, 25.0, loops=4)[10]["kLoop"] == 0.0 and sr.frame_times(40, 25.0, loops=4)[5]["kLoop"] == 0.5)
mono = so.sample_sound(4.0, 120.0, 44100)
times = [i / 25.0 for i in range(100)]
heard, lines = sr.sound_values(mono, 44100, times, sample=True)
check("sound: a level per frame for every name", set(heard) == {"kLevel", "kLow", "kMid", "kHigh", "kBeat", "kBeats"}
      and all(len(v) == 100 and min(v) >= 0 and max(v) <= 1 or k == "kBeats" for k, v in heard.items()))
check("sound: the kick is on the beat, the hats are between", heard["kLow"][0] > 0.5 and heard["kLow"][8] < heard["kLow"][0]
      and heard["kHigh"][6] > heard["kHigh"][0], f"{heard['kLow'][:13]} {heard['kHigh'][:13]}")
check("sound: kBeat is 1 on a beat and falls, kBeats counts", heard["kBeat"][0] == 1.0 and abs(heard["kBeat"][6] - 0.52) < 0.02
      and heard["kBeats"][0] == 1.0 and heard["kBeats"][13] == 2.0 and "120" in lines[1], f"{heard['kBeat'][:14]} {lines}")

# ---- GPU
try:
    sr.core().GLContext()
    gpu = True
except Exception as e:  # noqa: BLE001
    gpu = False
    print(f"skip the GPU part: no GL context here ({e})")

if gpu:
    def run(code, frames, w=160, h=90, static=None, textures=None):
        tr = st.translate(code)
        vals = {c.name: c.default for c in tr.controls}
        vals.update(static or {})
        out = []
        sr.render(tr.source, w, h, frames, vals, textures, lambda i, px: out.append(px.copy()))
        return out

    px = run("out vec4 c;\nvoid main(){ c = vec4(gl_FragCoord.xy / iResolution.xy, iTime, kLoop); }", [{"iTime": 0.25, "kLoop": 0.5}], 64, 32)[0]
    check("a frame: size, top row first, clock and loop arrive", px.shape == (32, 64, 4) and px[0, 0, 1] > 0.9 and px[-1, 0, 1] < 0.1
          and px[0, -1, 0] > 0.9 and abs(px[0, 0, 2] - 0.25) < 1e-6 and abs(px[0, 0, 3] - 0.5) < 1e-6, str(px[0, 0]))
    pic = np.zeros((1, 4, 8, 3), np.float32)
    pic[0, 0, :, 0] = 1.0                                 # the top row is red
    px = run("out vec4 c;\nvoid main(){ c = vec4(texture(iChannel0, gl_FragCoord.xy / iResolution.xy).rgb, iChannelResolution[0].x / 8.0); }",
             [{}], 64, 32, textures={0: lambda i: (0, pic[0])})[0]
    check("a picture on iChannel0 is upright and its size is known", px[3, 5, 0] > 0.9 and px[-1, 5, 0] < 0.1 and abs(px[0, 0, 3] - 1.0) < 1e-6)
    px = run("void mainImage(out vec4 c, in vec2 f){ c = vec4(kField(kUV), kMask(kUV), 0.0, 1.0); }", [{}], 64, 32)[0]
    check("no field connected: a left to right ramp, the mask is 1", px[5, 1, 0] < 0.05 and px[5, -2, 0] > 0.95 and px[5, 5, 1] == 1.0)
    px = run(GLSL, [{}], 64, 32, static={"amount": 0.75})[0]
    check("a plain fragment shader runs, its control arrives", abs(px[0, 0, 2] - 0.75) < 1e-6 and abs(px[0, 0, 3] - 0.5) < 1e-6 and px[0, -1, 0] > 0.9)
    ok, msg = raises(lambda: run("void mainImage(out vec4 c, in vec2 f){\n c = vec4(nothing);\n}", [{}]), "does not compile", "line 2")
    check("a mistake names the line of the pasted code", ok, msg)
    worst, pictures, feedback = [], [], []
    for n in names:
        tr = st.translate(st.preset_code(n))
        if "kPrev" in tr.uses:                                                            # checked below: they build on their last frame
            feedback.append(n)
            continue
        if tr.channels:                                                                   # these work on a picture
            pictures.append(n)
        a, b, c = run(st.preset_code(n), [{"kLoop": 0.0, "iTime": 0.0}, {"kLoop": 1.0, "iTime": 4.0}, {"kLoop": 0.37, "iTime": 1.5}])
        lit = n in pictures or float(max(a.max(), c.max())) > 0.2
        loops = n == "mapping grid" or float(np.abs(a - b).max()) < 3e-3                  # its scan line ends on the other side
        moves = n in pictures or n == "sound bars" or float(np.abs(a - c).max()) > 0.02   # sound bars move with a sound
        if not (lit and loops and moves and np.isfinite(c).all()):
            worst.append(f"{n}: max {a.max():.3f}, loop gap {np.abs(a - b).max():.4f}, move {np.abs(a - c).max():.3f}")
    check(f"all {len(names)} presets draw, close their loop and move", len(names) >= 48 and not worst, "; ".join(worst))
    check("which presets work on a picture, which on their own last frame", {"contour lines", "wireframe", "point cloud", "eight bit", "characters"} <= set(pictures)
          and feedback == ["slit scan", "datamosh", "trails"] and ns.node_only() == feedback, f"{pictures} {feedback}")

    # ---- feedback: a shader that reads the frame before
    grow = "void mainImage(out vec4 c, in vec2 f){ c = vec4(kPrev(kUV).r + 0.25, float(iFrame) * 0.1, 0.0, 1.0); }"
    seen = [float(p[3, 3, 0]) for p in run(grow, [{"iFrame": i} for i in range(4)], 32, 16)]
    check("kPrev is the frame before (black at the first frame)", np.allclose(seen, [0.25, 0.5, 0.75, 1.0], atol=1e-5), str(seen))
    clip = torch.zeros((12, 36, 64, 3))
    for k in range(12):
        clip[k, :, :, 0] = k / 11.0                                                       # the clip gets redder frame by frame
    isl, _, _, _, nsl, rsl = ns.KUBA_Shader.execute("slit scan", controls="show = 0", image0=clip, loop=False).result
    col = isl[-1, 18, :, 0]
    check("slit scan: each column keeps the moment the slit passed it", nsl == 12 and float(col[2]) < float(col[30]) < float(col[56])
          and abs(float(col[56]) - 1.0) < 0.1 and float(col[63]) < 0.1, f"{[round(float(v), 2) for v in col[::8]]} {rsl}")
    idm, _, _, _, ndm, rdm = ns.KUBA_Shader.execute("datamosh", width=96, height=54, seconds=0.8).result
    check("datamosh and trails run on their own moving pattern with nothing connected, and loop through a fade",
          ndm == 20 and float(idm.max()) > 0.3 and "fades into the start" in rdm and "sample facade" not in rdm
          and float(ns.KUBA_Shader.execute("trails", width=96, height=54, seconds=0.4).result[0].mean()) > 0.02, rdm)
    check("controls: a name with a trailing _ in the code can be typed without it", st.parse_values("length = 0.5", [st.Control("length_", 0.3, 0, 1)])[0] == {"length_": 0.5})
    quiet = run(st.preset_code("sound bars"), [{"kLow": 0.0}])[0]
    loud = run(st.preset_code("sound bars"), [{"kLow": 1.0, "kMid": 0.8, "kHigh": 0.9, "kBeat": 1.0, "kBeats": 2.0}])[0]
    check("preset 'sound bars' grows with the sound", float(loud[..., :3].mean()) > 3 * float(quiet[..., :3].mean()))

    # ---- the built-in noise for a channel nothing is connected to
    nz = sr.noise_picture(7)
    gl_rows = nz[::-1]                                                                    # bottom row first, as a shader reads it
    check("noise: 256 x 256, in steps of 1 / 255, every grey about equally often", nz.shape == (256, 256, 4) and nz.dtype == np.float32
          and abs(float(nz[..., 0].mean()) - 0.5) < 0.01 and np.allclose(nz * 255, np.round(nz * 255), atol=1e-4))
    check("noise: green is red moved by (37, 17), the layout Shadertoys build 3D noise from",
          gl_rows[100 + 17, 50, 0] == gl_rows[100, 50 + 37, 1] and gl_rows[3, 250, 2] == gl_rows[3 - 17, (250 + 37) % 256, 3])
    wrap = "void mainImage(out vec4 c, in vec2 f){ vec2 uv = f / iResolution.xy; c = vec4(texture(iChannel0, uv).r, texture(iChannel0, uv + 3.0).r, 0.0, 1.0); }"
    px = run(wrap, [{}], 64, 64, textures={0: sr.noise_texture(7)})[0]
    check("noise: repeats at its edges (a shader can walk far across it)", float(np.abs(px[..., 0] - px[..., 1]).max()) < 2e-3 and float(px[..., 0].std()) > 0.1)

    # ---- the node
    out = ns.KUBA_Shader.execute("warped noise", width=160, height=96, seconds=0.4)
    images, masks, fps, audio, count, report = out.result
    check("node, nothing connected: a preset plays", images.shape == (10, 96, 160, 3) and masks.shape == (10, 96, 160) and count == 10
          and fps == 25.0 and float(images.max()) > 0.2 and audio["waveform"].shape[-1] == 17640 and "warped noise" in report
          and "brightness" in report and "scale = 3" in report, report)
    check("node: a preview is written", out.ui and os.path.isfile(os.path.join(folder_paths.get_temp_directory(), out.ui["images"][0]["filename"])))
    out = ns.KUBA_Shader.execute("warped noise", controls="scale = 9", width=64, height=32, seconds=0.2, loops=2)
    check("node: controls text is used", "scale = 9" in out.result[5])
    drift = "out vec4 c;\nvoid main(){ c = vec4(vec3(fract(iTime * 0.37)), 0.25 + 0.5 * gl_FragCoord.x / iResolution.x); }"
    i1, m1, _, _, n1, r1 = ns.KUBA_Shader.execute("rings", code=drift, width=64, height=32, seconds=1.0, loop=True, loop_fade=0.4).result
    i0 = ns.KUBA_Shader.execute("rings", code=drift, width=64, height=32, seconds=1.0, loop=False).result[0]
    step = float((i0[1] - i0[0]).abs().max())
    check("node, own shader with loop: the end fades into the start", n1 == 25 and "fades into the start over 0.40 s" in r1
          and float((i1[0] - i0[0]).abs().max()) > 0.01 and torch.allclose(i1[12], i0[12])
          and abs(float(i1[0, 0, 0, 0]) - (0.37 * 25 / 25) % 1.0) < 1e-4, r1)
    check("node: the shader's alpha is the mask when it writes one", "alpha" in r1 and float(m1[0, 5, 0]) < 0.3 and float(m1[0, 5, -1]) > 0.7 and step > 0)
    toy = "void mainImage(out vec4 c, in vec2 f){ c = vec4(vec3(f.x / iResolution.x), 0.0); }"
    it, mt, _, _, _, rt = ns.KUBA_Shader.execute("rings", code=toy, width=64, height=32, seconds=0.2, mask_from="alpha").result
    check("node: a Shadertoy's alpha is ignored, as on shadertoy.com (alpha 0 is still a picture and a mask)",
          "brightness" in rt and "has no alpha" in rt and float(mt[0, 5, -1]) > 0.9 and float(it[0, 5, -1, 0]) > 0.9, rt)
    cut = torch.zeros((1, 32, 64))
    cut[:, :, :32] = 1.0
    ramp = torch.linspace(0, 1, 64)[None, None].repeat(1, 32, 1)
    ic, mc, _, _, _, rc = ns.KUBA_Shader.execute("scan band", width=0, height=0, seconds=0.2, mask=cut, field=1.0 - ramp).result
    check("node: size from the mask, pictures and masks are cut to it", ic.shape == (5, 32, 64, 3) and float(ic[:, :, 32:].max()) == 0.0
          and float(mc[:, :, 32:].max()) == 0.0 and float(ic[:, :, :32].max()) > 0.2, rc)
    check("node: the mask of a cut preset is the band, not the cut", "brightness" in rc and float(mc[:, :, :32].min()) < 0.1 < float(mc[:, :, :32].max()), rc)
    _, ma, _, _, _, ra = ns.KUBA_Shader.execute("rings", code=drift, width=64, height=32, seconds=0.2, loop=False, mask=cut).result
    check("node: a shader's own alpha is cut to the mask too", "alpha" in ra and float(ma[:, :, 32:].max()) == 0.0 and 0.2 < float(ma[0, 5, 0]) < 0.3, ra)
    listen = "void mainImage(out vec4 c, in vec2 f){ c = vec4(kLow, kBeat, kBeats / 8.0, 1.0); }"
    il, _, _, al, nl, rl = ns.KUBA_Shader.execute("rings", code=listen, width=16, height=8, loop=False).result
    check("node: a shader that listens gets the built-in beat", nl == 100 and "built-in beat" in rl and float(il[0, 0, 0, 1]) == 1.0
          and float(il[0, 0, 0, 0]) > 0.5 and float(il[6, 0, 0, 1]) < 0.6 and al["waveform"].shape[-1] == 4 * 44100, rl)
    wave = torch.from_numpy(so.sample_sound(2.0, 100.0, 22050))[None, None]
    _, _, _, a2, n2, r2 = ns.KUBA_Shader.execute("rings", width=16, height=8, audio={"waveform": wave, "sample_rate": 22050}, bpm=100.0).result
    check("node: the length is the sound's, the tempo can be set", n2 == 50 and a2["sample_rate"] == 22050 and "100.0 bpm (set)" in r2, r2)
    img = torch.rand((3, 20, 40, 3))
    ig, _, _, _, ng, rg = ns.KUBA_Shader.execute("chromatic split", controls="amount = 0, wave = 0, sound = 0", image0=img).result
    check("node: a picture batch is the clip, one picture per frame", ng == 3 and ig.shape == (3, 20, 40, 3)
          and float((ig - img).abs().max()) < 2e-3, f"{rg} {float((ig - img).abs().max())}")
    rs = ns.KUBA_Shader.execute("glass", width=64, height=32, seconds=0.1).result[5]
    check("node: a preset that needs a picture gets the sample facade", "sample facade" in rs and "image1" not in rs.split("glass", 1)[-1].split("controls")[1], rs)
    ok, msg = raises(lambda: ns.KUBA_Shader.execute("rings", code="void mainImage(out vec4 c, in vec2 f){ c = vec4(oops); }", width=16, height=8),
                     "kubakub shader", "line 1")
    check("node: a mistake in the code is a short message", ok, msg)
    ok, msg = raises(lambda: ns.KUBA_Shader.execute("rings", code=two, width=16, height=8), "kubakub shader", "Buffer")
    check("node: a Shadertoy with buffer tabs says so", ok, msg)
    ok, msg = raises(lambda: ns.KUBA_Shader.execute("rings", code=" https://www.shadertoy.com/view/XslGRr\n", width=16, height=8), "link", "Ctrl+A")
    check("node: a pasted link says how to copy the code instead", ok, msg)
    ok, msg = raises(lambda: ns.KUBA_Shader.execute("rings", code="float f(float x) { return x * 2.0; }", width=16, height=8), "whole shader", "Ctrl+A")
    check("node: half a shader says what is missing and how to copy all of it", ok, msg)
    reads = "void mainImage(out vec4 c, in vec2 f){ c = vec4(texture(iChannel0, f / iResolution.xy).rgb, 1.0); }"
    i_n, _, _, _, _, r_n = ns.KUBA_Shader.execute("rings", code=reads, width=64, height=64, seconds=0.04, loop=False).result
    check("node: your code reading a picture with none connected gets the built-in noise, and the report says so",
          "built-in noise" in r_n and "image0" in r_n and float(i_n.std()) > 0.08, r_n)
    i_b, _, _, _, _, r_b = ns.KUBA_Shader.execute("rings", code=reads, width=64, height=64, seconds=0.04, loop=False, unconnected="black").result
    check("node: unconnected = black", float(i_b.max()) == 0.0 and "black" in r_b, r_b)
    r_s = ns.KUBA_Shader.execute("mosaic", width=64, height=32, seconds=0.04, unconnected="noise").result[5]
    check("node: unconnected = noise for a preset too", "built-in noise" in r_s and "sample facade" not in r_s, r_s)
    cube = "void mainImage(out vec4 c, in vec2 f){ c = texture(iChannel0, vec3(f, 1.0)); }"
    ok, msg = raises(lambda: ns.KUBA_Shader.execute("rings", code=cube, width=16, height=8), "cubemap or a 3D texture")
    check("node: a shader that wants a cubemap is explained", ok, msg)
    from PIL import Image
    import io as _io
    sheet = Image.open(_io.BytesIO(ns.thumbs_sheet()))
    rows = (len(names) + ns.COLUMNS - 1) // ns.COLUMNS
    tiles = np.asarray(sheet.convert("RGB"), np.float32).reshape(rows, ns.TILE[1], ns.COLUMNS, ns.TILE[0], 3).max(axis=(1, 3, 4))
    check("pictures: one tile per preset on the sheet, none of them empty", sheet.size == (ns.COLUMNS * ns.TILE[0], rows * ns.TILE[1])
          and int((tiles.reshape(-1)[:len(names)] > 60).sum()) == len(names) and ns.thumbs_sheet() is ns.thumbs_sheet(), str(tiles.astype(int)))
    pick = open(os.path.join(os.path.dirname(HERE), "web", "kubakub_shader.js"), encoding="utf-8").read()
    check("pictures: the picker asks the node for its sheet and sets the preset widget", all(x in pick for x in
          ("/kubakub/shader/thumbs", "/kubakub/shader/presets", 'w.name === "preset"', "export async function pickShader")))

    # ---- the director: a shape layer filled by a shader
    from kubakub.director import render as rd
    base = np.zeros((90, 160, 3), np.float32)
    flat = "uniform float amount; // 0.25 0 1\nvoid mainImage(out vec4 c, in vec2 f){ c = vec4(kLoop, f.y / iResolution.y, amount, 1.0); }"
    shape = {"type": "rect", "points": [], "feather": 0, "shader": "", "code": flat, "loops": 2, "speed": 1, "c_amount": 0.75}
    doc = {"version": 1, "timeline": {"duration": 4, "fps": 25, "time": 0},
           "layers": [{"id": "S", "name": "fill", "kind": "shape", "x": 40, "y": 20, "w": 80, "h": 50, "color": "#ff0000", "shape": shape},
                      {"id": "base", "name": "base", "kind": "base"}]}
    sh = rd.parse(doc)["layers"][0]["shape"]
    check("director: a shape keeps its shader, loops and controls", sh["code"] == flat and sh["loops"] == 2 and sh["c_amount"] == 0.75 and sh["now"] is None)
    check("director: a plain shape has no shader keys", "shader" not in rd.parse({"version": 1, "layers": [dict(doc["layers"][0], shape={"type": "rect"}), doc["layers"][1]]})["layers"][0]["shape"])
    fr = rd.animate(doc, 1.0)
    now = fr["layers"][0]["shape"]["now"]
    check("director: a frame carries the shader's clock and loop", abs(now["kLoop"] - 0.5) < 1e-9 and now["iTime"] == 1.0 and now["kSeconds"] == 4.0, str(now))
    img = rd.render(fr, base, {}, image_only=True)["image"]
    check("director: the shader shows in the layer's box, the right way up, with its control",
          abs(float(img[45, 80, 0]) - 0.5) < 0.01 and abs(float(img[45, 80, 2]) - 0.75) < 0.01 and float(img[22, 80, 1]) > 0.9
          and float(img[68, 80, 1]) < 0.1 and float(img[10, 80].max()) == 0.0 and float(img[45, 20].max()) == 0.0, f"{img[45, 80]} {img[22, 80]} {img[68, 80]}")
    ell = dict(doc, layers=[dict(doc["layers"][0], shape=dict(shape, type="ellipse", feather=6)), doc["layers"][1]])
    ie = rd.render(rd.animate(ell, 1.0), base, {}, image_only=True)["image"]
    check("director: the shader is seen through the shape (ellipse, feather), nothing outside the box",
          float(ie[45, 80, 0]) > 0.45 and float(ie[22, 42].max()) < 0.08 and 0.05 < float(ie[45, 42, 0]) < 0.45 and float(ie[45, 36].max()) == 0.0, f"{ie[45, 80]} {ie[45, 42]} corner {ie[22, 42]} outside {ie[45, 36]} {ie[45, 39]}")
    bad = dict(doc, layers=[dict(doc["layers"][0], shape=dict(shape, code="void mainImage(out vec4 c, in vec2 f){ c = oops; }")), doc["layers"][1]])
    rb = rd.render(rd.animate(bad, 0.0), base, {}, image_only=True)
    check("director: a broken shader costs no frame, the colour shows and a note says why",
          abs(float(rb["image"][45, 80, 0]) - 1.0) < 0.01 and any("shader not drawn" in n for n in rb["notes"]), str(rb["notes"]))
    pre = dict(doc, layers=[dict(doc["layers"][0], shape={"type": "rect", "shader": "rings", "c_rings": 3}), doc["layers"][1]])
    ip = rd.render(pre, base, {}, image_only=True)["image"]
    check("director: a preset fill renders without a frame's clock (a still)", float(ip[20:70, 40:120].max()) > 0.3)
    # ---- a shader fill that works on the picture: in the director that picture is the layers below it
    from kubakub import sample_facade as _sf
    pic = _sf.render(160, 90)["image"]
    fx_doc = {"version": 1, "timeline": {"duration": 4, "fps": 25, "time": 0},
              "layers": [{"id": "S", "name": "fx", "kind": "shape", "x": 0, "y": 0, "w": 160, "h": 90, "shape": {"type": "rect", "shader": "mosaic", "c_tiles": 16, "c_breathe": 0, "c_gap": 0}},
                         {"id": "base", "name": "base", "kind": "base"}]}
    check("director: a preset that works on a picture reads the layers below; your own code reads the noise unless told otherwise",
          rd.parse(fx_doc)["layers"][0]["shape"]["input"] == "below" and sh["input"] == "noise" and rd.reads_below(rd.parse(fx_doc)["layers"][0])
          and not rd.reads_below(rd.parse(doc)["layers"][0]))
    mo_ = rd.render(rd.animate(fx_doc, 0.0), pic, {}, image_only=True)["image"]
    check("director: mosaic over the base is the base in tiles", float(np.corrcoef(mo_.mean(2).ravel(), pic.mean(2).ravel())[0, 1]) > 0.6
          and float(np.abs(mo_ - pic).mean()) > 0.01, f"corr {np.corrcoef(mo_.mean(2).ravel(), pic.mean(2).ravel())[0, 1]:.3f}")
    turn = "void mainImage(out vec4 c, in vec2 f){ c = vec4(1.0 - texture(iChannel0, f / iResolution.xy).rgb, 1.0); }"
    neg = {"type": "rect", "code": turn, "input": "below"}
    full = dict(fx_doc, layers=[dict(fx_doc["layers"][0], shape=neg), fx_doc["layers"][1]])
    fu = rd.render(rd.animate(full, 0.0), pic, {}, image_only=True)["image"]
    check("director: your own code with input = below gets the picture under it, pixel for pixel", float(np.abs(fu - (1.0 - pic)).max()) < 2e-3,
          f"{np.abs(fu - (1.0 - pic)).max():.4f}")
    half = dict(fx_doc, layers=[dict(fx_doc["layers"][0], x=80, w=80, shape=neg), fx_doc["layers"][1]])
    hf = rd.render(rd.animate(half, 0.0), pic, {}, image_only=True)["image"]
    check("director: the effect covers its box only, and reads the part of the picture under that box",
          float(np.abs(hf[:, :79] - pic[:, :79]).max()) < 1e-4 and float(np.abs(hf[:, 81:] - (1.0 - pic[:, 81:])).max()) < 5e-3,
          f"{np.abs(hf[:, :79] - pic[:, :79]).max():.5f} {np.abs(hf[:, 81:] - (1.0 - pic[:, 81:])).max():.5f}")
    two_up = dict(fx_doc, layers=[dict(fx_doc["layers"][0], shape=neg), {"id": "C", "name": "red", "kind": "shape", "x": 0, "y": 0, "w": 160, "h": 90,
                                                                         "color": "#ff0000", "shape": {"type": "rect"}}, fx_doc["layers"][1]])
    tu = rd.render(rd.animate(two_up, 0.0), pic, {}, image_only=True)["image"]
    check("director: it sees every layer under it, not only the base (red below, so cyan)", float(tu[..., 0].max()) < 0.01 and float(tu[..., 1:].min()) > 0.99)
    soft = dict(fx_doc, layers=[dict(fx_doc["layers"][0], x=40, y=20, w=80, h=50, opacity=0.5, shape=dict(neg, type="ellipse", feather=4)), fx_doc["layers"][1]])
    so_ = rd.render(rd.animate(soft, 0.0), pic, {}, image_only=True)["image"]
    check("director: shape, feather and opacity work on an effect as on any shape", float(np.abs(so_[45, 80] - 0.5).max()) < 0.02
          and float(np.abs(so_[5, 5] - pic[5, 5]).max()) < 1e-4, f"{so_[45, 80]}")
    gl_doc = dict(fx_doc, layers=[dict(fx_doc["layers"][0], shape={"type": "rect", "shader": "glass"}), fx_doc["layers"][1]])
    check("director: a picture a preset can do without is left empty (glass makes its own ripples)", rd.render(rd.animate(gl_doc, 0.0), pic, {}, image_only=True)["notes"] == []
          and st.translated(st.preset_code("glass")).optional == {1})
    import kubapack.nodes.director.nodes_director as nd
    with_fill = lambda name: dict(doc, layers=[dict(doc["layers"][0], shape={"type": "rect", "shader": name}), doc["layers"][1]])   # noqa: E731
    check("director: a shader fill that listens asks for the sound's level and bands, another one does not",
          nd.motion_needs(with_fill("sound bars")) == (True, True) and nd.motion_needs(with_fill("mapping grid")) == (False, False)
          and nd.motion_needs(dict(doc, layers=[doc["layers"][1]])) == (False, False))
    info = ns.shader_info("caustics")
    check("window: the shader text and its controls for WebGL", info["source"].startswith("#version 300 es") and
          [c["name"] for c in info["controls"]][:2] == ["scale", "sharp"] and "error" in ns.shader_info("", two))
    js = open(os.path.join(os.path.dirname(HERE), "web", "kubakub_director.js"), encoding="utf-8").read()
    check("window: the shader fill is wired (same names as the node)", all(s in js for s in
          ("/kubakub/shader/translate", "kFieldTex", "shaderFill(L)", '"c_" + d.shc', "kLoop: (t / D * Math.max(1, Math.round(+sh.loops || 1))) % 1",
           "byte(x - 37, y + 17, seed)", "gl.REPEAT", "pickShader(e.currentTarget")))
    check("window: a fill reads the layers below, as the render does", all(x in js for x in
          ("this._below = solo ? null : target", "shaderInput(L)", "UNPACK_FLIP_Y_WEBGL", '"iChannelResolution[0]"', 'data-id="sh-input"',
           "(info.optional || []).includes(i)")))
    check("window: review fixes (sliders follow the playhead, controls can be keyed, old keys go with the old fill, mistakes are shown)",
          all(x in js for x in ("syncShader()", "shaderDefaults(L, info)", 'p.startsWith("shape.c_")) delete q.anim[p]', "info.glError", "c.about || c.name")))
    cs = st.parse_controls("uniform float a; // 0.5 0 1  0 = hard edges, 1 = soft\nuniform float b; // 2 wide lines\nuniform float c;")
    check("controls: the comment after the numbers is what the control does", [(c.default, c.lo, c.hi, c.about) for c in cs] ==
          [(0.5, 0.0, 1.0, "0 = hard edges, 1 = soft"), (2.0, 0.0, 2.0, "wide lines"), (0.0, 0.0, 1.0, "")], str(cs))
    check("pictures: the node says what it will draw before a queue", all(x in pick for x in ('"draws"', "cannot run: ", "the preset is not used", "Hanken Grotesk KKD",
          "drawing the pictures")) and ns.shader_info("", "https://www.shadertoy.com/view/XslGRr")["error"].startswith("this is a link"))
    reads_doc = dict(doc, layers=[dict(doc["layers"][0], shape=dict(shape, code=reads)), doc["layers"][1]])
    i_d = rd.render(rd.animate(reads_doc, 0.0), base, {}, image_only=True)["image"]
    check("director: a shader that reads a picture gets the built-in noise (as the window)", float(i_d[25:65, 45:115].std()) > 0.1)

shutil.rmtree(TMP, ignore_errors=True)
print()
print("ALL PASSED" if not failures else f"{len(failures)} FAILED: {failures}")
sys.exit(1 if failures else 0)
