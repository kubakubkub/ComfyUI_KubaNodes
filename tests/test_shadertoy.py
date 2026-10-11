"""
Model free test of kubakub shader: the text side (kubakub/shadertoy.py: a Shadertoy or a fragment shader becomes one
GLSL ES 3.00 shader), time and sound per frame (kubakub/shader_render.py), every preset on the GPU and the node
(nodes/shader/nodes_shader.py). The GPU part is skipped with a note where ComfyUI's GL context does not start.

Run from the ComfyUI_windows_portable folder:
    python_embeded\\python.exe ComfyUI\\custom_nodes\\ComfyUI_KubaNodes\\tests\\test_shadertoy.py
"""

import os
import re
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
st.USER_DIR = tempfile.mkdtemp(prefix="kuba_my_shaders_")      # saved presets go here, not into the real user folder
names = list(st.presets())
check("presets are found, sorted by type and by file within a type", len(names) >= 30 and names[:6] == ["warped noise", "caustics", "voronoi cells",
      "rings", "clouds", "fire"] and len(set(names)) == len(names), str(names))
kinds = [st.preset_type(st.preset_code(n)) for n in names]
keys = [k for k, _, _ in st.TYPES if k != "mine"]             # 'my presets' is empty until one is saved
check("every preset names its type, and every type has presets", all(re.search(r"^// type: \w+$", st.preset_code(n), flags=re.M) for n in names)
      and sorted(set(kinds), key=keys.index) == keys and kinds == sorted(kinds, key=keys.index)
      and st.preset_type("// type: nonsense") == "pattern" and st.preset_type("void x;") == "pattern", str(sorted(set(kinds))))
by = dict(zip(names, kinds))

# ---- colours to pick: a vec3 with a hex colour in its comment
INK = "uniform vec3 ink;   // #ff8000 the line\nuniform float thick; // 1 0 4\nuniform vec3 plain;\nvoid mainImage(out vec4 c, in vec2 f){ c = vec4(ink * thick + plain, 1.0); }"
ti = st.translate(INK)
check("colour: a vec3 with a hex comment is three controls and one name in the shader", [(c.name, round(c.default, 3), c.colour) for c in ti.controls]
      == [("ink_r", 1.0, "ink"), ("ink_g", 0.502, "ink"), ("ink_b", 0.0, "ink"), ("thick", 1.0, "")] and "#define ink vec3(ink_r, ink_g, ink_b)" in ti.source
      and "uniform float ink_r; uniform float ink_g; uniform float ink_b;" in ti.source and "uniform vec3 plain;" in ti.source
      and ti.source.split("#line 1\n")[1].count("\n") == st.translate(INK.replace("// #ff8000 the line", "")).source.split("#line 1\n")[1].count("\n"))
_cv, _cn = st.parse_values("ink = #0080FF, thick = 2  # a note\nplain = #ffffff", ti.controls)
check("colour: 'ink = #0080ff' in controls sets the three, a comment after # still works", (_cv["ink_r"], round(_cv["ink_g"], 3), _cv["ink_b"], _cv["thick"]) == (0.0, 0.502, 1.0, 2.0)
      and st.colour_hex(_cv, "ink") == "#0080ff" and len(_cn) == 1 and "plain" in _cn[0], str(_cn))
_cw = st.with_defaults(INK, _cv)
check("colour: saved defaults are written back as a hex colour", "uniform vec3 ink;      // #0080ff   the line" in _cw and st.parse_controls(_cw)[2].default == 1.0
      and st.with_defaults(INK, {"thick": 3}) .count("#ff8000") == 1)
_house = [n for n in names if any(c.colour == "colour_a" for c in st.translate(st.preset_code(n)).controls)]
check("colour: the presets' two house colours can be picked", len(_house) >= 35 and "plasma" in _house and "melting forms" in _house
      and all(st.colour_hex({c.name: c.default for c in st.translate(st.preset_code(n)).controls}, "colour_a") == "#a187b7" for n in _house), str(len(_house)))

# ---- presets of your own: the sliders become the defaults of a saved file
_wd = st.with_defaults("uniform float warp;   // 1 0 4   the folds\nuniform float plain;\nuniform float iTime;\nvoid x;", {"warp": 2.5, "plain": 9, "nope": 1})
check("with_defaults: the values become the defaults, ranges and words stay", "uniform float warp;      // 2.5 0 4   the folds" in _wd
      and [(c.name, c.default, c.lo, c.hi) for c in st.parse_controls(_wd)] == [("warp", 2.5, 0.0, 4.0), ("plain", 1.0, 0.0, 1.0)] and "void x;" in _wd)
_saved = st.save_preset("  My  Smoke_2 ", st.preset_code("warped noise"), {"warp": 3.5})
_mine = list(st.presets())
check("save: a preset of my own is listed first, under 'my presets', with the sliders as its defaults", _saved == "my smoke 2" and _mine[0] == "my smoke 2"
      and st.type_of("my smoke 2") == "mine" and st.is_mine("my smoke 2") and not st.is_mine("warped noise") and st.type_of("warped noise") == "pattern"
      and {c.name: c.default for c in st.parse_controls(st.preset_code("my smoke 2"))}["warp"] == 3.5 and "// type:" not in st.preset_code("my smoke 2")
      and st.about(st.preset_code("my smoke 2")) == st.about(st.preset_code("warped noise")) and os.listdir(st.USER_DIR) == ["my_smoke_2.glsl"], str(_mine[:3]))
check("save: a built-in name, a bad name, a path and a shader that cannot run are refused", raises(lambda: st.save_preset("plasma", TOY), "built-in")[0]
      and raises(lambda: st.save_preset("../x", TOY), "letters")[0] and raises(lambda: st.save_preset("9 lives", TOY), "letters")[0]
      and raises(lambda: st.save_preset("x" * 41, TOY), "letters")[0] and raises(lambda: st.save_preset("broken", "float f() { return 1.0; }"), "")[0]
      and os.listdir(st.USER_DIR) == ["my_smoke_2.glsl"])
st.save_preset("my smoke 2", TOY)
check("save: my own preset can be saved again, and gets an about line", "mainImage" in st.preset_code("my smoke 2") and st.about(st.preset_code("my smoke 2")).startswith("my smoke 2")
      and ns.shader_info("my smoke 2")["mine"] and ns.shader_info("my smoke 2")["type"] == "mine" and ns.shader_info("plasma", controls="scale = 3")["values"]["scale"] == 3.0
      and ns.shader_info("plasma")["text"] == st.preset_code("plasma") and ns.shader_info("", TOY)["text"] == "")
_fp = ns.KUBA_Shader.fingerprint_inputs(preset="my smoke 2")
st.save_preset("my smoke 2", TOY.replace("0.5", "0.25") if "0.5" in TOY else TOY + "\n// again")
check("a preset saved again under its name runs again (the node's fingerprint is its text)", _fp and ns.KUBA_Shader.fingerprint_inputs(preset="my smoke 2") != _fp
      and ns.KUBA_Shader.fingerprint_inputs(preset="plasma") == ns.KUBA_Shader.fingerprint_inputs(preset="plasma")
      and ns.KUBA_Shader.fingerprint_inputs(preset="plasma", code=TOY) == "" and ns.KUBA_Shader.fingerprint_inputs(preset="no such") == "")
st.delete_preset("my smoke 2")
check("delete: mine goes, a built-in stays", "my smoke 2" not in st.presets() and os.listdir(st.USER_DIR) == [] and raises(lambda: st.delete_preset("plasma"), "not one of your")[0]
      and list(st.presets()) == names)
check("the types hold what their titles say", by["melting forms"] == "space" and by["vhs tape"] == "signal" and by["cartoon"] == "look"
      and by["extrude"] == "regions" and by["facade relief"] == "relief" and by["growth"] == "time" and by["plasma"] == "pattern"
      and [n for n in names if by[n] == "lens"] == ["blur", "motion blur", "zoom blur", "grade", "duotone", "vignette", "lens bend", "swirl", "under water",
                                                    "little planet", "glow", "mirrors"])
check("every preset has an about line and translates", all(st.about(st.preset_code(n)) and st.translate(st.preset_code(n)).kind == "shadertoy" for n in names))

# ---- 3D: distance functions and a raymarcher for a shader that uses their names
SCENE = """float map(vec3 p) { return opSmoothUnion(sdSphere(p, 0.8), p.z + 1.0, 0.2); }
void mainImage(out vec4 c, in vec2 f) {
    vec3 ro = vec3(0.0, 0.0, 3.0), rd = kCamera(f, ro, vec3(0.0), 2.0);
    float t = kMarch(ro, rd, 10.0, 64);
    kDepth = kDepthOf(t, 2.0, 4.2);
    c = vec4(t > 0.0 ? 0.5 + 0.5 * kNormal(ro + rd * t) : vec3(0.0), 1.0);
}
"""
sc = st.translate(SCENE)
head = sc.source.split("#line 1")[0]
check("3D: a shader gets the helpers it names, and only those", all(f" {n}(" in head for n in ("sdSphere", "opSmoothUnion", "kCamera", "kMarch", "kNormal", "kDepthOf"))
      and "float map(vec3 p);" in head and "sdBox" not in head and "kShadow" not in head and sc.depth and "kDepth" in sc.uses)
check("3D: a shader that names none gets none", "sdSphere" not in t.source and "float map(vec3 p);" not in t.source and not t.depth)
mine = st.translate("float sdBox(vec3 p, vec3 b) { return length(p) - b.x; }\n#define opRep(p, c) (mod(p, c))\n"
                    "void mainImage(out vec4 c, in vec2 f){ c = vec4(sdBox(opRep(vec3(f, 0.0), vec3(1.0)), vec3(0.5)) + sdTorus(vec3(f, 0.0), vec2(1.0, 0.2))); }")
check("3D: a pasted shader keeps the functions it brings itself", mine.source.count("float sdBox(") == 1 and "vec3 opRep(" not in mine.source
      and mine.source.count("float sdTorus(") == 1)
check("3D: the march without a scene says what is missing", raises(lambda: st.translate("void mainImage(out vec4 c, in vec2 f){ c = vec4(kMarch(vec3(0.0), vec3(0.0, 0.0, 1.0), 9.0, 8)); }"),
                                                               "kMarch", "float map(vec3 p)")[0])
check("3D: line numbers stay those of the pasted code", "#line 1\nfloat map(vec3 p) {" in sc.source)
SOLID = ("melting forms", "colonnade", "facade relief", "extrude", "growth")
check("3D: the presets that are a scene write a depth, the others do not", sorted(n for n in names if st.translate(st.preset_code(n)).depth) == sorted(SOLID),
      str([n for n in names if st.translate(st.preset_code(n)).depth]))
check("3D: extrude asks for the sample field, no other preset does", [n for n in names if st.sample_field(st.preset_code(n))] == ["extrude"])

# ---- review 2026-10-11: text that used to trip the translation
cm = st.translate("out vec4 o;\n// void main() is below\n/* old: void main() {} */\nvoid main(){ o = vec4(1.0); }")
check("review: a 'void main(' in a comment is left alone, the real one is renamed", cm.source.count("void kUserMain(") == 1 and "// void main() is below" in cm.source)
ext = st.translate("#extension GL_OES_standard_derivatives : enable\nuniform highp float iTime;\nvoid mainImage(out vec4 c, in vec2 f){ c = vec4(fwidth(f.x) + iTime); }")
check("review: #extension lines and built-in uniforms with a precision are dropped", "#extension" not in ext.source and ext.source.count("float iTime;") == 1)
check("review: map(const in vec3 p) is a scene, map(vec3, float) is not", "float map(vec3 p);" in st.translate(
      "float map(const in vec3 p) { return length(p) - 1.0; }\nvoid mainImage(out vec4 c, in vec2 f){ c = vec4(kMarch(vec3(0.0), vec3(0.0, 0.0, 1.0), 9.0, 8)); }").source
      and raises(lambda: st.translate("float map(vec3 p, float t) { return 1.0; }\nvoid mainImage(out vec4 c, in vec2 f){ c = vec4(kMarch(vec3(0.0), vec3(0.0, 0.0, 1.0), 9.0, 8)); }"), "float map(vec3 p)")[0])
import time as _time  # noqa: E402
_t0 = _time.perf_counter()
_nc = st._no_comments("/* " * 100000 + "\nvoid x; // y\n")
check("review: comments are found in one walk (a long broken text does not hold up the server), and a text has a limit",
      _time.perf_counter() - _t0 < 1.0 and st._no_comments("a // b\n/* c\nd */ e") == "a     \n    \n     e" and raises(lambda: st.translate("x" * (st.MAX_CODE + 1)), "limit")[0])
check("review: a compile message with a line break in it stays one line", st.explain("ERROR: 0:1: '\n' : extension directive must occur\nERROR: 0:9: bad")
      == "line 1: ' ' : extension directive must occur\nline 9: bad")

# ---- a Shadertoy with Buffer tabs
COUNT = """float gain() { return 0.25; }
// ---- Buffer A ----
void mainImage(out vec4 c, in vec2 f) { c = vec4(texture(iChannel0, f / iResolution.xy).r + gain(), 0.0, 0.0, 1.0); }
// Buffer B: 0 = A, 1 = noise
void mainImage(out vec4 c, in vec2 f) { c = vec4(texture(iChannel0, f / iResolution.xy).r * 2.0, texture(iChannel1, f / 256.0).r, 0.0, 1.0); }
// Image
void mainImage(out vec4 c, in vec2 f) {
    vec2 uv = f / iResolution.xy;
    c = vec4(texture(iChannel0, uv).r / 4.0, texture(iChannel1, uv).r / 8.0, texture(iChannel2, uv).r, 1.0);
}
"""
tb = st.translate(COUNT)
check("tabs: every Buffer tab is a shader of its own, with the text above the first tab in each",
      [b.name for b in tb.buffers] == ["A", "B"] and all("float gain()" in b.source and b.source.count("void mainImage") == 1 for b in tb.buffers)
      and tb.source.count("void mainImage") == 1 and "kFinish" in tb.source and "kFinish();\n}" not in tb.buffers[0].source.split("#line")[-1])
check("tabs: a channel reads the buffer of its letter, the tab's line can say otherwise, the rest are the node's pictures",
      tb.buffers[0].inputs == {0: "A"} and tb.buffers[1].inputs == {0: "A", 1: "noise"} and tb.inputs == {0: "A", 1: "B", 2: "image2"}
      and tb.channels == [2] and "Buffer B reads iChannel0 = Buffer A, iChannel1 = noise" in tb.notes[0], f"{tb.inputs} {tb.channels} {tb.notes}")
check("tabs: line numbers are those of the pasted text", "#line 3\nvoid mainImage(out vec4 c, in vec2 f) { c = vec4(texture(iChannel0, f / iResolution.xy).r + gain()" in tb.buffers[0].source
      and "#line 1\nfloat gain()" in tb.source and "#line 7\n" in tb.source)
check("tabs: only a line that is a tab's name counts", st._tab("// image plane distance") is None and st._tab("// Buffer A is the simulation") is None
      and st._tab("// Image") == ("image", {}) and st._tab("// === buffer b: iChannel0 = Buffer A, 2=image1 ===") == ("B", {0: "A", 2: "image1"})
      and st.tabs("// Common\nfloat f(){return 1.0;}\n// Image\nvoid mainImage(out vec4 c, in vec2 f){c=vec4(1.0);}") is None)
check("tabs: what is missing is said", raises(lambda: st.translate("// Buffer A\nvoid mainImage(out vec4 c, in vec2 f){c=vec4(1.0);}"), "no Image tab")[0]
      and raises(lambda: st.translate("// Buffer A: 0 = C\nvoid mainImage(out vec4 c, in vec2 f){c=texture(iChannel0, f);}\n// Image\nvoid mainImage(out vec4 c, in vec2 f){c=vec4(1.0);}"), "no tab '// Buffer C'")[0]
      and raises(lambda: st.translate(two), "// Buffer A", "// Image")[0])
check("tabs: a preset can have them (growth), and such a shader builds on its last frames", [b.name for b in st.translate(st.preset_code("growth")).buffers] == ["A", "B", "C", "D"])

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
        if "kPrev" in tr.uses or tr.buffers:                                              # checked below: they build on their last frames
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
          and feedback == ["slit scan", "datamosh", "trails", "growth"] and ns.node_only() == feedback, f"{pictures} {feedback}")
    # ---- a picture under a shader that does not read it
    _pic = torch.zeros((1, 48, 64, 3)); _pic[..., 0] = 0.5; _pic[:, :, 32:, 1] = 1.0
    _run = lambda **kw: ns.KUBA_Shader.execute("stripes", width=64, height=48, fps=5.0, seconds=0.4, **kw).result      # noqa: E731
    _alone = _run()
    _scr, _off, _mul, _half = _run(image0=_pic), _run(image0=_pic, on_picture="off"), _run(image0=_pic, on_picture="multiply"), _run(image0=_pic, picture_amount=0.5)
    _p, _s = _pic.expand(2, -1, -1, -1), _alone[0]
    check("on_picture: a pattern is laid over the picture on image0 (screen), off leaves it alone, masks stay the shader's",
          torch.allclose(_scr[0], 1 - (1 - _p) * (1 - _s), atol=1e-5) and torch.equal(_off[0], _s) and torch.equal(_scr[1], _alone[1])
          and torch.allclose(_mul[0], _p * _s, atol=1e-5) and torch.allclose(_half[0], _p + (_scr[0] - _p) * 0.5, atol=1e-5)
          and "laid over your picture" in _scr[5] and "only sets the size" in _off[5] and "image0" not in _alone[5])
    _m = torch.zeros((1, 48, 64)); _m[:, :, :20] = 1.0
    _mk = _run(image0=_pic, mask=_m, on_picture="multiply")
    check("on_picture: outside the mask the picture stays as it is", torch.allclose(_mk[0][:, :, 20:], _p[:, :, 20:], atol=1e-5)
          and torch.allclose(_mk[0][:, :, :20], (_p * _s)[:, :, :20], atol=1e-5))
    for _how in ns.ON_PICTURE:
        _o = _run(image0=_pic, on_picture=_how)[0]
        check(f"on_picture = {_how}: a whole picture in 0-1", bool(torch.isfinite(_o).all()) and float(_o.min()) >= 0.0 and float(_o.max()) <= 1.0 and _o.shape == _s.shape)
    _car = ns.KUBA_Shader.execute("cartoon", width=64, height=48, fps=5.0, seconds=0.4, image0=_pic).result
    check("on_picture: a preset that reads the picture is not laid over it", "laid over" not in _car[5]
          and torch.equal(_car[0], ns.KUBA_Shader.execute("cartoon", width=64, height=48, fps=5.0, seconds=0.4, image0=_pic, on_picture="off").result[0]))
    _big = ns.KUBA_Shader.execute("melting forms", width=64, height=48, fps=5.0, seconds=0.4, image0=torch.rand((1, 24, 32, 3))).result
    check("on_picture: a 3D scene over a picture of another size keeps its depth", _big[0].shape == (2, 48, 64, 3) and float(_big[6].max()) > 0.2 and "and depth are the shader's own" in _big[5])

    _snd = {"waveform": torch.sin(torch.linspace(0, 400, 8000))[None, None] * 0.8, "sample_rate": 8000}
    _deaf, _hear = _run(), _run(audio=_snd)
    _typed = _run(audio=_snd, controls="beat = 0")
    check("a connected sound opens a preset's beat / sound control, unless a value is typed",
          "a sound is connected: beat = 1" in _hear[5] and "a sound is connected" not in _typed[5] and "a sound is connected" not in _deaf[5]
          and "beat = 1 " in _hear[5].split("controls:")[1].splitlines()[0] and "beat = 0 " in _typed[5].split("controls:")[1].splitlines()[0])
    _idle = ns.KUBA_Shader.execute("clouds", width=64, height=48, fps=5.0, seconds=0.4, image1=_pic, field=torch.rand((1, 48, 64)), audio=_snd).result[5]
    check("an input the shader has no use for is named in the report", "image1: this shader does not read it" in _idle
          and "field: this shader does not run along a field" in _idle and "audio: this shader does not listen" in _idle)

    _red = ns.KUBA_Shader.execute("stripes", width=64, height=48, fps=5.0, seconds=0.4, controls="colour_a = #ff0000, colour_b = #ff0000").result
    check("colour: a picked colour is what the node draws, and the report names it", float(_red[0][..., 1].max()) < 0.02 and float(_red[0][..., 0].max()) > 0.5
          and float(_alone[0][..., 1].max()) > 0.3 and "colour_a = #ff0000" in _red[5] and "colour_a_r" not in _red[5])

    # ---- a drawn arrow and the presets that have no direction of their own
    _still = lambda n, **kw: ns.KUBA_Shader.execute(n, width=64, height=48, fps=5.0, seconds=0.2, **kw).result      # noqa: E731
    _f0, _fr, _fu = _still("fire"), _still("fire", vector="0.2, 0.5, 0.8, 0.5"), _still("fire", vector="0.5, 0.9, 0.5, 0.1")
    check("arrow: fire burns along a drawn arrow (to the right: bright on the left), upwards it is as without one",
          float(_fr[0][0, :, :8].mean()) > 0.6 and float(_fr[0][0, :, -8:].mean()) < 0.1 and float(_f0[0][0, -6:].mean()) > 0.6 and float(_f0[0][0, :6].mean()) < 0.1
          and torch.allclose(_fu[0], _f0[0], atol=2e-2) and "the shader follows it" in _fr[5])
    _s0, _s1 = _still("starfield"), _still("starfield", vector="0.5, 0.5, 0.9, 0.5")
    _s2 = _still("starfield", vector="0.25, 0.5, 0.9, 0.5")
    check("arrow: the stars fly out of the arrow's start (its middle moves there)", torch.allclose(_s1[0], _s0[0], atol=1e-4)
          and torch.allclose(_s2[0][0, :, :48], _s0[0][0, :, 16:], atol=1e-4) and not torch.allclose(_s2[0], _s0[0], atol=1e-2))
    _mk0 = torch.zeros((1, 48, 64)); _mk0[:, :, :20] = 1.0
    _fm = _still("fire", vector="0.2, 0.5, 0.8, 0.5", mask=_mk0)
    check("arrow: a mask stays where it is while the picture turns", float(_fm[0][0, :, 20:].max()) == 0.0 and torch.allclose(_fm[0][0, :, :20], _fr[0][0, :, :20], atol=1e-4))
    _marked = {n: st.arrow_mode(st.preset_code(n)) for n in names}
    check("arrow: every pattern and 3D scene does something with an arrow", all(st.translate(st.preset_code(n)).follows for n in names if by[n] in ("pattern", "space"))
          and _marked["fire"] == 2 and _marked["starfield"] == 1 and _marked["clouds"] == 3 and _marked["stripes"] == 0
          and st.arrow_mode("// arrow: sideways") == 0 and "#define K_ARROW 0" in st.translate(TOY).source and "kVector" not in st.translate(TOY).uses)

    check("what a preset reads and gives, in words for the picker", ns.preset_tags("cartoon") == ["your picture"] and ns.preset_tags("plasma") == []
          and ns.preset_tags("slit scan") == ["a clip", "node only"] and "gives a depth" in ns.preset_tags("melting forms")
          and ns.preset_tags("facade relief") == ["your picture", "a depth pass", "gives a depth"] and "a normal pass" in ns.preset_tags("glass")
          and "a field" in ns.preset_tags("region contours") and all(by[n] in ("look", "signal", "lens", "relief") for n in names if "your picture" in ns.preset_tags(n)),
          str({n: ns.preset_tags(n) for n in ("slit scan", "facade relief", "glass", "extrude", "scan band")}))
    import io as _pio
    from PIL import Image as _PImage
    for _n in ("plasma", "vhs tape", "extrude", "trails"):
        _im = _PImage.open(_pio.BytesIO(ns.preset_preview(_n)))
        _fr = []
        for _k in range(_im.n_frames):
            _im.seek(_k)
            _fr.append(np.asarray(_im.convert("RGB"), np.float32) / 255.0)
        check(f"the moving picture of '{_n}': one loop, lit and moving", _im.n_frames == ns.PREVIEW_FRAMES and _im.size == ns.PREVIEW
              and max(f.max() for f in _fr) > 0.2 and float(np.abs(_fr[0] - _fr[len(_fr) // 2]).max()) > 0.02, f"{_im.n_frames} {_im.size}")
    _a, _b = ns.look("plasma", "", "", False, ns.LOOK), ns.look("plasma", "", "scale = 6", False, ns.LOOK)
    _ia, _ib = (np.asarray(_PImage.open(_pio.BytesIO(q)).convert("RGB"), np.float32) / 255.0 for q in (_a, _b))
    check("adjust: one frame of a shader with its controls as they stand", _a[:2] == b"\xff\xd8" and _ia.shape == (ns.LOOK[1], ns.LOOK[0], 3)
          and float(np.abs(_ia - _ib).mean()) > 0.01 and ns.look("growth", "", "", False, (96, 54))[:2] == b"\xff\xd8"
          and ns.look("", TOY, "", True, (96, 54))[:4] == b"RIFF")
    check("a preview is kept, and a name that is no preset is refused", ns.preset_preview("plasma") is ns.preset_preview("plasma")
          and raises(lambda: ns.preset_preview("../x"), "no preset")[0])

    # ---- feedback: a shader that reads the frame before
    grow = "void mainImage(out vec4 c, in vec2 f){ c = vec4(kPrev(kUV).r + 0.25, float(iFrame) * 0.1, 0.0, 1.0); }"
    seen = [float(p[3, 3, 0]) for p in run(grow, [{"iFrame": i} for i in range(4)], 32, 16)]
    check("kPrev is the frame before (black at the first frame)", np.allclose(seen, [0.25, 0.5, 0.75, 1.0], atol=1e-5), str(seen))
    clip = torch.zeros((12, 36, 64, 3))
    for k in range(12):
        clip[k, :, :, 0] = k / 11.0                                                       # the clip gets redder frame by frame
    isl, _, _, _, nsl, rsl = ns.KUBA_Shader.execute("slit scan", controls="show = 0", image0=clip, loop=False).result[:6]
    col = isl[-1, 18, :, 0]
    check("slit scan: each column keeps the moment the slit passed it", nsl == 12 and float(col[2]) < float(col[30]) < float(col[56])
          and abs(float(col[56]) - 1.0) < 0.1 and float(col[63]) < 0.1, f"{[round(float(v), 2) for v in col[::8]]} {rsl}")
    idm, _, _, _, ndm, rdm = ns.KUBA_Shader.execute("datamosh", width=96, height=54, seconds=0.8).result[:6]
    check("datamosh and trails run on their own moving pattern with nothing connected, and loop through a fade",
          ndm == 20 and float(idm.max()) > 0.3 and "fades into the start" in rdm and "sample facade" not in rdm
          and float(ns.KUBA_Shader.execute("trails", width=96, height=54, seconds=0.4).result[0].mean()) > 0.02, rdm)
    check("controls: a name with a trailing _ in the code can be typed without it", st.parse_values("length = 0.5", [st.Control("length_", 0.3, 0, 1)])[0] == {"length_": 0.5})
    quiet = run(st.preset_code("sound bars"), [{"kLow": 0.0}])[0]
    loud = run(st.preset_code("sound bars"), [{"kLow": 1.0, "kMid": 0.8, "kHigh": 0.9, "kBeat": 1.0, "kBeats": 2.0}])[0]
    check("preset 'sound bars' grows with the sound", float(loud[..., :3].mean()) > 3 * float(quiet[..., :3].mean()))

    # ---- 3D: a raymarched scene, its depth
    px = run(SCENE, [{}], 96, 96, static={"kDepthOut": 1.0})[0]
    check("3D: a scene of the built-in helpers draws: a ball in front of a wall, lit side up, its depth in the alpha",
          abs(float(px[48, 48, 2]) - 1.0) < 0.02 and float(px[30, 48, 1]) > float(px[66, 48, 1]) + 0.2
          and abs(float(px[48, 48, 3]) - 0.909) < 0.02 and float(px[2, 2, 3]) < 0.2, f"{px[48, 48]} {px[30, 48]} {px[66, 48]} {px[2, 2]}")
    check("3D: without the node asking, the alpha stays the picture's", float(run(SCENE, [{}], 32, 32)[0][..., 3].min()) == 1.0)
    out = ns.KUBA_Shader.execute("melting forms", width=160, height=90, seconds=0.2)
    i3, m3, _, _, n3, r3, d3 = out.result
    check("node: a 3D preset gives its depth, the wall black and the forms near; the mask is the brightness",
          d3.shape == (5, 90, 160, 3) and float(d3.max()) > 0.5 and float(d3.min()) < 0.02 and "depth: near = white" in r3
          and "brightness" in r3 and torch.equal(d3[..., 0], d3[..., 2]) and float(m3.max()) > 0.3, r3)
    d0 = ns.KUBA_Shader.execute("rings", width=64, height=32, seconds=0.2).result
    check("node: a shader without a depth gives one black picture there", d0[6].shape == (1, 32, 64, 3) and float(d0[6].max()) == 0.0 and "depth" not in d0[5])
    half = torch.zeros((1, 90, 160))
    half[:, :, :80] = 1.0
    dc = ns.KUBA_Shader.execute("colonnade", seconds=0.04, mask=half).result[6]
    check("node: the depth is cut to the mask too", float(dc[:, :, 80:].max()) == 0.0 and float(dc[:, :, :80].max()) > 0.5)
    ex = ns.KUBA_Shader.execute("extrude", width=160, height=90, seconds=0.2, controls="rise = 0").result
    check("node: extrude with nothing connected rises from the windows of the sample facade", "sample facade" in ex[5] and float(ex[6].max()) > 0.9
          and float((ex[6][0, ..., 0] > 0.5).float().mean()) < 0.5 and float(ex[6][0, :3].max()) < 1e-3, ex[5])
    fr_ = torch.linspace(0, 1, 64)[None, None].repeat(1, 32, 1)
    ef = ns.KUBA_Shader.execute("extrude", seconds=0.04, field=fr_, controls="rise = 0, sway = 0").result
    check("node: extrude with a field connected uses it (no sample)", "sample facade" not in ef[5] and float(ef[6][0, 16, -2, 0]) > 0.9 and float(ef[6][0, 16, 0, 0]) < 0.1, ef[5])
    rl = ns.KUBA_Shader.execute("facade relief", width=160, height=90, seconds=0.2, controls="sway = 0, nod = 0, dark = 0, shade = 0").result
    from kubakub import sample_facade as _sf3
    flat3 = torch.from_numpy(_sf3.render(160, 90)["image"])
    check("node: facade relief seen straight on and unlit is the picture; its depth is the picture's brightness",
          float((rl[0][0] - flat3).abs().max()) < 0.02 and float(np.corrcoef(rl[6][0, ..., 0].numpy().ravel(), (flat3 @ torch.tensor([0.299, 0.587, 0.114])).numpy().ravel())[0, 1]) > 0.98,
          f"{float((rl[0][0] - flat3).abs().max()):.4f}")
    lean3 = ns.KUBA_Shader.execute("facade relief", width=160, height=90, seconds=0.2, loop=False).result[0]
    check("node: facade relief moves when the view leans", float((lean3[1] - lean3[0]).abs().mean()) > 0.003)

    # ---- Buffer tabs on the GPU
    rcnt = ns.KUBA_Shader.execute("rings", code=COUNT, width=32, height=16, seconds=0.16, loop=False).result
    check("tabs: a buffer that reads itself sees the frame before, one that reads an earlier tab sees this frame, the noise arrives",
          np.allclose(rcnt[0][:, 4, 4, 0].numpy(), [0.0625, 0.125, 0.1875, 0.25], atol=1e-4) and np.allclose(rcnt[0][:, 4, 4, 1].numpy(), [0.0625, 0.125, 0.1875, 0.25], atol=1e-4)
          and "2 buffer tabs (A, B)" in rcnt[5] and "Wired differently" in rcnt[5] and float(rcnt[0][0, :, :, 2].std()) > 0.05, rcnt[5])
    ok, msg = raises(lambda: ns.KUBA_Shader.execute("rings", code="// Buffer A\nvoid mainImage(out vec4 c, in vec2 f){\n c = vec4(oops);\n}\n// Image\nvoid mainImage(out vec4 c, in vec2 f){c=texture(iChannel0, f);}",
                                                    width=16, height=8), "does not compile", "line 3")
    check("tabs: a mistake in a buffer tab names its line in the pasted text", ok, msg)
    grow_mask = torch.zeros((1, 90, 160))
    grow_mask[:, :, :80] = 1.0
    gr = ns.KUBA_Shader.execute("growth", seconds=2.0, mask=grow_mask, controls="seeds = 1, size = 1", loop=False).result
    check("growth: it grows from its seeds, stays inside the mask and gives a depth",
          float(gr[0][45, :, :80].mean()) > float(gr[0][1, :, :80].mean()) + 0.02 and float(gr[0][:, :, 82:].max()) == 0.0 and float(gr[6][:, :, 82:].max()) == 0.0
          and float(gr[6].max()) > 0.5 and "Wired differently" not in gr[5],
          f"{float(gr[0][45, :, :80].mean()):.3f} {float(gr[0][1, :, :80].mean()):.3f} {float(gr[6].max()):.3f} {gr[5]}")
    check("growth: its loop is closed by a fade", "fades into the start" in ns.KUBA_Shader.execute("growth", width=64, height=36, seconds=0.4).result[5])
    ok, msg = raises(lambda: __import__("kubapack.nodes.shader.nodes_mix", fromlist=["x"]).KUBA_ShaderMix.execute(shader_a="growth", shader_b="rings", width=32, height=16), "last frame")
    check("mix: a shader with buffer tabs is refused with the reason", ok, msg)

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
    images, masks, fps, audio, count, report = out.result[:6]
    check("node, nothing connected: a preset plays", images.shape == (10, 96, 160, 3) and masks.shape == (10, 96, 160) and count == 10
          and fps == 25.0 and float(images.max()) > 0.2 and audio["waveform"].shape[-1] == 17640 and "warped noise" in report
          and "brightness" in report and "scale = 3" in report, report)
    check("node: a preview is written", out.ui and os.path.isfile(os.path.join(folder_paths.get_temp_directory(), out.ui["images"][0]["filename"])))
    out = ns.KUBA_Shader.execute("warped noise", controls="scale = 9", width=64, height=32, seconds=0.2, loops=2)
    check("node: controls text is used", "scale = 9" in out.result[5])
    drift = "out vec4 c;\nvoid main(){ c = vec4(vec3(fract(iTime * 0.37)), 0.25 + 0.5 * gl_FragCoord.x / iResolution.x); }"
    i1, m1, _, _, n1, r1 = ns.KUBA_Shader.execute("rings", code=drift, width=64, height=32, seconds=1.0, loop=True, loop_fade=0.4).result[:6]
    i0 = ns.KUBA_Shader.execute("rings", code=drift, width=64, height=32, seconds=1.0, loop=False).result[0]
    step = float((i0[1] - i0[0]).abs().max())
    check("node, own shader with loop: the end fades into the start", n1 == 25 and "fades into the start over 0.40 s" in r1
          and float((i1[0] - i0[0]).abs().max()) > 0.01 and torch.allclose(i1[12], i0[12])
          and abs(float(i1[0, 0, 0, 0]) - (0.37 * 25 / 25) % 1.0) < 1e-4, r1)
    check("node: the shader's alpha is the mask when it writes one", "alpha" in r1 and float(m1[0, 5, 0]) < 0.3 and float(m1[0, 5, -1]) > 0.7 and step > 0)
    toy = "void mainImage(out vec4 c, in vec2 f){ c = vec4(vec3(f.x / iResolution.x), 0.0); }"
    it, mt, _, _, _, rt = ns.KUBA_Shader.execute("rings", code=toy, width=64, height=32, seconds=0.2, mask_from="alpha").result[:6]
    check("node: a Shadertoy's alpha is ignored, as on shadertoy.com (alpha 0 is still a picture and a mask)",
          "brightness" in rt and "has no alpha" in rt and float(mt[0, 5, -1]) > 0.9 and float(it[0, 5, -1, 0]) > 0.9, rt)
    cut = torch.zeros((1, 32, 64))
    cut[:, :, :32] = 1.0
    ramp = torch.linspace(0, 1, 64)[None, None].repeat(1, 32, 1)
    ic, mc, _, _, _, rc = ns.KUBA_Shader.execute("scan band", width=0, height=0, seconds=0.2, mask=cut, field=1.0 - ramp).result[:6]
    check("node: size from the mask, pictures and masks are cut to it", ic.shape == (5, 32, 64, 3) and float(ic[:, :, 32:].max()) == 0.0
          and float(mc[:, :, 32:].max()) == 0.0 and float(ic[:, :, :32].max()) > 0.2, rc)
    check("node: the mask of a cut preset is the band, not the cut", "brightness" in rc and float(mc[:, :, :32].min()) < 0.1 < float(mc[:, :, :32].max()), rc)
    _, ma, _, _, _, ra = ns.KUBA_Shader.execute("rings", code=drift, width=64, height=32, seconds=0.2, loop=False, mask=cut).result[:6]
    check("node: a shader's own alpha is cut to the mask too", "alpha" in ra and float(ma[:, :, 32:].max()) == 0.0 and 0.2 < float(ma[0, 5, 0]) < 0.3, ra)
    listen = "void mainImage(out vec4 c, in vec2 f){ c = vec4(kLow, kBeat, kBeats / 8.0, 1.0); }"
    il, _, _, al, nl, rl = ns.KUBA_Shader.execute("rings", code=listen, width=16, height=8, loop=False).result[:6]
    check("node: a shader that listens gets the built-in beat", nl == 100 and "built-in beat" in rl and float(il[0, 0, 0, 1]) == 1.0
          and float(il[0, 0, 0, 0]) > 0.5 and float(il[6, 0, 0, 1]) < 0.6 and al["waveform"].shape[-1] == 4 * 44100, rl)
    wave = torch.from_numpy(so.sample_sound(2.0, 100.0, 22050))[None, None]
    _, _, _, a2, n2, r2 = ns.KUBA_Shader.execute("rings", width=16, height=8, audio={"waveform": wave, "sample_rate": 22050}, bpm=100.0).result[:6]
    check("node: the length is the sound's, the tempo can be set", n2 == 50 and a2["sample_rate"] == 22050 and "100.0 bpm (set)" in r2, r2)
    img = torch.rand((3, 20, 40, 3))
    ig, _, _, _, ng, rg = ns.KUBA_Shader.execute("chromatic split", controls="amount = 0, wave = 0, sound = 0", image0=img).result[:6]
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
    i_n, _, _, _, _, r_n = ns.KUBA_Shader.execute("rings", code=reads, width=64, height=64, seconds=0.04, loop=False).result[:6]
    check("node: your code reading a picture with none connected gets the built-in noise, and the report says so",
          "built-in noise" in r_n and "image0" in r_n and float(i_n.std()) > 0.08, r_n)
    i_b, _, _, _, _, r_b = ns.KUBA_Shader.execute("rings", code=reads, width=64, height=64, seconds=0.04, loop=False, unconnected="black").result[:6]
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
    # ---- a fill's own field: the shape's edges (extrude, region contours)
    probe = "void mainImage(out vec4 c, in vec2 f){ c = vec4(kField(kUV), kMask(kUV), 0.0, 1.0); }"
    fl = lambda **k: rd.parse(dict(doc, layers=[dict(doc["layers"][0], **k), doc["layers"][1]]))["layers"][0]   # noqa: E731
    Lr = fl(shape=dict(type="rect", code=probe, field="edge"))
    f_r = rd.shader_field(Lr)
    check("field: a rectangle rises from its outline to its middle, the mask is the shape", f_r.shape == (50, 80, 2) and float(f_r[0, 40, 0]) == 0.0
          and float(f_r[25, 40, 0]) > 0.95 and 0.2 < float(f_r[25, 6, 0]) < 0.35 and float(f_r[..., 1].min()) == 1.0, f"{f_r[25, 40]} {f_r[25, 6]}")
    f_e = rd.shader_field(fl(shape=dict(type="ellipse", code=probe, field="edge")))
    check("field: an ellipse is 0 in the corners of its box, 1 in the middle", float(f_e[1, 1, 1]) == 0.0 and float(f_e[1, 1, 0]) == 0.0 and float(f_e[25, 40, 0]) > 0.95)
    two_r = np.zeros((90, 160), bool)
    two_r[30:60, 45:70] = True
    two_r[30:60, 90:115] = True
    f_c = rd.shader_field(Lr, two_r, (160, 90))
    check("field: a layer clipped to regions has every region as a piece of its own", float(f_c[25, 17, 0]) > 0.9 and float(f_c[25, 62, 0]) > 0.9
          and float(f_c[25, 40, 1]) == 0.0 and float(f_c[25, 40, 0]) == 0.0 and float(f_c[5, 17, 1]) == 0.0, f"{f_c[25, 17]} {f_c[25, 62]} {f_c[25, 40]}")
    turned = rd.shader_field(dict(Lr, rotation=90.0), two_r, (160, 90))
    check("field: the regions follow a turned layer into its box", float(turned[..., 1].sum()) > 0 and not np.array_equal(turned[..., 1], f_c[..., 1]))
    edge_doc = dict(doc, layers=[dict(doc["layers"][0], shape=dict(type="rect", code=probe, field="edge")), doc["layers"][1]])
    ie_ = rd.render(rd.animate(edge_doc, 0.0), base, {}, image_only=True)["image"]
    ramp_doc = dict(doc, layers=[dict(doc["layers"][0], shape=dict(type="rect", code=probe)), doc["layers"][1]])
    ir_ = rd.render(rd.animate(ramp_doc, 0.0), base, {}, image_only=True)["image"]
    check("director: field = the shape's edges reaches the shader; without it the old left to right ramp stays",
          float(ie_[45, 80, 0]) > 0.9 and float(ie_[45, 42, 0]) < 0.1 and float(ie_[22, 80, 0]) < 0.1 and float(ir_[45, 42, 0]) < 0.1 and float(ir_[45, 117, 0]) > 0.9
          and abs(float(ir_[22, 80, 0]) - 0.5) < 0.05, f"{ie_[45, 80]} {ie_[45, 42]} {ir_[45, 117]}")
    check("director: extrude takes the shape's edges by itself, region contours and own code keep the ramp until told",
          fl(shape=dict(type="rect", shader="extrude"))["shape"]["field"] == "edge" and fl(shape=dict(type="rect", shader="region contours"))["shape"]["field"] == "ramp"
          and fl(shape=dict(type="rect", code=probe))["shape"]["field"] == "ramp" and fl(shape=dict(type="rect", shader="extrude", field="ramp"))["shape"]["field"] == "ramp"
          and ns.shader_info("extrude")["wants_field"] and not ns.shader_info("rings")["wants_field"])
    ex_doc = dict(doc, layers=[dict(doc["layers"][0], shape=dict(type="ellipse", shader="extrude", c_rise=0, c_sway=0, c_turns=0)), doc["layers"][1]])
    iex = rd.render(rd.animate(ex_doc, 0.0), np.full((90, 160, 3), 0.5, np.float32), {}, image_only=True)["image"]
    check("director: an ellipse filled with extrude lifts the picture under it as a block in its own shape: flat on top, lit and shaded on its slopes",
          abs(float(iex[45, 80, 0]) - 0.5) < 0.02 and float(iex[20:70, 40:120, 0].max()) > 0.6 and float(iex[20:70, 40:120, 0].min()) < 0.4
          and abs(float(iex[5, 5, 0]) - 0.5) < 1e-4, f"{iex[45, 80]} {iex[20:70, 40:120, 0].max():.3f} {iex[20:70, 40:120, 0].min():.3f}")
    buf_doc = dict(doc, layers=[dict(doc["layers"][0], shape=dict(type="rect", code=COUNT)), doc["layers"][1]])
    rbuf = rd.render(rd.animate(buf_doc, 0.0), base, {}, image_only=True)
    check("director: a Shadertoy with buffer tabs is named as node only, the colour shows", any("buffer tabs" in n for n in rbuf["notes"])
          and abs(float(rbuf["image"][45, 80, 0]) - 1.0) < 0.01 and ns.shader_info("", COUNT)["buffers"] == ["A", "B"], str(rbuf["notes"]))
    check("window: a fill's own field and the buffer refusal are wired", all(x in js for x in ("shaderField(L, info)", "shaderFieldMode(L, info)", 'data-id="sh-fieldsrc"',
          "gl.RGBA16F, field.w, field.h", 'fieldRamp(rp, fw, fh, { field: "edge distance", per: "each region" }, 1)', "info.buffers || []")))
    reads_doc = dict(doc, layers=[dict(doc["layers"][0], shape=dict(shape, code=reads)), doc["layers"][1]])
    i_d = rd.render(rd.animate(reads_doc, 0.0), base, {}, image_only=True)["image"]
    check("director: a shader that reads a picture gets the built-in noise (as the window)", float(i_d[25:65, 45:115].std()) > 0.1)

shutil.rmtree(TMP, ignore_errors=True)
print()
print("ALL PASSED" if not failures else f"{len(failures)} FAILED: {failures}")
sys.exit(1 if failures else 0)
