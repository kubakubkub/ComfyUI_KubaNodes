"""
Test of OSL shaders in kubakub shader (kubakub/osl.py through shadertoy.translate): the inputs become controls and
colours, the outputs and Ci are shown, OSL's loose numbers compile as GLSL, and five shaders as written for Blender
Cycles render. The text part needs nothing; the GPU part needs a GL context (skipped without one).

Run from the ComfyUI_windows_portable folder:
    python_embeded\\python.exe ComfyUI\\custom_nodes\\ComfyUI_KubaNodes\\tests\\test_osl.py
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(HERE))))     # ComfyUI
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import _pack  # noqa: E402,F401
from kubakub import osl  # noqa: E402
from kubakub import shadertoy as st  # noqa: E402

failures = []


def check(name, cond, detail=""):
    print(("ok   " if cond else "FAIL ") + name + (f"  ({detail})" if detail and not cond else ""))
    if not cond:
        failures.append(name)


BANDS = """// noise bands, written for Blender Cycles
shader noise_bands(
    float Scale = 5.0 [[ float min = 0.1, float max = 50, string help = "size of the pattern" ]],
    color ColorA = color(0.63, 0.53, 0.72),
    color ColorB = color(0.95, 0.54, 0.35),
    int Octaves = 4,
    float time = 0,
    point Pos = P,
    output color Col = 0,
    output float Fac = 0)
{
    float n = 0;
    float amp = 0.5;
    point p = Pos * Scale + vector(0, 0, time * 0.2);
    for (int i = 0; i < Octaves; i++) {
        n += amp * noise("perlin", p);
        p *= 2;
        amp *= 0.5;
    }
    Fac = smoothstep(-0.3, 0.3, n);
    Col = mix(ColorA, ColorB, Fac);
}
"""

GLOW = """surface glow(color Tint = color(1, 0.5, 0.2), float Rings = 8 [[ int min = 1, int max = 40 ]])
{
    float d = distance(P, point(0.89, 0.5, 0));
    float r = 0.5 + 0.5 * sin(d * Rings * M_2PI - time * 2);
    Ci = Tint * r * emission();
}
"""

CELLS = """#include "stdosl.h"
float hash1(point p) { return cellnoise(p); }
shader cells(float Scale = 8, int Invert = 0 [[ string widget = "checkBox" ]], output color Cout = 0,
             output float Dist = 0)
{
    point p = P * Scale;
    point cell = floor(p);
    float best = 10;
    for (int x = -1; x <= 1; x++)
        for (int y = -1; y <= 1; y++) {
            point c = cell + vector(x, y, 0);
            vector jitter = cellnoise(c);
            float d = distance(p, c + jitter * vector(1, 1, 0));
            if (d < best) best = d;
        }
    int idx = int(cell[0]) % 3;
    float pal[3] = {0.2, 0.5, 0.9};
    Dist = best;
    Cout = color("hsv", pal[idx], 0.6, Invert ? 1 - best : best);
    if (Invert) Dist = 1 - Dist;
}
"""

HEAT = """shader heat(string NoiseType = "perlin", float Temp = 3000 [[ float min = 1000, float max = 12000 ]],
            output color C = 0)
{
    point q = transform("object", P) * 6;
    float n = noise(NoiseType, q, time) * 0.5 + 0.5;
    C = blackbody(Temp * n) * n;
}
"""

DISPLACE = """displacement bumps(float Height = 0.1) { P += N * Height * noise(P * 10); }"""

# ---- text: what the node sees ---------------------------------------------------------------------------------------
check("an OSL shader is recognised", osl.is_osl(st._no_comments(BANDS)) and osl.is_osl(GLOW) and osl.is_osl(CELLS))
check("a Shadertoy is not", not osl.is_osl("void mainImage(out vec4 c, in vec2 f) { c = vec4(1.0); }"))
tr = st.translate(BANDS)
names = [c.name for c in tr.controls]
check("float inputs become sliders, colours become colours", names == ["Scale", "ColorA_r", "ColorA_g", "ColorA_b",
                                                                      "ColorB_r", "ColorB_g", "ColorB_b", "Octaves"], str(names))
sc = tr.controls[0]
check("the slider takes OSL's min / max / help", (sc.default, sc.lo, sc.hi) == (5.0, 0.1, 50.0) and sc.about == "size of the pattern",
      str(sc))
check("the colour keeps its value", abs(tr.controls[1].default - 0.63) < 0.01 and abs(tr.controls[4].default - 0.95) < 0.01)
check("an int input without a range gets one around its value", tr.controls[-1].default == 4.0 and tr.controls[-1].hi >= 8.0)
check("an input named time follows the clock (no slider)", "time" not in names and "follows the clock" in " ".join(tr.notes))
check("it runs as a Shadertoy", tr.kind == "shadertoy")
check("the note says which output is shown and which others there are", any("shows Col" in n and "Fac" in n for n in tr.notes),
      str(tr.notes))
check("whole numbers became floats, the loop counter too", "float n = 0.0;" in tr.source and "for (float i = 0.0;" in tr.source)
check("noise('perlin', p) became the signed perlin noise", "kosl_sperlin_f(p)" in tr.source)
pick = st.translate(BANDS.replace("// noise bands", "// output: Fac\n// noise bands"))
check("'// output: Fac' shows Fac as grey", "fragColor = vec4(kv3(vec3(Fac)), 1.0);" in pick.source)

tg = st.translate(GLOW)
check("a surface shader: Ci is shown, emission() counts as white", "fragColor = vec4(kv3(Ci), 1.0);" in tg.source
      and "vec3(1.0)" in tg.source and any("closures (emission)" in n for n in tg.notes), str(tg.notes))
check("an int range in [[ ]] is read", [(c.lo, c.hi) for c in tg.controls if c.name == "Rings"] == [(1.0, 40.0)])

tc = st.translate(CELLS)
check("a checkbox becomes a 0-1 slider", [(c.lo, c.hi) for c in tc.controls if c.name == "Invert"] == [(0.0, 1.0)])
check("a cellnoise given to a vector is the colour noise", "kosl_cell_v(c)" in tc.source)
check("an array in braces, an index cast back, % as mod", "float pal[3] = float[3](0.2, 0.5, 0.9);" in tc.source
      and "pal[int(idx)]" in tc.source and "mod(trunc(cell[0]), 3.0)" in tc.source, tc.source[-900:])
check("a number as a condition", "(Invert != 0.0) ?" in tc.source and "if (Invert != 0.0)" in tc.source)
check("hsv colours", "kosl_hsv(" in tc.source)

th = st.translate(HEAT)
check("a string input is put in where it is used (the noise type)", "kosl_sperlin_f(q, time)" in th.source)
check("transform() keeps the point, blackbody() is there", "(P) * 6.0" in th.source and "vec3 kosl_blackbody" in th.source)

for code, what in ((DISPLACE, "displacement"), ("shader x(output color C = 0) { C = texture(\"wood.png\", u, v); }", None),
                   ("shader x(float A = 1) { float b = A; }", "no output")):
    try:
        t = st.translate(code)
        ok = what is None and "kosl_tex(u, v)" in t.source and any("image0" in n for n in t.notes)
    except st.ShaderError as e:
        ok = what is not None and what in str(e)
    check(f"said clearly: {what or 'a texture file reads image0'}", ok)

# ---- GPU: they compile and draw -------------------------------------------------------------------------------------
try:
    from kubakub import shader_render as sr
    sr.core().GLContext()
    gpu = True
except Exception as e:  # noqa: BLE001
    gpu = False
    print(f"skip the GPU part: no GL context here ({e})")

if gpu:
    def draw(code, values=None, w=96, h=54, t=0.0):
        tr = st.translate(code)
        vals = {c.name: c.default for c in tr.controls}
        vals.update(values or {})
        out = []
        sr.render(tr.source, w, h, [{"iTime": t}], vals, None, lambda i, px: out.append(px.copy()))
        return out[0][..., :3]

    for name, code in (("noise bands", BANDS), ("glow", GLOW), ("cells", CELLS), ("heat", HEAT)):
        try:
            px = draw(code)
            check(f"renders: {name} (a picture with something in it)", px.std() > 0.01 and np.isfinite(px).all(),
                  f"std {px.std():.4f}")
        except Exception as e:  # noqa: BLE001
            check(f"renders: {name}", False, str(e)[:400])
    a, b = draw(BANDS), draw(BANDS, {"ColorA_r": 0.0, "ColorA_g": 0.0, "ColorA_b": 0.0})
    check("a colour control changes the picture", abs(float(a.mean()) - float(b.mean())) > 0.02)
    check("time moves the noise", float(np.abs(draw(BANDS, t=0.0) - draw(BANDS, t=3.0)).mean()) > 0.005)
    try:
        st.translate("shader x(output color C = 0) { C = foo(P); }")
        sr.render(st.translate("shader x(output color C = 0) { C = foo(P); }").source, 8, 8, [{}], {}, None, lambda i, px: None)
        check("a GLSL error names the line of the OSL text", False, "no error")
    except Exception as e:  # noqa: BLE001
        check("a GLSL error names the line of the OSL text", "line 1" in st.explain(str(e)), st.explain(str(e))[:200])

print()
print("FAILED: " + ", ".join(failures) if failures else "all OSL tests passed")
if __name__ == "__main__":
    sys.exit(1 if failures else 0)
