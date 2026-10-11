"""
shadertoy.py

kubakub shader, the text side: a pasted Shadertoy (or a plain GLSL fragment shader) becomes one complete GLSL ES 3.00
fragment shader. GLSL ES 3.00 is WebGL 2, so the same text compiles in ComfyUI's GL context and in a browser.

The names a shader can use:
    Shadertoy   iResolution, iTime, iTimeDelta, iFrame, iFrameRate, iMouse (never pressed), iDate, iSampleRate,
                iChannel0-3, iChannelResolution, iChannelTime
    kubakub     kLoop (0-1 through the clip), kSeconds, kLevel / kLow / kMid / kHigh (the sound, 0-1),
                kPrev(uv) (the frame before this one: trails, slit scan, datamosh; black at frame 0),
                kBeat (1 on a beat, falling to 0 before the next), kBeats (beats so far),
                kField(uv) and kMask(uv) (the grey ramp and the mask connected to the node), kUV,
                kDepth (a shader writes how near this pixel is, 0 far to 1 near: the node's depth output)
    3D          distance functions and a raymarcher, added to a shader that uses their names (SDF below)
    controls    every `uniform float name; // default min max` line of the shader

Text only, no ComfyUI and no GL imports (tests/test_shadertoy.py).
"""

from __future__ import annotations

import functools
import os
import re
from dataclasses import dataclass, field

SHADER_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "shaders")
CHANNELS = 4
MAX_CODE = 400_000                                   # characters of pasted shader text
MAX_CONTROLS = 32

FLOATS = ("iTime", "iTimeDelta", "iFrameRate", "iSampleRate", "kLoop", "kSeconds", "kLevel", "kLow", "kMid", "kHigh",
          "kBeat", "kBeats", "kAx", "kAy", "kBx", "kBy", "kVector", "kVecField", "kBend", "kDepthOut")
VECTOR = frozenset({"kFrom", "kTo", "kDir", "kVector"})      # a shader that reads one of these follows a drawn arrow
SOUND = frozenset({"kLevel", "kLow", "kMid", "kHigh", "kBeat", "kBeats"})      # a shader that reads one of these listens
_BUILTIN = set(FLOATS) | {"iResolution", "iFrame", "iMouse", "iDate", "iChannelTime", "iChannelResolution",
                          "kFieldTex", "kPrevTex", "kClip", "u_resolution", "kFrom", "kTo", "kDir", "kDirOf", "kMixTex", "kShift", "kDepth"} | {f"iChannel{i}" for i in range(CHANNELS)}

HEADER = """#version 300 es
precision highp float;
precision highp int;
precision highp sampler2D;
uniform vec3 iResolution;
uniform float iTime;
uniform float iTimeDelta;
uniform float iFrameRate;
uniform int iFrame;
uniform vec4 iMouse;
uniform vec4 iDate;
uniform float iSampleRate;
uniform float iChannelTime[4];
uniform vec3 iChannelResolution[4];
uniform sampler2D iChannel0;
uniform sampler2D iChannel1;
uniform sampler2D iChannel2;
uniform sampler2D iChannel3;
uniform float kLoop;
uniform float kSeconds;
uniform float kLevel;
uniform float kLow;
uniform float kMid;
uniform float kHigh;
uniform float kBeat;
uniform float kBeats;
uniform sampler2D kFieldTex;
uniform sampler2D kPrevTex;
uniform float kClip;
uniform float kAx;                                // an arrow drawn on the picture: its start and end, 0-1, (0, 0) bottom left
uniform float kAy;
uniform float kBx;
uniform float kBy;
uniform float kVector;                            // 1 when an arrow is drawn
uniform float kVecField;                          // 1 when the arrow is the field (no field of its own is connected)
uniform sampler2D kMixTex;                        // kubakub shader mix: the other shader's picture
uniform float kBend;                              // ... and how far it pushes this one's places (part of the height)
uniform float kDepthOut;                          // 1 when the node asks for the depth: it travels in the alpha
layout(location = 0) out vec4 kOut;
float kDepth = 0.0;                               // how near this pixel is, 0 far to 1 near: a shader that knows writes it
vec2 kShift = vec2(0.0);                          // pixels this pixel draws away from itself (set before the shader runs)
#define kUV ((gl_FragCoord.xy + kShift) / iResolution.xy)
#define kFrom vec2(kAx, kAy)
#define kTo vec2(kBx, kBy)
#define kDir kDirOf()
vec2 kDirOf() {                                   // the way the arrow points, length 1, measured in pixels
    vec2 d = (kTo - kFrom) * iResolution.xy;
    return dot(d, d) > 1e-9 ? normalize(d) : vec2(1.0, 0.0);
}
float kField(vec2 uv) {                           // 0 at the arrow's start, 1 at its end; else the connected field
    if (kVecField > 0.5) {
        vec2 d = (kTo - kFrom) * iResolution.xy;
        return clamp(dot((uv - kFrom) * iResolution.xy, d) / max(dot(d, d), 1e-9), 0.0, 1.0);
    }
    return texture(kFieldTex, uv).r;
}
float kMask(vec2 uv) { return texture(kFieldTex, uv).g; }
vec4 kPrev(vec2 uv) { return texture(kPrevTex, uv); }
void kStart() {                                   // red pushes sideways, green up and down; mid grey pushes nowhere
    if (kBend != 0.0) kShift = (texture(kMixTex, gl_FragCoord.xy / iResolution.xy).rg - 0.5) * 2.0 * kBend * iResolution.y;
}
void kFinish() {                                  // on the GPU: 0-1, and the picture cut to the mask when the node asks
    kOut = min(max(kOut, 0.0), 1.0);
    if (kClip > 0.5) kOut.rgb *= kMask(gl_FragCoord.xy / iResolution.xy);
    if (kDepthOut > 0.5) kOut.a = min(max(kDepth, 0.0), 1.0);
}
"""

# 3D for a shader that asks for it: (name, the names it needs, code). A shader gets the ones whose names it uses and
# does not define itself, so a pasted Shadertoy that brings its own sdBox keeps it. The march, the normal, the shadow
# and the occlusion walk the shader's own `float map(vec3 p)`: the distance from p to the nearest surface.
# Distance functions and smooth joins after Inigo Quilez, iquilezles.org/articles/distfunctions
SDF = (
    ("sdSphere", (), "float sdSphere(vec3 p, float r) { return length(p) - r; }"),
    ("sdBox", (), "float sdBox(vec3 p, vec3 b) { vec3 q = abs(p) - b; return length(max(q, 0.0)) + min(max(q.x, max(q.y, q.z)), 0.0); }"),
    ("sdRoundBox", (), "float sdRoundBox(vec3 p, vec3 b, float r) { vec3 q = abs(p) - b + r; "
                       "return length(max(q, 0.0)) + min(max(q.x, max(q.y, q.z)), 0.0) - r; }"),
    ("sdTorus", (), "float sdTorus(vec3 p, vec2 t) { return length(vec2(length(p.xz) - t.x, p.y)) - t.y; }"),
    ("sdCylinder", (), "float sdCylinder(vec3 p, float h, float r) { vec2 d = abs(vec2(length(p.xz), p.y)) - vec2(r, h); "
                       "return min(max(d.x, d.y), 0.0) + length(max(d, 0.0)); }"),
    ("sdCapsule", (), "float sdCapsule(vec3 p, vec3 a, vec3 b, float r) { vec3 pa = p - a, ba = b - a; "
                      "return length(pa - ba * clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0)) - r; }"),
    ("sdPlane", (), "float sdPlane(vec3 p, vec3 n, float h) { return dot(p, n) + h; }"),
    ("opSmoothUnion", (), "float opSmoothUnion(float a, float b, float k) { float h = clamp(0.5 + 0.5 * (b - a) / k, 0.0, 1.0); "
                          "return mix(b, a, h) - k * h * (1.0 - h); }"),
    # b with a taken out of it
    ("opSmoothSubtraction", (), "float opSmoothSubtraction(float a, float b, float k) { float h = clamp(0.5 - 0.5 * (b + a) / k, 0.0, 1.0); "
                                "return mix(b, -a, h) + k * h * (1.0 - h); }"),
    ("opSmoothIntersection", (), "float opSmoothIntersection(float a, float b, float k) { float h = clamp(0.5 - 0.5 * (b - a) / k, 0.0, 1.0); "
                                 "return mix(b, a, h) + k * h * (1.0 - h); }"),
    # the space repeated every c
    ("opRep", (), "vec3 opRep(vec3 p, vec3 c) { return p - c * round(p / c); }"),
    ("kRot", (), "mat2 kRot(float a) { float c = cos(a), s = sin(a); return mat2(c, -s, s, c); }"),
    # the ray through a pixel for a camera at ro that looks at target (zoom 1 = a wide view, 2 = a long lens)
    ("kCamera", (), """vec3 kCamera(vec2 fragCoord, vec3 ro, vec3 target, float zoom) {
    vec2 p = (2.0 * fragCoord - iResolution.xy) / iResolution.y;
    vec3 f = normalize(target - ro), r = normalize(cross(f, vec3(0.0, 1.0, 0.0))), u = cross(r, f);
    return normalize(p.x * r + p.y * u + zoom * f);
}"""),
    ("map", (), "float map(vec3 p);"),
    # how far along the ray the first surface is; -1 when there is none within `far`
    ("kMarch", ("map",), """float kMarch(vec3 ro, vec3 rd, float far, int steps) {
    float t = 0.0;
    for (int i = 0; i < steps; i++) {
        float d = map(ro + rd * t);
        if (abs(d) < 0.0006 * (1.0 + t) || t > far) break;
        t += d;
    }
    return t < far ? t : -1.0;
}"""),
    ("kNormal", ("map",), """vec3 kNormal(vec3 p) {
    const vec2 e = vec2(1.0, -1.0) * 0.0015;
    return normalize(e.xyy * map(p + e.xyy) + e.yyx * map(p + e.yyx) + e.yxy * map(p + e.yxy) + e.xxx * map(p + e.xxx));
}"""),
    # 1 = lit, 0 = in shadow; k = how hard the edge is (8 soft, 64 hard)
    ("kShadow", ("map",), """float kShadow(vec3 p, vec3 toLight, float far, float k) {
    float res = 1.0, t = 0.02;
    for (int i = 0; i < 48; i++) {
        float h = map(p + toLight * t);
        res = min(res, k * h / t);
        t += clamp(h, 0.01, 0.25);
        if (res < 0.002 || t > far) break;
    }
    return clamp(res, 0.0, 1.0);
}"""),
    # 1 = open, darker in corners and creases
    ("kAO", ("map",), """float kAO(vec3 p, vec3 n) {
    float occ = 0.0, sca = 1.0;
    for (int i = 0; i < 5; i++) {
        float h = 0.01 + 0.03 * float(i);
        occ += (h - map(p + n * h)) * sca;
        sca *= 0.95;
    }
    return clamp(1.0 - 3.0 * occ, 0.0, 1.0);
}"""),
    # a distance along the ray as a depth: 1 at `near`, 0 at `far` and where the ray hit nothing
    ("kDepthOf", (), "float kDepthOf(float t, float near, float far) { "
                     "return t < 0.0 ? 0.0 : clamp(1.0 - (t - near) / max(far - near, 1e-6), 0.0, 1.0); }"),
)

# a plain fragment shader written for ComfyUI's GLSL Shader node keeps working: its names map onto the same inputs
HEADER_GLSL = """in vec2 v_texCoord;
#define u_resolution (iResolution.xy)
#define u_image0 iChannel0
#define u_image1 iChannel1
#define u_image2 iChannel2
#define u_image3 iChannel3
#define fragColor0 kOut
"""

FOOTER_TOY = """
void main() {
    vec4 kColour = vec4(0.0, 0.0, 0.0, 1.0);
    kStart();
    mainImage(kColour, gl_FragCoord.xy + kShift);
    kOut = vec4(kColour.rgb, 1.0);                // Shadertoy shows the colour and ignores the alpha: many write 0 there
    kFinish();
}
"""

# a Buffer tab of a Shadertoy: its numbers go on as they are (a tab keeps positions, speeds, counters, not colours)
FOOTER_BUFFER = """
void main() {
    vec4 kColour = vec4(0.0);
    mainImage(kColour, gl_FragCoord.xy);
    kOut = kColour;
}
"""

FOOTER_GLSL = """
void main() {
    kStart();
    kUserMain();
    kFinish();
}
"""

_NUM = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"
_CONTROL = re.compile(r"^[ \t]*uniform\s+float\s+(\w+)\s*;[ \t]*(?://(.*))?$", re.M)
# a colour to pick: `uniform vec3 ink; // #f18a58 what it is`. In the shader it is three numbers (ink_r, ink_g, ink_b,
# each a control like any other, so the director and the keyframes need nothing new) and `ink` is a name for the three
_COLOUR = re.compile(r"^[ \t]*uniform\s+vec3\s+(\w+)\s*;[ \t]*//[ \t]*#([0-9a-fA-F]{6})\b[ \t]*(.*)$", re.M)
_UNIFORM = re.compile(r"^[ \t]*uniform\s+(?:(?:highp|mediump|lowp)\s+)?\w+\s+(\w+)\s*(?:\[\s*\d+\s*\])?\s*;.*$", re.M)
_TOY = re.compile(r"\bvoid\s+mainImage\s*\(")
_MAIN = re.compile(r"\bvoid\s+main\s*\(")


HOW_TO_COPY = ("On shadertoy.com: click into the code, press Ctrl+A, then Ctrl+C, and paste it into 'code' "
               "(Ctrl+V). If the shader has a 'Common' tab, paste that one first and the 'Image' tab below it.")


class ShaderError(ValueError):
    """Something in the shader text that the artist can fix; the message says what."""


@dataclass
class Control:
    name: str
    default: float = 0.0
    lo: float = 0.0
    hi: float = 1.0
    about: str = ""                   # what the comment says after the numbers
    colour: str = ""                  # the colour it is the red, green or blue of ('ink' for ink_r), else ""


@dataclass
class Translated:
    source: str                       # the complete fragment shader
    kind: str                         # shadertoy | glsl
    controls: list = field(default_factory=list)
    notes: list = field(default_factory=list)
    channels: list = field(default_factory=list)      # which of iChannel0-3 the shader reads
    optional: set = field(default_factory=set)        # those it has an answer of its own for ('// optional: image1')
    uses: set = field(default_factory=set)            # which kubakub / Shadertoy names it reads
    buffers: list = field(default_factory=list)       # the Buffer A-D tabs of a Shadertoy, in the order they are drawn (Pass)
    inputs: dict = field(default_factory=dict)        # what the Image tab reads on a channel when it is not the node's picture

    @property
    def listens(self) -> bool:
        return bool(self.uses & SOUND)

    @property
    def depth(self) -> bool:
        """It writes how near each pixel is (kDepth): the node's depth output."""
        return "kDepth" in self.uses

    @property
    def follows(self) -> bool:
        """It does something with a drawn arrow: it reads the arrow, or the field (which the arrow becomes)."""
        return bool(self.uses & (VECTOR | {"kField"}))


def _no_comments(code: str) -> str:
    """The code with comments blanked out (same length and lines), for looking at what it really uses."""
    out, i, n = [], 0, len(code)                     # one walk through the text (a regex takes minutes on a long broken one)
    while i < n:
        a, b = code.find("//", i), code.find("/*", i)
        if a < 0 and b < 0:
            break
        if b < 0 or 0 <= a < b:
            e = code.find("\n", a)
            e = n if e < 0 else e
            out += [code[i:a], " " * (e - a)]
        else:
            a, e = b, code.find("*/", b + 2)
            e = n if e < 0 else e + 2
            out += [code[i:a], re.sub(r"[^\n]", " ", code[a:e])]
        i = e
    return "".join(out) + code[i:]


def parse_controls(code: str) -> list:
    """`uniform float speed; // 1 0 4` -> Control(speed, 1, 0, 4). The numbers are default, min, max; all optional."""
    out = []
    for m in sorted([*_CONTROL.finditer(code), *_COLOUR.finditer(code)], key=lambda q: q.start()):
        name = m.group(1)
        if name in _BUILTIN or re.fullmatch(r"u_(float|int|bool)\d+", name):
            continue
        if m.re is _COLOUR:                           # a colour: its red, green and blue, 0-1
            out += [Control(f"{name}_{ch}", int(m.group(2)[2 * i:2 * i + 2], 16) / 255.0, 0.0, 1.0, m.group(3).strip(), name)
                    for i, ch in enumerate("rgb")]
            continue
        comment = m.group(2) or ""
        head = re.match(rf"\s*(?:{_NUM}\s+){{0,2}}(?:{_NUM})?(?![\w.])", comment)
        nums = [float(x) for x in re.findall(_NUM, head.group(0) if head else "")]
        d = nums[0] if nums else 0.0
        lo, hi = (nums[1], nums[2]) if len(nums) >= 3 else (min(0.0, d), max(1.0, d))
        out.append(Control(name, d, min(lo, hi), max(lo, hi), comment[head.end() if head else 0:].strip()))
    if len(out) > MAX_CONTROLS:
        raise ShaderError(f"the shader has {len(out)} controls (uniform float lines); {MAX_CONTROLS} is the limit.")
    return out


def _colours(body: str) -> tuple:
    """The colour lines of a shader as three numbers each, on the same line -> (body, the '#define's that give the
    three their one name back)."""
    names = [m.group(1) for m in _COLOUR.finditer(body) if m.group(1) not in _BUILTIN]
    body = _COLOUR.sub(lambda m: m.group(0) if m.group(1) in _BUILTIN else
                       "uniform float {0}_r; uniform float {0}_g; uniform float {0}_b;".format(m.group(1)), body)
    return body, "".join("#define {0} vec3({0}_r, {0}_g, {0}_b)\n".format(n) for n in dict.fromkeys(names))


def colour_hex(values: dict, name: str) -> str:
    """The colour `name` in these control values as '#rrggbb'."""
    return "#" + "".join(f"{int(round(min(max(float(values.get(f'{name}_{ch}', 0.0)), 0.0), 1.0) * 255)):02x}" for ch in "rgb")


def unsupported(code: str, tabbed=False) -> list:
    """What a pasted Shadertoy uses that cannot be done here, in words."""
    c = _no_comments(code)
    out = []
    if len(_TOY.findall(c)) > 1 and not tabbed:
        out.append("it has more than one mainImage: this Shadertoy has buffer tabs (Buffer A-D), each one a shader of "
                   "its own that feeds the next, and the text does not say where each one starts. " + HOW_TO_TABS)
    if re.search(r"\bvoid\s+mainSound\s*\(", c):
        out.append("it has a mainSound: sound shaders are not rendered (connect a sound to 'audio' instead).")
    if re.search(r"\bsamplerCube\b", c):
        out.append("it reads a cubemap (samplerCube): only flat pictures can be connected as iChannel0-3.")
    if re.search(r"\bsampler3D\b", c):
        out.append("it reads a 3D texture (sampler3D): only flat pictures can be connected as iChannel0-3.")
    return out


def helpers(body: str) -> str:
    """The 3D helpers (SDF) this shader uses and does not define itself, as GLSL to put in front of it."""
    used = _no_comments(body)

    def own(name):                                    # the shader brings its own
        return re.search(rf"\b(?:float|int|bool|void|vec[234]|mat[234])\s+{name}\s*\(|#[ \t]*define\s+{name}\b", used)

    want = {name for name, _, _ in SDF if name != "map" and re.search(rf"\b{name}\b", used) and not own(name)}
    walks = [name for name, needs, _ in SDF if name in want and "map" in needs]
    if walks:
        if not re.search(r"\bfloat\s+map\s*\(\s*(?:(?:const|in|highp|mediump|lowp)\s+)*vec3\s+\w+\s*\)", used):
            raise ShaderError(f"{', '.join(walks)} walk{'s' if len(walks) == 1 else ''} through your scene and "
                              f"need{'s' if len(walks) == 1 else ''} it as 'float map(vec3 p)': a function that gives "
                              "the distance from p to the nearest surface.")
        want.add("map")
    return "".join(code + "\n" for name, _, code in SDF if name in want)


def parse_values(text: str, controls: list) -> tuple:
    """'speed = 2, glow = 0.4' (commas or lines) -> ({name: value} for every control, notes about what did not fit)."""
    vals = {c.name: c.default for c in controls}
    by_lower = {c.name.lower(): c for c in controls}
    by_lower.update({c.name.lower().rstrip("_"): c for c in controls})      # 'length_' in the code is 'length' to type
    notes = []
    colours = {c.colour.lower(): c.colour for c in controls if c.colour}
    for part in re.split(r"[,;\n]+", str(text or "")):
        col = re.fullmatch(r"\s*(\w+)\s*[=:]\s*#([0-9a-fA-F]{6})\s*(?:#.*)?", part)
        if col and col.group(1).lower() in colours:   # a colour: 'ink = #ff8800'
            for i, ch in enumerate("rgb"):
                vals[f"{colours[col.group(1).lower()]}_{ch}"] = int(col.group(2)[2 * i:2 * i + 2], 16) / 255.0
            continue
        part = part.split("#", 1)[0].strip()
        if not part:
            continue
        m = re.fullmatch(rf"(\w+)\s*[=:]\s*({_NUM})", part)
        if not m:
            notes.append(f"controls: '{part}' is not 'name = number', ignored")
            continue
        c = by_lower.get(m.group(1).lower())
        if c is None:
            have = ", ".join(x.name for x in controls) or "none"
            notes.append(f"controls: the shader has no '{m.group(1)}' (it has: {have})")
            continue
        vals[c.name] = float(m.group(2))
    return vals, notes


HOW_TO_TABS = ("A Shadertoy with Buffer tabs: paste every tab under a line that names it, '// Common' (if it has "
               "one), '// Buffer A', '// Buffer B' ... and '// Image' last. A tab reads Buffer A on iChannel0, Buffer B "
               "on iChannel1 and so on; where the Shadertoy is wired differently, say it on the tab's line: "
               "'// Buffer A: 0 = A, 1 = noise' or '// Image: 0 = B, 1 = image0' (noise = the built-in noise, "
               "image0 - image3 = the node's pictures).")
_TAB = re.compile(r"^(common|image|buf(?:fer)?[ \t]*([a-d]))[ \t]*(?::(.*))?$", re.I)
_WIRE = re.compile(r"(?:ichannel)?[ \t]*([0-3])[ \t]*=[ \t]*(?:buf(?:fer)?[ \t]*([a-d])|([a-d])|(noise)|image[ \t]*([0-3]))(?![\w.])", re.I)


@dataclass
class Pass:
    name: str                         # A - D
    source: str                       # the complete fragment shader of the tab
    inputs: dict                      # {channel: "A" - "D" | "noise" | "image0" - "image3"}


def _tab(line: str):
    """A line that names a Shadertoy tab ('// Buffer A', '// --- Image ---', '// Buffer B: 0 = A, 1 = noise')
    -> (common | image | A - D, {channel: what it reads}); None for any other line."""
    m = re.match(r"^[ \t]*//(.*)$", line)
    t = _TAB.match(m.group(1).strip(" \t-=*#[]<>")) if m else None
    if not t:
        return None
    rest = t.group(3) or ""
    if _WIRE.sub("", rest).strip(" \t,;"):            # more words: an ordinary comment
        return None
    wires = {int(c): (b or a).upper() if (b or a) else ("noise" if n else f"image{i}") for c, b, a, n, i in _WIRE.findall(rest)}
    return (t.group(2).upper() if t.group(2) else t.group(1).lower()), wires


def tabs(code: str):
    """The pasted text cut into the tabs it names: [(name, {channel: source}, the line its text starts at, text)],
    or None when it names no Buffer tab (one shader, as before)."""
    out, name, wires, start, lines = [], "common", {}, 1, []
    found = False
    for n, line in enumerate(code.split("\n"), 1):
        t = _tab(line)
        if t:
            out.append((name, wires, start, "\n".join(lines)))
            (name, wires), start, lines = t, n + 1, []
            found = found or len(name) == 1
        else:
            lines.append(line)
    out.append((name, wires, start, "\n".join(lines)))
    return out if found else None


def _clean(text: str) -> str:
    """One tab's text as it goes into the shader: the uniforms the header has are dropped, old texture calls renamed."""
    def drop_builtin(m):
        return "" if m.group(1) in _BUILTIN or re.fullmatch(r"u_image\d", m.group(1)) else m.group(0)

    text = _UNIFORM.sub(drop_builtin, text)
    text = re.sub(r"^[ \t]*#[ \t]*(?:version|extension)[^\n]*$", "", text, flags=re.M)
    text = re.sub(r"\btexture2D\s*\(", "texture(", text)
    return re.sub(r"\btextureCube\s*\(", "texture(", text)


def _translate_tabs(code: str, parts: list) -> Translated:
    """A Shadertoy with Buffer tabs -> the Image tab's shader, and one shader per Buffer tab (Translated.buffers)."""
    problems = unsupported(code, tabbed=True)
    if problems:
        raise ShaderError("this shader cannot run here: " + " Also: ".join(problems))
    common = [(start, text) for name, _w, start, text in parts if name == "common" and _no_comments(text).strip()]
    named = [(name, w, start, text) for name, w, start, text in parts if name != "common"]
    letters = [name for name, _w, _s, _t in named if name != "image"]
    for name in dict.fromkeys(n for n, *_ in named):
        if sum(1 for n, *_ in named if n == name) > 1:
            raise ShaderError(f"the tab '{'Image' if name == 'image' else 'Buffer ' + name}' is there twice.")
    if not any(n == "image" for n, *_ in named):
        raise ShaderError("there is no Image tab: the picture comes from the tab under a line '// Image'. " + HOW_TO_TABS)
    notes, built, uses, channels, says = [], {}, set(), set(), []
    for name, wires, start, text in named:
        label = "Image" if name == "image" else f"Buffer {name}"
        if len(_TOY.findall(_no_comments(text))) != 1:
            raise ShaderError(f"the tab {label} must have one 'void mainImage(...)'; it has "
                              f"{len(_TOY.findall(_no_comments(text)))}. " + HOW_TO_TABS)
        body = "".join(f"#line {s}\n{_clean(t)}\n" for s, t in common) + f"#line {start}\n{_clean(text)}\n"
        body, picked = _colours(body)
        used = _no_comments(body)
        reads = {}
        for c in range(CHANNELS):
            if not re.search(rf"\biChannel{c}\b", used):
                continue
            src = wires.get(c) or ("ABCD"[c] if "ABCD"[c] in letters else f"image{c}")
            if len(src) == 1 and src not in letters:
                raise ShaderError(f"{label} reads Buffer {src} on iChannel{c}, and there is no tab '// Buffer {src}'.")
            reads[c] = src
            if src.startswith("image"):
                channels.add(int(src[5:]))
        says.append(f"{label} reads " + (", ".join(f"iChannel{c} = {('Buffer ' + s) if len(s) == 1 else s}" for c, s in reads.items()) or "no picture"))
        uses |= {n for n in ("iMouse", "iDate", "kLoop", "kLevel", "kLow", "kMid", "kHigh", "kBeat", "kBeats", "kField", "kMask", "kPrev", "kDepth", *VECTOR)
                 if re.search(rf"\b{n}\b", used)}
        built[name] = (HEADER + picked + helpers(body) + body + (FOOTER_TOY if name == "image" else FOOTER_BUFFER), reads)
    if "kPrev" in uses:
        raise ShaderError("kPrev (the frame before) cannot be used together with Buffer tabs: a Buffer tab that reads "
                          "itself is that already.")
    notes.append("; ".join(says) + ". Wired differently on shadertoy.com? Say it on the tab's line, e.g. '// Buffer A: 1 = noise'.")
    if "iMouse" in uses:
        notes.append("iMouse is used: there is no mouse here, it reads as never pressed (0, 0, 0, 0)")
    optional = {int(n) for n in re.findall(r"image(\d)", " ".join(re.findall(r"^//\s*optional:(.*)$", code, flags=re.M)))}
    return Translated(built["image"][0], "shadertoy", parse_controls(code), notes, sorted(channels), optional, uses,
                      [Pass(n, built[n][0], built[n][1]) for n in letters], built["image"][1])


def translate(code: str) -> Translated:
    """A Shadertoy or a plain fragment shader -> the complete GLSL ES 3.00 fragment shader and what it needs."""
    code = (code or "").replace("\r\n", "\n").replace("\r", "\n")
    if len(code) > MAX_CODE:
        raise ShaderError(f"the shader text is {len(code) // 1000} thousand characters long; {MAX_CODE // 1000} thousand is the limit.")
    bare = _no_comments(code)
    if re.fullmatch(r"\s*(https?://|www\.)\S+\s*", code):
        raise ShaderError("this is a link, not the shader's code, and the node does not download anything. " + HOW_TO_COPY)
    if not bare.strip():
        raise ShaderError("the shader is empty.")
    parts = tabs(code)
    if parts:
        return _translate_tabs(code, parts)
    if _TOY.search(bare):
        kind = "shadertoy"
    elif _MAIN.search(bare):
        kind = "glsl"
    else:
        raise ShaderError("this does not look like a whole shader: it has no 'void mainImage(...)' and no 'void main()'. "
                          "Part of the code may be missing. " + HOW_TO_COPY)
    problems = unsupported(code)
    if problems:
        raise ShaderError("this shader cannot run here: " + " Also: ".join(problems))

    controls = parse_controls(code)
    notes = []

    def drop_builtin(m):                              # the header declares these already
        return "" if m.group(1) in _BUILTIN or re.fullmatch(r"u_image\d", m.group(1)) else m.group(0)

    body = _UNIFORM.sub(drop_builtin, code)
    body = re.sub(r"^[ \t]*#[ \t]*(?:version|extension)[^\n]*$", "", body, flags=re.M)      # what an extension gave is built in here
    body = re.sub(r"\btexture2D\s*\(", "texture(", body)
    body = re.sub(r"\btextureCube\s*\(", "texture(", body)
    head = HEADER
    if kind == "glsl":
        head += HEADER_GLSL
        body = re.sub(r"^[ \t]*(?:in|varying)\s+vec2\s+v_texCoord\s*;[^\n]*$", "", body, flags=re.M)
        outs = re.findall(r"^[ \t]*(?:layout\s*\([^)]*\)\s*)?out\s+vec4\s+(\w+)\s*;", body, flags=re.M)
        body = re.sub(r"^[ \t]*(?:layout\s*\([^)]*\)\s*)?out\s+vec4\s+\w+\s*;[^\n]*$", "", body, flags=re.M)
        for name in dict.fromkeys(outs):
            if name not in ("fragColor0", "kOut"):
                head += f"#define {name} kOut\n"
        if len(set(outs)) > 1:
            notes.append("the shader writes several outputs: only the first one is used")
        body = re.sub(r"\bgl_FragColor\b", "kOut", body)
        at = _MAIN.search(_no_comments(body))          # the real one, not a 'void main(' in a comment
        body = body[:at.start()] + "void kUserMain(" + body[at.end():]
    body, picked = _colours(body)
    source = head + picked + helpers(body) + "#line 1\n" + body + "\n" + (FOOTER_TOY if kind == "shadertoy" else FOOTER_GLSL)

    used = _no_comments(body)
    channels = [i for i in range(CHANNELS) if re.search(rf"\b(?:iChannel|u_image){i}\b", used)]
    uses = {n for n in ("iMouse", "iDate", "kLoop", "kLevel", "kLow", "kMid", "kHigh", "kBeat", "kBeats", "kField", "kMask", "kPrev", "kDepth", *VECTOR)
            if re.search(rf"\b{n}\b", used)}
    if "iMouse" in uses:
        notes.append("iMouse is used: there is no mouse here, it reads as never pressed (0, 0, 0, 0)")
    optional = {int(n) for n in re.findall(r"image(\d)", " ".join(re.findall(r"^//\s*optional:(.*)$", code, flags=re.M)))}
    return Translated(source, kind, controls, notes, channels, optional, uses)


@functools.lru_cache(maxsize=64)
def translated(code: str) -> Translated:
    """translate(), remembered: the director asks for the same shader at every frame. Read only."""
    return translate(code)


def explain(error: str) -> str:
    """A GL compile log as short lines: 'line 12: what is wrong' (line numbers are those of the pasted shader)."""
    out = []
    for line in re.sub(r"\n(?!\s*(?:ERROR|WARNING):)", " ", str(error).strip()).splitlines():      # a message can hold a line break itself
        m = re.match(r"\s*ERROR:\s*\d+:(\d+):\s*(.*)", line)
        if m:
            out.append(f"line {m.group(1)}: {m.group(2).strip()}")
    text = "\n".join(out[:12]) or str(error).strip()
    if re.search(r"texture(Lod|Grad)?'? ?: no matching overloaded function|samplerCube|sampler3D", text):
        text += ("\nThis usually means the shader expects a cubemap or a 3D texture on one of its channels on "
                 "Shadertoy; only flat pictures work here.")
    elif len(re.findall(r"undeclared identifier", text)) > 3:
        text += "\nMany unknown names usually mean a part is missing: the shader's 'Common' tab goes above the 'Image' tab."
    return text


# --------------------------------------------------------------------------
# the presets: kubakub/shaders/*.glsl
# --------------------------------------------------------------------------

# The types of preset, in menu order: (key, title, what they are and what they need). A preset names its own in a
# '// type: key' line; the pickers on the node and in the director show the presets under these headings.
TYPES = (
    ("mine", "my presets", "Presets you saved yourself (adjust, then save as my preset). They are files in ComfyUI's user folder, under kubakub/shaders."),
    ("pattern", "patterns", "Moving pictures made from nothing: smoke, cells, stripes, light. They need no input; a picture on image0 goes under them (on_picture)."),
    ("space", "3d scenes", "A space you look into: forms, columns, corridors, fog. They need no input (a picture on image0 goes under them); they cost more to draw, and some give a depth."),
    ("regions", "on your regions", "Light that follows your masks and regions: connect kubakub mask field (the sample facade without one). For projection mapping."),
    ("look", "picture looks", "Your picture on image0, redrawn: cartoon, blueprint, dots, big pixels (the sample facade without one)."),
    ("signal", "screen and tape", "Your picture on image0 as on old video gear: a VHS tape, a tube television, a composite cable."),
    ("lens", "grade and lens", "Your picture on image0, finished: blur, colour grading, glow, and what a lens does to it (bend, swirl, mirrors). Most hold still unless you ask them to move."),
    ("relief", "picture with depth", "Your picture on image0 gets depth: relief, shadows, a wire mesh, points. A depth pass on image1 makes it true; without one its brightness is the depth."),
    ("time", "over time", "They build every frame on the one before: a clip that smears and leaves trails, a pattern that grows. On the node only."),
)
_TYPE_ORDER = {key: i for i, (key, _, _) in enumerate(TYPES)}


USER_DIR = None                                      # where saved presets live: set by the node (ComfyUI's user folder)


def preset_type(code: str) -> str:
    """The '// type: key' line of a preset -> a key of TYPES ('pattern' when it names none or one that is unknown)."""
    m = re.search(r"^//\s*type:\s*(\w+)", code, flags=re.M)
    return m.group(1) if m and m.group(1) in _TYPE_ORDER and m.group(1) != "mine" else "pattern"


def is_mine(name: str) -> bool:
    """A preset the user saved (a file in USER_DIR), not one of the pack's."""
    path = _NAMES.get(name) or presets().get(name)
    return bool(path and USER_DIR and os.path.dirname(os.path.abspath(path)) == os.path.abspath(USER_DIR))


def type_of(name: str) -> str:
    """The type a preset is listed under: 'mine' for a saved one, else what its file says."""
    return "mine" if is_mine(name) else preset_type(preset_code(name))


def with_defaults(code: str, values: dict) -> str:
    """The code with the values written into its control lines as their new defaults (the ranges and words stay):
    `uniform float warp; // 1 0 4 folds` and {warp: 2.5} -> `uniform float warp; // 2.5 0 4 folds`."""
    by = {c.name: c for c in parse_controls(code)}

    def line(m):
        c = by.get(m.group(1))
        if c is None or c.name not in values:
            return m.group(0)
        v = min(max(float(values[c.name]), c.lo), c.hi)
        return f"{m.group(0).split('//')[0].rstrip().split(';')[0]};{' ' * 6}// {v:g} {c.lo:g} {c.hi:g}{'   ' + c.about if c.about else ''}"

    def colour(m):
        n = m.group(1)
        if f"{n}_r" not in by or not any(f"{n}_{ch}" in values for ch in "rgb"):
            return m.group(0)
        now = {f"{n}_{ch}": values.get(f"{n}_{ch}", by[f"{n}_{ch}"].default) for ch in "rgb"}
        return f"{m.group(0).split('//')[0].rstrip()}{' ' * 6}// {colour_hex(now, n)}{'   ' + m.group(3).strip() if m.group(3).strip() else ''}"
    return _COLOUR.sub(colour, _CONTROL.sub(line, code))


def preset_file_name(name: str) -> str:
    """A name typed by the user -> the name a saved preset gets (lowercase letters, digits, spaces; it starts with a
    letter), or a ShaderError that says what is wrong."""
    name = re.sub(r"\s+", " ", str(name or "").strip().lower().replace("_", " "))
    if not re.fullmatch(r"[a-z][a-z0-9 ]{0,39}", name):
        raise ShaderError("a preset name is 1 to 40 letters, digits and spaces, and starts with a letter.")
    return name


def save_preset(name: str, code: str, values: dict = None) -> str:
    """Save a shader as a preset of the user's own (a .glsl file in USER_DIR), the values as its defaults -> its name.
    One of the pack's presets cannot be overwritten; one of the user's own with that name is."""
    if not USER_DIR:
        raise ShaderError("there is no folder for saved presets here.")
    name = preset_file_name(name)
    found = presets()
    if name in found and not is_mine(name):
        raise ShaderError(f"'{name}' is a built-in preset: pick another name.")
    code = str(code or "")
    translate(code)                                   # only a shader that runs is saved
    code = with_defaults(code, values or {})
    code = re.sub(r"^//[ \t]*type:.*\r?\n", "", code, flags=re.M)      # its type is 'my presets' now
    if not about(code):
        code = f"// about: {name}, a preset of my own.\n" + code
    os.makedirs(USER_DIR, exist_ok=True)
    path = os.path.join(USER_DIR, name.replace(" ", "_") + ".glsl")
    part = path + ".part"
    with open(part, "w", encoding="utf-8", newline="\n") as f:
        f.write(code if code.endswith("\n") else code + "\n")
    os.replace(part, path)
    presets()
    return name


def delete_preset(name: str) -> None:
    """Remove a preset the user saved. The pack's own presets stay."""
    if not is_mine(name):
        raise ShaderError(f"'{name}' is not one of your saved presets.")
    os.remove(_NAMES[name])
    presets()


def presets() -> dict:
    """{name: path}; the name is the file name with spaces. The menu order: by type (TYPES), then by file name."""
    if not os.path.isdir(SHADER_DIR):
        return {}
    found = {re.sub(r"^\d+_", "", os.path.splitext(f)[0]).replace("_", " "): os.path.join(SHADER_DIR, f)
             for f in sorted(os.listdir(SHADER_DIR)) if f.endswith(".glsl")}
    mine = set()
    if USER_DIR and os.path.isdir(USER_DIR):          # the user's own, saved from the node; a built-in name stays the built-in
        for f in sorted(os.listdir(USER_DIR)):
            name = os.path.splitext(f)[0].replace("_", " ")
            if f.endswith(".glsl") and name not in found:
                found[name] = os.path.join(USER_DIR, f)
                mine.add(name)

    def order(item):
        try:
            return _TYPE_ORDER["mine" if item[0] in mine else preset_type(_read(item[1]))]
        except OSError:
            return 0
    found = dict(sorted(found.items(), key=order))   # a stable sort: the file order stays within a type
    _NAMES.clear()
    _NAMES.update(found)
    return found


_NAMES = {}                                          # the last listing: a known name needs no new one


_CODE = {}                                           # path -> (mtime, text): the director asks at every frame


def _read(path: str) -> str:
    mtime = os.path.getmtime(path)
    hit = _CODE.get(path)
    if hit is None or hit[0] != mtime:
        with open(path, encoding="utf-8") as f:
            hit = _CODE[path] = (mtime, f.read())
    return hit[1]


def preset_code(name: str) -> str:
    path = _NAMES.get(name) or presets().get(name)
    if not path:
        raise ShaderError(f"no preset '{name}' (there are: {', '.join(presets()) or 'none'}).")
    try:
        return _read(path)
    except OSError:                                  # the file went away since the last listing
        _NAMES.clear()
        raise ShaderError(f"no preset '{name}' (there are: {', '.join(presets()) or 'none'}).") from None


def shader_text(preset="", code="") -> str:
    """Pasted code when there is any, else the preset's."""
    return code if (code or "").strip() else preset_code(preset)


def sample_field(code: str) -> bool:
    """The '// sample: field' line of a preset: it shows nothing without a field of regions, so with none connected
    the node gives it the windows of the sample facade."""
    return bool(re.search(r"^//\s*sample:\s*field\b", code, flags=re.M))


def about(code: str) -> str:
    """The '// about: ...' line of a preset: one sentence on what it draws."""
    m = re.search(r"^//\s*about:\s*(.+)$", code, flags=re.M)
    return m.group(1).strip() if m else ""
