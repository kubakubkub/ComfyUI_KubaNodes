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
                kField(uv) and kMask(uv) (the grey ramp and the mask connected to the node), kUV
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
MAX_CONTROLS = 32

FLOATS = ("iTime", "iTimeDelta", "iFrameRate", "iSampleRate", "kLoop", "kSeconds", "kLevel", "kLow", "kMid", "kHigh",
          "kBeat", "kBeats", "kAx", "kAy", "kBx", "kBy", "kVector", "kVecField", "kBend")
VECTOR = frozenset({"kFrom", "kTo", "kDir", "kVector"})      # a shader that reads one of these follows a drawn arrow
SOUND = frozenset({"kLevel", "kLow", "kMid", "kHigh", "kBeat", "kBeats"})      # a shader that reads one of these listens
_BUILTIN = set(FLOATS) | {"iResolution", "iFrame", "iMouse", "iDate", "iChannelTime", "iChannelResolution",
                          "kFieldTex", "kPrevTex", "kClip", "u_resolution", "kFrom", "kTo", "kDir", "kDirOf", "kMixTex", "kShift"} | {f"iChannel{i}" for i in range(CHANNELS)}

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
layout(location = 0) out vec4 kOut;
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
}
"""

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

FOOTER_GLSL = """
void main() {
    kStart();
    kUserMain();
    kFinish();
}
"""

_NUM = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"
_CONTROL = re.compile(r"^[ \t]*uniform\s+float\s+(\w+)\s*;[ \t]*(?://(.*))?$", re.M)
_UNIFORM = re.compile(r"^[ \t]*uniform\s+\w+\s+(\w+)\s*(?:\[\s*\d+\s*\])?\s*;.*$", re.M)
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


@dataclass
class Translated:
    source: str                       # the complete fragment shader
    kind: str                         # shadertoy | glsl
    controls: list = field(default_factory=list)
    notes: list = field(default_factory=list)
    channels: list = field(default_factory=list)      # which of iChannel0-3 the shader reads
    optional: set = field(default_factory=set)        # those it has an answer of its own for ('// optional: image1')
    uses: set = field(default_factory=set)            # which kubakub / Shadertoy names it reads

    @property
    def listens(self) -> bool:
        return bool(self.uses & SOUND)

    @property
    def follows(self) -> bool:
        """It does something with a drawn arrow: it reads the arrow, or the field (which the arrow becomes)."""
        return bool(self.uses & (VECTOR | {"kField"}))


def _no_comments(code: str) -> str:
    """The code with comments blanked out (same length and lines), for looking at what it really uses."""
    def blank(m):
        return re.sub(r"[^\n]", " ", m.group(0))
    return re.sub(r"//[^\n]*|/\*.*?\*/", blank, code, flags=re.S)


def parse_controls(code: str) -> list:
    """`uniform float speed; // 1 0 4` -> Control(speed, 1, 0, 4). The numbers are default, min, max; all optional."""
    out = []
    for m in _CONTROL.finditer(code):
        name = m.group(1)
        if name in _BUILTIN or re.fullmatch(r"u_(float|int|bool)\d+", name):
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


def unsupported(code: str) -> list:
    """What a pasted Shadertoy uses that one fragment shader cannot do here, in words."""
    c = _no_comments(code)
    out = []
    if len(_TOY.findall(c)) > 1:
        out.append("it has more than one mainImage: this Shadertoy has buffer tabs (Buffer A-D), each one a shader of "
                   "its own that feeds the next. Only single-tab shaders (Image, plus Common pasted above it) work.")
    if re.search(r"\bvoid\s+mainSound\s*\(", c):
        out.append("it has a mainSound: sound shaders are not rendered (connect a sound to 'audio' instead).")
    if re.search(r"\bsamplerCube\b", c):
        out.append("it reads a cubemap (samplerCube): only flat pictures can be connected as iChannel0-3.")
    if re.search(r"\bsampler3D\b", c):
        out.append("it reads a 3D texture (sampler3D): only flat pictures can be connected as iChannel0-3.")
    return out


def parse_values(text: str, controls: list) -> tuple:
    """'speed = 2, glow = 0.4' (commas or lines) -> ({name: value} for every control, notes about what did not fit)."""
    vals = {c.name: c.default for c in controls}
    by_lower = {c.name.lower(): c for c in controls}
    by_lower.update({c.name.lower().rstrip("_"): c for c in controls})      # 'length_' in the code is 'length' to type
    notes = []
    for part in re.split(r"[,;\n]+", str(text or "")):
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


def translate(code: str) -> Translated:
    """A Shadertoy or a plain fragment shader -> the complete GLSL ES 3.00 fragment shader and what it needs."""
    code = (code or "").replace("\r\n", "\n").replace("\r", "\n")
    bare = _no_comments(code)
    if re.fullmatch(r"\s*(https?://|www\.)\S+\s*", code):
        raise ShaderError("this is a link, not the shader's code, and the node does not download anything. " + HOW_TO_COPY)
    if not bare.strip():
        raise ShaderError("the shader is empty.")
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
    body = re.sub(r"^[ \t]*#version[^\n]*$", "", body, flags=re.M)
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
        body = _MAIN.sub("void kUserMain(", body, count=1)
    source = head + "#line 1\n" + body + "\n" + (FOOTER_TOY if kind == "shadertoy" else FOOTER_GLSL)

    used = _no_comments(body)
    channels = [i for i in range(CHANNELS) if re.search(rf"\b(?:iChannel|u_image){i}\b", used)]
    uses = {n for n in ("iMouse", "iDate", "kLoop", "kLevel", "kLow", "kMid", "kHigh", "kBeat", "kBeats", "kField", "kMask", "kPrev", *VECTOR)
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
    for line in str(error).splitlines():
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

def presets() -> dict:
    """{name: path}; the name is the file name with spaces. Sorted, the file order is the menu order."""
    if not os.path.isdir(SHADER_DIR):
        return {}
    found = {re.sub(r"^\d+_", "", os.path.splitext(f)[0]).replace("_", " "): os.path.join(SHADER_DIR, f)
             for f in sorted(os.listdir(SHADER_DIR)) if f.endswith(".glsl")}
    _NAMES.clear()
    _NAMES.update(found)
    return found


_NAMES = {}                                          # the last listing: a known name needs no new one


_CODE = {}                                           # path -> (mtime, text): the director asks at every frame


def preset_code(name: str) -> str:
    path = _NAMES.get(name) or presets().get(name)
    if not path:
        raise ShaderError(f"no preset '{name}' (there are: {', '.join(presets()) or 'none'}).")
    try:
        mtime = os.path.getmtime(path)
    except OSError:                                  # the file went away since the last listing
        _NAMES.clear()
        raise ShaderError(f"no preset '{name}' (there are: {', '.join(presets()) or 'none'}).") from None
    hit = _CODE.get(path)
    if hit is None or hit[0] != mtime:
        with open(path, encoding="utf-8") as f:
            hit = _CODE[path] = (mtime, f.read())
    return hit[1]


def shader_text(preset="", code="") -> str:
    """Pasted code when there is any, else the preset's."""
    return code if (code or "").strip() else preset_code(preset)


def about(code: str) -> str:
    """The '// about: ...' line of a preset: one sentence on what it draws."""
    m = re.search(r"^//\s*about:\s*(.+)$", code, flags=re.M)
    return m.group(1).strip() if m else ""
