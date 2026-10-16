"""
osl.py

kubakub shader, OSL: an Open Shading Language pattern shader (as written for Blender Cycles, Houdini, Arnold ...)
translated into the Shadertoy GLSL that kubakub shader runs (kubakub/shadertoy.py), so a pasted OSL shader gets the
node's sliders, colours, masks, regions and sound like any other.

What it becomes:
- the shader's inputs: `float` / `int` with a number -> a slider (`[[ float min, float max, string help ]]` give
  its range and text), `color` with numbers -> a colour to pick; a float input named time / frame follows the clock;
  other inputs (`point Pos = P`, an expression) keep their value; a string input is put in where it is used.
- the globals: P = the place in the picture (x across, y up, the picture height = 1, z = 0), u / v = 0-1 across and up,
  N = Ng = (0, 0, 1) towards the viewer, I = (0, 0, -1), time = the clock.
- the picture shown: the first `output color`, else `Ci`, else the first `output float` as grey; a line
  `// output: Name` picks another one.
- closures (diffuse(N), emission() ...) count as white: `Ci = col * emission()` shows col. The surface, the light and
  the camera of a renderer are not there, so a surface shader shows the colour it computes, not its look in a render.
- noise(): OSL's types (perlin, uperlin, snoise, cell, hash, simplex, gabor) as GLSL of our own: the same kind of
  noise, not the same numbers as Blender or Houdini, so a pattern looks alike but not identical.
Not translated: displacement and volume shaders, textures by file name (texture("file.png", ...) reads image0),
getattribute (gives 0), trace, spline, structs.

The number rule: OSL turns 1 into 1.0 where a float is needed, GLSL ES does not. Here every int is a float (int
variables, literals, loops), indices are cast back with int(), `a % b` becomes mod(a, b); `int(x)` becomes trunc(x).
"""

from __future__ import annotations

import re

_HEAD = re.compile(r"(?<![\w.])(shader|surface|displacement|volume)\s+(\w+)\s*\(")
_NUM = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"
_VEC = ("color", "point", "vector", "normal")
OSL_GLOBALS = ("P", "N", "Ng", "I", "u", "v", "dPdu", "dPdv", "time", "Ci", "dtime", "dPdtime", "Ps")

# GLSL ES 3.00 words that are plain names in OSL: a variable of that name gets a _ (input -> input_)
_GLSL_WORDS = ("input sample filter common partition active half fixed inline noinline volatile public static extern "
               "external interface long short double unsigned superp sizeof cast namespace using class union enum "
               "typedef template this packed goto smooth flat centroid layout uniform varying attribute patch "
               "subroutine precise invariant highp mediump lowp precision buffer shared coherent readonly writeonly "
               "resource in out bool ivec2 ivec3 ivec4 uvec2 uvec3 uvec4 bvec2 bvec3 bvec4 vec2 vec3 vec4 mat2 mat3 "
               "uint lowp sampler2D").split()

_RENAME = {"fabs": "abs", "atan2": "atan", "filterwidth": "fwidth", "Dx": "dFdx", "Dy": "dFdy", "fmod": "kosl_fmod",
           "hypot": "kosl_hypot", "log10": "kosl_log10", "luminance": "kosl_luminance", "select": "kosl_select",
           "aastep": "kosl_aastep", "linearstep": "kosl_linearstep", "smooth_linearstep": "kosl_smooth_linearstep",
           "erf": "kosl_erf", "erfc": "kosl_erfc", "blackbody": "kosl_blackbody", "rotate": "kosl_rotate",
           "sincos": "kosl_sincos", "wavelength_color": "kosl_wavelength", "calculatenormal": "kosl_calculatenormal"}

_CLOSURES = ("diffuse oren_nayar phong ward microfacet reflection refraction translucent emission background "
             "principled_bsdf generalized_schlick_bsdf dielectric_bsdf conductor_bsdf sheen_bsdf hair_bsdf "
             "subsurface_bssrdf bssrdf toon diffuse_toon glossy_toon ashikhmin_velvet velvet "
             "henyey_greenstein absorption").split()
_CLOSURES_DARK = ("transparent", "holdout", "debug")

# --------------------------------------------------------------------------------------------- the GLSL library
# name -> code; a part goes in only when the translated shader uses its name (or a part that needs it).
_LIB = {}
_NEEDS = {}


def _lib(name, code, needs=()):
    _LIB[name] = code
    _NEEDS[name] = tuple(needs)


_lib("kv3", "vec3 kv3(float x) { return vec3(x); }\nvec3 kv3(vec3 x) { return x; }")
_lib("kosl_hash", """uint kosl_hash(uint x) { x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; x ^= x >> 16; return x; }
uint kosl_hash3(ivec3 p, uint s) { return kosl_hash(uint(p.x) + kosl_hash(uint(p.y) + kosl_hash(uint(p.z) + s * 0x9e3779b9u))); }
float kosl_unit(uint h) { return float(h >> 8) / 16777216.0; }""")
_lib("kosl_perlin", """float kosl_grad(ivec3 c, vec3 f, uint s) {
    uint h = kosl_hash3(c, s) & 15u;
    float a = h < 8u ? f.x : f.y, b = h < 4u ? f.y : (h == 12u || h == 14u ? f.x : f.z);
    return ((h & 1u) == 0u ? a : -a) + ((h & 2u) == 0u ? b : -b);
}
float kosl_perlin(vec3 p, uint s) {                    // gradient noise, about -1 to 1
    vec3 i = floor(p), f = p - i;
    ivec3 c = ivec3(i);
    vec3 w = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
    float x00 = mix(kosl_grad(c, f, s), kosl_grad(c + ivec3(1, 0, 0), f - vec3(1, 0, 0), s), w.x);
    float x10 = mix(kosl_grad(c + ivec3(0, 1, 0), f - vec3(0, 1, 0), s), kosl_grad(c + ivec3(1, 1, 0), f - vec3(1, 1, 0), s), w.x);
    float x01 = mix(kosl_grad(c + ivec3(0, 0, 1), f - vec3(0, 0, 1), s), kosl_grad(c + ivec3(1, 0, 1), f - vec3(1, 0, 1), s), w.x);
    float x11 = mix(kosl_grad(c + ivec3(0, 1, 1), f - vec3(0, 1, 1), s), kosl_grad(c + ivec3(1, 1, 1), f - vec3(1, 1, 1), s), w.x);
    return 0.97 * mix(mix(x00, x10, w.y), mix(x01, x11, w.y), w.z);
}
float kosl_perlin4(vec3 p, float t, uint s) {          // a fourth dimension: two slices, blended
    float i = floor(t), f = t - i, w = f * f * (3.0 - 2.0 * f);
    return mix(kosl_perlin(p + vec3(0.0, 0.0, i * 31.17), s), kosl_perlin(p + vec3(0.0, 0.0, (i + 1.0) * 31.17), s), w);
}""", ("kosl_hash",))
_lib("kosl_cellh", """float kosl_cellh(vec3 p, uint s) { return kosl_unit(kosl_hash3(ivec3(floor(p)), s)); }
float kosl_cellh4(vec3 p, float t, uint s) { return kosl_unit(kosl_hash3(ivec3(floor(p)), s + uint(int(floor(t))) * 7919u)); }
float kosl_hashh(vec3 p, uint s) { uvec3 b = floatBitsToUint(p); return kosl_unit(kosl_hash(b.x ^ kosl_hash(b.y ^ kosl_hash(b.z + s)))); }
float kosl_hashh4(vec3 p, float t, uint s) { return kosl_hashh(p + vec3(t * 1.618), s); }""", ("kosl_hash",))


def _noise_family(name, base, base4, post, needs):
    """kosl_<name>_f / kosl_<name>_v for float, (float, float), point, (point, float): the OSL overloads."""
    f = (f"float kosl_{name}_f(vec3 p) {{ return {post.format(f'{base}(p, 0u)')}; }}\n"
         f"float kosl_{name}_f(vec3 p, float t) {{ return {post.format(f'{base4}(p, t, 0u)')}; }}\n"
         f"float kosl_{name}_f(float x) {{ return kosl_{name}_f(vec3(x, 0.0, 0.0)); }}\n"
         f"float kosl_{name}_f(float x, float y) {{ return kosl_{name}_f(vec3(x, y, 0.0)); }}\n"
         f"vec3 kosl_{name}_v(vec3 p) {{ return vec3({post.format(f'{base}(p, 0u)')}, {post.format(f'{base}(p, 1u)')}, {post.format(f'{base}(p, 2u)')}); }}\n"
         f"vec3 kosl_{name}_v(vec3 p, float t) {{ return vec3({post.format(f'{base4}(p, t, 0u)')}, {post.format(f'{base4}(p, t, 1u)')}, {post.format(f'{base4}(p, t, 2u)')}); }}\n"
         f"vec3 kosl_{name}_v(float x) {{ return kosl_{name}_v(vec3(x, 0.0, 0.0)); }}\n"
         f"vec3 kosl_{name}_v(float x, float y) {{ return kosl_{name}_v(vec3(x, y, 0.0)); }}")
    _lib(f"kosl_{name}", f, needs)


_noise_family("sperlin", "kosl_perlin", "kosl_perlin4", "{}", ("kosl_perlin",))
_noise_family("uperlin", "kosl_perlin", "kosl_perlin4", "(0.5 + 0.5 * {})", ("kosl_perlin",))
_noise_family("cell", "kosl_cellh", "kosl_cellh4", "{}", ("kosl_cellh",))
_noise_family("hashn", "kosl_hashh", "kosl_hashh4", "{}", ("kosl_cellh",))
for _n in ("sperlin", "uperlin", "cell", "hashn"):
    _NEEDS[f"kosl_{_n}_f"] = _NEEDS[f"kosl_{_n}_v"] = (f"kosl_{_n}",)

_lib("kosl_hsv", """vec3 kosl_hsv(float h, float s, float v) {
    vec3 k = clamp(abs(mod(h * 6.0 + vec3(0.0, 4.0, 2.0), 6.0) - 3.0) - 1.0, 0.0, 1.0);
    return v * mix(vec3(1.0), k, s);
}
vec3 kosl_hsv(vec3 c) { return kosl_hsv(c.x, c.y, c.z); }""")
_lib("kosl_fmod", "float kosl_fmod(float a, float b) { return b == 0.0 ? 0.0 : a - b * trunc(a / b); }\n"
                  "vec3 kosl_fmod(vec3 a, vec3 b) { return vec3(kosl_fmod(a.x, b.x), kosl_fmod(a.y, b.y), kosl_fmod(a.z, b.z)); }\n"
                  "vec3 kosl_fmod(vec3 a, float b) { return kosl_fmod(a, vec3(b)); }")
_lib("kosl_hypot", "float kosl_hypot(float a, float b) { return sqrt(a * a + b * b); }\n"
                   "float kosl_hypot(float a, float b, float c) { return sqrt(a * a + b * b + c * c); }")
_lib("kosl_log10", "float kosl_log10(float x) { return log(x) / log(10.0); }\nvec3 kosl_log10(vec3 x) { return log(x) / log(10.0); }")
_lib("kosl_luminance", "float kosl_luminance(vec3 c) { return dot(c, vec3(0.2126, 0.7152, 0.0722)); }")
_lib("kosl_select", "float kosl_select(float a, float b, float c) { return c != 0.0 ? b : a; }\n"
                    "vec3 kosl_select(vec3 a, vec3 b, float c) { return c != 0.0 ? b : a; }\n"
                    "vec3 kosl_select(vec3 a, vec3 b, vec3 c) { return mix(a, b, vec3(notEqual(c, vec3(0.0)))); }")
_lib("kosl_aastep", "float kosl_aastep(float e, float x) { float w = max(fwidth(x), 1e-6); return smoothstep(e - w, e + w, x); }\n"
                    "float kosl_aastep(float e, float x, float w) { w = max(w, 1e-6); return smoothstep(e - w, e + w, x); }")
_lib("kosl_linearstep", "float kosl_linearstep(float a, float b, float x) { return clamp((x - a) / max(b - a, 1e-9), 0.0, 1.0); }")
_lib("kosl_smooth_linearstep", "float kosl_smooth_linearstep(float a, float b, float x, float e) { "
                               "return smoothstep(a - e, a + e, x) * (1.0 - smoothstep(b - e, b + e, x)); }")
_lib("kosl_erf", "float kosl_erf(float x) { float t = 1.0 / (1.0 + 0.3275911 * abs(x)); float y = 1.0 - (((((1.061405429 * t "
                 "- 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * exp(-x * x); "
                 "return sign(x) * y; }\nfloat kosl_erfc(float x) { return 1.0 - kosl_erf(x); }")
_NEEDS["kosl_erfc"] = ("kosl_erf",)
_lib("kosl_blackbody", """vec3 kosl_blackbody(float k) {                 // the colour of a glowing body at k kelvin, 1 at its brightest channel
    float t = clamp(k, 800.0, 40000.0) / 100.0;
    float r = t <= 66.0 ? 1.0 : clamp(1.292936 * pow(t - 60.0, -0.1332047), 0.0, 1.0);
    float g = t <= 66.0 ? clamp(0.3900816 * log(t) - 0.6318414, 0.0, 1.0) : clamp(1.1298909 * pow(t - 60.0, -0.0755148), 0.0, 1.0);
    float b = t >= 66.0 ? 1.0 : (t <= 19.0 ? 0.0 : clamp(0.5432068 * log(t - 10.0) - 1.1962541, 0.0, 1.0));
    return vec3(r, g, b);
}""")
_lib("kosl_wavelength", """vec3 kosl_wavelength(float nm) {                 // a visible wavelength as a colour
    float t = clamp((nm - 380.0) / 400.0, 0.0, 1.0);
    return clamp(vec3(abs(t * 6.0 - 3.0) - 1.0, 2.0 - abs(t * 6.0 - 2.0), 2.0 - abs(t * 6.0 - 4.0)), 0.0, 1.0)
           * smoothstep(0.0, 0.1, t) * smoothstep(1.0, 0.9, t);
}""")
_lib("kosl_rotate", """vec3 kosl_rotate(vec3 p, float a, vec3 b, vec3 c) {   // p turned by a radians about the axis from b to c
    vec3 k = normalize(c - b), q = p - b;
    return b + q * cos(a) + cross(k, q) * sin(a) + k * dot(k, q) * (1.0 - cos(a));
}
vec3 kosl_rotate(vec3 p, float a, vec3 axis) { return kosl_rotate(p, a, vec3(0.0), axis); }""")
_lib("kosl_sincos", "void kosl_sincos(float x, out float s, out float c) { s = sin(x); c = cos(x); }\n"
                    "void kosl_sincos(vec3 x, out vec3 s, out vec3 c) { s = sin(x); c = cos(x); }")
_lib("kosl_tex", "vec3 kosl_tex(float s, float t) { return texture(iChannel0, vec2(s, t)).rgb; }")
_lib("kosl_calculatenormal", "vec3 kosl_calculatenormal(vec3 p) { return vec3(0.0, 0.0, 1.0); }")

_CONST = """#define M_PI 3.14159265358979323846
#define M_PI_2 1.57079632679489661923
#define M_PI_4 0.785398163397448309616
#define M_2PI 6.28318530717958647692
#define M_4PI 12.5663706143591729538
#define M_2_PI 0.636619772367581343076
#define M_2_SQRTPI 1.12837916709551257390
#define M_E 2.71828182845904523536
#define M_LN2 0.693147180559945309417
#define M_LN10 2.30258509299404568402
#define M_LOG2E 1.44269504088896340736
#define M_LOG10E 0.434294481903251827651
#define M_SQRT2 1.41421356237309504880
#define M_SQRT1_2 0.707106781186547524401
"""


def _library(text: str) -> str:
    """The parts of the library this text uses, in an order that compiles."""
    want, todo = [], [n for n in _LIB if re.search(rf"\b{n}(?:_f|_v)?\b", text)]
    todo += [n for n in ("kosl_sperlin", "kosl_uperlin", "kosl_cell", "kosl_hashn")
             if re.search(rf"\b{n}_[fv]\b", text)]

    def add(n):
        for d in _NEEDS.get(n, ()):
            add(d)
        if n in _LIB and n not in want:
            want.append(n)

    for n in todo:
        add(n)
    return "\n".join(_LIB[n] for n in want)


# --------------------------------------------------------------------------------------------- reading the text
def _error(msg):
    from .shadertoy import ShaderError
    return ShaderError(msg)


def _match(text: str, i: int, open_="(", close=")") -> int:
    """Index of the bracket closing the one at text[i]."""
    depth = 0
    for j in range(i, len(text)):
        c = text[j]
        if c == open_:
            depth += 1
        elif c == close:
            depth -= 1
            if depth == 0:
                return j
    raise _error(f"a '{open_}' is never closed (line {text.count(chr(10), 0, i) + 1}).")


def _split(text: str, sep=",") -> list:
    """Split at sep where no bracket is open."""
    out, depth, start = [], 0, 0
    for j, c in enumerate(text):
        if c in "([{":
            depth += 1
        elif c in ")]}":
            depth -= 1
        elif c == sep and depth == 0:
            out.append(text[start:j])
            start = j + 1
    out.append(text[start:])
    return out


def _calls(text: str, names, fn) -> str:
    """Every call of one of names( ... ) -> fn(name, [args], start) (a string), args split at the top level."""
    pat = re.compile(rf"(?<![\w.])({'|'.join(map(re.escape, names))})\s*\(")
    out, i = [], 0
    while True:
        m = pat.search(text, i)
        if not m:
            break
        close = _match(text, m.end() - 1)
        inner = text[m.end():close]
        args = [a.strip() for a in _split(inner)] if inner.strip() else []
        out += [text[i:m.start()], fn(m.group(1), args, m.start())]
        i = close + 1
    return "".join(out) + text[i:]


def is_osl(bare: str) -> bool:
    """A shader in OSL: a 'shader name(' / 'surface name(' head and no GLSL main."""
    if re.search(r"\bvoid\s+(?:mainImage|main)\s*\(", bare):
        return False
    return bool(_HEAD.search(bare)) or bool(re.search(r'#\s*include\s*[<"]stdosl\.h', bare))


# --------------------------------------------------------------------------------------------- the parameters
class Param:
    def __init__(self, text):
        meta = re.search(r"\[\[(.*?)\]\]", text, re.S)
        self.meta = meta.group(1) if meta else ""
        text = (text[:meta.start()] + text[meta.end():]) if meta else text
        m = re.fullmatch(r"\s*(output\s+)?(closure\s+color|color|point|vector|normal|float|int|string|matrix)\s+(\w+)"
                         r"\s*(\[\s*\d*\s*\])?\s*(?:=\s*(.*?))?\s*", text, re.S)
        if not m:
            raise _error(f"a shader input cannot be read: '{' '.join(text.split())[:80]}'.")
        self.output = bool(m.group(1))
        self.type = "color" if m.group(2).startswith("closure") else m.group(2)
        self.closure = m.group(2).startswith("closure")
        self.name, self.array, self.default = m.group(3), m.group(4), (m.group(5) or "").strip()
        if self.array:
            raise _error(f"the input '{self.name}' is an array: arrays as shader inputs are not supported here.")

    def meta_value(self, key):
        m = re.search(rf"\b(?:float|int|string)\s+{key}\s*=\s*({_NUM}|\"[^\"]*\")", self.meta)
        if not m:
            return None
        v = m.group(1)
        return v[1:-1] if v.startswith('"') else float(v)


def _numbers(default: str):
    """'color(1, 0.5, 0)' / '0.5' / 'color(.2)' -> [r, g, b] or [x]; None when it is not plain numbers."""
    d = re.sub(r"^(?:color|point|vector|normal)\s*\((.*)\)$", r"\1", default.strip(), flags=re.S)
    parts = [p.strip() for p in d.split(",")]
    if all(re.fullmatch(_NUM, p) for p in parts) and parts:
        return [float(p) for p in parts]
    return None


# --------------------------------------------------------------------------------------------- the expressions
def _protect_brackets(text: str):
    """[index] -> a placeholder (no digits in it, so _floats leaves it alone); the index is converted on its own and
    cast with int() unless it is a whole number. Inner brackets first."""
    keep = []
    pat = re.compile(r"\[([^\[\]]*)\]")
    while True:
        m = pat.search(text)
        if not m:
            return text, keep
        inner = m.group(1)
        keep.append(inner if re.fullmatch(r"\s*\d*\s*", inner) else f"int({_floats(inner)})")
        text = text[:m.start()] + "\x01" + chr(0xE000 + len(keep) - 1) + "\x02" + text[m.end():]


def _restore_brackets(text: str, keep) -> str:
    pat = re.compile("\x01([-])\x02")
    while True:
        new = pat.sub(lambda m: "[" + keep[ord(m.group(1)) - 0xE000] + "]", text)
        if new == text:
            return text
        text = new


def _floats(text: str) -> str:
    """Whole-number literals -> floats (1 -> 1.0), not inside names, not hex, not exponents of floats."""
    return re.sub(r"(?<![\w.])(\d+)(?![\w.])", r"\1.0", text)


def _vec_names(text: str) -> set:
    """Names declared as color / point / vector / normal (variables, function inputs)."""
    names = set()
    for m in re.finditer(r"(?<![\w.])(?:color|point|vector|normal)\s+(\w+)\s*(?=[=;,)\[])", text):
        names.add(m.group(1))
        # more names in the same declaration: 'color a = 0, b = 1;'
        end = text.find(";", m.end())
        if end > 0 and "(" not in text[m.start():m.end()]:
            for part in _split(text[m.end():end])[1:]:
                n = re.match(r"\s*(\w+)", part)
                if n:
                    names.add(n.group(1))
    return names


def _noise(text: str, vecs: set, notes: list) -> str:
    """noise() / snoise() / pnoise() / psnoise() / cellnoise() / hashnoise() -> kosl_<kind>_f or _v; _v when the call
    is the whole value given to a colour or vector."""

    def kind_of(fname, args):
        typ = None
        if args and re.fullmatch(r'"\w+"', args[0]):
            typ, args = args[0][1:-1], args[1:]
        periodic = fname in ("pnoise", "psnoise")
        if periodic:
            args = args[:len(args) // 2]               # pnoise(p, period) / pnoise(p, t, pp, pt)
            if "noise: periodic noise" not in " ".join(notes):
                notes.append("noise: periodic noise (pnoise) does not repeat here")
        if fname == "cellnoise":
            typ = "cell"
        elif fname == "hashnoise":
            typ = "hash"
        elif typ is None:
            typ = "perlin" if fname in ("snoise", "psnoise") else "uperlin"
        kind = {"perlin": "sperlin", "snoise": "sperlin", "uperlin": "uperlin", "noise": "uperlin", "cell": "cell",
                "cellnoise": "cell", "hash": "hashn", "simplex": "sperlin", "usimplex": "uperlin",
                "gabor": "sperlin"}.get(typ)
        if kind is None:
            raise _error(f'noise type "{typ}" is not known (perlin, uperlin, cell, hash, simplex, usimplex, gabor).')
        if typ in ("simplex", "usimplex", "gabor") and f"noise: {typ}" not in " ".join(notes):
            notes.append(f"noise: {typ} noise is drawn as perlin noise here")
        return kind, args

    def target_is_vec(start):
        before = text[:start]
        cut = max(before.rfind(";"), before.rfind("{"), before.rfind("}"))
        stmt = before[cut + 1:]
        m = re.search(r"(?:(?:color|point|vector|normal)\s+)?(\w+)\s*(?:[-+*/]?=)\s*$", stmt)
        if m and (re.search(r"(?:color|point|vector|normal)\s+\w+\s*=\s*$", stmt) or m.group(1) in vecs):
            return True
        return bool(re.search(r"\breturn\s*$", stmt)) and _in_vec_function(text, start)

    def fn(fname, args, start):
        kind, args = kind_of(fname, args)
        whole = _noise_whole(text, start)
        suffix = "v" if whole and target_is_vec(start) else "f"
        return f"kosl_{kind}_{suffix}({', '.join(args)})"

    return _calls(text, ("noise", "snoise", "pnoise", "psnoise", "cellnoise", "hashnoise"), fn)


def _noise_whole(text: str, start: int) -> bool:
    """The call at start is the whole right-hand side (what follows its closing bracket is ';' or ',')."""
    close = _match(text, text.index("(", start))
    rest = text[close + 1:].lstrip()
    return rest[:1] in (";", ",")


def _in_vec_function(text: str, pos: int) -> bool:
    for m in re.finditer(r"(?<![\w.])(?:color|point|vector|normal)\s+\w+\s*\([^;{]*\)\s*\{", text):
        if m.end() <= pos:
            try:
                if pos < _match(text, m.end() - 1, "{", "}"):
                    return True
            except Exception:  # noqa: BLE001
                pass
    return False


def _strings(text: str, notes: list) -> str:
    """The places OSL takes a string where GLSL has none."""
    text = re.sub(r'(?<![\w.])color\s*\(\s*"hsv"\s*,', "kosl_hsv(", text)
    text = re.sub(r'(?<![\w.])(?:color|point|vector|normal)\s*\(\s*"\w+"\s*,', "vec3(", text)

    def transform(fname, args, start):
        if len(args) >= 2 and "transform: spaces" not in " ".join(notes):
            notes.append("transform: spaces (object, world, camera) are all the picture here")
        return f"({args[-1]})" if args else "vec3(0.0)"

    text = _calls(text, ("transform", "transformu", "transformc"), transform)

    def texture(fname, args, start):
        if args and args[0].startswith('"'):
            if "texture: a file" not in " ".join(notes):
                notes.append("texture: a file cannot be read here; it reads the picture on image0")
            return f"kosl_tex({args[1] if len(args) > 1 else 'u'}, {args[2] if len(args) > 2 else 'v'})"
        return f"texture({', '.join(args)})"

    text = _calls(text, ("texture",), texture)

    def nothing(fname, args, start):
        if fname == "getattribute" and "getattribute" not in " ".join(notes):
            notes.append("getattribute: there are no attributes here, it gives 0")
        return {"raytype": "1.0", "backfacing": "0.0", "isconnected": "0.0", "getattribute": "0.0",
                "trace": "0.0", "surfacearea": "1.0", "area": "1.0"}.get(fname, "0.0")

    text = _calls(text, ("getattribute", "raytype", "backfacing", "isconnected", "trace", "surfacearea", "area"), nothing)
    text = _calls(text, ("printf", "warning", "error", "fprintf", "format", "concat"), lambda *a: "0.0")
    closures = []

    def closure(fname, args, start):
        closures.append(fname)
        return "vec3(0.0)" if fname in _CLOSURES_DARK else "vec3(1.0)"

    text = _calls(text, tuple(_CLOSURES) + _CLOSURES_DARK, closure)
    if closures:
        notes.append("closures (" + ", ".join(sorted(set(closures))) + ") count as white: the colour they are given "
                     "is shown, not how a renderer would light it")
    m = re.search(r'"[^"\n]*"', text)
    if m:
        line = text.count("\n", 0, m.start()) + 1
        raise _error(f"line {line}: the text {m.group(0)} cannot be used here (strings only work as a noise type, "
                     "a colour space or a string input).")
    return text


def _expr(text: str, vecs: set, notes: list) -> str:
    """OSL statements -> GLSL statements, line by line kept."""
    text = re.sub(r'^[ \t]*#[ \t]*include[^\n]*', "", text, flags=re.M)
    words = "|".join(_GLSL_WORDS)
    text = re.sub(rf"(?<![\w.#])({words})(?![\w])(?!\s*\()", r"\1_", text)       # input -> input_ (not calls)
    text = _noise(text, vecs, notes)
    text = _strings(text, notes)
    text = re.sub(r"(?<![\w.])exit\s*\(\s*\)", "return", text)
    for a, b in _RENAME.items():
        text = re.sub(rf"(?<![\w.]){a}\s*\(", f"{b}(", text)
    text = re.sub(r"(?<![\w.])int\s*\(", "trunc(", text)
    text = re.sub(r"(?<![\w.])closure\s+color(?!\w)", "vec3", text)
    text = re.sub(r"(?<![\w.])(?:color|point|vector|normal)(?!\w)", "vec3", text)
    text = re.sub(r"(?<![\w.])matrix(?!\w)", "mat4", text)
    text = re.sub(r"(?<![\w.])int(?!\w)", "float", text)
    text = re.sub(r"(?<![\w.])output(?!\w)", "inout", text)
    text = re.sub(r"(?<![\w.])string(?!\w)", "float", text)
    # arrays given in braces: 'float a[3] = {1, 2, 3}' -> float[3](...)
    def arr(m):
        n = m.group(3) or str(len(_split(m.group(4))))
        return f"{m.group(1)} {m.group(2)}[{n}] = {m.group(1)}[{n}]({m.group(4)})"
    text = re.sub(r"(?<![\w.])(float|vec3|mat4)\s+(\w+)\s*\[\s*(\d*)\s*\]\s*=\s*\{([^{}]*)\}", arr, text)
    text, keep = _protect_brackets(text)
    text = _floats(text)
    text = _restore_brackets(text, keep)
    # a % b -> mod(a, b); a %= b -> a = mod(a, b)
    text = re.sub(r"([\w.]+)\s*%=\s*([^;]+);", r"\1 = mod(\1, \2);", text)
    op = r"([\w.]+(?:\[[^\]]*\])?(?:\([^()]*\))?|\([^()]*\))"
    for _ in range(8):
        new = re.sub(op + r"\s*%\s*" + op, r"mod(\1, \2)", text)
        if new == text:
            break
        text = new
    # a number where a condition is wanted: if (x), while (!x), x ? a : b
    text = re.sub(r"(?<![\w.])(if|while)\s*\(\s*(!?)\s*([\w.]+)\s*\)",
                  lambda m: f"{m.group(1)} ({m.group(3)} {'==' if m.group(2) else '!='} 0.0)", text)
    text = re.sub(r"([=(,:]|return)\s*([A-Za-z_][\w.]*)\s*\?", r"\1 (\2 != 0.0) ?", text)
    return _wrap_vec(text, vecs)


def _wrap_vec(text: str, vecs: set) -> str:
    """What is given to a colour or vector goes through kv3(): a number there is a grey, as in OSL."""
    if not vecs:
        return text
    pat = re.compile(rf"(?<![\w.\]])({'|'.join(map(re.escape, sorted(vecs)))})\s*=(?!=)")
    out, i = [], 0
    while True:
        m = pat.search(text, i)
        if not m:
            break
        j, depth = m.end(), 0
        while j < len(text):
            c = text[j]
            if c in "([{":
                depth += 1
            elif c in ")]}":
                if depth == 0:
                    break
                depth -= 1
            elif c in ";," and depth == 0:
                break
            j += 1
        rhs = text[m.end():j]
        out += [text[i:m.end()], f" kv3({rhs.strip()})" if rhs.strip() else rhs]
        i = j
    return "".join(out) + text[i:]


# --------------------------------------------------------------------------------------------- the whole shader
def translate(code: str) -> tuple:
    """OSL text -> (Shadertoy GLSL text for shadertoy.translate, notes)."""
    from .shadertoy import _no_comments
    src = _no_comments(code.replace("\r\n", "\n").replace("\r", "\n"))
    head = _HEAD.search(src)
    if not head:
        raise _error("this looks like OSL but has no 'shader name(...)' (or surface): paste the whole shader.")
    kind = head.group(1)
    if kind in ("displacement", "volume"):
        raise _error(f"this is a {kind} shader: only shaders that make a colour or a value (shader / surface) "
                     "can be shown here.")
    p_open = head.end() - 1
    p_close = _match(src, p_open)
    b_open = src.find("{", p_close)
    if b_open < 0:
        raise _error("the shader has no body ({ ... }).")
    b_close = _match(src, b_open, "{", "}")
    params = [Param(t) for t in _split(src[p_open + 1:p_close]) if t.strip()]
    notes = []
    names = {p.name for p in params}
    strings = {p.name: p.default for p in params if p.type == "string" and not p.output}
    prefix, body = src[:head.start()], src[b_open + 1:b_close]
    for n, val in strings.items():                    # a string input is put in where it is used
        prefix = re.sub(rf"(?<![\w.]){n}(?!\w)", val or '""', prefix)
        body = re.sub(rf"(?<![\w.]){n}(?!\w)", val or '""', body)
    vecs = _vec_names(prefix + "\n" + body) | {p.name for p in params if p.type in _VEC}
    vecs |= {"P", "N", "Ng", "I", "Ci", "dPdu", "dPdv", "Ps"} - names
    prefix = _expr(prefix, vecs, notes)
    body = _expr(body, vecs, notes)

    controls, fixed, init, outs = [], [], [], []
    clock = {"time": "iTime", "frame": "(iTime * iFrameRate)"}
    for p in params:
        gtype = {"float": "float", "int": "float", "string": "float", "matrix": "mat4"}.get(p.type, "vec3")
        if p.type == "string":
            continue
        if p.output:
            outs.append(p)
            continue
        dflt = _expr(p.default, vecs, notes).strip() if p.default else ""
        nums = _numbers(p.default) if p.default else ([0.0] if p.type in ("float", "int") else None)
        if p.type in ("float", "int") and p.name.lower() in clock and not p.meta_value("min"):
            fixed.append(f"float {p.name};")
            init.append(f"{p.name} = {clock[p.name.lower()]};")
            notes.append(f"'{p.name}' follows the clock")
        elif p.type in ("float", "int") and nums and len(nums) == 1:
            d = nums[0]
            lo, hi = p.meta_value("min"), p.meta_value("max")
            widget = str(p.meta_value("widget") or "").lower()
            if widget in ("checkbox", "boolean"):
                lo, hi = 0.0, 1.0
            lo = min(0.0, d) if lo is None else lo
            hi = (max(1.0, 2.0 * abs(d)) if d != 0 else 1.0) if hi is None else hi
            about = str(p.meta_value("help") or p.meta_value("label") or "").replace("\n", " ")
            controls.append(f"uniform float {p.name}; // {d:g} {min(lo, hi):g} {max(lo, hi):g} {about}".rstrip())
        elif p.type == "color" and nums and len(nums) in (1, 3):
            rgb = nums * 3 if len(nums) == 1 else nums
            hexc = "".join(f"{int(round(min(max(c, 0.0), 1.0) * 255)):02x}" for c in rgb)
            about = str(p.meta_value("help") or p.meta_value("label") or "").replace("\n", " ")
            controls.append(f"uniform vec3 {p.name}; // #{hexc} {about}".rstrip())
        else:
            fixed.append(f"{gtype} {p.name};")
            if dflt:
                init.append(f"{p.name} = {'kv3(' + dflt + ')' if gtype == 'vec3' else dflt};")
    if not outs and not re.search(r"(?<![\w.])Ci\s*[-+*/]?=", body):
        raise _error("the shader has no output (no 'output color', 'output float' and no Ci): nothing to show.")
    pick = re.search(r"//\s*output\s*:\s*(\w+)", code)
    shown = None
    if pick:
        shown = next((p for p in outs if p.name == pick.group(1)), None)
        if shown is None and pick.group(1) != "Ci":
            notes.append(f"'// output: {pick.group(1)}': the shader has no such output")
    if shown is None and not (pick and pick.group(1) == "Ci"):
        shown = next((p for p in outs if p.type == "color" and not p.closure), None) \
            or (None if re.search(r"(?<![\w.])Ci\s*[-+*/]?=", body) else next(iter(outs), None))
    shown_expr = "Ci" if shown is None else (f"vec3({shown.name})" if shown.type in ("float", "int") else shown.name)
    others = [p.name for p in outs if p is not shown]
    notes.append(f"OSL shader '{head.group(2)}' shows {'Ci' if shown is None else shown.name}"
                 + (f" (also there: {', '.join(others)}; '// output: name' picks one)" if others else ""))

    globals_ = [g for g in ("P", "N", "Ng", "I", "u", "v", "dPdu", "dPdv", "time", "Ci", "dtime", "dPdtime", "Ps")
                if g not in names]
    gdecl = " ".join(f"{'float' if g in ('u', 'v', 'time', 'dtime') else 'vec3'} {g};" for g in globals_)
    sig = ", ".join(f"inout {('float' if p.type in ('float', 'int') else 'vec3')} {p.name}" for p in outs)
    out_init = []
    for p in outs:
        t = "float" if p.type in ("float", "int") else "vec3"
        d = _expr(p.default, vecs, notes).strip() if p.default else ("0.0" if t == "float" else "vec3(0.0)")
        out_init.append(f"    {t} {p.name} = {'kv3(' + d + ')' if t == 'vec3' else d};")
    set_g = {"P": "P = vec3(fragCoord / iResolution.y, 0.0);", "u": "u = fragCoord.x / iResolution.x;",
             "v": "v = fragCoord.y / iResolution.y;", "N": "N = vec3(0.0, 0.0, 1.0);", "Ng": "Ng = vec3(0.0, 0.0, 1.0);",
             "I": "I = vec3(0.0, 0.0, -1.0);", "dPdu": "dPdu = vec3(iResolution.x / iResolution.y, 0.0, 0.0);",
             "dPdv": "dPdv = vec3(0.0, 1.0, 0.0);", "time": "time = iTime;", "Ci": "Ci = vec3(0.0);",
             "dtime": "dtime = iTimeDelta;", "dPdtime": "dPdtime = vec3(0.0);", "Ps": "Ps = vec3(fragCoord / iResolution.y, 0.0);"}
    line_body = src.count("\n", 0, b_open) + 1
    user = prefix + "\n" + body
    lib = _library(user + " ".join(init) + " ".join(out_init) + " kv3")
    text = (f"// translated from OSL by kubakub shader: {head.group(2)}\n" + _CONST
            + "\n".join(controls) + "\n" + gdecl + "\n" + "\n".join(fixed) + "\n" + lib + "\n"
            + "#line 1\n" + prefix + f"\nvoid kOslShader({sig}) {{\n#line {line_body}\n" + body + "\n}\n"
            + "void mainImage(out vec4 fragColor, in vec2 fragCoord) {\n"
            + "".join(f"    {set_g[g]}\n" for g in globals_)
            + "".join(f"    {s}\n" for s in init)
            + "\n".join(out_init) + "\n"
            + f"    kOslShader({', '.join(p.name for p in outs)});\n"
            + f"    fragColor = vec4(kv3({shown_expr}), 1.0);\n}}\n")
    return text, notes
