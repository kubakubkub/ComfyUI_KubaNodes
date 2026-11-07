"""
shader_render.py

kubakub shader, the picture side: the frames of a translated shader (kubakub/shadertoy.py), and what the shader
hears of a sound per frame.

GL comes with ComfyUI (comfy_extras/nodes_glsl.py: ANGLE, GLSL ES 3.00), so no package is added; the context is
a second one of this module's own, because the core node keeps its context on the thread that ran it. The
shader compiles once, pictures that do not change are uploaded once, and every frame is handed on as soon as it is
read back, so a long clip needs no second copy in memory. GL is imported on the first render only: frame_times and
sound_values work without it (tests/test_shadertoy.py).
"""

from __future__ import annotations

import bisect
import contextlib
import ctypes
import functools
import itertools
import sys
import threading
import weakref

import numpy as np

from . import maskfields as mf
from . import shadertoy as st
from . import sound as so
from .director import motion as mo

FIELD_UNIT = st.CHANNELS                             # texture units 0-3 = iChannel0-3, 4 = field + mask
PREV_UNIT = FIELD_UNIT + 1                           # 5 = the frame before (kPrev), for shaders that feed back
MIX_UNIT = PREV_UNIT + 1                             # 6 = the other shader's picture (kubakub shader mix: it bends this one)
REGION_UNIT = MIX_UNIT + 1                           # 7 = the regions (kRegion and its kin): read pixel by pixel, never smoothed
SPARE_UNIT = REGION_UNIT + 1                         # where a new picture is made, so no unit above loses its own
_LOCK = threading.RLock()
_PROGRAMS = {}                                       # source -> compiled program (they live as long as the GL context)
_STAMPS = {}                                         # source -> which compile it is (a dropped shader is compiled again)
_COMPILES = itertools.count(1)
BANDS = (("kLevel", "all"), ("kLow", "low"), ("kMid", "mid"), ("kHigh", "high"))


def _beat(grid, period, t):
    """(kBeat, kBeats) at t: 1 on a beat of the grid and falling to 0 over one period, and how many beats so far."""
    k = bisect.bisect_right(grid, t + 1e-6)
    return (max(0.0, 1.0 - (t - grid[k - 1]) / period) if k and period > 0 else 0.0), float(k)


def now(loops=1, speed=1.0, t=0.0, duration=10.0, ctx=None, fps=25.0) -> dict:
    """What a shader on the director's timeline gets at time t: clock, loop phase and the sound, from the same
    beats and loudness curves the behaviours use (ctx = director/motion.context)."""
    duration = max(1e-6, float(duration))
    fps = float(fps) if fps and fps > 0 else 25.0
    out = {"iTime": float(t) * float(speed), "iTimeDelta": 1.0 / fps, "iFrame": int(round(t * fps)), "iFrameRate": fps,
           "iSampleRate": 44100.0, "kLoop": (float(t) / duration * max(1, int(loops))) % 1.0, "kSeconds": duration}
    if ctx:
        for name, band in BANDS:
            out[name] = min(1.0, max(0.0, mo._level(ctx, t, 0.15, band)))
        grid = ctx.get("beats") or []
        out["kBeat"], out["kBeats"] = _beat(grid, grid[1] - grid[0] if len(grid) > 1 else 0.0, t)
    return out


def vector_values(vector, as_field=True) -> dict:
    """The uniforms of an arrow drawn on the picture: vector = (x0, y0, x1, y1) in 0-1, y down (as it is drawn); a
    shader counts y from the bottom. as_field: the arrow is also the field (kField), when no field is connected.
    None = no arrow (render() starts every uniform at 0)."""
    if not vector:
        return {}
    x0, y0, x1, y1 = (float(v) for v in vector)
    return {"kAx": x0, "kAy": 1.0 - y0, "kBx": x1, "kBy": 1.0 - y1, "kVector": 1.0, "kVecField": 1.0 if as_field else 0.0}


_CORE = []                                           # the module core() found last (looking through sys.modules takes ms)


def core():
    """ComfyUI's GLSL module, the copy ComfyUI has loaded already when there is one (it holds the GL context)."""
    if _CORE and sys.modules.get(_CORE[0].__name__) is _CORE[0]:      # still loaded: it is still the first one found
        return _CORE[0]
    for m in list(sys.modules.values()):
        f = (getattr(m, "__file__", None) or "").replace("\\", "/")
        if f.endswith("comfy_extras/nodes_glsl.py") and hasattr(m, "GLContext"):
            _CORE[:] = [m]
            return m
    try:
        import comfy_extras.nodes_glsl as g
        _CORE[:] = [g]
        return g
    except Exception as e:  # noqa: BLE001
        raise RuntimeError("kubakub shader needs the GLSL support of ComfyUI (the core node 'GLSL Shader', ComfyUI "
                           f"0.16 or newer) and it did not load here: {e}") from e


# --------------------------------------------------------------------------
# time and sound per frame
# --------------------------------------------------------------------------

def frame_times(frames, fps, speed=1.0, start=0.0, extra=0, loops=1):
    """[{iTime, iTimeDelta, iFrame, kLoop}] for frames + extra frames. kLoop runs 0-1 `loops` times over `frames`
    and goes on in the extra ones (they are the tail a loop is blended from)."""
    dt = float(speed) / float(fps)
    loops = max(1, int(loops))
    return [{"iTime": float(start) + i * dt, "iTimeDelta": dt, "iFrame": i, "kLoop": ((i * loops) % frames) / float(frames)}
            for i in range(frames + extra)]


def sound_values(mono, sr, times, bpm=0.0, sample=False, release=0.15, gain=1.0):
    """
    What the shader hears at each time (s into the sound): ({kLevel, kLow, kMid, kHigh, kBeat, kBeats: [per time]},
    report lines). Levels are 0-1 with a release; kBeat is 1 on a beat and falls to 0 before the next; kBeats counts
    them. The same analysis as the mask tools and the director (kubakub/sound.py).
    """
    total = len(mono) / float(sr)
    lines = [f"sound {total:.2f} s" + (" (the built-in beat)" if sample else "")]
    energy = so.band_energy(mono, sr)
    ctx = {"level": mo.level_curve(mono, sr), "bands": {b: so._curve(v) for b, v in energy.items()}, "offset": 0.0}

    def level(t, band):
        return min(1.0, max(0.0, gain * mo._level(ctx, t, release, band)))

    out = {name: [level(t, band) for t in times] for name, band in BANDS}

    if bpm > 0:
        beat, how = {"bpm": float(bpm), "phase": so.beat_phase(mono, sr, bpm)}, "set"
    elif sample:
        beat, how = {"bpm": mf.SAMPLE_BPM, "phase": 0.0}, "the built-in beat"
    else:
        beat, how = so.analyze_beats(mono, sr), "found"
    out["kBeat"], out["kBeats"] = [0.0] * len(times), [0.0] * len(times)
    if beat:
        grid, period = mo.beat_times(beat, 0.0, total), 60.0 / beat["bpm"]
        for i, t in enumerate(times):
            out["kBeat"][i], out["kBeats"][i] = _beat(grid, period, t)
        lines.append(f"{beat['bpm']} bpm ({how}), first beat at {beat['phase']} s")
    else:
        lines.append("no tempo found (the sound is shorter than 2 s): kBeat stays 0, set bpm")
    return out, lines


def frame_sound(mono, sr, heard, fps, n):
    """sound_values() for n frames at fps, by plain names: ({level, low, mid, high, beat: [n values 0-1]}, lines)."""
    got, lines = sound_values(mono, sr, [f / fps for f in range(n)], float(heard["bpm"]), False,
                              float(heard["release"]), float(heard["gain"]))
    return ({k: [round(float(x), 4) for x in got[g]] for k, g in
             (("level", "kLevel"), ("low", "kLow"), ("mid", "kMid"), ("high", "kHigh"), ("beat", "kBeat"))}, lines)


# --------------------------------------------------------------------------
# pictures for the shader
# --------------------------------------------------------------------------

def rgba(a) -> np.ndarray:
    """A picture as float32 [h, w, 4]: grey [h, w], grey + mask [h, w, 2], RGB or RGBA."""
    a = np.asarray(a, np.float32)
    if a.ndim == 2:
        a = a[..., None]
    h, w, c = a.shape
    if c == 4:
        return a
    return _pad4(a, np.ones((h, w, 4), np.float32))


def _pad4(a, out):
    """a [h, w, 1 - 3] (numpy or torch) into out [h, w, 4] of ones: grey or RGB in red, green and blue, a field and
    its mask in red and green (blue 0); alpha stays 1."""
    if a.shape[-1] == 2:                             # field in red, mask in green
        out[..., :2] = a
        out[..., 2] = 0.0
    else:
        out[..., :3] = a[..., :3]
    return out


NOISE = 256


@functools.lru_cache(maxsize=4)
def noise_picture(seed=7) -> np.ndarray:
    """
    The built-in noise for a channel nothing is connected to: float32 [256, 256, 4] in steps of 1 / 255, top row
    first. Many Shadertoys expect a noise picture on iChannel0; green is red moved by (37, 17) and alpha is blue moved
    the same way, the layout they build smooth 3D noise from. The numbers are director/motion.hash01, which the
    window computes bit for bit, so the window and the render see the same noise.
    """
    def plane(sd):
        i = np.arange(NOISE * NOISE, dtype=np.uint64)
        h = (i * 374761393 + sd * 668265263) & 0xFFFFFFFF
        h = ((h ^ (h >> np.uint64(13))) * 1274126177) & 0xFFFFFFFF
        h ^= h >> np.uint64(16)
        return (h >> np.uint64(24)).astype(np.float32).reshape(NOISE, NOISE) / 255.0      # rows bottom first, as GL reads

    def moved(a):                                    # (x, y) reads a at (x - 37, y + 17)
        return np.roll(np.roll(a, 37, axis=1), -17, axis=0)

    r, b = plane(seed), plane(seed + 1)
    out = np.stack([r, moved(r), b, moved(b)], axis=-1)[::-1].copy()
    out.flags.writeable = False
    return out


def noise_texture(seed=7):
    """A texture source for render(): the built-in noise, repeated at its edges as on Shadertoy."""
    def source(i):
        return ("noise", noise_picture(seed))
    source.repeat = True
    return source


def region_picture(lab) -> np.ndarray:
    """
    What a shader knows of the regions (kRegion, kRegionUV, kRegionEdge), float32 [h, w, 4], top row first: red = the
    region's number + 1 (0 outside every region), green and blue = the place inside the region's own box ((0, 0) at its
    bottom left, (1, 1) at its top right), alpha = 0 on the region's edge to 1 at its deepest point.
    lab = int [h, w], -1 outside, else the region's number.
    """
    lab = np.asarray(lab)
    n = int(lab.max()) + 1 if lab.size else 0
    out = np.zeros((*lab.shape, 4), np.float32)
    if n < 1:
        return out
    out[..., 0] = lab + 1
    out[..., 3] = mf.field(lab, n, "edge distance", "each region")[0]
    for k, b in enumerate(mf.boxes(lab, n)):
        if not b:
            continue
        x, y, w, h = b
        on = lab[y:y + h, x:x + w] == k
        part = out[y:y + h, x:x + w]
        part[..., 1][on] = np.broadcast_to(np.linspace(0.0, 1.0, w, dtype=np.float32)[None, :], (h, w))[on]
        part[..., 2][on] = np.broadcast_to(np.linspace(1.0, 0.0, h, dtype=np.float32)[:, None], (h, w))[on]
    return out


def region_source(lab):
    """A texture source for render() (unit REGION_UNIT) and the kRegions value that goes with it."""
    pic = region_picture(lab)
    return (lambda i: ("regions", pic)), float(int(np.asarray(lab).max()) + 1 if np.asarray(lab).size else 0)


def default_field(n=256) -> np.ndarray:
    """Nothing connected: kField is a left to right ramp, kMask is 1 everywhere."""
    out = np.ones((1, n, 2), np.float32)
    out[0, :, 0] = np.linspace(0.0, 1.0, n, dtype=np.float32)
    return out


# --------------------------------------------------------------------------
# render
# --------------------------------------------------------------------------

def render(source, width, height, frames, static=None, textures=None, on_frame=None, check_interrupt=None,
           buffers=None, inputs=None):
    """
    Draw the shader once per entry of `frames`.

    source     the complete fragment shader (shadertoy.translate(...).source)
    frames     [{uniform name: number}]: what changes per frame (frame_times, sound_values)
    static     {uniform name: number or tuple}: set once (controls, kSeconds ...). iResolution and
               iChannelResolution are set here.
    textures   {unit: f(i) -> (key, picture)}: unit 0-3 = iChannel0-3, 4 = field and mask, 7 = the regions
               (region_picture). The picture is uploaded
               again only when its key changes, so a still costs one upload and a clip one per frame.
    on_frame   f(i, rgba float32 [height, width, 4], top row first). The array is reused: copy what you keep.
    buffers    the Buffer tabs of a Shadertoy (shadertoy.Pass: name, source, inputs), drawn in this order before the
               picture at every frame. Each keeps its last picture: a tab that reads itself or a later one sees the
               frame before, one that reads an earlier one sees this frame, as on shadertoy.com.
    inputs     {channel: "A" - "D" | "noise" | "image0" - "image3"}: what the picture's shader reads on a channel
               when it is not the node's picture of that number

    One GL context serves every thread (the director renders its frames in a pool): a render holds it under a lock
    and lets go of it at the end, so the next thread can take it.
    """
    with held() as g:
        _render(g, source, width, height, frames, static, textures, on_frame, check_interrupt, buffers or [], inputs or {})


@contextlib.contextmanager
def held():
    """This module's GL context, current on the calling thread for as long as the block runs -> ComfyUI's GLSL module
    (its .gl are the GL calls). Other threads wait."""
    g = core()
    EGL = g.EGL
    with _LOCK:
        display, surface, context = _context(g)
        EGL.eglBindAPI(EGL.EGL_OPENGL_ES_API)
        if not EGL.eglMakeCurrent(display, surface, surface, context):
            raise RuntimeError(f"kubakub shader: the graphics context is not free (EGL error 0x{EGL.eglGetError():04X}).")
        try:
            if _OWN.get("vao") is None:
                _OWN["vao"] = g.gl.glGenVertexArrays(1)
            g.gl.glBindVertexArray(_OWN["vao"])
            _bury_dead(g.gl)
            yield g
        finally:
            try:
                EGL.eglMakeCurrent(display, EGL.EGL_NO_SURFACE, EGL.EGL_NO_SURFACE, EGL.EGL_NO_CONTEXT)
            except Exception:  # noqa: BLE001      the frames are done
                pass


_OWN = {}                                            # this module's GL context: (display, surface, context), its vertex array


def _context(g):
    """
    A GL context of our own on ComfyUI's display, made once. Not the core's: the core 'GLSL Shader' node leaves its
    context current on the thread that ran it, a context serves one thread at a time, and the director draws its
    frames (and the picker its pictures) on other threads.
    """
    if "ctx" not in _OWN:
        EGL = g.EGL
        display = g._get_egl_display()[0]
        if not EGL.eglBindAPI(EGL.EGL_OPENGL_ES_API):
            raise RuntimeError("kubakub shader: eglBindAPI failed.")
        config, n = EGL.EGLConfig(), ctypes.c_int32(0)
        ok = EGL.eglChooseConfig(display, g._egl_attribs(EGL.EGL_RENDERABLE_TYPE, EGL.EGL_OPENGL_ES3_BIT, EGL.EGL_SURFACE_TYPE,
                                                        EGL.EGL_PBUFFER_BIT, EGL.EGL_RED_SIZE, 8, EGL.EGL_GREEN_SIZE, 8,
                                                        EGL.EGL_BLUE_SIZE, 8, EGL.EGL_ALPHA_SIZE, 8),
                                 ctypes.byref(config), 1, ctypes.byref(n))
        surface = EGL.eglCreatePbufferSurface(display, config, g._egl_attribs(EGL.EGL_WIDTH, 64, EGL.EGL_HEIGHT, 64)) if ok and n.value else None
        context = EGL.eglCreateContext(display, config, EGL.EGL_NO_CONTEXT, g._egl_attribs(EGL.EGL_CONTEXT_CLIENT_VERSION, 3)) if surface else None
        if not context:
            raise RuntimeError("kubakub shader: no graphics context here (the GLSL support of ComfyUI did not give one).")
        _OWN["ctx"] = (display, surface, context)
    return _OWN["ctx"]


def still(source, width, height, values=None, static=None, textures=None) -> np.ndarray:
    """One frame, float32 [height, width, 4]."""
    out = []
    render(source, width, height, [values or {}], static, textures, lambda i, px: out.append(px))   # one frame: the buffer is its own
    return out[0]


# --------------------------------------------------------------------------
# a session: the same shader frame after frame, set up once
# --------------------------------------------------------------------------

_SESSIONS = []                                       # weak refs to the sessions holding graphics memory, last drawn at the end
_KEEP_SESSIONS = 2                                   # how many keep it at once (a 1080p session takes about 120 MB)
_DEAD = []                                           # textures / framebuffers of sessions gone, deleted at the next held()


def _delete(gl, tex, fbos):
    """Delete textures and framebuffers (the context is current)."""
    if tex:
        gl.glDeleteTextures(len(tex), tex)
    for f in fbos:
        gl.glDeleteFramebuffers(1, [f])


def _bury_dead(gl):
    """Delete what sessions that are gone left behind (the context is current: called from held())."""
    while _DEAD:
        _delete(gl, *_DEAD.pop())


def _bury(objs):
    """A session went away without close(): its graphics objects are deleted the next time the context is held."""
    if objs.get("tex") or objs.get("fbos"):
        _DEAD.append((list(objs.get("tex", [])), list(objs.get("fbos", []))))
    objs.clear()


class _Prepared(np.ndarray):
    """A picture already in the form a texture takes (prepared())."""


def prepared(pic) -> np.ndarray:
    """
    A picture as a Session uploads it: rgba(), bottom row first, contiguous. Half floats [h, w, 4] go up as they are
    (every half float is a float the texture holds exactly); anything else as float32, which the card turns into half
    floats itself, as in render(). The card does not round every tiny number the way torch or numpy do (numbers below
    6e-5 can come out one step lower), so the conversion is left to it.
    """
    if getattr(pic, "dtype", None) == np.float16 and pic.ndim == 3 and pic.shape[-1] == 4:
        return np.ascontiguousarray(pic[::-1]).view(_Prepared)
    a = np.asarray(pic, np.float32)
    if a.ndim == 2:
        a = a[..., None]
    if a.size and a.shape[-1] in (1, 2, 3, 4):
        import torch                                  # turns a picture over several times faster than numpy
        a = np.ascontiguousarray(a)
        if a.flags.writeable:                         # (torch only takes a numpy array it may write to)
            src = torch.from_numpy(a).flip(0)
            if src.shape[-1] == 4:
                return src.numpy().view(_Prepared)
            return _pad4(src, torch.ones((*src.shape[:2], 4), dtype=torch.float32)).numpy().view(_Prepared)
    return np.ascontiguousarray(rgba(a)[::-1]).view(_Prepared)


def _names():
    """unit -> the sampler of every picture a shader can read (as render() sets them)."""
    return {**{i: f"iChannel{i}" for i in range(st.CHANNELS)}, FIELD_UNIT: "kFieldTex", MIX_UNIT: "kMixTex",
            REGION_UNIT: "kRegionTex"}


def _put(gl, at, name, v):
    """One uniform at its place `at` (below 0: the shader does not use it): iFrame an int, a tuple a vec2 - vec4,
    anything else a float."""
    if at < 0:
        return
    if name == "iFrame":
        gl.glUniform1i(at, int(v))
    elif isinstance(v, (tuple, list)):
        (gl.glUniform2f, gl.glUniform3f, gl.glUniform4f)[len(v) - 2](at, *[float(x) for x in v])
    else:
        gl.glUniform1f(at, float(v))


def _reset_uniforms(gl, put, loc, width, height, static):
    """A kept shader remembers its uniforms (the arrow, the bend, the sound of the run before): every one the engine
    knows starts at 0, then the picture's size, the static values and the sampler units (the shader is in use)."""
    for name in st.FLOATS + ("kClip", "iFrame"):
        put(name, 0.0)
    for unit in range(st.CHANNELS):                   # a picture input with nothing on it has no size (presets ask for it)
        put(f"iChannelResolution[{unit}]", (0.0, 0.0, 0.0))
    put("iResolution", (width, height, 1.0))
    for name, v in (static or {}).items():
        put(name, v)
    for unit, name in _names().items():               # every shader's samplers look at the same units
        if loc(name) >= 0:
            gl.glUniform1i(loc(name), unit)


def _make_samplers(gl, repeat, blank):
    """A texture for every sampler unit (_names()) -> {unit: texture}. Region numbers are read pixel by pixel (a
    region's number must not blend into its neighbour's), the others smoothed; units in `repeat` repeat at their
    edges, the others are clamped (a repeated picture bleeds its far edge into the near one); units in `blank` start
    as one black pixel."""
    tex = {}
    for unit in _names():
        tex[unit] = t = gl.glGenTextures(1)
        gl.glActiveTexture(gl.GL_TEXTURE0 + unit)
        gl.glBindTexture(gl.GL_TEXTURE_2D, t)
        smooth = gl.GL_NEAREST if unit == REGION_UNIT else gl.GL_LINEAR
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, smooth)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, smooth)
        wrap = gl.GL_REPEAT if unit in repeat else gl.GL_CLAMP_TO_EDGE
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, wrap)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, wrap)
        if unit in blank:
            gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA16F, 1, 1, 0, gl.GL_RGBA, gl.GL_FLOAT,
                            np.zeros((1, 1, 4), np.float32))
    return tex


def _float_picture(gl, w, h, wrap, data=None):
    """A smoothed float picture w x h on the active unit (black when data is None) -> its texture."""
    t = gl.glGenTextures(1)
    gl.glBindTexture(gl.GL_TEXTURE_2D, t)
    gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA32F, w, h, 0, gl.GL_RGBA, gl.GL_FLOAT,
                    np.zeros((h, w, 4), np.float32) if data is None else data)
    for name, v in ((gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR), (gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR),
                    (gl.GL_TEXTURE_WRAP_S, wrap), (gl.GL_TEXTURE_WRAP_T, wrap)):
        gl.glTexParameteri(gl.GL_TEXTURE_2D, name, v)
    return t


def _float_target(gl, w, h):
    """A float picture w x h to draw into, on the active unit, and its framebuffer (left bound) ->
    (texture, framebuffer, complete)."""
    t = gl.glGenTextures(1)
    gl.glBindTexture(gl.GL_TEXTURE_2D, t)
    gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA32F, w, h, 0, gl.GL_RGBA, gl.GL_FLOAT, None)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)
    f = gl.glGenFramebuffers(1)
    gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, f)
    gl.glFramebufferTexture2D(gl.GL_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT0, gl.GL_TEXTURE_2D, t, 0)
    gl.glDrawBuffers(1, [gl.GL_COLOR_ATTACHMENT0])
    return t, f, gl.glCheckFramebufferStatus(gl.GL_FRAMEBUFFER) == gl.GL_FRAMEBUFFER_COMPLETE


class Session:
    """
    One shader drawn frame after frame (a simulation's material: one call per frame, with its own work in between),
    with the graphics set-up made once: the textures, the frame's picture and its framebuffer stay, and a picture
    whose key did not change is not converted or uploaded again. Each draw gives the same numbers as render() of that
    one frame: every uniform starts at 0 again, then the static values, the picture sizes and the frame's values; a
    shader that reads its last frame (kPrev) reads black, as in a one-frame render().

        s = open_session(source, width, height)
        s.draw(values, static, textures, out)        # as often as needed
        s.close()                                    # or let it go: its graphics memory is freed later

    textures = {unit: (key, picture)}: as render()'s, without the function. A key that is None is uploaded at every
    draw, any other key only when it changed (the session keeps the picture with it: do not change it in place). A
    picture may also come prepared() already. A picture in half floats goes up as it is (half the bytes). A unit
    keeps its picture until a draw hands it another one.
    Only the last few sessions drawn keep their graphics memory; an older one makes it again at its next draw.
    One GL context serves every thread: a draw holds it under the same lock as render().
    """

    def __init__(self, source, width, height, repeat=()):
        self.source, self.width, self.height = source, int(width), int(height)
        self.repeat = frozenset(repeat)
        self._objs = {}                               # unit -> texture, "fbo", "fbo_up", "target", "prev", "tex", "fbos"
        self._keys = {}                               # unit -> (key, the picture kept with it)
        self._sizes = {}                              # unit -> iChannelResolution
        self._locs = {}
        self._stamp = None                            # which compile of the shader the uniform places are of
        self._buf = None
        weakref.finalize(self, _bury, self._objs)

    def __copy__(self):                               # a copy makes its own graphics objects (never shares them)
        return Session(self.source, self.width, self.height, self.repeat)

    def __deepcopy__(self, memo):
        return self.__copy__()

    def _make(self, gl, program):
        w, h, o = self.width, self.height, self._objs
        samplers = _make_samplers(gl, self.repeat, _names())       # black until a picture comes
        o.update(samplers)
        o["tex"], o["fbos"] = list(samplers.values()), []
        if gl.glGetUniformLocation(program, "kPrevTex") >= 0:      # the frame before: black, as in a one-frame render
            gl.glActiveTexture(gl.GL_TEXTURE0 + PREV_UNIT)
            o["prev"] = _float_picture(gl, w, h, gl.GL_CLAMP_TO_EDGE)
            o["tex"].append(o["prev"])
        gl.glActiveTexture(gl.GL_TEXTURE0)
        o["target"], o["fbo"], ok = _float_target(gl, w, h)
        # the frame turned upside down on the card (a copy, no filtering), so it is read top row first straight into
        # the caller's array
        up, o["fbo_up"], ok_up = _float_target(gl, w, h)
        o["tex"] += [o["target"], up]
        o["fbos"] += [o["fbo"], o["fbo_up"]]
        ok = ok and ok_up
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)
        self._keys, self._sizes = {}, {}
        if not ok:
            self._free(gl)
            raise RuntimeError(f"kubakub shader: the graphics card refuses a {w}x{h} float picture.")
        self._keys[FIELD_UNIT] = ("default", None)    # nothing on the field: render()'s default ramp
        self._upload(gl, FIELD_UNIT, prepared(default_field()))

    def _free(self, gl):
        o = self._objs
        _delete(gl, o.get("tex"), o.get("fbos", []))
        o.clear()
        self._keys, self._sizes = {}, {}
        _SESSIONS[:] = [r for r in _SESSIONS if r() is not None and r() is not self]

    def _upload(self, gl, unit, pic):
        """pic = prepared(): bottom row first, [h, w, 4] float16 or float32 (a half float texture, as render())."""
        gl.glActiveTexture(gl.GL_TEXTURE0 + unit)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self._objs[unit])
        kind = gl.GL_RGBA32F if unit == REGION_UNIT else gl.GL_RGBA16F
        data = gl.GL_HALF_FLOAT if pic.dtype == np.float16 else gl.GL_FLOAT
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, kind, pic.shape[1], pic.shape[0], 0, gl.GL_RGBA, data, pic)
        if unit < st.CHANNELS:
            self._sizes[unit] = (pic.shape[1], pic.shape[0], 1.0)

    def draw(self, values=None, static=None, textures=None, out=None) -> np.ndarray:
        """One frame -> out (float32 [height, width, 4], top row first; made when None)."""
        w, h = self.width, self.height
        if out is None:
            out = np.empty((h, w, 4), np.float32)
        with held() as g:
            gl = g.gl
            program = _program(g, self.source)
            if self._stamp != _STAMPS.get(self.source):         # compiled again since (the kept ones are few)
                self._stamp, self._locs = _STAMPS.get(self.source), {}
            if not self._objs:
                self._make(gl, program)
            _SESSIONS[:] = [r for r in _SESSIONS if r() is not None and r() is not self]
            _SESSIONS.append(weakref.ref(self))
            while len(_SESSIONS) > _KEEP_SESSIONS:   # the oldest gives its graphics memory back
                oldest = _SESSIONS.pop(0)()
                if oldest is not None:
                    oldest._free(gl)
            locs = self._locs

            def loc(name):
                at = locs.get(name)
                if at is None:
                    at = locs[name] = gl.glGetUniformLocation(program, name)
                return at

            def put(name, v):
                _put(gl, loc(name), name, v)

            try:
                gl.glUseProgram(program)
                # the kept shader is shared with render() and other sessions: every uniform starts as render() starts it
                _reset_uniforms(gl, put, loc, w, h, static)
                if "prev" in self._objs:
                    gl.glUniform1i(loc("kPrevTex"), PREV_UNIT)
                for unit, (key, pic) in (textures or {}).items():
                    was = self._keys.get(unit)
                    if key is None or was is None or was[0] != key:
                        self._keys[unit] = (key, None if key is None else pic)   # a frame's own picture is not kept
                        self._upload(gl, unit, pic if isinstance(pic, _Prepared) else prepared(pic))
                for unit in range(st.CHANNELS):       # render() sets the sizes with the upload, after the reset
                    if unit in self._sizes:
                        put(f"iChannelResolution[{unit}]", self._sizes[unit])
                for unit in _names():                 # other renders used the units in between
                    gl.glActiveTexture(gl.GL_TEXTURE0 + unit)
                    gl.glBindTexture(gl.GL_TEXTURE_2D, self._objs[unit])
                if "prev" in self._objs:
                    gl.glActiveTexture(gl.GL_TEXTURE0 + PREV_UNIT)
                    gl.glBindTexture(gl.GL_TEXTURE_2D, self._objs["prev"])
                gl.glActiveTexture(gl.GL_TEXTURE0)
                for name, v in (values or {}).items():
                    put(name, v)
                gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self._objs["fbo"])
                gl.glViewport(0, 0, w, h)
                gl.glDisable(gl.GL_BLEND)
                gl.glClearColor(0, 0, 0, 0)
                gl.glClear(gl.GL_COLOR_BUFFER_BIT)
                gl.glDrawArrays(gl.GL_TRIANGLES, 0, 3)
                gl.glBindFramebuffer(gl.GL_READ_FRAMEBUFFER, self._objs["fbo"])
                gl.glBindFramebuffer(gl.GL_DRAW_FRAMEBUFFER, self._objs["fbo_up"])
                gl.glBlitFramebuffer(0, 0, w, h, 0, h, w, 0, gl.GL_COLOR_BUFFER_BIT, gl.GL_NEAREST)
                gl.glBindFramebuffer(gl.GL_READ_FRAMEBUFFER, self._objs["fbo_up"])
                gl.glReadBuffer(gl.GL_COLOR_ATTACHMENT0)
                straight = out.dtype == np.float32 and out.flags.c_contiguous and out.shape == (h, w, 4)
                if not straight and self._buf is None:
                    self._buf = np.empty((h, w, 4), np.float32)
                gl.glReadPixels(0, 0, w, h, gl.GL_RGBA, gl.GL_FLOAT, out if straight else self._buf)
            finally:
                gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)
                gl.glUseProgram(0)
        if not straight:
            np.copyto(out, self._buf)
        return out

    def close(self):
        """Give the graphics memory back now (a later draw makes it again)."""
        if self._objs:
            with held() as g:
                self._free(g.gl)
        self._buf = None


def open_session(source, width, height, repeat=()) -> Session:
    """A Session (see there) for this shader at this size. repeat: the units whose picture repeats at its edges."""
    return Session(source, width, height, repeat)


def release_sessions():
    """Every session gives its graphics memory back now (a later draw makes it again) and what sessions that are
    gone left behind is deleted (held() does that). Any thread: the context is held under the lock. Nothing happens
    when no session holds any."""
    if "ctx" not in _OWN or not (_SESSIONS or _DEAD):
        return
    with held() as g:
        for r in list(_SESSIONS):
            s = r()
            if s is not None:
                s._free(g.gl)
        _SESSIONS.clear()


def _program(g, source):
    """The compiled shader, kept for the next frame (a sequence draws the same one hundreds of times)."""
    program = _PROGRAMS.get(source)
    if program is None:
        try:
            program = g._create_program(g.VERTEX_SHADER, source)
        except RuntimeError as e:
            raise st.ShaderError("the shader does not compile.\n" + st.explain(e)) from e
        if len(_PROGRAMS) >= 16:
            oldest = next(iter(_PROGRAMS))
            g.gl.glDeleteProgram(_PROGRAMS.pop(oldest))
            _STAMPS.pop(oldest, None)                # compiled again, it gets a new stamp
        _PROGRAMS[source] = program
        _STAMPS[source] = next(_COMPILES)            # a Session looks its uniform places up again
    return program


def _render(g, source, width, height, frames, static, textures, on_frame, check_interrupt, buffers, inputs):
    gl = g.gl
    width, height = int(width), int(height)
    fbo = target = prev = None
    tex, keys, sizes, extra, fbos = {}, {}, {}, [], []
    try:
        tabs = [{"name": b.name, "program": _program(g, b.source), "inputs": b.inputs} for b in buffers]
        program = _program(g, source)
        gl.glUseProgram(program)
        feedback = gl.glGetUniformLocation(program, "kPrevTex")
        if feedback >= 0:                             # the shader reads its own last frame: kept in a second picture
            gl.glActiveTexture(gl.GL_TEXTURE0 + PREV_UNIT)
            prev = _float_picture(gl, width, height, gl.GL_CLAMP_TO_EDGE)
            gl.glUniform1i(feedback, PREV_UNIT)
            gl.glActiveTexture(gl.GL_TEXTURE0)

        target, fbo, complete = _float_target(gl, width, height)
        if not complete:
            raise RuntimeError(f"kubakub shader: the graphics card refuses a {width}x{height} float picture.")

        locs = {}
        now = [program]                               # the shader that is being fed

        def loc(name):
            k = (now[0], name)
            if k not in locs:
                locs[k] = gl.glGetUniformLocation(now[0], name)
            return locs[k]

        def use(p):
            now[0] = p
            gl.glUseProgram(p)

        def put(name, v):
            _put(gl, loc(name), name, v)

        textures = dict(textures or {})
        textures.setdefault(FIELD_UNIT, lambda i: ("default", default_field()))
        # every sampler gets a texture (an unconnected one reads black); your pictures are clamped, not repeated as on
        # Shadertoy; the built-in noise repeats, shaders walk far across it
        tex = _make_samplers(gl, {u for u, f in textures.items() if getattr(f, "repeat", False)},
                             set(_names()) - set(textures))

        def upload(unit, i):
            key, pic = textures[unit](i)
            if keys.get(unit) == key:
                return
            keys[unit] = key
            pic = np.ascontiguousarray(rgba(pic)[::-1])           # GL reads pictures bottom row first
            gl.glActiveTexture(gl.GL_TEXTURE0 + unit)
            gl.glBindTexture(gl.GL_TEXTURE_2D, tex[unit])
            # half float: enough for pictures, depth and normals, and every GL ES 3 card filters it
            kind = gl.GL_RGBA32F if unit == REGION_UNIT else gl.GL_RGBA16F      # region numbers and places need every bit
            gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, kind, pic.shape[1], pic.shape[0], 0, gl.GL_RGBA, gl.GL_FLOAT, pic)
            if unit < st.CHANNELS:
                sizes[unit] = (pic.shape[1], pic.shape[0], 1.0)
                put(f"iChannelResolution[{unit}]", sizes[unit])

        def picture(wrap=gl.GL_CLAMP_TO_EDGE, data=None, w=width, h=height):      # a float picture of the frame's size
            gl.glActiveTexture(gl.GL_TEXTURE0 + SPARE_UNIT)
            extra.append(_float_picture(gl, w, h, wrap, data))
            return extra[-1]

        noise = []

        def feed(reads):                              # the channels of the shader in use, as its tab wires them
            for c in range(st.CHANNELS):
                src = reads.get(c, f"image{c}")
                if src == "noise":
                    if not noise:
                        noise.append(picture(gl.GL_REPEAT, np.ascontiguousarray(noise_picture(7)[::-1]), NOISE, NOISE))
                    t, size = noise[0], (float(NOISE), float(NOISE), 1.0)
                elif src.startswith("image"):
                    n = int(src[5:])
                    t, size = tex[n], sizes.get(n, (0.0, 0.0, 0.0))
                else:
                    t, size = next(b for b in tabs if b["name"] == src)["front"], (float(width), float(height), 1.0)
                gl.glActiveTexture(gl.GL_TEXTURE0 + c)
                gl.glBindTexture(gl.GL_TEXTURE_2D, t)
                put(f"iChannelResolution[{c}]", size)

        for b in tabs:                                # each tab draws into one picture while it reads its other one
            b["front"], b["back"] = picture(), picture()
            b["fbo"] = gl.glGenFramebuffers(1)
            fbos.append(b["fbo"])

        for p in [b["program"] for b in tabs] + [program]:
            use(p)
            _reset_uniforms(gl, put, loc, width, height, static)

        gl.glViewport(0, 0, width, height)
        gl.glDisable(gl.GL_BLEND)
        buf = np.empty((height, width, 4), np.float32)
        for i, values in enumerate(frames):
            if check_interrupt:
                check_interrupt()
            for unit in textures:
                upload(unit, i)
            for b in tabs:
                use(b["program"])
                for name, v in values.items():
                    put(name, v)
                feed(b["inputs"])
                gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, b["fbo"])
                gl.glFramebufferTexture2D(gl.GL_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT0, gl.GL_TEXTURE_2D, b["back"], 0)
                gl.glDrawBuffers(1, [gl.GL_COLOR_ATTACHMENT0])
                gl.glClearColor(0, 0, 0, 0)
                gl.glClear(gl.GL_COLOR_BUFFER_BIT)
                gl.glDrawArrays(gl.GL_TRIANGLES, 0, 3)
                b["front"], b["back"] = b["back"], b["front"]
            use(program)
            if tabs or inputs:
                feed(inputs)
            for name, v in values.items():
                put(name, v)
            gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, fbo)
            gl.glClearColor(0, 0, 0, 0)
            gl.glClear(gl.GL_COLOR_BUFFER_BIT)
            gl.glDrawArrays(gl.GL_TRIANGLES, 0, 3)
            gl.glReadBuffer(gl.GL_COLOR_ATTACHMENT0)
            gl.glReadPixels(0, 0, width, height, gl.GL_RGBA, gl.GL_FLOAT, buf)
            if prev is not None:                      # this frame is the next one's kPrev
                gl.glActiveTexture(gl.GL_TEXTURE0 + PREV_UNIT)
                gl.glBindTexture(gl.GL_TEXTURE_2D, prev)
                gl.glCopyTexSubImage2D(gl.GL_TEXTURE_2D, 0, 0, 0, 0, 0, width, height)
            if on_frame:
                on_frame(i, buf[::-1])
    finally:
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)
        gl.glUseProgram(0)
        _delete(gl, list(tex.values()) + [t for t in (target, prev) if t is not None] + extra,
                fbos + [f for f in (fbo,) if f is not None])
