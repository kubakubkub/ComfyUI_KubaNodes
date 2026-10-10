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
import ctypes
import functools
import sys
import threading

import numpy as np

from . import maskfields as mf
from . import shadertoy as st
from . import sound as so
from .director import motion as mo

FIELD_UNIT = st.CHANNELS                             # texture units 0-3 = iChannel0-3, 4 = field + mask
PREV_UNIT = FIELD_UNIT + 1                           # 5 = the frame before (kPrev), for shaders that feed back
MIX_UNIT = PREV_UNIT + 1                             # 6 = the other shader's picture (kubakub shader mix: it bends this one)
_LOCK = threading.RLock()
_PROGRAMS = {}                                       # source -> compiled program (they live as long as the GL context)
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


def core():
    """ComfyUI's GLSL module, the copy ComfyUI has loaded already when there is one (it holds the GL context)."""
    for m in list(sys.modules.values()):
        f = (getattr(m, "__file__", None) or "").replace("\\", "/")
        if f.endswith("comfy_extras/nodes_glsl.py") and hasattr(m, "GLContext"):
            return m
    try:
        import comfy_extras.nodes_glsl as g
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
    out = np.ones((h, w, 4), np.float32)
    if c == 1:
        out[..., :3] = a
    elif c == 2:                                     # field in red, mask in green
        out[..., 0], out[..., 1], out[..., 2] = a[..., 0], a[..., 1], 0.0
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


def default_field(n=256) -> np.ndarray:
    """Nothing connected: kField is a left to right ramp, kMask is 1 everywhere."""
    out = np.ones((1, n, 2), np.float32)
    out[0, :, 0] = np.linspace(0.0, 1.0, n, dtype=np.float32)
    return out


# --------------------------------------------------------------------------
# render
# --------------------------------------------------------------------------

def render(source, width, height, frames, static=None, textures=None, on_frame=None, check_interrupt=None):
    """
    Draw the shader once per entry of `frames`.

    source     the complete fragment shader (shadertoy.translate(...).source)
    frames     [{uniform name: number}]: what changes per frame (frame_times, sound_values)
    static     {uniform name: number or tuple}: set once (controls, kSeconds ...). iResolution and
               iChannelResolution are set here.
    textures   {unit: f(i) -> (key, picture)}: unit 0-3 = iChannel0-3, 4 = field and mask. The picture is uploaded
               again only when its key changes, so a still costs one upload and a clip one per frame.
    on_frame   f(i, rgba float32 [height, width, 4], top row first). The array is reused: copy what you keep.

    One GL context serves every thread (the director renders its frames in a pool): a render holds it under a lock
    and lets go of it at the end, so the next thread can take it.
    """
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
            _render(g, source, width, height, frames, static, textures, on_frame, check_interrupt)
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


def _program(g, source):
    """The compiled shader, kept for the next frame (a sequence draws the same one hundreds of times)."""
    program = _PROGRAMS.get(source)
    if program is None:
        try:
            program = g._create_program(g.VERTEX_SHADER, source)
        except RuntimeError as e:
            raise st.ShaderError("the shader does not compile.\n" + st.explain(e)) from e
        if len(_PROGRAMS) >= 16:
            g.gl.glDeleteProgram(_PROGRAMS.pop(next(iter(_PROGRAMS))))
        _PROGRAMS[source] = program
    return program


def _render(g, source, width, height, frames, static, textures, on_frame, check_interrupt):
    gl = g.gl
    width, height = int(width), int(height)
    fbo = target = prev = None
    tex, keys = {}, {}
    try:
        program = _program(g, source)
        gl.glUseProgram(program)
        feedback = gl.glGetUniformLocation(program, "kPrevTex")
        if feedback >= 0:                             # the shader reads its own last frame: kept in a second picture
            prev = gl.glGenTextures(1)
            gl.glActiveTexture(gl.GL_TEXTURE0 + PREV_UNIT)
            gl.glBindTexture(gl.GL_TEXTURE_2D, prev)
            gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA32F, width, height, 0, gl.GL_RGBA, gl.GL_FLOAT,
                            np.zeros((height, width, 4), np.float32))
            for name, v in ((gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR), (gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR),
                            (gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE), (gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)):
                gl.glTexParameteri(gl.GL_TEXTURE_2D, name, v)
            gl.glUniform1i(feedback, PREV_UNIT)
            gl.glActiveTexture(gl.GL_TEXTURE0)

        target = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, target)
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA32F, width, height, 0, gl.GL_RGBA, gl.GL_FLOAT, None)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)
        fbo = gl.glGenFramebuffers(1)
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, fbo)
        gl.glFramebufferTexture2D(gl.GL_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT0, gl.GL_TEXTURE_2D, target, 0)
        gl.glDrawBuffers(1, [gl.GL_COLOR_ATTACHMENT0])
        if gl.glCheckFramebufferStatus(gl.GL_FRAMEBUFFER) != gl.GL_FRAMEBUFFER_COMPLETE:
            raise RuntimeError(f"kubakub shader: the graphics card refuses a {width}x{height} float picture.")

        locs = {}

        def loc(name):
            if name not in locs:
                locs[name] = gl.glGetUniformLocation(program, name)
            return locs[name]

        def put(name, v):
            at = loc(name)
            if at < 0:                                # the shader does not use it
                return
            if name == "iFrame":
                gl.glUniform1i(at, int(v))
            elif isinstance(v, (tuple, list)):
                (gl.glUniform2f, gl.glUniform3f, gl.glUniform4f)[len(v) - 2](at, *[float(x) for x in v])
            else:
                gl.glUniform1f(at, float(v))

        textures = dict(textures or {})
        textures.setdefault(FIELD_UNIT, lambda i: ("default", default_field()))
        names = {**{i: f"iChannel{i}" for i in range(st.CHANNELS)}, FIELD_UNIT: "kFieldTex", MIX_UNIT: "kMixTex"}
        for unit in names:                            # every sampler gets a texture: an unconnected one reads black
            tex[unit] = gl.glGenTextures(1)
            gl.glActiveTexture(gl.GL_TEXTURE0 + unit)
            gl.glBindTexture(gl.GL_TEXTURE_2D, tex[unit])
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
            # your pictures are clamped, not repeated as on Shadertoy (a repeated picture bleeds its far edge into the
            # near one); the built-in noise repeats, shaders walk far across it
            wrap = gl.GL_REPEAT if getattr(textures.get(unit), "repeat", False) else gl.GL_CLAMP_TO_EDGE
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, wrap)
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, wrap)
            if unit not in textures:
                gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA16F, 1, 1, 0, gl.GL_RGBA, gl.GL_FLOAT,
                                np.zeros((1, 1, 4), np.float32))
            if loc(names[unit]) >= 0:
                gl.glUniform1i(loc(names[unit]), unit)

        def upload(unit, i):
            key, pic = textures[unit](i)
            if keys.get(unit) == key:
                return
            keys[unit] = key
            pic = np.ascontiguousarray(rgba(pic)[::-1])           # GL reads pictures bottom row first
            gl.glActiveTexture(gl.GL_TEXTURE0 + unit)
            gl.glBindTexture(gl.GL_TEXTURE_2D, tex[unit])
            # half float: enough for pictures, depth and normals, and every GL ES 3 card filters it
            gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA16F, pic.shape[1], pic.shape[0], 0, gl.GL_RGBA, gl.GL_FLOAT, pic)
            if unit < st.CHANNELS:
                put(f"iChannelResolution[{unit}]", (pic.shape[1], pic.shape[0], 1.0))

        # a compiled shader is kept between renders and remembers its uniforms (the arrow, the bend, the sound of the
        # run before): every one the engine knows starts at 0, then the caller's values
        for name in st.FLOATS + ("kClip", "iFrame"):
            put(name, 0.0)
        for unit in range(st.CHANNELS):               # a picture input with nothing on it has no size (presets ask for it)
            put(f"iChannelResolution[{unit}]", (0.0, 0.0, 0.0))
        put("iResolution", (width, height, 1.0))
        for name, v in (static or {}).items():
            put(name, v)

        gl.glViewport(0, 0, width, height)
        gl.glDisable(gl.GL_BLEND)
        buf = np.empty((height, width, 4), np.float32)
        for i, values in enumerate(frames):
            if check_interrupt:
                check_interrupt()
            for unit in textures:
                upload(unit, i)
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
        ids = list(tex.values()) + [t for t in (target, prev) if t is not None]
        if ids:
            gl.glDeleteTextures(len(ids), ids)
        if fbo is not None:
            gl.glDeleteFramebuffers(1, [fbo])
