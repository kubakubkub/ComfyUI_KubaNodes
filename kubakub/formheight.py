"""
formheight.py

kubakub form height: how far a 3D model stands out of the real wall, per pixel of the projection view. The model (a
generated one that kubakub align 3d model has laid onto the building) is drawn from the scene's camera on the graphics
card (the GL context of kubakub/shader_render.py, a depth picture in a few milliseconds), and the real wall the scene
render saw is taken away from it. What is left is a height map: 0 on the wall, metres where a form stands in front
of it. The wall itself stays the real one, so windows the generator drew beside the real ones do not show.

numpy + opencv; GL only in model_depth. No ComfyUI imports (tests/test_formheight.py).
"""

from __future__ import annotations

import contextlib
import ctypes

import cv2
import numpy as np

from .scene3d import scene_ids as si
from .scene3d import scene_view as sv

UPS = ("auto", "y up (.glb, .obj)", "z up (Blender)")
FROMS = ("the wall's plane", "the real surface")

_VERT = """#version 300 es
layout(location = 0) in vec4 aClip;
layout(location = 1) in float aDepth;
layout(location = 2) in vec2 aUV;
layout(location = 3) in vec3 aCol;
out float vDepth;
out vec2 vUV;
out vec3 vCol;
void main() { vDepth = aDepth; vUV = aUV; vCol = aCol; gl_Position = aClip; }
"""
_FRAG = """#version 300 es
precision highp float;
uniform sampler2D uTex;
uniform float uUseTex;
in float vDepth;
in vec2 vUV;
in vec3 vCol;
layout(location = 0) out vec4 o;
layout(location = 1) out vec4 c;
void main() {
    o = vec4(vDepth, 1.0, 0.0, 1.0);
    c = vec4(vCol * (uUseTex > 0.5 ? texture(uTex, vUV).rgb : vec3(1.0)), 1.0);
}
"""


def to_scene(vertices, y_up, unit_scale=1.0) -> np.ndarray:
    """Model points in the scene's world (z up, metres). y_up: the file counts y as up, as .glb and .obj do, and
    Blender turns it on import: (x, y, z) -> (x, -z, y)."""
    v = np.asarray(vertices, np.float64)
    if y_up:
        v = np.stack([v[:, 0], -v[:, 2], v[:, 1]], 1)
    return v * float(unit_scale)


def model_depth(cam, vertices, faces) -> np.ndarray:
    """The model seen from the scene's camera: depth along the view axis per pixel (float32 [H, W], metres, 0 = no
    model there). vertices in the scene's world."""
    return model_view(cam, vertices, faces)[0]


def model_view(cam, vertices, faces, look=None) -> tuple:
    """The model seen from the scene's camera -> (depth as model_depth gives it, its colours float32 [H, W, 3] or
    None without `look`). The colours are the model's own, with no light on them.

    look  {"uv": [V, 2] texture places, v counted from the bottom (as trimesh holds them), or None;
           "col": [V, 3] a colour per point in 0-1, or None;
           "parts": [(first face, how many, picture uint8 [h, w, 3 or 4] or None)]: which faces wear which texture}
    """
    from . import shader_render as sr
    pc = (np.asarray(vertices, np.float64) - cam.C) @ cam.R
    clip = np.concatenate([pc, np.ones((len(pc), 1))], 1) @ cam.P.T
    n = len(pc)
    uv = look.get("uv") if look else None
    col = look.get("col") if look else None
    data = np.ascontiguousarray(np.concatenate([
        clip, -pc[:, 2:3], np.zeros((n, 2)) if uv is None else uv, np.ones((n, 3)) if col is None else col], 1), np.float32)
    index = np.ascontiguousarray(faces, np.uint32).ravel()
    parts = look["parts"] if look else [(0, len(index) // 3, None)]
    W, H = cam.W, cam.H
    with sr.held() as g, contextlib.ExitStack() as undo:
        gl = g.gl

        def reset():                                      # the shader renders that follow expect no depth test
            gl.glDisable(gl.GL_DEPTH_TEST)
            gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)
            gl.glBindVertexArray(0)
            gl.glUseProgram(0)
            gl.glActiveTexture(gl.GL_TEXTURE0)

        program = g._create_program(_VERT, _FRAG)
        undo.callback(gl.glDeleteProgram, program)
        undo.callback(reset)
        vao = gl.glGenVertexArrays(1)
        undo.callback(gl.glDeleteVertexArrays, 1, [vao])
        gl.glBindVertexArray(vao)
        buffers = [int(x) for x in gl.glGenBuffers(2)]
        undo.callback(gl.glDeleteBuffers, 2, buffers)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, buffers[0])
        gl.glBufferData(gl.GL_ARRAY_BUFFER, data.nbytes, data, gl.GL_STATIC_DRAW)
        for at, (size, offset) in enumerate(((4, 0), (1, 16), (2, 20), (3, 28))):
            gl.glEnableVertexAttribArray(at)
            gl.glVertexAttribPointer(at, size, gl.GL_FLOAT, False, 40, ctypes.c_void_p(offset))
        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, buffers[1])
        gl.glBufferData(gl.GL_ELEMENT_ARRAY_BUFFER, index.nbytes, index, gl.GL_STATIC_DRAW)

        fbo = gl.glGenFramebuffers(1)
        undo.callback(gl.glDeleteFramebuffers, 1, [fbo])
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, fbo)
        targets = [gl.GL_COLOR_ATTACHMENT0, gl.GL_COLOR_ATTACHMENT1][:2 if look else 1]
        for attachment in targets:
            target = gl.glGenTextures(1)
            undo.callback(gl.glDeleteTextures, 1, [target])
            gl.glBindTexture(gl.GL_TEXTURE_2D, target)
            gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA32F, W, H, 0, gl.GL_RGBA, gl.GL_FLOAT, None)
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST)
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)
            gl.glFramebufferTexture2D(gl.GL_FRAMEBUFFER, attachment, gl.GL_TEXTURE_2D, target, 0)
        zbuf = gl.glGenRenderbuffers(1)
        undo.callback(gl.glDeleteRenderbuffers, 1, [zbuf])
        gl.glBindRenderbuffer(gl.GL_RENDERBUFFER, zbuf)
        gl.glRenderbufferStorage(gl.GL_RENDERBUFFER, gl.GL_DEPTH_COMPONENT32F, W, H)
        gl.glFramebufferRenderbuffer(gl.GL_FRAMEBUFFER, gl.GL_DEPTH_ATTACHMENT, gl.GL_RENDERBUFFER, zbuf)
        gl.glDrawBuffers(len(targets), targets)
        if gl.glCheckFramebufferStatus(gl.GL_FRAMEBUFFER) != gl.GL_FRAMEBUFFER_COMPLETE:
            raise RuntimeError(f"the graphics card refuses a {W}x{H} depth picture.")

        gl.glUseProgram(program)
        gl.glUniform1i(gl.glGetUniformLocation(program, "uTex"), 0)
        use_tex = gl.glGetUniformLocation(program, "uUseTex")
        gl.glViewport(0, 0, W, H)
        gl.glDisable(gl.GL_BLEND)
        gl.glDisable(gl.GL_CULL_FACE)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glDepthFunc(gl.GL_LESS)
        gl.glDepthMask(True)
        gl.glClearColor(0, 0, 0, 0)
        gl.glClearDepthf(1.0)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        gl.glActiveTexture(gl.GL_TEXTURE0)
        gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 1)
        for first, count, picture in parts:
            if picture is not None and uv is not None:
                px = np.ascontiguousarray(picture[::-1, :, :3])      # GL counts a picture's rows from the bottom, as the uv
                tex = gl.glGenTextures(1)
                undo.callback(gl.glDeleteTextures, 1, [tex])
                gl.glBindTexture(gl.GL_TEXTURE_2D, tex)
                gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGB8, px.shape[1], px.shape[0], 0, gl.GL_RGB, gl.GL_UNSIGNED_BYTE, px)
                gl.glGenerateMipmap(gl.GL_TEXTURE_2D)
                gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR_MIPMAP_LINEAR)
                gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
                gl.glUniform1f(use_tex, 1.0)
            else:
                gl.glUniform1f(use_tex, 0.0)
            gl.glDrawElements(gl.GL_TRIANGLES, int(count) * 3, gl.GL_UNSIGNED_INT, ctypes.c_void_p(int(first) * 12))
        buf = np.empty((H, W, 4), np.float32)
        gl.glReadBuffer(gl.GL_COLOR_ATTACHMENT0)
        gl.glReadPixels(0, 0, W, H, gl.GL_RGBA, gl.GL_FLOAT, buf)
        depth = np.where(buf[::-1, :, 1] > 0.5, buf[::-1, :, 0], 0.0).astype(np.float32)
        rgb = None
        if look:
            gl.glReadBuffer(gl.GL_COLOR_ATTACHMENT1)
            gl.glReadPixels(0, 0, W, H, gl.GL_RGBA, gl.GL_FLOAT, buf)
            rgb = np.clip(buf[::-1, :, :3], 0.0, 1.0)
    return depth, rgb


def shown(values) -> np.ndarray:
    """Colour values a model holds as light (the colours on its points, a material's plain colour) as picture values."""
    return np.clip(np.asarray(values, np.float64), 0.0, 1.0) ** (1.0 / 2.2)


def _over_view(cam, values, u, v, inverse=False) -> np.ndarray:
    """A value known at the view points (u, v) for every pixel: float32 [H, W]. For what is linear across a camera's
    picture (inverse: what is 1 / linear, as the depth of a plane is in a perspective view), so a few points give it
    exactly and no ray per pixel is needed."""
    A = np.stack([u.ravel(), v.ravel(), np.ones(u.size)], 1)
    q = values.ravel()
    c = np.linalg.lstsq(A, 1.0 / q if inverse else q, rcond=None)[0]
    lin = c[0] * (np.arange(cam.W) + 0.5)[None, :] + c[1] * (np.arange(cam.H) + 0.5)[:, None] + c[2]
    return (1.0 / lin if inverse else lin).astype(np.float32)


def wall(scene) -> dict:
    """What the height is counted from, per pixel of the scene's view: the real model where the scene render saw it,
    the main wall's plane beside it (a form may reach past the building's outline).
    -> {depth [H, W] along the view axis, plane [H, W]: the same for the main wall's plane alone, seen [H, W] bool,
    out [H, W]: metres out of the wall per metre of depth towards the camera, pixel_m: size of one pixel on the wall,
    size_m: the building's largest length, cam}"""
    cam = sv.Camera(scene["info"])
    pt, nrm = si.main_plane(scene)
    depth, _dist = si.depth_maps(scene)
    seen = scene["faceid"] > 0
    u, v = np.meshgrid(np.linspace(0.5, cam.W - 0.5, 9), np.linspace(0.5, cam.H - 0.5, 9))
    origin, d = cam.ray(u, v)
    fwd = -cam.R[:, 2]
    along = np.maximum(d @ fwd, 1e-6)
    on_plane = _over_view(cam, (sv.plane_hit(origin, d, pt, nrm) - cam.C) @ fwd, u, v, inverse=not cam.ortho)
    far, size = (float(np.median(depth[seen])), float(np.ptp(scene["position"][seen], axis=0).max())) if seen.any() else (1.0, 1.0)
    return {"depth": np.where(seen, depth, on_plane).astype(np.float32), "plane": on_plane, "seen": seen,
            "out": np.abs(_over_view(cam, (d @ nrm) / along, u, v)), "pixel_m": float(cam.pixel_size_m(far)), "cam": cam, "size_m": size}


def lies_on(model, w, near_m) -> tuple:
    """How well the model lies on the template, seen from the camera, as shares of the building's pixels (0-1):
    (where the model's surface is within near_m of the real one, where the model is in front of the building at all)."""
    n = int(w["seen"].sum())
    if not n:
        return 0.0, 0.0
    both = w["seen"] & (model > 0)
    return float((np.abs(model - w["depth"])[both] < near_m).sum()) / n, float(both.sum()) / n


def height(model, w, wall_m=0.3, smallest_form_m=0.5, flat=True) -> tuple:
    """Depth of the model and the wall -> (metres out of the wall per pixel, float32 [H, W], 0 on the wall;
    forms [H, W] bool). wall_m: what stands out of the real surface less than this is the wall (a generated wall
    never lies exactly on the real one) and the height is counted from there. smallest_form_m: what is thinner than
    this is dropped (specks, and the lines along cornices and window edges that sit a little off). flat: the height
    is counted from the main wall's plane, so the real wall's own relief does not show in the forms; else from the
    real surface under each pixel (what a displacement of the real model needs)."""
    rise = np.where(model > 0, (w["depth"] - model) * w["out"], 0.0)
    forms = rise > float(wall_m)
    if flat:                                              # a model that only fills a niche up to the wall is no form
        rise = np.where(model > 0, (w["plane"] - model) * w["out"], 0.0)
        forms &= rise > float(wall_m)
    r = 0.5 * float(smallest_form_m) / w["pixel_m"]
    if r >= 0.5 and forms.any():                          # by distance, so a large r costs no more than a small one
        thick = cv2.distanceTransform(forms.astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE) > r
        if thick.any():                                   # the thick parts keep their own outline
            forms &= cv2.distanceTransform((~thick).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE) <= 2.0 * r
        else:
            forms = thick
    return np.where(forms, np.maximum(rise - float(wall_m), 0.0), 0.0).astype(np.float32), forms


def with_forms(model, forms, w) -> np.ndarray:
    """The depth of the real building with the forms standing on it: the model where a form is, the real wall
    elsewhere. Grey as depth_gray of kubakub scene render: near = 1, far = 0, 0 where there is neither."""
    return si.grey_depth(np.where(forms, model, w["depth"]), w["seen"] | forms)


def ball(scene, share=0.18, n=48) -> tuple:
    """A sample model: half a ball standing on the middle of the main wall (share of the building's size as its
    radius). -> (vertices in the scene's world, faces)"""
    pt, nrm = si.main_plane(scene)
    seen = scene["faceid"] > 0
    pos = scene["position"][seen]
    centre = pos.mean(0)
    centre = centre - ((centre - pt) @ nrm) * nrm
    r = share * float(np.ptp(pos, axis=0).max())
    a = np.array([0.0, 0.0, 1.0]) if abs(nrm[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    ex = np.cross(a, nrm)
    ex /= np.linalg.norm(ex)
    ey = np.cross(nrm, ex)
    lat = np.linspace(0, np.pi / 2, n // 4 + 1)
    lon = np.linspace(0, 2 * np.pi, n, endpoint=False)
    rings = [centre + r * (np.cos(t) * (np.cos(lon)[:, None] * ex + np.sin(lon)[:, None] * ey) + np.sin(t) * nrm) for t in lat[:-1]]
    verts = np.concatenate(rings + [(centre + r * nrm)[None]])
    faces = []
    for k in range(len(rings) - 1):
        for j in range(n):
            p, q = k * n + j, k * n + (j + 1) % n
            faces += [[p, q, q + n], [p, q + n, p + n]]
    top, last = len(verts) - 1, (len(rings) - 1) * n
    faces += [[last + j, last + (j + 1) % n, top] for j in range(n)]
    return verts, np.array(faces, np.int64)
