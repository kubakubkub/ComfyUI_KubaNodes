"""
heightanim.py

kubakub height animate: a height map that grows. Any grey picture where white stands out works: the height of
kubakub form height, or a depth map from Depth Anything or Marigold. One height map per frame, and the same lit from
the side, so the growth can be seen (and used as the moving clip of kubakub motion guide).

numpy + opencv, no ComfyUI imports (tests/test_heightanim.py).
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from . import maskfields as mf

GROWS = ("all together", "peaks first", "built up in layers", "along the field")
LOOKS = ("height", "relief", "shading")               # what a director layer shows of its height
LAYER_BLUR = 3.0                                     # the same for a director layer (its window works on 8 bit pictures)
FLAT = 0.8                                           # the grey of a lit relief where the wall is flat
COMES = 0.04                                         # a form shows its own picture from this height on
BLUR = 1.2                                           # the smoothing before the light, pixels at 1920 wide
LIGHT_HEIGHT = math.radians(35.0)                    # the light stands this high above the wall


def from_grey(grey, wall=0.0) -> np.ndarray:
    """A grey picture [H, W] or [H, W, C] as a height 0-1: `wall` is the grey that counts as the wall (0), white is 1."""
    g = np.asarray(grey, np.float32)
    if g.ndim == 3:
        g = g[..., :3].mean(-1)
    w = min(max(float(wall), 0.0), 0.999)
    return np.clip((g - w) / (1.0 - w), 0.0, 1.0).astype(np.float32)


def sample(W=1280, H=720) -> np.ndarray:
    """Nothing connected: every window of the sample facade as a soft hill."""
    lab, n, _ = mf.sample_regions(W, H)
    f, _inside = mf.field(lab, n, "edge distance")
    return f.astype(np.float32)


def ramp_over(h) -> np.ndarray:
    """No field given for 'along the field': left to right over where the height is."""
    lab, n = mf.from_masks((h > 0).astype(np.float32)[None], split=False, min_area=1)
    if n == 0:
        return np.zeros(h.shape, np.float32)
    return mf.field(lab, n, "direction", "all together")[0]


def grown(h, v, grow=GROWS[0], field=None, soft=0.1) -> np.ndarray:
    """The height at value v (0 = the bare wall, 1 = the whole height; above 1 'all together' overshoots).

    all together        every form rises at once: h * v
    peaks first         the forms are pushed out through the wall, the highest points come first: h - (1 - v), not below 0
    built up in layers  from the wall up, cut flat at the level reached: the lower of h and v
    along the field     the forms rise where the field (black first, white last) has been passed; soft = how wide the
                        rising edge is
    """
    v = float(v)
    if grow == "peaks first":
        return np.clip(h - (1.0 - min(v, 1.0)), 0.0, None).astype(np.float32, copy=False)      # above 1 the wall would rise too
    if grow == "built up in layers":
        return np.minimum(h, max(v, 0.0)).astype(np.float32, copy=False)
    if grow == "along the field" and field is not None:
        return (h * mf.along(field, min(max(v, 0.0), 1.0), "reveal", soft)).astype(np.float32, copy=False)
    return (h * max(v, 0.0)).astype(np.float32, copy=False)


def lit(h, base=None, angle=0.0, relief=1.0, width=None, blur=BLUR) -> np.ndarray:
    """The height as a relief lit from the side: float32 [H, W, 3] in 0-1.

    base    the picture on the relief [H, W, 3] (the facade, a clay render, a texture); None = clay grey
    angle   where the light comes from, degrees: 0 = from the left, 90 = from above, 180 = from the right
    relief  how deep the relief reads: at 1 a height of 1 stands a tenth of the picture's width out
    width   the width of the whole picture when h is a part of it (a layer's box); None = h's own
    blur    pixels the height is smoothed by first (an 8 bit depth map has steps: they would show as lines)
    """
    smooth = cv2.GaussianBlur(h, (0, 0), float(blur)) if blur >= 0.3 else h
    k = float(relief) * 0.1 * (width or h.shape[1])
    gx = cv2.Sobel(smooth, cv2.CV_32F, 1, 0, ksize=3, scale=0.125) * k
    gy = cv2.Sobel(smooth, cv2.CV_32F, 0, 1, ksize=3, scale=0.125) * k
    a = math.radians(float(angle))
    lx, ly, lz = -math.cos(a) * math.cos(LIGHT_HEIGHT), -math.sin(a) * math.cos(LIGHT_HEIGHT), math.sin(LIGHT_HEIGHT)
    shade = (-gx * lx - gy * ly + lz) / (np.sqrt(gx * gx + gy * gy + 1.0) * lz)      # 1 on the flat wall
    shade = np.clip(0.75 * shade + 0.05, 0.0, 1.25).astype(np.float32, copy=False)[..., None]
    out = np.repeat(shade, 3, -1) if base is None else base * shade
    return np.clip(out, 0.0, 1.0, out=out)


def layer_look(rgb, value, mode=GROWS[0], wall=0.0, look=LOOKS[0], angle=35.0, relief=1.0, width=None) -> np.ndarray:
    """A director layer whose picture is a height map, at value: the grown height as grey, or as a lit relief
    (float32 [H, W, 3]), or as shading: the relief with the flat wall white, which changes nothing there under blend
    multiply and darkens what is under a form's shadow side. rgb is the layer's placed picture, width the canvas
    width (the look is the same in the small window and the full render). web/kubakub_director.js heightGrown /
    heightShade are the twins."""
    g = np.clip(grown(from_grey(rgb, wall), value, mode), 0.0, 1.0)
    if look in LOOKS[1:]:
        w = width or g.shape[1]
        out = lit(g, None, angle, relief, w, LAYER_BLUR * w / 1920.0)
        return np.minimum(out / FLAT, 1.0, out=out) if look == "shading" else out
    return np.repeat(g[..., None], 3, -1)
