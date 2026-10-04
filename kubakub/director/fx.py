"""
fx.py

Layer effects of the kubakub director: blur, sharpen, glow and grain, applied to a layer's own pixels (after its
region clip, before masks by other layers) or, on a colour-grade layer, to everything below it. Distances are in
pixels of the delivery size. The window runs the same chain as one SVG filter per layer (kubakub_director.js
fxFilter); this is the render's version. numpy + OpenCV, no ComfyUI imports (tests/test_fx.py).

Order (as in the window): blur -> sharpen (unsharp mask) -> glow (bright parts, blurred, added) -> grain.
Grain is seeded per frame: the same frame renders the same grain; it matches the window's grain in strength and
size, not pixel for pixel (the browser's noise generator is its own).
"""

from __future__ import annotations

import cv2
import numpy as np

RANGES = {"blur": (0.0, 200.0), "sharpen": (0.0, 5.0), "sharpen_radius": (0.3, 20.0), "glow": (0.0, 5.0),
          "glow_radius": (0.0, 300.0), "glow_threshold": (0.0, 0.99), "grain": (0.0, 1.0), "grain_size": (0.5, 20.0)}
DEFAULTS = {"blur": 0.0, "sharpen": 0.0, "sharpen_radius": 1.5, "glow": 0.0, "glow_radius": 20.0, "glow_threshold": 0.6,
            "grain": 0.0, "grain_size": 1.5}


def parse(fx) -> dict:
    fx = fx if isinstance(fx, dict) else {}
    out = {}
    for k, d in DEFAULTS.items():
        v = fx.get(k, d)
        v = float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else d
        lo, hi = RANGES[k]
        out[k] = min(max(v, lo), hi)
    return out


def active(fx) -> bool:
    return bool(fx) and (fx["blur"] > 0.05 or fx["sharpen"] > 0.001 or (fx["glow"] > 0.001 and fx["glow_radius"] > 0) or fx["grain"] > 0.001)


def margin(fx) -> int:
    """How far the effects spread past the layer's pixels (px)."""
    if not active(fx):
        return 0
    m = 3 * fx["blur"]
    if fx["glow"] > 0.001:
        m += 3 * fx["glow_radius"]
    return int(np.ceil(m))


def _blur(a, sigma):
    return cv2.GaussianBlur(a, (0, 0), sigmaX=sigma, borderType=cv2.BORDER_CONSTANT) if sigma > 0.05 else a


def apply(rgb, alpha, fx, scale=1.0, seed=0):
    """rgb (h, w, 3) and alpha (h, w) float -> (rgb, alpha) with the effects; `scale` = pixels per delivery pixel
    (a smaller sequence render). Works premultiplied like SVG filters."""
    if not active(fx):
        return rgb, alpha
    if alpha is None and fx["blur"] <= 0.05 and fx["sharpen"] <= 0.001:
        return _apply_opaque(rgb, fx, scale, seed)
    a = np.ones(rgb.shape[:2], np.float32) if alpha is None else alpha.astype(np.float32)
    pre = np.concatenate([rgb.astype(np.float32) * a[..., None], a[..., None]], -1)       # premultiplied RGBA
    if fx["blur"] > 0.05:
        pre = _blur(pre, fx["blur"] * scale)
    if fx["sharpen"] > 0.001:                              # unsharp mask: (1 + s) x - s blur(x)
        s = fx["sharpen"]
        pre = np.clip((1 + s) * pre - s * _blur(pre, fx["sharpen_radius"] * scale), 0, None)
        pre[..., :3] = np.minimum(pre[..., :3], pre[..., 3:4])
    if fx["glow"] > 0.001 and fx["glow_radius"] > 0:       # the bright parts, blurred, added on top
        th = fx["glow_threshold"]
        al = np.maximum(pre[..., 3:4], 1e-6)
        bright = np.clip((pre[..., :3] / al - th) / (1 - th), 0, 1) * pre[..., 3:4]
        g = _blur(np.concatenate([bright, pre[..., 3:4]], -1), fx["glow_radius"] * scale)
        pre = np.clip(pre + fx["glow"] * g, 0, 1)
    a = np.clip(pre[..., 3], 0, 1)
    out = np.where(a[..., None] > 1e-6, pre[..., :3] / np.maximum(a[..., None], 1e-6), 0)
    out = _grain(out, fx, scale, seed)
    return np.clip(out, 0, 1).astype(np.float32), a.astype(np.float32)


def _grain(out, fx, scale, seed):
    if fx["grain"] > 0.001:                                # grey noise blended (overlay) into the picture
        h, w = out.shape[:2]
        size = max(0.5, fx["grain_size"] * scale)
        gh, gw = max(2, int(round(h / size))), max(2, int(round(w / size)))
        n = np.random.default_rng(int(seed) & 0x7FFFFFFF).normal(0.5, 0.15, (gh, gw)).astype(np.float32)
        n = np.clip(cv2.resize(n, (w, h), interpolation=cv2.INTER_CUBIC), 0, 1)[..., None]
        ov = np.where(out < 0.5, 2 * out * n, 1 - 2 * (1 - out) * (1 - n))
        out = out * (1 - fx["grain"]) + ov * fx["grain"]
    return out


def _apply_opaque(rgb, fx, scale, seed):
    """apply() for an opaque picture (a colour-grade layer's frame) with glow / grain only. alpha = 1 everywhere, so
    every '* a' and '/ a' of the premultiplied path is an exact no-op and the glow's blurred alpha always clips back
    to 1 (1 + glow * g >= 1): the same numbers from 3 channels, no 4-channel copies (bit-identical)."""
    out = rgb.astype(np.float32)
    if fx["glow"] > 0.001 and fx["glow_radius"] > 0:
        th = fx["glow_threshold"]
        bright = out - th
        bright /= (1 - th)
        np.clip(bright, 0, 1, out=bright)
        g = _blur(bright, fx["glow_radius"] * scale)      # OpenCV blurs each channel on its own: = the RGB of the 4-channel blur
        g *= fx["glow"]
        g += out                                          # pre + glow * g (the sum does not depend on the order)
        out = np.clip(g, 0, 1, out=g)
    out = _grain(out, fx, scale, seed)
    return np.clip(out, 0, 1).astype(np.float32), np.ones(rgb.shape[:2], np.float32)
