"""
blend.py

Blend modes and colour grading exactly as the browser canvas does them, so the kubakub director's
preview (Canvas2D globalCompositeOperation and ctx.filter) and the full-resolution render in Python
look the same. numpy + OpenCV (tests/test_director.py).

Sources: W3C "Compositing and Blending Level 1" (separable and non-separable blend modes, source-over
with an opaque backdrop) and W3C "Filter Effects Module Level 1" (brightness, contrast, saturate,
hue-rotate, sepia as colour matrices), both in sRGB like the canvas implementations. The layout of the
table follows ComfyUI core's comfy_extras/compositor_blend.py (Comfy Org); the soft-light and
non-separable formulas here are the W3C ones the browser uses.
"""

from __future__ import annotations

import cv2
import numpy as np

EPS = 1e-6


def _soft_light(b, s):
    if b.dtype != np.float32 or s.dtype != np.float32 or b.shape != s.shape:
        d = np.where(b <= 0.25, ((16 * b - 12) * b + 4) * b, np.sqrt(np.maximum(b, 0)))
        return np.where(s <= 0.5, b - (1 - 2 * s) * b * (1 - b), b + (2 * s - 1) * (d - b))
    # the same arithmetic in the same order, in place (a few temporaries instead of ~16): bit-identical
    d = np.multiply(b, 16, dtype=b.dtype)
    d -= 12; d *= b; d += 4; d *= b                       # ((16 b - 12) b + 4) b
    sq = np.maximum(b, 0); np.sqrt(sq, out=sq)
    np.copyto(d, sq, where=~(b <= 0.25))                  # d = where(b <= 0.25, poly, sqrt)
    lo = np.multiply(s, 2, dtype=b.dtype); np.subtract(1, lo, out=lo)          # 1 - 2 s
    lo *= b; lo *= np.subtract(1, b, dtype=b.dtype); np.subtract(b, lo, out=lo)   # b - (1 - 2 s) b (1 - b)
    hi = np.multiply(s, 2, dtype=b.dtype); hi -= 1       # 2 s - 1
    d -= b; hi *= d; hi += b                              # b + (2 s - 1)(d - b)
    np.copyto(hi, lo, where=(s <= 0.5))
    return hi


def _hard_light(b, s):
    return np.where(s <= 0.5, b * 2 * s, 1 - (1 - b) * (1 - (2 * s - 1)))


SEPARABLE = {
    "normal": lambda b, s: s,
    "multiply": lambda b, s: b * s,
    "screen": lambda b, s: b + s - b * s,
    "overlay": lambda b, s: _hard_light(s, b),
    "darken": np.minimum,
    "lighten": np.maximum,
    "color dodge": lambda b, s: np.where(b <= 0, 0.0, np.where(s >= 1, 1.0, np.minimum(1.0, b / np.maximum(1 - s, EPS)))),
    "color burn": lambda b, s: np.where(b >= 1, 1.0, np.where(s <= 0, 0.0, 1 - np.minimum(1.0, (1 - b) / np.maximum(s, EPS)))),
    "hard light": _hard_light,
    "soft light": _soft_light,
    "difference": lambda b, s: np.abs(b - s),
    "exclusion": lambda b, s: b + s - 2 * b * s,
}


def _lum(c):
    return 0.3 * c[..., 0] + 0.59 * c[..., 1] + 0.11 * c[..., 2]


def _clip_color(c):
    l = _lum(c)[..., None]
    n = c.min(axis=-1, keepdims=True)
    x = c.max(axis=-1, keepdims=True)
    c = np.where(n < 0, l + (c - l) * l / np.maximum(l - n, EPS), c)
    c = np.where(x > 1, l + (c - l) * (1 - l) / np.maximum(x - l, EPS), c)
    return c


def _set_lum(c, l):
    return _clip_color(c + (l - _lum(c))[..., None])


def _sat(c):
    return c.max(axis=-1) - c.min(axis=-1)


def _set_sat(c, s):
    mx = c.max(axis=-1, keepdims=True)
    mn = c.min(axis=-1, keepdims=True)
    rng = mx - mn
    out = np.where(rng > EPS, (c - mn) * s[..., None] / np.maximum(rng, EPS), 0.0)
    return out


NON_SEPARABLE = {
    "hue": lambda b, s: _set_lum(_set_sat(s, _sat(b)), _lum(b)),
    "saturation": lambda b, s: _set_lum(_set_sat(b, _sat(s)), _lum(b)),
    "color": lambda b, s: _set_lum(s, _lum(b)),
    "luminosity": lambda b, s: _set_lum(b, _lum(s)),
}

MODES = tuple(SEPARABLE) + tuple(NON_SEPARABLE)


def blend(mode: str, backdrop: np.ndarray, source: np.ndarray) -> np.ndarray:
    """B(Cb, Cs) for RGB arrays in 0..1."""
    if mode in SEPARABLE:
        return SEPARABLE[mode](backdrop, source)
    if mode in NON_SEPARABLE:
        return NON_SEPARABLE[mode](backdrop, source)
    raise ValueError(f"unknown blend mode {mode!r}; known: {', '.join(MODES)}")


def composite_over(backdrop: np.ndarray, source: np.ndarray, alpha: np.ndarray, mode: str = "normal") -> np.ndarray:
    """Source-over onto an opaque backdrop: Cb * (1 - a) + B(Cb, Cs) * a (W3C, alpha_b = 1)."""
    a = np.clip(alpha, 0, 1)[..., None]
    return backdrop * (1 - a) + np.clip(blend(mode, backdrop, source), 0, 1) * a


def composite_into(sub: np.ndarray, source: np.ndarray, alpha: np.ndarray, mode: str = "normal") -> np.ndarray:
    """composite_over written into `sub` (a strip of the frame): the same operations in the same order
    (Cb * (1 - a) + clip(B) * a), fewer temporaries, bit-identical. Returns sub."""
    a = np.clip(alpha, 0, 1)[..., None]
    b = blend(mode, sub, source)
    own = b.dtype == sub.dtype and not np.may_share_memory(b, source) and not np.may_share_memory(b, sub)
    t = np.clip(b, 0, 1, out=b if own else None)          # 'normal' hands back the source itself: never clip that in place
    t *= a
    sub *= (1 - a)
    sub += t
    return sub


# ------------------------------------------------------------------------------------------------
# colour grade = the CSS filter chain brightness() contrast() saturate() hue-rotate() sepia()
# ------------------------------------------------------------------------------------------------

def _saturate_matrix(s):
    return np.array([[0.213 + 0.787 * s, 0.715 - 0.715 * s, 0.072 - 0.072 * s],
                     [0.213 - 0.213 * s, 0.715 + 0.285 * s, 0.072 - 0.072 * s],
                     [0.213 - 0.213 * s, 0.715 - 0.715 * s, 0.072 + 0.928 * s]])


def _hue_matrix(deg):
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    return (np.array([[0.213, 0.715, 0.072], [0.213, 0.715, 0.072], [0.213, 0.715, 0.072]])
            + c * np.array([[0.787, -0.715, -0.072], [-0.213, 0.285, -0.072], [-0.213, -0.715, 0.928]])
            + s * np.array([[-0.213, -0.715, 0.928], [0.143, 0.140, -0.283], [-0.787, 0.715, 0.072]]))


def _sepia_matrix(amount):
    t = 1 - min(max(amount, 0.0), 1.0)
    return np.array([[0.393 + 0.607 * t, 0.769 - 0.769 * t, 0.189 - 0.189 * t],
                     [0.349 - 0.349 * t, 0.686 + 0.314 * t, 0.168 - 0.168 * t],
                     [0.272 - 0.272 * t, 0.534 - 0.534 * t, 0.131 + 0.869 * t]])


def grade(rgb: np.ndarray, brightness=1.0, contrast=1.0, saturate=1.0, hue=0.0, sepia=0.0) -> np.ndarray:
    """The CSS filter chain in the order the director writes it; every step clamps to 0..1 like the browser."""
    out = np.multiply(rgb, brightness)                     # a new array: the steps below work in place
    np.clip(out, 0, 1, out=out)
    out -= 0.5; out *= contrast; out += 0.5
    np.clip(out, 0, 1, out=out)
    for m in (_saturate_matrix(saturate) if saturate != 1 else None,
              _hue_matrix(hue) if hue else None,
              _sepia_matrix(sepia) if sepia else None):
        if m is not None:
            if out.dtype == np.float32 and out.ndim == 3 and out.shape[-1] == 3:
                out = cv2.transform(out, m.astype(np.float32))   # per pixel m @ c (rounds within 3e-7 of the matmul)
                np.clip(out, 0, 1, out=out)
            else:
                out = np.clip(out @ m.T.astype(np.float32), 0, 1)   # float32: no upcast of the frame
    return out.astype(np.float32, copy=False)
