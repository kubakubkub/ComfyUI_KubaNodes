"""
compensate.py

Brightness compensation: a projector lights near, frontal surfaces more than far or turned-away ones
(cos(incidence) / distance^2, the 'brightness' map of scene_view.measure_maps). A gain map evens that out, so the
content reads the same brightness everywhere on the building. Light can only be taken away (the brightest white stays
the brightest white), so by default the bright parts are darkened towards the dim ones; max_boost > 1 lets dim parts
be lifted as well, at the cost of clipping highlights there.

Gains work on linear light (sRGB decoded), so the look of the content stays right. numpy + opencv, no ComfyUI imports
(tests/test_compensate.py).
"""

from __future__ import annotations

import cv2
import numpy as np

MATCH = {  # name -> percentile of the building's brightness that everything is evened to
    "dim areas": 10.0,
    "average": 50.0,
}


def _fill_outside(g, fg):
    """Values outside fg = the nearest inside value, so resizing / smoothing makes no halo at the building's edge."""
    if fg.all() or not fg.any():
        return g
    from scipy.ndimage import distance_transform_edt
    iy, ix = distance_transform_edt(~fg, return_distances=False, return_indices=True)
    return g[iy, ix].astype(np.float32)


def gain_map(brightness, fg, match="dim areas", strength=1.0, max_boost=1.0, smooth_px=0.0, labels=None,
             incidence=None, grazing_deg=60.0):
    """(gain HxW float32, info). gain = (target / brightness) ** strength, capped at max_boost; 1 outside fg.

    labels (HxW int, -1 = none): one gain per region (its median), so a region never gets a gradient inside.
    incidence (HxW degrees): surfaces hit steeper than grazing_deg (side faces of mouldings, reveals) do not set the
    target; they are dim by nature and no gain can fix them, but they would drag the whole building down."""
    b = np.asarray(brightness, np.float32)
    fg = np.asarray(fg, bool) & (b > 1e-6)
    if not fg.any():
        return np.ones_like(b), {"target": 1.0, "kept": 1.0, "min": 1.0, "max": 1.0, "grazing": 0.0}
    pct = MATCH.get(match, 10.0)
    facing = fg if incidence is None else fg & (np.asarray(incidence) <= grazing_deg)
    if facing.sum() < 0.05 * fg.sum():                 # almost everything grazing: use all
        facing = fg
    target = float(np.percentile(b[facing], pct))
    g = np.ones_like(b)
    g[fg] = np.power(target / b[fg], float(strength))
    g = np.minimum(g, float(max_boost))
    if labels is not None:
        lab = np.asarray(labels)
        if lab.shape != b.shape:
            lab = cv2.resize(lab.astype(np.int32), (b.shape[1], b.shape[0]), interpolation=cv2.INTER_NEAREST)
        m = fg & (lab >= 0)
        if m.any():
            ids = lab[m]
            order = np.argsort(ids, kind="stable")
            ids_s, vals_s = ids[order], g[m][order]
            starts = np.r_[0, np.flatnonzero(np.diff(ids_s)) + 1]
            med = np.array([np.median(vals_s[a:e]) for a, e in zip(starts, np.r_[starts[1:], len(ids_s)])], np.float32)
            lut = np.ones(int(ids_s.max()) + 1, np.float32)
            lut[ids_s[starts]] = med
            g = g.copy()
            g[m] = lut[lab[m]]
    g = _fill_outside(g, fg)
    if smooth_px > 0:
        k = int(round(smooth_px)) * 2 + 1
        g = cv2.GaussianBlur(g, (k, k), 0)
    inside = g[fg]
    kept = float(np.mean(np.minimum(inside, 1.0) * b[fg]) / max(float(np.mean(b[fg])), 1e-12))
    return g.astype(np.float32), {"target": target, "kept": kept, "min": float(inside.min()),
                                  "max": float(inside.max()), "grazing": 1.0 - float(facing.sum()) / float(fg.sum())}


def srgb_to_linear(x):
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.04045, x / 12.92, np.power((x + 0.055) / 1.055, 2.4)).astype(np.float32)


def linear_to_srgb(x):
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * np.power(x, 1 / 2.4) - 0.055).astype(np.float32)


def resize_gain(gain, w, h):
    """The gain at the content's size (the projection view is often smaller than the matrix)."""
    if gain.shape == (h, w):
        return gain
    return cv2.resize(gain, (w, h), interpolation=cv2.INTER_LINEAR)


_N = 65535                       # 16-bit lookup tables: exact enough for 8 / 10 / 16 bit delivery, ~10x faster than pow
_DEC = srgb_to_linear(np.arange(_N + 1, dtype=np.float32) / _N)
_ENC = linear_to_srgb(np.arange(_N + 1, dtype=np.float32) / _N)


def apply(frame, gain):
    """One sRGB frame (HxWx3, 0..1) times the gain in linear light -> (frame, share of clipped pixels)."""
    idx = np.clip(frame, 0.0, 1.0) * _N
    lin = _DEC[(idx + 0.5).astype(np.uint16)] * gain[..., None]
    clipped = float(np.mean(lin.max(-1) > 1.0 + 1e-4)) if gain.max() > 1.0 else 0.0
    out = _ENC[(np.clip(lin, 0.0, 1.0) * _N + 0.5).astype(np.uint16)]
    return out, clipped
