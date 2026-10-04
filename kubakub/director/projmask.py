"""
projmask.py

The projection mask of the kubakub director: the building's silhouette at the delivery resolution. Everything
outside it is black in the window, the still, the sequence and the export (transparent in an export with alpha).
Pure numpy + OpenCV, no ComfyUI imports (tests/test_projmask.py).

Document key (optional; missing or on=false = no mask):
    "projection_mask": {"on": true, "source": "file:kuba_director/x.png" | "path:D:/.../x.png" | "scene",
                        "invert": false, "grow": 0, "feather": 0, "view": "black" | "dim" | "off"}
- source "scene": the 3D scene's silhouette (every pixel the model covers).
- a mask file comes in one of the usual template forms, read automatically:
  an overlay with alpha that is dark where it is opaque (opaque black outside, transparent building, as
  festivals deliver it), a cutout with alpha that is bright where it is opaque (the building opaque), or no alpha:
  white building on black. invert flips the result.
- grow: pixels to grow (+) or shrink (-) the building; feather: soft edge width in pixels.
- view: only the window's display (black = as projected, dim = outside darkened, off = while working). The render
  and the export always apply the mask when it is on.
"""

from __future__ import annotations

import cv2
import numpy as np

VIEWS = ("black", "dim", "off")


def settings(doc) -> dict | None:
    """The document's projection mask settings, or None when there is none / it is off."""
    pm = (doc or {}).get("projection_mask") if isinstance(doc, dict) else None
    if not isinstance(pm, dict) or not pm.get("on") or not str(pm.get("source") or ""):
        return None

    def num(k, lo, hi):
        v = pm.get(k, 0)
        return float(min(max(v, lo), hi)) if isinstance(v, (int, float)) and not isinstance(v, bool) else 0.0

    return {"source": str(pm["source"]), "invert": bool(pm.get("invert")), "grow": num("grow", -200, 200),
            "feather": num("feather", 0, 400), "view": pm.get("view") if pm.get("view") in VIEWS else "black"}


def from_image(img) -> tuple[np.ndarray, str]:
    """A template mask image (float or uint8, HxW, HxWx3 or HxWx4) -> (building mask float32 0..1, how it was read)."""
    a = np.asarray(img)
    a = a.astype(np.float32) / (65535.0 if a.dtype == np.uint16 else 255.0) if a.dtype != np.float32 else a
    if a.ndim == 2:
        return np.clip(a, 0, 1), "grey: white = building"
    rgb = a[..., :3]
    lum = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
    if a.shape[-1] == 4:
        al = a[..., 3]
        opaque = al > 0.5
        if 0.001 < opaque.mean() < 0.999:            # a real alpha: overlay (dark where opaque) or cutout (bright)
            if float(lum[opaque].mean()) < 0.5:
                return np.clip(1 - al, 0, 1), "overlay: transparent = building"
            return np.clip(al, 0, 1), "cutout: opaque = building"
    return np.clip(lum, 0, 1), "white = building"


def refine(mask, grow=0.0, feather=0.0, invert=False) -> np.ndarray:
    """Invert, grow (+) / shrink (-) by pixels, then a soft edge of `feather` pixels."""
    m = np.asarray(mask, np.float32)
    if invert:
        m = 1 - m
    g = int(round(grow))
    if g:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * abs(g) + 1, 2 * abs(g) + 1))
        m = cv2.dilate(m, k) if g > 0 else cv2.erode(m, k)
    if feather > 0.5:
        m = cv2.GaussianBlur(m, (0, 0), sigmaX=feather / 2.0, borderType=cv2.BORDER_REPLICATE)
    return np.clip(m, 0, 1).astype(np.float32)


def fit(mask, W, H) -> tuple[np.ndarray, str]:
    """The mask at W x H (a note when it had to be scaled: a template mask should match the delivery size)."""
    h, w = mask.shape[:2]
    if (w, h) == (W, H):
        return mask, ""
    return cv2.resize(mask, (W, H), interpolation=cv2.INTER_AREA if w > W else cv2.INTER_LINEAR), \
        f"projection mask {w}x{h} scaled to {W}x{H} (the template mask should have the delivery size)"


def apply(rgb, mask) -> np.ndarray:
    """Outside the building black (in place when possible, strip by strip for large frames)."""
    out = rgb
    H = rgb.shape[0]
    for r0 in range(0, H, 512):
        r1 = min(H, r0 + 512)
        out[r0:r1] *= mask[r0:r1, :, None]
    return out
