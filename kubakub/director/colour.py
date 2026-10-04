"""
colour.py

Colour management of the kubakub director. One working space: display-referred sRGB / Rec.709 (what projectors
and media servers are calibrated to), float 0..1 end to end, blending in that space like the browser and After
Effects' default. This module brings images into it: an embedded ICC profile (Adobe RGB, P3, ProPhoto, CMYK from
Illustrator / Photoshop ...) is converted to sRGB with LittleCMS (Pillow's ImageCms), the way the browser shows the
file in the window. ComfyUI's own loaders ignore these profiles, which is the usual "the colours changed" case.
No ComfyUI imports (tests/test_colour.py).
"""

from __future__ import annotations

import io
import os

import cv2
import numpy as np


def _profile_name(icc: bytes) -> str:
    from PIL import ImageCms
    try:
        return ImageCms.getProfileDescription(ImageCms.ImageCmsProfile(io.BytesIO(icc))).strip()
    except Exception:  # noqa: BLE001
        return "unknown profile"


def is_srgb_name(name: str) -> bool:
    n = name.lower().replace(" ", "")
    return "srgb" in n or "iec61966-2.1" in n or "iec61966-2-1" in n


def read_image(path: str):
    """
    An image file -> (RGBA float32 0..1 in sRGB, note). Colour-managed: an embedded non-sRGB ICC profile is converted
    to sRGB (relative colorimetric, as browsers do); without a profile (or an sRGB one) the pixels are read as they
    are, 16-bit kept. Returns (None, note) when the file cannot be read.
    """
    from PIL import Image, ImageCms
    note = ""
    try:
        im = Image.open(path)
        icc = im.info.get("icc_profile")
    except Exception as e:  # noqa: BLE001
        im, icc = None, None
        note = f"{os.path.basename(path)}: read without colour profile ({e})"
    if im is not None and icc:
        name = _profile_name(icc)
        if not is_srgb_name(name):
            try:
                if im.mode in ("I;16", "I;16B", "I;16L", "I", "F"):
                    note = f"{os.path.basename(path)}: 16-bit / float with profile '{name}' kept as it is (not converted)"
                else:
                    has_a = "A" in im.getbands() or "transparency" in im.info
                    alpha = im.convert("RGBA").getchannel("A") if has_a else None
                    src = im if im.mode in ("RGB", "CMYK", "L") else im.convert("RGB")
                    conv = ImageCms.profileToProfile(src, ImageCms.ImageCmsProfile(io.BytesIO(icc)), ImageCms.createProfile("sRGB"),
                                                     renderingIntent=ImageCms.Intent.RELATIVE_COLORIMETRIC, outputMode="RGB")
                    rgb = np.asarray(conv, np.float32) / 255.0
                    a = np.asarray(alpha, np.float32)[..., None] / 255.0 if alpha is not None else np.ones(rgb.shape[:2] + (1,), np.float32)
                    return np.concatenate([rgb, a], -1), f"{os.path.basename(path)}: colour profile '{name}' converted to sRGB"
            except Exception as e:  # noqa: BLE001
                note = f"{os.path.basename(path)}: colour profile '{name}' not applied ({e})"
    img = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_UNCHANGED)
    if img is None:
        return None, note or f"{os.path.basename(path)}: not readable"
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
    elif img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
    scale = 65535.0 if img.dtype == np.uint16 else 255.0
    return img.astype(np.float32) / scale, note
