"""
imio.py

cv2.imread / cv2.imwrite for any path. On Windows OpenCV opens files with the ANSI code page, so a path with other
characters (a user folder named Zażółć, a project folder with an umlaut) reads as None and writes nothing,
without an error. These go through numpy's file access instead and behave the same everywhere.
"""

from __future__ import annotations

import os

import cv2
import numpy as np


def imread(path, flags=cv2.IMREAD_COLOR):
    """The image as cv2.imread returns it (BGR / BGRA / grey, uint8 or uint16), or None when it cannot be read."""
    try:
        data = np.fromfile(path, np.uint8)
    except OSError:
        return None
    return cv2.imdecode(data, flags) if data.size else None


def pil_rgb01(img) -> np.ndarray:
    """A PIL image as float32 H x W x 3 in 0..1. 16-bit and float greyscale (depth passes, After Effects masks) keep
    their range; PIL's own convert('RGB') clips them to white."""
    if img.mode in ("I;16", "I;16B", "I;16L", "I", "F"):
        a = np.asarray(img).astype(np.float32)
        if img.mode != "F":
            a = a / (65535.0 if a.max() > 255 else 255.0)
        return np.repeat(np.clip(a, 0, 1)[..., None], 3, axis=2)
    return np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0


def imwrite(path, img, params=None) -> bool:
    """Writes img in the format of the file's extension (.png when there is none). False when encoding fails."""
    ok, buf = cv2.imencode(os.path.splitext(path)[1] or ".png", img, params or [])
    if not ok:
        return False
    buf.tofile(path)
    return True
