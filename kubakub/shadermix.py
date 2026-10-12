"""
shadermix.py

Two shaders as one picture, with one mix value: 0 is shader A, 1 is shader B. Pure numpy (tests/test_shadermix.py);
the node (nodes/shader/nodes_mix.py) draws A and B with shader_render and joins them here.

How A and B meet in between (`how`):
  b works on a   B takes A's picture as its picture (image0), so B is an effect on A; the mix fades it in
  a bends b      A's picture pushes the places B draws at (red = sideways, green = up and down); B melts out of A
  a cuts b       B shows first where A is bright and takes over from there: a wipe in the shape of A's pattern
  blend          a plain fade
"""

from __future__ import annotations

import math

import numpy as np

HOW = ("b works on a", "a bends b", "a cuts b", "blend")
LUMA = np.array((0.2126, 0.7152, 0.0722), np.float32)


def bend(m: float, strength: float) -> float:
    """How far A bends B at mix m, as a part of the picture's height: nothing at both ends (A alone, B alone), the
    most half way."""
    m = min(1.0, max(0.0, float(m)))
    return float(strength) * math.sin(math.pi * m)


def bend_picture(a: np.ndarray, b: np.ndarray, amount: float) -> np.ndarray:
    """B's finished picture with its places pushed around by A's red and green, as far as `amount` of the height:
    what 'a bends b' does inside shader B, for a B that comes as a picture. [h, w, 3] float each."""
    if amount == 0.0:
        return b
    import cv2
    h, w = b.shape[:2]
    reach = 2.0 * float(amount) * h
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    return cv2.remap(np.ascontiguousarray(b[..., :3], np.float32), xx + (a[..., 0] - 0.5) * reach, yy - (a[..., 1] - 0.5) * reach,
                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def cut(a: np.ndarray, m: float, soft: float = 0.1) -> np.ndarray:
    """Where B shows at mix m when A cuts it: [h, w, 1] 0-1. Nothing at 0, everything at 1; in between B has taken
    the places where A is brighter than a level that falls with m."""
    s = max(1e-4, float(soft))
    level = (1.0 - min(1.0, max(0.0, float(m)))) * (1.0 + 2.0 * s) - s
    k = np.clip((a[..., :3] @ LUMA - (level - s)) / (2.0 * s), 0.0, 1.0)
    return (k * k * (3.0 - 2.0 * k))[..., None].astype(np.float32)


def join(a: np.ndarray, b: np.ndarray, m: float, how: str = "blend", soft: float = 0.1) -> np.ndarray:
    """A and B ([h, w, 3+] float 0-1) at mix m -> [h, w, 3] float32. For 'b works on a' and 'a bends b' the meeting
    has happened while B was drawn; here it only fades from A to that B."""
    m = min(1.0, max(0.0, float(m)))
    if m <= 0.0:
        return np.asarray(a[..., :3], np.float32)
    if m >= 1.0:
        return np.asarray(b[..., :3], np.float32)
    k = cut(a, m, soft) if how == "a cuts b" else np.float32(m)
    return (a[..., :3] * (1.0 - k) + b[..., :3] * k).astype(np.float32)
