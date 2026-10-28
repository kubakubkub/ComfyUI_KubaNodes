"""
project.py

The project settings of one show: matrix size, frame rate, the facade in metres and where the audience stands.
Pure logic for kubakub project settings (nodes_project.py); no ComfyUI imports, so the tests run without a server.
"""

from __future__ import annotations

import re

from . import viewer as vw

# name -> (width, height); "custom" uses the width / height inputs
PRESETS = {
    "custom": None,
    "hd 1920x1080": (1920, 1080),
    "uhd 3840x2160": (3840, 2160),
    "dci 4k 4096x2160": (4096, 2160),
    "8k 7680x4320": (7680, 4320),
}


def safe_name(name: str) -> str:
    """A name usable for folders and files on every system ('' -> 'project')."""
    s = re.sub(r"[^\w\-. ]+", " ", str(name or "")).strip(" .")
    return re.sub(r"\s+", "_", s) or "project"


def resolve_size(preset: str, width: int, height: int, matrix_shape=None) -> tuple[int, int, str]:
    """(width, height, where it came from): a connected matrix wins over the preset, the preset over width / height."""
    if matrix_shape is not None:
        return int(matrix_shape[2]), int(matrix_shape[1]), "matrix"
    wh = PRESETS.get(preset)
    if wh:
        return wh[0], wh[1], "preset"
    return int(width), int(height), "custom"


def eye_preview(viewer, picture=None):
    """The matrix (float32 [h, w, 3]; None = a dark frame of its shape, at most 960 wide) with a red dot where the
    audience's eye line meets the facade -> float32 [h, w, 3]."""
    import cv2
    import numpy as np
    W, H = int(viewer.width_px), int(viewer.height_px)
    if picture is None:
        k = min(1.0, 960.0 / max(W, 1))
        prev = np.full((max(1, int(round(H * k))), max(1, int(round(W * k))), 3), 28, np.uint8)
    else:
        k = 1.0
        prev = (np.clip(picture[..., :3], 0.0, 1.0) * 255).astype(np.uint8).copy()
    h, w = prev.shape[:2]
    fx, fy = viewer.foot_px()
    r = max(6, w // 200)
    cv2.circle(prev, (int(round(fx * k)), int(round(min(max(fy * k, r), h - r)))), r, (255, 40, 40), -1)
    return prev.astype(np.float32) / 255


def whole_fps(fps) -> int:
    """Frames per second as the director uses them: a whole number from 1 to 120 (29.97 -> 30)."""
    try:
        f = float(fps)
    except (TypeError, ValueError):
        return 25
    return max(1, min(120, int(round(f)))) if f == f else 25


def build(name, preset, width, height, fps, facade_width_m, bottom_m, viewer_x_m, eye_height_m, distance_m,
          matrix_shape=None) -> dict:
    """Everything the node outputs, plus a one-line report."""
    w, h, src = resolve_size(preset, width, height, matrix_shape)
    fps = whole_fps(fps)
    viewer = vw.Viewer(w, h, float(facade_width_m), float(bottom_m), None if viewer_x_m < 0 else float(viewer_x_m),
                       float(eye_height_m), float(distance_m))
    nm = safe_name(name)
    report = (f"{nm}: matrix {w}x{h} ({src}), {fps:g} fps; facade {facade_width_m:g} x {viewer.facade_height_m:.1f} m, "
              f"{viewer.m_per_px * 1000:.1f} mm per px; audience {distance_m:g} m in front, eye {eye_height_m:g} m")
    return {"name": nm, "width": w, "height": h, "fps": float(fps), "viewer": viewer, "report": report}
