"""
bg.py

Background removal for kubakub director layers with ComfyUI core's own background removal model
(comfy/bg_removal_model.py, the model in models/background_removal - BiRefNet ships with core).
The window asks for a preview cutout through a server route; the node recomputes the mask at full size.
"""

from __future__ import annotations

import hashlib
import threading

import cv2
import numpy as np

_lock = threading.Lock()
_model = None
_masks: dict[str, np.ndarray] = {}       # sha1 of the pixels -> mask (a few recent ones)


def model():
    global _model
    with _lock:
        if _model is None:
            import folder_paths
            from comfy.bg_removal_model import load
            files = sorted(folder_paths.get_filename_list("background_removal"))
            if not files:
                raise RuntimeError("no background removal model in models/background_removal (core's BiRefNet: put "
                                   "birefnet.safetensors there)")
            pick = next((f for f in files if "birefnet" in f.lower()), files[0])
            _model = load(folder_paths.get_full_path_or_raise("background_removal", pick))
            if _model is None:
                raise RuntimeError(f"{pick} is not a valid background removal model")
        return _model


def mask_for(rgb: np.ndarray) -> np.ndarray:
    """Foreground mask (H, W) float 0..1 for an RGB float image; cached by pixel content."""
    import torch
    rgb = np.ascontiguousarray(rgb[..., :3], np.float32)
    key = hashlib.sha1(rgb.tobytes()).hexdigest()
    if key in _masks:
        return _masks[key]
    m = model().encode_image(torch.from_numpy(rgb)[None])
    m = m.detach().float().cpu().numpy()
    m = m.reshape(m.shape[-2], m.shape[-1]) if m.ndim > 2 else m
    if m.shape != rgb.shape[:2]:
        m = cv2.resize(m, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
    m = np.clip(m, 0, 1).astype(np.float32)
    if len(_masks) > 16:
        _masks.pop(next(iter(_masks)))
    _masks[key] = m
    return m
