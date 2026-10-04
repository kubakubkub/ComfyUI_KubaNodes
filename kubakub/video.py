"""
video.py

Pure logic of the region video sampler.
numpy / opencv only, no ComfyUI imports (tests/test_video.py).

A still (e.g. the frame-in-frame result) becomes a video in which only chosen
regions move: regions with the plan key `animate` are grouped into clips (same
video prompt, time range and motion), each clip is cropped to the LTX grid,
sampled as a masked video (the rest of the crop is the still, held fixed by the
noise mask), and pasted back into the untouched still frame by frame.

LTX geometry: 32 px per latent pixel, 8 frames per latent frame, frame counts 8k + 1.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

SPATIAL = 32
TEMPORAL = 8


def ltx_frames(n: int) -> int:
    """The nearest valid LTX frame count >= n (8k + 1)."""
    n = max(1, int(n))
    return ((n - 1 + TEMPORAL - 1) // TEMPORAL) * TEMPORAL + 1


def latent_frames(frames: int) -> int:
    return (ltx_frames(frames) - 1) // TEMPORAL + 1


def latent_frame_times(frames: int, fps: float):
    """Time (s) each latent frame stands for: frame 0 is the first image, latent frame i >= 1
    covers pixel frames 8(i-1)+1 .. 8i (its centre is taken)."""
    t = [0.0]
    for i in range(1, latent_frames(frames)):
        t.append((TEMPORAL * (i - 1) + 1 + TEMPORAL * i) / 2.0 / fps)
    return np.asarray(t)


def clip_groups(entries, only=None):
    """
    Animated regions grouped into clips: one clip per (video prompt, t_start, t_end, motion).
    entries: resolved plan entries (plan.py). Returns [{"ids", "prompt", "t_start", "t_end", "motion"}].
    """
    import fnmatch
    groups = {}
    pats = [p.strip().lower() for p in (only or "").replace("\n", ",").split(",") if p.strip()]
    for e in entries:
        if not e.get("animate"):
            continue
        if pats and not any(fnmatch.fnmatchcase(str(e.get("name", "")).lower(), p) for p in pats):
            continue
        prompt = (e.get("video_prompt") or e.get("prompt") or "").strip()
        key = (prompt, float(e.get("t_start", 0.0)), float(e.get("t_end", -1.0)), float(e.get("motion", 1.0)))
        groups.setdefault(key, []).append(int(e["region_id"]))
    return [{"ids": ids, "prompt": k[0], "t_start": k[1], "t_end": k[2], "motion": k[3]}
            for k, ids in groups.items()]


def video_range(t_start: float, t_end: float, fps: float, n: int):
    """Frames i0..i1 of an n-frame background video a clip covers (t_end < 0 = to the end); None when empty."""
    i0 = int(round(max(0.0, float(t_start)) * fps))
    i1 = n - 1 if t_end < 0 else min(n - 1, int(round(float(t_end) * fps)))
    return None if i1 <= i0 else (i0, i1)


def groups_with_work(groups, label_ids, n_bg=None, fps=25.0):
    """The clips that will be rendered: some of their regions are in the label map (label_ids: the ids present)
    and, into a background video of n_bg frames, their time range is inside it."""
    present = set(int(i) for i in label_ids)
    return [g for g in groups if present.intersection(g["ids"])
            and (n_bg is None or video_range(g["t_start"], g["t_end"], fps, n_bg) is not None)]


def _r32(v: float) -> int:
    return max(SPATIAL, int(round(v / SPATIAL)) * SPATIAL)


def crop_plan(mask: np.ndarray, target_px: float = 450_000, context_px: int = 64):
    """
    Crop box around a mask for the first LTX stage: (x0, y0, x1, y1) in image pixels and the
    stage-1 size (w1, h1), multiples of 32, with the box's aspect equal to w1 / h1 (uniform
    scale; the box grows around the mask's centre and is shifted, never stretched, at borders).
    """
    H, W = mask.shape
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        raise ValueError("empty region mask")
    bx0, by0 = max(0, xs.min() - context_px), max(0, ys.min() - context_px)
    bx1, by1 = min(W, xs.max() + 1 + context_px), min(H, ys.max() + 1 + context_px)
    bw, bh = bx1 - bx0, by1 - by0
    a = bw / bh
    w1, h1 = _r32(math.sqrt(target_px * a)), _r32(math.sqrt(target_px / a))
    # grow the box to the exact aspect w1 / h1
    if bw / bh < w1 / h1:
        bw = bh * w1 / h1
    else:
        bh = bw * h1 / w1
    if bw > W:
        bw, bh = W, W * h1 / w1
    if bh > H:
        bh, bw = H, H * w1 / h1
    cx, cy = (bx0 + bx1) / 2, (by0 + by1) / 2
    x0 = int(round(min(max(cx - bw / 2, 0), W - bw)))
    y0 = int(round(min(max(cy - bh / 2, 0), H - bh)))
    return (x0, y0, x0 + int(round(bw)), y0 + int(round(bh))), (w1, h1)


def latent_mask(region_crop: np.ndarray, lat_w: int, lat_h: int, frames: int, fps: float,
                t_start: float = 0.0, t_end: float = -1.0, motion: float = 1.0, dilate: int = 1,
                loop: bool = False):
    """
    Noise mask [1, 1, T_lat, lat_h, lat_w] (float32): motion inside the region during
    t_start..t_end (t_end < 0 = to the end), 0 elsewhere; latent frame 0 is always 0 (the
    first frame stays the still). loop: the last latent frame is 0 too, so the clip comes
    back to the still (its last 8 frames) and can repeat.
    """
    m = cv2.resize(region_crop.astype(np.float32), (lat_w, lat_h), interpolation=cv2.INTER_AREA) > 0.02
    if dilate > 0:
        m = cv2.dilate(m.astype(np.uint8), np.ones((2 * dilate + 1, 2 * dilate + 1), np.uint8)) > 0
    times = latent_frame_times(frames, fps)
    end = float("inf") if t_end < 0 else t_end
    active = (times >= t_start) & (times <= end)
    active[0] = False
    if loop and len(times) > 2:
        active[-1] = False
    out = np.zeros((1, 1, len(times), lat_h, lat_w), np.float32)
    out[0, 0, active] = m.astype(np.float32) * float(np.clip(motion, 0, 1))
    return out


def audio_range(first_frame: int, frames: int, fps: float, sample_rate: int):
    """The samples of a sound a clip covers: (first sample, sample count) for `frames` frames from first_frame."""
    return int(round(first_frame / fps * sample_rate)), int(round(frames / fps * sample_rate))


def close_loop(frames: np.ndarray, tail: int = TEMPORAL) -> np.ndarray:
    """A looped clip ends exactly on its first frame: the last `tail` frames fade from the frame before them onto
    frame 0. The held last latent frame is the still, but the VAE decodes its 8 pixel frames with a drift (measured:
    back at the still at frame F - 9, then away again), so they are replaced. In place; float frames."""
    if len(frames) <= 2 * tail:
        return frames
    w = np.linspace(0, 1, tail + 1, dtype=np.float32)[1:, None, None, None]
    a, b = frames[-tail - 1].astype(np.float32), frames[0].astype(np.float32)
    frames[-tail:] = (a[None] * (1 - w) + b[None] * w).astype(frames.dtype)
    return frames


def feather_mask(mask: np.ndarray, feather_px: float) -> np.ndarray:
    """Soft paste mask: 1 inside, falling to 0 over feather_px just outside (the region itself is fully pasted)."""
    if feather_px <= 0:
        return mask.astype(np.float32)
    # distance of every outside pixel to the region (0 inside)
    dist = cv2.distanceTransform((~mask.astype(bool)).astype(np.uint8), cv2.DIST_L2, 5)
    return np.clip(1.0 - dist / float(feather_px), 0, 1).astype(np.float32)


def paste_frame(frame: np.ndarray, clip_frame: np.ndarray, box, alpha: np.ndarray):
    """Paste one clip frame (any size) into frame at box, weighted by alpha (full-size float mask)."""
    x0, y0, x1, y1 = box
    c = cv2.resize(np.asarray(clip_frame, np.float32), (x1 - x0, y1 - y0),
                   interpolation=cv2.INTER_AREA if clip_frame.shape[1] > x1 - x0 else cv2.INTER_CUBIC)
    a = alpha[y0:y1, x0:x1, None]
    frame[y0:y1, x0:x1] = frame[y0:y1, x0:x1] * (1 - a) + c * a
    return frame
