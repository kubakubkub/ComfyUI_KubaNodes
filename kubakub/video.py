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


def only_filter(only):
    """
    The 'only' text as a predicate on plan entries, or None (empty = all). Comma or line separated; each part is
    a name wildcard ('W_F1_*') or a plan selector (plan.py: 'group:Windows', 'tag:front', 'region:3-7',
    'tag:front group:Windows' = and, '!x' = not). A part that is no valid selector stays a name wildcard.
    """
    import fnmatch
    from . import plan as rp
    pats = [p.strip().lower() for p in (only or "").replace("\n", ",").split(",") if p.strip()]
    if not pats:
        return None
    sections = []
    for p in pats:
        try:
            sections.append(rp.selector(p))
        except rp.PlanError:
            sections.append(None)

    def hit(e):
        name = str(e.get("name", "")).lower()
        return any(fnmatch.fnmatchcase(name, p) or (s is not None and s.match(e) is not None)
                   for p, s in zip(pats, sections))
    return hit


def clip_groups(entries, only=None):
    """
    Animated regions grouped into clips: one clip per (video prompt, t_start, t_end, motion, material_repaint).
    entries: resolved plan entries (plan.py). only: see only_filter.
    Returns [{"ids", "prompt", "t_start", "t_end", "motion", "repaint"}]; repaint = the noise level the clip is
    repainted from when a material clip guides it.
    """
    groups = {}
    wanted = only_filter(only)
    for e in entries:
        if not e.get("animate"):
            continue
        if wanted is not None and not wanted(e):
            continue
        prompt = (e.get("video_prompt") or e.get("prompt") or "").strip()
        key = (prompt, float(e.get("t_start", 0.0)), float(e.get("t_end", -1.0)), float(e.get("motion", 1.0)),
               float(e.get("material_repaint", 0.8)))
        groups.setdefault(key, []).append(int(e["region_id"]))
    return [{"ids": ids, "prompt": k[0], "t_start": k[1], "t_end": k[2], "motion": k[3], "repaint": k[4]}
            for k, ids in groups.items()]


def video_range(t_start: float, t_end: float, fps: float, n: int):
    """Frames i0..i1 of an n-frame background video a clip covers (t_end < 0 = to the end); None when empty."""
    i0 = int(round(max(0.0, float(t_start)) * fps))
    i1 = n - 1 if t_end < 0 else min(n - 1, int(round(float(t_end) * fps)))
    return None if i1 <= i0 else (i0, i1)


def frames_in_range(frames: int, fps: float, t_start: float = 0.0, t_end: float = -1.0):
    """Which frames of a clip lie in t_start..t_end (t_end < 0 = to the end): a bool array, one per frame."""
    t = np.arange(int(frames), dtype=np.float64) / float(fps)
    return (t >= float(t_start) - 1e-9) & (t <= (float("inf") if t_end < 0 else float(t_end)) + 1e-9)


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
                loop: bool = False, hold_first: bool = True):
    """
    Noise mask [1, 1, T_lat, lat_h, lat_w] (float32): motion inside the region during
    t_start..t_end (t_end < 0 = to the end), 0 elsewhere; latent frame 0 is always 0 (the
    first frame stays the still). loop: the last latent frame is 0 too, so the clip comes
    back to the still (its last 8 frames) and can repeat. hold_first off: the first and the last
    frame are repainted like the others (a clip that follows a moving guide: the guide is its start).
    """
    m = cv2.resize(region_crop.astype(np.float32), (lat_w, lat_h), interpolation=cv2.INTER_AREA) > 0.02
    if dilate > 0:
        m = cv2.dilate(m.astype(np.uint8), np.ones((2 * dilate + 1, 2 * dilate + 1), np.uint8)) > 0
    times = latent_frame_times(frames, fps)
    end = float("inf") if t_end < 0 else t_end
    active = (times >= t_start) & (times <= end)
    if hold_first:
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


# ---- motion guide: a clip LTX repaints while it keeps the motion ---------------------------------------------------

GUIDE_SIGMAS = (1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0)     # LTX 2.5 distilled, stage 1
GUIDE_SIGMAS_2 = (0.85, 0.725, 0.4219, 0.0)                                                # after the x2 upscaler


def guide_sigmas(repaint: float, base=GUIDE_SIGMAS) -> list:
    """The noise levels to sample through when the start is a guide clip and not pure noise: `repaint` first (1 =
    pure noise, the whole schedule), then the schedule's own steps below it. A step closer than 0.02 under the start
    is left out (it would be a step that does nothing)."""
    r = float(min(1.0, max(0.05, repaint)))
    if r >= base[0]:
        return list(base)
    return [r] + [s for s in base if s < r - 0.02]


def guide_size(w: int, h: int, megapixels: float) -> tuple:
    """The LTX size (multiples of 32) for a guide of w x h at about `megapixels`: of the grid sizes around it, the
    one closest to the guide's own shape (the guide is scaled to it, so the less it is stretched the better)."""
    a = float(w) / float(h)
    px = max(0.01, float(megapixels)) * 1e6
    fw, fh = math.sqrt(px * a) / SPATIAL, math.sqrt(px / a) / SPATIAL
    sizes = [(max(1, x), max(1, y)) for x in (math.floor(fw), math.ceil(fw)) for y in (math.floor(fh), math.ceil(fh))]
    x, y = min(sizes, key=lambda s: (abs(math.log(s[0] / s[1] / a)), abs(s[0] * s[1] * SPATIAL * SPATIAL - px)))
    return x * SPATIAL, y * SPATIAL


def guide_frames(frames: np.ndarray, w: int, h: int) -> np.ndarray:
    """A guide clip (F, H, W, 3) at the LTX size and a valid LTX length: the last frame is repeated up to 8k + 1."""
    out = [cv2.resize(f, (w, h), interpolation=cv2.INTER_AREA if f.shape[1] >= w else cv2.INTER_CUBIC)
           if f.shape[:2] != (h, w) else f for f in frames]
    out += [out[-1]] * (ltx_frames(len(out)) - len(out))
    return np.ascontiguousarray(np.stack(out), dtype=np.float32)


def _strength(mask, values, f, n):
    """How much frame f may be repainted: the mask's frame (a shorter batch holds its last one) times the curve."""
    v = 1.0 if values is None else float(values[min(f, n - 1)])
    return v if mask is None else mask[min(f, n - 1, len(mask) - 1)] * v


REPAINT_FLOOR = 0.70          # LTX hardly repaints below this noise level (measured: 0.725 = a trace, 0.91 = everything)


def repaint_level(amount: np.ndarray, start: float) -> np.ndarray:
    """Amounts (0-1) as the noise mask LTX reads, for a stage that starts at noise level `start`. LTX takes the mask
    as the noise level of every place and only changes things near full noise (half the level changes nothing), so
    an amount above 0 is laid onto the stretch of levels where something happens, REPAINT_FLOOR .. start, with a
    square root (half the amount = clearly changed). Nothing stays nothing, all stays all."""
    floor = min(1.0, REPAINT_FLOOR / max(float(start), 1e-6))
    return np.where(amount > 0.002, floor + (1.0 - floor) * np.sqrt(amount), 0.0).astype(np.float32)


def repaint_latent(mask, values, n: int, lat_t: int, lat_w: int, lat_h: int):
    """How much of every place at every moment is open (0-1), at the latent's size: [1, 1, lat_t, lat_h, lat_w]
    float32, or None when neither a mask nor a curve is given.
    mask    None or [M, H, W] 0-1: white = repaint, black = the guide stays (M frames; fewer than n: the last holds)
    values  None or one number per frame, 0-1: how much of the repaint that frame gets (a curve over time)
    n       the frames of the guide; a latent frame stands for 8 of them (the first for frame 0) and takes their mean
    """
    if mask is None and values is None:
        return None
    small = {}

    def at(f):
        k = min(f, n - 1, len(mask) - 1)
        if k not in small:
            small[k] = cv2.resize(np.ascontiguousarray(mask[k], dtype=np.float32), (lat_w, lat_h), interpolation=cv2.INTER_AREA)
        return small[k]

    out = np.zeros((lat_t, lat_h, lat_w), np.float32)
    for i in range(lat_t):
        fs = [0] if i == 0 else range(TEMPORAL * (i - 1) + 1, TEMPORAL * i + 1)
        for f in fs:
            v = 1.0 if values is None else float(values[min(f, n - 1)])
            out[i] += v if mask is None else at(f) * v
        out[i] /= len(fs)
    return np.clip(out, 0.0, 1.0)[None, None]


def repaint_share(mask, values, n: int) -> float:
    """How much of the whole clip is open, 0-1 (the mean of mask x curve over the frames)."""
    means = {}
    total = 0.0
    for f in range(n):
        v = 1.0 if values is None else float(values[min(f, n - 1)])
        if mask is not None:
            k = min(f, len(mask) - 1)
            if k not in means:
                means[k] = float(np.clip(mask[k], 0.0, 1.0).mean())
            v *= means[k]
        total += v
    return total / max(1, n)


def repaint_alpha(mask, values, f: int, n: int, w: int, h: int, step: bool = True):
    """Where frame f of the result is LTX's picture (1) and where the guide's own pixels (0). [h, w] float32, or a
    number when there is no mask. step: everything that may be repainted at all, with a short soft edge (the guide's
    pixels back where LTX was held); else the amount itself (the repainted clip shown through the mask)."""
    s = _strength(mask, values, f, n)
    k = 0.1 if step else 1.0
    if np.isscalar(s):
        return float(min(1.0, max(0.0, s / k)))
    return np.clip(cv2.resize(np.ascontiguousarray(s, dtype=np.float32), (w, h), interpolation=cv2.INTER_LINEAR) / k, 0.0, 1.0)


def sample_guide(w: int = 768, h: int = 448, frames: int = 49) -> np.ndarray:
    """The built-in guide: the sample facade in the dark, a band of light sweeping over it from left to right."""
    from . import sample_facade as sf
    pic = np.asarray(sf.render(w, h, style="night")["image"], np.float32)[..., :3]
    x = np.linspace(0, 1, w, dtype=np.float32)[None, :, None]
    out = np.empty((frames, h, w, 3), np.float32)
    for i in range(frames):
        k = np.clip(1 - np.abs(x - (0.08 + 0.84 * i / max(1, frames - 1))) / 0.09, 0, 1)
        out[i] = np.clip(pic * (0.22 + 1.3 * k * k * (3 - 2 * k)), 0, 1)
    return out
