"""
soundswap.py

kubakub sound mask swap: a set of layers trade their masks in time with a sound. The pictures stay where they are,
the masks move on: one step per beat, bar, fixed time, or per hit in the low / mid / high band, in a loop, there
and back, or shuffled. The same engine as the director's "swap masks" behaviour (director/motion.py swap_*), so a
set-up tried here plays the same in the window. Pure numpy + OpenCV (tests/test_soundswap.py).
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from . import sample_facade as sf
from . import sound as so
from .director import motion as mo

COLOURS = ((241, 138, 88), (161, 135, 183), (122, 196, 186), (238, 208, 112), (236, 236, 236), (96, 142, 216))
SAMPLE_BPM = 120.0


@dataclass
class Settings:
    order: str = "loop"            # loop | pingpong | random
    step_on: str = "beats"         # beats | bars | every | low | mid | high | signal | list
    nth: int = 1                   # every nth beat / bar / hit
    every: float = 0.5             # s, step_on = every
    threshold: float = 0.3         # 0-1: how strong a hit has to be (low / mid / high)
    gap: float = 0.1               # s: the shortest time between two steps (low / mid / high)
    fade: float = 0.0              # s fade into the new mask, 0 = the layer just appears
    transition: str = "cross"      # with a fade: cross (crossfade) | dip (out of the old mask, then into the new one)
    feather: float = 0.0           # px (of the background): soft mask edges
    bpm: float = 0.0               # 0 = found in the sound
    fps: float = 25.0
    start: float = 0.0             # s into the sound
    seconds: float = 0.0           # 0 = to the end of the sound
    scale: float = 0.5             # size of the frames relative to the background
    seed: int = 1


def sample_layers(W, H):
    """(background HxWx3 0..1, masks [4, H, W] 0..1): the sample facade and its windows, one layer per floor."""
    r = sf.render(W, H)
    masks = []
    for fl in (3, 2, 1, 0):
        m = np.zeros((H, W), np.float32)
        for name, (_, a) in r["masks"].items():
            if name.startswith(f"W_F{fl}_") or (fl == 0 and name == "M_Door"):
                m = np.maximum(m, a.astype(np.float32) / 255.0)
        masks.append(m)
    return r["image"], np.stack(masks)


def _fit(a, W, H):
    if a.shape[1] == W and a.shape[0] == H:
        return np.ascontiguousarray(a, np.float32)
    shrink = a.shape[1] > W or a.shape[0] > H
    return cv2.resize(np.ascontiguousarray(a, np.float32), (W, H), interpolation=cv2.INTER_AREA if shrink else cv2.INTER_LINEAR)


def timing(mono, sr, st: Settings, sample=False, times=None):
    """-> (behaviour, ctx, frame times, lines): what the engine needs and what the report says about the sound.
    times: the step times (s) for step_on = signal / list."""
    total, fps = len(mono) / float(sr), min(max(float(st.fps), 1.0), 120.0)
    frames, start, seconds = frame_count(len(mono), sr, st)
    own = st.step_on in ("signal", "list")
    b = {"type": "swap", "order": st.order, "trigger": "markers" if own else st.step_on, "nth": int(st.nth), "every": float(st.every),
         "threshold": float(st.threshold), "gap": float(st.gap), "fade": float(st.fade), "seed": int(st.seed),
         "transition": st.transition,
         "t_start": start, "t_end": start + seconds}
    ctx = {"beats": [], "markers": [], "offset": 0.0, "hits": {}}
    lines = [f"sound {total:.2f} s" + (" (the built-in beat)" if sample else "") + f", {start:.2f} to {start + seconds:.2f} s"]
    if own:
        ctx["markers"] = sorted(float(t) for t in times or [])
        lines.append(f"{len(ctx['markers'])} " + ("rises found in the signal" if st.step_on == "signal" else "times in the list"))
    elif st.step_on in ("beats", "bars"):
        if st.bpm > 0:
            beat, how = {"bpm": float(st.bpm), "phase": so.beat_phase(mono, sr, st.bpm)}, "set"
        elif sample:
            beat, how = {"bpm": SAMPLE_BPM, "phase": 0.0}, "the built-in beat"
        else:
            beat, how = so.analyze_beats(mono, sr), "found"
        if not beat:
            lines.append("no tempo found (the sound is shorter than 2 s): set bpm, or step on low / high / every")
        else:
            ctx["beats"] = mo.beat_times(beat, 0.0, total)
            lines.append(f"{beat['bpm']} bpm ({how}), first beat at {beat['phase']} s")
    elif st.step_on in mo.BANDS:
        ctx["hits"] = so.analyze(mono, sr)["hits"]
        hits = ctx["hits"].get(st.step_on) or []
        lines.append(f"{len(hits)} {st.step_on} hits, {sum(1 for h in hits if h[1] >= st.threshold)} of them at or above {st.threshold}")
    steps = mo.swap_times(b, ctx, start + seconds)
    lines.append(f"{len(steps)} steps ({st.order}, on {st.step_on}" + (f", every {st.nth}." if st.nth > 1 else "") + ")"
                 + ("" if steps else ": nothing moves. Lower the threshold or pick another 'step on'."))
    return b, ctx, [start + f / fps for f in range(frames)], lines


def run(background, masks, pictures, mono, sr, st: Settings, mask_of=1, sample=False, progress=None, check_interrupt=None,
        times=None):
    """
    background HxWx3 0..1 (gives the size), masks [n, h, w] 0..1 (one per layer), pictures [m, h, w, 3] or None
    (one per layer, repeated when there are fewer; None = a flat colour per layer), mono float [S].
    Returns (frames [F, H', W', 3], what is seen of layer `mask_of` per frame [F, H', W'], lines).
    """
    H0, W0 = background.shape[:2]
    s = min(max(float(st.scale), 0.05), 1.0)
    W, H = max(8, int(round(W0 * s))), max(8, int(round(H0 * s)))
    n = len(masks)
    if n < 2:
        raise ValueError("kubakub sound mask swap needs at least two masks (one per layer): a mask batch.")
    bg = _fit(background[..., :3], W, H)
    M = np.stack([np.clip(_fit(m, W, H), 0.0, 1.0) for m in masks])
    if st.feather * s >= 0.5:                       # soft edges: the feather is given in background pixels
        M = np.stack([np.clip(cv2.GaussianBlur(m, (0, 0), st.feather * s / 2.0), 0.0, 1.0) for m in M])
    if pictures is not None and len(pictures):
        P = [_fit(pictures[i % len(pictures)][..., :3], W, H) for i in range(n)]
    else:
        P = [np.broadcast_to(np.array(COLOURS[i % len(COLOURS)], np.float32) / 255.0, (H, W, 3)) for i in range(n)]
    b, ctx, frame_t, lines = timing(mono, sr, st, sample, times)
    dur = b["t_end"]
    F = len(frame_t)
    which = min(max(int(mask_of), 1), n) - 1
    frames = np.empty((F, H, W, 3), np.float32)
    seq = np.empty((F, H, W), np.float32)
    held = {}                                       # a mask arrangement that holds is composed once
    for f, t in enumerate(frame_t):
        if check_interrupt:
            check_interrupt()
        mix = mo.swap_mix(b, t, dur, n, ctx)
        key = tuple(tuple((m, round(w, 6)) for m, w in one) for one in mix)
        got = held.get(key)
        if got is None:
            now = [M[one[0][0]] if len(one) == 1 else np.minimum(1.0, sum(w * M[m] for m, w in one)) for one in mix]
            img, seen = bg.copy(), None
            for i in range(n - 1, -1, -1):          # layer 1 on top
                a = now[i][..., None]
                img = img * (1.0 - a) + P[i] * a
            free = np.ones((H, W), np.float32)      # what the layers above leave of each one: the mask that is seen
            for i in range(which + 1):
                seen = now[i] * free
                free = free * (1.0 - now[i])
            got = (img, seen)
            if len(held) > 64:
                held.clear()
            if all(len(one) == 1 for one in mix):
                held[key] = got
        frames[f], seq[f] = got
        if progress:
            progress(f + 1, F)
    lines.append(f"{F} frames at {W}x{H}, {n} layers" + ("" if pictures is not None and len(pictures) else " (no pictures: a colour per layer)"))
    return frames, seq, lines


def ram_bytes(frames, W, H):
    return frames * W * H * 4 * 4                   # RGB + the mask, float32


def frame_count(mono_len, sr, st: Settings):
    total = mono_len / float(sr)
    start = min(max(0.0, st.start), max(0.0, total - 1e-3))
    seconds = total - start if st.seconds <= 0 else min(st.seconds, total - start)
    return max(1, int(round(seconds * min(max(float(st.fps), 1.0), 120.0)))), start, seconds
