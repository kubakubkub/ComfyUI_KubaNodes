"""
maskfields.py

Masks that are more than on / off. A field is a grey ramp over regions (0 = where a move starts, 1 = where it ends):
the distance from each region's edge, a direction, the distance from a point, one step per region in an order, noise.
A mask moves along a field (reveal, hide, a band, rings) or as a whole (move, rotate, scale, opacity), following a
curve over time: a ramp, saw, triangle, sine, square, random steps, noise, the level of a sound or its beats and hits. numpy + OpenCV, no ComfyUI imports (tests/test_maskfields.py).
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from . import sample_facade as sf
from . import sound as so
from .director import motion as mo

FIELDS, PER, ORDERS, EASES = mo.FIELD_KINDS, mo.FIELD_PER, mo.STAGGER_ORDERS, mo.EASES    # one list, shared with the director
FIELD_EFFECTS = mo.FIELD_EFFECTS                    # these run along a field, the others move the mask itself
EFFECTS = FIELD_EFFECTS + ("move x", "move y", "rotate", "scale", "opacity")
CURVES = mo.FIELD_SHAPES + ("sound level", "beats", "bars", "low hits", "mid hits", "high hits")
TRIGGERS = {"beats": "beats", "bars": "bars", "low hits": "low", "mid hits": "mid", "high hits": "high"}
LISTEN = ("everything", "low", "mid", "high")
PIVOTS = ("mask centre", "canvas centre")
SAMPLE_BPM = 120.0
NOISE_BINS = 2048


# --------------------------------------------------------------------------
# regions as one label map: 0..n-1 inside, -1 outside
# --------------------------------------------------------------------------

def compact(labels, ids):
    """labels int [H, W] (-1 = no region) and the ids to keep -> (int32 map with 0..n-1 in the order of ids, n)."""
    labels = np.asarray(labels)
    lut = np.full(max(int(labels.max()), max(ids, default=0)) + 2, -1, np.int32)
    for k, i in enumerate(ids):
        lut[int(i) + 1] = k
    return lut[np.maximum(labels, -1).astype(np.int64) + 1], len(ids)


def from_masks(masks, split=True, min_area=16):
    """masks float [N, H, W] -> (label map, n): one region per mask, or per separate shape of each mask (split).
    Where masks overlap the later one is on top. Shapes under min_area pixels are left out."""
    masks = np.asarray(masks)
    if masks.ndim == 2:
        masks = masks[None]
    lab = np.full(masks.shape[1:], -1, np.int32)
    n = 0
    for m in masks:
        on = np.ascontiguousarray(m > 0.5, np.uint8)
        if not split:
            if int(on.sum()) >= min_area:
                lab[on > 0] = n
                n += 1
            continue
        count, cc, stats, _ = cv2.connectedComponentsWithStats(on, connectivity=8)
        for c in range(1, count):
            if stats[c, cv2.CC_STAT_AREA] >= min_area:
                lab[cc == c] = n
                n += 1
    return compact(lab, sorted(set(np.unique(lab).tolist()) - {-1}))      # a mask fully covered by a later one is gone


def sample_regions(W=1920, H=1080):
    """(label map, n, names): the windows of the sample facade, one region each."""
    r = sf.render(W, H)
    names = [k for k in r["masks"] if k.startswith("W_F")]
    lab = np.full((H, W), -1, np.int32)
    for k, name in enumerate(names):
        lab[r["masks"][name][1] > 0] = k
    return lab, len(names), names


def boxes(lab, n):
    """[(x, y, w, h) or None] per region."""
    from scipy import ndimage
    return [None if s is None else (s[1].start, s[0].start, s[1].stop - s[1].start, s[0].stop - s[0].start)
            for s in ndimage.find_objects(lab + 1, max_label=n)]


# --------------------------------------------------------------------------
# fields
# --------------------------------------------------------------------------

def _unit(v, on):
    """v scaled to 0..1 over the pixels in on."""
    if not on.any():
        return np.zeros_like(v, np.float32)
    lo, hi = float(v[on].min()), float(v[on].max())
    return ((v - lo) / (hi - lo)).astype(np.float32) if hi - lo > 1e-9 else np.zeros_like(v, np.float32)


def _hash01(i, seed):
    """motion.hash01 for an array of whole numbers."""
    m = np.uint64(0xFFFFFFFF)
    h = ((i.astype(np.int64).astype(np.uint64) & m) * np.uint64(374761393) + np.uint64((int(seed) * 668265263) & 0xFFFFFFFF)) & m
    h = ((h ^ (h >> np.uint64(13))) * np.uint64(1274126177)) & m
    return ((h ^ (h >> np.uint64(16))) & m) / 4294967296.0


def field(lab, n, kind="edge distance", per="each region", angle=0.0, centre=(0.5, 0.5), order="left", seed=1,
          noise_px=64.0, invert=False, px_scale=1.0, ids=None):
    """-> (field float32 [H, W] 0..1, 0 outside the regions; inside bool [H, W]).

    edge distance  0 on the edge of a region, 1 at its deepest point
    direction      0 to 1 along angle (0 = left to right, 90 = top to bottom)
    radial         0 at the centre (of each region, or the point centre = (x, y) in 0..1 of the canvas), 1 farthest
    region order   one flat value per region, in the order left / right / top / bottom / centre / outside / size / random
    noise          soft random values (cells of noise_px), every value equally often; px_scale = picture pixels per
                   pixel of lab (a small preview map draws the same noise)
    per            each region = every region runs 0..1 on its own; all together = one ramp over all of them
    ids            the regions' own ids, in the order of lab's numbers (ties and 'random' in region order use them)
    """
    H, W = lab.shape
    inside = lab >= 0
    f = np.zeros((H, W), np.float32)
    bx = boxes(lab, n)
    live = [(k, b) for k, b in enumerate(bx) if b]
    together = per == "all together"
    ca, sa = round(math.cos(math.radians(angle)), 9), round(math.sin(math.radians(angle)), 9)
    if not live:
        return f, inside
    if kind == "region order":
        val = np.zeros(n + 1, np.float32)
        own = list(ids) if ids is not None else list(range(n))
        back = {int(own[k]): k for k, _b in live}
        for rank, rid in enumerate(mo.stagger_order([(int(own[k]), list(b)) for k, b in live], order, W, H, seed)):
            val[back[rid] + 1] = (rank + 0.5) / len(live)
        f = val[lab + 1]
    elif kind == "noise":
        cell = max(2.0, float(noise_px))
        u, v = (np.arange(W, dtype=np.float64) * px_scale / cell)[None, :], (np.arange(H, dtype=np.float64) * px_scale / cell)[:, None]
        ix, iy = np.floor(u), np.floor(v)
        fx, fy = u - ix, v - iy
        fx, fy = fx * fx * (3 - 2 * fx), fy * fy * (3 - 2 * fy)
        h = lambda dx, dy: _hash01((ix + dx) + 7919 * (iy + dy), seed)      # noqa: E731
        big = (h(0, 0) * (1 - fx) + h(1, 0) * fx) * (1 - fy) + (h(0, 1) * (1 - fx) + h(1, 1) * fx) * fy
        hist = np.bincount(np.minimum(NOISE_BINS - 1, (big[inside] * NOISE_BINS).astype(np.int64)), minlength=NOISE_BINS)
        cdf = np.cumsum(hist) / max(1, int(hist.sum()))          # every grey equally often
        f = cdf[np.minimum(NOISE_BINS - 1, (big * NOISE_BINS).astype(np.int64))].astype(np.float32)
    elif together and kind in ("direction", "radial"):
        yy, xx = np.arange(H, dtype=np.float32)[:, None], np.arange(W, dtype=np.float32)[None, :]
        v = xx * ca + yy * sa if kind == "direction" else np.hypot(xx - centre[0] * (W - 1), yy - centre[1] * (H - 1))
        if kind == "radial":
            top = float(v[inside].max())
            f = (v / top).astype(np.float32) if top > 1e-9 else f
        else:
            f = _unit(v, inside)
    else:
        top = 0.0
        for k, (x, y, w, h) in live:
            on = lab[y:y + h, x:x + w] == k
            if kind == "edge distance":
                d = cv2.distanceTransform(np.pad(on, 1).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)[1:-1, 1:-1] - 1.0
                top = max(top, float(d.max()))
                v = d if together else (d / d.max() if d.max() > 1e-9 else d * 0)
            else:
                yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
                if kind == "direction":
                    v = _unit(xx * ca + yy * sa, on)
                else:
                    r = np.hypot(xx - xx[on].mean(), yy - yy[on].mean())
                    v = r / r[on].max() if r[on].max() > 1e-9 else r * 0
            f[y:y + h, x:x + w][on] = v[on]
        if kind == "edge distance" and together and top > 1e-9:
            f /= top
    f = np.clip(f, 0.0, 1.0)
    if invert:
        f = 1.0 - f
    f[~inside] = 0.0
    return f, inside


# --------------------------------------------------------------------------
# curves: a value over time
# --------------------------------------------------------------------------

ease = mo.ease                                      # the time curves, shared with the director's field behaviour
shape = mo.curve_shape
progress_triggers = mo.progress_triggers


def sound_context(mono, sr, curve, bpm=0.0, sample=False, listen="everything"):
    """-> (what a sound curve needs: beats, the hits of its band, or one loudness curve; report lines)."""
    total = len(mono) / float(sr)
    ctx = {"beats": [], "offset": 0.0, "hits": {}}
    lines = [f"sound {total:.2f} s" + (" (the built-in beat)" if sample else "")]
    if curve in ("beats", "bars"):
        if bpm > 0:
            beat, how = {"bpm": float(bpm), "phase": so.beat_phase(mono, sr, bpm)}, "set"
        elif sample:
            beat, how = {"bpm": SAMPLE_BPM, "phase": 0.0}, "the built-in beat"
        else:
            beat, how = so.analyze_beats(mono, sr), "found"
        if beat:
            ctx["beats"] = mo.beat_times(beat, 0.0, total)
            lines.append(f"{beat['bpm']} bpm ({how}), first beat at {beat['phase']} s")
        else:
            lines.append("no tempo found (the sound is shorter than 2 s): set bpm, or use low hits / high hits")
    elif curve in TRIGGERS:
        band = TRIGGERS[curve]
        ctx["hits"] = {band: so._hits(so.band_energy(mono, sr)[band])}
    elif curve == "sound level":
        if listen in so.BANDS:
            ctx["bands"] = {listen: so._curve(so.band_energy(mono, sr)[listen])}
        else:
            ctx["level"] = mo.level_curve(mono, sr)
    return ctx, lines


def triggers(curve, ctx, t0, t1, nth=1, threshold=0.3, gap=0.1):
    """The times (s) a ramp starts at between t0 and t1; one on t0 counts."""
    if curve in ("beats", "bars"):
        ts = (ctx.get("beats") or [])[::4 if curve == "bars" else 1]
    else:
        ts, last = [], None
        for h in (ctx.get("hits") or {}).get(TRIGGERS.get(curve, curve)) or []:
            if h[1] >= threshold and (last is None or h[0] - last >= gap - 1e-9):
                ts.append(h[0])
                last = h[0]
    return [x for x in ts if t0 - 1e-6 <= x <= t1 + 1e-6][::max(1, int(nth))]


def curve_values(times, curve="ramp", cycle=2.0, phase=0.0, easing="linear", duty=0.5, seed=1, lo=0.0, hi=1.0,
                 ctx=None, nth=1, threshold=0.3, gap=0.1, steps=1, listen="everything", release=0.15, gain=1.0):
    """-> ([value per time, between lo and hi], report lines). cycle = seconds per cycle; phase = cycles to start at."""
    lines = []
    if curve in TRIGGERS:
        trig = triggers(curve, ctx or {}, times[0], times[-1], nth, threshold, gap)
        lines.append(f"{len(trig)} triggers ({curve}" + (f", every {nth}." if nth > 1 else "") + ")"
                     + ("" if trig else ": nothing moves. Lower the threshold or pick another curve."))
        v = [progress_triggers(t, trig, cycle, easing, steps) for t in times]
    elif curve == "sound level":
        band = listen if listen in so.BANDS else "all"
        v = [min(1.0, max(0.0, gain * mo._level(ctx or {}, t, release, band))) for t in times]
    else:
        c = max(1e-6, float(cycle))
        v = [shape(curve, max(0.0, (t - times[0]) / c + phase), easing, duty, seed) for t in times]
    return [lo + x * (hi - lo) for x in v], lines


# --------------------------------------------------------------------------
# the mask at one value
# --------------------------------------------------------------------------

def along(f, p, effect="reveal", soft=0.05, width=0.2, rings=3):
    """The mask at progress p (0..1) along field f: float32 [H, W].

    reveal  on where the field is below p (nothing at 0, everything at 1); hide = the other way round
    band    a stripe of `width` (in field units) that travels from just before 0 to just past 1
    rings   `rings` stripes at even distances that move on by one distance from p = 0 to 1 (loops without a jump)
    soft    the width of the soft edge, in field units
    """
    s = max(1e-6, float(soft))
    if effect in ("reveal", "hide"):
        if p <= 0 or p >= 1:
            return np.full(f.shape, float((p >= 1) != (effect == "hide")), np.float32)
        m = np.clip((p * (1.0 + s) - f) / s, 0.0, 1.0)
        if effect == "hide":
            np.subtract(1.0, m, out=m)
    else:
        w = max(1e-6, float(width))
        if effect == "band":
            d = np.abs(f - (p * (1.0 + w + 2.0 * s) - w / 2.0 - s))
        else:
            n = max(1, int(rings))
            u = f * n - p
            u -= np.floor(u)
            d = np.minimum(u, 1.0 - u) / n
        m = np.clip((w / 2.0 - d) / s + 0.5, 0.0, 1.0)
    return (m * m * (3.0 - 2.0 * m)).astype(np.float32, copy=False)


def pruned(values, effect="reveal"):
    """[[progress, weight], ...] without the entries that cannot win the per-pixel maximum: one per progress (its
    highest weight); and for reveal (the mask only grows with the progress) / hide (it only shrinks) none that
    another entry beats in both. The mask is exactly the same; a rising ramp with a trail collapses to one entry."""
    best = {}
    for p, w in values:
        best[p] = max(w, best.get(p, 0.0))
    vals = sorted(best.items())
    if effect in ("reveal", "hide"):
        out, top = [], -1.0
        for p, w in (reversed(vals) if effect == "reveal" else vals):     # from the fullest mask down
            if w > top:
                out.append((p, w))
                top = w
        vals = out
    return vals


def mask_at(f, values, effect="reveal", soft=0.05, width=0.2, rings=3):
    """The mask for [[progress, weight], ...] (motion.field_values): now and its fading trail, the brightest wins."""
    out = None
    for p, w in pruned(values, effect):
        m = along(f, p, effect, soft, width, rings)
        if w != 1.0:
            m *= w
        out = m if out is None else np.maximum(out, m, out=out)
    return out


def centre_of(m):
    """The centre of weight of a mask (x, y), the canvas centre when it is empty."""
    H, W = m.shape
    tot = float(m.sum())
    if tot <= 1e-6:
        return (W - 1) / 2.0, (H - 1) / 2.0
    return float((m.sum(axis=0) * np.arange(W)).sum() / tot), float((m.sum(axis=1) * np.arange(H)).sum() / tot)


def moved(m, v, effect, pivot=None, wrap=False):
    """The mask as a whole at value v: move x / move y (v = canvas widths / heights), rotate (v = turns, clockwise),
    scale (v = factor), opacity (v = 0..1)."""
    H, W = m.shape
    if effect == "opacity":
        return (m * min(1.0, max(0.0, v))).astype(np.float32)
    if effect in ("move x", "move y"):
        dx, dy = (v * W, 0.0) if effect == "move x" else (0.0, v * H)
        M = np.float32([[1, 0, dx], [0, 1, dy]])
    else:
        cx, cy = pivot if pivot is not None else ((W - 1) / 2.0, (H - 1) / 2.0)
        if effect == "scale" and v <= 1e-4:
            return np.zeros_like(m, np.float32)
        M = cv2.getRotationMatrix2D((cx, cy), -360.0 * v if effect == "rotate" else 0.0, v if effect == "scale" else 1.0)
    border = cv2.BORDER_WRAP if wrap and effect.startswith("move") else cv2.BORDER_CONSTANT
    return cv2.warpAffine(np.ascontiguousarray(m, np.float32), M, (W, H), flags=cv2.INTER_LINEAR, borderMode=border, borderValue=0)


def fit(a, scale):
    """A mask or field [H, W] at another size."""
    H, W = a.shape
    w, h = max(8, int(round(W * scale))), max(8, int(round(H * scale)))
    if (w, h) == (W, H):
        return np.ascontiguousarray(a, np.float32)
    return cv2.resize(np.ascontiguousarray(a, np.float32), (w, h), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)


def animate(masks, f, values, effect="reveal", soft=0.05, width=0.2, rings=3, trail=0.0, fps=25.0,
            pivot="mask centre", wrap=False, progress=None, check_interrupt=None):
    """One mask per value: float32 [len(values), H, W].

    masks  float [N, H, W]: what is animated (frame i takes mask i, the last one repeats). For reveal / hide / band /
           rings it limits the result; None = everywhere the field is above 0.
    f      the field [H, W] for reveal / hide / band / rings.
    trail  seconds a mask keeps glowing after it has moved on (0 = none).
    """
    H, W = (f if masks is None else masks[0]).shape
    out = np.empty((len(values), H, W), np.float32)
    base0 = (f > 0).astype(np.float32) if masks is None else None
    pv = centre_of(masks.max(axis=0)) if masks is not None and pivot == "mask centre" else None   # of everywhere the batch ever is
    keep = math.exp(-1.0 / (trail * fps)) if trail > 0 else 0.0
    still = masks is None or len(masks) == 1            # one mask for every frame: a held value gives the same picture
    for i, v in enumerate(values):
        if check_interrupt:
            check_interrupt()
        if i and still and not keep and v == values[i - 1]:
            out[i] = out[i - 1]
        else:
            base = base0 if masks is None else masks[min(i, len(masks) - 1)]
            if effect in FIELD_EFFECTS:
                np.multiply(along(f, v, effect, soft, width, rings), base, out=out[i])
            else:
                out[i] = moved(base, v, effect, pv, wrap)
            if keep and i:
                np.maximum(out[i], out[i - 1] * keep, out=out[i])
        if progress:
            progress(i + 1, len(values))
    return out
