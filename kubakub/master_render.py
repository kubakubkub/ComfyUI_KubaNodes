"""
master_render.py

Renders the show of the master timeline (workflow.extra.kubakub_master, see master.py) at the project size x scale
and the project fps: the frames and the sound. Used by kubakub timeline render (nodes/master/nodes_master.py).
No ComfyUI imports, so the tests run without a server; the directors' frames come in through a callback.

The picture of a piece is placed like the panel's preview (web/kubakub_master.js layoutRect): fit / fill / stretch
into the project frame, times size, moved by x / y project pixels, at its opacity, with its keys (opacity, x, y,
size) at every frame; the bottom track first, the top track over it. Clips are timed by the director's own video
model (media.segments / source_time), so a clip renders here exactly as a video layer does in the director, and
their sound is left out at a speed other than 1 like there.
"""

from __future__ import annotations

import math
import os

import cv2
import numpy as np

from . import imio
from .director import media as md
from .director.render import key_value, sorted_keys
from .keyframes import resample
from .project import whole_fps

SR = 44100
CHUNK_BYTES = 512 << 20             # decoded frames of one clip held at a time (memory)
ANIM = ("opacity", "x", "y", "scale")


def frame_count(duration, fps) -> int:
    """Frames of the show: duration x fps, half up (as the director counts its timeline)."""
    return max(1, int(math.floor(float(duration) * fps + 0.5)))


def out_size(W, H, scale):
    """The frames' size at `scale` of the project size, even (video encoders need it)."""
    w, h = max(16, round(W * scale)), max(16, round(H * scale))
    return w - w % 2, h - h % 2


def layout_rect(props, sw, sh, W, H):
    """= layoutRect in the panel: (x, y, w, h) of a sw x sh picture in the W x H project frame."""
    p = props or {}
    fit, scale = p.get("fit", "fit"), float(p.get("scale", 1) or 1)
    if fit == "stretch":
        w, h = float(W), float(H)
    else:
        k = max(W / sw, H / sh) if fit == "fill" else min(W / sw, H / sh)
        w, h = sw * k, sh * k
    w, h = w * scale, h * scale
    return (W - w) / 2 + float(p.get("x", 0) or 0), (H - h) / 2 + float(p.get("y", 0) or 0), w, h


def props_at(it, t):
    """= propsAt in the panel: the item's picture settings at timeline time t, the keyed ones (anim, keys in seconds
    from the item's start sorted by time as the panel saves them, the director's smooth / linear / hold) at t."""
    p = dict(it.get("props") or {})
    anim, lt = it.get("anim") or {}, t - float(it.get("start", 0) or 0)
    for name in ANIM:
        v = key_value(anim.get(name), lt, presorted=True)
        if v is not None:
            p[name] = v
    return p


def _top_scale(it):
    """The item's largest size: its size, or its biggest size key."""
    return max([float((it.get("props") or {}).get("scale", 1) or 1)] + [float(q["v"]) for q in (it.get("anim") or {}).get("scale", [])
                                                                if isinstance(q.get("v"), (int, float))])


def _shown_size(it, sw, sh, W, H, k):
    """The largest size (w, h) a sw x sh picture of the item is drawn at in the frames, at most its own size: decoded /
    resized once to that."""
    _, _, w, h = layout_rect({**(it.get("props") or {}), "scale": _top_scale(it)}, sw, sh, W, H)
    return max(2, min(sw, int(round(w * k)))), max(2, min(sh, int(round(h * k))))


def _frames(start, length, fps, N):
    """The show frames inside [start, start + length)."""
    return range(max(0, math.ceil((start - 1e-9) * fps)), min(N, math.ceil((start + length - 1e-9) * fps)))


def paste(canvas, img, rect, opacity):
    """Put img (h, w, 3 or 4, float 0..1) into canvas (H, W, 3) at rect = (x, y, w, h) in canvas pixels, alpha over."""
    ch, cw = canvas.shape[:2]
    x, y, w, h = rect
    x0, y0, x1, y1 = int(round(x)), int(round(y)), int(round(x + w)), int(round(y + h))
    rw, rh = x1 - x0, y1 - y0
    vx0, vy0, vx1, vy1 = max(0, x0), max(0, y0), min(cw, x1), min(ch, y1)
    if rw <= 0 or rh <= 0 or vx1 <= vx0 or vy1 <= vy0 or opacity <= 0:
        return
    sh, sw = img.shape[:2]
    if rw * rh <= 4 * cw * ch:                         # the usual case: resize all of it, cut the visible part
        interp = cv2.INTER_AREA if rw < sw and rh < sh else cv2.INTER_LINEAR
        big = cv2.resize(img, (rw, rh), interpolation=interp) if (rw, rh) != (sw, sh) else img
        part = big[vy0 - y0:vy1 - y0, vx0 - x0:vx1 - x0]
    else:                                               # zoomed far in: map only the visible part
        M = np.float32([[rw / sw, 0, x0 - vx0], [0, rh / sh, y0 - vy0]])
        part = cv2.warpAffine(img, M, (vx1 - vx0, vy1 - vy0), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    dst = canvas[vy0:vy1, vx0:vx1]
    if part.shape[2] == 3 and opacity >= 1:            # opaque: no blending
        dst[...] = part
        return
    a = (part[..., 3:4] if part.shape[2] > 3 else np.ones_like(part[..., :1])) * float(opacity)
    dst *= 1.0 - a
    dst += part[..., :3] * a


def _float_rgba(fr):
    """A decoded frame or picture (uint8 or uint16) -> float32 0..1."""
    return fr.astype(np.float32) / (65535.0 if fr.dtype == np.uint16 else 255.0)


def _read_picture(path):
    im = imio.imread(path, cv2.IMREAD_UNCHANGED)
    if im is None:
        return None
    im = _float_rgba(im)
    if im.ndim == 2:
        im = np.repeat(im[..., None], 3, axis=2)
    if im.shape[2] == 4:
        return cv2.cvtColor(im, cv2.COLOR_BGRA2RGBA)
    return cv2.cvtColor(im[..., :3], cv2.COLOR_BGR2RGB)


def _mix_into(sound, part, a0):
    """part [2, n] added into sound [2, S] from sample a0 on (cut at both ends)."""
    a1 = min(sound.shape[1], a0 + part.shape[1])
    if a1 > max(a0, 0):
        sound[:, max(a0, 0):a1] += part[:, max(0, -a0):max(0, -a0) + a1 - max(a0, 0)]


def _add_sound(sound, path, seg, sr, name, notes):
    """One clip's sound into sound [2, S] (= media.layer_audio for one segment, which only takes videos; also sound
    files). Left out at a speed other than 1, as in the director."""
    if abs(seg["speed"] - 1.0) > 1e-3:
        notes.append(f"{name}: sound left out at speed {seg['speed']:g} (as in the director)")
        return
    n = sound.shape[1]
    a0, a1 = int(round(seg["start"] * sr)), min(n, int(round((seg["start"] + seg["length"]) * sr)))
    if a1 <= a0 or a0 >= n:
        return
    clip = md.read_audio(path, seg["in"], seg["out"] - seg["in"], sr)
    if seg.get("reverse"):
        clip = np.ascontiguousarray(clip[:, ::-1])
    if seg["fill"] == "loop" and clip.shape[1]:
        clip = np.tile(clip, (1, int(np.ceil((a1 - a0) / clip.shape[1]))))
    _mix_into(sound, clip[:, :a1 - a0], a0)


def render(doc, scale, resolve, director, sr=SR):
    """
    The show -> (frames float32 [N, h, w, 3], sound float32 [2, S], fps, report lines).
    resolve(source) -> file path or None ('file:<input/>' / 'output:<output/>').
    director(node_id, w, h) -> (frames [n, h', w', 3] float, fps, sound dict or None) or None when that director is not
    connected to the render node.
    """
    doc = doc if isinstance(doc, dict) else {}
    fps = whole_fps(doc.get("fps", 25))
    dur = max(1.0 / fps, float(doc.get("duration", 10) or 10))
    W, H = int(doc.get("width", 1920) or 1920), int(doc.get("height", 1080) or 1080)
    ow, oh = out_size(W, H, scale)
    k = ow / W
    N = frame_count(dur, fps)
    S = int(round(N / fps * sr))
    frames = np.zeros((N, oh, ow, 3), np.float32)
    sound = np.zeros((2, S), np.float32)
    media = {m.get("id"): m for m in doc.get("media", []) if isinstance(m, dict)}
    notes, counts = [], {"clips": 0, "directors": 0, "pictures": 0, "sounds": 0}
    missing_dirs = set()

    def put(f, img, it, sw, sh):                                  # img (of a sw x sh source) into frame f, placed as the item is then
        p = props_at(it, f / fps)
        x, y, w, h = layout_rect(p, sw, sh, W, H)
        o = p.get("opacity")
        paste(frames[f], img, (x * k, y * k, w * k, h * k), 1.0 if o is None else min(1.0, float(o)))

    tracks = [t for t in doc.get("tracks", []) if isinstance(t, dict)]
    for tr in reversed(tracks):                                   # the bottom track first
        seen, heard = not tr.get("hidden"), not tr.get("muted")
        for it in sorted((i for i in tr.get("items", []) if isinstance(i, dict)), key=lambda i: float(i.get("start", 0) or 0)):
            if not (seen or heard):
                continue
            it = {**it, "anim": {n: sorted_keys(ks) for n, ks in (it.get("anim") or {}).items() if n in ANIM}}
            start, length = float(it.get("start", 0) or 0), float(it.get("length", -1) or -1)
            if it.get("type") == "director":
                top = _top_scale(it)                              # big enough for its largest size, not blown up
                got = director(it.get("node"), round(ow * top), round(oh * top))
                if got is None:
                    missing_dirs.add(str(it.get("node")))
                    continue
                dfr, dfps, dsound = got
                n = int(dfr.shape[0])
                tin = max(0.0, float(it.get("in", 0) or 0))
                length = length if length > 0 else max(1.0 / fps, n / dfps - tin)
                counts["directors"] += 1
                if seen:
                    for f in _frames(start, length, fps, N):
                        idx = min(n - 1, max(0, int(math.floor((tin + f / fps - start) * dfps + 1e-6))))
                        put(f, np.asarray(dfr[idx], np.float32), it, dfr.shape[2], dfr.shape[1])
                if heard and dsound is not None:
                    wave = np.asarray(dsound["waveform"])[0].astype(np.float32)
                    if wave.shape[0] == 1:
                        wave = np.repeat(wave, 2, axis=0)
                    wave = resample(wave, int(dsound.get("sample_rate", sr)), sr)
                    i0 = int(round(tin * sr))
                    _mix_into(sound, wave[:2, i0:i0 + int(round(length * sr))], int(round(start * sr)))
                continue

            m = media.get(it.get("media"))
            path = resolve(m.get("source", "")) if m else None
            if not path or not os.path.isfile(path):
                notes.append(f"missing file: {(m or {}).get('name') or it.get('media')}")
                continue
            kind, name = m.get("kind"), m.get("name")
            if kind == "video":
                info = md.probe(path)
                seg = md.segments({"segments": [it]}, info["duration"])
                src_fps, src_frames = float(info["fps"]), int(info.get("frames") or 0) or max(1, int(round(info["duration"] * info["fps"])))
                counts["clips"] += 1
                if seen:
                    want = []                                  # (output frame, source frame)
                    for f in _frames(seg[0]["start"], seg[0]["length"], fps, N):
                        r = md.source_time(seg, f / fps)
                        if r is not None:
                            want.append((f, md.frame_pick(r[1], src_fps, src_frames)[0][0]))
                    size, ch = _shown_size(it, info["w"], info["h"], W, H, k), 4 if info.get("alpha") else 3
                    step = max(1, CHUNK_BYTES // (size[0] * size[1] * 8))
                    for c in range(0, len(want), step):
                        part = want[c:c + step]
                        dec = md.decode(path, sorted({s for _, s in part}), size=size, info=info)
                        for f, s in part:
                            put(f, _float_rgba(dec[s][..., :ch]), it, info["w"], info["h"])
                if heard and not it.get("muted") and info.get("audio"):
                    _add_sound(sound, path, seg[0], sr, name, notes)
            elif kind == "audio":
                counts["sounds"] += 1
                if heard and not it.get("muted"):
                    seg = md.segments({"segments": [it]}, float((m.get("info") or {}).get("duration") or 0))[0]
                    _add_sound(sound, path, seg, sr, name, notes)
            else:
                pic = _read_picture(path)
                if pic is None:
                    notes.append(f"cannot read {name}")
                    continue
                counts["pictures"] += 1
                if seen:
                    sw, sh = pic.shape[1], pic.shape[0]
                    size = _shown_size(it, sw, sh, W, H, k)
                    if size != (sw, sh):
                        pic = cv2.resize(pic, size, interpolation=cv2.INTER_AREA)
                    for f in _frames(start, max(0.0, length), fps, N):
                        put(f, pic, it, sw, sh)

    np.clip(frames, 0.0, 1.0, out=frames)
    np.clip(sound, -1.0, 1.0, out=sound)
    if missing_dirs:
        notes.append("not connected to this node, left out: director node " + ", ".join(sorted(missing_dirs))
                     + " (connect its 'director' output)")
    report = [f"{N} frames at {fps} fps, {ow}x{oh} ({scale:g} of {W}x{H}), {N / fps:.2f} s: "
              + ", ".join(f"{v} {k_}" for k_, v in counts.items() if v)] + notes
    return frames, sound, fps, report
