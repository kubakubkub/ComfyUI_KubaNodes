"""
render.py

Full-resolution render of a kubakub director document (the JSON the window saves in the node) and the
outputs that go back into the workflow. numpy + opencv, no ComfyUI imports (tests/test_director.py).

Document (version 1):
    {"version": 1, "canvas": [W, H],
     "layers": [ ...top first (= in front)... {
        "id": "L3", "name": "kiosk", "kind": "image" | "paint" | "adjust" | "base" | "light" | "shape",
        "source": "input:0" | "file:<name in input/>" | "",      image / paint pixels (base = the node's image)
        "visible": true, "opacity": 1.0, "blend": "normal",
        "x": 0, "y": 0, "w": 100, "h": 100, "rotation": 0, "flip_h": false, "flip_v": false,   image placement
        "clip": "",                  region selector (plan syntax): the layer only shows inside those regions
        "clip_feather": 0,           soft edge of the clip, canvas px
        "mask": {"by": "", "mode": "shape" | "brightness", "invert": false},   by = layer id or "below"
        "holes": "",                 base only: region selector cut open, layers below show through
        "adjust": {"brightness": 1, "contrast": 1, "saturate": 1, "hue": 0, "sepia": 0},
        "action": "rediffuse" | "edges" | "keep", "prompt": "", "denoise": 0.5,
        "shape": {"type": "rect" | "ellipse" | "poly", "points": [[u, v] 0..1 in the box], "feather": px}, "color": "#rrggbb",
                                   shape only: a solid colour shape in the layer's box (a solid = a rect over the canvas;
                                   hidden and used by another layer's mask = a shape mask); feather = soft edge (sigma, px)
        "light": {...}}]}          light only: the rig (scene3d/scene_view.rig_from_doc); the node renders it in
                                   Cycles and hands the pixels in as sources["light:<id>"] (a full-canvas layer)

What is in front of what is the list order: a layer below the base is seen only through its holes.
"""

from __future__ import annotations

import json
import os
import re
import threading

import cv2
import numpy as np

from . import blend as bl
from . import fx as fxm

ACTIONS = ("rediffuse", "edges", "keep")
KINDS = ("image", "paint", "adjust", "base", "light", "shape")
SHAPES = ("rect", "ellipse", "poly")


class DocumentError(ValueError):
    pass


# ------------------------------------------------------------------------------------------------
# document
# ------------------------------------------------------------------------------------------------

def _num(d, k, default):
    v = d.get(k, default)
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else float(default)


def _video(v) -> dict:
    """LTX animation of a placed layer: {on, prompt, t_start, t_end (-1 = to the end), motion} -> plan keys."""
    v = v if isinstance(v, dict) else {}
    return {"on": bool(v.get("on")), "prompt": " ".join(str(v.get("prompt") or "").split()).replace("//", "/"),
            "t_start": max(0.0, _num(v, "t_start", 0.0)), "t_end": _num(v, "t_end", -1.0),
            "motion": min(max(_num(v, "motion", 1.0), 0.0), 1.0)}


def _shape(sh) -> dict:
    sh = sh if isinstance(sh, dict) else {}
    pts = []
    for q in sh.get("points") or []:
        if isinstance(q, (list, tuple)) and len(q) == 2 and all(_is_num(v) for v in q):
            pts.append([min(max(float(q[0]), 0.0), 1.0), min(max(float(q[1]), 0.0), 1.0)])
    t = sh.get("type") if sh.get("type") in SHAPES else "rect"
    if t == "poly" and len(pts) < 3:
        t = "rect"
    return {"type": t, "points": pts, "feather": min(max(_num(sh, "feather", 0.0), 0.0), 500.0)}


def _react(r) -> dict | None:
    """'light_react': {"by": light layer id ("" = the first light layer), "amount": 0..1} or None (off)."""
    if not isinstance(r, dict) or r.get("on") is False:
        return None
    amt = min(max(_num(r, "amount", 1.0), 0.0), 1.0)
    return {"by": str(r.get("by") or ""), "amount": amt} if amt > 0 else None


def _hex(c):
    c = str(c or "")
    if re.fullmatch(r"#[0-9a-fA-F]{6}", c):
        return tuple(int(c[i:i + 2], 16) / 255.0 for i in (1, 3, 5))
    return (1.0, 1.0, 1.0)


class _Memo:
    """A small cache bounded by bytes (oldest entries dropped first), safe to share between frame threads."""

    def __init__(self, limit=256 << 20):
        self.limit, self.size, self.d, self.lock = limit, 0, {}, threading.Lock()

    def get(self, k):
        return self.d.get(k)

    def put(self, k, v, nbytes):
        with self.lock:
            if k in self.d:
                return
            while self.d and self.size + nbytes > self.limit:
                self.size -= self.d.pop(next(iter(self.d)))[1]
            if nbytes <= self.limit:
                self.d[k], self.size = (v, nbytes), self.size + nbytes


def shape_source(L, max_side=4096, cache=None):
    """A shape layer's pixels (RGBA float) and the layer with its box grown by the feather margin, for place().
    Drawn like the window: the path filled in the box, then a Gaussian blur (sigma = feather) that spreads out.
    cache: a _Memo (render: one per render / sequence): the same shape is drawn once, its pixels read only."""
    sh, (r, g, b) = L["shape"], _hex(L["color"])
    f = sh["feather"]
    if sh["type"] == "rect" and f <= 0.5:                 # a constant colour: 2 px are enough, place() scales them
        out = np.empty((2, 2, 4), np.float32)
        out[..., :3], out[..., 3] = (r, g, b), 1.0
        return out, L
    key = (sh["type"], tuple(map(tuple, sh["points"])), f, L["color"], L["w"], L["h"], max_side)   # all it reads but x, y
    hit = cache.get(key) if cache is not None else None
    if hit is None:
        got = _shape_pixels(L, max_side)
        if cache is not None:
            got[0].flags.writeable = False
            cache.put(key, got, got[0].nbytes)
    else:
        got = hit[0]
    out, pad, gw, gh = got
    if pad <= 0:
        return out, L
    return out, dict(L, x=L["x"] - pad, y=L["y"] - pad, w=gw, h=gh)   # the grown box, relative to this x, y


def _shape_pixels(L, max_side):
    """shape_source's drawing: (pixels, pad, grown w, grown h)."""
    sh, (r, g, b) = L["shape"], _hex(L["color"])
    f = sh["feather"]
    bw, bh = max(1.0, abs(L["w"])), max(1.0, abs(L["h"]))
    pad = 3.0 * f
    s = min(1.0, max_side / max(bw + 2 * pad, bh + 2 * pad))
    w, h, p = max(2, int(round((bw + 2 * pad) * s))), max(2, int(round((bh + 2 * pad) * s))), pad * s
    a = np.zeros((h, w), np.uint8)
    S = 16                                                # 4 bits of sub-pixel precision
    if sh["type"] == "ellipse":
        cv2.ellipse(a, (int(round(w / 2 * S)), int(round(h / 2 * S))), (int(round(bw * s / 2 * S)), int(round(bh * s / 2 * S))),
                    0, 0, 360, 255, -1, cv2.LINE_AA, 4)
    elif sh["type"] == "poly":
        pts = np.array([[p + u * bw * s, p + v * bh * s] for u, v in sh["points"]]) * S
        cv2.fillPoly(a, [np.round(pts).astype(np.int32)], 255, cv2.LINE_AA, 4)
    else:
        a[int(round(p)):int(round(p + bh * s)), int(round(p)):int(round(p + bw * s))] = 255
    alpha = a.astype(np.float32) / 255.0
    if f > 0.5:
        alpha = cv2.GaussianBlur(alpha, (0, 0), sigmaX=f * s, borderType=cv2.BORDER_CONSTANT)
    out = np.empty((h, w, 4), np.float32)
    out[..., :3], out[..., 3] = (r, g, b), alpha
    sx, sy = (1 if L["w"] >= 0 else -1), (1 if L["h"] >= 0 else -1)
    return out, pad, (bw + 2 * pad) * sx, (bh + 2 * pad) * sy


def parse(doc) -> dict:
    """Validate and fill defaults. Accepts the dict or its JSON text; empty -> a document with just the base."""
    if isinstance(doc, str):
        doc = json.loads(doc) if doc.strip() else {}
    if not isinstance(doc, dict):
        raise DocumentError("the director document must be a JSON object")
    if doc.get("version", 1) != 1:
        raise DocumentError(f"director document version {doc.get('version')} is not supported")
    layers = doc.get("layers") or [{"id": "base", "name": "base", "kind": "base"}]
    out, ids = [], set()
    for i, raw in enumerate(layers):
        if not isinstance(raw, dict):
            raise DocumentError(f"layer {i} is not an object")
        kind = raw.get("kind", "image")
        if kind not in KINDS:
            raise DocumentError(f"layer {i}: unknown kind {kind!r}")
        lid = str(raw.get("id") or f"L{i}")
        if lid in ids:
            raise DocumentError(f"duplicate layer id {lid!r}")
        ids.add(lid)
        mode = raw.get("blend", "normal")
        if mode not in bl.MODES:
            raise DocumentError(f"layer {lid}: unknown blend mode {mode!r}")
        m = raw.get("mask") or {}
        adj = raw.get("adjust") or {}
        action = raw.get("action", "keep" if kind in ("base", "adjust", "light", "shape") else "rediffuse")
        if action not in ACTIONS:
            raise DocumentError(f"layer {lid}: unknown action {action!r}")
        out.append({
            "id": lid, "name": str(raw.get("name") or lid), "kind": kind, "source": str(raw.get("source") or ""),
            "visible": bool(raw.get("visible", True)), "opacity": min(max(_num(raw, "opacity", 1.0), 0.0), 1.0),
            "blend": mode, "x": _num(raw, "x", 0), "y": _num(raw, "y", 0), "w": _num(raw, "w", 0), "h": _num(raw, "h", 0),
            "rotation": _num(raw, "rotation", 0), "flip_h": bool(raw.get("flip_h")), "flip_v": bool(raw.get("flip_v")),
            "clip": str(raw.get("clip") or ""), "holes": str(raw.get("holes") or ""),
            "mask": {"by": str(m.get("by") or ""), "mode": m.get("mode", "shape") if m.get("mode") in ("shape", "brightness") else "shape",
                     "invert": bool(m.get("invert"))},
            "adjust": {k: _num(adj, k, d) for k, d in (("brightness", 1), ("contrast", 1), ("saturate", 1), ("hue", 0), ("sepia", 0))},
            "action": action, "prompt": str(raw.get("prompt") or ""), "denoise": min(max(_num(raw, "denoise", 0.5), 0.0), 1.0),
            "light": dict(raw.get("light")) if kind == "light" and isinstance(raw.get("light"), dict) else {},
            "video": _video(raw.get("video")),
            "color": str(raw.get("color") or "#ffffff"),
            "fx": fxm.parse(raw.get("fx")),
            "light_react": _react(raw.get("light_react")),
            "shape": _shape(raw.get("shape")) if kind == "shape" else None,
            "stagger": [b for b in raw.get("motion") or [] if isinstance(b, dict) and b.get("type") == "stagger" and b.get("on") is not False]
                       if raw.get("clip") else [],
            "clip_feather": max(0.0, _num(raw, "clip_feather", 0)) if raw.get("clip") else 0.0,
            "clip_field": raw.get("clip_field") if raw.get("clip") and isinstance(raw.get("clip_field"), dict)
                          and isinstance(raw["clip_field"].get("values"), list) else None,
            "clip_mix": [[str(m[0]), min(max(float(m[1]), 0.0), 1.0)] for m in raw.get("clip_mix") or []
                         if isinstance(m, (list, tuple)) and len(m) == 2 and _is_num(m[1])] if raw.get("clip") else [],
        })
    if sum(1 for q in out if q["kind"] == "base") != 1:
        raise DocumentError("the document needs exactly one base layer")
    canvas = doc.get("canvas")
    return {"version": 1, "canvas": list(canvas) if isinstance(canvas, (list, tuple)) and len(canvas) == 2 else None,
            "layers": out}


# ------------------------------------------------------------------------------------------------
# regions
# ------------------------------------------------------------------------------------------------

def select_regions(table: dict | None, selector: str) -> list[int]:
    """Region ids matching a plan selector ('W_F1_*', 'group:Windows', 'tag:front', 'a, b', '!x')."""
    if not selector.strip() or not table:
        return []
    regions = table.get("regions", [])
    want = selector.strip().lower()
    exact = [r["region_id"] for r in regions if str(r.get("name", "")).lower() == want]
    if exact:
        return exact
    from .. import plan as pl
    try:
        sec = pl.parse_rules(f"[{selector}]\ndenoise = 0.5\n")[0]
    except pl.PlanError:
        return []
    return [r["region_id"] for r in regions if sec.match(r) is not None]


def region_mask(labels: np.ndarray | None, table: dict | None, selector: str) -> np.ndarray | None:
    ids = select_regions(table, selector)
    if labels is None or not ids:
        return None if not selector.strip() else np.zeros(labels.shape if labels is not None else (1, 1), np.float32)
    return np.isin(labels, ids).astype(np.float32)


# ------------------------------------------------------------------------------------------------
# layer pixels, each in its own bounding box (x0, y0, x1, y1): a 10k matrix with many small layers
# never holds more than a few full-canvas arrays
# ------------------------------------------------------------------------------------------------

STRIP = 256                                    # rows per strip for full-canvas work


def _box_of(corners, W, H):
    x0 = max(0, int(np.floor(corners[:, 0].min())) - 2)       # 2 px margin: anti-aliased edges stay inside
    y0 = max(0, int(np.floor(corners[:, 1].min())) - 2)
    x1 = min(W, int(np.ceil(corners[:, 0].max())) + 3)
    y1 = min(H, int(np.ceil(corners[:, 1].max())) + 3)
    return (x0, y0, x1, y1) if x1 > x0 and y1 > y0 else None


def premultiply(rgba):
    """The premultiplied float32 RGBA place() warps: (rgb * a, a)."""
    pre = np.empty(rgba.shape[:2] + (4,), np.float32)
    np.multiply(rgba[..., :3], rgba[..., 3:4], out=pre[..., :3])
    pre[..., 3] = rgba[..., 3]
    return pre


def place(rgba: np.ndarray, layer: dict, W: int, H: int, pre=None):
    """
    Warp a source (h, w, 4) by the layer's box, rotation (deg, clockwise) and flips.
    Returns (box, rgb, alpha) cut to the layer's bounding box on the W x H canvas, or None if outside.
    pre: premultiply(rgba) when the caller keeps it (a source placed many times).
    """
    h, w = rgba.shape[:2]
    bw = layer["w"] or w
    bh = layer["h"] or h
    sx = bw / w * (-1 if layer["flip_h"] else 1)
    sy = bh / h * (-1 if layer["flip_v"] else 1)
    a = np.radians(layer["rotation"])
    c, s_ = np.cos(a), np.sin(a)
    cx, cy = layer["x"] + bw / 2, layer["y"] + bh / 2
    # dst = T(cx, cy) R(a) S(sx, sy) T(-w/2, -h/2) src, in pixel-corner coordinates; OpenCV works on
    # pixel centres, hence the half-pixel shifts
    A = np.array([[c * sx, -s_ * sy], [s_ * sx, c * sy]], np.float64)
    t = np.array([cx, cy]) - A @ np.array([w / 2, h / 2])
    corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float64) @ A.T + t - 0.5
    box = _box_of(corners + 0.5, W, H)
    if box is None:
        return None
    x0, y0, x1, y1 = box
    M = np.concatenate([A, (A @ np.array([0.5, 0.5]) + t - 0.5 - [x0, y0])[:, None]], 1)
    # premultiplied warp, edge pixels clamped like the browser's drawImage ...
    if pre is None:
        pre = premultiply(rgba)
    out = cv2.warpAffine(pre, M, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    # ... and the box itself cut with an anti-aliased outline exactly at its edges
    cover = np.zeros((y1 - y0, x1 - x0), np.uint8)
    cv2.fillConvexPoly(cover, np.round((corners - [x0, y0]) * 16).astype(np.int32), 255, lineType=cv2.LINE_AA, shift=4)
    alpha = np.clip(out[..., 3], 0, 1)
    rgb = np.clip(out[..., :3] / np.maximum(alpha, 1e-6)[..., None], 0, 1)
    return box, rgb, alpha * (cover.astype(np.float32) / 255)


def _full(rgba, W, H):
    """A full-canvas source (paint layers) cut to the box of its visible pixels, or None if empty."""
    if rgba.shape[:2] != (H, W):
        rgba = cv2.resize(rgba, (W, H), interpolation=cv2.INTER_AREA if rgba.shape[1] > W else cv2.INTER_LINEAR)
    alpha = rgba[..., 3]
    rows = np.flatnonzero(alpha.max(axis=1) > 0)
    if not len(rows):
        return None
    cols = np.flatnonzero(alpha[rows[0]:rows[-1] + 1].max(axis=0) > 0)
    x0, y0, x1, y1 = int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1
    return (x0, y0, x1, y1), np.clip(rgba[y0:y1, x0:x1, :3], 0, 1), np.clip(alpha[y0:y1, x0:x1], 0, 1).copy()


def _per_source(mc, tag, src, make, *key):
    """make() once per source array in mc (a render / sequence), read only. The entry keeps src alive, so its id()
    can not be reused by another array while cached. Never for video / light frames (new arrays every frame)."""
    k = (tag, id(src)) + key
    hit = mc.get(k)
    if hit is None or hit[0] is not src:
        got = make()
        for a in (got if isinstance(got, tuple) else (got,)):
            if isinstance(a, np.ndarray):
                a.flags.writeable = False
        hit = mc[k] = (src, got)
    return hit[1]


def _as_rgba(img):
    img = np.asarray(img, np.float32)
    if img.ndim == 2:
        img = np.repeat(img[..., None], 3, -1)
    if img.shape[-1] == 3:
        img = np.concatenate([img, np.ones(img.shape[:2] + (1,), np.float32)], -1)
    return img


def mask_full(entry, W, H) -> np.ndarray:
    """A layer mask entry (id, name, box, crop) as a full W x H float32 mask."""
    _, _, (x0, y0, x1, y1), crop = entry
    m = np.zeros((H, W), np.float32)
    m[y0:y1, x0:x1] = crop
    return m


# ------------------------------------------------------------------------------------------------
# timeline: keyframes per value path, evaluated exactly like the window (web/kubakub_director.js keyValue)
# ------------------------------------------------------------------------------------------------

_HEX = re.compile(r"^#[0-9a-fA-F]{6}$")


def _step(o, seg):
    if isinstance(o, list) and seg.startswith("#"):
        return next((e for e in o if isinstance(e, dict) and e.get("id") == seg[1:]), None)
    return o.get(seg) if isinstance(o, dict) else None


def path_get(o, path):
    for seg in path.split("."):
        if o is None:
            return None
        o = _step(o, seg)
    return o


def path_set(o, path, v):
    *head, last = path.split(".")
    for seg in head:
        o = _step(o, seg)
        if o is None:
            return False
    if isinstance(o, dict):
        o[last] = v
        return True
    return False


def _is_num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _js_round(x):
    return int(np.floor(x + 0.5))


def sorted_keys(keys):
    """The valid keys, sorted by time (key_value(..., presorted=True) takes them as they are)."""
    return sorted((k for k in keys or [] if isinstance(k, dict) and _is_num(k.get("t"))), key=lambda k: k["t"])


def key_value(keys, t, presorted=False):
    """keys [{t, v, e: smooth | linear | hold}] (any order) -> the value at t (None without keys)."""
    ks = keys if presorted else sorted_keys(keys)
    if not ks:
        return None
    if t <= ks[0]["t"]:
        return ks[0].get("v")
    if t >= ks[-1]["t"]:
        return ks[-1].get("v")
    i = 0
    while i < len(ks) - 2 and ks[i + 1]["t"] <= t:
        i += 1
    a, b = ks[i], ks[i + 1]
    av, bv = a.get("v"), b.get("v")
    hexes = isinstance(av, str) and isinstance(bv, str) and _HEX.match(av) and _HEX.match(bv)
    if a.get("e") == "hold" or not (_is_num(av) and _is_num(bv) or hexes):
        return av
    u = (t - a["t"]) / max(1e-9, b["t"] - a["t"])
    if a.get("e") != "linear":
        u = u * u * (3 - 2 * u)
    if _is_num(av):
        return av + (bv - av) * u
    A = [int(av[i:i + 2], 16) for i in (1, 3, 5)]
    B = [int(bv[i:i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{_js_round(x + (y - x) * u):02x}" for x, y in zip(A, B))


def timeline(doc) -> dict:
    tl = (doc or {}).get("timeline") or {}
    dur = min(max(float(tl.get("duration") or 10), 0.5), 3600.0)
    fps = int(min(max(round(float(tl.get("fps") or 25)), 1), 120))
    return {"duration": dur, "fps": fps, "time": min(max(float(tl.get("time") or 0), 0.0), dur),
            "frames": max(1, int(round(dur * fps)))}


def _copy1(L):
    return {k: dict(v) if isinstance(v, dict) else list(v) if isinstance(v, list) else v for k, v in L.items()}


def animate(doc, t, ctx=None):
    """The document at time t (a copy): every keyframed value (layer 'anim', loops included) written in, then the
    behaviours (layer 'motion', director/motion.py) added on top. ctx: motion.context(doc, level) for beat / sound
    behaviours (built from the document when missing, without the loudness curve)."""
    from . import motion as mo
    d = json.loads(doc) if isinstance(doc, str) else json.loads(json.dumps(doc))
    if not isinstance(d, dict):
        return d
    dur = timeline(d)["duration"]
    if not isinstance(d.get("timeline"), dict):
        d["timeline"] = {}
    d["timeline"]["time"] = t                         # stagger weights are evaluated at the document time
    if ctx is None and any(isinstance(L, dict) and L.get("motion") for L in d.get("layers", [])):
        ctx = mo.context(d)

    kv = lambda ks, tt: key_value(ks, tt, True)       # noqa: E731  (keys sorted once per frame below)

    def at(L, tt, anim):                              # keys (loops) + behaviours of one layer at tt, in place
        mv = L.get("motion") if isinstance(L.get("motion"), list) else []
        loops = mo.loop_modes(mv)
        for path, keys in anim.items():
            v = mo.keyed(keys, tt, loops.get(path), kv)
            if v is not None:
                path_set(L, path, v)
        offs = mo.offsets(mv, tt, dur, ctx)
        if offs:
            for path, v in mo.compose(offs, lambda p: path_get(L, p)).items():
                path_set(L, path, v)

    out = []
    src_layers = d.get("layers", [])
    swaps = mo.swap_clips(src_layers, t, dur, ctx) if any(isinstance(L, dict) and mo.swap_of(L.get("motion")) for L in src_layers) else {}
    for idx, L in enumerate(src_layers):
        if not isinstance(L, dict):
            out.append(L)
            continue
        mix = swaps.get(str(L.get("id")))
        if mix and mix != [[L.get("clip"), 1.0]]:     # the masks this layer shows through now (swap behaviour)
            L["clip_mix"] = mix
        else:
            L.pop("clip_mix", None)
        fb = mo.field_of(L.get("motion")) if L.get("clip") else None
        if fb:                                        # where its field move is now (and its trail), for the clip
            L["clip_field"] = dict(fb, values=mo.field_values(fb, t, dur, ctx))
        else:
            L.pop("clip_field", None)
        rb = mo.repeat_of(L.get("motion")) if L.get("kind") in ("image", "shape") else None
        if rb and isinstance(L.get("mask"), dict) and L["mask"].get("by") == "below":
            nxt = src_layers[idx + 1] if idx + 1 < len(src_layers) and isinstance(src_layers[idx + 1], dict) else None
            L["mask"] = dict(L["mask"], by=str(nxt.get("id")) if nxt else "")   # the real layer below, as in the window
        anim = {p: sorted_keys(ks) for p, ks in (L.get("anim") or {}).items() if isinstance(ks, list)}
        raw = json.loads(json.dumps({k: v for k, v in L.items() if k != "anim"})) if rb else None   # before this frame, no keys
        at(L, t, anim)
        out.append(L)
        if rb:                                        # the copies, below the layer, copy 1 nearest
            delay = rb.get("delay") if _is_num(rb.get("delay")) else 0.0
            for k in range(1, mo.repeat_count(rb)):
                # one level deep is enough: keys, behaviours and the placement write top-level values and
                # values one dict down (fx.blur, shape.feather, light_react.amount); deeper data is shared, read only
                if delay and not L.get("media"):
                    c = _copy1(raw)
                    at(c, t - k * delay, anim)
                else:
                    c = _copy1(L)
                c["id"] = f"{L.get('id')}~{k}"
                out.append(mo.repeat_apply(c, rb, k))
    for L in out:                                     # applied: the frame needs no keys, only the stagger (clip weights; swaps are in clip_mix)
        if isinstance(L, dict):
            L.pop("anim", None)
            if L.get("motion"):
                L["motion"] = [b for b in L["motion"] if isinstance(b, dict) and b.get("type") == "stagger"]
    d["layers"] = out
    return d


def scale_doc(doc, s):
    """Placements scaled for a smaller render of the same composition (sequence previews)."""
    d = json.loads(json.dumps(doc))
    for L in d.get("layers", []):
        for key in ("x", "y", "w", "h"):
            if _is_num(L.get(key)):
                L[key] = L[key] * s
        if isinstance(L.get("fx"), dict):             # effect distances are pixels too
            for key in ("blur", "sharpen_radius", "glow_radius", "grain_size"):
                if _is_num(L["fx"].get(key)):
                    L["fx"][key] = L["fx"][key] * s
        if isinstance(L.get("shape"), dict) and _is_num(L["shape"].get("feather")):
            L["shape"]["feather"] = L["shape"]["feather"] * s
        if _is_num(L.get("clip_feather")):
            L["clip_feather"] = L["clip_feather"] * s
        for b in [*(L.get("motion") or []), L.get("clip_field")]:     # a noise field's blobs are pixels too
            if isinstance(b, dict) and (b.get("type") == "field" or "values" in b) and b.get("field") == "noise":
                b["noise_px"] = (b["noise_px"] if _is_num(b.get("noise_px")) else 64.0) * s
    if isinstance(d.get("canvas"), list) and len(d["canvas"]) == 2:
        d["canvas"] = [max(1, round(d["canvas"][0] * s)), max(1, round(d["canvas"][1] * s))]
    return d


def layer_shape(doc, layer_id, sources, labels=None, table=None, W=0, H=0):
    """A layer's own placed alpha (with its clip) as a full W x H mask, or None (glow sources of light layers)."""
    d = parse(doc)
    L = next((q for q in d["layers"] if q["id"] == layer_id), None)
    if L is None:
        return None
    if L["kind"] == "base":
        m = np.ones((H, W), np.float32)
    elif L["kind"] == "shape":
        src, Lp = shape_source(L)
        got = place(src, Lp, W, H)
    elif L["kind"] in ("image", "paint"):
        src = sources.get(L["source"])
        got = None if src is None else (place(_as_rgba(src), L, W, H) if L["kind"] == "image" else _full(_as_rgba(src), W, H))
        if got is None:
            return None
        m = np.zeros((H, W), np.float32)
        (x0, y0, x1, y1), _, a = got
        m[y0:y1, x0:x1] = a
    else:
        return None
    if L["clip"]:
        cm = region_mask(labels, table, L["clip"])
        if L["clip_mix"]:
            cm = None
            for sel, wt in L["clip_mix"]:
                one = region_mask(labels, table, sel)
                if one is not None:
                    cm = one * wt if cm is None else np.minimum(1.0, cm + one * wt)
        if cm is not None and L["clip_feather"] >= 0.5:
            cm = cv2.GaussianBlur(np.ascontiguousarray(cm, np.float32), (0, 0), L["clip_feather"] / 2.0, borderType=cv2.BORDER_REPLICATE)
        m = m * cm if cm is not None and cm.shape == m.shape else np.zeros_like(m)
    return m


def glow_masks(doc, rig, sources, labels=None, table=None, W=0, H=0):
    """
    The glow entries of a light rig (scene_view.rig_from_doc) -> [(mask W x H, colour, strength, label)].
    'by' = 'layer:<id>' (that layer's placed shape incl. its clip) or a region selector (plan syntax, like clip to).
    """
    out = []
    for g in rig.get("glow", []):
        by = g["by"].strip()
        m = layer_shape(doc, by[6:], sources, labels, table, W, H) if by.startswith("layer:") else region_mask(labels, table, by)
        if m is None or not m.any():
            continue
        out.append((m, tuple(g["color"]), float(g["strength"]), f"glow {by}"))
    return out


def _strips(y0, y1):
    for r in range(y0, y1, STRIP):
        yield r, min(r + STRIP, y1)


# ------------------------------------------------------------------------------------------------
# render
# ------------------------------------------------------------------------------------------------

_POOL, _POOL_LOCK = None, threading.Lock()
_FIELD_LOCK = threading.Lock()


def _pool():
    """Threads for the strips of the still (numpy / OpenCV release the GIL)."""
    global _POOL
    with _POOL_LOCK:
        if _POOL is None:
            from concurrent.futures import ThreadPoolExecutor
            _POOL = ThreadPoolExecutor(max_workers=max(1, min(12, (os.cpu_count() or 4) - 2)), thread_name_prefix="kuba-strips")
    return _POOL


def render(doc, base: np.ndarray, sources: dict | None = None, labels: np.ndarray | None = None,
           table: dict | None = None, seam_px: int = 24, image_only: bool = False, mask_cache: dict | None = None,
           frame_seed: int = 0, with_alpha: bool = False):
    """
    doc: document (dict or JSON); base: (H, W, 3|4) float 0..1 = the base layer's pixels; sources:
    {"input:0": rgba, "file:x.png": rgba, ...}; labels / table: KUBA_REGIONS as numpy + atlas (optional).
    Returns {"image", "layer_masks" [(id, name, box, crop)] (mask_full() expands one), "changed",
    "labels", "table", "rules", "notes", "layer_count"}.
    image_only: just {"image", "notes"} (sequence frames, projector composites: no masks, bands or regions).
    mask_cache: a dict shared by calls with the same labels / table (a sequence): region selections and their
    masks are computed once instead of per frame. frame_seed: the frame number (the grain effect changes per frame).
    with_alpha (with image_only): also "alpha", how much the layers cover each pixel (an export with alpha: hide the
    base and the image is the colour premultiplied by that coverage, over black).
    """
    import zlib
    from . import motion as mo
    mc = mask_cache if mask_cache is not None else {}
    d = parse(doc)
    tl_now = timeline((json.loads(doc) if doc.strip() else {}) if isinstance(doc, str) else doc)   # "" = no composition yet
    H, W = base.shape[:2]
    FULL = (0, 0, W, H)
    sources = sources or {}
    notes = []
    layers = d["layers"]
    n = len(layers)
    base_rgb = base[..., :3] if base.dtype == np.float32 else base[..., :3].astype(np.float32)
    base_a = base[..., 3].astype(np.float32) if base.shape[-1] == 4 else None

    def ids_of(selector, what):
        k = ("ids", selector)
        ids = mc.get(k)
        if ids is None:
            ids = mc[k] = select_regions(table, selector) if labels is not None else []
        if not ids:
            notes.append(f"{what}: '{selector}' matches no region")
        return ids

    def full_mask(ids):                                  # bool, the whole canvas, cached per selection
        k = ("mask", tuple(ids))
        m = mc.get(k)
        if m is None:
            m = mc[k] = np.isin(labels, ids)
        return m

    def region_crop(ids, box):
        x0, y0, x1, y1 = box
        if labels is None or not ids:
            return np.zeros((y1 - y0, x1 - x0), np.float32)
        return full_mask(ids)[y0:y1, x0:x1].astype(np.float32)

    stagger_luts = {}                                   # this render (one time): per selection + stagger settings

    def clip_crop(L, ids, box):
        """The clip mask of a box, with the layer's clip feather: a soft edge, in canvas pixels."""
        f = L["clip_feather"]
        x0, y0, x1, y1 = box
        if f < 0.5 or x1 <= x0 or y1 <= y0:
            return clip_hard(L, ids, box)
        mg = int(f * 1.5) + 2                           # the blur reaches 3 sigma, sigma = feather / 2
        X0, Y0, X1, Y1 = max(0, x0 - mg), max(0, y0 - mg), min(W, x1 + mg), min(H, y1 + mg)
        big = cv2.GaussianBlur(np.ascontiguousarray(clip_hard(L, ids, (X0, Y0, X1, Y1)), np.float32), (0, 0), f / 2.0,
                               borderType=cv2.BORDER_REPLICATE)
        return big[y0 - Y0:y1 - Y0, x0 - X0:x1 - X0]

    def clip_hard(L, ids, box):
        """The clip mask, and with a field behaviour (clip_field from animate) only as far as its move has come."""
        m = clip_regions(L, ids, box)
        fb = L["clip_field"]
        if not fb or labels is None:
            return m
        from .. import maskfields as mf
        if L["clip_mix"]:                               # in a swap: the field runs over the masks it holds now
            ids = sorted({i for sel, _wt in L["clip_mix"] for i in ids_of(sel, f"{L['name']} swap")})
        num = lambda k, d: float(fb[k]) if _is_num(fb.get(k)) else d      # noqa: E731
        kind = fb.get("field") if fb.get("field") in mo.FIELD_KINDS else "edge distance"
        per = fb.get("per") if fb.get("per") in mo.FIELD_PER else "each region"
        order, effect = str(fb.get("order") or "left"), fb.get("effect") if fb.get("effect") in mo.FIELD_EFFECTS else "reveal"
        key = ("field", tuple(ids), kind, per, num("angle", 0.0), num("cx", 0.5), num("cy", 0.5), order, int(num("seed", 1)),
               num("noise_px", 64.0), bool(fb.get("invert")))
        f = mc.get(key)
        if f is None:
            with _FIELD_LOCK:                           # the frames of a sequence render in threads: built once
                f = mc.get(key)
                if f is None:
                    own = [i for i in ids if i >= 0]
                    lab, n = mf.compact(labels, own)
                    f = mc[key] = mf.field(lab, n, kind, per, key[4], (key[5], key[6]), order, key[8], key[9], key[10], ids=own)[0]
        x0, y0, x1, y1 = box
        vals = [[float(v[0]), float(v[1])] for v in fb["values"] if isinstance(v, (list, tuple)) and len(v) == 2]
        if not vals:
            return m
        return m * mf.mask_at(f[y0:y1, x0:x1], vals, effect, num("soft", 0.05), num("width", 0.2), int(num("rings", 3)))

    def clip_regions(L, ids, box):
        """The clip regions; with stagger behaviours each region weighted by its stagger value at the document time.
        A layer in a swap (clip_mix from animate) shows through the masks it holds now, each at its weight."""
        mix = L["clip_mix"]
        if (not L["stagger"] and not mix) or labels is None or not (ids or mix):
            return region_crop(ids, box)
        key = (tuple(ids), json.dumps([L["stagger"], mix], sort_keys=True))
        lut = stagger_luts.get(key)
        if lut is None:
            lmax = mc.get(("lmax",))
            if lmax is None:
                lmax = mc[("lmax",)] = int(labels.max()) if labels.size else 0
            lut = np.zeros(lmax + 1, np.float32)
            if mix:
                ids = []
                for sel, wt in mix:
                    for i in ids_of(sel, f"{L['name']} swap"):
                        if 0 <= i <= lmax:
                            lut[i] = min(1.0, lut[i] + wt)
                            ids.append(i)
                ids = sorted(set(ids))
            else:
                lut[[i for i in ids if 0 <= i <= lmax]] = 1.0
            if L["stagger"]:
                geo = (table or {}).get("canvas_bbox") or {g["region_id"]: g.get("bbox", [0, 0, 0, 0]) for g in (table or {}).get("regions", [])}
                cw, ch = (table or {}).get("canvas_size") or (W, H)
                regs = [(int(i), list(geo[i])) for i in ids if i in geo]
                wts = mo.stagger_weights(L["stagger"], tl_now["time"], tl_now["duration"], regs, cw, ch) or {}
                for i in ids:
                    if 0 <= i <= lmax:
                        lut[i] *= wts.get(int(i), 1.0)
            stagger_luts[key] = lut
        x0, y0, x1, y1 = box
        return lut[labels[y0:y1, x0:x1]]

    # every layer's own pixels: (box, rgb or None = the base image, alpha or None = opaque)
    own = [None] * n
    for i, L in enumerate(layers):
        if L["kind"] == "base":
            alpha = None if base_a is None else base_a.copy()
            if L["holes"]:
                ids = ids_of(L["holes"], "base holes")
                if ids:
                    alpha = np.ones((H, W), np.float32) if alpha is None else alpha
                    hm = full_mask(ids)
                    for r0, r1 in _strips(0, H):
                        alpha[r0:r1] *= 1 - hm[r0:r1]
            piece = [FULL, None, alpha]
        elif L["kind"] == "shape":
            src, Lp = shape_source(L, cache=mc.setdefault(("shapes",), _Memo()))
            got = place(src, Lp, W, H)
            if got is None:
                notes.append(f"{L['name']}: outside the canvas or empty")
                continue
            piece = list(got)
        elif L["kind"] in ("image", "paint", "light"):
            key = f"light:{L['id']}" if L["kind"] == "light" else L["source"]
            src = sources.get(key)
            if src is None:
                if L["visible"]:
                    notes.append(f"{L['name']}: " + ("not rendered (connect a scene)" if L["kind"] == "light"
                                                     else f"source '{key}' not found") + ", layer skipped")
                continue
            if key.startswith(("media:", "light:")):    # video / light frames: new pictures every frame
                got = place(_as_rgba(src), L, W, H) if L["kind"] == "image" else _full(_as_rgba(src), W, H)
            elif L["kind"] == "image":                  # premultiplied once per source (repeat copies, frames)
                pre = _per_source(mc, "pre", src, lambda: premultiply(_as_rgba(src)))
                got = place(pre, L, W, H, pre=pre)
            else:
                got = _per_source(mc, "full", src, lambda: _full(_as_rgba(src), W, H), W, H)
            if got is None:
                notes.append(f"{L['name']}: outside the canvas or empty")
                continue
            piece = list(got)
        else:
            continue
        if L["clip"]:
            cm = clip_crop(L, ids_of(L["clip"], f"{L['name']} clip"), piece[0])
            piece[2] = cm if piece[2] is None else piece[2] * cm
        if fxm.active(L["fx"]):                         # effects on the layer's own pixels: its box grows by the spread
            (x0, y0, x1, y1), prgb, pal = piece
            mg = fxm.margin(L["fx"])
            X0, Y0, X1, Y1 = max(0, x0 - mg), max(0, y0 - mg), min(W, x1 + mg), min(H, y1 + mg)
            big_rgb = np.zeros((Y1 - Y0, X1 - X0, 3), np.float32)
            big_a = np.zeros((Y1 - Y0, X1 - X0), np.float32)
            big_rgb[y0 - Y0:y1 - Y0, x0 - X0:x1 - X0] = base_rgb[y0:y1, x0:x1] if prgb is None else prgb
            big_a[y0 - Y0:y1 - Y0, x0 - X0:x1 - X0] = 1.0 if pal is None else pal
            frgb, fa = fxm.apply(big_rgb, big_a, L["fx"], seed=frame_seed * 1000003 + zlib.crc32(L["id"].encode()))
            piece = [(X0, Y0, X1, Y1), frgb, fa]
        rr = L["light_react"]
        if rr and L["kind"] in ("image", "paint", "shape"):
            # shaded by the light: x (light on the clay / the clay's own brightness), coloured lamps tint it
            lt = next((q for q in layers if q["kind"] == "light" and (q["id"] == rr["by"] if rr["by"] else True)), None)
            lsrc = sources.get(f"light:{lt['id']}") if lt else None
            if lsrc is None:
                notes.append(f"{L['name']}: no light layer rendered to react to")
            else:
                (x0, y0, x1, y1), prgb, pal = piece
                albedo = max(0.05, float(_num(lt["light"], "clay", 0.7)))
                shade = np.clip(np.asarray(lsrc)[y0:y1, x0:x1, :3] / albedo, 0, 4)
                base_part = base_rgb[y0:y1, x0:x1] if prgb is None else prgb
                piece = [(x0, y0, x1, y1), np.clip(base_part * (1 - rr["amount"] + rr["amount"] * shade), 0, 1).astype(np.float32), pal]
        own[i] = piece

    def mask_crop(i, box):
        """The mask layer's value (shape / brightness, inverted) over box, or None when there is no mask."""
        m = layers[i]["mask"]
        if not m["by"]:
            return None
        j = i + 1 if m["by"] == "below" else next((k for k, q in enumerate(layers) if q["id"] == m["by"]), None)
        if j is None or j >= n or j == i or own[j] is None:
            if (i, "mask") not in warned:
                notes.append(f"{layers[i]['name']}: mask layer '{m['by']}' not usable, mask ignored")
                warned.add((i, "mask"))
            return None
        (sx0, sy0, sx1, sy1), srgb, sa = own[j]
        x0, y0, x1, y1 = box
        out = np.zeros((y1 - y0, x1 - x0), np.float32)
        ix0, iy0, ix1, iy1 = max(x0, sx0), max(y0, sy0), min(x1, sx1), min(y1, sy1)
        if ix1 > ix0 and iy1 > iy0:
            a_ = np.ones((iy1 - iy0, ix1 - ix0), np.float32) if sa is None else sa[iy0 - sy0:iy1 - sy0, ix0 - sx0:ix1 - sx0]
            if m["mode"] == "brightness":
                c = base_rgb[iy0:iy1, ix0:ix1] if srgb is None else srgb[iy0 - sy0:iy1 - sy0, ix0 - sx0:ix1 - sx0]
                a_ = a_ * (0.2126 * c[..., 0] + 0.7152 * c[..., 1] + 0.0722 * c[..., 2])
            out[iy0 - y0:iy1 - y0, ix0 - x0:ix1 - x0] = a_
        return 1 - out if m["invert"] else out

    warned = set()
    # the still: the strips of one layer in threads (they write disjoint rows); not the sequence frames and
    # projector renders (image_only), which already run in parallel
    par = not image_only and W * H >= 2_000_000

    def run_strips(f, strips):
        if par and len(strips) > 1:
            list(_pool().map(f, strips))
        else:
            for rr in strips:
                f(rr)

    # composite bottom to top onto opaque black (as the window does), strip by strip
    comp = np.zeros((H, W, 3), np.float32)
    cov = np.zeros((H, W), np.float32) if with_alpha else None
    shape = [None] * n                   # (box, alpha incl. clip and mask, or None = opaque full canvas)
    for i in range(n - 1, -1, -1):
        L = layers[i]
        if not L["visible"]:
            continue
        op = L["opacity"]
        if L["kind"] == "adjust":
            a_ = L["adjust"]
            clip_ids = ids_of(L["clip"], f"{L['name']} clip") if L["clip"] else None
            fx_full = None
            if fxm.active(L["fx"]):                     # effects on everything below (they need the whole frame)
                graded = bl.grade(comp, a_["brightness"], a_["contrast"], a_["saturate"], a_["hue"], a_["sepia"])
                fx_full = fxm.apply(graded, None, L["fx"], seed=frame_seed * 1000003 + zlib.crc32(L["id"].encode()))[0]
            if par:                                     # shared lookups (region mask, stagger weights, mask note) once, before the threads
                if clip_ids is not None:
                    clip_crop(L, clip_ids, (0, 0, W, 0))
                mask_crop(i, (0, 0, W, 0))

            def adj_strip(rr, i=i, L=L, a_=a_, op=op, clip_ids=clip_ids, fx_full=fx_full):
                r0, r1 = rr
                sub = comp[r0:r1]
                al = np.full((r1 - r0, W), op, np.float32)
                if clip_ids is not None:
                    al *= clip_crop(L, clip_ids, (0, r0, W, r1))
                mk = mask_crop(i, (0, r0, W, r1))
                if mk is not None:
                    al *= mk
                src_ = fx_full[r0:r1] if fx_full is not None else bl.grade(sub, a_["brightness"], a_["contrast"], a_["saturate"],
                                                                           a_["hue"], a_["sepia"])
                bl.composite_into(sub, src_, al, L["blend"])
            run_strips(adj_strip, list(_strips(0, H)))
            continue
        if own[i] is None:
            continue
        box, rgb, alpha = own[i]
        mk = mask_crop(i, box)
        if mk is not None:
            alpha = mk if alpha is None else alpha * mk
        shape[i] = (box, alpha)
        x0, y0, x1, y1 = box

        def lay_strip(rr, L=L, rgb=rgb, alpha=alpha, op=op, x0=x0, y0=y0, x1=x1):
            r0, r1 = rr
            sub = comp[r0:r1, x0:x1]
            src = base_rgb[r0:r1, x0:x1] if rgb is None else rgb[r0 - y0:r1 - y0]
            al = np.full((r1 - r0, x1 - x0), op, np.float32) if alpha is None else alpha[r0 - y0:r1 - y0] * op
            bl.composite_into(sub, src, al, L["blend"])
            if cov is not None:
                cv_ = cov[r0:r1, x0:x1]
                cv_ *= (1 - al)
                cv_ += al
        run_strips(lay_strip, list(_strips(y0, y1)))

    if image_only:
        for r0, r1 in _strips(0, H):
            np.clip(comp[r0:r1], 0, 1, out=comp[r0:r1])
        return {"image": comp, "notes": notes, **({"alpha": np.clip(cov, 0, 1)} if cov is not None else {})}

    # where each placed layer is seen: its shape minus the shapes in front of it (regions follow the
    # shape, so a half transparent layer is still an object)
    seen = [None] * n
    above = np.zeros((H, W), np.float32)
    for i in range(n):
        if shape[i] is None or layers[i]["kind"] not in ("image", "paint", "base"):
            continue
        (x0, y0, x1, y1), alpha = shape[i]
        if layers[i]["kind"] == "base":
            for r0, r1 in _strips(y0, y1):
                al = 1.0 if alpha is None else alpha[r0 - y0:r1 - y0]
                above[r0:r1] = np.minimum(above[r0:r1] + al * (1 - above[r0:r1]), 1)
            continue
        ab = above[y0:y1, x0:x1]
        seen[i] = alpha * (1 - ab)
        above[y0:y1, x0:x1] = np.minimum(ab + seen[i], 1)
    del above
    placed = [i for i in range(n) if seen[i] is not None]
    layer_masks = [(layers[i]["id"], layers[i]["name"], shape[i][0], np.clip(seen[i], 0, 1)) for i in placed]

    # changed: re-diffused layers + an edge band around every non-kept layer, each in its padded box
    changed = np.zeros((H, W), np.uint8)
    k = max(1, int(seam_px))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))
    for i in placed:
        L = layers[i]
        if L["action"] == "keep":
            continue
        (x0, y0, x1, y1) = shape[i][0]
        hard = (seen[i] > 0.5).astype(np.uint8)
        pad = cv2.copyMakeBorder(hard, k + 1, k + 1, k + 1, k + 1, cv2.BORDER_CONSTANT, value=0)
        band = cv2.dilate(pad, kernel) - cv2.erode(pad, kernel)
        if L["action"] == "rediffuse":
            band = np.maximum(band, pad)
        px0, py0 = x0 - k - 1, y0 - k - 1
        cx0, cy0, cx1, cy1 = max(0, px0), max(0, py0), min(W, px0 + pad.shape[1]), min(H, py0 + pad.shape[0])
        np.maximum(changed[cy0:cy1, cx0:cx1], band[cy0 - py0:cy1 - py0, cx0 - px0:cx1 - px0],
                   out=changed[cy0:cy1, cx0:cx1])

    for r0, r1 in _strips(0, H):
        np.clip(comp[r0:r1], 0, 1, out=comp[r0:r1])
    out_labels, out_table, names = _regions_with_layers(labels, table, layers, seen, placed, shape, W, H, notes)
    return {"image": comp, "layer_masks": layer_masks, "changed": changed.astype(np.float32),
            "labels": out_labels, "table": out_table, "rules": rules_text(layers, placed, names), "notes": notes,
            "layer_count": n}


def region_stats(lab, n):
    """(boxes [x0, y0, x1, y1], areas, centroids) per label 0..n-1, counted strip by strip (no full xs / ys)."""
    from scipy import ndimage
    areas = np.zeros(n, np.int64)
    sx = np.zeros(n, np.float64)
    sy = np.zeros(n, np.float64)
    H, W = lab.shape
    xs = np.arange(W, dtype=np.float64)
    for r0, r1 in _strips(0, H):
        part = lab[r0:r1]
        m = part >= 0
        if not m.any():
            continue
        ids = part[m]
        areas += np.bincount(ids, minlength=n)[:n]
        sx += np.bincount(ids, weights=np.broadcast_to(xs, part.shape)[m], minlength=n)[:n]
        sy += np.bincount(ids, weights=np.broadcast_to(np.arange(r0, r1, dtype=np.float64)[:, None], part.shape)[m], minlength=n)[:n]
    boxes = np.zeros((n, 4), np.int64)
    for k, sl in enumerate(ndimage.find_objects(lab + 1, max_label=n)):
        if sl is not None:
            boxes[k] = [sl[1].start, sl[0].start, sl[1].stop, sl[0].stop]
    cent = np.stack([sx / np.maximum(areas, 1), sy / np.maximum(areas, 1)], 1)
    return boxes, areas, cent


def _slug(s):
    return re.sub(r"[^\w.+\-]+", "_", s).strip("_") or "layer"


def _regions_with_layers(labels, table, layers, seen, placed, shape, W, H, notes):
    """
    The input regions (ids and fields unchanged) with every placed layer cut in where it is seen (> 0.5)
    as a new region appended after them. Returns (labels, table, {layer index: region name}).
    """
    import copy
    lab = np.full((H, W), -1, np.int32) if labels is None else labels.astype(np.int32).copy()
    if table and labels is not None:
        out = copy.deepcopy(table)
    else:
        out = {"format": "kubakub.regions.atlas", "version": 1, "mode": "director", "regions": [], "groups": {}, "notes": []}
    out["width"], out["height"] = W, H
    regs = out["regions"]
    used = {str(r.get("name", "")) for r in regs}
    next_id = max([r["region_id"] for r in regs], default=-1) + 1
    names = {}
    for i in reversed(placed):             # bottom first, so a layer in front wins where they overlap
        L = layers[i]
        (x0, y0, x1, y1) = shape[i][0]
        hit = seen[i] > 0.5
        if not hit.any():
            notes.append(f"{L['name']}: not visible, no region")
            continue
        name, k = _slug(L["name"]), 2
        while name in used:
            name, k = f"{_slug(L['name'])}_{k}", k + 1
        used.add(name)
        names[i] = name
        lab[y0:y1, x0:x1][hit] = next_id
        regs.append({"region_id": next_id, "name": name, "group_id": "director", "tags": ["director", L["action"]],
                     "source": f"director:{L['id']}"})
        out.setdefault("groups", {}).setdefault("director", []).append(next_id)
        next_id += 1
    if regs:
        boxes, areas, centroids = region_stats(lab, next_id)
        for r in regs:
            k = r["region_id"]
            x0, y0, x1, y1 = (int(v) for v in boxes[k])
            r["bbox"] = [x0, y0, x1 - x0, y1 - y0]                  # x, y, w, h like every regions table
            r["area"] = int(areas[k])
            r["centroid"] = [round(float(v), 1) for v in centroids[k]]
        hidden = [r["name"] for r in regs if r["area"] == 0]
        if hidden:
            notes.append(f"covered completely by placed layers: {', '.join(hidden[:12])}" + (" ..." if len(hidden) > 12 else ""))
    out["unassigned_px"] = int((lab < 0).sum())
    return lab, out, names


def rules_text(layers, placed, names) -> str:
    """Plan rules for the placed layers that became regions (names as in the region table)."""
    lines = []
    for i in reversed(placed):
        if i not in names:
            continue
        L = layers[i]
        lines.append(f"[name:{names[i]}]")
        if L["action"] == "rediffuse":
            lines.append("strategy = inpaint")
            lines.append(f"denoise = {L['denoise']:.2f}")
            prompt = " ".join(L["prompt"].split()).replace("//", "/")
            if prompt:
                lines.append(f"prompt = {prompt}")
        else:
            lines.append("strategy = keep")
            lines.append(f"blend = {'on' if L['action'] == 'edges' else 'off'}")
        v = L.get("video") or {}
        if v.get("on"):                                # animated with LTX in the Region Video Sampler
            lines.append("animate = on")
            if v["prompt"]:
                lines.append(f"video_prompt = {v['prompt']}")
            lines.append(f"t_start = {v['t_start']:g}")
            lines.append(f"t_end = {v['t_end']:g}")
            lines.append(f"motion = {v['motion']:g}")
        lines.append("")
    return "\n".join(lines)
