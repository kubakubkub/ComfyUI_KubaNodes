"""
trackdraw.py

kubakub track draw: the tracks of kubakub tracker (kubakub/tracking.py) drawn as a diagram over the video or on
black. numpy / opencv / scipy only (tests/test_tracking.py).

The same look is drawn live in the browser by web/kubakub_track.js; what decides WHAT is drawn (states, labels,
which points connect, where stamps land) is written the same way in both files and checked against each other in
the test. Sizes are in pixels of a 1080 high picture and scale with the video.

The look (the node's hidden "look" text, JSON; DEFAULT_LOOK, PRESETS):
    background  video | black | transparent
    marks       [{type: box | brackets | outline | circle | cross | dot | label | skeleton | points,
                  kinds: [] (all) | [...], colour: "#hex" | "track", width, size, fill, opacity, text (label)}]
    connect     [{mode: nearest | within | network | tree | hull | hub | chain | trail | corners, kinds, same_kind,
                  k, distance (0-1 of the width), hub: biggest | first | centre | corner name, length (trail frames),
                  style: straight | curved | angle, dash, colour, width, opacity, fade}]
    inside      [{type: reveal | pixels | freeze | invert, kinds, block (pixels), shape: box | outline}]
    stamp       {on, kinds, every (px walked), size: blob | fixed, fixed (px), patch: now | first, stay (s, 0 = forever),
                 outline (draw a thin frame round each stamp)}
    span, fade  every mark, line set, inside, depth entry and the stamp trail has its own time: "span" [from, to] as
                0-1 of the clip (none = all of it) and "span_fade" (seconds in and out)
    blend       every mark, line set, inside, depth entry and the stamp trail has its own "blend" (the director's
                modes: normal, multiply, screen, overlay, darken, lighten, color dodge, color burn, hard light,
                soft light, difference, exclusion, hue, saturation, color, luminosity)
    depth       [{type: layers | lines | time | fog | slice, count, near / far (colours), colour, width, opacity,
                  delay (frames per layer), amount, from / to (0 far .. 1 near)}]    (needs the tracker's depth)
                marks: depth_scale (0 = off: bigger when near); labels: {z}; lines: same_depth (how close in depth)
    states      {words: [new, moving, still, lost], moving (px per frame), new (frames), lost (frames)}
"""

from __future__ import annotations

import copy
import json
import math

import cv2
import numpy as np

from .director import blend as bl
from .optional import need

STATE_WORDS = ["INPUT", "PROCESS", "STABLE", "ARCHIVED"]

DEFAULT_LOOK = {
    "background": "video",
    "depth": [],
    "marks": [{"type": "box", "colour": "#ffffff", "width": 1.0},
              {"type": "label", "colour": "#ffffff", "text": "{id}  {state}", "size": 14}],
    "connect": [{"mode": "within", "distance": 0.18, "colour": "#ffffff", "width": 0.8, "opacity": 0.6, "fade": True}],
    "inside": [],
    "stamp": {"on": False, "kinds": [], "every": 14, "size": "blob", "fixed": 40, "patch": "now", "stay": 0, "outline": False},
    "states": {"words": list(STATE_WORDS), "moving": 1.2, "new": 8, "lost": 24},
    # with a sound on the tracker: marks pulse (loudness + beat), lines flash on the beat, more 'within' lines when
    # it is loud, thicker lines when it is loud
    "sound": {"pulse": 0.5, "flash": 0.4, "reach": 0.5, "width": 0.0},
}

PRESETS = {
    "diagnostic": {                                   # thin boxes, state words, numbers, a fan of lines to the corners
        "background": "video",
        "marks": [{"type": "box", "colour": "#39ff6a", "width": 1.0},
                  {"type": "label", "colour": "#39ff6a", "text": "{data}", "size": 11, "at": "top right"},
                  {"type": "label", "colour": "#39ff6a", "text": "{state}", "size": 17, "at": "bottom left"},
                  {"type": "dot", "colour": "#39ff6a", "size": 2.5}],
        "connect": [{"mode": "corners", "colour": "#39ff6a", "width": 0.7, "opacity": 0.75},
                    {"mode": "nearest", "k": 2, "colour": "#ffffff", "width": 0.6, "opacity": 0.5}],
        "inside": [], "stamp": {"on": False}, "states": {"words": list(STATE_WORDS)}},
    "constellation": {                                # gold circles on black, ids, a network between them
        "background": "black",
        "marks": [{"type": "circle", "colour": "#e8b04a", "width": 1.0, "size": 1.0},
                  {"type": "dot", "colour": "#e8b04a", "size": 2.0},
                  {"type": "label", "colour": "#e8b04a", "text": "{id_hex}", "size": 11, "at": "below"}],
        "connect": [{"mode": "network", "colour": "#e8b04a", "width": 0.7, "opacity": 0.55, "fade": True}],
        "inside": [{"type": "reveal", "shape": "box"}], "stamp": {"on": False}},
    "stamp trail": {                                  # pixels of every walker stamped along its path, small id boxes
        "background": "video",
        "marks": [{"type": "box", "colour": "#ffffff", "width": 1.0, "size": 0.35},
                  {"type": "label", "colour": "#ffffff", "text": "ID {id_long}", "size": 10, "at": "below"}],
        "connect": [], "inside": [],
        "stamp": {"on": True, "every": 10, "size": "blob", "patch": "now", "stay": 0, "outline": False}},
}


def look(raw) -> dict:
    """The node's look text merged over DEFAULT_LOOK (lists replace, dicts merge); a preset name is a look too."""
    if isinstance(raw, str) and raw.strip() in PRESETS:
        raw = PRESETS[raw.strip()]
    try:
        got = json.loads(raw) if isinstance(raw, str) and raw.strip() else (raw if isinstance(raw, dict) else {})
    except json.JSONDecodeError:
        got = {}
    out = copy.deepcopy(DEFAULT_LOOK)
    for k, v in (got or {}).items():
        if isinstance(out.get(k), dict) and isinstance(v, dict):
            out[k].update(v)
        elif k in out:
            out[k] = v
    out["states"] = {**DEFAULT_LOOK["states"], **(out.get("states") or {})}
    out["stamp"] = {**DEFAULT_LOOK["stamp"], **(out.get("stamp") or {})}
    out["sound"] = {**DEFAULT_LOOK["sound"], **(out.get("sound") if isinstance(out.get("sound"), dict) else {})}
    return out


def sound_at(data, t):
    """What the tracker heard at frame t ({level, beat} 0-1), or None when it had no sound (= web soundAt)."""
    c = data.get("curves") or {}
    if not c.get("beat"):
        return None
    i = min(max(t, 0), len(c["beat"]) - 1)
    lv = c.get("level") or []
    return {"level": float(lv[i]) if i < len(lv) else 0.0, "beat": float(c["beat"][i])}


# ------------------------------------------------------------------------------------------------
# what is where at frame t (= web/kubakub_track.js)
# ------------------------------------------------------------------------------------------------

def centre(b):
    return (b[0] + b[2] / 2, b[1] + b[3] / 2)


def state_of(tr, t, st, unit):
    """(state index 0 new / 1 moving / 2 still / 3 lost, box) at frame t, or None when the track is not there."""
    n = len(tr["box"])
    k = t - tr["t0"]
    lost = int(st.get("lost", 24))
    if k < 0 or k >= n + lost:
        return None
    if k >= n:
        return 3, tr["box"][n - 1]
    b = tr["box"][k]
    if k < int(st.get("new", 8)):
        return 0, b
    p = tr["box"][k - 1]
    c, q = centre(b), centre(p)
    speed = math.hypot(c[0] - q[0], c[1] - q[1]) / unit
    return (1 if speed > float(st.get("moving", 1.2)) else 2), b


def speed_at(tr, t, unit):
    k = t - tr["t0"]
    if k < 1 or k >= len(tr["box"]):
        return 0.0
    c, q = centre(tr["box"][k]), centre(tr["box"][k - 1])
    return math.hypot(c[0] - q[0], c[1] - q[1]) / unit


def visible(data, t, lk, kinds=None):
    """[(track, state, box)] at frame t, by id; lost tracks only while the lost word is shown."""
    unit = data["H"] / 1080.0
    out = []
    for tr in data["tracks"]:
        if kinds and tr["kind"] not in kinds:
            continue
        s = state_of(tr, t, lk["states"], unit)
        if s is not None:
            out.append((tr, s[0], s[1]))
    out.sort(key=lambda e: e[0]["id"])
    return out


def z_at(tr, t):
    """A track's depth at frame t (0 far .. 1 near; held after its end), None without depth."""
    z = tr.get("z")
    if not z:
        return None
    return float(z[min(max(t - tr["t0"], 0), len(z) - 1)])


def span_weight(e, t, data):
    """How much of an element is there at frame t: 0 outside its span, 1 inside, ramping over its fade (= web)."""
    sp = (e or {}).get("span")
    if not sp or len(sp) != 2:
        return 1.0
    last = max(1, int(data["n"]) - 1)
    a, b = float(sp[0]) * last, float(sp[1]) * last
    if t < a - 1e-9 or t > b + 1e-9:
        return 0.0
    f = float(e.get("span_fade", 0) or 0) * float(data.get("fps") or 25)
    if f <= 0:
        return 1.0
    return max(0.0, min(1.0, (t - a) / f, (b - t) / f))


def span_start(e, data):
    sp = (e or {}).get("span")
    return float(sp[0]) * max(1, int(data["n"]) - 1) if sp and len(sp) == 2 else -1.0


def fmt(v, digits):
    return f"{v:.{digits}f}"


def label_text(text, tr, state, b, t, data, lk):
    W, H, fps = data["W"], data["H"], data["fps"] or 25.0
    unit = H / 1080.0
    words = (lk["states"].get("words") or STATE_WORDS) + STATE_WORDS
    cx, cy = centre(b)
    sp = speed_at(tr, t, unit)
    rep = {"{id}": str(tr["id"]), "{id_hex}": "%04X" % (tr["id"] * 2654435761 % 65536),
           "{id_long}": "%07d" % (1006000 + tr["id"] * 7919 % 1000), "{kind}": tr["kind"], "{label}": tr.get("label", ""),
           "{state}": str(words[state]), "{x}": fmt(cx / W, 3), "{y}": fmt(cy / H, 3), "{speed}": fmt(sp, 1),
           "{size}": fmt(b[2] * b[3] / (W * H) * 100, 2), "{frame}": str(t), "{t}": fmt(t / fps, 2),
           "{data}": "%04d.%04d" % (int(cx) % 10000, int(cy * 10000 / H) % 10000),
           "{z}": fmt(z_at(tr, t), 3) if z_at(tr, t) is not None else ""}
    out = str(text)
    for k, v in rep.items():
        out = out.replace(k, v)
    return out


# ---- the graph of lines (= web/kubakub_track.js connect())
def _d(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def hull_idx(P):
    """Convex hull (monotone chain), indices in order; fewer than 3 points = all of them."""
    idx = sorted(range(len(P)), key=lambda i: (P[i][0], P[i][1]))
    if len(idx) < 3:
        return idx

    def cross(o, a, b):
        return (P[a][0] - P[o][0]) * (P[b][1] - P[o][1]) - (P[a][1] - P[o][1]) * (P[b][0] - P[o][0])
    lower, upper = [], []
    for i in idx:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], i) <= 0:
            lower.pop()
        lower.append(i)
    for i in reversed(idx):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], i) <= 0:
            upper.pop()
        upper.append(i)
    return lower[:-1] + upper[:-1]


def delaunay_edges(P):
    """Bowyer-Watson (the same steps as the web file, so both draw the same network)."""
    n = len(P)
    if n < 2:
        return []
    if n == 2:
        return [(0, 1)]
    xs, ys = [p[0] for p in P], [p[1] for p in P]
    mx, my = min(xs), min(ys)
    dm = max(max(xs) - mx, max(ys) - my, 1.0) * 20
    pts = list(P) + [(mx - dm, my - dm), (mx + dm * 2, my - dm), (mx - dm, my + dm * 2)]
    tris = [(n, n + 1, n + 2)]

    def circ(tri):
        (ax, ay), (bx, by), (cx, cy) = pts[tri[0]], pts[tri[1]], pts[tri[2]]
        d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
        if abs(d) < 1e-12:
            return (0.0, 0.0, float("inf"))
        ux = ((ax * ax + ay * ay) * (by - cy) + (bx * bx + by * by) * (cy - ay) + (cx * cx + cy * cy) * (ay - by)) / d
        uy = ((ax * ax + ay * ay) * (cx - bx) + (bx * bx + by * by) * (ax - cx) + (cx * cx + cy * cy) * (bx - ax)) / d
        return (ux, uy, (ax - ux) ** 2 + (ay - uy) ** 2)
    cc = {tris[0]: circ(tris[0])}
    for i in range(n):
        px, py = pts[i]
        bad = [tr for tr in tris if (px - cc[tr][0]) ** 2 + (py - cc[tr][1]) ** 2 < cc[tr][2] - 1e-9]
        edges = {}
        for tr in bad:
            for e in ((tr[0], tr[1]), (tr[1], tr[2]), (tr[2], tr[0])):
                key = (min(e), max(e))
                edges[key] = edges.get(key, 0) + 1
        tris = [tr for tr in tris if tr not in bad]
        for (a, b), c in edges.items():
            if c == 1:
                tri = (a, b, i)
                tris.append(tri)
                cc[tri] = circ(tri)
    out = set()
    for tr in tris:
        for a, b in ((tr[0], tr[1]), (tr[1], tr[2]), (tr[2], tr[0])):
            if a < n and b < n:
                out.add((min(a, b), max(a, b)))
    return sorted(out)


def mst_edges(P):
    n = len(P)
    if n < 2:
        return []
    inside, best, frm, out = [False] * n, [float("inf")] * n, [-1] * n, []
    best[0] = 0.0
    for _ in range(n):
        u = min((i for i in range(n) if not inside[i]), key=lambda i: (best[i], i))
        inside[u] = True
        if frm[u] >= 0:
            out.append((min(u, frm[u]), max(u, frm[u])))
        for v in range(n):
            if not inside[v]:
                d = _d(P[u], P[v])
                if d < best[v] - 1e-9:
                    best[v], frm[v] = d, u
    return sorted(out)


def connect_edges(c, items, data, lk, t):
    """Lines of one connect entry at frame t: [((x0, y0), (x1, y1), length)]."""
    W, H = data["W"], data["H"]
    mode = c.get("mode", "within")
    kinds = c.get("kinds") or []
    its = [e for e in items if not kinds or e[0]["kind"] in kinds]
    P = [centre(e[2]) for e in its]
    groups = [list(range(len(its)))]
    if c.get("same_kind"):
        gs = {}
        for i, e in enumerate(its):
            gs.setdefault(e[0]["kind"], []).append(i)
        groups = [gs[k] for k in sorted(gs)]
    segs = []
    for g in groups:
        Q = [P[i] for i in g]
        pairs = []
        if mode == "nearest":
            k = max(1, int(c.get("k", 1)))
            s = set()
            for a in range(len(Q)):
                near = sorted((b for b in range(len(Q)) if b != a), key=lambda b: (_d(Q[a], Q[b]), b))[:k]
                s |= {(min(a, b), max(a, b)) for b in near}
            pairs = sorted(s)
        elif mode == "within":
            r = float(c.get("distance", 0.18)) * W
            pairs = [(a, b) for a in range(len(Q)) for b in range(a + 1, len(Q)) if _d(Q[a], Q[b]) < r]
        elif mode == "network":
            pairs = delaunay_edges(Q)
        elif mode == "tree":
            pairs = mst_edges(Q)
        elif mode == "hull":
            h = hull_idx(Q)
            pairs = [(h[i], h[(i + 1) % len(h)]) for i in range(len(h))] if len(h) > 2 else ([(h[0], h[1])] if len(h) == 2 else [])
        elif mode == "chain":
            pairs = [(a, a + 1) for a in range(len(Q) - 1)]          # by id = in the order they appeared
        elif mode == "hub" and Q:
            how = c.get("hub", "biggest")
            if how in ("biggest", "first"):
                hi = 0 if how == "first" else max(range(len(Q)), key=lambda i: (its[g[i]][2][2] * its[g[i]][2][3], -i))
                segs += [(Q[hi], Q[b]) for b in range(len(Q)) if b != hi]
            else:
                hp = {"centre": (W / 2, H / 2), "top left": (0, 0), "top right": (W, 0), "bottom left": (0, H),
                      "bottom right": (W, H)}.get(how, (W / 2, H / 2))
                segs += [(hp, q) for q in Q]
        elif mode == "corners" and len(Q) > 1:
            big = max(range(len(Q)), key=lambda i: (its[g[i]][2][2] * its[g[i]][2][3], -i))
            bx, by, bw, bh = its[g[big]][2]
            cs = [(bx, by), (bx + bw, by), (bx, by + bh), (bx + bw, by + bh)]
            for a in range(len(Q)):
                if a != big:
                    segs.append((Q[a], cs[min(range(4), key=lambda j: (_d(cs[j], Q[a]), j))]))
        elif mode == "trail":
            n = max(2, int(c.get("length", 20)))
            for i in g:
                tr = its[i][0]
                k = t - tr["t0"]
                path = [centre(tr["box"][j]) for j in range(max(0, k - n), min(k, len(tr["box"]) - 1) + 1)]
                segs += [(path[j], path[j + 1], (j + 1) / len(path)) for j in range(len(path) - 1)]
        if c.get("same_depth") is not None and c.get("same_depth") is not False:
            tol = 0.15 if c.get("same_depth") is True else float(c.get("same_depth"))
            zs = [z_at(its[i][0], t) for i in g]
            pairs = [(a, b) for a, b in pairs if zs[a] is None or zs[b] is None or abs(zs[a] - zs[b]) <= tol]
        segs += [(Q[a], Q[b]) for a, b in pairs]
    return segs


# ---- stamps (= web stampsAt): where a track leaves a patch, every `every` px it walks
def stamp_points(tr, every, unit):
    """[(frame index into the track, x, y)] - the first frame, then each time the path has grown by `every`."""
    step = max(1.0, float(every)) * unit
    out, walked, last = [], 0.0, None
    seen = tr.get("seen") or [1] * len(tr["box"])
    for k, b in enumerate(tr["box"]):
        c = centre(b)
        if last is not None:
            walked += math.hypot(c[0] - last[0], c[1] - last[1])
        last = c
        if (not out or walked >= step) and seen[k]:
            out.append((k, c[0], c[1]))
            walked = 0.0
    return out


# ------------------------------------------------------------------------------------------------
# drawing (OpenCV, full size)
# ------------------------------------------------------------------------------------------------

def _rgb(h, tr=None):
    if h == "track" and tr is not None:
        h = tr.get("colour", "#ffffff")
    h = str(h or "#ffffff").lstrip("#")
    try:
        return tuple(int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4))
    except ValueError:
        return (1.0, 1.0, 1.0)


def put(dst, src, a, mode="normal"):
    """dst (h, w, 3) in place: src laid on with coverage a (h, w) or a number, in a blend mode (= the browser's
    globalCompositeOperation, kubakub/director/blend.py)."""
    src = np.asarray(src, np.float32)
    if mode and mode != "normal" and mode in bl.MODES:
        src = np.clip(bl.blend(mode, dst, src), 0, 1)
    dst += (src - dst) * (a[..., None] if isinstance(a, np.ndarray) else a)


class Canvas:
    """RGB float picture + alpha. Everything of one colour in one step goes into one 8 bit coverage mask (drawn
    anti-aliased, 1/16 px precision, its opacity as the mask value), laid on the picture at flush()."""

    S = 16

    def __init__(self, rgb, alpha):
        self.rgb, self.a = rgb, alpha
        self.H, self.W = alpha.shape
        self.layers = {}

    def _m(self, colour):
        key = tuple(round(float(c), 4) for c in colour)
        m = self.layers.get(key)
        if m is None:
            m = self.layers[key] = np.zeros((self.H, self.W), np.uint8)
        return m

    def flush(self, mode="normal"):
        for colour, m in self.layers.items():
            ys, xs = np.nonzero(m.any(1))[0], np.nonzero(m.any(0))[0]
            if not len(ys):
                continue
            y0, y1, x0, x1 = ys[0], ys[-1] + 1, xs[0], xs[-1] + 1
            a = m[y0:y1, x0:x1].astype(np.float32) / 255.0
            r = self.rgb[y0:y1, x0:x1]
            put(r, np.broadcast_to(np.array(colour, np.float32), r.shape), a, mode)
            self.a[y0:y1, x0:x1] = np.maximum(self.a[y0:y1, x0:x1], a)
        self.layers = {}

    @staticmethod
    def _v(opacity):
        return int(round(255 * min(1.0, max(0.0, float(opacity)))))

    def lines(self, polys, colour, width, opacity=1.0, closed=False, dash=None):
        S = self.S
        if width < 1:
            opacity *= width
        v, th, m = self._v(opacity), max(1, int(round(width))), self._m(colour)
        if v == 0:
            return
        for p in polys:
            pts = np.round(np.asarray(p, np.float64) * S).astype(np.int32)
            if len(pts) < 2:
                continue
            if dash:
                on, off = dash
                for a, b in (zip(pts, np.roll(pts, -1, 0)) if closed else zip(pts[:-1], pts[1:])):
                    L = math.hypot(*(b - a)) / S
                    s = 0.0
                    while s < L:
                        e = min(L, s + on)
                        cv2.line(m, tuple((a + (b - a) * s / max(L, 1e-6)).astype(int)), tuple((a + (b - a) * e / max(L, 1e-6)).astype(int)),
                                 v, th, cv2.LINE_AA, shift=4)
                        s = e + off
            else:
                cv2.polylines(m, [pts], closed, v, th, cv2.LINE_AA, shift=4)

    def circle(self, c, r, colour, width, opacity=1.0, fill=0.0):
        S, m = self.S, self._m(colour)
        ctr, rad = (int(round(c[0] * S)), int(round(c[1] * S))), max(1, int(round(r * S)))
        if fill > 0:
            cv2.circle(m, ctr, rad, self._v(opacity * fill), -1, cv2.LINE_AA, shift=4)
        if width > 0:
            cv2.circle(m, ctr, rad, self._v(opacity * min(1.0, width)), max(1, int(round(width))), cv2.LINE_AA, shift=4)

    def poly_fill(self, poly, colour, opacity):
        cv2.fillPoly(self._m(colour), [np.round(np.asarray(poly) * self.S).astype(np.int32)], self._v(opacity), cv2.LINE_AA, shift=4)

    def text(self, s, org, colour, size, opacity=1.0):
        cv2.putText(self._m(colour), s, (int(org[0]), int(org[1])), cv2.FONT_HERSHEY_SIMPLEX, size / 22.0, self._v(opacity),
                    max(1, int(round(size / 12))), cv2.LINE_AA)


def _region_mask(tr, b, t, shape, H, W):
    m = np.zeros((H, W), np.uint8)
    o = (tr.get("outline") or {}).get(str(t)) if shape == "outline" else None
    if o and len(o) >= 3:
        cv2.fillPoly(m, [np.round(np.asarray(o) * 4).astype(np.int32)], 255, cv2.LINE_AA, shift=2)
    else:
        x0, y0 = int(max(0, math.floor(b[0]))), int(max(0, math.floor(b[1])))
        x1, y1 = int(min(W, math.ceil(b[0] + b[2]))), int(min(H, math.ceil(b[1] + b[3])))
        m[y0:y1, x0:x1] = 255
    return m.astype(np.float32) / 255.0


DEPTH_GRID = 480                                     # the depth work happens on a grid this wide (= web file)


def depth_grid(d, W, H):
    """The depth picture on the work grid (gw x gh, float 0-1, near = 1), lightly blurred."""
    gw = min(W, DEPTH_GRID)
    gh = max(2, int(round(gw * H / W)))
    g = cv2.resize(np.asarray(d, np.float32), (gw, gh), interpolation=cv2.INTER_AREA)
    return cv2.GaussianBlur(g, (0, 0), 1.0)


def _hex01(h):
    return np.array(_rgb(h), np.float32)


def depth_under(cv, lk, t, depth_at, frame_at, W, H, data=None):
    """The depth part under everything else: far = earlier frames, a slice, fog, coloured layers."""
    if depth_at is None:
        return
    entries = [e for e in lk.get("depth") or [] if e.get("type") in ("time", "slice", "fog", "layers")
               and (data is None or span_weight(e, t, data) > 0)]
    if not entries:
        return
    d = np.asarray(depth_at(t), np.float32)
    if d.shape != (H, W):
        d = cv2.resize(d, (W, H), interpolation=cv2.INTER_LINEAR)
    for e in entries:
        typ = e.get("type")
        sw = span_weight(e, t, data) if data is not None else 1.0
        cnt = max(2, int(e.get("count", 6)))
        band = np.minimum(cnt - 1, ((1.0 - d) * cnt).astype(np.int32))      # 0 = the nearest layer
        if typ == "time" and frame_at is not None:
            delay = max(0, int(e.get("delay", 4)))
            for k in range(1, cnt):
                m = band == k
                if m.any():
                    cv.rgb[m] = np.asarray(frame_at(max(0, t - k * delay)), np.float32)[..., :3][m]
        elif typ == "slice":
            lo, hi = float(e.get("from", 0.5)), float(e.get("to", 1.0))
            soft = 0.03
            m = np.clip((d - lo) / soft + 0.5, 0, 1) * np.clip((hi - d) / soft + 0.5, 0, 1)
            m = 1.0 - sw * (1.0 - m)
            cv.rgb *= m[..., None]
            cv.a *= m
        elif typ == "fog":
            f = np.clip(float(e.get("amount", 0.7)) * sw * (1.0 - d), 0, 1)
            put(cv.rgb, np.broadcast_to(_hex01(e.get("colour", "#000000")), cv.rgb.shape), f, e.get("blend", "normal"))
            cv.a = np.maximum(cv.a, f)
        elif typ == "layers":
            near, far = _hex01(e.get("near", "#f18a58")), _hex01(e.get("far", "#a187b7"))
            cols = np.stack([near + (far - near) * (k / (cnt - 1)) for k in range(cnt)])
            op = float(e.get("opacity", 0.6)) * sw
            put(cv.rgb, cols[band], op, e.get("blend", "normal"))
            cv.a = np.maximum(cv.a, op)


def depth_lines(lk, t, depth_at, W, H):
    """Contour lines of the depth (marching squares on the work grid): [(entry, [polyline in px ...])]."""
    if depth_at is None:
        return []
    entries = [e for e in lk.get("depth") or [] if e.get("type") == "lines"]
    if not entries:
        return []
    measure = need("skimage.measure", "track draw depth lines")
    g = depth_grid(depth_at(t), W, H)
    sx, sy = W / g.shape[1], H / g.shape[0]
    out = []
    for e in entries:
        cnt = max(2, int(e.get("count", 12)))
        polys = []
        for k in range(1, cnt):
            for c in measure.find_contours(g, k / cnt):
                if len(c) >= 3:
                    polys.append(np.stack([(c[:, 1] + 0.5) * sx, (c[:, 0] + 0.5) * sy], 1))
        out.append((e, polys))
    return out


def render_frame(data, lk, t, frame=None, frame_at=None, depth_at=None):
    """One frame: (rgb (H, W, 3) float, alpha (H, W)). frame = the video's frame t (float 0-1) or None;
    frame_at(i) gives any frame (stamps, freeze); depth_at(i) the depth (near = 1) for the look's depth part."""
    W, H = data["W"], data["H"]
    unit = H / 1080.0
    bgk = lk.get("background", "video")
    if bgk == "video" and frame is not None:
        rgb = np.array(frame[..., :3], np.float32, copy=True)
        alpha = np.ones((H, W), np.float32)
    else:
        rgb = np.zeros((H, W, 3), np.float32)
        alpha = np.zeros((H, W), np.float32) if bgk == "transparent" or frame is None and bgk == "video" else np.ones((H, W), np.float32)
    cv = Canvas(rgb, alpha)
    items = visible(data, t, lk)
    depth_under(cv, lk, t, depth_at, frame_at, W, H, data)

    # stamps: pixels left behind along each path
    st = lk.get("stamp") or {}
    sw_st = span_weight(st, t, data)
    if st.get("on") and frame_at is not None and sw_st > 0:
        st_from = span_start(st, data)
        stay = float(st.get("stay", 0)) * (data["fps"] or 25)
        for tr in data["tracks"]:
            if st.get("kinds") and tr["kind"] not in st["kinds"]:
                continue
            for k, x, y in stamp_points(tr, st.get("every", 14), unit):
                ts = tr["t0"] + k
                if ts > t or ts < st_from - 1e-9 or (stay > 0 and t - ts > stay):
                    continue
                b = tr["box"][0 if st.get("patch") == "first" else k]
                s = float(st.get("fixed", 40)) * unit if st.get("size") == "fixed" else max(4.0, min(b[2], b[3]))
                src_t = tr["t0"] if st.get("patch") == "first" else ts
                sx, sy = (centre(b) if st.get("patch") == "first" else (x, y))
                src = frame_at(src_t)
                x0, y0, x1, y1 = int(round(sx - s / 2)), int(round(sy - s / 2)), int(round(sx + s / 2)), int(round(sy + s / 2))
                dx0, dy0 = int(round(x - s / 2)), int(round(y - s / 2))
                cx0, cy0, cx1, cy1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
                if cx1 <= cx0 or cy1 <= cy0:
                    continue
                patch = src[cy0:cy1, cx0:cx1, :3]
                ox, oy = dx0 + (cx0 - x0), dy0 + (cy0 - y0)
                px0, py0, px1, py1 = max(0, ox), max(0, oy), min(W, ox + patch.shape[1]), min(H, oy + patch.shape[0])
                if px1 <= px0 or py1 <= py0:
                    continue
                fade = (1.0 if stay <= 0 else max(0.0, 1 - (t - ts) / stay)) * sw_st
                pp = patch[py0 - oy:py1 - oy, px0 - ox:px1 - ox]
                put(cv.rgb[py0:py1, px0:px1], pp, fade, st.get("blend", "normal"))
                cv.a[py0:py1, px0:px1] = np.maximum(cv.a[py0:py1, px0:px1], fade)
                if st.get("outline"):
                    cv.lines([[(px0, py0), (px1, py0), (px1, py1), (px0, py1)]], (1, 1, 1), 0.7 * unit, 0.6 * fade, closed=True)
        cv.flush()

    # inside the blobs: reveal, pixels, freeze, invert
    for e in lk.get("inside") or []:
        kinds = e.get("kinds") or []
        sw = span_weight(e, t, data)
        if sw <= 0:
            continue
        for tr, s, b in items:
            if (kinds and tr["kind"] not in kinds) or s == 3:
                continue
            m = _region_mask(tr, b, t, e.get("shape", "box"), H, W) * sw
            typ = e.get("type")
            if typ == "reveal" and frame is not None:
                src = frame[..., :3]
            elif typ == "pixels" and frame is not None:
                blk = max(2, int(float(e.get("block", 12)) * unit))
                small = cv2.resize(np.ascontiguousarray(frame[..., :3]), (max(1, W // blk), max(1, H // blk)), interpolation=cv2.INTER_AREA)
                src = cv2.resize(small, (W, H), interpolation=cv2.INTER_NEAREST)
            elif typ == "freeze" and frame_at is not None:
                f0, b0 = frame_at(tr["t0"]), tr["box"][0]
                src = cv.rgb.copy()
                x0, y0 = int(b0[0]), int(b0[1])
                x1, y1 = int(b0[0] + b0[2]), int(b0[1] + b0[3])
                dx, dy = int(b[0]), int(b[1])
                piece = f0[max(0, y0):min(H, y1), max(0, x0):min(W, x1), :3]
                ph, pw = piece.shape[:2]
                ph, pw = min(ph, H - max(0, dy)), min(pw, W - max(0, dx))
                if ph > 0 and pw > 0 and dx >= 0 and dy >= 0:
                    src[dy:dy + ph, dx:dx + pw] = piece[:ph, :pw]
            elif typ == "invert":
                src = 1.0 - cv.rgb
            else:
                continue
            put(cv.rgb, np.asarray(src, np.float32), m, e.get("blend", "normal"))
            cv.a = np.maximum(cv.a, m)

    # depth contour lines
    for e, polys in depth_lines(lk, t, depth_at, W, H):
        cv.lines(polys, _rgb(e.get("colour", "#ffffff")), float(e.get("width", 0.8)) * unit, float(e.get("opacity", 0.7)) * span_weight(e, t, data))
        cv.flush(e.get("blend", "normal"))

    sn, so = sound_at(data, t), lk.get("sound") or {}
    # lines
    for c in lk.get("connect") or []:
        if sn and c.get("mode", "within") == "within":
            c = dict(c, distance=float(c.get("distance", 0.18)) * (1 + float(so.get("reach", 0)) * sn["level"]))
        col = _rgb(c.get("colour", "#ffffff"))
        width = float(c.get("width", 1.0)) * unit * (1 + float(so.get("width", 0)) * sn["level"] if sn else 1.0)
        op = float(c.get("opacity", 1.0)) * span_weight(c, t, data) * (1 - float(so.get("flash", 0)) * 0.7 * (1 - sn["beat"]) if sn else 1.0)
        if op <= 0:
            continue
        dash = (6 * unit, 4 * unit) if c.get("dash") else None
        segs = connect_edges(c, items, data, lk, t)
        maxd = max([_d(s_[0], s_[1]) for s_ in segs] or [1.0])
        for sgm in segs:
            a, b = sgm[0], sgm[1]
            o = op * (sgm[2] if len(sgm) > 2 else 1.0)
            if c.get("fade") and len(sgm) == 2:
                o *= max(0.15, 1 - _d(a, b) / (maxd * 1.05))
            cv.lines([_styled(a, b, c.get("style", "straight"))], col, width, o, dash=dash)
        cv.flush(c.get("blend", "normal"))

    # marks
    for mk in lk.get("marks") or []:
        kinds = mk.get("kinds") or []
        typ = mk.get("type", "box")
        width = float(mk.get("width", 1.0)) * unit
        op = float(mk.get("opacity", 1.0)) * span_weight(mk, t, data)
        if op <= 0:
            continue
        for tr, s, b in items:
            if kinds and tr["kind"] not in kinds:
                continue
            col = _rgb(mk.get("colour", "#ffffff"), tr)
            o = op * (0.45 if s == 3 else 1.0)
            x, y, w, h = b
            sz = float(mk.get("size", 1.0))
            ds = float(mk.get("depth_scale", 0) or 0)
            zz = z_at(tr, t) if ds else None
            if zz is not None:                            # bigger when near, smaller when far
                sz *= max(0.1, 1.0 + ds * (zz - 0.5) * 2.0)
            if sn:                                        # the sound: they pulse with the loudness and the beat
                sz *= 1 + float(so.get("pulse", 0)) * (0.3 * sn["level"] + 0.5 * sn["beat"])
            if typ in ("box", "brackets") and sz != 1.0:
                cx, cy = centre(b)
                w, h = w * sz, h * sz
                x, y = cx - w / 2, cy - h / 2
            if typ == "box":
                cv.lines([[(x, y), (x + w, y), (x + w, y + h), (x, y + h)]], col, width, o, closed=True)
                if mk.get("fill"):
                    cv.poly_fill([(x, y), (x + w, y), (x + w, y + h), (x, y + h)], col, o * float(mk["fill"]))
            elif typ == "brackets":
                L = min(w, h) * 0.22
                cv.lines([[(x, y + L), (x, y), (x + L, y)], [(x + w - L, y), (x + w, y), (x + w, y + L)],
                          [(x + w, y + h - L), (x + w, y + h), (x + w - L, y + h)], [(x + L, y + h), (x, y + h), (x, y + h - L)]], col, width, o)
            elif typ == "outline":
                ol = (tr.get("outline") or {}).get(str(t))
                if ol and len(ol) >= 3:
                    cv.lines([ol], col, width, o, closed=True)
                    if mk.get("fill"):
                        cv.poly_fill(ol, col, o * float(mk["fill"]))
            elif typ == "circle":
                cv.circle(centre(b), max(2.0, math.hypot(w, h) / 2 * sz), col, width, o, float(mk.get("fill", 0)))
            elif typ == "cross":
                cx, cy = centre(b)
                L = max(4.0, 7 * sz * unit)
                cv.lines([[(cx - L, cy), (cx + L, cy)], [(cx, cy - L), (cx, cy + L)]], col, width, o)
            elif typ == "dot":
                cv.circle(centre(b), max(1.0, sz * unit), col, 0, o, 1.0)
            elif typ == "label":
                txt = label_text(mk.get("text", "{id}"), tr, s, b, t, data, lk)
                size = float(mk.get("size", 14)) * unit * (sz / float(mk.get("size", 1.0) or 1.0) if zz is not None else 1.0)
                at = mk.get("at", "top left")
                org = {"top left": (x, y - 4 * unit), "top right": (x + w + 4 * unit, y + size * 0.7), "below": (x, y + h + size + 2 * unit),
                       "bottom left": (x, y + h - 4 * unit), "centre": (x + w / 2, y + h / 2)}.get(at, (x, y - 4 * unit))
                cv.text(txt, org, col, size, o)
            elif typ in ("skeleton", "points"):
                pts = (tr.get("points") or {}).get(str(t))
                if not pts:
                    continue
                if typ == "points":
                    for p in pts.values():
                        cv.circle(p, max(1.0, 2 * sz * unit), col, 0, o, 1.0)
                else:
                    names = data.get("names", {}).get(tr["kind"], [])
                    edges = data.get("skeleton", {}).get(tr["kind"], [])
                    segs = [[pts[names[a]], pts[names[b_]]] for a, b_ in edges
                            if a < len(names) and b_ < len(names) and names[a] in pts and names[b_] in pts]
                    cv.lines(segs, col, width, o)
        cv.flush(mk.get("blend", "normal"))
    return np.clip(cv.rgb, 0, 1), np.clip(cv.a, 0, 1)


def _styled(a, b, style):
    """A line from a to b as a polyline: straight, a gentle curve, or a right angle."""
    if style == "angle":
        return [a, (b[0], a[1]), b]
    if style == "curved":
        mx, my = (a[0] + b[0]) / 2, (a[1] + b[1]) / 2
        nx, ny = -(b[1] - a[1]) * 0.2, (b[0] - a[0]) * 0.2
        cx, cy = mx + nx, my + ny
        return [((1 - u) ** 2 * a[0] + 2 * (1 - u) * u * cx + u * u * b[0], (1 - u) ** 2 * a[1] + 2 * (1 - u) * u * cy + u * u * b[1])
                for u in np.linspace(0, 1, 13)]
    return [a, b]
