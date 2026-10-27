"""kubakub point cloud: a picture or a video lifted into points by its depth, moved and filmed.

Every point is a sample of the picture on a grid (a little jittered); its place comes from the depth at that spot,
and every motion is a formula of the time and the point's own random numbers. Nothing is simulated, so any frame can
be drawn on its own: here with torch at full size, and live on the node (web/kubakub_pointcloud.js, WebGL2) with the
same formulas. Keep both in step: grid, pcg hash, place, motion, camera, morph timeline, point size and blends.

From the start view the points cover the picture exactly (a point further away is also spread wider), so a camera
move is what shows the depth. Pictures in a batch morph into each other: a point goes to the point of the next
picture with the same brightness rank.
"""

from __future__ import annotations

import json
import math
import os

import numpy as np
import torch
import torch.nn.functional as F

M32 = 0xFFFFFFFF
LUMA = (0.2126, 0.7152, 0.0722)
COLOURS = ("true", "mono", "tint", "negative")
BLENDS = ("glow", "ink", "solid")
WHERE = ("everywhere", "field", "tracked", "forces")  # where the flow and the beat act ("forces" = near the curves and regions)
CURVE_KINDS = ("wind", "attract", "repel", "curl+", "curl-")      # what a drawn curve does to the flowing points near it
REGION_KINDS = ("off", "attract", "repel", "curl+", "curl-")      # what a group of regions does
LINES = ("off", "within", "network", "tree")
EASES = ("smooth", "linear", "still")
MAX_POINT = 64.0                                    # the largest point in pixels (WebGL draws at least that much)
NEAR_Z = 0.05
FINE = 1.5                                          # points smaller than this (pixels) are shared bilinearly (glow / ink)
RING = 14.0                                         # a tracked thing's ring radius in pixels of a 1080 high picture (x its size)

DEFAULT_LOOK = {
    "density": 1920,            # points across the long side of the picture
    "jitter": 1.0,              # 0 = an exact grid, 1 = each point anywhere in its cell
    "depth": 1.0,               # how deep the picture goes: the farthest points are 1 + 2 x this away
    "far_cut": 0.0,             # points farther than this (0 far .. 1 near) are left out
    "size": 0.55,               # point size, in grid cells (1 = the points just touch from the start view)
    "opacity": 1.6,             # above 1 a point smaller than a pixel still shines (its light = opacity x its area, at most 1)
    "colour": "true",           # true / mono / tint / negative
    "tint": ["#a187b7", "#f18a58"],   # tint: dark -> bright
    "blend": "glow",            # glow (light adds up, on black) / ink (on white) / solid (the nearest point wins)
    "near": "bright",           # which side of the depth picture is near (depth nodes differ)
    "speed": 0.6,
    "drift": 0.0, "breathe": 0.0, "wind": 0.0, "scan": 0.0,
    "curl": 0.0,                # the cloud sways on big curly loops (neighbours together)
    # the fine motion (2026-10-27, after his note: subtle, every point on its own, not big groups): a scanner's look
    "shiver": 0.3,              # every point trembles along its own line of sight, at its own rhythm (in point spacings)
    "dropout": 0.0,             # which share of the points blinks out now and then, each on its own clock
    "rows": 0.0,                # the points gather into horizontal scan lines (0 = none, 1 = on the lines)
    "flow": 0.0,                # flowing points stream along smooth swirls (0 = none): born at their place, they flow, fade, are born again
    "swirl": 0.5,               # the size of the swirls (0..1 of the picture's height)
    "life": 3.0,                # seconds a flowing point lives
    "flowing": 0.35,            # which share of the points flows; the others hold the form
    "where": "everywhere",      # where the flow and the beat act: everywhere / field (white) / tracked (near the tracked things) / forces (near the curves and regions)
    "push": 0.6,                # how far the field pushes points out
    "swarm": 0.5,               # pictures: how far the points fly out while they travel to the next picture
    "trails": 0.0,              # seconds of each point's own way drawn behind it as a fine line (0 = none)
    "trail_share": 0.1,         # which share of the points draws a trail
    "trail_opacity": 0.5,
    "curves": [],               # curves drawn on the picture: {"kind": CURVE_KINDS, "pts": [[x, y] ...] in 0..1, y down} (an old
                                # look's arrows [x0, y0, x1, y1] under "forces" become wind curves)
    "regions": None,            # {group: REGION_KINDS} for the connected regions; None = the group with the most regions attracts
    "force": 1.0,               # how strong the curves and regions act
    "reach": 0.15,              # how far from a curve, a region (or a tracked thing) it still acts (0..1 of the picture's height)
    "sound": {"burst": 0.0, "swell": 0.2, "size": 0.0, "sparkle": 0.3, "motion": 0.0,
              "spectrum": 0.8,  # the depth hears the song: far = the bass ... near = the highs, each slice comes forward and lights up
              "sweep": 0.6,     # on every beat a thin bright plane rushes through the depth
              "shiver": 1.0,    # the highs make the points tremble more
              "fill": 0.5},     # when it is loud the blinked out points come back
    "camera": {"start": {"yaw": 0.0, "pitch": 0.0, "dolly": 0.0, "x": 0.0, "y": 0.0},
               "end": {"yaw": 18.0, "pitch": -6.0, "dolly": 0.35, "x": 0.0, "y": 0.0},
               "ease": "smooth",
               "shake": 0.0,    # handheld: the camera shakes a little (0 = steady)
               "path": []},     # a recorded move: [[0..1 of the move, yaw, pitch, dolly, x, y] ...]; it beats start / end
    "tracks": {"marks": True, "lines": "network", "distance": 0.2, "colour": "#f18a58", "opacity": 0.85, "size": 1.0},
}
_RANGES = {"density": (128, 3072), "jitter": (0, 1), "depth": (0, 4), "far_cut": (0, 1), "size": (0.1, 8), "opacity": (0.01, 4),
           "speed": (0, 4), "drift": (0, 2), "breathe": (0, 2), "wind": (0, 2), "scan": (0, 2), "curl": (0, 2), "push": (0, 3),
           "shiver": (0, 3), "dropout": (0, 1), "rows": (0, 1), "flow": (0, 3), "swirl": (0.05, 2), "life": (0.3, 12), "flowing": (0, 1),
           "swarm": (0, 2), "trails": (0, 3), "trail_share": (0.01, 1), "trail_opacity": (0, 1), "force": (0, 3), "reach": (0.02, 1)}
_SOUND_RANGES = {"burst": (0, 2), "swell": (0, 2), "size": (0, 2), "sparkle": (0, 1), "motion": (0, 2), "spectrum": (0, 2), "sweep": (0, 2),
                 "shiver": (0, 3), "fill": (0, 1)}
MAX_CURVES = 8
MAX_CURVE_PTS = 64
MAX_GROUPS = 12                                     # region groups with a force (the ones with the most regions)
FIELD_H = 270                                       # rows of the force field (the curves and regions), the same on the node
CLOSED = 0.06                                       # a curve whose ends are this near (of the height) is a loop
PULL = 0.12                                         # speed of a curve's or region's current at strength 1 (heights a second)
CURL_SCALE = 2.5                                    # a curl's eddies are this much smaller than the swirls
CURL_SPEED = 0.2                                    # and this fast at strength 1
# a region group's basis (signed distance, the way to the group, the place in its region), 8 bits each on the node
BASIS = ((-0.25, 1.0), (-1.0, 1.0), (-1.0, 1.0), (-2.0, 2.0), (-2.0, 2.0), (0.0, 1.0))
MAX_PATH = 600                                   # keys of a recorded camera move
TRAIL_STEPS = 12                                    # segments of a trail
FLOW_STEPS = 12                                     # steps a flowing point takes over its life (the same on the node)
MAX_TRACKED = 32                                    # tracked things the flow and the beat know of (the shader's array)
# the swirls: a stream function of four waves (angle, frequency, amplitude, tempo), so the flow has no sources and no
# sinks: the points stream around each other like a liquid instead of bunching up
WAVES = ((0.3, 1.0, 1.0, 0.31), (1.9, 1.7, 0.6, 0.23), (5.5, 2.3, 0.45, 0.11), (3.4, 2.9, 0.35, 0.17), (4.6, 4.3, 0.2, 0.13),
         (2.6, 6.1, 0.12, 0.29))
_CAM_RANGES = {"yaw": (-180, 180), "pitch": (-89, 89), "dolly": (-3, 0.95), "x": (-3, 3), "y": (-3, 3)}


def _num(v, default, lo, hi):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return default
    return default if not math.isfinite(f) else min(max(f, lo), hi)


def _hex(h, default):
    h = str(h or "").strip()
    return h if len(h) == 7 and h[0] == "#" and all(c in "0123456789abcdefABCDEF" for c in h[1:]) else default


def hex01(h):
    return tuple(int(h[i:i + 2], 16) / 255.0 for i in (1, 3, 5))


def look(raw) -> dict:
    """The look as set on the node (JSON or dict; missing or wrong values -> the defaults, numbers clamped)."""
    if isinstance(raw, str):
        try:
            raw = json.loads(raw) if raw.strip() else {}
        except ValueError:
            raw = {}
    raw = raw if isinstance(raw, dict) else {}
    lk = json.loads(json.dumps(DEFAULT_LOOK))
    for k, (lo, hi) in _RANGES.items():
        lk[k] = _num(raw.get(k), lk[k], lo, hi)
    lk["density"] = int(round(lk["density"]))
    if raw.get("where") == "arrows":                # an older look
        raw = dict(raw, where="forces")
    for k, opts in (("colour", COLOURS), ("blend", BLENDS), ("near", ("bright", "dark")), ("where", WHERE)):
        if raw.get(k) in opts:
            lk[k] = raw[k]
    t = raw.get("tint")
    if isinstance(t, (list, tuple)) and len(t) == 2:
        lk["tint"] = [_hex(t[0], lk["tint"][0]), _hex(t[1], lk["tint"][1])]
    cam = raw.get("camera") if isinstance(raw.get("camera"), dict) else {}
    for key in ("start", "end"):
        src = cam.get(key) if isinstance(cam.get(key), dict) else {}
        for k, (lo, hi) in _CAM_RANGES.items():
            lk["camera"][key][k] = _num(src.get(k), lk["camera"][key][k], lo, hi)
    if cam.get("ease") in EASES:
        lk["camera"]["ease"] = cam["ease"]
    lk["camera"]["shake"] = _num(cam.get("shake"), 0.0, 0, 2)
    path = []
    for key in cam.get("path") if isinstance(cam.get("path"), list) else []:
        if isinstance(key, (list, tuple)) and len(key) == 6:
            v = [_num(key[0], None, 0, 1)] + [_num(x, None, lo, hi) for x, (lo, hi) in zip(key[1:], _CAM_RANGES.values())]
            if None not in v:
                path.append(v)
    path.sort(key=lambda k: k[0])
    if len(path) > MAX_PATH:                       # evenly thinned, the first and the last key kept
        path = [path[round(k * (len(path) - 1) / (MAX_PATH - 1))] for k in range(MAX_PATH)]
    lk["camera"]["path"] = path if len(path) >= 2 else []
    lk["curves"] = curves_of(raw)
    reg = raw.get("regions")
    lk["regions"] = None if not isinstance(reg, dict) else {str(k): v for k, v in reg.items() if v in REGION_KINDS}
    snd = raw.get("sound") if isinstance(raw.get("sound"), dict) else {}
    for k, (lo, hi) in _SOUND_RANGES.items():
        lk["sound"][k] = _num(snd.get(k), lk["sound"][k], lo, hi)
    tr = raw.get("tracks") if isinstance(raw.get("tracks"), dict) else {}
    lk["tracks"]["marks"] = bool(tr.get("marks", lk["tracks"]["marks"]))
    if tr.get("lines") in LINES:
        lk["tracks"]["lines"] = tr["lines"]
    lk["tracks"]["distance"] = _num(tr.get("distance"), lk["tracks"]["distance"], 0.01, 1)
    lk["tracks"]["colour"] = _hex(tr.get("colour"), lk["tracks"]["colour"])
    lk["tracks"]["opacity"] = _num(tr.get("opacity"), lk["tracks"]["opacity"], 0, 1)
    lk["tracks"]["size"] = _num(tr.get("size"), lk["tracks"]["size"], 0.2, 5)
    return lk


def curves_of(raw) -> list:
    """The drawn curves of a raw look, checked: an older look's arrows ("forces": [x0, y0, x1, y1]) are wind curves
    first, then the curves; at most MAX_CURVES (the last ones), each 2..MAX_CURVE_PTS points in -0.5..1.5."""
    out = []
    for f in raw.get("forces") if isinstance(raw.get("forces"), list) else []:
        if isinstance(f, (list, tuple)) and len(f) == 4:
            out.append({"kind": "wind", "pts": [f[:2], f[2:]]})
    out += [c for c in raw.get("curves") if isinstance(c, dict)] if isinstance(raw.get("curves"), list) else []
    good = []
    for c in out:
        pts = [[_num(x, None, -0.5, 1.5), _num(y, None, -0.5, 1.5)] for x, y in
               (p for p in (c.get("pts") or [])[:MAX_CURVE_PTS] if isinstance(p, (list, tuple)) and len(p) == 2)]
        pts = [p for p in pts if None not in p]
        if c.get("kind") in CURVE_KINDS and len(pts) >= 2 and sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(pts, pts[1:])) > 1e-3:
            good.append({"kind": c["kind"], "pts": pts})
    return good[-MAX_CURVES:]


# ------------------------------------------------------------------------------------------ the forces
# The drawn curves and the regions make one force field over the picture (FIELD_H rows): per cell the current (vx,
# vy down, in heights a second), the curl (signed: how much of the small eddies and which way) and how near a force
# is (0..1: where they act). The flowing points read it at their place every step. The node computes the same field
# in kubakub_pointcloud.js from the same numbers (the regions' basis as 8 bits, made here) - keep both in step.
def field_size(aspect):
    return max(2, min(4 * FIELD_H, int(round(FIELD_H * aspect)))), FIELD_H


def _sm(e):
    e = np.clip(e, 0.0, 1.0)
    return e * e * (3 - 2 * e)


def _turn(rx, ry, a):
    """(vx, vy) round a middle, at (rx, ry) from it in its radii (slower inside one radius); a = signed strength, + =
    anticlockwise."""
    n = np.maximum(1.0, np.hypot(rx, ry))
    return a * ry / n, -a * rx / n


def _inside(px, py, X, Y):
    """Even-odd: which cells are inside the closed polygon px, py."""
    ins = np.zeros(X.shape, bool)
    j = len(px) - 1
    for i in range(len(px)):
        yi, yj = py[i], py[j]
        if yi != yj:
            ins ^= ((yi > Y) != (yj > Y)) & (X < (px[j] - px[i]) * (Y - yi) / (yj - yi) + px[i])
        j = i
    return ins


def curve_field(lk, aspect):
    """The drawn curves' share of the force field: (4, fh, fw) float64 (vx, vy, curl, near). wind = along the curve
    (in the way it was drawn), attract / repel = to / from it, curl + / - = the points turn round it (a loop: round
    its middle; + = anticlockwise) with small eddies; near = how near it is (inside a loop: 1). At strength 1: the flow
    scales the currents by the look's strength."""
    fw, fh = field_size(aspect)
    X, Y = np.meshgrid((np.arange(fw) + 0.5) / fw * aspect, (np.arange(fh) + 0.5) / fh)
    out = np.zeros((4, *X.shape))
    S, R = PULL, lk["reach"]
    for c in lk["curves"]:
        px = np.array([p[0] for p in c["pts"]], float) * aspect
        py = np.array([p[1] for p in c["pts"]], float)
        dd = np.full(X.shape, np.inf)
        qx, qy, tx, ty = (np.zeros(X.shape) for _ in range(4))
        for a in range(len(px) - 1):
            abx, aby = px[a + 1] - px[a], py[a + 1] - py[a]
            ln2 = abx * abx + aby * aby
            if ln2 < 1e-12:
                continue
            k = np.clip(((X - px[a]) * abx + (Y - py[a]) * aby) / ln2, 0.0, 1.0)
            sx, sy = px[a] + k * abx, py[a] + k * aby
            d = np.hypot(X - sx, Y - sy)
            near = d < dd
            ln = math.sqrt(ln2)
            dd, qx, qy = np.where(near, d, dd), np.where(near, sx, qx), np.where(near, sy, qy)
            tx, ty = np.where(near, abx / ln, tx), np.where(near, aby / ln, ty)
        w = _sm(1.0 - dd / R)
        loop = len(px) >= 4 and math.hypot(px[0] - px[-1], py[0] - py[-1]) < CLOSED
        if loop:
            w = np.where(_inside(px, py, X, Y), 1.0, w)
        safe = np.maximum(dd, 1e-9)
        ex, ey = (X - qx) / safe, (Y - qy) / safe               # from the curve out to the cell
        g = _sm(dd / 0.01)                                      # slower on the curve itself (no overshoot)
        kind = c["kind"]
        if kind == "wind":
            vx, vy = tx * S * w, ty * S * w
        elif kind == "attract":
            vx, vy = -ex * S * w * g, -ey * S * w * g
        elif kind == "repel":
            vx, vy = ex * S * w, ey * S * w
        else:
            s = 1.0 if kind == "curl+" else -1.0
            if loop:
                cx, cy = px.mean(), py.mean()
                rc = max(float(np.hypot(px - cx, py - cy).mean()), 1e-3)
                vx, vy = _turn((X - cx) / rc, (Y - cy) / rc, s * S * w)
            else:
                vx, vy = s * ey * S * w * g, s * -ex * S * w * g
            out[2] += s * w
        out[0] += vx
        out[1] += vy
        out[3] = np.maximum(out[3], w)
    return out


def region_basis(labels, groups, aspect):
    """What the regions' forces are made of, per group (up to MAX_GROUPS, the ones with the most regions): the
    labels (H, W) int, -1 = none, region ids; groups {name: [region ids]} -> {"names", "counts", "default" (the group
    with the most regions), "q": (G, fh, fw, 6) uint8} with per cell: the signed distance to the group (heights, < 0
    inside), the way to its nearest pixel (outside), the place in the nearest region (from its middle, in its radii).
    The labels are stretched over the picture."""
    import cv2
    fw, fh = field_size(aspect)
    W2, H2 = 2 * fw, 2 * fh
    lab = cv2.resize(np.asarray(labels, np.int32), (W2, H2), interpolation=cv2.INTER_NEAREST)
    there = set(np.unique(lab).tolist())                     # a region too small for the field's grid is not there
    items = sorted(((str(k), [int(i) for i in v if int(i) in there]) for k, v in (groups or {}).items()), key=lambda kv: -len(kv[1]))
    names, counts, qs = [], [], []
    px, py = np.meshgrid(np.arange(W2) + 0.5, np.arange(H2) + 0.5)
    top = int(lab.max()) + 1
    for name, ids in items:
        m = np.isin(lab, ids)
        if not m.any():
            continue
        dout, lbl = cv2.distanceTransformWithLabels((~m).astype(np.uint8), cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
        lut = np.zeros((int(lbl.max()) + 1, 2), np.int64)
        ys, xs = np.nonzero(m)
        lut[lbl[ys, xs]] = np.stack([xs, ys], -1)
        near = lut[lbl]                                          # the nearest pixel of the group
        nx, ny = near[..., 0] + 0.5 - px, near[..., 1] + 0.5 - py
        ln = np.maximum(np.hypot(nx, ny), 1e-9)
        nx, ny = np.where(m, 0.0, nx / ln), np.where(m, 0.0, ny / ln)
        din = cv2.distanceTransform(m.astype(np.uint8), cv2.DIST_L2, 5)
        sd = np.where(m, -din, dout) / H2
        rid = np.where(m, lab, lab[near[..., 1], near[..., 0]])
        cnt = np.bincount(lab[m], minlength=top).astype(float)
        cx = np.bincount(lab[m], weights=px[m], minlength=top) / np.maximum(cnt, 1)
        cy = np.bincount(lab[m], weights=py[m], minlength=top) / np.maximum(cnt, 1)
        rr = np.sqrt(np.maximum(cnt, 1) / math.pi)
        rx, ry = (px - cx[rid]) / rr[rid], (py - cy[rid]) / rr[rid]
        ch = [sd, nx, ny, rx, ry, np.zeros_like(sd)]
        q = []
        for v, (lo, hi) in zip(ch, BASIS):
            v = cv2.resize(np.clip(v, lo, hi).astype(np.float32), (fw, fh), interpolation=cv2.INTER_AREA)
            q.append(np.clip(np.round((v - lo) / (hi - lo) * 255), 0, 255).astype(np.uint8))
        names.append(name)
        counts.append(len(ids))
        qs.append(np.stack(q, -1))
        if len(names) >= MAX_GROUPS:
            break
    return {"names": names, "counts": counts, "default": names[0] if names else None,
            "q": np.stack(qs) if qs else np.zeros((0, fh, fw, 6), np.uint8)}


def region_kinds(lk, reg):
    """{group: kind} of the groups that act: the look's, else the default group attracts."""
    if not reg or not reg["names"]:
        return {}
    kinds = lk.get("regions")
    if kinds is None:
        kinds = {reg["default"]: "attract"}
    return {n: kinds[n] for n in reg["names"] if kinds.get(n, "off") != "off"}


def region_field(lk, reg, out):
    """The regions' share of the force field, added to out (4, fh, fw): attract = the points near a region stream into
    it and gather in it, repel = they leave it and keep away, curl + / - = they turn round each region's middle."""
    S, R = PULL, lk["reach"]
    for name, kind in region_kinds(lk, reg).items():
        q = reg["q"][reg["names"].index(name)].astype(np.float64) / 255.0
        sd, nx, ny, rx, ry = (BASIS[k][0] + q[..., k] * (BASIS[k][1] - BASIS[k][0]) for k in range(5))
        w = np.where(sd <= 0, 1.0, _sm(1.0 - sd / R))
        ins = np.clip(0.5 - sd / 0.008, 0.0, 1.0)
        if kind == "attract":
            vx, vy = (1 - ins) * nx * S * w - ins * rx * 0.5 * S, (1 - ins) * ny * S * w - ins * ry * 0.5 * S
        elif kind == "repel":
            vx, vy = -(1 - ins) * nx * S * w + ins * rx * S, -(1 - ins) * ny * S * w + ins * ry * S
        else:
            s = 1.0 if kind == "curl+" else -1.0
            vx, vy = _turn(rx, ry, s * S * w)
            out[2] += s * w
        out[0] += vx
        out[1] += vy
        out[3] = np.maximum(out[3], w)
    return out


def force_field(lk, aspect, reg=None, device="cpu"):
    """The whole force field (4, fh, fw) float32 on the device (at strength 1), or None when no curve and no region
    acts. Made once per render and handed to every points() call."""
    kinds = region_kinds(lk, reg)
    if lk["force"] <= 0 or (not lk["curves"] and not kinds):
        return None
    out = curve_field(lk, aspect)
    if kinds:
        region_field(lk, reg, out)
    return torch.from_numpy(out.astype(np.float32)).to(device)


# ------------------------------------------------------------------------------------------ the points
def grid_size(density, W, H):
    """(gw, gh): `density` points across the long side, the other side in the picture's proportion."""
    d = int(density)
    if W >= H:
        return d, max(2, int(round(d * H / W)))
    return max(2, int(round(d * W / H))), d


def pcg(x):
    """The pcg hash on uint32 (in int64, masked): the same numbers as the shader's pcg()."""
    s = (x * 747796405 + 2891336453) & M32
    w = (((s >> ((s >> 28) + 4)) ^ s) * 277803737) & M32
    return ((w >> 22) ^ w) & M32


def rnd(i, k):
    """The point's k-th random number 0..1 (k < 8)."""
    return (pcg(i * 8 + k).double() / 4294967295.0).float()


def grid_uv(i, gw, gh, jitter):
    gx, gy = (i % gw).float(), (i // gw).float()
    u = (gx + 0.5 + (rnd(i, 0) - 0.5) * jitter) / gw
    v = (gy + 0.5 + (rnd(i, 1) - 0.5) * jitter) / gh
    return u, v


def sample(img, u, v):
    """img (C, h, w) at the points (bilinear, edges held, like a WebGL texture) -> (N, C)."""
    g = torch.stack([u * 2 - 1, v * 2 - 1], -1).view(1, 1, -1, 2).to(img.dtype)
    return F.grid_sample(img[None], g, mode="bilinear", padding_mode="border", align_corners=False)[0, :, 0].T


def distance(d, depth):
    return 1.0 + depth * 2.0 * (1.0 - d)


def place(u, v, d, depth, aspect):
    dist = distance(d, depth)
    return torch.stack([(u - 0.5) * aspect * dist, (0.5 - v) * dist, -dist], -1), dist


def morph_at(t, T, K):
    """Pictures: (which picture, how far into the next one 0..1) at t seconds of T. Each picture holds, then the last
    45 % of its time travels to the next one; the last one stays."""
    if K <= 1 or T <= 0:
        return 0, 0.0
    seg = T / K
    k = min(int(t / seg), K - 1)
    if k >= K - 1:
        return K - 1, 0.0
    m = 0.45 * seg
    x = min(max((t - k * seg - (seg - m)) / m, 0.0), 1.0)
    return k, x * x * (3 - 2 * x)


def rank_perm(lum_a, lum_b):
    """Point i of picture a goes to point perm[i] of picture b: the one with the same brightness rank (ties by index)."""
    ra = torch.argsort(lum_a, stable=True)
    rb = torch.argsort(lum_b, stable=True)
    perm = torch.empty_like(ra)
    perm[ra] = rb
    return perm


def _colour_mode(c, lk):
    if lk["colour"] == "mono":
        return (c * torch.tensor(LUMA, device=c.device)).sum(-1, keepdim=True).expand(-1, 3)
    if lk["colour"] == "tint":
        lum = (c * torch.tensor(LUMA, device=c.device)).sum(-1, keepdim=True)
        a, b = (torch.tensor(hex01(h), device=c.device) for h in lk["tint"])
        return a + (b - a) * lum
    if lk["colour"] == "negative":
        return 1 - c
    return c


BANDS = 8                                           # the song's bands the depth hears (far = the lowest)
ROW = 6                                             # scan lines: one every this many points
BLINK = 4.0                                         # a blinking point's clock (a second, times the speed)
SILENT = {"level": 0.0, "low": 0.0, "mid": 0.0, "high": 0.0, "beat": 0.0, "bands": [0.0] * BANDS}


def spectrum(mono, hz, fps, n, bands=BANDS):
    """The song in `bands` bands (40 Hz .. 12 kHz, even in octaves) for n frames at fps: [n][bands] 0..1, every band
    measured against itself (its quiet = 0, its loud = 1), up at once and down over about 0.15 s."""
    mono = np.asarray(mono, np.float32)
    win = 2048
    w = np.hanning(win).astype(np.float32)
    edges = np.geomspace(40.0, min(12000.0, hz / 2 - 1), bands + 1)
    freqs = np.fft.rfftfreq(win, 1.0 / hz)
    idx = [np.nonzero((freqs >= edges[b]) & (freqs < edges[b + 1]))[0] for b in range(bands)]
    idx = [i if len(i) else np.array([int(np.argmin(np.abs(freqs - edges[b])))]) for b, i in enumerate(idx)]
    tail = max(win, int(round((n - 1) / fps * hz)) + win - len(mono))   # a render longer than the sound hears silence
    pad = np.concatenate([np.zeros(win // 2, np.float32), mono, np.zeros(tail, np.float32)])
    e = np.zeros((n, bands), np.float32)
    for f in range(n):
        a = int(round(f / fps * hz))
        mag = np.abs(np.fft.rfft(pad[a:a + win] * w))
        e[f] = [np.log1p(mag[i].mean()) for i in idx]
    lo, hi = np.percentile(e, 30, axis=0), np.percentile(e, 98, axis=0)
    e = np.clip((e - lo) / np.maximum(hi - lo, 1e-6), 0, 1)
    keep = math.exp(-1.0 / (0.15 * fps))
    for f in range(1, n):
        e[f] = np.maximum(e[f], e[f - 1] * keep)
    return [[round(float(x), 4) for x in row] for row in e]


def blink(i, t):
    """0..1 of point i in the blink step t (an integer tensor): the same numbers as the shader's blink()."""
    return (pcg((i * 1664525 + t * 1013904223) & M32).double() / 4294967295.0).float()
_GRIDS: dict = {}


def grid(gw, gh, jitter, device, stride=1):
    """What never changes of one point in every stride (index, grid place, its own random numbers and direction), made
    once and kept until forget(): points() runs every frame, the trails 13 times a frame."""
    key = (gw, gh, float(jitter), str(device), int(stride))
    g = _GRIDS.get(key)
    if g is None:
        if len(_GRIDS) >= 4:
            _GRIDS.clear()
        i = trail_points(gw * gh, stride, device)
        u, v = grid_uv(i, gw, gh, jitter)
        g = _GRIDS[key] = {"i": i, "u": u, "v": v, "r2": rnd(i, 2), "r3": rnd(i, 3), "r4": rnd(i, 4), "dir": _dir(i)}
    return g


def forget():
    """Let go of the kept grids (the node calls it after a render)."""
    _GRIDS.clear()


def points(lk, t, col_a, dep_a, aspect, col_b=None, dep_b=None, perm=None, s=0.0, field=None, device="cpu", snd=None,
           stride=1, tracked=None, now=None, forces=None):
    """Every point at t seconds: (P (N, 3), colour (N, 3), dist (N,), alive (N,) bool, fade (N,) 0..1). col (3, h, w),
    dep (1, h, w) 0 far .. 1 near; with a next picture, perm and s > 0 the points travel to it; field (1, h, w) pushes
    them out (and is where they flow with where = field); snd = what is heard now (SILENT keys, 0-1); stride = only
    every stride-th point (the trails); tracked = (M, 2) u, v of the tracked things now; now = the frame's own time when
    t is earlier (a trail): a flowing point of another life is then not alive; forces = force_field() (4, fh, fw) or
    None."""
    gw, gh = grid_size(lk["density"], col_a.shape[2], col_a.shape[1])
    g = grid(gw, gh, lk["jitter"], device, stride)
    # what every point does on its own is measured in point spacings (cell), what neighbours do together in the
    # picture's height: so the picture shimmers instead of falling apart, at any number of points and any size
    cell = 1.0 / gh
    i, u, v, dirs = g["i"], g["u"], g["v"], g["dir"]
    c = sample(col_a, u, v)[:, :3]
    d = sample(dep_a, u, v)[:, 0]
    def row(vv):                                    # scan lines: the place pulled onto its row (the colour stays its own)
        return vv + ((torch.floor(vv * gh / ROW) + 0.5) * ROW / gh - vv) * lk["rows"] if lk["rows"] > 0 else vv
    P, dist = place(u, row(v), d, lk["depth"], aspect)
    if col_b is not None and perm is not None and s > 0:
        j = perm.to(device) if stride == 1 else perm.to(device)[i]
        u2, v2 = grid_uv(j, gw, gh, lk["jitter"])
        c2, d2 = sample(col_b, u2, v2)[:, :3], sample(dep_b, u2, v2)[:, 0]
        P2, dist2 = place(u2, row(v2), d2, lk["depth"], aspect)
        P, c, d, dist = P + (P2 - P) * s, c + (c2 - c) * s, d + (d2 - d) * s, dist + (dist2 - dist) * s
        P = P + dirs * (lk["swarm"] * math.sin(math.pi * s) * 0.1 * dist)[:, None]
    tt = t * lk["speed"]
    r2, r3, r4 = g["r2"], g["r3"], g["r4"]
    tau = 6.2831853
    sn, so = {**SILENT, **(snd or {})}, lk["sound"]
    m = 1.0 + so["motion"] * sn["level"]           # the sound makes every motion bigger
    if tracked is not None:
        tracked = torch.as_tensor(tracked, dtype=u.dtype, device=u.device).reshape(-1, 2)[:MAX_TRACKED]
    alive = d >= lk["far_cut"]
    fade = torch.ones_like(d)
    ray = (P / P.norm(dim=-1, keepdim=True).clamp(min=1e-6)       # the point's line of sight (from the picture's camera)
           if lk["shiver"] > 0 or so["spectrum"] > 0 or so["sweep"] > 0 else None)
    if lk["shiver"] > 0:                            # each point trembles along its line of sight, its own rhythm
        fq = 2.0 + 5.0 * r4
        sv = 0.6 * torch.sin(tau * (fq * tt + r2)) + 0.4 * torch.sin(tau * (1.7 * fq * tt + r3))
        P = P + ray * (lk["shiver"] * 2.0 * cell * dist * (1 + so["shiver"] * sn["high"]) * sv)[:, None]
    if lk["dropout"] > 0:                           # a scanner losing its returns: each point blinks on its own clock
        drop = lk["dropout"] * (1 - so["fill"] * sn["level"])
        alive = alive & (blink(i, torch.floor(tt * BLINK + r2 * 7.0).long()) >= drop)
    if lk["drift"] > 0:
        P = P + torch.stack([torch.sin(0.8 * tt + tau * r2), torch.sin(0.7 * tt + tau * r3), torch.sin(0.6 * tt + tau * r4)], -1) * (lk["drift"] * 4.0 * cell * m * dist)[:, None]
    if lk["breathe"] > 0:
        P = P * (1 + lk["breathe"] * 0.025 * m * torch.sin(1.6 * tt - d * 6.0))[:, None]
    if lk["wind"] > 0:                              # neighbours together: no point shakes on its own
        wx = lk["wind"] * m * dist * 0.02 * torch.sin(1.3 * tt + v * 9.0 + u * 2.0)
        wy = lk["wind"] * m * dist * 0.007 * torch.sin(1.7 * tt + u * 7.0 + v * 3.0)
        P = P + torch.stack([wx, wy, torch.zeros_like(wx)], -1)
    if lk["curl"] > 0:                              # big loops, smooth over the picture
        ph = tau * (0.5 + 0.5 * torch.sin(5.3 * u + 3.1 * torch.sin(4.1 * v)))
        w = 0.9 * tt
        P = P + torch.stack([torch.cos(w + ph) + 0.45 * torch.cos(2.7 * w + 1.7 * ph),
                             torch.sin(w + ph) + 0.45 * torch.sin(2.7 * w + 1.7 * ph),
                             0.5 * torch.sin(0.6 * w + 2.0 * ph)], -1) * (lk["curl"] * 0.008 * m * dist)[:, None]
    if (lk["flow"] > 0 or forces is not None) and lk["flowing"] > 0:
        k = torch.nonzero(r3 < lk["flowing"]).squeeze(1)
        if len(k):
            dP, f, dead = flow_path(lk, tt, u[k], v[k], r4[k], aspect, field, tracked,
                                    None if now is None else now * lk["speed"], forces)
            P[k] = P[k] + dP * dist[k][:, None]
            fade[k] = f
            alive[k] = alive[k] & ~dead
    if field is not None and lk["push"] > 0:
        f = sample(field, u, v)[:, 0]
        P = P + dirs * (lk["push"] * f * 0.3 * dist * (0.4 + r2))[:, None]
    if so["swell"] > 0 and sn["low"] > 0:
        P = P * (1 + so["swell"] * 0.06 * sn["low"])
    if so["burst"] > 0 and sn["beat"] > 0:          # a short hit where the forces act: those parts pop out and kick along the swirls
        x, y = u * aspect, v
        b = so["burst"] * sn["beat"] ** 3 * zone(lk, x, y, field, tracked, aspect, forces)
        fx, fy, fz = swirl(lk, x, y, tt, 0.35)
        n = torch.sqrt(fx * fx + fy * fy + fz * fz) + 1e-6
        kick = torch.stack([fx, -fy, fz], -1) / n[:, None] * (0.6 + 0.4 * r2)[:, None]
        pop = torch.stack([(u - 0.5) * aspect, 0.5 - v, 0.5 * d], -1)
        P = P + (pop * 0.06 + kick * 0.05) * (b * dist)[:, None]
    bands = sn["bands"]
    if so["spectrum"] > 0 and max(bands) > 0:       # every slice of the depth hears its band: far = the bass, near = the highs
        x = (d.clamp(0, 1) * (BANDS - 1))
        b0 = torch.floor(x).long().clamp(0, BANDS - 2)
        bt = torch.tensor(bands, dtype=d.dtype, device=d.device)
        e = bt[b0] + (bt[b0 + 1] - bt[b0]) * (x - b0.float())
        P = P - ray * (so["spectrum"] * e * e * 0.05 * dist)[:, None]
        c = c + (1 - c.clamp(0, 1)) * (so["spectrum"] * e * e * 0.35).clamp(max=1.0)[:, None]
    if so["sweep"] > 0 and sn["beat"] > 0.05:       # on the beat a thin bright plane rushes from near to far
        w = (1.0 - (d - sn["beat"]).abs() / 0.035).clamp(min=0.0) * so["sweep"]
        c = c + (1 - c.clamp(0, 1)) * (w * 0.7).clamp(max=1.0)[:, None]
        P = P - ray * (w * 0.012 * dist)[:, None]
    c = _colour_mode(c.clamp(0, 1), lk)
    if lk["scan"] > 0:
        sc = 1.0 - ((tt * 0.25) % 1.0)
        w = (1.0 - (d - sc).abs() / 0.04).clamp(min=0.0) * lk["scan"]
        c = c + (1 - c) * (w * 0.8).clamp(max=1.0)[:, None]
        P = P + dirs * (w * 0.02 * dist)[:, None]
    if so["sparkle"] > 0 and sn["high"] > 0:        # the highs light up a few points
        k = ((r2 + r3) % 1.0 - (1.0 - so["sparkle"] * sn["high"] * 0.3)) / 0.02
        c = c + (1 - c) * k.clamp(0, 1)[:, None]
    return P, c, dist, alive, fade


def swirl(lk, x, y, tt, sp=None):
    """The swirls' velocity at picture places x (0..aspect), y (0..1, down) at the time tt, in picture heights a
    second (sp = the speed, by default the flow's): (vx, vy down, vz towards the camera). The curl of a stream
    function, so it never bunches the points up."""
    K = 6.2831853 / lk["swirl"]
    norm = sum(a * f for _ang, f, a, _w in WAVES)
    amp = sum(a for _ang, _f, a, _w in WAVES)
    sp = lk["flow"] * 0.35 if sp is None else sp
    vx, vy, vz = torch.zeros_like(x), torch.zeros_like(x), torch.zeros_like(x)
    for ang, f, a, w in WAVES:
        ca, sa = math.cos(ang), math.sin(ang)
        arg = (ca * x + sa * y) * (f * K) + w * tt + 3.7 * ang
        cs = torch.cos(arg) * (a * f / norm)
        vx = vx + cs * sa
        vy = vy - cs * ca
        vz = vz + torch.sin(arg) * (a / amp)
    return vx * sp, vy * sp, vz * (sp * 0.5)


def forces_at(forces, x, y, aspect):
    """The force field at picture places -> (N, 4): vx, vy, curl, near (bilinear, like the node's texture)."""
    return sample(forces, x / aspect, y)


def zone(lk, x, y, field, tracked, aspect, forces=None, near=None):
    """Where the flow and the beat act (0..1) at picture places: everywhere, the field's white, near the tracked
    things or near the curves and regions (within 'reach'; near = the field's near there when already read)."""
    wh = lk["where"]
    if wh == "field":
        return sample(field, x / aspect, y)[:, 0].clamp(0, 1) if field is not None else torch.zeros_like(x)
    if wh == "forces":
        if forces is None:
            return torch.zeros_like(x)
        return (forces_at(forces, x, y, aspect)[:, 3] if near is None else near).clamp(0, 1)
    if wh == "tracked":
        w = torch.zeros_like(x)
        for tu, tv in (tracked.tolist() if tracked is not None else []):
            e = (1.0 - torch.hypot(x - tu * aspect, y - tv) / lk["reach"]).clamp(0, 1)
            w = torch.maximum(w, e * e * (3 - 2 * e))
        return w
    return torch.ones_like(x)


def flow_path(lk, tt, u, v, ph, aspect, field, tracked, now=None, forces=None):
    """The flowing points: each lives 'life' seconds (its own phase ph 0..1) and is born at its place; over its age it
    takes FLOW_STEPS fixed steps through the swirls (where the zone lets it) and the force field, so the same
    point a little earlier lies on the same line (its trail is its way). -> (offset (N, 3) in picture heights, x right,
    y up, z near; fade 0..1: in at birth, out at the end of its life; dead: of another life than at `now`)."""
    L = lk["life"]
    cyc = tt / L + ph
    life = torch.floor(cyc)
    age = (cyc - life) * L
    born = tt - age
    h = L / FLOW_STEPS
    x0, y0 = u * aspect, v
    x, y, z = x0.clone(), y0.clone(), torch.zeros_like(u)
    swirls = lk["flow"] > 0
    curly = forces is not None and any(k.startswith("curl") for k in [c["kind"] for c in lk["curves"]] + list((lk.get("regions") or {}).values()))
    for k in range(FLOW_STEPS):
        dt = (age - k * h).clamp(0, h)
        zero = torch.zeros_like(u)
        mx, my, mz, near = zero, zero, zero, None
        if forces is not None:                      # the currents, and the curls' small eddies, of the curves and regions
            F = forces_at(forces, x, y, aspect)
            mx, my, near = F[:, 0] * lk["force"], F[:, 1] * lk["force"], F[:, 3]
            if curly:
                cx, cy, cz = swirl(lk, x * CURL_SCALE, y * CURL_SCALE, born + k * h, CURL_SPEED * lk["force"])
                mx, my, mz = mx + F[:, 2] * cx, my + F[:, 2] * cy, F[:, 2] * cz
        if swirls:
            vx, vy, vz = swirl(lk, x, y, born + k * h)
            w = zone(lk, x, y, field, tracked, aspect, forces, near)
            mx, my, mz = mx + vx * w, my + vy * w, mz + vz * w
        x, y, z = x + mx * dt, y + my * dt, z + mz * dt
    a = age / L
    near0 = forces_at(forces, x0, y0, aspect)[:, 3] if forces is not None else None
    w0 = zone(lk, x0, y0, field, tracked, aspect, forces, near0) if swirls else torch.zeros_like(u)
    if near0 is not None:
        w0 = torch.maximum(w0, near0)
    fade = 1.0 - w0.clamp(0, 1) * (1.0 - _smoothstep(0.0, 0.08, a) * (1.0 - _smoothstep(0.55, 1.0, a)))
    dead = torch.zeros_like(u, dtype=torch.bool) if now is None else torch.floor(now / L + ph) != life
    return torch.stack([x - x0, y0 - y, z], -1), fade, dead


def _dir(i):
    q = torch.stack([rnd(i, 5) - 0.5, rnd(i, 6) - 0.5, rnd(i, 7) - 0.5], -1)
    return q / (q.norm(dim=-1, keepdim=True) + 1e-6)


# ------------------------------------------------------------------------------------------ the camera
def ease(a, how):
    a = min(max(float(a), 0.0), 1.0)
    return 0.0 if how == "still" else a if how == "linear" else a * a * (3 - 2 * a)


def camera_at(lk, a, t=0.0):
    """The camera at 0..1 of the move (t seconds in): a recorded path, else start -> end with the ease; a handheld
    shake on top."""
    cam = lk["camera"]
    path = cam.get("path") or []
    if len(path) >= 2:
        a = min(max(float(a), 0.0), 1.0)
        j = 0
        while j < len(path) - 2 and path[j + 1][0] < a:
            j += 1
        p, q = path[j], path[j + 1]
        k = min(max((a - p[0]) / max(q[0] - p[0], 1e-9), 0.0), 1.0)
        out = {key: p[n + 1] + (q[n + 1] - p[n + 1]) * k for n, key in enumerate(_CAM_RANGES)}
    else:
        k = ease(a, cam["ease"])
        out = {key: cam["start"][key] + (cam["end"][key] - cam["start"][key]) * k for key in _CAM_RANGES}
    sh = cam.get("shake", 0.0)
    if sh > 0:
        out["yaw"] += sh * 1.5 * (math.sin(1.1 * t) + 0.5 * math.sin(2.7 * t + 1.3) + 0.25 * math.sin(6.1 * t + 0.7))
        out["pitch"] += sh * 1.0 * (math.sin(0.9 * t + 2.1) + 0.5 * math.sin(3.1 * t + 0.4) + 0.25 * math.sin(5.3 * t + 2.9))
        out["x"] += sh * 0.01 * (math.sin(1.7 * t + 0.3) + 0.5 * math.sin(4.3 * t + 1.9))
        out["y"] += sh * 0.01 * (math.sin(1.3 * t + 1.1) + 0.5 * math.sin(3.7 * t + 0.8))
    return out


def basis(cam, depth):
    """(position, right, up, forward) of a camera that orbits the middle of the cloud: yaw / pitch in degrees, dolly
    0..1 towards the middle, x / y a sideways shift. All zero = the picture's own view from the origin."""
    R = 1.0 + depth
    ya, pa = math.radians(cam["yaw"]), math.radians(cam["pitch"])

    def rot(x, y, z):
        y1, z1 = y * math.cos(pa) - z * math.sin(pa), y * math.sin(pa) + z * math.cos(pa)
        return np.array([x * math.cos(ya) + z1 * math.sin(ya), y1, -x * math.sin(ya) + z1 * math.cos(ya)])
    right, up, fwd = rot(1, 0, 0), rot(0, 1, 0), rot(0, 0, -1)
    c = np.array([0.0, 0.0, -R]) + rot(0, 0, R * (1 - cam["dolly"])) + right * cam["x"] * R + up * cam["y"] * R
    return c, right, up, fwd


def project(P, cam_basis, W, H):
    """World points -> (sx, sy, z) in output pixels (y down) and the depth along the view."""
    c, r, u, f = (torch.tensor(a, dtype=P.dtype, device=P.device) for a in cam_basis)
    q = P - c
    x, y, z = q @ r, q @ u, q @ f
    zs = z.clamp(min=1e-6)
    sx = (x / (zs * 0.5 * (W / H)) + 1) * 0.5 * W
    sy = (1 - y / (zs * 0.5)) * 0.5 * H
    return sx, sy, z


# ------------------------------------------------------------------------------------------ drawing
def trail_stride(lk):
    """Every k-th point draws a trail (k = 1 / trail_share), the same on the node."""
    return max(1, int(math.floor(1.0 / lk["trail_share"] + 0.5)))      # half up, like Math.round


def trail_points(n, stride, device="cpu"):
    """One point out of every `stride`, picked by a hash in its block (not every stride-th: that would be columns of
    the grid); stride 1 = all of them. The same on the node (kubakub_pointcloud_math.trailPoint)."""
    stride = int(stride)
    if stride <= 1:
        return torch.arange(0, n, device=device, dtype=torch.int64)
    b = torch.arange(0, (n + stride - 1) // stride, device=device, dtype=torch.int64)
    return (b * stride + torch.floor(rnd(b, 3) * stride).long()).clamp(max=n - 1)



TRAIL_CHUNK = 8_000_000                             # 1 pixel samples rasterised at a time (bounds the memory)


def trail_lines(lk, t, frame, cam_basis, W, H):
    """The trails as chunks of 1 pixel samples [(pixel index, colour (M, 3), weight (M,)) ...]: each chosen point's way
    over the last 'trails' seconds, TRAIL_STEPS straight pieces fading out to the oldest, rasterised one sample per
    pixel along the longer axis (like the node's WebGL lines); a piece shows when both its ends are alive and in front
    of the camera. frame = the keyword arguments of points() for this frame."""
    S, stride = TRAIL_STEPS, trail_stride(lk)
    xs, ys, oks, col, fade = [], [], [], None, None
    for k in range(S + 1):
        P, c, _dist, alive, f = points(lk, t - lk["trails"] * k / S, stride=stride, now=t, **frame)
        sx, sy, z = project(P, cam_basis, W, H)
        xs.append(sx)
        ys.append(sy)
        oks.append(alive & (z > NEAR_Z))
        if k == 0:
            col, fade = c, f
    X, Y, OK = torch.stack(xs), torch.stack(ys), torch.stack(oks)          # (S + 1, M)
    x0, y0, dx, dy = X[:-1].flatten(), Y[:-1].flatten(), (X[1:] - X[:-1]).flatten(), (Y[1:] - Y[:-1]).flatten()
    ln = torch.maximum(dx.abs(), dy.abs())
    n = torch.ceil(ln).clamp(1, 512).long() * ((OK[:-1] & OK[1:]).flatten() & (ln > 0.35))     # a point that holds draws no trail
    M = X.shape[1]
    w = lk["trail_opacity"] * min(lk["opacity"], 1.0) * (1.0 - torch.arange(S, device=n.device).float() / S)    # per piece
    ends = torch.cumsum(n, 0)
    total = int(ends[-1]) if len(ends) else 0
    chunks, a = [], 0
    while a < len(n) and total:                     # whole pieces, about TRAIL_CHUNK samples each
        b = max(a + 1, int(torch.searchsorted(ends, int(ends[a - 1] if a else 0) + TRAIL_CHUNK, right=True)))
        nn = n[a:b]
        seg = torch.repeat_interleave(torch.arange(a, b, device=n.device), nn)
        if len(seg):
            f = (torch.arange(len(seg), device=n.device) - (torch.cumsum(nn, 0) - nn)[seg - a]).float() / n[seg].float()
            px, py = torch.floor(x0[seg] + dx[seg] * f), torch.floor(y0[seg] + dy[seg] * f)
            inside = (px >= 0) & (px < W) & (py >= 0) & (py < H)
            seg = seg[inside]
            chunks.append(((py[inside] * W + px[inside]).long(), col[seg % M], w[seg // M] * fade[seg % M]))
        a = b
    return chunks or None


def splat(P, c, dist, alive, lk, cam_basis, W, H, gh, lines=None, size=None, fade=None):
    """The points as the node's WebGL draws them -> (image (H, W, 3), alpha (H, W)). glow = screen blend on black
    (1 - product of 1 - c*a), ink = multiply on white, solid = the nearest point wins. lines = trail_lines() (they
    blend like the points; on solid they lie over them). size = the point size now (the sound can change it); fade =
    each point's light 0..1 (a flowing point fades in and out)."""
    dev = P.device
    sx, sy, z = project(P, cam_basis, W, H)
    px = (lk["size"] if size is None else size) * (H / gh) * dist / z.clamp(min=1e-6)
    a = (lk["opacity"] * px * px).clamp(max=1.0)
    if fade is not None:
        a = a * fade
    s = px.clamp(1.0, MAX_POINT)
    keep = alive & (z > NEAR_Z) & (sx + s > 0) & (sx - s < W) & (sy + s > 0) & (sy - s < H)
    keep = keep & (a > 0.002)
    sx, sy, z, s, a, c, px = sx[keep], sy[keep], z[keep], s[keep], a[keep], c[keep], px[keep]
    blend = lk["blend"]
    # a point smaller than FINE pixels shares its light with the 4 pixels around it by where it falls (bilinear), so
    # very many small points read as a smooth fine picture instead of grain (the node draws them as a soft 2 px sprite)
    fine = (px < FINE) if blend != "solid" else torch.zeros_like(px, dtype=torch.bool)
    tiny = (sx[fine], sy[fine], a[fine], c[fine])
    big = ~fine
    sx, sy, z, s, a, c = sx[big], sy[big], z[big], s[big], a[big], c[big]
    lox, loy = torch.ceil(sx - s / 2 - 0.5), torch.ceil(sy - s / 2 - 0.5)
    nx = (torch.floor(sx + s / 2 - 0.5) - lox + 1).clamp(min=0)
    ny = (torch.floor(sy + s / 2 - 0.5) - loy + 1).clamp(min=0)
    span = torch.maximum(nx, ny)
    order = torch.argsort(span, descending=True)
    sx, sy, z, s, a, c, lox, loy, nx, ny, span = (t[order] for t in (sx, sy, z, s, a, c, lox, loy, nx, ny, span))
    top = int(span.max().item()) if len(span) else 0
    counts = (len(span) - torch.searchsorted(span.flip(0).contiguous(), torch.arange(top, device=dev, dtype=span.dtype), right=True)).tolist()
    acc = torch.zeros((3, H * W), device=dev)
    cov = torch.zeros(H * W, device=dev)
    zbuf = torch.full((H * W,), float("inf"), device=dev) if blend == "solid" else None
    taps = []

    def add(pix, cc, w):                            # glow: 1 - product of (1 - c a); ink: product of (1 - a (1 - c)) on white
        if blend == "glow":
            acc.index_add_(1, pix, torch.log1p(-(cc * w[:, None]).clamp(max=0.999)).T)
        else:
            acc.index_add_(1, pix, torch.log((1 - w[:, None] * (1 - cc)).clamp(min=1e-4)).T)
        cov.index_add_(0, pix, torch.log1p(-w.clamp(max=0.999)))
    tx, ty, ta, tc = tiny
    if len(tx):
        x0, y0 = torch.floor(tx - 0.5), torch.floor(ty - 0.5)
        fx, fy = tx - 0.5 - x0, ty - 0.5 - y0
        for ox, oy, wt in ((0, 0, (1 - fx) * (1 - fy)), (1, 0, fx * (1 - fy)), (0, 1, (1 - fx) * fy), (1, 1, fx * fy)):
            ix, iy = x0 + ox, y0 + oy
            ok = (ix >= 0) & (ix < W) & (iy >= 0) & (iy < H) & (wt > 0)
            add((iy * W + ix).long()[ok], tc[ok], (ta * wt)[ok])
    for oy in range(top):
        for ox in range(top):
            m = counts[max(ox, oy)]
            if m == 0:
                continue
            sel = (nx[:m] > ox) & (ny[:m] > oy)
            ix, iy = lox[:m][sel] + ox, loy[:m][sel] + oy
            inside = (ix >= 0) & (ix < W) & (iy >= 0) & (iy < H)
            idx_all = torch.nonzero(sel).squeeze(1)[inside]
            ix, iy = ix[inside], iy[inside]
            ss = s[idx_all]
            w = a[idx_all]
            pcx, pcy = (ix + 0.5 - (sx[idx_all] - ss / 2)) / ss, (iy + 0.5 - (sy[idx_all] - ss / 2)) / ss
            r = torch.hypot(pcx - 0.5, pcy - 0.5) * 2
            big = ss >= 2.5
            ok = ~big | (r <= 1.0)
            fall = torch.where(big, 1 - _smoothstep(0.6, 1.0, r), torch.ones_like(r))
            w = (w * fall)[ok]
            pix = (iy * W + ix).long()[ok]
            cc = c[idx_all][ok]
            if blend == "solid":
                zz = z[idx_all][ok]
                zbuf.scatter_reduce_(0, pix, zz, "amin")
                taps.append((pix, zz, cc))
            else:
                add(pix, cc, w)
    if lines is not None and blend != "solid":
        for chunk in lines:
            add(*chunk)
    if blend == "glow":
        img = 1 - torch.exp(acc)
        alpha = 1 - torch.exp(cov)
    elif blend == "ink":
        img = torch.exp(acc)
        alpha = 1 - torch.exp(cov)
    else:
        img = torch.zeros((3, H * W), device=dev)
        for pix, zz, cc in taps:
            win = zz <= zbuf[pix]
            img[:, pix[win]] = cc[win].T
        alpha = torch.isfinite(zbuf).float()
        if lines is not None:                       # the trails over the points, alpha over alpha
            keep_, sw, sc = torch.zeros(H * W, device=dev), torch.zeros(H * W, device=dev), torch.zeros((3, H * W), device=dev)
            for pix, cc, w in lines:
                keep_.index_add_(0, pix, torch.log1p(-w.clamp(max=0.999)))
                sw.index_add_(0, pix, w)
                sc.index_add_(1, pix, (cc * w[:, None]).T)
            la = 1 - torch.exp(keep_)
            img = img * (1 - la) + sc / sw.clamp(min=1e-6) * la
            alpha = torch.maximum(alpha, la)
    return img.T.reshape(H, W, 3).clamp(0, 1), alpha.reshape(H, W).clamp(0, 1)


def _smoothstep(e0, e1, x):
    t = ((x - e0) / (e1 - e0)).clamp(0, 1)
    return t * t * (3 - 2 * t)


# ------------------------------------------------------------------------------------------ tracks in the cloud
def track_points(tracks, frame, dep_at):
    """The tracked things seen at `frame` of the tracker's video: [(id, u, v, d)], d = the cloud's own depth there."""
    out = []
    W, H = float(tracks["W"]), float(tracks["H"])
    for tr in tracks["tracks"]:
        k = frame - tr["t0"]
        if k < 0 or k >= len(tr["box"]):
            continue
        seen = tr.get("seen")
        if seen and not seen[min(k, len(seen) - 1)]:
            continue
        b = tr["box"][k]
        u, v = (b[0] + b[2] / 2) / W, (b[1] + b[3] / 2) / H
        out.append((int(tr["id"]), round(u, 5), round(v, 5), round(float(dep_at(u, v)), 4)))
    return out


def track_edges(pts, lk, aspect):
    """Pairs of indices into pts, joined the way the look says (in picture space, like kubakub track draw)."""
    from . import trackdraw as td
    mode = lk["tracks"]["lines"]
    P = [[p[1] * 1000.0 * aspect, p[2] * 1000.0] for p in pts]
    if mode == "network":
        return [tuple(e) for e in td.delaunay_edges(P)]
    if mode == "tree":
        return [tuple(e) for e in td.mst_edges(P)]
    if mode == "within":
        r = lk["tracks"]["distance"] * 1000.0 * aspect
        return [(a, b) for a in range(len(P)) for b in range(a + 1, len(P)) if math.hypot(P[a][0] - P[b][0], P[a][1] - P[b][1]) < r]
    return []


def draw_tracks(img, pts, lk, cam_basis, aspect):
    """Lines between the tracked things and a ring on each, at their place in the cloud. img = torch (H, W, 3) on any
    device -> the same with the marks over it. cv2 draws only a one-channel mask; the colour is blended on the image's
    own device, inside the marks' box, so a big frame is not copied around."""
    import cv2
    if not pts:
        return img
    H, W = int(img.shape[0]), int(img.shape[1])
    tk_ = lk["tracks"]
    u, v, d = torch.tensor([p[1:4] for p in pts], dtype=torch.float64).T
    P, _ = place(u, v, d, lk["depth"], aspect)
    sx, sy, z = project(P, cam_basis, W, H)
    xy = [(float(a), float(b)) if zz > NEAR_Z else None for a, b, zz in zip(sx, sy, z)]
    m = np.zeros((H, W), np.uint8)
    unit = H / 1080.0
    lw = max(1, int(round(1.2 * unit)))
    sh = 4                                         # sub-pixel positions (cv2 shift)
    for a, b in track_edges(pts, lk, aspect):
        if xy[a] and xy[b]:
            cv2.line(m, tuple(int(round(q * 16)) for q in xy[a]), tuple(int(round(q * 16)) for q in xy[b]), 255, lw, cv2.LINE_AA, sh)
    if tk_["marks"]:
        rad = int(round(RING * tk_["size"] * unit * 16))
        for q in xy:
            if q:
                cv2.circle(m, tuple(int(round(v * 16)) for v in q), rad, 255, lw, cv2.LINE_AA, sh)
    ys, xs = np.nonzero(m.any(1))[0], np.nonzero(m.any(0))[0]
    if not len(ys):
        return img
    y0, y1, x0, x1 = int(ys[0]), int(ys[-1]) + 1, int(xs[0]), int(xs[-1]) + 1
    w = torch.from_numpy(m[y0:y1, x0:x1]).to(img.device).to(img.dtype)[..., None] * (tk_["opacity"] / 255.0)
    col = torch.tensor(hex01(tk_["colour"]), dtype=img.dtype, device=img.device)
    img = img.clone()
    img[y0:y1, x0:x1] = img[y0:y1, x0:x1] * (1 - w) + col * w
    return img


# ------------------------------------------------------------------------------------------ without inputs
def sample_picture(W=1280, H=720):
    """The built-in facade and its own depth (wall, deeper windows and doors, the cornice a little out, the sky far):
    what the node shows with nothing connected. -> (image (H, W, 3), depth (H, W) near = 1)."""
    from . import sample_facade as sf
    r = sf.render(W, H)
    kinds = {name: kind for name, _g, _b, kind in sf.layout(W, H)}
    d = np.full((H, W), 0.6, np.float32)
    d[r["silhouette"] < 0.5] = 0.0
    for name, (_g, m) in r["masks"].items():
        k = kinds.get(name, "wall")
        if k in ("glass", "door"):
            d[m > 0] = 0.45
        elif k == "trim":
            d[m > 0] = 0.68
    return r["image"].astype(np.float32), d


def depth_from_light(frames):
    """No depth connected: a soft relief from the brightness (bright = a little nearer), so the points still have
    some depth; a depth node gives the real one."""
    import cv2
    out = []
    for f in frames:
        lum = (np.asarray(f, np.float32)[..., :3] * np.array(LUMA, np.float32)).sum(-1)
        k = max(3, int(max(lum.shape) / 60) | 1)
        out.append(0.35 + 0.3 * cv2.GaussianBlur(lum, (k, k), 0))
    return np.stack(out).astype(np.float32)


# ------------------------------------------------------------------------------------------ PLY
def write_ply(path, P, c):
    """Binary little-endian PLY: x y z float, red green blue uchar."""
    P = P.detach().cpu().numpy().astype("<f4")
    c8 = (c.detach().cpu().clamp(0, 1).numpy() * 255 + 0.5).astype(np.uint8)
    rec = np.empty(len(P), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("r", "u1"), ("g", "u1"), ("b", "u1")])
    rec["x"], rec["y"], rec["z"] = P[:, 0], P[:, 1], P[:, 2]
    rec["r"], rec["g"], rec["b"] = c8[:, 0], c8[:, 1], c8[:, 2]
    head = ("ply\nformat binary_little_endian 1.0\ncomment kubakub point cloud\n"
            f"element vertex {len(P)}\nproperty float x\nproperty float y\nproperty float z\n"
            "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        f.write(head.encode("ascii"))
        f.write(rec.tobytes())
    return path


def read_ply_count(path):
    with open(path, "rb") as f:
        head = f.read(512).split(b"end_header")[0].decode("ascii", "replace")
    for line in head.splitlines():
        if line.startswith("element vertex"):
            return int(line.split()[-1])
    return 0


