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
LINES = ("off", "within", "network", "tree")
EASES = ("smooth", "linear", "still")
MAX_POINT = 64.0                                    # the largest point in pixels (WebGL draws at least that much)
NEAR_Z = 0.05
RING = 14.0                                         # a tracked thing's ring radius in pixels of a 1080 high picture (x its size)

DEFAULT_LOOK = {
    "density": 1280,            # points across the long side of the picture
    "jitter": 1.0,              # 0 = an exact grid, 1 = each point anywhere in its cell
    "depth": 1.0,               # how deep the picture goes: the farthest points are 1 + 2 x this away
    "far_cut": 0.0,             # points farther than this (0 far .. 1 near) are left out
    "size": 1.0,                # point size, in grid cells (1 = the points just touch from the start view)
    "opacity": 0.8,
    "colour": "true",           # true / mono / tint / negative
    "tint": ["#a187b7", "#f18a58"],   # tint: dark -> bright
    "blend": "glow",            # glow (light adds up, on black) / ink (on white) / solid (the nearest point wins)
    "near": "bright",           # which side of the depth picture is near (depth nodes differ)
    "speed": 0.6,
    "drift": 0.15, "breathe": 0.0, "wind": 0.0, "scan": 0.0,
    "curl": 0.0,                # every point runs on small curly loops (neighbours curl together)
    "push": 0.6,                # how far the field pushes points out
    "swarm": 0.5,               # pictures: how far the points fly out while they travel to the next picture
    "trails": 0.0,              # seconds of each point's own way drawn behind it as a fine line (0 = none)
    "trail_share": 0.1,         # which share of the points draws a trail
    "trail_opacity": 0.5,
    "forces": [],               # arrows drawn on the picture, [x0, y0, x1, y1] in 0..1 (y down): they push the points
    "force": 1.0,               # how far the arrows push
    "reach": 0.15,              # how far from an arrow it still pushes (0..1 of the picture's height)
    "sound": {"burst": 0.5, "swell": 0.3, "size": 0.0, "sparkle": 0.0, "motion": 0.4},
    "camera": {"start": {"yaw": 0.0, "pitch": 0.0, "dolly": 0.0, "x": 0.0, "y": 0.0},
               "end": {"yaw": 18.0, "pitch": -6.0, "dolly": 0.35, "x": 0.0, "y": 0.0},
               "ease": "smooth",
               "shake": 0.0,    # handheld: the camera shakes a little (0 = steady)
               "path": []},     # a recorded move: [[0..1 of the move, yaw, pitch, dolly, x, y] ...]; it beats start / end
    "tracks": {"marks": True, "lines": "network", "distance": 0.2, "colour": "#f18a58", "opacity": 0.85, "size": 1.0},
}
_RANGES = {"density": (128, 2048), "jitter": (0, 1), "depth": (0, 4), "far_cut": (0, 1), "size": (0.1, 8), "opacity": (0.01, 1),
           "speed": (0, 4), "drift": (0, 2), "breathe": (0, 2), "wind": (0, 2), "scan": (0, 2), "curl": (0, 2), "push": (0, 3),
           "swarm": (0, 2), "trails": (0, 3), "trail_share": (0.01, 1), "trail_opacity": (0, 1), "force": (0, 3), "reach": (0.02, 1)}
_SOUND_RANGES = {"burst": (0, 2), "swell": (0, 2), "size": (0, 2), "sparkle": (0, 1), "motion": (0, 2)}
MAX_FORCES = 8
MAX_PATH = 600                                      # keys of a recorded camera move
TRAIL_STEPS = 12                                    # segments of a trail
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
    for k, opts in (("colour", COLOURS), ("blend", BLENDS), ("near", ("bright", "dark"))):
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
    lk["camera"]["path"] = path[:MAX_PATH] if len(path) >= 2 else []
    forces = []
    for f in raw.get("forces") if isinstance(raw.get("forces"), list) else []:
        if isinstance(f, (list, tuple)) and len(f) == 4:
            v = [_num(x, None, -0.5, 1.5) for x in f]
            if None not in v and math.hypot(v[2] - v[0], v[3] - v[1]) > 1e-4:
                forces.append(v)
    lk["forces"] = forces[:MAX_FORCES]
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


SILENT = {"level": 0.0, "low": 0.0, "mid": 0.0, "high": 0.0, "beat": 0.0}


def points(lk, t, col_a, dep_a, aspect, col_b=None, dep_b=None, perm=None, s=0.0, field=None, device="cpu", snd=None,
           idx=None):
    """Every point at t seconds: (P (N, 3), colour (N, 3), dist (N,), alive (N,) bool). col (3, h, w), dep (1, h, w)
    0 far .. 1 near; with a next picture, perm and s > 0 the points travel to it; field (1, h, w) pushes them out;
    snd = what is heard now (SILENT keys, 0-1); idx = only these points (the trails)."""
    gw, gh = grid_size(lk["density"], col_a.shape[2], col_a.shape[1])
    i = torch.arange(gw * gh, device=device, dtype=torch.int64) if idx is None else idx.to(device)
    u, v = grid_uv(i, gw, gh, lk["jitter"])
    c = sample(col_a, u, v)[:, :3]
    d = sample(dep_a, u, v)[:, 0]
    P, dist = place(u, v, d, lk["depth"], aspect)
    if col_b is not None and perm is not None and s > 0:
        j = perm.to(device) if idx is None else perm.to(device)[i]
        u2, v2 = grid_uv(j, gw, gh, lk["jitter"])
        c2, d2 = sample(col_b, u2, v2)[:, :3], sample(dep_b, u2, v2)[:, 0]
        P2, dist2 = place(u2, v2, d2, lk["depth"], aspect)
        P, c, d, dist = P + (P2 - P) * s, c + (c2 - c) * s, d + (d2 - d) * s, dist + (dist2 - dist) * s
        P = P + _dir(i) * (lk["swarm"] * math.sin(math.pi * s) * 0.25 * dist)[:, None]
    tt = t * lk["speed"]
    r2, r3, r4 = rnd(i, 2), rnd(i, 3), rnd(i, 4)
    tau = 6.2831853
    sn, so = snd or SILENT, lk["sound"]
    m = 1.0 + so["motion"] * sn["level"]           # the sound makes every motion bigger
    if lk["drift"] > 0:
        P = P + torch.stack([torch.sin(0.8 * tt + tau * r2), torch.sin(0.7 * tt + tau * r3), torch.sin(0.6 * tt + tau * r4)], -1) * (lk["drift"] * 0.01 * m * dist)[:, None]
    if lk["breathe"] > 0:
        P = P * (1 + lk["breathe"] * 0.025 * m * torch.sin(1.6 * tt - d * 6.0))[:, None]
    if lk["wind"] > 0:
        wx = lk["wind"] * 0.03 * m * dist * (torch.sin(1.3 * tt + v * 9.0 + r2 * 3.0) + 0.5 * torch.sin(2.3 * tt + r3 * tau))
        wy = lk["wind"] * 0.01 * m * dist * torch.sin(1.7 * tt + u * 7.0 + r4 * tau)
        P = P + torch.stack([wx, wy, torch.zeros_like(wx)], -1)
    if lk["curl"] > 0:                              # loops on loops; the phase is smooth over the picture
        ph = tau * (0.5 + 0.5 * torch.sin(5.3 * u + 3.1 * torch.sin(4.1 * v))) + 1.3 * r3
        a, w = lk["curl"] * 0.012 * m * dist, 0.9 * tt
        P = P + torch.stack([torch.cos(w + ph) + 0.45 * torch.cos(2.7 * w + 1.7 * ph),
                             torch.sin(w + ph) + 0.45 * torch.sin(2.7 * w + 1.7 * ph),
                             0.5 * torch.sin(0.6 * w + 2.0 * ph)], -1) * a[:, None]
    if lk["forces"] and lk["force"] > 0:
        P = P + force_push(lk, u, v, dist, tt, r2, aspect, m)
    if field is not None and lk["push"] > 0:
        f = sample(field, u, v)[:, 0]
        P = P + _dir(i) * (lk["push"] * f * 0.3 * dist * (0.4 + r2))[:, None]
    if so["swell"] > 0 and sn["low"] > 0:
        P = P * (1 + so["swell"] * 0.06 * sn["low"])
    if so["burst"] > 0 and sn["beat"] > 0:
        P = P + _dir(i) * (so["burst"] * sn["beat"] * 0.08 * dist * (0.4 + r2))[:, None]
    c = _colour_mode(c.clamp(0, 1), lk)
    if lk["scan"] > 0:
        sc = 1.0 - ((tt * 0.25) % 1.0)
        w = (1.0 - (d - sc).abs() / 0.04).clamp(min=0.0) * lk["scan"]
        c = c + (1 - c) * (w * 0.8).clamp(max=1.0)[:, None]
        P = P + _dir(i) * (w * 0.02 * dist)[:, None]
    if so["sparkle"] > 0 and sn["high"] > 0:        # the highs light up a few points
        k = ((r2 + r3) % 1.0 - (1.0 - so["sparkle"] * sn["high"] * 0.3)) / 0.02
        c = c + (1 - c) * k.clamp(0, 1)[:, None]
    alive = d >= lk["far_cut"]
    return P, c, dist, alive


def force_push(lk, u, v, dist, tt, r2, aspect, m=1.0):
    """How far the drawn arrows push each point (N, 3): along the arrow, as far as it is long, fading out at 'reach'
    from it (in the picture, its height = 1), with a slow sway."""
    out = torch.zeros((len(u), 3), dtype=u.dtype, device=u.device)
    px, py = u * aspect, v
    for x0, y0, x1, y1 in lk["forces"]:
        ax, ay, bx, by = x0 * aspect, y0, x1 * aspect, y1
        dx, dy = bx - ax, by - ay
        k = (((px - ax) * dx + (py - ay) * dy) / max(dx * dx + dy * dy, 1e-9)).clamp(0, 1)
        e = torch.hypot(px - (ax + k * dx), py - (ay + k * dy))
        w = (1.0 - e / lk["reach"]).clamp(0, 1)
        w = w * w * (3 - 2 * w) * dist * (lk["force"] * 0.5 * m) * (1 + 0.3 * torch.sin(1.1 * tt + 6.2831853 * r2))
        out[:, 0] += w * dx
        out[:, 1] -= w * dy
    return out


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
def trail_index(lk, n):
    """The points that draw a trail: every k-th (k = 1 / trail_share), the same on the node."""
    return torch.arange(0, n, max(1, int(round(1.0 / lk["trail_share"]))), dtype=torch.int64)


def trail_lines(lk, t, frame, cam_basis, W, H):
    """The trails as 1 pixel samples (pixel index, colour (M, 3), weight (M,)): each chosen point's way over the last
    'trails' seconds, TRAIL_STEPS straight pieces fading out to the oldest, rasterised one sample per pixel along the
    longer axis (like the node's WebGL lines). frame = the keyword arguments of points() for this frame."""
    gw, gh = grid_size(lk["density"], frame["col_a"].shape[2], frame["col_a"].shape[1])
    idx = trail_index(lk, gw * gh)
    S = TRAIL_STEPS
    xs, ys, oks, col = [], [], [], None
    for k in range(S + 1):
        P, c, _dist, alive = points(lk, t - lk["trails"] * k / S, idx=idx, **frame)
        sx, sy, z = project(P, cam_basis, W, H)
        xs.append(sx)
        ys.append(sy)
        oks.append(alive & (z > NEAR_Z))
        col = c if col is None else col
    pix_all, c_all, w_all = [], [], []
    base = lk["trail_opacity"] * lk["opacity"]
    for k in range(S):
        x0, y0, x1, y1 = xs[k], ys[k], xs[k + 1], ys[k + 1]
        ok = oks[k] & oks[k + 1]
        if not bool(ok.any()):
            continue
        x0, y0, x1, y1, cc = x0[ok], y0[ok], x1[ok], y1[ok], col[ok]
        n = torch.ceil(torch.maximum((x1 - x0).abs(), (y1 - y0).abs())).clamp(1, 512).long()
        seg = torch.repeat_interleave(torch.arange(len(n), device=n.device), n)
        start = torch.cumsum(n, 0) - n
        f = (torch.arange(len(seg), device=n.device) - start[seg]).float() / n[seg].float()
        px = torch.floor(x0[seg] + (x1 - x0)[seg] * f)
        py = torch.floor(y0[seg] + (y1 - y0)[seg] * f)
        inside = (px >= 0) & (px < W) & (py >= 0) & (py < H)
        pix_all.append((py[inside] * W + px[inside]).long())
        c_all.append(cc[seg][inside])
        w_all.append(torch.full((int(inside.sum()),), base * (1.0 - k / S), device=n.device))
    if not pix_all:
        return None
    return torch.cat(pix_all), torch.cat(c_all), torch.cat(w_all)


def splat(P, c, dist, alive, lk, cam_basis, W, H, gh, lines=None, size=None):
    """The points as the node's WebGL draws them -> (image (H, W, 3), alpha (H, W)). glow = screen blend on black
    (1 - product of 1 - c*a), ink = multiply on white, solid = the nearest point wins. lines = trail_lines() (they
    blend like the points; on solid they lie over them). size = the point size now (the sound can change it)."""
    dev = P.device
    sx, sy, z = project(P, cam_basis, W, H)
    px = (lk["size"] if size is None else size) * (H / gh) * dist / z.clamp(min=1e-6)
    a = lk["opacity"] * (px * px).clamp(max=1.0)
    s = px.clamp(1.0, MAX_POINT)
    keep = alive & (z > NEAR_Z) & (sx + s > 0) & (sx - s < W) & (sy + s > 0) & (sy - s < H)
    sx, sy, z, s, a, c = sx[keep], sy[keep], z[keep], s[keep], a[keep], c[keep]
    lox, loy = torch.ceil(sx - s / 2 - 0.5), torch.ceil(sy - s / 2 - 0.5)
    nx = (torch.floor(sx + s / 2 - 0.5) - lox + 1).clamp(min=0)
    ny = (torch.floor(sy + s / 2 - 0.5) - loy + 1).clamp(min=0)
    span = torch.maximum(nx, ny)
    order = torch.argsort(span, descending=True)
    sx, sy, z, s, a, c, lox, loy, nx, ny, span = (t[order] for t in (sx, sy, z, s, a, c, lox, loy, nx, ny, span))
    top = int(span.max().item()) if len(span) else 0
    counts = [int((span > k).sum().item()) for k in range(top)]
    blend = lk["blend"]
    acc = torch.zeros((3, H * W), device=dev)
    cov = torch.zeros(H * W, device=dev)
    zbuf = torch.full((H * W,), float("inf"), device=dev) if blend == "solid" else None
    taps = []
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
            if blend == "glow":
                acc.index_add_(1, pix, torch.log1p(-(cc * w[:, None]).clamp(max=0.999)).T)
            elif blend == "ink":
                acc.index_add_(1, pix, torch.log((1 - w[:, None] * (1 - cc)).clamp(min=1e-4)).T)
            if blend == "solid":
                zz = z[idx_all][ok]
                zbuf.scatter_reduce_(0, pix, zz, "amin")
                taps.append((pix, zz, cc))
            else:
                cov.index_add_(0, pix, torch.log1p(-w.clamp(max=0.999)))
    if lines is not None and blend != "solid":
        pix, cc, w = lines
        if blend == "glow":
            acc.index_add_(1, pix, torch.log1p(-(cc * w[:, None]).clamp(max=0.999)).T)
        else:
            acc.index_add_(1, pix, torch.log((1 - w[:, None] * (1 - cc)).clamp(min=1e-4)).T)
        cov.index_add_(0, pix, torch.log1p(-w.clamp(max=0.999)))
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
            pix, cc, w = lines
            keep_ = torch.zeros(H * W, device=dev).index_add_(0, pix, torch.log1p(-w.clamp(max=0.999)))
            sw = torch.zeros(H * W, device=dev).index_add_(0, pix, w)
            sc = torch.zeros((3, H * W), device=dev).index_add_(1, pix, (cc * w[:, None]).T)
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


def draw_tracks(img, pts, lk, cam_basis, aspect, depth):
    """Lines between the tracked things and a ring on each, at their place in the cloud. img = torch (H, W, 3) on any
    device -> the same with the marks over it. cv2 draws only a one-channel mask; the colour is blended on the image's
    own device, inside the marks' box, so a big frame is not copied around."""
    import cv2
    if not pts:
        return img
    H, W = int(img.shape[0]), int(img.shape[1])
    tk_ = lk["tracks"]
    P = torch.tensor([[(p[1] - 0.5) * aspect * distance(p[3], depth), (0.5 - p[2]) * distance(p[3], depth), -distance(p[3], depth)] for p in pts], dtype=torch.float64)
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


