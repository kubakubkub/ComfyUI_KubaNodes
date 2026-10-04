"""
pieces.py

Motion for the pieces of a facade model, in the spirit of MOPS for Houdini: the model's loose parts (stones, window
frames, cornice blocks ...) are pieces; operators (push out of the wall, move, rotate, scale, jitter) act on them,
each weighted by a falloff (a wave running across the facade, a pulse, a stagger in some order, noise). Everything is
evaluated here in numpy into one 4x4 matrix per piece and frame; Blender only moves the vertices and renders
(blender_export.relight, job part "pieces").

Coordinates: world metres (the scene's mesh.npz). Facade axes from the wall frame (scene_view.wall_frame): ex =
along the facade (left -> right), up = world Z, out = the wall normal (towards the audience).

No ComfyUI imports (tests/test_pieces.py).
"""

from __future__ import annotations

import numpy as np

AXES = ("left to right", "right to left", "bottom to top", "top to bottom", "from the centre", "towards the centre")
ORDERS = ("left to right", "right to left", "bottom to top", "top to bottom", "from the centre", "random")


# ------------------------------------------------------------------------------------------------
# pieces
# ------------------------------------------------------------------------------------------------

def piece_stats(mesh, labels):
    """labels: (F,) piece per face (-1 = not a piece). -> dict of (P, ...) arrays: centre (area-weighted face centres),
    normal (area-weighted, unit), bbox_min / bbox_max, area, size (bbox diagonal)."""
    P = int(labels.max()) + 1 if len(labels) and labels.max() >= 0 else 0
    ok = labels >= 0
    lab = labels[ok]
    area = np.asarray(mesh["area"], np.float64)[ok]
    cen = np.asarray(mesh["centre"], np.float64)[ok]
    nrm = np.asarray(mesh["normal"], np.float64)[ok]
    A = np.bincount(lab, area, P)
    C = np.stack([np.bincount(lab, area * cen[:, k], P) for k in range(3)], 1) / np.maximum(A, 1e-12)[:, None]
    N = np.stack([np.bincount(lab, area * nrm[:, k], P) for k in range(3)], 1)
    openness = np.linalg.norm(N, axis=1) / np.maximum(A, 1e-12)   # ~1 for a flat panel, ~0 for a closed stone
    N /= np.maximum(np.linalg.norm(N, axis=1), 1e-12)[:, None]
    # bbox from the vertices of each face
    lt = np.asarray(mesh["loop_total"], np.int64)
    face_of_loop = np.repeat(np.arange(len(lt)), lt)
    vpos = np.asarray(mesh["vert"], np.float64)[np.asarray(mesh["loop_vert"], np.int64)]
    lp = labels[face_of_loop]
    keep = lp >= 0
    bmin = np.full((P, 3), np.inf)
    bmax = np.full((P, 3), -np.inf)
    np.minimum.at(bmin, lp[keep], vpos[keep])
    np.maximum.at(bmax, lp[keep], vpos[keep])
    return {"centre": C, "normal": N, "openness": openness, "bbox_min": bmin, "bbox_max": bmax, "area": A,
            "size": np.linalg.norm(bmax - bmin, axis=1)}


def pieces_from_parts(parts, mesh, min_size_m=0.2, max_size_m=0.0, keep_largest=True, max_pieces=4000):
    """parts: (F,) loose-part label per face (scene_ids.loose_parts). Parts smaller than min_size_m (bbox diagonal),
    larger than max_size_m (0 = no limit) and, with keep_largest, the biggest part (the wall shell the rest sits on)
    stay still (-1). Returns (labels (F,) 0..P-1 or -1, stats)."""
    parts = np.asarray(parts, np.int64)
    st = piece_stats(mesh, parts)
    move = st["size"] >= float(min_size_m)
    if max_size_m and max_size_m > 0:
        move &= st["size"] <= float(max_size_m)
    if keep_largest and len(st["area"]):
        move[int(np.argmax(st["area"]))] = False
    ids = np.flatnonzero(move)
    if len(ids) > max_pieces:                                 # keep the biggest ones
        ids = ids[np.argsort(-st["area"][ids])[:max_pieces]]
        ids.sort()
    remap = np.full(len(move), -1, np.int64)
    remap[ids] = np.arange(len(ids))
    labels = remap[parts]
    return labels, piece_stats(mesh, labels)


def facade_coords(stats, frame):
    """Piece centres in facade metres: u from the left edge, v above the ground, d out of the wall."""
    C = stats["centre"]
    ex = np.asarray(frame["ex"], np.float64)
    nrm = np.asarray(frame["normal"], np.float64)
    left = np.asarray(frame["left"], np.float64)
    centre = np.asarray(frame["centre"], np.float64)
    u = (C - left) @ ex
    v = C[:, 2] - float(frame["ground_z"])
    d = (C - centre) @ nrm
    return u, v, d


def part_regions(faceid, region_labels, parts):
    """Region per loose part: the region most of the part's visible pixels fall in (-1 = none / not visible).
    faceid (H, W): 1 + face index, 0 = background; region_labels (H, W): region id or -1 (same size as faceid);
    parts (F,): loose part per face."""
    fid = np.asarray(faceid, np.int64).ravel()
    reg = np.asarray(region_labels, np.int64).ravel()
    ok = (fid > 0) & (reg >= 0)
    part = np.asarray(parts, np.int64)[fid[ok] - 1]
    reg = reg[ok]
    n_parts = int(np.max(parts)) + 1 if len(parts) else 0
    out = np.full(n_parts, -1, np.int64)
    if not len(part):
        return out
    key = part * (int(reg.max()) + 1) + reg
    uk, cnt = np.unique(key, return_counts=True)
    kp, kr = uk // (int(reg.max()) + 1), uk % (int(reg.max()) + 1)
    order = np.lexsort((-cnt, kp))                          # per part, the most counted region first
    kp, kr = kp[order], kr[order]
    first = np.r_[True, kp[1:] != kp[:-1]]
    out[kp[first]] = kr[first]
    return out


def pieces_from_regions(parts, part_region, mesh, wanted=None, group=True, keep_largest=True, min_size_m=0.0,
                        max_size_m=0.0, max_pieces=4000):
    """Pieces from region-tagged loose parts. wanted: region ids that move (None = every region). group: every
    region is one rigid piece (all its parts move together); else every part in a wanted region is its own piece
    (parts under min_size_m / over max_size_m stay still, at most max_pieces, the biggest first).
    -> (labels (F,), stats, region id per piece)."""
    parts = np.asarray(parts, np.int64)
    pr = np.asarray(part_region, np.int64).copy()
    if not group and (min_size_m > 0 or max_size_m > 0):
        size = piece_stats(mesh, parts)["size"]
        small = size < float(min_size_m)
        big = size > float(max_size_m) if max_size_m and max_size_m > 0 else np.zeros(len(size), bool)
        pr[(small | big)[:len(pr)]] = -1
    if keep_largest and len(pr):
        area = np.bincount(parts, np.asarray(mesh["area"], np.float64), len(pr))
        pr[int(np.argmax(area))] = -1                        # the wall they sit on
    if wanted is not None:
        pr[~np.isin(pr, np.asarray(list(wanted), np.int64))] = -1
    if group:
        regs = np.unique(pr[pr >= 0])
        remap = {int(r): k for k, r in enumerate(regs)}
        per_part = np.array([remap.get(int(r), -1) for r in pr], np.int64)
        piece_region = [int(r) for r in regs]
    else:
        movers = np.flatnonzero(pr >= 0)
        if len(movers) > max_pieces:                          # keep the biggest
            area = np.bincount(parts, np.asarray(mesh["area"], np.float64), len(pr))
            movers = np.sort(movers[np.argsort(-area[movers])[:max_pieces]])
            pr[np.setdiff1d(np.arange(len(pr)), movers)] = -1
        per_part = np.full(len(pr), -1, np.int64)
        per_part[movers] = np.arange(len(movers))
        piece_region = [int(pr[m]) for m in movers]
    labels = per_part[parts]
    return labels, piece_stats(mesh, labels), piece_region


def audience_view(frame, cam_pos, distance_m=15.0, offset_m=0.0, eye_m=1.7, lens_mm=24.0, look_height_m=None):
    """An audience camera in facade terms: `distance_m` in front of the wall (on the projector's side), `offset_m`
    left (-) / right (+) of the facade's middle, eyes `eye_m` above the ground, looking at the facade's middle
    (at `look_height_m`, default half its height). -> {"location", "look_at", "lens"} in world metres."""
    ex = np.asarray(frame["ex"], np.float64)
    n = np.asarray(frame["normal"], np.float64)
    centre = np.asarray(frame["centre"], np.float64)
    if (np.asarray(cam_pos, np.float64) - centre) @ n < 0:  # the audience stands where the projector is
        n = -n
    W = float(frame.get("width_m", 20.0))
    left = np.asarray(frame["left"], np.float64)
    gz = float(frame["ground_z"])
    base = left + ex * (W / 2.0 + float(offset_m))
    loc = base + n * float(distance_m)
    loc[2] = gz + float(eye_m)
    look = left + ex * (W / 2.0 + float(offset_m) * 0.3)
    look = look - n * ((look - centre) @ n)                  # on the wall plane
    look[2] = gz + (float(look_height_m) if look_height_m is not None else float(frame.get("top_m", 10.0)) / 2.0)
    return {"location": [float(v) for v in loc], "look_at": [float(v) for v in look], "lens": float(lens_mm)}


# ------------------------------------------------------------------------------------------------
# falloffs: weight 0..1 per piece at time t (noise: -1..1)
# ------------------------------------------------------------------------------------------------

def _smooth(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


def _hash01(i, seed):
    """Deterministic per-piece random numbers in [0, 1) (integer hash, the same on every machine)."""
    x = (np.asarray(i, np.uint64) * np.uint64(2654435761) + np.uint64((int(seed) * 97 + 13) & 0xFFFFFFFF)) \
        & np.uint64(0xFFFFFFFF)
    x ^= x >> np.uint64(16)
    x = (x * np.uint64(0x45D9F3B)) & np.uint64(0xFFFFFFFF)
    x ^= x >> np.uint64(16)
    return (x & np.uint64(0xFFFFFF)).astype(np.float64) / float(0x1000000)


def _order(kind, u, v, W, H, seed, n):
    """0..1 per piece along an order (for the stagger) or the travel distance in metres (for the wave)."""
    cu, cv = W / 2.0, H / 2.0
    r = np.hypot(u - cu, v - cv)
    rmax = max(float(np.hypot(cu, cv)), 1e-6)
    return {"left to right": u / max(W, 1e-6), "right to left": (W - u) / max(W, 1e-6),
            "bottom to top": v / max(H, 1e-6), "top to bottom": (H - v) / max(H, 1e-6),
            "from the centre": r / rmax, "towards the centre": 1 - r / rmax,
            "random": _hash01(np.arange(n), seed)}[kind]


def falloff(f, t, u, v, W, H):
    """Evaluate a falloff dict for all pieces at time t (seconds). u, v: facade coords (metres); W, H: facade size."""
    n = len(u)
    typ = (f or {}).get("type", "all")
    if typ == "all":
        w = np.ones(n)
    elif typ in ("wave", "pulse"):
        kind = f.get("axis", "left to right")
        pos = _order(kind, u, v, W, H, int(f.get("seed", 1)), n)
        if kind in ("from the centre", "towards the centre"):
            span = float(np.hypot(W, H)) / 2.0                 # pos = r / rmax, rmax = half the diagonal
        elif kind == "random":
            span = W                                          # a random ripple: every piece its own place in line
        else:
            span = W if "right" in kind else H
        c = pos * span                                        # metres along the travel direction
        tt = t - float(f.get("start", 0.0))
        if tt < 0:                                            # nothing before the start, looped or not
            return np.zeros(n)
        loop = float(f.get("loop", 0.0))
        if loop > 0:
            tt = np.mod(tt, loop)
        front = float(f.get("speed", 5.0)) * tt
        width = max(float(f.get("width", 2.0)), 1e-3)
        if typ == "wave":                                      # the front passes and the pieces stay changed
            w = _smooth((front - c) / width)
        else:                                                  # a band of this width travels across
            w = _smooth(1.0 - np.abs(front - c) / width)
    elif typ == "stagger":
        o = _order(f.get("order", "left to right"), u, v, W, H, int(f.get("seed", 1)), n)
        if n > 1 and np.ptp(o) > 1e-9:                         # the first piece starts at 'start', the last after 'spread'
            o = (o - o.min()) / np.ptp(o)
        delay = float(f.get("start", 0.0)) + float(f.get("spread", 2.0)) * o
        w = _smooth((t - delay) / max(float(f.get("duration", 1.0)), 1e-3))
        back = float(f.get("hold", -1.0))                       # >= 0: back to rest after holding this long
        if back >= 0:
            w = w * (1 - _smooth((t - delay - float(f.get("duration", 1.0)) - back) / max(float(f.get("duration", 1.0)), 1e-3)))
    elif typ == "triggers":                                    # beats / bars / markers of the director timeline
        times = np.asarray(sorted(f.get("times") or []), np.float64)
        o = _order(f.get("order", "left to right"), u, v, W, H, int(f.get("seed", 1)), n)
        if n > 1 and np.ptp(o) > 1e-9:
            o = (o - o.min()) / np.ptp(o)
        local = t - float(f.get("spread", 0.0)) * o              # the kick reaches later pieces later
        w = np.zeros(n)
        if len(times):
            i = np.searchsorted(times, local, side="right") - 1
            has = i >= 0
            dt = np.where(has, local - times[np.maximum(i, 0)], -1.0)
            a = max(float(f.get("attack", 0.03)), 0.0)
            d = max(float(f.get("decay", 0.3)), 1e-3)
            rise = _smooth(dt / a) if a > 0 else np.ones(n)
            w = np.where(dt < 0, 0.0, np.where(dt < a, rise, np.exp(-(dt - a) / d)))
            # a kick still rising: the previous one may still be higher (decaying)
            j = i - 1
            prev = np.where(j >= 0, local - times[np.maximum(j, 0)], -1.0)
            pw = np.where(prev < a, 0.0, np.exp(-(prev - a) / d))
            w = np.maximum(w, np.where((dt < a) & (j >= 0), pw, 0.0))
    elif typ == "noise":
        seed = int(f.get("seed", 1))
        freq = float(f.get("freq", 0.5))
        x = t * freq + _hash01(np.arange(n), seed) * 100.0      # value noise over time, one phase per piece
        i0 = np.floor(x)
        a = _hash01(i0.astype(np.int64) + np.arange(n) * 7919, seed + 1) * 2 - 1
        b = _hash01(i0.astype(np.int64) + 1 + np.arange(n) * 7919, seed + 1) * 2 - 1
        w = a + (b - a) * _smooth(x - i0)
    else:
        raise ValueError(f"unknown falloff {typ!r}")
    if f and f.get("invert"):
        w = 1.0 - w
    return w * float((f or {}).get("amount", 1.0))


# ------------------------------------------------------------------------------------------------
# operators -> matrices
# ------------------------------------------------------------------------------------------------

def _rot(axis, deg):
    """(n, 3, 3) rotations about a unit axis by per-piece angles (degrees)."""
    a = np.radians(np.asarray(deg, np.float64))
    k = np.asarray(axis, np.float64)
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    s, c = np.sin(a)[:, None, None], np.cos(a)[:, None, None]
    return np.eye(3)[None] + s * K[None] + (1 - c) * (K @ K)[None]


def evaluate(stats, frame, ops, t, W=None, H=None):
    """All operators at time t -> (P, 4, 4) world matrices (about each piece's centre)."""
    C = stats["centre"]
    N = stats["normal"]
    P = len(C)
    u, v, _ = facade_coords(stats, frame)
    W = float(W if W is not None else frame.get("width_m", max(u.max(initial=1), 1)))
    H = float(H if H is not None else frame.get("top_m", max(v.max(initial=1), 1)))
    ex = np.asarray(frame["ex"], np.float64)
    up = np.array([0.0, 0.0, 1.0])
    out = np.asarray(frame["normal"], np.float64)
    # push direction: the piece's own facing where it has one (panels, reliefs), else out of the facade (closed stones)
    own = (stats.get("openness", np.ones(P)) > 0.3) & (N @ np.asarray(frame["normal"], np.float64) > -0.2)
    PD = np.where(own[:, None], N, np.asarray(frame["normal"], np.float64)[None])
    T = np.zeros((P, 3))
    R = np.repeat(np.eye(3)[None], P, 0)
    S = np.ones(P)
    for op in ops:
        w = falloff(op.get("falloff"), t, u, v, W, H)
        if op.get("push"):
            T += (w * float(op["push"]))[:, None] * PD
        mv = op.get("move") or (0, 0, 0)
        if any(mv):
            T += w[:, None] * (float(mv[0]) * ex + float(mv[1]) * up + float(mv[2]) * out)[None]
        jit = float(op.get("jitter", 0.0))
        if jit:
            seed = int(op.get("seed", 1))
            rnd = np.stack([_hash01(np.arange(P), seed + k) * 2 - 1 for k in range(3)], 1)
            T += (w * jit)[:, None] * rnd
        rot = op.get("rotate") or (0, 0, 0)                 # tilt (about ex), turn (about up), roll (about out)
        for axis, deg in ((ex, rot[0]), (up, rot[1]), (out, rot[2])):
            if deg:
                R = _rot(axis, w * float(deg)) @ R
        rr = float(op.get("random_rotate", 0.0))
        if rr:
            seed = int(op.get("seed", 1))
            for k, axis in enumerate((ex, up, out)):
                R = _rot(axis, w * rr * (_hash01(np.arange(P), seed + 10 + k) * 2 - 1)) @ R
        sc = op.get("scale")
        if sc is not None and float(sc) != 1.0:
            S *= np.maximum(1.0 + (float(sc) - 1.0) * w, 0.0)
    M = np.zeros((P, 4, 4))
    M[:, :3, :3] = R * S[:, None, None]
    M[:, :3, 3] = C + T - np.einsum("pij,pj->pi", M[:, :3, :3], C)
    M[:, 3, 3] = 1.0
    return M


def sequence(stats, frame, ops, duration, fps):
    """(frames, P, 4, 4) float32 for t = 0, 1/fps, ... < duration."""
    n = max(1, int(round(float(duration) * float(fps))))
    out = np.empty((n, len(stats["centre"]), 4, 4), np.float32)
    for k in range(n):
        out[k] = evaluate(stats, frame, ops, k / float(fps))
    return out


def apply_to_points(M, piece_of_point, pts):
    """What Blender does per frame: every point of piece k moved by M[k] (piece -1 stays). pts (n, 3)."""
    out = np.array(pts, np.float64, copy=True)
    m = piece_of_point >= 0
    Mk = M[piece_of_point[m]]
    out[m] = np.einsum("nij,nj->ni", Mk[:, :3, :3], pts[m]) + Mk[:, :3, 3]
    return out
