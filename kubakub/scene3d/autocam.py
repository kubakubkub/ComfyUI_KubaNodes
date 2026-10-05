"""
autocam.py

Where the projector stands when the file has no camera, or when kubakub projector says so: finds the side of the model
that is the facade and places a camera in front of it, in metres from the wall.

numpy only: imported by blender_export.py inside Blender and by tests/test_autocam.py. World axes as in Blender
(z up). All functions take the flattened mesh: vertices (N, 3), face normals (F, 3), face areas (F), face centres (F, 3).
"""

from __future__ import annotations

import math

import numpy as np

SIDES = {"front (-y)": (0.0, -1.0), "back (+y)": (0.0, 1.0), "left (-x)": (-1.0, 0.0), "right (+x)": (1.0, 0.0)}
SIDE_OPTIONS = ("auto",) + tuple(SIDES)
AIMS = ("lens shift", "tilt", "straight")
DEFAULTS = {"side": "auto", "distance_m": 0.0, "offset_m": 0.0, "height_m": 1.5, "throw_ratio": 0.0, "aim": "lens shift",
            "margin": 0.05, "turn_deg": 0.0, "name": ""}
AUTO_THROW = 1.5            # distance / picture width when neither a distance nor a throw ratio is given


def spec(d=None):
    """A projector dict with every key, numbers checked. Unknown keys are dropped."""
    out = dict(DEFAULTS)
    for k, v in (d or {}).items():
        if k in out and v is not None:
            out[k] = v
    for k in ("distance_m", "offset_m", "height_m", "throw_ratio", "margin", "turn_deg"):
        out[k] = float(out[k])
    if out["side"] not in SIDE_OPTIONS:
        raise ValueError(f"projector side '{out['side']}': one of {', '.join(SIDE_OPTIONS)}")
    if out["aim"] not in AIMS:
        raise ValueError(f"projector aim '{out['aim']}': one of {', '.join(AIMS)}")
    if out["distance_m"] < 0 or out["throw_ratio"] < 0:
        raise ValueError("projector: distance_m and throw_ratio cannot be negative")
    out["name"] = str(out["name"])
    return out


def facade_direction(normal, area, side="auto", vert=None):
    """
    Unit horizontal vector from the facade towards the audience. auto: of the directions much upright surface faces
    (by area, in 5 degree steps), the one the model is widest across (a facade is wide and shallow, so the sides of
    its stones do not count as a wall); the front and the back of a model are equally wide, there the one nearer to
    Blender's front view (-y) wins.
    """
    if side in SIDES:
        return np.array([*SIDES[side], 0.0])
    n = np.asarray(normal, np.float64)
    a = np.asarray(area, np.float64)
    upright = np.abs(n[:, 2]) < 0.5
    if not upright.any() or a[upright].sum() <= 0:
        return np.array([0.0, -1.0, 0.0])
    n, a = n[upright], a[upright]
    ang = np.arctan2(n[:, 1], n[:, 0])
    bins = 72
    k = np.floor((ang + math.pi) / (2 * math.pi) * bins + 0.5).astype(np.int64) % bins
    w = np.bincount(k, weights=a, minlength=bins)
    ws = w + np.roll(w, 1) + np.roll(w, -1)                  # a facade a little off an axis falls into two bins
    peaks = [i for i in range(bins) if ws[i] >= 0.25 * ws.max() and ws[i] >= ws[i - 1] and ws[i] >= ws[(i + 1) % bins]]

    def direction(i):
        near = np.minimum((k - i) % bins, (i - k) % bins) <= 1
        d = (n[near, :2] * a[near, None]).sum(0)
        return d / max(np.linalg.norm(d), 1e-12)

    dirs = [direction(i) for i in peaks]
    if vert is not None and len(vert):
        V = np.asarray(vert, np.float64)
        across = [float(np.ptp(V[:, 0] * -d[1] + V[:, 1] * d[0])) for d in dirs]
    else:
        across = [float(ws[i]) for i in peaks]
    top = max(across) or 1.0
    best = max(range(len(peaks)), key=lambda j: (round(across[j] / top, 1), round(-dirs[j][1], 3), -j))
    d = dirs[best]
    return np.array([d[0], d[1], 0.0])


def facade_frame(vert, normal, area, centre, side="auto"):
    """
    The facade as a frame: n (towards the audience), ex (to the right, seen from the audience), the wall's offset
    along n (the depth where most of the surface facing n lies), the model's extent across (u0, u1) and up (z0, z1),
    and how far it reaches towards the audience (front).
    """
    V = np.asarray(vert, np.float64)
    n = facade_direction(normal, area, side, V)
    ex = np.cross([0.0, 0.0, 1.0], n)
    ex /= np.linalg.norm(ex)
    nn, aa, cc = np.asarray(normal, np.float64), np.asarray(area, np.float64), np.asarray(centre, np.float64)
    facing = (nn @ n) > 0.9
    if facing.any() and aa[facing].sum() > 0:
        s = cc[facing] @ n
        order = np.argsort(s)
        cum = np.cumsum(aa[facing][order])
        wall = float(s[order][np.searchsorted(cum, cum[-1] / 2)])        # area-weighted median
    else:
        wall = float((V @ n).max())
    u = V @ ex
    return {"n": n, "ex": ex, "wall": wall, "u0": float(u.min()), "u1": float(u.max()),
            "z0": float(V[:, 2].min()), "z1": float(V[:, 2].max()), "front": float((V @ n).max())}


def _matrix(right, up, back, loc):
    m = np.eye(4)
    m[:3, 0], m[:3, 1], m[:3, 2], m[:3, 3] = right, up, back, loc
    return m


def _lens(distance, picture_w, W, H, sensor=36.0):
    """Lens (mm) for a picture picture_w metres wide at this distance; sensor fit AUTO = the long side is 36 mm."""
    long_side = picture_w if W >= H else picture_w * H / W
    return sensor * distance / max(long_side, 1e-9)


def front_camera(vert, normal, area, centre, W, H, side="auto", lens=36.0):
    """
    The camera for a file without one: straight in front of the facade, level with its middle, far enough to frame
    the whole model. -> {"matrix": 4x4 world matrix (Blender camera: looks along -z), "lens", "shift_x", "shift_y",
    "distance_m" from the wall, "note"}.
    """
    f = facade_frame(vert, normal, area, centre, side)
    n, ex = f["n"], f["ex"]
    width, height, depth = f["u1"] - f["u0"], f["z1"] - f["z0"], f["front"] - float((np.asarray(vert) @ n).min())
    tan_h = 18.0 / lens if W >= H else 18.0 / lens * W / H
    tan_v = tan_h * H / W
    dist = 1.1 * max(width / 2 / tan_h, height / 2 / tan_v) + depth / 2
    mid = f["front"] - depth / 2
    loc = ex * (f["u0"] + f["u1"]) / 2 + n * (mid + dist) + np.array([0.0, 0.0, (f["z0"] + f["z1"]) / 2])
    return {"matrix": _matrix(ex, [0.0, 0.0, 1.0], n, loc), "lens": lens, "shift_x": 0.0, "shift_y": 0.0,
            "distance_m": float(mid + dist - f["wall"]), "side": _side_name(n),
            "note": f"no camera in the file: front camera {dist:.1f} m in front of the model, looking at its {_side_name(n)} side"}


def _side_name(n):
    best = max(SIDES, key=lambda k: n[0] * SIDES[k][0] + n[1] * SIDES[k][1])
    exact = abs(n[0] * SIDES[best][0] + n[1] * SIDES[best][1]) > 0.999
    return best if exact else f"{best}, turned {math.degrees(math.atan2(n[1], n[0])):.0f} deg"


def projector_camera(vert, normal, area, centre, W, H, proj):
    """
    A projector placed like on site: distance_m from the wall, offset_m along it from the middle of the facade (+ = to
    the right, seen from the audience), height_m above the ground. throw_ratio = distance / picture width; 0 = the
    picture just covers the model (plus margin). distance_m 0 = as far as the throw ratio needs to cover the model.
    aim: 'lens shift' keeps the projector square to the wall and shifts the picture onto the model (verticals stay
    vertical), 'tilt' turns the projector to the middle of the model, 'straight' looks square at the wall without shift.
    turn_deg turns the whole setup around the building (a projector that stands at an angle to the facade).
    -> as front_camera, plus "picture_w_m", "picture_h_m" on the wall and "covers" (share of the model's box in the picture).
    """
    p = spec(proj)
    f = facade_frame(vert, normal, area, centre, p["side"])
    n, ex = f["n"], f["ex"]
    cu, cz = (f["u0"] + f["u1"]) / 2, (f["z0"] + f["z1"]) / 2
    bw, bh = f["u1"] - f["u0"], f["z1"] - f["z0"]
    fit_w = (1 + 2 * p["margin"]) * max(bw, bh * W / H)                   # the picture that just covers the model
    d, throw = p["distance_m"], p["throw_ratio"]
    if d <= 0:
        d = (throw or AUTO_THROW) * fit_w
    pic_w = d / throw if throw > 0 else fit_w
    if p["turn_deg"]:                                                     # walk around the middle of the wall
        t = math.radians(p["turn_deg"])
        n = n * math.cos(t) + ex * math.sin(t)
        ex = np.cross([0.0, 0.0, 1.0], n)
        ex /= np.linalg.norm(ex)
    wall_mid = f["ex"] * cu + f["n"] * f["wall"]                          # on the ground line, middle of the facade
    loc = wall_mid + ex * p["offset_m"] + n * d + np.array([0.0, 0.0, f["z0"] + p["height_m"]])
    target = wall_mid + np.array([0.0, 0.0, cz])
    sx = sy = 0.0
    reach = d
    if p["aim"] == "tilt":
        back = loc - target
        dist = float(np.linalg.norm(back))
        back /= dist
        right = np.cross([0.0, 0.0, 1.0], back)
        right /= np.linalg.norm(right)
        pic_w = dist / throw if throw > 0 else fit_w                      # the picture at the middle of the model
        reach = dist
        lens = _lens(dist, pic_w, W, H)
        m = _matrix(right, np.cross(back, right), back, loc)
    else:
        lens = _lens(d, pic_w, W, H)
        if p["aim"] == "lens shift":
            long_side = pic_w if W >= H else pic_w * H / W
            sx = float((target - loc) @ ex) / long_side
            sy = float(target[2] - loc[2]) / long_side
        m = _matrix(ex, [0.0, 0.0, 1.0], n, loc)
    pic_h = pic_w * H / W
    covers = min(1.0, pic_w / max(bw, 1e-9)) * min(1.0, pic_h / max(bh, 1e-9)) if p["aim"] != "straight" else None
    note = (f"projector{' ' + p['name'] if p['name'] else ''}: {d:.1f} m from the wall, {p['offset_m']:+.1f} m along it, "
            f"{p['height_m']:.1f} m up, {p['aim']}, throw {reach / pic_w:.2f}, picture {pic_w:.1f} x {pic_h:.1f} m on the wall, "
            f"lens {lens:.1f} mm, looking at the {_side_name(f['n'])} side")
    if covers is not None and covers < 0.98:
        note += f"; the picture covers about {covers * 100:.0f} % of the model ({bw:.1f} x {bh:.1f} m)"
    return {"matrix": m, "lens": float(lens), "shift_x": sx, "shift_y": sy, "distance_m": float(d),
            "picture_w_m": float(pic_w), "picture_h_m": float(pic_h), "covers": covers, "side": _side_name(f["n"]), "note": note}
