"""
scene_view.py

Real distances and previz from a Blender export (blender_export.py). numpy + opencv, no ComfyUI imports
(tests/test_scene3d.py).

Measure (3D-2): per pixel of the projection view the distance to the projector, the angle at which the
projector ray hits the surface, the size of one matrix pixel on the building (mm), the relative
brightness a projector gives there (cos(incidence) / distance^2, 1 = the typical facade), and the offset
from the main wall; per region the medians of these plus area in m^2. The projector frame is intersected
with the main plane to give the real facade width / bottom edge for KUBA_Viewer.

Preview (3D-4): the model rendered from an audience spot (a second export with a 'view' camera); every
pixel of that view is projected into the projector camera. Where the projector reaches the point (depth
test against the projection view) the matrix colour lands there, elsewhere the building stays in
ambient light (projection shadow). The lookup maps are computed once, so video frames are one remap each.
"""

from __future__ import annotations

import cv2
import numpy as np

try:
    from . import scene_ids as si
except ImportError:          # tests: this folder is on sys.path
    import scene_ids as si


# --------------------------------------------------------------------------
# cameras
# --------------------------------------------------------------------------

class Camera:
    """Blender camera from scene.json: world <-> pixel (top-left origin, pixel centres at +0.5)."""

    def __init__(self, info):
        c = info["camera"]
        M = np.array(c["matrix_world"], float)
        self.R = M[:3, :3] / np.linalg.norm(M[:3, :3], axis=0)      # columns: right, up, back
        self.C = M[:3, 3].copy()
        self.P = np.array(c["projection"], float)
        self.Pinv = np.linalg.inv(self.P)
        self.W, self.H = int(info["width"]), int(info["height"])
        self.ortho = c.get("type") == "ORTHO"

    def project(self, pts):
        """World points (..., 3) -> (u, v, camera-space depth along the view axis)."""
        pc = (pts - self.C) @ self.R                    # camera space (x right, y up, -z forward)
        h = np.concatenate([pc, np.ones(pc.shape[:-1] + (1,))], -1) @ self.P.T
        w = np.where(np.abs(h[..., 3]) < 1e-12, 1e-12, h[..., 3])
        u = (h[..., 0] / w + 1) * 0.5 * self.W
        v = (1 - h[..., 1] / w) * 0.5 * self.H
        return u, v, -pc[..., 2]

    def ray(self, u, v):
        """Pixel -> (origin, unit direction) in world space."""
        x = np.asarray(u, float) / self.W * 2 - 1
        y = 1 - np.asarray(v, float) / self.H * 2
        pts = []
        for z in (-1.0, 1.0):
            h = np.stack([x, y, np.full_like(x, z), np.ones_like(x)], -1) @ self.Pinv.T
            pts.append((h[..., :3] / h[..., 3:4]) @ self.R.T + self.C)
        d = pts[1] - pts[0]
        return pts[0], d / np.linalg.norm(d, axis=-1, keepdims=True)

    def towards(self, pts):
        """(unit direction from the points back to the camera, distance) - parallel rays for ortho."""
        d = self.C - pts
        dist = np.linalg.norm(d, axis=-1)
        if self.ortho:
            back = self.R[:, 2]
            return np.broadcast_to(back, d.shape), d @ back
        return d / np.maximum(dist, 1e-9)[..., None], dist

    def pixel_size_m(self, dist):
        """Size of one pixel (m) on a surface facing the camera at distance dist."""
        fx = self.P[0, 0] * self.W / 2
        if self.ortho:
            return np.full_like(np.asarray(dist, float), 1.0 / fx)
        return np.asarray(dist, float) / fx


def plane_hit(origin, direction, pt, nrm):
    denom = direction @ nrm
    t = ((pt - origin) @ nrm) / np.where(np.abs(denom) < 1e-9, 1e-9, denom)
    return origin + direction * t[..., None]


# --------------------------------------------------------------------------
# measure
# --------------------------------------------------------------------------

def wall_frame(info, pt, nrm, ground_z):
    """The projector frame on the main plane: width / height (m), bottom edge above the ground, axes."""
    cam = Camera(info)
    W, H = cam.W, cam.H
    uv = np.array([[0, H / 2], [W, H / 2], [W / 2, 0], [W / 2, H], [W / 2, H / 2],
                   [0, 0], [W, 0], [W, H], [0, H]], float)
    o, d = cam.ray(uv[:, 0], uv[:, 1])
    p = plane_hit(o, d, pt, nrm)
    left, right, top, bottom, centre = p[:5]
    ex = right - left
    width = float(np.linalg.norm(ex))
    ex /= width
    return {"width_m": width, "height_m": float(np.linalg.norm(top - bottom)),
            "bottom_m": float(bottom[2] - ground_z), "top_m": float(top[2] - ground_z),
            "centre": centre, "left": left, "right": right, "ex": ex, "normal": nrm, "ground_z": float(ground_z),
            "corners": p[5:], "projector_distance_m": float(abs((cam.C - pt) @ nrm))}


def measure_maps(scene, pt, nrm):
    """Per-pixel maps of the projection view (0 / nan-free on background)."""
    info = scene["info"]
    cam = Camera(info)
    pos, nor, fid = scene["position"], scene["normal"], scene["faceid"]
    fg = fid > 0
    dirs, dist = cam.towards(pos)
    cos_inc = np.clip((nor * dirs).sum(-1), 0.0, 1.0)
    size_m = cam.pixel_size_m(dist)
    area = size_m ** 2 / np.maximum(cos_inc, 0.05)
    bright = cos_inc if cam.ortho else cos_inc / np.maximum(dist, 1e-6) ** 2
    ref = np.median(bright[fg & (cos_inc > 0.9)]) if (fg & (cos_inc > 0.9)).any() else (
        np.median(bright[fg]) if fg.any() else 1.0)
    offset = (pos - pt) @ nrm
    maps = {"distance_m": dist, "incidence_deg": np.degrees(np.arccos(cos_inc)), "pixel_mm": size_m * 1000,
            "stretch": 1 / np.maximum(cos_inc, 0.05), "area_m2": area, "brightness": bright / max(ref, 1e-12),
            "offset_m": offset}
    for k in maps:
        maps[k] = np.where(fg, maps[k], 0).astype(np.float32)
    return maps


def region_stats(labels, maps, viewer_pos=None, position=None):
    """{region id: {...}} medians per region (labels: -1 = none, same size as the maps)."""
    out = {}
    fg = labels >= 0
    ids = np.unique(labels[fg])
    lab = labels[fg]
    order = np.argsort(lab, kind="stable")
    lab_s = lab[order]
    starts = np.searchsorted(lab_s, ids)
    ends = np.searchsorted(lab_s, ids, side="right")
    cols = {k: v[fg][order] for k, v in maps.items()}
    vd = None
    if viewer_pos is not None and position is not None:
        vd = np.linalg.norm(position[fg][order] - np.asarray(viewer_pos, float), axis=-1)
    for i, a, b in zip(ids, starts, ends):
        sl = slice(a, b)
        e = {"pixels": int(b - a),
             "distance_m": round(float(np.median(cols["distance_m"][sl])), 3),
             "offset_m": round(float(np.median(cols["offset_m"][sl])), 3),
             "incidence_deg": round(float(np.median(cols["incidence_deg"][sl])), 1),
             "pixel_mm": round(float(np.median(cols["pixel_mm"][sl])), 2),
             "brightness": round(float(np.median(cols["brightness"][sl])), 3),
             "area_m2": round(float(cols["area_m2"][sl].sum()), 3)}
        if vd is not None:
            e["viewer_distance_m"] = round(float(np.median(vd[sl])), 2)
        out[int(i)] = e
    return out


def spot_position(frame, x_m, distance_m, eye_m):
    """Audience spot in world space: x along the wall from the frame centre, distance from the wall, eye height."""
    c = frame["centre"]
    p = c + frame["ex"] * x_m + frame["normal"] * distance_m
    p = p.copy()
    p[2] = frame["ground_z"] + eye_m
    return p


def parse_spots(text):
    """'x, distance, eye' per line (m; x from the frame centre along the wall) -> [(x, d, eye)]."""
    spots = []
    for raw in (text or "").replace(";", "\n").splitlines():
        line = raw.split("//", 1)[0].strip()
        if not line:
            continue
        v = [float(t) for t in line.replace(",", " ").split()]
        if len(v) == 2:
            v.append(1.7)
        if len(v) != 3:
            raise ValueError(f"spot '{raw}': need 'x, distance[, eye height]' in metres")
        spots.append(tuple(v))
    return spots or [(0.0, 15.0, 1.7)]


# --------------------------------------------------------------------------
# preview
# --------------------------------------------------------------------------

def reprojection(view_scene, proj_scene, tol_frac=0.01, tol_min_m=0.03):
    """
    For every pixel of an audience view: where it lands in the projector image (map_x, map_y for
    cv2.remap), whether the projector reaches it (lit), and the projector brightness there. Surroundings in the
    view (kubakub scene surroundings: face ids above the model's) are drawn like the model but never lit and never
    counted: 'model' = everything solid, 'building' = the model alone.
    """
    vcam = Camera(view_scene["info"])
    pcam = Camera(proj_scene["info"])
    pos, nor, fid = view_scene["position"], view_scene["normal"], view_scene["faceid"]
    fg = fid > 0
    faces = view_scene["info"].get("faces")                     # ids above the model's faces are its surroundings
    own = fg if faces is None else fg & (fid <= int(faces))
    u, v, _ = pcam.project(pos)
    dirs, dist = pcam.towards(pos)
    inside = own & (u >= 0) & (u < pcam.W) & (v >= 0) & (v < pcam.H)
    ui = np.clip(u.astype(np.int64), 0, pcam.W - 1)
    vi = np.clip(v.astype(np.int64), 0, pcam.H - 1)
    cached = proj_scene.get("_pdist")                  # once per projection view (walkthroughs reuse it),
    if cached is not None and cached[0] is proj_scene["position"]:     # tied to this exact position array
        pdist = cached[1]
    else:
        pdist = pcam.towards(proj_scene["position"])[1]
        proj_scene["_pdist"] = (proj_scene["position"], pdist)
    pfg = proj_scene["faceid"] > 0
    seen = pdist[vi, ui]
    cos_inc = np.clip((nor * dirs).sum(-1), 0, 1)
    # tolerance grows on grazing surfaces, where one projector pixel spans a long depth range
    tan = np.sqrt(np.maximum(1 - cos_inc ** 2, 0)) / np.maximum(cos_inc, 0.17)
    tol = np.maximum(tol_min_m, tol_frac * dist) + 2 * pcam.pixel_size_m(dist) * tan
    lit = inside & pfg[vi, ui] & (np.abs(seen - dist) <= tol) & (cos_inc > 0)
    bright = cos_inc if pcam.ortho else cos_inc / np.maximum(dist, 1e-6) ** 2
    ref = np.median(bright[lit & (cos_inc > 0.9)]) if (lit & (cos_inc > 0.9)).any() else (
        np.median(bright[lit]) if lit.any() else 1.0)
    return {"map_x": (u - 0.5).astype(np.float32), "map_y": (v - 0.5).astype(np.float32), "lit": lit,
            "model": fg, "building": own, "brightness": (bright / max(ref, 1e-12)).astype(np.float32),
            "shadow": own & ~lit & inside}


def render_preview(frame_rgb, rp, clay, ambient=0.12, gain=1.0, physical=1.0, background=None):
    """
    One matrix frame (H, W, 3 float 0..1, projector size) seen from the audience view. background
    (view size, already fitted) fills everything that is not the model.
    """
    src = np.ascontiguousarray(frame_rgb, np.float32)
    col = cv2.remap(src, rp["map_x"], rp["map_y"], cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    b = 1.0 + physical * (np.clip(rp["brightness"], 0, 1.5) - 1.0)
    model = rp["model"][..., None]
    out = clay * ambient * model
    if background is not None:
        out = np.where(model, out, background)
    lit = rp["lit"][..., None]
    return np.clip(np.where(lit, out + col * gain * b[..., None], out), 0, 1).astype(np.float32)


CLAY_GREY = 0.8             # the grey blender_export.py renders the clay picture in
CLAY_COLOR = "#f2f2f2"      # the previz default: almost white, like stone under work light
RELIGHT_CLAY_COLOR = "#ffffff"   # relights: the colour multiplies the 'clay' brightness, white = the clay as it was


def _clay_rgb(color):
    return np.array(_hex_rgb(str(color or CLAY_COLOR), _hex_rgb(CLAY_COLOR)), np.float32)


def tint_clay(clay, color):
    """The clay render (H, W, 3 float, grey) in another clay colour ('#rrggbb'): its base grey becomes that colour,
    the shading stays."""
    return np.clip(clay * (_clay_rgb(color) / CLAY_GREY), 0, 1).astype(np.float32)


SHADERS = ("clay", "wireframe", "clay + wireframe")


def wire_lines(view_scene, angle_deg=20.0, step_m=0.25, soft=0.6):
    """
    The edges of a view as a line drawing (H, W float 0..1), from the passes the view already has: a line where two
    neighbouring pixels see different faces, where the surface turns by more than angle_deg, or where it jumps by
    more than step_m in depth (one building in front of another). Surroundings share one face id, so their flat
    walls stay clean and only their corners draw.
    """
    fid, nor, pos = view_scene["faceid"], view_scene["normal"], view_scene["position"]
    cos = np.cos(np.radians(angle_deg))
    lines = np.zeros(fid.shape, bool)
    for a, b in (((slice(None), slice(0, -1)), (slice(None), slice(1, None))),
                 ((slice(0, -1), slice(None)), (slice(1, None), slice(None)))):
        fa, fb = fid[a], fid[b]
        both = (fa > 0) & (fb > 0)
        edge = fa != fb
        edge |= both & ((nor[a] * nor[b]).sum(-1) < cos)
        edge |= both & (np.abs(((pos[a] - pos[b]) * nor[a]).sum(-1)) > step_m)
        lines[a] |= edge & (fa > 0)
        lines[b] |= edge & (fb > 0) & (fa == 0)               # the outline belongs to the model's side
    out = lines.astype(np.float32)
    if soft > 0:
        out = (np.clip(cv2.GaussianBlur(out, (0, 0), soft) * 1.8, 0, 1) * (fid > 0)).astype(np.float32)
    return out


def look_picture(view_scene, clay, shader="clay", color=CLAY_COLOR):
    """
    What the unlit model looks like in a previz: 'clay' (the clay render in the clay colour), 'wireframe' (its edges
    as lines in that colour on black) or 'clay + wireframe' (the clay with dark edges).
    -> (picture H x W x 3 float, full: True when the picture is shown as it is instead of dimmed to 'ambient').
    """
    if shader == "wireframe":
        return wire_lines(view_scene)[..., None] * _clay_rgb(color), True
    pic = tint_clay(clay, color)
    if shader == "clay + wireframe":
        pic = pic * (1.0 - 0.7 * wire_lines(view_scene)[..., None])
    return pic, False


def clay_albedo(brightness, color):
    """Cycles albedo of the clay: the colour ('#rrggbb', as seen on screen -> linear light) x the brightness."""
    c = np.array(_hex_rgb(str(color or RELIGHT_CLAY_COLOR)), float)
    lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    return [round(float(v), 5) for v in lin * float(brightness)]


def fit_background(img, W, H):
    """Scale to cover W x H (keeps the aspect, crops the centre)."""
    h, w = img.shape[:2]
    s = max(W / w, H / h)
    r = cv2.resize(img, (max(W, int(round(w * s))), max(H, int(round(h * s)))), interpolation=cv2.INTER_AREA)
    y0, x0 = (r.shape[0] - H) // 2, (r.shape[1] - W) // 2
    return np.ascontiguousarray(r[y0:y0 + H, x0:x0 + W, :3], np.float32)


# --------------------------------------------------------------------------
# walkthrough camera path
# --------------------------------------------------------------------------

def parse_path(text):
    """
    Keyframes, one per line: 'x, distance, eye[, look_x, look_height]' (m). x / look_x along the wall from
    the frame centre, distance from the wall, eye and look height above the ground. The keys are spread
    evenly over the walk. Missing look = the frame centre.
    """
    keys = []
    for raw in (text or "").replace(";", "\n").splitlines():
        line = raw.split("//", 1)[0].strip()
        if not line:
            continue
        v = [float(t) for t in line.replace(",", " ").split()]
        if len(v) < 2 or len(v) > 5:
            raise ValueError(f"path key '{raw}': need 'x, distance[, eye[, look_x, look_height]]'")
        v += [1.7, np.nan, np.nan][len(v) - 2:]
        keys.append(v[:5])
    if not keys:
        raise ValueError("the walkthrough path needs at least one key")
    return np.array(keys, float)


def _catmull_rom(keys, t):
    """keys (K, D), t in [0, 1] -> (D,) on a uniform Catmull-Rom spline through the keys."""
    K = len(keys)
    if K == 1:
        return keys[0]
    f = t * (K - 1)
    i = min(int(f), K - 2)
    u = f - i
    p0, p1, p2, p3 = keys[max(i - 1, 0)], keys[i], keys[i + 1], keys[min(i + 2, K - 1)]
    return 0.5 * ((2 * p1) + (-p0 + p2) * u + (2 * p0 - 5 * p1 + 4 * p2 - p3) * u ** 2
                  + (-p0 + 3 * p1 - 3 * p2 + p3) * u ** 3)


def camera_path(frame, keys, n):
    """n (location, look_at) pairs in world space along the keyframes."""
    centre_h = float(frame["centre"][2] - frame["ground_z"])
    k = keys.copy()
    k[:, 3] = np.where(np.isnan(k[:, 3]), 0.0, k[:, 3])
    k[:, 4] = np.where(np.isnan(k[:, 4]), centre_h, k[:, 4])
    out = []
    for j in range(n):
        x, d, eye, lx, lh = _catmull_rom(k, j / max(n - 1, 1))
        loc = spot_position(frame, x, d, eye)
        look = frame["centre"] + frame["ex"] * lx
        look = look.copy()
        look[2] = frame["ground_z"] + lh
        out.append((loc, look))
    return out


def colorize(values, fg, lo=None, hi=None, invert=False):
    """Turbo colour map of a metric map (for the node outputs); returns (rgb 0..1, lo, hi)."""
    v = values[fg]
    if lo is None:
        lo, hi = (float(np.percentile(v, 1)), float(np.percentile(v, 99))) if v.size else (0.0, 1.0)
    t = np.clip((values - lo) / max(hi - lo, 1e-9), 0, 1)
    if invert:
        t = 1 - t
    rgb = cv2.applyColorMap((t * 255).astype(np.uint8), cv2.COLORMAP_TURBO)[..., ::-1] / 255.0
    return (rgb * fg[..., None]).astype(np.float32), lo, hi


def main_plane_and_ground(scene):
    pt, nrm = si.main_plane(scene)
    ground = float(scene["mesh"]["vert"][:, 2].min())
    return pt, nrm, ground


# --------------------------------------------------------------------------
# several projectors: who reaches what, soft-edge blend masks
# --------------------------------------------------------------------------

def _edge_weight(u, v, W, H, ramp_px):
    """0 at the border of a projector's picture, 1 from ramp_px inside (u, v in pixels of that picture)."""
    d = np.minimum(np.minimum(u, W - u), np.minimum(v, H - v))
    return np.clip(d / max(ramp_px, 1e-6), 0.0, 1.0)


def projector_blend(scenes, ramp=0.1, gamma=2.2):
    """
    Soft-edge blend masks for projectors that light the same model. scenes: the loaded projection views
    (scene_ids.load), one per projector. Where two or more projectors reach the same surface, each one's share falls
    off towards the border of its own picture (over ramp x its short side) and the shares add up to 1 in light;
    gamma turns that into the pixel value the projector needs (2.2 for video content, 1 = linear).
    -> one dict per projector: mask (H, W) float32, shared (H, W) bool, counts {model, shared, alone} in pixels,
       partners (how many of its pixels each other projector also reaches).
    """
    out = []
    for i, s in enumerate(scenes):
        cam = Camera(s["info"])
        fg = s["faceid"] > 0
        uu, vv = np.meshgrid(np.arange(cam.W) + 0.5, np.arange(cam.H) + 0.5)
        own = _edge_weight(uu, vv, cam.W, cam.H, ramp * min(cam.W, cam.H))
        total = own.copy()
        shared = np.zeros(fg.shape, bool)
        partners = []
        for j, o in enumerate(scenes):
            if j == i:
                partners.append(0)
                continue
            rp = reprojection(s, o)
            oc = Camera(o["info"])
            w = _edge_weight(rp["map_x"] + 0.5, rp["map_y"] + 0.5, oc.W, oc.H, ramp * min(oc.W, oc.H)) * rp["lit"]
            total += w
            shared |= rp["lit"]
            partners.append(int(rp["lit"].sum()))
        share = np.where(total > 1e-6, own / np.maximum(total, 1e-6), 1.0)
        share = np.where(shared, share, 1.0)                 # alone: the full picture, up to its border
        mask = np.power(np.clip(share, 0.0, 1.0), 1.0 / max(gamma, 1e-3)).astype(np.float32)
        out.append({"mask": mask, "shared": shared & fg, "partners": partners,
                    "counts": {"model": int(fg.sum()), "shared": int((shared & fg).sum()),
                               "alone": int((fg & ~shared).sum())}})
    return out


# --------------------------------------------------------------------------
# relight (3D-3): light rigs in facade coordinates, masks -> emissive faces
# --------------------------------------------------------------------------

LIGHT_TYPES = ("point", "area", "spot", "sun")
ENVIRONMENTS = ("none", "night", "city", "courtyard", "sunset", "sunrise", "forest", "studio", "interior", "file")


def _hex_rgb(tok, default=(1.0, 1.0, 1.0)):
    t = tok.strip().lstrip("#")
    if len(t) == 6:
        try:
            return tuple(int(t[k:k + 2], 16) / 255 for k in (0, 2, 4))
        except ValueError:
            pass
    return default


def parse_lights(text):
    """
    One light per line (metres; x along the wall from the frame centre, height above the ground,
    distance from the wall towards the audience):
        point x height distance watts [#rrggbb] [radius]
        area  x height distance watts [#rrggbb] [size]      faces the wall
        spot  x height distance watts [#rrggbb] [angle_deg]  aims at the wall
        sun   azimuth_deg elevation_deg strength [#rrggbb]   azimuth 0 = from the audience side
    '//' starts a comment.
    """
    out = []
    for n, raw in enumerate((text or "").replace(";", "\n").splitlines(), 1):
        line = raw.split("//", 1)[0].strip()
        if not line:
            continue
        tok = line.replace(",", " ").split()
        kind = tok[0].lower()
        if kind not in LIGHT_TYPES:
            raise ValueError(f"light line {n}: type must be one of {', '.join(LIGHT_TYPES)}")
        nums = [t for t in tok[1:] if not t.startswith("#")]
        color = next((_hex_rgb(t) for t in tok[1:] if t.startswith("#")), (1.0, 1.0, 1.0))
        try:
            vals = [float(v) for v in nums]
        except ValueError:
            raise ValueError(f"light line {n}: numbers expected, got '{line}'")
        need = 3 if kind == "sun" else 4
        if len(vals) < need:
            raise ValueError(f"light line {n}: {kind} needs {need} numbers")
        L = {"type": kind, "color": list(color)}
        if kind == "sun":
            L.update(azimuth=vals[0], elevation=vals[1], power=vals[2])
        else:
            L.update(x=vals[0], height=vals[1], distance=vals[2], power=vals[3])
            if len(vals) > 4:
                L["angle" if kind == "spot" else "size"] = vals[4]
        out.append(L)
    return out


RIG_DEFAULTS = {"environment": "night", "env_strength": 0.3, "env_rotation": 0.0, "exposure": 0.0, "clay": 0.7,
                "roughness": 0.8, "samples": 64, "background": "black", "resolution_scale": 1.0, "hdri_file": "",
                "view": "AgX"}


def projector_power(brightness, distance_m, albedo):
    """
    Spot power (W) so the matrix lands at its own brightness: a white pixel on a white-ish frontal wall at the
    projector distance comes out ~1 (checked in Cycles: 0.86-0.93 with the Standard view). Radiance of a diffuse
    wall under a point source: albedo * P / (4 pi^2 d^2).
    """
    return float(brightness) * 4 * np.pi ** 2 * max(float(distance_m), 0.1) ** 2 / max(float(albedo), 0.05)


def _f(d, k, default, lo, hi):
    v = d.get(k, default)
    try:
        v = float(v)
    except (TypeError, ValueError):
        v = float(default)
    return min(max(v, lo), hi) if np.isfinite(v) else float(default)


def rig_from_doc(raw):
    """
    A light rig as the director window stores it (light layer's 'light'):
        {"environment", "env_strength", "env_rotation", "exposure", "clay", "roughness", "samples", "background",
         "resolution_scale", "hdri_file",
         "lights": [{"type", "x", "height", "distance", "power", "color": "#rrggbb", "size", "angle",
                     "azimuth", "elevation", "on"}],
         "glow": [{"by": "layer:<id>" | region selector, "color": "#rrggbb", "strength"}],
         "projector": {"on", "source": "below" | "base" | "layer:<id>", "brightness",
                       "mode": "light" (cast from the camera) | "paint" (the model's colour, lit by the lamps)},
         "view": "AgX" | "Standard" | "Neutral" (keeps a projection's colours)}
    -> the same with defaults and clamped numbers; 'lights' in parse_lights() form (switched-off lights dropped).
    """
    raw = raw if isinstance(raw, dict) else {}
    env = raw.get("environment", RIG_DEFAULTS["environment"])
    rig = {"environment": env if env in ENVIRONMENTS else RIG_DEFAULTS["environment"],
           "env_strength": _f(raw, "env_strength", 0.3, 0, 50), "env_rotation": _f(raw, "env_rotation", 0, -360, 360),
           "exposure": _f(raw, "exposure", 0, -10, 10), "clay": _f(raw, "clay", 0.7, 0, 1),
           "roughness": _f(raw, "roughness", 0.8, 0, 1), "samples": int(_f(raw, "samples", 64, 1, 4096)),
           "background": raw.get("background") if raw.get("background") in ("black", "environment", "transparent") else "black",
           "resolution_scale": _f(raw, "resolution_scale", 1.0, 0.1, 1.0), "hdri_file": str(raw.get("hdri_file") or "")}
    lights = []
    for L in raw.get("lights") or []:
        if not isinstance(L, dict) or L.get("on") is False or L.get("type") not in LIGHT_TYPES:
            continue
        out = {"type": L["type"], "color": list(_hex_rgb(str(L.get("color") or "#ffffff")))}
        if L["type"] == "sun":
            out.update(azimuth=_f(L, "azimuth", 30, -360, 360), elevation=_f(L, "elevation", 35, -90, 90),
                       power=_f(L, "power", 3, 0, 1e4))
        else:
            out.update(x=_f(L, "x", 0, -1e4, 1e4), height=_f(L, "height", 5, -1e4, 1e4),
                       distance=_f(L, "distance", 5, -1e4, 1e4), power=_f(L, "power", 1000, 0, 1e7))
            if L["type"] == "spot":
                out["angle"] = _f(L, "angle", 45, 1, 180)
            out["size"] = _f(L, "size", 4 if L["type"] == "area" else 0.3, 0.001, 1000)
        lights.append(out)
    rig["lights"] = lights
    cc = str(raw.get("clay_color") or RELIGHT_CLAY_COLOR)
    rig["clay_color"] = cc if len(cc) == 7 and cc.startswith("#") else RELIGHT_CLAY_COLOR
    rig["view"] = raw.get("view") if raw.get("view") in ("AgX", "Standard", "Neutral") else "AgX"
    pj = raw.get("projector") if isinstance(raw.get("projector"), dict) else {}
    src = str(pj.get("source") or "below")
    rig["projector"] = {"on": bool(pj.get("on")), "brightness": _f(pj, "brightness", 1.0, 0, 100),
                        "mode": pj.get("mode") if pj.get("mode") in ("light", "paint") else "light",
                        "source": src if src in ("below", "base") or src.startswith("layer:") else "below"}
    rig["glow"] = [{"by": str(g.get("by") or ""), "color": list(_hex_rgb(str(g.get("color") or "#ffb060"))),
                    "strength": _f(g, "strength", 20, 0, 1e5)}
                   for g in raw.get("glow") or [] if isinstance(g, dict) and str(g.get("by") or "").strip()
                   and g.get("on") is not False]
    return rig


def lights_to_world(lights, frame):
    """Facade coordinates -> world location / target / direction for Blender (frame from wall_frame())."""
    up = np.array([0.0, 0.0, 1.0])
    out = []
    for L in lights:
        W_ = dict(L)
        if L["type"] == "sun":
            az, el = np.radians(L["azimuth"]), np.radians(L["elevation"])
            toward_light = np.cos(el) * (np.cos(az) * frame["normal"] + np.sin(az) * frame["ex"]) + np.sin(el) * up
            W_["direction"] = (-toward_light).tolist()          # the way the light travels
            W_["location"] = (frame["centre"] + toward_light * 50).tolist()
        else:
            wall = frame["centre"] + frame["ex"] * L["x"]
            wall = wall.copy()
            wall[2] = frame["ground_z"] + L["height"]
            loc = wall + frame["normal"] * L["distance"]
            W_["location"] = loc.tolist()
            W_["target"] = wall.tolist()
        out.append(W_)
    return out


def emissive_faces(faceid, mask, min_share=0.5):
    """Global face indices (0-based) whose visible pixels are covered by the mask to at least min_share."""
    if mask.shape != faceid.shape:
        mask = cv2.resize(mask.astype(np.float32), (faceid.shape[1], faceid.shape[0]), interpolation=cv2.INTER_LINEAR)
    fid = faceid.astype(np.int64)
    vis = fid > 0
    n = int(fid.max()) + 1 if vis.any() else 1
    total = np.bincount(fid[vis], minlength=n)
    hit = np.bincount(fid[vis & (mask > 0.5)], minlength=n)
    ok = (total > 0) & (hit >= min_share * np.maximum(total, 1))
    ok[0] = False
    return np.flatnonzero(ok) - 1
