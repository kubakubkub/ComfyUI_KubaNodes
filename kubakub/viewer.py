"""
viewer.py

Viewer camera for frame in frame.
Pure numpy / opencv, no ComfyUI imports (tests/test_viewer.py).

The facade is the plane z = 0, the matrix maps onto it with one uniform scale
(facade_width_m / matrix width). The viewer stands at (x, eye height) at a
distance d in front. A point at depth D behind the facade (D > 0) appears on
the facade scaled towards the viewer's foot point F by

    s = d / (d + D)

and a point D in front (D < 0, a "parasite" sticking out) is scaled away
from F (s > 1). Everything here works in matrix pixels.
"""

from __future__ import annotations

import fnmatch
from dataclasses import asdict, dataclass

import cv2
import numpy as np

VIEWER_FORMAT = "kubakub.regions.viewer"


@dataclass
class Viewer:
    width_px: int
    height_px: int
    facade_width_m: float = 40.0
    bottom_m: float = 0.0          # height of the matrix's bottom edge above the ground
    x_m: float | None = None       # viewer position along the facade (None = centre)
    eye_m: float = 1.7             # eye height above the ground
    distance_m: float = 30.0       # distance from the facade plane

    @property
    def m_per_px(self) -> float:
        return self.facade_width_m / self.width_px

    @property
    def facade_height_m(self) -> float:
        return self.height_px * self.m_per_px

    def foot_px(self) -> tuple[float, float]:
        """The point of the facade straight ahead of the viewer's eye, in matrix pixels (may lie outside)."""
        x_m = self.facade_width_m / 2 if self.x_m is None else self.x_m
        fx = x_m / self.m_per_px
        fy = self.height_px - (self.eye_m - self.bottom_m) / self.m_per_px
        return fx, fy

    def scale(self, depth_m: float) -> float:
        """Apparent scale of a plane depth_m behind (> 0) or in front of (< 0) the facade."""
        if depth_m <= -self.distance_m:
            raise ValueError(f"depth {depth_m} m is at or behind the viewer ({self.distance_m} m away)")
        return self.distance_m / (self.distance_m + depth_m)

    def to_dict(self) -> dict:
        return {"format": VIEWER_FORMAT, **asdict(self)}

    @classmethod
    def from_dict(cls, d: dict) -> "Viewer":
        return cls(**{k: v for k, v in d.items() if k != "format"})


def scale_points(pts: np.ndarray, foot, s: float) -> np.ndarray:
    f = np.asarray(foot, np.float64)
    return f + (np.asarray(pts, np.float64) - f) * s


def frame_polygon(mask: np.ndarray, eps_frac: float = 0.01, hull: bool = True) -> np.ndarray | None:
    """
    Outline of a frame as a simplified polygon (N x 2, pixel coordinates). hull=True: the
    convex hull of all parts, so a window made of several glass panes is one opening (the
    mullions stay in front as occlusion); hull=False: the largest part only.
    """
    cnt, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnt:
        return None
    c = cv2.convexHull(np.concatenate(cnt)) if hull else max(cnt, key=cv2.contourArea)
    poly = cv2.approxPolyDP(c, max(1.0, eps_frac * cv2.arcLength(c, True)), True).reshape(-1, 2)
    if len(poly) < 3:
        return None
    if cv2.contourArea(poly.astype(np.float32)) < 0:        # make orientation consistent
        poly = poly[::-1]
    return poly.astype(np.float64)


def back_quad(poly: np.ndarray, foot, s: float) -> np.ndarray:
    """4 corners (tl, tr, br, bl) of the scaled frame's minimum-area rectangle: where media is corner-pinned."""
    q = scale_points(poly, foot, s).astype(np.float32)
    box = cv2.boxPoints(cv2.minAreaRect(q))
    # order: top-left, top-right, bottom-right, bottom-left
    c = box.mean(axis=0)
    ang = np.arctan2(box[:, 1] - c[1], box[:, 0] - c[0])
    box = box[np.argsort(ang)]                     # clockwise from the left-top-ish corner
    start = int(np.argmin(box.sum(axis=1)))
    return np.roll(box, -start, axis=0)


def parse_depth_rules(text: str):
    """
    'W_F1_* = 3', 'group:Windows = 2.5', 'tag:M_FLOOR_F0 = -1.5' (metres; < 0 sticks out).
    Later lines win. Returns ([(kind, pattern, depth)], notes).
    """
    rules, notes = [], []
    for n, raw in enumerate((text or "").splitlines(), 1):
        line = raw.split("//", 1)[0].strip()
        if not line:
            continue
        sel, eq, val = line.rpartition("=")
        try:
            depth = float(val)
        except ValueError:
            notes.append(f"depth rules line {n}: '{line}' needs '<selector> = <metres>'")
            continue
        sel = sel.strip()
        kind, _, pat = sel.partition(":") if sel.lower().startswith(("group:", "tag:", "name:")) else ("name", "", sel)
        if not eq or not pat:
            notes.append(f"depth rules line {n}: '{line}' needs '<selector> = <metres>'")
            continue
        rules.append((kind.lower(), pat.strip().lower(), depth))
    return rules, notes


def region_depths(table: dict, rules, default_m: float = 0.0) -> dict:
    """{region_id: depth_m} for every region a rule (or a non-zero default) gives a depth."""
    out = {}
    for r in table.get("regions", []):
        d = default_m
        for kind, pat, depth in rules:
            if kind == "name":
                hit = fnmatch.fnmatchcase(str(r.get("name", "")).lower(), pat)
            elif kind == "group":
                hit = fnmatch.fnmatchcase(str(r.get("group_id", "")).lower(), pat)
            else:
                hit = any(fnmatch.fnmatchcase(str(t).lower(), pat) for t in r.get("tags", []))
            if hit:
                d = depth
        if d != 0:
            out[int(r["region_id"])] = float(d)
    return out


def _shade(p_mid, q_mid):
    """Brightness of a side wall from the direction it recedes in (floors light, ceilings dark)."""
    v = np.asarray(q_mid, np.float64) - np.asarray(p_mid, np.float64)
    n = np.linalg.norm(v)
    if n < 1e-6:
        return 0.5
    v /= n
    # receding upwards (a floor seen from above) = bright, downwards (a ceiling) = dark, sideways in between
    return float(np.clip(0.5 - 0.3 * v[1] + 0.08 * v[0], 0.15, 0.9))


def render_frames(labels: np.ndarray, depths: dict, viewer: Viewer, background=None, hull: bool = True,
                  dim: float = 0.55):
    """
    Perspective guide for frame in frame. labels: int32 [H, W] (region ids, -1 none);
    depths: {region_id: metres}. Returns a dict:
      guide    [H, W, 3] float: background dimmed, frames drawn as tunnels / extrusions (shaded walls,
               back / front face, edges)
      area     bool [H, W]: pixels a generation may change (window for tunnels; frame + walls + front
               face for parasites)
      back     bool: back faces (tunnels) and front faces (parasites), where media / worlds go
      walls    bool: side walls
      frames   list of dicts: region_id, depth_m, scale, polygon, back_quad (for corner pin)
    """
    H, W = labels.shape
    fx, fy = viewer.foot_px()
    guide = np.empty((H, W, 3), np.float32)
    if background is None:
        guide[:] = 0.5
    else:
        bg = np.asarray(background, np.float32)
        guide[:] = bg[..., :3] * (1.0 - dim) + 0.2 * dim      # dim 0 = the matrix unchanged (generation input)
    area = np.zeros((H, W), bool)
    back = np.zeros((H, W), bool)
    walls = np.zeros((H, W), bool)
    edges = np.zeros((H, W), np.uint8)
    back_id = np.full((H, W), -1, np.int32)      # which frame each back / front face pixel belongs to
    wall_id = np.full((H, W), -1, np.int32)
    edge_id = np.full((H, W), -1, np.int32)
    shade_map = np.zeros((H, W), np.float32)     # wall brightness (0.15 .. 0.9), for tinting media walls
    area_id = np.full((H, W), -1, np.int32)      # which frame owns each changed pixel
    frames = []
    # far things first: deep tunnels, then shallow, then parasites (nearest last)
    for rid, depth in sorted(depths.items(), key=lambda kv: -kv[1]):
        win = labels == rid                # the visible part (glass panes); the opening is its hull
        poly = frame_polygon(win, hull=hull)
        if poly is None:
            continue
        s = viewer.scale(depth)
        q = scale_points(poly, (fx, fy), s)
        wall_layer = np.zeros((H, W), np.float32)
        wall_mask = np.zeros((H, W), np.uint8)
        n = len(poly)
        quads = []
        for i in range(n):
            p0, p1 = poly[i], poly[(i + 1) % n]
            q0, q1 = q[i], q[(i + 1) % n]
            quad = np.round(np.array([p0, p1, q1, q0])).astype(np.int32)
            quads.append((_shade((p0 + p1) / 2, (q0 + q1) / 2), quad))
        # walls farther from the viewer's foot point first
        for shade, quad in sorted(quads, key=lambda sq: -np.hypot(*(sq[1].mean(axis=0) - (fx, fy)))):
            tmp = np.zeros((H, W), np.uint8)
            cv2.fillPoly(tmp, [quad], 1)
            wall_layer[tmp > 0] = shade
            wall_mask |= tmp
        face = np.zeros((H, W), np.uint8)
        cv2.fillPoly(face, [np.round(q).astype(np.int32)], 1)
        face = face > 0
        wall_b = wall_mask > 0
        if depth > 0:                      # tunnel: the facade hides everything outside the window
            face &= win
            wall_b &= win & ~face
            region = win
            face_val = 0.22
        else:                              # parasite: drawn over the facade
            wall_b &= ~face
            region = win | wall_b | face
            # window pixels neither wall nor front face are hidden inside the solid: count them as wall
            rest = region & ~wall_b & ~face
            wall_layer[rest] = 0.5
            wall_b |= rest
            face_val = 0.85
        guide[wall_b] = wall_layer[wall_b][:, None]
        guide[face] = face_val
        shade_map[wall_b] = wall_layer[wall_b]
        wall_id[wall_b] = rid
        back_id[face] = rid
        # a nearer frame drawn later owns its pixels
        wall_id[face] = -1
        back_id[wall_b] = -1
        area |= region
        area_id[region] = rid
        back |= face
        walls |= wall_b
        e = np.zeros((H, W), np.uint8)
        cv2.polylines(e, [np.round(q).astype(np.int32)], True, 1, 2)
        for i in range(n):
            cv2.line(e, tuple(np.round(poly[i]).astype(int)), tuple(np.round(q[i]).astype(int)), 1, 2)
        e = (e > 0) & region
        edges |= e.astype(np.uint8)
        edge_id[e] = rid
        frames.append({"region_id": int(rid), "depth_m": depth, "scale": round(s, 4),
                       "polygon": np.round(poly, 1).tolist(), "face": np.round(q, 1).tolist(),
                       "back_quad": np.round(back_quad(poly, (fx, fy), s), 1).tolist()})
    guide[edges > 0] = 1.0
    return {"guide": guide, "area": area, "back": back, "walls": walls, "frames": frames,
            "foot": (fx, fy), "back_id": back_id, "wall_id": wall_id, "edge_id": edge_id, "shade": shade_map,
            "area_id": area_id}


def place_media(canvas: np.ndarray, media: np.ndarray, quad, clip: np.ndarray) -> np.ndarray:
    """Corner-pin media (cover-fit to the quad's aspect) into quad, only where clip is True."""
    quad = np.asarray(quad, np.float32)
    qw = float(np.linalg.norm(quad[1] - quad[0]))
    qh = float(np.linalg.norm(quad[3] - quad[0]))
    mh, mw = media.shape[:2]
    # crop the media to the quad's aspect (cover), centred
    target = qw / max(qh, 1e-6)
    if mw / mh > target:
        cw = int(round(mh * target))
        x0 = (mw - cw) // 2
        media = media[:, x0:x0 + cw]
    else:
        ch = int(round(mw / target))
        y0 = (mh - ch) // 2
        media = media[y0:y0 + ch]
    mh, mw = media.shape[:2]
    src = np.float32([[0, 0], [mw, 0], [mw, mh], [0, mh]])
    M = cv2.getPerspectiveTransform(src, quad)
    H, W = canvas.shape[:2]
    warped = cv2.warpPerspective(np.asarray(media, np.float32), M, (W, H), flags=cv2.INTER_LINEAR)
    cover = cv2.warpPerspective(np.ones((mh, mw), np.float32), M, (W, H)) > 0.5
    out = canvas.copy()
    sel = cover & clip
    out[sel] = warped[sel]
    return out


SOURCES = ("generate", "media", "world")


def pick_media(key: str, media: list, names: list, counter: list):
    """Media by name (file stem, case-insensitive), by index, or the next one in turn."""
    if not media:
        return None, "no media connected"
    k = (key or "").strip()
    low = [str(n).strip().lower() for n in names]
    if k:
        if k.lower() in low:
            return media[low.index(k.lower())], None
        if k.isdigit() and int(k) < len(media):
            return media[int(k)], None
        return None, f"media '{k}' not found (names: {', '.join(names) or 'none'})"
    m = media[counter[0] % len(media)]
    counter[0] += 1
    return m, None


def cast_shadow(poly, depth_m: float, viewer: Viewer, angle_deg: float = 60.0, length: float = 0.7):
    """
    Shadow of a parasite on the facade: the solid sticks out |depth_m| metres; directional light throws
    the outline of its front (same x, y as the frame, |depth_m| in front) length * |depth_m| metres along
    angle_deg in the facade plane (0 = to the right, 90 = down). Returns the shadow polygon (pixels).
    """
    off_px = abs(depth_m) * length / viewer.m_per_px
    a = np.deg2rad(angle_deg)
    shifted = np.asarray(poly, np.float64) + off_px * np.array([np.cos(a), np.sin(a)])
    return cv2.convexHull(np.concatenate([poly, shifted]).astype(np.float32)).reshape(-1, 2)


def compose(matrix, labels, specs: dict, viewer: Viewer, media=None, media_names=None, world=None,
            hull: bool = True, wall_level: float = 0.7, shadow_angle_deg: float = 60.0,
            shadow_length: float = 0.7, shadow_strength: float = 0.45):
    """
    The frame in frame composite on the matrix. specs: {region_id: {"depth": m, "source": generate |
    media | world, "media": key}}. generate frames get the perspective guide (for the Region Sampler),
    media frames the media corner-pinned onto the back wall, world frames the world image (it maps 1:1
    onto the facade through the viewer projection); walls of media / world frames are the guide's
    shading tinted with the content's mean colour. Returns (image, render dict, notes).
    """
    matrix = np.asarray(matrix, np.float32)[..., :3]
    depths = {rid: sp["depth"] for rid, sp in specs.items() if sp["depth"] != 0}
    H, W = labels.shape
    base = matrix
    shadow = np.zeros((H, W), bool)
    if shadow_strength > 0:
        for rid, d in depths.items():
            if d < 0:
                poly = frame_polygon(labels == rid, hull=hull)
                if poly is not None:
                    m = np.zeros((H, W), np.uint8)
                    cv2.fillPoly(m, [np.round(cast_shadow(poly, d, viewer, shadow_angle_deg,
                                                          shadow_length)).astype(np.int32)], 1)
                    shadow |= m > 0
        if shadow.any():
            soft = cv2.GaussianBlur(shadow.astype(np.float32), (0, 0), max(1.0, W / 1000))
            base = matrix * (1.0 - shadow_strength * soft[..., None])
    r = render_frames(labels, depths, viewer, background=base, hull=hull, dim=0.0)
    r["shadow"] = shadow & ~r["area"]
    out = r["guide"].copy()
    notes, counter = [], [0]
    names = list(media_names or [])
    media = list(media or [])
    world_img = None
    if world is not None:
        world_img = cv2.resize(np.asarray(world, np.float32)[..., :3], (W, H), interpolation=cv2.INTER_AREA)
    for f in r["frames"]:
        rid = f["region_id"]
        src = specs[rid]["source"]
        f["source"] = src
        if src == "generate":
            continue
        bk, wl, ed = r["back_id"] == rid, r["wall_id"] == rid, r["edge_id"] == rid
        if src == "media":
            img, err = pick_media(specs[rid].get("media", ""), media, names, counter)
            if img is None:
                notes.append(f"region {rid}: {err}; left as guide")
                continue
            out = place_media(out, img, f["back_quad"], bk)
            tint = np.asarray(img, np.float32)[..., :3].reshape(-1, 3).mean(axis=0)
        else:
            if world_img is None:
                notes.append(f"region {rid}: fif_source = world but no world image; left as guide")
                continue
            out[bk] = world_img[bk]
            sel = bk if bk.any() else (labels == rid)
            tint = world_img[sel].mean(axis=0)
        walls = wl | (ed & ~bk)
        out[walls] = np.clip(tint[None, :] * (r["shade"][walls][:, None] / 0.5) * wall_level, 0, 1)
    return out, r, notes


def extend_plan(plan: dict, table: dict, labels: np.ndarray, area_id: np.ndarray, parasite_ids):
    """
    Parasites reach beyond their region, and the Region Sampler only paints inside regions. For every
    parasite frame, add a region '<name>_out' covering its whole area (frame + walls + front face) with
    the frame's plan settings; the original region is set to keep (its pixels now belong to the new one).
    Returns (labels, table, plan) as new objects; the inputs are not changed.
    """
    import copy
    labels = labels.copy()
    table = copy.deepcopy(table)
    plan = copy.deepcopy(plan)
    by_id = {int(e["region_id"]): e for e in plan["regions"]}
    order = list(plan.get("order", []))
    for rid in parasite_ids:
        m = area_id == rid
        if not m.any():
            continue
        new_id = len(table["regions"])
        labels[m] = new_id
        ys, xs = np.nonzero(m)
        src = next(r for r in table["regions"] if int(r["region_id"]) == rid)
        name = f"{src.get('name', f'r{rid}')}_out"
        rec = {"region_id": new_id, "name": name, "group_id": src.get("group_id"),
               "bbox": [int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)],
               "area": int(m.sum()), "centroid": [round(float(xs.mean()), 1), round(float(ys.mean()), 1)],
               "tags": list(src.get("tags", [])) + [src.get("name", "")], "source": "frame_in_frame parasite"}
        table["regions"].append(rec)
        table.setdefault("groups", {}).setdefault(rec["group_id"], []).append(new_id)
        e = dict(by_id[rid])
        e.update({"region_id": new_id, "name": name, "bbox": rec["bbox"], "area": rec["area"], "tags": rec["tags"],
                  "fif_depth_m": 0.0})           # drawn already; the sampler just paints it
        plan["regions"].append(e)
        by_id[rid]["strategy"] = "keep"
        order = [new_id if o == rid else o for o in order] if rid in order else order + [new_id]
        # keep remaining pixels of the original region (none left if the parasite covers it all)
        rest = labels == rid
        by_id[rid]["area"] = int(rest.sum())
    plan["order"] = order
    return labels, table, plan
