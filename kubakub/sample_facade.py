"""
A synthetic facade to try the pack without any files: a classical elevation (ground floor with arches and a door,
upper floors with framed windows, pilasters, cornices, an attic) drawn from a few parameters.

numpy + OpenCV only (tests/test_sample_facade.py). Gives the look (shaded, a little texture so diffusion has
something to hold on to), the flat colour matrix (one colour per element, for kubakub facade mask atlas in
color_regions mode), the building silhouette and one mask per element, named like an After Effects mask folder
(Groups/M_*.png, Windows/W_F<floor>_C<column>.png).
"""

from __future__ import annotations

import os

import cv2
import numpy as np

STYLES = {                          # wall, trim, glass, frame, sky, ground
    "sandstone": ((196, 170, 128), (226, 208, 172), (38, 52, 66), (120, 98, 70), (22, 30, 52), (70, 64, 58)),
    "brick": ((150, 72, 52), (212, 200, 184), (30, 44, 58), (236, 232, 222), (26, 34, 56), (62, 58, 54)),
    "concrete": ((150, 150, 146), (182, 182, 176), (44, 56, 64), (80, 80, 80), (30, 38, 58), (66, 66, 66)),
    "night": ((74, 70, 78), (104, 98, 106), (236, 170, 88), (40, 36, 44), (6, 8, 18), (30, 28, 30)),
}


def layout(W, H, floors=3, bays=7):
    """-> [(name, group, (x0, y0, x1, y1), kind)] in canvas pixels; kind in wall / trim / glass / door / sky."""
    parts = []
    sky_h = int(H * 0.10)
    ground_y = int(H * 0.94)
    top = sky_h
    attic_h = int((ground_y - top) * 0.10)
    cornice_h = max(4, int(H * 0.018))
    ground_h = int((ground_y - top) * 0.28)
    body_top = top + attic_h + cornice_h
    ground_top = ground_y - ground_h
    floor_h = (ground_top - body_top) / max(1, floors)
    margin = int(W * 0.06)
    bay_w = (W - 2 * margin) / bays
    parts.append(("M_Sky", "Groups", (0, 0, W, ground_y), "sky"))          # behind the building, left and right too
    parts.append(("M_Attic", "Groups", (margin, top, W - margin, top + attic_h), "wall"))
    parts.append(("M_Cornice_Top", "Groups", (margin - int(W * 0.01), top + attic_h, W - margin + int(W * 0.01), body_top), "trim"))
    for f in range(floors):
        y0 = int(body_top + f * floor_h)
        fl = floors - f                                         # floors counted from the ground floor up
        parts.append((f"M_FLOOR_F{fl}", "Groups", (margin, y0, W - margin, int(y0 + floor_h)), "wall"))
        for c in range(bays):
            cx = margin + (c + 0.5) * bay_w
            ww, wh = bay_w * 0.46, floor_h * 0.56
            x0, y1 = int(cx - ww / 2), int(y0 + floor_h * 0.82)
            parts.append((f"W_F{fl}_C{c + 1:02d}", "Windows", (x0, int(y1 - wh), int(x0 + ww), y1), "glass"))
        parts.append((f"M_Sill_F{fl}", "Groups", (margin, int(y0 + floor_h * 0.86), W - margin, int(y0 + floor_h * 0.9)), "trim"))
    parts.append(("M_Cornice_Ground", "Groups", (margin - int(W * 0.005), ground_top - cornice_h, W - margin + int(W * 0.005), ground_top), "trim"))
    parts.append(("M_FLOOR_F0", "Groups", (margin, ground_top, W - margin, ground_y), "wall"))
    door = bays // 2
    for c in range(bays):
        cx = margin + (c + 0.5) * bay_w
        aw = bay_w * (0.62 if c == door else 0.5)
        name, kind = ("M_Door", "door") if c == door else (f"W_F0_C{c + 1:02d}", "glass")
        parts.append((name, "Groups" if c == door else "Windows",
                      (int(cx - aw / 2), int(ground_top + ground_h * 0.18), int(cx + aw / 2), ground_y), kind))
    for c in range(bays + 1):                                   # pilasters between the bays
        x = int(margin + c * bay_w)
        pw = max(3, int(bay_w * 0.06))
        parts.append((f"M_Pilaster_{c + 1:02d}", "Groups", (x - pw // 2, body_top, x + pw // 2 + 1, ground_top - cornice_h), "trim"))
    parts.append(("M_Ground", "Groups", (0, ground_y, W, H), "ground"))
    return parts


def _shape(name, box, W, H):
    """The element's mask: arched tops for the ground floor openings and the door, rectangles otherwise."""
    x0, y0, x1, y1 = box
    m = np.zeros((H, W), np.uint8)
    if name.startswith("W_F0_") or name == "M_Door":
        r = (x1 - x0) // 2
        cv2.rectangle(m, (x0, y0 + r), (x1 - 1, y1 - 1), 255, -1)
        cv2.ellipse(m, ((x0 + x1) // 2, y0 + r), (r, r), 0, 180, 360, 255, -1)
    else:
        cv2.rectangle(m, (x0, y0), (x1 - 1, y1 - 1), 255, -1)
    return m


def render(W=1920, H=1080, floors=3, bays=7, style="sandstone", seed=0):
    """-> dict(image float32 HxWx3 0..1, matrix float32 HxWx3 flat colours, silhouette float32 HxW, masks {name: (group, uint8 HxW)})"""
    wall, trim, glass, frame, sky, ground = [np.array(c, np.float32) for c in STYLES.get(style, STYLES["sandstone"])]
    rng = np.random.default_rng(int(seed))
    img = np.empty((H, W, 3), np.float32)
    matrix = np.zeros((H, W, 3), np.uint8)
    masks = {}
    order = {"sky": 0, "ground": 0, "wall": 1, "trim": 3, "glass": 4, "door": 4}
    parts = sorted(layout(W, H, floors, bays), key=lambda p: order[p[3]])
    palette = rng.permutation(np.arange(40, 255, 7))
    for k, (name, group, box, kind) in enumerate(parts):
        m = _shape(name, box, W, H)
        masks[name] = (group, m)
        col = {"sky": sky, "ground": ground, "wall": wall, "trim": trim, "glass": glass, "door": frame}[kind].copy()
        if kind == "glass":
            col = col * (0.8 + 0.4 * rng.random())               # every window a little different
        sel = m > 0
        img[sel] = col
        matrix[sel] = (palette[k % len(palette)], palette[(k * 5 + 3) % len(palette)], palette[(k * 11 + 7) % len(palette)])
        if kind in ("glass", "door"):                            # a frame around openings
            ring = cv2.dilate(m, np.ones((5, 5), np.uint8), iterations=max(1, W // 900)) & ~m
            img[ring > 0] = frame
    built = np.zeros((H, W), bool)
    for name, (_, m) in masks.items():
        if name != "M_Sky":
            built |= m > 0
    masks["M_Sky"] = ("Groups", ((masks["M_Sky"][1] > 0) & ~built).astype(np.uint8) * 255)   # only what the building leaves
    yy = np.linspace(0, 1, H, dtype=np.float32)[:, None, None]
    sky_sel = masks["M_Sky"][1] > 0
    img[sky_sel] = (sky * (1.0 + 0.8 * yy) * np.ones((1, W, 1), np.float32))[sky_sel]
    grain = rng.normal(0, 4.0, (H // 4 + 1, W // 4 + 1)).astype(np.float32)
    grain = cv2.resize(grain, (W, H), interpolation=cv2.INTER_CUBIC)[..., None]
    img = np.clip((img + grain * (~sky_sel)[..., None]) / 255.0, 0, 1)
    img = cv2.GaussianBlur(img, (0, 0), 0.6)                     # soft edges, like a rendered elevation
    sil = (~sky_sel & (masks["M_Ground"][1] == 0)).astype(np.float32)
    return {"image": img.astype(np.float32), "matrix": matrix.astype(np.float32) / 255.0, "silhouette": sil, "masks": masks}


def write_mask_folder(masks, folder):
    """Groups/ and Windows/ subfolders with one PNG per element (white on black); the sky and the ground are left out."""
    for name, (group, m) in masks.items():
        if name in ("M_Sky", "M_Ground"):
            continue
        d = os.path.join(folder, group)
        os.makedirs(d, exist_ok=True)
        cv2.imencode(".png", m)[1].tofile(os.path.join(d, f"{name}.png"))
    return folder


def sketch_photo(W=1920, H=1080, floors=3, bays=7, seed=0):
    """A phone photo of a hand pencil sketch of the same facade (for kubakub scan to line): wobbly grey strokes that
    stop short of the corners, a red marker dot on the door, blue dots in the first floor windows, cream paper on a
    dark table at an angle, lamp from the left. -> float32 (W*3/4 x W x 3)."""
    rng = np.random.default_rng(int(seed) + 101)
    paper = np.ones((H, W, 3), np.float32)
    lw = max(2, W // 640)
    gap = max(4, W // 200)

    def stroke(pts):
        pts = np.asarray(pts, np.float32)
        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1).sum()
        n = max(8, int(seg / 12))
        t = np.linspace(0, 1, len(pts))
        ts = np.linspace(gap * rng.random() / max(seg, 1), 1 - gap * rng.random() / max(seg, 1), n)
        p = np.c_[np.interp(ts, t, pts[:, 0]), np.interp(ts, t, pts[:, 1])] + rng.normal(0, 0.7, (n, 2))
        g = 0.3 + 0.25 * rng.random()
        cv2.polylines(paper, [p.astype(np.int32)], False, (g, g, g), lw, cv2.LINE_AA)

    for name, _group, (x0, y0, x1, y1), kind in layout(W, H, floors, bays):
        if kind in ("sky", "ground"):
            continue
        if name.startswith("W_F0_") or name == "M_Door":
            r = (x1 - x0) / 2
            arc = [((x0 + x1) / 2 + r * np.cos(a), y0 + r + r * np.sin(a)) for a in np.linspace(np.pi, 2 * np.pi, 24)]
            stroke([(x0, y1), (x0, y0 + r)]); stroke(arc); stroke([(x1, y0 + r), (x1, y1)]); stroke([(x0, y1), (x1, y1)])
        else:
            stroke([(x0, y0), (x1, y0)]); stroke([(x1, y0), (x1, y1)]); stroke([(x1, y1), (x0, y1)]); stroke([(x0, y1), (x0, y0)])
    rdot = max(5, W // 140)
    for name, _group, (x0, y0, x1, y1), _k in layout(W, H, floors, bays):
        col = (0.85, 0.12, 0.1) if name == "M_Door" else (0.12, 0.3, 0.85) if name.startswith("W_F1_") else None
        if col:
            cv2.circle(paper, ((x0 + x1) // 2, (y0 + y1) // 2 + (y1 - y0) // 6), rdot, col, -1, cv2.LINE_AA)
    paper *= np.array([0.97, 0.94, 0.86], np.float32)
    paper *= 1 - 0.05 * rng.random((H, W, 1)).astype(np.float32)
    PW, PH = W, W * 3 // 4
    cx, cy, sw, sh = PW / 2, PH / 2, PW * 0.42, PW * 0.42 * H / W
    corners = np.array([[cx - sw + PW * 0.03, cy - sh], [cx + sw, cy - sh + PH * 0.03],
                        [cx + sw + PW * 0.02, cy + sh], [cx - sw - PW * 0.02, cy + sh - PH * 0.02]], np.float32)
    m = cv2.getPerspectiveTransform(np.array([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]], np.float32), corners)
    table = np.full((PH, PW, 3), (0.22, 0.15, 0.1), np.float32) + rng.normal(0, 0.02, (PH, PW, 3)).astype(np.float32)
    warped = cv2.warpPerspective(paper, m, (PW, PH), flags=cv2.INTER_LINEAR, borderValue=(-1, -1, -1))
    photo = np.where(warped[..., :1] >= 0, warped, table)
    photo *= np.linspace(1.0, 0.75, PW, dtype=np.float32)[None, :, None]
    return np.clip(photo, 0, 1).astype(np.float32)
