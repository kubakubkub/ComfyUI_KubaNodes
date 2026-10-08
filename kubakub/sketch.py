"""
sketch.py

Hand drawings into the region framework (logic for nodes_sketch.py; numpy + opencv + scipy, no ComfyUI imports,
tests/test_sketch.py):

- scan: a phone photo of a drawing -> find the paper (or 4 given corners) -> perspective warp to the matrix ->
  remove paper colour, shadows and texture -> line strength 0..1 (pencil greys kept, or clean ink).
- regions: closed shapes of the drawing -> cells. Hand-drawn strokes rarely meet, so every stroke end is extended
  in its own direction to the next line within gap_px (thin skeleton, endpoints, cone search). Coloured marks
  (a dot of red marker in a cell) tag the cell with the colour name.
- overlay: the artist's line back on top of a render.
"""

from __future__ import annotations

import re

import cv2
import numpy as np

# ---------------------------------------------------------------------------------------------------------- scan


def parse_corners(text: str):
    """'x,y x,y x,y x,y' (any separators) -> 4x2 float array, or None when empty / not 8 numbers."""
    nums = [float(v) for v in re.findall(r"-?\d+(?:\.\d+)?", text or "")]
    if len(nums) != 8:
        return None
    return np.array(nums, np.float32).reshape(4, 2)


def order_corners(pts):
    """Top-left, top-right, bottom-right, bottom-left."""
    pts = np.asarray(pts, np.float32).reshape(4, 2)
    s, d = pts.sum(1), np.diff(pts, axis=1)[:, 0]
    return np.array([pts[np.argmin(s)], pts[np.argmin(d)], pts[np.argmax(s)], pts[np.argmax(d)]], np.float32)


def find_paper(rgb):
    """(corners 4x2 in image pixels, found?) - the largest bright four-sided shape; the whole image if none."""
    h, w = rgb.shape[:2]
    sc = 1000.0 / max(h, w) if max(h, w) > 1000 else 1.0
    small = cv2.resize(rgb, (max(1, int(w * sc)), max(1, int(h * sc))), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor((np.clip(small, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))   # pencil lines don't split the paper
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    full = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], np.float32)
    if not cnts:
        return full, False
    c = max(cnts, key=cv2.contourArea)
    area = cv2.contourArea(c)
    if area < 0.15 * small.shape[0] * small.shape[1] or area > 0.985 * small.shape[0] * small.shape[1]:
        return full, False
    hull = cv2.convexHull(c)
    peri = cv2.arcLength(hull, True)
    for eps in (0.02, 0.03, 0.05, 0.08):
        ap = cv2.approxPolyDP(hull, eps * peri, True)
        if len(ap) == 4:
            return order_corners(ap.reshape(4, 2) / sc), True
    box = cv2.boxPoints(cv2.minAreaRect(hull))
    return order_corners(box / sc), True


def paper_size(corners):
    """Width and height of the paper in photo pixels (mean of opposite sides)."""
    tl, tr, br, bl = corners
    w = (np.linalg.norm(tr - tl) + np.linalg.norm(br - bl)) / 2
    h = (np.linalg.norm(bl - tl) + np.linalg.norm(br - tr)) / 2
    return float(w), float(h)


def warp(rgb, corners, width, height):
    dst = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], np.float32)
    m = cv2.getPerspectiveTransform(order_corners(corners), dst)
    return cv2.warpPerspective(rgb, m, (int(width), int(height)), flags=cv2.INTER_CUBIC,
                               borderMode=cv2.BORDER_REPLICATE)


def flatten(rgb):
    """Divide out the paper: its colour, the phone's shading and shadows -> paper = 1.0 (float32 HxWx3)."""
    h, w = rgb.shape[:2]
    k = max(15, int(round(max(h, w) * 0.02)) | 1)                  # bigger than any line is wide
    sc = 512.0 / max(h, w) if max(h, w) > 512 else 1.0             # the paper estimate is smooth: work small
    small = cv2.resize(rgb, (max(1, int(w * sc)), max(1, int(h * sc))), interpolation=cv2.INTER_AREA)
    ks = max(3, int(k * sc) | 1)
    bg = cv2.dilate(small, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks, ks)))   # max filter: lines vanish
    bg = cv2.GaussianBlur(bg, (0, 0), ks / 2)
    bg = cv2.resize(bg, (w, h), interpolation=cv2.INTER_LINEAR)
    return np.clip(rgb / np.maximum(bg, 1e-3), 0.0, 1.0).astype(np.float32)


def line_strength(flat, mode="pencil", clean=0.5, boost=1.0, min_speck_px=6):
    """0..1 how much line there is per pixel. pencil keeps the greys, ink = clean black / white."""
    dark = 1.0 - flat.min(axis=2)                                  # coloured pencil counts as line too
    fg = dark[dark > 0.02]
    top = float(np.percentile(fg, 99.5)) if fg.size else 1.0
    floor = 0.04 + 0.2 * float(clean)                              # paper grain below this goes
    s = np.clip((dark - floor) / max(top - floor, 0.05), 0.0, 1.0)
    if mode == "ink":
        s = (s > 0.35).astype(np.float32)
    if boost != 1.0:
        s = np.power(s, 1.0 / max(float(boost), 0.05))
    if min_speck_px > 0:                                            # dust, crumbs, pores of the paper
        n, cc, st, _ = cv2.connectedComponentsWithStats((s > 0.2).astype(np.uint8), connectivity=8)
        small = st[:, cv2.CC_STAT_AREA] < int(min_speck_px)
        small[0] = False
        if small.any():
            s = np.where(small[cc], 0.0, s)
    return s.astype(np.float32)


def clean_drawing(flat, strength):
    """The drawing on pure white: colours of the strokes kept, paper and grain gone."""
    dark = 1.0 - flat.min(axis=2)
    k = np.clip(strength / np.maximum(dark, 1e-3), 0.0, 1.0)[..., None]
    return (1.0 - (1.0 - flat) * k).astype(np.float32)


def scan(rgb, width=0, height=0, corners_text="", mode="pencil", clean=0.5, boost=1.0, dots_in_line=False):
    """Photo -> dict(drawing, strength, corners, found, note). strength leaves the marker dots out unless
    dots_in_line (the drawing keeps them: regions from sketch reads the names there)."""
    rgb = np.clip(np.asarray(rgb, np.float32)[..., :3], 0, 1)
    given = parse_corners(corners_text)
    if given is not None:
        corners, found, how = order_corners(given), True, "given corners"
    else:
        corners, found = find_paper(rgb)
        how = "paper found" if found else "no paper edge found: the whole photo is used"
    pw, ph = paper_size(corners)
    if width <= 0 and height <= 0:
        width, height = int(round(pw)), int(round(ph))
    elif width <= 0:
        width = int(round(height * pw / max(ph, 1)))
    elif height <= 0:
        height = int(round(width * ph / max(pw, 1)))
    flat = flatten(warp(rgb, corners, width, height))
    s = line_strength(flat, mode, clean, boost)
    drawing = clean_drawing(flat, s)
    if not dots_in_line:
        s = hide_marks(s, drawing)
    return {"drawing": drawing, "strength": s, "corners": corners, "found": found,
            "note": f"{how}; paper {pw:.0f}x{ph:.0f} px in the photo -> {width}x{height}"}


def corners_text(corners):
    return " ".join(f"{x:.0f},{y:.0f}" for x, y in corners)


def draw_quad(rgb, corners):
    out = (np.clip(rgb[..., :3], 0, 1) * 255).astype(np.uint8).copy()
    t = max(2, max(out.shape[:2]) // 300)
    cv2.polylines(out, [np.round(corners).astype(np.int32)], True, (241, 138, 88), t, cv2.LINE_AA)
    for i, (x, y) in enumerate(corners):
        cv2.circle(out, (int(x), int(y)), t * 4, (161, 135, 183), -1, cv2.LINE_AA)
        cv2.putText(out, "tl tr br bl".split()[i], (int(x) + t * 5, int(y) + t * 5), cv2.FONT_HERSHEY_SIMPLEX,
                    t * 0.5, (161, 135, 183), max(1, t // 2), cv2.LINE_AA)
    return out.astype(np.float32) / 255


# ------------------------------------------------------------------------------------------------------- regions


def thin(mask):
    """Zhang-Suen thinning (bool HxW -> 1 px wide skeleton), vectorised."""
    img = np.pad(mask.astype(np.uint8), 1)
    while True:
        changed = False
        for step in (0, 1):
            p = img
            n = [p[:-2, 1:-1], p[:-2, 2:], p[1:-1, 2:], p[2:, 2:], p[2:, 1:-1], p[2:, :-2], p[1:-1, :-2], p[:-2, :-2]]
            c = p[1:-1, 1:-1]
            b = sum(x.astype(np.int32) for x in n)
            a = sum(((n[i] == 0) & (n[(i + 1) % 8] == 1)).astype(np.int32) for i in range(8))
            if step == 0:
                cond = (n[0] * n[2] * n[4] == 0) & (n[2] * n[4] * n[6] == 0)
            else:
                cond = (n[0] * n[2] * n[6] == 0) & (n[0] * n[4] * n[6] == 0)
            rm = (c == 1) & (b >= 2) & (b <= 6) & (a == 1) & cond
            if rm.any():
                img[1:-1, 1:-1][rm] = 0
                changed = True
        if not changed:
            return img[1:-1, 1:-1].astype(bool)


def endpoints(skel):
    """(ys, xs) of skeleton pixels with exactly one neighbour."""
    k = np.ones((3, 3), np.float32)
    k[1, 1] = 0
    nb = cv2.filter2D(skel.astype(np.float32), -1, k, borderType=cv2.BORDER_CONSTANT)
    return np.nonzero(skel & (nb == 1))


def bridge_gaps(lines, gap_px, cone_deg=40.0):
    """Extend every stroke end along its direction to the nearest other line within gap_px -> (lines, bridges)."""
    if gap_px <= 0:
        return lines, 0
    skel = thin(lines)
    ey, ex = endpoints(skel)
    if len(ey) == 0:
        return lines, 0
    sy, sx = np.nonzero(skel)
    from scipy.spatial import cKDTree
    tree = cKDTree(np.c_[sx, sy])
    out = lines.astype(np.uint8).copy()
    back = max(4, int(gap_px // 3))
    cos_cone = np.cos(np.radians(cone_deg))
    _, comp = cv2.connectedComponents(skel.astype(np.uint8), connectivity=8)
    ends = np.c_[ex, ey].astype(np.float32)
    n = 0
    for y, x in zip(ey, ex):
        own = comp[y, x]
        near = [j for j in tree.query_ball_point([x, y], back) if comp[sy[j], sx[j]] == own]   # own last pixels
        if len(near) < 2:
            continue
        v = np.array([x, y], np.float32) - np.c_[sx[near], sy[near]].mean(0)
        if np.linalg.norm(v) < 1e-3:
            continue
        v /= np.linalg.norm(v)
        # 1) a line straight ahead (the stroke should have met it: T-junctions, crossings)
        best, bd = None, 1e9
        for j in tree.query_ball_point([x, y], gap_px):
            d = np.array([sx[j] - x, sy[j] - y], np.float32)
            dist = float(np.linalg.norm(d))
            if dist >= bd or (comp[sy[j], sx[j]] == own and dist <= 2 * back):   # own stroke / not closer
                continue
            if float(d @ v) / max(dist, 1e-6) >= cos_cone:
                best, bd = (int(sx[j]), int(sy[j])), dist
        # 2) else the nearest other stroke end in front (corners: both strokes stopped short)
        if best is None:
            d = ends - np.array([x, y], np.float32)
            dist = np.linalg.norm(d, axis=1)
            fwd = (d @ v) > 0.2 * dist
            other = np.array([comp[int(b), int(a)] != own for a, b in ends])
            ok = fwd & other & (dist > 0) & (dist <= gap_px)
            if ok.any():
                j = int(np.argmin(np.where(ok, dist, np.inf)))
                best = (int(ends[j, 0]), int(ends[j, 1]))
        if best is not None:
            cv2.line(out, (int(x), int(y)), best, 1, 2)
            n += 1
    return out.astype(bool), n


HUES = [("red", 0), ("orange", 25), ("yellow", 55), ("green", 115), ("cyan", 180), ("blue", 225), ("purple", 275),
        ("pink", 320), ("red", 360)]


def hue_name(h_deg):
    return min(HUES, key=lambda t: abs(t[1] - h_deg))[0]


def colour_marks(drawing, min_sat=0.35, min_px=20, max_frac=0.04):
    """(bool mask of marker DOTS, hue in degrees HxW). A dot = a compact saturated blob (filled, not longer than
    max_frac of the drawing); coloured pencil LINES are long and thin and stay lines."""
    hsv = cv2.cvtColor(np.clip(drawing, 0, 1).astype(np.float32), cv2.COLOR_RGB2HSV)   # H 0..360, S 0..1
    m = (hsv[..., 1] >= min_sat) & (hsv[..., 2] > 0.15)
    m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)).astype(bool)
    n, cc, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=8)
    bw, bh, area = st[:, cv2.CC_STAT_WIDTH], st[:, cv2.CC_STAT_HEIGHT], st[:, cv2.CC_STAT_AREA]
    compact = area >= 0.45 * bw * bh                                 # a disc fills ~79 % of its box, a stroke little
    small = np.maximum(bw, bh) <= max_frac * max(drawing.shape[:2])
    keep = (area >= min_px) & compact & small
    keep[0] = False
    return keep[cc], hsv[..., 0]


def hide_marks(strength, drawing):
    """The line without the marker dots (they name shapes; they are no part of the drawing's line)."""
    marks, _ = colour_marks(drawing)
    if not marks.any():
        return strength
    marks = cv2.dilate(marks.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    return np.where(marks, 0.0, strength).astype(np.float32)


def sketch_cells(strength, drawing=None, gap_px=24, threshold=0.3, work_px=1600, use_colour=True):
    """-> (cells HxW int32, -1 on lines; tags per cell {id: [colour names]}; bridges; the line mask used)."""
    h, w = strength.shape
    sc = min(1.0, float(work_px) / max(h, w))
    small_s = cv2.resize(strength, (max(1, int(w * sc)), max(1, int(h * sc))), interpolation=cv2.INTER_AREA)
    lines = small_s > threshold
    marks = hue = None
    if use_colour and drawing is not None:
        small_d = cv2.resize(drawing, (small_s.shape[1], small_s.shape[0]), interpolation=cv2.INTER_AREA)
        marks, hue = colour_marks(small_d, min_px=max(4, int(20 * sc * sc)))
        marks = cv2.dilate(marks.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)   # with its dark rim
        lines &= ~marks                                              # a marker dot is a name, not a wall
    lines = cv2.morphologyEx(lines.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)).astype(bool)
    lines, n_br = bridge_gaps(lines, gap_px * sc)
    k, cc = cv2.connectedComponents((~lines).astype(np.uint8), connectivity=4, ltype=cv2.CV_32S)
    cells = cc - 1
    tags = {}
    if marks is not None and marks.any():
        for cid in np.unique(cells[marks]):
            if cid < 0:
                continue
            sel = marks & (cells == cid)
            names = [hue_name(float(v)) for v in hue[sel]]
            best = max(set(names), key=names.count)
            tags[int(cid)] = [best]
        cells = np.where(marks & (cells < 0), -1, cells)
    if sc < 1.0:
        cells = cv2.resize(cells.astype(np.int32), (w, h), interpolation=cv2.INTER_NEAREST)
        lines = cv2.resize(lines.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
    return cells.astype(np.int32), tags, n_br, lines


# ------------------------------------------------------------------------------------------------------- overlay


def hex_rgb(text, default=(0.0, 0.0, 0.0)):
    t = (text or "").strip().lstrip("#")
    if len(t) == 3:
        t = "".join(c * 2 for c in t)
    try:
        return tuple(int(t[i:i + 2], 16) / 255 for i in (0, 2, 4))
    except (ValueError, IndexError):
        return default


def line_shape(strength, height, width, grow_px=0, soften_px=0.0):
    """The line as it is laid on an image: at the image's size, grown (bolder) and softened -> float32 HxW 0..1."""
    s = np.clip(strength, 0, 1).astype(np.float32)
    if s.shape != (height, width):
        s = cv2.resize(s, (width, height), interpolation=cv2.INTER_LINEAR)
    if grow_px > 0:
        k = 2 * int(grow_px) + 1
        s = cv2.dilate(s, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    if soften_px > 0:
        s = cv2.GaussianBlur(s, (0, 0), float(soften_px))
    return np.clip(s, 0, 1)


def overlay(image, strength, amount=1.0, mode="multiply", colour="#000000", grow_px=0, soften_px=0.0):
    """The line on top of an image: multiply (dark line), screen (light line), or colour (paint it)."""
    s = line_shape(strength, image.shape[0], image.shape[1], grow_px, soften_px)
    a = (s * float(amount))[..., None]
    c = np.array(hex_rgb(colour), np.float32)
    img = image[..., :3].astype(np.float32)
    if mode == "multiply":
        target = img * c                                            # black = a dark line, a colour tints
    elif mode == "screen":
        target = 1 - (1 - img) * (1 - c)
    else:
        target = np.broadcast_to(c, img.shape)
    return np.clip(img * (1 - a) + target * a, 0, 1).astype(np.float32)
