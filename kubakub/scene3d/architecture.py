"""
architecture.py

Architectural grouping of a facade seen from the projection camera, from the per-pixel 3D data of a Blender
export (scene_ids.load): world position, normal and the main facade plane. No AI model, deterministic.
numpy + opencv only, no ComfyUI imports (tests/test_scene3d.py).

Three labellings (pixel level, -1 = background), written as ID passes by scene_ids.build:
    elements   windows (openings behind the wall), window_frames (what stands out around them), columns
               (tall narrow protrusions), cornices (long horizontal ones), relief (other protrusions), wall, roof
    sections   left / centre / right ... : where the wall surface steps forward or back (an avant-corps), else
               thirds of the facade
    floors     ground_floor, floor_1, floor_2 ... between the cornice lines; roof above the highest line
From ID Maps turns 'elements' into regions (split: every window, column ... its own region, grouped by class)
and takes 'sections' / 'floors' as tags: group:windows, tag:left, 'tag:floor_1 group:windows'.
"""

from __future__ import annotations

import cv2
import numpy as np

ELEMENTS = ("wall", "windows", "window_frames", "columns", "cornices", "relief", "roof")


def _px_m(position, fg):
    """Metres per pixel along the rows (median neighbour distance on the model)."""
    d = np.linalg.norm(np.diff(position, axis=1), axis=-1)
    ok = fg[:, 1:] & fg[:, :-1]
    v = d[ok]
    v = v[(v > 0) & (v < np.percentile(v, 90))] if v.size else v
    return float(np.median(v)) if v.size else 0.01


def _kernel(w, h):
    return cv2.getStructuringElement(cv2.MORPH_RECT, (max(1, int(w)), max(1, int(h))))


def _runs(mask_1d, min_len):
    """[start, end) of the True runs of at least min_len."""
    m = np.concatenate([[False], mask_1d, [False]])
    edges = np.flatnonzero(m[1:] != m[:-1])
    return [(a, b) for a, b in zip(edges[::2], edges[1::2]) if b - a >= min_len]


def wall_depth_per_column(rel, fg, front=None, px_m=0.01, window_m=1.5, bin_m=0.04):
    """
    The wall surface around every pixel column: the most common depth of the front-facing surfaces within
    +-window_m / 2 (over a window, not one column: a column through a window sees mostly glass). nan = empty.
    """
    H, W = rel.shape
    use = fg if front is None else fg & front
    out = np.full(W, np.nan)
    if not use.any():
        return out
    lo, hi = np.nanpercentile(rel[use], [0.5, 99.5])
    nb = max(2, int(np.ceil((hi - lo) / bin_m)) + 2)
    idx = np.clip(np.nan_to_num((rel - lo) / bin_m, nan=0.0).astype(np.int64), 0, nb - 1)
    hist = np.zeros((W, nb), np.int64)                     # per column: count of front pixels in each depth bin
    ys, xs = np.nonzero(use)
    np.add.at(hist, (xs, idx[ys, xs]), 1)
    cs = np.concatenate([np.zeros((1, nb), np.int64), np.cumsum(hist, axis=0)])
    r = max(1, int(round(window_m / px_m / 2)))
    for x in range(W):
        if not use[:, x].any():
            continue
        h = cs[min(W, x + r + 1)] - cs[max(0, x - r)]
        k = int(np.argmax(h))
        sel = rel[:, x][use[:, x] & (idx[:, x] == k)]
        out[x] = float(np.median(sel)) if sel.size else lo + (k + 0.5) * bin_m
    return out


def sections(wall_d, fg, px_m, step_m=0.15, min_width_m=3.0):
    """Column ranges of the facade sections: split where the wall depth steps by more than step_m."""
    cols = np.flatnonzero(fg.any(axis=0))
    if not len(cols):
        return [], []
    x0, x1 = int(cols[0]), int(cols[-1]) + 1
    wd = wall_d.copy()
    idx = np.arange(len(wd))
    good = ~np.isnan(wd)
    if good.sum() < 2:
        return [(x0, x1)], ["facade"]
    wd = np.interp(idx, idx[good], wd[good])
    k = max(3, int(0.5 / px_m) | 1)                      # median over half a metre: single mouldings vanish
    pad = np.pad(wd[x0:x1], k // 2, mode="edge")
    sm = np.array([np.median(pad[i:i + k]) for i in range(x1 - x0)])
    level = np.round(sm / step_m)                        # quantised wall level; a change = a section border
    cuts = [0] + [i for i in range(1, len(level)) if level[i] != level[i - 1]] + [len(level)]
    segs = [[x0 + a, x0 + b] for a, b in zip(cuts[:-1], cuts[1:])]
    min_w = min_width_m / px_m
    changed = True
    while changed and len(segs) > 1:                     # narrow pieces join the neighbour they resemble most
        changed = False
        for i, (a, b) in enumerate(segs):
            if b - a < min_w:
                j = i - 1 if i == len(segs) - 1 or (i > 0 and segs[i - 1][1] - segs[i - 1][0] >= segs[i + 1][1] - segs[i + 1][0]) else i + 1
                lo_, hi_ = min(i, j), max(i, j)
                segs[lo_:hi_ + 1] = [[segs[lo_][0], segs[hi_][1]]]
                changed = True
                break
    # neighbours at the same wall level merge (a pilaster broke a wall into two)
    merged = [segs[0]]
    for a, b in segs[1:]:
        pa, pb = merged[-1]
        if abs(np.median(sm[pa - x0:pb - x0]) - np.median(sm[a - x0:b - x0])) < step_m * 0.75:
            merged[-1] = [pa, b]
        else:
            merged.append([a, b])
    segs = merged
    if len(segs) == 1:                                   # a flat facade: thirds
        t = (x1 - x0) / 3
        segs = [[x0, int(x0 + t)], [int(x0 + t), int(x0 + 2 * t)], [int(x0 + 2 * t), x1]]
    return [tuple(s) for s in segs], section_names(len(segs))


def section_names(n):
    if n == 1:
        return ["facade"]
    if n == 2:
        return ["left", "right"]
    if n == 3:
        return ["left", "centre", "right"]
    half = n // 2
    left = ["left"] + [f"left_{i}" for i in range(2, half + 1)]
    right = [f"right_{i}" for i in range(half, 1, -1)] + ["right"]
    return left + (["centre"] if n % 2 else []) + right


def _split_necks(mask, r):
    """
    Connected parts of mask, big ones split where they narrow below 2r (an arcade = one part per arch).
    Yields (y0, x0, part): part is a bool crop placed at (y0, x0). Each component is worked on inside its
    bounding box padded by r + 2, so the erosion and the 5x5 distance mask see what the full image shows.
    """
    n, cc, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    H, W = mask.shape[:2]
    pad = r + 2
    ker = _kernel(2 * r + 1, 2 * r + 1)
    for i in range(1, n):
        x, y, w, h = (int(v) for v in stats[i, :4])
        y0, x0 = max(0, y - pad), max(0, x - pad)
        comp = (cc[y0:min(H, y + h + pad), x0:min(W, x + w + pad)] == i).astype(np.uint8)
        seeds_n, seeds = cv2.connectedComponents(cv2.erode(comp, ker), connectivity=8)
        if seeds_n <= 2:
            yield y0, x0, comp.astype(bool)
            continue
        # every pixel to its nearest seed (Voronoi of the eroded cores), limited to the component
        inv = (seeds == 0).astype(np.uint8)
        _, lab = cv2.distanceTransformWithLabels(inv, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_CCOMP)
        # map the distance-transform labels (one per zero pixel group) back to the seeds they belong to
        lut = np.zeros(int(lab.max()) + 1, np.int64)
        lut[lab[seeds > 0]] = seeds[seeds > 0]
        part = lut[lab] * comp
        for k in range(1, seeds_n):                       # 1 px gap to the neighbours: they stay apart as regions
            yield y0, x0, cv2.erode((part == k).astype(np.uint8), _kernel(3, 3)).astype(bool)


def _window_rows(windows, height):
    """Vertical extents (m) of the window rows: windows whose height ranges overlap form one row."""
    n, cc, stats, _ = cv2.connectedComponentsWithStats(windows.astype(np.uint8), 8)
    spans = []
    for i in range(1, n):
        hv = height[cc == i]
        hv = hv[np.isfinite(hv)]
        if hv.size:
            lo_, hi_ = float(np.percentile(hv, 5)), float(np.percentile(hv, 95))
            if hi_ - lo_ >= 0.4:                          # a window row needs real windows, not crumbs
                spans.append([lo_, hi_])
    spans.sort()
    rows = []
    for lo, hi in spans:
        if rows and lo < rows[-1][1] - 0.3 * min(hi - lo, rows[-1][1] - rows[-1][0]):
            rows[-1] = [min(rows[-1][0], lo), max(rows[-1][1], hi)]
        else:
            rows.append([lo, hi])
    return rows


def analyse(scene, pt, nrm, ground_z, up=(0.0, 0.0, 1.0)):
    """
    -> {"elements": (labels, names), "sections": (labels, names), "floors": (labels, names), "info": {...}}
    labels are int32 (H, W), -1 = background.
    """
    pos, fid = scene["position"], scene["faceid"]
    fg = fid > 0
    H, W = fg.shape
    px_m = _px_m(pos, fg)
    m = lambda metres: max(1, int(round(metres / px_m)))  # noqa: E731
    rel0 = np.where(fg, (pos - pt) @ nrm, np.nan)
    height = np.where(fg, pos @ np.asarray(up, float) - ground_z, np.nan)
    normal = scene.get("normal")
    front = fg if normal is None else fg & ((np.asarray(normal, np.float32) @ np.asarray(nrm, np.float32)) > 0.9)

    # sections: the wall surface per column, split where it steps
    wall_d = wall_depth_per_column(rel0, fg, front, px_m)
    segs, sec_names = sections(wall_d, fg, px_m)
    sec = np.full((H, W), -1, np.int32)
    wall_ref = np.zeros(W)
    for k, (a, b) in enumerate(segs):
        sec[:, a:b] = np.where(fg[:, a:b], k, -1)
        v = rel0[:, a:b][front[:, a:b]]
        if v.size:
            hist, e = np.histogram(v, np.arange(np.nanmin(v) - 0.04, np.nanmax(v) + 0.08, 0.04))
            j = int(np.argmax(hist))
            wall_ref[a:b] = float(np.median(v[(v >= e[j]) & (v < e[j + 1])]))
    rel = np.where(fg, rel0 - wall_ref[None, :], 0.0)    # depth relative to the section's own wall

    # openings: well behind the wall, glazing bars closed, big enough to be a window or a door
    behind = (fg & (rel < -0.10)).astype(np.uint8)
    behind = cv2.morphologyEx(behind, cv2.MORPH_CLOSE, _kernel(m(0.15), m(0.15)))
    min_px, max_px, side_px = 0.25 / px_m ** 2, 6.0 / px_m ** 2, 0.6 / px_m
    windows = np.zeros((H, W), bool)
    for y0, x0, comp in _split_necks(behind, m(0.3)):
        ys, xs = np.nonzero(comp)
        if not len(ys):
            continue                                      # a split part the 1 px gap ate completely
        w, h, area = xs.max() - xs.min() + 1, ys.max() - ys.min() + 1, len(ys)
        if area < min_px or min(w, h) < side_px:
            continue                                      # slivers, pediment triangles, gaps at section edges
        if area > 4.0 / px_m ** 2 and area < 0.5 * w * h:
            continue                                      # a big triangle (tympanum) is not an opening
        windows[y0:y0 + comp.shape[0], x0:x0 + comp.shape[1]] |= comp
    # blind windows / panels: wall-level areas a frame closes in (window size, roughly rectangular)
    out0 = (fg & (rel > 0.03)).astype(np.uint8)
    ring = cv2.morphologyEx(out0, cv2.MORPH_CLOSE, _kernel(m(0.12), m(0.12)))
    inside = (fg & ~ring.astype(bool) & ~windows).astype(np.uint8)
    n, cc, stats, _ = cv2.connectedComponentsWithStats(inside, 4)
    for i in range(1, n):
        x, y, w, h, area = (int(v) for v in stats[i])
        if not (min_px <= area <= max_px) or area < 0.7 * w * h or not (0.25 <= w / max(h, 1) <= 4) or min(w, h) < side_px:
            continue
        if x == 0 or y == 0 or x + w >= W or y + h >= H:
            continue
        box = ring[max(0, y - 2):y + h + 3, max(0, x - 2):x + w + 3]
        if box.size and box.mean() > 0.02:                # really framed, not a patch of plain wall
            windows[y:y + h, x:x + w] |= cc[y:y + h, x:x + w] == i
    ff = np.zeros((H, W), np.uint8)                       # fill what the glazing encloses, window by window
    n, cc = cv2.connectedComponents(windows.astype(np.uint8), connectivity=8)
    for i in range(1, n):
        one = (cc == i).astype(np.uint8)
        cnts, _ = cv2.findContours(one, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(ff, cnts, -1, 1, thickness=cv2.FILLED)
    windows = ff.astype(bool) & fg
    n, cc, st, _ = cv2.connectedComponentsWithStats(windows.astype(np.uint8), 8)
    for i in range(1, n):                                 # crumbs from the arch split: not windows
        if st[i, cv2.CC_STAT_AREA] < min_px or min(st[i, cv2.CC_STAT_WIDTH], st[i, cv2.CC_STAT_HEIGHT]) < side_px:
            windows[cc == i] = False

    # what stands out: frames around openings, long horizontals (cornices), long verticals (columns), the rest
    out = fg & (rel > 0.03) & ~windows
    near = cv2.distanceTransform((~windows).astype(np.uint8), cv2.DIST_L2, 5) * px_m < 0.35
    frames = out & near
    rest = (out & ~frames).astype(np.uint8)
    cornice = cv2.morphologyEx(rest, cv2.MORPH_OPEN, _kernel(m(1.5), 1)).astype(bool)
    column = cv2.morphologyEx(rest, cv2.MORPH_OPEN, _kernel(1, m(2.5))).astype(bool) & ~cornice
    relief = rest.astype(bool) & ~cornice & ~column

    # floor lines: cornices that run on across most of the facade (a belt, not a row of window pediments)
    cols = np.flatnonzero(fg.any(axis=0))
    width_px = max(1, int(cols[-1] - cols[0] + 1)) if len(cols) else 1
    belt = np.zeros(H, bool)
    band = cv2.dilate((cornice | frames).astype(np.uint8), _kernel(m(0.4), 1)).astype(bool)   # small breaks close
    for y in range(H):
        runs = _runs(band[y], 1)
        belt[y] = bool(runs) and max(b - a for a, b in runs) >= 0.6 * width_px
    lines = []
    for a, b in _runs(belt, 1):
        hv = height[a:b][band[a:b] & fg[a:b]]
        if hv.size:
            lines.append(float(np.median(hv)))
    merged_lines = []
    for hl in sorted(lines):                              # a cornice has several mouldings: one line per 0.8 m
        if merged_lines and hl - merged_lines[-1] < 0.8:
            merged_lines[-1] = max(merged_lines[-1], hl)
        else:
            merged_lines.append(hl)
    lines = [hl for hl in merged_lines if hl > 1.0]
    top = lines[-1] if lines else np.inf                  # above the highest belt: the roof (attic, dormers, pediment) ...
    if np.isfinite(top):                                  # ... only when the facade narrows there, not a full storey
        above = fg & (np.nan_to_num(height, nan=-1e9) >= top)
        rows_above = np.flatnonzero(above.any(axis=1))
        if rows_above.size and above[rows_above].sum(axis=1).mean() >= 0.6 * width_px:
            top = np.inf
    below_top = windows & (np.nan_to_num(height, nan=-1e9) < top)
    rows = _window_rows(below_top, height)
    borders = []
    for (lo1, hi1), (lo2, hi2) in zip(rows[:-1], rows[1:]):
        inside_belts = [hl for hl in lines if hi1 - 0.3 <= hl <= lo2 + 0.3]
        borders.append(inside_belts[0] if inside_belts else (hi1 + lo2) / 2)
    edges = [-np.inf] + borders + ([top] if np.isfinite(top) else []) + [np.inf]
    fl = np.full((H, W), -1, np.int32)
    hh = np.where(fg, height, -1e9)
    fl_names = []
    n_floors = len(edges) - 1 - (1 if np.isfinite(top) else 0)
    for k in range(len(edges) - 1):
        name = "roof" if k >= n_floors else ("ground_floor" if k == 0 else f"floor_{k}")
        fl_names.append(name)
        fl[fg & (hh >= edges[k]) & (hh < edges[k + 1])] = k
    roof_zone = fg & (hh >= top)

    el = np.full((H, W), -1, np.int32)
    el[fg] = ELEMENTS.index("wall")
    el[relief] = ELEMENTS.index("relief")
    el[column] = ELEMENTS.index("columns")
    el[cornice] = ELEMENTS.index("cornices")
    el[frames] = ELEMENTS.index("window_frames")
    el[roof_zone & ~windows & ~frames] = ELEMENTS.index("roof")
    el[windows] = ELEMENTS.index("windows")
    info = {"px_m": round(px_m, 4), "sections_px": [list(map(int, s_)) for s_ in segs], "section_names": sec_names,
            "floor_lines_m": [round(v, 2) for v in lines], "floor_borders_m": [round(v, 2) for v in borders],
            "window_rows_m": [[round(a_, 2), round(b_, 2)] for a_, b_ in rows],
            "wall_steps_m": [round(float(np.nanmedian(wall_d[a_:b_])), 2) for a_, b_ in segs],
            "windows": int(cv2.connectedComponents(windows.astype(np.uint8), connectivity=8)[0] - 1)}
    return {"elements": (el, list(ELEMENTS)), "sections": (sec, sec_names), "floors": (fl, fl_names), "info": info}
