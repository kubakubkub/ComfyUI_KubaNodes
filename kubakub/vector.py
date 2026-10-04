"""
vector.py

Regions to vectors for kubakub regions to vector.
numpy / opencv / scikit-image / shapely, PyMuPDF only for PDF; no ComfyUI imports
(tests/test_vector.py).

Two modes share the tracing:
- outline: every region as closed paths (outer ring + holes). Vertices where the
  outline turns sharply stay corners (straight edges stay straight); gentle
  vertices become smooth cubic Beziers (Catmull-Rom through the simplified
  points). SVG (groups = region groups, ids = region names), PDF (one layer per
  group) and DXF.
- centerline: the skeleton of each region as open polylines (for plotters,
  lasers, engraving), short spurs removed, ordered for short travel.
Units: pixels, or millimetres from one uniform scale (width_mm / matrix width).
Kerf (outline, mm): each outline offset by kerf / 2 with a mitred buffer.
"""

from __future__ import annotations

import math
import os

import cv2
import numpy as np

from . import optional

# ---------------------------------------------------------------------------
# tracing
# ---------------------------------------------------------------------------


def trace_mask(mask: np.ndarray, simplify_px: float = 1.0, min_area_px: float = 0.0):
    """
    Outlines of a bool mask: [(outer Nx2, [hole Mx2, ...]), ...] in pixel coordinates
    (pixel edges, not centres: the rings are grown by half a pixel). Rings smaller than
    min_area_px are dropped.
    """
    m = np.ascontiguousarray(mask.astype(np.uint8))
    if not m.any():
        return []
    cnts, hier = cv2.findContours(m, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    if hier is None:
        return []
    hier = hier[0]
    keep = max(min_area_px, 1e-9)
    # Parts touching only diagonally are traced as one self-touching contour; _rings splits it
    # into all its pieces. Holes then go to the outer that contains them.
    outers, holes = [], []
    for i, c in enumerate(cnts):
        if hier[i][3] == -1:
            outers += [r for r in _rings(c, simplify_px, +0.5) if abs(_area(r)) >= keep]
        else:
            holes += [r for r in _rings(c, simplify_px, -0.5) if abs(_area(r)) >= keep]
    shapes = [(o, []) for o in outers]
    for h in holes:
        probe = tuple(float(v) for v in h.mean(axis=0)) if len(h) else (0.0, 0.0)
        for o, hs in sorted(shapes, key=lambda s: abs(_area(s[0]))):      # the smallest outer that holds it
            if cv2.pointPolygonTest(o.astype(np.float32).reshape(-1, 1, 2), probe, False) >= 0:
                hs.append(h)
                break
    return shapes


def _area(ring):
    x, y = ring[:, 0], ring[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _rings(contour, simplify_px, grow):
    """
    Simplified rings of one contour, moved from pixel centres to pixel edges by growing
    grow px along the normals. A self-touching contour (parts meeting at a corner pixel)
    gives several rings; all are kept.
    """
    pts = contour.reshape(-1, 2).astype(np.float64)
    if len(pts) < 3:
        # a 1-2 pixel blob: its pixel square
        x0, y0 = pts.min(axis=0)
        x1, y1 = pts.max(axis=0)
        return [np.array([[x0 - 0.5, y0 - 0.5], [x1 + 0.5, y0 - 0.5], [x1 + 0.5, y1 + 0.5], [x0 - 0.5, y1 + 0.5]])]
    Polygon = optional.need("shapely.geometry", "regions to vector (outlines)").Polygon
    poly = Polygon(pts)
    if not poly.is_valid:
        poly = poly.buffer(0)
    poly = poly.buffer(grow, join_style=2, mitre_limit=2.0)
    if poly.is_empty:
        return []
    geoms = poly.geoms if poly.geom_type == "MultiPolygon" else [poly]
    out = []
    for g in geoms:
        p = np.asarray(g.exterior.coords)[:-1]
        approx = cv2.approxPolyDP(p.astype(np.float32).reshape(-1, 1, 2), max(0.01, simplify_px), True)
        ring = approx.reshape(-1, 2).astype(np.float64)
        if len(ring) >= 3:
            out.append(ring)
    return out


# ---------------------------------------------------------------------------
# curves
# ---------------------------------------------------------------------------


def smooth_path(ring: np.ndarray, corner_deg: float = 35.0, closed: bool = True):
    """
    Path segments through the ring's points: [("M", p), ("L", p) | ("C", c1, c2, p), ...].
    A vertex turning more than corner_deg is a corner (lines meet there); segments
    between two corners stay straight; the rest are Catmull-Rom curves.
    """
    n = len(ring)
    if n < 3 or corner_deg >= 180:
        segs = [("M", ring[0])] + [("L", p) for p in ring[1:]]
        return segs
    corner = np.zeros(n, bool)
    for i in range(n):
        if not closed and (i == 0 or i == n - 1):
            corner[i] = True
            continue
        a, b, c = ring[i - 1], ring[i], ring[(i + 1) % n]
        v1, v2 = b - a, c - b
        l1, l2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if l1 < 1e-9 or l2 < 1e-9:
            corner[i] = True
            continue
        ang = math.degrees(math.acos(float(np.clip(np.dot(v1, v2) / (l1 * l2), -1, 1))))
        corner[i] = ang > corner_deg
    segs = [("M", ring[0])]
    count = n if closed else n - 1
    for i in range(count):
        p1, p2 = ring[i], ring[(i + 1) % n]
        i2 = (i + 1) % n
        if corner[i] and corner[i2]:
            segs.append(("L", p2))
            continue
        p0 = ring[i - 1] if (closed or i > 0) else p1
        p3 = ring[(i + 2) % n] if (closed or i + 2 < n) else p2
        t1 = np.zeros(2) if corner[i] else (p2 - p0) / 6.0
        t2 = np.zeros(2) if corner[i2] else (p3 - p1) / 6.0
        segs.append(("C", p1 + t1, p2 - t2, p2))
    return segs


def flatten(segs, step_px: float = 2.0) -> np.ndarray:
    """Segments to a polyline (Beziers sampled about every step_px)."""
    pts = [np.asarray(segs[0][1], np.float64)]
    for s in segs[1:]:
        if s[0] == "L":
            pts.append(np.asarray(s[1], np.float64))
        else:
            p0 = pts[-1]
            c1, c2, p3 = (np.asarray(v, np.float64) for v in s[1:])
            length = np.linalg.norm(c1 - p0) + np.linalg.norm(c2 - c1) + np.linalg.norm(p3 - c2)
            k = max(2, int(length / max(step_px, 0.1)))
            for t in np.linspace(0, 1, k + 1)[1:]:
                u = 1 - t
                pts.append(u ** 3 * p0 + 3 * u * u * t * c1 + 3 * u * t * t * c2 + t ** 3 * p3)
    return np.asarray(pts)


# ---------------------------------------------------------------------------
# centerlines
# ---------------------------------------------------------------------------

_NB = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def centerlines(mask: np.ndarray, simplify_px: float = 1.0, spur_px: float = 6.0):
    """Skeleton of a mask as open polylines (pixel centres); spurs shorter than spur_px dropped."""
    skeletonize = optional.need("skimage.morphology", "regions to vector (centerlines)").skeletonize
    sk = skeletonize(mask.astype(bool))
    if not sk.any():
        return []
    H, W = sk.shape
    ys, xs = np.nonzero(sk)
    pix = set(zip(ys.tolist(), xs.tolist()))

    def nbrs(p):
        y, x = p
        return [(y + dy, x + dx) for dy, dx in _NB if (y + dy, x + dx) in pix]

    deg = {p: len(nbrs(p)) for p in pix}
    nodes = {p for p, d in deg.items() if d != 2}
    visited_edges = set()
    lines = []

    def walk(start, first):
        path = [start, first]
        visited_edges.add((start, first))
        visited_edges.add((first, start))
        prev, cur = start, first
        while cur not in nodes:
            nxt = [q for q in nbrs(cur) if q != prev and (cur, q) not in visited_edges]
            if not nxt:
                break
            prev, cur = cur, nxt[0]
            visited_edges.add((prev, cur))
            visited_edges.add((cur, prev))
            path.append(cur)
            if cur == start:
                break
        return path

    for n in nodes:
        for q in nbrs(n):
            if (n, q) not in visited_edges:
                lines.append((walk(n, q), deg[n] == 1))
    # closed loops without nodes
    rest = [p for p in pix if deg[p] == 2 and not any((p, q) in visited_edges for q in nbrs(p))]
    for p in rest:
        if any((p, q) in visited_edges for q in nbrs(p)):
            continue
        q = nbrs(p)[0]
        lines.append((walk(p, q), False))
    out = []
    for path, from_end in lines:
        arr = np.array([(x, y) for y, x in path], np.float64)
        length = float(np.linalg.norm(np.diff(arr, axis=0), axis=1).sum()) if len(arr) > 1 else 0.0
        end_to_end = deg.get(path[0], 0) == 1 or deg.get(path[-1], 0) == 1
        if end_to_end and length < spur_px and len(lines) > 1:
            continue                                   # a spur from the skeleton's noise
        if len(arr) >= 2:
            ap = cv2.approxPolyDP(arr.astype(np.float32).reshape(-1, 1, 2), max(0.01, simplify_px), False)
            out.append(ap.reshape(-1, 2).astype(np.float64))
    return out


def order_paths(paths):
    """
    Greedy nearest-neighbour order from the top-left corner, paths may be reversed.
    Returns ([(index, reversed), ...], travel between paths in px). A KD tree over the
    2N endpoints keeps it N log N (a line drawing gives tens of thousands of paths).
    """
    n = len(paths)
    if n == 0:
        return [], 0.0
    from scipy.spatial import cKDTree
    ends = np.empty((2 * n, 2))
    for i, p in enumerate(paths):
        ends[2 * i], ends[2 * i + 1] = p[0], p[-1]
    tree = cKDTree(ends)
    used = np.zeros(n, bool)
    pos = np.array([0.0, 0.0])
    out, travel = [], 0.0
    for _ in range(n):
        k = 8
        while True:
            dist, idx = tree.query(pos, k=min(k, 2 * n))
            dist, idx = np.atleast_1d(dist), np.atleast_1d(idx)
            free = [(d, j) for d, j in zip(dist, idx) if j < 2 * n and not used[j // 2]]
            if free or k >= 2 * n:
                break
            k *= 4
        d, j = free[0]
        i, rev = int(j // 2), bool(j % 2)
        used[i] = True
        out.append((i, rev))
        travel += float(d)
        pos = paths[i][0] if rev else paths[i][-1]
    return out, travel


def kerf_offset(outer, holes, offset):
    """Offset a shape (outer + holes) by offset units with mitred corners; returns [(outer, holes), ...]."""
    Polygon = optional.need("shapely.geometry", "regions to vector (outlines)").Polygon
    poly = Polygon(outer, holes).buffer(0).buffer(offset, join_style=2, mitre_limit=2.0)
    if poly.is_empty:
        return []
    geoms = poly.geoms if poly.geom_type == "MultiPolygon" else [poly]
    return [(np.asarray(g.exterior.coords)[:-1], [np.asarray(h.coords)[:-1] for h in g.interiors]) for g in geoms]


# ---------------------------------------------------------------------------
# document
# ---------------------------------------------------------------------------


def build(labels: np.ndarray, table: dict, region_ids, mode: str = "outline", simplify_px: float = 1.0,
          corner_deg: float = 35.0, min_area_px: float = 16.0, spur_px: float = 6.0, width_mm: float = 0.0,
          kerf_mm: float = 0.0):
    """
    Vector document: {"width", "height", "unit" ("px" or "mm"), "scale" (unit per px), "layers":
    {group: [{"name", "rings": [segments...] (closed) | "lines": [polyline...] (open)}]}, "stats"}.
    """
    H, W = labels.shape
    scale = (width_mm / W) if width_mm > 0 else 1.0
    unit = "mm" if width_mm > 0 else "px"
    names = {int(r["region_id"]): r for r in table.get("regions", [])}
    layers, stats = {}, {"paths": 0, "nodes": 0, "length": 0.0, "travel": 0.0}
    all_lines = []
    for rid in region_ids:
        r = names.get(int(rid), {"name": f"r{rid}", "group_id": "regions"})
        group = str(r.get("group_id") or "regions")
        item = {"name": str(r.get("name", f"r{rid}")), "region_id": int(rid)}
        mask = labels == rid
        if mode == "outline":
            rings = []
            for outer, holes in trace_mask(mask, simplify_px, min_area_px):
                if kerf_mm and width_mm > 0:
                    for o2, h2 in kerf_offset(outer, holes, kerf_mm / 2.0 / scale):
                        rings += [[("M", o2[0])] + [("L", p) for p in o2[1:]]]
                        rings += [[("M", h[0])] + [("L", p) for p in h[1:]] for h in h2]
                else:
                    rings.append(smooth_path(outer, corner_deg))
                    rings += [smooth_path(h, corner_deg) for h in holes]
            item["rings"] = [[(s[0], *[np.asarray(v) * scale for v in s[1:]]) for s in ring] for ring in rings]
            stats["paths"] += len(rings)
            stats["nodes"] += sum(len(g) for g in rings)
        else:
            lines = centerlines(mask, simplify_px, spur_px)
            all_lines += [(group, item, ln) for ln in lines]
            item["lines"] = []
        if item.get("rings") or mode != "outline":
            layers.setdefault(group, []).append(item)
    if mode != "outline":
        order, travel = order_paths([ln for _, _, ln in all_lines])
        for idx, rev in order:
            _, it, ln = all_lines[idx]
            ln = ln[::-1] if rev else ln
            it["lines"].append(ln * scale)
            stats["length"] += float(np.linalg.norm(np.diff(ln, axis=0), axis=1).sum()) * scale
            stats["nodes"] += len(ln)
        stats["paths"] = len(order)
        stats["travel"] = travel * scale
    return {"width": W * scale, "height": H * scale, "unit": unit, "scale": scale, "layers": layers, "stats": stats}


def _fmt(v):
    return f"{v:.3f}".rstrip("0").rstrip(".")


def to_svg(doc, colors=None) -> str:
    w, h, u = doc["width"], doc["height"], doc["unit"]
    size = f'width="{_fmt(w)}mm" height="{_fmt(h)}mm"' if u == "mm" else f'width="{_fmt(w)}" height="{_fmt(h)}"'
    out = ['<?xml version="1.0" encoding="UTF-8"?>',
           f'<svg xmlns="http://www.w3.org/2000/svg" {size} viewBox="0 0 {_fmt(w)} {_fmt(h)}">']
    for gi, (group, items) in enumerate(doc["layers"].items()):
        col = (colors or {}).get(group, "#%02x%02x%02x" % tuple(int(c) for c in _palette(gi)))
        out.append(f'  <g id="{_xml(group)}">')
        for it in items:
            if "rings" in it:
                d = " ".join(_svg_d(r) + " Z" for r in it["rings"])
                out.append(f'    <path id="{_xml(it["name"])}" d="{d}" fill="{col}" fill-rule="evenodd" stroke="none"/>')
            else:
                for k, ln in enumerate(it["lines"]):
                    pts = " ".join(f"{_fmt(x)},{_fmt(y)}" for x, y in ln)
                    sw = 0.2 if u == "mm" else 1.0
                    out.append(f'    <polyline id="{_xml(it["name"])}_{k + 1:03d}" points="{pts}" fill="none" '
                               f'stroke="{col}" stroke-width="{sw}"/>')
        out.append("  </g>")
    out.append("</svg>")
    return "\n".join(out) + "\n"


def _svg_d(segs):
    parts = []
    for s in segs:
        if s[0] == "M":
            parts.append(f"M {_fmt(s[1][0])} {_fmt(s[1][1])}")
        elif s[0] == "L":
            parts.append(f"L {_fmt(s[1][0])} {_fmt(s[1][1])}")
        else:
            parts.append("C " + " ".join(f"{_fmt(p[0])} {_fmt(p[1])}" for p in s[1:]))
    return " ".join(parts)


def _xml(s):
    return str(s).replace("&", "&amp;").replace('"', "&quot;").replace("<", "&lt;").replace(">", "&gt;")


def _palette(i):
    import colorsys
    r, g, b = colorsys.hsv_to_rgb((0.07 + i * 0.618033988749895) % 1.0, 0.6, 0.9)
    return (r * 255, g * 255, b * 255)


def to_dxf(doc, flatten_step: float | None = None) -> str:
    """DXF R12 (ASCII): one layer per group, closed POLYLINEs for outlines, open ones for centerlines.
    Y is flipped (DXF y goes up). Curves are flattened (step in document units)."""
    h = doc["height"]
    step = flatten_step if flatten_step else (0.2 if doc["unit"] == "mm" else 1.0)
    lines = ["0", "SECTION", "2", "HEADER", "9", "$ACADVER", "1", "AC1009",
             "9", "$MEASUREMENT", "70", "1" if doc["unit"] == "mm" else "0",
             "9", "$EXTMIN", "10", "0.0", "20", "0.0", "9", "$EXTMAX", "10", _fmt(doc["width"]), "20", _fmt(h),
             "0", "ENDSEC", "0", "SECTION", "2", "TABLES", "0", "TABLE", "2", "LAYER", "70", str(len(doc["layers"]))]
    for gi, group in enumerate(doc["layers"]):
        lines += ["0", "LAYER", "2", _dxf_name(group), "70", "0", "62", str(1 + gi % 254), "6", "CONTINUOUS"]
    lines += ["0", "ENDTAB", "0", "ENDSEC", "0", "SECTION", "2", "ENTITIES"]
    for group, items in doc["layers"].items():
        lay = _dxf_name(group)
        for it in items:
            polys = [(flatten(r, step), True) for r in it.get("rings", [])]
            polys += [(ln, False) for ln in it.get("lines", [])]
            for pts, closed in polys:
                if closed and len(pts) > 1 and np.allclose(pts[0], pts[-1]):
                    pts = pts[:-1]
                lines += ["0", "POLYLINE", "8", lay, "66", "1", "70", "1" if closed else "0"]
                for x, y in pts:
                    lines += ["0", "VERTEX", "8", lay, "10", _fmt(x), "20", _fmt(h - y)]
                lines += ["0", "SEQEND", "8", lay]
    lines += ["0", "ENDSEC", "0", "EOF"]
    return "\n".join(lines) + "\n"


def _dxf_name(s):
    import re
    return re.sub(r"[^A-Za-z0-9_\-]", "_", str(s))[:31] or "0"


def to_pdf(doc, path: str):
    """PDF with one optional content layer per group (Illustrator shows them as layers); mm or px (1 px = 1 pt)."""
    fitz = optional.need("fitz", "regions to vector (pdf)")
    k = 72 / 25.4 if doc["unit"] == "mm" else 1.0
    pdf = fitz.open()
    page = pdf.new_page(width=doc["width"] * k, height=doc["height"] * k)
    for gi, (group, items) in enumerate(doc["layers"].items()):
        oc = pdf.add_ocg(str(group), on=True)
        col = tuple(c / 255 for c in _palette(gi))
        for it in items:
            sh = page.new_shape()
            for ring in it.get("rings", []):
                cur = None
                for s in ring:
                    if s[0] == "M":
                        cur = fitz.Point(*(s[1] * k))
                        start = cur
                    elif s[0] == "L":
                        nxt = fitz.Point(*(s[1] * k))
                        sh.draw_line(cur, nxt)
                        cur = nxt
                    else:
                        c1, c2, p = (fitz.Point(*(v * k)) for v in s[1:])
                        sh.draw_bezier(cur, c1, c2, p)
                        cur = p
                if cur is not None and (cur.x, cur.y) != (start.x, start.y):
                    sh.draw_line(cur, start)
            if it.get("rings"):
                sh.finish(fill=col, color=None, even_odd=True, closePath=True, oc=oc)
            for ln in it.get("lines", []):
                sh.draw_polyline([fitz.Point(*(p * k)) for p in ln])
                sh.finish(color=col, width=0.3 if doc["unit"] == "mm" else 1.0, closePath=False, oc=oc)
            sh.commit()
    pdf.save(path)
    pdf.close()


def render_preview(doc, H, W, background=None) -> np.ndarray:
    """RGB float preview of the paths at matrix resolution."""
    img = np.zeros((H, W, 3), np.float32) if background is None else np.asarray(background, np.float32) * 0.35
    s = 1.0 / doc["scale"]
    thick = max(1, W // 1200)
    for gi, (group, items) in enumerate(doc["layers"].items()):
        col = tuple(float(c) / 255 for c in _palette(gi))
        for it in items:
            for ring in it.get("rings", []):
                pts = np.round(flatten([(q[0], *[np.asarray(v) * s for v in q[1:]]) for q in ring], 3.0)).astype(np.int32)
                cv2.polylines(img, [pts], True, col, thick, cv2.LINE_AA)
            for ln in it.get("lines", []):
                cv2.polylines(img, [np.round(ln * s).astype(np.int32)], False, col, thick, cv2.LINE_AA)
    return np.clip(img, 0, 1)


def write_all(doc, folder: str, prefix: str, formats=("svg", "pdf", "dxf")):
    """Write the document; the next free counter keeps earlier files. Returns the written paths."""
    sub, prefix = os.path.split(prefix.replace("\\", "/"))    # 'project/regions' = a subfolder, as in ComfyUI's save nodes
    if sub and not os.path.isabs(sub) and ".." not in sub.split("/"):
        folder = os.path.join(folder, sub)
    os.makedirs(folder, exist_ok=True)
    n = 1
    while any(os.path.exists(os.path.join(folder, f"{prefix}_{n:05d}.{f}")) for f in formats):
        n += 1
    out = []
    for f in formats:
        p = os.path.join(folder, f"{prefix}_{n:05d}.{f}")
        if f == "svg":
            open(p, "w", encoding="utf-8").write(to_svg(doc))
        elif f == "dxf":
            open(p, "w", encoding="ascii", errors="replace").write(to_dxf(doc))
        elif f == "pdf":
            to_pdf(doc, p)
        out.append(p)
    return out
