"""
facade_core.py

Array logic for the Facade nodes (category KUBAKUB/facade). Only numpy, opencv
and scipy, no ComfyUI imports, so tests/test_facade_masks.py can run it without
starting ComfyUI or loading a model.

Convention used throughout: a region partition is an int32 label map the size
of the facade matrix, -1 for unassigned pixels and 0..N-1 for regions. Region
ids are assigned in reading order (rows top to bottom, left to right inside a
row) and equal the index of the region in the MASK batch.
"""

from __future__ import annotations

import colorsys
import fnmatch
import os
import re

import cv2
import numpy as np

ATLAS_FORMAT = "kubakub.facade.atlas"
ATLAS_VERSION = 1

MODES = ("color_regions", "line_drawing", "mask_folder")
GROUP_BY = ("stem", "folder")
LINE_POLARITIES = ("auto", "dark_lines", "light_lines")

_CROSS = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------

def to_uint8_rgb(image: np.ndarray) -> np.ndarray:
    """HxWxC float image in 0..1 to HxWx3 uint8."""
    a = np.asarray(image, dtype=np.float32)
    if a.ndim == 2:
        a = a[..., None]
    a = a[..., :3]
    if a.shape[2] == 1:
        a = np.repeat(a, 3, axis=2)
    return np.clip(a * 255.0 + 0.5, 0, 255).astype(np.uint8)


def _pack(rgb: np.ndarray) -> np.ndarray:
    rgb = rgb.astype(np.int32)
    return (rgb[..., 0] << 16) | (rgb[..., 1] << 8) | rgb[..., 2]


def _hex(color) -> str:
    r, g, b = (int(round(float(c))) for c in color)
    return f"#{r:02x}{g:02x}{b:02x}"


def parse_color(text: str):
    """'#2b3a55', '2b3a55' or '43, 58, 85' to an (r, g, b) tuple, else None."""
    t = text.strip().lower()
    m = re.fullmatch(r"#?([0-9a-f]{6})", t)
    if m:
        v = int(m.group(1), 16)
        return ((v >> 16) & 255, (v >> 8) & 255, v & 255)
    parts = re.split(r"[\s,;]+", t)
    if len(parts) == 3 and all(p.isdigit() for p in parts):
        vals = tuple(int(p) for p in parts)
        if all(0 <= v <= 255 for v in vals):
            return vals
    return None


def parse_color_names(text: str):
    """Lines like '#2b3a55 = windows' or '200,180,160 = wall'. Returns ([(rgb, name)], [errors])."""
    entries, errors = [], []
    for n, raw in enumerate((text or "").splitlines(), 1):
        line = raw.strip()  # '#' starts a hex colour here, so comments use '//'
        if not line or line.startswith("//"):
            continue
        if "=" not in line:
            errors.append(f"color_names line {n}: expected '<color> = <name>', got '{line}'")
            continue
        key, name = (s.strip() for s in line.split("=", 1))
        rgb = parse_color(key)
        if rgb is None or not re.fullmatch(r"[A-Za-z0-9_\-]+", name or ""):
            errors.append(f"color_names line {n}: cannot read '{line}' "
                          "(color as #rrggbb or r,g,b; name of letters, digits, _ or -)")
            continue
        entries.append((rgb, name))
    return entries, errors


# --------------------------------------------------------------------------
# mode 1: flat colour regions
# --------------------------------------------------------------------------

def build_palette(rgb: np.ndarray, tolerance: float, min_pixels: int, max_colors: int = 256):
    """
    The distinct flat colours of a matrix, most common first.

    Only pixels whose four neighbours have exactly the same colour are counted.
    Anti-aliased edges between two fills are one or two pixels wide and almost
    never pass that test, so they do not turn into palette entries of their
    own, however long the edges are in total. A colour closer than `tolerance`
    (euclidean, 0..255 units) to a more common one is folded into it.
    """
    packed = _pack(rgb)
    uniform = np.ones(packed.shape, dtype=bool)
    uniform[1:, :] &= packed[1:, :] == packed[:-1, :]
    uniform[:-1, :] &= packed[:-1, :] == packed[1:, :]
    uniform[:, 1:] &= packed[:, 1:] == packed[:, :-1]
    uniform[:, :-1] &= packed[:, :-1] == packed[:, 1:]
    # A noisy (e.g. JPEG) matrix has almost no uniform pixels; count all of them then.
    sample = packed[uniform] if uniform.mean() > 0.10 else packed.ravel()

    uniq, counts = np.unique(sample, return_counts=True)
    order = np.argsort(-counts, kind="stable")
    colors = np.stack([(uniq >> 16) & 255, (uniq >> 8) & 255, uniq & 255], axis=1).astype(np.float32)

    palette = []
    for i in order:
        if counts[i] < min_pixels and palette:
            break
        c = colors[i]
        if palette and np.min(np.linalg.norm(np.asarray(palette) - c, axis=1)) <= tolerance:
            continue
        palette.append(c)
        if len(palette) >= max_colors:
            break
    return np.asarray(palette, dtype=np.float32)


def quantize_to_palette(rgb: np.ndarray, palette: np.ndarray, max_distance: float) -> np.ndarray:
    """
    Index of the nearest palette colour for every pixel (HxW int32), or -1 where
    no palette colour is within max_distance.

    Anti-aliased edge pixels are a mix of the two fills they sit between, and
    that mix is often closest to some *third* colour (window blue + wall beige
    lands on the ground floor grey). Snapping them to it would ring every
    window with a thin region of the wrong colour, so they are left at -1 and
    given to the spatially nearest region instead (fill_unassigned).
    """
    packed = _pack(rgb).ravel()
    uniq, inverse = np.unique(packed, return_inverse=True)
    colors = np.stack([(uniq >> 16) & 255, (uniq >> 8) & 255, uniq & 255], axis=1).astype(np.float32)
    nearest = np.empty(len(uniq), dtype=np.int32)
    step = 65536
    for s in range(0, len(uniq), step):
        d = ((colors[s:s + step, None, :] - palette[None, :, :]) ** 2).sum(-1)
        idx = d.argmin(axis=1)
        far = d[np.arange(len(idx)), idx] > max_distance ** 2
        nearest[s:s + step] = np.where(far, -1, idx)
    return nearest[inverse].reshape(rgb.shape[:2])


def label_color_regions(rgb, tolerance, min_area, split_disconnected):
    """Returns (labels, meta) where meta[i] = {'color': (r,g,b)}. Edge pixels come back as -1."""
    palette = build_palette(rgb, tolerance, max(16, min_area // 2))
    # a floor under the match distance, so tolerance 0 still accepts a flat fill with tiny noise
    pal_idx = quantize_to_palette(rgb, palette, max(float(tolerance), 8.0))
    labels = np.full(pal_idx.shape, -1, dtype=np.int32)
    meta = []
    for p, color in enumerate(palette):
        m = pal_idx == p
        if not m.any():
            continue
        if split_disconnected:
            k, cc = cv2.connectedComponents(m.astype(np.uint8), connectivity=4, ltype=cv2.CV_32S)
            labels[m] = cc[m] - 1 + len(meta)
            meta.extend({"color": tuple(color)} for _ in range(k - 1))
        else:
            labels[m] = len(meta)
            meta.append({"color": tuple(color)})
    return labels, meta


# --------------------------------------------------------------------------
# mode 2: line drawing
# --------------------------------------------------------------------------

def label_line_cells(rgb, line_threshold, polarity, gap_close_px):
    """
    Closed cells between the lines of a line drawing. Line pixels come back as -1
    and are handed to the nearest cell afterwards by fill_unassigned().
    """
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    dark = gray < line_threshold
    if polarity == "dark_lines":
        lines = dark
    elif polarity == "light_lines":
        lines = ~dark
    else:  # auto: lines are the minority of the drawing
        lines = dark if dark.mean() <= 0.5 else ~dark
    lines = lines.astype(np.uint8)
    if gap_close_px > 0:
        k = 2 * int(gap_close_px) + 1
        lines = cv2.dilate(lines, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    k, cc = cv2.connectedComponents((lines == 0).astype(np.uint8), connectivity=4, ltype=cv2.CV_32S)
    labels = cc - 1  # background of connectedComponents (the lines) becomes -1
    return labels.astype(np.int32), [{} for _ in range(k - 1)]


def fill_unassigned(labels: np.ndarray) -> np.ndarray:
    """Give every -1 pixel the label of the nearest labelled pixel."""
    holes = labels < 0
    if not holes.any() or holes.all():
        return labels
    from scipy import ndimage
    _, (iy, ix) = ndimage.distance_transform_edt(holes, return_indices=True)
    out = labels.copy()
    out[holes] = labels[iy[holes], ix[holes]]
    return out


# --------------------------------------------------------------------------
# mode 3: folder of masks (After Effects export)
# --------------------------------------------------------------------------

def _natural_key(name):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", name)]


def clean_folder_path(path: str) -> str:
    """Accept paths pasted from Explorer's 'Copy as path', which adds quotes."""
    return os.path.expandvars(os.path.expanduser((path or "").strip().strip('"').strip("'")))


def list_mask_files(folder: str, recursive: bool = False):
    """(folder, [relative paths with forward slashes]) of the .png files, natural order."""
    folder = clean_folder_path(folder)
    if not folder or not os.path.isdir(folder):
        return folder, []
    if not recursive:
        files = [f for f in os.listdir(folder) if f.lower().endswith(".png")]
    else:
        files = []
        for root, dirs, names in os.walk(folder):
            dirs[:] = sorted(d for d in dirs if not d.startswith("."))
            rel = os.path.relpath(root, folder).replace("\\", "/")
            for f in names:
                if f.lower().endswith(".png"):
                    files.append(f if rel == "." else f"{rel}/{f}")
    return folder, sorted(files, key=_natural_key)


def parse_patterns(text: str):
    """'M_Pilasters, M_Stone_*' or one per line -> lower case fnmatch patterns. // starts a comment."""
    out = []
    for line in (text or "").splitlines():
        line = line.split("//", 1)[0]
        out += [p.strip().lower() for p in line.split(",") if p.strip()]
    return out


def matches(patterns, rel_path: str) -> bool:
    """A pattern matches the file stem ('M_Pilasters'), the file name or the relative path."""
    rel = rel_path.lower()
    name = rel.rsplit("/", 1)[-1]
    cands = (os.path.splitext(name)[0], name, rel, os.path.splitext(rel)[0])
    return any(fnmatch.fnmatchcase(c, p) for p in patterns for c in cands)


def read_mask_png(path: str) -> np.ndarray:
    """
    One mask PNG as a bool array. Uses the alpha channel when it actually varies
    (After Effects exports with transparency), otherwise the brightness.
    """
    # imdecode instead of imread: imread cannot open non-ASCII paths on Windows.
    data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"cannot decode {path}")
    scale = 65535.0 if img.dtype == np.uint16 else 255.0
    img = img.astype(np.float32) / scale
    if img.ndim == 3 and img.shape[2] == 4 and img[..., 3].min() < img[..., 3].max():
        value = img[..., 3]
    elif img.ndim == 3:
        value = img[..., :3].mean(axis=2)
    else:
        value = img
    return value > 0.5


def group_name_from_stem(stem: str) -> str:
    """'windows_03' -> 'windows', 'pier-12' -> 'pier', 'cornice' -> 'cornice'."""
    base = re.sub(r"[\s_\-\.]*\d+$", "", stem)
    return base or stem


def split_components(mask: np.ndarray, min_area: int = 1):
    """Connected parts (8-connected) of a bool mask in reading order; specks under min_area dropped."""
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    parts = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= max(1, min_area)]
    if not parts:
        return []
    boxes = np.asarray([[stats[i, 0], stats[i, 1], stats[i, 0] + stats[i, 2], stats[i, 1] + stats[i, 3]]
                        for i in parts], dtype=np.int64)
    return [lab == parts[j] for j in reading_order(boxes)]


def label_mask_folder(folder, width, height, recursive=False, scope_masks="", tag_only_masks="",
                      split_masks="", group_by="stem", min_part_area=1):
    """
    Returns (labels, meta, notes, scope, tag_masks) for the PNGs of a folder,
    see label_masks(). The file name without extension is the mask name; with
    group_by="folder" the subfolder name is the group.
    """
    folder, files = list_mask_files(folder, recursive)
    if not folder:
        raise ValueError("mode is mask_folder but mask_folder is empty.")
    if not os.path.isdir(folder):
        raise ValueError(f"mask_folder does not exist: {folder}")
    if not files:
        raise ValueError(f"no .png files in mask_folder{' or its subfolders' if recursive else ''}: {folder}")
    if group_by not in GROUP_BY:
        raise ValueError(f"group_by must be one of {GROUP_BY}")

    top = os.path.basename(folder.rstrip("\\/")) or "masks"
    entries = []
    for f in files:
        group = None
        if group_by == "folder":
            group = f.rsplit("/", 1)[0].replace("/", "_") if "/" in f else top
        entries.append({"source": f, "name": os.path.splitext(f.rsplit("/", 1)[-1])[0],
                        "mask": read_mask_png(os.path.join(folder, f)), "group": group})
    return label_masks(entries, width, height, scope_masks=scope_masks, tag_only_masks=tag_only_masks,
                       split_masks=split_masks, min_part_area=min_part_area)


def label_masks(entries, width, height, scope_masks="", tag_only_masks="", split_masks="",
                min_part_area=1):
    """
    Returns (labels, meta, notes, scope, tag_masks).

    entries: dicts with "source" (file path or 'mask 3', matched by the
    patterns), "name" (region name), "mask" (HxW bool) and "group" (None =
    the name without its trailing number, 'windows_03' -> 'windows').

    Where masks overlap, the smaller mask wins, so a window cut out of a wall
    mask stays a window. Masks of another size are resized (nearest) to
    width x height. Scope masks are no regions: their union limits all regions
    (outside becomes -1). Tag-only masks are no regions either, they only tag
    the regions they cover. Split masks give one region per connected part,
    named <name>_01.. in reading order and grouped as <name>. tag_masks lists
    (name, bool mask) of every non-scope mask, for region_tags().
    """
    scope_p, tag_p, split_p = (parse_patterns(t) for t in (scope_masks, tag_only_masks, split_masks))
    notes, regions, tag_masks, scope = [], [], [], None
    used = {"scope": 0, "tag": 0, "split": 0}
    for e in entries:
        f, stem, m = e["source"], e["name"], np.asarray(e["mask"], dtype=bool)
        if m.shape != (height, width):
            notes.append(f"{f}: {m.shape[1]}x{m.shape[0]} resized to {width}x{height}")
            m = cv2.resize(m.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST) > 0
        if not m.any():
            notes.append(f"{f}: empty mask, skipped")
            continue
        if matches(scope_p, f) or matches(scope_p, stem):
            scope = m if scope is None else (scope | m)
            used["scope"] += 1
            continue
        tag_masks.append((stem, m))
        if matches(tag_p, f) or matches(tag_p, stem):
            used["tag"] += 1
            continue
        if matches(split_p, f) or matches(split_p, stem):
            used["split"] += 1
            parts = split_components(m, min_part_area)
            notes.append(f"{f}: split into {len(parts)} parts")
            for j, part in enumerate(parts):
                regions.append((part, {"name": f"{stem}_{j + 1:02d}", "group": stem,
                                       "source": f, "own_mask": stem}))
            continue
        group = e.get("group") or group_name_from_stem(stem)
        regions.append((m, {"name": stem, "group": group, "source": f, "own_mask": stem}))

    for kind, pats in (("scope", scope_p), ("tag", tag_p), ("split", split_p)):
        if pats and not used[kind]:
            notes.append(f"{kind} patterns '{', '.join(pats)}' match no mask")
    if scope is None and scope_p:
        notes.append("no scope mask found, nothing is limited")

    labels = np.full((height, width), -1, dtype=np.int32)
    coverage = np.zeros((height, width), dtype=np.uint16)
    meta = []
    for m, info in sorted(regions, key=lambda r: -int(r[0].sum())):
        labels[m] = len(meta)
        coverage += m
        meta.append(info)
    overlap = int((coverage > 1).sum())
    if overlap:
        notes.append(f"{overlap} px covered by more than one mask; the smaller mask won there")
    if scope is not None:
        outside = int(((labels >= 0) & ~scope).sum())
        labels[~scope] = -1
        notes.append(f"scope {int(scope.sum())} px: {outside} region px outside it removed, "
                     f"{int((scope & (labels < 0)).sum())} px inside it belong to no region")
    return labels, meta, notes, scope, tag_masks


def region_tags(labels, n, tag_masks, own_masks, min_cover=0.5):
    """
    Per region, the names of the masks covering at least min_cover of it,
    broadest first, without the region's own mask. This is the hierarchy the
    overlapping masks describe (facade > floor > ...), for group rules in the plan.
    """
    areas = np.bincount(labels[labels >= 0], minlength=n).astype(np.float64)
    tags = [[] for _ in range(n)]
    for name, m in sorted(tag_masks, key=lambda nm: -int(nm[1].sum())):
        inside = labels[m]
        counts = np.bincount(inside[inside >= 0], minlength=n)
        for i in np.flatnonzero(counts >= min_cover * np.maximum(areas, 1)):
            if name != own_masks[i]:
                tags[i].append(name)
    return tags


# --------------------------------------------------------------------------
# small regions
# --------------------------------------------------------------------------

def _bboxes(labels, n):
    """(n, 4) array of x0, y0, x1, y1 (exclusive), -1 rows for empty labels."""
    from scipy import ndimage
    out = np.full((n, 4), -1, dtype=np.int64)
    for i, sl in enumerate(ndimage.find_objects(labels + 1, max_label=n)):
        if sl is not None:
            out[i] = (sl[1].start, sl[0].start, sl[1].stop, sl[0].stop)
    return out


def handle_small_regions(labels, min_area, merge):
    """
    Regions under min_area are merged into the neighbour they share the longest
    border with (merge=True) or set to -1 (merge=False). Smallest first, so specks
    collapse into each other before anything touches a real region.
    Returns (labels, kept_old_ids, n_merged, n_dropped).
    """
    n = int(labels.max()) + 1 if labels.size and labels.max() >= 0 else 0
    if n == 0:
        return labels, np.zeros(0, dtype=np.int64), 0, 0
    labels = labels.copy()
    areas = np.bincount(labels[labels >= 0], minlength=n).astype(np.int64)
    small = [int(i) for i in np.argsort(areas, kind="stable") if 0 < areas[i] < min_area]
    merged = dropped = 0

    if small and not merge:
        drop = np.zeros(n, dtype=bool)
        drop[small] = True
        labels[(labels >= 0) & drop[np.maximum(labels, 0)]] = -1
        areas[small] = 0
        dropped = len(small)
    elif small:
        boxes = _bboxes(labels, n)
        h, w = labels.shape
        for i in small:
            if areas[i] == 0 or areas[i] >= min_area:
                continue
            x0, y0, x1, y1 = boxes[i]
            x0, y0, x1, y1 = max(0, x0 - 1), max(0, y0 - 1), min(w, x1 + 1), min(h, y1 + 1)
            sub = labels[y0:y1, x0:x1]
            m = sub == i
            ring = (cv2.dilate(m.astype(np.uint8), _CROSS) > 0) & ~m
            neigh = sub[ring]
            neigh = neigh[neigh >= 0]
            if neigh.size == 0:
                continue  # nothing to merge into (isolated); keep it
            target = int(np.bincount(neigh).argmax())
            sub[m] = target
            areas[target] += areas[i]
            areas[i] = 0
            tb = boxes[target]
            boxes[target] = (min(tb[0], boxes[i][0]), min(tb[1], boxes[i][1]),
                             max(tb[2], boxes[i][2]), max(tb[3], boxes[i][3]))
            merged += 1

    kept = np.flatnonzero(areas > 0)
    remap = np.full(n, -1, dtype=np.int32)
    remap[kept] = np.arange(len(kept), dtype=np.int32)
    out = np.where(labels >= 0, remap[np.maximum(labels, 0)], -1).astype(np.int32)
    return out, kept, merged, dropped


# --------------------------------------------------------------------------
# region stats, reading order, grouping
# --------------------------------------------------------------------------

def region_stats(labels, n):
    boxes = _bboxes(labels, n)
    areas = np.bincount(labels[labels >= 0], minlength=n)
    ys, xs = np.nonzero(labels >= 0)
    lab = labels[ys, xs]
    cx = np.bincount(lab, weights=xs, minlength=n) / np.maximum(areas, 1)
    cy = np.bincount(lab, weights=ys, minlength=n) / np.maximum(areas, 1)
    return boxes, areas, np.stack([cx, cy], axis=1)


def reading_order(boxes):
    """
    Order regions in rows top to bottom, then left to right. A region joins the
    current row while its top edge is within half the row's smallest height of
    the row's first top edge, so a row of windows drawn a pixel or two apart
    still reads as one row.
    """
    n = len(boxes)
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    tops = boxes[:, 1]
    heights = np.maximum(boxes[:, 3] - boxes[:, 1], 1)
    by_top = np.argsort(tops, kind="stable")
    rows, row, row_top, row_h = [], [], None, None
    for i in by_top:
        if row and tops[i] - row_top > 0.5 * row_h:
            rows.append(row)
            row = []
        if not row:
            row_top, row_h = tops[i], heights[i]
        row.append(i)
        row_h = min(row_h, heights[i])
    rows.append(row)
    return np.asarray([i for r in rows for i in sorted(r, key=lambda j: boxes[j, 0])], dtype=np.int64)


def _thumbnails(labels, boxes, size=24):
    thumbs = np.zeros((len(boxes), size * size), dtype=np.float32)
    for i, (x0, y0, x1, y1) in enumerate(boxes):
        crop = (labels[y0:y1, x0:x1] == i).astype(np.float32)
        thumbs[i] = cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA).ravel() > 0.5
    return thumbs


def group_by_shape(labels, boxes, keys, tolerance):
    """
    Group regions that are near copies of each other: same key (palette colour in
    colour mode), bbox width and height within `tolerance` (relative), and mask
    shapes overlapping with IoU >= 1 - tolerance after scaling to a common size.
    Returns an int group index per region (0-based, in order of first member).
    """
    n = len(boxes)
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    w = (boxes[:, 2] - boxes[:, 0]).astype(np.float32)
    h = (boxes[:, 3] - boxes[:, 1]).astype(np.float32)
    rel = lambda a: np.abs(a[:, None] - a[None, :]) / np.maximum(np.maximum(a[:, None], a[None, :]), 1)
    cand = (rel(w) <= tolerance) & (rel(h) <= tolerance)
    if keys is not None:
        k = np.asarray(keys)
        cand &= k[:, None] == k[None, :]
    thumbs = _thumbnails(labels, boxes)
    inter = thumbs @ thumbs.T
    s = thumbs.sum(axis=1)
    iou = inter / np.maximum(s[:, None] + s[None, :] - inter, 1)
    cand &= iou >= 1.0 - tolerance

    parent = np.arange(n)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a, b in np.argwhere(np.triu(cand, 1)):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    roots = np.array([find(i) for i in range(n)])
    _, first, index = np.unique(roots, return_index=True, return_inverse=True)
    # renumber groups by their first member so g1 is the top-left group
    rank = np.argsort(np.argsort(first))
    return rank[index]


# --------------------------------------------------------------------------
# the whole atlas
# --------------------------------------------------------------------------

def build_atlas(image, mode="color_regions", min_region_area=400, merge_small_regions=True,
                color_tolerance=24.0, split_disconnected=True, group_tolerance=0.08,
                line_threshold=0.5, line_polarity="auto", line_gap_close_px=1,
                mask_folder="", color_names="", recursive=False, scope_masks="",
                tag_only_masks="", split_masks="", group_by="stem", scope=None,
                with_scope=False):
    """
    Split a facade matrix into regions.

    image: HxWxC float array in 0..1.
    scope: optional HxW bool array; pixels outside it are never part of a region
    (combined with the scope masks of mask_folder mode).
    Returns (labels, atlas) where labels is the int32 partition (-1 = unassigned)
    and atlas is the JSON-ready dict described in the README. With with_scope,
    returns (labels, atlas, scope) where scope is the bool array used or None.
    """
    if mode not in MODES:
        raise ValueError(f"unknown mode '{mode}', expected one of {MODES}")
    rgb = to_uint8_rgb(image)
    height, width = rgb.shape[:2]
    notes = []
    tag_masks = []
    if scope is not None:
        scope = np.asarray(scope, dtype=bool)
        if scope.shape != (height, width):
            raise ValueError(f"scope is {scope.shape[1]}x{scope.shape[0]}, the matrix {width}x{height}")

    if mode == "color_regions":
        labels, meta = label_color_regions(rgb, float(color_tolerance), int(min_region_area),
                                           bool(split_disconnected))
        labels = fill_unassigned(labels)
    elif mode == "line_drawing":
        labels, meta = label_line_cells(rgb, float(line_threshold), line_polarity, int(line_gap_close_px))
        labels = fill_unassigned(labels)
    else:
        labels, meta, folder_notes, folder_scope, tag_masks = label_mask_folder(
            mask_folder, width, height, recursive=recursive, scope_masks=scope_masks,
            tag_only_masks=tag_only_masks, split_masks=split_masks, group_by=group_by,
            min_part_area=int(min_region_area))
        notes.extend(folder_notes)
        if folder_scope is not None:
            scope = folder_scope if scope is None else (scope & folder_scope)
    return _finish_atlas(labels, meta, notes, tag_masks, scope, mode, min_region_area,
                         merge_small_regions, color_names, color_tolerance, group_tolerance, with_scope)


def _finish_atlas(labels, meta, notes, tag_masks, scope, mode, min_region_area, merge_small_regions,
                  color_names="", color_tolerance=24.0, group_tolerance=0.08, with_scope=False):
    """
    Shared tail of build_atlas() and atlas_from_masks(): scope, small regions,
    reading order, groups, tags, the atlas dict. meta entries that carry a
    "group" keep it; a "tags" list in meta is kept and extended by tag_masks.
    """
    height, width = labels.shape
    if scope is not None:
        labels[~scope] = -1

    labels, kept, merged, dropped = handle_small_regions(labels, int(min_region_area), bool(merge_small_regions))
    meta = [meta[k] for k in kept]
    if merged:
        notes.append(f"{merged} region(s) under {min_region_area} px merged into a neighbour")
    if dropped:
        notes.append(f"{dropped} region(s) under {min_region_area} px dropped (left unassigned)")

    n = len(meta)
    boxes, areas, centroids = region_stats(labels, n)

    # final ids in reading order
    order = reading_order(boxes)
    remap = np.empty(n, dtype=np.int32)
    remap[order] = np.arange(n, dtype=np.int32)
    labels = np.where(labels >= 0, remap[np.maximum(labels, 0)], -1).astype(np.int32)
    boxes, areas, centroids = boxes[order], areas[order], centroids[order]
    meta = [meta[i] for i in order]

    # groups
    named, errors = parse_color_names(color_names)
    notes.extend(errors)
    group_ids = [None] * n
    if mode == "mask_folder" or (meta and all(m.get("group") for m in meta)):
        group_ids = [m["group"] for m in meta]
    else:
        if mode == "color_regions" and named:
            colors = np.asarray([m["color"] for m in meta], dtype=np.float32).reshape(-1, 3)
            for rgb_key, name in named:
                if n == 0:
                    break
                d = np.linalg.norm(colors - np.asarray(rgb_key, np.float32), axis=1)
                hit = d <= max(float(color_tolerance), 30.0)
                if not hit.any():
                    notes.append(f"color_names: {_hex(rgb_key)} = {name} matches no region colour")
                    continue
                for i in np.flatnonzero(hit):
                    if group_ids[i] is None:
                        group_ids[i] = name
        rest = [i for i in range(n) if group_ids[i] is None]
        if rest:
            keys = None
            if mode == "color_regions":
                keys = [_hex(meta[i]["color"]) for i in rest]
            sub_boxes = boxes[rest]
            sub_labels = np.where(np.isin(labels, rest), labels, -1)
            # relabel the subset to 0..len(rest)-1 for group_by_shape
            local = np.full(n, -1, dtype=np.int32)
            local[rest] = np.arange(len(rest), dtype=np.int32)
            sub_labels = np.where(sub_labels >= 0, local[np.maximum(sub_labels, 0)], -1)
            gidx = group_by_shape(sub_labels, sub_boxes, keys, float(group_tolerance))
            for i, g in zip(rest, gidx):
                group_ids[i] = f"g{int(g) + 1}"

    regions = []
    for i in range(n):
        x0, y0, x1, y1 = (int(v) for v in boxes[i])
        rec = {
            "region_id": i,
            "name": meta[i].get("name", f"r{i}"),
            "group_id": group_ids[i],
            "bbox": [x0, y0, x1 - x0, y1 - y0],
            "area": int(areas[i]),
            "centroid": [round(float(centroids[i][0]), 1), round(float(centroids[i][1]), 1)],
        }
        if "color" in meta[i]:
            rec["color"] = _hex(meta[i]["color"])
        if "source" in meta[i]:
            rec["source"] = meta[i]["source"]
        regions.append(rec)

    if tag_masks or any("tags" in m for m in meta):
        tags = region_tags(labels, n, tag_masks, [m.get("own_mask") for m in meta])
        for r, m, t in zip(regions, meta, tags):
            r["tags"] = list(m.get("tags", [])) + [x for x in t if x not in m.get("tags", [])]

    groups = {}
    for r in regions:
        groups.setdefault(r["group_id"], []).append(r["region_id"])

    atlas = {
        "format": ATLAS_FORMAT,
        "version": ATLAS_VERSION,
        "width": int(width),
        "height": int(height),
        "mode": mode,
        "regions": regions,
        "groups": groups,
        "unassigned_px": int((labels < 0).sum()),
        "notes": notes,
    }
    if scope is not None:
        atlas["scope_px"] = int(scope.sum())
        atlas["unassigned_in_scope_px"] = int((scope & (labels < 0)).sum())
    if with_scope:
        return labels, atlas, scope
    return labels, atlas


PATCH_MODES = ("on_top", "underneath")


def parse_mask_names(text: str, count: int):
    """
    One name per line ('W_F1_01' or 'W_F1_01 = windows' to set the group), or
    one comma separated line. // starts a comment. Missing names become
    mask_01.., repeated names get _2, _3. Returns ([(name, group or None)], notes).
    """
    lines = []
    for raw in (text or "").splitlines():
        line = raw.split("//", 1)[0].strip()
        if line:
            lines.append(line)
    if len(lines) == 1 and "," in lines[0]:
        lines = [p.strip() for p in lines[0].split(",") if p.strip()]
    notes, out, seen = [], [], {}
    for line in lines[:count]:
        name, _, group = (s.strip() for s in line.partition("="))
        name = re.sub(r"\s+", "_", name) or "mask"
        out.append((name, re.sub(r"\s+", "_", group) or None))
    if len(lines) > count:
        notes.append(f"{len(lines) - count} name(s) more than masks, ignored")
    if lines and len(lines) < count:
        notes.append(f"{count - len(lines)} mask(s) without a name, named mask_NN")
    out += [(f"mask_{i + 1:02d}", None) for i in range(len(out), count)]
    final = []
    for name, group in out:
        seen[name] = seen.get(name, 0) + 1
        if seen[name] > 1:
            notes.append(f"name '{name}' repeated, the copy is '{name}_{seen[name]}'")
            name = f"{name}_{seen[name]}"
        final.append((name, group))
    return final, notes


def atlas_from_masks(entries, width, height, base=None, patch="on_top", scope=None,
                     min_region_area=64, merge_small_regions=True, scope_masks="",
                     tag_only_masks="", split_masks="", with_scope=False):
    """
    Regions from a list of masks (entries as in label_masks()), optionally
    patched into an existing atlas.

    base: (labels, atlas, scope or None) of an existing atlas at width x height.
    patch="on_top": every new mask cuts itself out of the base regions below it
    (a window out of a wall); "underneath": new masks only fill pixels no base
    region has. Base regions keep their name, group, colour, source and tags; a
    base region that loses all its pixels disappears. New regions are tagged
    with the base region that covered most of them and that region's tags, so
    rules for the wall's floor still reach the window cut out of it.
    Returns (labels, atlas[, scope]) like build_atlas().
    """
    if patch not in PATCH_MODES:
        raise ValueError(f"patch must be one of {PATCH_MODES}")
    labels, meta, notes, mask_scope, tag_masks = label_masks(
        entries, width, height, scope_masks=scope_masks, tag_only_masks=tag_only_masks,
        split_masks=split_masks, min_part_area=int(min_region_area))
    if scope is not None:
        scope = np.asarray(scope, dtype=bool)
        if scope.shape != (height, width):
            raise ValueError(f"scope is {scope.shape[1]}x{scope.shape[0]}, the regions {width}x{height}")
    if mask_scope is not None:
        scope = mask_scope if scope is None else (scope & mask_scope)
    mode = "masks"

    if base is not None:
        base_labels, base_atlas, base_scope = base
        base_labels = np.asarray(base_labels, dtype=np.int32)
        if base_labels.shape != (height, width):
            raise ValueError(f"base regions are {base_labels.shape[1]}x{base_labels.shape[0]}, "
                             f"expected {width}x{height}")
        if base_scope is not None:
            base_scope = np.asarray(base_scope, dtype=bool)
            scope = base_scope if scope is None else (scope & base_scope)
        base_meta = []
        for r in base_atlas["regions"]:
            m = {"name": r.get("name", f"r{r['region_id']}"), "group": r.get("group_id") or "base",
                 "tags": list(r.get("tags", []))}
            if r.get("source"):
                m["source"] = r["source"]
            if r.get("color") and parse_color(r["color"]):
                m["color"] = parse_color(r["color"])
            base_meta.append(m)
        k = len(base_meta)
        # host: the base region under most of each new region, before patching
        both = (labels >= 0) & (base_labels >= 0)
        if k and meta and both.any():
            pair = np.bincount(labels[both].astype(np.int64) * k + base_labels[both],
                               minlength=len(meta) * k).reshape(len(meta), k)
            for i, m in enumerate(meta):
                if pair[i].any():
                    h = base_meta[int(pair[i].argmax())]
                    m["tags"] = h["tags"] + [h["name"]]
        shifted = np.where(labels >= 0, labels + k, -1).astype(np.int32)
        before = int((base_labels >= 0).sum())
        if patch == "on_top":
            merged_labels = np.where(shifted >= 0, shifted, base_labels).astype(np.int32)
        else:
            merged_labels = np.where(base_labels >= 0, base_labels, shifted).astype(np.int32)
        left = np.bincount(merged_labels[(merged_labels >= 0) & (merged_labels < k)], minlength=k)
        gone = [base_meta[i]["name"] for i in np.flatnonzero(left[:k] == 0)]
        notes.append(f"patch {patch}: {len(meta)} new region(s) into {k} base region(s) "
                     f"({before} px); {int(((shifted >= 0) & (base_labels >= 0)).sum())} px overlap")
        if gone:
            notes.append(f"base region(s) fully covered and removed: {', '.join(gone[:20])}"
                         + (" ..." if len(gone) > 20 else ""))
        labels, meta = merged_labels, base_meta + meta
        mode = f"{base_atlas.get('mode', 'base')}+masks"

    if not meta:
        raise ValueError("no regions: every mask is empty, a scope mask or a tag-only mask.")
    return _finish_atlas(labels, meta, notes, tag_masks, scope, mode, min_region_area,
                         merge_small_regions, with_scope=with_scope)


# --------------------------------------------------------------------------
# preview
# --------------------------------------------------------------------------

def group_colors(group_names):
    """A stable, well separated colour per group (golden-ratio hue walk)."""
    out = {}
    for k, g in enumerate(group_names):
        hue = (0.07 + k * 0.618033988749895) % 1.0
        sat = 0.55 + 0.35 * ((k * 7) % 3) / 2.0
        val = 0.95 - 0.2 * ((k * 5) % 2)
        out[g] = np.asarray(colorsys.hsv_to_rgb(hue, sat, val), dtype=np.float32)
    return out


def render_preview(image, labels, atlas):
    """
    Colour coded preview: one colour per group over a grey copy of the matrix,
    region borders in black, the region id (and group id when it fits) written
    at the point of each region furthest from its border. Returns HxWx3 float.
    """
    rgb = to_uint8_rgb(image)
    h, w = labels.shape
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    regions = atlas["regions"]
    colors = group_colors(list(atlas["groups"].keys()))

    lut = np.zeros((len(regions) + 1, 3), dtype=np.float32)
    for r in regions:
        lut[r["region_id"] + 1] = colors[r["group_id"]]
    out = lut[labels + 1] * 0.75 + gray[..., None] * 0.25
    out[labels < 0] = gray[labels < 0, None] * 0.35

    border = np.zeros((h, w), dtype=bool)
    border[:, 1:] |= labels[:, 1:] != labels[:, :-1]
    border[1:, :] |= labels[1:, :] != labels[:-1, :]
    thick = max(1, round(min(h, w) / 1080))
    if thick > 1:
        border = cv2.dilate(border.astype(np.uint8), np.ones((thick, thick), np.uint8)) > 0
    out[border] = 0.0

    canvas = np.ascontiguousarray((np.clip(out, 0, 1) * 255).astype(np.uint8))
    font = cv2.FONT_HERSHEY_SIMPLEX
    base_scale = max(0.4, min(h, w) / 1080 * 0.9)
    for r in regions:
        x, y, bw, bh = r["bbox"]
        crop = np.pad((labels[y:y + bh, x:x + bw] == r["region_id"]).astype(np.uint8), 1)
        dist = cv2.distanceTransform(crop, cv2.DIST_L2, 3)
        # Among the points nearly as deep inside as the deepest one, take the one
        # nearest the centroid: on a long band the deepest points form a whole line,
        # and the first of them would put the label at the band's left end.
        ys, xs = np.nonzero(dist >= 0.9 * dist.max())
        gx, gy = r["centroid"][0] - x + 1, r["centroid"][1] - y + 1
        k = int(np.argmin((xs - gx) ** 2 + (ys - gy) ** 2))
        py, px = ys[k], xs[k]
        radius = float(dist[py, px])
        cx, cy = x + px - 1, y + py - 1

        lines = [str(r["region_id"])]
        if radius > 30 * base_scale:
            lines.append(str(r["group_id"]))
        stroke = max(1, round(2 * base_scale))
        (tw, th), _ = cv2.getTextSize(lines[0], font, 1.0, stroke)
        scale = min(base_scale, 1.8 * radius / max(tw, 1), 1.2 * radius / max(th, 1))
        scale = max(scale, 0.3)
        sub_scale = scale * 0.55
        stroke = max(1, round(2 * scale))

        (tw, th), _ = cv2.getTextSize(lines[0], font, scale, stroke)
        total_h = th + (int(th * 0.55) + 6 if len(lines) > 1 else 0)
        org = (int(cx - tw / 2), int(cy - total_h / 2 + th))
        cv2.putText(canvas, lines[0], org, font, scale, (0, 0, 0), stroke + 2, cv2.LINE_AA)
        cv2.putText(canvas, lines[0], org, font, scale, (255, 255, 255), stroke, cv2.LINE_AA)
        if len(lines) > 1:
            s2 = max(1, round(2 * sub_scale))
            (tw2, th2), _ = cv2.getTextSize(lines[1], font, sub_scale, s2)
            org2 = (int(cx - tw2 / 2), org[1] + th2 + 6)
            cv2.putText(canvas, lines[1], org2, font, sub_scale, (0, 0, 0), s2 + 2, cv2.LINE_AA)
            cv2.putText(canvas, lines[1], org2, font, sub_scale, (255, 255, 255), s2, cv2.LINE_AA)

    return canvas.astype(np.float32) / 255.0
