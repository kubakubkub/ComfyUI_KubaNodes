"""
sam_prompts.py

Pure helpers for kubakub regions sam3 masks (nodes_sources.py): prompt lines,
point / box parsing and the crop window of the detail pass. numpy only, no
ComfyUI imports, so tests/test_sam_prompts.py runs without a model.
"""

from __future__ import annotations

import json
import re

import numpy as np

DEFAULT_MAX_DETECTIONS = 50


def safe_name(text: str, fallback: str = "mask") -> str:
    """Region / file name: letters, digits, _ and -; spaces become _."""
    s = re.sub(r"\s+", "_", (text or "").strip())
    s = re.sub(r"[^A-Za-z0-9_\-]", "", s)
    return s or fallback


def parse_prompt_lines(text: str, default_max: int = DEFAULT_MAX_DETECTIONS):
    """
    'window = window : 40', 'balcony = wrought iron balcony', or just 'door'
    (name = the text). // starts a comment. Returns ([(name, text, max_det)], notes).
    Core SAM3 finds 1 object per prompt unless ':N' is given, so the default here is default_max.
    """
    out, notes = [], []
    for n, raw in enumerate((text or "").splitlines(), 1):
        line = raw.split("//", 1)[0].strip()
        if not line:
            continue
        name, eq, prompt = line.partition("=")
        if not eq:
            name, prompt = "", line
        prompt = prompt.strip().replace(",", " ")
        m = re.match(r"^(.*?)\s*:\s*(\d+)\s*$", prompt)
        max_det = default_max
        if m:
            prompt, max_det = m.group(1).strip(), max(1, int(m.group(2)))
        if not prompt:
            notes.append(f"prompt line {n}: no text, skipped")
            continue
        out.append((safe_name(name or prompt), prompt, max_det))
    return out, notes


def parse_points(text, width: int, height: int):
    """
    Points JSON as the KJNodes Points Editor writes it: [{"x": .., "y": ..}, ...].
    Normalized points (all within 0..1) are scaled to the image. Also accepts
    [[x, y], ...]. Returns a list of (x, y) floats in image pixels.
    """
    if text is None:
        return []
    if isinstance(text, str):
        text = text.strip()
        if not text:
            return []
        data = json.loads(text)
    else:
        data = text
    pts = []
    for p in data or []:
        if isinstance(p, dict):
            pts.append((float(p["x"]), float(p["y"])))
        else:
            pts.append((float(p[0]), float(p[1])))
    if pts and all(0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 for x, y in pts) and any(
            isinstance(v, float) and v != int(v) for pt in pts for v in pt):
        pts = [(x * width, y * height) for x, y in pts]
    return pts


def parse_boxes(boxes):
    """
    Boxes as (x0, y0, x1, y1): accepts the KJNodes BBOX list of xyxy tuples,
    core BOUNDING_BOX dicts {x, y, width, height} or a JSON string of either.
    """
    if boxes is None:
        return []
    if isinstance(boxes, str):
        boxes = json.loads(boxes) if boxes.strip() else []
    if isinstance(boxes, dict):
        boxes = [boxes]
    out = []
    for b in boxes:
        if isinstance(b, dict):
            x, y = float(b["x"]), float(b["y"])
            out.append((x, y, x + float(b["width"]), y + float(b["height"])))
        elif isinstance(b, (list, tuple)) and len(b) == 4:
            x0, y0, x1, y1 = (float(v) for v in b)
            out.append((min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)))
    return [b for b in out if b[2] > b[0] and b[3] > b[1]]


def detail_window(bbox, width: int, height: int, pad: float = 0.25, min_px: int = 256):
    """
    Crop (x0, y0, x1, y1) around an object's bbox for the detail pass: padded by
    pad of its size on every side, at least min_px wide and high, clamped to the
    image. Returns None when the crop would cover most of the image (no gain).
    """
    x0, y0, x1, y1 = bbox
    bw, bh = x1 - x0, y1 - y0
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    # square: SAM3 resizes every input to 1008x1008, a long crop would be stretched
    side = max(bw * (1 + 2 * pad), bh * (1 + 2 * pad), min_px)
    w, h = min(side, width), min(side, height)
    cx0 = int(round(min(max(cx - w / 2, 0), width - w)))
    cy0 = int(round(min(max(cy - h / 2, 0), height - h)))
    cx1, cy1 = int(round(cx0 + w)), int(round(cy0 + h))
    if (cx1 - cx0) * (cy1 - cy0) > 0.6 * width * height:
        return None
    return cx0, cy0, min(cx1, width), min(cy1, height)


def mask_bbox(mask: np.ndarray):
    """(x0, y0, x1, y1) exclusive of a bool mask, or None when empty."""
    ys = np.flatnonzero(mask.any(axis=1))
    if ys.size == 0:
        return None
    xs = np.flatnonzero(mask.any(axis=0))
    return int(xs[0]), int(ys[0]), int(xs[-1]) + 1, int(ys[-1]) + 1


def parts_at(mask: np.ndarray, points) -> np.ndarray:
    """
    The connected parts of mask under any of the points (x, y); the largest
    part when no point hits the mask. Removes the specks SAM leaves elsewhere.
    """
    import cv2
    n, lab = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    if n <= 2:
        return mask
    h, w = mask.shape
    hit = {int(lab[min(h - 1, max(0, int(y))), min(w - 1, max(0, int(x)))]) for x, y in points}
    hit.discard(0)
    if not hit:
        hit = {int(np.bincount(lab[lab > 0]).argmax())}
    return np.isin(lab, list(hit))


def container_hits(masks, min_inside: float = 0.8):
    """
    Indices of masks that contain at least two other masks of the list (each
    with min_inside of its pixels inside): SAM text hits that cover a whole row
    of objects besides the single objects. One-in-one (window surround + its
    glass) is kept.
    """
    boxes = [mask_bbox(m) for m in masks]
    areas = [int(m.sum()) for m in masks]
    out = []
    for i, (bi, mi) in enumerate(zip(boxes, masks)):
        if bi is None:
            continue
        inside = 0
        for j, bj in enumerate(boxes):
            if j == i or bj is None or areas[j] >= areas[i]:
                continue
            if bj[0] < bi[0] or bj[1] < bi[1] or bj[2] > bi[2] or bj[3] > bi[3]:
                # bbox not inside: allow a few px of slack before the pixel test
                if (bj[0] < bi[0] - 4 or bj[1] < bi[1] - 4 or bj[2] > bi[2] + 4 or bj[3] > bi[3] + 4):
                    continue
            x0, y0, x1, y1 = bj
            sub = masks[j][y0:y1, x0:x1]
            if (sub & mi[y0:y1, x0:x1]).sum() >= min_inside * areas[j]:
                inside += 1
                if inside >= 2:
                    out.append(i)
                    break
    return out


def numbered(names):
    """['window', 'window', 'door'] -> ['window_01', 'window_02', 'door'] (only repeated names get numbers)."""
    total = {n: names.count(n) for n in names}
    seen, out = {}, []
    for n in names:
        if total[n] > 1:
            seen[n] = seen.get(n, 0) + 1
            out.append(f"{n}_{seen[n]:02d}")
        else:
            out.append(n)
    return out
