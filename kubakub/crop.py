"""
crop.py

The crop box of kubakub crop / uncrop and the director's layer crop. Pure numpy / opencv, no ComfyUI imports
(tests/test_crop.py).

The box is drawn on the picture (web/kubakub_crop.js) and kept as 0-1 of the picture ("x y w h"), so it stays
the same part of the picture when the picture comes at another size (a 1024 draft, then the 3840 final). The
pixels are found here, once: the ratio is kept by shrinking the longer side around the centre, and a model-safe
size rounds width and height down to a multiple of `snap` (16 for Flux) around the centre too.

The box may reach past the picture (as Photoshop's crop tool): the new area is padded (edge, mirror, black, white,
grey) and comes out as its own mask, ready for an outpaint.
"""

from __future__ import annotations

import cv2
import numpy as np

CROP_FORMAT = "kubakub.crop"
RATIOS = ["free", "original", "picture", "1:1", "4:5", "5:4", "3:2", "2:3", "16:9", "9:16", "21:9"]
SIZES = ["as cut", "1024", "1536", "2048", "3840"]
FILLS = ["edge", "mirror", "black", "white", "grey"]
FULL = (0.0, 0.0, 1.0, 1.0)
REACH = 2.0                                          # how far past the picture a box may reach (pictures)


def parse_box(text) -> tuple[float, float, float, float]:
    """'x y w h' (0-1 of the picture, commas allowed; past 0 / 1 = padding, up to REACH pictures away) -> the box;
    anything else, or a box of no size, = all of it."""
    try:
        v = [float(t) for t in str(text or "").replace(",", " ").split()]
    except ValueError:
        return FULL
    if len(v) != 4 or not all(np.isfinite(v)):
        return FULL
    x0, y0 = min(max(v[0], -REACH), 1.0), min(max(v[1], -REACH), 1.0)
    x1, y1 = min(max(v[0] + v[2], 0.0), 1.0 + REACH), min(max(v[1] + v[3], 0.0), 1.0 + REACH)
    return FULL if x1 - x0 <= 0 or y1 - y0 <= 0 else (x0, y0, x1 - x0, y1 - y0)


def box_text(box) -> str:
    return " ".join(f"{round(float(v), 5):g}" for v in box)


def ratio_value(name: str, W: int, H: int, of: tuple[int, int] | None = None) -> float | None:
    """Width / height of a ratio pill, None = free. 'original' = the picture's own; 'picture' = of the
    'ratio of' input (W, H) (falls back to original without one); 'a:b' as written."""
    name = str(name or "free").strip().lower()
    if name == "original":
        return W / H
    if name == "picture":
        return (of[0] / of[1]) if of and of[1] else W / H
    if ":" in name:
        a, b = name.split(":", 1)
        try:
            a, b = float(a), float(b)
        except ValueError:
            return None
        return a / b if a > 0 and b > 0 else None
    return None


def fit_ratio(x: float, y: float, w: float, h: float, r: float | None):
    """The largest box of ratio r inside (x, y, w, h), same centre (pixels, floats)."""
    if not r or w <= 0 or h <= 0:
        return x, y, w, h
    if w / h > r:
        nw = h * r
        return x + (w - nw) / 2, y, nw, h
    nh = w / r
    return x, y + (h - nh) / 2, w, nh


def pixel_box(box, W: int, H: int, ratio: float | None = None, snap: int = 1) -> tuple[int, int, int, int]:
    """The 0-1 box on a W x H picture -> integer (x, y, w, h): ratio kept, sizes multiples of snap (never below
    one snap). It may reach past the picture (padding); a box inside the picture stays inside it."""
    x, y, w, h = parse_box(box if isinstance(box, str) else box_text(box))
    inside = x >= -1e-6 and y >= -1e-6 and x + w <= 1 + 1e-6 and y + h <= 1 + 1e-6
    x, y, w, h = fit_ratio(x * W, y * H, w * W, h * H, ratio)
    snap = max(1, int(snap))
    cx, cy = x + w / 2, y + h / 2
    pw = max(snap, int(round(w)) // snap * snap)
    ph = max(snap, int(round(h)) // snap * snap)
    if inside:
        pw = min(pw, W // snap * snap or W)
        ph = min(ph, H // snap * snap or H)
    if ratio and snap > 1:                               # snapping both sides must not drift the ratio by more than a snap
        if pw / ph > ratio * 1.0001 and pw > snap:
            pw = max(snap, int(round(ph * ratio)) // snap * snap)
        elif pw / ph < ratio / 1.0001 and ph > snap:
            ph = max(snap, int(round(pw / ratio)) // snap * snap)
    px = int(round(cx - pw / 2))
    py = int(round(cy - ph / 2))
    if inside:
        px, py = min(max(px, 0), max(0, W - pw)), min(max(py, 0), max(0, H - ph))
    else:                                                # at least one pixel of the picture stays in the box
        px, py = min(max(px, 1 - pw), W - 1), min(max(py, 1 - ph), H - 1)
    return px, py, pw, ph


def long_side(w: int, h: int, size: str, snap: int = 1) -> tuple[int, int]:
    """The output size for a size pill: 'as cut' = (w, h); a number = that long side, the ratio kept, multiples of snap."""
    try:
        target = int(size)
    except (TypeError, ValueError):
        return w, h
    k = target / max(w, h)
    snap = max(1, int(snap))
    return (max(snap, int(round(w * k / snap)) * snap), max(snap, int(round(h * k / snap)) * snap))


def crop_info(px_box, W: int, H: int) -> dict:
    x, y, w, h = (int(v) for v in px_box)
    return {"format": CROP_FORMAT, "x": x, "y": y, "w": w, "h": h, "W": int(W), "H": int(H)}


def padded(info: dict) -> bool:
    return info["x"] < 0 or info["y"] < 0 or info["x"] + info["w"] > info["W"] or info["y"] + info["h"] > info["H"]


def box_mask(info: dict) -> np.ndarray:
    """(H, W) float32: 1 where the crop box covers the picture."""
    m = np.zeros((info["H"], info["W"]), np.float32)
    x0, y0 = max(0, info["x"]), max(0, info["y"])
    m[y0:max(y0, info["y"] + info["h"]), x0:max(x0, info["x"] + info["w"])] = 1.0
    return m


def new_area(info: dict) -> np.ndarray:
    """(h, w) float32 over the crop: 1 on the padding (past the picture), 0 on the picture."""
    m = np.ones((info["h"], info["w"]), np.float32)
    x0, y0 = max(0, -info["x"]), max(0, -info["y"])
    x1, y1 = min(info["w"], info["W"] - info["x"]), min(info["h"], info["H"] - info["y"])
    if x1 > x0 and y1 > y0:
        m[y0:y1, x0:x1] = 0.0
    return m


_BORDER = {"edge": cv2.BORDER_REPLICATE, "mirror": cv2.BORDER_REFLECT_101}
_VALUE = {"black": 0.0, "white": 1.0, "grey": 0.5}


def _pad(part, t, b, l, r, mode):
    """copyMakeBorder that also pads by more than the picture (a mirror repeats the mirrored copy)."""
    out = part
    while t or b or l or r:
        h, w = out.shape[:2]
        if mode == cv2.BORDER_REFLECT_101 and min(h, w) < 2:
            mode = cv2.BORDER_REPLICATE                 # one pixel has nothing to mirror
        lim_h, lim_w = (h - 1, w - 1) if mode == cv2.BORDER_REFLECT_101 else (h, w)
        st, sb, sl, sr = min(t, lim_h), min(b, lim_h), min(l, lim_w), min(r, lim_w)
        out = cv2.copyMakeBorder(out, st, sb, sl, sr, mode)
        if out.ndim == 2:
            out = out[..., None]
        t, b, l, r = t - st, b - sb, l - sl, r - sr
    return out


def cut(img: np.ndarray, info: dict, fill: str = "edge") -> np.ndarray:
    """(H, W, c) float -> the crop (h, w, c); the part past the picture filled by `fill` (FILLS)."""
    H, W = img.shape[:2]
    x, y, w, h = info["x"], info["y"], info["w"], info["h"]
    x0, y0, x1, y1 = max(0, x), max(0, y), min(W, x + w), min(H, y + h)
    if x0 == x and y0 == y and x1 == x + w and y1 == y + h:
        return np.ascontiguousarray(img[y0:y1, x0:x1])
    if x1 <= x0 or y1 <= y0:
        raise ValueError("crop: the box does not touch the picture")
    part = np.ascontiguousarray(img[y0:y1, x0:x1], np.float32)
    t, l = y0 - y, x0 - x
    b, r = (y + h) - y1, (x + w) - x1
    if fill in _BORDER:
        return _pad(part, t, b, l, r, _BORDER[fill])
    out = np.full((h, w, part.shape[2]), _VALUE.get(fill, 0.0), np.float32)
    out[t:t + part.shape[0], l:l + part.shape[1]] = part
    return out


def feather_alpha(info: dict, feather: float) -> np.ndarray:
    """(h, w) float32 over the box: 0 at the box edge rising to 1 at `feather` px inside, smoothstep. An edge that
    lies on or past the picture's border stays hard (nothing is there to blend with)."""
    w, h = info["w"], info["h"]
    a = np.ones((h, w), np.float32)
    f = float(feather)
    if f <= 0:
        return a
    xs = np.arange(w, dtype=np.float32) + 0.5
    ys = np.arange(h, dtype=np.float32) + 0.5
    big = np.float32(1e9)
    dl = xs if info["x"] > 0 else np.full(w, big, np.float32)
    dr = w - xs if info["x"] + w < info["W"] else np.full(w, big, np.float32)
    dt = ys if info["y"] > 0 else np.full(h, big, np.float32)
    db = h - ys if info["y"] + h < info["H"] else np.full(h, big, np.float32)
    d = np.minimum(np.minimum(dt, db)[:, None], np.minimum(dl, dr)[None, :])
    u = np.clip(d / f, 0, 1)
    return (u * u * (3 - 2 * u)).astype(np.float32)


def fit_to(img: np.ndarray, w: int, h: int) -> np.ndarray:
    """(h0, w0[, c]) -> (h, w[, c]): area when it shrinks, cubic when it grows."""
    if img.shape[:2] == (h, w):
        return img
    inter = cv2.INTER_AREA if img.shape[1] > w and img.shape[0] > h else cv2.INTER_CUBIC
    out = cv2.resize(np.ascontiguousarray(img), (w, h), interpolation=inter)
    if out.ndim == 2 and img.ndim == 3:
        out = out[..., None]
    return np.clip(out, 0, 1) if out.dtype.kind == "f" else out


def mask_in_box(mask: np.ndarray | None, info: dict) -> np.ndarray | None:
    """A paste mask as (h, w) over the box: a whole-picture mask is cut to the box (0 past the picture), a
    crop-size one resized to it."""
    if mask is None:
        return None
    m = np.asarray(mask, np.float32)
    if m.shape == (info["H"], info["W"]):
        m = cut(m[..., None], info, "black")[..., 0]
    return np.clip(fit_to(m, info["w"], info["h"]), 0, 1)


def scaled_info(info: dict, W: int, H: int) -> dict:
    """The same box on the same picture at another size."""
    if (H, W) == (info["H"], info["W"]):
        return info
    sx, sy = W / info["W"], H / info["H"]
    x, y = int(round(info["x"] * sx)), int(round(info["y"] * sy))
    return dict(info, x=x, y=y, w=max(1, int(round(info["w"] * sx))), h=max(1, int(round(info["h"] * sy))), W=W, H=H)


def paste(original: np.ndarray, piece: np.ndarray, info: dict, feather: float = 0.0, mask=None) -> np.ndarray:
    """The edited crop back into the original picture (H, W, c): resized to the box, blended over the feather and
    the optional mask. Padding past the picture is dropped (only the part over the picture comes back). An
    original of another size than the crop was cut from gets the box at the same spot of it."""
    H, W = original.shape[:2]
    info = scaled_info(info, W, H)
    x, y, w, h = info["x"], info["y"], info["w"], info["h"]
    c = original.shape[2]
    p = fit_to(np.asarray(piece, np.float32), w, h)
    if p.shape[2] != c:
        p = p[..., :c] if p.shape[2] > c else np.concatenate([p, np.ones(p.shape[:2] + (c - p.shape[2],), np.float32)], -1)
    a = feather_alpha(info, feather)
    m = mask_in_box(mask, info)
    if m is not None:
        a = a * m
    x0, y0, x1, y1 = max(0, x), max(0, y), min(W, x + w), min(H, y + h)
    out = np.array(original, np.float32, copy=True)
    if x1 <= x0 or y1 <= y0:
        return out
    region = out[y0:y1, x0:x1]
    pp, aa = p[y0 - y:y1 - y, x0 - x:x1 - x], a[y0 - y:y1 - y, x0 - x:x1 - x]
    out[y0:y1, x0:x1] = region + (pp - region) * aa[..., None]
    return out
