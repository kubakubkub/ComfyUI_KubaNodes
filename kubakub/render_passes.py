"""
render_passes.py

A folder of render passes from any 3D tool (Houdini, Blender, C4D, Maya ...) -> the picture, depth, normals and
every other pass as a mask. Files are named <render>_<pass>.<ext> (facade_beauty.png, facade_depth.png,
facade_cut.png) or just <pass>.<ext>; several renders can share a folder, and files that belong to no render
(facade_mask.png next to facade_a_beauty.png, facade_b_beauty.png) are masks for all of them.

numpy + OpenCV only (tests/test_render_passes.py).
"""

from __future__ import annotations

import fnmatch
import os
import re

import cv2
import numpy as np

from . import imio

EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp", ".bmp")
ROLES = {
    "beauty": ("beauty", "rgb", "rgba", "combined", "color", "colour", "clay", "render", "image", "diffuse", "c"),
    "depth": ("depth", "z", "zdepth", "depthmap", "dist"),
    "normal": ("normal", "normals", "nrm", "n"),
}
_ROLE_OF = {tok: role for role, toks in ROLES.items() for tok in toks}


def _split(stem):
    """'facade_a_beauty' -> ('facade_a', 'beauty'); 'beauty' -> ('', 'beauty')."""
    m = re.match(r"^(.*?)[_\-. ]+([^_\-. ]+)$", stem)
    return (m.group(1), m.group(2)) if m else ("", stem)


def discover(folder):
    """
    -> {"renders": {name: {"beauty": path, "depth": path, "normal": path, "masks": {pass: path}}},
        "shared": {stem: path}, "skipped": [file names]}.
    A render is every name that has a beauty pass. Files starting with '_' (contact sheets) and other file types
    are skipped.
    """
    if not os.path.isdir(folder):
        raise ValueError(f"render passes: folder not found: '{folder}'")
    files, skipped = [], []
    for fn in sorted(os.listdir(folder), key=str.lower):
        p = os.path.join(folder, fn)
        if not os.path.isfile(p):
            continue
        stem, ext = os.path.splitext(fn)
        if ext.lower() not in EXTS or fn.startswith(("_", ".")):
            if ext.lower() in (".exr", ".hdr") or fn.startswith("_"):
                skipped.append(fn)
            continue
        files.append((stem, p))
    split = [(_split(stem), stem, p) for stem, p in files]
    names = {pre.lower(): pre for (pre, tok), _s, _p in split if _ROLE_OF.get(tok.lower()) == "beauty"}
    renders = {n: {"beauty": None, "depth": None, "normal": None, "masks": {}} for n in names.values()}
    shared = {}
    for (pre, tok), stem, p in split:
        r = renders.get(names.get(pre.lower(), None))
        if r is None:
            shared[stem] = p
            continue
        role = _ROLE_OF.get(tok.lower())
        if role and r[role] is None:
            r[role] = p
        else:
            r["masks"][tok] = p
    return {"renders": renders, "shared": shared, "skipped": skipped}


def _read(path):
    """-> float32 (H, W, C) in 0..1 (C = 1, 3 or 4, RGB order), 16-bit files keep their range."""
    img = imio.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"render passes: cannot read {os.path.basename(path)}")
    scale = 65535.0 if img.dtype == np.uint16 else 255.0 if img.dtype == np.uint8 else 1.0
    a = img.astype(np.float32) / scale
    if a.ndim == 2:
        return a[..., None]
    if a.shape[2] == 4:
        return a[..., [2, 1, 0, 3]]
    return a[..., ::-1]


def _rgb(a):
    return np.repeat(a[..., :1], 3, 2) if a.shape[2] == 1 else np.ascontiguousarray(a[..., :3])


def _is_colour(a):
    """An ID map or a picture, not a mask: its channels differ."""
    if a.shape[2] < 3:
        return False
    s = a[::4, ::4, :3]
    return float(np.abs(s - s.mean(2, keepdims=True)).max()) > 0.08


def _mask(a):
    """White (or opaque, when the picture itself is flat) = inside."""
    if a.shape[2] == 4 and float(np.ptp(a[..., 3])) > 0.5 and float(np.ptp(a[..., :3])) < 0.05:
        return a[..., 3]
    return a[..., :3].mean(2) if a.shape[2] >= 3 else a[..., 0]


def _fit(a, W, H, nearest=False):
    if a.shape[:2] == (H, W):
        return a
    out = cv2.resize(a, (W, H), interpolation=cv2.INTER_NEAREST if nearest else cv2.INTER_AREA)
    return out[..., None] if out.ndim == 2 and a.ndim == 3 else out


def load(folder, name="", invert="", exclude="", depth="as rendered"):
    """
    -> dict(name, beauty (H, W, 3), depth (H, W, 3) or None, normal (H, W, 3) or None, masks [(name, (H, W) float32)],
            notes [str], renders [names in the folder]).
    name: which render ('' = the first). invert / exclude: mask names, wildcards, comma separated ('cut' = black means
    inside). depth: 'as rendered', 'near is white' or 'near is black' (the last two stretch it to 0..1).
    """
    found = discover(folder)
    renders = found["renders"]
    notes = []
    if not renders:
        raise ValueError(f"render passes: no picture in '{folder}': name it <render>_beauty.png "
                         f"(also: {', '.join(ROLES['beauty'][1:8])})")
    key = {n.lower(): n for n in renders}
    name = (name or "").strip()
    if name and name.lower() not in key:
        raise ValueError(f"render passes: no render '{name}' in the folder; it has: {', '.join(sorted(renders)) or '-'}")
    pick = key[name.lower()] if name else sorted(renders, key=str.lower)[0]
    r = renders[pick]
    beauty = _rgb(_read(r["beauty"]))
    H, W = beauty.shape[:2]

    def aux(path, what):
        if not path:
            return None
        a = _rgb(_read(path))
        if a.shape[:2] != (H, W):
            notes.append(f"{what} {a.shape[1]}x{a.shape[0]} resized to the picture {W}x{H}")
        return _fit(a, W, H)

    dep = aux(r["depth"], "depth")
    if dep is not None and depth != "as rendered":
        g = dep[..., 0]
        lo, hi = float(g.min()), float(g.max())
        g = (g - lo) / max(hi - lo, 1e-9)
        far_white = float(g[: max(1, H // 20)].mean()) > float(g[H // 3: 2 * H // 3].mean())   # the sky row is far
        if (depth == "near is white") == far_white:
            g = 1 - g
        dep = np.repeat(g[..., None], 3, 2).astype(np.float32)
    nrm = aux(r["normal"], "normal")
    pats = lambda t: [x.strip().lower() for x in (t or "").replace("\n", ",").split(",") if x.strip()]  # noqa: E731
    inv, exc = pats(invert), pats(exclude)
    hit = lambda nm, pp: any(fnmatch.fnmatch(nm.lower(), q) for q in pp)  # noqa: E731
    masks = []
    for nm, path in list(r["masks"].items()) + list(found["shared"].items()):
        if hit(nm, exc):
            continue
        a = _read(path)
        if _is_colour(a):
            notes.append(f"{os.path.basename(path)} is a colour picture (an ID map?), not a mask: left out "
                         f"(kubakub regions from id renders reads ID maps)")
            continue
        m = _mask(a)
        if m.shape != (H, W):
            notes.append(f"mask {nm} {m.shape[1]}x{m.shape[0]} resized to the picture {W}x{H}")
            m = _fit(m, W, H, nearest=True)
        if hit(nm, inv):
            m = 1.0 - m
        masks.append((nm, np.ascontiguousarray(m, dtype=np.float32)))
    if found["skipped"]:
        notes.append(f"not read: {', '.join(found['skipped'][:6])}" + (" ..." if len(found["skipped"]) > 6 else ""))
    return {"name": pick, "beauty": beauty, "depth": dep, "normal": nrm, "masks": masks, "notes": notes,
            "renders": sorted(renders, key=str.lower)}
