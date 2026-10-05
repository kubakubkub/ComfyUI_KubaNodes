"""
align.py

Puts an edited picture back onto the picture it was made from. An image model that repaints a whole facade returns
it a little moved or scaled (a few pixels; one or two percent), which a projection on a real building cannot afford.

The fit starts from "nothing moved" and only looks for a small shift, scale and shear, on the edges of both pictures
(edges survive a restyle, colours do not), coarse to fine with OpenCV's ECC. A facade of identical windows cannot
fool it the way feature matching can, because it never searches far. A fit that does not improve the match, or that
asks for more than a small move, is refused and the picture is returned as it is.

numpy + OpenCV only (tests/test_align.py).
The problem and the idea of a node for it: github.com/Mozer/ComfyUI-PixelDriftFix (Mozer); the method here is another one.
"""

from __future__ import annotations

import cv2
import numpy as np

WORK = 1024                 # long edge of the finest level the fit runs on
MAX_SHIFT = 0.04            # of the long edge
MAX_SCALE = 0.05            # scale / shear away from 1
SECOND_START = 0.012       # a second fit starts this far off (of the long edge) ...
CERTAIN_PX = 1.0            # ... and must end within this many pixels of the first (at 1024 px; scaled with the size)
BORDER = 0.03               # share of each side the fit leaves out
MIN_GAIN = 0.01             # the edge match has to get better by at least this (correlation, 0..1)


def _gray(img):
    a = np.asarray(img)
    if a.dtype != np.float32:
        a = a.astype(np.float32) / (255.0 if a.dtype == np.uint8 else 1.0)
    return a if a.ndim == 2 else a[..., :3].mean(2)


def edges(gray, sigma=1.5):
    """Edge strength, soft and normalised: what both pictures still have in common after a restyle."""
    g = cv2.GaussianBlur(gray, (0, 0), sigma)
    gx, gy = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    e = np.sqrt(gx * gx + gy * gy)
    e = np.minimum(e, np.percentile(e, 99.5) + 1e-6)       # a few very hard edges must not carry the whole fit
    return (e / (e.max() + 1e-9)).astype(np.float32)


def _corr(a, b):
    a = a - a.mean()
    b = b - b.mean()
    return float((a * b).sum() / (np.sqrt((a * a).sum() * (b * b).sum()) + 1e-12))


def estimate(source, edited, model="affine", work=WORK):
    """
    The warp that maps source pixels to where they sit in `edited` (2x3, at the size of `source`), found on edges.
    model: 'shift' (move only), 'rigid' (move + turn), 'affine' (move, scale, shear).
    -> dict(matrix 2x3 float64, ok bool, why str, before / after edge match, shift_px, scale_x, scale_y, corner_px).
    """
    src, edt = _gray(source), _gray(edited)
    H, W = src.shape
    if edt.shape != (H, W):
        edt = cv2.resize(edt, (W, H), interpolation=cv2.INTER_AREA)
    longest = max(W, H)
    sizes = [s for s in (160, 320, 640) if s < min(work, longest)] + [min(work, longest)]
    levels = []
    for size in sizes:
        f = size / longest
        w, h = max(16, int(round(W * f))), max(16, int(round(H * f)))
        inner = np.zeros((h, w), np.uint8)             # the fit ignores a band along the border: a picture that was
        by, bx = max(2, int(h * BORDER)), max(2, int(w * BORDER))   # shifted or padded has edges there that mean nothing
        inner[by:h - by, bx:w - bx] = 255
        levels.append((f, edges(cv2.resize(src, (w, h), interpolation=cv2.INTER_AREA)),
                       edges(cv2.resize(edt, (w, h), interpolation=cv2.INTER_AREA)), inner))
    before = _corr(levels[-1][1], levels[-1][2])
    pts = np.array([[0, 0], [W, 0], [W, H], [0, H]], np.float64)
    apart = lambda m, n: float(np.linalg.norm(pts @ (m[:, :2] - n[:, :2]).T + m[:, 2] - n[:, 2], axis=1).max())  # noqa: E731
    limit = max(CERTAIN_PX, CERTAIN_PX * longest / 1024)
    first, after = _fit(levels, model, np.eye(2, 3))
    if first is None:
        return _result(np.eye(2, 3), False, "the fit did not converge (the pictures have too little in common)",
                       before, None, W, H)
    # the fit is run again from where it ended (coarse to fine once more) until it stays put: a first pass that
    # started far off can stop early on rows of identical windows
    gap = None
    for _ in range(3):
        again, cc = _fit(levels, model, first)
        if again is None:
            break
        gap = apart(first, again)
        first, after = again, cc
        if gap <= limit / 2:
            break
    r = _judge(first, before, after, W, H)
    if not (r["ok"] and r["why"] == "aligned"):
        return r
    if gap is None or gap > limit / 2:
        return _result(np.eye(2, 3), False, "the fit does not settle (too little of the source is left in the picture): "
                       "not applied", before, after, W, H)
    # and from other starts it must land on the same place. Where the picture moved in parts (the gable one way,
    # the floors another), several fits are equally good and none of them is the truth: then nothing is moved
    d = SECOND_START * longest
    worst = 0.0
    for sc, dx, dy in ((1.012, d, -d), (0.988, -d, d), (1.0, d, d), (1.0, -d, -d)):
        start = np.array([[sc, 0.0, dx + (1 - sc) * W / 2], [0.0, sc, dy + (1 - sc) * H / 2]])
        coarse = levels[:-1] if len(levels) > 2 else levels       # where it lands shows without the finest level
        other, _ = _fit(coarse, model, start)
        if other is not None:
            again, _ = _fit(coarse, model, other)
            other = again if again is not None else other
        if other is None:                              # this start found nothing: it says nothing about the first
            continue
        worst = max(worst, apart(first, other))
        if worst > 3 * limit:
            return _result(np.eye(2, 3), False, f"the fit is not certain (another start ends {worst:.1f} px away): "
                           "not applied", before, after, W, H)
    return r


def _fit(levels, model, start):
    """Coarse to fine ECC on the edge pictures. start: a 2x3 warp at full size. -> (2x3 at full size, edge match) or (None, None)."""
    motion = {"shift": cv2.MOTION_TRANSLATION, "rigid": cv2.MOTION_EUCLIDEAN, "affine": cv2.MOTION_AFFINE}[model]
    M = np.asarray(start, np.float32).copy()
    if model != "affine":
        M[:, :2] = np.eye(2)
    f_last, cc = 1.0, None
    for k, (f, a, b, inner) in enumerate(levels):
        M[:, 2] *= f / f_last
        f_last = f
        try:
            # the first level only moves: scale and shear come in once the pictures roughly sit
            cc, M = cv2.findTransformECC(a, b, M, cv2.MOTION_TRANSLATION if k == 0 and len(levels) > 1 else motion,
                                         (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 120, 1e-6), inner, 5)
        except cv2.error:
            return None, None
    full = M.astype(np.float64)                        # to the full size: pixel centres sit at +0.5
    c = 0.5 * f_last - 0.5
    full[:, 2] = (full[:, :2] @ np.array([c, c]) + full[:, 2] + 0.5) / f_last - 0.5
    return full, float(cc)


def _measure(M, W, H):
    pts = np.array([[0, 0], [W, 0], [W, H], [0, H], [W / 2, H / 2]], np.float64)
    moved = pts @ M[:, :2].T + M[:, 2] - pts
    d = np.linalg.norm(moved, axis=1)
    sx, sy = float(np.linalg.norm(M[:, 0])), float(np.linalg.norm(M[:, 1]))
    return float(d[4]), sx, sy, float(d[:4].max())


def _result(M, ok, why, before, after, W, H):
    shift, sx, sy, corner = _measure(M, W, H)
    return {"matrix": M, "ok": bool(ok), "why": why, "before": before, "after": after, "shift_px": shift,
            "scale_x": sx, "scale_y": sy, "corner_px": corner}


def _judge(M, before, after, W, H):
    shift, sx, sy, corner = _measure(M, W, H)
    longest = max(W, H)
    shear = abs(float(M[0, 0] * M[0, 1] + M[1, 0] * M[1, 1]))
    if shift > MAX_SHIFT * longest or corner > 2 * MAX_SHIFT * longest:
        return _result(np.eye(2, 3), False, f"the fit asks for a move of {corner:.0f} px, more than a drift: not applied",
                       before, after, W, H)
    if abs(sx - 1) > MAX_SCALE or abs(sy - 1) > MAX_SCALE or shear > MAX_SCALE:
        return _result(np.eye(2, 3), False, f"the fit asks for a scale of {sx:.3f} x {sy:.3f}, more than a drift: not applied",
                       before, after, W, H)
    if after is None or after < before + MIN_GAIN:
        if corner < 0.5:
            return _result(np.eye(2, 3), True, "already in place (under half a pixel)", before, after, W, H)
        return _result(np.eye(2, 3), False, "the fit does not make the pictures match better: not applied", before, after, W, H)
    return _result(M, True, "aligned", before, after, W, H)


def apply(edited, matrix, size=None):
    """`edited` (H, W, C float or uint8) warped back onto the source. matrix: from estimate(), at the source's size.
    size: (W, H) of the source when `edited` has another size (it is fitted to it first)."""
    img = np.asarray(edited)
    H, W = img.shape[:2]
    if size is not None and (W, H) != tuple(size):
        img = cv2.resize(img, tuple(size), interpolation=cv2.INTER_LANCZOS4 if size[0] > W else cv2.INTER_AREA)
        W, H = size
    M = np.asarray(matrix, np.float64)
    if np.allclose(M, np.eye(2, 3), atol=1e-9):
        return img.copy()
    out = cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE)
    return np.clip(out, 0, 1) if out.dtype.kind == "f" else out


def align(source, edited, model="affine"):
    """-> (edited put back onto source, result dict of estimate())."""
    r = estimate(source, edited, model)
    src = np.asarray(source)
    return apply(edited, r["matrix"], (src.shape[1], src.shape[0])), r


def describe(r):
    """One line for a node's report."""
    if r["before"] is None:
        return r["why"]
    head = f"{r['why']}: "
    if r["ok"] and r["why"] == "aligned":
        head += (f"it sat {r['corner_px']:.1f} px off at the corners ({r['shift_px']:.1f} px in the middle, scale "
                 f"{r['scale_x']:.4f} x {r['scale_y']:.4f}); ")
    return head + f"edge match {r['before']:.3f}" + (f" -> {r['after']:.3f}" if r["after"] is not None else "")
