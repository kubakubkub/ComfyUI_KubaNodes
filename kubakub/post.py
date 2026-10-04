"""
post.py

Post production for frames (logic for nodes_post.py; numpy + opencv + Pillow, no ComfyUI imports, tests/test_post.py):
colour match to a reference (one transform for all frames, bakeable as a .cube LUT), .cube LUTs, deflicker (per
frame colour statistics on a coarse grid smoothed over time), retime (optical flow or blend) and burn-in text.
Frames are sRGB float32 HxWx3 in 0..1; gains and blends work in linear light where it matters.
"""

from __future__ import annotations

import math
import os
import re
import threading

import cv2
import numpy as np

from .scene3d.compensate import _DEC, _ENC, _N


def to_linear(x):
    return _DEC[(np.clip(x, 0, 1) * _N + 0.5).astype(np.uint16)]


def to_srgb(x):
    return _ENC[(np.clip(x, 0, 1) * _N + 0.5).astype(np.uint16)]


# ------------------------------------------------------------------------------------------------- colour match


def _sample(img, n=200_000, seed=0):
    px = img.reshape(-1, 3)
    if len(px) > n:
        px = px[np.random.default_rng(seed).choice(len(px), n, replace=False)]
    return px.astype(np.float64)


def colour_fit(src, ref, method="mkl"):
    """A transform (mean_s, matrix, mean_r) that moves src's colours to ref's: out = (x - ms) @ M.T + mr."""
    a, b = _sample(src), _sample(ref)
    ma, mb = a.mean(0), b.mean(0)
    if method == "mean_std":
        m = np.diag(np.clip(b.std(0) / np.maximum(a.std(0), 1e-4), 0.2, 5.0))
    elif method == "mkl":                                    # Monge-Kantorovich linear (Pitie & Kokaram 2007)
        ca = np.cov(a.T) + np.eye(3) * 1e-6
        cb = np.cov(b.T) + np.eye(3) * 1e-6
        ea, va = np.linalg.eigh(ca)
        ca_h = va @ np.diag(np.sqrt(np.maximum(ea, 1e-10))) @ va.T
        ca_ih = va @ np.diag(1 / np.sqrt(np.maximum(ea, 1e-10))) @ va.T
        em, vm = np.linalg.eigh(ca_h @ cb @ ca_h)
        m = ca_ih @ (vm @ np.diag(np.sqrt(np.maximum(em, 0))) @ vm.T) @ ca_ih
    else:
        raise ValueError(f"unknown method {method}")
    return ma, m, mb


def colour_apply(img, fit, strength=1.0):
    ma, m, mb = fit
    out = (img.reshape(-1, 3).astype(np.float64) - ma) @ m.T + mb
    out = np.clip(out.reshape(img.shape), 0, 1).astype(np.float32)
    return img + (out - img) * float(strength) if strength < 1 else out


# ---------------------------------------------------------------------------------------------------------- LUTs


def parse_cube(text):
    """.cube (Adobe / Resolve) -> {"size", "dim": 1 | 3, "table": (N,3) or (N,N,N,3) indexed [b, g, r], "min", "max"}."""
    size, dim, dmin, dmax, rows = None, 3, np.zeros(3), np.ones(3), []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key = line.split()[0].upper()
        if key == "LUT_3D_SIZE":
            size, dim = int(line.split()[1]), 3
        elif key == "LUT_1D_SIZE":
            size, dim = int(line.split()[1]), 1
        elif key == "DOMAIN_MIN":
            dmin = np.array([float(v) for v in line.split()[1:4]])
        elif key == "DOMAIN_MAX":
            dmax = np.array([float(v) for v in line.split()[1:4]])
        elif re.match(r"^[-+0-9.eE]", line):
            rows.append([float(v) for v in line.split()[:3]])
    if size is None:
        raise ValueError(".cube without LUT_3D_SIZE / LUT_1D_SIZE")
    t = np.asarray(rows, np.float32)
    need = size ** 3 if dim == 3 else size
    if len(t) != need:
        raise ValueError(f".cube has {len(t)} rows, expected {need}")
    table = t.reshape(size, size, size, 3) if dim == 3 else t
    return {"size": size, "dim": dim, "table": table, "min": dmin.astype(np.float32), "max": dmax.astype(np.float32)}


def apply_lut(img, lut, strength=1.0):
    """Trilinear 3D (or per channel 1D) lookup."""
    n = lut["size"]
    x = (np.clip(img, lut["min"], lut["max"]) - lut["min"]) / np.maximum(lut["max"] - lut["min"], 1e-6) * (n - 1)
    if lut["dim"] == 1:
        out = np.stack([np.interp(x[..., c], np.arange(n), lut["table"][:, c]) for c in range(3)], -1)
    else:
        i0 = np.minimum(np.floor(x).astype(np.int32), n - 2)
        f = (x - i0).astype(np.float32)
        r0, g0, b0 = i0[..., 0], i0[..., 1], i0[..., 2]
        fr, fg, fb = f[..., :1], f[..., 1:2], f[..., 2:3]
        T = lut["table"]

        def at(db, dg, dr):
            return T[b0 + db, g0 + dg, r0 + dr]
        c00 = at(0, 0, 0) * (1 - fr) + at(0, 0, 1) * fr
        c01 = at(0, 1, 0) * (1 - fr) + at(0, 1, 1) * fr
        c10 = at(1, 0, 0) * (1 - fr) + at(1, 0, 1) * fr
        c11 = at(1, 1, 0) * (1 - fr) + at(1, 1, 1) * fr
        out = (c00 * (1 - fg) + c01 * fg) * (1 - fb) + (c10 * (1 - fg) + c11 * fg) * fb
    out = np.clip(out, 0, 1).astype(np.float32)
    return img + (out - img) * float(strength) if strength < 1 else out


def write_cube(path, fn, size=33, title="kubakub"):
    """Bake any colour function (HxWx3 -> HxWx3) into a .cube file."""
    g = np.linspace(0, 1, size, dtype=np.float32)
    b, gg, r = np.meshgrid(g, g, g, indexing="ij")
    grid = np.stack([r, gg, b], -1).reshape(1, -1, 3)             # red fastest, as .cube wants
    out = fn(grid).reshape(-1, 3)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(f'TITLE "{title}"\nLUT_3D_SIZE {size}\nDOMAIN_MIN 0 0 0\nDOMAIN_MAX 1 1 1\n')
        f.writelines(f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n" for v in out)
    return path


# ------------------------------------------------------------------------------------------------------ deflicker


def deflicker_gains(frames, window=9, grid=8, strength=1.0):
    """Per frame a gain (grid x grid x 3, linear light) that moves its local colour means to their average over
    `window` frames around it. frames: list / array of sRGB HxWx3."""
    means = np.stack([cv2.resize(to_linear(f), (grid, grid), interpolation=cv2.INTER_AREA) for f in frames])
    logm = np.log(np.maximum(means, 1e-4))
    sigma = max(window / 4.0, 0.5)
    r = int(math.ceil(2.5 * sigma))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    pad = np.pad(logm, ((r, r), (0, 0), (0, 0), (0, 0)), mode="reflect" if len(frames) > r else "edge")
    smooth = np.stack([np.tensordot(k, pad[i:i + 2 * r + 1], axes=(0, 0)) for i in range(len(frames))])
    return np.exp((smooth - logm) * float(strength)).astype(np.float32)


def apply_gain(frame, gain_grid):
    h, w = frame.shape[:2]
    g = cv2.resize(gain_grid, (w, h), interpolation=cv2.INTER_CUBIC)
    return to_srgb(to_linear(frame) * np.maximum(g, 0))


# --------------------------------------------------------------------------------------------------------- retime


def retime_times(n, speed=1.0, out_frames=0):
    """Source times (float frame positions) of the output frames: speed 0.5 = twice as many frames (slow motion)."""
    if n <= 1:
        return np.zeros(max(1, out_frames or 1))
    m = int(out_frames) if out_frames > 0 else max(1, int(round((n - 1) / float(speed))) + 1)
    return np.linspace(0, n - 1, m) if out_frames > 0 else np.minimum(np.arange(m) * float(speed), n - 1)


_TLS = threading.local()                 # one DIS object per thread: sharing one crashes the process (access violation)


def _flow(a, b, scale=0.5):
    dis = getattr(_TLS, "dis", None)
    if dis is None:
        dis = _TLS.dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    ga = cv2.cvtColor((a * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    gb = cv2.cvtColor((b * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    h, w = ga.shape
    if scale < 1:
        ga = cv2.resize(ga, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        gb = cv2.resize(gb, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    f = dis.calc(ga, gb, None)
    if scale < 1:
        f = cv2.resize(f, (w, h), interpolation=cv2.INTER_LINEAR) / scale
    return f


def between(a, b, t, mode="flow"):
    """The frame at t (0..1) between a and b: blend, or both warped along the optical flow and blended."""
    if t <= 1e-6:
        return a
    if t >= 1 - 1e-6:
        return b
    if mode == "blend":
        return (a * (1 - t) + b * t).astype(np.float32)
    h, w = a.shape[:2]
    fab, fba = _flow(a, b), _flow(b, a)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    wa = cv2.remap(a, xx - fab[..., 0] * t, yy - fab[..., 1] * t, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    wb = cv2.remap(b, xx - fba[..., 0] * (1 - t), yy - fba[..., 1] * (1 - t), cv2.INTER_LINEAR,
                   borderMode=cv2.BORDER_REPLICATE)
    return (wa * (1 - t) + wb * t).astype(np.float32)


# -------------------------------------------------------------------------------------------------------- burn in


def timecode(i, fps):
    fr = int(round(fps))
    s = i // fr
    return f"{s // 3600:02d}:{s // 60 % 60:02d}:{s % 60:02d}:{i % fr:02d}"


def burn_text(frame, text, position="bottom left", size=0.03, opacity=0.8):
    """Text in a dark box on the frame (PIL's built-in font, scalable)."""
    from PIL import Image, ImageDraw, ImageFont
    h, w = frame.shape[:2]
    px = max(10, int(h * size))
    try:
        font = ImageFont.load_default(size=px)
    except TypeError:                                          # Pillow < 10.1
        font = ImageFont.load_default()
    img = Image.fromarray((np.clip(frame, 0, 1) * 255).astype(np.uint8))
    over = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(over)
    box = d.multiline_textbbox((0, 0), text, font=font, spacing=px // 4)
    tw, th = box[2] - box[0], box[3] - box[1]
    m = px // 2
    x = m if "left" in position else (w - tw - 3 * m if "right" in position else (w - tw) // 2 - m)
    y = m if "top" in position else h - th - 3 * m
    d.rectangle([x, y, x + tw + 2 * m, y + th + 2 * m], fill=(0, 0, 0, int(160 * opacity)))
    d.multiline_text((x + m - box[0], y + m - box[1]), text, font=font, fill=(255, 255, 255, int(255 * opacity)),
                     spacing=px // 4)
    out = Image.alpha_composite(img.convert("RGBA"), over).convert("RGB")
    return np.asarray(out, np.float32) / 255.0
