"""
relief_core.py

Pure numpy geometry core for ReliefForge. No torch, no scipy, no ComfyUI
imports, so it can be unit tested outside of ComfyUI.

Everything works on a heightfield H in millimetres:

    H[row, col]  = relief depth in mm, measured outward from the panel plane
    px_mm        = size of one pixel in mm (isotropic)

Row 0 is the TOP of the panel. Physical height above ground of row y is

    z(y) = panel_bottom_mm + (rows - 1 - y) * px_mm
"""

from __future__ import annotations

import math
import struct

import numpy as np

__all__ = [
    "depth_to_height",
    "round_edges",
    "limit_slope",
    "ledge_analysis",
    "deslope_ledges",
    "mass_estimate",
    "grid_mesh",
    "write_binary_stl",
    "write_prow_obj",
]


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------

def _shift(a: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """Shift with edge replication (no wraparound)."""
    pad_y = (max(dy, 0), max(-dy, 0))
    pad_x = (max(dx, 0), max(-dx, 0))
    p = np.pad(a, (pad_y, pad_x), mode="edge")
    h, w = a.shape
    y0 = pad_y[0] - dy
    x0 = pad_x[0] - dx
    return p[y0:y0 + h, x0:x0 + w]


def _disk_offsets(radius_px: int):
    """(dy, dx, distance_in_px) for every offset inside the disk."""
    out = []
    r = int(radius_px)
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            d = math.hypot(dy, dx)
            if d <= r:
                out.append((dy, dx, d))
    return out


def _resize_nearest(a: np.ndarray, shape) -> np.ndarray:
    hy = (np.linspace(0, a.shape[0] - 1, shape[0])).round().astype(np.int64)
    hx = (np.linspace(0, a.shape[1] - 1, shape[1])).round().astype(np.int64)
    return a[hy][:, hx]


# --------------------------------------------------------------------------
# 1. depth map to physical heightfield
# --------------------------------------------------------------------------

def depth_to_height(
    depth: np.ndarray,
    max_relief_mm: float,
    invert: bool = False,
    gamma: float = 1.0,
    flatten_percentile: float = 0.0,
) -> np.ndarray:
    """Normalise an arbitrary depth map to a heightfield in mm.

    depth              2D float array, any range
    max_relief_mm      the deepest point of the relief, in mm
    invert             True if the source is "near = bright" the wrong way round
    gamma              < 1 lifts the mid tones (more shallow detail), > 1 deepens
    flatten_percentile clip this percentile off both ends before normalising,
                       which kills single-pixel outliers from depth models
    """
    d = np.asarray(depth, dtype=np.float64)
    if d.ndim == 3:
        d = d.mean(axis=2)

    if flatten_percentile > 0.0:
        lo = np.percentile(d, flatten_percentile)
        hi = np.percentile(d, 100.0 - flatten_percentile)
    else:
        lo, hi = float(d.min()), float(d.max())
    if hi - lo < 1e-9:
        return np.zeros_like(d, dtype=np.float32)

    n = np.clip((d - lo) / (hi - lo), 0.0, 1.0)
    if invert:
        n = 1.0 - n
    if gamma != 1.0:
        n = np.power(n, float(gamma))
    return (n * float(max_relief_mm)).astype(np.float32)


# --------------------------------------------------------------------------
# 2. minimum edge radius (rolling ball)
# --------------------------------------------------------------------------

def _ball_dilate(h, offsets, radius_mm, px_mm):
    out = np.full_like(h, -np.inf)
    r2 = radius_mm * radius_mm
    for dy, dx, d in offsets:
        dmm = d * px_mm
        if dmm > radius_mm:
            continue
        cap = math.sqrt(max(r2 - dmm * dmm, 0.0))
        np.maximum(out, _shift(h, dy, dx) + cap, out=out)
    return out


def _ball_erode(h, offsets, radius_mm, px_mm):
    out = np.full_like(h, np.inf)
    r2 = radius_mm * radius_mm
    for dy, dx, d in offsets:
        dmm = d * px_mm
        if dmm > radius_mm:
            continue
        cap = math.sqrt(max(r2 - dmm * dmm, 0.0))
        np.minimum(out, _shift(h, dy, dx) - cap, out=out)
    return out


def round_edges(
    h: np.ndarray,
    px_mm: float,
    radius_mm: float,
    max_work_px: int = 1024,
) -> np.ndarray:
    """Guarantee a minimum edge radius everywhere.

    Grayscale closing then opening with a spherical structuring element. A
    physical ball of radius r can then touch every point of the surface from
    both sides, which is exactly what "no sharp edges" means to a safety
    inspector and to a CNC cutter with a ball nose bit.
    """
    if radius_mm <= 0:
        return h.astype(np.float32)

    work = h.astype(np.float64)
    src_shape = h.shape
    scale = 1.0
    long_side = max(src_shape)
    if max_work_px and long_side > max_work_px:
        scale = max_work_px / float(long_side)
        new_shape = (max(int(round(src_shape[0] * scale)), 8),
                     max(int(round(src_shape[1] * scale)), 8))
        work = _resize_nearest(work, new_shape)
    work_px_mm = px_mm / scale

    r_px = int(math.ceil(radius_mm / work_px_mm))
    r_px = max(1, min(r_px, 32))
    offsets = _disk_offsets(r_px)

    closed = _ball_erode(_ball_dilate(work, offsets, radius_mm, work_px_mm),
                         offsets, radius_mm, work_px_mm)
    opened = _ball_dilate(_ball_erode(closed, offsets, radius_mm, work_px_mm),
                          offsets, radius_mm, work_px_mm)

    if opened.shape != src_shape:
        opened = _resize_nearest(opened, src_shape)
    return opened.astype(np.float32)


# --------------------------------------------------------------------------
# 3. draft angle / demouldability
# --------------------------------------------------------------------------

def limit_slope(
    h: np.ndarray,
    px_mm: float,
    draft_angle_deg: float,
    max_work_px: int = 1024,
) -> np.ndarray:
    """Limit the surface slope so the panel releases from a one piece mould.

    draft_angle_deg is measured from the pull direction (the panel normal).
    3 degrees is the usual minimum for GFRC on a milled foam mould, 5 is
    comfortable. The result is the largest surface below h that is Lipschitz
    with constant cot(draft), computed as an infimal convolution with a cone.
    """
    a = math.radians(max(float(draft_angle_deg), 0.05))
    slope = 1.0 / math.tan(a)          # mm of height per mm of run

    work = h.astype(np.float64)
    src_shape = h.shape
    scale = 1.0
    long_side = max(src_shape)
    if max_work_px and long_side > max_work_px:
        scale = max_work_px / float(long_side)
        work = _resize_nearest(work, (max(int(round(src_shape[0] * scale)), 8),
                                      max(int(round(src_shape[1] * scale)), 8)))
    work_px_mm = px_mm / scale

    span = float(work.max() - work.min())
    r_px = int(math.ceil(span / max(slope * work_px_mm, 1e-6)))
    r_px = max(1, min(r_px, 48))

    out = np.full_like(work, np.inf)
    for dy, dx, d in _disk_offsets(r_px):
        np.minimum(out, _shift(work, dy, dx) + slope * d * work_px_mm, out=out)

    if out.shape != src_shape:
        out = _resize_nearest(out, src_shape)
    return np.minimum(out, h).astype(np.float32)


# --------------------------------------------------------------------------
# 4. the rule check
# --------------------------------------------------------------------------

def ledge_analysis(
    h: np.ndarray,
    px_mm: float,
    panel_bottom_mm: float = 0.0,
    reach_mm: float = 2500.0,
    foothold_depth_mm: float = 25.0,
    min_width_mm: float = 80.0,
    clearance_mm: float = 150.0,
    sit_depth_mm: float = 150.0,
    sit_low_mm: float = 300.0,
    sit_high_mm: float = 750.0,
    lie_depth_mm: float = 450.0,
    lie_max_mm: float = 700.0,
):
    """Find every surface a person could stand, climb, sit or lie on.

    A foothold exists where the relief steps outward by at least
    foothold_depth_mm relative to everything within clearance_mm above it,
    over a continuous horizontal run of at least min_width_mm, and where that
    surface is within reach from the ground.

    Returns (masks dict, report dict).
    """
    h = h.astype(np.float64)
    rows, cols = h.shape

    clear_px = max(int(round(clearance_mm / px_mm)), 1)
    width_px = max(int(round(min_width_mm / px_mm)), 1)

    # highest relief found in the band directly above each pixel
    above = np.full_like(h, -np.inf)
    for k in range(1, clear_px + 1):
        np.maximum(above, _shift(h, k, 0), out=above)   # dy>0 pulls from above
    step_out = h - above

    z = panel_bottom_mm + (rows - 1 - np.arange(rows)) * px_mm
    z_map = np.repeat(z[:, None], cols, axis=1)

    def _run_filter(mask, w):
        """Keep only pixels sitting in a horizontal run of at least w."""
        m = mask.astype(bool)
        eroded = m.copy()
        half = w // 2
        for dx in range(-half, w - half):
            eroded &= _shift(m.astype(np.float64), 0, dx) > 0.5
        out = np.zeros_like(m)
        for dx in range(-half, w - half):
            out |= _shift(eroded.astype(np.float64), 0, dx) > 0.5
        return out

    raw_foot = (step_out >= foothold_depth_mm) & (z_map <= reach_mm)
    foot = _run_filter(raw_foot, width_px)

    raw_sit = ((step_out >= sit_depth_mm)
               & (z_map >= sit_low_mm) & (z_map <= sit_high_mm))
    sit = _run_filter(raw_sit, max(int(round(400.0 / px_mm)), 1))

    raw_lie = (step_out >= lie_depth_mm) & (z_map <= lie_max_mm)
    lie = _run_filter(raw_lie, max(int(round(1200.0 / px_mm)), 1))

    px_area_m2 = (px_mm / 1000.0) ** 2
    report = {
        "footholds_found": int(foot.sum()),
        "foothold_area_m2": round(float(foot.sum()) * px_area_m2, 4),
        "deepest_foothold_mm": round(float(step_out[foot].max()), 1) if foot.any() else 0.0,
        "highest_foothold_mm_above_ground": round(float(z_map[foot].max()), 0) if foot.any() else 0.0,
        "sitting_surfaces_found": int(sit.sum()),
        "lying_surfaces_found": int(lie.sum()),
        "max_relief_mm": round(float(h.max() - h.min()), 1),
        "passes_climbing_rule": bool(not foot.any()),
        "passes_sitting_rule": bool(not sit.any()),
        "passes_lying_rule": bool(not lie.any()),
    }
    report["passes"] = (report["passes_climbing_rule"]
                        and report["passes_sitting_rule"]
                        and report["passes_lying_rule"])
    masks = {"foothold": foot, "sitting": sit, "lying": lie, "step_out": step_out}
    return masks, report


def deslope_ledges(
    h: np.ndarray,
    px_mm: float,
    sill_angle_deg: float = 45.0,
    max_span_mm: float = 400.0,
) -> np.ndarray:
    """Shave every horizontal ledge into a sloped sill.

    Anisotropic infimal convolution, upward direction only: the relief may
    only grow outward as it descends, and no faster than the sill angle
    allows. Vertical faces, reveals and cornice undersides are untouched, the
    flat top faces that a foot needs are gone. Runs after round_edges so the
    new sill still carries its radius.

    sill_angle_deg is measured from horizontal. 45 is the common anti perch
    value, 60 is aggressive and eats more of the facade.
    """
    ang = math.radians(min(max(float(sill_angle_deg), 5.0), 85.0))
    ratio = 1.0 / math.tan(ang)          # mm of extra protrusion per mm down
    k_max = max(int(round(max_span_mm / px_mm)), 1)

    work = h.astype(np.float64)
    out = work.copy()
    for k in range(1, k_max + 1):
        np.minimum(out, _shift(work, k, 0) + ratio * k * px_mm, out=out)
    return out.astype(np.float32)


# --------------------------------------------------------------------------
# 5. mass and cost sanity check
# --------------------------------------------------------------------------

def mass_estimate(
    h: np.ndarray,
    px_mm: float,
    shell_thickness_mm: float = 15.0,
    density_kg_m3: float = 2100.0,
    frame_kg_per_m2: float = 18.0,
    footprint_m2: float = 1.0,
):
    """Shell mass for a GFRC panel plus a steel frame allowance.

    The shell follows the relief, so its area is the true surface area, not the
    flat projection. Returns a dict in kg, and the load per square metre of
    footprint, which is the number the call caps at 1000 kg/m2.
    """
    h = h.astype(np.float64)
    rows, cols = h.shape
    gy, gx = np.gradient(h, px_mm)
    # true surface area of a heightfield
    area_px_m2 = ((px_mm / 1000.0) ** 2) * np.sqrt(1.0 + gx ** 2 + gy ** 2)
    surface_m2 = float(area_px_m2.sum())
    flat_m2 = (rows * px_mm / 1000.0) * (cols * px_mm / 1000.0)

    shell_kg = surface_m2 * (shell_thickness_mm / 1000.0) * density_kg_m3
    frame_kg = flat_m2 * frame_kg_per_m2
    total = shell_kg + frame_kg
    return {
        "flat_panel_m2": round(flat_m2, 3),
        "true_surface_m2": round(surface_m2, 3),
        "surface_factor": round(surface_m2 / flat_m2, 3) if flat_m2 else 0.0,
        "shell_kg": round(shell_kg, 1),
        "frame_kg": round(frame_kg, 1),
        "panel_total_kg": round(total, 1),
        "kg_per_m2_of_footprint": round(total / footprint_m2, 1) if footprint_m2 else 0.0,
    }


# --------------------------------------------------------------------------
# 6. mesh output
# --------------------------------------------------------------------------

def grid_mesh(h: np.ndarray, px_mm: float, base_mm: float = 0.0, step: int = 1):
    """Heightfield to a closed solid: top surface, flat back, side walls.

    Local frame: x across the panel, y up the panel, z outward.
    """
    hh = h[::step, ::step].astype(np.float64)
    rows, cols = hh.shape
    sx = px_mm * step

    xs = np.arange(cols) * sx
    ys = (rows - 1 - np.arange(rows)) * sx
    X, Y = np.meshgrid(xs, ys)
    Ztop = base_mm + hh

    top = np.stack([X, Y, Ztop], axis=-1).reshape(-1, 3)
    back = np.stack([X, Y, np.zeros_like(Ztop)], axis=-1).reshape(-1, 3)
    verts = np.vstack([top, back])
    n = rows * cols

    faces = []

    def idx(r, c, layer=0):
        return layer * n + r * cols + c

    for r in range(rows - 1):
        for c in range(cols - 1):
            a, b = idx(r, c), idx(r, c + 1)
            d, e = idx(r + 1, c), idx(r + 1, c + 1)
            faces.append((a, d, b))
            faces.append((b, d, e))
            a, b = idx(r, c, 1), idx(r, c + 1, 1)
            d, e = idx(r + 1, c, 1), idx(r + 1, c + 1, 1)
            faces.append((a, b, d))
            faces.append((b, e, d))

    for c in range(cols - 1):
        for r in (0, rows - 1):
            a, b = idx(r, c), idx(r, c + 1)
            a2, b2 = idx(r, c, 1), idx(r, c + 1, 1)
            if r == 0:
                faces.append((a, b, a2))
                faces.append((b, b2, a2))
            else:
                faces.append((a, a2, b))
                faces.append((b, a2, b2))
    for r in range(rows - 1):
        for c in (0, cols - 1):
            a, b = idx(r, c), idx(r + 1, c)
            a2, b2 = idx(r, c, 1), idx(r + 1, c, 1)
            if c == 0:
                faces.append((a, a2, b))
                faces.append((b, a2, b2))
            else:
                faces.append((a, b, a2))
                faces.append((b, b2, a2))

    return verts.astype(np.float32), np.asarray(faces, dtype=np.int64)


def write_binary_stl(path, verts: np.ndarray, faces: np.ndarray) -> str:
    v = verts.astype(np.float32)
    tri = v[faces]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    ln = np.linalg.norm(n, axis=1, keepdims=True)
    n = np.divide(n, np.where(ln == 0, 1.0, ln))
    with open(path, "wb") as f:
        f.write(b"ReliefForge" + b"\0" * (80 - 11))
        f.write(struct.pack("<I", len(faces)))
        for i in range(len(faces)):
            f.write(struct.pack("<3f", *n[i]))
            f.write(struct.pack("<3f", *tri[i, 0]))
            f.write(struct.pack("<3f", *tri[i, 1]))
            f.write(struct.pack("<3f", *tri[i, 2]))
            f.write(b"\0\0")
    return str(path)


def _fold(verts: np.ndarray, angle_deg: float, mirror: bool) -> np.ndarray:
    """Place a panel on one wing of a prow.

    The fold edge is the world Z axis. The prow tip points towards -Y, the two
    wings open towards +Y with the full interior angle angle_deg between them.
    Panel local x runs away from the tip, local y is up, local z is outward.
    """
    a = math.radians(angle_deg) / 2.0
    s = -1.0 if mirror else 1.0
    x, y, z = verts[:, 0], verts[:, 1], verts[:, 2]
    dir_x, dir_y = s * math.sin(a), math.cos(a)
    nor_x, nor_y = s * math.cos(a), -math.sin(a)
    wx = x * dir_x + z * nor_x
    wy = x * dir_y + z * nor_y
    return np.stack([wx, wy, y], axis=-1).astype(np.float32)


def write_prow_obj(path, panels, px_mm: float, angle_deg: float,
                   base_mm: float = 0.0, step: int = 2) -> str:
    """Two heightfields folded into a prow, written as a single OBJ.

    panels is a sequence of one or two 2D arrays. With one panel it is
    mirrored onto both wings.
    """
    if len(panels) == 1:
        panels = [panels[0], panels[0][:, ::-1]]
    chunks, offset = [], 0
    lines = ["# ReliefForge prow", f"# interior angle {angle_deg} deg"]
    for i, hp in enumerate(panels[:2]):
        v, f = grid_mesh(hp, px_mm, base_mm, step=step)
        v = _fold(v, angle_deg, mirror=(i == 1))
        chunks.append((v, f + offset))
        offset += len(v)
    for v, _ in chunks:
        for p in v:
            lines.append(f"v {p[0]:.3f} {p[1]:.3f} {p[2]:.3f}")
    for i, (_, f) in enumerate(chunks):
        lines.append(f"g wing_{i}")
        for t in f:
            lines.append(f"f {t[0] + 1} {t[1] + 1} {t[2] + 1}")
    with open(path, "w") as fh:
        fh.write("\n".join(lines))
    return str(path)
