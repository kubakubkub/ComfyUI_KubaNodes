"""
mesh_to_field_Kub.py

Brings a finished mesh back into the ReliefForge heightfield world.

Export the kitbashed block from Houdini as OBJ, point these nodes at it, and
you get the millimetre heightfield that kubakub relief safety check, kubakub relief mould prep
and kubakub relief mass estimate already speak. Works the same for a TRELLIS.2 mesh.

OBJ is read by a reader in this file, so nothing new has to be installed.
GLB, GLTF, PLY and STL are read through trimesh if it happens to be available.

Requires kubakub/relief_core.py.
"""

from __future__ import annotations

import json
import math
import os

import numpy as np
import torch

import folder_paths

from ...kubakub import relief_core as rc

CAT = "kubakub/3d/fabricate/relief"
FIELD = "RELIEFFORGE_FIELD"

_UNIT_MM = {"mm": 1.0, "cm": 10.0, "m": 1000.0, "inch": 25.4}


# --------------------------------------------------------------------------
# mesh loading
# --------------------------------------------------------------------------

def load_obj(path):
    """Minimal OBJ reader: vertices and faces, n-gons fanned into triangles.

    Ignores normals, uvs, materials and groups on purpose. Handles the
    v/vt/vn index form and negative (relative) indices.
    """
    verts, faces = [], []
    with open(path, "r", errors="ignore") as fh:
        for line in fh:
            if not line or line[0] not in "vf":
                continue
            parts = line.split()
            if not parts:
                continue
            if parts[0] == "v":
                verts.append((float(parts[1]), float(parts[2]), float(parts[3])))
            elif parts[0] == "f":
                idx = []
                for tok in parts[1:]:
                    s = tok.split("/")[0]
                    if not s:
                        continue
                    i = int(s)
                    idx.append(i - 1 if i > 0 else len(verts) + i)
                for k in range(1, len(idx) - 1):
                    faces.append((idx[0], idx[k], idx[k + 1]))
    if not verts or not faces:
        raise ValueError(f"no geometry found in {path}")
    return np.asarray(verts, np.float64), np.asarray(faces, np.int64)


def load_mesh(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".obj":
        return load_obj(path)
    try:
        import trimesh
    except ImportError as exc:
        raise ImportError(
            f"{ext} needs trimesh. Either pip install trimesh into the ComfyUI "
            "environment, or export the mesh as OBJ, which is read natively."
        ) from exc
    m = trimesh.load(path, force="mesh")
    return np.asarray(m.vertices, np.float64), np.asarray(m.faces, np.int64)


# --------------------------------------------------------------------------
# orthographic rasterization
# --------------------------------------------------------------------------

def _basis(azimuth_deg, elevation_deg, up_axis="Y"):
    """View direction and the screen axes for an orthographic camera.

    Returns (right, up, view) where view points from the camera into the scene.
    Azimuth turns around the up axis, elevation tilts down from horizontal.
    """
    az = math.radians(azimuth_deg)
    el = math.radians(elevation_deg)
    # direction from the scene towards the camera, in a Y-up frame
    dx = math.cos(el) * math.sin(az)
    dy = math.sin(el)
    dz = math.cos(el) * math.cos(az)
    if up_axis.upper() == "Y":
        to_cam = np.array([dx, dy, dz])
        up_world = np.array([0.0, 1.0, 0.0])
    else:  # Z up
        to_cam = np.array([dx, dz, dy])
        up_world = np.array([0.0, 0.0, 1.0])
    to_cam /= np.linalg.norm(to_cam)
    if abs(float(np.dot(to_cam, up_world))) > 0.999:
        up_world = np.array([0.0, 0.0, 1.0]) if up_axis.upper() == "Y" else np.array([0.0, 1.0, 0.0])
    right = np.cross(up_world, to_cam)
    right /= np.linalg.norm(right)
    up = np.cross(to_cam, right)
    return right, up, -to_cam


def rasterize_depth(verts, faces, azimuth_deg=0.0, elevation_deg=0.0,
                    resolution=768, up_axis="Y", pad_frac=0.02):
    """Orthographic depth buffer plus a count of front facing surfaces.

    Returns (depth, hit, front_layers, info). depth is in mesh units measured
    towards the camera, so larger is nearer. front_layers above 1 marks an
    undercut: geometry hidden behind other geometry along this direction, which
    no single sided mould or single milling setup can reach.
    """
    right, up, view = _basis(azimuth_deg, elevation_deg, up_axis)
    P = np.stack([verts @ right, verts @ up, verts @ (-view)], axis=-1)

    xmin, ymin = P[:, 0].min(), P[:, 1].min()
    xmax, ymax = P[:, 0].max(), P[:, 1].max()
    bbox_min_up = float(ymin)          # before padding, so ground level stays true
    w, h = xmax - xmin, ymax - ymin
    pad = max(w, h) * pad_frac
    xmin -= pad; xmax += pad; ymin -= pad; ymax += pad
    span = max(xmax - xmin, 1e-9)
    px = span / float(resolution)
    cols = int(resolution)
    rows = max(int(math.ceil((ymax - ymin) / px)), 2)

    tri = P[faces]
    # screen space winding tells us which triangles face the camera
    e1 = tri[:, 1, :2] - tri[:, 0, :2]
    e2 = tri[:, 2, :2] - tri[:, 0, :2]
    area2 = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]
    keep = np.abs(area2) > 1e-12
    tri, area2 = tri[keep], area2[keep]
    front = area2 > 0

    depth = np.full((rows, cols), -np.inf)
    layers = np.zeros((rows, cols), np.int32)

    gx = xmin + (np.arange(cols) + 0.5) * px
    gy = ymax - (np.arange(rows) + 0.5) * px      # row 0 is the top

    for t in range(len(tri)):
        a, b, c = tri[t]
        x0 = max(int((min(a[0], b[0], c[0]) - xmin) / px) - 1, 0)
        x1 = min(int((max(a[0], b[0], c[0]) - xmin) / px) + 2, cols)
        y0 = max(int((ymax - max(a[1], b[1], c[1])) / px) - 1, 0)
        y1 = min(int((ymax - min(a[1], b[1], c[1])) / px) + 2, rows)
        if x0 >= x1 or y0 >= y1:
            continue

        X, Y = np.meshgrid(gx[x0:x1], gy[y0:y1])
        d = area2[t]
        w0 = ((b[0] - a[0]) * (Y - a[1]) - (b[1] - a[1]) * (X - a[0])) / d
        w1 = ((c[0] - b[0]) * (Y - b[1]) - (c[1] - b[1]) * (X - b[0])) / d
        w2 = ((a[0] - c[0]) * (Y - c[1]) - (a[1] - c[1]) * (X - c[0])) / d
        inside = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
        if not inside.any():
            continue
        z = w1 * a[2] + w2 * b[2] + w0 * c[2]

        sub = depth[y0:y1, x0:x1]
        np.copyto(sub, np.maximum(sub, z), where=inside)
        if front[t]:
            layers[y0:y1, x0:x1] += inside.astype(np.int32)

    hit = np.isfinite(depth)
    info = {
        "triangles": int(len(tri)),
        "pixel_size_mesh_units": float(px),
        "view_width_units": float(xmax - xmin),
        "view_height_units": float(ymax - ymin),
        "bbox_min_up_units": bbox_min_up,
        "resolution": [rows, cols],
    }
    return depth, hit, layers, info


# --------------------------------------------------------------------------
# node 1: one view to a field
# --------------------------------------------------------------------------

def _mesh_stamp(mesh_path, **_):
    """The mesh file's path + size + mtime: an edited mesh re-runs the node (the widget text alone does not change)."""
    p = str(mesh_path or "").strip().strip('"')
    try:
        st = os.stat(p)
        return f"{p}|{st.st_size}|{st.st_mtime_ns}"
    except OSError:
        return p


class MeshOrthoFieldKub:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh_path": ("STRING", {"default": "", "multiline": False,
                                         "tooltip": "Full path of the mesh file. OBJ always works; GLB, GLTF, PLY "
                                                    "and STL need trimesh installed. The node re-runs when the file "
                                                    "changes."}),
                "mesh_units": (["mm", "cm", "m", "inch"], {"default": "m",
                                                          "tooltip": "The unit the mesh was modelled in, so the "
                                                                     "relief comes out in real millimetres."}),
                "up_axis": (["Y", "Z"], {"default": "Y",
                                         "tooltip": "Which axis points up in the mesh file (Y for most OBJ / GLB "
                                                    "exports, Z for some CAD tools)."}),
                "azimuth_deg": ("FLOAT", {"default": 0.0, "min": -360.0,
                                          "max": 360.0, "step": 1.0,
                                          "tooltip": "Direction you look at the mesh from, turning around the up "
                                                     "axis, in degrees. 0 = from the front (+Z for Y up)."}),
                "elevation_deg": ("FLOAT", {"default": 0.0, "min": -89.0,
                                            "max": 89.0, "step": 1.0,
                                            "tooltip": "Tilt of the view, in degrees: above 0 looks down from above, "
                                                       "0 = straight on."}),
                "resolution": ("INT", {"default": 768, "min": 128,
                                       "max": 4096, "step": 64,
                                       "tooltip": "Width of the relief in pixels; the height follows the mesh. "
                                                  "Higher = finer but slower."}),
                "ground_offset_mm": ("FLOAT", {"default": 0.0, "min": -5000.0,
                                               "max": 5000.0, "step": 10.0,
                                               "tooltip": "Added to the mesh's own height, in mm: how high its "
                                                          "lowest point sits above the ground (for the safety "
                                                          "check's reach). 0 = as modelled."}),
                "clamp_relief_mm": ("FLOAT", {"default": 0.0, "min": 0.0,
                                              "max": 5000.0, "step": 10.0,
                                              "tooltip": "Cut the relief off at this depth, in mm, measured from "
                                                         "the back. 0 = no limit."}),
            }
        }

    DESCRIPTION = ("Turns a finished 3D mesh (OBJ, or GLB / PLY / STL with trimesh) into a relief field seen from "
                   "one direction, so relief safety check, mould prep and mass estimate can work on it. The "
                   "undercut mask shows what a one-piece mould or a single milling setup cannot reach.")
    RETURN_TYPES = (FIELD, "IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("field", "preview", "undercut", "report")
    OUTPUT_TOOLTIPS = ("The mesh as a relief in mm, for the other relief nodes.",
                       "Grey preview of the relief (white = nearest to the viewer).",
                       "White where geometry hides behind other geometry from this direction (an undercut).",
                       "Size, depth, height above ground and the share of undercuts (json).")
    @classmethod
    def IS_CHANGED(cls, mesh_path, **kw):
        return _mesh_stamp(mesh_path)

    FUNCTION = "run"
    CATEGORY = CAT

    def run(self, mesh_path, mesh_units, up_axis, azimuth_deg, elevation_deg,
            resolution, ground_offset_mm, clamp_relief_mm):
        path = mesh_path.strip().strip('"')
        if not os.path.isfile(path):
            raise FileNotFoundError(f"mesh not found: {path}")
        verts, faces = load_mesh(path)
        depth, hit, layers, info = rasterize_depth(
            verts, faces, azimuth_deg, elevation_deg, resolution, up_axis)

        scale = _UNIT_MM[mesh_units]
        px_mm = info["pixel_size_mesh_units"] * scale

        h = np.where(hit, depth, -np.inf)
        back = float(h[hit].min()) if hit.any() else 0.0
        h = np.where(hit, (h - back) * scale, 0.0).astype(np.float32)
        if clamp_relief_mm > 0:
            h = np.minimum(h, float(clamp_relief_mm))

        panel_bottom_mm = info["bbox_min_up_units"] * scale + float(ground_offset_mm)
        undercut = (layers > 1) & hit

        field = {
            "height_mm": h,
            "px_mm": float(px_mm),
            "panel_bottom_mm": float(panel_bottom_mm),
            "meta": {
                "source_mesh": os.path.basename(path),
                "azimuth_deg": float(azimuth_deg),
                "elevation_deg": float(elevation_deg),
                "stage": "from_mesh",
            },
        }

        span = float(h.max() - h.min())
        n = (h - h.min()) / span if span > 1e-6 else np.zeros_like(h)
        prev = torch.from_numpy(np.stack([n] * 3, -1).astype(np.float32))[None, ...]

        rep = dict(info)
        rep.update({
            "px_mm": round(px_mm, 3),
            "panel_width_mm": round(info["view_width_units"] * scale, 1),
            "panel_height_mm": round(info["view_height_units"] * scale, 1),
            "depth_range_mm": round(span, 1),
            "panel_bottom_mm": round(panel_bottom_mm, 1),
            "undercut_pixels": int(undercut.sum()),
            "undercut_pct_of_silhouette": round(
                100.0 * undercut.sum() / max(int(hit.sum()), 1), 2),
            "single_mould_possible": bool(not undercut.any()),
        })
        mask = torch.from_numpy(undercut.astype(np.float32))[None, ...]
        return (field, prev, mask, json.dumps(rep, indent=2))


# --------------------------------------------------------------------------
# node 2: turntable rule check on the assembled block
# --------------------------------------------------------------------------

class MeshTurntableCheckKub:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh_path": ("STRING", {"default": "", "multiline": False,
                                         "tooltip": "Full path of the mesh file. OBJ always works; GLB, GLTF, PLY "
                                                    "and STL need trimesh installed. The node re-runs when the file "
                                                    "changes."}),
                "mesh_units": (["mm", "cm", "m", "inch"], {"default": "m",
                                                          "tooltip": "The unit the mesh was modelled in, so the "
                                                                     "check works in real millimetres."}),
                "up_axis": (["Y", "Z"], {"default": "Y",
                                         "tooltip": "Which axis points up in the mesh file (Y for most OBJ / GLB "
                                                    "exports, Z for some CAD tools)."}),
                "views": ("INT", {"default": 8, "min": 1, "max": 24,
                                  "tooltip": "How many directions around the mesh are checked, evenly spaced "
                                             "(8 = every 45 degrees)."}),
                "start_azimuth_deg": ("FLOAT", {"default": 0.0, "min": -360.0,
                                                "max": 360.0, "step": 1.0,
                                                "tooltip": "Direction of the first view around the up axis, in "
                                                           "degrees. 0 = from the front."}),
                "resolution": ("INT", {"default": 512, "min": 128,
                                       "max": 2048, "step": 64,
                                       "tooltip": "Width of each view in pixels. Higher = finer but slower."}),
                "ground_offset_mm": ("FLOAT", {"default": 0.0, "min": -5000.0,
                                               "max": 5000.0, "step": 10.0,
                                               "tooltip": "Added to the mesh's own height, in mm: how high its "
                                                          "lowest point sits above the ground. 0 = as modelled."}),
                "reach_mm": ("FLOAT", {"default": 2500.0, "min": 500.0,
                                       "max": 6000.0, "step": 50.0,
                                       "tooltip": "Height above the ground, in mm, up to which a foothold counts "
                                                  "as reachable."}),
                "foothold_depth_mm": ("FLOAT", {"default": 25.0, "min": 5.0,
                                                "max": 300.0, "step": 1.0,
                                                "tooltip": "A step outward at least this deep, in mm, counts as a "
                                                           "foothold."}),
                "min_width_mm": ("FLOAT", {"default": 80.0, "min": 10.0,
                                           "max": 1000.0, "step": 5.0,
                                           "tooltip": "A foothold must run at least this wide, in mm, to count."}),
            }
        }

    DESCRIPTION = ("Walks around a finished 3D mesh and checks every side for places a person could climb, sit or "
                   "lie on (the same rules as relief safety check). One overlay per view: red = foothold, "
                   "orange = seat, yellow = lying surface.")
    RETURN_TYPES = ("IMAGE", "STRING", "BOOLEAN")
    RETURN_NAMES = ("overlays", "report", "passes")
    OUTPUT_TOOLTIPS = ("One image per view with the flagged areas coloured.",
                       "Failing views, footholds, the largest undercut share and the result per view (json).",
                       "True when every view passes.")
    @classmethod
    def IS_CHANGED(cls, mesh_path, **kw):
        return _mesh_stamp(mesh_path)

    FUNCTION = "run"
    CATEGORY = CAT

    def run(self, mesh_path, mesh_units, up_axis, views, start_azimuth_deg,
            resolution, ground_offset_mm, reach_mm, foothold_depth_mm,
            min_width_mm):
        path = mesh_path.strip().strip('"')
        if not os.path.isfile(path):
            raise FileNotFoundError(f"mesh not found: {path}")
        verts, faces = load_mesh(path)
        scale = _UNIT_MM[mesh_units]

        frames, per_view = [], []
        worst_h, shape = None, None
        for i in range(int(views)):
            az = float(start_azimuth_deg) + 360.0 * i / float(views)
            depth, hit, layers, info = rasterize_depth(
                verts, faces, az, 0.0, resolution, up_axis)
            px_mm = info["pixel_size_mesh_units"] * scale
            h = np.where(hit, depth, -np.inf)
            back = float(h[hit].min()) if hit.any() else 0.0
            h = np.where(hit, (h - back) * scale, 0.0).astype(np.float32)
            bottom = info["bbox_min_up_units"] * scale + float(ground_offset_mm)

            masks, rep = rc.ledge_analysis(
                h, px_mm, panel_bottom_mm=bottom, reach_mm=reach_mm,
                foothold_depth_mm=foothold_depth_mm, min_width_mm=min_width_mm)

            span = float(h.max() - h.min())
            n = (h - h.min()) / span if span > 1e-6 else np.zeros_like(h)
            img = np.stack([n] * 3, -1).astype(np.float32)
            img[~hit] = 0.06
            for key, col in (("foothold", (0.90, 0.10, 0.10)),
                             ("sitting", (1.00, 0.55, 0.00)),
                             ("lying", (1.00, 0.95, 0.20))):
                m = masks[key] & hit
                if m.any():
                    img[m] = img[m] * 0.25 + np.array(col, np.float32) * 0.75

            undercut = (layers > 1) & hit
            rep["azimuth_deg"] = round(az, 1)
            rep["undercut_pct_of_silhouette"] = round(
                100.0 * undercut.sum() / max(int(hit.sum()), 1), 2)
            per_view.append(rep)
            if shape is None:
                shape = img.shape
            if img.shape != shape:  # views can differ by a row, keep the batch square
                img = img[:shape[0], :shape[1]]
                if img.shape != shape:
                    pad = [(0, shape[0] - img.shape[0]), (0, shape[1] - img.shape[1]), (0, 0)]
                    img = np.pad(img, pad, mode="edge")
            frames.append(img)

        overlays = torch.from_numpy(np.stack(frames, 0))
        summary = {
            "views": int(views),
            "views_failing": [v["azimuth_deg"] for v in per_view if not v["passes"]],
            "total_footholds": int(sum(v["footholds_found"] for v in per_view)),
            "max_undercut_pct": max(v["undercut_pct_of_silhouette"] for v in per_view),
            "passes": all(v["passes"] for v in per_view),
            "per_view": per_view,
        }
        return (overlays, json.dumps(summary, indent=2), bool(summary["passes"]))


NODE_CLASS_MAPPINGS = {
    "MeshOrthoFieldKub": MeshOrthoFieldKub,
    "MeshTurntableCheckKub": MeshTurntableCheckKub,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MeshOrthoFieldKub": "kubakub mesh to relief field",
    "MeshTurntableCheckKub": "kubakub mesh turntable rule check",
}
