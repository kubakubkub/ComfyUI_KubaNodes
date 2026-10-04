"""
relief_forge_Kub.py

ReliefForge in the legacy V1 node schema, so it lives inside an existing
NODE_CLASS_MAPPINGS pack instead of needing its own folder.

Requires kubakub/relief_core.py. Only numpy beyond ComfyUI.
"""

from __future__ import annotations

import json
import os

import numpy as np
import torch

import folder_paths

from ...kubakub import relief_core as rc

CAT = "kubakub/3d/fabricate/relief"
FIELD = "RELIEFFORGE_FIELD"


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _img_to_np(image):
    a = image[0].detach().cpu().numpy()
    if a.ndim == 3:
        a = a[..., :3].mean(axis=2)          # the colour channels only: an alpha channel is not depth
    return a.astype(np.float32)


def _np_to_img(a):
    a = np.asarray(a, dtype=np.float32)
    if a.ndim == 2:
        a = np.stack([a] * 3, axis=-1)
    return torch.from_numpy(a)[None, ...]


def _preview(field):
    h = field["height_mm"]
    span = float(h.max() - h.min())
    n = (h - h.min()) / span if span > 1e-6 else np.zeros_like(h)
    return _np_to_img(n)


def _make_field(height_mm, px_mm, panel_bottom_mm=0.0, meta=None):
    return {
        "height_mm": np.asarray(height_mm, dtype=np.float32),
        "px_mm": float(px_mm),
        "panel_bottom_mm": float(panel_bottom_mm),
        "meta": dict(meta or {}),
    }


def _stamp(field, **kw):
    out = dict(field)
    out["meta"] = dict(field.get("meta", {}))
    out["meta"].update(kw)
    return out


# --------------------------------------------------------------------------
# 1. depth map to physical relief
# --------------------------------------------------------------------------

class ReliefFieldKub:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "depth": ("IMAGE", {"tooltip": "Depth map of the relief (bright = sticks out, unless invert is on). "
                                               "Only the first image of a batch is used."}),
                "panel_width_mm": ("FLOAT", {"default": 2400.0, "min": 100.0,
                                             "max": 20000.0, "step": 10.0,
                                             "tooltip": "Real width of the panel in mm. The depth map is stretched to "
                                                        "it; the panel height follows from the image aspect."}),
                "max_relief_mm": ("FLOAT", {"default": 120.0, "min": 1.0,
                                            "max": 1000.0, "step": 1.0,
                                            "tooltip": "How far the brightest point sticks out from the back, in mm."}),
                "panel_bottom_mm": ("FLOAT", {"default": 300.0, "min": 0.0,
                                              "max": 10000.0, "step": 10.0,
                                              "tooltip": "Height of the panel's lower edge above the ground, in mm. "
                                                         "The safety check uses it to know what a person can reach."}),
                "invert": ("BOOLEAN", {"default": False,
                                       "tooltip": "Turn the depth around: on when your depth map has near = dark."}),
                "gamma": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 4.0,
                                    "step": 0.05,
                                    "tooltip": "Shapes the depth curve: below 1 lifts the mid tones (more shallow "
                                               "detail), above 1 pushes them back. 1 = linear."}),
                "flatten_percentile": ("FLOAT", {"default": 0.5, "min": 0.0,
                                                 "max": 10.0, "step": 0.1,
                                                 "tooltip": "Percent of the darkest and brightest pixels clipped "
                                                            "before scaling, so single stray pixels from a depth "
                                                            "model do not set the range. 0 = no clipping."}),
            }
        }

    DESCRIPTION = ("Turns a depth map into a physical relief panel in millimetres, the start of every relief chain. "
                   "Set the real panel width and the deepest relief; the other relief nodes then shape, check, "
                   "weigh and export it.")
    RETURN_TYPES = (FIELD, "IMAGE")
    RETURN_NAMES = ("field", "preview")
    OUTPUT_TOOLTIPS = ("The relief as heights in mm with its real size, for the other relief nodes.",
                       "Grey preview of the relief (white = sticks out furthest).")
    FUNCTION = "run"
    CATEGORY = CAT

    def run(self, depth, panel_width_mm, max_relief_mm, panel_bottom_mm,
            invert, gamma, flatten_percentile):
        d = _img_to_np(depth)
        px_mm = float(panel_width_mm) / d.shape[1]
        h = rc.depth_to_height(d, max_relief_mm, invert, gamma, flatten_percentile)
        field = _make_field(h, px_mm, panel_bottom_mm, {
            "panel_width_mm": float(panel_width_mm),
            "panel_height_mm": round(d.shape[0] * px_mm, 1),
            "stage": "raw",
        })
        return (field, _preview(field))


# --------------------------------------------------------------------------
# 2. mould prep
# --------------------------------------------------------------------------

class ReliefMouldKub:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "field": (FIELD, {"tooltip": "The relief from relief field (or another relief node)."}),
                "min_edge_radius_mm": ("FLOAT", {"default": 20.0, "min": 0.0,
                                                 "max": 200.0, "step": 1.0,
                                                 "tooltip": "Smallest rounding on every edge and corner, in mm (no "
                                                            "sharp edges, and a ball-nose cutter reaches everything). "
                                                            "0 = no rounding."}),
                "draft_angle_deg": ("FLOAT", {"default": 4.0, "min": 0.0,
                                              "max": 45.0, "step": 0.5,
                                              "tooltip": "Minimum draft angle in degrees, measured from the pull "
                                                         "direction, so the cast releases from a one-piece mould. "
                                                         "3 is the usual minimum, 5 is comfortable."}),
                "max_work_px": ("INT", {"default": 1024, "min": 128,
                                        "max": 4096, "step": 64,
                                        "tooltip": "The relief is worked on at most this many pixels on its long "
                                                   "side, then scaled back. Higher = finer but slower."}),
            }
        }

    DESCRIPTION = ("Makes a relief castable: rounds every sharp edge and limits steep walls so the panel comes out "
                   "of a one-piece mould. Some depth is lost; the report says how much.")
    RETURN_TYPES = (FIELD, "IMAGE", "STRING")
    RETURN_NAMES = ("field", "preview", "report")
    OUTPUT_TOOLTIPS = ("The mould-ready relief.",
                       "Grey preview of the mould-ready relief.",
                       "Relief depth before and after, the share of detail lost and the settings used (json).")
    FUNCTION = "run"
    CATEGORY = CAT

    def run(self, field, min_edge_radius_mm, draft_angle_deg, max_work_px):
        h, px = field["height_mm"], field["px_mm"]
        before = float(h.max() - h.min())
        h2 = rc.round_edges(h, px, min_edge_radius_mm, max_work_px)
        h2 = rc.limit_slope(h2, px, draft_angle_deg, max_work_px)
        after = float(h2.max() - h2.min())
        out = _stamp(_make_field(h2, px, field["panel_bottom_mm"], field["meta"]),
                     stage="mould_ready",
                     min_edge_radius_mm=float(min_edge_radius_mm),
                     draft_angle_deg=float(draft_angle_deg))
        rep = json.dumps({
            "relief_before_mm": round(before, 1),
            "relief_after_mm": round(after, 1),
            "detail_lost_pct": round(100.0 * (1.0 - after / before), 1) if before else 0.0,
            "min_edge_radius_mm": float(min_edge_radius_mm),
            "draft_angle_deg": float(draft_angle_deg),
        }, indent=2)
        return (out, _preview(out), rep)


# --------------------------------------------------------------------------
# 3. anti perch
# --------------------------------------------------------------------------

class ReliefAntiPerchKub:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "field": (FIELD, {"tooltip": "The relief to treat, usually after relief mould prep."}),
                "sill_angle_deg": ("FLOAT", {"default": 45.0, "min": 10.0,
                                             "max": 80.0, "step": 1.0,
                                             "tooltip": "Slope of the new sills, in degrees from horizontal. 45 is "
                                                        "the common anti-perch value; 60 is aggressive and eats more "
                                                        "of the relief."}),
                "max_span_mm": ("FLOAT", {"default": 400.0, "min": 20.0,
                                          "max": 2000.0, "step": 10.0,
                                          "tooltip": "How far down, in mm, a sill slope can reach below a ledge. "
                                                     "Larger = deep ledges get fully sloped, but slower."}),
            }
        }

    DESCRIPTION = ("Shaves every flat ledge of a relief into a sloped sill so birds and people cannot perch, stand "
                   "or sit on it. Vertical faces and undersides stay as they are.")
    RETURN_TYPES = (FIELD, "IMAGE")
    RETURN_NAMES = ("field", "preview")
    OUTPUT_TOOLTIPS = ("The relief with sloped sills.",
                       "Grey preview of the treated relief.")
    FUNCTION = "run"
    CATEGORY = CAT

    def run(self, field, sill_angle_deg, max_span_mm):
        h2 = rc.deslope_ledges(field["height_mm"], field["px_mm"],
                               sill_angle_deg, max_span_mm)
        out = _stamp(_make_field(h2, field["px_mm"], field["panel_bottom_mm"],
                                 field["meta"]),
                     sill_angle_deg=float(sill_angle_deg))
        return (out, _preview(out))


# --------------------------------------------------------------------------
# 4. safety check
# --------------------------------------------------------------------------

class ReliefSafetyCheckKub:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "field": (FIELD, {"tooltip": "The relief to check (its panel_bottom_mm sets the height above ground)."}),
                "reach_mm": ("FLOAT", {"default": 2500.0, "min": 500.0,
                                       "max": 6000.0, "step": 50.0,
                                       "tooltip": "Height above the ground, in mm, up to which a foothold counts "
                                                  "as reachable. Footholds higher up are ignored."}),
                "foothold_depth_mm": ("FLOAT", {"default": 25.0, "min": 5.0,
                                                "max": 300.0, "step": 1.0,
                                                "tooltip": "A step outward at least this deep, in mm, counts as a "
                                                           "foothold."}),
                "min_width_mm": ("FLOAT", {"default": 80.0, "min": 10.0,
                                           "max": 1000.0, "step": 5.0,
                                           "tooltip": "A foothold must run at least this wide, in mm, to count."}),
                "clearance_mm": ("FLOAT", {"default": 150.0, "min": 20.0,
                                           "max": 1000.0, "step": 10.0,
                                           "tooltip": "Band above a step, in mm, that it is measured against: the "
                                                      "step must stick out beyond everything in this band."}),
            },
            "optional": {"over_image": ("IMAGE", {"tooltip": "Optional picture to draw the flags on (same size as "
                                                             "the relief). Without it the grey relief is used."})},
        }

    DESCRIPTION = ("Checks a relief for places a person could climb, sit or lie on, and marks them: red = foothold "
                   "within reach, orange = seat height, yellow = deep enough to lie on. Fix what it flags with "
                   "relief anti perch or a new depth map.")
    RETURN_TYPES = ("IMAGE", "MASK", "STRING", "BOOLEAN")
    RETURN_NAMES = ("overlay", "flagged", "report", "passes")
    OUTPUT_TOOLTIPS = ("The relief (or over_image) with the flagged areas coloured.",
                       "White where anything was flagged.",
                       "Counts, areas and which rules pass (json).",
                       "True when no foothold, seat or lying surface was found.")
    FUNCTION = "run"
    CATEGORY = CAT

    def run(self, field, reach_mm, foothold_depth_mm, min_width_mm,
            clearance_mm, over_image=None):
        h = field["height_mm"]
        masks, report = rc.ledge_analysis(
            h, field["px_mm"],
            panel_bottom_mm=field["panel_bottom_mm"],
            reach_mm=reach_mm,
            foothold_depth_mm=foothold_depth_mm,
            min_width_mm=min_width_mm,
            clearance_mm=clearance_mm,
        )

        base = None
        if over_image is not None:
            b = over_image[0].detach().cpu().numpy().astype(np.float32)
            if b.ndim == 2:
                b = np.stack([b] * 3, -1)
            if b.shape[:2] == h.shape:
                base = b[..., :3].copy()
        if base is None:
            base = _preview(field)[0].numpy().copy()

        for key, col in (("foothold", (0.90, 0.10, 0.10)),
                         ("sitting", (1.00, 0.55, 0.00)),
                         ("lying", (1.00, 0.95, 0.20))):
            m = masks[key]
            if m.any():
                base[m] = base[m] * 0.25 + np.array(col, np.float32) * 0.75

        flag = masks["foothold"] | masks["sitting"] | masks["lying"]
        report["panel_bottom_mm"] = field["panel_bottom_mm"]
        report["px_mm"] = round(field["px_mm"], 3)
        overlay = torch.from_numpy(np.clip(base, 0, 1))[None, ...]
        mask = torch.from_numpy(flag.astype(np.float32))[None, ...]
        return (overlay, mask, json.dumps(report, indent=2), bool(report["passes"]))


# --------------------------------------------------------------------------
# 5. mass
# --------------------------------------------------------------------------

class ReliefMassKub:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "field": (FIELD, {"tooltip": "The relief to weigh (one panel)."}),
                "shell_thickness_mm": ("FLOAT", {"default": 15.0, "min": 2.0,
                                                 "max": 200.0, "step": 0.5,
                                                 "tooltip": "Thickness of the cast shell, in mm. The shell follows the "
                                                            "relief, so its true surface area is used."}),
                "density_kg_m3": ("FLOAT", {"default": 2100.0, "min": 100.0,
                                            "max": 8000.0, "step": 10.0,
                                            "tooltip": "Density of the shell material in kg/m3 (2100 = glass fibre "
                                                       "reinforced concrete)."}),
                "frame_kg_per_m2": ("FLOAT", {"default": 18.0, "min": 0.0,
                                              "max": 200.0, "step": 1.0,
                                              "tooltip": "Weight of the backing frame per square metre of flat panel, "
                                                         "in kg. 0 = shell only."}),
                "footprint_m2": ("FLOAT", {"default": 2.0, "min": 0.05,
                                           "max": 100.0, "step": 0.05,
                                           "tooltip": "Ground area the whole piece stands on, in m2, for the load "
                                                      "per square metre."}),
                "panel_count": ("INT", {"default": 2, "min": 1, "max": 20,
                                        "tooltip": "How many panels like this one the piece has."}),
            }
        }

    DESCRIPTION = ("Estimates how heavy a relief piece will be: shell plus frame, times the number of panels, and "
                   "the load per square metre of footprint. Ballast for wind is not included.")
    RETURN_TYPES = ("STRING", "FLOAT", "FLOAT")
    RETURN_NAMES = ("report", "total_kg", "kg_per_m2")
    OUTPUT_TOOLTIPS = ("Areas, shell and frame weight per panel and the totals (json).",
                       "Weight of all panels together, in kg.",
                       "Total weight divided by the footprint, in kg per m2.")
    FUNCTION = "run"
    CATEGORY = CAT

    def run(self, field, shell_thickness_mm, density_kg_m3, frame_kg_per_m2,
            footprint_m2, panel_count):
        r = rc.mass_estimate(field["height_mm"], field["px_mm"],
                             shell_thickness_mm, density_kg_m3,
                             frame_kg_per_m2, footprint_m2)
        total = r["panel_total_kg"] * int(panel_count)
        per_m2 = total / float(footprint_m2)
        r["panel_count"] = int(panel_count)
        r["all_panels_kg"] = round(total, 1)
        r["kg_per_m2_of_footprint"] = round(per_m2, 1)
        r["note"] = "Ballast for wind loading is not included."
        return (json.dumps(r, indent=2), float(total), float(per_m2))


# --------------------------------------------------------------------------
# 6. export
# --------------------------------------------------------------------------

class ReliefExportPanelKub:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "field": (FIELD, {"tooltip": "The finished relief to export."}),
                "filename_prefix": ("STRING", {"default": "reliefforge/panel",
                                               "tooltip": "File name in the ComfyUI output folder; a part before / "
                                                          "makes a subfolder. A counter is added."}),
                "base_mm": ("FLOAT", {"default": 40.0, "min": 0.0,
                                      "max": 500.0, "step": 1.0,
                                      "tooltip": "Solid back plate under the relief, in mm (0 = the relief starts "
                                                 "right at the flat back)."}),
                "decimate": ("INT", {"default": 2, "min": 1, "max": 16,
                                     "tooltip": "Use every n-th pixel for the mesh. 1 = full detail, higher = "
                                                "smaller, lighter file."}),
            }
        }

    DESCRIPTION = ("Saves a relief as a closed STL solid in millimetres for printing or milling (relief on top, "
                   "flat back, side walls).")
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("path",)
    OUTPUT_TOOLTIPS = ("Full path of the saved .stl file.",)
    FUNCTION = "run"
    OUTPUT_NODE = True
    CATEGORY = CAT

    def run(self, field, filename_prefix, base_mm, decimate):
        out_dir = folder_paths.get_output_directory()
        full, name, counter, subfolder, _ = folder_paths.get_save_image_path(
            filename_prefix, out_dir)
        path = os.path.join(full, f"{name}_{counter:05}.stl")
        v, f = rc.grid_mesh(field["height_mm"], field["px_mm"], base_mm, decimate)
        rc.write_binary_stl(path, v, f)
        return (path,)


class ReliefExportProwKub:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "field_a": (FIELD, {"tooltip": "The relief for the first wing of the prow."}),
                "interior_angle_deg": ("FLOAT", {"default": 28.0, "min": 5.0,
                                                 "max": 170.0, "step": 0.5,
                                                 "tooltip": "Full angle between the two wings, in degrees (small = "
                                                            "sharp prow)."}),
                "filename_prefix": ("STRING", {"default": "reliefforge/prow",
                                               "tooltip": "File name in the ComfyUI output folder; a part before / "
                                                          "makes a subfolder. A counter is added."}),
                "base_mm": ("FLOAT", {"default": 40.0, "min": 0.0,
                                      "max": 500.0, "step": 1.0,
                                      "tooltip": "Solid back plate under each wing's relief, in mm."}),
                "decimate": ("INT", {"default": 3, "min": 1, "max": 16,
                                     "tooltip": "Use every n-th pixel for the mesh. 1 = full detail, higher = "
                                                "smaller, lighter file."}),
            },
            "optional": {"field_b": (FIELD, {"tooltip": "Optional relief for the second wing. Without it, field_a "
                                                         "is mirrored onto both wings."})},
        }

    DESCRIPTION = ("Folds one or two reliefs into a prow (two wings meeting at an edge) and saves it as one OBJ "
                   "in millimetres.")
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("path",)
    OUTPUT_TOOLTIPS = ("Full path of the saved .obj file.",)
    FUNCTION = "run"
    OUTPUT_NODE = True
    CATEGORY = CAT

    def run(self, field_a, interior_angle_deg, filename_prefix, base_mm,
            decimate, field_b=None):
        out_dir = folder_paths.get_output_directory()
        full, name, counter, subfolder, _ = folder_paths.get_save_image_path(
            filename_prefix, out_dir)
        path = os.path.join(full, f"{name}_{counter:05}.obj")
        panels = [field_a["height_mm"]]
        if field_b is not None:
            panels.append(field_b["height_mm"])
        rc.write_prow_obj(path, panels, field_a["px_mm"], interior_angle_deg,
                          base_mm, decimate)
        return (path,)


NODE_CLASS_MAPPINGS = {
    "ReliefFieldKub": ReliefFieldKub,
    "ReliefMouldKub": ReliefMouldKub,
    "ReliefAntiPerchKub": ReliefAntiPerchKub,
    "ReliefSafetyCheckKub": ReliefSafetyCheckKub,
    "ReliefMassKub": ReliefMassKub,
    "ReliefExportPanelKub": ReliefExportPanelKub,
    "ReliefExportProwKub": ReliefExportProwKub,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ReliefFieldKub": "kubakub relief field (from depth)",
    "ReliefMouldKub": "kubakub relief mould prep",
    "ReliefAntiPerchKub": "kubakub relief anti perch",
    "ReliefSafetyCheckKub": "kubakub relief safety check",
    "ReliefMassKub": "kubakub relief mass estimate",
    "ReliefExportPanelKub": "kubakub relief export panel (stl)",
    "ReliefExportProwKub": "kubakub relief export prow (obj)",
}
