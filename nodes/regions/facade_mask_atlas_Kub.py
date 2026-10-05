"""
facade_mask_atlas_Kub.py

KUBA_FacadeMaskAtlas: splits a facade matrix into region masks, groups repeated
elements (window rows, piers) and writes the region table as JSON for
KUBA_FacadeRegionPlan. The array work lives in kubakub/facade_core.py.

V3 node schema (comfy_api.latest), registered through the pack's
NODE_CLASS_MAPPINGS loader in __init__.py.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time

import torch

from comfy_api.latest import io, ui

from ...kubakub import facade_core as fc
from ...kubakub.io_types import RegionsType
from ...kubakub.types import Regions

log = logging.getLogger("KUBA.facade")

CATEGORY = "kubakub/2d/regions"


class KUBA_FacadeMaskAtlas(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="KUBA_FacadeMaskAtlas",
            display_name="kubakub regions from matrix / masks",
            category="kubakub/2d/regions",
            search_aliases=['facade', 'projection mapping', 'mask atlas', 'after effects masks'],
            description=(
                "Splits a facade matrix (the full projection image at delivery size) into regions, so "
                "each window, pier or wall part can be treated on its own. color_regions: one region per flat "
                "colour (per connected patch when split_disconnected is on). line_drawing: the "
                "closed cells between lines. mask_folder: one region per PNG, named after the "
                "file (subfolders too). Scope masks limit where regions may be, split masks give one "
                "region per separate shape. Repeated elements get a shared group_id; every region is "
                "tagged with the larger masks that cover it (floor, facade ...)."),
            inputs=[
                io.Image.Input("matrix", tooltip="The facade matrix: the full projection image at delivery "
                                                 "size (e.g. a 3840x2160 PNG). Only the first image of a batch "
                                                 "is used."),
                io.Combo.Input("mode", options=list(fc.MODES), default="color_regions",
                               tooltip="Where the regions come from. color_regions: flat colour fills in the "
                                       "matrix. line_drawing: closed cells of a line drawing. mask_folder: a "
                                       "folder of PNG masks (e.g. exported from After Effects)."),
                io.Int.Input("min_region_area", default=400, min=0, max=100_000_000, step=10,
                             tooltip="Regions smaller than this (in pixels) are merged or dropped."),
                io.Boolean.Input("merge_small_regions", default=True,
                                 tooltip="On: merge small regions into the neighbour with the longest "
                                         "shared border. Off: drop them (they stay unassigned)."),
                io.Int.Input("color_tolerance", advanced=True, default=24, min=0, max=442, step=1,
                             tooltip="color_regions: colours closer than this (RGB distance, 0-255 "
                                     "units) count as the same fill."),
                io.Boolean.Input("split_disconnected", advanced=True, default=True,
                                 tooltip="color_regions: on = every separate patch of a colour is its own "
                                         "region (each window), off = one region per colour."),
                io.Float.Input("group_tolerance", advanced=True, default=0.08, min=0.0, max=0.5, step=0.01,
                               tooltip="How different two regions may be (relative bbox size, 1 - shape "
                                       "IoU) and still share a group_id. 0 = identical only."),
                io.String.Input("color_names", advanced=True, multiline=True, default="", optional=True,
                                placeholder="#2b3a55 = windows\n200,180,160 = wall",
                                tooltip="color_regions: name the groups by matrix colour. Every region "
                                        "of that colour gets the name as group_id."),
                io.String.Input("mask_folder", default="", optional=True,
                                placeholder="paste your mask folder  (empty = the sample facade's masks)",
                                tooltip="mask_folder: folder of PNG masks (alpha or white = inside). "
                                        "'windows_03.png' becomes region 'windows_03' in group 'windows'. Empty: the "
                                        "sample facade's masks at the matrix size, so the node runs as it is."),
                io.Boolean.Input("recursive", default=True,
                                 tooltip="mask_folder: also read PNGs in subfolders (Groups/, Windows/ ...)."),
                io.String.Input("scope_masks", advanced=True, default="", optional=True,
                                placeholder="M_Projection_Range",
                                tooltip="mask_folder: masks (names or wildcards, comma or one per line) "
                                        "that are not regions but the area regions may use. Nothing "
                                        "outside their union is ever a region."),
                io.String.Input("split_masks", advanced=True, default="", optional=True,
                                placeholder="M_Pilasters, M_Stone_Frames, M_Lintels",
                                tooltip="mask_folder: masks holding several separate shapes. Each shape "
                                        "becomes a region <name>_01, _02 ... in reading order, all in "
                                        "group <name>."),
                io.String.Input("tag_only_masks", advanced=True, default="", optional=True,
                                placeholder="M_FLOOR_*",
                                tooltip="mask_folder: masks that only tag the regions they cover and are "
                                        "no region themselves."),
                io.Combo.Input("group_by", options=list(fc.GROUP_BY), default="stem",
                               tooltip="mask_folder: group_id from the file name without its trailing "
                                       "number (stem) or from the subfolder name (folder)."),
                io.Mask.Input("scope", optional=True,
                              tooltip="Any mode: pixels outside this mask are never part of a region."),
                io.Boolean.Input("output_masks", default=True, advanced=True,
                                 tooltip="Off: the masks output is a single empty mask, saving RAM when "
                                         "only the regions output is used."),
                io.Float.Input("line_threshold", default=0.5, min=0.0, max=1.0, step=0.01, advanced=True,
                               tooltip="line_drawing: brightness that separates lines from paper."),
                io.Combo.Input("line_polarity", options=list(fc.LINE_POLARITIES), default="auto",
                               advanced=True,
                               tooltip="line_drawing: auto takes whichever of dark/light is the minority "
                                       "as the lines."),
                io.Int.Input("line_gap_close_px", default=1, min=0, max=32, step=1, advanced=True,
                             tooltip="line_drawing: thicken lines by this much first, so small gaps "
                                     "don't leak one cell into the next."),
            ],
            outputs=[
                io.Mask.Output("masks", tooltip="One mask per region; batch index = region_id."),
                io.String.Output("regions_json", tooltip="The region table as text: ids, groups, tags, boxes "
                                                         "and notes."),
                io.Image.Output("preview", tooltip="The matrix in grey with one colour per group, region "
                                                   "borders and ids."),
                io.Mask.Output("scope", tooltip="1 where regions may be (all 1 without a scope)."),
                RegionsType.Output("regions", tooltip="Label map + region table for the kubakub region nodes."),
            ],
        )

    @classmethod
    def fingerprint_inputs(cls, mode=None, mask_folder="", recursive=True, **kwargs):
        # The PNGs in the folder can change without any input changing.
        if mode != "mask_folder":
            return ""
        folder, files = fc.list_mask_files(mask_folder, recursive)
        h = hashlib.sha256(folder.encode("utf-8", "replace"))
        for f in files:
            try:
                st = os.stat(os.path.join(folder, f))
                h.update(f"{f}|{st.st_size}|{st.st_mtime_ns}".encode("utf-8", "replace"))
            except OSError:
                pass
        return h.hexdigest()

    @classmethod
    def execute(cls, matrix, mode, min_region_area, merge_small_regions, color_tolerance,
                split_disconnected, group_tolerance, color_names="", mask_folder="",
                recursive=True, scope_masks="", split_masks="", tag_only_masks="", group_by="stem",
                scope=None, output_masks=True,
                line_threshold=0.5, line_polarity="auto", line_gap_close_px=1) -> io.NodeOutput:
        t0 = time.perf_counter()
        if matrix.shape[0] > 1:
            log.warning("[KUBA facade] matrix batch has %d images; using the first.", matrix.shape[0])
        image = matrix[0].detach().cpu().float().numpy()
        scope_np = None
        if scope is not None:
            sc = scope[0] if scope.ndim == 3 else scope
            if tuple(sc.shape) != tuple(image.shape[:2]):
                raise ValueError(f"scope mask is {sc.shape[1]}x{sc.shape[0]}, the matrix "
                                 f"{image.shape[1]}x{image.shape[0]}; they must match exactly.")
            scope_np = sc.detach().cpu().numpy() > 0.5

        sample = mode == "mask_folder" and not fc.clean_folder_path(mask_folder or "")
        if sample:                                     # nothing pasted or linked: the sample facade's masks at this size
            import folder_paths
            from ...kubakub import samples
            mask_folder, recursive = samples.mask_folder(folder_paths.get_temp_directory(), image.shape[1], image.shape[0]), True
        labels, atlas, scope_used = fc.build_atlas(
            image, mode=mode, min_region_area=min_region_area,
            merge_small_regions=merge_small_regions, color_tolerance=color_tolerance,
            split_disconnected=split_disconnected, group_tolerance=group_tolerance,
            line_threshold=line_threshold, line_polarity=line_polarity,
            line_gap_close_px=line_gap_close_px, mask_folder=mask_folder or "",
            color_names=color_names or "", recursive=recursive, scope_masks=scope_masks or "",
            tag_only_masks=tag_only_masks or "", split_masks=split_masks or "", group_by=group_by,
            scope=scope_np, with_scope=True)

        if sample:
            atlas["notes"].append("no mask folder given: the sample facade's masks are used. Paste the path of your own "
                                  "mask folder into 'mask_folder'.")
        n = len(atlas["regions"])
        h, w = labels.shape
        if n == 0:
            raise ValueError("kubakub regions from matrix / masks found no regions. Check the mode and "
                             "min_region_area. Notes: " + "; ".join(atlas["notes"]))

        gb = n * h * w * 4 / 1024 ** 3
        if output_masks and gb > 2.0:
            log.warning("[KUBA facade] %d masks at %dx%d take %.1f GB of RAM. Raise "
                        "min_region_area or turn split_disconnected off to reduce.", n, w, h, gb)
        regions = Regions.from_numpy(labels, atlas, scope_used)
        masks = regions.masks() if output_masks else torch.zeros((1, h, w), dtype=torch.float32)
        scope_out = regions.scope if regions.scope is not None else torch.ones((1, h, w))

        preview = torch.from_numpy(fc.render_preview(image, labels, atlas))[None]

        log.info("[KUBA facade] atlas: mode=%s, %d regions in %d groups, %d px unassigned, %.2fs",
                 mode, n, len(atlas["groups"]), atlas["unassigned_px"], time.perf_counter() - t0)
        for note in atlas["notes"]:
            log.info("[KUBA facade]   %s", note)

        return io.NodeOutput(masks, json.dumps(atlas, indent=1), preview, scope_out, regions,
                             ui=ui.PreviewImage(preview, cls=cls))


NODE_CLASS_MAPPINGS = {
    "KUBA_FacadeMaskAtlas": KUBA_FacadeMaskAtlas,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "KUBA_FacadeMaskAtlas": "kubakub regions from matrix / masks",
}
