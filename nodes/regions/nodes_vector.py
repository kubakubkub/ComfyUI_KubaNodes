"""
nodes_vector.py

kubakub regions to vector (category KUBAKUB/regions): regions or a mask
as SVG / PDF / DXF, as filled outlines (Illustrator, After Effects shape layers)
or as centerlines (plotter, laser, engraving), in pixels or millimetres.
Logic in vector.py. V3 node schema, registered via ../__init__.py.
"""

from __future__ import annotations

import fnmatch
import logging
import os

import numpy as np
import torch

from comfy_api.latest import io, ui

from ...kubakub import vector as vec
from ...kubakub.io_types import RegionsType

log = logging.getLogger("KUBA.regions")

CATEGORY = "kubakub/2d/regions"
MODES = ("outline", "centerline")


def select_ids(table: dict, text: str):
    """'W_F1_*, group:Windows' (comma or line separated, name wildcards or group:/tag:) -> region ids; empty = all."""
    pats = [p.strip().lower() for p in (text or "").replace("\n", ",").split(",") if p.strip()]
    ids = []
    for r in table.get("regions", []):
        name, group = str(r.get("name", "")).lower(), str(r.get("group_id", "")).lower()
        tags = [str(t).lower() for t in r.get("tags", [])]
        if not pats:
            ids.append(int(r["region_id"]))
            continue
        for p in pats:
            if p.startswith("group:") and fnmatch.fnmatchcase(group, p[6:]) or \
               p.startswith("tag:") and any(fnmatch.fnmatchcase(t, p[4:]) for t in tags) or \
               not p.startswith(("group:", "tag:")) and fnmatch.fnmatchcase(name, p):
                ids.append(int(r["region_id"]))
                break
    return ids


class KUBA_RegionsToVector(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="KUBA_RegionsToVector",
            display_name="kubakub regions to svg / pdf / dxf",
            category="kubakub/2d/regions",
            search_aliases=['vector', 'plotter', 'laser', 'cnc'],
            is_output_node=True,        # the written files are the product: runs with its outputs unconnected
            description=(
                "Save regions (or a mask) as vector files for Illustrator, After Effects, a plotter or a laser cutter. outline: filled paths per region, straight edges stay "
                "straight, curves become Beziers; SVG with groups = region groups and ids = region names, PDF "
                "with one layer per group, DXF. centerline: the skeleton as open lines ordered for short travel "
                "(plotter, laser, engraving). width_mm > 0 writes real millimetres; kerf_mm offsets outlines "
                "for cutting."),
            inputs=[
                RegionsType.Input("regions", optional=True,
                                  tooltip="The regions to vectorise (from any kubakub regions node)."),
                io.Mask.Input("mask", optional=True, tooltip="Vectorise a mask instead (e.g. a painted line layer)."),
                io.String.Input("select", default="", optional=True, placeholder="W_F1_*, group:Windows",
                                tooltip="Which regions (name wildcards, group:, tag:); empty = all."),
                io.Combo.Input("mode", options=list(MODES), default="outline",
                               tooltip="outline: filled shapes per region (Illustrator, After Effects). "
                                       "centerline: single lines through the middle (plotter, laser, engraving)."),
                io.Float.Input("width_mm", default=0.0, min=0.0, max=1_000_000.0, step=0.1,
                               tooltip="Real width of the whole image in mm (one uniform scale); 0 = pixels."),
                io.Float.Input("simplify_px", default=1.0, min=0.0, max=50.0, step=0.1,
                               tooltip="Largest deviation from the pixel outline when simplifying."),
                io.Float.Input("corner_deg", default=35.0, min=0.0, max=180.0, step=1.0,
                               tooltip="outline: vertices turning more than this stay corners; 180 = no curves."),
                io.Float.Input("min_area_px", default=16.0, min=0.0, max=1_000_000.0, step=1.0,
                               tooltip="Drop specks and holes smaller than this."),
                io.Float.Input("spur_px", default=6.0, min=0.0, max=1000.0, step=0.5,
                               tooltip="centerline: drop side branches shorter than this."),
                io.Float.Input("kerf_mm", default=0.0, min=-100.0, max=100.0, step=0.01,
                               tooltip="outline in mm: grow (+) or shrink (-) every outline by kerf / 2."),
                io.Boolean.Input("svg", default=True, tooltip="Write an .svg (groups = region groups, ids = region names)."),
                io.Boolean.Input("pdf", default=True, tooltip="Write a .pdf with one layer per region group."),
                io.Boolean.Input("dxf", default=True, tooltip="Write a .dxf for CAD, CNC and laser software."),
                io.String.Input("folder", default="", optional=True,
                                tooltip="Where to write; empty = ComfyUI output/kuba_vector."),
                io.String.Input("filename_prefix", default="regions",
                                tooltip="Start of the file names (a number is added, earlier files are kept); "
                                        "'sub/name' writes into a subfolder."),
                io.Image.Input("matrix", optional=True, tooltip="Preview background."),
            ],
            outputs=[
                io.Image.Output("preview", tooltip="The vector paths drawn over the matrix."),
                io.String.Output("files", tooltip="The written files, one per line."),
                io.String.Output("report", tooltip="Regions, paths, size in px or mm, line length for "
                                                   "centerlines, and the written files."),
            ],
        )

    @classmethod
    def execute(cls, mode, width_mm, simplify_px, corner_deg, min_area_px, spur_px, kerf_mm, svg, pdf, dxf,
                filename_prefix, regions=None, mask=None, select="", folder="", matrix=None) -> io.NodeOutput:
        if regions is None and mask is None:            # left in a graph unconnected: nothing to write, no error
            empty = torch.zeros((1, 64, 64, 3))
            return io.NodeOutput(empty, "", "nothing to vectorise: connect regions or a mask")
        if mask is not None:
            m = (mask[0] if mask.ndim == 3 else mask).cpu().float().numpy() > 0.5
            labels = np.where(m, 0, -1).astype(np.int32)
            table = {"regions": [{"region_id": 0, "name": filename_prefix, "group_id": filename_prefix}]}
            ids = [0]
        else:
            labels = regions.labels[0].cpu().numpy()
            table = regions.table
            ids = select_ids(table, select)
            if not ids:
                raise ValueError(f"select '{select}' matches no region")
        formats = [f for f, on in (("svg", svg), ("pdf", pdf), ("dxf", dxf)) if on]
        if not formats:
            raise ValueError("switch on at least one of svg / pdf / dxf")
        doc = vec.build(labels, table, ids, mode=mode, simplify_px=simplify_px, corner_deg=corner_deg,
                        min_area_px=min_area_px, spur_px=spur_px, width_mm=width_mm, kerf_mm=kerf_mm)
        if not folder:
            import folder_paths
            folder = os.path.join(folder_paths.get_output_directory(), "kuba_vector")
        files = vec.write_all(doc, os.path.expandvars(folder.strip().strip('"')), filename_prefix, formats)
        H, W = labels.shape
        bg = matrix[0, ..., :3].cpu().float().numpy() if matrix is not None else None
        if bg is not None and bg.shape[:2] != (H, W):
            import cv2
            bg = cv2.resize(bg, (W, H))
        prev = torch.from_numpy(vec.render_preview(doc, H, W, bg))[None]
        st = doc["stats"]
        u = doc["unit"]
        report = (f"{mode}: {len(ids)} regions in {len(doc['layers'])} layers, {st['paths']} paths, {st['nodes']} "
                  f"nodes; {doc['width']:.1f} x {doc['height']:.1f} {u}"
                  + (f"; line length {st['length']:.1f} {u}, travel {st['travel']:.1f} {u}" if mode == "centerline"
                     else "") + (f"; kerf {kerf_mm:g} mm" if kerf_mm and u == "mm" else "")
                  + "\n" + "\n".join(files))
        log.info("[KUBA regions] to vector: %s", report.split("\n")[0])
        return io.NodeOutput(prev, "\n".join(files), report, ui=ui.PreviewImage(prev, cls=cls))


NODE_CLASS_MAPPINGS = {"KUBA_RegionsToVector": KUBA_RegionsToVector}
NODE_DISPLAY_NAME_MAPPINGS = {"KUBA_RegionsToVector": "kubakub regions to svg / pdf / dxf"}
