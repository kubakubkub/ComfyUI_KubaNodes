"""kubakub render passes: a folder of render passes from any 3D tool -> picture, depth, normals, masks (logic in render_passes.py)."""

from __future__ import annotations

import logging
import os

import numpy as np
import torch

import folder_paths

from comfy_api.latest import io, ui

from ...kubakub import render_passes as rp
from ...kubakub import samples

log = logging.getLogger("KUBA.regions")

CATEGORY = "kubakub/3d/from renders"


def _clean(path):
    return os.path.expandvars(os.path.expanduser((path or "").strip().strip('"').strip("'")))


class KUBA_RenderPasses(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="KUBA_RenderPasses",
            display_name="kubakub render passes",
            category="kubakub/3d/from renders",
            search_aliases=['aov', 'beauty', 'depth pass', 'houdini', 'karma', 'blender', 'c4d', 'redshift', 'load passes'],
            description=("Your own render as the start: reads a folder of passes written by any 3D tool, named "
                         "<render>_<pass>.png (facade_beauty.png, facade_depth.png, facade_normal.png, facade_cut.png). "
                         "Gives the picture, depth and normals, and every other pass as a mask with its name. Connect "
                         "masks and names to kubakub regions from masks: each mask becomes a region you can repaint "
                         "while the rest of the render stays as it is."),
            inputs=[
                io.String.Input("folder", default="", placeholder="paste the folder of your passes  (empty = sample passes)",
                                tooltip="The folder with your passes: paste its path (Explorer: Copy as path; the quotes "
                                        "are fine). Empty: built-in sample passes, so the node runs as it is."),
                io.String.Input("render", default="", optional=True,
                                tooltip="Which render, when the folder holds several (facade_a for facade_a_beauty.png). "
                                        "Empty: the first one; the report lists them."),
                io.String.Input("invert", default="", optional=True, placeholder="cut",
                                tooltip="Masks where black means inside (names, wildcards, comma separated)."),
                io.String.Input("exclude", default="", optional=True, advanced=True, placeholder="alpha, wire*",
                                tooltip="Passes that are no masks (names, wildcards, comma separated)."),
                io.Combo.Input("depth", options=["as rendered", "near is white", "near is black"], default="as rendered",
                               optional=True, advanced=True,
                               tooltip="How the depth output reads. The last two stretch it over the whole range."),
            ],
            outputs=[
                io.Image.Output("beauty", tooltip="The picture: the image for the region sampler / versions."),
                io.Image.Output("depth", tooltip="The depth pass (black when the folder has none)."),
                io.Image.Output("normal", tooltip="The normal pass (black when the folder has none)."),
                io.Mask.Output("masks", tooltip="Every other pass as a mask (batch) -> kubakub regions from masks, masks."),
                io.String.Output("names", tooltip="The masks' names, one per line -> kubakub regions from masks, names."),
                io.String.Output("report", tooltip="What was found in the folder."),
            ],
        )

    @classmethod
    def fingerprint_inputs(cls, folder, **kw):
        p = _clean(folder)
        try:
            stamp = sorted((e.name, e.stat().st_size, e.stat().st_mtime_ns) for e in os.scandir(p) if e.is_file())
        except OSError:
            stamp = []
        return f"{p}|{stamp}|{sorted(kw.items())}"

    @classmethod
    def execute(cls, folder, render="", invert="", exclude="", depth="as rendered") -> io.NodeOutput:
        sample = not _clean(folder)
        r = rp.load(samples.passes(folder_paths.get_temp_directory()) if sample else _clean(folder),
                    render, invert, exclude, depth)
        if sample:
            r["notes"].append(samples.note("folder of render passes", "folder"))
        H, W = r["beauty"].shape[:2]
        img = lambda a: torch.from_numpy(np.ascontiguousarray(a if a is not None else np.zeros((H, W, 3), np.float32)))[None]  # noqa: E731
        if r["masks"]:
            masks = torch.from_numpy(np.stack([m for _, m in r["masks"]]))
        else:
            masks = torch.zeros((1, H, W), dtype=torch.float32)
        names = "\n".join(n for n, _ in r["masks"])
        report = "\n".join([
            f"render '{r['name'] or 'beauty'}' {W}x{H}" + (f"; also in the folder: {', '.join(n for n in r['renders'] if n != r['name'])}"
                                                           if len(r["renders"]) > 1 else ""),
            f"depth: {'yes' if r['depth'] is not None else 'none'}, normal: {'yes' if r['normal'] is not None else 'none'}",
            "masks: " + (", ".join(f"{n} ({m.mean() * 100:.0f} %)" for n, m in r["masks"])
                         or "none (the output is one empty mask)"),
            *r["notes"]])
        log.info("[KUBA render passes] %s", report.replace("\n", " | "))
        beauty = img(r["beauty"])
        return io.NodeOutput(beauty, img(r["depth"]), img(r["normal"]), masks, names, report,
                             ui=ui.PreviewImage(beauty, cls=cls))


NODE_CLASS_MAPPINGS = {"KUBA_RenderPasses": KUBA_RenderPasses}
NODE_DISPLAY_NAME_MAPPINGS = {"KUBA_RenderPasses": "kubakub render passes"}
