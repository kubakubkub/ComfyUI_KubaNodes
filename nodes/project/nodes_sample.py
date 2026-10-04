"""kubakub sample facade: a synthetic facade to try every node without files (logic in sample_facade.py)."""

from __future__ import annotations

import hashlib
import os

import torch

import folder_paths
from comfy_api.latest import io

from ...kubakub import sample_facade as sf

CATEGORY = "kubakub/project"


class KUBA_SampleFacade(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="KUBA_SampleFacade",
            display_name="kubakub sample facade",
            category="kubakub/project",
            search_aliases=['demo', 'example', 'test image'],
            description=("A synthetic facade to try the nodes without your own files: arches and a door on the ground "
                         "floor, framed windows above, pilasters and cornices. Gives the look, a flat colour matrix for "
                         "the mask atlas (color_regions), the building silhouette and a mask folder like an After "
                         "Effects export (Groups / Windows). Swap in your own matrix any time."),
            inputs=[
                io.Int.Input("width", default=1920, min=256, max=8192, step=16, tooltip="Size of the matrix in pixels."),
                io.Int.Input("height", default=1080, min=256, max=8192, step=16,
                             tooltip="Height of the matrix in pixels."),
                io.Int.Input("floors", default=3, min=1, max=10, tooltip="Upper floors (above the ground floor)."),
                io.Int.Input("bays", default=7, min=2, max=20, tooltip="Window columns."),
                io.Combo.Input("style", options=list(sf.STYLES), default="sandstone",
                               tooltip="Colour scheme of wall, trim, glass and sky (night = lit windows)."),
                io.Int.Input("seed", default=0, min=0, max=2 ** 31 - 1, control_after_generate=False,
                             tooltip="Small variations of the window glass and the texture."),
            ],
            outputs=[
                io.Image.Output("image", tooltip="The facade as a shaded elevation."),
                io.Image.Output("color_matrix", tooltip="One flat colour per element: kubakub regions from matrix / masks, mode color_regions."),
                io.Mask.Output("silhouette", tooltip="1 = the building, 0 = sky and ground (a projection mask)."),
                io.String.Output("mask_folder", tooltip="A folder with one mask per element (Groups/M_*.png, Windows/W_F*_C*.png): "
                                                        "kubakub facade mask atlas, mode mask_folder, recursive on."),
                io.Image.Output("sketch_photo", tooltip="A phone photo of a hand pencil sketch of this facade, for "
                                                        "kubakub scan to line (red dot = door, blue dots = first floor windows)."),
            ],
        )

    @classmethod
    def execute(cls, width, height, floors, bays, style, seed) -> io.NodeOutput:
        r = sf.render(int(width), int(height), int(floors), int(bays), style, int(seed))
        key = hashlib.sha1(f"{width}|{height}|{floors}|{bays}|{style}|{seed}".encode()).hexdigest()[:12]
        folder = os.path.join(folder_paths.get_temp_directory(), "kubakub_sample", key, "Masks")
        sf.write_mask_folder(r["masks"], folder)
        photo = sf.sketch_photo(int(width), int(height), int(floors), int(bays), int(seed))
        return io.NodeOutput(torch.from_numpy(r["image"])[None], torch.from_numpy(r["matrix"])[None],
                             torch.from_numpy(r["silhouette"])[None], folder, torch.from_numpy(photo)[None])


NODE_CLASS_MAPPINGS = {"KUBA_SampleFacade": KUBA_SampleFacade}
NODE_DISPLAY_NAME_MAPPINGS = {"KUBA_SampleFacade": "kubakub sample facade"}
