"""kubakub sample facade / sample model: a synthetic facade, as a picture and as a 3D file, to try every node without
files (logic in sample_facade.py, sample_model.py)."""

from __future__ import annotations

import hashlib
import json
import os

import torch

import folder_paths
from comfy_api.latest import io

from ...kubakub import sample_facade as sf
from ...kubakub import sample_model as sm

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


class KUBA_SampleModel(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="KUBA_SampleModel",
            display_name="kubakub sample model",
            category="kubakub/project",
            search_aliases=['demo', 'example', '3d', 'obj', 'test model'],
            description=("A 3D model of the sample facade to try the 3d nodes without your own file: stones in "
                         "courses, arches, framed windows, pilasters and cornices, in metres. Connect 'file' to "
                         "kubakub scene render. Every stone is a part of its own (kubakub scene pieces moves them); "
                         "objects are named by element and have materials. The file has no camera: add kubakub "
                         "projector, or scene render frames the model by itself."),
            inputs=[
                io.Int.Input("floors", default=3, min=1, max=10, tooltip="Upper floors (above the ground floor)."),
                io.Int.Input("bays", default=7, min=2, max=20, tooltip="Window columns."),
                io.Float.Input("width_m", default=24.0, min=4.0, max=200.0, step=0.5, tooltip="Width of the facade in metres."),
                io.Float.Input("height_m", default=16.0, min=4.0, max=200.0, step=0.5, tooltip="Height of the facade in metres."),
                io.Float.Input("relief_m", default=0.015, min=0.0, max=0.5, step=0.005, advanced=True,
                               tooltip="How far single stones stand out of the wall (rustication)."),
                io.Int.Input("seed", default=0, min=0, max=2 ** 31 - 1, control_after_generate=False,
                             tooltip="Which stones stand out."),
            ],
            outputs=[
                io.String.Output("file", tooltip="The model as an .obj (in ComfyUI's temp folder) -> kubakub scene render, file."),
                io.String.Output("report", tooltip="Size, objects, parts and faces."),
            ],
        )

    @classmethod
    def execute(cls, floors, bays, width_m, height_m, relief_m, seed) -> io.NodeOutput:
        key = hashlib.sha1(f"{floors}|{bays}|{width_m}|{height_m}|{relief_m}|{seed}|{sm.VERSION}".encode()).hexdigest()[:12]
        path = os.path.join(folder_paths.get_temp_directory(), "kubakub_sample", key, "sample_facade.obj")
        if os.path.isfile(path) and os.path.isfile(path[:-4] + ".json"):     # written once: scene render keeps its cache
            with open(path[:-4] + ".json", encoding="utf-8") as f:
                info = json.load(f)
        else:
            info = sm.write_obj(path, int(floors), int(bays), float(width_m), float(height_m), float(relief_m), int(seed))
            with open(path[:-4] + ".json", "w", encoding="utf-8") as f:
                json.dump(info, f)
        report = (f"sample facade {info['width_m']:.1f} x {info['height_m']:.1f} m: {info['objects']} objects, "
                  f"{info['pieces']} parts, {info['faces']} faces\n{path}")
        return io.NodeOutput(path, report)


NODE_CLASS_MAPPINGS = {"KUBA_SampleFacade": KUBA_SampleFacade, "KUBA_SampleModel": KUBA_SampleModel}
NODE_DISPLAY_NAME_MAPPINGS = {"KUBA_SampleFacade": "kubakub sample facade", "KUBA_SampleModel": "kubakub sample model"}
