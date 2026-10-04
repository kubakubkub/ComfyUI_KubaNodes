"""kubakub project settings: one place for the matrix size, fps, name and the audience (logic in project.py)."""

from __future__ import annotations

from comfy_api.latest import io

from ...kubakub import project as pj
from ...kubakub.io_types import ViewerType

CATEGORY = "kubakub/project"


class KUBA_Project(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="KUBA_Project",
            display_name="kubakub project settings",
            category="kubakub/project",
            search_aliases=['settings', 'matrix size', 'fps', 'template', 'show'],
            description=("Set the show once and link it everywhere: width / height into scene render, preview or "
                         "walkthrough, fps into export, the video sampler or keyframe clips, name into export, "
                         "viewer into frame in frame. A connected matrix gives the size by itself."),
            inputs=[
                io.String.Input("name", default="my show", tooltip="Project name; export folders and files use it."),
                io.Combo.Input("preset", options=list(pj.PRESETS), default="uhd 3840x2160",
                               tooltip="Matrix size; custom = width / height below. A connected matrix wins."),
                io.Int.Input("width", default=3840, min=64, max=16384, step=8, tooltip="Matrix width (preset custom)."),
                io.Int.Input("height", default=2160, min=64, max=16384, step=8, tooltip="Matrix height (preset custom)."),
                io.Float.Input("fps", default=25.0, min=1.0, max=120.0, step=0.001,
                               tooltip="Frames per second of the show (e.g. 25, 30 or 29.97)."),
                io.Float.Input("facade_width_m", default=40.0, min=0.1, max=10000.0, step=0.1,
                               tooltip="Real width of the whole matrix on the building, in metres."),
                io.Float.Input("bottom_m", default=0.0, min=-100.0, max=1000.0, step=0.1, advanced=True,
                               tooltip="Height of the matrix's bottom edge above the ground where people stand."),
                io.Float.Input("viewer_x_m", default=-1.0, min=-10000.0, max=10000.0, step=0.1, advanced=True,
                               tooltip="Audience position along the facade from its left edge; -1 = centre."),
                io.Float.Input("eye_height_m", default=1.7, min=0.0, max=1000.0, step=0.05, advanced=True,
                               tooltip="Eye height of the audience above the ground (1.7 = a standing adult)."),
                io.Float.Input("distance_m", default=30.0, min=0.5, max=10000.0, step=0.5,
                               tooltip="Distance of the audience from the facade, in metres."),
                io.Image.Input("matrix", optional=True, tooltip="The festival's matrix / template: its size is used."),
            ],
            outputs=[
                io.Int.Output("width", tooltip="Matrix width in pixels."),
                io.Int.Output("height", tooltip="Matrix height in pixels."),
                io.Float.Output("fps", tooltip="Frames per second."),
                io.String.Output("name", tooltip="The project name, safe for folders and files."),
                ViewerType.Output("viewer", tooltip="The audience viewpoint (same as kubakub audience viewpoint)."),
                io.String.Output("report", tooltip="Size, fps, millimetres per pixel and the audience position."),
            ],
        )

    @classmethod
    def execute(cls, name, preset, width, height, fps, facade_width_m, bottom_m, viewer_x_m, eye_height_m,
                distance_m, matrix=None) -> io.NodeOutput:
        p = pj.build(name, preset, width, height, fps, facade_width_m, bottom_m, viewer_x_m, eye_height_m, distance_m,
                     None if matrix is None else tuple(matrix.shape))
        return io.NodeOutput(p["width"], p["height"], p["fps"], p["name"], p["viewer"], p["report"])


NODE_CLASS_MAPPINGS = {"KUBA_Project": KUBA_Project}
NODE_DISPLAY_NAME_MAPPINGS = {"KUBA_Project": "kubakub project settings"}
