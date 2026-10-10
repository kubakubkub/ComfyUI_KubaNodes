"""
nodes_formheight.py

kubakub form height: a 3D model that lies on the real building (a generated one after kubakub align 3d model) becomes
a height map in the projection view: 0 on the wall, white where a form stands furthest out of it. A picture, so
everything that works on pictures and masks works on it: grown from 0 to 1 the forms come out of the wall, a mask
field or the sound says where and when, a depth shader throws their shadows. The model is drawn from the scene's
camera on the graphics card and the real wall of the scene render is taken away (kubakub/formheight.py).
V3 node schema, registered via ../__init__.py.
"""

from __future__ import annotations

import logging
import os
import time

import numpy as np
import torch

from comfy_api.latest import io, ui

from ...kubakub import formheight as fh
from ...kubakub import imio
from ...kubakub.io_types import SceneType
from ...kubakub.scene3d import scene_ids
from ...kubakub.scene3d.bridge import clean_path
from .nodes_align3d import _from_mesh, _load, _look_of_file, _look_of_mesh

log = logging.getLogger("KUBA.scene3d")
NAME = "kubakub form height"
SAMPLE_COLOUR = (0.945, 0.541, 0.345)             # the sample form with colours on: kubakub orange


class KUBA_FormHeight(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="KUBA_FormHeight",
            display_name=NAME,
            category="kubakub/3d/scene",
            search_aliases=["height map", "displacement", "relief", "depth", "trellis", "image to 3d", "grow"],
            description=(
                "How far a 3D model stands out of the real wall, as a picture in the projection view: black on the "
                "wall, white where a form stands furthest out. Give the scene of your real building (kubakub scene "
                "render) and the generated model after kubakub align 3d model. The wall stays your real one; only "
                "what stands in front of it counts. Use the height like any picture or mask: brought up from black "
                "the forms grow out of the wall, a mask field or the sound says where and when, and 'depth' is the "
                "building with the forms on it for the depth shaders (hard shadow, sheet of light) and for "
                "displacement in Houdini or Blender. No model connected = a sample form."),
            inputs=[
                SceneType.Input("scene", tooltip="From kubakub scene render of the real model: its camera is the view, "
                                                 "its wall is where the height is counted from."),
                io.Mesh.Input("model", optional=True,
                              tooltip="The model that lies on the building, e.g. 'model' of kubakub align 3d model. Or "
                                      "give a file in model_file."),
                io.String.Input("model_file", default="", optional=True,
                                tooltip="The model as a file (.glb, .gltf, .obj, .ply, .stl), e.g. 'file' of kubakub "
                                        "align 3d model. Empty and nothing connected: a sample form on the wall."),
                io.Float.Input("wall_m", default=0.5, min=0.0, max=20.0, step=0.01,
                               tooltip="What stands out less than this (metres) is the wall. A generated wall never "
                                       "lies exactly on the real one; raise it until the wall is black, lower it to "
                                       "keep flat forms. The height is counted from here."),
                io.Float.Input("smallest_form_m", default=0.5, min=0.0, max=50.0, step=0.05,
                               tooltip="What is thinner than this (metres) is dropped: specks, and the lines along "
                                       "cornices and window edges where the generated wall sits a little off. 0 keeps "
                                       "everything."),
                io.Combo.Input("counted_from", options=list(fh.FROMS), default=fh.FROMS[0],
                               tooltip="the wall's plane: the height is counted from the flat main wall, so the real "
                                       "wall's own cornices and window niches do not show inside the forms; the "
                                       "clean picture. the real surface: counted from the real model under each "
                                       "pixel, which is what a displacement of the real model needs."),
                io.Float.Input("white_at_m", default=0.0, min=0.0, max=100.0, step=0.05,
                               tooltip="The height (metres) that is white. 0 = the highest point of this model. Set a "
                                       "number to give several models the same scale."),
                io.Combo.Input("up", options=list(fh.UPS), default=fh.UPS[0], optional=True, advanced=True,
                               tooltip="Which way is up in the model. .glb and .obj files count y as up and Blender "
                                       "turns them when it opens them; the scene counts z. auto tries both and takes "
                                       "the one that lies on the building."),
                io.Boolean.Input("colours", default=False, optional=True,
                                 tooltip="On: the 'colour' output shows the forms in the model's own colours (its "
                                         "texture, or the colours on its points), seen from the scene's camera, on the "
                                         "clay render of the real building. Off: 'colour' is the clay render alone."),
            ],
            outputs=[
                io.Image.Output("height", tooltip="The height map: black = the wall, white = white_at_m (or the highest "
                                                  "point). The report says how many metres white is."),
                io.Mask.Output("forms", tooltip="1 where a form stands in front of the wall."),
                io.Image.Output("depth", tooltip="The real building with the forms on it as grey depth, near = white: "
                                                 "for depth shaders and for displacement."),
                io.String.Output("report", tooltip="How well the model lies on the building, how high the forms stand, "
                                                   "how much of the picture they cover."),
                io.Image.Output("colour", tooltip="With 'colours' on: the real building as clay with the forms in the "
                                                  "model's own colours, no light on them. Connect it to 'image' of "
                                                  "kubakub height animate and the forms grow in colour."),
            ],
        )

    @classmethod
    def execute(cls, scene, model=None, model_file="", wall_m=0.5, smallest_form_m=0.5, counted_from=fh.FROMS[0],
                white_at_m=0.0, up=fh.UPS[0], colours=False) -> io.NodeOutput:
        t0 = time.perf_counter()
        s = scene_ids.load(scene["folder"], mesh=False)
        try:
            w = fh.wall(s)
        except ValueError as e:
            raise ValueError(f"{NAME}: the scene cannot be used ({e}).") from e
        cam, notes = w["cam"], []
        counted_from = counted_from if counted_from in fh.FROMS else fh.FROMS[0]
        path = clean_path(model_file)
        scale = float(scene.get("unit_scale") or 1.0)     # the model is in the units of the scene's file
        look = None                                       # what the model wears, when its colours are asked for
        if model is not None:
            mv, mf = _from_mesh(model)
            if path:
                notes.append("model and model_file are both given: the connected model is used")
            if colours:
                look = _look_of_mesh(model, len(mv), len(mf), fh.shown)
        elif path:
            loaded, mv, mf = _load(path, "model_file", NAME)
            if colours:
                look = _look_of_file(loaded, fh.shown)
        else:
            mv, mf = fh.ball(s)                           # made in the scene's world
            up, scale = fh.UPS[2], 1.0
            look = {"col": np.tile(SAMPLE_COLOUR, (len(mv), 1)), "parts": [(0, len(mf), None)]}
            notes.append("no model given: a sample form is used (half a ball on the middle of the wall)")
        near = max(2.0 * float(wall_m), 0.02 * w["size_m"])
        tries = []
        try:
            for y_up in ((True, False) if up == fh.UPS[0] else (up == fh.UPS[1],)):
                depth = fh.model_depth(cam, fh.to_scene(mv, y_up, scale), mf)
                tries.append((*fh.lies_on(depth, w, near), y_up, depth))
        except RuntimeError as e:
            raise RuntimeError(f"{NAME}: {e}") from e
        on, cover, y_up, depth = max(tries, key=lambda t: t[:2])
        if len(tries) == 2 and cover >= 0.01:
            notes.append(f"up = auto: the model is read with {'y' if y_up else 'z'} up (it lies on the building that way)")
        elif len(tries) == 2:
            notes.append("up = auto: the model is in front of the building neither with y up nor with z up")
        how = ("Lay it onto the real model with kubakub align 3d model first (give the same model file there as "
               "'template_file' that kubakub scene render reads), or try the other 'up'.")
        if cover < 0.01:
            notes.append("WARNING: the model is not in front of the building in this view. " + how)
        elif cover > 0.5 and on < 0.3:                    # it has a wall of its own, and that wall is somewhere else
            notes.append(f"WARNING: the model covers the building but lies on only {on * 100:.0f} % of it. " + how)
        h, forms = fh.height(depth, w, wall_m, smallest_form_m, counted_from == fh.FROMS[0])
        top = float(h.max())
        white = float(white_at_m) if white_at_m > 0 else top
        grey = np.clip(h / max(white, 1e-6), 0.0, 1.0).astype(np.float32)
        if white_at_m > 0 and top > white:
            notes.append(f"the highest form stands {top:.2f} m out, above white_at_m: it is clipped to white")
        if not forms.any():
            notes.append("no form stands in front of the wall: the height map is black. Lower wall_m or smallest_form_m.")
        clay_file = os.path.join(scene["folder"], "clay.png")
        clay = imio.imread(clay_file) if os.path.isfile(clay_file) else None
        colour = np.full(grey.shape + (3,), 0.5, np.float32) if clay is None else clay[..., 2::-1].astype(np.float32) / 255.0
        if colours and look is None:
            notes.append("colours is on, but the model carries no texture and no colours: 'colour' is the clay render")
        elif colours and forms.any():
            try:
                rgb = fh.model_view(cam, fh.to_scene(mv, y_up, scale), mf, look)[1]
            except RuntimeError as e:
                raise RuntimeError(f"{NAME}: {e}") from e
            colour = np.where(forms[..., None], rgb, colour).astype(np.float32)
        report = "\n".join([
            f"model {len(mv)} vertices, {len(mf)} faces, view {cam.W}x{cam.H}",
            f"the model covers {cover * 100:.0f} % of the building and lies on {on * 100:.0f} % of it (within {near:.2f} m)",
            f"forms cover {forms.mean() * 100:.1f} % of the picture and stand up to {top:.2f} m out of the wall; "
            f"white = {white:.2f} m, black = the wall (what stands out less than {wall_m:g} m); counted from "
            f"{counted_from}",
            f"{time.perf_counter() - t0:.1f} s", *notes])
        log.info("[KUBA form height] %s", report.replace("\n", "\n    "))
        img =torch.from_numpy(np.repeat(grey[..., None], 3, -1))[None]
        dep = torch.from_numpy(np.repeat(fh.with_forms(depth, forms, w)[..., None], 3, -1))[None]
        return io.NodeOutput(img, torch.from_numpy(forms.astype(np.float32))[None], dep, report,
                             torch.from_numpy(np.ascontiguousarray(colour))[None], ui=ui.PreviewImage(img, cls=cls))


NODE_CLASS_MAPPINGS = {"KUBA_FormHeight": KUBA_FormHeight}
NODE_DISPLAY_NAME_MAPPINGS = {"KUBA_FormHeight": NAME}
