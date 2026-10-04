"""kubakub brightness compensation: even out the projector's light over the building (logic in scene3d/compensate.py)."""

from __future__ import annotations

import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

from comfy_api.latest import io, ui

from ...kubakub.io_types import RegionsType, SceneType
from .nodes_scene3d import load_scene_cached
from ...kubakub.scene3d import compensate as cp, scene_view as sv

log = logging.getLogger("KUBA.regions")

CATEGORY = "kubakub/3d/scene"


class KUBA_BrightnessCompensation(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="KUBA_BrightnessCompensation",
            display_name="kubakub brightness compensation",
            category="kubakub/3d/scene",
            search_aliases=['even', 'hotspot', 'falloff', 'projector light', 'uniform'],
            description=("The projector lights near, frontal parts of the building brighter than far or turned-away "
                         "ones. This darkens the bright parts (and optionally lifts the dim ones) so your content "
                         "looks equally bright everywhere on the building. Put it last, before the export."),
            inputs=[
                io.Image.Input("images", tooltip="Your finished frames at matrix size (one still or a video batch)."),
                SceneType.Input("scene", tooltip="From kubakub scene render (the projector's view of the building)."),
                io.Combo.Input("match", options=list(cp.MATCH), default="dim areas",
                               tooltip="dim areas = everything as even as the dimmest parts (most even, darker); "
                                       "average = only the brightest parts come down."),
                io.Float.Input("strength", default=1.0, min=0.0, max=1.0, step=0.05,
                               tooltip="0 = unchanged, 1 = fully even. 0.5-0.7 keeps a little of the natural falloff."),
                io.Float.Input("lift_dim_up_to", default=1.0, min=1.0, max=4.0, step=0.05,
                               tooltip="1 = never brighter than your content (safe). Above 1 dim parts get lifted too; "
                                       "their highlights may clip."),
                io.Float.Input("smooth_px", default=8.0, min=0.0, max=256.0, step=1.0,
                               tooltip="Softens the gain map (pixels of the projection view), hides small steps."),
                io.Float.Input("grazing_deg", default=60.0, min=0.0, max=90.0, step=1.0, advanced=True,
                               tooltip="Surfaces hit steeper than this (side faces of mouldings) don't set the level: "
                                       "they are dim by nature and would darken everything else."),
                RegionsType.Input("regions", optional=True,
                                  tooltip="One gain per region instead of per pixel, so no gradient inside a window."),
            ],
            outputs=[
                io.Image.Output("images", tooltip="The compensated frames."),
                io.Image.Output("gain_map", tooltip="How much each part is changed (dark = darkened, colours as in scene measure)."),
                io.String.Output("report", tooltip="Target, how much light is kept overall and how much clips."),
            ],
        )

    @classmethod
    def execute(cls, images, scene, match, strength, lift_dim_up_to, smooth_px, grazing_deg=60.0, regions=None) -> io.NodeOutput:
        t0 = time.perf_counter()
        s, pt, nrm, _ground, _fr = load_scene_cached(scene)
        maps = sv.measure_maps(s, pt, nrm)
        fg = s["faceid"] > 0
        labels = regions.labels[0].cpu().numpy() if regions is not None else None
        gain, info = cp.gain_map(maps["brightness"], fg, match, strength, lift_dim_up_to, smooth_px, labels,
                                 maps["incidence_deg"], grazing_deg)
        n, h, w = int(images.shape[0]), int(images.shape[1]), int(images.shape[2])
        g = cp.resize_gain(gain, w, h)
        out = np.empty((n, h, w, 3), np.float32)      # numpy: the worker threads are outside inference mode
        clipped = [0.0] * n

        def one(i):
            out[i], clipped[i] = cp.apply(images[i, ..., :3].cpu().float().numpy(), g)

        workers = max(1, min(8, (os.cpu_count() or 4) - 2, n))
        with ThreadPoolExecutor(max_workers=workers) as pool:        # one frame at a time per worker, in place
            list(pool.map(one, range(n)))
        rgb, lo, hi = sv.colorize(gain, fg, lo=min(info["min"], 1.0), hi=max(info["max"], 1.0))
        preview = torch.from_numpy(np.ascontiguousarray(rgb, np.float32))[None]
        report = (f"{n} frame(s) {w}x{h}; evened to the {match} level ({info['target']:.2f} of a typical facade); "
                  f"({info['grazing'] * 100:.0f} % grazing surfaces ignored); gain {info['min']:.2f} .. {info['max']:.2f}; overall light kept {info['kept'] * 100:.0f} %"
                  + (f"; clipped {max(clipped) * 100:.1f} % of pixels (lift_dim_up_to)" if max(clipped) > 0 else "")
                  + ("; one gain per region" if labels is not None else "")
                  + f"; {time.perf_counter() - t0:.1f} s")
        log.info("[KUBA compensate] %s", report)
        return io.NodeOutput(torch.from_numpy(out), preview, report, ui=ui.PreviewImage(preview, cls=cls))


NODE_CLASS_MAPPINGS = {"KUBA_BrightnessCompensation": KUBA_BrightnessCompensation}
NODE_DISPLAY_NAME_MAPPINGS = {"KUBA_BrightnessCompensation": "kubakub brightness compensation"}
