"""
nodes_canvas.py

Kuba Regions geometry nodes (category KUBAKUB/regions):
  KUBA_CanvasPlan     target size + model family -> work canvas on the model grid
  KUBA_CanvasToWork   image / mask -> work canvas (uniform resize, pad, scope mask)
  KUBA_CanvasRestore  generated canvas -> exact target (crop padding, uniform resize)

The logic lives in geometry.py. V3 node schema (comfy_api.latest), registered
through the pack's NODE_CLASS_MAPPINGS loader in ../__init__.py.
"""

from __future__ import annotations

import json
import logging

from comfy_api.latest import io, ui

import comfy.utils

from ...kubakub import geometry as geo
from ...kubakub.io_types import CanvasPlanType

log = logging.getLogger("KUBA.regions")

CATEGORY = "kubakub/2d/regions/canvas"

RESIZE_METHODS = ("auto", "lanczos", "bicubic", "bilinear", "area", "nearest-exact")


def comfy_resize(method: str) -> geo.Resize:
    """geometry.Resize backed by comfy.utils.common_upscale. auto = area down, lanczos up."""
    def run(x, w, h):
        if x.shape[-1] == w and x.shape[-2] == h:
            return x
        m = method
        if m == "auto":
            m = "area" if w < x.shape[-1] else "lanczos"
        if m == "lanczos" and x.shape[1] not in (1, 3, 4):
            m = "bicubic"       # the lanczos path goes through PIL images
        return comfy.utils.common_upscale(x, w, h, m, "disabled").clamp(0.0, 1.0)
    return run


class KUBA_CanvasPlan(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="KUBA_CanvasPlan",
            display_name="kubakub canvas plan",
            category="kubakub/2d/regions/canvas",
            description=(
                "Work out the size a model should generate at, so the result scales back to your "
                "target without stretching. The work canvas is target / k for one "
                "uniform factor k, padded to the model's size grid, never stretched. Feed "
                "width/height to the empty latent and the plan to Canvas To Work / Canvas Restore."),
            inputs=[
                io.Int.Input("target_width", default=3840, min=16, max=32768, step=1,
                             tooltip="Width of the final image in pixels (the matrix = the full "
                                     "projection image at delivery size)."),
                io.Int.Input("target_height", default=2160, min=16, max=32768, step=1,
                             tooltip="Height of the final image in pixels."),
                io.Combo.Input("model_family", options=list(geo.GEOMETRIES), default="flux2",
                               tooltip="flux2: divisor 16, ~1 MP. ltxav: divisor 32, 8k+1 frames. "
                                       "minimax_h3: divisor 32, 17k+5 frames, 24 fps. sd_8: divisor 8."),
                io.String.Input("k", default="2",
                                tooltip="Downscale factor target -> work. A number (2, 2.5), a fraction "
                                        "(15/4) or 'auto' to pick from the MP budget. It must divide "
                                        "the target into whole pixels."),
                io.Float.Input("budget_mp", default=0.0, min=0.0, max=16.0, step=0.05,
                               tooltip="auto k: target megapixels on the model grid. 0 = the "
                                       "model's trained budget."),
                io.Boolean.Input("integer_k", default=False,
                                 tooltip="auto k: only whole factors (needed for latent x2 paths)."),
                io.Int.Input("target_frames", default=0, min=0, max=10000, step=1,
                             tooltip="Video models: frames wanted. Rounded up to the model's frame "
                                     "rule; Canvas Restore trims the extra frames. 0 = still."),
                io.Image.Input("size_from", optional=True,
                               tooltip="Take the target size from this image instead of the widgets."),
                io.Combo.Input("pad_align", options=list(geo.PAD_ALIGNS), default="center",
                               advanced=True,
                               tooltip="Where the grid padding goes: split on both sides, or all at "
                                       "the end (right/bottom) or start (left/top)."),
                io.Float.Input("max_pad_percent", default=2.0, min=0.0, max=50.0, step=0.1,
                               advanced=True, tooltip="auto k: skip factors needing more padding "
                                                      "(extra area in percent)."),
                io.Int.Input("divisor_override", default=0, min=0, max=256, step=1, advanced=True,
                             tooltip="Force another size grid (e.g. 64). 0 = the model's own."),
            ],
            outputs=[
                CanvasPlanType.Output("canvas_plan", tooltip="The plan; connect to canvas to work and "
                                                             "canvas restore."),
                io.Int.Output("width", tooltip="Model grid width (padded). Use for the empty latent."),
                io.Int.Output("height", tooltip="Model grid height (padded)."),
                io.Int.Output("frames", tooltip="Frames to generate (1 for stills)."),
                io.Float.Output("fps", tooltip="Frame rate the model generates at."),
                io.String.Output("plan_json", tooltip="The plan as text, for saving or checking."),
                io.String.Output("report", tooltip="Readable summary: work size, padding, k, frames."),
            ],
        )

    @classmethod
    def execute(cls, target_width, target_height, model_family, k, budget_mp, integer_k,
                target_frames, size_from=None, pad_align="center", max_pad_percent=2.0,
                divisor_override=0) -> io.NodeOutput:
        if size_from is not None:
            target_height, target_width = int(size_from.shape[1]), int(size_from.shape[2])
        g = geo.get_geometry(model_family, divisor_override)
        k_text = (k or "").strip().lower()
        plan = geo.plan_canvas(
            target_width, target_height, g, k=None if k_text in ("", "auto", "0") else k_text,
            budget_mp=budget_mp, integer_k=integer_k, pad_align=pad_align,
            max_pad_percent=max_pad_percent, target_frames=target_frames)
        report = plan.report()
        log.info("[KUBA regions] canvas plan\n%s", report)
        frames = plan.work_frames or 1
        return io.NodeOutput(plan, plan.padded_w, plan.padded_h, frames, float(plan.fps or g.fps),
                             json.dumps(plan.to_dict(), indent=1), report,
                             ui=ui.PreviewText(report))


class KUBA_CanvasToWork(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="KUBA_CanvasToWork",
            display_name="kubakub canvas to work",
            category="kubakub/2d/regions/canvas",
            description=(
                "Bring an image (matrix, start image, reference or first frame) and an optional "
                "mask onto the planned work canvas: uniform resize to target / k, then pad to the "
                "model grid. scope is 1 on the real canvas and 0 on padding."),
            inputs=[
                io.Image.Input("image", tooltip="The image to bring onto the work canvas (matrix, "
                                                "start image, reference or first frame)."),
                CanvasPlanType.Input("canvas_plan", tooltip="From kubakub canvas plan."),
                io.Mask.Input("mask", optional=True,
                              tooltip="Optional mask, resized and padded the same way as the image."),
                io.Combo.Input("aspect_mode", options=list(geo.ASPECT_MODES), default="error",
                               tooltip="If the image's aspect differs from the target: error, "
                                       "center_crop, or pad (for references and first frames). "
                                       "Never stretch."),
                io.Combo.Input("pad_mode", options=list(geo.PAD_MODES), default="replicate",
                               tooltip="What the model sees in the grid padding."),
                io.Combo.Input("resize_method", options=list(RESIZE_METHODS), default="auto",
                               advanced=True,
                               tooltip="Resize filter: auto = area when shrinking, lanczos when growing."),
            ],
            outputs=[
                io.Image.Output("image", tooltip="The image on the work canvas, padded to the model grid."),
                io.Mask.Output("mask", tooltip="The input mask on the work canvas (zeros if none)."),
                io.Mask.Output("scope", tooltip="1 = target canvas, 0 = padding."),
            ],
        )

    @classmethod
    def execute(cls, image, canvas_plan, mask=None, aspect_mode="error", pad_mode="replicate",
                resize_method="auto") -> io.NodeOutput:
        plan = canvas_plan
        if mask is not None and mask.ndim == 2:
            mask = mask[None]
        img, m, scope = geo.to_work(image, plan, mask=mask, aspect_mode=aspect_mode,
                                    pad_mode=pad_mode, resize=comfy_resize(resize_method))
        if m is None:
            m = scope.new_zeros((1, plan.padded_h, plan.padded_w))
        log.info("[KUBA regions] to work: %dx%d -> %dx%d (work %dx%d + padding)",
                 image.shape[2], image.shape[1], plan.padded_w, plan.padded_h,
                 plan.work_w, plan.work_h)
        return io.NodeOutput(img, m, scope)


class KUBA_CanvasRestore(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="KUBA_CanvasRestore",
            display_name="kubakub canvas restore",
            category="kubakub/2d/regions/canvas",
            description=(
                "Bring a generated canvas back to the exact target: crop the grid padding, resize "
                "uniformly to the target size, trim extra video frames, check the size. Put a "
                "model upscaler (4x, SeedVR2) before it; any uniform scale of the padded or "
                "cropped canvas is accepted."),
            inputs=[
                io.Image.Input("image", tooltip="The generated (or upscaled) work canvas."),
                CanvasPlanType.Input("canvas_plan", tooltip="The same plan used for canvas to work."),
                io.Mask.Input("mask", optional=True,
                              tooltip="Optional mask on the work canvas, restored the same way."),
                io.Combo.Input("resize_method", options=list(RESIZE_METHODS), default="auto",
                               tooltip="Final resize to the target: auto = area when shrinking, "
                                       "lanczos when growing."),
                io.Boolean.Input("trim_frames", default=True, advanced=True,
                                 tooltip="Video: drop the frames added by the model's frame rule."),
            ],
            outputs=[
                io.Image.Output("image", tooltip="The image at the exact target size."),
                io.Mask.Output("mask", tooltip="The mask at the target size (zeros if none)."),
            ],
        )

    @classmethod
    def execute(cls, image, canvas_plan, mask=None, resize_method="auto",
                trim_frames=True) -> io.NodeOutput:
        plan = canvas_plan
        if mask is not None and mask.ndim == 2:
            mask = mask[None]
        in_w, in_h = image.shape[2], image.shape[1]
        img, m = geo.restore(image, plan, mask=mask, resize=comfy_resize(resize_method),
                             trim_frames=trim_frames)
        if m is None:
            m = img.new_zeros((1, plan.target_h, plan.target_w))
        log.info("[KUBA regions] restore: %dx%d -> %dx%d, %d frames",
                 in_w, in_h, plan.target_w, plan.target_h, img.shape[0])
        return io.NodeOutput(img, m)


NODE_CLASS_MAPPINGS = {
    "KUBA_CanvasPlan": KUBA_CanvasPlan,
    "KUBA_CanvasToWork": KUBA_CanvasToWork,
    "KUBA_CanvasRestore": KUBA_CanvasRestore,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "KUBA_CanvasPlan": "kubakub canvas plan",
    "KUBA_CanvasToWork": "kubakub canvas to work",
    "KUBA_CanvasRestore": "kubakub canvas restore",
}
