"""
nodes_sampler.py

KUBA_RegionSampler (category KUBAKUB/regions): runs a KUBA_PLAN on a canvas with
one model backend, region by region in plan order: inpaint and frame in
frame, through the model adapter (Flux 2 Klein, or the generic KSampler path).

V3 node schema (comfy_api.latest), registered through the pack's
NODE_CLASS_MAPPINGS loader in ../__init__.py.
"""

from __future__ import annotations

import logging

from comfy_api.latest import io

import comfy.model_management
import comfy.samplers
import comfy.utils

from ...kubakub import adapters
from ...kubakub import plan as rp
from ...kubakub import region_cache as rc
from ...kubakub import schedules
from ...kubakub import seams
from ...kubakub import strategies as st
from ...kubakub.io_types import PlanType

log = logging.getLogger("KUBA.regions")

CATEGORY = "kubakub/2d/generate"


def only_filter(text: str):
    """'W_F1_*, group:M_Pilasters' -> predicate on plan entries (plan selector syntax), or None."""
    text = (text or "").strip()
    if not text:
        return None
    try:
        section = rp.parse_rules(f"[{text}]\n")[0]
    except rp.PlanError as e:
        raise ValueError(f"kubakub region sampler only: {e}") from None
    return lambda entry: section.match(entry) is not None


class KUBA_RegionSampler(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="KUBA_RegionSampler",
            display_name="kubakub region sampler",
            category="kubakub/2d/generate",
            search_aliases=['inpaint', 'regional prompt', 'klein', 'qwen'],
            description=(
                "Run a region plan on an image, region by region. inpaint: the region is cut out "
                "with context, scaled uniformly to about region_mp on the model grid, inpainted "
                "with its own prompt and the crop as reference, colour matched and pasted back. "
                "frame_in_frame: the region gets its own scene generated from scratch, placed "
                "through its mask, and its border is harmonized. Later regions see earlier ones. "
                "Flux 2 Klein and Qwen Image 2.1 are detected automatically. Results are kept in RAM "
                "across queues: a region whose crop pixels, masks, prompt, seed, settings and models are "
                "unchanged is pasted from the cache, so changing one prompt re-samples only that region "
                "(and later regions whose crops overlap it)."),
            inputs=[
                io.Model.Input("model", tooltip="The image model (Flux 2 Klein, Qwen Image 2.1 or a "
                                                "generic model), with any LoRAs already applied."),
                io.Clip.Input("clip", tooltip="The text encoder that matches the model."),
                io.Vae.Input("vae", tooltip="The VAE that matches the model."),
                PlanType.Input("plan", tooltip="From kubakub region plan."),
                io.Image.Input("image", tooltip="The canvas, exactly the size of the regions "
                                                "(e.g. the 3840x2160 facade render or photo)."),
                io.Int.Input("steps", advanced=True, default=0, min=0, max=200,
                             tooltip="Used where the plan says steps = 0. 0 = the model's default "
                                     "(Klein 4, Qwen Image 2.1 25)."),
                io.Float.Input("cfg", advanced=True, default=0.0, min=0.0, max=30.0, step=0.1, round=0.01,
                               tooltip="Used where the plan says cfg = 0. 0 = the model's default (1.0)."),
                io.Combo.Input("sampler_name", advanced=True, options=comfy.samplers.KSampler.SAMPLERS, default="euler",
                               tooltip="Sampler for every region (euler is a safe default)."),
                io.Combo.Input("scheduler", advanced=True, options=comfy.samplers.KSampler.SCHEDULERS, default="simple",
                               tooltip="Generic models only; Flux 2 always uses its own schedule "
                                       "(core Flux2Scheduler for the crop size)."),
                io.Float.Input("region_mp", default=1.0, min=0.1, max=4.0, step=0.05,
                               tooltip="Work megapixels per region crop (region + context), and the "
                                       "size of frame_in_frame scenes. Small regions are scaled up "
                                       "to this, large ones down."),
                io.Float.Input("max_upscale", advanced=True, default=4.0, min=1.0, max=8.0, step=0.25,
                               tooltip="Small regions are scaled up at most this much."),
                io.Combo.Input("schedule", options=list(schedules.SCHEDULES), default="auto",
                               tooltip="auto: steps / sampler / scheduler as set. qwen21_turbo_5: the "
                                       "5 trained steps of the Viggle Qwen-Image-2.1 turbo LoRA (load it "
                                       "with LoraLoaderModelOnly); pruna_qwen21_8: the 8 trained steps of "
                                       "the Pruna Qwen-Image-2.1 LoRA (strength 1.0). Partial denoise starts "
                                       "at the matching step, steps is ignored. Sampler as set (euler, or "
                                       "res_2s / deis_2m for more detail)."),
                io.String.Input("only", default="", optional=True,
                                placeholder="W_F1_*, group:M_Pilasters",
                                tooltip="Process only the regions matching these selectors (plan "
                                        "syntax: space = AND, comma = OR). Empty = all."),
                io.Int.Input("seed_offset", default=0, min=0, max=0xFFFFFFFF, advanced=True,
                             tooltip="Added to every region seed, for a new variation of all regions."),
                io.Combo.Input("adapter", options=adapters.families(), default="auto", advanced=True,
                               tooltip="Model family; auto detects Flux 2 and Qwen Image 2.1."),
                # new inputs go last: saved workflows store the widget values by position
                io.Image.Input("style", optional=True,
                               tooltip="A style image for the regions whose plan says reference = style: the "
                                       "model sees it as a second reference next to the region's own crop "
                                       "(scaled to about 1 megapixel). Of a batch the first image is used."),
            ],
            outputs=[
                io.Image.Output("image", tooltip="The image with all processed regions pasted in."),
                io.Mask.Output("changed", tooltip="Where pixels were replaced (paste alpha)."),
                io.String.Output("report", tooltip="Model, cache use, time, and one line per region "
                                                   "(status, crop box and scale)."),
            ],
        )

    @classmethod
    def execute(cls, model, clip, vae, plan, image, steps, cfg, sampler_name, scheduler, region_mp,
                max_upscale, schedule="auto", only="", seed_offset=0, adapter="auto",
                style=None) -> io.NodeOutput:
        ad = adapters.make_adapter(model, clip, vae, adapter)
        ad.set_schedule(schedule)
        log.info("[KUBA regions] sampler: adapter %s, grid %d px, schedule %s", ad.family, ad.grid, ad.schedule)
        settings = st.SamplerSettings(steps=steps, cfg=cfg, sampler_name=sampler_name,
                                      scheduler=scheduler, region_mp=region_mp,
                                      max_upscale=max_upscale, seed_offset=seed_offset,
                                      style=style[:1] if style is not None else None)
        pick = only_filter(only)
        notes = []
        if image.shape[0] > 1:
            log.warning("[KUBA regions] image batch has %d images; using the first.", image.shape[0])
            notes.append(f"the image input is a batch of {image.shape[0]} images: only the first one is used")
        # regions of this run that ask for the style image (frame_in_frame scenes take no reference)
        entries = plan.plan["regions"]
        want_style = [i for i in plan.order
                      if entries[i]["strategy"] == "inpaint" and entries[i].get("reference", "self") == "style"
                      and (pick is None or pick(entries[i]))]
        if want_style and style is None:
            notes.append(f"{len(want_style)} region(s) have reference = style but no style image is connected: "
                         f"they use their own crop (reference = self). Connect an image to 'style'.")
        elif style is not None and not want_style:
            notes.append("a style image is connected but no region of this run has reference = style: "
                         "it is not used. Write 'reference = style' into the plan.")
        elif style is not None:
            notes.append(f"style image used by {len(want_style)} region(s)")
        for n in notes:
            log.info("[KUBA regions] sampler: %s", n)
        pbar = None
        n_todo = len(plan.order)
        if n_todo:
            pbar = comfy.utils.ProgressBar(n_todo)
        out, changed, results = st.run_plan(
            ad, image, plan, settings, only=pick,
            progress=(lambda i, n: pbar.update_absolute(i, n)) if pbar else None,
            check_interrupt=comfy.model_management.throw_exception_if_processing_interrupted)
        report = (f"adapter {ad.family}, grid {ad.grid}, schedule {ad.schedule}; cache {rc.RESULTS.stats()}\n"
                  + st.results_report(results) + "".join(f"\nnote: {n}" for n in notes))
        log.info("[KUBA regions] %s", report.splitlines()[1])
        return io.NodeOutput(out, changed, report)


class KUBA_SeamPass(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="KUBA_SeamPass",
            display_name="kubakub seam pass",
            category="kubakub/2d/generate",
            search_aliases=['seams', 'blend edges'],
            description=(
                "Blend the borders between regions after the region sampler, so neighbouring regions "
                "grow together instead of meeting at a hard edge. A soft band along the region borders that "
                "changed (both sides blend = on, keep regions untouched) is resampled tile by tile "
                "at full resolution with a low denoise, the crop as reference and optional "
                "differential diffusion."),
            inputs=[
                io.Model.Input("model", tooltip="The same image model as in the region sampler."),
                io.Clip.Input("clip", tooltip="The text encoder that matches the model."),
                io.Vae.Input("vae", tooltip="The VAE that matches the model."),
                PlanType.Input("plan", tooltip="The same plan as the region sampler: gives the borders "
                                               "and each region's prompt."),
                io.Image.Input("image", tooltip="The region sampler's output."),
                io.Mask.Input("changed", optional=True,
                              tooltip="The region sampler's changed output: only borders next to "
                                      "changed pixels get a seam. Without it every border does."),
                io.Int.Input("seam_px", default=32, min=4, max=256, step=2,
                             tooltip="Width of the soft band across a border (target pixels)."),
                io.Float.Input("denoise", default=0.3, min=0.0, max=1.0, step=0.01,
                               tooltip="How much the seam band may change: low keeps the regions, "
                                       "higher blends more (0.3 = gentle)."),
                io.Int.Input("steps", advanced=True, default=0, min=0, max=200,
                             tooltip="0 = the model's default (Klein 4, Qwen Image 2.1 25)."),
                io.Float.Input("cfg", advanced=True, default=0.0, min=0.0, max=30.0, step=0.1, round=0.01,
                               tooltip="0 = the model's default (1.0)."),
                io.Combo.Input("sampler_name", advanced=True, options=comfy.samplers.KSampler.SAMPLERS,
                               default="euler", tooltip="Sampler for the seam tiles (euler is a safe default)."),
                io.Combo.Input("scheduler", advanced=True, options=comfy.samplers.KSampler.SCHEDULERS,
                               default="simple", tooltip="Generic models only; Flux 2 uses its own schedule."),
                io.Boolean.Input("differential", default=True,
                                 tooltip="Differential diffusion: the soft band edge is denoised less "
                                         "than its centre, step by step (core DifferentialDiffusion)."),
                io.Combo.Input("schedule", options=list(schedules.SCHEDULES), default="auto",
                               tooltip="auto: steps / sampler / scheduler as set. qwen21_turbo_5: the "
                                       "5 trained steps of the Viggle Qwen-Image-2.1 turbo LoRA (load it "
                                       "with LoraLoaderModelOnly); pruna_qwen21_8: the 8 trained steps of "
                                       "the Pruna Qwen-Image-2.1 LoRA (strength 1.0). Partial denoise starts "
                                       "at the matching step, steps is ignored. Sampler as set (euler, or "
                                       "res_2s / deis_2m for more detail)."),
                io.Int.Input("seed", default=0, min=0, max=0xFFFFFFFFFFFFFFFF, control_after_generate=False,
                             tooltip="Noise seed for the seam tiles; change it for another variation."),
                io.String.Input("prompt", default="", optional=True, multiline=True, dynamic_prompts=False,
                                tooltip="Empty: each tile uses the prompt of the region most present "
                                        "in its seams."),
                io.Int.Input("tile_px", default=1024, min=256, max=2048, step=16, advanced=True,
                             tooltip="Tile size at 1:1 (about 1 MP for 1024)."),
                io.Combo.Input("adapter", options=adapters.families(), default="auto", advanced=True,
                               tooltip="Model family; auto detects Flux 2 and Qwen Image 2.1."),
            ],
            outputs=[
                io.Image.Output("image", tooltip="The image with blended seams."),
                io.Mask.Output("seam_mask", tooltip="The soft band that was resampled."),
                io.String.Output("report", tooltip="Band width, tile count and the prompt used per tile."),
            ],
        )

    @classmethod
    def execute(cls, model, clip, vae, plan, image, seam_px, denoise, steps, cfg, sampler_name,
                scheduler, differential, seed, schedule="auto", changed=None, prompt="", tile_px=1024,
                adapter="auto") -> io.NodeOutput:
        if differential:
            model = seams.differential_diffusion_patch(model, 1.0)
        ad = adapters.make_adapter(model, clip, vae, adapter)
        ad.set_schedule(schedule)
        settings = seams.SeamSettings(seam_px=seam_px, denoise=denoise, tile_px=tile_px, steps=steps,
                                      cfg=cfg, sampler_name=sampler_name, scheduler=scheduler,
                                      seed=seed, prompt=(prompt or "").strip())
        if changed is not None and changed.ndim == 2:
            changed = changed[None]
        pbar = comfy.utils.ProgressBar(1)
        out, mask, lines = seams.seam_pass(
            ad, image, plan, settings, changed=changed,
            progress=lambda i, n: pbar.update_absolute(i, n),
            check_interrupt=comfy.model_management.throw_exception_if_processing_interrupted)
        report = "\n".join([f"adapter {ad.family}, differential {'on' if differential else 'off'}"] + lines)
        log.info("[KUBA regions] seam pass: %s", lines[0])
        return io.NodeOutput(out, mask, report)


NODE_CLASS_MAPPINGS = {"KUBA_RegionSampler": KUBA_RegionSampler, "KUBA_SeamPass": KUBA_SeamPass}
NODE_DISPLAY_NAME_MAPPINGS = {"KUBA_RegionSampler": "kubakub region sampler",
                              "KUBA_SeamPass": "kubakub seam pass"}
