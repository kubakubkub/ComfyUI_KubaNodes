"""
nodes_versions.py

KUBA_Versions (category kubakub/2d/generate): many versions of one facade from
one queue. The versions sheet (versions.py) lists what changes: looks, ideas per
region, prompt rotation, LoRAs, style reference images, seeds, plan settings.
Every version is the region plan plus that version's rules, run by the region
sampler's own code (strategies.run_plan), so the region cache works across
versions and queues. draft = small and fast with a numbered contact sheet,
final = only the picked numbers at full quality with the seam pass.

V3 node schema (comfy_api.latest), registered through the pack's
NODE_CLASS_MAPPINGS loader in ../__init__.py.
"""

from __future__ import annotations

import json
import logging
import os
import time
import weakref
from collections import OrderedDict

import numpy as np
import torch
from PIL import Image, ImageOps
from PIL.PngImagePlugin import PngInfo

from comfy_api.latest import io, ui

from comfy.cli_args import args
import comfy.model_management
import comfy.samplers
import comfy.sd
import comfy.utils
import folder_paths

from ...kubakub import adapters
from ...kubakub import align
from ...kubakub import ops
from ...kubakub import plan as rp
from ...kubakub import region_cache as rc
from ...kubakub import runtime
from ...kubakub import save_paths
from ...kubakub import schedules
from ...kubakub import seams
from ...kubakub import strategies as st
from ...kubakub import versions as vs
from ...kubakub.io_types import PlanType
from ...kubakub.types import RegionPlan

log = logging.getLogger("KUBA.regions")

# auto, every core scheduler, then the fixed schedules of the Qwen turbo LoRAs
SCHEDULERS = ["auto"] + list(comfy.samplers.KSampler.SCHEDULERS) + [s for s in schedules.SCHEDULES if s != "auto"]


def _adapter(model, clip, vae, scheduler: str):
    ad = adapters.make_adapter(model, clip, vae)
    if scheduler in schedules.SCHEDULES:
        ad.set_schedule(scheduler)
    else:
        ad.core_scheduler = scheduler
    return ad

# LoRA-patched models are kept between queues: the region cache is keyed on the model object, so a version
# whose rules did not change is pasted from the cache instead of sampled again.
_PATCHED: OrderedDict = OrderedDict()      # (id(model), identity, lora file, strength) -> (weakref(model), patched)
_PATCHED_MAX = 12


def _with_lora(model, name: str, strength: float):
    if not name or strength == 0:
        return model
    file = vs.match_lora(name, folder_paths.get_filename_list("loras")) if not os.path.isfile(name) else name
    key = (id(model), rc.identity(model), file, float(strength))
    hit = _PATCHED.get(key)
    if hit is not None and hit[0]() is model:
        _PATCHED.move_to_end(key)
        return hit[1]
    path = file if os.path.isfile(file) else folder_paths.get_full_path_or_raise("loras", file)
    lora = comfy.utils.load_torch_file(path, safe_load=True)
    patched, _ = comfy.sd.load_lora_for_models(model, None, lora, strength, 0)
    for k in [k for k, v in _PATCHED.items() if v[0]() is None]:
        del _PATCHED[k]
    _PATCHED[key] = (weakref.ref(model), patched)
    while len(_PATCHED) > _PATCHED_MAX:
        _PATCHED.popitem(last=False)
    log.info("[KUBA regions] versions: lora %s at %g", os.path.basename(file), strength)
    return patched


def _load_reference(path: str) -> torch.Tensor:
    from ...kubakub import imio
    return torch.from_numpy(imio.pil_rgb01(ImageOps.exif_transpose(Image.open(path))))[None]


class KUBA_Versions(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="KUBA_Versions",
            display_name="kubakub versions",
            category="kubakub/2d/generate",
            search_aliases=['variations', 'iterations', 'contact sheet', 'prompt rotation', 'lora compare',
                            'xy plot'],
            description=(
                "Many versions of one facade from one queue. The versions sheet lists what changes, one option "
                "per line: [look] whole-facade looks, [region windows] ideas for some regions, [rotate a | b | c] "
                "ideas that move on by one region per version, [lora] your LoRAs (name : strength : words; a chain with ' + '), "
                "[reference] style images, [image] the picture a version starts from, [set denoise] any plan "
                "setting, [seed]. Every version is the region "
                "plan plus its own rules. quality draft renders small and fast and gives a numbered contact "
                "sheet; write the numbers you like into pick, switch to final, and only those are scaled up to "
                "the full canvas and resampled tile by tile for full-size detail: a final is its draft, sharper. "
                "Versions that did not change come from the region cache."),
            inputs=[
                io.Model.Input("model", tooltip="The image model (Flux 2 Klein or Qwen Image 2.1). LoRAs of the "
                                                "sheet are added to it per version."),
                io.Clip.Input("clip", tooltip="The text encoder that matches the model."),
                io.Vae.Input("vae", tooltip="The VAE that matches the model."),
                PlanType.Input("plan", tooltip="From kubakub region plan: what every version has in common "
                                               "(the base prompt, denoise, order). Each version adds its rules."),
                io.Image.Input("image", tooltip="The canvas every version starts from, e.g. the clay render, "
                                                "exactly the size of the regions. A batch of several images "
                                                "(clay, wireframe, canny ...): each one is an option of the "
                                                "image list, like [image] in the sheet."),
                io.String.Input("versions", multiline=True, default=vs.EXAMPLE_SHEET, dynamic_prompts=False,
                                tooltip="The versions sheet: lists of options, one per line. "
                                        "See docs/generate.md (kubakub versions)."),
                io.Combo.Input("mode", options=list(vs.MODES), default="rotate",
                               tooltip="rotate: version n takes option n of every list (short lists start "
                                       "over). combine: every combination. one by one: the first option of "
                                       "everything, then one change at a time."),
                io.Combo.Input("quality", options=["draft", "final"], default="draft",
                               tooltip="draft: small and fast, for the contact sheet. final: the same picture "
                                       "at the full canvas size with full-size detail, for the picked versions."),
                io.String.Input("pick", default="", placeholder="3, 7-9",
                                tooltip="Version numbers to render (as on the contact sheet). Empty = all."),
                io.Int.Input("max_versions", advanced=True, default=12, min=1, max=200,
                             tooltip="The sheet never gives more versions than this (combine grows fast)."),
                io.Int.Input("seed", default=0, min=0, max=0xFFFFFFFFFFFFFFFF, control_after_generate=False,
                             tooltip="Added to the plan's seed; a [seed] list in the sheet adds to it per version."),
                io.Float.Input("draft_mp", advanced=True, default=0.5, min=0.1, max=2.0, step=0.05,
                               tooltip="Work megapixels per region of the draft (small = fast)."),
                io.Float.Input("final_denoise", advanced=True, default=0.4, min=0.0, max=1.0, step=0.01,
                               tooltip="final: the draft is scaled up to the full canvas and resampled tile by "
                                       "tile at this denoise. Low stays close to the draft, higher adds more "
                                       "detail and changes more; 0 = the scaled-up draft."),
                io.Int.Input("draft_long_edge", advanced=True, default=1600, min=256, max=8192, step=16,
                             tooltip="The picture is made on the canvas scaled down to this long edge (draft "
                                     "and final); final scales it up again and adds the detail."),
                io.Image.Input("references", optional=True,
                               tooltip="Style reference images (a batch): each one is an option of the "
                                       "reference list, after those written in the sheet."),
                io.Int.Input("count", default=0, min=0, max=200, optional=True, advanced=True,
                             tooltip="rotate: how many versions; 0 = as many as the longest list."),
                io.Int.Input("sheet_columns", default=0, min=0, max=16, optional=True, advanced=True,
                             tooltip="Columns of the contact sheet; 0 = automatic."),
                io.Int.Input("steps", default=0, min=0, max=200, optional=True, advanced=True,
                             tooltip="0 = the model's default (Klein 4, Qwen Image 2.1 25)."),
                io.Combo.Input("sampler_name", options=comfy.samplers.KSampler.SAMPLERS, default="euler",
                               optional=True, advanced=True, tooltip="Sampler for every region."),
                io.Combo.Input("scheduler", advanced=True, options=SCHEDULERS, default="auto", optional=True,
                               tooltip="auto: the model's own schedule (Flux 2: core's Flux2Scheduler for the "
                                       "crop size; Qwen Image 2.1: simple). simple, beta, sgm_uniform ...: that "
                                       "core scheduler, as in KSampler. qwen21_turbo_5 / pruna_qwen21_8: the "
                                       "trained steps of the Qwen turbo LoRAs."),
                # new widgets go last: saved workflows store the widget values by position
                io.String.Input("picks_folder", default="", optional=True,
                                tooltip="final: a folder with the drafts you like (copy them there from the save "
                                        "folder). Every image in it is scaled up to the full canvas and gets its "
                                        "full-size detail, with the look, LoRAs and style image it was made with; "
                                        "the sheet and pick are not used. Empty = the numbers in pick."),
                io.String.Input("save_folder", advanced=True, default="kubakub/versions", optional=True,
                                tooltip="Where the versions are saved (inside ComfyUI's output folder, or a full "
                                        "path): drafts as r<run>_v<number>_<what it is made of>.png, finals in the "
                                        "subfolder 'final'. Each file knows its version, so it can go into a "
                                        "picks folder. Empty = nothing is saved."),
                io.Float.Input("unify", default=0.0, min=0.0, max=1.0, step=0.05, optional=True,
                               tooltip="One more pass over the whole picture after the regions, at this denoise: "
                                       "it ties regions with different prompts into one structure. 0 = off; "
                                       "0.3 keeps every region's idea, 0.5 and more lets them grow into each "
                                       "other. Costs about one more sample per version."),
                io.Combo.Input("generate", options=["regions", "whole picture"], default="regions", optional=True,
                               tooltip="regions: every region is repainted inside its own mask (the facade keeps "
                                       "its elements). whole picture: one free sample of the whole picture from "
                                       "nothing, with the image only as reference image 1 and the style image as "
                                       "image 2, no masks: forms may grow across the facade and out of it. "
                                       "Per-region rules then only add their words to the one prompt."),
            ],
            outputs=[
                io.Image.Output("images", tooltip="One image per rendered version, in number order."),
                io.Image.Output("sheet", tooltip="The contact sheet: every rendered version with its number "
                                                 "and what it is made of."),
                io.String.Output("report", tooltip="The lists, every version with its number, and each "
                                                   "rendered version's rules and time."),
            ],
            hidden=[io.Hidden.prompt, io.Hidden.extra_pnginfo],
            is_output_node=True,        # the saved versions are the product: runs with its outputs unconnected
        )

    @classmethod
    def execute(cls, model, clip, vae, plan, image, versions, mode, quality, pick, max_versions, seed,
                draft_mp, final_denoise, draft_long_edge, references=None, count=0, sheet_columns=0,
                steps=0, sampler_name="euler", scheduler="auto", picks_folder="",
                save_folder="kubakub/versions", unify=0.0, generate="regions") -> io.NodeOutput:
        base_rules = plan.plan.get("rules_text")
        if base_rules is None:
            raise ValueError("kubakub versions: this plan has no rules text; connect kubakub region plan.")
        n_refs = int(references.shape[0]) if references is not None else 0
        n_imgs = int(image.shape[0])
        final = quality == "final"
        plan_seed0 = int(plan.plan.get("seed", 0))
        picks = (picks_folder or "").strip().strip('"') if final else ""
        # per version number: the rules every version shares, the seed, what goes into the saved file,
        # and (picks folder) the draft file the final is made of
        base_of, seed_of, recipes, drafts, unknown = {}, {}, {}, {}, []
        free = set()            # version numbers made as a whole picture (no masks)
        try:
            if picks:
                all_versions, axes, left_out, picked = [], [], 0, []
                for n, path in enumerate(vs.pick_files(picks), 1):
                    stem = os.path.splitext(os.path.basename(path))[0]
                    r = vs.read_recipe(path, folder_paths.get_output_directory())
                    if r is None:           # not a draft of this node: the plan as it is, no LoRA, no style image
                        unknown.append(stem)
                        r = vs.recipe(vs.Version(n, stem, ""), None, None, int(seed))
                    all_versions.append(vs.Version(n, stem, r["rules"], reference=r["reference"],
                                                   image=r.get("image", ""),
                                                   chain=tuple((a, float(s)) for a, s in r["chain"])))
                    base_of[n] = base_rules if r["base_rules"] is None else r["base_rules"]
                    seed_of[n] = (plan_seed0 if r["plan_seed"] is None else int(r["plan_seed"])) + int(r["offset"])
                    recipes[n], drafts[n] = r, path
                    if r.get("whole"):
                        free.add(n)
                    picked.append(n)
            else:
                all_versions, axes, left_out = vs.build(versions, mode, int(count), int(max_versions), n_refs, n_imgs)
                picked = vs.parse_pick(pick, len(all_versions))
                for v in all_versions:
                    base_of[v.number], seed_of[v.number] = base_rules, plan_seed0 + int(seed) + v.seed
                    recipes[v.number] = vs.recipe(v, base_rules, plan_seed0, int(seed) + v.seed)
                    if generate == "whole picture":
                        recipes[v.number]["whole"] = True
                        free.add(v.number)
        except vs.SheetError as e:
            raise ValueError(f"kubakub versions: {e}") from None
        todo = [v for v in all_versions if v.number in picked]

        H, W = image.shape[1], image.shape[2]
        if plan.regions.size != (W, H):
            raise ValueError(f"kubakub versions: the image is {W}x{H} but the regions are "
                             f"{plan.regions.size[0]}x{plan.regions.size[1]}; they must match.")
        # the picture is always made on a small canvas (crops, masks and pasting at full size cost as much as
        # the sampling, and regions that span the facade gain nothing from it); final then scales that draft
        # up and resamples it tile by tile at 1:1, so a final is its draft with full-size detail
        full_regions = plan.regions
        s = min(1.0, draft_long_edge / max(W, H))
        dw, dh = max(8, round(W * s)), max(8, round(H * s))
        regions = plan.regions.resized(dw, dh)
        ow, oh = (W, H) if final else (dw, dh)

        starts: dict[str, tuple] = {}

        def start_of(v):
            """(full size, draft size) of the image the version starts from: one of the input batch or a file."""
            key = v.image or "input:0"
            if key not in starts:
                i = int(key[6:]) if key.startswith("input:") else 0
                if not key.startswith("input:") and os.path.isfile(key):
                    full = _load_reference(key)
                    fh, fw = full.shape[1], full.shape[2]
                    if abs(fw / fh - W / H) > 0.02 * W / H:
                        raise ValueError(f"kubakub versions: the image {key} is {fw}x{fh}, another shape than "
                                         f"the regions ({W}x{H}); it must show the same view.")
                    if (fw, fh) != (W, H):
                        full = ops.resize(full, W, H)
                else:                       # a pick whose image is gone, or a smaller batch: the first input
                    full = image[i:i + 1, ..., :3] if i < n_imgs else image[:1, ..., :3]
                full = full.float().cpu()
                starts[key] = (full, ops.resize(full, dw, dh))
            return starts[key]

        # resolve every version's plan first: a typo in the sheet stops the queue before any sampling
        plans, full_plans = {}, {}
        for v in todo:
            rules = base_of[v.number].rstrip("\n") + "\n" + v.rules
            plan_seed = seed_of[v.number]
            start_of(v)                     # an image of another shape stops the queue here, too
            try:
                if final:
                    full_plans[v.number] = RegionPlan(rp.resolve(full_regions.table, rules, seed=plan_seed),
                                                      full_regions)
                if v.number in drafts:      # the picture is already there
                    continue
                d = rp.resolve(regions.table, rules, seed=plan_seed)
            except rp.PlanError as e:
                raise ValueError(f"kubakub versions: version {v.number} ({v.label}): {e} (line numbers count "
                                 f"from the start of the plan's rules, then this version's rules)") from None
            # regions with the same prompt and settings are sampled together (one sample, not seven)
            plans[v.number] = RegionPlan(*vs.merge_same(d, regions))
        models = {}
        for v in todo:
            m = model
            for name, strength in v.chain:          # a chain of LoRAs, one on top of the other
                m = _with_lora(m, name, strength)
            models[v.number] = m

        settings = dict(steps=steps, cfg=0.0, sampler_name=sampler_name, scheduler="simple",
                        region_mp=draft_mp, max_upscale=4.0, style_mp=0.5)

        # every prompt of every version through the text encoder first, so it loads once
        ad0 = adapters.make_adapter(model, clip, vae)
        texts = []
        for v in todo:
            if v.number in drafts:
                continue
            entries = plans[v.number].plan["regions"]
            texts += [entries[i][k] for i in plans[v.number].order for k in ("prompt", "negative")]
        ad0.encode_prompts(texts)

        style_cache: dict[str, torch.Tensor | None] = {}

        def style_of(v):
            if not v.reference:
                return None
            if v.reference not in style_cache:
                if v.reference.startswith("input:"):
                    i = int(v.reference[6:])
                    style_cache[v.reference] = references[i][None] if i < n_refs else None
                elif v.number in drafts and not os.path.isfile(v.reference):
                    style_cache[v.reference] = None     # a pick whose style image is gone: without it
                else:
                    style_cache[v.reference] = _load_reference(v.reference)
            return style_cache[v.reference]

        # versions of one LoRA after each other: the model is patched once per LoRA
        run_order = sorted(todo, key=lambda v: (str(v.chain).lower(), v.number))
        pbar = comfy.utils.ProgressBar(len(run_order))
        outs, lines = {}, []
        t_all = time.perf_counter()
        for n, v in enumerate(run_order):
            comfy.model_management.throw_exception_if_processing_interrupted()
            t0 = time.perf_counter()
            ad = _adapter(models[v.number], clip, vae, scheduler)
            if v.number in drafts:
                canvas, results = _load_reference(drafts[v.number]), []
            elif v.number in free:
                canvas, results = cls._whole(ad, plans[v.number], start_of(v)[1], style_of(v), draft_mp,
                                             st.SamplerSettings(**settings), dw, dh), []
            else:
                canvas, changed, results = st.run_plan(
                    ad, start_of(v)[1], plans[v.number], st.SamplerSettings(style=style_of(v), **settings),
                    check_interrupt=comfy.model_management.throw_exception_if_processing_interrupted)
            note = ", whole picture" if v.number in free else ""
            if unify > 0 and v.number not in drafts:
                # the whole draft as one tile, with the prompts of its largest regions and itself as reference
                canvas, _ = seams.refine_tiles(
                    ad, canvas, plans[v.number], denoise=unify, whole=True,
                    seed=int(plans[v.number].plan["seed"]), steps=steps, sampler_name=sampler_name,
                    scheduler="simple", cache=rc.run_cache(ad),
                    style=st.StyleReference(ad, style_of(v), 1.0) if style_of(v) is not None else None,
                    check_interrupt=comfy.model_management.throw_exception_if_processing_interrupted)
                note += f", unify {unify:g}"
            if final:
                up = ops.resize(canvas, W, H, "bicubic")
                keep = (full_regions.labels[:1] < 0)[..., None]          # outside the regions: the input image
                if v.number not in free:                                # a whole picture keeps its own background
                    up = torch.where(keep, start_of(v)[0], up)
                style = st.StyleReference(ad, style_of(v), 1.0) if style_of(v) is not None else None
                tiles = comfy.utils.ProgressBar(1)
                canvas, tile_lines = seams.refine_tiles(
                    ad, up, full_plans[v.number], denoise=final_denoise, tile_px=1024,
                    seed=int(full_plans[v.number].plan["seed"]), steps=steps, sampler_name=sampler_name,
                    scheduler="simple",
                    style=style, cache=rc.run_cache(ad),
                    progress=lambda i, n: tiles.update_absolute(i, n),
                    check_interrupt=comfy.model_management.throw_exception_if_processing_interrupted)
                note += ", " + tile_lines[0]
            outs[v.number] = canvas
            runtime.between_runs()
            done = [r for r in results if r.status == "done"]
            cached = sum(1 for r in done if "cached" in r.note)
            dt = time.perf_counter() - t0
            lines.append((v.number, f"{v.number:02d}  {v.label}: {len(done)} regions"
                          + (f" ({cached} cached)" if cached else "") + f", {dt:.1f}s{note}"))
            log.info("[KUBA regions] versions %d/%d: %s", n + 1, len(run_order), lines[-1][1])
            pbar.update_absolute(n + 1, len(run_order))

        numbers = sorted(outs)
        images = torch.cat([outs[k] for k in numbers], dim=0)
        by_number = {v.number: v for v in todo}
        labels = [f"{k:02d}  {by_number[k].label}" for k in numbers]
        sheet = torch.from_numpy(vs.contact_sheet([outs[k][0].numpy() for k in numbers], labels,
                                                  int(sheet_columns)))[None]
        if picks:
            report = [f"{len(todo)} picked draft(s) from {picks}"]
            if unknown:
                report += ["", "these files do not say which version they are, so their final uses the plan as it "
                               "is (no look, no LoRA, no style image): " + ", ".join(unknown)]
        else:
            report = [vs.report(all_versions, axes, mode, left_out, picked if pick.strip() else None)]
        report += ["", f"{quality}: {len(numbers)} version(s) at {ow}x{oh} in {time.perf_counter() - t_all:.1f}s; "
                       f"cache {rc.RESULTS.stats()}"]
        report += [text for _, text in sorted(lines)]
        saved = cls._save({k: outs[k] for k in numbers}, by_number, recipes, save_folder, final, bool(picks))
        if saved:
            report += ["", f"saved {len(saved)} file(s) to {os.path.dirname(saved[0])}"]
        report.append("")
        for k in numbers:
            report.append(f"--- version {k:02d} adds to the plan"
                          + (" (lora " + " + ".join(f"{n} {s:g}" for n, s in by_number[k].chain) + ")"
                             if by_number[k].chain else "")
                          + (f" (reference {os.path.basename(by_number[k].reference)})"
                             if by_number[k].reference else "")
                          + (f" (image {os.path.basename(by_number[k].image)})"
                             if by_number[k].image not in ("", "input:0") else "") + ":")
            report.append(by_number[k].rules.rstrip() or "(nothing)")
        return io.NodeOutput(images, sheet, "\n".join(report), ui=ui.PreviewImage(sheet, cls=cls))

    @staticmethod
    def _whole(ad, rplan, start, style, mp: float, settings, dw: int, dh: int) -> torch.Tensor:
        """One free sample of the whole picture: an empty latent of about mp megapixels in the canvas shape, the
        starting image (1 MP) as reference image 1 and the style image (1 MP) as image 2, no mask. The prompt is
        that of the largest regions. Returns the picture at dw x dh."""
        entries = rplan.plan["regions"]
        lab = rplan.regions.labels[:1]
        prompt, negative, _ = seams.tile_prompt(entries, lab, (lab >= 0).float())
        e = entries[rplan.order[0]] if rplan.order else entries[0]
        first = st.StyleReference(ad, start, 1.0)
        pos, neg = ad.conds_for(prompt, negative, first.pixels, first.latent)
        if style is not None:
            pos = ad.add_reference(pos, st.StyleReference(ad, style, 1.0).latent)
        gw, gh = st.fif_size([0, 0, dw, dh], mp * 1024 * 1024, ad.grid)
        out = ad.sample(ad.empty_latent(gw, gh), pos, neg, denoise=1.0, pixel_size=(gw, gh),
                        **st._entry_sampling(e, settings))
        pic = ops.resize(ad.decode(out)[:1, ..., :3].float().cpu(), dw, dh)
        try:        # a free sample comes back a few pixels moved or scaled: back onto the starting image (align.py)
            src = ops.resize(start[:1, ..., :3].float().cpu(), dw, dh)[0].numpy()
            fixed, r = align.align(src, pic[0].numpy())
            log.info("[KUBA versions] whole picture, registration: %s", align.describe(r))
            if r["ok"] and r["why"] == "aligned":
                pic = torch.from_numpy(np.ascontiguousarray(fixed, dtype=np.float32))[None]
        except Exception as e:  # noqa: BLE001  (the picture is worth more than its registration)
            log.warning("[KUBA versions] whole picture: registration skipped (%s)", e)
        return pic

    @classmethod
    def _save(cls, outs: dict, by_number: dict, recipes: dict, save_folder: str, final: bool,
              from_picks: bool) -> list[str]:
        """Write every version as a PNG that holds its recipe (and the workflow, as Save Image does)."""
        folder = save_paths.save_folder(save_folder, folder_paths.get_output_directory())
        if not folder:
            return []
        drafts_dir = folder
        if final:
            folder = os.path.join(folder, "final")
        os.makedirs(folder, exist_ok=True)
        run = vs.next_run(os.listdir(drafts_dir))
        hid = getattr(cls, "hidden", None)
        paths = []
        for k, img in outs.items():
            v = by_number[k]
            # a pick keeps the name of its draft file
            stem = v.label if from_picks else f"r{run:02d}_v{k:02d}_{vs.slug(v.label)}"
            path, n = os.path.join(folder, stem + ("_final" if final else "") + ".png"), 1
            while os.path.exists(path):
                n += 1
                path = os.path.join(folder, f"{stem}_final_{n}.png" if final else f"{stem}_{n}.png")
            meta = PngInfo()
            meta.add_text(vs.RECIPE_KEY, json.dumps(recipes[k]))
            if not args.disable_metadata:
                if getattr(hid, "prompt", None) is not None:
                    meta.add_text("prompt", json.dumps(hid.prompt))
                for name, value in (getattr(hid, "extra_pnginfo", None) or {}).items():
                    meta.add_text(name, json.dumps(value))
            arr = (img[0].clamp(0, 1) * 255 + 0.5).to(torch.uint8).cpu().numpy()
            Image.fromarray(arr).save(path, pnginfo=meta, compress_level=4)
            paths.append(path)
        return paths


NODE_CLASS_MAPPINGS = {"KUBA_Versions": KUBA_Versions}
NODE_DISPLAY_NAME_MAPPINGS = {"KUBA_Versions": "kubakub versions"}
