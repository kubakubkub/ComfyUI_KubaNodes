"""
strategies.py

Kuba Regions strategies. Each one takes
an adapter, the canvas, the regions and the plan, and returns the new canvas.

run_plan goes through the regions in plan order and dispatches:
S1 inpaint: cut a box around the region with context, scale it uniformly
(exactly, grid / block) to about the model's budget, inpaint the dilated
region with the crop itself as reference, scale back, colour match, paste with
a feathered alpha.
S4 frame_in_frame: generate the region's own scene from scratch, place it
through the region mask, then harmonize a ring along the border.
The canvas updates after every region, so later regions see earlier ones.

No ComfyUI imports: the adapter is passed in, so tests/test_sequential.py can
drive this with a fake adapter and no model.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Callable

import torch

from . import ops
from . import region_cache as rc
from .types import RegionPlan

log = logging.getLogger("KUBA.regions")


@dataclass
class SamplerSettings:
    steps: int = 0                  # 0 = adapter default; a region's own value wins
    cfg: float = 0.0
    sampler_name: str = ""
    scheduler: str = ""
    region_mp: float = 0.0          # 0 = adapter default
    max_upscale: float = 4.0
    seed_offset: int = 0
    style: torch.Tensor | None = None   # IMAGE [1,h,w,3]: second reference of regions with reference = style
    style_mp: float = 1.0               # megapixels the style image is scaled to


@dataclass
class RegionResult:
    region_id: int
    name: str
    status: str                     # done | skipped | empty
    seconds: float = 0.0
    crop: dict = field(default_factory=dict)
    note: str = ""


def _crop_zero_pad(t: torch.Tensor, cp: ops.CropPlan, fill) -> torch.Tensor:
    """Cut the box from [1,H,W] without touching the rest of the map; outside the canvas = fill."""
    H, W = t.shape[-2], t.shape[-1]
    out = torch.full((1, cp.h, cp.w), fill, dtype=t.dtype, device=t.device)
    x0, y0 = max(cp.x0, 0), max(cp.y0, 0)
    x1, y1 = min(cp.x0 + cp.w, W), min(cp.y0 + cp.h, H)
    if x1 > x0 and y1 > y0:
        out[:, y0 - cp.y0:y1 - cp.y0, x0 - cp.x0:x1 - cp.x0] = t[:1, y0:y1, x0:x1]
    return out


def region_masks(rp: RegionPlan, region_id: int, cp: ops.CropPlan):
    """Exact region mask and scope inside the crop box (MASK [1,h,w]); 0 outside the canvas.
    Crops the label map first and compares only the box (the whole map per region cost ~9 ms at 7 MP)."""
    lab = _crop_zero_pad(rp.regions.labels, cp, -1)
    m = (lab == region_id).float()
    if rp.regions.scope is not None:
        scope = _crop_zero_pad(rp.regions.scope, cp, 0).float()
    else:
        scope = _ones_in_canvas(rp.regions.labels, cp)
    return m, scope


def _ones_in_canvas(lab: torch.Tensor, cp: ops.CropPlan) -> torch.Tensor:
    H, W = lab.shape[-2], lab.shape[-1]
    out = torch.zeros((1, cp.h, cp.w))
    x0, y0 = max(cp.x0, 0), max(cp.y0, 0)
    x1, y1 = min(cp.x0 + cp.w, W), min(cp.y0 + cp.h, H)
    if x1 > x0 and y1 > y0:
        out[:, y0 - cp.y0:y1 - cp.y0, x0 - cp.x0:x1 - cp.x0] = 1.0
    return out


def _paste_(canvas: torch.Tensor, patch: torch.Tensor, alpha: torch.Tensor, cp: ops.CropPlan) -> torch.Tensor:
    """ops.paste in place (same arithmetic, no copy of the whole canvas): the caller owns canvas."""
    H, W = canvas.shape[1], canvas.shape[2]
    x0, y0 = max(cp.x0, 0), max(cp.y0, 0)
    x1, y1 = min(cp.x0 + cp.w, W), min(cp.y0 + cp.h, H)
    if x1 <= x0 or y1 <= y0:
        return canvas
    px0, py0 = x0 - cp.x0, y0 - cp.y0
    p = patch[:, py0:py0 + (y1 - y0), px0:px0 + (x1 - x0)]
    a = alpha[:, py0:py0 + (y1 - y0), px0:px0 + (x1 - x0), None]
    region = canvas[:, y0:y1, x0:x1]
    canvas[:, y0:y1, x0:x1] = region * (1 - a) + p.to(region) * a
    return canvas


def _paste_changed_(changed: torch.Tensor, alpha: torch.Tensor, cp: ops.CropPlan) -> torch.Tensor:
    """The changed mask [1,H,W] gets alpha pasted in place (as ops.paste of ones)."""
    ones = torch.ones((alpha.shape[0], alpha.shape[1], alpha.shape[2], 1))
    _paste_(changed[..., None], ones, alpha, cp)
    return changed


def _own(t: torch.Tensor) -> torch.Tensor:
    """t itself, or a compact copy when it is a view into a larger buffer (a cache must not hold that buffer)."""
    if t.is_contiguous() and t.untyped_storage().nbytes() == t.numel() * t.element_size():
        return t
    return t.clone(memory_format=torch.contiguous_format)


def _s1_key(cache, src, inpaint, prompt, negative, denoise, reference, color_match, sampling, cp, extra):
    """Everything an S1 sample depends on besides the adapter (whose identities are in cache.base)."""
    return cache.key("s1", rc.digest(src), rc.digest(inpaint), prompt, negative, float(denoise), reference,
                     color_match, tuple(sorted(sampling.items())), (cp.w, cp.h, cp.work_w, cp.work_h), extra)


def _boxes_overlap(a: ops.CropPlan, b: ops.CropPlan, W: int, H: int) -> bool:
    """Do the on-canvas parts of two boxes overlap (a paste into one can change the other's crop)?"""
    ax0, ay0, ax1, ay1 = max(a.x0, 0), max(a.y0, 0), min(a.x0 + a.w, W), min(a.y0 + a.h, H)
    bx0, by0, bx1, by1 = max(b.x0, 0), max(b.y0, 0), min(b.x0 + b.w, W), min(b.y0 + b.h, H)
    return ax0 < bx1 and bx0 < ax1 and ay0 < by1 and by0 < ay1


class StyleReference:
    """A style image as a second reference next to the crop itself (reference = style): scaled uniformly to
    about mp megapixels and centre-cropped to the model grid; its latent is encoded once, when the first region
    needs it. (The style image alone, without the crop, loses the building at a high denoise.)"""

    def __init__(self, adapter, image: torch.Tensor, mp: float = 1.0):
        img = image[:1, ..., :3].float().cpu()
        g = adapter.grid
        s = math.sqrt(mp * 1024 * 1024 / (img.shape[1] * img.shape[2]))
        w = max(g, int(img.shape[2] * s) // g * g)
        h = max(g, int(img.shape[1] * s) // g * g)
        self.pixels = cover(img, w, h)
        self.digest = rc.digest(self.pixels)
        self._adapter, self._latent = adapter, None

    @property
    def latent(self) -> torch.Tensor:
        if self._latent is None:
            self._latent = self._adapter.encode(self.pixels)
        return self._latent


def _entry_sampling(e: dict, settings: SamplerSettings) -> dict:
    return dict(seed=int(e["seed"]) + settings.seed_offset, steps=int(e["steps"] or settings.steps),
                cfg=float(e["cfg"] or settings.cfg), sampler_name=settings.sampler_name,
                scheduler=settings.scheduler)


def _inpaint_crop(adapter, canvas, changed, cp: ops.CropPlan, inpaint, alpha, prompt, negative,
                  denoise, reference, color_match, sampling, where="", key=None, cache=None, cache_extra=None,
                  on_miss=None, style=None):
    """
    One masked sample of the box cp: returns (canvas, changed). Masks are at target scale in the box.
    canvas and changed are updated in place (the callers own their copies). With a cache
    (region_cache.RunCache) the generated patch is looked up by everything it depends on: the crop's
    actual pixels, the inpaint mask, prompts, denoise, reference, colour match, sampling, the box and work
    size, cache_extra (e.g. the hash of the image a prefetched conditioning was encoded from) and the
    adapter's model identities; a hit pastes the stored patch without touching the model. on_miss runs
    before the model is used (run_plan encodes a region it did not prefetch).
    """
    src = ops.crop(canvas, cp)                                   # target scale
    ckey = None
    if cache is not None:
        ckey = _s1_key(cache, src, inpaint, prompt, negative, denoise, reference, color_match, sampling, cp,
                       cache_extra)
        gen = cache.get(ckey)
        if gen is not None:
            _paste_(canvas, gen, alpha, cp)
            _paste_changed_(changed, alpha, cp)
            return canvas, changed
    if on_miss is not None:
        on_miss()
    work_img = ops.resize(src, cp.work_w, cp.work_h)             # exact uniform scale
    work_mask = ops.resize_mask(inpaint, cp.work_w, cp.work_h)
    latent = adapter.encode(work_img)
    if reference == "self" or reference.startswith("style"):
        if getattr(adapter, "prefetches", False):          # its conditioning was encoded up front (run_plan)
            pos, neg = adapter.conds_for(prompt, negative, work_img, latent, key=key)
        else:
            pos, neg = adapter.conds_for(prompt, negative, work_img, latent)
        if reference.startswith("style") and style is not None:
            pos = adapter.add_reference(pos, style.latent)       # crop first, style image second
    else:
        pos, neg = adapter.conds_for(prompt, negative)
    out = adapter.sample(latent, pos, neg, noise_mask=work_mask[:, None], denoise=float(denoise),
                         pixel_size=(cp.work_w, cp.work_h), **sampling)
    gen = adapter.decode(out)[:1, ..., :3].float().cpu()
    if tuple(gen.shape[1:3]) != (cp.work_h, cp.work_w):
        raise AssertionError(f"{where}: decoded {gen.shape[2]}x{gen.shape[1]}, expected "
                             f"{cp.work_w}x{cp.work_h}. Registration would break.")
    gen = ops.resize(gen, cp.w, cp.h)                            # exact inverse scale
    gen = ops.color_match(gen, src, inpaint, color_match)
    if ckey is not None:
        cache.put(ckey, _own(gen))
    _paste_(canvas, gen, alpha, cp)
    _paste_changed_(changed, alpha, cp)
    return canvas, changed


def fif_size(bbox, budget_px: float, grid: int, fif_w: int = 0, fif_h: int = 0) -> tuple[int, int]:
    """Sub-generation size: fif_width/height if given, else the region box aspect at the budget, on the grid."""
    if fif_w > 0 and fif_h > 0:
        return max(grid, fif_w // grid * grid), max(grid, fif_h // grid * grid)
    w, h = max(1, bbox[2]), max(1, bbox[3])
    a = w / h
    gw = max(grid, round(math.sqrt(budget_px * a) / grid) * grid)
    gh = max(grid, round(math.sqrt(budget_px / a) / grid) * grid)
    return gw, gh


def cover(img: torch.Tensor, w: int, h: int) -> torch.Tensor:
    """Scale uniformly so the image covers w x h, then centre-crop to exactly w x h."""
    ih, iw = img.shape[1], img.shape[2]
    s = max(w / iw, h / ih)
    nw, nh = max(w, math.ceil(iw * s - 1e-6)), max(h, math.ceil(ih * s - 1e-6))
    r = ops.resize(img, nw, nh)
    x0, y0 = (nw - w) // 2, (nh - h) // 2
    return r[:, y0:y0 + h, x0:x0 + w]


def frame_in_frame(adapter, canvas, changed, rp: RegionPlan, rid: int, e: dict,
                   settings: SamplerSettings, region_mp: float, cache=None):
    """
    S4: the region gets its own scene, generated from scratch with its prompt at
    about region_mp in the region's aspect, scaled uniformly to cover the region
    box and placed through the exact region mask (inside the scope). Then an
    optional harmonizing inpaint over a ring along the region border
    (fif_border_px wide, denoise fif_harmonize) ties the scene into its frame.
    The scene itself has no registration; its mask does.
    """
    H, W = canvas.shape[1], canvas.shape[2]
    x, y, bw, bh = (int(v) for v in e["bbox"])
    box = ops.CropPlan(x, y, bw, bh, 1, 1, bw, bh)
    region, scope = region_masks(rp, rid, box)
    if float(region.sum()) == 0:
        return canvas, changed, None
    mp = e.get("region_mp") or region_mp
    gw, gh = fif_size(e["bbox"], mp * 1024 * 1024, adapter.grid, int(e.get("fif_width", 0)),
                      int(e.get("fif_height", 0)))
    sampling = _entry_sampling(e, settings)
    # the scene depends only on the prompts, its size and sampling (not on the canvas)
    skey = scene = None
    if cache is not None:
        skey = cache.key("s4", e["prompt"], e["negative"], gw, gh, tuple(sorted(sampling.items())))
        scene = cache.get(skey)
    if scene is None:
        latent = adapter.empty_latent(gw, gh)
        pos, neg = adapter.conds_for(e["prompt"], e["negative"])
        out = adapter.sample(latent, pos, neg, denoise=1.0, pixel_size=(gw, gh), **sampling)
        scene = adapter.decode(out)[:1, ..., :3].float().cpu()
        if cache is not None:
            cache.put(skey, _own(scene))
    scene = cover(scene, bw, bh)
    feather = int(e["feather_px"])
    alpha = ops.feather_alpha(ops.dilate(region, -(feather // 2)), feather, limit=region) * scope
    _paste_(canvas, scene, alpha, box)
    _paste_changed_(changed, alpha, box)
    info = {"scene": [gw, gh], "box": [x, y, bw, bh]}

    harm, border = float(e.get("fif_harmonize", 0.0)), int(e.get("fif_border_px", 24))
    if harm > 0 and border > 0:
        cp = ops.plan_crop(e["bbox"], max(int(e["context_px"]), border), W, H, grid=adapter.grid,
                           budget_px=mp * 1024 * 1024, max_upscale=settings.max_upscale)
        reg, sc = region_masks(rp, rid, cp)
        half = max(1, border // 2)
        ring = (ops.dilate(reg, half) - ops.dilate(reg, -half)).clamp(0, 1) * sc
        alpha_r = ops.gaussian_blur(ring, half / 2.0).clamp(0, 1) * ring
        canvas, changed = _inpaint_crop(adapter, canvas, changed, cp, ring, alpha_r, e["prompt"],
                                        e["negative"], harm, "self", "none", sampling,
                                        where=f"region {rid} harmonize", cache=cache)
        info["harmonize"] = {"denoise": harm, "border_px": border, **cp.to_dict()}
    return canvas, changed, info


def _predicted_misses(cache, todo, entries, crop_plan, s1_masks, s1_args, prefetchable, pre_digest, canvas,
                      adapter, settings, region_mp, W, H) -> list[int]:
    """
    Which prefetchable regions will miss the result cache: a dry run of the plan on a copy of the canvas
    that pastes the cached hits. A region whose box overlaps a box that is sampled anew (a miss, or any
    frame_in_frame region) cannot be predicted and counts as a miss. So an unchanged plan encodes no crop,
    and one changed prompt encodes only the regions it can affect.
    """
    sim = canvas.clone()
    dirty: list[ops.CropPlan] = []
    want = []
    for rid in todo:
        e = entries[rid]
        if e["strategy"] == "frame_in_frame":
            x, y, bw, bh = (int(v) for v in e["bbox"])
            dirty.append(ops.CropPlan(x, y, bw, bh, 1, 1, bw, bh))
            border = int(e.get("fif_border_px", 24))
            if float(e.get("fif_harmonize", 0.0)) > 0 and border > 0:
                mp = e.get("region_mp") or region_mp
                dirty.append(ops.plan_crop(e["bbox"], max(int(e["context_px"]), border), W, H, grid=adapter.grid,
                                           budget_px=mp * 1024 * 1024, max_upscale=settings.max_upscale))
            continue
        cp = crop_plan(e)
        masks = s1_masks(rid, e, cp)
        if masks is None:
            continue
        hit = None
        if not any(_boxes_overlap(cp, d, W, H) for d in dirty):
            key = _s1_key(cache, ops.crop(sim, cp), masks[0], *s1_args(e), cp, pre_digest.get(rid))
            hit = cache.store.get(key)
        if hit is None:
            dirty.append(cp)
            if prefetchable(e):
                want.append(rid)
        else:
            _paste_(sim, hit, masks[1], cp)
    return want


def run_plan(adapter, canvas: torch.Tensor, rp: RegionPlan, settings: SamplerSettings,
             only: Callable[[dict], bool] | None = None,
             progress: Callable[[int, int], None] | None = None,
             check_interrupt: Callable[[], None] | None = None,
             use_cache: bool = True):
    """
    Run every region of the plan in plan order with its strategy: inpaint (S1)
    or frame_in_frame (S4); keep is skipped. Returns (canvas, changed_mask
    [1,H,W], results). canvas is IMAGE [1,H,W,C] at the regions' size.
    With use_cache (and an adapter with a model and vae) every region's result
    is kept across queues (region_cache.RESULTS): an unchanged region whose
    crop pixels are unchanged is pasted from the cache instead of sampled.
    """
    H, W = canvas.shape[1], canvas.shape[2]
    rw, rh = rp.regions.size
    if (rw, rh) != (W, H):
        raise ValueError(f"The image is {W}x{H} but the regions are {rw}x{rh}; they must match "
                         f"(the canvas is never resized to fit).")
    canvas = canvas[:1, ..., :3].float().cpu().clone()
    changed = torch.zeros((1, H, W))
    entries = rp.plan["regions"]
    todo = [i for i in rp.order if entries[i]["strategy"] in ("inpaint", "frame_in_frame")
            and (only is None or only(entries[i]))]

    region_mp = settings.region_mp or getattr(adapter, "default_region_mp", 1.0)
    adapter.encode_prompts([entries[i]["prompt"] for i in todo] + [entries[i]["negative"] for i in todo])
    cache = rc.run_cache(adapter, use_cache)
    prefetches = bool(getattr(adapter, "prefetches", False))

    def crop_plan(e):
        mp = e.get("region_mp") or region_mp
        return ops.plan_crop(e["bbox"], int(e["context_px"]), W, H, grid=adapter.grid,
                             budget_px=mp * 1024 * 1024, max_upscale=settings.max_upscale)

    def s1_masks(rid, e, cp):
        """(inpaint, alpha) of an S1 region in its box, or None when it has no pixels inside the scope."""
        region, scope = region_masks(rp, rid, cp)
        if float(region.sum()) == 0:
            return None
        inpaint = ops.dilate(region, int(e["dilate_px"])) * scope
        return inpaint, ops.feather_alpha(region, int(e["feather_px"]), limit=inpaint)

    style = StyleReference(adapter, settings.style, settings.style_mp) if settings.style is not None else None

    def reference_of(e):
        """The region's reference; 'style' carries the style image's hash (it is part of the cache key) and
        falls back to self when no style image is connected."""
        ref = e.get("reference", "self")
        if ref == "style":
            return f"style:{style.digest}" if style is not None else "self"
        return ref

    def s1_args(e):
        return (e["prompt"], e["negative"], e["denoise"], reference_of(e),
                e.get("color_match", "mean_std"), _entry_sampling(e, settings))

    def prefetchable(e):
        return prefetches and e["strategy"] == "inpaint" and reference_of(e) != "none"

    # adapters that encode the crop image with the text (Qwen Image 2.1): encode every region first, so the
    # text encoder loads once instead of swapping with the DiT for each region. The encoder sees the crops
    # as they are before any region is painted (they only differ where regions overlap).
    canvas0 = canvas.clone() if prefetches else None          # the canvas is painted in place
    pre_digest: dict[int, str] = {}
    prefetched: set[int] = set()

    def pre_image(cp):
        return ops.resize(ops.crop(canvas0, cp), cp.work_w, cp.work_h)

    def need_negative(e):
        return (_entry_sampling(e, settings)["cfg"] or getattr(adapter, "default_cfg", 1.0)) > 1.0

    if prefetches:
        t0 = time.perf_counter()
        want = [rid for rid in todo if prefetchable(entries[rid])]
        if cache is not None:
            for rid in want:
                pre_digest[rid] = rc.digest(pre_image(crop_plan(entries[rid])))
            want = _predicted_misses(cache, todo, entries, crop_plan, s1_masks, s1_args, prefetchable, pre_digest,
                                     canvas, adapter, settings, region_mp, W, H)
        for rid in want:
            e = entries[rid]
            adapter.prefetch(rid, e["prompt"], e["negative"], pre_image(crop_plan(e)), need_negative=need_negative(e))
            prefetched.add(rid)
        if want:
            log.info("[KUBA regions] encoded %d region(s) with their images first, %.1fs", len(want),
                     time.perf_counter() - t0)

    results: list[RegionResult] = []
    for n, rid in enumerate(todo):
        if check_interrupt:
            check_interrupt()
        e = entries[rid]
        t0 = time.perf_counter()
        hits0, misses0 = (cache.hits, cache.misses) if cache is not None else (0, 0)
        if e["strategy"] == "frame_in_frame":
            canvas, changed, info = frame_in_frame(adapter, canvas, changed, rp, rid, e, settings, region_mp,
                                                   cache=cache)
            if info is None:
                results.append(RegionResult(rid, e["name"], "empty", note="no pixels inside the scope"))
                continue
            dt = time.perf_counter() - t0
            cached = cache is not None and cache.hits > hits0 and cache.misses == misses0
            results.append(RegionResult(rid, e["name"], "done", dt, {}, note=f"fif scene {info['scene']}"
                                        + (" + harmonize" if "harmonize" in info else "")
                                        + (" (cached)" if cached else "")))
            log.info("[KUBA regions] S4 %d/%d region %d %s: scene %s -> box %s, %.1fs%s",
                     n + 1, len(todo), rid, e["name"], info["scene"], info["box"], dt, " cached" if cached else "")
        else:
            cp = crop_plan(e)
            masks = s1_masks(rid, e, cp)
            if masks is None:
                results.append(RegionResult(rid, e["name"], "empty", note="no pixels inside the scope"))
                continue
            inpaint, alpha = masks
            on_miss = None
            if prefetchable(e) and rid not in prefetched:
                # predicted as a cache hit but missed (e.g. evicted meanwhile): encode it from the same image
                # the up-front pass would have used, so the result equals a fresh run
                on_miss = (lambda e=e, cp=cp, rid=rid: adapter.prefetch(
                    rid, e["prompt"], e["negative"], pre_image(cp), need_negative=need_negative(e)))
            canvas, changed = _inpaint_crop(adapter, canvas, changed, cp, inpaint, alpha, *s1_args(e),
                                            where=f"region {rid}", key=rid, cache=cache,
                                            cache_extra=pre_digest.get(rid), on_miss=on_miss, style=style)
            dt = time.perf_counter() - t0
            cached = cache is not None and cache.hits > hits0 and cache.misses == misses0
            results.append(RegionResult(rid, e["name"], "done", dt, cp.to_dict(), note="cached" if cached else ""))
            log.info("[KUBA regions] S1 %d/%d region %d %s: box %s x%s -> %dx%d, %.1fs%s",
                     n + 1, len(todo), rid, e["name"], cp.to_dict()["box"], cp.scale, cp.work_w, cp.work_h, dt,
                     " cached" if cached else "")
        if progress:
            progress(n + 1, len(todo))

    for e in entries:
        if e["strategy"] == "keep":
            results.append(RegionResult(e["region_id"], e["name"], "skipped", note="keep"))
    return canvas, changed, results


sequential_inpaint = run_plan   # the M1 step 4 name


def results_report(results: list[RegionResult]) -> str:
    done = [r for r in results if r.status == "done"]
    total = sum(r.seconds for r in done)
    n_cached = sum(1 for r in done if "cached" in r.note)
    lines = [f"{len(done)} regions done in {total:.1f}s"
             + (f" ({total / len(done):.1f}s each)" if done else "")
             + (f", {n_cached} from the cache" if n_cached else "")]
    for r in results:
        box = r.crop.get("box", "")
        extra = f"box {box} x{r.crop.get('scale')} work {r.crop.get('work')}" if r.crop else ""
        lines.append(f"{r.region_id:>3} {r.name[:18]:<18} {r.status:<7} {r.seconds:5.1f}s {extra} {r.note}".rstrip())
    return "\n".join(lines)
