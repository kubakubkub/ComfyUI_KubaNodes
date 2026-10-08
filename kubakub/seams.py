"""
seams.py

The seam pass of Kuba Regions: a soft band
along the region borders that changed, sampled tile by tile at full resolution
with a low denoise, so neighbouring regions grow together. Pure torch, no
ComfyUI imports except in differential_diffusion_patch (tests/test_seams.py).
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass

import torch

from . import ops

log = logging.getLogger("KUBA.regions")


# --------------------------------------------------------------------------
# seam mask
# --------------------------------------------------------------------------

def border_pixels(labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Pixels on a border between two regions (4-neighbourhood, both labels >= 0).
    Returns (border [H,W] bool, a [H,W], b [H,W]) where a/b are the labels on the
    two sides (-1 where there is no border).
    """
    lab = labels
    H, W = lab.shape
    border = torch.zeros((H, W), dtype=torch.bool)
    a = torch.full((H, W), -1, dtype=lab.dtype)
    b = torch.full((H, W), -1, dtype=lab.dtype)
    for dy, dx in ((0, 1), (1, 0)):
        p = lab[:H - dy, :W - dx]
        q = lab[dy:, dx:]
        m = (p != q) & (p >= 0) & (q >= 0)
        for sl, other in (((slice(0, H - dy), slice(0, W - dx)), q), ((slice(dy, H), slice(dx, W)), p)):
            border[sl] |= m
            a_sl, b_sl = a[sl], b[sl]
            own = lab[sl]
            a_sl[m] = own[m]
            b_sl[m] = other[m]
    return border, a, b


def seam_mask(labels: torch.Tensor, seam_px: int, changed: torch.Tensor | None = None,
              blend: torch.Tensor | None = None, keep: torch.Tensor | None = None,
              scope: torch.Tensor | None = None) -> torch.Tensor:
    """
    Soft seam band [1,H,W] in 0..1, 1 on the border fading to 0 at seam_px/2 on
    each side. A border counts when both regions have blend on (bool per region
    id), and at least one side changed (changed MASK > 0 near the border, if
    given). keep regions (bool per region id) and pixels outside the scope are
    never part of the band.
    """
    lab = labels[0] if labels.ndim == 3 else labels
    border, a, b = border_pixels(lab)
    if blend is not None:
        ok = torch.zeros_like(border)
        ok[border] = blend[a[border].long()] & blend[b[border].long()]
        border &= ok
    if changed is not None:
        near = ops.dilate((changed[:1] > 0.01).float(), 2)[0] > 0
        border &= near
    half = max(1, seam_px // 2)
    # linear falloff with the distance to the border: 1 on it, 0 at half + 1
    cur = border.float()[None]
    acc = torch.zeros_like(cur)
    for _ in range(half + 1):
        acc += cur
        cur = ops.dilate(cur, 1)
    soft = acc / (half + 1)
    if keep is not None and bool(keep.any()):
        k = keep[lab.clamp_min(0).long()] & (lab >= 0)
        soft = soft * (~k).float()[None]
    if scope is not None:
        soft = soft * scope[:1].float()
    soft = soft * (lab >= 0).float()[None]
    return soft


# --------------------------------------------------------------------------
# tiles
# --------------------------------------------------------------------------

def tile_starts(total: int, tile: int, overlap: int) -> list[int]:
    if tile >= total:
        return [0]
    n = math.ceil((total - overlap) / (tile - overlap))
    step = (total - tile) / max(n - 1, 1)
    return sorted({int(round(i * step)) for i in range(n)})


def seam_tiles(mask: torch.Tensor, tile_px: int, overlap: int, grid: int = 16) -> list[ops.CropPlan]:
    """Tiles at 1:1 scale (work = box) covering the canvas; only those containing seam pixels."""
    H, W = mask.shape[-2], mask.shape[-1]
    tw = min(tile_px, W) // grid * grid
    th = min(tile_px, H) // grid * grid
    out = []
    for y in tile_starts(H, th, overlap):
        for x in tile_starts(W, tw, overlap):
            if float(mask[..., y:y + th, x:x + tw].max()) > 0.05:
                out.append(ops.CropPlan(x, y, tw, th, grid, grid, tw, th))
    return out


def dominant_region(labels: torch.Tensor, weight: torch.Tensor) -> int:
    """The region id carrying most of the weight inside a crop (-1 if none)."""
    lab = labels.flatten()
    w = weight.flatten()
    sel = lab >= 0
    if not bool(sel.any()):
        return -1
    sums = torch.bincount(lab[sel].long(), weights=w[sel].double())
    return int(sums.argmax()) if float(sums.max()) > 0 else -1


# --------------------------------------------------------------------------
# the pass
# --------------------------------------------------------------------------

@dataclass
class SeamSettings:
    seam_px: int = 32
    denoise: float = 0.3
    tile_px: int = 1024
    steps: int = 0
    cfg: float = 0.0
    sampler_name: str = ""
    scheduler: str = ""
    seed: int = 0
    prompt: str = ""          # empty = the prompt of the region most present in each tile


def plan_seam_mask(rp, seam_px: int, changed: torch.Tensor | None = None) -> torch.Tensor:
    entries = rp.plan["regions"]
    blend = torch.tensor([bool(e.get("blend", True)) for e in entries])
    keep = torch.tensor([e["strategy"] == "keep" for e in entries])
    return seam_mask(rp.regions.labels[:1], seam_px, changed=changed, blend=blend, keep=keep,
                     scope=rp.regions.scope)


def seam_pass(adapter, canvas: torch.Tensor, rp, settings: SeamSettings,
              changed: torch.Tensor | None = None, progress=None, check_interrupt=None):
    """Returns (canvas, seam_mask [1,H,W], report lines)."""
    from .strategies import _inpaint_crop    # shared with S1

    H, W = canvas.shape[1], canvas.shape[2]
    if rp.regions.size != (W, H):
        raise ValueError(f"The image is {W}x{H} but the regions are {rp.regions.size[0]}x"
                         f"{rp.regions.size[1]}; they must match.")
    canvas = canvas[:1, ..., :3].float().cpu().clone()
    mask = plan_seam_mask(rp, settings.seam_px, changed)
    tiles = seam_tiles(mask, settings.tile_px, overlap=max(64, settings.seam_px * 2), grid=adapter.grid)
    entries = rp.plan["regions"]
    lines = [f"seam band {settings.seam_px}px, {int((mask > 0.05).sum())} px, {len(tiles)} tile(s), "
             f"denoise {settings.denoise}"]
    tile_prompts = []
    for cp in tiles:
        rid = dominant_region(ops.crop(rp.regions.labels[:1].float(), cp).long(), ops.crop(mask, cp))
        e = entries[rid] if rid >= 0 else None
        tile_prompts.append((settings.prompt or (e["prompt"] if e else ""), e["negative"] if e else "", rid))
    adapter.encode_prompts([p for p, _, _ in tile_prompts] + [n for _, n, _ in tile_prompts])

    dummy = torch.zeros((1, H, W))
    for i, (cp, (prompt, negative, rid)) in enumerate(zip(tiles, tile_prompts)):
        if check_interrupt:
            check_interrupt()
        t0 = time.perf_counter()
        m = ops.crop(mask, cp)
        sampling = dict(seed=int(settings.seed) + 100_000 + i, steps=settings.steps, cfg=settings.cfg,
                        sampler_name=settings.sampler_name, scheduler=settings.scheduler)
        canvas, dummy = _inpaint_crop(adapter, canvas, dummy, cp, m, m, prompt, negative,
                                      settings.denoise, "self", "none", sampling, where=f"seam tile {i}")
        name = entries[rid]["name"] if rid >= 0 else "-"
        lines.append(f"tile {i + 1}: box {[cp.x0, cp.y0, cp.w, cp.h]}, prompt of {name}, "
                     f"{time.perf_counter() - t0:.1f}s")
        if progress:
            progress(i + 1, len(tiles))
    return canvas, mask, lines


# --------------------------------------------------------------------------
# refine: the whole canvas tile by tile at 1:1 (the final of kubakub versions)
# --------------------------------------------------------------------------

def tile_ramp(cp: ops.CropPlan, W: int, H: int, overlap: int) -> torch.Tensor:
    """[1,h,w] paste weight of a tile: 1 inside, a linear ramp over the overlap towards every tile edge
    that is not a canvas edge, so overlapping tiles blend instead of meeting at a line."""
    def axis(n, start, total):
        r = torch.ones(n)
        k = max(1, min(overlap, n // 2))
        up = (torch.arange(k) + 1) / (k + 1)
        if start > 0:
            r[:k] = up
        if start + n < total:
            r[n - k:] = torch.minimum(r[n - k:], up.flip(0))
        return r
    return (axis(cp.h, cp.y0, H)[:, None] * axis(cp.w, cp.x0, W)[None, :])[None]


def tile_prompt(entries: list, labels: torch.Tensor, weight: torch.Tensor, share: float = 0.2, most: int = 3):
    """(prompt, negative, names) for a tile: the prompts of the regions that fill it most (each at least
    `share` of the tile's region pixels, at most `most`), largest first, joined without repeats."""
    lab, w = labels.flatten(), weight.flatten()
    sel = lab >= 0
    if not bool(sel.any()):
        return "", "", []
    sums = torch.bincount(lab[sel].long(), weights=w[sel].double(), minlength=len(entries))
    total = float(sums.sum())
    if total <= 0:
        return "", "", []
    ids = [int(i) for i in torch.argsort(sums, descending=True)[:most] if float(sums[i]) >= share * total]
    ids = ids or [int(sums.argmax())]
    prompts, negatives = [], []
    for i in ids:
        if entries[i]["strategy"] == "keep":
            continue
        for part in entries[i]["prompt"].split(","):
            if part.strip() and part.strip() not in prompts:
                prompts.append(part.strip())
        if entries[i]["negative"] and entries[i]["negative"] not in negatives:
            negatives.append(entries[i]["negative"])
    return ", ".join(prompts), ", ".join(negatives), [entries[i]["name"] for i in ids]


def refine_tiles(adapter, canvas: torch.Tensor, rp, denoise: float = 0.4, tile_px: int = 1024, seed: int = 0,
                 steps: int = 0, sampler_name: str = "", scheduler: str = "", style=None, cache=None,
                 progress=None, check_interrupt=None, whole: bool = False, cfg: float = 0.0):
    """
    Resample everything inside the regions (and the scope) tile by tile at 1:1 with a low denoise, each tile
    with the prompts of the regions in it and the tile itself as reference: detail at full resolution on a
    canvas that already has the picture (an upscaled draft). keep regions stay untouched.
    whole = the canvas as a single tile (the unify pass of kubakub versions, on a draft).
    cfg 0 = the adapter's default.
    Returns (canvas, report lines).
    """
    from .strategies import _inpaint_crop

    H, W = canvas.shape[1], canvas.shape[2]
    if rp.regions.size != (W, H):
        raise ValueError(f"The image is {W}x{H} but the regions are {rp.regions.size[0]}x"
                         f"{rp.regions.size[1]}; they must match.")
    canvas = canvas[:1, ..., :3].float().cpu().clone()
    entries = rp.plan["regions"]
    lab = rp.regions.labels[:1]
    keep = torch.tensor([e["strategy"] == "keep" for e in entries] or [False])
    mask = ((lab >= 0) & ~keep[lab.clamp_min(0).long()]).float()
    if rp.regions.scope is not None:
        mask = mask * rp.regions.scope[:1].float()
    overlap = max(64, tile_px // 8)
    if whole:                   # one tile: the canvas, grown past its right / bottom edge to the grid
        g = adapter.grid
        tw, th = -(-W // g) * g, -(-H // g) * g
        tiles = [ops.CropPlan(0, 0, tw, th, g, g, tw, th)]
    else:
        tiles = seam_tiles(mask, tile_px, overlap=overlap, grid=adapter.grid)
    prompts = [tile_prompt(entries, ops.crop(lab.float(), cp).long(), ops.crop(mask, cp)) for cp in tiles]
    adapter.encode_prompts([p for p, _, _ in prompts] + [n for _, n, _ in prompts])
    lines = [f"refine {len(tiles)} tile(s) of {tiles[0].w}x{tiles[0].h}, denoise {denoise}" if tiles
             else "refine: nothing inside the regions"]
    dummy = torch.zeros((1, H, W))
    reference = f"style:{style.digest}" if style is not None else "self"
    for i, (cp, (prompt, negative, names)) in enumerate(zip(tiles, prompts)):
        if check_interrupt:
            check_interrupt()
        t0 = time.perf_counter()
        m = ops.crop(mask, cp)
        sampling = dict(seed=int(seed) + 200_000 + i, steps=steps, cfg=float(cfg), sampler_name=sampler_name,
                        scheduler=scheduler)
        canvas, dummy = _inpaint_crop(adapter, canvas, dummy, cp, m, m * tile_ramp(cp, W, H, overlap), prompt,
                                      negative, denoise, reference, "none", sampling, where=f"refine tile {i}",
                                      cache=cache, style=style)
        lines.append(f"tile {i + 1}: box {[cp.x0, cp.y0, cp.w, cp.h]}, prompts of {', '.join(names) or '-'}, "
                     f"{time.perf_counter() - t0:.1f}s")
        if progress:
            progress(i + 1, len(tiles))
    return canvas, lines


# Adapted from ComfyUI core comfy_extras/nodes_differential_diffusion.py
# (github.com/comfyanonymous/ComfyUI, from github.com/exx8/differential-diffusion).
def _dd_forward(sigma, denoise_mask, extra_options, strength):
    model = extra_options["model"]
    step_sigmas = extra_options["sigmas"]
    sigma_to = model.inner_model.model_sampling.sigma_min
    if step_sigmas[-1] > sigma_to:
        sigma_to = step_sigmas[-1]
    sigma_from = step_sigmas[0]
    ts_from = model.inner_model.model_sampling.timestep(sigma_from)
    ts_to = model.inner_model.model_sampling.timestep(sigma_to)
    current_ts = model.inner_model.model_sampling.timestep(sigma[0])
    threshold = (current_ts - ts_to) / (ts_from - ts_to)
    binary = (denoise_mask >= threshold).to(denoise_mask.dtype)
    if strength and strength < 1:
        return strength * binary + (1 - strength) * denoise_mask
    return binary


def differential_diffusion_patch(model, strength: float = 1.0):
    """A clone of the MODEL whose soft denoise mask becomes a per-step threshold (soft edges)."""
    m = model.clone()
    m.set_model_denoise_mask_function(lambda *a, **k: _dd_forward(*a, **k, strength=strength))
    return m
