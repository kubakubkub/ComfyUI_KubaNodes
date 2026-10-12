"""
materials.py

A shader's picture as the material of regions. The picture comes from a kubakub shader node: its `regions` input
makes the shader know the regions, `only` chooses which of them it draws in, `on_picture` lays it over a picture,
and a second shader node on top of the first is a second layer. Two cables carry it into the region samplers:

    material   IMAGE: the regions the sampler paints start from it, so the model paints from the shader's picture
               (a region whose plan says `reference = material` keeps its own start and is shown the material)
    change     MASK: how much each pixel of a painted region may change (its brightest place = all of 'denoise',
               black = none)

blend() is how the shader node lays a shader over a picture.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

BLENDS = ("normal", "screen", "add", "multiply", "overlay", "difference", "displace")
LUMA = (0.2126, 0.7152, 0.0722)


def blend(shader, picture, how="screen", amount=1.0, mask=None):
    """A shader's picture laid over a picture: float32 [h, w, 3] each -> the shader's array, changed in place.
    amount 0 = the picture alone, 1 = all of the blend; mask [h, w]: the picture stays as it is outside it.
    normal = the shader in place of the picture; multiply = the shader as light on the picture; displace = the
    shader's light and dark bend the picture like glass."""
    p, s = picture[..., :3], shader
    if how == "normal":
        out = s
    elif how == "add":
        out = np.minimum(p + s, 1.0)
    elif how == "multiply":
        out = p * s
    elif how == "overlay":
        out = np.where(p < 0.5, 2.0 * p * s, 1.0 - 2.0 * (1.0 - p) * (1.0 - s))
    elif how == "difference":
        out = np.abs(p - s)
    elif how == "displace":
        import cv2
        h, w = s.shape[:2]
        lum = cv2.GaussianBlur((s * np.asarray(LUMA, np.float32)).sum(-1), (0, 0), max(1.0, w / 160.0))
        reach = 0.25 * w                              # soft slopes of light push the picture a few pixels, like glass
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        out = cv2.remap(np.ascontiguousarray(p), xx + cv2.Sobel(lum, cv2.CV_32F, 1, 0, ksize=3) * reach / 8.0,
                        yy + cv2.Sobel(lum, cv2.CV_32F, 0, 1, ksize=3) * reach / 8.0, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    else:                                             # screen
        out = 1.0 - (1.0 - p) * (1.0 - s)
    k = float(min(max(amount, 0.0), 1.0))
    k = k * mask[..., None] if mask is not None else k
    shader[...] = p + (out - p) * k
    return shader


def frame_at(n: int, at: float = 0.3) -> int:
    """Which frame of a clip of n frames a still takes: `at` of the way through it (0-1)."""
    return min(max(int(n) - 1, 0), int(round(min(max(float(at), 0.0), 1.0) * (int(n) - 1))))


def _one(x, at, W, H, lines, what):
    """One frame of an IMAGE / MASK batch, float32 on the CPU at W x H; says so when it picked or resized."""
    import torch
    x = x.float().cpu()
    x = x[None] if x.ndim == 2 else x
    if len(x) > 1:
        i = frame_at(len(x), at)
        lines.append(f"{what}: a clip of {len(x)} frames, frame {i + 1} is used (material_at = {float(at):g})")
        x = x[i:i + 1]
    if (x.shape[2], x.shape[1]) != (W, H):
        lines.append(f"{what}: {x.shape[2]}x{x.shape[1]}, resized to the picture's {W}x{H}")
        chan = x.ndim == 4
        y = x.movedim(-1, 1) if chan else x[:, None]
        y = torch.nn.functional.interpolate(y, size=(H, W), mode="bilinear", align_corners=False)
        x = y.movedim(1, -1) if chan else y[:, 0]
    return x


@dataclass
class Prepared:
    canvas: object                    # IMAGE [1, H, W, 3]: the picture with the material laid in (the sampler starts here)
    change: object = None             # MASK [1, H, W]: how much each pixel may change; None = none connected
    reference: object = None          # IMAGE [1, H, W, 3]: the material, for the regions with reference = material
    lines: tuple = ()


def before(rp, image, material=None, change=None, pick=None, at=0.3) -> Prepared:
    """What a material gives the sampler. image = IMAGE [1, H, W, 3+] at the regions' size; material = IMAGE,
    change = MASK (a batch: the frame at `at`; another size: resized). The material is laid into the regions this
    run paints (strategy other than keep, and `pick`); those whose reference is 'material' keep their own start and
    are shown the material instead. The change map is scaled so that its brightest place on the painted regions
    is 1 (a coloured shader never reaches white)."""
    import torch
    canvas = image[:1, ..., :3].float().cpu()
    if material is None and change is None:
        return Prepared(canvas)
    H, W = canvas.shape[1:3]
    entries = rp.plan["regions"]
    lab = rp.regions.labels[0]
    lab = lab.cpu().numpy() if hasattr(lab, "cpu") else np.asarray(lab)
    painted = [e for e in entries if e.get("strategy") != "keep" and (pick is None or pick(e))]
    as_ref = [e for e in painted if str(e.get("reference", "")).strip().lower() == "material"]
    lines, reference = [], None
    if not painted:
        lines.append("material / change: no region of this run is painted (strategy = keep everywhere), so they are not used")
    if material is not None and painted:
        mat = _one(material[..., :3], at, W, H, lines, "material")
        start = [int(e["region_id"]) for e in painted if e not in as_ref]
        if start:
            on = torch.from_numpy(np.isin(lab, start))[None, ..., None]
            canvas = torch.where(on, mat, canvas)
            lines.append(f"material: the start picture of {len(start)} region{'s' if len(start) != 1 else ''}")
        if as_ref:
            reference = mat
            lines.append(f"material: shown as the reference to {len(as_ref)} region{'s' if len(as_ref) != 1 else ''} (reference = material)")
    if change is not None and painted:
        chg = _one(change, at, W, H, lines, "change").clamp(0.0, 1.0)
        on = torch.from_numpy(np.isin(lab, [int(e["region_id"]) for e in painted]))[None]
        top = float(chg[on].max()) if bool(on.any()) else 0.0
        if top > 1e-3:
            chg = (chg / top).clamp(0.0, 1.0)
            lines.append("change: white = all of a region's denoise, black = nothing"
                         + (f" (its brightest place was {top:.2f}, counted as white)" if top < 0.98 else ""))
        else:
            lines.append("change: the mask is black on the painted regions, so nothing changes there")
        change = chg
    else:
        change = None
    return Prepared(canvas, change, reference, tuple(lines))


def guide(pixels, clip, region, moving):
    """
    A material clip as the guide of one video clip: the frames of the crop with the material inside the region.

    pixels   float32 [F, h, w, 3]: the crop's frames (the still repeated, or the background), not changed
    clip     F pictures float32 [h, w, 3]: the material, cut and scaled to the same crop, one per frame
    region   bool [h, w]: where the clip's regions are
    moving   bool [F]: the frames inside the clip's time range (outside it the crop stays as it is)
    -> float32 [F, h, w, 3], or None when the material does not differ from the crop there (nothing to follow)
    """
    out = np.array(pixels, np.float32, copy=True)
    differs = False
    for i in np.flatnonzero(moving):
        m = clip[int(i)]
        differs = differs or float(np.abs(m[region] - out[i][region]).max(initial=0.0)) > 2.0 / 255.0
        out[i][region] = m[region]
    return out if differs else None
