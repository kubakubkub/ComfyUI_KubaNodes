"""
geometry.py

The geometry contract of Kuba Regions:
the matrix is the rule, every output pixel lands exactly where the template
says. No step may scale non-uniformly. A work canvas is `target / k` for one
uniform factor k and is fitted to the model grid by padding, never stretching.
After generation the padding is cropped away and the result is scaled back to
the exact target size.

Pure logic: torch and the standard library only, no ComfyUI imports, so
tests/test_geometry.py runs it without starting ComfyUI.

Tensor layout follows ComfyUI: IMAGE is [B, H, W, C] float 0..1, MASK is
[B, H, W]. For video the batch axis is the frame axis.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from fractions import Fraction
from typing import Callable

import torch
import torch.nn.functional as F

PLAN_FORMAT = "kubakub.regions.canvas_plan"
PLAN_VERSION = 1

PAD_ALIGNS = ("center", "end", "start")
PAD_MODES = ("replicate", "reflect", "gray", "black")
ASPECT_MODES = ("error", "center_crop", "pad")


# --------------------------------------------------------------------------
# model geometry
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Geometry:
    """Size rules of one model family. frame_step 0 means a still model."""
    family: str
    divisor: int
    budget_mp: float
    max_mp: float
    frame_step: int = 0
    frame_offset: int = 1
    fps: float = 0.0
    fps_fixed: bool = False
    note: str = ""

    @property
    def is_video(self) -> bool:
        return self.frame_step > 0

    def snap_frames(self, n: int) -> int:
        """Smallest valid frame count >= n (frame_step * j + frame_offset)."""
        if not self.is_video or n <= 0:
            return max(n, 1)
        j = max(0, math.ceil((n - self.frame_offset) / self.frame_step))
        return self.frame_step * j + self.frame_offset


# Research 2026-09-23 against core 0.35.0.
GEOMETRIES: dict[str, Geometry] = {
    "flux2": Geometry(
        "flux2", divisor=16, budget_mp=1.0, max_mp=4.0,
        note="Flux 2 Klein: VAE 8x + 2x2 patchify. About 1 MP trained, up to 4 MP."),
    "ltxav": Geometry(
        "ltxav", divisor=32, budget_mp=960 * 544 / 1e6, max_mp=1920 * 1088 / 1e6,
        frame_step=8, frame_offset=1, fps=25.0,
        note="LTX-2.3/2.5: 8k+1 frames. Stage 1 at about 960x544, then latent x2."),
    "minimax_h3": Geometry(
        "minimax_h3", divisor=32, budget_mp=1.03, max_mp=1.03,
        frame_step=17, frame_offset=5, fps=24.0, fps_fixed=True,
        note="MiniMax H3: 17k+5 frames, 24 fps fixed, short edge 768 trained."),
    "qwen_image21": Geometry(
        "qwen_image21", divisor=32, budget_mp=2048 * 2048 / 1e6, max_mp=2048 * 2048 / 1e6 * 1.05,
        note="Qwen Image 2.1: native 2048x2048 (4 MP), 16x VAE, edit refs on multiples of 32."),
    "sd_8": Geometry(
        "sd_8", divisor=8, budget_mp=1.0, max_mp=2.0,
        note="Generic latent model with an 8x VAE (SD1.5, SDXL)."),
}


def get_geometry(family: str, divisor: int = 0) -> Geometry:
    """The geometry for a family name, with an optional divisor override."""
    if family not in GEOMETRIES:
        raise ValueError(f"Unknown model family '{family}'. Known: {', '.join(GEOMETRIES)}.")
    g = GEOMETRIES[family]
    if divisor and divisor != g.divisor:
        g = Geometry(**{**asdict(g), "divisor": int(divisor),
                        "note": g.note + f" Divisor overridden to {divisor}."})
    return g


# --------------------------------------------------------------------------
# canvas plan
# --------------------------------------------------------------------------

@dataclass
class CanvasPlan:
    """Where a target canvas lives while a model works on it.

    work_w x work_h is exactly target / k (same aspect, integer pixels).
    padded_w x padded_h is the work canvas grown to the model grid. pad_* is
    the padding per side at work scale. Restoring = crop the padding away,
    then scale the work canvas by k to the target.
    """
    target_w: int
    target_h: int
    k: str                      # exact factor as a fraction string, e.g. "2" or "5/2"
    work_w: int
    work_h: int
    padded_w: int
    padded_h: int
    pad_left: int
    pad_right: int
    pad_top: int
    pad_bottom: int
    family: str
    divisor: int
    target_frames: int = 0
    work_frames: int = 0
    fps: float = 0.0
    notes: list = field(default_factory=list)
    format: str = PLAN_FORMAT
    version: int = PLAN_VERSION

    @property
    def k_fraction(self) -> Fraction:
        return Fraction(self.k)

    @property
    def k_float(self) -> float:
        return float(Fraction(self.k))

    @property
    def pad_fraction(self) -> float:
        return self.padded_w * self.padded_h / (self.work_w * self.work_h) - 1.0

    @property
    def padded_mp(self) -> float:
        return self.padded_w * self.padded_h / 1e6

    def to_dict(self) -> dict:
        d = asdict(self)
        d["k_float"] = round(self.k_float, 6)
        d["pad_percent"] = round(self.pad_fraction * 100, 3)
        d["padded_mp"] = round(self.padded_mp, 4)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "CanvasPlan":
        if d.get("format") != PLAN_FORMAT:
            raise ValueError(f"Not a canvas plan (format {d.get('format')!r}).")
        if int(d.get("version", 0)) > PLAN_VERSION:
            raise ValueError(f"Canvas plan version {d.get('version')} is newer than this "
                             f"code ({PLAN_VERSION}). Update ComfyUI_KubaNodes.")
        names = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in names})

    def report(self) -> str:
        k = self.k_fraction
        lines = [
            f"target      {self.target_w} x {self.target_h}",
            f"k           {k} ({self.k_float:g}) uniform, {'integer' if k.denominator == 1 else 'non-integer'}",
            f"work        {self.work_w} x {self.work_h}  (target / k, aspect exact)",
            f"model grid  {self.padded_w} x {self.padded_h}  ({self.family}, divisor {self.divisor}, "
            f"{self.padded_mp:.2f} MP)",
            f"padding     left {self.pad_left}, right {self.pad_right}, top {self.pad_top}, "
            f"bottom {self.pad_bottom} px at work scale ({self.pad_fraction * 100:.2f} % extra area)",
            f"restore     crop padding -> {self.work_w} x {self.work_h} -> x{self.k_float:g} -> "
            f"{self.target_w} x {self.target_h}",
        ]
        if self.target_frames:
            lines.append(f"frames      {self.work_frames} generated at {self.fps:g} fps, "
                         f"trim to {self.target_frames}")
        lines += [f"note        {n}" for n in self.notes]
        return "\n".join(lines)


def _aspect(w: int, h: int) -> tuple[int, int, int]:
    g = math.gcd(w, h)
    return w // g, h // g, g


def exact_factors(target_w: int, target_h: int, min_side: int = 64) -> list[Fraction]:
    """Every k (k >= 1) for which target / k is a whole-pixel canvas of the same aspect."""
    aw, ah, g = _aspect(target_w, target_h)
    out = []
    for j in range(g, 0, -1):
        if min(aw * j, ah * j) < min_side:
            break
        out.append(Fraction(g, j))
    return out


def _ceil_to(n: int, d: int) -> int:
    return -(-n // d) * d


def _split(pad: int, align: str) -> tuple[int, int]:
    if align == "start":        # padding before the image (left / top)
        return pad, 0
    if align == "end":          # padding after the image (right / bottom)
        return 0, pad
    return pad // 2, pad - pad // 2


def _parse_k(k) -> Fraction:
    if isinstance(k, Fraction):
        return k
    if isinstance(k, str):
        return Fraction(k.strip())
    return Fraction(str(round(float(k), 6))).limit_denominator(10000)


def plan_canvas(target_w: int, target_h: int, geometry: Geometry, k=None,
                budget_mp: float = 0.0, integer_k: bool = False, pad_align: str = "center",
                max_pad_percent: float = 2.0, target_frames: int = 0) -> CanvasPlan:
    """Choose the work canvas for a target and a model.

    k given: use it; it must divide the target into whole pixels.
    k None: pick the exact factor whose grid canvas is closest to budget_mp
    (the model's trained budget when 0), with the least padding.
    """
    target_w, target_h = int(target_w), int(target_h)
    if target_w < 16 or target_h < 16:
        raise ValueError(f"Target {target_w}x{target_h} is too small.")
    if pad_align not in PAD_ALIGNS:
        raise ValueError(f"pad_align must be one of {PAD_ALIGNS}.")
    d = geometry.divisor
    max_pad = max_pad_percent / 100.0

    def build(kf: Fraction) -> CanvasPlan:
        ww, wh = Fraction(target_w) / kf, Fraction(target_h) / kf
        if ww.denominator != 1 or wh.denominator != 1:
            near = sorted(exact_factors(target_w, target_h), key=lambda f: abs(float(f - kf)))[:4]
            raise ValueError(
                f"k = {kf} does not divide {target_w}x{target_h} into whole pixels "
                f"({float(ww):.3f} x {float(wh):.3f}), so the aspect ratio would change. "
                f"Nearest exact factors: {', '.join(f'{float(f):g}' for f in sorted(near))}.")
        ww, wh = int(ww), int(wh)
        pw, ph = _ceil_to(ww, d), _ceil_to(wh, d)
        pl, pr = _split(pw - ww, pad_align)
        pt, pb = _split(ph - wh, pad_align)
        plan = CanvasPlan(target_w, target_h, str(kf), ww, wh, pw, ph, pl, pr, pt, pb,
                          geometry.family, d)
        if geometry.is_video and target_frames > 0:
            plan.target_frames = int(target_frames)
            plan.work_frames = geometry.snap_frames(int(target_frames))
            plan.fps = geometry.fps
        return plan

    kf = _parse_k(k) if k is not None and k != "" else Fraction(0)
    if kf > 0:
        if kf < 1:
            raise ValueError(f"k = {kf} would make the work canvas larger than the target; use k >= 1.")
        plan = build(kf)
        if plan.pad_fraction > max_pad:
            plan.notes.append(f"padding is {plan.pad_fraction * 100:.2f} % of the area, above "
                              f"max_pad_percent {max_pad_percent:g}")
    else:
        budget = budget_mp if budget_mp > 0 else geometry.budget_mp
        best, best_score = None, None
        for kf in exact_factors(target_w, target_h):
            if integer_k and kf.denominator != 1:
                continue
            plan = build(kf)
            if plan.pad_fraction > max_pad or plan.padded_mp > geometry.max_mp * 1.001:
                continue
            # Integer k is preferred: restore stays a clean xN and latent x2 paths can use it.
            score = (abs(math.log(plan.padded_mp / budget)) + 4.0 * plan.pad_fraction
                     + (0.1 if kf.denominator != 1 else 0.0))
            if best_score is None or score < best_score:
                best, best_score = plan, score
        if best is None:
            raise ValueError(
                f"No exact factor fits {target_w}x{target_h} on the {geometry.family} grid "
                f"(divisor {d}) with at most {max_pad_percent:g} % padding"
                + (" and an integer k" if integer_k else "") + ". Raise max_pad_percent or give k.")
        plan = best
        plan.notes.append(f"k chosen for a budget of {budget:.2f} MP")

    if plan.padded_mp > geometry.max_mp * 1.001:
        plan.notes.append(f"{plan.padded_mp:.2f} MP is above the {geometry.max_mp:.2f} MP this "
                          f"model handles well; expect repetition or OOM")
    if plan.k_fraction.denominator != 1:
        plan.notes.append("non-integer k: restore needs a pixel resize (a latent x2 upscaler "
                          "cannot reach the target alone)")
    if geometry.fps_fixed and plan.target_frames:
        plan.notes.append(f"{geometry.family} always runs at {geometry.fps:g} fps")
    return plan


# --------------------------------------------------------------------------
# tensor ops
# --------------------------------------------------------------------------

Resize = Callable[[torch.Tensor, int, int], torch.Tensor]  # BCHW -> BCHW


def torch_resize(method: str = "auto") -> Resize:
    """A resize function on BCHW tensors. auto = area when shrinking, bicubic when growing."""
    def run(x: torch.Tensor, w: int, h: int) -> torch.Tensor:
        if x.shape[-1] == w and x.shape[-2] == h:
            return x
        m = method
        if m == "auto":
            m = "area" if w < x.shape[-1] else "bicubic"
        if m in ("bicubic", "bilinear"):
            y = F.interpolate(x, size=(h, w), mode=m, align_corners=False, antialias=True)
        elif m == "area":
            y = F.interpolate(x, size=(h, w), mode="area")
        else:
            y = F.interpolate(x, size=(h, w), mode="nearest-exact")
        return y.clamp(0.0, 1.0)
    return run


def _bhwc_resize(img: torch.Tensor, w: int, h: int, resize: Resize) -> torch.Tensor:
    return resize(img.movedim(-1, 1), w, h).movedim(1, -1)


def _mask_resize(mask: torch.Tensor, w: int, h: int, resize: Resize) -> torch.Tensor:
    return resize(mask[:, None], w, h)[:, 0]


def fit_aspect(img: torch.Tensor, target_w: int, target_h: int, mode: str = "error",
               mask: torch.Tensor | None = None):
    """Make an image (and mask) the exact aspect of the target by cropping or padding.

    Returns (image, mask, valid) where valid is 1 on original pixels and 0 on
    padding. The aspect after this is exact, so a later resize is uniform.
    """
    b, h, w, _ = img.shape
    aw, ah, _ = _aspect(target_w, target_h)
    valid = torch.ones((b, h, w), dtype=img.dtype, device=img.device)
    if w * ah == h * aw:
        return img, mask, valid
    if mode == "error":
        raise ValueError(
            f"The image is {w}x{h} but the target is {target_w}x{target_h}; the aspect ratios "
            f"differ and nothing may be stretched. Set aspect_mode to center_crop or pad.")
    if mode == "center_crop":
        j = min(w // aw, h // ah)
        cw, ch = aw * j, ah * j
        x0, y0 = (w - cw) // 2, (h - ch) // 2
        img = img[:, y0:y0 + ch, x0:x0 + cw]
        valid = valid[:, y0:y0 + ch, x0:x0 + cw]
        if mask is not None:
            mask = mask[:, y0:y0 + ch, x0:x0 + cw]
        return img, mask, valid
    if mode == "pad":
        j = max(-(-w // aw), -(-h // ah))
        cw, ch = aw * j, ah * j
        l, t = (cw - w) // 2, (ch - h) // 2
        pads = (l, cw - w - l, t, ch - h - t)
        img = pad_image(img, pads, "replicate")
        valid = F.pad(valid, pads, value=0.0)
        if mask is not None:
            mask = F.pad(mask, pads, value=0.0)
        return img, mask, valid
    raise ValueError(f"aspect_mode must be one of {ASPECT_MODES}.")


def pad_image(img: torch.Tensor, pads: tuple[int, int, int, int], mode: str) -> torch.Tensor:
    """Pad BHWC by (left, right, top, bottom). Never resamples."""
    if not any(pads):
        return img
    if mode == "gray":
        return F.pad(img.movedim(-1, 1), pads, value=0.5).movedim(1, -1)
    if mode == "black":
        return F.pad(img.movedim(-1, 1), pads, value=0.0).movedim(1, -1)
    x = img.movedim(-1, 1)
    if mode == "reflect" and (pads[0] >= x.shape[-1] or pads[1] >= x.shape[-1]
                              or pads[2] >= x.shape[-2] or pads[3] >= x.shape[-2]):
        mode = "replicate"      # reflect needs padding smaller than the image
    return F.pad(x, pads, mode="reflect" if mode == "reflect" else "replicate").movedim(1, -1)


def to_work(img: torch.Tensor, plan: CanvasPlan, mask: torch.Tensor | None = None,
            aspect_mode: str = "error", pad_mode: str = "replicate",
            resize: Resize | None = None):
    """Bring an image (and mask) from any size of the target's aspect onto the model grid.

    Returns (image, mask, scope) at padded_w x padded_h. scope is 1 where the
    pixel belongs to the target canvas and 0 on padding (and on aspect padding).
    """
    resize = resize or torch_resize("auto")
    img, mask, valid = fit_aspect(img, plan.target_w, plan.target_h, aspect_mode, mask)
    img = _bhwc_resize(img, plan.work_w, plan.work_h, resize)
    valid = (_mask_resize(valid, plan.work_w, plan.work_h, torch_resize("area")) > 0.5).to(img.dtype)
    if mask is not None:
        if mask.shape[0] not in (1, img.shape[0]) and img.shape[0] != 1:
            raise ValueError(f"mask batch {mask.shape[0]} does not match image batch {img.shape[0]}.")
        mask = _mask_resize(mask.to(img.dtype), plan.work_w, plan.work_h, torch_resize("area"))
    pads = (plan.pad_left, plan.pad_right, plan.pad_top, plan.pad_bottom)
    img = pad_image(img, pads, pad_mode)
    scope = F.pad(valid, pads, value=0.0)
    if mask is not None:
        mask = F.pad(mask, pads, value=0.0)
    assert_size(img, plan.padded_w, plan.padded_h, "to_work")
    return img, mask, scope


def _crop_scale(w: int, h: int, plan: CanvasPlan):
    """Find the scale s of an image relative to the plan. Returns (s, padded: bool)."""
    for base_w, base_h, padded in ((plan.padded_w, plan.padded_h, True),
                                   (plan.work_w, plan.work_h, False),
                                   (plan.target_w, plan.target_h, None)):
        s = Fraction(w, base_w)
        if Fraction(h, base_h) == s:
            return s, padded
    raise ValueError(
        f"An image of {w}x{h} is no uniform scale of this canvas plan (padded "
        f"{plan.padded_w}x{plan.padded_h}, work {plan.work_w}x{plan.work_h}, target "
        f"{plan.target_w}x{plan.target_h}). Something resized it non-uniformly or cropped it.")


def restore(img: torch.Tensor, plan: CanvasPlan, mask: torch.Tensor | None = None,
            resize: Resize | None = None, trim_frames: bool = True):
    """Back from the model grid to the exact target: crop padding, scale uniformly, check size.

    Accepts the padded canvas at any uniform scale (e.g. after a 4x model
    upscaler), an already cropped work canvas at any scale, or the target.
    """
    resize = resize or torch_resize("auto")
    _, h, w, _ = img.shape
    s, padded = _crop_scale(w, h, plan)
    if padded:
        box = [plan.pad_left * s, plan.pad_top * s, plan.work_w * s, plan.work_h * s]
        if any(v.denominator != 1 for v in box):
            raise ValueError(f"At scale {s} the padding is not a whole number of pixels "
                             f"({[float(v) for v in box]}). Upscale by an integer factor first.")
        x0, y0, cw, ch = (int(v) for v in box)
        img = img[:, y0:y0 + ch, x0:x0 + cw]
        if mask is not None:
            mask = mask[:, y0:y0 + ch, x0:x0 + cw]
    img = _bhwc_resize(img, plan.target_w, plan.target_h, resize)
    if mask is not None:
        mask = _mask_resize(mask, plan.target_w, plan.target_h, torch_resize("auto"))
    if trim_frames and plan.target_frames and img.shape[0] == plan.work_frames:
        img = img[:plan.target_frames]
        if mask is not None and mask.shape[0] == plan.work_frames:
            mask = mask[:plan.target_frames]
    assert_size(img, plan.target_w, plan.target_h, "restore")
    return img, mask


def assert_size(img: torch.Tensor, w: int, h: int, where: str = "") -> None:
    ih, iw = img.shape[-3], img.shape[-2]
    if (iw, ih) != (w, h):
        raise AssertionError(f"{where}: expected {w}x{h}, got {iw}x{ih}. Registration is broken.")
