"""
ops.py

Shared array helpers of Kuba Regions strategies: crop planning under the
geometry contract, crop/paste with padding, mask morphology, feathering and
colour matching. Pure torch, no ComfyUI imports (tests/test_ops.py).

Layout as in ComfyUI: IMAGE [B, H, W, C] float 0..1, MASK [B, H, W].
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction

import torch
import torch.nn.functional as F


# --------------------------------------------------------------------------
# crop planning
# --------------------------------------------------------------------------

@dataclass
class CropPlan:
    """A box on the canvas (may reach past its edges) and its exact uniform scale to work size."""
    x0: int
    y0: int
    w: int
    h: int
    block: int          # target pixels per grid cell; scale = grid / block exactly
    grid: int
    work_w: int
    work_h: int

    @property
    def scale(self) -> Fraction:
        return Fraction(self.grid, self.block)

    def to_dict(self) -> dict:
        return {"box": [self.x0, self.y0, self.w, self.h], "scale": str(self.scale),
                "work": [self.work_w, self.work_h]}


def plan_crop(bbox, context_px: int, canvas_w: int, canvas_h: int, grid: int = 16,
              budget_px: float = 1024 * 1024, max_upscale: float = 4.0) -> CropPlan:
    """
    Box around a region (bbox = x, y, w, h) plus context, scaled uniformly to about
    budget_px on the model grid. The scale is exactly grid / block for a whole
    block, and the box grows (more context, never a stretch) to whole blocks, so
    work = box * scale is an exact grid multiple in both axes. The box is shifted
    to stay on the canvas where it fits; a box larger than the canvas is centred
    and the outside is padded when cropping.
    """
    x, y, w, h = (int(v) for v in bbox)
    cx0, cy0 = x - context_px, y - context_px
    cw, ch = w + 2 * context_px, h + 2 * context_px
    s = min(math.sqrt(budget_px / max(cw * ch, 1)), max_upscale)
    block = max(1, math.ceil(grid / s))            # scale = grid / block <= s
    bw, bh = -(-cw // block) * block, -(-ch // block) * block
    cx0 -= (bw - cw) // 2
    cy0 -= (bh - ch) // 2
    cx0 = _fit(cx0, bw, canvas_w)
    cy0 = _fit(cy0, bh, canvas_h)
    return CropPlan(cx0, cy0, bw, bh, block, grid, bw // block * grid, bh // block * grid)


def _fit(start: int, size: int, total: int) -> int:
    if size >= total:
        return -((size - total) // 2)
    return min(max(start, 0), total - size)


def crop(t: torch.Tensor, cp: CropPlan, mode: str = "replicate") -> torch.Tensor:
    """Cut the box from IMAGE [B,H,W,C] or MASK [B,H,W]; outside the canvas is padded."""
    is_mask = t.ndim == 3
    x = t[..., None] if is_mask else t
    H, W = x.shape[1], x.shape[2]
    l, tp = max(0, -cp.x0), max(0, -cp.y0)
    r, b = max(0, cp.x0 + cp.w - W), max(0, cp.y0 + cp.h - H)
    x = x[:, max(cp.y0, 0):min(cp.y0 + cp.h, H), max(cp.x0, 0):min(cp.x0 + cp.w, W)]
    if l or r or tp or b:
        y = x.movedim(-1, 1)
        y = F.pad(y, (l, r, tp, b), value=0.0) if (is_mask or mode == "zeros") else \
            F.pad(y, (l, r, tp, b), mode="replicate")
        x = y.movedim(1, -1)
    return x[..., 0] if is_mask else x


def paste(canvas: torch.Tensor, patch: torch.Tensor, alpha: torch.Tensor, cp: CropPlan) -> torch.Tensor:
    """Blend patch [B,h,w,C] into canvas at the box with alpha [B,h,w]; parts off the canvas are dropped."""
    H, W = canvas.shape[1], canvas.shape[2]
    x0, y0 = max(cp.x0, 0), max(cp.y0, 0)
    x1, y1 = min(cp.x0 + cp.w, W), min(cp.y0 + cp.h, H)
    px0, py0 = x0 - cp.x0, y0 - cp.y0
    p = patch[:, py0:py0 + (y1 - y0), px0:px0 + (x1 - x0)]
    a = alpha[:, py0:py0 + (y1 - y0), px0:px0 + (x1 - x0), None]
    out = canvas.clone()
    region = out[:, y0:y1, x0:x1]
    out[:, y0:y1, x0:x1] = region * (1 - a) + p.to(region) * a
    return out


def resize(img: torch.Tensor, w: int, h: int, method: str = "auto") -> torch.Tensor:
    """IMAGE resize; auto = area when shrinking, bicubic (antialiased) when growing."""
    if img.shape[2] == w and img.shape[1] == h:
        return img
    x = img.movedim(-1, 1)
    m = method
    if m == "auto":
        m = "area" if w < img.shape[2] else "bicubic"
    if m == "area":
        y = F.interpolate(x, size=(h, w), mode="area")
    else:
        y = F.interpolate(x, size=(h, w), mode="bicubic", align_corners=False, antialias=True)
    return y.clamp(0, 1).movedim(1, -1)


def resize_mask(m: torch.Tensor, w: int, h: int) -> torch.Tensor:
    if m.shape[-1] == w and m.shape[-2] == h:
        return m
    mode = "area" if w < m.shape[-1] else "bilinear"
    kw = {} if mode == "area" else {"align_corners": False}
    return F.interpolate(m[:, None], size=(h, w), mode=mode, **kw)[:, 0].clamp(0, 1)


# --------------------------------------------------------------------------
# masks
# --------------------------------------------------------------------------

def dilate(m: torch.Tensor, r: int) -> torch.Tensor:
    """Grow a MASK by r pixels (square). Negative r erodes."""
    if r == 0:
        return m
    k = 2 * abs(r) + 1
    if r > 0:
        return F.max_pool2d(m[:, None], k, stride=1, padding=abs(r))[:, 0]
    return -F.max_pool2d(-m[:, None], k, stride=1, padding=abs(r))[:, 0]


def gaussian_blur(m: torch.Tensor, sigma: float) -> torch.Tensor:
    """Separable gaussian blur of a MASK [B,H,W]; edges replicate."""
    if sigma <= 0:
        return m
    r = max(1, int(math.ceil(3 * sigma)))
    xs = torch.arange(-r, r + 1, dtype=m.dtype, device=m.device)
    k = torch.exp(-(xs ** 2) / (2 * sigma * sigma))
    k = k / k.sum()
    x = F.pad(m[:, None], (r, r, r, r), mode="replicate")
    x = F.conv2d(x, k.view(1, 1, 1, -1))
    x = F.conv2d(x, k.view(1, 1, -1, 1))
    return x[:, 0]


def feather_alpha(region: torch.Tensor, feather_px: int, limit: torch.Tensor | None = None) -> torch.Tensor:
    """
    Paste alpha: the region grown by half the feather, then blurred, so the
    transition is centred on the region's border. Clipped to `limit` (where
    new pixels exist and may go) so nothing outside it changes.
    """
    a = region
    if feather_px > 0:
        a = gaussian_blur(dilate(region, feather_px // 2), feather_px / 3.0)
    a = torch.maximum(a, region).clamp(0, 1)
    if limit is not None:
        a = a * limit
    return a


# --------------------------------------------------------------------------
# colour matching
# --------------------------------------------------------------------------

def color_match(img: torch.Tensor, ref: torch.Tensor, mask: torch.Tensor, method: str = "mean_std",
                strength: float = 1.0) -> torch.Tensor:
    """
    Match the colours of img to ref, with statistics taken only where mask > 0.5.
    mean_std: per channel, the std ratio clamped to [0.75, 1.33] so a real change
    of brightness or contrast survives (idea from KleinTiledUpscaler).
    mkl: Monge-Kantorovich linear transfer (Pitie & Kokaram 2007) of the RGB
    covariance, as in the color-matcher library, written here in torch.
    """
    if method == "none" or strength <= 0:
        return img
    sel = mask[0] > 0.5
    if int(sel.sum()) < 16:
        return img
    a = img[0][sel][:, :3].double()        # N x 3
    b = ref[0][sel][:, :3].double()
    ma, mb = a.mean(0), b.mean(0)
    flat = img[..., :3].double().reshape(-1, 3)
    if method == "mean_std":
        sa, sb = a.std(0).clamp_min(1e-4), b.std(0).clamp_min(1e-4)
        ratio = (sb / sa).clamp(0.75, 1.33)
        out = (flat - ma) * ratio + mb
    elif method == "mkl":
        ca = torch.cov(a.T) + torch.eye(3, dtype=a.dtype) * 1e-6
        cb = torch.cov(b.T) + torch.eye(3, dtype=a.dtype) * 1e-6
        ea, va = torch.linalg.eigh(ca)
        ca_h = va @ torch.diag(ea.clamp_min(1e-10).sqrt()) @ va.T
        ca_ih = va @ torch.diag(1.0 / ea.clamp_min(1e-10).sqrt()) @ va.T
        m = ca_h @ cb @ ca_h
        em, vm = torch.linalg.eigh(m)
        m_h = vm @ torch.diag(em.clamp_min(0).sqrt()) @ vm.T
        t = ca_ih @ m_h @ ca_ih
        out = (flat - ma) @ t.T + mb
    else:
        raise ValueError(f"unknown color_match method '{method}'")
    out = out.reshape(img[..., :3].shape).to(img.dtype).clamp(0, 1)
    res = img.clone()
    res[..., :3] = img[..., :3] * (1 - strength) + out * strength
    return res
