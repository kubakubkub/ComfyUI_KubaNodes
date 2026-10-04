"""
MosaicIllusion v2 for ComfyUI_KubaNodes

Changes from v1
  - the target is handled automatically. Any size, any aspect, any framing.
    Dark border trimmed, levels stretched, fitted to the texture, blurred.
  - adaptive tiles. A quadtree splits cells where the motif has detail and
    leaves them large where it is flat, so the grid follows the image.
  - mortar is drawn on the real cell outlines, so it works with either grid.
  - third output shows the grid itself.

adaptive = false reproduces v1 behaviour.
"""

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------- helpers

def _to_bchw(img):
    return img.permute(0, 3, 1, 2)


def _to_bhwc(t):
    return t.permute(0, 2, 3, 1)


def _luma(t):
    if t.shape[1] == 1:
        return t
    return 0.2126 * t[:, 0:1] + 0.7152 * t[:, 1:2] + 0.0722 * t[:, 2:3]


def _gauss(t, radius):
    if radius <= 0:
        return t
    k = int(radius) * 2 + 1
    sig = max(radius / 2.0, 0.1)
    x = torch.arange(k, dtype=torch.float32, device=t.device) - (k - 1) / 2.0
    g = torch.exp(-(x ** 2) / (2 * sig * sig))
    g = g / g.sum()
    c = t.shape[1]
    t = F.pad(t, (k // 2, k // 2, 0, 0), mode="reflect")
    t = F.conv2d(t, g.view(1, 1, 1, k).expand(c, 1, 1, k), groups=c)
    t = F.pad(t, (0, 0, k // 2, k // 2), mode="reflect")
    t = F.conv2d(t, g.view(1, 1, k, 1).expand(c, 1, k, 1), groups=c)
    return t


def _auto_crop(t, thresh=0.04):
    l = _luma(t)[0, 0]
    rows = (l.max(dim=1).values > thresh).nonzero()
    cols = (l.max(dim=0).values > thresh).nonzero()
    if rows.numel() < 2 or cols.numel() < 2:
        return t
    y0, y1 = int(rows[0]), int(rows[-1]) + 1
    x0, x1 = int(cols[0]), int(cols[-1]) + 1
    if (y1 - y0) < 8 or (x1 - x0) < 8:
        return t
    return t[:, :, y0:y1, x0:x1]


def _auto_levels(t, lo_pct=1.0, hi_pct=99.0):
    l = t.flatten()
    if l.numel() > 2000000:
        l = l[torch.randperm(l.numel(), device=l.device)[:2000000]]
    lo = torch.quantile(l, lo_pct / 100.0)
    hi = torch.quantile(l, hi_pct / 100.0)
    if float(hi - lo) < 1e-4:
        return t
    return torch.clamp((t - lo) / (hi - lo), 0.0, 1.0)


def _fit(t, H, W, mode):
    _, c, h, w = t.shape
    if mode == "stretch":
        return F.interpolate(t, size=(H, W), mode="bilinear", align_corners=False)
    s = max(H / h, W / w) if mode == "cover" else min(H / h, W / w)
    nh, nw = max(1, int(round(h * s))), max(1, int(round(w * s)))
    t = F.interpolate(t, size=(nh, nw), mode="bilinear", align_corners=False)
    if mode == "cover":
        y0 = max(0, (nh - H) // 2)
        x0 = max(0, (nw - W) // 2)
        return t[:, :, y0:y0 + H, x0:x0 + W]
    out = torch.full((1, c, H, W), 0.5, device=t.device, dtype=t.dtype)
    y0, x0 = (H - nh) // 2, (W - nw) // 2
    out[:, :, y0:y0 + nh, x0:x0 + nw] = t
    return out


def _quadtree_ids(tv, base, min_cells, detail, max_depth):
    """tv is a cpu (H, W) tensor. Returns an int32 id map and the cell count."""
    H, W = tv.shape
    ids = torch.zeros((H, W), dtype=torch.int32)
    nxt = 0
    stack = []
    for y in range(0, H, base):
        for x in range(0, W, base):
            stack.append((y, x, min(base, H - y), min(base, W - x), 0))
    while stack:
        y, x, h, w, d = stack.pop()
        cell = tv[y:y + h, x:x + w]
        if (d < max_depth and h >= min_cells * 2 and w >= min_cells * 2
                and float(cell.std()) > detail):
            hh, ww = h // 2, w // 2
            stack.append((y, x, hh, ww, d + 1))
            stack.append((y, x + ww, hh, w - ww, d + 1))
            stack.append((y + hh, x, h - hh, ww, d + 1))
            stack.append((y + hh, x + ww, h - hh, w - ww, d + 1))
        else:
            ids[y:y + h, x:x + w] = nxt
            nxt += 1
    return ids, nxt


def _scatter_mean(src, ids, n):
    """src (C, H, W), ids (H, W) -> per cell mean painted back at full size."""
    C = src.shape[0]
    flat = src.reshape(C, -1)
    idx = ids.reshape(-1).long()
    sums = torch.zeros((C, n), dtype=flat.dtype, device=flat.device)
    sums.index_add_(1, idx, flat)
    cnt = torch.zeros(n, dtype=flat.dtype, device=flat.device)
    cnt.index_add_(0, idx, torch.ones(idx.numel(), dtype=flat.dtype,
                                      device=flat.device))
    means = sums / torch.clamp(cnt, min=1.0)
    return means[:, idx].reshape(src.shape)


def _edge_noise(ids, tile_px, amount, scale, seed):
    """
    Displace the cell lookup with low frequency noise so the boundaries
    wander and chip, the way hand-cut tesserae do. Cell count is unchanged,
    only the outlines move.
    """
    if amount <= 0.0:
        return ids
    H, W = ids.shape
    dev = ids.device
    sc = max(2.0, float(scale))
    nh, nw = max(2, int(H / sc)), max(2, int(W / sc))
    g = torch.Generator(device="cpu").manual_seed(int(seed) + 7717)
    n = (torch.rand((1, 2, nh, nw), generator=g) * 2.0 - 1.0).to(dev)
    n = F.interpolate(n, size=(H, W), mode="bilinear", align_corners=False)
    amp = float(amount) * float(tile_px)
    yy = torch.arange(H, device=dev).view(H, 1).expand(H, W)
    xx = torch.arange(W, device=dev).view(1, W).expand(H, W)
    sy = torch.clamp(yy + n[0, 0] * amp, 0, H - 1).long()
    sx = torch.clamp(xx + n[0, 1] * amp, 0, W - 1).long()
    return ids[sy, sx]


def _select(base, tvq, k, strength, dither, level_radius, tile_px, seed):
    """
    Palette selection. Build a tray of k colours out of the texture itself,
    then for every cell pick the tile whose value the picture needs, keeping
    that cell's deviation from its neighbours so the material's own pattern
    survives. Nothing is tinted, every tile in the result is a real tile.
    """
    C, H, W = base.shape
    dev = base.device
    L = _luma(base.unsqueeze(0))[0, 0]

    flatL = L.reshape(-1)
    flatC = base.reshape(C, -1)
    qs = torch.linspace(0.02, 0.98, k, device=dev)
    samp = flatL
    if samp.numel() > 500000:
        samp = samp[torch.randperm(samp.numel(), device=dev)[:500000]]
    targets = torch.quantile(samp, qs)
    idxs = torch.argmin((flatL.unsqueeze(0) - targets.unsqueeze(1)).abs(), dim=1)
    pal = flatC[:, idxs]
    palL = (0.2126 * pal[0] + 0.7152 * pal[1] + 0.0722 * pal[2])
    order = torch.argsort(palL)
    pal = pal[:, order]
    palL = palL[order]

    lo, hi = float(palL.min()), float(palL.max())
    want = tvq[0] * (hi - lo) + lo
    Lb = _gauss(L.view(1, 1, H, W), max(1.0, level_radius * tile_px / 4.0))[0, 0]
    D = L + strength * (want - Lb)

    if dither > 0.0:
        g = torch.Generator(device="cpu").manual_seed(int(seed) + 4243)
        step = (hi - lo) / max(k - 1, 1)
        nz = (torch.rand((H, W), generator=g) - 0.5).to(dev) * dither * step
        D = D + nz

    pick = torch.argmin((D.reshape(-1, 1) - palL.reshape(1, k)).abs(), dim=1)
    return pal[:, pick].reshape(C, H, W)


def _cell_borders(ids):
    e = torch.zeros_like(ids, dtype=torch.bool)
    ne = ids[:, :-1] != ids[:, 1:]
    e[:, :-1] |= ne
    e[:, 1:] |= ne
    ne = ids[:-1, :] != ids[1:, :]
    e[:-1, :] |= ne
    e[1:, :] |= ne
    e[0, :] = True
    e[-1, :] = True
    e[:, 0] = True
    e[:, -1] = True
    return e


# ---------------------------------------------------------------- node

class MosaicIllusion:
    """Tile a texture and steer each tile toward a target image."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "texture": ("IMAGE", {"tooltip": "The material photo (stone, tiles, "
                               "bricks) the mosaic is cut from. Sets the output "
                               "size; a batch gives one mosaic per image."}),
                "target": ("IMAGE", {"tooltip": "The picture hidden in the "
                               "mosaic. Only its brightness is used (first image "
                               "of a batch)."}),
                "tile_px": ("INT", {"default": 48, "min": 2, "max": 2048,
                    "tooltip": "Tile size in pixels. With adaptive on this is "
                               "the largest tile."}),
                "strength": ("FLOAT", {"default": 0.55, "min": 0.0, "max": 1.0,
                    "step": 0.01,
                    "tooltip": "0 = material only, 1 = target fully readable. "
                               "Animate to move the resolving distance."}),
                "mode": (["select", "multiply", "mix", "lift"], {"default": "select",
                    "tooltip": "select picks the tile from the material's own "
                               "palette whose value the picture needs, the way "
                               "a mosaicist picks from a tray. Nothing is "
                               "tinted. multiply darkens / brightens the tiles, "
                               "mix blends them toward the target grey, lift "
                               "only brightens where the target is light."}),
                "quantise_texture": ("BOOLEAN", {"default": True,
                    "tooltip": "Flatten each tile to one colour. Turn off when "
                               "the texture is already tiled."}),
            },
            "optional": {
                "target_fit": (["cover", "contain", "stretch"],
                               {"default": "cover",
                    "tooltip": "How the target fits the texture size. cover "
                               "fills and crops, contain fits it whole on mid "
                               "grey, stretch distorts it to fit."}),
                "target_autocrop": ("BOOLEAN", {"default": True,
                    "tooltip": "Trim a dark border so a cutout fills the frame."}),
                "target_autolevels": ("BOOLEAN", {"default": True,
                    "tooltip": "Stretch the target's levels to full black to "
                               "white, so a flat photo still reads."}),
                "target_blur": ("INT", {"default": 6, "min": 0, "max": 200,
                    "tooltip": "Soften the target so it emerges instead of "
                               "stamping a silhouette."}),
                "target_invert": ("BOOLEAN", {"default": False,
                    "tooltip": "Swap light and dark in the target."}),
                "contrast": ("FLOAT", {"default": 1.2, "min": 0.0, "max": 4.0,
                                       "step": 0.05,
                    "tooltip": "Contrast of the target around mid grey. 1 = "
                               "unchanged, higher makes the hidden picture "
                               "punchier."}),
                "adaptive": ("BOOLEAN", {"default": False,
                    "tooltip": "Split tiles where the motif has detail, keep "
                               "them large where it is flat."}),
                "min_tile_px": ("INT", {"default": 8, "min": 2, "max": 512,
                    "tooltip": "adaptive only. Smallest tile in pixels; a tile "
                               "is not split below this."}),
                "detail": ("FLOAT", {"default": 0.06, "min": 0.005, "max": 0.5,
                    "step": 0.005, "tooltip": "Lower splits more."}),
                "max_depth": ("INT", {"default": 3, "min": 0, "max": 6,
                    "tooltip": "adaptive only. How many times a tile may be "
                               "halved (3 = down to 1/8 of tile_px)."}),
                "edge_noise": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 0.6,
                    "step": 0.01,
                    "tooltip": "Roughness of the tile outlines, as a fraction "
                               "of tile size. 0.10 to 0.20 looks hand cut."}),
                "edge_scale": ("FLOAT", {"default": 12.0, "min": 2.0,
                    "max": 200.0, "step": 1.0,
                    "tooltip": "Noise wavelength in pixels. Small is jagged "
                               "and chipped, large is wobbly."}),
                "mortar": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 0.45,
                                     "step": 0.01,
                    "tooltip": "Width of the joints between tiles, as a "
                               "fraction of tile size. 0 = no joints."}),
                "mortar_value": ("FLOAT", {"default": 0.0, "min": 0.0,
                                           "max": 1.0, "step": 0.01,
                    "tooltip": "Grey level of the joints: 0 = black, "
                               "1 = white."}),
                "jitter": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 0.5,
                                     "step": 0.01,
                    "tooltip": "Random change of each tile's target value, so "
                               "the hidden picture looks less mechanical. "
                               "0 = off."}),
                "palette_size": ("INT", {"default": 12, "min": 2, "max": 64,
                    "tooltip": "select mode. How many tile colours are in "
                               "the tray. Taken from the texture itself."}),
                "dither": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 3.0,
                    "step": 0.05,
                    "tooltip": "select mode. Breaks banding at the edges "
                               "of the hidden shape."}),
                "level_radius": ("FLOAT", {"default": 6.0, "min": 1.0,
                    "max": 80.0, "step": 1.0,
                    "tooltip": "select mode. Neighbourhood, in tiles, "
                               "whose level gets shifted. Larger keeps more "
                               "of the pattern."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFF,
                    "tooltip": "Random seed for jitter, edge noise and dither. "
                               "Same seed = same mosaic."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("image", "tile_values", "grid")
    OUTPUT_TOOLTIPS = ("The mosaic.",
                       "The grey value each tile was asked for (the target as "
                       "the tiles see it).",
                       "The tile outlines: white tiles, black borders.")
    DESCRIPTION = ("Cuts a texture photo into mosaic tiles and picks or shades "
                   "each tile so a target picture appears when seen from a "
                   "distance. The tiles stay real material; strength sets how "
                   "readable the hidden picture is.")
    FUNCTION = "build"
    CATEGORY = "kubakub/lab/mosaic"

    def build(self, texture, target, tile_px, strength, mode, quantise_texture,
              target_fit="cover", target_autocrop=True, target_autolevels=True,
              target_blur=6, target_invert=False, contrast=1.2,
              adaptive=False, min_tile_px=8, detail=0.06, max_depth=3,
              edge_noise=0.0, edge_scale=12.0,
              palette_size=12, dither=0.9, level_radius=6.0,
              mortar=0.0, mortar_value=0.0, jitter=0.0, seed=0):

        tex = _to_bchw(texture).float()
        B, C, H, W = tex.shape
        dev = tex.device

        # target, handled automatically
        tgt = _to_bchw(target).float()[:1].to(dev)
        if target_autocrop:
            tgt = _auto_crop(tgt)
        tgt = _fit(tgt, H, W, target_fit)
        tv = _luma(tgt)
        if target_autolevels:
            tv = _auto_levels(tv)
        if target_invert:
            tv = 1.0 - tv
        tv = _gauss(tv, target_blur)
        if contrast != 1.0:
            tv = torch.clamp((tv - 0.5) * contrast + 0.5, 0.0, 1.0)

        # the grid
        if adaptive:
            ids, n = _quadtree_ids(tv[0, 0].detach().cpu(), int(tile_px),
                                   int(min_tile_px), float(detail),
                                   int(max_depth))
            ids = ids.to(dev)
        else:
            ny = max(1, int(round(H / float(tile_px))))
            nx = max(1, int(round(W / float(tile_px))))
            gy = (torch.arange(H, device=dev) * ny // H).view(H, 1)
            gx = (torch.arange(W, device=dev) * nx // W).view(1, W)
            ids = (gy * nx + gx).to(torch.int32)
            n = ny * nx

        ids = _edge_noise(ids, tile_px, edge_noise, edge_scale, seed)

        tvq = _scatter_mean(tv[0], ids, n)[0:1]

        if jitter > 0.0:
            g = torch.Generator(device="cpu").manual_seed(int(seed))
            noise = ((torch.rand(n, generator=g) - 0.5) * 2.0 * jitter).to(dev)
            tvq = torch.clamp(
                tvq + noise[ids.reshape(-1).long()].reshape(1, H, W), 0.0, 1.0)

        outs = []
        for b in range(B):
            base = _scatter_mean(tex[b], ids, n) if quantise_texture else tex[b]
            if mode == "select":
                outs.append(_select(base, tvq, int(palette_size), strength,
                                    float(dither), float(level_radius),
                                    float(tile_px), int(seed) + b))
                continue
            if mode == "multiply":
                o = base * (1.0 - strength + strength * 2.0 * tvq)
            elif mode == "lift":
                o = base + strength * torch.clamp(tvq - 0.5, min=0.0) * 2.0
            else:
                o = base * (1.0 - strength) + tvq.expand_as(base) * strength
            outs.append(torch.clamp(o, 0.0, 1.0))
        out = torch.stack(outs, 0)

        if mortar > 0.0:
            edge = _cell_borders(ids)
            w = max(0, int(round(mortar * float(tile_px) * 0.5)) - 1)
            if w > 0:
                e = edge.float().view(1, 1, H, W)
                e = F.max_pool2d(e, kernel_size=w * 2 + 1, stride=1, padding=w)
                edge = e[0, 0] > 0.5
            m = (~edge).float().view(1, 1, H, W)
            out = out * m + mortar_value * (1.0 - m)

        vis = tvq.expand(3, -1, -1).unsqueeze(0)
        grid = (~_cell_borders(ids)).float().view(1, 1, H, W).expand(1, 3, H, W)
        return (_to_bhwc(out), _to_bhwc(vis), _to_bhwc(grid.contiguous()))


class MosaicGridFromCamera:
    """Derive tile_px from real dimensions instead of guessing."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "facade_width_m": ("FLOAT", {"default": 33.93, "min": 0.1,
                                         "max": 500.0, "step": 0.01,
                "tooltip": "Real width of the facade the image covers, in "
                           "metres."}),
            "feature_width_m": ("FLOAT", {"default": 0.10, "min": 0.001,
                                          "max": 50.0, "step": 0.001,
                "tooltip": "Real width of one tile on the building, in metres "
                           "(e.g. a brick or a stone)."}),
            "render_width_px": ("INT", {"default": 3200, "min": 8, "max": 16384,
                "tooltip": "Width in pixels of the image the mosaic is made "
                           "at."}),
            "divide": ("INT", {"default": 1, "min": 1, "max": 16,
                "tooltip": "Split each real feature into this many tiles "
                           "across. 1 = one tile per feature."}),
        }}

    RETURN_TYPES = ("INT", "STRING")
    RETURN_NAMES = ("tile_px", "info")
    OUTPUT_TOOLTIPS = ("Tile size in pixels, for the mosaic illusion's "
                       "tile_px.",
                       "Pixels per metre, tile size and tiles across, as text.")
    DESCRIPTION = ("Works out the mosaic tile size in pixels from real sizes: "
                   "the facade width, the width of one tile on the building "
                   "and the image width. No camera is involved.")
    FUNCTION = "calc"
    CATEGORY = "kubakub/lab/mosaic"

    def calc(self, facade_width_m, feature_width_m, render_width_px, divide):
        ppm = render_width_px / float(facade_width_m)
        tile = max(2, int(round(feature_width_m * ppm / divide)))
        across = int(round(render_width_px / float(tile)))
        return (tile, "{:.1f} px/m, tile {} px, {} across".format(
            ppm, tile, across))


NODE_CLASS_MAPPINGS = {
    "MosaicIllusion": MosaicIllusion,
    "MosaicGridFromCamera": MosaicGridFromCamera,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "MosaicIllusion": "kubakub mosaic illusion",
    "MosaicGridFromCamera": "kubakub mosaic grid from camera",
}
