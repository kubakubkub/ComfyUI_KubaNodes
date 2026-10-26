"""kubakub crop and kubakub uncrop: a crop drawn on the picture (web/kubakub_crop.js), and the edited crop pasted back.

The box, ratio, size and fill are set on the node's picture with the mouse and pills; their widgets are only where
the drawing keeps them (hidden by the web file, so a workflow without it still runs). The logic is kubakub/crop.py.
"""

from __future__ import annotations

import hashlib
import logging
import os
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import torch

import folder_paths
from comfy_api.latest import io

from ...kubakub import crop as cr

log = logging.getLogger("KUBA.regions")

CATEGORY = "kubakub/utils"
CropBoxType = io.Custom("KUBA_CROP_BOX")             # crop.crop_info(): the box in pixels of the picture it was cut from
PREVIEW_SIDE = 1600


def save_preview(frame: np.ndarray, folder: str, side: int = PREVIEW_SIDE) -> dict:
    """One frame (h, w, 3) float as a JPEG in ComfyUI's temp folder (same pixels = same file) -> /view arguments."""
    h, w = frame.shape[:2]
    k = min(1.0, side / max(h, w))
    small = cv2.resize(frame, (max(1, round(w * k)), max(1, round(h * k))), interpolation=cv2.INTER_AREA) if k < 1 else frame
    u8 = (np.clip(small[..., :3], 0, 1) * 255 + 0.5).astype(np.uint8)
    name = hashlib.sha1(u8.tobytes()).hexdigest()[:20] + ".jpg"
    out = os.path.join(folder_paths.get_temp_directory(), folder)
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, name)
    if not os.path.isfile(path):
        cv2.imwrite(path, cv2.cvtColor(u8, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])
    return {"filename": name, "subfolder": folder, "type": "temp"}


def _each(fn, n):
    with ThreadPoolExecutor(max_workers=max(1, min(8, (os.cpu_count() or 4) - 2, n))) as pool:
        return list(pool.map(fn, range(n)))


def _resize(t: torch.Tensor, w: int, h: int) -> torch.Tensor:
    """(n, h0, w0, c) -> (n, h, w, c): area down, bicubic up (antialiased)."""
    if t.shape[1:3] == (h, w):
        return t
    x = t.movedim(-1, 1)
    down = w < t.shape[2] and h < t.shape[1]
    x = torch.nn.functional.interpolate(x, size=(h, w), mode="area") if down else \
        torch.nn.functional.interpolate(x, size=(h, w), mode="bicubic", align_corners=False, antialias=True)
    return x.clamp(0, 1).movedim(1, -1)


class KUBA_Crop(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="KUBA_Crop",
            display_name="kubakub crop",
            category=CATEGORY,
            search_aliases=["crop", "cut", "trim", "canvas size", "pad", "extend", "outpaint", "aspect ratio", "frame"],
            description=("Crop by drawing on the picture, as in Photoshop: drag a box, drag inside it to move it, the "
                         "handles to size it (Shift keeps the ratio, Alt sizes from the centre), double click = all of "
                         "it. Pick a ratio on the pills (original, 'picture' = the ratio of a connected picture, e.g. "
                         "the facade matrix), and the size that comes out. Drag past the edge to add canvas: the new "
                         "area is filled (edge, mirror, black, white, grey) and comes out as its own mask, for an "
                         "outpaint. A batch or video is cut the same on every frame. The box is kept as a part of the "
                         "picture, so it stays right when the picture comes at another size."),
            inputs=[
                io.Image.Input("image", tooltip="The picture (or the frames of a video) to crop."),
                io.String.Input("box", default="0 0 1 1",
                                tooltip="The box, drawn on the picture (kept as 0-1 of it). You never need to type it."),
                io.Combo.Input("ratio", options=cr.RATIOS, default="free",
                               tooltip="The box's shape (the pills on the picture)."),
                io.Combo.Input("size", options=cr.SIZES, default="as cut",
                               tooltip="What comes out: the pixels as cut, or scaled to this long side."),
                io.Boolean.Input("model_safe", default=True,
                                 tooltip="Width and height in steps of 16 px (what Flux and most models want)."),
                io.Combo.Input("fill", options=cr.FILLS, default="edge",
                               tooltip="What fills the canvas added past the picture's edge."),
                io.Image.Input("ratio_of", optional=True,
                               tooltip="A picture whose shape the box takes (the 'picture' pill), e.g. the facade matrix."),
            ],
            outputs=[
                io.Image.Output("image", tooltip="The crop (every frame)."),
                io.Mask.Output("box_mask", tooltip="White where the box lies on the picture (the picture's size)."),
                io.Mask.Output("new_area", tooltip="White on the canvas added past the picture's edge (the crop's size): "
                                                  "the mask for an outpaint. All black when nothing was added."),
                CropBoxType.Output("crop_box", tooltip="Where the crop came from: for kubakub uncrop."),
            ],
        )

    @classmethod
    def execute(cls, image, box="0 0 1 1", ratio="free", size="as cut", model_safe=True, fill="edge",
                ratio_of=None) -> io.NodeOutput:
        n, H, W = int(image.shape[0]), int(image.shape[1]), int(image.shape[2])
        of = (int(ratio_of.shape[2]), int(ratio_of.shape[1])) if ratio_of is not None else None
        if ratio_of is not None and ratio == "free":
            ratio = "picture"                            # a connected 'ratio of' always shapes the box
        snap = 16 if model_safe else 1
        r = cr.ratio_value(ratio, W, H, of)
        info = cr.crop_info(cr.pixel_box(cr.parse_box(box), W, H, r, snap), W, H)
        if cr.padded(info):
            frames = image.cpu().float().numpy()
            out = torch.from_numpy(np.stack(_each(lambda i: cr.cut(frames[i], info, fill), n)))
        else:
            out = image[:, info["y"]:info["y"] + info["h"], info["x"]:info["x"] + info["w"]]
        ow, oh = cr.long_side(info["w"], info["h"], size, snap)
        out = _resize(out.float(), ow, oh)
        new = torch.from_numpy(cr.fit_to(cr.new_area(info), ow, oh))[None]
        log.info("[KUBA crop] %dx%d -> box %d,%d %dx%d -> %dx%d%s, %d frame(s)", W, H, info["x"], info["y"],
                 info["w"], info["h"], ow, oh, " (padded: " + fill + ")" if cr.padded(info) else "", n)
        shown = dict(save_preview(image[0].cpu().float().numpy(), "kubakub_crop"), W=W, H=H,
                     ratio_of=list(of) if of else None, px=[info["x"], info["y"], info["w"], info["h"]], out=[ow, oh])
        return io.NodeOutput(out, torch.from_numpy(cr.box_mask(info))[None], new, info, ui={"kubakub_crop": [shown]})


class KUBA_Uncrop(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="KUBA_Uncrop",
            display_name="kubakub uncrop",
            category=CATEGORY,
            search_aliases=["paste back", "uncrop", "stitch", "composite", "crop back"],
            description=("Puts an edited crop back where kubakub crop cut it from: scaled to the box, with a soft "
                         "edge (none where the box lies on the picture's border). Only the part over the picture comes "
                         "back (canvas added past the edge is left out). The original may come at another size: the "
                         "box goes to the same spot of it. A mask limits the paste (white = paste)."),
            inputs=[
                io.Image.Input("original", tooltip="The picture the crop was cut from (or the same at another size)."),
                io.Image.Input("crop", tooltip="The edited crop (any size: it is scaled to the box)."),
                CropBoxType.Input("crop_box", tooltip="From kubakub crop."),
                io.Int.Input("soft_edge", default=16, min=0, max=512, step=1, display_mode=io.NumberDisplay.slider,
                             tooltip="How wide the blend is at the box's edges, in pixels of the original."),
                io.Mask.Input("mask", optional=True,
                              tooltip="Paste only where this is white: a mask of the crop, or of the whole original."),
            ],
            outputs=[io.Image.Output("image")],
        )

    @classmethod
    def execute(cls, original, crop, crop_box, soft_edge=16, mask=None) -> io.NodeOutput:
        if not isinstance(crop_box, dict) or crop_box.get("format") != cr.CROP_FORMAT:
            raise ValueError("kubakub uncrop: 'crop_box' must come from kubakub crop.")
        orig, piece = original.cpu().float().numpy(), crop.cpu().float().numpy()
        masks = None if mask is None else mask.reshape((-1,) + tuple(mask.shape[-2:])).cpu().float().numpy()
        n = max(len(orig), len(piece))

        def one(i):
            m = None if masks is None else masks[min(i, len(masks) - 1)]
            return cr.paste(orig[min(i, len(orig) - 1)], piece[min(i, len(piece) - 1)], crop_box, soft_edge, m)
        return io.NodeOutput(torch.from_numpy(np.stack(_each(one, n))))


NODE_CLASS_MAPPINGS = {"KUBA_Crop": KUBA_Crop, "KUBA_Uncrop": KUBA_Uncrop}
NODE_DISPLAY_NAME_MAPPINGS = {"KUBA_Crop": "kubakub crop", "KUBA_Uncrop": "kubakub uncrop"}
