"""kubakub post nodes: align to source (align.py), colour match, apply lut, deflicker, retime, burn in (post.py)."""

from __future__ import annotations

import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

import folder_paths
from comfy_api.latest import io

from ...kubakub import align as al
from ...kubakub import post as pp

log = logging.getLogger("KUBA.regions")

CATEGORY = "kubakub/2d/post"


def _frames(images):
    return [images[i, ..., :3].cpu().float().numpy() for i in range(int(images.shape[0]))]


def _map(fn, n):
    """fn(i) for every frame into one numpy batch (threads: numpy / opencv release the GIL)."""
    out = [None] * n
    with ThreadPoolExecutor(max_workers=max(1, min(8, (os.cpu_count() or 4) - 2, n))) as pool:
        for i, r in zip(range(n), pool.map(fn, range(n))):
            out[i] = r
    return torch.from_numpy(np.stack(out).astype(np.float32))


class KUBA_ColourMatch(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="KUBA_ColourMatch",
            display_name="kubakub colour match",
            category="kubakub/2d/post",
            search_aliases=['color match', 'grade', 'look', 'reference', 'lut'],
            description=("Gives your frames the colours of a reference picture (a still you like, a frame of "
                         "another scene). One transform for all frames, so nothing flickers. save_lut writes it as "
                         "a .cube for After Effects, Resolume, MadMapper, Resolve."),
            inputs=[
                io.Image.Input("images", tooltip="The frames to grade."),
                io.Image.Input("reference", tooltip="The look to match (any size, any content)."),
                io.Combo.Input("method", options=["mkl", "mean_std"], default="mkl",
                               tooltip="mkl = colours and their mix (richer); mean_std = per channel brightness and contrast."),
                io.Float.Input("strength", default=1.0, min=0.0, max=1.0, step=0.05),
                io.String.Input("save_lut", default="", placeholder="my_look",
                                tooltip="Name for a .cube file in output/kubakub_luts/ (empty = no file)."),
            ],
            outputs=[io.Image.Output("images"), io.String.Output("lut_file", tooltip="The written .cube (or empty).")],
        )

    @classmethod
    def execute(cls, images, reference, method, strength, save_lut) -> io.NodeOutput:
        fr = _frames(images)
        pick = sorted({0, len(fr) // 2, len(fr) - 1})
        src = np.concatenate([fr[i].reshape(-1, 3) for i in pick])[None]
        fit = pp.colour_fit(src, reference[0, ..., :3].cpu().float().numpy(), method)
        out = _map(lambda i: pp.colour_apply(fr[i], fit, strength), len(fr))
        path = ""
        name = re.sub(r"[^\w\-]+", "_", (save_lut or "").strip())
        if name:
            path = pp.write_cube(os.path.join(folder_paths.get_output_directory(), "kubakub_luts", name + ".cube"),
                                 lambda g: pp.colour_apply(g, fit, strength), title=name)
        return io.NodeOutput(out, path)


def lut_dirs():
    return [os.path.join(folder_paths.get_input_directory(), "luts"), os.path.join(folder_paths.models_dir, "luts"),
            os.path.join(folder_paths.get_output_directory(), "kubakub_luts")]


def list_luts():
    names = []
    for d in lut_dirs():
        if os.path.isdir(d):
            names += [f for f in sorted(os.listdir(d)) if f.lower().endswith(".cube") and f not in names]
    return names or ["(none: put .cube files in input/luts)"]


class KUBA_ApplyLUT(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="KUBA_ApplyLUT",
            display_name="kubakub apply lut",
            category="kubakub/2d/post",
            search_aliases=['lut', 'cube', 'grade', 'look'],
            description="A .cube look (1D or 3D) on the frames. Files from input/luts, models/luts or output/kubakub_luts, or a path.",
            inputs=[
                io.Image.Input("images"),
                io.Combo.Input("lut", options=list_luts(), tooltip="A .cube from input/luts, models/luts, output/kubakub_luts."),
                io.Float.Input("strength", default=1.0, min=0.0, max=1.0, step=0.05),
                io.String.Input("lut_path", default="", optional=True, placeholder="looks/film.cube",
                                tooltip="Any .cube file; wins over the list."),
            ],
            outputs=[io.Image.Output("images")],
        )

    @classmethod
    def execute(cls, images, lut, strength, lut_path="") -> io.NodeOutput:
        path = (lut_path or "").strip().strip('"')
        if not path:
            path = next((os.path.join(d, lut) for d in lut_dirs() if os.path.isfile(os.path.join(d, lut))), "")
        if not os.path.isfile(path):
            raise ValueError(f"kubakub apply lut: no .cube file '{path or lut}'")
        with open(path, encoding="utf-8", errors="replace") as f:
            table = pp.parse_cube(f.read())
        fr = _frames(images)
        return io.NodeOutput(_map(lambda i: pp.apply_lut(fr[i], table, strength), len(fr)))


class KUBA_Deflicker(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="KUBA_Deflicker",
            display_name="kubakub deflicker",
            category="kubakub/2d/post",
            search_aliases=['flicker', 'temporal', 'stabilize brightness', 'smooth'],
            description=("Calms brightness and colour jumps between frames (diffusion video, timelapse): each part "
                         "of the frame is moved to its average over the neighbouring frames. Movement stays sharp."),
            inputs=[
                io.Image.Input("images", tooltip="The frames (at least 3)."),
                io.Int.Input("window", default=9, min=3, max=121, tooltip="Frames to average over; more = calmer, slower changes survive."),
                io.Int.Input("grid", default=8, min=1, max=64, tooltip="1 = the whole frame at once; 8 = local flicker in 8 x 8 zones."),
                io.Float.Input("strength", default=1.0, min=0.0, max=1.0, step=0.05),
            ],
            outputs=[io.Image.Output("images")],
        )

    @classmethod
    def execute(cls, images, window, grid, strength) -> io.NodeOutput:
        fr = _frames(images)
        if len(fr) < 3:
            return io.NodeOutput(images)
        gains = pp.deflicker_gains(fr, int(window), int(grid), float(strength))
        return io.NodeOutput(_map(lambda i: pp.apply_gain(fr[i], gains[i]), len(fr)))


class KUBA_Retime(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="KUBA_Retime",
            display_name="kubakub retime",
            category="kubakub/2d/post",
            search_aliases=['slow motion', 'speed', 'interpolate', 'frame rate', 'optical flow'],
            description=("Changes the speed of frames with in-between frames from optical flow (or a blend): 0.5 = "
                         "half speed, twice the frames; 2 = double speed. Or give the exact number of frames."),
            inputs=[
                io.Image.Input("images"),
                io.Float.Input("speed", default=0.5, min=0.05, max=16.0, step=0.05, tooltip="Below 1 = slow motion."),
                io.Int.Input("frames", default=0, min=0, max=100000, tooltip="Exact output length instead of speed (0 = use speed)."),
                io.Combo.Input("mode", options=["flow", "blend", "nearest"], default="flow",
                               tooltip="flow = moving in-betweens (optical flow); blend = cross-fade; nearest = repeat / drop frames."),
            ],
            outputs=[io.Image.Output("images"), io.Int.Output("frame_count")],
        )

    @classmethod
    def execute(cls, images, speed, frames, mode) -> io.NodeOutput:
        t0 = time.perf_counter()
        fr = _frames(images)
        ts = pp.retime_times(len(fr), float(speed), int(frames))

        def one(k):
            t = float(ts[k])
            i = int(np.floor(t))
            if mode == "nearest" or i >= len(fr) - 1:
                return fr[min(int(round(t)), len(fr) - 1)]
            return pp.between(fr[i], fr[i + 1], t - i, mode)
        out = _map(one, len(ts))
        log.info("[KUBA post] retime %d -> %d frames (%s), %.1f s", len(fr), len(ts), mode, time.perf_counter() - t0)
        return io.NodeOutput(out, len(ts))


class KUBA_BurnIn(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="KUBA_BurnIn",
            display_name="kubakub burn in",
            category="kubakub/2d/post",
            search_aliases=['timecode', 'frame number', 'slate', 'review', 'watermark'],
            description="Frame number, timecode and a name on every frame, for review copies. {name} {frame} {timecode} {total}.",
            inputs=[
                io.Image.Input("images"),
                io.String.Input("text", default="{name}   {frame}   {timecode}", multiline=True,
                                tooltip="{name}, {frame} (from start_frame), {timecode} (hh:mm:ss:ff), {total} = frame count."),
                io.String.Input("name", default="", tooltip="Link it from project settings."),
                io.Float.Input("fps", default=25.0, min=1.0, max=120.0, step=0.001),
                io.Int.Input("start_frame", default=0, min=0, max=10_000_000),
                io.Combo.Input("position", options=["bottom left", "bottom right", "bottom centre", "top left", "top right", "top centre"],
                               default="bottom left"),
                io.Float.Input("size", default=0.03, min=0.005, max=0.2, step=0.005, tooltip="Text height as a share of the frame height."),
                io.Float.Input("opacity", default=0.8, min=0.1, max=1.0, step=0.05),
            ],
            outputs=[io.Image.Output("images")],
        )

    @classmethod
    def execute(cls, images, text, name, fps, start_frame, position, size, opacity) -> io.NodeOutput:
        fr = _frames(images)
        n = len(fr)

        def one(i):
            f = int(start_frame) + i
            t = (text or "").replace("{name}", name or "").replace("{frame}", f"{f:05d}") \
                .replace("{timecode}", pp.timecode(f, float(fps))).replace("{total}", str(n))
            return pp.burn_text(fr[i], t, position, float(size), float(opacity))
        return io.NodeOutput(_map(one, n))


class KUBA_AlignToSource(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="KUBA_AlignToSource",
            display_name="kubakub align to source",
            category="kubakub/2d/post",
            search_aliases=['pixel drift', 'registration', 'shift', 'align', 'qwen edit', 'kontext', 'klein'],
            description=("Puts an edited picture back onto the picture it was made from. An image model that repaints "
                         "a whole facade returns it a few pixels moved or scaled; on a building that shows. Finds the "
                         "small shift and scale on the edges both pictures share and undoes it. When the two have too "
                         "little in common, or the fit is not certain, the picture is returned as it is and the "
                         "report says why. Not needed after the region sampler (it pastes inside masks)."),
            inputs=[
                io.Image.Input("images", tooltip="The edited pictures (one, or the frames of a clip)."),
                io.Image.Input("source", tooltip="What they were made from: the clay, the matrix, the render. One "
                                                 "picture for all, or one per image. Another size is fine: the result "
                                                 "has the size of the source."),
                io.Combo.Input("fit", options=["move and scale", "move only"], default="move and scale",
                               tooltip="move and scale: shift, scale and a little shear (what image models do). "
                                       "move only: a shift, when the size is known to be right."),
                io.Combo.Input("frames", options=["each on its own", "one fit for all"], default="each on its own",
                               optional=True, advanced=True,
                               tooltip="For a batch. one fit for all: the first picture is fitted and every frame gets "
                                       "the same correction (a clip that drifted as a whole stays steady)."),
            ],
            outputs=[io.Image.Output("images", tooltip="The pictures on the source, at its size."),
                     io.String.Output("report", tooltip="Per picture: how far it sat off, or why it was left as it is.")],
        )

    @classmethod
    def execute(cls, images, source, fit, frames="each on its own") -> io.NodeOutput:
        t0 = time.perf_counter()
        model = "affine" if fit == "move and scale" else "shift"
        srcs, imgs = _frames(source), _frames(images)
        H, W = srcs[0].shape[:2]
        results = [None] * len(imgs)

        def one(i):
            src = srcs[i] if i < len(srcs) else srcs[-1]
            if frames == "one fit for all" and i > 0:
                r = results[0]
            else:
                r = al.estimate(src, imgs[i], model)
            results[i] = r
            return al.apply(imgs[i], r["matrix"], (W, H)).astype(np.float32)

        first = one(0)                                   # the first one alone: 'one fit for all' needs its result
        rest = _map(lambda k: one(k + 1), len(imgs) - 1) if len(imgs) > 1 else None
        out = torch.from_numpy(first)[None] if rest is None else torch.cat([torch.from_numpy(first)[None], rest])
        moved = sum(1 for r in results if r["ok"] and r["why"] == "aligned")
        lines = [f"{moved} of {len(imgs)} pictures moved back onto the source, {time.perf_counter() - t0:.1f} s"]
        lines += [f"{i + 1}: {al.describe(r)}" for i, r in enumerate(results[:24])]
        if len(results) > 24:
            lines.append(f"... and {len(results) - 24} more")
        report = "\n".join(lines)
        log.info("[KUBA post] align to source: %s", lines[0])
        return io.NodeOutput(out, report)


NODE_CLASS_MAPPINGS = {"KUBA_AlignToSource": KUBA_AlignToSource, "KUBA_ColourMatch": KUBA_ColourMatch, "KUBA_ApplyLUT": KUBA_ApplyLUT,
                       "KUBA_Deflicker": KUBA_Deflicker, "KUBA_Retime": KUBA_Retime, "KUBA_BurnIn": KUBA_BurnIn}
NODE_DISPLAY_NAME_MAPPINGS = {"KUBA_AlignToSource": "kubakub align to source", "KUBA_ColourMatch": "kubakub colour match", "KUBA_ApplyLUT": "kubakub apply lut",
                              "KUBA_Deflicker": "kubakub deflicker", "KUBA_Retime": "kubakub retime",
                              "KUBA_BurnIn": "kubakub burn in"}
