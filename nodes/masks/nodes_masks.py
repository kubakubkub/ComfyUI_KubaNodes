"""kubakub mask field and kubakub mask animate: masks as grey ramps over regions, and masks that move
(logic in kubakub/maskfields.py; the curve on the animate node is web/kubakub_mask_animate.js)."""

from __future__ import annotations

import logging
import os
import random

import numpy as np
import torch

import comfy.model_management
import comfy.utils
import folder_paths
from comfy_api.latest import io, ui

from ...kubakub import maskfields as mf
from ...kubakub import sound as so
from ...kubakub.io_types import RegionsType

log = logging.getLogger("KUBA.masks")
PREVIEW_PX, PREVIEW_FRAMES = 384, 240


def _np(mask):
    m = mask.float().cpu().numpy()
    return m[None] if m.ndim == 2 else m


class KUBA_MaskField(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="KUBA_MaskField",
            display_name="kubakub mask field",
            category="kubakub/2d/masks",
            search_aliases=["gradient", "ramp", "distance", "wipe", "reveal", "stagger", "matte"],
            description=(
                "Regions as a grey ramp instead of on / off: black is where a move starts, white where it ends. "
                "The distance from each region's edge, a direction, the distance from a point, one step per region "
                "in an order, or noise. kubakub mask animate runs a reveal, a band or rings along it; it also "
                "works as a soft mask or a time offset in any other tool. Nothing connected = the windows of the "
                "sample facade."),
            inputs=[
                RegionsType.Input("regions", optional=True,
                                  tooltip="From any kubakub regions node. Empty (and no mask) = the windows of the sample facade."),
                io.Mask.Input("mask", optional=True,
                              tooltip="Instead of regions: a mask or a mask batch. Each mask is a region; with "
                                      "split_shapes each separate shape is one."),
                io.String.Input("select", default="*",
                                tooltip="Which regions, in plan syntax: 'W_F1_*', 'group:Windows', 'tag:front', "
                                        "comma = or, space = and, '!' = not. * = all."),
                io.Combo.Input("field", options=list(mf.FIELDS), default="edge distance",
                               tooltip="edge distance: black on the edge of a region, white at its deepest point. "
                                       "direction: black to white along 'angle'. radial: black at the centre, white "
                                       "farthest out. region order: one flat grey per region, in 'order'. noise: "
                                       "soft random greys."),
                io.Combo.Input("per", options=list(mf.PER), default="each region",
                               tooltip="each region: every region runs from black to white on its own (all windows "
                                       "open at once). all together: one ramp over all of them (a wipe across the "
                                       "facade). No effect on region order and noise."),
                io.Float.Input("angle", default=0.0, min=-360.0, max=360.0, step=1.0,
                               tooltip="direction: degrees. 0 = left to right, 90 = top to bottom, 180 = right to left."),
                io.Combo.Input("order", options=list(mf.ORDERS), default="left",
                               tooltip="region order: which region comes first. left / right / top / bottom = from "
                                       "that side; centre = from the middle out; outside = towards the middle; "
                                       "size = the biggest first; random."),
                io.Float.Input("centre_x", default=0.5, min=-1.0, max=2.0, step=0.01,
                               tooltip="radial, all together: the centre, 0 = left edge of the picture, 1 = right edge."),
                io.Float.Input("centre_y", default=0.5, min=-1.0, max=2.0, step=0.01,
                               tooltip="radial, all together: the centre, 0 = top edge, 1 = bottom edge."),
                io.Float.Input("noise_px", default=64.0, min=2.0, max=4096.0, step=1.0,
                               tooltip="noise: the size of the blobs in pixels."),
                io.Int.Input("seed", default=1, min=0, max=2 ** 31 - 1, control_after_generate=False,
                             tooltip="noise and region order = random: another pattern."),
                io.Boolean.Input("invert", default=False, tooltip="Black and white swapped: the move runs the other way."),
                io.Boolean.Input("split_shapes", default=True, optional=True, advanced=True,
                                 tooltip="mask input: on = every separate shape is its own region; off = one region per mask."),
            ],
            outputs=[
                io.Mask.Output("field", tooltip="The grey ramp, 0 to 1 inside the regions, 0 outside."),
                io.Mask.Output("mask", tooltip="1 on the regions the field covers. Connect it to 'mask' of kubakub mask animate."),
                io.String.Output("report", tooltip="How many regions, the size, and what was used when nothing was connected."),
            ],
        )

    @classmethod
    def execute(cls, regions=None, mask=None, select="*", field="edge distance", per="each region", angle=0.0,
                order="left", centre_x=0.5, centre_y=0.5, noise_px=64.0, seed=1, invert=False,
                split_shapes=True) -> io.NodeOutput:
        notes = []
        if regions is not None:
            from ...kubakub.director.render import select_regions
            ids = [r["region_id"] for r in regions.table.get("regions", [])]
            if select.strip() not in ("", "*"):
                ids = select_regions(regions.table, select)
                if not ids:
                    raise ValueError(f"kubakub mask field: select '{select.strip()}' matches no region.")
            lab, n = mf.compact(regions.labels[0].cpu().numpy(), ids)
            if mask is not None:
                notes.append("regions and mask are both connected: the regions are used")
        elif mask is not None:
            lab, n = mf.from_masks(_np(mask), bool(split_shapes))
            if n == 0:
                raise ValueError("kubakub mask field: the mask is empty.")
        else:
            lab, n, _ = mf.sample_regions()
            notes.append("nothing connected: the windows of the sample facade are used (connect regions or a mask)")
        f, inside = mf.field(lab, n, field, per, float(angle), (float(centre_x), float(centre_y)), order, int(seed),
                             float(noise_px), bool(invert))
        report = "\n".join([f"{n} regions, {lab.shape[1]}x{lab.shape[0]}, field = {field}"
                            + (f" ({per})" if field not in ("region order", "noise") else "")] + notes)
        log.info("[KUBA mask field] %s", report.replace("\n", "\n    "))
        out = torch.from_numpy(f)[None]
        return io.NodeOutput(out, torch.from_numpy(inside.astype(np.float32))[None], report, ui=ui.PreviewMask(out, cls=cls))


class KUBA_MaskAnimate(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="KUBA_MaskAnimate",
            display_name="kubakub mask animate",
            category="kubakub/2d/masks",
            search_aliases=["wipe", "reveal", "strobe", "trail", "rotate", "move", "lfo", "sine", "audio reactive", "vj"],
            description=(
                "A mask that moves: one mask per frame. The effect says what moves (a reveal, a band or rings along "
                "a field from kubakub mask field; or the mask itself: move, rotate, scale, opacity). The curve says "
                "how it runs over time: ramp, saw, triangle, sine, square (strobe), random, noise, or a sound (its "
                "level, beats, hits). 'trail' leaves a fading trace. Chain two of these nodes for two moves at "
                "once (masks into mask). The curve is drawn on the node. Nothing connected = a sample."),
            inputs=[
                io.Mask.Input("mask", optional=True,
                              tooltip="What is animated: a mask, or a mask batch (one per frame, e.g. from another "
                                      "kubakub mask animate). For reveal / hide / band / rings it limits the result."),
                io.Mask.Input("field", optional=True,
                              tooltip="For reveal / hide / band / rings: the grey ramp from kubakub mask field. Empty "
                                      "= a left to right ramp over the mask."),
                io.Combo.Input("effect", options=list(mf.EFFECTS), default="reveal",
                               tooltip="Along the field: reveal (appears from black to white), hide, band (a stripe "
                                       "travels through), rings (stripes that keep running). The mask itself: move "
                                       "x / move y (value = picture widths / heights), rotate (value = turns), scale "
                                       "(value = size, 1 = as it is), opacity (value 0 to 1; with a square curve = "
                                       "strobe)."),
                io.Combo.Input("curve", options=list(mf.CURVES), default="ramp",
                               tooltip="How the value runs. ramp: once, then it stays. saw: again and again. "
                                       "triangle: there and back. sine: a soft there and back. square: on / off. "
                                       "random: a new value every cycle. noise: wanders. sound level: follows the "
                                       "loudness. beats / bars / low / mid / high hits: each one starts a ramp."),
                io.Float.Input("cycle", default=2.0, min=0.01, max=3600.0, step=0.01,
                               tooltip="Speed: seconds for one cycle of the curve (for beats and hits: seconds the "
                                       "ramp takes after each one)."),
                io.Float.Input("value_from", default=0.0, min=-100.0, max=100.0, step=0.01,
                               tooltip="The value at the bottom of the curve. reveal / band / rings / opacity: 0 to "
                                       "1. move: picture widths or heights (-0.5 = half a picture to the left / "
                                       "up). rotate: turns. scale: size."),
                io.Float.Input("value_to", default=1.0, min=-100.0, max=100.0, step=0.01,
                               tooltip="The value at the top of the curve."),
                io.Combo.Input("easing", options=list(mf.EASES), default="linear",
                               tooltip="ramp, saw, triangle, beats and hits: how the ramp starts and stops."),
                io.Float.Input("soft", default=0.05, min=0.0, max=1.0, step=0.01,
                               tooltip="reveal / hide / band / rings: how soft the moving edge is (0 = hard, 1 = the "
                                       "whole field fades at once)."),
                io.Float.Input("width", default=0.2, min=0.01, max=1.0, step=0.01,
                               tooltip="band / rings: how wide a stripe is (1 = the whole field)."),
                io.Int.Input("rings", default=3, min=1, max=64, tooltip="rings: how many stripes."),
                io.Float.Input("trail", default=0.0, min=0.0, max=60.0, step=0.05,
                               tooltip="Seconds a place keeps glowing after the mask has moved on. 0 = no trail."),
                io.Float.Input("fps", default=25.0, min=1.0, max=120.0, step=0.01, tooltip="Frames per second."),
                io.Float.Input("seconds", default=0.0, min=0.0, max=3600.0, step=0.1,
                               tooltip="Length. 0 = the length of the sound, else of the mask batch, else 4 s."),
                io.Float.Input("scale", default=1.0, min=0.05, max=1.0, step=0.05,
                               tooltip="Size of the masks relative to the input (RAM: 250 masks at 3840x2160 = 8 GB)."),
                io.Audio.Input("audio", optional=True,
                               tooltip="For the sound curves, and passed on to 'audio' for Create Video. A sound "
                                       "curve without a sound runs on a built-in beat (120 bpm)."),
                io.Float.Input("phase", default=0.0, min=0.0, max=1.0, step=0.01, optional=True, advanced=True,
                               tooltip="Where in its cycle the curve starts (0.5 = half a cycle later)."),
                io.Float.Input("duty", default=0.5, min=0.01, max=0.99, step=0.01, optional=True, advanced=True,
                               tooltip="square: the part of the cycle it is on (0.1 = short flashes)."),
                io.Int.Input("seed", default=1, min=0, max=2 ** 31 - 1, optional=True, advanced=True,
                             control_after_generate=False, tooltip="random / noise: another run."),
                io.Combo.Input("listen", options=list(mf.LISTEN), default="everything", optional=True, advanced=True,
                               tooltip="sound level: the whole sound, or only its low (kick, bass), mid or high "
                                       "(hats, clicks) frequencies."),
                io.Float.Input("gain", default=1.0, min=0.0, max=10.0, step=0.05, optional=True, advanced=True,
                               tooltip="sound level: how strongly the loudness drives the value."),
                io.Float.Input("release", default=0.15, min=0.0, max=5.0, step=0.01, optional=True, advanced=True,
                               tooltip="sound level: seconds the value takes to fall after a peak."),
                io.Int.Input("nth", default=1, min=1, max=64, optional=True, advanced=True,
                             tooltip="beats / bars / hits: only every nth one counts."),
                io.Int.Input("steps", default=1, min=1, max=512, optional=True, advanced=True,
                             tooltip="beats / bars / hits: 1 = each one runs the whole move again. More = each one "
                                     "moves on by one step of that many (e.g. the number of regions with a 'region "
                                     "order' field: one more region per beat)."),
                io.Float.Input("threshold", default=0.3, min=0.0, max=1.0, step=0.01, optional=True, advanced=True,
                               tooltip="low / mid / high hits: how strong a hit has to be to count."),
                io.Float.Input("gap", default=0.1, min=0.0, max=10.0, step=0.01, optional=True, advanced=True,
                               tooltip="low / mid / high hits: the shortest time between two."),
                io.Float.Input("bpm", default=0.0, min=0.0, max=400.0, step=0.1, optional=True, advanced=True,
                               tooltip="beats / bars: a tempo you know. 0 = found in the sound."),
                io.Float.Input("start", default=0.0, min=0.0, max=3600.0, step=0.1, optional=True, advanced=True,
                               tooltip="Seconds into the sound where the frames start."),
                io.Combo.Input("pivot", options=list(mf.PIVOTS), default="mask centre", optional=True, advanced=True,
                               tooltip="rotate / scale: around the mask's own centre or the middle of the picture."),
                io.Boolean.Input("wrap", default=False, optional=True, advanced=True,
                                 tooltip="move: what leaves on one side comes back in on the other."),
            ],
            outputs=[
                io.Mask.Output("masks", tooltip="One mask per frame."),
                io.Float.Output("fps", tooltip="Frames per second."),
                io.Audio.Output("audio", tooltip="The sound, cut to the frames (silence without one)."),
                io.Int.Output("frame_count", tooltip="How many frames."),
                io.String.Output("report", tooltip="Frames, size, the range of the curve, tempo or hits found."),
            ],
        )

    @classmethod
    def execute(cls, mask=None, field=None, effect="reveal", curve="ramp", cycle=2.0, value_from=0.0, value_to=1.0,
                easing="linear", soft=0.05, width=0.2, rings=3, trail=0.0, fps=25.0, seconds=0.0, scale=1.0, audio=None,
                phase=0.0, duty=0.5, seed=1, listen="everything", gain=1.0, release=0.15, nth=1, steps=1, threshold=0.3,
                gap=0.1, bpm=0.0, start=0.0, pivot="mask centre", wrap=False) -> io.NodeOutput:
        import cv2
        name, notes = "kubakub mask animate", []
        fps = float(min(max(fps, 1.0), 120.0))
        along = effect in mf.FIELD_EFFECTS
        ms = _np(mask) if mask is not None else None
        f = _np(field)[0] if field is not None else None
        if ms is None and f is None:
            lab, n, _ = mf.sample_regions()
            f, inside = mf.field(lab, n, "edge distance")
            ms = inside.astype(np.float32)[None]
            notes.append("nothing connected: the windows of the sample facade are used (connect a mask, and a field "
                         "from kubakub mask field for reveal / hide / band / rings)")
        elif along and f is None:
            lab, n = mf.from_masks(ms.max(axis=0, keepdims=True), split=False, min_area=1)   # everywhere the batch ever is
            if n == 0:
                raise ValueError(f"{name}: the mask is empty.")
            f, _ = mf.field(lab, n, "direction", "all together")
            notes.append("no field connected: a left to right ramp over the mask is used (kubakub mask field makes others)")
        elif not along and ms is None:
            ms = (f > 0).astype(np.float32)[None]
        if ms is not None and f is not None and ms.shape[1:] != f.shape:
            f = cv2.resize(f, (ms.shape[2], ms.shape[1]), interpolation=cv2.INTER_LINEAR)
            notes.append("the field has another size than the mask: it was resized to the mask")

        sound_curve = curve in mf.TRIGGERS or curve == "sound level"
        wave = sr = mono = None
        sample_sound = False
        if audio is not None:
            wave, sr = audio["waveform"], int(audio["sample_rate"])
            wave = wave if wave.dim() == 3 else wave.reshape(1, -1, wave.shape[-1])
            mono = so.to_mono(wave)
        elif sound_curve:
            sr, sample_sound = 44100, True
            wave = torch.from_numpy(so.sample_sound(8.0, mf.SAMPLE_BPM, sr))[None, None]
            mono = so.to_mono(wave)
            notes.append("a sound curve without a sound: the built-in beat is used (connect yours to 'audio')")
        total = len(mono) / float(sr) if mono is not None else 0.0
        start = min(max(0.0, float(start)), max(0.0, total - 1e-3)) if mono is not None else 0.0
        if seconds > 0:
            secs = float(seconds)
        elif mono is not None:
            secs = total - start
        elif ms is not None and len(ms) > 1:
            secs = len(ms) / fps
        else:
            secs = 4.0
        frames = max(1, int(round(secs * fps)))
        H, W = (ms[0] if ms is not None else f).shape
        w, h = max(8, int(round(W * scale))), max(8, int(round(H * scale)))
        need = frames * w * h * 4
        try:
            import psutil
            free = int(psutil.virtual_memory().available)
        except Exception:  # noqa: BLE001
            free = 0
        if free and need > 0.7 * free:
            raise ValueError(f"{name}: {frames} masks at {w}x{h} need about {need / 2 ** 30:.0f} GB of RAM, "
                             f"{free / 2 ** 30:.0f} GB are free. Lower 'scale' or 'seconds'.")
        if scale < 1.0:
            ms = np.stack([mf.fit(m, scale) for m in ms]) if ms is not None else None
            f = mf.fit(f, scale) if f is not None else None

        ctx, lines = ({}, [])
        if sound_curve:
            ctx, lines = mf.sound_context(mono, sr, curve, float(bpm), sample_sound, listen)
        times = [start + i / fps for i in range(frames)]
        values, more = mf.curve_values(times, curve, float(cycle), float(phase), easing, float(duty), int(seed),
                                       float(value_from), float(value_to), ctx, int(nth), float(threshold), float(gap),
                                       int(steps), listen, float(release), float(gain))
        pbar = comfy.utils.ProgressBar(frames)
        out = mf.animate(ms, f, values, effect, float(soft), float(width), int(rings), float(trail), fps, pivot, bool(wrap),
                         progress=lambda i, n: pbar.update_absolute(i, n),
                         check_interrupt=comfy.model_management.throw_exception_if_processing_interrupted)

        if wave is not None:
            a, n = int(round(start * sr)), max(1, int(round(secs * sr)))
            cut = wave[..., a:a + n].float().cpu()
            if cut.shape[-1] < n:                     # longer than the sound: silence to the end of the frames
                cut = torch.nn.functional.pad(cut, (0, n - cut.shape[-1]))
            sound = {"waveform": cut.contiguous(), "sample_rate": sr}
        else:
            sound = {"waveform": torch.zeros((1, 1, max(1, int(round(secs * 44100)))), dtype=torch.float32), "sample_rate": 44100}
        report = "\n".join([f"{frames} masks, {w}x{h}, {secs:.2f} s at {fps:g} fps",
                            f"{effect} on a {curve} curve, value {min(values):.3g} to {max(values):.3g}"] + lines + more + notes)
        log.info("[KUBA mask animate] %s", report.replace("\n", "\n    "))
        return io.NodeOutput(torch.from_numpy(out), fps, sound, frames, report, ui=_preview(out, fps))


def _preview(out, fps):
    """The masks as a small animated WebP in ComfyUI's temp folder, shown on the node."""
    try:
        import cv2
        from PIL import Image
        step = max(1, int(np.ceil(len(out) / PREVIEW_FRAMES)))
        k = min(1.0, PREVIEW_PX / float(out.shape[2]))
        size = (max(8, int(out.shape[2] * k)), max(8, int(out.shape[1] * k)))
        imgs = [Image.fromarray((np.clip(cv2.resize(m, size, interpolation=cv2.INTER_AREA), 0, 1) * 255).astype(np.uint8))
                for m in out[::step]]
        fn = "kubakub_mask_animate_" + "".join(random.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(6)) + ".webp"
        folder = folder_paths.get_temp_directory()
        os.makedirs(folder, exist_ok=True)
        imgs[0].save(os.path.join(folder, fn), save_all=True, append_images=imgs[1:], duration=int(round(1000.0 * step / fps)),
                     loop=0, quality=70, method=2)
        return {"images": [{"filename": fn, "subfolder": "", "type": "temp"}], "animated": (True,)}
    except Exception as e:  # noqa: BLE001      the preview must never cost the masks
        log.warning("[KUBA mask animate] no preview: %s", e)
        return None


NODE_CLASS_MAPPINGS = {"KUBA_MaskField": KUBA_MaskField, "KUBA_MaskAnimate": KUBA_MaskAnimate}
NODE_DISPLAY_NAME_MAPPINGS = {"KUBA_MaskField": "kubakub mask field", "KUBA_MaskAnimate": "kubakub mask animate"}
