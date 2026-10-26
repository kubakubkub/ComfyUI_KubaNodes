"""
nodes_height.py

kubakub height animate: a height map that grows, one per frame, and the same as a relief lit from the side. The height
is any grey picture where white stands out: 'height' of kubakub form height, or a depth map from Depth Anything or
Marigold. The curve is the one of kubakub mask animate (ramp, sine, the sound ...), the field one from kubakub mask
field. Logic in kubakub/heightanim.py.
V3 node schema, registered via ../__init__.py.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np
import torch

import comfy.model_management
import comfy.utils
from comfy_api.latest import io

from ...kubakub import heightanim as ha
from ...kubakub import maskfields as mf
from ...kubakub import sound as so
from ...kubakub.io_types import SoundType
from ..shader.nodes_shader import _built_in_beat, _preview, _ram_check

log = logging.getLogger("KUBA.motion")
NAME = "kubakub height animate"


class KUBA_HeightAnimate(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="KUBA_HeightAnimate",
            display_name=NAME,
            category="kubakub/2d/motion",
            search_aliases=["depth", "depth anything", "marigold", "displacement", "grow", "relief", "height map", "emerge"],
            description=(
                "A height map that grows: forms come out of the wall, frame by frame. Give any grey picture where "
                "white stands out: 'height' of kubakub form height, or a depth map from Depth Anything or Marigold. "
                "'grow' says how the forms come (all together, peaks first, built up in layers, or along a field "
                "from kubakub mask field), the curve says how it runs over time (ramp, sine, the sound ...). Gives "
                "the height per frame, the same lit from the side so the growth can be seen, and it as masks. "
                "Nothing connected = a sample."),
            inputs=[
                io.Image.Input("height", optional=True,
                               tooltip="The height as a grey picture: black = the wall, white = furthest out. From "
                                       "kubakub form height, or a depth map (near = white). Empty = a sample."),
                io.Image.Input("image", optional=True,
                               tooltip="The picture on the relief for 'lit': the facade, a clay render, the picture the "
                                       "depth map was made from. Empty = clay grey."),
                io.Image.Input("wall_image", optional=True,
                               tooltip="With 'image': the picture of the bare wall (e.g. the clay render). It shows "
                                       "where a form has not come out yet, so the forms bring their colours with "
                                       "them. Empty = 'image' is there from the start."),
                io.Mask.Input("field", optional=True,
                              tooltip="For grow = along the field: the grey ramp from kubakub mask field, black comes "
                                      "first, white last. Empty = left to right over the forms."),
                io.Combo.Input("grow", options=list(ha.GROWS), default=ha.GROWS[0],
                               tooltip="all together: every form rises at once. peaks first: the forms are pushed out "
                                       "through the wall, the highest points come first. built up in layers: from the "
                                       "wall up, cut flat at the level reached. along the field: the forms rise where "
                                       "the field has been passed."),
                io.Combo.Input("curve", options=list(mf.CURVES), default="ramp",
                               tooltip="How the value runs. ramp: once, then it stays. saw: again and again. "
                                       "triangle: there and back. sine: a soft there and back. square: on / off. "
                                       "random, noise. sound level: follows the loudness. beats / bars / low / mid / "
                                       "high hits: each one starts a ramp."),
                io.Float.Input("cycle", default=3.0, min=0.01, max=3600.0, step=0.01,
                               tooltip="Speed: seconds for one cycle of the curve (for beats and hits: seconds the "
                                       "ramp takes after each one)."),
                io.Float.Input("value_from", default=0.0, min=-10.0, max=10.0, step=0.01,
                               tooltip="The value at the bottom of the curve: 0 = the bare wall."),
                io.Float.Input("value_to", default=1.0, min=-10.0, max=10.0, step=0.01,
                               tooltip="The value at the top of the curve: 1 = the whole height. More overshoots "
                                       "(all together), less stops on the way."),
                io.Combo.Input("easing", options=list(mf.EASES), default="ease in out",
                               tooltip="ramp, saw, triangle, beats and hits: how the ramp starts and stops."),
                io.Float.Input("wall", default=0.0, min=0.0, max=0.99, step=0.01,
                               tooltip="The grey that counts as the wall. 0 for kubakub form height. For a depth map of "
                                       "a whole picture raise it to the grey of the wall, so only what stands in "
                                       "front of it grows."),
                io.Float.Input("soft", default=0.15, min=0.0, max=1.0, step=0.01,
                               tooltip="along the field: how wide the rising edge is (0 = the forms pop up, 1 = the "
                                       "whole field rises at once)."),
                io.Float.Input("light_angle", default=35.0, min=-360.0, max=360.0, step=1.0,
                               tooltip="lit: where the light comes from, degrees. 0 = from the left, 90 = from above, "
                                       "180 = from the right."),
                io.Float.Input("relief", default=1.0, min=0.0, max=20.0, step=0.05,
                               tooltip="lit: how deep the relief reads. Raise it for flat forms."),
                io.Float.Input("fps", default=25.0, min=1.0, max=120.0, step=0.01, tooltip="Frames per second."),
                io.Float.Input("seconds", default=0.0, min=0.0, max=3600.0, step=0.1,
                               tooltip="Length. 0 = the length of the sound, else 4 s."),
                io.Float.Input("scale", default=1.0, min=0.05, max=1.0, step=0.05,
                               tooltip="Size of the frames relative to the height map (RAM: 100 frames at 1920x1080 "
                                       "= 6 GB)."),
                io.Audio.Input("audio", optional=True,
                               tooltip="For the sound curves, and passed on to 'audio' for Create Video. A sound "
                                       "curve without a sound runs on a built-in beat (120 bpm)."),
                SoundType.Input("sound", optional=True,
                                tooltip="From kubakub sound: the sound and how it is heard (tempo, gain, release, band, threshold, gap), set "
                                        "once for every node that listens, so all of them step on the same beat. Connected, it comes before '"
                                        "audio' and before this node's own sound settings."),
            ],
            outputs=[
                io.Image.Output("height", tooltip="The height per frame, grey: for the depth presets of kubakub shader "
                                                  "(hard shadow, sheet of light), or as displacement."),
                io.Image.Output("lit", tooltip="The same as a relief lit from the side, on 'image': the growth as a "
                                               "clip, e.g. the guide of kubakub motion guide."),
                io.Mask.Output("masks", tooltip="The height per frame as masks."),
                io.Float.Output("fps", tooltip="Frames per second."),
                io.Audio.Output("audio", tooltip="The sound, cut to the frames (silence without one)."),
                io.Int.Output("frame_count", tooltip="How many frames."),
                io.String.Output("report", tooltip="Frames, size, the range of the curve, tempo or hits found."),
            ],
        )

    @classmethod
    def execute(cls, height=None, image=None, field=None, grow=ha.GROWS[0], curve="ramp", cycle=3.0, value_from=0.0,
                value_to=1.0, easing="ease in out", wall=0.0, soft=0.15, light_angle=35.0, relief=1.0, fps=25.0,
                seconds=0.0, scale=1.0, audio=None, wall_image=None, sound=None) -> io.NodeOutput:
        audio, heard, cabled = so.take(sound, audio)
        notes = []
        fps = float(min(max(fps, 1.0), 120.0))
        grow = grow if grow in ha.GROWS else ha.GROWS[0]
        if height is None:
            h = ha.sample()
            notes.append("no height connected: a sample is used (the windows of the sample facade as soft hills)")
        else:
            if height.shape[0] > 1:
                notes.append(f"height holds {height.shape[0]} pictures: the first one is used")
            h = ha.from_grey(height[0].cpu().numpy(), wall)
            if not (h > 0).any():
                notes.append("the height map is black above 'wall': nothing grows. Lower 'wall'.")
        h = mf.fit(h, scale)
        H, W = h.shape
        base = None
        if image is not None:
            base = np.ascontiguousarray(image[0, ..., :3].cpu().numpy(), np.float32)
            if base.shape[:2] != (H, W):
                base = cv2.resize(base, (W, H), interpolation=cv2.INTER_AREA)
        under = None
        if wall_image is not None and base is not None:
            under = np.ascontiguousarray(wall_image[0, ..., :3].cpu().numpy(), np.float32)
            if under.shape[:2] != (H, W):
                under = cv2.resize(under, (W, H), interpolation=cv2.INTER_AREA)
        elif wall_image is not None:
            notes.append("wall_image is not used without 'image' (it is what shows where 'image' has not come out yet)")
        f = None
        if grow == ha.GROWS[3]:
            if field is not None:
                f = field.float().cpu().numpy()
                f = np.ascontiguousarray(f[0] if f.ndim == 3 else f, np.float32)
                if f.shape != (H, W):
                    f = cv2.resize(f, (W, H), interpolation=cv2.INTER_LINEAR)
            else:
                f = ha.ramp_over(h)
                notes.append("no field connected: the forms rise from left to right (kubakub mask field makes others)")
        elif field is not None:
            notes.append(f"the field is not used by grow = {grow} (it is for 'along the field')")

        wave, sr, mono = so.sound_in(audio)
        sample_sound = wave is None and mf.needs_sound(curve)
        secs = float(seconds) if seconds > 0 else (len(mono) / float(sr) if mono is not None else 4.0)
        if sample_sound:
            wave, sr, mono = _built_in_beat(secs)
            notes.append("a sound curve without a sound: the built-in beat is used (connect yours to 'audio')")
        frames = max(1, int(round(secs * fps)))
        _ram_check(NAME, frames, W, H, 28)                # height and lit as pictures, the masks: 7 floats per pixel

        times = [i / fps for i in range(frames)]
        values, lines = mf.timed_values(times, curve, cycle, value_from, value_to, easing, mono, sr or 44100, sample_sound,
                                        heard if cabled else None)
        masks = torch.empty((frames, H, W), dtype=torch.float32)
        lit = torch.empty((frames, H, W, 3), dtype=torch.float32)
        pbar = comfy.utils.ProgressBar(frames)
        for i, v in enumerate(values):
            comfy.model_management.throw_exception_if_processing_interrupted()
            if i and v == values[i - 1]:                   # a held value: the same picture
                masks[i], lit[i] = masks[i - 1], lit[i - 1]
            else:
                g = np.clip(ha.grown(h, v, grow, f, soft), 0.0, 1.0)
                masks[i] = torch.from_numpy(g)
                on = base if under is None else under + (base - under) * np.clip(g / ha.COMES, 0.0, 1.0)[..., None]
                lit[i] = torch.from_numpy(ha.lit(g, on, light_angle, relief))
            pbar.update_absolute(i + 1, frames)
        grey = masks[..., None].repeat(1, 1, 1, 3)

        report = "\n".join([f"{frames} frames, {W}x{H}, {secs:.2f} s at {fps:g} fps",
                            f"grow = {grow} on a {curve} curve, value {min(values):.3g} to {max(values):.3g}; the height "
                            f"reaches {float(masks.max()):.2f} of white"] + lines + notes)
        log.info("[KUBA height animate] %s", report.replace("\n", "\n    "))
        return io.NodeOutput(grey, lit, masks, fps, so.sound_out(wave, sr, secs), frames, report, ui=_preview(lit.numpy(), fps))


NODE_CLASS_MAPPINGS = {"KUBA_HeightAnimate": KUBA_HeightAnimate}
NODE_DISPLAY_NAME_MAPPINGS = {"KUBA_HeightAnimate": NAME}
