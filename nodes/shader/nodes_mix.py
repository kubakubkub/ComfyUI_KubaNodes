"""
nodes_mix.py

kubakub shader mix: two clips, A and B, and one mix value, 0 = A, 1 = B. A and B come from two kubakub shader
nodes (inputs a and b; any picture or clip works); in between A's pattern cuts B in, A bends B, or they fade
(kubakub/shadermix.py). The mix can run on a curve or with a sound. With nothing connected the node draws two
presets itself, as before the two inputs were there (its preset lists, controls and size are advanced now); then B
can also work on A's picture inside the shader. Draws with kubakub/shader_render.py, as kubakub shader.
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

from ...kubakub import maskfields as mf
from ...kubakub import shader_render as sr
from ...kubakub import shadermix as mx
from ...kubakub import shadertoy as st
from ...kubakub import sound as so
from ...kubakub.io_types import SoundType
from .nodes_shader import _built_in_beat, _controls_text, _fit, _luma_masks, _preview, _ram_check

log = logging.getLogger("KUBA.shader")
NAME = "kubakub shader mix"


def _shader(which, preset, code, controls):
    try:
        t = st.translate(st.shader_text(preset, code))
        values, notes = st.parse_values(controls, t.controls)
    except st.ShaderError as e:
        raise ValueError(f"{NAME}, shader {which}: {e}") from e
    own = bool((code or "").strip())
    return t, values, [f"{which}: {n}" for n in t.notes + notes], ("your code" if own else f"'{preset}'")


class KUBA_ShaderMix(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        names = list(st.presets()) or ["(no presets found)"]

        def pick(want):
            return want if want in names else names[0]
        return io.Schema(
            node_id="KUBA_ShaderMix",
            display_name=NAME,
            category="kubakub/2d/shader",
            search_aliases=["shadertoy", "glsl", "blend", "transition", "morph", "combine", "vj", "crossfade"],
            description=(
                "Mix two kubakub shaders: connect the images of one to 'a' and of another to 'b'. mix 0 is A, 1 is "
                "B; in between A's pattern cuts B in, A bends B, or they fade. The mix can run on a curve or with "
                "a sound. Any two pictures or clips work. Nothing connected mixes two presets of the node's "
                "own lists. No AI model."),
            inputs=[
                io.Combo.Input("shader_a", options=names, default=pick("plasma"), advanced=True,
                               tooltip="Nothing connected to 'a': this preset is A."),
                io.Combo.Input("shader_b", options=names, default=pick("eight bit"), advanced=True,
                               tooltip="Nothing connected to 'b': this preset is B."),
                io.Combo.Input("how", options=list(mx.HOW), default="a cuts b",
                               tooltip="How they meet between 0 and 1. a cuts b: B shows first where A is bright, a wipe "
                                       "in the shape of A's pattern. a bends b: A's picture pushes B's places around, B "
                                       "melts out of A. blend: a plain fade. b works on a: B is an effect on A; for two "
                                       "kubakub shaders do that with a cable instead (A's images into image0 of shader "
                                       "B), here it works when B is the preset of this node."),
                io.Float.Input("mix", default=0.5, min=0.0, max=1.0, step=0.01,
                               tooltip="0 = shader A, 1 = shader B. With a curve: the value the curve reaches."),
                io.Combo.Input("curve", options=list(mf.TIMED), default="constant",
                               tooltip="How the mix runs over time. constant = stays at 'mix'. ramp: from A to B once. "
                                       "saw / triangle / sine / square: again and again. sound level, beats, bars, "
                                       "hits: with the sound on 'audio'."),
                io.Float.Input("cycle", default=2.0, min=0.02, max=3600.0, step=0.01,
                               tooltip="curve: seconds for one run (beats and hits: how long each one's ramp is)."),
                io.Image.Input("image", optional=True,
                               tooltip="For the presets of this node (nothing on 'a' / 'b'): a picture or a clip they "
                                       "work on (their image0). Without one, a preset that needs a picture takes the "
                                       "sample facade, and B takes A's picture."),
                io.Audio.Input("audio", optional=True,
                               tooltip="A sound for the curve (sound level, beats ...) and for shaders that listen. "
                                       "Passed on to 'audio'."),
                io.Int.Input("width", default=1280, min=0, max=8192, step=8, advanced=True,
                             tooltip="Nothing on 'a' / 'b': the width. 0 = the image's width. A connected clip sets the size."),
                io.Int.Input("height", default=720, min=0, max=8192, step=8, advanced=True,
                             tooltip="Nothing on 'a' / 'b': the height. 0 = the image's height."),
                io.Float.Input("fps", default=25.0, min=1.0, max=120.0, step=0.01,
                               tooltip="Frames per second: connect the fps of the kubakub shader on 'a', so the curve runs "
                                       "at the clip's pace."),
                io.Float.Input("seconds", default=4.0, min=0.0, max=3600.0, step=0.01, advanced=True,
                               tooltip="Nothing on 'a' / 'b' (or two stills): how long. 0 = as long as the sound, else "
                                       "the clip on 'image', else 4 s. A connected clip sets the length."),
                io.Int.Input("loops", default=1, min=1, max=1000, advanced=True,
                             tooltip="The presets of this node: how many times they run through."),
                io.String.Input("controls_a", default="", multiline=True, optional=True, advanced=True,
                                placeholder="controls of preset A: speed = 2, glow = 0.4",
                                tooltip="Preset A's controls as 'name = number', commas or lines. The report lists them."),
                io.String.Input("controls_b", default="", multiline=True, optional=True, advanced=True, placeholder="controls of preset B",
                                tooltip="Preset B's controls."),
                io.String.Input("code_a", default="", multiline=True, optional=True, advanced=True,
                                placeholder="Shader A from shadertoy.com instead of the preset (Ctrl+A, Ctrl+C there, paste here)",
                                tooltip="Shader A as pasted code (a Shadertoy or a GLSL fragment shader) instead of the preset."),
                io.String.Input("code_b", default="", multiline=True, optional=True, advanced=True,
                                placeholder="Shader B from shadertoy.com instead of the preset",
                                tooltip="Shader B as pasted code instead of the preset."),
                io.Float.Input("mix_from", default=0.0, min=0.0, max=1.0, step=0.01, optional=True, advanced=True,
                               tooltip="curve: the mix at its low end ('mix' is its high end)."),
                io.Float.Input("strength", default=0.15, min=0.0, max=1.0, step=0.01, optional=True, advanced=True,
                               tooltip="a bends b: how far A pushes B half way through, as a part of the picture's height."),
                io.Float.Input("soft", default=0.1, min=0.0, max=1.0, step=0.01, optional=True, advanced=True,
                               tooltip="a cuts b: how soft the edge of the wipe is."),
                io.Image.Input("a", optional=True,
                               tooltip="A: what you see at mix = 0. The images of a kubakub shader (or any picture or "
                                       "clip). It sets the size and the length."),
                io.Image.Input("b", optional=True,
                               tooltip="B: what you see at mix = 1. The images of another kubakub shader. Another size is "
                                       "resized to A's, a shorter clip plays again from its start."),
                SoundType.Input("sound", optional=True,
                                tooltip="From kubakub sound: the sound and how it is heard (tempo, gain, release, band, threshold, gap), set "
                                        "once for every node that listens, so all of them step on the same beat. Connected, it comes before '"
                                        "audio' and before this node's own sound settings."),
            ],
            outputs=[
                io.Image.Output("images", tooltip="One picture per frame."),
                io.Mask.Output("masks", tooltip="One mask per frame: how bright the picture is."),
                io.Float.Output("fps", tooltip="Frames per second."),
                io.Audio.Output("audio", tooltip="The sound, cut to the frames (silence without one)."),
                io.Int.Output("frame_count", tooltip="How many frames."),
                io.String.Output("report", tooltip="Size, length, the two shaders and their controls, how the mix runs."),
            ],
        )

    @classmethod
    def execute(cls, shader_a, shader_b, how=mx.HOW[0], mix=0.5, curve="constant", cycle=2.0, image=None, audio=None,
                width=1280, height=720, fps=25.0, seconds=4.0, loops=1, controls_a="", controls_b="", code_a="", code_b="",
                mix_from=0.0, strength=0.15, soft=0.1, a=None, b=None, sound=None) -> io.NodeOutput:
        how = how if how in mx.HOW else mx.HOW[0]
        audio, heard, cabled = so.take(sound, audio)
        pa = a[..., :3].float().cpu().numpy() if a is not None and len(a) else None      # A and B as finished clips
        pb = b[..., :3].float().cpu().numpy() if b is not None and len(b) else None
        notes, ta, tb, va, vb = [], None, None, {}, {}
        what_a, what_b = "the clip on 'a'", "the clip on 'b'"
        if pa is None:
            ta, va, more, what_a = _shader("A", shader_a, code_a, controls_a)
            notes += more
        if pb is None:
            tb, vb, more, what_b = _shader("B", shader_b, code_b, controls_b)
            notes += more
        for t, which in ((ta, "A"), (tb, "B")):
            if t is not None and ("kPrev" in t.uses or t.buffers):
                raise ValueError(f"{NAME}: preset {which} builds on its own last frame (slit scan, datamosh, trails, a "
                                 "Shadertoy with buffer tabs): "
                                 f"that works in kubakub shader only. Connect its images to '{which.lower()}' instead.")
        listens = bool((ta is not None and ta.listens) or (tb is not None and tb.listens))
        fps = float(min(max(fps, 1.0), 120.0))
        pic = image[..., :3].float().cpu().numpy() if image is not None else None
        given = pa if pa is not None else pb          # a connected clip sets the size and the length
        if given is not None:
            w, h = given.shape[2], given.shape[1]
            if pa is not None and pb is not None and pb.shape[1:3] != pa.shape[1:3]:
                notes.append(f"b is {pb.shape[2]}x{pb.shape[1]}: resized to A's {w}x{h}")
        else:
            w = int(width) or (pic.shape[2] if pic is not None else 1280)
            h = int(height) or (pic.shape[1] if pic is not None else 720)
        longest = max([len(p) for p in (pa, pb) if p is not None] + [1])

        wave, hz, mono = so.sound_in(audio)
        sample = False
        if pa is not None and pb is not None and image is not None:
            notes.append("image: not used while 'a' and 'b' are both connected (it is the picture a preset reads)")
        if longest > 1:
            secs = longest / fps
            short = [n for n, p in (("a", pa), ("b", pb)) if p is not None and 1 < len(p) < longest]
            if short:
                notes.append(f"{short[0]} is shorter ({len(pa if short[0] == 'a' else pb)} of {longest} frames): it plays again from its start")
        elif seconds > 0:
            secs = float(seconds)
        elif mono is not None:
            secs = len(mono) / float(hz)
        elif pic is not None and len(pic) > 1:
            secs = len(pic) / fps
        else:
            secs = 4.0
        frames = max(1, int(round(secs * fps)))
        if mono is None and (listens or mf.needs_sound(curve)):
            (wave, hz, mono), sample = _built_in_beat(secs), True
            notes.append("a shader or the curve listens to a sound and none is connected: the built-in beat is used "
                         "(connect yours to 'audio')")

        _ram_check(NAME, frames, w, h, 15)            # the frames out (float) and shader A kept for B (bytes)

        # the mix over time
        times = [i / fps for i in range(frames)]
        lo, hi = float(min(max(mix_from, 0.0), 1.0)), float(min(max(mix, 0.0), 1.0))
        ms, curve_lines = mf.timed_values(times, curve, cycle, lo, hi, mono=mono, sr=hz or 44100, sample=sample,
                                          heard=heard if cabled else None)
        ms = [hi] * frames if ms is None else ms

        per_frame = sr.frame_times(frames, fps, 1.0, 0.0, 0, int(loops))
        sound_lines = []
        if mono is not None and listens:
            levels, sound_lines = sr.sound_values(mono, hz, times, heard["bpm"] if cabled else 0.0, sample,
                                                  *((heard["release"], heard["gain"]) if cabled else ()))
            for i, d in enumerate(per_frame):
                d.update({k: v[i] for k, v in levels.items()})
        base = {"iFrameRate": fps, "iSampleRate": float(hz or 44100), "kSeconds": secs}

        def picture(t, which, other=None):
            """What a shader reads as its pictures: the connected image, else (B) shader A's picture, else the sample."""
            tex = {}
            for i in t.channels:
                if i == 0 and other is not None:
                    tex[0] = other
                elif i == 0 and pic is not None:
                    tex[0] = lambda k: (k % len(pic), pic[k % len(pic)])
                elif i == 0 and i not in t.optional:
                    from ...kubakub import sample_facade as sf
                    still = sf.render(w, h)["image"]
                    tex[0] = lambda k, still=still: (0, still)
                    notes.append(f"shader {which} works on a picture and none is connected: the sample facade is used")
                elif i not in t.optional:
                    tex[i] = sr.noise_texture(7 + 2 * i)
            return tex

        pbar = comfy.utils.ProgressBar(2 * frames)
        stop = comfy.model_management.throw_exception_if_processing_interrupted
        bad = [False]

        a_clip = np.empty((frames, h, w, 3), np.uint8)       # shader A, kept: B reads it, and the two are joined at the end

        def numbers(px):
            """The frame's colours, 0-1; values that are no numbers (a division by zero in the shader) are black."""
            rgb = px[..., :3]
            if not np.isfinite(rgb).all():
                bad[0] = True
                rgb = np.nan_to_num(rgb, nan=0.0, posinf=1.0, neginf=0.0)
            return rgb

        def keep_a(i, px):
            a_clip[i] = cv2.convertScaleAbs(numbers(px), alpha=255.0)       # 0-1 -> bytes, rounded and clipped
            pbar.update_absolute(i + 1, 2 * frames)

        a_now = [-1, np.empty((h, w, 3), np.float32)]         # frame k of A as float, made once for B and for the join

        def a_float(k):
            if a_now[0] != k:
                np.multiply(a_clip[k], np.float32(1.0 / 255.0), out=a_now[1])
                a_now[0] = k
            return a_now[1]

        def a_at(k):
            return (("a", k), a_float(k))

        images = torch.empty((frames, h, w, 3), dtype=torch.float32)
        out = images.numpy()

        def keep_b(i, px):
            out[i] = mx.join(a_float(i), np.clip(numbers(px), 0.0, 1.0), ms[i], how, float(soft))
            pbar.update_absolute(frames + i + 1, 2 * frames)

        try:
            if pa is not None:                        # A is a finished clip
                for i in range(frames):
                    stop()
                    keep_a(i, _fit(np.ascontiguousarray(pa[i % len(pa)]), w, h))
            else:
                sr.render(ta.source, w, h, per_frame, {**base, **va}, picture(ta, "A"), keep_a, stop)
            if pb is not None:                        # B is a finished clip: A bends or cuts its picture
                if how == "b works on a":
                    notes.append("how = b works on a: B is a finished clip here, so this is a plain fade. To make B an "
                                 "effect on A, connect A's images to image0 of the kubakub shader that draws B.")
                for i in range(frames):
                    stop()
                    px = _fit(np.ascontiguousarray(pb[i % len(pb)]), w, h)
                    if how == "a bends b":
                        px = mx.bend_picture(a_float(i), px, mx.bend(ms[i], strength))
                    keep_b(i, px)
            else:
                b_frames = per_frame
                b_tex = picture(tb, "B", a_at if how == "b works on a" or (pic is None and 0 in tb.channels) else None)
                if how == "a bends b":                # A's picture on the bend unit, and how far it pushes at each frame
                    b_frames = [dict(d, kBend=mx.bend(ms[i], strength)) for i, d in enumerate(per_frame)]
                    b_tex[sr.MIX_UNIT] = a_at
                if how == "a bends b" and tb.kind != "shadertoy":
                    notes.append("how = a bends b: shader B is a plain fragment shader (void main); it is only bent where it "
                                 "reads kUV, not where it reads gl_FragCoord or v_texCoord. A Shadertoy (mainImage) is bent whole.")
                if how == "b works on a" and 0 not in tb.channels:
                    notes.append(f"how = b works on a: shader B ({what_b}) does not read a picture, so it cannot work on A; "
                                 "this is a plain fade. Pick a B that works on a picture (eight bit, vhs tape, halftone ...).")
                sr.render(tb.source, w, h, b_frames, {**base, **vb}, b_tex, keep_b, stop)
        except st.ShaderError as e:
            raise ValueError(f"{NAME}: {e}") from e
        del a_clip
        if bad[0]:
            notes.append("a shader wrote values that are no numbers (a division by zero?): they are black")

        masks, sound = _luma_masks(images), so.sound_out(wave, hz, secs)
        lines = [f"{frames} frames, {w}x{h}, {secs:.2f} s at {fps:g} fps",
                 f"A = {what_a}, B = {what_b}, how = {how}",
                 (f"mix: {curve}, {cycle:g} s, from {lo:g} to {hi:g}" if curve != "constant" else f"mix: {hi:g}")
                 + (" (0 = only A, 1 = only B)" if curve == "constant" and hi in (0.0, 1.0) else ""),
                 *(f"controls of {n}: {_controls_text(t, v) or 'none'}" for n, t, v in (("A", ta, va), ("B", tb, vb)) if t is not None)]
        report = "\n".join(lines + curve_lines + sound_lines + list(dict.fromkeys(notes)))
        log.info("[KUBA shader mix] %s", report.replace("\n", "\n    "))
        return io.NodeOutput(images, masks, fps, sound, frames, report, ui=_preview(out, fps))


NODE_CLASS_MAPPINGS = {"KUBA_ShaderMix": KUBA_ShaderMix}
NODE_DISPLAY_NAME_MAPPINGS = {"KUBA_ShaderMix": NAME}
