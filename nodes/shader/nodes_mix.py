"""
nodes_mix.py

kubakub shader mix: two shaders (presets or pasted code) and one mix value, 0 = shader A, 1 = shader B. In between
they meet in the shaders themselves, not as two finished pictures laid over each other: B works on A's picture, A
bends the places B draws at, or A's pattern cuts B in (kubakub/shadermix.py). An optional picture or clip is what
both work on. The mix can run on a curve or with a sound. Draws with kubakub/shader_render.py, as kubakub shader.
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
from .nodes_shader import _built_in_beat, _controls_text, _luma_masks, _preview, _ram_check, _sound_in, _sound_out

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
                "Two shaders and one mix value: 0 is shader A, 1 is shader B. In between they meet in the shaders, "
                "not as two pictures laid over each other: B works on A's picture, A bends B, or A's pattern cuts B "
                "in. Connect a picture or a clip to 'image' and both work on it. The mix can run on a curve or with "
                "a sound. Nothing connected mixes two presets. No AI model."),
            inputs=[
                io.Combo.Input("shader_a", options=names, default=pick("plasma"), tooltip="Shader A: what you see at mix = 0."),
                io.Combo.Input("shader_b", options=names, default=pick("eight bit"), tooltip="Shader B: what you see at mix = 1."),
                io.Combo.Input("how", options=list(mx.HOW), default=mx.HOW[0],
                               tooltip="How they meet between 0 and 1. b works on a: B takes A's picture as its own, so B "
                                       "is an effect on A (pick a B that works on a picture: eight bit, vhs tape, "
                                       "halftone ...). a bends b: A's picture pushes B's places around, B melts out of "
                                       "A. a cuts b: B shows first where A is bright, a wipe in the shape of A's "
                                       "pattern. blend: a plain fade."),
                io.Float.Input("mix", default=0.5, min=0.0, max=1.0, step=0.01,
                               tooltip="0 = shader A, 1 = shader B. With a curve: the value the curve reaches."),
                io.Combo.Input("curve", options=list(mf.TIMED), default="constant",
                               tooltip="How the mix runs over time. constant = stays at 'mix'. ramp: from A to B once. "
                                       "saw / triangle / sine / square: again and again. sound level, beats, bars, "
                                       "hits: with the sound on 'audio'."),
                io.Float.Input("cycle", default=2.0, min=0.02, max=3600.0, step=0.01,
                               tooltip="curve: seconds for one run (beats and hits: how long each one's ramp is)."),
                io.Image.Input("image", optional=True,
                               tooltip="A picture or a clip both shaders work on (their image0). Without one, a shader "
                                       "that needs a picture takes the sample facade, and B takes A's picture."),
                io.Audio.Input("audio", optional=True,
                               tooltip="A sound for the curve (sound level, beats ...) and for shaders that listen. "
                                       "Passed on to 'audio'."),
                io.Int.Input("width", default=1280, min=0, max=8192, step=8, tooltip="0 = the image's width."),
                io.Int.Input("height", default=720, min=0, max=8192, step=8, tooltip="0 = the image's height."),
                io.Float.Input("fps", default=25.0, min=1.0, max=120.0, step=0.01),
                io.Float.Input("seconds", default=4.0, min=0.0, max=3600.0, step=0.01,
                               tooltip="How long. 0 = as long as the sound, else the clip on 'image', else 4 s."),
                io.Int.Input("loops", default=1, min=1, max=1000, tooltip="How many times the shaders run through."),
                io.String.Input("controls_a", default="", multiline=True, optional=True, placeholder="controls of shader A: speed = 2, glow = 0.4",
                                tooltip="Shader A's controls as 'name = number', commas or lines. The report lists them."),
                io.String.Input("controls_b", default="", multiline=True, optional=True, placeholder="controls of shader B",
                                tooltip="Shader B's controls."),
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
                mix_from=0.0, strength=0.15, soft=0.1) -> io.NodeOutput:
        how = how if how in mx.HOW else mx.HOW[0]
        ta, va, notes_a, what_a = _shader("A", shader_a, code_a, controls_a)
        tb, vb, notes_b, what_b = _shader("B", shader_b, code_b, controls_b)
        notes = notes_a + notes_b
        for t, which in ((ta, "A"), (tb, "B")):
            if "kPrev" in t.uses or t.buffers:
                raise ValueError(f"{NAME}: shader {which} builds on its own last frame (slit scan, datamosh, trails, a "
                                 "Shadertoy with buffer tabs): "
                                 "that works in kubakub shader only. Mix its result as 'image' instead.")
        fps = float(min(max(fps, 1.0), 120.0))
        pic = image[..., :3].float().cpu().numpy() if image is not None else None
        w = int(width) or (pic.shape[2] if pic is not None else 1280)
        h = int(height) or (pic.shape[1] if pic is not None else 720)

        wave, hz, mono = _sound_in(audio)
        sample = False
        if seconds > 0:
            secs = float(seconds)
        elif mono is not None:
            secs = len(mono) / float(hz)
        elif pic is not None and len(pic) > 1:
            secs = len(pic) / fps
        else:
            secs = 4.0
        frames = max(1, int(round(secs * fps)))
        if mono is None and (ta.listens or tb.listens or mf.needs_sound(curve)):
            (wave, hz, mono), sample = _built_in_beat(secs), True
            notes.append("a shader or the curve listens to a sound and none is connected: the built-in beat is used "
                         "(connect yours to 'audio')")

        _ram_check(NAME, frames, w, h, 15)            # the frames out (float) and shader A kept for B (bytes)

        # the mix over time
        times = [i / fps for i in range(frames)]
        lo, hi = float(min(max(mix_from, 0.0), 1.0)), float(min(max(mix, 0.0), 1.0))
        ms, curve_lines = mf.timed_values(times, curve, cycle, lo, hi, mono=mono, sr=hz or 44100, sample=sample)
        ms = [hi] * frames if ms is None else ms

        per_frame = sr.frame_times(frames, fps, 1.0, 0.0, 0, int(loops))
        sound_lines = []
        if mono is not None and (ta.listens or tb.listens):
            heard, sound_lines = sr.sound_values(mono, hz, times, 0.0, sample)
            for i, d in enumerate(per_frame):
                d.update({k: v[i] for k, v in heard.items()})
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
            sr.render(ta.source, w, h, per_frame, {**base, **va}, picture(ta, "A"), keep_a, stop)
            b_frames = per_frame
            b_tex = picture(tb, "B", a_at if how == "b works on a" or (pic is None and 0 in tb.channels) else None)
            if how == "a bends b":                    # A's picture on the bend unit, and how far it pushes at each frame
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

        masks, sound = _luma_masks(images), _sound_out(wave, hz, secs)
        lines = [f"{frames} frames, {w}x{h}, {secs:.2f} s at {fps:g} fps",
                 f"A = {what_a}, B = {what_b}, how = {how}",
                 (f"mix: {curve}, {cycle:g} s, from {lo:g} to {hi:g}" if curve != "constant" else f"mix: {hi:g}")
                 + (" (0 = only A, 1 = only B)" if curve == "constant" and hi in (0.0, 1.0) else ""),
                 f"controls of A: {_controls_text(ta, va) or 'none'}", f"controls of B: {_controls_text(tb, vb) or 'none'}"]
        report = "\n".join(lines + curve_lines + sound_lines + list(dict.fromkeys(notes)))
        log.info("[KUBA shader mix] %s", report.replace("\n", "\n    "))
        return io.NodeOutput(images, masks, fps, sound, frames, report, ui=_preview(out, fps))


NODE_CLASS_MAPPINGS = {"KUBA_ShaderMix": KUBA_ShaderMix}
NODE_DISPLAY_NAME_MAPPINGS = {"KUBA_ShaderMix": NAME}
