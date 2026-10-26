"""kubakub shader: frames and masks from a preset, a pasted Shadertoy, an OSL shader or a GLSL fragment shader, with time, a loop,
a sound, a field and pictures as inputs (text side kubakub/shadertoy.py, picture side kubakub/shader_render.py)."""

from __future__ import annotations

import functools
import hashlib
import logging
import os
import random
import re
import time

import numpy as np
import torch

import comfy.model_management
import comfy.utils
import folder_paths
from comfy_api.latest import io

from ...kubakub import maskfields as mf
from ...kubakub import materials as ma
from ...kubakub import plan as rp
from ...kubakub import shader_render as sr
from ...kubakub import shadertoy as st
from ...kubakub import sound as so
from ...kubakub import tracking as tk
from ...kubakub.io_types import RegionsType, SoundType, TracksType

log = logging.getLogger("KUBA.shader")
st.USER_DIR = os.path.join(folder_paths.get_user_directory(), "kubakub", "shaders")      # presets saved from the node
NAME = "kubakub shader"
PREVIEW_PX, PREVIEW_FRAMES = 384, 240
SLOW = 0.25                                          # seconds a frame may take before the report says how to make it faster
MASK_FROM = ("auto", "alpha", "brightness")
UNCONNECTED = ("auto", "noise", "sample facade", "black")
LUMA = ma.LUMA
ON_PICTURE = ("screen", "add", "multiply", "overlay", "difference", "displace", "off")


def _fit(a, w, h):
    """[h0, w0] or [h0, w0, c] float32 at w x h."""
    if a.shape[1] == w and a.shape[0] == h:
        return a
    import cv2
    return cv2.resize(a, (w, h), interpolation=cv2.INTER_LINEAR)


def _sound_in(audio):
    """A connected sound as (waveform [1, channels, samples], sample rate, mono), or (None, None, None)."""
    if audio is None:
        return None, None, None
    wave, hz = audio["waveform"], int(audio["sample_rate"])
    wave = wave if wave.dim() == 3 else wave.reshape(1, -1, wave.shape[-1])
    return wave, hz, so.to_mono(wave)


def _built_in_beat(secs, hz=44100):
    """The beat a shader hears when it listens and no sound is connected: (waveform, sample rate, mono)."""
    wave = torch.from_numpy(so.sample_sound(secs, mf.SAMPLE_BPM, hz))[None, None]
    return wave, hz, so.to_mono(wave)


def _sound_out(wave, hz, secs):
    """The sound cut or padded to the clip's length; silence of that length without one."""
    if wave is None:
        return {"waveform": torch.zeros((1, 1, max(1, int(round(secs * 44100)))), dtype=torch.float32), "sample_rate": 44100}
    n = max(1, int(round(secs * hz)))
    snd = wave[..., :n].float().cpu()
    if snd.shape[-1] < n:
        snd = torch.nn.functional.pad(snd, (0, n - snd.shape[-1]))
    return {"waveform": snd.contiguous(), "sample_rate": hz}


def _ram_check(name, frames, w, h, per_pixel):
    """Says so before a clip that does not fit into the free RAM is started (per_pixel = bytes it needs per pixel)."""
    need = frames * w * h * per_pixel
    try:
        import psutil
        free = int(psutil.virtual_memory().available)
    except Exception:  # noqa: BLE001
        return
    if need > 0.7 * free:
        raise ValueError(f"{name}: {frames} frames at {w}x{h} need about {need / 2 ** 30:.0f} GB of RAM, "
                         f"{free / 2 ** 30:.0f} GB are free. Lower 'seconds' or the size.")


lay_over = ma.blend                                  # a shader that does not read the picture, laid over it


def _luma_masks(images):
    """How bright each frame is, as masks [frames, h, w]; in blocks, so there is no second copy of the whole clip."""
    masks = torch.empty(images.shape[:3], dtype=torch.float32)
    lum = torch.tensor(LUMA)
    for a in range(0, len(images), 16):
        masks[a:a + 16] = (images[a:a + 16] * lum).sum(-1).clamp_(0.0, 1.0)
    return masks


def _controls_text(t, values):
    return ", ".join(f"{c.colour} = {st.colour_hex(values, c.colour)}" if c.colour else f"{c.name.rstrip('_')} = {values[c.name]:g} ({c.lo:g} to {c.hi:g})"
                     for c in t.controls if not c.colour or c.name.endswith("_r"))


class KUBA_Shader(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        names = list(st.presets()) or ["(no presets found)"]
        return io.Schema(
            node_id="KUBA_Shader",
            display_name=NAME,
            category="kubakub/2d/shader",
            search_aliases=["shadertoy", "glsl", "procedural", "generator", "noise", "pattern", "caustics", "voronoi",
                            "grid", "test pattern", "glass", "refraction", "vj", "loop", "raymarch", "sdf", "3d", "relief",
                            "depth", "fog"],
            description=(
                "Moving pictures and masks drawn by a shader: pick a preset, or paste the code of a Shadertoy (the "
                "Image tab), an OSL pattern shader (as written for Blender, Houdini, Arnold) or your own GLSL fragment "
                "shader. The node gives it the time, a loop, the sound (level, "
                "low / mid / high, beats), a field and a mask from kubakub mask field, your regions, and up to four "
                "pictures. Every "
                "'uniform float name; // default min max' line of the shader is a control you set in 'controls'. "
                "A shader can be a 3D scene: distance functions and a raymarcher are built in, and such a "
                "shader gives its depth. Nothing connected runs a preset. No AI model."),
            inputs=[
                io.Combo.Input("preset", options=names, default="warped noise" if "warped noise" in names else names[0],
                               tooltip="A built-in shader. Used while 'code' is empty. The list is sorted by type: "
                                       + ", ".join(title for _, title, _ in st.TYPES) + ". The button under it shows "
                                       "them as moving pictures. The report lists its controls."),
                io.String.Input("code", default="", multiline=True,
                                placeholder="A shader from shadertoy.com instead of the preset: open it there, click "
                                            "into the code, Ctrl+A, Ctrl+C, then paste here. No need to understand it.",
                                tooltip="A shader from shadertoy.com instead of the preset. On its page click into the "
                                        "code, press Ctrl+A, then Ctrl+C, and paste here; a 'Common' tab goes first, "
                                        "the 'Image' tab below it. You do not need to read the code: the node says in "
                                        "plain words what a shader needs and what cannot work (buffer tabs, cubemaps, "
                                        "sound shaders). A complete GLSL ES 3.00 fragment shader works too, and an OSL shader "
                                        "('shader name(...)'): its inputs become the controls, see the shader guide."),
                io.String.Input("controls", default="", multiline=True,
                                tooltip="Values for the shader's controls, e.g. 'scale = 5, warp = 2' (commas or "
                                        "lines). Empty = the defaults. The report lists every control with its range."),
                io.Int.Input("width", default=0, min=0, max=16384, step=8,
                             tooltip="0 = the size of the first connected picture or mask, else 1280."),
                io.Int.Input("height", default=0, min=0, max=16384, step=8,
                             tooltip="0 = the size of the first connected picture or mask, else 720."),
                io.Float.Input("fps", default=25.0, min=1.0, max=120.0, step=0.01, tooltip="Frames per second."),
                io.Float.Input("seconds", default=0.0, min=0.0, max=3600.0, step=0.1,
                               tooltip="Length. 0 = the length of the sound, else of the longest picture or mask "
                                       "batch, else 4 s."),
                io.Float.Input("speed", default=1.0, min=-100.0, max=100.0, step=0.05,
                               tooltip="How fast the shader's clock (iTime) runs. 0 = a still, negative = backwards. "
                                       "The presets run on the loop, not on the clock: their pace is 'loops'."),
                io.Int.Input("loops", default=1, min=1, max=1000,
                             tooltip="How many times the presets run through within the clip (kLoop in a shader "
                                     "goes 0 to 1 that often). More = faster."),
                io.Boolean.Input("loop", default=True,
                                 tooltip="The last frame meets the first. The presets loop by themselves; a pasted "
                                         "shader is rendered a little longer and its end is faded into its start."),
                io.Image.Input("image0", optional=True,
                               tooltip="Your picture. A batch plays as a clip, one picture per frame. The presets "
                                       "that work on a picture redraw it; every other shader is laid over it "
                                       "(on_picture). iChannel0 in the shader; one that reads it with nothing "
                                       "connected gets the sample facade."),
                io.Image.Input("image1", optional=True, tooltip="iChannel1 (e.g. a normal pass for the glass preset)."),
                io.Image.Input("image2", optional=True, tooltip="iChannel2."),
                io.Image.Input("image3", optional=True, tooltip="iChannel3."),
                io.Mask.Input("mask", optional=True,
                              tooltip="Where the shader shows: kMask(uv) in the shader, and with 'clip' the pictures "
                                      "and masks are cut to it. A batch plays one mask per frame (e.g. from kubakub "
                                      "mask animate)."),
                io.Mask.Input("field", optional=True,
                              tooltip="A grey ramp from kubakub mask field: kField(uv) in the shader, the path a "
                                      "band runs along. Empty = a left to right ramp."),
                io.Audio.Input("audio", optional=True,
                               tooltip="What the shader hears: kLevel, kLow, kMid, kHigh (0-1), kBeat (1 on a beat, "
                                       "then falling), kBeats (a count). Passed on to 'audio'. Connecting a sound "
                                       "makes a preset follow it: its 'sound' or 'beat' control goes to 1 unless you "
                                       "type a value. A shader that listens without a sound runs on a built-in beat "
                                       "(120 bpm)."),
                io.Float.Input("time_offset", default=0.0, min=-36000.0, max=36000.0, step=0.1, optional=True, advanced=True,
                               tooltip="The shader's clock at the first frame, in seconds."),
                io.Float.Input("loop_fade", default=1.0, min=0.0, max=60.0, step=0.1, optional=True, advanced=True,
                               tooltip="loop: seconds over which the end fades into the start. 0 = no fade (for "
                                       "shaders that loop by themselves)."),
                io.Combo.Input("mask_from", options=list(MASK_FROM), default="auto", optional=True, advanced=True,
                               tooltip="What 'masks' is made of: the shader's alpha, or how bright the picture is. "
                                       "auto = alpha when the shader writes one, else brightness. A Shadertoy "
                                       "(mainImage) has no alpha, as on shadertoy.com: only a plain fragment shader "
                                       "(void main) can write one."),
                io.Boolean.Input("clip", default=True, optional=True, advanced=True,
                                 tooltip="With a mask connected: pictures and masks are black outside it."),
                io.Float.Input("bpm", default=0.0, min=0.0, max=400.0, step=0.1, optional=True, advanced=True,
                               tooltip="A tempo you know. 0 = found in the sound."),
                io.Float.Input("gain", default=1.0, min=0.0, max=10.0, step=0.05, optional=True, advanced=True,
                               tooltip="How loud the shader hears the sound."),
                io.Float.Input("release", default=0.15, min=0.0, max=5.0, step=0.01, optional=True, advanced=True,
                               tooltip="Seconds a sound level takes to fall after a peak."),
                io.Combo.Input("unconnected", options=list(UNCONNECTED), default="auto", optional=True, advanced=True,
                               tooltip="What a shader reads on image0 - image3 when nothing is connected there. auto = "
                                       "the built-in noise for your own code (most Shadertoys expect a noise picture), "
                                       "the sample facade for a preset."),
                io.String.Input("vector", default="", optional=True,
                                tooltip="An arrow for the shader to follow: draw it on the node (drag on the picture "
                                        "under 'draw the direction'), or type x, y of its start and x, y of its end, "
                                        "each 0 to 1 of the picture. Presets that run along the field (scan band, "
                                        "region contours ...) run from its start to its end when no field is connected; "
                                        "stripes, rain and pixel smear take its direction; rings, ripples and radar "
                                        "start at its start. Your own code reads kFrom, kTo, kDir and kVector. Empty = none."),
                io.Combo.Input("on_picture", options=list(ON_PICTURE), default="screen", optional=True,
                               tooltip="A picture on image0 and a shader that does not read it (the patterns, the 3d "
                                       "scenes, most pasted code): how the shader is laid over your picture. screen and "
                                       "add = its light on top; multiply = the shader as the light that falls on the "
                                       "picture (dark where the shader is dark); overlay = stronger contrast; "
                                       "difference = inverted where it is bright; displace = its light and dark bend "
                                       "the picture like glass; off = the shader alone (the picture only sets the "
                                       "size). With a mask the picture stays as it is outside it. 'masks' and 'depth' "
                                       "stay those of the shader. Presets that read the picture (picture looks, screen "
                                       "and tape, picture with depth) do not use this."),
                io.Float.Input("picture_amount", default=1.0, min=0.0, max=1.0, step=0.05, optional=True, advanced=True,
                               tooltip="on_picture: how much of the shader is laid over the picture. 0 = the picture alone."),
                RegionsType.Input("regions", optional=True,
                                  tooltip="Your regions (from kubakub regions from masks, from illustrator, from id maps "
                                          "...). The presets 'on your regions' treat every region on its own: its own "
                                          "colour, its own pace, a pattern that fits its box. Every other shader shows "
                                          "on the regions only (they are its mask when no mask is connected) and runs "
                                          "from each region's edge to its middle when it follows a field and none is "
                                          "connected. Your own code reads kRegion, kRegionUV, kRegionEdge, kRegionRand "
                                          "and kRegions. With image0 connected too, the picture stays as it is outside "
                                          "the regions: that picture is a material for kubakub region sampler ('material') "
                                          "and kubakub region video sampler."),
                io.String.Input("only", display_name=rp.WHICH, default="", optional=True, placeholder=rp.WHICH_HINT,
                                tooltip="With regions connected: the shader draws in these regions only. Empty = all "
                                        "regions. For another shader on other regions, put a second kubakub shader "
                                        "behind this one (its images into image0 of the next). " + rp.WHICH_TIP),
                SoundType.Input("sound", optional=True,
                                tooltip="From kubakub sound: the sound and how it is heard (tempo, gain, release, band, threshold, gap), set "
                                        "once for every node that listens, so all of them step on the same beat. Connected, it comes before '"
                                        "audio' and before this node's own sound settings."),
                TracksType.Input("tracks", optional=True,
                                 tooltip="From kubakub tracker: the shader follows what was tracked. Its arrow is the "
                                         "biggest thing's way, frame by frame (instead of a drawn one), and when no field "
                                         "is connected the field is a glow around everything tracked, so field presets "
                                         "(scan band, rings, contours ...) run from the tracked things. The clip is as "
                                         "long as the tracks unless 'seconds' or a sound says otherwise."),
            ],
            outputs=[
                io.Image.Output("images", tooltip="One picture per frame."),
                io.Mask.Output("masks", tooltip="One mask per frame: the shader's alpha or its brightness (mask_from)."),
                io.Float.Output("fps", tooltip="Frames per second."),
                io.Audio.Output("audio", tooltip="The sound, cut to the frames (silence without one)."),
                io.Int.Output("frame_count", tooltip="How many frames."),
                io.String.Output("report", tooltip="Size, length, what the shader uses, its controls and their values."),
                io.Image.Output("depth", tooltip="How near each pixel is, grey, near = white: from a shader that is a 3D scene "
                                                 "(the presets melting forms, colonnade, facade relief, extrude, growth; your own "
                                                 "code writes kDepth). For the depth presets on image1 of another kubakub "
                                                 "shader, kubakub height animate or a depth ControlNet. One black picture "
                                                 "from a shader that writes none."),
            ],
        )

    @classmethod
    def fingerprint_inputs(cls, preset="", code="", **kwargs):
        # a preset is a file: one that was saved again under its name (adjust > save as my preset) runs again
        if (code or "").strip():
            return ""
        try:
            return hashlib.sha256(st.preset_code(preset).encode("utf-8", "replace")).hexdigest()
        except st.ShaderError:
            return ""

    @classmethod
    def execute(cls, preset, code="", controls="", width=0, height=0, fps=25.0, seconds=0.0, speed=1.0, loops=1, loop=True,
                image0=None, image1=None, image2=None, image3=None, mask=None, field=None, audio=None,
                time_offset=0.0, loop_fade=1.0, mask_from="auto", clip=True, bpm=0.0, gain=1.0,
                release=0.15, unconnected="auto", vector="", on_picture="screen", picture_amount=1.0,
                regions=None, only="", sound=None, tracks=None) -> io.NodeOutput:
        notes = []
        audio, heard, cabled = so.take(sound, audio, bpm=bpm, gain=gain, release=release)
        bpm, gain, release = heard["bpm"], heard["gain"], heard["release"]
        try:
            arrow = mf.parse_vector(vector)
        except ValueError as e:
            raise ValueError(f"{NAME}: {e}") from e
        own = bool((code or "").strip())
        try:
            text = st.shader_text(preset, code)
            t = st.translate(text)
            values, more = st.parse_values(controls, t.controls)
        except st.ShaderError as e:
            raise ValueError(f"{NAME}: {e}") from e
        notes += [n for n in t.notes if own or "Wired differently" not in n] + more      # a preset's tabs are its own business

        fps = float(min(max(fps, 1.0), 120.0))
        pics = {i: p.float().cpu().numpy() for i, p in enumerate((image0, image1, image2, image3)) if p is not None and len(p)}
        ms = mask.float().cpu().numpy() if mask is not None else None
        fs = field.float().cpu().numpy() if field is not None else None
        ms = ms[None] if ms is not None and ms.ndim == 2 else ms
        fs = fs[None] if fs is not None and fs.ndim == 2 else fs
        track_arrows = None
        if tracks is not None:
            if not isinstance(tracks, dict) or tracks.get("format") != tk.TRACKS_FORMAT:
                raise ValueError(f"{NAME}: 'tracks' must come from kubakub tracker.")
            tn = int(tracks["n"])
            track_arrows = [tk.main_arrow(tracks, i) for i in range(tn)]
            if arrow:
                notes.append("tracks: the tracked arrow is used instead of the drawn one")
            arrow = next((a for a in track_arrows if a), None)       # so the shader knows it has an arrow at all
            if fs is None:
                fw_ = 480
                fh_ = max(1, int(round(fw_ * tracks["H"] / tracks["W"])))
                fs = np.stack([tk.near_field(tracks, i, fw_, fh_) for i in range(tn)])
                notes.append(f"tracks: the field is a glow around the {len(tracks['tracks'])} tracked things, frame by frame")
            notes.append(f"tracks: {tn} frames; the arrow follows the biggest tracked thing")
        first = next(iter(pics.values()), None)
        lab = regions.labels[0].cpu().numpy() if regions is not None and regions.labels.numel() else None
        if (only or "").strip() not in ("", "*"):
            if lab is None:
                notes.append(f"{rp.WHICH}: no regions are connected, so there is nothing to choose from (connect 'regions')")
            else:
                ids = rp.pick_or_say(regions.table, only, NAME)
                lab = np.where(np.isin(lab, ids), lab, -1)
                notes.append(f"{rp.WHICH}: the shader draws in {len(ids)} of your regions")
        size_of = first if first is not None else (ms if ms is not None else (fs if fs is not None else (lab[None] if lab is not None else None)))
        w = int(width) or (size_of.shape[2] if size_of is not None else 1280)
        h = int(height) or (size_of.shape[1] if size_of is not None else 720)
        if lab is None and t.regions:
            lab = mf.sample_regions(w, h)[0]
            notes.append("the shader works on regions and none are connected: the windows of the sample facade are used "
                         "(connect yours to 'regions')")
        elif lab is not None:
            if lab.shape != (h, w):
                import cv2
                lab = cv2.resize(lab.astype(np.int32), (w, h), interpolation=cv2.INTER_NEAREST)
            if ms is None:                            # whatever is connected does something: the regions are where it shows
                ms = (lab >= 0).astype(np.float32)[None]
                notes.append("regions: the shader shows on your regions (connect a mask to choose another place)"
                             + ("" if t.regions else "; the presets 'on your regions' treat each one on its own"))
            if fs is None and not arrow and "kField" in t.uses:
                fs = mf.field(lab, int(lab.max()) + 1, "edge distance", "each region")[0][None]
                notes.append("regions: the shader runs from each region's edge to its middle (connect a field of kubakub "
                             "mask field for another path)")

        wave, sr_hz, mono = _sound_in(audio)
        sample_sound = False
        # whatever is connected does something: a sound opens a preset's 'sound' / 'beat' control (they rest at 0),
        # and an input the shader has no use for is named in the report
        if mono is not None and not own:
            typed = {m.lower() for m in re.findall(r"(\w+?)_?\s*=", str(controls or ""))}
            opened = [c for c in t.controls if c.name in ("sound", "beat") and c.default == 0 and c.name not in typed]
            for c in opened:
                values[c.name] = min(1.0, c.hi)
            if opened:
                notes.append("a sound is connected: " + ", ".join(f"{c.name} = {values[c.name]:g}" for c in opened)
                             + " (it rests at 0 without one; type another value in controls)")
            elif not t.listens:
                notes.append("audio: this shader does not listen to the sound; it is passed on to 'audio'")
        idle = [f"image{i}" for i in sorted(pics) if i and i not in t.channels]
        if idle:
            notes.append(f"{', '.join(idle)}: this shader does not read " + ("them" if len(idle) > 1 else "it"))
        if fs is not None and "kField" not in t.uses:
            notes.append("field: this shader does not run along a field (scan band, brick wave, region contours and extrude do)")
        longest = max([len(p) for p in pics.values()] + [len(x) for x in (ms, fs) if x is not None] + [1])
        if seconds > 0:
            secs = float(seconds)
        elif mono is not None:
            secs = len(mono) / float(sr_hz)
        elif longest > 1:
            secs = longest / fps
        else:
            secs = 4.0
        frames = max(1, int(round(secs * fps)))
        secs = frames / fps                           # the sound and kSeconds are as long as the frames, to the sample
        if mono is None and t.listens:
            (wave, sr_hz, mono), sample_sound = _built_in_beat(secs), True
            notes.append("the shader listens to a sound and none is connected: the built-in beat is used (connect "
                         "yours to 'audio')")
        feedback = "kPrev" in t.uses or bool(t.buffers)      # it builds on its own last frame: no loop of its own, node only
        fade = min(frames, int(round(float(loop_fade) * fps))) if loop and (own or feedback) and frames > 1 else 0
        if fs is None and not arrow and not own and st.sample_field(text):
            lab, n_regions, _ = mf.sample_regions()
            fs = mf.field(lab, n_regions, "edge distance", "each region")[0][None]
            notes.append("the shader works on the regions of a field and none is connected: the windows of the sample "
                         "facade are used (connect the field of kubakub mask field)")

        _ram_check(NAME, frames, w, h, 28 if t.depth else 16)

        textures, noisy = {}, []
        empty = unconnected if unconnected in UNCONNECTED[1:] else None      # auto: noise for your code, the sample for a preset
        for i in t.channels:
            if i in pics:
                textures[i] = (lambda a: lambda k: (k % len(a), a[k % len(a)]))(pics[i])
            elif empty == "noise" or (empty is None and own):
                textures[i] = sr.noise_texture(7 + 2 * i)
                noisy.append(f"image{i}")
            elif empty is None and i in t.optional:
                pass                                  # the preset draws something of its own when no picture is there
            elif empty == "sample facade" or (empty is None and i == 0):
                from ...kubakub import sample_facade as sf
                pic = sf.render(w, h)["image"]
                textures[i] = lambda k, pic=pic: (0, pic)
                notes.append(f"the shader reads image{i} and nothing is connected: the sample facade is used")
            elif i not in t.optional:
                notes.append(f"the shader reads image{i} and nothing is connected: it is black there")
        if noisy:
            notes.append(f"the shader reads {', '.join(noisy)} and nothing is connected: the built-in noise is used (on "
                         "Shadertoy a noise or a picture sits there; connect the picture it expects if it looks wrong)")
        if ms is not None or fs is not None:
            fw, fh = (fs if fs is not None else ms).shape[2], (fs if fs is not None else ms).shape[1]

            def both(k):                              # field in the first channel, mask in the second
                pair = np.ones((fh, fw, 2), np.float32)
                pair[..., 0] = fs[k % len(fs)] if fs is not None else sr.default_field(fw)[..., 0]
                if ms is not None:
                    pair[..., 1] = _fit(ms[k % len(ms)], fw, fh)
                return pair

            n_f, n_m = (len(fs) if fs is not None else 1), (len(ms) if ms is not None else 1)
            held = [None, None]                       # a still field is put together once, not at every frame

            def field_source(k):
                key = (k % n_f, k % n_m)
                if held[0] != key:
                    held[0], held[1] = key, both(k)
                return key, held[1]

            textures[sr.FIELD_UNIT] = field_source

        per_frame = sr.frame_times(frames, fps, float(speed), float(time_offset), fade, int(loops))
        sound_lines = []
        if mono is not None:
            heard, sound_lines = sr.sound_values(mono, sr_hz, [i / fps for i in range(frames)], float(bpm), sample_sound,
                                                 float(release), float(gain))
            for i, d in enumerate(per_frame):
                d.update({k: v[i % frames] for k, v in heard.items()})
        if track_arrows is not None:                  # the tracked arrow, frame by frame (none on a frame = no arrow there)
            for i, d in enumerate(per_frame):
                a = track_arrows[i % len(track_arrows)]
                d.update(sr.vector_values(a, as_field=False) if a else {"kVector": 0.0, "kVecField": 0.0})
        if lab is not None:
            textures[sr.REGION_UNIT], values["kRegions"] = sr.region_source(lab)
        static = {"iFrameRate": fps, "iSampleRate": float(sr_hz or 44100), "kSeconds": secs, "kDepthOut": 1.0 if t.depth else 0.0,
                  "kClip": 1.0 if clip and ms is not None else 0.0, **sr.vector_values(arrow, as_field=fs is None), **values}
        if arrow:
            reads, as_field = t.uses & st.VECTOR, fs is None and "kField" in t.uses
            if reads or as_field:
                notes.append("the drawn arrow: the shader follows it" + (" (as its field)" if as_field else ""))
            elif t.follows:
                notes.append("the drawn arrow: this shader does not read it (a connected field comes first)")
            else:
                notes.append("the drawn arrow: this shader does not use it (it has no direction to set)")

        images = torch.empty((frames, h, w, 3), dtype=torch.float32)
        alphas = torch.empty((frames, h, w), dtype=torch.float32)
        img, alp = images.numpy(), alphas.numpy()
        pbar = comfy.utils.ProgressBar(frames + fade)

        bad, clock = [False], []

        def on_frame(i, px):                          # 0-1 and cut to the mask already, on the GPU
            j = i % frames
            clock.append(time.perf_counter()) if i == 0 or i == frames + fade - 1 else None      # the first frame also compiles
            bad[0] = bad[0] or not bool(np.isfinite(px).all())
            if i < frames:
                img[j], alp[j] = px[..., :3], px[..., 3]
            else:                                     # the tail of a loop: it fades out over the first frames
                k = (i - frames) / float(fade)
                img[j] = px[..., :3] * (1.0 - k) + img[j] * k
                alp[j] = px[..., 3] * (1.0 - k) + alp[j] * k
            pbar.update_absolute(i + 1, frames + fade)

        try:
            sr.render(t.source, w, h, per_frame, static, textures, on_frame,
                      comfy.model_management.throw_exception_if_processing_interrupted, t.buffers, t.inputs)
        except st.ShaderError as e:
            raise ValueError(f"{NAME}: {e}") from e
        each = (clock[-1] - clock[0]) / (frames + fade - 1) if frames + fade > 1 else 0.0
        if each > SLOW:
            notes.append(f"slow: {each:.2f} s per frame. " + ("Lower 'steps' in controls, or the size."
                         if any(c.name == "steps" for c in t.controls) else "A smaller size is faster."))

        if bad[0]:                                    # a shader that divides by zero
            images.nan_to_num_(nan=0.0, posinf=1.0, neginf=0.0).clamp_(0.0, 1.0)
            alphas.nan_to_num_(nan=0.0, posinf=1.0, neginf=0.0).clamp_(0.0, 1.0)
            notes.append("the shader wrote values that are no numbers (a division by zero?): they are black")
        how = mask_from if mask_from in MASK_FROM else "auto"
        depth = None
        if t.depth:                                   # the alpha carried the depth: the mask is the brightness
            if how == "alpha":
                notes.append("mask_from = alpha: this shader gives a depth instead of an alpha, so the mask is its brightness")
            how = "brightness"
            if clip and ms is not None:
                for a in range(frames):
                    alphas[a] *= torch.from_numpy(_fit(ms[a % len(ms)], w, h))
            depth = alphas[..., None].repeat(1, 1, 1, 3)
        if how == "alpha" and t.kind == "shadertoy":
            how = "brightness"
            notes.append("mask_from = alpha: a Shadertoy (mainImage) has no alpha, as on shadertoy.com, so the mask is "
                         "its brightness. A plain fragment shader (void main) can write one.")
        if how == "auto":
            how = "alpha" if float(alphas.min()) < 0.999 else "brightness"
        if how == "brightness":
            masks = _luma_masks(images)
        else:
            masks = alphas
            if clip and ms is not None:               # the picture was cut on the GPU; the alpha is cut here
                for a in range(frames):
                    masks[a] *= torch.from_numpy(_fit(ms[a % len(ms)], w, h))

        if 0 in pics and 0 not in t.channels:         # a picture is there and the shader does not read it: it goes under
            how_over = on_picture if on_picture in ON_PICTURE else "screen"
            if how_over == "off":
                notes.append("image0: this shader does not read it, and on_picture is off: the picture only sets the size")
            else:
                under = pics[0]
                for a in range(frames):
                    comfy.model_management.throw_exception_if_processing_interrupted()
                    lay_over(img[a], _fit(np.ascontiguousarray(under[a % len(under)][..., :3]), w, h), how_over, picture_amount,
                             _fit(ms[a % len(ms)], w, h) if clip and ms is not None else None)
                notes.append(f"image0: this shader does not read it, so it is laid over your picture (on_picture = {how_over}"
                             + (f", {picture_amount:g} of it" if picture_amount < 1.0 else "") + "); masks"
                             + (" and depth" if t.depth else "") + " are the shader's own. off = the shader alone")

        sound = _sound_out(wave, sr_hz, secs)

        what = ("your " + ("OSL shader" if t.language == "osl" else "Shadertoy" if t.kind == "shadertoy" else "fragment shader")) if own else f"preset '{preset}'"
        if t.buffers:
            what += f" with {len(t.buffers)} buffer tab{'s' if len(t.buffers) > 1 else ''} ({', '.join(b.name for b in t.buffers)})"
        lines = [f"{frames} frames, {w}x{h}, {secs:.2f} s at {fps:g} fps", what + (f": {st.about(text)}" if not own and st.about(text) else "")]
        if loop:
            lines.append("loop: " + (f"the end fades into the start over {fade / fps:.2f} s" if fade else
                                     ("the preset loops by itself" if not own and not feedback else "no fade (loop_fade = 0)")))
        if t.controls:
            lines.append("controls: " + _controls_text(t, values))
        lines.append(f"masks: the shader's {how}")
        if depth is not None:
            lines.append("depth: near = white")
        report = "\n".join(lines + sound_lines + notes)
        log.info("[KUBA shader] %s", report.replace("\n", "\n    "))
        if depth is None:
            depth = torch.zeros((1, h, w, 3), dtype=torch.float32)
        return io.NodeOutput(images, masks, fps, sound, frames, report, depth, ui=_preview(img, fps))


def _preview(out, fps):
    """The frames as a small animated WebP in ComfyUI's temp folder, shown on the node."""
    try:
        import cv2
        from PIL import Image
        step = max(1, int(np.ceil(len(out) / PREVIEW_FRAMES)))
        k = min(1.0, PREVIEW_PX / float(out.shape[2]))
        size = (max(8, int(out.shape[2] * k)), max(8, int(out.shape[1] * k)))
        imgs = [Image.fromarray((np.clip(cv2.resize(m, size, interpolation=cv2.INTER_AREA), 0, 1) * 255).astype(np.uint8))
                for m in out[::step]]
        fn = "kubakub_shader_" + "".join(random.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(6)) + ".webp"
        folder = folder_paths.get_temp_directory()
        os.makedirs(folder, exist_ok=True)
        imgs[0].save(os.path.join(folder, fn), save_all=True, append_images=imgs[1:], duration=int(round(1000.0 * step / fps)),
                     loop=0, quality=80, method=2)
        return {"images": [{"filename": fn, "subfolder": "", "type": "temp"}], "animated": (True,)}
    except Exception as e:  # noqa: BLE001      the preview must never cost the frames
        log.warning("[KUBA shader] no preview: %s", e)
        return None


def shader_info(preset="", code="", controls="") -> dict:
    """What the director window needs to draw a shader itself (WebGL 2 runs the same text): the complete fragment
    shader, its controls, or the reason it cannot run."""
    try:
        text = st.shader_text(preset, code)
        t = st.translated(text)
        own = bool((code or "").strip())
        return {"source": t.source, "about": st.about(text), "notes": t.notes, "channels": t.channels,
                "type": "" if own else st.type_of(preset), "mine": not own and st.is_mine(preset),
                "text": "" if own else text, "values": st.parse_values(controls, t.controls)[0],
                "optional": sorted(t.optional), "buffers": [b.name for b in t.buffers], "wants_field": st.sample_field(text),
                "kind": t.kind, "uses": sorted(t.uses),
                "controls": [{"name": c.name, "default": c.default, "min": c.lo, "max": c.hi, "about": c.about, "colour": c.colour} for c in t.controls]}
    except st.ShaderError as e:
        return {"error": str(e)}


def _builds_up(t) -> bool:
    """The shader builds every frame on the ones before (kPrev, or the Buffer tabs of a Shadertoy)."""
    return "kPrev" in t.uses or bool(t.buffers)


def node_only() -> list:
    """Presets that need what only the node has: their own last frame (slit scan, datamosh, trails). The director's
    fill menu leaves them out. A preset that works on a picture is an effect on the layers below there."""
    return [name for name in st.presets() if _builds_up(st.translated(st.preset_code(name)))]


def preset_tags(name) -> list:
    """What a preset needs and gives, as short words for the pickers."""
    text = st.preset_code(name)
    t = st.translated(text)
    tags = []
    if 0 in t.channels and 0 not in t.optional:
        tags.append("your picture")
    if _builds_up(t) and 0 in t.optional:
        tags.append("a clip")
    if 1 in t.channels:
        tags.append("a depth pass" if "depth pass" in st.about(text) else "a normal pass")
    if st.sample_field(text) or "kField" in t.uses:
        tags.append("a field")
    if t.regions:
        tags.append("your regions")
    if t.depth:
        tags.append("gives a depth")
    if _builds_up(t):
        tags.append("node only")
    return tags


TILE, COLUMNS = (160, 90), 6
PREVIEW, PREVIEW_FRAMES, PREVIEW_FPS = (320, 180), 36, 12
_SHEET = {}
_PREVIEWS = {}
_HEARD = {"kLoop": 0.3, "iTime": 1.2, "kLevel": 0.8, "kLow": 0.9, "kMid": 0.5, "kHigh": 0.7, "kBeat": 0.6, "kBeats": 3.0}


@functools.lru_cache(maxsize=4)
def _sample_region_source(w, h):
    """The windows of the sample facade as a shader's regions, at one size (the pickers ask again and again)."""
    return sr.region_source(mf.sample_regions(w, h)[0])


def _preset_setup(name, w, h, facade, text=None, values=None):
    """(translated, static values, pictures) to draw a preset with nothing connected, as the node does. text: pasted
    code instead of the preset (it reads the built-in noise, as on the node); values: its controls."""
    own = text is not None
    text = text if own else st.preset_code(name)
    t = st.translated(text)
    static = {"kSeconds": 4.0, "iFrameRate": 25.0, **{c.name: c.default for c in t.controls}, **(values or {})}
    pictures = {0: lambda k: (0, facade)} if 0 in t.channels and 0 not in t.optional else None
    if own:
        pictures = {i: sr.noise_texture(7 + 2 * i) for i in t.channels} or None
    if st.sample_field(text):                         # it works on regions: the windows of the sample facade, as on the node
        lab, n_regions, _ = mf.sample_regions()
        pair = np.stack([mf.field(lab, n_regions)[0], np.ones(lab.shape, np.float32)], -1)
        pictures = {**(pictures or {}), sr.FIELD_UNIT: lambda k, pair=pair: ("field", pair)}
    if t.regions:
        source, static["kRegions"] = _sample_region_source(w, h)
        pictures = {**(pictures or {}), sr.REGION_UNIT: source}
    return t, static, pictures


def preset_preview(name) -> bytes:
    """One loop of a preset as a small moving picture (animated WebP): what the pickers play under the pointer.
    Drawn on the first request, and again when the preset file changes."""
    path = st.presets().get(name)
    if not path:
        raise st.ShaderError(f"no preset '{name}'.")
    key = os.path.getmtime(path)
    hit = _PREVIEWS.get(name)
    if hit is None or hit[0] != key:
        hit = _PREVIEWS[name] = (key, look(name))
    return hit[1]


LOOK = (480, 270)


def look(preset="", code="", controls="", moving=True, size=PREVIEW) -> bytes:
    """A shader with these controls and nothing connected, as the node draws it: one loop as an animated WebP
    (moving), or one frame as a JPEG. What the pickers play, and what 'adjust' shows while a slider moves."""
    import io as _io
    from PIL import Image
    from ...kubakub import sample_facade as sf
    own = bool((code or "").strip())
    values = st.parse_values(controls, st.translated(st.shader_text(preset, code)).controls)[0]
    w, h = size
    t, static, pictures = _preset_setup(preset, w, h, sf.render(w, h)["image"], code if own else None, values)
    n = PREVIEW_FRAMES
    warm = (150 if t.buffers else 40) if _builds_up(t) else 0      # it builds up: show it after a while
    frames = []
    for k in range(warm + n):
        u = k / n
        frames.append(dict(_HEARD, kLoop=u % 1.0, iTime=4.0 * u, iTimeDelta=4.0 / n, iFrame=k, kBeat=max(0.0, 1.0 - (4.0 * u) % 1.0),
                           kBeats=float(int(4.0 * u)), kLow=0.55 + 0.4 * np.cos(2 * np.pi * 2 * u), kLevel=0.6 + 0.3 * np.cos(2 * np.pi * 2 * u)))
    if not moving:                                    # one frame, a third into the loop (where the sheet of pictures looks too)
        frames = frames[:warm + n // 3 + 1] if warm else [dict(_HEARD, iFrame=0, iTimeDelta=0.04)]
    got = []
    last = len(frames) - 1
    sr.render(t.source, w, h, frames, static, pictures,
              lambda k, px: got.append((np.clip(px[..., :3], 0, 1) * 255).astype(np.uint8)) if k >= (warm if moving else last) else None,
              buffers=t.buffers, inputs=t.inputs)
    imgs = [Image.fromarray(g) for g in got]
    buf = _io.BytesIO()
    if moving:
        imgs[0].save(buf, "WEBP", save_all=True, append_images=imgs[1:], duration=int(round(1000.0 / PREVIEW_FPS)), loop=0, quality=72, method=2)
    else:
        imgs[0].save(buf, "JPEG", quality=88)
    return buf.getvalue()


def thumbs_sheet() -> bytes:
    """One JPEG with a small picture of every preset (rows of COLUMNS tiles, in menu order): what the preset pickers
    on the node and in the director show. Drawn once, and again when a preset file changes."""
    import io as _io
    from PIL import Image
    from ...kubakub import sample_facade as sf
    found = st.presets()
    key = tuple((n, os.path.getmtime(p)) for n, p in found.items())
    if _SHEET.get("key") != key:
        w, h = TILE
        rows = (len(found) + COLUMNS - 1) // COLUMNS
        sheet = np.zeros((max(1, rows) * h, COLUMNS * w, 3), np.float32)
        facade = sf.render(w, h)["image"]
        heard = _HEARD
        for i, name in enumerate(found):
            try:
                t, static, pictures = _preset_setup(name, w, h, facade)
                if _builds_up(t):                     # it builds up over frames: the picture is what it shows after a while
                    last, n = [], 150 if t.buffers else 40
                    sr.render(t.source, w, h, [dict(heard, kLoop=k / 100.0, iTime=k / 25.0, iFrame=k) for k in range(n)],
                              static, pictures, lambda k, p: last.append(p.copy()) if k == n - 1 else None,
                              buffers=t.buffers, inputs=t.inputs)
                    px = last[0]
                else:
                    px = sr.still(t.source, w, h, heard, static, pictures)
                sheet[(i // COLUMNS) * h:(i // COLUMNS + 1) * h, (i % COLUMNS) * w:(i % COLUMNS + 1) * w] = px[..., :3]
            except Exception as e:  # noqa: BLE001      one broken preset leaves a black tile
                log.warning("[KUBA shader] no picture of preset '%s': %s", name, str(e).splitlines()[0])
        buf = _io.BytesIO()
        Image.fromarray((np.clip(sheet, 0, 1) * 255).astype(np.uint8)).save(buf, "JPEG", quality=88)
        _SHEET.update(key=key, jpeg=buf.getvalue())
    return _SHEET["jpeg"]


def _routes():
    try:
        from aiohttp import web
        from server import PromptServer
        routes = PromptServer.instance.routes
    except Exception:  # noqa: BLE001      no server (the tests import the nodes without one)
        return

    async def presets(request):
        found = st.presets()
        return web.json_response({"presets": list(found), "about": {n: st.about(st.preset_code(n)) for n in found},
                                  "types": [{"key": k, "title": title, "about": about} for k, title, about in st.TYPES],
                                  "type": {n: st.type_of(n) for n in found},
                                  "tags": {n: preset_tags(n) for n in found},
                                  "node_only": node_only(), "tile": list(TILE), "columns": COLUMNS})

    async def preview(request):
        import asyncio
        name = str(request.query.get("name") or "")
        if name not in st.presets():                  # only the names of the list, never a path
            return web.Response(status=404, text="no such preset")
        try:
            data = await asyncio.get_running_loop().run_in_executor(None, preset_preview, name)
        except Exception as e:  # noqa: BLE001      no GL here, or a broken preset: the picker keeps its still
            return web.Response(status=503, text=str(e).splitlines()[0] if str(e) else "no preview")
        return web.Response(body=data, content_type="image/webp", headers={"Cache-Control": "no-cache"})

    async def thumbs(request):
        import asyncio
        try:
            data = await asyncio.get_running_loop().run_in_executor(None, thumbs_sheet)
        except Exception as e:  # noqa: BLE001      no GL here: the pickers show names only
            return web.Response(status=503, text=str(e))
        return web.Response(body=data, content_type="image/jpeg", headers={"Cache-Control": "no-cache"})

    async def translate(request):
        try:
            body = await request.json()
        except Exception:  # noqa: BLE001
            body = {}
        import asyncio                                # off the server's own thread: a long text must not hold up the UI
        return web.json_response(await asyncio.get_running_loop().run_in_executor(
            None, shader_info, str(body.get("preset") or ""), str(body.get("code") or ""), str(body.get("controls") or "")))

    async def look_route(request):                    # what 'adjust' shows: the shader with the controls as they stand
        import asyncio
        try:
            body = await request.json()
        except Exception:  # noqa: BLE001
            body = {}
        moving = bool(body.get("moving"))
        try:
            data = await asyncio.get_running_loop().run_in_executor(
                None, look, str(body.get("preset") or ""), str(body.get("code") or ""), str(body.get("controls") or ""), moving, LOOK)
        except Exception as e:  # noqa: BLE001
            return web.Response(status=422, text=str(e).splitlines()[0] if str(e) else "no picture")
        return web.Response(body=data, content_type="image/webp" if data[:4] == b"RIFF" else "image/jpeg", headers={"Cache-Control": "no-store"})

    async def save(request):                          # 'save as my preset': a .glsl file in ComfyUI's user folder
        try:
            body = await request.json()
            preset, code = str(body.get("preset") or ""), str(body.get("code") or "")
            text = st.shader_text(preset, code)
            values = st.parse_values(str(body.get("controls") or ""), st.translated(text).controls)[0]
            return web.json_response({"name": st.save_preset(str(body.get("name") or ""), text, values)})
        except (st.ShaderError, OSError) as e:
            return web.json_response({"error": str(e)})
        except Exception:  # noqa: BLE001
            return web.json_response({"error": "nothing to save."})

    async def delete(request):
        try:
            body = await request.json()
            st.delete_preset(str(body.get("name") or ""))
            return web.json_response({"ok": True})
        except (st.ShaderError, OSError) as e:
            return web.json_response({"error": str(e)})
        except Exception:  # noqa: BLE001
            return web.json_response({"error": "nothing to delete."})

    routes.get("/kubakub/shader/presets")(presets)
    routes.get("/kubakub/shader/thumbs")(thumbs)
    routes.get("/kubakub/shader/preview")(preview)
    routes.post("/kubakub/shader/look")(look_route)
    routes.post("/kubakub/shader/save")(save)
    routes.post("/kubakub/shader/delete")(delete)
    routes.post("/kubakub/shader/translate")(translate)


_routes()

NODE_CLASS_MAPPINGS = {"KUBA_Shader": KUBA_Shader}
NODE_DISPLAY_NAME_MAPPINGS = {"KUBA_Shader": NAME}
