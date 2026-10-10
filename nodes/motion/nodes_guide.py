"""
nodes_guide.py

kubakub motion guide (ltx): a rough moving clip (a director sequence, a shader, a 3D previz) becomes a finished video
that keeps the clip's motion. The guide is VAE-encoded and LTX 2.x (22B distilled) samples from it instead of from
pure noise: `repaint` is the noise level it starts at. `repaint_mask` (where) and `curve` (when) come in two ways
(`mask_does`), because of what was measured on the GPU: any guide left untouched next to a repainted place anchors
the look, so a noise mask never gives 'stone here, previz there'. 'shows the repaint' therefore repaints the whole
clip and shows it through the mask and the curve over the guide; 'limits the change' is the noise mask: LTX may only
change things inside it and keeps the guide's look (for a guide that is a finished picture already). Measured on a band of light sweeping over a facade (512x320,
25 frames): from 0.91 the stone, windows and light are repainted and the band is 5 px off the guide's; from 0.725 it
follows and barely repaints; from 1.0 the guide is ignored. An optional second stage after the x2 latent upscaler.
The LTX steps are the region video sampler's (nodes_video._OPS, core's own node implementations); the logic is in
kubakub/video.py. V3 node schema, registered via ../__init__.py.
"""

from __future__ import annotations

import logging
import time

import cv2
import numpy as np
import torch

from comfy_api.latest import io

from ...kubakub import maskfields as mf
from ...kubakub import sound as so
from ...kubakub import video as vd
from . import nodes_video as nv

log = logging.getLogger("KUBA.regions")

PROMPT = ("a sandstone building facade at night, a bright vertical band of warm light sweeps across the facade from "
          "left to right, windows glow, cinematic, detailed stone")
NEGATIVE = "camera movement, zoom, pan, shaky, blurry, low quality, text, watermark"
MASK_DOES = ("shows the repaint", "limits the change")


def _text(sigmas):
    return ", ".join(f"{s:g}" for s in sigmas)


class KUBA_MotionGuide(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="KUBA_MotionGuide",
            display_name="kubakub motion guide (ltx)",
            category="kubakub/2d/motion",
            search_aliases=["ltx", "video to video", "v2v", "previz", "restyle", "motion control"],
            description=(
                "A rough moving clip becomes a finished video that keeps its motion. Connect a clip as 'guide' (the "
                "frames of a kubakub director sequence, a kubakub shader, a 3D previz, moving masks over a "
                "picture), say in 'prompt' what it should look like, and LTX 2.x repaints it: what moves, where "
                "and when stays yours. 'repaint' is how far it may go: 0.91 repaints everything and still "
                "follows, lower keeps more of the guide's own look, 1 ignores the guide. No guide connected = a "
                "sample (a band of light sweeping over the sample facade)."),
            inputs=[
                io.Model.Input("model", tooltip="LTX 2.5 22B distilled (UNETLoader)."),
                io.Clip.Input("clip", tooltip="Gemma text encoder for LTX (CLIPLoader, type ltxv)."),
                io.Vae.Input("vae", tooltip="LTX video VAE."),
                io.Vae.Input("audio_vae", tooltip="LTX audio VAE (the model is audio-video)."),
                io.Image.Input("guide", optional=True,
                               tooltip="The clip whose motion is kept: director sequence frames, a shader, a previz "
                                       "render. Rough is fine, LTX repaints it. Empty = a built-in sample."),
                io.String.Input("prompt", multiline=True, default=PROMPT,
                                tooltip="What the video should look like. Also say what moves (the same as in the "
                                        "guide): the prompt and the guide should agree."),
                io.Float.Input("repaint", default=0.91, min=0.3, max=1.0, step=0.01,
                               tooltip="How much LTX repaints. 0.91 = everything is repainted, the motion still "
                                       "follows the guide (the tested value). 0.95-0.98 = freer, follows less. "
                                       "0.7 = follows closely and keeps the guide's own look. 1 = the guide is ignored."),
                io.Int.Input("seed", default=42, min=0, max=0xFFFFFFFFFFFFFFFF,
                             control_after_generate=io.ControlAfterGenerate.fixed,
                             tooltip="Noise seed. Change it for another take of the same motion."),
                io.Float.Input("fps", default=25.0, min=1.0, max=120.0, step=0.01,
                               tooltip="Frame rate of the guide and of the video."),
                io.Float.Input("megapixels", default=0.2, min=0.05, max=2.0, step=0.05,
                               tooltip="Size of the first stage (the guide is scaled to it, sides in steps of 32). "
                                       "With the upscaler the video is twice as wide and high. 0.2 is safe on 12 GB."),
                io.LatentUpscaleModel.Input("upscale_model", optional=True,
                                            tooltip="LTX latent spatial upscaler x2: adds a sharper second stage."),
                io.Audio.Input("audio", optional=True,
                               tooltip="A sound the video is sampled with (held as it is), e.g. the director's. "
                                       "It is passed on, cut to the video's length."),
                io.Int.Input("max_frames", advanced=True, default=121, min=9, max=1025, step=8,
                             tooltip="A longer guide is cut here (121 frames at 0.6 megapixels is safe on 12 GB)."),
                io.Float.Input("stage2_max_mp", advanced=True, default=0.6, min=0.1, max=8.0, step=0.05,
                               tooltip="Cap for the second stage (4x the first stage's pixels): the first stage "
                                       "shrinks to fit."),
                io.Boolean.Input("guide_size", advanced=True, default=False,
                                 tooltip="Scale the frames back to the guide's own size (needs RAM for a large "
                                         "guide). Off = the size LTX rendered."),
                io.String.Input("negative", advanced=True, multiline=True, optional=True, default=NEGATIVE,
                                tooltip="What the video should avoid."),
                io.Mask.Input("repaint_mask", optional=True,
                              tooltip="Where the repaint is: white = LTX's picture, black = your guide as it is, grey = "
                                      "a mix. One mask, or one per frame (kubakub mask animate, a shader's masks, an ID "
                                      "pass): a moving mask lets the repaint travel over the picture. Empty = "
                                      "everywhere."),
                io.Combo.Input("curve", options=list(mf.TIMED), default="constant", optional=True,
                               tooltip="When the repaint is there: how its amount runs over the clip. constant = all "
                                       "the time. ramp: from your guide to fully repainted, once. saw / triangle / sine "
                                       "/ square: again and again. sound level, beats, bars, hits: with the sound "
                                       "connected to 'audio'. Multiplies the repaint mask."),
                io.Float.Input("cycle", default=2.0, min=0.02, max=3600.0, step=0.01, optional=True,
                               tooltip="curve: seconds for one run (ramp: how long it takes; beats and hits: how long "
                                       "each one's ramp is)."),
                io.Float.Input("curve_from", default=0.0, min=0.0, max=1.0, step=0.01, optional=True,
                               tooltip="curve: the amount of repaint at its low end. 0 = the guide as it is."),
                io.Float.Input("curve_to", default=1.0, min=0.0, max=1.0, step=0.01, optional=True,
                               tooltip="curve: the amount at its high end. 1 = the full 'repaint'."),
                io.Combo.Input("easing", options=list(mf.EASES), default="linear", optional=True, advanced=True,
                               tooltip="curve: how a ramp starts and stops."),
                io.Boolean.Input("keep_exact", default=True, optional=True, advanced=True,
                                 tooltip="mask_does = limits the change: where nothing may change the guide's own pixels "
                                         "are put back after decoding, so they stay exact. Off = LTX's decoded picture "
                                         "everywhere."),
                io.Combo.Input("mask_does", options=list(MASK_DOES), default=MASK_DOES[0], optional=True,
                               tooltip="What the mask and the curve do. shows the repaint: the whole clip is repainted "
                                       "and shown through them over your guide, so the new look travels over the "
                                       "picture or comes in over time. limits the change: LTX may only change things "
                                       "inside the mask and keeps your guide's look everywhere (for a guide that is a "
                                       "finished picture, when only a part should come alive)."),
            ],
            outputs=[
                io.Image.Output("frames", tooltip="The video, as long as the guide."),
                io.Float.Output("fps", tooltip="The frame rate, passed through for Create Video."),
                io.Audio.Output("audio", tooltip="The connected audio cut or padded with silence to the length of "
                                                 "the frames; silence of that length when none is connected."),
                io.String.Output("report", tooltip="Sizes, the noise levels sampled through, time, and what to try."),
            ],
        )

    @classmethod
    def execute(cls, model, clip, vae, audio_vae, prompt, repaint, seed, fps, megapixels, guide=None,
                upscale_model=None, audio=None, max_frames=121, stage2_max_mp=0.6, guide_size=False,
                negative=NEGATIVE, repaint_mask=None, curve="constant", cycle=2.0, curve_from=0.0, curve_to=1.0,
                easing="linear", keep_exact=True, mask_does=MASK_DOES[0]) -> io.NodeOutput:
        ops = nv._OPS
        t0 = time.perf_counter()
        notes = []
        if audio is not None and audio["waveform"].dim() != 3:              # [channels, samples] -> a batch of one
            audio = dict(audio, waveform=audio["waveform"].reshape(1, -1, audio["waveform"].shape[-1]))
        if guide is None:
            src = vd.sample_guide()
            notes.append("no guide connected: the built-in sample is used (a band of light over the sample facade)")
        else:
            src = guide[..., :3].float().cpu().numpy()
        if len(src) < 2:
            notes.append("the guide is one picture: there is no motion to keep, LTX invents its own")
        if len(src) > int(max_frames):
            notes.append(f"the guide has {len(src)} frames: cut to max_frames = {int(max_frames)}")
            src = src[:int(max_frames)]
        n, gh, gw = len(src), src.shape[1], src.shape[2]
        mp = float(megapixels)
        if upscale_model is not None and mp * 4 > stage2_max_mp:
            mp = float(stage2_max_mp) / 4              # the second stage is 2x per side: keep it inside the cap
            notes.append(f"first stage at {mp:.2f} megapixels, so that the second stays under stage2_max_mp")
        w1, h1 = vd.guide_size(gw, gh, mp)
        pixels = torch.from_numpy(vd.guide_frames(src, w1, h1))
        F = pixels.shape[0]
        sig1 = vd.guide_sigmas(repaint)
        sig2 = vd.guide_sigmas(min(float(repaint), vd.GUIDE_SIGMAS_2[0]), vd.GUIDE_SIGMAS_2)

        # where and when: a mask and a curve over the frames, 0-1 each
        where = None
        if repaint_mask is not None:
            where = repaint_mask.float().cpu().numpy()             # 0-1 as masks are; clipped where it is used
            where = where[None] if where.ndim == 2 else where.reshape(-1, *where.shape[-2:])
        mono, hz, own = None, 44100, False
        if mf.needs_sound(curve):
            if audio is not None:
                mono, hz = so.to_mono(audio["waveform"]), int(audio["sample_rate"])
            else:
                own, mono = True, so.sample_sound(max(1.0, n / float(fps)), mf.SAMPLE_BPM, hz)
                notes.append(f"curve = {curve} and no sound is connected: the built-in beat is used (connect yours to 'audio')")
        when, curve_lines = mf.timed_values([i / float(fps) for i in range(n)], curve, cycle, min(max(curve_from, 0.0), 1.0),
                                            min(max(curve_to, 0.0), 1.0), easing, mono, hz, own)
        masked = where is not None or when is not None
        limit = masked and mask_does == MASK_DOES[1]   # the noise mask; else the repainted clip is shown through the mask
        share = vd.repaint_share(where, when, n) if masked else None

        def noise_mask(video, start):
            """What may change, as LTX reads it for a stage that starts at noise level `start`."""
            s = video["samples"]
            return torch.from_numpy(vd.repaint_level(vd.repaint_latent(where, when, n, s.shape[2], s.shape[4], s.shape[3]), start))

        pbar = ops.progress(4 if upscale_model is not None else 3)
        latent = ops.vae_encode(vae, pixels)
        del pixels
        if limit:
            latent["noise_mask"] = noise_mask(latent, sig1[0])
        text = (prompt or "").strip() or PROMPT
        pos, neg = ops.encode_text(clip, text), ops.encode_text(clip, negative or "")
        pbar.update(1)
        held = None
        if audio is not None:                          # the sound, held: both stages get this same latent
            sr = int(audio["sample_rate"])
            a0, an = vd.audio_range(0, F, fps, sr)
            seg = audio["waveform"][:1, :, a0:a0 + an].float().cpu()
            if seg.shape[-1] < an:
                seg = torch.nn.functional.pad(seg, (0, an - seg.shape[-1]))
            held = ops.sound_audio(audio_vae, seg, sr)
        guider = ops.guider(model, pos, neg, fps)
        video, sound = ops.sample(model, guider, int(seed), _text(sig1), latent, held or ops.empty_audio(audio_vae, F, fps))
        del latent
        ops.free()
        pbar.update(1)
        if upscale_model is not None:
            video = ops.upsample(video, upscale_model, vae)
            if limit:
                video["noise_mask"] = noise_mask(video, sig2[0])
            video, _ = ops.sample(model, guider, int(seed) + 1000, _text(sig2), video, held or sound)
            ops.free()
            pbar.update(1)
        decoded = ops.decode(vae, video)[:n, ..., :3].float().cpu().clamp_(0, 1)
        del video
        ops.free()
        pbar.update(1)
        oh, ow = decoded.shape[1:3]
        if masked and (keep_exact or not limit):       # the guide's own pixels where the mask and the curve say so
            dec = decoded.numpy()
            for i in range(n):
                a = vd.repaint_alpha(where, when, i, n, ow, oh, step=limit)
                if np.ndim(a) or a < 1.0:
                    g = src[i] if src[i].shape[:2] == (oh, ow) else cv2.resize(
                        src[i], (ow, oh), interpolation=cv2.INTER_AREA if src[i].shape[1] > ow else cv2.INTER_CUBIC)
                    a = a[..., None] if np.ndim(a) else a
                    dec[i] = np.clip(g * (1.0 - a) + dec[i] * a, 0.0, 1.0)
        back = bool(guide_size) and (oh, ow) != (gh, gw)
        if back:
            big = np.empty((n, gh, gw, 3), np.float32)
            for i in range(n):
                big[i] = cv2.resize(decoded[i].numpy(), (gw, gh), interpolation=cv2.INTER_CUBIC if gw > ow else cv2.INTER_AREA)
            decoded = torch.from_numpy(np.clip(big, 0, 1, out=big))

        r = float(repaint)
        lines = [f"{n} frames at {fps:g} fps ({n / fps:.2f} s), guide {gw}x{gh} -> {ow}x{oh}"
                 + (f", scaled back to {gw}x{gh}" if back else "")
                 + f", {'two stages' if upscale_model is not None else 'one stage'}, {time.perf_counter() - t0:.0f} s",
                 f"repaint {r:g}: sampled through {_text(sig1)}" + (f"; second stage {_text(sig2)}" if upscale_model is not None else "")]
        if masked:
            lines.append((f"repaint mask: {len(where)} frame(s)" if where is not None else "no repaint mask (everywhere)")
                         + (f"; curve {curve}, {cycle:g} s, from {curve_from:g} to {curve_to:g}" if when is not None else "")
                         + (f": LTX may change {share * 100:.0f} % of the clip and keeps the guide's look"
                            + ("; the guide's own pixels where nothing may change" if keep_exact else "") if limit else
                            f": the repainted clip shows on {share * 100:.0f} % of the clip, your guide on the rest"))
            lines += curve_lines
            if share < 0.02:
                lines.append("almost nothing of the repaint shows: the mask is black or the curve stays at 0 (check repaint_mask, curve_from / curve_to)")
        if r >= 1.0:
            lines.append("repaint 1: the guide is ignored, this is a plain text to video. Lower it to about 0.91 to follow the guide.")
        elif r > 0.93:
            lines.append("if the video does not follow the guide enough: lower repaint towards 0.91")
        elif r >= 0.85:
            lines.append("small moving details of the guide can get lost at this level: make them bigger or brighter "
                         "in the guide, or lower repaint (more of the guide's own look stays then)")
        else:
            lines.append("at this level the guide's own look mostly stays. For a real repaint raise repaint towards 0.91.")
        report = "\n".join(lines + notes)
        log.info("[KUBA regions] motion guide: %s", lines[0])
        return io.NodeOutput(decoded, float(fps), nv._audio_out(audio, n, float(fps)), report)


NODE_CLASS_MAPPINGS = {"KUBA_MotionGuide": KUBA_MotionGuide}
NODE_DISPLAY_NAME_MAPPINGS = {"KUBA_MotionGuide": "kubakub motion guide (ltx)"}
