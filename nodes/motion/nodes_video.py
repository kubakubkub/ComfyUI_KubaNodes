"""
nodes_video.py

kubakub region video sampler (category KUBAKUB/regions): a still becomes
a video in which only regions with `animate = on` move, with LTX 2.x (22B distilled).
Each clip (animated regions sharing video prompt, time range and motion) is cropped
to the LTX grid, the crop of the still is VAE-encoded as a static video, a noise
mask lets only the regions move during their time range (first frame fixed), the
clip is sampled with the LTX 2.5 blueprint's distilled schedule (optionally a second
stage after the x2 latent upscaler) and pasted back into the untouched still frame
by frame. The LTX steps call core's own node implementations.
loop: the last latent frame is held like the first, so every clip returns to the still.
audio: a sound becomes each clip's audio latent, held as it is (noise mask 0), so the
picture is sampled together with it (for the audio-reactive LoRA).
material: a clip on the 'material' input (a kubakub shader with the regions connected) moves the
animated regions: its frames are laid into each clip's crop where the regions are (kubakub/materials.py),
and the clip is sampled from the noise level material_repaint instead of from pure noise, so LTX keeps
the movement and paints the look (the way of kubakub motion guide).
The clips run in phases (VAE encode all, enhance all, text encode all, stage 1 all, upsample + stage 2 all, decode
all), so each model is loaded once per run, not once per clip; rendered clips are cached (clip_cache.py), so an
unchanged clip is not rendered again. The models are lazy inputs: without an animated region (or while the
director holds) no model is loaded and the frames pass through.
Logic in video.py. V3 node schema, registered via ../__init__.py.
"""

from __future__ import annotations

import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import torch

from comfy_api.latest import io, ui

from ...kubakub import clip_cache as cc
from ...kubakub import materials as ma
from ...kubakub import plan as rp
from ...kubakub import save_paths as sp
from ...kubakub import video as vd
from ...kubakub import video_models as vm
from ...kubakub.io_types import PlanType

log = logging.getLogger("KUBA.regions")

CATEGORY = "kubakub/2d/motion"
SIGMAS_1 = "1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0"   # LTX 2.5 blueprint stage 1
SIGMAS_2 = "0.85, 0.7250, 0.4219, 0.0"                                                 # stage 2 after the x2 upscaler


def _sigmas(text):
    return torch.tensor([float(v) for v in text.replace(",", " ").split()], dtype=torch.float32)


def _text(sigmas) -> str:
    return ", ".join(f"{float(s):g}" for s in sigmas)


def _ltx():
    from comfy_extras import nodes_custom_sampler as cs
    from comfy_extras import nodes_lt as lt
    from comfy_extras import nodes_lt_audio as lta
    from comfy_extras import nodes_lt_upsampler as ltu
    return cs, lt, lta, ltu


# rendered clips, so that changing one clip (its prompt, seed, regions) does not render the others again:
# key = the clip's pixels, mask, prompt, seed and settings; valid while the same model objects are alive
_CLIP_CACHE = cc.ClipCache(8, max_bytes=6 << 30)          # float16 frames: ~0.4 GB per 121-frame 0.6 MP clip
_digest = cc.digest
_WRITE_THREADS = max(1, min(8, (os.cpu_count() or 4)))


class _LtxOps:
    """The model steps for LTX (core nodes). Tests put a stub with the same methods in its place (_OPS); _H3Ops
    has the same methods for MiniMax H3 (ops_for picks by the model)."""

    family = "ltx"

    def progress(self, n):
        import comfy.utils
        return comfy.utils.ProgressBar(n)

    def free(self):
        import comfy.model_management as mm
        mm.soft_empty_cache()

    def vae_encode(self, vae, pixels):
        return {"samples": vae.encode(pixels)}

    def enhance(self, enhance_clip, text, first, seed):
        from comfy_extras import nodes_textgen as tg
        from ...kubakub.runtime import between_runs
        try:
            return tg.TextGenerate.execute(enhance_clip, text, 512, {"sampling_mode": "on", "temperature": 0.6, "top_k": 64,
                                           "top_p": 0.95, "min_p": 0.05, "repetition_penalty": 1.05, "seed": int(seed)},
                                           image=first, system_prompt=tg.LTX2_I2V_SYSTEM_PROMPT).args[0]
        finally:
            between_runs()                  # one enhance per clip inside one node: see runtime.between_runs

    def encode_text(self, clip, text):
        return clip.encode_from_tokens_scheduled(clip.tokenize(text))

    def condition(self, clip, vae, text, first, last, frames, task=None):
        """The prompt's conditioning; LTX takes the still through the noise mask, so the frames are not needed."""
        return self.encode_text(clip, text)

    def guider(self, model, pos, neg, fps, prof=None):
        cs, lt, _lta, _ltu = _ltx()
        pos, neg = lt.LTXVConditioning.execute(pos, neg, fps).args
        cfg = float((prof or {}).get("cfg", 1.0))
        if cfg != 1.0:                                  # a dev model: core's dev templates use CFGGuider
            return cs.CFGGuider.execute(model, pos, neg, cfg).args[0]
        return lt.LTXVDualCFGGuider.execute(model, pos, neg, 1.0, 1.0).args[0]

    def base_sigmas(self, model, prof, latent=None):
        """The schedule a stage starts from: the profile's own sigmas, or its scheduler at its steps."""
        if prof.get("sigmas"):
            return [float(s) for s in prof["sigmas"]]
        cs, lt, _lta, _ltu = _ltx()
        if prof.get("scheduler") == "ltxv":
            sig = lt.LTXVScheduler.execute(int(prof["steps"]), 2.05, 0.95, True, 0.1, latent).args[0]
        else:
            sig = cs.BasicScheduler.execute(model, prof.get("scheduler") or "simple", int(prof["steps"]), 1.0).args[0]
        return [float(s) for s in sig]

    def empty_audio(self, audio_vae, frames, fps):
        if audio_vae is None:                           # LTX video only: no audio stream
            return None
        _cs, _lt, lta, _ltu = _ltx()
        return lta.LTXVEmptyLatentAudio.execute(frames_number=frames, frame_rate=int(round(fps)), batch_size=1,
                                                audio_vae=audio_vae).args[0]

    def sound_audio(self, audio_vae, waveform, sample_rate):
        """A sound as the audio latent, held as it is: core's audio encode + a noise mask of 0 (the wiring of
        RuneXX's custom-audio LTX workflows, https://huggingface.co/RuneXX/LTX-2.3-Workflows)."""
        import nodes
        _cs, _lt, lta, _ltu = _ltx()
        latent = lta.LTXVAudioVAEEncode.execute({"waveform": waveform, "sample_rate": int(sample_rate)}, audio_vae).args[0]
        return nodes.SetLatentNoiseMask().set_mask(latent, torch.zeros((1, 512, 512)))[0]

    def sample(self, model, guider, seed, sigmas, video, audio, sampler_name="euler_ancestral"):
        """One stage on the audio-video latent (the video alone for a video-only model) -> (video, audio latent)."""
        import comfy.samplers
        cs, lt, _lta, _ltu = _ltx()
        sampler = comfy.samplers.sampler_object(sampler_name)
        if audio is None:
            out = cs.SamplerCustomAdvanced.execute(cs.Noise_RandomNoise(int(seed)), guider, sampler, _sigmas(sigmas), video).args[0]
            return out, None
        av = lt.LTXVConcatAVLatent.execute(video, audio).args[0]
        out = cs.SamplerCustomAdvanced.execute(cs.Noise_RandomNoise(int(seed)), guider, sampler, _sigmas(sigmas), av).args[0]
        return tuple(lt.LTXVSeparateAVLatent.execute(out).args)

    def upsample(self, video, upscale_model, vae):
        _cs, _lt, _lta, ltu = _ltx()
        return ltu.LTXVLatentUpsampler.execute(video, upscale_model, vae).args[0]

    def decode(self, vae, video):
        import nodes
        decoded = nodes.VAEDecodeTiled().decode(vae, video, 512, 64, 64, 16)[0]
        return decoded.reshape(-1, *decoded.shape[-3:])


class _H3Ops(_LtxOps):
    """The same steps for MiniMax H3 (core comfy_extras/nodes_minimax_h3.py): the AV latent is a NestedTensor
    (video [1, 24, T, H/16, W/16], audio [1, 32, 2, T40]); core's sampler takes a nested noise mask, so the regions
    are held the same way as with LTX. fl2va gets the still as its first (and with a loop last) keyframe, ref2va as
    <Picture 1>. No latent upscaler exists for H3."""

    family = "h3"

    def condition(self, clip, vae, text, first, last, frames, task="fl2va"):
        import node_helpers
        from comfy_extras import nodes_minimax_h3 as h3
        if first is None:
            return self.encode_text(clip, text)
        H, W = first.shape[1:3]
        if task == "ref2va":
            if "<Picture 1>" not in text:
                text = "<Picture 1> is the first frame: keep its building, framing and light. " + text
            return h3.MiniMaxH3ReferenceToVideo.execute(clip, text, W, H, frames, "match", vae, None,
                                                        ref_images={"ref_image_1": first}).args[0]
        n = vm.h3_frames(frames)
        images, keys = [first], [{"resolved_frame_index": 0, "latent": vae.encode(first[..., :3])}]
        if last is not None:
            images.append(last)
            keys.append({"resolved_frame_index": n - 1, "latent": vae.encode(last[..., :3])})
        cond = clip.encode_from_tokens_scheduled(clip.tokenize(text, images=images))
        return node_helpers.conditioning_set_values(cond, {"minimax_keyframes": keys})

    def guider(self, model, pos, neg, fps, prof=None):
        from comfy_extras import nodes_custom_sampler as cs
        cfg = float((prof or {}).get("cfg", 1.0))
        if cfg != 1.0:
            return cs.CFGGuider.execute(model, pos, neg, cfg).args[0]
        return cs.BasicGuider.execute(model, pos).args[0]       # core's H3 templates: no negative at cfg 1

    def empty_audio(self, audio_vae, frames, fps):
        import comfy.model_management as mm
        return {"samples": torch.zeros((1, 32, 2, vm.h3_audio_frames(frames)), device=mm.intermediate_device())}

    def sound_audio(self, audio_vae, waveform, sample_rate):
        """A sound as H3's audio latent, held (noise mask 0): core's H3 reference-audio encode."""
        import comfy.audio
        sr = getattr(audio_vae, "audio_sample_rate", 32000)
        if int(sample_rate) != sr:
            waveform = comfy.audio.resample(waveform, int(sample_rate), sr)
        z = audio_vae.encode(waveform[:1].movedim(1, -1))
        return {"samples": z, "noise_mask": torch.zeros_like(z)}

    def sample(self, model, guider, seed, sigmas, video, audio, sampler_name="res_multistep"):
        import comfy.nested_tensor
        import comfy.samplers
        from comfy_extras import nodes_custom_sampler as cs
        v = video["samples"]
        t_lat = v.shape[2]
        frames = 5 if t_lat <= 2 else (t_lat - 2) // 5 * 17 + 5
        a = (audio or self.empty_audio(None, frames, vm.H3_FPS))["samples"]
        am = (audio or {}).get("noise_mask")
        want = vm.h3_audio_frames(frames)                 # the sound cut / padded to the clip's audio length
        if a.shape[-1] != want:
            a = a[..., :want] if a.shape[-1] > want else torch.nn.functional.pad(a, (0, want - a.shape[-1]))
            if am is not None:
                am = torch.zeros_like(a)
        latent = {"samples": comfy.nested_tensor.NestedTensor((v, a))}
        v_mask = video.get("noise_mask")
        if v_mask is not None or am is not None:
            latent["noise_mask"] = comfy.nested_tensor.NestedTensor((
                v_mask if v_mask is not None else torch.ones((1, 1) + tuple(v.shape[2:])),
                am if am is not None else torch.ones_like(a)))
        sampler = comfy.samplers.sampler_object(sampler_name)
        out = cs.SamplerCustomAdvanced.execute(cs.Noise_RandomNoise(int(seed)), guider, sampler, _sigmas(sigmas), latent).args[0]
        streams = out["samples"].unbind()
        return {"samples": streams[0]}, {"samples": streams[1]}

    def upsample(self, video, upscale_model, vae):
        raise ValueError("there is no latent upscaler for MiniMax H3")

    def decode(self, vae, video):
        import nodes
        decoded = nodes.VAEDecode().decode(vae, video)[0]
        return decoded.reshape(-1, *decoded.shape[-3:])


_OPS = _LtxOps()
_H3_OPS = _H3Ops()


def ops_for(model):
    """The model steps for this model's family (LTX unless it is MiniMax H3)."""
    return _H3_OPS if vm.family(model) == "h3" else _OPS


MODEL_TIP = ("The video model, any variant: LTX 2.x (dev or distilled, fp8 / nvfp4 / int8 / GGUF) or MiniMax H3 "
             "(fl2va or ref2va, pruned or full), with or without a speed LoRA on it. The node sees which it is.")
CLIP_TIP = ("The text encoder of the same model: Gemma for LTX (CLIPLoader type ltxv), Qwen3-VL for H3 "
            "(CLIPLoader type minimax).")
VAE_TIP = "The video VAE of the same model (LTX or H3)."
AUDIO_VAE_TIP = ("The audio VAE of the same model (LTX 2.x and H3 make sound with the picture; a video-only LTX "
                 "needs none). Needed for the 'audio' input; without it the clip's sound is empty.")


def schedule_inputs():
    """The two advanced widgets every video node has: how to sample the model when its file names do not say."""
    return [
        io.Combo.Input("schedule", options=list(vm.SCHEDULES), default=vm.SCHEDULES[0], optional=True, advanced=True,
                       tooltip="auto = from the model's file names (a 'distilled' / '4step' / 'flashgen' / 'taomate' "
                               "name = fast, a 'dev' LTX or an H3 without a speed LoRA = full steps); the report says "
                               "what it took. Set it when your files have other names."),
        io.Int.Input("steps", default=0, min=0, max=100, optional=True, advanced=True,
                     tooltip="0 = what the model needs (8 distilled LTX, 20 dev LTX, 3 / 4 / 8 with an H3 speed LoRA, "
                             "20 H3 without). Another number overrides it."),
    ]


def model_setup(node, model, schedule="auto", steps=0, input_name="model", default="ltx", **parts):
    """(family, profile, ops) for the connected model: its family from the model, the schedule from the file names
    upstream of it (the hidden prompt) and the widgets; a VAE / text encoder of the other family is an error."""
    fam = vm.family(model) or default
    vm.check_set(fam, **parts)
    hid = getattr(node, "hidden", None)
    files = vm.upstream_files(getattr(hid, "prompt", None), getattr(hid, "unique_id", None), input_name)
    return fam, vm.profile(fam, files, schedule or "auto", int(steps or 0)), (_H3_OPS if fam == "h3" else _OPS)


def _label_ids(plan):
    """The region ids in the plan's label map (-1 = no region)."""
    lab = plan.regions.labels[0].cpu().numpy().ravel()
    if lab.size and lab.min() >= -1 and lab.max() < (1 << 24):
        return np.nonzero(np.bincount(lab.astype(np.int64) + 1))[0] - 1
    return np.unique(lab)


def _composite(still, bg, clips, F, H, W, oh, ow, folder, prefix):
    """The output frames: the untouched still (or background frame) with each clip pasted in; with a folder every
    frame is also written at full size as PNG. Frames in parallel threads (cv2 / numpy release the GIL), each into
    its own slot, so the pixels are the same as one after the other."""
    out_frames = np.empty((F, oh, ow, 3), np.float32)

    def one(i):
        frame = still.copy() if bg is None else bg[i].copy()
        for box, cf, alpha, start in clips:
            if 0 <= i - start < len(cf):
                frame = vd.paste_frame(frame, cf[i - start].astype(np.float32), box, alpha)
        if folder:
            png = (np.clip(frame, 0, 1) * 255 + 0.5).astype(np.uint8)[..., ::-1]
            cv2.imencode(".png", png, [cv2.IMWRITE_PNG_COMPRESSION, 1])[1].tofile(
                os.path.join(folder, f"{prefix}_{i:05d}.png"))
        out_frames[i] = cv2.resize(frame, (ow, oh), interpolation=cv2.INTER_AREA) if (oh, ow) != (H, W) else frame

    if F <= 1 or _WRITE_THREADS <= 1:
        for i in range(F):
            one(i)
    else:
        with ThreadPoolExecutor(_WRITE_THREADS) as ex:
            list(ex.map(one, range(F)))                     # list(): re-raises a worker's error here
    return out_frames


def _audio_out(audio, F, fps):
    """The sound for the frames: the connected audio cut / padded with silence to F frames, silence without one."""
    if audio is None:
        return {"waveform": torch.zeros((1, 2, vd.audio_range(0, F, fps, 44100)[1])), "sample_rate": 44100}
    sr = int(audio["sample_rate"])
    n = vd.audio_range(0, F, fps, sr)[1]
    wave = audio["waveform"][:1, :, :n].float().cpu().clone()
    if wave.shape[-1] < n:
        wave = torch.nn.functional.pad(wave, (0, n - wave.shape[-1]))
    return {"waveform": wave, "sample_rate": sr}


class KUBA_RegionVideoSampler(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="KUBA_RegionVideoSampler",
            display_name="kubakub region video sampler",
            category="kubakub/2d/motion",
            search_aliases=['ltx', 'h3', 'minimax', 'animate', 'video'],
            description=(
                "Animate chosen regions of a still: only regions with animate = on move, the rest of the "
                "facade stays still. Plug in any LTX 2.x or MiniMax H3 model (dev or distilled, fl2va or ref2va, "
                "with or without a speed LoRA): the node sees which one it is and samples it the way it needs. "
                "Regions sharing video_prompt, t_start / t_end and motion form one clip; each clip runs on its "
                "crop, the rest of the crop is held by the noise mask, the first frame is the still, "
                "and the result is pasted back into the untouched still frame by frame, so the facade stays "
                "pixel exact. Plan keys: animate, video_prompt, t_start, t_end, motion. With a clip on "
                "'material' (a kubakub shader) the animated regions move as it moves, and the model repaints "
                "them (plan: material_repaint)."),
            inputs=[
                io.Model.Input("model", lazy=True, tooltip=MODEL_TIP),
                io.Clip.Input("clip", lazy=True, tooltip=CLIP_TIP),
                io.Vae.Input("vae", lazy=True, tooltip=VAE_TIP),
                io.Vae.Input("audio_vae", lazy=True, optional=True, tooltip=AUDIO_VAE_TIP),
                PlanType.Input("plan", tooltip="From kubakub region plan: animate = on, video_prompt, t_start, t_end "
                                               "and motion say which regions move and how."),
                io.Image.Input("image", optional=True,
                               tooltip="The still (matrix size), e.g. frame in frame + region sampler result. Not "
                                       "needed (and not used) when background is connected."),
                io.LatentUpscaleModel.Input("upscale_model", optional=True, lazy=True,
                                            tooltip="LTX latent spatial upscaler x2: adds a sharper second stage "
                                                    "(LTX only; H3 has none)."),
                io.Int.Input("frames", default=121, min=9, max=1025, step=8,
                             tooltip="Video length in frames; rounded up to 8k + 1. Ignored when background is "
                                     "connected (the video is as long as the background)."),
                io.Float.Input("fps", default=25.0, min=1.0, max=120.0, step=0.01,
                               tooltip="Frame rate of the video (the plan's t_start / t_end are in seconds)."),
                io.Int.Input("seed", default=42, min=0, max=0xFFFFFFFFFFFFFFFF,
                             control_after_generate=io.ControlAfterGenerate.fixed,
                             tooltip="Noise seed; each clip adds its number, so clips differ. Change it for another take."),
                io.Float.Input("stage1_mp", advanced=True, default=0.2, min=0.05, max=4.0, step=0.05,
                               tooltip="Megapixels of each clip in the first stage (x2 per side with the upscaler)."),
                io.Int.Input("context_px", advanced=True, default=64, min=0, max=2048, step=8,
                             tooltip="Still around the regions each clip sees."),
                io.Int.Input("feather_px", advanced=True, default=6, min=0, max=256,
                             tooltip="Soft edge when pasting back; beyond it the still is untouched."),
                io.String.Input("negative", advanced=True, multiline=True, optional=True,
                                default="camera movement, zoom, pan, shaky, morphing facade, blurry, low quality, text, watermark",
                                tooltip="What the clips should avoid (one negative prompt for all clips)."),
                io.String.Input("only", display_name=rp.WHICH, advanced=True, default="", optional=True,
                                placeholder=rp.WHICH_HINT,
                                tooltip="Only these animated regions. Empty = all. " + rp.WHICH_TIP),
                io.Float.Input("output_scale", default=0.5, min=0.05, max=1.0, step=0.05,
                               tooltip="Size of the IMAGE output (a 4K batch of 121 frames needs 12 GB RAM); use "
                                       "save_folder for the full-size frames. Ignored when background is connected "
                                       "(the frames keep the background's size)."),
                io.String.Input("save_folder", default="", optional=True,
                                tooltip="Write every frame at matrix size as PNG (an image sequence for After "
                                        "Effects)."),
                io.String.Input("filename_prefix", advanced=True, default="region_video", optional=True,
                                tooltip="Start of the PNG names in save_folder (<prefix>_00000.png ...)."),
                io.Float.Input("stage2_max_mp", advanced=True, default=0.6, min=0.1, max=8.0, step=0.05, optional=True,
                               tooltip="Cap for the second stage (4x the stage-1 pixels): stage 1 shrinks to fit. "
                                       "0.6 MP x 121 frames is safe on a 12 GB card; 1.8 MP crashed it."),
                io.Boolean.Input("enhance_prompt", default=False, optional=True,
                                 tooltip="Expand each clip's video prompt with a writing text encoder (enhance_clip) and "
                                         "core's LTX prompt instructions, looking at the clip's first frame."),
                io.Clip.Input("enhance_clip", optional=True, lazy=True,
                              tooltip="A text encoder that can write, e.g. qwen3.5_9b ... pe_i2i (CLIPLoader type qwen_image). "
                                      "The Gemma 4 LTX int8 encoder cannot generate text."),
                io.Image.Input("background", optional=True,
                               tooltip="A video to animate into instead of the still (e.g. the director's frames): "
                                       "clips start at their region's t_start and continue its frames; everything "
                                       "else of the background plays as it is."),
                io.Boolean.Input("loop", default=False, optional=True,
                                 tooltip="Each clip comes back to the still and its last frame is its first frame, so "
                                         "the video can repeat. The motion has to return within the clip, so it is "
                                         "calmer near the end. On a still only, not into a background video."),
                io.Audio.Input("audio", optional=True,
                               tooltip="A sound the clips are sampled with (held as it is). With an audio-reactive LoRA on "
                                       "the model and its trigger words in video_prompt the regions move with the sound. "
                                       "The frames carry no sound: add the same audio in Create Video."),
                io.Image.Input("material", optional=True,
                               tooltip="A clip the animated regions follow, e.g. the images of a kubakub shader with "
                                       "the same regions connected: each region moves as the clip moves there, and the "
                                       "model repaints it in its own look (plan: material_repaint, 0.8; 0.91 is "
                                       "freer, small regions can lose the timing). Frame by frame at this node's fps; a "
                                       "shorter clip holds its last frame. Where the clip is the same as the picture "
                                       "the model moves the region freely."),
                *schedule_inputs(),
            ],
            hidden=[io.Hidden.prompt, io.Hidden.unique_id],
            outputs=[
                io.Image.Output("frames", tooltip="The video at output_scale."),
                io.String.Output("report", tooltip="Clips, frames, stages and time, one line per clip (size, time range, "
                                                   "cache, prompt)."),
                io.Float.Output("fps", tooltip="The frame rate (frames per second), passed through for Create Video."),
                io.Audio.Output("audio", tooltip="The connected audio cut or padded with silence to the length of "
                                                 "the frames; silence of that length when no audio is connected."),
                io.Mask.Output("mask", tooltip="The regions that were animated, as one mask at the size of the "
                                               "frames (white = moves at some time in the video)."),
            ],
        )

    @classmethod
    def check_lazy_status(cls, plan=None, fps=25.0, only="", enhance_prompt=False, background=None, **kw):
        """The video models only when a clip will be rendered (lazy inputs): a held director (blocked plan / image /
        background: this is not even called) or a plan without an animated region loads no model. An input that
        is not connected is not in kw; a connected one that has not run yet is None."""
        if plan is None:
            return []
        groups = vd.clip_groups(plan.plan["regions"], only)
        n_bg = int(background.shape[0]) if background is not None else None
        if not groups or not vd.groups_with_work(groups, _label_ids(plan), n_bg, float(fps)):
            return []
        want = ["model", "clip", "vae", "audio_vae", "upscale_model"] + (["enhance_clip"] if enhance_prompt else [])
        return [x for x in want if x in kw and kw[x] is None]

    @classmethod
    def execute(cls, model, clip, vae, plan, frames, fps, seed, stage1_mp, context_px,
                feather_px, output_scale, audio_vae=None, image=None, upscale_model=None, negative="", only="",
                save_folder="", filename_prefix="region_video", stage2_max_mp=0.6, background=None,
                enhance_prompt=False, enhance_clip=None, loop=False, audio=None, material=None, schedule="auto",
                steps=0) -> io.NodeOutput:
        t0 = time.perf_counter()
        if image is None and background is None:
            raise ValueError("kubakub region video sampler: connect 'image' (the still to animate) or 'background' "
                             "(a video to animate into).")
        fam, prof, ops = model_setup(cls, model, schedule, steps, clip=clip, vae=vae, audio_vae=audio_vae)
        labels = plan.regions.labels[0].cpu().numpy()
        bg = None
        if background is not None:                     # animate into a video: regions follow its size
            bg = background[..., :3].cpu().float().numpy()
            H, W = bg.shape[1:3]
            if labels.shape != (H, W):
                labels = cv2.resize(labels.astype(np.int32), (W, H), interpolation=cv2.INTER_NEAREST)
            still = bg[0]
        else:
            H, W = labels.shape
            still = image[0, ..., :3].cpu().float().numpy()
            if still.shape[:2] != (H, W):
                still = cv2.resize(still, (W, H), interpolation=cv2.INTER_AREA)
        mat = None
        if material is not None and len(material):   # the clip the animated regions follow, at the frames' size
            mat = material[..., :3].cpu().float().numpy()   # another size stays as it is: only each clip's crop is scaled
        groups = vd.clip_groups(plan.plan["regions"], only)
        model_rows = []
        if upscale_model is not None and fam == "h3":
            model_rows.append("upscale_model: there is no latent upscaler for MiniMax H3, so one stage (raise stage1_mp "
                              "for a sharper clip)")
            upscale_model = None
        if upscale_model is not None and stage1_mp * 4 > stage2_max_mp:
            stage1_mp = stage2_max_mp / 4              # the second stage is 2x per side: keep it inside the cap
        F = vd.ltx_frames(frames) if bg is None else len(bg)
        loop = bool(loop) and bg is None
        use_enh = enhance_clip if enhance_prompt else None
        objs = (model, clip, vae, audio_vae, upscale_model, use_enh)
        rows, jobs = [], []
        if not groups and (only or "").strip() and vd.clip_groups(plan.plan["regions"]):
            names = [str(e.get("name", "")) for e in plan.plan["regions"] if e.get("animate")]
            rows.append(f"{rp.WHICH} = '{only.strip()}' matches no animated region: frames passed through, no video "
                        f"model loaded (animated regions: {', '.join(names[:8])}{', ...' if len(names) > 8 else ''})")
        elif not groups:
            rows.append("no animated regions: frames passed through (set 'animate = on' and a video_prompt in the "
                        "region plan to animate), no video model loaded")
        # 1. per clip: the crop, the cache lookup and the VAE encode of the clips to render (the video VAE only)
        for gi, g in enumerate(groups):
            tg = time.perf_counter()
            mask = np.isin(labels, g["ids"])
            if not mask.any():
                rows.append(f"clip {gi + 1}: its regions are not in the frame, skipped")
                continue
            (x0, y0, x1, y1), (w1, h1) = vd.crop_plan(mask, stage1_mp * 1e6, context_px)
            if bg is None:                             # a still: the crop repeated, the time range in the mask
                i0, n, ts, te = 0, F, g["t_start"], g["t_end"]
                Fc, mfps = vm.model_frames(fam, (n - 1) / fps, fps)
                crop = cv2.resize(still[y0:y1, x0:x1], (w1, h1), interpolation=cv2.INTER_AREA)
                pixels = torch.from_numpy(np.ascontiguousarray(crop))[None].repeat(Fc, 1, 1, 1)
            else:                                      # a video: the clip covers the time range, its first frame = the background
                rng = vd.video_range(g["t_start"], g["t_end"], fps, len(bg))
                if rng is None:
                    rows.append(f"clip {gi + 1}: time range outside the video, skipped")
                    continue
                i0, i1 = rng
                n, ts, te = i1 - i0 + 1, 0.0, -1.0
                Fc, mfps = vm.model_frames(fam, (n - 1) / fps, fps)
                idx = [i0 + k for k in vm.time_index(Fc, mfps, n, fps)]     # the model's frames on the video's time
                pixels = torch.from_numpy(np.stack([cv2.resize(bg[k, y0:y1, x0:x1], (w1, h1), interpolation=cv2.INTER_AREA)
                                                    for k in idx]))
            region_crop = mask[y0:y1, x0:x1]
            sound = None
            if audio is not None:                      # the clip's stretch of the sound, silence where it has ended
                sr = int(audio["sample_rate"])
                a0, an = int(round(i0 / fps * sr)), int(round(Fc / mfps * sr))
                seg = audio["waveform"][:1, :, a0:a0 + an].float().cpu()
                if seg.shape[-1] < an:
                    seg = torch.nn.functional.pad(seg, (0, an - seg.shape[-1]))
                sound = (seg, sr)
                if audio_vae is None:
                    rows.append(f"clip {gi + 1}: the sound needs audio_vae (the model's audio VAE): sampled without it")
                    sound = None
            repaint, said, guided = None, [], False
            if mat is not None:                        # the material clip moves the regions: it is laid into the crop
                want_m = [i0 + k for k in vm.time_index(Fc, mfps, 1 << 30, fps)]   # the material's frames on the clip's time
                idx_m = [min(k, len(mat) - 1) for k in want_m]
                held = {}                              # a still material is cut and scaled once
                for k in idx_m:
                    if k not in held:
                        mh, mw = mat.shape[1:3]        # the crop's box in the material's own pixels
                        mx0, my0 = min(x0 * mw // W, mw - 1), min(y0 * mh // H, mh - 1)
                        cut = mat[k, my0:max(my0 + 1, -(-y1 * mh // H)), mx0:max(mx0 + 1, -(-x1 * mw // W))]
                        held[k] = cv2.resize(cut, (w1, h1),
                                             interpolation=cv2.INTER_AREA if cut.shape[1] >= w1 else cv2.INTER_LINEAR)
                small = cv2.resize(region_crop.astype(np.uint8), (w1, h1), interpolation=cv2.INTER_NEAREST) > 0
                guide = ma.guide(pixels.numpy(), [held[k] for k in idx_m], small,
                                 vd.frames_in_range(pixels.shape[0], mfps, ts, te))
                last_m = want_m[-1] + 1
                if 1 < len(mat) < last_m:
                    said.append(f"it has {len(mat)} frames, the clip runs to frame {last_m}: its last frame is held")
                elif len(mat) == 1:
                    said.append("it is one picture, so the regions take its look but barely move; for movement give it a "
                                "clip (a kubakub shader with seconds > 0)")
                if guide is None:
                    said = [f"clip {gi + 1} material: it does not differ from the picture on these regions, so the model moves them freely"]
                else:
                    guided, pixels, repaint = True, torch.from_numpy(guide), float(g["repaint"])
                    said = [f"clip {gi + 1} material: {m}" for m in said] + [
                        f"clip {gi + 1} material: the regions move as the material clip moves, repainted from noise "
                        f"level {repaint:g} (plan: material_repaint)"]
            key = _digest(fam, vm.describe(prof), pixels, region_crop, g["prompt"], negative or "", bool(enhance_prompt),
                          int(seed) + gi, float(fps), n, float(ts), float(te), float(g["motion"]), repaint,
                          upscale_model is not None, loop, *(sound if sound is not None else ()))
            job = {"gi": gi, "g": g, "box": (x0, y0, x1, y1), "mask": mask, "region_crop": region_crop, "i0": i0,
                   "n": n, "ts": ts, "te": te, "Fc": Fc, "mfps": mfps, "key": key, "hit": _CLIP_CACHE.get(key, objs),
                   "text": g["prompt"], "enhanced": None, "rows": list(said), "sound": sound, "repaint": repaint,
                   "guided": guided}
            rows.append(job["rows"])
            if enhance_prompt and enhance_clip is None:
                job["rows"].append(f"clip {gi + 1}: enhance_prompt needs enhance_clip (a text encoder that can write), plan prompt used")
            if job["hit"] is None:
                job["first"] = pixels[0:1].clone() if enhance_prompt and enhance_clip is not None else None
                if fam == "h3" and (not guided or prof["task"] == "ref2va"):
                    job["still"] = pixels[0:1].clone()     # H3's first (and loop: last) keyframe / picture reference
                job["latent"] = ops.vae_encode(vae, pixels)
            del pixels
            job["t"] = time.perf_counter() - tg
            jobs.append(job)
        todo = [j for j in jobs if j["hit"] is None]
        pbar = ops.progress(len(groups)) if todo else None
        # 2. the prompt enhancer, once for all clips
        for j in todo:
            if j.get("first") is None:
                continue
            tg = time.perf_counter()
            from ...kubakub import keyframes as kf              # core's LTX instructions on a writing encoder
            try:
                better = (ops.enhance(enhance_clip, j["text"], j.pop("first"), int(seed) + j["gi"]) or "").strip()
                if kf.plausible_prompt(better, ""):
                    j["enhanced"] = j["text"] = better
            except Exception as e:  # noqa: BLE001
                log.warning("[KUBA regions] prompt enhance failed, using the plan prompt: %s", e)
            j["t"] += time.perf_counter() - tg
        # 3. the text encoder (LTX: each text once per run; H3: per clip, with the clip's still as its keyframe)
        conds = {}
        use_neg = fam != "h3" or float(prof["cfg"]) != 1.0
        for j in todo:
            tg = time.perf_counter()
            still_px = j.pop("still", None)
            ck = j["text"] if still_px is None else ("clip", j["gi"])
            if ck not in conds:
                conds[ck] = ops.condition(clip, vae, j["text"], still_px, still_px if loop else None, j["Fc"],
                                          prof["task"])
            j["ck"] = ck
            if use_neg and (negative or "") not in conds:
                conds[negative or ""] = ops.encode_text(clip, negative or "")
            j["t"] += time.perf_counter() - tg
        # 4. stage 1 for every clip (the DiT loaded once)
        for j in todo:
            tg = time.perf_counter()
            latent, Fc, mfps = j.pop("latent"), j["Fc"], j["mfps"]
            lat_t, lat_h, lat_w = latent["samples"].shape[2:]
            times = np.linspace(0.0, (Fc - 1) / mfps, lat_t) if fam == "h3" else None
            j["times"] = times
            latent["noise_mask"] = torch.from_numpy(vd.latent_mask(j["region_crop"], lat_w, lat_h, Fc, mfps, j["ts"], j["te"],
                                                                   j["g"]["motion"], loop=loop, hold_first=not j["guided"],
                                                                   times=times))
            if j["sound"] is not None:                 # the sound, held: both stages get this same latent
                clip_audio = j["sound_latent"] = ops.sound_audio(audio_vae, *j.pop("sound"))
            else:
                clip_audio = ops.empty_audio(audio_vae, Fc, mfps)
            base = ops.base_sigmas(model, prof, latent)
            sig1 = base if j["repaint"] is None else vd.guide_sigmas(j["repaint"], base)
            j["sig1"] = _text(sig1)
            if prof.get("sigmas_2"):
                s2 = prof["sigmas_2"]
                j["sig2"] = _text(s2 if j["repaint"] is None else vd.guide_sigmas(min(j["repaint"], s2[0]), s2))
            j["guider"] = ops.guider(model, conds[j["ck"]], conds.get(negative or ""), mfps, prof)
            j["video"], j["audio"] = ops.sample(model, j["guider"], int(seed) + j["gi"], j["sig1"], latent, clip_audio,
                                                prof["sampler"])
            del latent
            ops.free()                                 # stage 1 activations out before the next clip / stage 2
            j["t"] += time.perf_counter() - tg
        # 5. with the upscaler: upsample every clip, then stage 2 for every clip
        if upscale_model is not None:
            for j in todo:
                tg = time.perf_counter()
                video = ops.upsample(j["video"], upscale_model, vae)
                h2, w2 = video["samples"].shape[3:]
                video["noise_mask"] = torch.from_numpy(vd.latent_mask(j["region_crop"], w2, h2, j["Fc"], j["mfps"], j["ts"],
                                                                      j["te"], j["g"]["motion"], loop=loop,
                                                                      hold_first=not j["guided"], times=j["times"]))
                j["video"] = video
                j["t"] += time.perf_counter() - tg
            for j in todo:
                tg = time.perf_counter()
                stage1_audio = j.pop("audio")
                j["video"], _ = ops.sample(model, j["guider"], int(seed) + 1000 + j["gi"], j["sig2"], j["video"],
                                           j.pop("sound_latent", None) or stage1_audio, prof["sampler"])
                ops.free()
                j["t"] += time.perf_counter() - tg
        conds.clear()
        # 6. the VAE decode of every clip, on the node's frames (H3 renders at its own 24 fps: mapped by time)
        for j in todo:
            tg = time.perf_counter()
            j.pop("guider", None)
            j.pop("audio", None)
            j.pop("sound_latent", None)
            decoded = ops.decode(vae, j.pop("video"))
            decoded = decoded[vm.time_index(j["n"], fps, decoded.shape[0], j["mfps"]), ..., :3].cpu().to(torch.float16).numpy()
            if loop:
                decoded = vd.close_loop(decoded)       # the last frame = the first frame, exactly
            ops.free()
            j["hit"] = (decoded, j["enhanced"])
            _CLIP_CACHE.put(j["key"], objs, j["hit"])
            j["t"] += time.perf_counter() - tg
            j["fresh"] = True
            pbar.update(1)
        clips = []
        moving = np.zeros((H, W), bool)                # the regions of every clip: the mask output
        for j in jobs:
            decoded, enhanced = j.pop("hit")
            g, (x0, y0, x1, y1) = j["g"], j["box"]
            if enhanced:
                j["rows"].append(f"clip {j['gi'] + 1} prompt (enhanced): {enhanced[:300]}")
            moving |= j["mask"]
            clips.append(((x0, y0, x1, y1), decoded, vd.feather_mask(j.pop("mask"), feather_px), j["i0"]))
            t_end = "end" if g["t_end"] < 0 else f"{g['t_end']:g}"
            rows_t = f"{j['t']:.0f} s" if j.get("fresh") else "unchanged, from the cache"
            j["rows"].append(f"clip {j['gi'] + 1}: {len(g['ids'])} region(s), {decoded.shape[2]}x{decoded.shape[1]} "
                             f"({x1 - x0}x{y1 - y0} px of the still), t {g['t_start']:g}-{t_end} s, "
                             f"motion {g['motion']:g}, {rows_t}: {g['prompt'][:60]}")
        rows = [r for x in rows for r in (x if isinstance(x, list) else [x])]
        if jobs:
            model_rows.insert(0, vm.describe(prof) + ("; 'negative' is not used at cfg 1" if not use_neg else ""))

        # composite: the untouched still, each clip pasted in (the first frame is the still itself)
        if bg is not None:
            output_scale = 1.0                         # the background already has the size it should have
        oh, ow = max(1, int(round(H * output_scale))), max(1, int(round(W * output_scale)))
        if output_scale < 1:                           # scaled output: even sizes for H.264 / yuv420
            oh, ow = max(2, 2 * round(oh / 2)), max(2, 2 * round(ow / 2))
        import folder_paths
        folder = sp.save_folder(save_folder, folder_paths.get_output_directory())
        if folder:
            os.makedirs(folder, exist_ok=True)
        out_frames = _composite(still, bg, clips, F, H, W, oh, ow, folder, sp.file_stem(filename_prefix, "region_video"))
        report = "\n".join([f"{len(clips)} clip(s), {F} frames at {fps:g} fps ({F / fps:.2f} s), "
                            f"{'two stages' if upscale_model is not None else 'one stage'} (stage 1 {stage1_mp:.2f} MP), "
                            f"{'into the background video, ' if bg is not None else ''}"
                            f"{'loop, ' if loop else ''}{'with the sound, ' if audio is not None else ''}"
                            f"{time.perf_counter() - t0:.0f} s", *model_rows, *rows]
                           + ([f"frames written to {folder}"] if folder else []))
        log.info("[KUBA regions] video: %s", report.split("\n")[0])
        moving = moving.astype(np.float32)
        if (oh, ow) != (H, W):
            moving = cv2.resize(moving, (ow, oh), interpolation=cv2.INTER_AREA)
        return io.NodeOutput(torch.from_numpy(out_frames), report, float(fps), _audio_out(audio, F, float(fps)),
                             torch.from_numpy(moving)[None])


NODE_CLASS_MAPPINGS = {"KUBA_RegionVideoSampler": KUBA_RegionVideoSampler}
NODE_DISPLAY_NAME_MAPPINGS = {"KUBA_RegionVideoSampler": "kubakub region video sampler"}
