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
from ...kubakub import save_paths as sp
from ...kubakub import video as vd
from ...kubakub.io_types import PlanType

log = logging.getLogger("KUBA.regions")

CATEGORY = "kubakub/2d/motion"
SIGMAS_1 = "1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0"   # LTX 2.5 blueprint stage 1
SIGMAS_2 = "0.85, 0.7250, 0.4219, 0.0"                                                 # stage 2 after the x2 upscaler


def _sigmas(text):
    return torch.tensor([float(v) for v in text.replace(",", " ").split()], dtype=torch.float32)


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
    """The model steps (core nodes). Tests put a stub with the same methods in its place (_OPS)."""

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

    def guider(self, model, pos, neg, fps):
        _cs, lt, _lta, _ltu = _ltx()
        pos, neg = lt.LTXVConditioning.execute(pos, neg, fps).args
        return lt.LTXVDualCFGGuider.execute(model, pos, neg, 1.0, 1.0).args[0]

    def empty_audio(self, audio_vae, frames, fps):
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

    def sample(self, model, guider, seed, sigmas, video, audio):
        """One LTX stage on the audio-video latent -> (video latent, audio latent)."""
        import comfy.samplers
        cs, lt, _lta, _ltu = _ltx()
        av = lt.LTXVConcatAVLatent.execute(video, audio).args[0]
        sampler = comfy.samplers.sampler_object("euler_ancestral")
        out = cs.SamplerCustomAdvanced.execute(cs.Noise_RandomNoise(int(seed)), guider, sampler, _sigmas(sigmas), av).args[0]
        return tuple(lt.LTXVSeparateAVLatent.execute(out).args)

    def upsample(self, video, upscale_model, vae):
        _cs, _lt, _lta, ltu = _ltx()
        return ltu.LTXVLatentUpsampler.execute(video, upscale_model, vae).args[0]

    def decode(self, vae, video):
        import nodes
        decoded = nodes.VAEDecodeTiled().decode(vae, video, 512, 64, 64, 16)[0]
        return decoded.reshape(-1, *decoded.shape[-3:])


_OPS = _LtxOps()


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


class KUBA_RegionVideoSampler(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="KUBA_RegionVideoSampler",
            display_name="kubakub region video sampler (ltx)",
            category="kubakub/2d/motion",
            search_aliases=['ltx', 'animate', 'video'],
            description=(
                "Animate chosen regions of a still: only regions with animate = on move, the rest of the "
                "facade stays still (LTX 2.x distilled). "
                "Regions sharing video_prompt, t_start / t_end and motion form one clip; each clip runs on its "
                "crop (LTX grid), the rest of the crop is held by the noise mask, the first frame is the still, "
                "and the result is pasted back into the untouched still frame by frame, so the facade stays "
                "pixel exact. Plan keys: animate, video_prompt, t_start, t_end, motion."),
            inputs=[
                io.Model.Input("model", lazy=True, tooltip="LTX 2.5 22B distilled (UNETLoader)."),
                io.Clip.Input("clip", lazy=True, tooltip="Gemma text encoder for LTX (CLIPLoader, type ltxv)."),
                io.Vae.Input("vae", lazy=True, tooltip="LTX video VAE."),
                io.Vae.Input("audio_vae", lazy=True, tooltip="LTX audio VAE (the model is audio-video; without the audio input the audio stays silent)."),
                PlanType.Input("plan", tooltip="From kubakub region plan: animate = on, video_prompt, t_start, t_end "
                                               "and motion say which regions move and how."),
                io.Image.Input("image", tooltip="The still (matrix size), e.g. frame in frame + region sampler result."),
                io.LatentUpscaleModel.Input("upscale_model", optional=True, lazy=True,
                                            tooltip="LTX latent spatial upscaler x2: adds a sharper second stage."),
                io.Int.Input("frames", default=121, min=9, max=1025, step=8,
                             tooltip="Video length; rounded up to 8k + 1."),
                io.Float.Input("fps", default=25.0, min=1.0, max=120.0, step=1.0,
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
                io.String.Input("only", advanced=True, default="", optional=True, placeholder="W_F1_*",
                                tooltip="Only these animated regions (name wildcards); empty = all."),
                io.Float.Input("output_scale", default=0.5, min=0.05, max=1.0, step=0.05,
                               tooltip="Size of the IMAGE output (a 4K batch of 121 frames needs 12 GB RAM); use "
                                       "save_folder for the full-size frames."),
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
            ],
            outputs=[
                io.Image.Output("frames", tooltip="The video at output_scale."),
                io.String.Output("report", tooltip="Clips, frames, stages and time, one line per clip (size, time range, "
                                                   "cache, prompt)."),
            ],
        )

    @classmethod
    def check_lazy_status(cls, plan=None, fps=25.0, only="", enhance_prompt=False, background=None, **kw):
        """The LTX models only when a clip will be rendered (lazy inputs): a held director (blocked plan / image /
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
    def execute(cls, model, clip, vae, audio_vae, plan, image, frames, fps, seed, stage1_mp, context_px,
                feather_px, output_scale, upscale_model=None, negative="", only="", save_folder="",
                filename_prefix="region_video", stage2_max_mp=0.6, background=None, enhance_prompt=False,
                enhance_clip=None, loop=False, audio=None) -> io.NodeOutput:
        ops = _OPS
        t0 = time.perf_counter()
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
        groups = vd.clip_groups(plan.plan["regions"], only)
        if upscale_model is not None and stage1_mp * 4 > stage2_max_mp:
            stage1_mp = stage2_max_mp / 4              # the second stage is 2x per side: keep it inside the cap
        F = vd.ltx_frames(frames) if bg is None else len(bg)
        loop = bool(loop) and bg is None
        use_enh = enhance_clip if enhance_prompt else None
        objs = (model, clip, vae, audio_vae, upscale_model, use_enh)
        rows, jobs = [], []
        if not groups:
            rows.append("no animated regions: frames passed through (set 'animate = on' and a video_prompt in the "
                        "region plan to animate), no LTX model loaded")
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
                crop = cv2.resize(still[y0:y1, x0:x1], (w1, h1), interpolation=cv2.INTER_AREA)
                pixels = torch.from_numpy(np.ascontiguousarray(crop))[None].repeat(F, 1, 1, 1)
            else:                                      # a video: the clip covers the time range, its first frame = the background
                rng = vd.video_range(g["t_start"], g["t_end"], fps, len(bg))
                if rng is None:
                    rows.append(f"clip {gi + 1}: time range outside the video, skipped")
                    continue
                i0, i1 = rng
                n, ts, te = i1 - i0 + 1, 0.0, -1.0
                Fc = vd.ltx_frames(n)
                idx = [min(i0 + k, i1) for k in range(Fc)]
                pixels = torch.from_numpy(np.stack([cv2.resize(bg[k, y0:y1, x0:x1], (w1, h1), interpolation=cv2.INTER_AREA)
                                                    for k in idx]))
            region_crop = mask[y0:y1, x0:x1]
            sound = None
            if audio is not None:                      # the clip's stretch of the sound, silence where it has ended
                sr = int(audio["sample_rate"])
                a0, an = vd.audio_range(i0, pixels.shape[0], fps, sr)
                seg = audio["waveform"][:1, :, a0:a0 + an].float().cpu()
                if seg.shape[-1] < an:
                    seg = torch.nn.functional.pad(seg, (0, an - seg.shape[-1]))
                sound = (seg, sr)
            key = _digest("ltx", pixels, region_crop, g["prompt"], negative or "", bool(enhance_prompt), int(seed) + gi,
                          float(fps), n, float(ts), float(te), float(g["motion"]), SIGMAS_1,
                          SIGMAS_2 if upscale_model is not None else None, loop,
                          *(sound if sound is not None else ()))
            job = {"gi": gi, "g": g, "box": (x0, y0, x1, y1), "mask": mask, "region_crop": region_crop, "i0": i0,
                   "n": n, "ts": ts, "te": te, "Fc": pixels.shape[0], "key": key, "hit": _CLIP_CACHE.get(key, objs),
                   "text": g["prompt"], "enhanced": None, "rows": [], "sound": sound}
            rows.append(job["rows"])
            if enhance_prompt and enhance_clip is None:
                job["rows"].append(f"clip {gi + 1}: enhance_prompt needs enhance_clip (a text encoder that can write), plan prompt used")
            if job["hit"] is None:
                job["first"] = pixels[0:1].clone() if enhance_prompt and enhance_clip is not None else None
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
                log.warning("[KUBA regions] LTX prompt enhance failed, using the plan prompt: %s", e)
            j["t"] += time.perf_counter() - tg
        # 3. the LTX text encoder (each text encoded once per run)
        conds = {}
        for j in todo:
            tg = time.perf_counter()
            for text in (j["text"], negative or ""):
                if text not in conds:
                    conds[text] = ops.encode_text(clip, text)
            j["t"] += time.perf_counter() - tg
        # 4. stage 1 for every clip (the DiT loaded once)
        for j in todo:
            tg = time.perf_counter()
            latent, Fc = j.pop("latent"), j["Fc"]
            lat_h, lat_w = latent["samples"].shape[3:]
            latent["noise_mask"] = torch.from_numpy(vd.latent_mask(j["region_crop"], lat_w, lat_h, Fc, fps, j["ts"], j["te"],
                                                                   j["g"]["motion"], loop=loop))
            if j["sound"] is not None:                 # the sound, held: both stages get this same latent
                clip_audio = j["sound_latent"] = ops.sound_audio(audio_vae, *j.pop("sound"))
            else:
                clip_audio = ops.empty_audio(audio_vae, Fc, fps)
            j["guider"] = ops.guider(model, conds[j["text"]], conds[negative or ""], fps)
            j["video"], j["audio"] = ops.sample(model, j["guider"], int(seed) + j["gi"], SIGMAS_1, latent, clip_audio)
            del latent
            ops.free()                                 # stage 1 activations out before the next clip / stage 2
            j["t"] += time.perf_counter() - tg
        # 5. with the upscaler: upsample every clip, then stage 2 for every clip
        if upscale_model is not None:
            for j in todo:
                tg = time.perf_counter()
                video = ops.upsample(j["video"], upscale_model, vae)
                h2, w2 = video["samples"].shape[3:]
                video["noise_mask"] = torch.from_numpy(vd.latent_mask(j["region_crop"], w2, h2, j["Fc"], fps, j["ts"], j["te"],
                                                                      j["g"]["motion"], loop=loop))
                j["video"] = video
                j["t"] += time.perf_counter() - tg
            for j in todo:
                tg = time.perf_counter()
                stage1_audio = j.pop("audio")
                j["video"], _ = ops.sample(model, j["guider"], int(seed) + 1000 + j["gi"], SIGMAS_2, j["video"],
                                           j.pop("sound_latent", None) or stage1_audio)
                ops.free()
                j["t"] += time.perf_counter() - tg
        conds.clear()
        # 6. the VAE decode of every clip
        for j in todo:
            tg = time.perf_counter()
            j.pop("guider", None)
            j.pop("audio", None)
            j.pop("sound_latent", None)
            decoded = ops.decode(vae, j.pop("video"))
            decoded = decoded[:j["n"], ..., :3].cpu().to(torch.float16).numpy()
            if loop:
                decoded = vd.close_loop(decoded)       # the last frame = the first frame, exactly
            ops.free()
            j["hit"] = (decoded, j["enhanced"])
            _CLIP_CACHE.put(j["key"], objs, j["hit"])
            j["t"] += time.perf_counter() - tg
            j["fresh"] = True
            pbar.update(1)
        clips = []
        for j in jobs:
            decoded, enhanced = j.pop("hit")
            g, (x0, y0, x1, y1) = j["g"], j["box"]
            if enhanced:
                j["rows"].append(f"clip {j['gi'] + 1} prompt (enhanced): {enhanced[:300]}")
            clips.append(((x0, y0, x1, y1), decoded, vd.feather_mask(j.pop("mask"), feather_px), j["i0"]))
            t_end = "end" if g["t_end"] < 0 else f"{g['t_end']:g}"
            rows_t = f"{j['t']:.0f} s" if j.get("fresh") else "unchanged, from the cache"
            j["rows"].append(f"clip {j['gi'] + 1}: {len(g['ids'])} region(s), {decoded.shape[2]}x{decoded.shape[1]} "
                             f"({x1 - x0}x{y1 - y0} px of the still), t {g['t_start']:g}-{t_end} s, "
                             f"motion {g['motion']:g}, {rows_t}: {g['prompt'][:60]}")
        rows = [r for x in rows for r in (x if isinstance(x, list) else [x])]

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
                            f"{time.perf_counter() - t0:.0f} s", *rows]
                           + ([f"frames written to {folder}"] if folder else []))
        log.info("[KUBA regions] video: %s", report.split("\n")[0])
        return io.NodeOutput(torch.from_numpy(out_frames), report)


NODE_CLASS_MAPPINGS = {"KUBA_RegionVideoSampler": KUBA_RegionVideoSampler}
NODE_DISPLAY_NAME_MAPPINGS = {"KUBA_RegionVideoSampler": "kubakub region video sampler (ltx)"}
