"""
nodes_keyframes.py

kubakub keyframe clips (category kubakub/2d/motion): the director's timeline clips rendered with MiniMax H3 or
LTX 2.x (local, core nodes; any variant, see kubakub/video_models.py) and put into the director's sequence.
- keyframes clip (fl2va model): the sequence frames at the clip's start and end are H3's first and last frame;
  H3 makes the in-between (with sound). Anchored at both ends, so it joins the sequence without a seam.
- reference clip (ref2va model): no anchors; references instead - director layers (image references), the
  timeline music of that stretch (audio reference) and/or the sequence stretch itself (reference video with its
  sound). Joined with a short crossfade.
The H3 chain follows core's templates video_minimax_h3_i2v / _r2v, the LTX chain video_ltx2_5_flf2v (LTXVAddGuide at
the first and the last frame); the schedule comes from the model's file names (a speed LoRA's own sigmas). The sound
is mixed into the timeline sound. The clips run in phases (enhance all, encode all, sample all per model, decode
all), so each model is loaded once per run; rendered clips and enhanced prompts are cached (clip_cache.py). The
models are lazy inputs: without a clip to render (or while the director holds) no H3 model is loaded.
Logic in keyframes.py. V3 node schema, registered via ../__init__.py.
"""

from __future__ import annotations

import json
import logging
import time

import cv2
import numpy as np
import torch

from comfy_api.latest import io

from ...kubakub import clip_cache as cc
from ...kubakub import keyframes as kf
from ...kubakub import video_models as vm
from . import nodes_video as nv

log = logging.getLogger("KUBA.regions")

CATEGORY = "kubakub/2d/motion"


# rendered clips, so that changing one clip of a timeline does not render the others again (and an unchanged
# clip skips the prompt enhance too): key = everything that shapes the clip -> (video uint8, sound, prompt).
# Enhanced prompts have their own cache (draft, first frame, seed, enhancer): changing steps, size or the LoRA
# renders again but does not run the enhancer again.
_CLIP_CACHE = cc.ClipCache(6)                              # ~150 MB per 56-frame clip at 1120x768
_PROMPT_CACHE = cc.ClipCache(64)
_digest = cc.digest
_LAZY = ("model", "clip", "vae", "audio_vae", "model_ref", "enhance_clip")


class _H3Ops:
    """The model steps (core nodes). Tests put a stub with the same methods in its place (_OPS)."""

    def progress(self, n):
        import comfy.utils
        return comfy.utils.ProgressBar(n)

    def free(self):
        import comfy.model_management as mm
        mm.soft_empty_cache()

    def enhance(self, enhance_clip, draft, first_frame, seed):
        from comfy_extras import nodes_textgen as tg
        from ...kubakub.runtime import between_runs
        try:
            return tg.TextGenerate.execute(enhance_clip, draft, 700, {"sampling_mode": "on", "temperature": 0.6, "top_k": 64,
                                           "top_p": 0.95, "min_p": 0.05, "repetition_penalty": 1.05, "seed": int(seed)},
                                           image=first_frame, system_prompt=kf.H3_ENHANCE_SYSTEM).args[0]
        finally:
            between_runs()                  # one enhance per clip inside one node: see runtime.between_runs

    def keyframes_cond(self, clip, vae, prompt, W, H, length, first, last, guides):
        from comfy_extras import nodes_minimax_h3 as h3
        cond, latent = h3.MiniMaxH3ImageToVideo.execute(clip, vae, prompt, W, H, length,
                                                        first_frame=first, last_frame=last).args
        for hf, img in guides:                                   # H3 passes through the director's picture there
            cond = h3.MiniMaxH3AddGuide.execute(cond, latent, hf, vae=vae, image=img).args[0]
        return cond, latent

    def reference_cond(self, clip, vae, audio_vae, prompt, W, H, length, ref_images, ref_videos, ref_video_audios,
                       ref_audios):
        from comfy_extras import nodes_minimax_h3 as h3
        return h3.MiniMaxH3ReferenceToVideo.execute(
            clip, prompt, W, H, length, "match", vae, audio_vae, ref_images=ref_images or None,
            ref_videos=ref_videos or None, ref_video_audios=ref_video_audios or None, ref_audios=ref_audios or None).args

    def sample(self, mdl, cond, latent, prof, seed):
        """One clip with the model's schedule (prof: video_models.profile)."""
        from comfy_extras import nodes_custom_sampler as cs
        guider = cs.BasicGuider.execute(mdl, cond).args[0]
        sampler = cs.KSamplerSelect.execute(prof["sampler"]).args[0]
        if prof.get("sigmas"):
            sigmas = torch.tensor([float(v) for v in prof["sigmas"]], dtype=torch.float32)
        else:
            sigmas = cs.BasicScheduler.execute(mdl, prof.get("scheduler") or "simple", int(prof["steps"]), 1.0).args[0]
        noise = cs.RandomNoise.execute(int(seed)).args[0]
        return cs.SamplerCustomAdvanced.execute(noise, guider, sampler, sigmas, latent).args[0]

    def decode_video(self, vae, res):
        import nodes
        video = nodes.VAEDecode().decode(vae, res)[0]
        return video.reshape(-1, *video.shape[-3:])[..., :3].float().cpu()

    def decode_audio(self, audio_vae, res):
        from comfy_extras import nodes_audio as na
        snd = na.VAEDecodeAudio.execute(vae=audio_vae, samples=res).args[0]
        return {"waveform": snd["waveform"].cpu(), "sample_rate": snd["sample_rate"]}


class _LtxKeyOps(_H3Ops):
    """The same steps for LTX 2.x (core template video_ltx2_5_flf2v): the first and the last frame (and pinned
    frames) as LTXVAddGuide at strength 0.7 after LTXVPreprocess 18, the audio-video latent, the guide frames cropped
    off before decoding. LTX has no reference mode: a reference clip starts from the stretch's first frame."""

    NEGATIVE = "camera movement, zoom, pan, shaky, morphing facade, blurry, low quality, text, watermark"

    def keyframes_cond(self, clip, vae, prompt, W, H, length, first, last, guides, fps=25.0, audio_vae=None):
        from comfy_extras import nodes_lt as lt
        pos = clip.encode_from_tokens_scheduled(clip.tokenize(prompt))
        neg = clip.encode_from_tokens_scheduled(clip.tokenize(self.NEGATIVE))
        latent = lt.EmptyLTXVLatentVideo.execute(W, H, length, 1).args[0]
        for idx, img in [(0, first), *guides, (-1, last)]:
            if img is None:
                continue
            img = lt.LTXVPreprocess.execute(img[..., :3], 18).args[0]
            pos, neg, latent = lt.LTXVAddGuide.execute(pos, neg, vae, latent, img, idx, 0.7).args
        return {"pos": pos, "neg": neg, "fps": float(fps), "audio": nv._OPS.empty_audio(audio_vae, length, fps)}, latent

    def reference_cond(self, clip, vae, audio_vae, prompt, W, H, length, ref_images, ref_videos, ref_video_audios,
                       ref_audios, first=None, fps=25.0):
        return self.keyframes_cond(clip, vae, prompt, W, H, length, first, None, [], fps, audio_vae)

    def sample(self, mdl, cond, latent, prof, seed):
        from comfy_extras import nodes_lt as lt
        ops = nv._OPS
        guider = ops.guider(mdl, cond["pos"], cond["neg"], cond["fps"], prof)
        sigmas = nv._text(ops.base_sigmas(mdl, prof, latent))
        video, audio = ops.sample(mdl, guider, int(seed), sigmas, latent, cond["audio"], prof["sampler"])
        video = lt.LTXVCropGuides.execute(cond["pos"], cond["neg"], video).args[2]
        return {"video": video, "audio": audio}

    def decode_video(self, vae, res):
        return nv._OPS.decode(vae, res["video"])[..., :3].float().cpu()

    def decode_audio(self, audio_vae, res):
        if res.get("audio") is None or audio_vae is None:
            return {"waveform": torch.zeros((1, 2, 1600)), "sample_rate": 16000}
        from comfy_extras import nodes_lt_audio as lta
        snd = lta.LTXVAudioVAEDecode.execute(res["audio"], audio_vae).args[0]
        return {"waveform": snd["waveform"].cpu(), "sample_rate": snd["sample_rate"]}


_OPS = _H3Ops()
_LTX_OPS = _LtxKeyOps()


def _by_model(jobs):
    """Jobs grouped by their model (first appearance order), so each model is loaded once for all its clips."""
    order = {}
    for j in jobs:
        order.setdefault(id(j["mdl"]), len(order))
    return sorted(jobs, key=lambda j: order[id(j["mdl"])])


def _audio_slice(audio, t0, t1):
    """Core AUDIO for t0..t1 of the timeline sound (None without sound)."""
    if audio is None:
        return None
    sr = int(audio["sample_rate"])
    w = audio["waveform"]
    a, b = int(round(t0 * sr)), int(round(t1 * sr))
    return {"waveform": w[..., a:b].clone(), "sample_rate": sr}


def _frames_at_24(frames, i0, i1, fps):
    """The sequence frames i0..i1 resampled to 24 fps (H3's reference video rate)."""
    dur = (i1 - i0) / fps
    n = max(2, int(round(dur * kf.H3_FPS)) + 1)
    idx = [min(i1, i0 + int(round(k / kf.H3_FPS * fps))) for k in range(n)]
    return frames[idx]


def _layer_image(doc, layer_id, layer_images):
    """A director layer as an RGB reference image: imported / painted files from input/, node inputs by index."""
    L = next((q for q in doc.get("layers", []) if str(q.get("id")) == layer_id), None)
    if L is None:
        return None
    src = str(L.get("source") or "")
    if src.startswith("file:"):
        from ..director.nodes_director import _load_input_file
        rgba = _load_input_file(src[5:])
        if rgba is None:
            return None
        rgb = rgba[..., :3] * rgba[..., 3:4] + (1 - rgba[..., 3:4]) * 0.5     # transparent parts on grey
        return torch.from_numpy(np.ascontiguousarray(rgb, dtype=np.float32))[None]
    if src.startswith("input:") and layer_images is not None:
        k = int(src.split(":")[1])
        if k < len(layer_images):
            return layer_images[k:k + 1, ..., :3].float().cpu()
    return None


class KUBA_KeyframeClips(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="KUBA_KeyframeClips",
            display_name="kubakub keyframe clips",
            category="kubakub/2d/motion",
            search_aliases=['h3', 'minimax', 'ltx', 'video with sound'],
            description=(
                "Turns the clips you set on the director's timeline into generated video with sound and puts them "
                "into its sequence (MiniMax H3 or LTX 2.x, any variant, run locally). Keyframe clips (H3 fl2va or "
                "any LTX): the frames at the clip's start and end become the first and last frame, the model "
                "animates the way between them. Reference clips (H3 ref2va on model_ref): director layers, the music "
                "of that stretch and / or the stretch itself as references. The model's sound is mixed into the "
                "timeline sound."),
            inputs=[
                io.Model.Input("model", lazy=True, tooltip="For keyframe clips: H3 fl2va or any LTX 2.x, with or "
                                                            "without a speed LoRA (FlashGen, TaoMate, distilled ...)."),
                io.Clip.Input("clip", lazy=True, tooltip=nv.CLIP_TIP),
                io.Vae.Input("vae", lazy=True, tooltip=nv.VAE_TIP),
                io.Vae.Input("audio_vae", lazy=True, tooltip="The audio VAE of the same model (its sound is mixed in)."),
                io.Image.Input("frames", tooltip="The frames output of kubakub director sequence."),
                io.Float.Input("fps", default=25.0, min=1.0, max=120.0, tooltip="The director's fps output."),
                io.String.Input("document", multiline=True, default="", tooltip="The director's document output (its timeline clips)."),
                io.Audio.Input("audio", optional=True, tooltip="The director's audio output (timeline sound)."),
                io.Model.Input("model_ref", optional=True, lazy=True, tooltip="H3 ref2va for reference clips (+ a turbo LoRA -> fewer steps). "
                                                                              "An LTX model here makes first-frame clips (LTX has no references)."),
                io.Image.Input("layer_images", optional=True,
                               tooltip="The images connected to the director's 'layers' input (for references to those layers)."),
                io.Int.Input("steps", advanced=True, default=0, min=0, max=100,
                             tooltip="Keyframe clips. 0 = what the model needs (its speed LoRA's steps, 20 without one); "
                                     "another number overrides it."),
                io.Int.Input("steps_ref", advanced=True, default=0, min=0, max=100,
                             tooltip="Reference clips. 0 = what model_ref needs; another number overrides it."),
                io.Float.Input("megapixels", default=0.85, min=0.1, max=2.0, step=0.05,
                               tooltip="H3 work size (the aspect of the sequence); scaled to the sequence size after."),
                io.Int.Input("seed", default=42, min=0, max=0xFFFFFFFFFFFFFFFF,
                             control_after_generate=io.ControlAfterGenerate.fixed,
                             tooltip="Random seed for the clips (each clip adds its number to it); another seed gives "
                                     "other motion."),
                io.Float.Input("h3_audio", advanced=True, default=0.6, min=0.0, max=2.0, step=0.05,
                               tooltip="Level of H3's own sound in the mix (0 = keep only the timeline sound)."),
                io.Clip.Input("enhance_clip", optional=True, lazy=True,
                              tooltip="A text encoder that can write (the Gemma 4 LTX encoder: CLIPLoader type ltxv) - it "
                                      "expands each clip's prompt into H3's format, looking at the clip's first frame."),
                io.Boolean.Input("enhance", default=True, optional=True, tooltip="Use enhance_clip when connected."),
                io.Int.Input("crossfade", advanced=True, default=6, min=0, max=48, tooltip="Frames blended into the director's own frames at each clip end (H3 does not always land exactly on the end frame). At most 12 are used."),
                nv.schedule_inputs()[0],
            ],
            hidden=[io.Hidden.prompt, io.Hidden.unique_id],
            outputs=[
                io.Image.Output("frames", tooltip="The sequence with the clips in."),
                io.String.Output("prompts", tooltip="The prompts the model got, per clip (copy one into a clip with 'raw' on to tune it)."),
                io.Audio.Output("audio", tooltip="Timeline sound + the model's sound."),
                io.String.Output("report", tooltip="Per clip: mode, time span, size, frames, steps and time (or "
                                                   "'from the cache')."),
                io.Float.Output("fps", tooltip="The frame rate (frames per second), passed through for Create Video."),
            ],
        )

    @classmethod
    def check_lazy_status(cls, frames=None, fps=25.0, document="", enhance=True, **kw):
        """Only the models this document needs (lazy inputs): none without a clip to render, so a held director or a
        timeline without H3 clips loads no H3 model. Not called at all while frames / document are blocked."""
        try:
            doc = json.loads(document) if (document or "").strip() else {}
        except ValueError:
            return []                                           # execute reports the broken document
        n = int(frames.shape[0]) if frames is not None else 0
        need = kf.models_needed(doc, n, float(fps), bool(enhance), "model_ref" in kw)
        return [x for x in _LAZY if x in need and x in kw and kw[x] is None]

    @classmethod
    def execute(cls, model, clip, vae, audio_vae, frames, fps, document, audio=None, model_ref=None, layer_images=None,
                steps=0, steps_ref=0, megapixels=0.85, seed=42, h3_audio=0.6, crossfade=6, enhance_clip=None,
                enhance=True, schedule="auto") -> io.NodeOutput:
        ops = _OPS
        t0 = time.perf_counter()
        setups = {}                                             # per model input: (family, profile, ops)
        for name, mdl, n_steps in (("model", model, steps), ("model_ref", model_ref, steps_ref)):
            if mdl is not None:
                fam, prof, _o = nv.model_setup(cls, mdl, schedule, n_steps, input_name=name, default="h3", clip=clip, vae=vae,
                                               audio_vae=audio_vae)
                setups[name] = (fam, prof, _OPS if fam == "h3" else _LTX_OPS)
        doc = json.loads(document) if (document or "").strip() else {}
        out = frames[..., :3].float().cpu().clone()
        N, fh, fw = out.shape[0], out.shape[1], out.shape[2]
        duration = (N - 1) / float(fps) if N > 1 else 0.0
        clips = kf.parse_clips(doc, duration)
        work = {k: (mode, i0, i1) for k, _c, mode, i0, i1 in kf.clip_work(doc, N, fps)}
        raw = {str(c.get("id")): c for c in ((doc.get("timeline") or {}).get("clips") or []) if isinstance(c, dict)}
        W, H = kf.h3_size(fw, fh, megapixels)
        segments, rows, prompts = [], [], []
        tl = doc.get("timeline") or {}
        h3cfg = tl.get("h3") if isinstance(tl.get("h3"), dict) else {}
        pbar = ops.progress(max(1, len(clips))) if work else None
        jobs = []
        for k, c in enumerate(clips):
            if k not in work:
                rows.append(f"clip {c['id']}: too short at this fps, skipped")
                continue
            mode, i0, i1 = work[k]
            if mode == "reference" and model_ref is None:
                rows.append(f"clip {c['id']}: reference clip needs model_ref (the ref2va model), skipped")
                continue
            jobs.append({"k": k, "c": c, "mode": mode, "i0": i0, "i1": i1, "rows": []})
            rows.append(jobs[-1]["rows"])                       # this clip's lines, filled in below
        # clips are prepared together (enhance all, encode all, sample all per model, decode all), except a clip
        # that reads frames an earlier clip writes (a reference clip ending where the next one starts): it waits
        spans = [(j["i0"], j["i1"]) + ((j["i0"] + 1, j["i1"] - 1) if j["mode"] == "keyframes" else (j["i0"], j["i1"]))
                 for j in jobs]
        for run in kf.dependency_runs(spans):
            run_jobs = [jobs[i] for i in run]
            for j in run_jobs:
                cls._prepare(j, doc, raw, tl, h3cfg, out, fps, W, H, audio, layer_images, model, model_ref, clip, vae,
                             audio_vae, enhance_clip, enhance, setups, seed)
            todo = [j for j in run_jobs if j["hit"] is None]
            for j in todo:                                      # the enhancer, once for all clips
                tc = time.perf_counter()
                j["prompt"] = cls._prompt(j["draft"], j["meta"], enhance_clip, enhance, out[j["i0"]:j["i0"] + 1],
                                          seed + j["k"])
                j["t"] += time.perf_counter() - tc
            for j in todo:                                      # the text encoder (+ VAE for the anchors)
                tc = time.perf_counter()
                jo = j["ops"]
                if j["mode"] == "keyframes":
                    i0, i1 = j["i0"], j["i1"]
                    extra = {} if jo is ops else {"fps": j["mfps"], "audio_vae": audio_vae}
                    j["cond"], j["latent"] = jo.keyframes_cond(clip, vae, j["prompt"], W, H, j["length"], out[i0:i0 + 1],
                                                               out[i1:i1 + 1], [(hf, out[ia:ia + 1]) for ia, hf in j["pins"]],
                                                               **extra)
                    if j["pins"]:
                        j["rows"].append(f"clip {j['c']['id']}: pinned {len(j['pins'])} frame(s) at "
                                         + ", ".join(f"{ia / fps:.2f} s" for ia, _ in j["pins"]))
                else:
                    extra = {} if jo is ops else {"first": out[j["i0"]:j["i0"] + 1], "fps": j["mfps"]}
                    j["cond"], j["latent"] = jo.reference_cond(clip, vae, audio_vae, j["prompt"], W, H, j["length"],
                                                               *j["refs"], **extra)
                j["t"] += time.perf_counter() - tc
            for j in _by_model(todo):                           # each diffusion model loaded once for its clips
                tc = time.perf_counter()
                j["res"] = j["ops"].sample(j["mdl"], j.pop("cond"), j.pop("latent"), j["prof"], int(seed) + j["k"])
                ops.free()
                j["t"] += time.perf_counter() - tc
            for j in todo:                                      # the video VAE, then the audio VAE
                tc = time.perf_counter()
                video = j["ops"].decode_video(vae, j["res"])
                j["video_u8"] = (video.clamp(0, 1) * 255).round().to(torch.uint8)
                del video
                j["t"] += time.perf_counter() - tc
            ops.free()
            for j in todo:
                tc = time.perf_counter()
                j["snd"] = j["ops"].decode_audio(audio_vae, j.pop("res"))
                j["t"] += time.perf_counter() - tc
            if todo:
                ops.free()
            for j in run_jobs:
                if j["hit"] is not None:                        # this clip is unchanged: no enhance, no sampling
                    j["video_u8"], j["snd"], j["prompt"] = j["hit"]
                else:
                    _CLIP_CACHE.put(j["ckey"], j["objs"], (j["video_u8"], j["snd"], j["prompt"]))
                cls._paste(j, out, fps, crossfade, fh, fw)
                c, snd = j["c"], j["snd"]
                prompts.append(f"[clip {c['id']} {c['t_start']:g}-{c['t_end']:g} s{', reference' if j['mode'] == 'reference' else ''}]\n{j['prompt']}")
                wave = snd["waveform"][0].float().cpu().numpy()
                segments.append((c["t_start"], wave, int(snd["sample_rate"])))
                j["rows"].append(f"clip {c['id']} ({j['mode']}): {c['t_start']:.2f}-{c['t_end']:.2f} s, "
                                 f"{vm.NAMES[j['fam']]} {W}x{H} x {j['length']} frames, {j['n_steps']} steps, "
                                 + ("unchanged, from the cache" if j["hit"] is not None else f"{j['t']:.0f} s"))
                for key in ("video_u8", "hit", "snd"):
                    j.pop(key, None)
                pbar.update(1)
        rows = [r for x in rows for r in (x if isinstance(x, list) else [x])]
        # sound: the timeline sound (or silence of the sequence length) + H3's sound at the clips
        sr = int(audio["sample_rate"]) if audio is not None else 44100
        base = audio["waveform"][0].float().cpu().numpy() if audio is not None else np.zeros((2, int(round(duration * sr))), np.float32)
        mixed = kf.mix_audio(base, sr, segments, float(h3_audio)) if segments and h3_audio > 0 else base
        audio_out = {"waveform": torch.from_numpy(np.ascontiguousarray(mixed))[None], "sample_rate": sr}
        models = [vm.describe(prof).replace("model:", f"{name}:", 1) for name, (_f, prof, _o) in setups.items() if work]
        report = "\n".join([f"{len(clips)} clip(s) in {N} frames at {fps:g} fps, {time.perf_counter() - t0:.0f} s", *models, *rows]
                           + ([] if work else ["no clip to render: frames passed through, no video model loaded"]))
        log.info("[KUBA director] keyframe clips: %s", report.split("\n")[0])
        return io.NodeOutput(out, "\n\n".join(prompts), audio_out, report, float(fps))

    @staticmethod
    def _prepare(j, doc, raw, tl, h3cfg, out, fps, W, H, audio, layer_images, model, model_ref, clip, vae, audio_vae,
                 enhance_clip, enhance, setups, seed):
        """A clip's draft prompt, references, cache key and cache lookup (no model runs here)."""
        tc = time.perf_counter()
        c, k, i0, i1 = j["c"], j["k"], j["i0"], j["i1"]
        meta = raw.get(c["id"], {})
        dur = (i1 - i0) / float(fps)
        fam, prof, jops = setups.get("model" if j["mode"] == "keyframes" else "model_ref",
                                     ("h3", vm.profile("h3"), _OPS))
        if jops is not _LTX_OPS:
            jops = _OPS                                         # the module's ops (tests replace them)
        length, mfps = kf.model_length(fam, dur, fps)
        ltx_ref = j["mode"] == "reference" and fam != "h3"
        if ltx_ref:
            j["rows"].append(f"clip {c['id']}: LTX has no reference mode, so this clip starts from the stretch's first "
                             f"frame (put an H3 ref2va model on model_ref for references)")
        beats = kf.clip_beats(tl.get("markers"), c["t_start"], c["t_end"])
        meta_c = {**meta, "prompt": c["prompt"], "mode": j["mode"]}
        use_enh = enhance_clip if (enhance and not meta.get("raw")) else None
        if j["mode"] == "keyframes":
            draft = kf.build_h3_prompt(meta_c, h3cfg, dur, beats)
            mdl = model
            pins = kf.clip_anchors(tl.get("markers"), fps, i0, i1, length, mfps)  # markers set to pin their frame
            ckey = _digest("keyframes", fam, vm.describe(prof), draft, W, H, length, seed + k, out[i0], out[i1],
                           [(hf, out[ia]) for ia, hf in pins])
            j.update(pins=pins)
        else:
            refs = [] if ltx_ref else [str(r) for r in meta.get("refs") or []]
            ref_images = {}
            for r in refs:
                if r.startswith("layer:"):
                    img = _layer_image(doc, r[6:], layer_images)
                    if img is not None:
                        ref_images[f"ref_image_{len(ref_images) + 1}"] = img
            ref_audios, ref_videos, ref_video_audios = {}, {}, {}
            if "music" in refs and audio is not None:
                ref_audios["ref_audio_1"] = _audio_slice(audio, c["t_start"], c["t_end"])
            if "sequence" in refs:
                if dur >= 2.0:
                    ref_videos["ref_video_1"] = _frames_at_24(out, i0, i1, fps)
                    if audio is not None:
                        ref_video_audios["ref_video_audio_1"] = _audio_slice(audio, c["t_start"], c["t_end"])
                else:
                    j["rows"].append(f"clip {c['id']}: the sequence as reference needs >= 2 s, left out")
            tags = [(f"<Picture {n + 1}>", "the look of " + (next((str(q.get("name")) for q in doc.get("layers", [])
                     if str(q.get("id")) == r[6:]), "the layer"))) for n, r in enumerate(x for x in refs if x.startswith("layer:"))][:len(ref_images)]
            if ref_videos:
                tags.append(("<Video 1>", "the layout, the light and the motion"))
            if ref_audios:
                tags.append(("<Audio 1>", "the music"))
            draft = kf.build_h3_prompt(meta_c, h3cfg, dur, beats, tags)
            mdl = model_ref
            ckey = _digest("reference", fam, vm.describe(prof), draft, W, H, length, seed + k, out[i0], ref_images,
                           ref_videos, ref_video_audios, ref_audios)
            j.update(refs=(ref_images, ref_videos, ref_video_audios, ref_audios))
        objs = (mdl, clip, vae, audio_vae, use_enh)
        j.update(meta=meta_c, draft=draft, length=length, dur=dur, mdl=mdl, n_steps=prof["steps"], objs=objs, ckey=ckey,
                 hit=_CLIP_CACHE.get(ckey, objs), t=time.perf_counter() - tc, fam=fam, prof=prof, ops=jops, mfps=mfps)

    @staticmethod
    def _paste(j, out, fps, crossfade, fh, fw):
        """The clip into the sequence, by time; keyframe clips keep the director's own end frames (the anchors)."""
        i0, i1 = j["i0"], j["i1"]
        crossfade = min(crossfade, kf.HIDDEN_MARGIN)            # skip_h3_frames keeps only that many director frames
        video = j["video_u8"].float() / 255.0                   # the same 8 bits whether cached or not
        idx = kf.frame_map(i1 - i0 + 1, j["dur"], video.shape[0], j["mfps"])
        lo, hi = (i0 + 1, i1 - 1) if j["mode"] == "keyframes" else (i0, i1)
        for jj in range(lo, hi + 1):
            f = video[idx[jj - i0]].numpy()
            if f.shape[:2] != (fh, fw):
                f = cv2.resize(f, (fw, fh), interpolation=cv2.INTER_CUBIC)
            f = torch.from_numpy(np.clip(f, 0, 1))
            if crossfade > 0:                                   # soft joins into the director's own frames
                a = min(1.0, (jj - i0 + 1) / (crossfade + 1), (i1 - jj + 1) / (crossfade + 1))
                f = out[jj] * (1 - a) + f * a
            out[jj] = f

    @staticmethod
    def _prompt(draft, meta, enhance_clip, enhance, first_frame, seed):
        """The draft in H3's format, expanded by a writing text encoder (core Generate Text) when connected. Cached
        by draft, first frame, seed and enhancer, so a clip rendered again with other steps / size / LoRA reuses it."""
        if enhance_clip is None or not enhance or meta.get("raw"):
            return draft
        pkey = _digest("enhance", draft, first_frame, int(seed))
        hit = _PROMPT_CACHE.get(pkey, (enhance_clip,))
        if hit is not None:
            log.info("[KUBA director] enhanced prompt unchanged, from the cache")
            return hit
        t0 = time.time()
        try:
            text = kf.restore_anchors((_OPS.enhance(enhance_clip, draft, first_frame, seed) or "").strip(), draft)
        except Exception as e:  # noqa: BLE001  (an encoder that cannot write: keep the draft)
            log.warning("[KUBA director] prompt enhance failed, using the draft: %s", e)
            return draft
        if not kf.plausible_prompt(text, draft):
            log.warning("[KUBA director] the prompt enhancer returned unusable text (can this encoder write?), using the draft: %r", text[:120])
            text = draft                                        # the same seed gives the same text: cached as well
        else:
            log.info("[KUBA director] prompt enhanced in %.0f s", time.time() - t0)
        _PROMPT_CACHE.put(pkey, (enhance_clip,), text)
        return text


NODE_CLASS_MAPPINGS = {"KUBA_KeyframeClips": KUBA_KeyframeClips}
NODE_DISPLAY_NAME_MAPPINGS = {"KUBA_KeyframeClips": "kubakub keyframe clips"}
