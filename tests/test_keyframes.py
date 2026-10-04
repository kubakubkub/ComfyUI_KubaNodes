"""
Model free test for kubakub keyframe clips (kubakub/keyframes.py): clip parsing, H3 lengths and sizes, the
24 fps -> sequence fps mapping, audio mixing; the node with stub models: lazy model inputs, pass-through without clips,
phases (each model once), the clip cache and the separate enhanced-prompt cache.

Run from the ComfyUI_windows_portable folder:
    python_embeded\\python.exe ComfyUI\\custom_nodes\\ComfyUI_KubaNodes\\tests\\test_keyframes.py
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from kubakub import keyframes as kf  # noqa: E402

failures = []


def check(name, cond, detail=""):
    print(("ok   " if cond else "FAIL ") + name + (f"  ({detail})" if detail and not cond else ""))
    if not cond:
        failures.append(name)


doc = {"timeline": {"clips": [
    {"id": "b", "t_start": 4, "t_end": 7, "prompt": "  lights   on "},
    {"id": "a", "t_start": 1, "t_end": 4.5, "prompt": "dawn"},
    {"id": "off", "t_start": 8, "t_end": 9, "on": False},
    {"id": "tiny", "t_start": 9, "t_end": 9.1},
    {"id": "late", "t_start": 9.5, "t_end": 30},
    {"id": "bad", "t_start": "x", "t_end": 2}]}}
c = kf.parse_clips(doc, 10.0)
check("clips sorted, off / tiny / bad dropped", [x["id"] for x in c] == ["a", "b", "late"], str(c))
check("overlap trimmed: b starts where a ends", c[1]["t_start"] == 4.5 and c[1]["prompt"] == "lights on")
check("clip end clamped to the timeline", c[2]["t_end"] == 10.0)
check("no timeline -> no clips", kf.parse_clips({}, 5) == [])

check("h3_length snaps to 17k+5", all((kf.h3_length(d) - 5) % 17 == 0 and kf.h3_length(d) >= round(d * 24) + 1
                                      for d in (0.2, 1, 2.3, 3, 5, 12.4)))
check("h3_length 5 s = 124 (the model's default)", kf.h3_length(5.0) == 124, str(kf.h3_length(5.0)))
w, h = kf.h3_size(3200, 2160, 0.85)
check("h3_size keeps the aspect on the grid", w % 32 == 0 and h % 32 == 0 and abs(w / h - 3200 / 2160) < 0.05
      and 0.75e6 < w * h < 0.95e6, f"{w}x{h}")

m = kf.frame_map(76, 3.0, kf.h3_length(3.0))             # 3 s at 25 fps: 76 sequence frames, 73 H3 frames used
check("frame map: first / last at the anchors", m[0] == 0 and m[-1] == 72, str(m[-3:]))
check("frame map monotonic", all(b >= a for a, b in zip(m, m[1:])))
check("frame map 24 -> 24 fps is the identity", kf.frame_map(49, 2.0, kf.h3_length(2.0)) == list(range(49)))

base = np.zeros((2, 44100 * 4), np.float32)
seg = np.ones((1, 32000), np.float32) * 0.5               # 1 s mono at 32 kHz
out = kf.mix_audio(base, 44100, [(1.0, seg, 32000)], gain=0.6)
check("mix: segment placed at its start, resampled, channels matched",
      out[0, 44100 - 10] == 0 and abs(out[1, 44100 + 100] - 0.3) < 1e-4 and abs(out[0, 2 * 44100 + 100]) < 1e-6)
check("mix: clipped to [-1, 1], base untouched", kf.mix_audio(base + 0.9, 44100, [(0, seg, 32000)], 1.0).max() == 1.0 and base.max() == 0)
check("mix: segment past the end ignored", kf.mix_audio(base, 44100, [(10, seg, 32000)]).max() == 0)

# H3 prompt language
b = kf.clip_beats([{"t": 2.0, "name": "arcade lights on"}, {"t": 2.5, "name": "marker 3"}, {"t": 5, "name": "outside"},
                   {"t": "x"}], 1.0, 4.0)
check("beats: named markers inside the clip, relative", b == [(1.0, "arcade lights on")], str(b))
p = kf.build_h3_prompt({"prompt": "the facade wakes up", "audio": "soft synth pad"}, {"look": "Warm night."}, 3.0, b)
lines = p.split("\n")
check("keyframes prompt structure", lines[0] == "Warm night. The environment is constant throughout."
      and lines[1] == "The scene opens exactly on image 1 and ends exactly on image 2."
      and "[0s-1s] the facade wakes up" in lines and "[1s-3s] arcade lights on" in lines
      and "Audio: soft synth pad" in lines and lines[-1] == kf.H3_DEFAULTS["avoid"], p)
check("camera line is the locked-off default", kf.H3_DEFAULTS["camera"] in lines)
r = kf.build_h3_prompt({"prompt": "gold lines trace the arches", "mode": "reference"}, {}, 2.5, (),
                       [("<Video 1>", "the layout and motion"), ("<Picture 1>", "the kiosk"), ("<Audio 1>", "the music")])
check("reference prompt: tags in order, audio exactly as it is",
      "Use <Video 1> as the layout and motion, <Picture 1> as the kiosk, <Audio 1> exactly as it is." in r
      and "image 1" not in r and "[0s-2.5s] gold lines trace the arches" in r, r)
check("raw prompt is sent as written", kf.build_h3_prompt({"prompt": " my own  prompt ", "raw": True}, {}, 3) == "my own prompt")
garbage = "FSOKNZIUBPYLAOtRECTEVYGEYDHvLlfRITXTKKMGOkIEPBtnSNUSVLMORFSIBRVLUMHAAERDDVDOrtPCBVGLPBNOSASZPFRTMOF"
check("plausible: tonight's garbage output is rejected", not kf.plausible_prompt(garbage, p))
good = p.replace("Warm night.", "A warm night, sodium light on pale limestone, calm and still.")
check("plausible: a real rewrite keeping the structure is used", kf.plausible_prompt(good, p))
check("plausible: a rewrite that drops the anchor line is rejected", not kf.plausible_prompt(good.replace("image 1", "the start"), p))
check("plausible: a rewrite that drops a reference tag is rejected", not kf.plausible_prompt(r.replace("<Audio 1>", "the music"), r))
anchor = "The scene opens exactly on image 1 and ends exactly on image 2."
renamed = good.replace(anchor, "The scene opens exactly on <Picture 1> and ends exactly on <Picture 2>.")
fixed = kf.restore_anchors(renamed, p)
check("restore: '<Picture 1>' rewording of the anchor line is put back", anchor in fixed and "<Picture" not in fixed
      and kf.plausible_prompt(fixed, p) and "sodium light" in fixed, fixed)
dropped = "\n".join(ln for ln in good.split("\n") if ln != anchor)
fixed = kf.restore_anchors(dropped, p)
check("restore: a dropped anchor line goes in before Timeline:",
      fixed.split("\n").index(anchor) == fixed.split("\n").index("Timeline:") - 1, fixed)
check("restore: an untouched rewrite stays as it is", kf.restore_anchors(good, p) == good)
check("restore: reference tags are kept", kf.restore_anchors(r, r) == r)
hd = {"timeline": {"clips": [{"id": "c1", "t_start": 1.0, "t_end": 3.0}]}}
hid = kf.hidden_frames(hd, 100, 25)
check("hidden: frames 25..75 minus 12 at each end", hid == list(range(38, 63)), str(hid[:3]) + str(hid[-3:]))
check("hidden: crossfade frames stay rendered", all(i not in hid for i in list(range(25, 38)) + list(range(63, 76))))
ak = {"timeline": {"clips": [{"id": "c1", "t_start": 1.0, "t_end": 3.0}], "markers": [{"t": 2.0, "name": "peak", "anchor": True},
                                                                                     {"t": 2.4, "name": "beat only"}, {"t": 1.02, "anchor": True}]}}
anc = kf.clip_anchors(ak["timeline"]["markers"], 25, 25, 75, kf.h3_length(2.0))
check("anchors: only pinned markers inside the clip (not at its ends) -> (sequence, H3 frame)", anc == [(50, 24)], str(anc))
check("hidden: a pinned frame is rendered (not skipped)", 50 not in kf.hidden_frames(ak, 100, 25) and 49 in kf.hidden_frames(ak, 100, 25))
check("hidden: a reference clip hides nothing",
      kf.hidden_frames({"timeline": {"clips": [{"id": "c1", "t_start": 1.0, "t_end": 3.0, "mode": "reference"}]}}, 100, 25) == [])
check("hidden: a clip that is off hides nothing",
      kf.hidden_frames({"timeline": {"clips": [{"id": "c1", "t_start": 1.0, "t_end": 3.0, "on": False}]}}, 100, 25) == [])
check("hidden: a short clip keeps all its frames", kf.hidden_frames({"timeline": {"clips": [{"id": "c1", "t_start": 1.0, "t_end": 1.8}]}}, 100, 25) == [])
check("hidden: no timeline, nothing hidden", kf.hidden_frames({}, 100, 25) == [] and kf.hidden_frames(hd, 1, 25) == [])
check("enhancer guide keeps the structure", "Timeline:" in kf.H3_ENHANCE_SYSTEM and "word for word" in kf.H3_ENHANCE_SYSTEM)

# ---- the node KUBA_KeyframeClips with stub models ----------------------------------------------------------------
import json  # noqa: E402
import types  # noqa: E402

import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(HERE))))     # ComfyUI (comfy_api)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))     # embedded Python leaves the script folder off the path
import _pack  # noqa: E402,F401  (the pack as a package, see tests/_pack.py)
from kubapack.nodes.motion import nodes_keyframes as nk  # noqa: E402


class _Model:
    def __init__(self, name):
        self.name = name


class _StubH3:
    """nk._H3Ops with no model: records (step, model name); results depend on the inputs."""

    def __init__(self):
        self.calls = []

    def progress(self, n):
        return types.SimpleNamespace(update=lambda k: None)

    def free(self):
        pass

    def enhance(self, enhance_clip, draft, first_frame, seed):
        self.calls.append(("enhance", enhance_clip.name))
        return draft + "\nSoft warm light slowly moves over the pale stone of the facade tonight."

    def keyframes_cond(self, clip, vae, prompt, W, H, length, first, last, guides):
        self.calls.append(("encode", clip.name))
        return float(len(prompt)) + float(first.mean()) + float(last.mean()), {"length": length, "W": W, "H": H}

    def reference_cond(self, clip, vae, audio_vae, prompt, W, H, length, ref_images, ref_videos, ref_video_audios,
                       ref_audios):
        self.calls.append(("encode", clip.name))
        return float(len(prompt)), {"length": length, "W": W, "H": H}

    def sample(self, mdl, cond, latent, steps, seed):
        self.calls.append(("sample", mdl.name))
        return {"v": (cond * 0.37 + steps * 0.01 + (seed % 7) * 0.05) % 1.0, **latent}

    def decode_video(self, vae, res):
        self.calls.append(("decode_video", vae.name))
        return torch.full((res["length"], res["H"] // 8, res["W"] // 8, 3), res["v"])

    def decode_audio(self, audio_vae, res):
        self.calls.append(("decode_audio", audio_vae.name))
        return {"waveform": torch.full((1, 1, 16000), 0.1), "sample_rate": 16000}


class _NoH3(_StubH3):
    def __getattribute__(self, k):
        if k in ("enhance", "keyframes_cond", "reference_cond", "sample", "decode_video", "decode_audio", "progress"):
            raise AssertionError(f"a model step ran: {k}")
        return object.__getattribute__(self, k)


def _blocks(calls):
    steps = [c[0] for c in calls]
    return [s for i, s in enumerate(steps) if i == 0 or s != steps[i - 1]]


H3M = {k: _Model(k) for k in ("model", "clip", "vae", "audio_vae", "model_ref", "enhance_clip")}
SEQ = torch.rand(101, 48, 64, 3)                          # 4 s at 25 fps


def _doc(*clips, markers=()):
    return json.dumps({"timeline": {"clips": list(clips), "markers": list(markers)}})


KC1 = {"id": "c1", "t_start": 0.4, "t_end": 1.4, "prompt": "the facade wakes up"}
KC2 = {"id": "c2", "t_start": 2.6, "t_end": 3.6, "prompt": "lights go out"}
RC = {"id": "r1", "t_start": 1.6, "t_end": 2.4, "prompt": "gold lines", "mode": "reference"}


def _h3(doc, ops, models=None, **kw):
    nk._OPS = ops
    m = H3M if models is None else models
    args = dict(frames=SEQ, fps=25.0, document=doc, steps=4, seed=3, crossfade=2, megapixels=0.05)
    args.update({k: m.get(k) for k in H3M})
    args.update(kw)
    return nk.KUBA_KeyframeClips.execute(**args)


lz = nk.KUBA_KeyframeClips.check_lazy_status
unset = {k: None for k in ("model", "clip", "vae", "audio_vae")}
check("lazy: no clip -> no model", lz(frames=SEQ, fps=25.0, document=_doc(), enhance=True, enhance_clip=None, **unset) == [])
check("lazy: empty document -> no model", lz(frames=SEQ, fps=25.0, document="", **unset) == [])
check("lazy: broken document -> no model (execute reports it)", lz(frames=SEQ, fps=25.0, document="{", **unset) == [])
check("lazy: clips off -> no model", lz(frames=SEQ, fps=25.0, document=_doc({**KC1, "on": False}), **unset) == [])
check("lazy: a clip too short at this fps -> no model",
      lz(frames=SEQ[:3], fps=25.0, document=_doc({"id": "t", "t_start": 0.0, "t_end": 0.25}), **unset) == [])
got = lz(frames=SEQ, fps=25.0, document=_doc(KC1), enhance=True, model_ref=None, enhance_clip=None, **unset)
check("lazy: keyframe clip -> model, clip, vaes, enhancer (not model_ref)",
      got == ["model", "clip", "vae", "audio_vae", "enhance_clip"], str(got))
got = lz(frames=SEQ, fps=25.0, document=_doc(RC), enhance=False, model_ref=None, enhance_clip=None, **unset)
check("lazy: reference clip, enhance off -> model_ref, clip, vaes (not model, not the enhancer)",
      got == ["clip", "vae", "audio_vae", "model_ref"], str(got))
check("lazy: reference clip without model_ref connected -> nothing (the clip is skipped)",
      lz(frames=SEQ, fps=25.0, document=_doc(RC), **unset) == [])
got = lz(frames=SEQ, fps=25.0, document=_doc({**KC1, "raw": True}), enhance=True, enhance_clip=None, **unset)
check("lazy: raw clip -> no enhancer", "enhance_clip" not in got and "model" in got, str(got))
check("lazy: everything loaded -> nothing more", lz(frames=SEQ, fps=25.0, document=_doc(KC1, RC), **H3M) == [])

r = _h3(_doc(), _NoH3(), models={})
check("no clip: frames pass through, no model step", torch.equal(r.args[0], SEQ[..., :3]) and "no clip to render" in r.args[3],
      r.args[3])
check("no clip: silent sound of the sequence length", r.args[2]["waveform"].shape[-1] == 4 * 44100)

nk._CLIP_CACHE.clear()
nk._PROMPT_CACHE.clear()
ops = _StubH3()
a = _h3(_doc(KC1, RC, KC2), ops)
b = _blocks(ops.calls)
check("phases: enhance all > encode all > sample per model > decode video all > decode audio all",
      b == ["enhance", "encode", "sample", "decode_video", "decode_audio"], str(b))
samp = [c[1] for c in ops.calls if c[0] == "sample"]
check("each diffusion model once for its clips (keyframes model, then ref model)",
      samp == ["model", "model", "model_ref"], str(samp))
check("clips in, outside untouched", not torch.equal(a.args[0][20], SEQ[20]) and torch.equal(a.args[0][38], SEQ[38])
      and torch.equal(a.args[0][10], SEQ[10]))
check("prompts in clip order", [p.split("]")[0] for p in a.args[1].split("\n\n")] == ["[clip c1 0.4-1.4 s",
                                                                                     "[clip r1 1.6-2.4 s, reference",
                                                                                     "[clip c2 2.6-3.6 s"], a.args[1][:200])
ops = _StubH3()
b2 = _h3(_doc(KC1, RC, KC2), ops)
check("unchanged: all from the cache, no model step, same frames and sound",
      ops.calls == [] and torch.equal(a.args[0], b2.args[0]) and torch.equal(a.args[2]["waveform"], b2.args[2]["waveform"]))
ops = _StubH3()
_h3(_doc(KC1, RC, {**KC2, "prompt": "lights go out slowly"}), ops)
check("one prompt changed: only that clip enhances and renders",
      [c[0] for c in ops.calls] == ["enhance", "encode", "sample", "decode_video", "decode_audio"], str(ops.calls))
ops = _StubH3()
_h3(_doc(KC1, RC, KC2), ops, steps=6)
check("steps changed: keyframe clips sample again, the enhancer does not run",
      [c[0] for c in ops.calls].count("sample") == 2 and "enhance" not in [c[0] for c in ops.calls], str(ops.calls))
ops = _StubH3()
_h3(_doc(KC1, RC, KC2), ops, model=_Model("model"))
check("LoRA changed (a new model object): sampled again, no enhance",
      [c[0] for c in ops.calls].count("sample") == 2 and "enhance" not in [c[0] for c in ops.calls], str(ops.calls))
ops = _StubH3()
_h3(_doc(KC1, RC, KC2), ops, enhance_clip=_Model("enhance_clip"))
check("another enhancer: enhanced again", [c[0] for c in ops.calls].count("enhance") == 3, str(ops.calls))

# a reference clip that ends where the next clip starts: the next clip reads the frame it writes -> it waits
check("dependency runs: keyframe clips never wait", kf.dependency_runs([(10, 20, 11, 19), (20, 30, 21, 29)]) == [[0, 1]])
check("dependency runs: after an adjacent reference clip, a new run",
      kf.dependency_runs([(10, 20, 10, 20), (20, 30, 21, 29), (40, 50, 41, 49)]) == [[0], [1, 2]])
nk._CLIP_CACHE.clear()
nk._PROMPT_CACHE.clear()
RA = {**RC, "t_start": 1.0, "t_end": 2.0}
KA = {**KC1, "t_start": 2.0, "t_end": 3.0}
ops = _StubH3()
adj = _h3(_doc(RA, KA), ops)
check("adjacent reference > keyframes: two runs (the second after the first is pasted)",
      _blocks(ops.calls) == ["enhance", "encode", "sample", "decode_video", "decode_audio"] * 2, str(_blocks(ops.calls)))

print()
if failures:
    print(f"{len(failures)} FAILED: {', '.join(failures)}")
    sys.exit(1)
print("all keyframe tests passed")
