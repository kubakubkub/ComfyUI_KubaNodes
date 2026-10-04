"""
Model free test for the region video sampler logic (kubakub/video.py) and the video plan keys, and the node
KUBA_RegionVideoSampler with stub models (lazy model inputs, pass-through without animated regions, the clip cache,
one load per model, parallel frame writing). Does not start ComfyUI.

Run from the ComfyUI_windows_portable folder:
    python_embeded\\python.exe ComfyUI\\custom_nodes\\ComfyUI_KubaNodes\\tests\\test_video.py
"""

import os
import shutil
import sys
import tempfile
import types

import cv2
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(HERE))))     # ComfyUI (comfy_api)
sys.path.insert(0, os.path.dirname(HERE))
from kubakub import clip_cache as cc  # noqa: E402
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))     # embedded Python leaves the script folder off the path
import _pack  # noqa: E402,F401  (the pack as a package, see tests/_pack.py)
from kubapack.nodes.motion import nodes_video as nv  # noqa: E402
from kubakub import plan as rp  # noqa: E402
from kubakub import video as vd  # noqa: E402


def test_frames():
    assert [vd.ltx_frames(n) for n in (1, 2, 9, 10, 121, 125)] == [1, 9, 9, 17, 121, 129]
    assert vd.latent_frames(121) == 16
    t = vd.latent_frame_times(121, 25.0)
    assert t[0] == 0 and abs(t[1] - 4.5 / 25) < 1e-9 and abs(t[-1] - 116.5 / 25) < 1e-9
    return "8k+1 frames, 121 -> 16 latent frames"


def test_plan_keys_and_groups():
    table = {"regions": [{"region_id": i, "name": n, "group_id": g} for i, (n, g) in
                         enumerate([("W_F1_C01", "Windows"), ("W_F1_C02", "Windows"), ("W_F2_C01", "Windows"),
                                    ("wall", "Groups")])]}
    rules = ("[default]\nstrategy = keep\n[W_F1_*]\nanimate = on\nvideo_prompt = lamps flicker\n"
             "[W_F2_*]\nanimate = on\nvideo_prompt = a bird flies past\nt_start = 2\nt_end = 3.5\nmotion = 0.7\n")
    plan = rp.resolve(table, rules, seed=1)
    e = {x["name"]: x for x in plan["regions"]}
    assert e["W_F1_C01"]["animate"] is True and e["wall"]["animate"] is False
    assert e["W_F2_C01"]["t_start"] == 2.0 and e["W_F2_C01"]["motion"] == 0.7
    groups = vd.clip_groups(plan["regions"])
    assert sorted(len(g["ids"]) for g in groups) == [1, 2], groups
    only = vd.clip_groups(plan["regions"], only="W_F2_*")
    assert len(only) == 1 and only[0]["t_end"] == 3.5
    return f"{len(groups)} clips: floor 1 together, floor 2 own time range"


def test_crop_plan():
    m = np.zeros((2160, 3840), bool)
    m[975:1420, 1012:2833] = True                             # the five floor 1 windows
    (x0, y0, x1, y1), (w1, h1) = vd.crop_plan(m, target_px=450_000, context_px=64)
    assert w1 % 32 == 0 and h1 % 32 == 0 and abs(w1 * h1 - 450_000) < 60_000
    assert abs((x1 - x0) / (y1 - y0) - w1 / h1) < 0.01, "uniform scale"
    assert x0 <= 1012 - 64 and x1 >= 2833 + 64 and y0 <= 975 - 64 and y1 >= 1420 + 64
    edge = np.zeros((2160, 3840), bool)
    edge[2100:2160, 3700:3840] = True                         # at the corner: the box is shifted inside
    (x0, y0, x1, y1), _ = vd.crop_plan(edge)
    assert 0 <= x0 and x1 <= 3840 and 0 <= y0 and y1 <= 2160
    return f"floor 1 -> {w1}x{h1} from {x1 - x0}x{y1 - y0} px"


def test_latent_mask_timing():
    region = np.zeros((384, 1152), bool)
    region[100:300, 200:400] = True
    m = vd.latent_mask(region, 36, 12, 121, 25.0, t_start=2.0, t_end=3.0, motion=0.8)
    assert m.shape == (1, 1, 16, 12, 36)
    assert m[0, 0, 0].max() == 0, "first frame stays the still"
    t = vd.latent_frame_times(121, 25.0)
    active = [i for i in range(16) if m[0, 0, i].max() > 0]
    assert all(2.0 <= t[i] <= 3.0 for i in active) and active, active
    assert abs(m.max() - 0.8) < 1e-6 and m[0, 0, active[0], 0, 0] == 0, "only the region (+1 latent px)"
    assert vd.latent_mask(region, 36, 12, 121, 25.0)[0, 0, -1].max() > 0, "no loop: the clip ends anywhere"
    lp = vd.latent_mask(region, 36, 12, 121, 25.0, loop=True)
    assert lp[0, 0, 0].max() == 0 and lp[0, 0, -1].max() == 0 and lp[0, 0, 1:-1].max() == 1, "loop: both ends are the still"
    assert vd.audio_range(25, 121, 25.0, 48000) == (48000, 232320), "the sound of a clip from 1 s on"
    return f"active latent frames {active}; loop holds the last one"


def test_paste_exact_outside():
    frame = np.full((200, 300, 3), 0.5, np.float32)
    mask = np.zeros((200, 300), bool)
    mask[50:100, 100:200] = True
    alpha = vd.feather_mask(mask, 4)
    assert alpha[mask].min() == 1.0 and alpha[0, 0] == 0.0 and 0 < alpha[48, 150] < 1
    clip = np.ones((64, 96, 3), np.float32)
    out = vd.paste_frame(frame.copy(), clip, (80, 30, 230, 130), alpha)
    far = alpha == 0
    assert (out[far] == 0.5).all(), "pixels outside the feather keep the still exactly"
    assert np.allclose(out[mask], 1.0)
    return "region replaced, facade outside the feather untouched"


# ---- the node with stub models ----------------------------------------------------------------------------------

class _Model:
    """A stand-in for a loaded model (weak-referenceable, like ModelPatcher / CLIP / VAE)."""

    def __init__(self, name):
        self.name = name


class _StubOps:
    """nv._LtxOps with no model: records every step as (step, model name); results depend on every input."""

    def __init__(self):
        self.calls = []

    def progress(self, n):
        return types.SimpleNamespace(update=lambda k: None)

    def free(self):
        pass

    def vae_encode(self, vae, pixels):
        self.calls.append(("vae_encode", vae.name))
        f, h, w = pixels.shape[:3]
        return {"samples": torch.full((1, 4, (f - 1) // 8 + 1, h // 32, w // 32), float(pixels.mean()))}

    def enhance(self, enhance_clip, text, first, seed):
        self.calls.append(("enhance", enhance_clip.name))
        return f"{text}, rendered in a slow and calm way with soft light moving, seed {seed}"

    def encode_text(self, clip, text):
        self.calls.append(("encode_text", clip.name))
        return [[torch.tensor([float(len(text))]), {}]]

    def guider(self, model, pos, neg, fps):
        return (float(pos[0][0][0]), float(neg[0][0][0]))

    def empty_audio(self, audio_vae, frames, fps):
        return {"samples": torch.zeros(1, 1, 1, 1)}

    def sound_audio(self, audio_vae, waveform, sample_rate):
        self.calls.append(("sound_audio", audio_vae.name))
        return {"samples": waveform.mean().reshape(1, 1, 1, 1) + 0.25, "noise_mask": torch.zeros(1, 1, 1, 1)}

    def sample(self, model, guider, seed, sigmas, video, audio):
        self.calls.append(("sample", model.name))
        self.audio_seen = getattr(self, "audio_seen", []) + [audio]
        s = (video["samples"] + float(video["noise_mask"].mean()) + guider[0] * 1e-3 + (seed % 97) * 1e-4
             + float(audio["samples"].mean()))
        return {"samples": s}, {"samples": audio["samples"]}

    def upsample(self, video, upscale_model, vae):
        self.calls.append(("upsample", upscale_model.name))
        return {"samples": video["samples"].repeat_interleave(2, 3).repeat_interleave(2, 4)}

    def decode(self, vae, video):
        self.calls.append(("decode", vae.name))
        s = video["samples"]
        t, h, w = (s.shape[2] - 1) * 8 + 1, s.shape[3] * 32, s.shape[4] * 32
        ramp = torch.linspace(0, 1, t)[:, None, None, None]
        return (torch.sigmoid(s.mean()) * 0.5 + 0.4 * ramp).expand(t, h, w, 3).clone()


class _NoOps(_StubOps):
    """Any model step fails the test."""

    def __getattribute__(self, k):
        if k in ("vae_encode", "enhance", "encode_text", "sample", "upsample", "decode", "progress"):
            raise AssertionError(f"a model step ran: {k}")
        return object.__getattribute__(self, k)


def _plan(rules, H=192, W=320):
    names = ["W_F1_C01", "W_F1_C02", "W_F2_C01", "wall"]
    table = {"regions": [{"region_id": i, "name": n, "group_id": "Windows"} for i, n in enumerate(names)]}
    lab = np.full((H, W), 3, np.int32)
    lab[20:60, 30:90], lab[20:60, 120:180], lab[110:150, 200:280] = 0, 1, 2
    regions = types.SimpleNamespace(labels=torch.from_numpy(lab)[None])
    return types.SimpleNamespace(plan=rp.resolve(table, rules, seed=1), regions=regions)


RULES2 = ("[default]\nstrategy = keep\n[W_F1_*]\nanimate = on\nvideo_prompt = lamps flicker\n"
          "[W_F2_*]\nanimate = on\nvideo_prompt = a bird flies past\n")
RULES0 = "[default]\nstrategy = keep\n"
MODELS = {k: _Model(k) for k in ("model", "clip", "vae", "audio_vae", "upscale_model", "enhance_clip")}
STILL = torch.from_numpy(np.dstack([np.tile(np.linspace(0, 1, 320, dtype=np.float32), (192, 1))] * 3))[None]


def _run(plan, ops, models=None, **kw):
    nv._OPS = ops
    args = dict(plan=plan, image=STILL, frames=17, fps=8.0, seed=5, stage1_mp=0.02, context_px=8, feather_px=2,
                output_scale=1.0)
    m = MODELS if models is None else models
    args.update({k: m.get(k) for k in ("model", "clip", "vae", "audio_vae")})
    args.update(kw)
    return nv.KUBA_RegionVideoSampler.execute(**args)


def _blocks(calls):
    steps = [c[0] for c in calls]
    return [s for i, s in enumerate(steps) if i == 0 or s != steps[i - 1]]


def test_lazy_status():
    lz = nv.KUBA_RegionVideoSampler.check_lazy_status
    none = {k: None for k in ("model", "clip", "vae", "audio_vae")}
    assert lz(plan=_plan(RULES0), fps=8.0, **none) == [], "no animated region: no model"
    assert lz(plan=_plan(RULES2), fps=8.0, **none) == ["model", "clip", "vae", "audio_vae"]
    got = lz(plan=_plan(RULES2), fps=8.0, enhance_prompt=True, upscale_model=None, enhance_clip=None, **none)
    assert got == ["model", "clip", "vae", "audio_vae", "upscale_model", "enhance_clip"], got
    assert lz(plan=_plan(RULES2), fps=8.0, enhance_prompt=True, **none) == ["model", "clip", "vae", "audio_vae"], \
        "inputs that are not connected are never requested"
    assert lz(plan=_plan(RULES2), fps=8.0, enhance_prompt=False, enhance_clip=None, **none)[-1] == "audio_vae", \
        "enhance off: the enhancer is not loaded"
    assert lz(plan=_plan(RULES2), fps=8.0, **MODELS) == [], "all loaded: nothing more"
    assert lz(plan=_plan(RULES2), fps=8.0, only="X_*", **none) == [], "'only' matches nothing"
    bg = torch.zeros(9, 192, 320, 3)
    far = RULES2.replace("lamps flicker", "lamps flicker\nt_start = 5").replace("flies past", "flies past\nt_start = 5")
    assert lz(plan=_plan(far), fps=8.0, background=bg, **none) == [], "time range outside the background video"
    assert lz(plan=_plan(far), fps=8.0, **none) != [], "on a still the time range is in the mask"
    gone = _plan(RULES2)
    gone.regions.labels[:] = 3
    assert lz(plan=gone, fps=8.0, **none) == [], "animated regions not in the label map"
    return "models requested only with work; unconnected and loaded inputs never"


def test_pass_through():
    frames, report = _run(_plan(RULES0), _NoOps(), models={}).args
    assert frames.shape == (17, 192, 320, 3) and torch.equal(frames[5], STILL[0]), "the still, F times"
    assert "no animated regions" in report, report
    bg = torch.rand(11, 192, 320, 3)
    frames, report = _run(_plan(RULES0), _NoOps(), models={}, background=bg, output_scale=0.5).args
    assert torch.equal(frames, bg), "the background passes through unchanged (at its own size)"
    return "no animated region: frames through, no model step, no error"


def test_phases_and_cache():
    up = MODELS["upscale_model"]
    nv._CLIP_CACHE.clear()
    ops = _StubOps()
    a = _run(_plan(RULES2), ops, upscale_model=up)
    steps = [c[0] for c in ops.calls]
    assert steps.count("vae_encode") == 2 and steps.count("sample") == 4 and steps.count("decode") == 2, steps
    assert _blocks(ops.calls) == ["vae_encode", "encode_text", "sample", "upsample", "sample", "decode"], \
        f"each model loaded once per phase, not per clip: {_blocks(ops.calls)}"
    frames = a.args[0]
    assert not torch.equal(frames[8], STILL[0]), "the clips are pasted in"
    assert np.array_equal(frames[8].numpy()[100:, :150], STILL[0].numpy()[100:, :150]), "far from the regions: the still"

    ops2 = _StubOps()
    b = _run(_plan(RULES2), ops2, upscale_model=up)
    assert ops2.calls == [] and torch.equal(a.args[0], b.args[0]), "unchanged: every clip from the cache, same frames"
    assert b.args[1].count("from the cache") == 2, b.args[1]

    ops6 = _StubOps()
    d = _run(_plan(RULES2), ops6, upscale_model=up, feather_px=6)
    assert ops6.calls == [] and not torch.equal(d.args[0], a.args[0]), "feather is paste only: from the cache"

    ops3 = _StubOps()
    c = _run(_plan(RULES2.replace("a bird flies past", "a bat flies past")), ops3, upscale_model=up)
    assert [x[0] for x in ops3.calls].count("vae_encode") == 1 and c.args[1].count("from the cache") == 1, \
        "one prompt changed: only that clip renders"

    for name, kw in (("a new seed", {"seed": 6}), ("a reloaded model (new object)", {"model": _Model("model")}),
                     ("no upscaler", {"upscale_model": None}), ("another context", {"context_px": 24})):
        o = _StubOps()
        _run(_plan(RULES2), o, **{"upscale_model": up, **kw})
        assert [x[0] for x in o.calls].count("vae_encode") == 2, f"{name} renders again"

    nv._CLIP_CACHE.clear()
    e = _run(_plan(RULES2), _StubOps(), upscale_model=up)
    assert torch.equal(e.args[0], a.args[0]), "a fresh render gives the same frames as before"

    ops8 = _StubOps()
    f = _run(_plan(RULES2), ops8, enhance_prompt=True, enhance_clip=MODELS["enhance_clip"])
    assert _blocks(ops8.calls) == ["vae_encode", "enhance", "encode_text", "sample", "decode"], _blocks(ops8.calls)
    assert "prompt (enhanced)" in f.args[1]
    ops9 = _StubOps()
    g = _run(_plan(RULES2), ops9, enhance_prompt=True, enhance_clip=MODELS["enhance_clip"])
    assert ops9.calls == [] and "prompt (enhanced)" in g.args[1], "cached together with its enhanced prompt"
    return "encode > text > stage 1 > upsample > stage 2 > decode; hits, misses, invalidation"


def test_loop_and_sound():
    up = MODELS["upscale_model"]
    nv._CLIP_CACHE.clear()
    plain = _run(_plan(RULES2), _StubOps(), upscale_model=up)
    lo = _StubOps()
    looped = _run(_plan(RULES2), lo, upscale_model=up, loop=True)
    assert [c[0] for c in lo.calls].count("vae_encode") == 2 and "loop" in looped.args[1], "loop renders again and is reported"
    assert not torch.equal(plain.args[0], looped.args[0]), "the mask differs: other frames"
    assert torch.equal(looped.args[0][-1], looped.args[0][0]) and not torch.equal(plain.args[0][-1], plain.args[0][0]), \
        "a loop ends on its first frame"
    ramp = np.linspace(0, 1, 33, dtype=np.float16)[:, None, None, None] * np.ones((1, 2, 2, 3), np.float16)
    closed = vd.close_loop(ramp.copy())
    assert np.array_equal(closed[:25], ramp[:25]) and np.array_equal(closed[-1], ramp[0]), "only the tail changes"
    assert np.all(np.diff(closed[24:, 0, 0, 0].astype(np.float32)) < 0), "the tail fades evenly onto frame 0"
    assert np.array_equal(vd.close_loop(ramp[:9].copy()), ramp[:9]), "a clip too short for a tail is left alone"
    bg = torch.rand(17, 192, 320, 3)
    head = _run(_plan(RULES2), _StubOps(), background=bg, loop=True).args[1].split(chr(10))[0]
    assert "loop" not in head, "not into a background video"

    sound = {"waveform": torch.linspace(0, 1, 2 * 8000 * 3).reshape(1, 2, -1), "sample_rate": 8000}
    so = _StubOps()
    a = _run(_plan(RULES2), so, upscale_model=up, audio=sound)
    assert _blocks(so.calls) == ["vae_encode", "encode_text", "sound_audio", "sample", "sound_audio", "sample", "upsample",
                                 "sample", "decode"], _blocks(so.calls)
    assert all("noise_mask" in x for x in so.audio_seen), "both stages get the held sound, not a sampled audio latent"
    assert "with the sound" in a.args[1] and not torch.equal(a.args[0], plain.args[0])
    again = _StubOps()
    _run(_plan(RULES2), again, upscale_model=up, audio=sound)
    assert again.calls == [], "the same sound: from the cache"
    other = _StubOps()
    _run(_plan(RULES2), other, upscale_model=up, audio={"waveform": sound["waveform"] * 0.5, "sample_rate": 8000})
    assert [c[0] for c in other.calls].count("vae_encode") == 2, "another sound renders again"
    short = {"waveform": torch.ones(1, 1, 800), "sample_rate": 8000}                  # 0.1 s for a 2.1 s clip
    assert _run(_plan(RULES2), _StubOps(), audio=short).args[0].shape[0] == 17, "a short sound is padded with silence"
    nv._CLIP_CACHE.clear()
    return "loop holds both ends (still only); a sound is held through both stages and is part of the cache key"


def test_cache_limits():
    c = cc.ClipCache(2, max_bytes=100)
    m = _Model("m")
    for k in "abc":
        c.put(k, (m,), np.zeros(10, np.uint8))
    assert c.get("a", (m,)) is None and c.get("c", (m,)) is not None, "LRU by count"
    c.put("d", (m,), np.zeros(95, np.uint8))
    assert len(c) == 1 and c.get("d", (m,)) is not None, "capped by bytes"
    assert c.get("d", (_Model("m"),)) is None and c.get("d", (m, None)) is None, "other model objects miss"
    assert cc.digest(torch.ones(3)) == cc.digest(torch.ones(3)) != cc.digest(torch.ones(3, dtype=torch.float16))
    assert cc.digest(np.ones((2, 3))) != cc.digest(np.ones((3, 2))), "shape is part of the key"
    assert cc.digest(torch.ones(2, dtype=torch.bool)) != cc.digest(torch.zeros(2, dtype=torch.bool))
    return "count and byte caps, weak model identity, content digest"


def test_parallel_writing():
    rng = np.random.default_rng(3)
    still = rng.random((96, 128, 3), dtype=np.float32)
    mask = np.zeros((96, 128), bool)
    mask[20:50, 30:90] = True
    clips = [((20, 10, 100, 60), rng.random((5, 32, 48, 3)).astype(np.float16), vd.feather_mask(mask, 3), 2)]
    tmp = tempfile.mkdtemp(prefix="kkv_")
    threads = nv._WRITE_THREADS
    try:
        for sub in "ab":
            os.makedirs(os.path.join(tmp, sub))
        nv._WRITE_THREADS = 1
        serial = nv._composite(still, None, clips, 9, 96, 128, 48, 64, os.path.join(tmp, "a"), "f")
        nv._WRITE_THREADS = 8
        par = nv._composite(still, None, clips, 9, 96, 128, 48, 64, os.path.join(tmp, "b"), "f")
        assert np.array_equal(serial, par), "threads give the same frames"
        for i in range(9):
            pa = cv2.imread(os.path.join(tmp, "a", f"f_{i:05d}.png"))
            pb = cv2.imread(os.path.join(tmp, "b", f"f_{i:05d}.png"))
            assert pa is not None and np.array_equal(pa, pb)
        ref = (np.clip(still, 0, 1) * 255 + 0.5).astype(np.uint8)[..., ::-1]
        assert np.array_equal(cv2.imread(os.path.join(tmp, "b", "f_00000.png")), ref), "PNG level 1 is lossless"
    finally:
        nv._WRITE_THREADS = threads
        shutil.rmtree(tmp, ignore_errors=True)
    return "9 frames, 8 threads = serial, PNGs identical"


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    ok = 0
    for t in tests:
        try:
            print(f"PASS {t.__name__}: {t()}")
            ok += 1
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            print(f"FAIL {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{ok}/{len(tests)} passed.")
    sys.exit(0 if ok == len(tests) else 1)
