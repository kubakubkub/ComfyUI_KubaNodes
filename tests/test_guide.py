"""
Model free test for kubakub motion guide: the noise levels for a start level, the LTX size and length of a guide,
the built-in sample, and the node KUBA_MotionGuide with stub models (the steps of one and two stages, the sound, the
sizes of what comes out). Does not start ComfyUI.

Run from the ComfyUI_windows_portable folder:
    python_embeded\\python.exe ComfyUI\\custom_nodes\\ComfyUI_KubaNodes\\tests\\test_guide.py
"""

import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(HERE))))     # ComfyUI (comfy_api)
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)                                    # embedded Python leaves the script folder off the path
import _pack  # noqa: E402,F401  (the pack as a package, see tests/_pack.py)
from kubapack.nodes.motion import nodes_guide as ng  # noqa: E402
from kubapack.nodes.motion import nodes_video as nv  # noqa: E402
from kubakub import video as vd  # noqa: E402
from test_video import MODELS, _StubOps  # noqa: E402


class _Ops(_StubOps):
    """The stub steps, and what each sample call was given."""

    def sample(self, model, guider, seed, sigmas, video, audio, sampler_name="euler_ancestral"):
        self.sampled = getattr(self, "sampled", []) + [(sigmas, tuple(video["samples"].shape), seed)]
        self.masks = getattr(self, "masks", []) + [video.get("noise_mask")]
        video = dict(video, noise_mask=video.get("noise_mask") if video.get("noise_mask") is not None else torch.zeros(1))
        return super().sample(model, guider, seed, sigmas, video, audio, sampler_name)


def _run(ops, **kw):
    nv._OPS = ops
    args = dict(prompt="light sweeps", repaint=0.91, seed=5, fps=8.0, megapixels=0.02)
    args.update({k: MODELS[k] for k in ("model", "clip", "vae", "audio_vae")})
    args.update(kw)
    return ng.KUBA_MotionGuide.execute(**args).args


def test_sigmas():
    assert vd.guide_sigmas(1.0) == list(vd.GUIDE_SIGMAS), "1 = the whole schedule"
    assert vd.guide_sigmas(0.91) == [0.91, 0.725, 0.421875, 0.0], vd.guide_sigmas(0.91)
    assert vd.guide_sigmas(0.95) == [0.95, 0.909375, 0.725, 0.421875, 0.0]
    assert vd.guide_sigmas(0.98) == [0.98, 0.909375, 0.725, 0.421875, 0.0], "no step of 0.005"
    assert vd.guide_sigmas(0.5) == [0.5, 0.421875, 0.0]
    assert vd.guide_sigmas(0.7, vd.GUIDE_SIGMAS_2) == [0.7, 0.4219, 0.0]
    for r in (0.3, 0.6, 0.85, 0.93, 0.999):
        s = vd.guide_sigmas(r)
        assert s[0] == r and s[-1] == 0.0 and all(a > b for a, b in zip(s, s[1:])), s
    return "the start level first, then the schedule's steps below it"


def test_size_and_frames():
    assert vd.guide_size(1920, 1080, 0.2) == (576, 320), vd.guide_size(1920, 1080, 0.2)
    w, h = vd.guide_size(1080, 1920, 0.2)
    assert w % 32 == 0 and h % 32 == 0 and h > w
    clip = np.random.rand(10, 90, 160, 3).astype(np.float32)
    g = vd.guide_frames(clip, 64, 32)
    assert g.shape == (17, 32, 64, 3) and np.array_equal(g[9], g[16]) and not np.array_equal(g[8], g[9]), "padded with the last frame"
    same = vd.guide_frames(clip[:9], 160, 90)
    assert same.shape == (9, 90, 160, 3) and np.array_equal(same, clip[:9]), "a guide that fits is not touched"
    s = vd.sample_guide(160, 96, 9)
    cols = s.mean((1, 3)).argmax(1)
    assert s.shape == (9, 96, 160, 3) and 0 <= s.min() and s.max() <= 1 and cols[0] < 40 and cols[-1] > 120 \
        and all(a <= b for a, b in zip(cols, cols[1:])), cols
    return f"1920x1080 at 0.2 MP -> 576x320; the sample's band runs from x {cols[0]} to {cols[-1]}"


def test_node_one_stage():
    ops = _Ops()
    guide = torch.rand(12, 96, 160, 3)
    frames, fps, audio, report = _run(ops, guide=guide)
    steps = [c[0] for c in ops.calls]
    assert steps == ["vae_encode", "encode_text", "encode_text", "sample", "decode"], steps
    assert frames.shape[0] == 12 and frames.shape[1] % 32 == 0 and frames.shape[2] % 32 == 0 and frames.shape[3] == 3, frames.shape
    assert ops.sampled[0][0] == "0.91, 0.725, 0.421875, 0" and ops.sampled[0][1][2] == 3, ops.sampled      # 17 frames -> 3 latent frames
    assert fps == 8.0 and audio["waveform"].shape[-1] == int(round(12 / 8.0 * 44100)) and float(audio["waveform"].abs().max()) == 0
    assert "12 frames" in report and "one stage" in report and "repaint 0.91" in report and "small moving details" in report, report
    back = _run(_Ops(), guide=guide, guide_size=True)[0]
    assert back.shape == (12, 96, 160, 3), back.shape
    return f"12 frames in, 12 out at {frames.shape[2]}x{frames.shape[1]}; guide_size gives 160x96"


def test_node_two_stages_and_sound():
    ops = _Ops()
    wave = torch.rand(1, 2, 3 * 22050)
    frames, fps, audio, report = _run(ops, guide=torch.rand(9, 64, 128, 3), upscale_model=MODELS["upscale_model"],
                                      audio={"waveform": wave, "sample_rate": 22050}, repaint=0.7, megapixels=0.5, stage2_max_mp=0.08)
    steps = [c[0] for c in ops.calls]
    assert steps == ["vae_encode", "encode_text", "encode_text", "sound_audio", "sample", "upsample", "sample", "decode"], steps
    (s1, shape1, seed1), (s2, shape2, seed2) = ops.sampled
    assert s1 == "0.7, 0.421875, 0" and s2 == "0.7, 0.4219, 0", (s1, s2)
    assert shape2[3] == 2 * shape1[3] and shape2[4] == 2 * shape1[4] and seed2 == seed1 + 1000
    assert all("noise_mask" in a for a in ops.audio_seen), "the sound is held in both stages"
    assert shape1[3] * shape1[4] * 1024 <= 0.03e6, "the first stage shrank to fit the cap of the second"
    assert audio["sample_rate"] == 22050 and audio["waveform"].shape[-1] == int(round(9 / 8.0 * 22050))
    assert "two stages" in report and "stage2_max_mp" in report and "own look mostly stays" in report, report
    return f"stage 1 {shape1[4] * 32}x{shape1[3] * 32}, stage 2 twice that, the sound held"


def test_node_sample_and_limits():
    ops = _Ops()
    frames, _, _, report = _run(ops)
    assert frames.shape[0] == 49 and "built-in sample" in report, report
    frames, _, _, report = _run(_Ops(), guide=torch.rand(30, 64, 96, 3), max_frames=17, repaint=1.0)
    assert frames.shape[0] == 17 and "cut to max_frames = 17" in report and "guide is ignored" in report, report
    _, _, _, report = _run(_Ops(), guide=torch.rand(1, 64, 96, 3), repaint=0.97)
    assert "one picture" in report and "lower repaint towards 0.91" in report, report
    return "no guide = the sample (49 frames); a long guide is cut; the report says what to try"


def test_repaint_where_and_when():
    assert vd.repaint_latent(None, None, 17, 3, 6, 4) is None, "nothing set: no mask, everything is repainted"
    half = np.zeros((1, 64, 96), np.float32)
    half[:, :, 48:] = 1.0
    m = vd.repaint_latent(half, None, 17, 3, 6, 4)
    assert m.shape == (1, 1, 3, 4, 6) and m[0, 0, :, :, :3].max() == 0 and m[0, 0, :, :, 3:].min() == 1, "white = repaint, black = keep"
    ramp = [i / 16 for i in range(17)]
    t = vd.repaint_latent(None, ramp, 17, 3, 6, 4)
    assert t[0, 0, 0].max() == 0 and abs(float(t[0, 0, 1, 0, 0]) - np.mean(ramp[1:9])) < 1e-6 and abs(float(t[0, 0, 2, 0, 0]) - np.mean(ramp[9:17])) < 1e-6, \
        "a latent frame is the mean of its 8 frames"
    both = vd.repaint_latent(half, ramp, 17, 3, 6, 4)
    assert both[0, 0, 2, 0, 0] == 0 and abs(float(both[0, 0, 2, 0, 5]) - np.mean(ramp[9:17])) < 1e-6, "mask times curve"
    moving = np.zeros((17, 64, 96), np.float32)
    moving[9:] = 1.0
    mv = vd.repaint_latent(moving, None, 17, 3, 6, 4)
    assert mv[0, 0, 1].max() == 0 and mv[0, 0, 2].min() == 1, "a mask per frame: its own time"
    felt = vd.repaint_level(np.array([0.0, 0.25, 1.0], np.float32), 0.91)
    assert felt[0] == 0 and abs(float(felt[2]) - 1) < 1e-6 and 0.70 / 0.91 < float(felt[1]) < 1, \
        "as LTX reads it: nothing stays nothing, all stays all, anything between lies where LTX changes things"
    assert abs(vd.repaint_share(half, None, 17) - 0.5) < 1e-6 and abs(vd.repaint_share(None, ramp, 17) - 0.5) < 1e-6 \
        and abs(vd.repaint_share(half, ramp, 17) - 0.25) < 1e-6
    assert vd.repaint_alpha(None, ramp, 0, 17, 8, 8) == 0.0 and vd.repaint_alpha(None, ramp, 8, 17, 8, 8) == 1.0
    a = vd.repaint_alpha(half, None, 3, 17, 96, 64)
    assert a.shape == (64, 96) and a[:, :40].max() == 0 and a[:, 56:].min() == 1

    guide = torch.rand(17, 64, 96, 3)
    mask = torch.from_numpy(half)
    LIMIT = dict(mask_does="limits the change")
    # limits the change: the noise mask, and the guide's own pixels back where nothing may change
    ops = _Ops()
    frames, _, _, report = _run(ops, guide=guide, repaint_mask=mask, megapixels=0.006, **LIMIT)      # 96x64: the guide's own size
    assert frames.shape == (17, 64, 96, 3), frames.shape
    lm = ops.masks[0]
    assert lm is not None and tuple(lm.shape) == (1, 1, 3, 2, 3) and float(lm[0, 0, :, :, 0].max()) == 0 and float(lm[0, 0, :, :, 2].min()) == 1
    assert torch.allclose(frames[:, :, :40], guide[:, :, :40]), "black in the mask: the guide's own pixels, exactly"
    assert not torch.allclose(frames[:, :, 60:], guide[:, :, 60:]), "white: LTX's picture"
    assert "repaint mask: 1 frame(s)" in report and "may change 50 % of the clip" in report and "own pixels" in report, report
    loose = _run(_Ops(), guide=guide, repaint_mask=mask, megapixels=0.006, keep_exact=False, **LIMIT)[0]
    assert not torch.allclose(loose[:, :, :40], guide[:, :, :40]), "keep_exact off: the decoded picture everywhere"
    ops = _Ops()
    frames, _, _, report = _run(ops, guide=guide, curve="ramp", cycle=17 / 8.0, megapixels=0.006, upscale_model=MODELS["upscale_model"], stage2_max_mp=0.03, **LIMIT)
    assert len(ops.masks) == 2 and ops.masks[1].shape[-1] == 2 * ops.masks[0].shape[-1], "both stages get the mask, at their size"
    assert float(ops.masks[0][0, 0, 0].max()) == 0 and float(ops.masks[0][0, 0, 2].min()) > 0.5, ops.masks[0][0, 0, :, 0, 0]

    # shows the repaint (the default): everything is repainted, no noise mask, and shown through the mask and the curve
    ops, whole = _Ops(), _Ops()
    frames, _, _, report = _run(ops, guide=guide, repaint_mask=mask, megapixels=0.006)
    full = _run(whole, guide=guide, megapixels=0.006)[0]
    assert ops.masks == [None] and whole.masks == [None], "no noise mask: LTX repaints the whole clip"
    assert torch.allclose(frames[:, :, :40], guide[:, :, :40]) and torch.allclose(frames[:, :, 56:], full[:, :, 56:]), "the repaint through the mask, the guide outside"
    assert "the repainted clip shows on 50 % of the clip" in report, report
    grey = _run(_Ops(), guide=guide, repaint_mask=torch.full((1, 64, 96), 0.5), megapixels=0.006)[0]
    assert torch.allclose(grey, guide * 0.5 + full * 0.5, atol=1e-5), "grey = half of each"
    one, _, _, report = _run(_Ops(), guide=guide, curve="ramp", cycle=2.0, megapixels=0.006)
    assert torch.allclose(one[0], guide[0]) and torch.allclose(one[8], guide[8] * 0.5 + full[8] * 0.5, atol=1e-5) and torch.allclose(one[16], full[16]),         "a ramp: the guide at the start, the repaint at the end"
    assert "curve ramp" in report and "no repaint mask (everywhere)" in report, report
    _, _, _, report = _run(_Ops(), guide=guide, curve="beats", cycle=0.2, megapixels=0.006)
    assert "built-in beat" in report and "triggers" in report, report
    _, _, _, report = _run(_Ops(), guide=guide, repaint_mask=torch.zeros(1, 64, 96), megapixels=0.006)
    assert "almost nothing of the repaint shows" in report, report
    return "shows the repaint through mask and curve; or limits the change with the noise mask, both stages"


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
