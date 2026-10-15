"""
test_video_models.py: the video nodes take any LTX or MiniMax H3 variant (kubakub/video_models.py), no model loaded.

Run:  python_embeded\\python.exe custom_nodes\\ComfyUI_KubaNodes\\tests\\test_video_models.py
"""

import os
import sys
import types

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))   # ComfyUI root

import _pack  # noqa: E402,F401
from kubapack.kubakub import video_models as vm  # noqa: E402
from kubapack.nodes.motion import nodes_video as nv  # noqa: E402
from test_video import MODELS, RULES2, _Model, _plan, _run, _StubOps  # noqa: E402


def _files(*names):
    return [(n.lower(), None) for n in names]


def test_profiles():
    p = vm.profile("h3", _files("minimax_h3_fl2va_pruned_w4a8_mixed.safetensors",
                                "minimax_h3_taomate_fl2va_3step_avg_rank_19_bf16.safetensors"))
    assert p["task"] == "fl2va" and p["sigmas"] == [1.0, 0.961165, 0.853333, 0.0] and p["steps"] == 3, p
    assert p["sampler"] == "euler" and "TaoMate" in p["label"]
    p = vm.profile("h3", _files("minimax_h3_fl2va_pruned_w4a8_mixed.safetensors",
                                "minimax_h3_4step_lora_flashgen_v1.0_768p_fl2va_pruned_avg_rank_13_bf16.safetensors"))
    assert p["sigmas"] == [1.0, 0.965517, 0.888889, 0.679245, 0.0], p
    p = vm.profile("h3", _files("minimax_h3_ref2va_pruned_w4a8_mixed.safetensors",
                                "MiniMax-H3-Ref2VA-Acc-8Step_pruned_comfy.safetensors"))
    assert p["task"] == "ref2va" and p["steps"] == 8 and p["scheduler"] == "simple" and p["sigmas"] is None, p
    p = vm.profile("h3", _files("fastvideo_fasth3_8step_v2_pruned_int8_convrot.safetensors"))
    assert p["steps"] == 8, "a distilled H3 named by its steps"
    p = vm.profile("h3", _files("minimax_h3_fl2va_pruned_w4a8_mixed.safetensors"))
    assert p["steps"] == 20 and p["sampler"] == "res_multistep" and "full steps" in p["label"], p
    p = vm.profile("h3", _files("x_flashgen.safetensors"), steps=6)
    assert p["steps"] == 6 and p["sigmas"] is None, "other steps than the LoRA's own sigmas: its scheduler"
    p = vm.profile("h3", _files("x_flashgen.safetensors"), schedule=vm.SCHEDULES[2])
    assert p["steps"] == 20, "schedule = full steps wins over the names"
    p = vm.profile("h3", _files("my_own_model.gguf"), schedule=vm.SCHEDULES[1])
    assert p["steps"] == 4, "fast without a name: 4 steps"
    # LTX
    p = vm.profile("ltx", _files("ltx-2.5-22b-distilled-transformer-nvfp4-comfy-v2.safetensors"))
    assert p["sigmas"] == list(vm.LTX_DISTILLED) and p["cfg"] == 1.0 and p["sampler"] == "euler_ancestral", p
    p = vm.profile("ltx", _files("ltx-2.3-22b-dev_transformer_only_fp8_scaled.safetensors"))
    assert p["scheduler"] == "ltxv" and p["steps"] == 20 and p["cfg"] == 4.0, "a dev model: full steps with cfg"
    p = vm.profile("ltx", _files("ltx-2.3-22b-dev_transformer_only_fp8_scaled.safetensors",
                                 "ltx-2.3-22b-distilled-lora-dynamic_fro09_avg_rank_105_bf16.safetensors"))
    assert p["sigmas"] == list(vm.LTX_DISTILLED), "dev + the distilled LoRA = distilled"
    p = vm.profile("ltx", _files("some-ltx.gguf"))
    assert p["sigmas"] and "taken as distilled" in p["why"], p
    p = vm.profile("ltxv", _files("ltxv-13b-0.9.8-distilled.safetensors"))
    assert "video only" in p["label"]
    assert vm.describe(p).startswith("model: LTX video only")
    return "profiles: TaoMate / FlashGen sigmas, n-step LoRAs, raw H3, ref2va, dev / distilled LTX, overrides"


def test_upstream_files():
    prompt = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": "H3\\minimax_h3_fl2va_pruned_w4a8_mixed.safetensors",
                                                    "weight_dtype": "default"}},
        "2": {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["1", 0], "strength_model": 0.9,
                                                             "lora_name": "H3/minimax_h3_taomate_fl2va_3step_avg_rank_19_bf16.safetensors"}},
        "3": {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["2", 0], "strength_model": 0.0,
                                                             "lora_name": "H3/minimax_h3_DMAD_4step_full_lora_avg_rank_39_bf16.safetensors"}},
        "4": {"class_type": "Power Lora Loader (rgthree)", "inputs": {"model": ["3", 0],
              "lora_1": {"on": True, "lora": "style.safetensors", "strength": 0.5},
              "lora_2": {"on": False, "lora": "off.safetensors", "strength": 1.0}}},
        "9": {"class_type": "KUBA_RegionVideoSampler", "inputs": {"model": ["4", 0], "clip": ["5", 0]}},
    }
    files = vm.upstream_files(prompt, "9")
    names = [f for f, _ in files]
    assert names[0] == "minimax_h3_fl2va_pruned_w4a8_mixed.safetensors", "the loader leads"
    assert "minimax_h3_taomate_fl2va_3step_avg_rank_19_bf16.safetensors" in names
    assert not any("dmad" in f for f in names), "a LoRA at strength 0 does not count"
    assert "style.safetensors" in names and "off.safetensors" not in names, "LoRA stacks: only the ones that are on"
    assert vm.profile("h3", files)["steps"] == 3
    assert vm.upstream_files(None, "9") == [] and vm.upstream_files(prompt, None) == []
    loop = {"1": {"inputs": {"model": ["2", 0]}}, "2": {"inputs": {"model": ["1", 0]}}}
    assert vm.upstream_files(loop, "1") == [], "a cycle ends"
    return "file names walked up the model chain (loader first, LoRAs, stacks, strength 0 left out)"


def test_family_and_checks():
    LTXAV = type("LTXAV", (), {})
    MiniMaxH3 = type("MiniMaxH3", (), {})
    assert vm.family(types.SimpleNamespace(model=LTXAV())) == "ltx"
    assert vm.family(types.SimpleNamespace(model=MiniMaxH3())) == "h3"
    assert vm.family(types.SimpleNamespace(model=object())) is None and vm.family(None) is None
    h3vae = types.SimpleNamespace(first_stage_model=type("MiniMaxH3VideoVAE", (), {"__module__": "comfy.ldm.minimax.vae"})())
    ltxvae = types.SimpleNamespace(first_stage_model=type("VideoVAE", (), {"__module__": "comfy.ldm.lightricks.vae.causal_video_autoencoder"})())
    gemma = types.SimpleNamespace(cond_stage_model=type("LTXAVTEModel_", (), {"__module__": "comfy.text_encoders.lt"})())
    assert vm.part_family(h3vae) == "h3" and vm.part_family(ltxvae) == "ltx" and vm.part_family(gemma) == "ltx"
    vm.check_set("ltx", vae=ltxvae, clip=gemma)
    try:
        vm.check_set("h3", vae=ltxvae)
        raise AssertionError("an LTX VAE with an H3 model must stop")
    except ValueError as e:
        assert "MiniMax H3" in str(e) and "vae" in str(e)
    return "family from the model class, VAE / encoder from their modules, a mixed set stops with a plain message"


def test_frames():
    assert [vm.h3_frames(n) for n in (1, 5, 6, 22, 23, 124)] == [5, 5, 22, 22, 39, 124]
    assert vm.h3_latent_frames(124) == 37 and vm.h3_audio_frames(124) == round(124 / 24 * 40)
    assert vm.model_frames("h3", 2.0, 8.0) == (56, 24.0) and vm.model_frames("ltx", 2.0, 8.0) == (17, 8.0)
    assert vm.time_index(5, 8.0, 100, 24.0) == [0, 3, 6, 9, 12] and vm.time_index(3, 24.0, 2, 8.0) == [0, 0, 1]
    return "H3 17k + 5 at 24 fps, LTX 8k + 1 at the node's fps, frames mapped by time"


class _H3Stub(_StubOps):
    """The region sampler's steps for H3: records the frames each step got."""

    def vae_encode(self, vae, pixels):
        self.calls.append(("vae_encode", vae.name))
        f, h, w = pixels.shape[:3]
        self.encoded = f
        lat_t = vm.h3_latent_frames(f)
        return {"samples": torch.full((1, 24, lat_t, h // 16, w // 16), float(pixels.mean()))}

    def condition(self, clip, vae, text, first, last, frames, task=None):
        self.cond = (first is not None, last is not None, frames, task)
        return super().condition(clip, vae, text, first, last, frames, task)

    def decode(self, vae, video):
        self.calls.append(("decode", vae.name))
        s = video["samples"]
        f = (s.shape[2] - 2) // 5 * 17 + 5
        ramp = torch.linspace(0, 1, f)[:, None, None, None]
        return (0.3 + 0.4 * ramp).expand(f, s.shape[3] * 16, s.shape[4] * 16, 3).clone()


def test_region_sampler_h3():
    h3 = {k: _Model(k) for k in MODELS}
    h3["model"].video_family = "h3"
    ops = _H3Stub()
    nv._H3_OPS = ops
    try:
        nv._CLIP_CACHE.clear()
        out = _run(_plan(RULES2), _StubOps(), models=h3, upscale_model=_Model("upscale_model"), loop=True).args
        frames, report = out[0], out[1]
        assert frames.shape[0] == 17, "the node's frames, whatever the model's grid"
        assert ops.encoded == vm.h3_frames(int(np.ceil(16 / 8.0 * 24)) + 1), "H3 renders 2 s at 24 fps (17k + 5)"
        assert ops.cond[:2] == (True, True) and ops.cond[3] == "fl2va", "fl2va: the still as first and (loop) last frame"
        assert "MiniMax H3" in report and "no latent upscaler" in report and "one stage" in report, report
        assert not any(c[0] == "upsample" for c in ops.calls)
        assert "'negative' is not used at cfg 1" in report
        sig, smp = ops.sigmas_seen[0]
        assert smp == "res_multistep" and sig.count(",") == 20, "a raw H3: 20 steps of res_multistep"
        assert torch.allclose(frames[0], frames[-1], atol=1e-3), "loop: the last frame is the first"
    finally:
        nv._H3_OPS = nv._H3Ops()
        nv._CLIP_CACHE.clear()
    return "region video sampler on H3: 24 fps grid mapped to the node's frames, keyframes, no upscaler, its schedule"


def test_widgets():
    names = {i.id for i in nv.KUBA_RegionVideoSampler.define_schema().inputs}
    assert {"schedule", "steps"} <= names
    sch = nv.KUBA_RegionVideoSampler.define_schema()
    assert sch.display_name == "kubakub region video sampler" and "h3" in sch.search_aliases
    aud = next(i for i in sch.inputs if i.id == "audio_vae")
    assert aud.optional, "a video-only LTX has no audio VAE"
    return "schedule / steps widgets, neutral name, audio VAE optional"


if __name__ == "__main__":
    for t in (test_profiles, test_upstream_files, test_family_and_checks, test_frames, test_region_sampler_h3,
              test_widgets):
        print("ok  ", t())
    print("\nall video model tests passed")
