"""
adapters.py

Model adapters of Kuba Regions. The
strategies only talk to this interface, so the same plan can drive Flux 2
Klein now and LTX / MiniMax H3 later. One adapter wraps one MODEL + CLIP + VAE.

Uses ComfyUI (comfy.sample, node_helpers); not imported by the model free tests.
"""

from __future__ import annotations

import logging
import math

import torch

import comfy.model_base
import comfy.model_management
import comfy.sample
import comfy.samplers
import node_helpers

from . import geometry as geo
from . import region_cache as rc
from . import schedules

log = logging.getLogger("KUBA.regions")


class Adapter:
    """Generic latent model: core KSampler path, no reference latents."""
    family = "generic"
    default_sampler = "euler"
    default_scheduler = "normal"
    default_steps = 20
    default_cfg = 5.0
    default_region_mp = 1.0
    schedule_names = ("auto",)      # fixed schedules this family understands

    def __init__(self, model, clip, vae):
        self.model, self.clip, self.vae = model, clip, vae
        self._conds: dict[str, list] = {}
        self.schedule = "auto"
        self.core_scheduler = ""    # a core scheduler name (simple, beta ...) instead of the family's own schedule

    def set_schedule(self, name: str) -> None:
        if name not in self.schedule_names:
            log.warning("[KUBA regions] schedule %s is not for %s; using auto", name, self.family)
            name = "auto"
        self.schedule = name

    @classmethod
    def detect(cls, model) -> bool:
        return True

    # ---- geometry --------------------------------------------------------
    @property
    def grid(self) -> int:
        """Pixels per model token: the latent downscale times the DiT patch (16 for Flux 2)."""
        try:
            down = int(self.model.get_model_object("latent_format").spacial_downscale_ratio)
        except Exception:
            down = 8
        patch = 1
        try:
            dm = self.model.get_model_object("diffusion_model")
            p = getattr(dm, "patch_size", 1)
            patch = int(p[-1] if isinstance(p, (tuple, list)) else p)
        except Exception:
            pass
        return max(1, down * patch)

    @property
    def geometry(self) -> geo.Geometry:
        return geo.GEOMETRIES.get(self.family, geo.GEOMETRIES["sd_8"])

    # ---- text ------------------------------------------------------------
    def _encode_text(self, text: str) -> list:
        tokens = self.clip.tokenize(text)
        return self.clip.encode_from_tokens_scheduled(tokens)

    def encode_prompts(self, texts) -> None:
        """Encode every distinct prompt once, up front, so the text encoder loads once. Encodings are kept
        across queues (region_cache.CONDS, per text encoder and text), so a queue encodes only new prompts."""
        todo = [t for t in dict.fromkeys(texts) if t not in self._conds]
        n = 0
        for t in todo:
            def enc(t=t):
                nonlocal n
                n += 1
                return self._encode_text(t)
            self._conds[t] = rc.cached_cond(self.clip, f"text:{type(self).__name__}", t, None, enc)
        if todo:
            log.info("[KUBA regions] encoded %d prompt(s)%s", n,
                     f", {len(todo) - n} from cache" if len(todo) > n else "")

    def cond(self, text: str) -> list:
        if text not in self._conds:
            self.encode_prompts([text])
        return self._conds[text]

    # ---- latents ---------------------------------------------------------
    def encode(self, pixels: torch.Tensor) -> torch.Tensor:
        return self.vae.encode(pixels[..., :3])

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        img = self.vae.decode(latent)
        if img.ndim == 5:                       # video VAEs: [B, T, H, W, C]
            img = img.reshape(-1, *img.shape[-3:])
        return img

    def empty_latent(self, width: int, height: int) -> torch.Tensor:
        """Zero latent for a text-to-image run, shaped by the model's latent format."""
        lf = self.model.get_model_object("latent_format")
        down = int(lf.spacial_downscale_ratio)
        return torch.zeros([1, int(lf.latent_channels), height // down, width // down],
                           device=comfy.model_management.intermediate_device())

    def add_reference(self, cond: list, latent: torch.Tensor) -> list:
        return cond

    prefetches = False                 # True: encodes each region's crop image with its text (encode them all first)

    def prefetch(self, key, prompt: str, negative: str, pixels: torch.Tensor, need_negative: bool = True) -> None:
        pass

    def conds_for(self, prompt: str, negative: str, pixels: torch.Tensor | None = None,
                  latent: torch.Tensor | None = None, key=None) -> tuple[list, list]:
        """(positive, negative) for one sample; with pixels/latent the crop is the reference."""
        pos, neg = self.cond(prompt), self.cond(negative)
        if latent is not None:
            pos = self.add_reference(pos, latent)
        return pos, neg

    def sigmas(self, steps: int, denoise: float, width: int, height: int):
        return None

    # ---- sampling --------------------------------------------------------
    def sample(self, latent: torch.Tensor, positive: list, negative: list, *, noise_mask=None,
               seed: int = 0, steps: int = 0, cfg: float = 0.0, denoise: float = 1.0,
               sampler_name: str = "", scheduler: str = "", pixel_size=(0, 0)) -> torch.Tensor:
        steps = steps or self.default_steps
        cfg = cfg or self.default_cfg
        sampler_name = sampler_name or self.default_sampler
        scheduler = self.core_scheduler or scheduler or self.default_scheduler
        latent = comfy.sample.fix_empty_latent_channels(self.model, latent)
        noise = comfy.sample.prepare_noise(latent, seed)
        # core_scheduler: the KSampler path with that scheduler, not the family's own sigmas (Flux2Scheduler ...)
        sig = None if self.core_scheduler else self.sigmas(steps, denoise, *pixel_size)
        if sig is not None:
            if sig.numel() < 2:
                return latent
            return comfy.sample.sample_custom(
                self.model, noise, cfg, comfy.samplers.sampler_object(sampler_name), sig,
                positive, negative, latent, noise_mask=noise_mask, seed=seed)
        return comfy.sample.sample(self.model, noise, steps, cfg, sampler_name, scheduler, positive,
                                   negative, latent, denoise=denoise, noise_mask=noise_mask, seed=seed)


# Flux 2 schedule, copied from ComfyUI core comfy_extras/nodes_flux.py
# (github.com/comfyanonymous/ComfyUI, Comfy Org) so a core refactor cannot break it.
def _flux2_mu(image_seq_len: int, num_steps: int) -> float:
    a1, b1 = 8.73809524e-05, 1.89833333
    a2, b2 = 0.00016927, 0.45666666
    if image_seq_len > 4300:
        return float(a2 * image_seq_len + b2)
    m_200 = a2 * image_seq_len + b2
    m_10 = a1 * image_seq_len + b1
    a = (m_200 - m_10) / 190.0
    b = m_200 - 200.0 * a
    return float(a * num_steps + b)


def flux2_schedule(num_steps: int, image_seq_len: int) -> torch.Tensor:
    mu = _flux2_mu(image_seq_len, num_steps)
    t = torch.linspace(1, 0, num_steps + 1)
    return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** 1.0)   # t = 0 gives inf -> sigma 0


class Flux2Adapter(Adapter):
    """Flux 2 (Klein 4B / 9B, dev): reference latents, Flux2Scheduler sigmas, distilled defaults."""
    family = "flux2"
    default_sampler = "euler"
    default_scheduler = "simple"
    default_steps = 4
    default_cfg = 1.0
    default_region_mp = 1.0

    @classmethod
    def detect(cls, model) -> bool:
        return isinstance(getattr(model, "model", None), comfy.model_base.Flux2)

    def add_reference(self, cond: list, latent: torch.Tensor) -> list:
        # As core ReferenceLatent: the crop's own latent as an extra reference token set.
        return node_helpers.conditioning_set_values(cond, {"reference_latents": [latent]}, append=True)

    def sigmas(self, steps: int, denoise: float, width: int, height: int):
        """Core Flux2Scheduler for the crop size; denoise as in KSampler (longer schedule, keep the tail)."""
        seq = round(width * height / (16 * 16))
        if denoise <= 0:
            return torch.zeros(1)
        total = steps if denoise >= 0.9999 else max(steps, int(steps / denoise))
        return flux2_schedule(total, seq)[-(steps + 1):]


class QwenImage21Adapter(Adapter):
    """
    Qwen Image 2.1 (core QwenImage21): the reference crop goes through the
    Qwen3-VL vision tower and in as a VAE latent, as core TextEncodeQwenImage21
    does, so every region with a reference is encoded on its own (the text
    encoder and the DiT take turns on the GPU). Grid 32: vision slots cover 2x2
    latents and an edit's output must match its reference size.
    """
    family = "qwen_image21"
    default_sampler = "euler"
    default_scheduler = "simple"
    default_steps = 25
    default_cfg = 1.0
    default_region_mp = 1.0
    schedule_names = ("auto", "qwen21_turbo_5", "pruna_qwen21_8")

    @classmethod
    def detect(cls, model) -> bool:
        klass = getattr(comfy.model_base, "QwenImage21", None)
        return klass is not None and isinstance(getattr(model, "model", None), klass)

    def __init__(self, model, clip, vae):
        super().__init__(model, clip, vae)
        self._prefetched = {}
        opts = model.model_options.get("transformer_options", {}).get("qwen_image21_cache")
        if not opts or opts.get("device", "auto") == "auto":
            # On this machine the auto cache (pinned RAM) aborted ComfyUI at sampling
            # cleanup (core 0.37.0, see ENVIRONMENT.md "Known runtime issues").
            log.warning("[KUBA regions] Qwen Image 2.1 prefix cache is on auto. If ComfyUI aborts at "
                        "step 0, add core 'Qwen Image 2.1 Cache' before this node with device = gpu "
                        "(dtype int8 halves its VRAM).")

    @property
    def grid(self) -> int:
        return 32

    def sigmas(self, steps: int, denoise: float, width: int, height: int):
        """Turbo LoRA schedules: the LoRA's trained timesteps (steps is ignored)."""
        if self.schedule == "qwen21_turbo_5":
            return schedules.qwen21_turbo_sigmas(denoise, width, height)
        if self.schedule == "pruna_qwen21_8":
            return schedules.pruna_qwen21_8_sigmas(denoise)
        return None

    def _encode(self, text: str, images: list) -> list:
        tokens = self.clip.tokenize(text, images=images, keep_vision=len(images) == 0,
                                    prevent_empty_text=True)
        return self.clip.encode_from_tokens_scheduled(tokens)

    def _encode_text(self, text: str) -> list:
        return self._encode(text, [])

    def _encode_image(self, text: str, rgb: torch.Tensor) -> list:
        """Text + vision encoding of one crop, kept across queues (per text encoder, text and crop pixels)."""
        return rc.cached_cond(self.clip, f"vision:{type(self).__name__}", text, rgb,
                              lambda: self._encode(text, [rgb]))

    prefetches = True

    def prefetch(self, key, prompt, negative, pixels, need_negative=True) -> None:
        """Text + vision encoding of one region's crop ahead of sampling: run_plan encodes every region first,
        so the 7 GB text encoder loads once instead of taking turns with the DiT for every region. At cfg 1
        (turbo LoRAs) the negative is never evaluated: its text-only encoding (cached) stands in."""
        rgb = pixels[:1, ..., :3]
        neg = self._encode_image(negative, rgb) if need_negative else self.cond(negative)
        self._prefetched[key] = (self._encode_image(prompt, rgb), neg)

    def conds_for(self, prompt, negative, pixels=None, latent=None, key=None):
        if pixels is None or latent is None:
            return self.cond(prompt), self.cond(negative)
        if key is not None and key in self._prefetched:
            pos, neg = self._prefetched.pop(key)
        else:
            rgb = pixels[:1, ..., :3]
            pos = self._encode_image(prompt, rgb)
            neg = self._encode_image(negative, rgb)
        ref = {"reference_latents": [latent]}
        return (node_helpers.conditioning_set_values(pos, ref, append=True),
                node_helpers.conditioning_set_values(neg, ref, append=True))

    def add_reference(self, cond: list, latent: torch.Tensor) -> list:
        # a further reference (the style image of kubakub versions): as a latent only, without vision tokens
        return node_helpers.conditioning_set_values(cond, {"reference_latents": [latent]}, append=True)


ADAPTERS = [Flux2Adapter, QwenImage21Adapter]


def make_adapter(model, clip, vae, family: str = "auto") -> Adapter:
    if family != "auto":
        for a in ADAPTERS + [Adapter]:
            if a.family == family:
                return a(model, clip, vae)
        raise ValueError(f"unknown adapter family '{family}'")
    for a in ADAPTERS:
        if a.detect(model):
            return a(model, clip, vae)
    log.warning("[KUBA regions] no adapter recognises %s; using the generic one",
                type(getattr(model, "model", model)).__name__)
    return Adapter(model, clip, vae)


def families() -> list[str]:
    return ["auto"] + [a.family for a in ADAPTERS] + ["generic"]
