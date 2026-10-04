"""
schedules.py

Fixed few-step sigma schedules for distilled models and LoRAs, with partial
denoise for inpainting. Pure torch (tests/test_schedules.py).

qwen21_turbo_5: Viggle Qwen-Image-2.1-viggle-turbo v0.2 (5-step LoRA, DMD).
Source: huggingface.co/Viggle/Qwen-Image-2.1-viggle-turbo (README: sigmas
[1.0, 0.875, 0.75, 0.5, 0.25]; scheduler_config.json: dynamic exponential shift,
base_shift 0.5 at 256 tokens, max_shift 0.9 at 8192 tokens).

pruna_qwen21_8: Pruna Qwen-Image-2.1 8-step LoRA v0.1 (p_qwen_image_2.1_8step_v0.1).
Source: huggingface.co/PrunaAI/Pruna-Qwen-Image-2.1 (model card: sigmas
[1.0, 14/15, 6/7, 10/13, 2/3, 6/11, 0.4, 2/9], shift 1.0, no dynamic shifting,
LoRA strength 1.0, no CFG, trained at 1K). These are shift 2 applied to the
8 linear timesteps k/8, so they do not depend on the image size.
"""

from __future__ import annotations

import math

import torch

SCHEDULES = ("auto", "qwen21_turbo_5", "pruna_qwen21_8")

_QWEN21_TURBO_T = (1.0, 0.875, 0.75, 0.5, 0.25)
_PRUNA8_T = tuple(k / 8 for k in range(8, 0, -1))
_PRUNA8_SHIFT = 2.0


def qwen21_mu(width: int, height: int) -> float:
    """Dynamic shift of Qwen Image 2.1 (diffusers FlowMatchEuler, linear in the token count)."""
    seq = width * height / (16 * 16)              # 16x VAE, patch 1
    base_seq, max_seq, base_shift, max_shift = 256, 8192, 0.5, 0.9
    m = (max_shift - base_shift) / (max_seq - base_seq)
    return base_shift + m * (seq - base_seq)


def exp_shift(t: float, mu: float) -> float:
    if t <= 0:
        return 0.0
    return math.exp(mu) / (math.exp(mu) + (1.0 / t - 1.0))


def qwen21_turbo_sigmas(denoise: float, width: int, height: int) -> torch.Tensor:
    """
    The five trained timesteps, shifted for this size, ending in 0. With
    denoise < 1 the run starts at the first trained timestep <= denoise (the
    student only knows these), so 0.75 -> 3 steps, 0.5 -> 2, 0.3 -> 1; below the
    last timestep it starts at denoise itself.
    """
    if denoise <= 0:
        return torch.zeros(1)
    ts = [t for t in _QWEN21_TURBO_T if t <= denoise + 1e-6] or [float(denoise)]
    mu = qwen21_mu(width, height)
    return torch.tensor([exp_shift(t, mu) for t in ts] + [0.0], dtype=torch.float32)


def flow_shift(t: float, shift: float) -> float:
    """Constant flow-matching shift (diffusers FlowMatchEuler, shift != 1)."""
    return shift * t / (1.0 + (shift - 1.0) * t)


def pruna_qwen21_8_sigmas(denoise: float) -> torch.Tensor:
    """
    The eight trained sigmas of the Pruna LoRA, ending in 0. denoise is read on
    the unshifted timestep axis like qwen21_turbo_sigmas, so both LoRAs start at
    about the same noise level for the same denoise (0.75 -> 6/7, 6 steps;
    0.875 -> 14/15, 7 steps). Below the last timestep it starts at denoise itself.
    """
    if denoise <= 0:
        return torch.zeros(1)
    ts = [t for t in _PRUNA8_T if t <= denoise + 1e-6] or [float(denoise)]
    return torch.tensor([flow_shift(t, _PRUNA8_SHIFT) for t in ts] + [0.0], dtype=torch.float32)
