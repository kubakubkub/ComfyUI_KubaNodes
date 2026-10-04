r"""
TrimToExactDuration  --  drop-in replacement for the (wrap-127 + TrimAudioDuration) pair.

Why this exists
---------------
In LTX-2 AV workflows the audio latent is concatenated with the video latent and the
transformer's guide_mask computes guide_start from the VIDEO token count. If the audio
block ever changes length, the combined sequence no longer matches guide_start and you
get the classic:

    RuntimeError: The expanded size of the tensor (A) must match the existing size (B)
    at non-singleton dimension 1   (inside _attention_with_guide_mask)

The usual culprit is a start-index that wraps near the tail of the track, so the trim
window overruns and returns a SHORTER clip (or throws "start must be less than end").
This node removes that failure mode by construction: it always emits EXACTLY the same
number of samples for a given duration, whatever the start position is.
"""

import torch


def _as_bcn(waveform: torch.Tensor):
    """Normalise a ComfyUI AUDIO waveform to shape [B, C, N]."""
    if waveform.dim() == 1:          # [N]
        waveform = waveform.unsqueeze(0).unsqueeze(0)
    elif waveform.dim() == 2:        # [C, N]
        waveform = waveform.unsqueeze(0)
    elif waveform.dim() > 3:         # squeeze stray leading dims
        waveform = waveform.reshape(-1, waveform.shape[-2], waveform.shape[-1])
    return waveform


class TrimToExactDuration:
    """
    Returns audio of a guaranteed constant sample count.

    start_seconds   : where to start reading (e.g. your raw cumulative-seconds value).
    duration_seconds: the clip length you always want (e.g. 249/25 = 9.96).
    wrap_start      : if True, start is taken modulo the real track length, so you can
                      feed huge cumulative values directly and delete the magic-127 wrap.
    overrun_mode    : what to do when the window runs past the end of the track ->
                        loop        : seamlessly wrap to the track start (best for
                                      continuous / audio-reactive material, no silence gaps)
                        silence_pad : take what's there, pad the rest with zeros
                        clamp       : slide the window back so it ends exactly at the tail
    snap_samples_to : 0 = off. If you ever see a one-frame audio/video drift after the
                      audio VAE, set this to the audio VAE's sample stride so the output
                      length is rounded up to a clean multiple. Leave 0 unless needed.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "start_seconds": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0e9, "step": 0.001,
                    "tooltip": "Start position. Feed your raw cumulative seconds here; "
                               "with wrap_start on, no upstream modulo is needed."}),
                "duration_seconds": ("FLOAT", {
                    "default": 9.96, "min": 0.001, "max": 1.0e9, "step": 0.001,
                    "tooltip": "Constant clip length. Output sample count is fixed by this."}),
                "wrap_start": ("BOOLEAN", {"default": True}),
                "overrun_mode": (["loop", "silence_pad", "clamp"], {"default": "loop"}),
            },
            "optional": {
                "snap_samples_to": ("INT", {"default": 0, "min": 0, "max": 1 << 20, "step": 1}),
            },
        }

    RETURN_TYPES = ("AUDIO", "FLOAT", "INT")
    RETURN_NAMES = ("audio", "actual_start_seconds", "num_samples")
    FUNCTION = "trim"
    CATEGORY = "kubakub/2d/motion"

    def trim(self, audio, start_seconds, duration_seconds, wrap_start,
             overrun_mode, snap_samples_to=0):

        wf = _as_bcn(audio["waveform"])
        sr = int(audio["sample_rate"])
        B, C, total = wf.shape

        target = max(1, int(round(duration_seconds * sr)))
        if snap_samples_to and snap_samples_to > 0:
            # round UP to the nearest multiple so we never lose a frame
            r = target % snap_samples_to
            if r:
                target += (snap_samples_to - r)

        start = int(round(start_seconds * sr))
        if wrap_start and total > 0:
            start %= total

        if total == 0:
            out = torch.zeros((B, C, target), dtype=wf.dtype, device=wf.device)
            return ({"waveform": out, "sample_rate": sr}, 0.0, target)

        if overrun_mode == "loop":
            # circular read -> always exactly `target`, seamless wrap, no silence
            idx = (torch.arange(target, device=wf.device) + start) % total
            out = wf.index_select(-1, idx)

        elif overrun_mode == "clamp":
            # slide window so it ends at the tail; pad only if the track itself is shorter
            s = min(start, max(0, total - target))
            chunk = wf[..., s:s + target]
            if chunk.shape[-1] < target:
                pad = torch.zeros((B, C, target - chunk.shape[-1]),
                                  dtype=wf.dtype, device=wf.device)
                chunk = torch.cat([chunk, pad], dim=-1)
            out = chunk
            start = s

        else:  # silence_pad
            end = min(start + target, total)
            chunk = wf[..., start:end] if start < total else wf[..., 0:0]
            if chunk.shape[-1] < target:
                pad = torch.zeros((B, C, target - chunk.shape[-1]),
                                  dtype=wf.dtype, device=wf.device)
                chunk = torch.cat([chunk, pad], dim=-1)
            out = chunk

        return ({"waveform": out.contiguous(), "sample_rate": sr},
                float(start) / float(sr),
                int(out.shape[-1]))


NODE_CLASS_MAPPINGS = {"TrimToExactDuration": TrimToExactDuration}
NODE_DISPLAY_NAME_MAPPINGS = {"TrimToExactDuration": "kubakub trim to exact duration (ltx safe)"}
