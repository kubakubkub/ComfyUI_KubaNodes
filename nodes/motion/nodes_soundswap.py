"""kubakub sound mask swap: layers trade their masks in time with a sound (logic in kubakub/soundswap.py, the engine
it shares with the director's "swap masks" behaviour in kubakub/director/motion.py)."""

from __future__ import annotations

import logging

import torch

import comfy.model_management
import comfy.utils
from comfy_api.latest import io

from ...kubakub import sound as so
from ...kubakub import soundswap as sw
from ...kubakub.director import motion as mo

log = logging.getLogger("KUBA.regions")

STEP_ON = ["beats", "bars", "every", "low", "mid", "high", "signal", "list"]
FADE_STYLES = {"crossfade": "cross", "out, then in": "dip"}


class KUBA_SoundMaskSwap(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="KUBA_SoundMaskSwap",
            display_name="kubakub sound mask swap",
            category="kubakub/2d/motion",
            search_aliases=["audio reactive", "beat", "bpm", "music", "rotate masks", "vj"],
            description=(
                "Layers trade their masks in time with a sound. Each layer has a mask and, if you connect them, a "
                "picture; the pictures stay where they are and the masks move on by one step per beat, bar, fixed "
                "time, or per hit in the low, mid or high frequencies. Gives the frames, the moving mask of one "
                "layer, fps and the sound, ready for Create Video. Nothing connected = a sample facade and a "
                "built-in beat. The kubakub director has the same thing as the behaviour 'swap masks'."),
            inputs=[
                io.Audio.Input("audio", optional=True, tooltip="The sound the masks react to. Empty = a built-in beat (120 bpm)."),
                io.Image.Input("background", optional=True,
                               tooltip="The picture behind the layers; it gives the size. Empty = the sample facade."),
                io.Mask.Input("masks", optional=True,
                              tooltip="One mask per layer (a mask batch, at least two). Empty = the windows of the "
                                      "sample facade, one layer per floor."),
                io.Image.Input("pictures", optional=True,
                               tooltip="One whole picture per layer (an image batch, repeated when there are fewer "
                                       "than masks). Each shows through the mask its layer holds at that moment. "
                                       "Empty = a flat colour per layer, to see the masks move."),
                io.Combo.Input("order", options=list(mo.SWAP_ORDERS), default="loop",
                               tooltip="loop: every layer moves on to the next mask. pingpong: there and back. "
                                       "random: a new shuffle every step."),
                io.Combo.Input("step_on", options=STEP_ON, default="beats",
                               tooltip="What makes a step: beats or bars (tempo of the sound), every (a fixed time), "
                                       "a hit in the low (kick, bass), mid or high (hats, clicks) frequencies, signal "
                                       "(the control wav in 'signal' rises through the threshold) or list (the "
                                       "times in 'step_times')."),
                io.Int.Input("nth", default=1, min=1, max=64, tooltip="Step on every nth beat, bar or hit."),
                io.Float.Input("fade", default=0.0, min=0.0, max=10.0, step=0.01,
                               tooltip="Seconds of fade into the new mask. 0 = the layer just appears there."),
                io.Combo.Input("fade_style", options=list(FADE_STYLES), default="crossfade",
                               tooltip="With a fade: crossfade = the new picture fades in over the old one. out, then "
                                       "in = each layer fades out of its old mask first, then into the new one."),
                io.Float.Input("feather", default=0.0, min=0.0, max=500.0, step=1.0,
                               tooltip="Soft mask edges, in pixels of the background. 0 = the masks as they are."),
                io.Float.Input("threshold", default=0.3, min=0.0, max=1.0, step=0.01,
                               tooltip="low / mid / high: how strong a hit has to be to count (0 = every hit, 1 = only "
                                       "the strongest). signal: the level it has to rise through (0-1 of its peak)."),
                io.Float.Input("fps", default=25.0, min=1.0, max=120.0, step=0.01, tooltip="Frames per second."),
                io.Float.Input("seconds", default=0.0, min=0.0, max=3600.0, step=0.1,
                               tooltip="Length to render. 0 = to the end of the sound."),
                io.Float.Input("scale", default=0.5, min=0.05, max=1.0, step=0.05,
                               tooltip="Size of the frames relative to the background (RAM: 250 frames at 1600x1080 = ~7 GB)."),
                io.Int.Input("mask_of", default=1, min=1, max=64,
                             tooltip="Which layer's moving mask goes out on 'mask' (1 = the first mask of the batch)."),
                io.Audio.Input("signal", optional=True,
                               tooltip="step_on = signal: a control track as a wav (a gate or trigger track, an LFO, an "
                                       "envelope, CV recorded from a synth). A step each time it rises through "
                                       "'threshold'. Its time runs with the sound's."),
                io.String.Input("step_times", default="", multiline=True, optional=True,
                                tooltip="step_on = list: the times of the steps in seconds, separated by commas, "
                                        "spaces or new lines (e.g. 0.5, 1, 1.75, 3)."),
                io.Float.Input("start", default=0.0, min=0.0, max=3600.0, step=0.1, optional=True, advanced=True,
                               tooltip="Seconds into the sound where the frames start."),
                io.Float.Input("bpm", default=0.0, min=0.0, max=400.0, step=0.1, optional=True, advanced=True,
                               tooltip="Tempo for beats / bars. 0 = found in the sound."),
                io.Float.Input("every", default=0.5, min=0.02, max=600.0, step=0.01, optional=True, advanced=True,
                               tooltip="step_on = every: seconds between two steps."),
                io.Float.Input("gap", default=0.1, min=0.0, max=10.0, step=0.01, optional=True, advanced=True,
                               tooltip="low / mid / high: the shortest time between two steps, so a roll does not flicker."),
                io.Int.Input("seed", default=1, min=0, max=2 ** 31 - 1, optional=True, advanced=True,
                             control_after_generate=False, tooltip="order = random: another shuffle."),
            ],
            outputs=[
                io.Image.Output("frames", tooltip="Every frame: the background with each picture (or colour) in the mask its layer holds."),
                io.Mask.Output("mask", tooltip="The moving mask of the layer chosen in mask_of, one per frame."),
                io.Float.Output("fps", tooltip="Frames per second."),
                io.Audio.Output("audio", tooltip="The sound, cut to the frames."),
                io.String.Output("report", tooltip="Tempo or hits found, the number of steps, frames and size."),
            ],
        )

    @classmethod
    def execute(cls, audio=None, background=None, masks=None, pictures=None, order="loop", step_on="beats", nth=1,
                fade=0.0, fade_style="crossfade", feather=0.0, threshold=0.3, fps=25.0, seconds=0.0, scale=0.5, mask_of=1,
                signal=None, step_times="", start=0.0, bpm=0.0, every=0.5, gap=0.1, seed=1) -> io.NodeOutput:
        st = sw.Settings(order=order, step_on=step_on, nth=int(nth), every=float(every), threshold=float(threshold),
                         gap=float(gap), fade=float(fade), transition=FADE_STYLES.get(fade_style, "cross"),
                         feather=float(feather), bpm=float(bpm), fps=float(fps), start=float(start),
                         seconds=float(seconds), scale=float(scale), seed=int(seed))
        notes = []
        sample_sound = audio is None
        if sample_sound:
            sr = 44100
            wave = torch.from_numpy(so.sample_sound(8.0, sw.SAMPLE_BPM, sr))[None, None]
            notes.append("no sound connected: the built-in beat is used (connect yours to 'audio')")
        else:
            wave, sr = audio["waveform"], int(audio["sample_rate"])
            wave = wave if wave.dim() == 3 else wave.reshape(1, -1, wave.shape[-1])
        mono = so.to_mono(wave)
        if mono.shape[0] < sr // 10:
            raise ValueError("kubakub sound mask swap: the sound is empty.")
        bg = background[0].float().cpu().numpy() if background is not None else None
        ms = masks.float().cpu().numpy() if masks is not None else None
        if ms is not None and ms.ndim == 2:
            ms = ms[None]
        if bg is None or ms is None:
            W, H = (bg.shape[1], bg.shape[0]) if bg is not None else ((ms.shape[2], ms.shape[1]) if ms is not None else (1920, 1080))
            s_bg, s_ms = sw.sample_layers(W, H)
            if bg is None:
                bg = s_bg
                notes.append("no background connected: the sample facade is used")
            if ms is None:
                ms = s_ms
                notes.append("no masks connected: the windows of the sample facade are used, one layer per floor")
        pics = pictures.float().cpu().numpy() if pictures is not None else None
        times = None
        if step_on == "signal":
            if signal is None:
                raise ValueError("kubakub sound mask swap: step_on = signal needs a control wav in 'signal' (Load Audio).")
            times = so.gate_times(so.to_mono(signal["waveform"]), int(signal["sample_rate"]), st.threshold, st.gap)
        elif step_on == "list":
            times = so.parse_times(step_times)
            if not times:
                raise ValueError("kubakub sound mask swap: step_on = list needs times in 'step_times', e.g. 0.5, 1, 1.75")

        frames, t0, secs = sw.frame_count(len(mono), sr, st)
        W, H = max(8, int(round(bg.shape[1] * st.scale))), max(8, int(round(bg.shape[0] * st.scale)))
        need = sw.ram_bytes(frames, W, H)
        try:
            import psutil
            free = int(psutil.virtual_memory().available)
        except Exception:  # noqa: BLE001
            free = 0
        if free and need > 0.7 * free:
            raise ValueError(f"kubakub sound mask swap: {frames} frames at {W}x{H} need about {need / 2 ** 30:.0f} GB of RAM, "
                             f"{free / 2 ** 30:.0f} GB are free. Lower 'scale' or 'seconds'.")
        pbar = comfy.utils.ProgressBar(frames)
        out, seq, lines = sw.run(bg, ms, pics, mono, sr, st, int(mask_of), sample=sample_sound, times=times,
                                 progress=lambda i, n: pbar.update_absolute(i, n),
                                 check_interrupt=comfy.model_management.throw_exception_if_processing_interrupted)
        a, b = int(round(t0 * sr)), int(round((t0 + secs) * sr))
        report = "\n".join(lines + notes)
        log.info("[KUBA sound mask swap] %s", report.replace("\n", "\n    "))
        return io.NodeOutput(torch.from_numpy(out), torch.from_numpy(seq), float(min(max(st.fps, 1.0), 120.0)),
                             {"waveform": wave[..., a:b].float().cpu().contiguous(), "sample_rate": sr}, report)


NODE_CLASS_MAPPINGS = {"KUBA_SoundMaskSwap": KUBA_SoundMaskSwap}
NODE_DISPLAY_NAME_MAPPINGS = {"KUBA_SoundMaskSwap": "kubakub sound mask swap"}
