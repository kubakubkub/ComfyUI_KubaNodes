"""
nodes_sound.py

kubakub sound: a sound and how it is heard, set once and carried on one 'sound' cable to every node that listens
(kubakub shader, mask animate, height animate, motion guide, shader mix, sound mask swap). The node finds the tempo
once, so all of them agree on the beat, and draws what they will react to: the loudness, the beats, the hits.
A node with only a plain 'audio' connected works as before, with its own sound settings (kubakub/sound.py take()).
V3 node schema, registered via ../__init__.py.
"""

from __future__ import annotations

import logging

import numpy as np
import torch

from comfy_api.latest import io

from ...kubakub import maskfields as mf
from ...kubakub import sound as so
from ...kubakub.io_types import SoundType
from ..shader.nodes_shader import _preview

log = logging.getLogger("KUBA.sound")
NAME = "kubakub sound"
SAMPLE_SECONDS = 8.0


class KUBA_Sound(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="KUBA_Sound",
            display_name=NAME,
            category="kubakub/2d/motion",
            search_aliases=["audio", "beat", "bpm", "tempo", "audio reactive", "music", "hits", "kick"],
            description=(
                "A sound and how it is heard, set once for every node that listens: connect 'sound' to the 'sound' "
                "input of kubakub shader, mask animate, height animate, motion guide, shader mix or sound mask swap. "
                "The tempo is found once, so all of them step on the same beat. The node draws what they react to: "
                "the loudness as a line, the beats as bars, the hits in the low, mid and high frequencies as dots. "
                "Nothing connected = a built-in beat. No AI model."),
            inputs=[
                io.Audio.Input("audio", optional=True, tooltip="Your sound (Load Audio). Empty = a built-in beat (120 bpm)."),
                io.Float.Input("bpm", default=0.0, min=0.0, max=400.0, step=0.1,
                               tooltip="The tempo, when you know it. 0 = found in the sound (the report says which)."),
                io.Float.Input("gain", default=1.0, min=0.0, max=10.0, step=0.05,
                               tooltip="How loud the sound is heard: the orange line. Above 1 for a quiet recording."),
                io.Float.Input("release", default=0.15, min=0.0, max=5.0, step=0.01,
                               tooltip="Seconds the loudness takes to fall after a peak: low = nervous, high = slow and round."),
                io.Combo.Input("listen", options=list(mf.LISTEN), default="everything",
                               tooltip="Which part of the sound the loudness follows: everything, or only the low (kick, "
                                       "bass), mid or high (hats, clicks) frequencies."),
                io.Float.Input("threshold", default=0.3, min=0.0, max=1.0, step=0.01,
                               tooltip="How strong a hit has to be to count (0 = every hit, 1 = only the strongest): "
                                       "the bright dots count, the dim ones do not."),
                io.Float.Input("gap", default=0.1, min=0.0, max=10.0, step=0.01,
                               tooltip="The shortest time between two hits that count, so a roll does not flicker."),
                io.Float.Input("start", default=0.0, min=0.0, max=36000.0, step=0.1,
                               tooltip="Seconds into the sound where it starts."),
                io.Float.Input("seconds", default=0.0, min=0.0, max=36000.0, step=0.1,
                               tooltip="How much of the sound is used. 0 = to its end."),
            ],
            outputs=[
                SoundType.Output("sound", tooltip="The sound with these settings: into 'sound' of every node that listens."),
                io.Audio.Output("audio", tooltip="The same stretch of the sound, for Create Video."),
                io.Float.Output("bpm", tooltip="The tempo: yours, or the one found (0 = none found)."),
                io.String.Output("report", tooltip="Length, tempo and first beat, and how many hits count in each band."),
            ],
        )

    @classmethod
    def execute(cls, audio=None, bpm=0.0, gain=1.0, release=0.15, listen="everything", threshold=0.3, gap=0.1,
                start=0.0, seconds=0.0) -> io.NodeOutput:
        notes = []
        if audio is None:
            hz = 44100
            wave = torch.from_numpy(so.sample_sound(max(SAMPLE_SECONDS, float(start) + float(seconds)), mf.SAMPLE_BPM, hz))[None, None]
            bpm = float(bpm) or float(mf.SAMPLE_BPM)
            notes.append("no sound connected: the built-in beat is used (connect yours to 'audio')")
        else:
            wave, hz = audio["waveform"], int(audio["sample_rate"])
            wave = wave if wave.dim() == 3 else wave.reshape(1, -1, wave.shape[-1])
        total = wave.shape[-1] / float(hz)
        if total < 0.1:
            raise ValueError(f"{NAME}: the sound is empty.")
        a = int(round(min(max(0.0, float(start)), max(0.0, total - 0.1)) * hz))
        b = wave.shape[-1] if seconds <= 0 else min(wave.shape[-1], a + max(1, int(round(float(seconds) * hz))))
        wave = wave[..., a:b].float().cpu().contiguous()
        mono = so.to_mono(wave)
        secs = len(mono) / float(hz)

        if bpm > 0:
            beat, how = {"bpm": float(bpm), "phase": so.beat_phase(mono, hz, float(bpm))}, "set"
        else:
            beat, how = so.analyze_beats(mono, hz), "found"
        heard = dict(bpm=float(beat["bpm"]) if beat else 0.0, gain=float(gain), release=float(release), listen=listen,
                     threshold=float(threshold), gap=float(gap))
        lines = [f"sound {secs:.2f} s" + (f", from {a / float(hz):.2f} s of the file" if a else "")]
        lines.append(f"{beat['bpm']} bpm ({how}), first beat at {beat['phase']} s" if beat
                     else "no tempo found (the sound is shorter than 2 s): set bpm; beats and bars will not step")
        energy = so.band_energy(mono, hz)
        counted = []
        for band in ("low", "mid", "high"):
            hits, last, n = so._hits(energy[band]), None, 0
            for t, s in hits:
                if s >= threshold and (last is None or t - last >= gap - 1e-9):
                    n, last = n + 1, t
            counted.append(f"{band} {n} of {len(hits)}")
        lines.append("hits that count: " + ", ".join(counted) + f" (threshold {threshold:g}, gap {gap:g} s)")
        lines.append(f"loudness: {listen}, gain {gain:g}, release {release:g} s")
        out_audio = {"waveform": wave, "sample_rate": hz}
        report = "\n".join(lines + notes)
        log.info("[KUBA sound] %s", report.replace("\n", "\n    "))
        pic = so.picture(mono, hz, beat, heard)
        return io.NodeOutput(so.bundle(out_audio, **heard), out_audio, float(heard["bpm"]), report,
                             ui=_preview(np.ascontiguousarray(pic)[None], 1.0))


NODE_CLASS_MAPPINGS = {"KUBA_Sound": KUBA_Sound}
NODE_DISPLAY_NAME_MAPPINGS = {"KUBA_Sound": NAME}
