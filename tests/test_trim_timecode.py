"""
Model free test for trim to exact duration (constant sample count in every overrun mode) and timecode filename
prefix (frame-accurate mm-ss-ff).

Run from the ComfyUI_windows_portable folder:
    python_embeded\\python.exe ComfyUI\\custom_nodes\\ComfyUI_KubaNodes\\tests\\test_trim_timecode.py
"""

import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))     # embedded Python leaves the script folder off the path
import _pack  # noqa: E402,F401  (the pack as a package, see tests/_pack.py)
from kubapack.nodes.motion import ltxv_trim_exact_Kub as tr  # noqa: E402
from kubapack.nodes.utils import timecode_filename_Kub as tc  # noqa: E402

failures = []


def check(name, cond, detail=""):
    print(("ok   " if cond else "FAIL ") + name + (f"  ({detail})" if detail and not cond else ""))
    if not cond:
        failures.append(name)


sr = 100
wave = torch.arange(1000, dtype=torch.float32).reshape(1, 1, 1000)     # 10 s, sample value = its index
audio = {"waveform": wave, "sample_rate": sr}
node = tr.TrimToExactDuration()

for mode in ("loop", "silence_pad", "clamp"):
    for start in (0.0, 5.0, 9.5, 123.4):
        out, _, n = node.trim(audio, start, 2.0, True, mode)
        check(f"{mode} start {start}: always 200 samples", n == 200 and out["waveform"].shape == (1, 1, 200), str(n))

out, s, _ = node.trim(audio, 9.5, 2.0, True, "loop")
w = out["waveform"][0, 0]
check("loop wraps to the track start", w[49].item() == 999 and w[50].item() == 0)
out, _, _ = node.trim(audio, 9.5, 2.0, True, "silence_pad")
check("silence_pad pads with zeros", out["waveform"][0, 0, 50:].abs().sum().item() == 0)
out, s, _ = node.trim(audio, 9.5, 2.0, True, "clamp")
check("clamp ends at the tail", out["waveform"][0, 0, -1].item() == 999 and abs(s - 8.0) < 1e-9, str(s))
_, s, _ = node.trim(audio, 12.0, 1.0, True, "loop")
check("wrap_start takes the start modulo the track", abs(s - 2.0) < 1e-9, str(s))
_, _, n = node.trim(audio, 0.0, 1.03, True, "loop", snap_samples_to=8)
check("snap_samples_to rounds up to a multiple", n == 104, str(n))
out, _, n = node.trim({"waveform": torch.zeros(2, 0), "sample_rate": sr}, 0.0, 1.0, True, "loop")
check("empty track gives silence of the right length", n == 100 and out["waveform"].shape == (1, 2, 100))

build = tc.TimecodeFilenamePrefix().build
check("12 s at 25 fps", build("clip", 12.0, 25.0, "_") == ("clip_00-12-00", "00-12-00", 300))
check("frames and minutes", build("a", 75.52, 25.0, "-") == ("a-01-15-13", "01-15-13", 1888))

print(f"\n{len(failures)} failed" if failures else "\nall passed")
sys.exit(1 if failures else 0)
