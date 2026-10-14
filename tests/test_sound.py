"""
Model free test for the sound cable: kubakub sound (nodes/motion/nodes_sound.py), what travels on its 'sound' output
(kubakub/sound.py bundle / take / picture) and the nodes that take it in place of their own sound settings (mask
animate, height animate, shader mix, sound mask swap, and with a GL context kubakub shader). Does not start ComfyUI.

Run from the ComfyUI_windows_portable folder:
    python_embeded\\python.exe ComfyUI\\custom_nodes\\ComfyUI_KubaNodes\\tests\\test_sound.py
"""

import os
import shutil
import sys
import tempfile

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(HERE))))     # ComfyUI (comfy_api)
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)     # embedded Python leaves the script folder off the path

sys.argv = sys.argv[:1] + ["--cpu"]
import comfy.options  # noqa: E402
comfy.options.enable_args_parsing()
import folder_paths  # noqa: E402

TMP = tempfile.mkdtemp(prefix="kkd_sound_")
folder_paths.set_temp_directory(os.path.join(TMP, "temp"))

import _pack  # noqa: E402,F401  (the pack as a package, see tests/_pack.py)
from kubapack.nodes.masks import nodes_masks as nm  # noqa: E402
from kubapack.nodes.motion import nodes_height as nh  # noqa: E402
from kubapack.nodes.motion import nodes_sound as ns  # noqa: E402
from kubapack.nodes.motion import nodes_soundswap as nw  # noqa: E402
from kubapack.nodes.motion import nodes_guide as ng  # noqa: E402
from kubapack.nodes.shader import nodes_mix as nx  # noqa: E402
from kubapack.nodes.shader import nodes_shader as nsh  # noqa: E402
from kubakub import maskfields as mf, sound as so  # noqa: E402

failures = []


def check(name, cond, detail=""):
    print(("ok   " if cond else "FAIL ") + name + (f"  ({detail})" if detail and not cond else ""))
    if not cond:
        failures.append(name)


SR = 44100
wave = torch.from_numpy(so.sample_sound(8.0, 120.0, SR))[None, None]
AUDIO = {"waveform": wave, "sample_rate": SR}

# ---- what travels on the cable
b = so.bundle(AUDIO, bpm=120, gain=2)
check("bundle: the sound and every setting, the ones not named at their defaults", b["audio"] is AUDIO and b["bpm"] == 120.0
      and b["gain"] == 2.0 and b["release"] == 0.15 and b["listen"] == "everything" and set(b) == {"audio", *so.HEARD})
audio, heard, cabled = so.take(b, {"waveform": wave[..., :100], "sample_rate": SR}, gain=5, bpm=90)
check("take: a cable comes before the node's own audio and settings", cabled and audio is AUDIO and heard["gain"] == 2.0 and heard["bpm"] == 120.0)
audio, heard, cabled = so.take(None, AUDIO, gain=5)
check("take: without a cable the node's own, the rest at the defaults", not cabled and audio is AUDIO and heard["gain"] == 5 and heard["threshold"] == 0.3)
pic = so.picture(so.to_mono(wave), SR, {"bpm": 120.0, "phase": 0.0}, {"threshold": 0.3})
check("picture: the loudness line, the beat bars and the hit dots are drawn", pic.shape == (200, 640, 3) and pic.dtype == np.float32
      and float(pic[..., 0].max()) > 0.9 and float(pic[20:150].max()) > 0.3 and float(pic[160:].max()) > 0.5)

# ---- the node
schema = ns.KUBA_Sound.define_schema()
check("node: id, name, outputs", schema.node_id == "KUBA_Sound" and schema.display_name == "kubakub sound"
      and [o.id for o in schema.outputs] == ["sound", "audio", "bpm", "report"] and all(i.tooltip for i in schema.inputs))
out = ns.KUBA_Sound.execute().args
check("node, nothing connected: the built-in beat at 120 bpm, and said", out[2] == 120.0 and "built-in beat" in out[3]
      and out[0]["audio"]["waveform"].shape[-1] == 8 * SR, out[3])
out = ns.KUBA_Sound.execute(audio=AUDIO, gain=1.5, threshold=0.4, start=2.0, seconds=4.0).args
check("node: the tempo is found once and travels as a number", abs(out[2] - 120.0) < 0.6 and out[0]["bpm"] == out[2] and "found" in out[3], out[3])
check("node: start and seconds cut the sound for the cable and for 'audio'", out[0]["audio"]["waveform"].shape[-1] == 4 * SR
      and out[1]["waveform"].shape[-1] == 4 * SR and "from 2.00 s" in out[3])
check("node: the settings travel", out[0]["gain"] == 1.5 and out[0]["threshold"] == 0.4 and "low 8 of" in out[3], out[3])
out = ns.KUBA_Sound.execute(audio=AUDIO, start=500.0, seconds=2.0).args
check("node: a start past the end begins at 0 and says so", out[1]["waveform"].shape[-1] == 2 * SR
      and "past the end" in out[3] and "from " not in out[3].splitlines()[0], out[3])
out = ns.KUBA_Sound.execute(audio=AUDIO, bpm=100.0).args
check("node: a tempo you set wins", out[2] == 100.0 and "(set)" in out[3], out[3])
try:
    ns.KUBA_Sound.execute(audio={"waveform": torch.zeros(1, 1, 10), "sample_rate": SR})
    check("node: an empty sound is refused", False)
except ValueError as e:
    check("node: an empty sound is refused", "empty" in str(e))

# ---- the nodes that listen
SOUND = ns.KUBA_Sound.execute(audio=AUDIO, gain=0.25).args[0]
for cls in (nm.KUBA_MaskAnimate, nh.KUBA_HeightAnimate, ng.KUBA_MotionGuide, nx.KUBA_ShaderMix, nw.KUBA_SoundMaskSwap, nsh.KUBA_Shader):
    last = cls.define_schema().inputs[-1]
    check(f"{cls.__name__}: 'sound' is its last input, optional", last.id == "sound" and last.optional and bool(last.tooltip))

own = nm.KUBA_MaskAnimate.execute(effect="opacity", curve="sound level", fps=20.0, seconds=1.0, scale=0.25, audio=AUDIO).result
cab = nm.KUBA_MaskAnimate.execute(effect="opacity", curve="sound level", fps=20.0, seconds=1.0, scale=0.25, sound=SOUND, gain=9.0).result
check("mask animate: the cable's gain is used, not the node's own", float(cab[0].max()) < 0.5 * float(own[0].max()) and float(cab[0].max()) > 0.0,
      f"{float(cab[0].max()):.3f} vs {float(own[0].max()):.3f}")
late = ns.KUBA_Sound.execute(audio=AUDIO, start=0.25).args[0]
a = nm.KUBA_MaskAnimate.execute(effect="opacity", curve="low hits", cycle=0.2, fps=20.0, seconds=1.0, scale=0.25, audio=AUDIO, start=0.25).result[0]
c = nm.KUBA_MaskAnimate.execute(effect="opacity", curve="low hits", cycle=0.2, fps=20.0, seconds=1.0, scale=0.25, sound=late, start=3.0).result[0]
check("mask animate: a cable's sound is cut already, the node's own start does not apply", torch.allclose(a, c, atol=0.02),
      f"{float((a - c).abs().max()):.3f}")

t = [i / 20.0 for i in range(20)]
mono = so.to_mono(wave)
v0, _ = mf.timed_values(t, "sound level", 1.0, 0.0, 1.0, mono=mono, sr=SR)
v1, _ = mf.timed_values(t, "sound level", 1.0, 0.0, 1.0, mono=mono, sr=SR, heard={"gain": 0.25})
v2, lines = mf.timed_values(t, "beats", 0.2, 0.0, 1.0, mono=mono, sr=SR, heard={"bpm": 60.0})
check("a curve hears as the cable says: gain", max(v1) < 0.5 * max(v0) and max(v1) > 0)
check("a curve hears as the cable says: the tempo", any("60.0 bpm (set)" in x for x in lines), str(lines))
h0 = nh.KUBA_HeightAnimate.execute(curve="sound level", fps=10.0, seconds=1.0, scale=0.25, audio=AUDIO).result
h1 = nh.KUBA_HeightAnimate.execute(curve="sound level", fps=10.0, seconds=1.0, scale=0.25, sound=SOUND).result
check("height animate: grows less with a quieter cable, and gives its sound out", float(h1[0].mean()) < float(h0[0].mean())
      and h1[4]["waveform"].shape[-1] == SR, f"{float(h1[0].mean()):.4f} vs {float(h0[0].mean()):.4f}")
sw0 = nw.KUBA_SoundMaskSwap.execute(audio=AUDIO, seconds=2.0, fps=10.0, scale=0.1).result
sw1 = nw.KUBA_SoundMaskSwap.execute(sound=ns.KUBA_Sound.execute(audio=AUDIO, bpm=60.0).args[0], seconds=2.0, fps=10.0, scale=0.1, bpm=200.0).result
check("sound mask swap: steps on the cable's tempo", "60" in sw1[4] and "200" not in sw1[4] and sw1[4] != sw0[4], sw1[4])

try:
    red, blue = torch.zeros(1, 16, 32, 3), torch.zeros(1, 16, 32, 3)
    red[..., 0], blue[..., 2] = 1.0, 1.0
    m0 = nx.KUBA_ShaderMix.execute("rings", "rings", how="blend", mix=1.0, curve="sound level", a=red, b=blue, seconds=1.0, fps=10.0, audio=AUDIO).result
    m1 = nx.KUBA_ShaderMix.execute("rings", "rings", how="blend", mix=1.0, curve="sound level", a=red, b=blue, seconds=1.0, fps=10.0, sound=SOUND).result
    check("shader mix: the mix follows the cable's sound, quieter = more of A", float(m1[0][..., 2].mean()) < float(m0[0][..., 2].mean())
          and float(m1[0][..., 2].mean()) > 0.0, f"{float(m1[0][..., 2].mean()):.3f} vs {float(m0[0][..., 2].mean()):.3f}")
    LEVEL = "void mainImage(out vec4 c, in vec2 f){ c = vec4(vec3(kLevel), 1.0); }"
    s0 = nsh.KUBA_Shader.execute("rings", code=LEVEL, width=16, height=8, fps=10.0, seconds=1.0, audio=AUDIO).result
    s1 = nsh.KUBA_Shader.execute("rings", code=LEVEL, width=16, height=8, fps=10.0, seconds=1.0, sound=SOUND, gain=9.0).result
    check("shader: kLevel follows the cable's gain, and the cable's sound goes out on 'audio'", float(s1[0].max()) < 0.5 * float(s0[0].max())
          and float(s1[0].max()) > 0.0 and s1[3]["waveform"].shape[-1] == SR, f"{float(s1[0].max()):.3f} vs {float(s0[0].max()):.3f}")
except Exception as e:  # noqa: BLE001
    if "graphics context" in str(e):
        print("skip the GPU part: no GL context here")
    else:
        raise

shutil.rmtree(TMP, ignore_errors=True)
print("\nALL PASSED" if not failures else f"\n{len(failures)} FAILED: {failures}")
sys.exit(1 if failures else 0)
