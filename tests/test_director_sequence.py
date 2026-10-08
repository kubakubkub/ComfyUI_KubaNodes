"""
Model free test of kubakub director sequence (KUBA_DirectorSequence): the director hands its composition over
(director output) and the sequence node renders the same frames, sound and export as the director's own
render_sequence / export inputs (kept for older workflows). Also: frames off = the still, no document = the still.

Run from the ComfyUI_windows_portable folder:
    python_embeded\\python.exe ComfyUI\\custom_nodes\\ComfyUI_KubaNodes\\tests\\test_director_sequence.py
"""

import dataclasses
import glob
import json
import os
import shutil
import sys
import tempfile
import types

import cv2
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(HERE))))     # ComfyUI (folder_paths, comfy_api)
sys.path.insert(0, os.path.dirname(HERE))
import folder_paths  # noqa: E402

TMP = tempfile.mkdtemp(prefix="kkd_seq_")
folder_paths.set_temp_directory(os.path.join(TMP, "temp"))
folder_paths.set_output_directory(os.path.join(TMP, "output"))
folder_paths.set_input_directory(os.path.join(TMP, "input"))
os.makedirs(os.path.join(TMP, "input"), exist_ok=True)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))     # embedded Python leaves the script folder off the path

import _pack  # noqa: E402,F401  (the pack as a package, see tests/_pack.py)

import kubapack.nodes.director.nodes_director as nd  # noqa: E402

nd.KUBA_Director.hidden = types.SimpleNamespace(unique_id="seqtest", prompt=None, extra_pnginfo=None)
failures = []


def check(name, cond, detail=""):
    print(("ok   " if cond else "FAIL ") + name + (f"  ({detail})" if detail else ""))
    if not cond:
        failures.append(name)


W, H = 320, 200
yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
base = torch.from_numpy(np.stack([xx / W, yy / H, np.full_like(xx, 0.3)], -1))[None]
doc = {"version": 1, "canvas": [W, H], "timeline": {"duration": 1.0, "fps": 10, "time": 0.4},
       "layers": [{"id": "sq", "name": "square", "kind": "shape", "shape": {"type": "rect"}, "color": "#ff8800",
                   "x": 20, "y": 30, "w": 80, "h": 50, "opacity": 0.9,
                   "anim": {"x": [{"t": 0, "v": 20}, {"t": 1, "v": 200}]},
                   "motion": [{"id": "m1", "type": "wiggle", "path": "rotation", "amount": 12, "freq": 2, "seed": 3, "on": True}]},
                  {"id": "base", "kind": "base", "name": "clay"}]}
D = json.dumps(doc)


def run_director(**kw):
    return nd.KUBA_Director.execute(image=base, document=D, **kw)


legacy = run_director(render_sequence=True, sequence_scale=0.5, skip_h3_frames=True)
new_d = run_director()
state = new_d.result[11]
check("director has a 12th output: the director state", isinstance(state, nd.DirectorState) and state.doc is not None)
check("still is the same with and without render_sequence", torch.equal(legacy.result[0], new_d.result[0]))
check("director frames output without render_sequence = the still", torch.equal(new_d.result[7], new_d.result[0]))

seq = nd.KUBA_DirectorSequence.execute(director=state, frames=True, sequence_scale=0.5, skip_h3_frames=True)
fa, fb = legacy.result[7], seq.result[0]
check("sequence frames = the director's render_sequence frames", fa.shape == fb.shape and torch.equal(fa, fb),
      f"{tuple(fa.shape)} vs {tuple(fb.shape)}")
check("10 frames of 160x100 (1 s at 10 fps, scale 0.5)", tuple(fb.shape) == (10, 100, 160, 3), str(tuple(fb.shape)))
check("the square moves between frames", not torch.equal(fb[0], fb[-1]))
check("fps and audio passed through", seq.result[1] == legacy.result[8] == 10.0
      and torch.equal(seq.result[2]["waveform"], legacy.result[9]["waveform"]))

again = nd.KUBA_DirectorSequence.execute(director=state, frames=True, sequence_scale=0.5)
check("the state is not changed by a sequence run (same frames twice)", torch.equal(again.result[0], fb))

off = nd.KUBA_DirectorSequence.execute(director=state, frames=False)
check("frames off = the still, nothing rendered", torch.equal(off.result[0], new_d.result[0]) and "nothing to do" in off.result[3])

# export: the same PNG files as the director's own export input
out_root = os.path.join(TMP, "output", "kubakub_director")
run_director(export="png8", export_scale=0.5, export_name="legacyexp")
nd.KUBA_DirectorSequence.execute(director=state, frames=False, export="png8", export_scale=0.5, export_name="seqexp")
la = sorted(glob.glob(os.path.join(out_root, "legacyexp_*", "**", "*.png"), recursive=True))
lb = sorted(glob.glob(os.path.join(out_root, "seqexp_*", "**", "*.png"), recursive=True))
same = len(la) == len(lb) == 10 and all(np.array_equal(cv2.imread(a, cv2.IMREAD_UNCHANGED), cv2.imread(b, cv2.IMREAD_UNCHANGED)) for a, b in zip(la, lb))
check("export png8 from the sequence node = the director's export (10 identical frames)", same, f"{len(la)} / {len(lb)} files")

# the outputs added after 'report': folder, files, frame_count, projection_mask, document
check("sequence outputs: the old four first, the new ones appended",
      [o.id for o in nd.KUBA_DirectorSequence.define_schema().outputs]
      == ["frames", "fps", "audio", "report", "folder", "files", "frame_count", "projection_mask", "document"])
check("every sequence / export output and input has a tooltip",
      all(getattr(x, "tooltip", None) for c in (nd.KUBA_DirectorSequence, nd.KUBA_Export)
          for x in [*c.define_schema().inputs, *c.define_schema().outputs]))
check("no export: folder and files are empty", seq.result[4] == "" and seq.result[5] == "")
check("frame_count = the timeline's frames = the frames returned", seq.result[6] == 10 == fb.shape[0] and off.result[6] == 10)
check("projection_mask at the frame size, all ones without a projection mask",
      tuple(seq.result[7].shape) == (1, 100, 160) and bool((seq.result[7] == 1).all())
      and tuple(off.result[7].shape) == (1, H, W), f"{tuple(seq.result[7].shape)} / {tuple(off.result[7].shape)}")
check("document = the director's document output", seq.result[8] == new_d.result[5] and json.loads(seq.result[8]) == doc)
ex1 = nd.KUBA_DirectorSequence.execute(director=state, frames=False, export="png8", export_scale=0.5, export_name="pathsexp")
fo, fi = ex1.result[4], ex1.result[5].split("\n")
check("export: folder = the export folder, files = what was written (one path per line)",
      os.path.isdir(fo) and os.path.basename(fo).startswith("pathsexp_") and fo in ex1.result[3]
      and all(os.path.exists(f) and f.startswith(fo) for f in fi) and len(glob.glob(os.path.join(fi[0], "*.png"))) == 10, ex1.result[5])
ex2 = nd.KUBA_DirectorSequence.execute(director=state, frames=False, export="png8", export_scale=0.5, export_name="pathsexp")
check("export not written again: folder and files point to the earlier export",
      "not written again" in ex2.result[3] and ex2.result[4] == fo and ex2.result[5] == ex1.result[5])
try:
    nd.KUBA_DirectorSequence.execute(director=state, frames=False, export="preview", export_scale=0.5, export_alpha=True, export_name="noalpha")
    check("export_alpha with a format without alpha is refused", False)
except ValueError as e:
    check("export_alpha with a format without alpha is refused, naming the formats that can",
          all(f in str(e) for f in ("png8", "png16", "prores4444")) and "export_alpha" in str(e), str(e))
pm = np.zeros((H, W), np.float32)
pm[:, :W // 2] = 1
m_seq = nd._pmask_out(dataclasses.replace(state, pmask=pm), 0.5)
check("projection_mask follows the frames: the left half at half size", tuple(m_seq.shape) == (1, 100, 160)
      and float(m_seq[0, :, :78].min()) == 1.0 and float(m_seq[0, :, 82:].max()) == 0.0)

# kubakub export video / frames: alpha of another size is resized (it was cropped), paths out, alpha the format cannot carry
imgs = torch.rand((3, 50, 80, 3))
big = torch.zeros((100, 160))
big[:, 80:] = 1                                              # full size mask on half size frames: the right half
r_big = nd.KUBA_Export.execute(images=imgs, fps=10.0, format="png8", name="alphabig", alpha=big)
png = cv2.imread(sorted(glob.glob(os.path.join(r_big.result[2].split("\n")[0], "*.png")))[0], cv2.IMREAD_UNCHANGED)
check("export: a larger alpha is resized to the frames, not cropped", png.shape == (50, 80, 4)
      and int(png[:, :38, 3].max()) == 0 and int(png[:, 42:, 3].min()) == 255 and "alpha 160x100 resized" in r_big.result[0],
      f"{png.shape} left {int(png[:, :38, 3].max())} right {int(png[:, 42:, 3].min())}")
small = torch.zeros((3, 25, 40))
small[:, :, 20:] = 1
r_small = nd.KUBA_Export.execute(images=imgs, fps=10.0, format="png8", name="alphasmall", alpha=small)
png = cv2.imread(sorted(glob.glob(os.path.join(r_small.result[2].split("\n")[0], "*.png")))[0], cv2.IMREAD_UNCHANGED)
check("export: a smaller alpha is resized too (it crashed)", png.shape == (50, 80, 4) and int(png[:, :36, 3].max()) == 0
      and int(png[:, 44:, 3].min()) == 255 and "resized" in r_small.result[0])
r_same = nd.KUBA_Export.execute(images=imgs, fps=10.0, format="png8", name="alphasame", alpha=torch.ones((50, 80)))
check("export: an alpha of the frame size is taken as it is (no note)", "resized" not in r_same.result[0])
check("export outputs: report, folder, files", [o.id for o in nd.KUBA_Export.define_schema().outputs] == ["report", "folder", "files"]
      and os.path.isdir(r_same.result[1]) and os.path.basename(r_same.result[1]).startswith("alphasame_")
      and all(os.path.exists(f) and f.startswith(r_same.result[1]) for f in r_same.result[2].split("\n")))
r_na = nd.KUBA_Export.execute(images=imgs, fps=10.0, format="preview", name="alphanone", alpha=big)
check("export: alpha on a format without alpha is said in the report (and the file is written)",
      "alpha not written" in r_na.result[0] and "prores4444" in r_na.result[0] and "resized" not in r_na.result[0]
      and os.path.isfile(r_na.result[2]), r_na.result[0])
check("the director's old sequence widgets say so, layers promises no missing inputs",
      all(next(i for i in nd.KUBA_Director.define_schema().inputs if i.id == k).tooltip.startswith("(old: use kubakub director sequence)")
          for k in ("render_sequence", "skip_h3_frames", "sequence_scale"))
      and "director inputs" not in next(i for i in nd.KUBA_Director.define_schema().inputs if i.id == "layers").tooltip
      and "fps output" in next(i for i in nd.KUBA_Export.define_schema().inputs if i.id == "fps").tooltip)

held = nd.KUBA_Director.execute(image=base, document="")
from comfy_execution.graph_utils import ExecutionBlocker  # noqa: E402
blocked = [i for i, v in enumerate(held.result) if isinstance(v, ExecutionBlocker)]
check("flow 'hold until apply' + nothing applied: every output held except the report", blocked == [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11]
      and "waiting for apply" in held.result[6], str(blocked))
check("held: the window still gets its preview / manifest", "kuba_director" in (held.ui or {}))
check("flow is the last widget (older workflows keep their widget order); only the audio socket comes after it",
      [i.id for i in nd.KUBA_Director.define_schema().inputs][-2:] == ["flow", "audio"])
applied = nd.KUBA_Director.execute(image=base, document=D)
check("hold: an applied composition passes on", not any(isinstance(v, ExecutionBlocker) for v in applied.result))
empty = nd.KUBA_Director.execute(image=base, document="", flow="always")
st0 = empty.result[11]
e = nd.KUBA_DirectorSequence.execute(director=st0)
check("no composition yet: the sequence node returns the still", st0.doc is None and torch.equal(e.result[0], empty.result[0]))

schema = nd.KUBA_DirectorSequence.define_schema()
check("node id / display name / category", schema.node_id == "KUBA_DirectorSequence"
      and schema.display_name == "kubakub director sequence" and schema.category == "kubakub/2d/director")
ds = nd.KUBA_Director.define_schema()
adv = {i.id for i in ds.inputs if getattr(i, "advanced", None)}
check("the director's old sequence / export inputs are advanced (still there for saved workflows)",
      adv == {"render_sequence", "skip_h3_frames", "export", "export_scale", "export_alpha", "export_name", "sequence_scale"}, str(sorted(adv)))
check("director outputs keep their order, director appended last",
      [o.id if hasattr(o, "id") else None for o in ds.outputs][-1] == "director" and len(ds.outputs) == 12)

# ---- a sound on the director's audio input is the timeline's sound
from kubakub import sound as so  # noqa: E402
snd_in = {"waveform": torch.from_numpy(so.sample_sound(3.0, 120.0, 44100))[None, None], "sample_rate": 44100}
ADOC = json.dumps({"canvas": {"w": W, "h": H}, "timeline": {"duration": 2, "fps": 10},
                   "layers": [{"id": "s", "kind": "shape", "shape": {"type": "solid"}, "color": "#ff0000", "x": 0, "y": 0, "w": 40, "h": 40},
                              {"id": "base", "kind": "base"}]})
img_in = torch.zeros((1, H, W, 3))
da = nd.KUBA_Director.execute(img_in, document=ADOC, audio=snd_in)
docA = json.loads(da.result[5])
fileA = docA["timeline"]["audio"]["file"]
check("audio input: written once into input/kuba_director and named in the document",
      fileA.startswith("kuba_director/node_audio_") and os.path.isfile(os.path.join(TMP, "input", *fileA.split("/"))), fileA)
check("audio input: the tempo of the sound is found (120 bpm)", abs(docA["timeline"].get("beat", {}).get("bpm", 0) - 120) < 1, str(docA["timeline"].get("beat")))
check("audio input: the audio output is that sound, cut to the timeline (2 s)",
      da.result[9]["waveform"].shape[-1] == 2 * 44100 and float(da.result[9]["waveform"].abs().max()) > 0.1)
check("audio input: the window is told (manifest audio_in)", da.ui["kuba_director"][0]["audio_in"] == fileA)
da2 = nd.KUBA_Director.execute(img_in, document=json.dumps(dict(docA, timeline=dict(docA["timeline"], beat={"bpm": 60, "phase": 0.1}))), audio=snd_in)
check("audio input: the same sound again keeps the tempo set in the window",
      json.loads(da2.result[5])["timeline"]["beat"] == {"bpm": 60, "phase": 0.1})
de = nd.KUBA_Director.execute(img_in, document="", audio=snd_in, flow="always")
check("audio input without a composition: the document stays empty, the sound still goes out (its own length)",
      de.result[5] == "" and de.result[9]["waveform"].shape[-1] == 3 * 44100 and de.ui["kuba_director"][0]["audio_in"] == fileA,
      f"{de.result[5]!r} {de.result[9]['waveform'].shape}")
dn = nd.KUBA_Director.execute(img_in, document=ADOC)
check("no audio input: nothing changes (silence, no audio_in)", "audio" not in json.loads(dn.result[5])["timeline"]
      and float(dn.result[9]["waveform"].abs().max()) == 0 and dn.ui["kuba_director"][0]["audio_in"] == "")
try:
    nd.KUBA_Director.execute(img_in, document=ADOC, export="preview", export_alpha=True)
    check("director: export_alpha with a format without alpha is refused", False)
except ValueError as e:
    check("director: export_alpha with a format without alpha is refused", "export_alpha" in str(e))
check("director schema: audio is the last input and optional",
      nd.KUBA_Director.define_schema().inputs[-1].id == "audio" and nd.KUBA_Director.define_schema().inputs[-1].optional)

shutil.rmtree(TMP, ignore_errors=True)
print()
print("all director sequence tests passed" if not failures else f"{len(failures)} FAILED: " + ", ".join(failures))
sys.exit(1 if failures else 0)
