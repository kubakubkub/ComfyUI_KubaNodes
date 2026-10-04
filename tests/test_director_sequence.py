"""
Model free test of kubakub director sequence (KUBA_DirectorSequence): the director hands its composition over
(director output) and the sequence node renders the same frames, sound and export as the director's own
render_sequence / export inputs (kept for older workflows). Also: frames off = the still, no document = the still.

Run from the ComfyUI_windows_portable folder:
    python_embeded\\python.exe ComfyUI\\custom_nodes\\ComfyUI_KubaNodes\\tests\\test_director_sequence.py
"""

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

held = nd.KUBA_Director.execute(image=base, document="")
from comfy_execution.graph_utils import ExecutionBlocker  # noqa: E402
blocked = [i for i, v in enumerate(held.result) if isinstance(v, ExecutionBlocker)]
check("flow 'hold until apply' + nothing applied: every output held except the report", blocked == [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11]
      and "waiting for apply" in held.result[6], str(blocked))
check("held: the window still gets its preview / manifest", "kuba_director" in (held.ui or {}))
check("flow input is the last input (older workflows keep their widget order)",
      [i.id for i in nd.KUBA_Director.define_schema().inputs][-1] == "flow")
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

shutil.rmtree(TMP, ignore_errors=True)
print()
print("all director sequence tests passed" if not failures else f"{len(failures)} FAILED: " + ", ".join(failures))
sys.exit(1 if failures else 0)
