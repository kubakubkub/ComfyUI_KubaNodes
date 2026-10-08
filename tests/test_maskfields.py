"""
Model free test of the mask tools: fields over regions and masks that move (kubakub/maskfields.py), and the two nodes
kubakub mask field / kubakub mask animate (nodes/masks/nodes_masks.py).

Run from the ComfyUI_windows_portable folder:
    python_embeded\\python.exe ComfyUI\\custom_nodes\\ComfyUI_KubaNodes\\tests\\test_maskfields.py
"""

import os
import shutil
import sys
import tempfile

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(HERE))))     # ComfyUI (folder_paths, comfy_api)
sys.path.insert(0, os.path.dirname(HERE))
import folder_paths  # noqa: E402

TMP = tempfile.mkdtemp(prefix="kkd_masks_")
folder_paths.set_temp_directory(os.path.join(TMP, "temp"))

sys.path.insert(0, HERE)     # embedded Python leaves the script folder off the path
import _pack  # noqa: E402,F401  (the pack as a package, see tests/_pack.py)

import kubapack.nodes.masks.nodes_masks as nm  # noqa: E402
from kubakub import maskfields as mf  # noqa: E402
from kubakub import sound as so  # noqa: E402
from kubakub.types import Regions  # noqa: E402

failures = []


def check(name, cond, detail=""):
    print(("ok   " if cond else "FAIL ") + name + (f"  ({detail})" if detail and not cond else ""))
    if not cond:
        failures.append(name)


# ---- regions: two boxes on a 200 x 100 canvas, the left one 40 x 40, the right one 60 x 20
lab = np.full((100, 200), -1, np.int32)
lab[30:70, 20:60] = 0
lab[40:60, 120:180] = 1
check("boxes of the regions", mf.boxes(lab, 2) == [(20, 30, 40, 40), (120, 40, 60, 20)], str(mf.boxes(lab, 2)))
c, n = mf.compact(np.where(lab == 1, 7, np.where(lab == 0, 3, -1)), [7, 3])
check("compact: ids in the given order, the rest -1", n == 2 and c[50, 150] == 0 and c[50, 40] == 1 and c[0, 0] == -1)
m2 = np.zeros((1, 100, 200), np.float32)
m2[0][lab >= 0] = 1
ls, ns = mf.from_masks(m2, split=True)
check("from_masks: split = one region per shape", ns == 2 and ls[50, 40] != ls[50, 150] and ls[0, 0] == -1)
check("from_masks: no split = one region per mask", mf.from_masks(m2, split=False)[1] == 1)
sl, sn, names = mf.sample_regions(960, 540)
check("sample: the windows of the sample facade", sn == len(names) == 27 and (sl >= 0).any())

# ---- fields
f, ins = mf.field(lab, 2, "edge distance")
check("edge distance: 0 on the edge, 1 in the middle, 0 outside", f[30, 20] == 0 and f[49, 39] > 0.9 and f[0, 0] == 0
      and abs(f[40:60, 120:180].max() - 1) < 1e-6 and ins.sum() == 2800, f"{f[30, 20]} {f[49, 39]}")
ft, _ = mf.field(lab, 2, "edge distance", "all together")
check("edge distance, all together: the thin region stays darker", abs(ft[30:70, 20:60].max() - 1) < 1e-6 and ft[40:60, 120:180].max() < 0.6)
f, _ = mf.field(lab, 2, "direction")
check("direction, each region: every region runs 0..1 left to right", f[50, 20] == 0 and f[50, 59] == 1 and f[50, 120] == 0 and f[50, 179] == 1)
f, _ = mf.field(lab, 2, "direction", "all together")
check("direction, all together: one ramp over both", f[50, 20] == 0 and f[50, 179] == 1 and 0.2 < f[50, 59] < 0.3)
f, _ = mf.field(lab, 2, "direction", angle=90.0)
check("direction 90 = top to bottom", f[30, 40] == 0 and f[69, 40] == 1 and abs(f[50, 25] - f[50, 55]) < 1e-5)
f, _ = mf.field(lab, 2, "radial")
check("radial, each region: 0 at its own centre", f[50, 40] < 0.05 and f[30, 20] > 0.95 and f[50, 150] < 0.05)
f, _ = mf.field(lab, 2, "radial", "all together", centre=(0.0, 0.5))
check("radial, all together: from the point", f[50, 20] < f[50, 59] < f[50, 120] < f[50, 179] and f[40, 179] == 1)
f, _ = mf.field(lab, 2, "region order", order="left")
g, _ = mf.field(lab, 2, "region order", order="right")
check("region order: one flat value per region, in order", f[50, 40] == 0.25 and f[50, 150] == 0.75 and g[50, 40] == 0.75
      and len(np.unique(f)) == 3)
f, _ = mf.field(sl, sn, "noise", noise_px=40, seed=3)
h = np.histogram(f[sl >= 0], bins=4, range=(0, 1))[0] / (sl >= 0).sum()
check("noise: every grey about equally often, another seed another pattern", h.min() > 0.2 and h.max() < 0.3
      and not np.array_equal(f, mf.field(sl, sn, "noise", noise_px=40, seed=4)[0]), str(h))
f, _ = mf.field(lab, 2, "direction", invert=True)
check("invert: the other way, outside stays 0", f[50, 20] == 1 and f[50, 59] == 0 and f[0, 0] == 0)

# ---- curves
check("ramp: once, then it stays", [mf.shape("ramp", u) for u in (0, 0.5, 1, 3)] == [0, 0.5, 1, 1])
check("saw / triangle / sine / square", mf.shape("saw", 1.25) == 0.25 and mf.shape("triangle", 0.5) == 1 and mf.shape("triangle", 1.75) == 0.5
      and abs(mf.shape("sine", 0.5) - 1) < 1e-9 and mf.shape("sine", 1) < 1e-9
      and mf.shape("square", 0.2, duty=0.3) == 1 and mf.shape("square", 0.4, duty=0.3) == 0)
r = [mf.shape("random", u, seed=5) for u in (0.1, 0.9, 1.1, 2.5)]
check("random: one value per cycle", r[0] == r[1] and len(set(r)) == 3 and all(0 <= x < 1 for x in r))
nz = [mf.shape("noise", u / 10, seed=2) for u in range(100)]
check("noise: 0..1, moves softly", 0 <= min(nz) and max(nz) <= 1 and max(nz) - min(nz) > 0.3 and max(abs(a - b) for a, b in zip(nz, nz[1:])) < 0.35)
check("easing", mf.ease(0.5, "ease in") == 0.25 and mf.ease(0.5, "ease out") == 0.75 and mf.ease(0.5, "ease in out") == 0.5)
t = [i / 10 for i in range(41)]
v, _ = mf.curve_values(t, "saw", cycle=2.0, lo=-0.5, hi=0.5)
check("curve_values: from / to and the cycle", v[0] == -0.5 and abs(v[10]) < 1e-9 and abs(v[19] - 0.45) < 1e-9 and v[20] == -0.5)
v, _ = mf.curve_values(t, "saw", cycle=2.0, phase=0.5)
check("phase: half a cycle later", abs(v[0] - 0.5) < 1e-9)
tr = [0.0, 1.0, 2.0]
check("triggers: each one runs the ramp again", mf.progress_triggers(0.25, tr, 0.5) == 0.5 and mf.progress_triggers(0.9, tr, 0.5) == 1
      and mf.progress_triggers(1.0, tr, 0.5) == 0 and mf.progress_triggers(-0.1, tr, 0.5) == 0)
check("triggers with steps: each one moves on by a step", abs(mf.progress_triggers(0.9, tr, 0.5, steps=4) - 0.25) < 1e-9
      and abs(mf.progress_triggers(1.9, tr, 0.5, steps=4) - 0.5) < 1e-9)
SR = 44100
x = so.sample_sound(4.0, 120.0, SR)
ctx, lines = mf.sound_context(x, SR, "beats", sample=True)
bt = mf.triggers("beats", ctx, 0.0, 3.99)
check("beats of the built-in sound: every 0.5 s, the one on 0 counts", len(bt) == 8 and bt[0] == 0 and abs(bt[1] - 0.5) < 1e-6, str(bt[:3]))
check("bars = every 4th beat, nth", len(mf.triggers("bars", ctx, 0, 3.99)) == 2 and len(mf.triggers("beats", ctx, 0, 3.99, nth=2)) == 4)
ctx, _ = mf.sound_context(x, SR, "low hits", sample=True)
lh = mf.triggers("low hits", ctx, 0.0, 3.99)
check("low hits = the kicks", len(lh) == 8 and all(abs(a - k * 0.5) < 0.03 for k, a in enumerate(lh)), str(lh))
ctx, _ = mf.sound_context(x, SR, "sound level", sample=True, listen="low")
lv, _ = mf.curve_values([i / 50 for i in range(200)], "sound level", ctx=ctx, listen="low")
check("sound level (low): up on a kick, down between", max(lv[24:30]) > 0.6 and lv[20] < 0.3 and 0 <= min(lv) and max(lv) <= 1, f"{max(lv[24:30])} {lv[20]}")

# ---- the mask at one value
fd, _ = mf.field(lab, 2, "direction")
check("reveal: nothing at 0, everything at 1, half at 0.5", mf.along(fd, 0.0).max() == 0 and mf.along(fd, 1.0)[lab >= 0].min() == 1
      and mf.along(fd, 0.5, soft=0.01)[50, 25] == 1 and mf.along(fd, 0.5, soft=0.01)[50, 55] == 0)
check("hide = the other way round", mf.along(fd, 0.0, "hide").min() == 1 and mf.along(fd, 1.0, "hide")[lab >= 0].max() == 0)
b0, b5, b1 = (mf.along(fd, p, "band", soft=0.02, width=0.2) for p in (0.0, 0.5, 1.0))
check("band: empty at both ends, a stripe in the middle", b0.max() == 0 and b1[lab >= 0].max() == 0 and b5[50, 40] == 1 and b5[50, 22] == 0 and b5[50, 57] == 0)
r0, r1 = mf.along(fd, 0.0, "rings", rings=4, width=0.05), mf.along(fd, 1.0, "rings", rings=4, width=0.05)
check("rings: the same at 0 and 1 (loops)", np.abs(r0 - r1).max() < 1e-4 and 0 < r0[lab >= 0].mean() < 0.5)
box = np.zeros((100, 200), np.float32)
box[40:60, 20:40] = 1
check("move x: half a picture to the right", mf.moved(box, 0.5, "move x")[50, 130] == 1 and mf.moved(box, 0.5, "move x")[50, 30] == 0)
check("move y / wrap", mf.moved(box, 0.2, "move y")[70, 30] == 1 and mf.moved(box, 1.0, "move x", wrap=True)[50, 30] == 1
      and mf.moved(box, 1.0, "move x").max() == 0)
tall = np.zeros((101, 201), np.float32)
tall[20:81, 95:106] = 1
q = mf.moved(tall, 0.25, "rotate", mf.centre_of(tall))
check("rotate a quarter turn around the mask centre: tall becomes wide", q[50, 75] == 1 and q[25, 100] == 0 and abs(q.sum() - tall.sum()) < 20)
check("scale 2 around the centre, scale 0 = nothing", abs(mf.moved(tall, 2.0, "scale", mf.centre_of(tall)).sum() / tall.sum() - 4) < 0.7
      and mf.moved(tall, 0.0, "scale").max() == 0)
check("opacity", abs(mf.moved(box, 0.3, "opacity").max() - 0.3) < 1e-6)

# ---- animate
ins = (lab >= 0).astype(np.float32)[None]
out = mf.animate(ins, fd, [0.0, 0.5, 1.0], "reveal", soft=0.01)
check("animate along a field: limited to the mask", out.shape == (3, 100, 200) and out[0].max() == 0 and np.array_equal(out[2], ins[0]))
vals, _ = mf.curve_values([i / 10 for i in range(10)], "saw", cycle=1.0, lo=0.0, hi=0.5)
mv = mf.animate(box[None], None, vals, "move x")
tr = mf.animate(box[None], None, vals, "move x", trail=0.5, fps=10)
check("trail: where the mask was still glows, fading", mv[5][50, 30] == 0 and 0 < tr[5][50, 30] < 1 and tr[5][50, 30] < tr[2][50, 30]
      and np.array_equal(tr[0], mv[0]))
ch = mf.animate(mv, None, [1.0, 0.0] * 5, "opacity")
check("a mask batch in: frame i takes mask i (two moves in a chain)", np.array_equal(ch[4], mv[4]) and ch[5].max() == 0)

# ---- the nodes
res = nm.KUBA_MaskField.execute()
check("mask field node, nothing connected: the sample, with a note", res.result[0].shape == (1, 1080, 1920) and "sample" in res.result[2]
      and float(res.result[0].max()) == 1.0 and res.result[1].shape == (1, 1080, 1920))
regs = Regions(torch.from_numpy(lab)[None], {"regions": [{"region_id": 0, "name": "a_01", "group_id": "a", "bbox": [20, 30, 40, 40]},
                                                          {"region_id": 1, "name": "b_01", "group_id": "b", "bbox": [120, 40, 60, 20]}]})
res = nm.KUBA_MaskField.execute(regions=regs, select="b_*", field="direction")
check("mask field node: select picks regions", float(res.result[1][0, 50, 150]) == 1 and float(res.result[1][0, 50, 40]) == 0
      and float(res.result[0][0, 50, 179]) == 1)
try:
    nm.KUBA_MaskField.execute(regions=regs, select="nothing_*")
    check("mask field node: a select that matches nothing is refused", False)
except ValueError as e:
    check("mask field node: a select that matches nothing is refused", "matches no region" in str(e))
res = nm.KUBA_MaskAnimate.execute(seconds=1.0, fps=10.0, scale=0.25)
check("mask animate node, nothing connected: the sample", res.result[0].shape == (10, 270, 480) and res.result[1] == 10.0
      and res.result[3] == 10 and res.result[2]["waveform"].shape == (1, 1, 44100) and "sample" in res.result[4])
check("mask animate node: an animated preview in the temp folder",
      res.ui and res.ui["animated"] == (True,) and os.path.isfile(os.path.join(TMP, "temp", res.ui["images"][0]["filename"])))
fld = torch.from_numpy(fd)[None]
res = nm.KUBA_MaskAnimate.execute(mask=torch.from_numpy(ins), field=fld, effect="band", curve="beats", cycle=0.4, fps=20.0)
check("mask animate node, beats without a sound: the built-in beat, its length, its sound out",
      res.result[0].shape[0] == 160 and res.result[2]["waveform"].shape[-1] == 8 * 44100 and "120.0 bpm" in res.result[4], res.result[4])
k = res.result[0].reshape(160, -1).sum(1)
check("... a band runs after every beat and is gone before the next", float(k[4]) > 0 and float(k[9]) == 0 and float(k[14]) > 0)
res = nm.KUBA_MaskAnimate.execute(mask=torch.from_numpy(box)[None], effect="move x", curve="sine", value_from=-0.1, value_to=0.1,
                                  cycle=1.0, seconds=2.0, fps=10.0)
cx = [float((m.sum(0) * torch.arange(200)).sum() / m.sum()) for m in res.result[0]]
check("mask animate node: move x on a sine swings around", abs(cx[0] - 9.5) < 1 and abs(cx[5] - 49.5) < 1 and abs(cx[10] - 9.5) < 1, str(cx[:11]))
res = nm.KUBA_MaskAnimate.execute(mask=torch.from_numpy(ins), effect="reveal", seconds=0.5, fps=10.0)
check("mask animate node: reveal without a field uses a left to right ramp", "no field" in res.result[4] and float(res.result[0][0].max()) == 0)
first = nm.KUBA_MaskAnimate.execute(mask=torch.from_numpy(ins), field=fld, effect="reveal", seconds=1.0, fps=10.0).result[0]
res = nm.KUBA_MaskAnimate.execute(mask=first, effect="band", curve="saw", cycle=0.5, seconds=1.0, fps=10.0)
check("two mask animate nodes in a chain: a batch that starts empty still gets its ramp (from everywhere it ever is)",
      float(first[0].max()) == 0 and res.result[0].shape[0] == 10 and float(res.result[0].max()) > 0.5, res.result[4])
sch = nm.KUBA_MaskAnimate.define_schema()
check("schemas: names, category, every input and output has a tooltip",
      sch.display_name == "kubakub mask animate" and sch.category == "kubakub/2d/masks"
      and nm.KUBA_MaskField.define_schema().category == "kubakub/2d/masks"
      and all(getattr(i, "tooltip", None) for s in (sch, nm.KUBA_MaskField.define_schema()) for i in list(s.inputs) + list(s.outputs)))
js = open(os.path.join(os.path.dirname(HERE), "web", "kubakub_mask_animate.js"), encoding="utf-8").read()
check("the curve on the node knows every curve of the node", all(f'"{c}"' in js for c in mf.CURVES))
import json as _json  # noqa: E402
import re as _re  # noqa: E402
import subprocess  # noqa: E402
node_exe = shutil.which("node")
if not node_exe:
    print("skip the curve twin: node.js not found")
else:
    engine = js[js.index("function hash01"):js.index("// 0..1 at t seconds")].replace("export function", "function")
    cases = [[k, u / 8, es, 0.3, 5] for k in mf.CURVES[:7] for u in range(0, 41) for es in ("linear", "ease in", "ease out", "ease in out")]
    tmp = os.path.join(HERE, "_curve_twin.cjs")
    open(tmp, "w", encoding="utf-8").write(engine + "\nconsole.log(JSON.stringify(" + _json.dumps(cases) + ".map(c => shape(...c))));\n")
    try:
        res = subprocess.run([node_exe, tmp], capture_output=True, text=True, timeout=60)
    finally:
        os.remove(tmp)
    got = _json.loads(res.stdout) if res.returncode == 0 else []
    check(f"the curve drawn on the node = the engine ({len(cases)} cases)", len(got) == len(cases)
          and max(abs(a - mf.shape(*c)) for a, c in zip(got, cases)) < 1e-12, res.stderr[-300:])

shutil.rmtree(TMP, ignore_errors=True)
print()
print("all mask field tests passed" if not failures else f"{len(failures)} FAILED: " + ", ".join(failures))
sys.exit(1 if failures else 0)
