"""
Model free test of the director's content caches and flow fixes: the sequence / export are kept per content (an apply
that only moves the playhead or changes window state hands the same frames on), skip_h3_frames only with a keyframe
clips (h3) node after the sequence, preview state per node + workflow, imported files decoded once, proxies written
once, the next window of video frames decoded in the background (same frames as decoding each frame on its own).

Run from the ComfyUI_windows_portable folder:
    python_embeded\\python.exe ComfyUI\\custom_nodes\\ComfyUI_KubaNodes\\tests\\test_director_cache.py
"""

import glob
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import types

import cv2
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(HERE))))     # ComfyUI (folder_paths, comfy_api)
sys.path.insert(0, os.path.dirname(HERE))
import folder_paths  # noqa: E402

TMP = tempfile.mkdtemp(prefix="kkd_cache_")
folder_paths.set_temp_directory(os.path.join(TMP, "temp"))
folder_paths.set_output_directory(os.path.join(TMP, "output"))
folder_paths.set_input_directory(os.path.join(TMP, "input"))
UP = os.path.join(TMP, "input", "kuba_director")
os.makedirs(UP, exist_ok=True)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))     # embedded Python leaves the script folder off the path

import _pack  # noqa: E402,F401  (the pack as a package, see tests/_pack.py)

import kubapack.nodes.director.nodes_director as nd  # noqa: E402
from kubakub.director import media as md  # noqa: E402
from kubakub.director import render as rd  # noqa: E402

failures = []


def check(name, cond, detail=""):
    print(("ok   " if cond else "FAIL ") + name + (f"  ({detail})" if detail else ""))
    if not cond:
        failures.append(name)


def hidden(uid="5", wf=None, prompt=None):
    return types.SimpleNamespace(unique_id=uid, prompt=prompt, extra_pnginfo={"workflow": {"id": wf}} if wf else None)


W, H = 240, 150
rng = np.random.default_rng(3)
yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
base = torch.from_numpy(np.stack([xx / W, yy / H, np.full_like(xx, 0.3)], -1))[None]
imp = (rng.random((60, 80, 4)) * 255).astype(np.uint8)
imp[..., 3] = 220
cv2.imwrite(os.path.join(UP, "imp.png"), imp)
doc = {"version": 1, "canvas": [W, H], "timeline": {"duration": 2.0, "fps": 10, "time": 0.4, "snap": True,
                                                    "clips": [{"id": "c1", "t_start": 0.0, "t_end": 2.0, "prompt": "a"}],
                                                    "h3": {"look": "x"}},
       "diffusion": {"steps": 4}, "dismissed": ["hint"],
       "layers": [{"id": "sq", "name": "square", "kind": "shape", "shape": {"type": "rect"}, "color": "#ff8800",
                   "x": 10, "y": 20, "w": 60, "h": 40, "anim": {"x": [{"t": 0, "v": 10}, {"t": 2, "v": 150}]},
                   "motion": [{"id": "m", "type": "wiggle", "path": "rotation", "amount": 8, "freq": 2, "seed": 1, "on": True}]},
                  {"id": "im", "name": "imp", "kind": "image", "source": "file:kuba_director/imp.png", "x": 90, "y": 30, "w": 80, "h": 60},
                  {"id": "base", "kind": "base", "name": "clay"}]}


def director(d, uid="5", wf=None, **kw):
    nd.KUBA_Director.hidden = hidden(uid, wf)
    return nd.KUBA_Director.execute(image=base, document=json.dumps(d), **kw)


def sequence(state, prompt=None, uid="9", **kw):
    nd.KUBA_DirectorSequence.hidden = hidden(uid, prompt=prompt)
    return nd.KUBA_DirectorSequence.execute(director=state, **kw)


def variant(**tl):
    d = json.loads(json.dumps(doc))
    d["timeline"].update(tl)
    return d


# ------------------------------------------------------------------ what counts as content
k0 = nd.pixel_doc(doc, True)
ui_only = json.loads(json.dumps(doc))
ui_only["timeline"].update(time=1.7, snap=False, h3={"look": "y"})
ui_only["timeline"]["clips"][0]["prompt"] = "another prompt"
ui_only["diffusion"] = {"steps": 8}
ui_only["dismissed"] = []
check("pixel_doc: playhead, snap, H3 texts, clip prompt, diffusion, dismissed do not count",
      nd.pixel_doc(ui_only, True) == k0 and nd.pixel_doc(ui_only, False) == nd.pixel_doc(doc, False))
check("pixel_doc: clip timing counts when skipping, clips do not count without skipping",
      nd.pixel_doc(variant(clips=[{"id": "c1", "t_start": 0.5, "t_end": 2.0}]), True) != k0
      and nd.pixel_doc(variant(clips=[]), False) == nd.pixel_doc(doc, False))
check("pixel_doc: layers, duration, fps, sound count", all(nd.pixel_doc(d_, True) != k0 for d_ in (
    variant(duration=3.0), variant(fps=12), variant(audio={"file": "x.wav"}),
    {**doc, "layers": doc["layers"][1:]})))

# ------------------------------------------------------------------ the sequence cache
nd._SEQ_CACHE.clear()
r1 = director(doc)
s1 = sequence(r1.result[11], frames=True, sequence_scale=0.5)
r2 = director(ui_only)                                      # an apply that only moved the playhead etc.
check("the director re-renders the still at the new playhead", not torch.equal(r1.result[0], r2.result[0]))
t = time.perf_counter()
s2 = sequence(r2.result[11], frames=True, sequence_scale=0.5)
check("unchanged content: the same frames tensor, no render", s2.result[0] is s1.result[0]
      and "sequence unchanged" in s2.result[3], s2.result[3].split("\n")[0])
s3 = sequence(r2.result[11], frames=True, sequence_scale=1.0)
check("another scale renders", s3.result[0] is not s1.result[0] and s3.result[0].shape[1] == H)
nd.SEQ_CACHE_ITEMS, keep = 0, nd.SEQ_CACHE_ITEMS             # cache off: a fresh render gives the same pixels
fresh = sequence(r2.result[11], frames=True, sequence_scale=0.5)
nd.SEQ_CACHE_ITEMS = keep
check("cached frames = a fresh render of the new document", torch.equal(fresh.result[0], s1.result[0]))
moved = json.loads(json.dumps(doc))
moved["layers"][1]["x"] = 100
s4 = sequence(director(moved).result[11], frames=True, sequence_scale=0.5)
check("a moved layer renders again", not torch.equal(s4.result[0], s1.result[0]) and "unchanged" not in s4.result[3])
check("at most SEQ_CACHE_ITEMS results kept", len(nd._SEQ_CACHE) <= nd.SEQ_CACHE_ITEMS)
cv2.imwrite(os.path.join(UP, "imp.png"), 255 - imp)          # the imported file changes on disk, the document not
os.utime(os.path.join(UP, "imp.png"), (time.time() + 5, time.time() + 5))
s5 = sequence(director(doc).result[11], frames=True, sequence_scale=0.5)
check("an edited import renders again", not torch.equal(s5.result[0], s1.result[0]) and "unchanged" not in s5.result[3])
cv2.imwrite(os.path.join(UP, "imp.png"), imp)
os.utime(os.path.join(UP, "imp.png"), (time.time() + 10, time.time() + 10))

# legacy render_sequence path uses the same cache
leg1 = director(doc, render_sequence=True, sequence_scale=0.5)
leg2 = director(ui_only, render_sequence=True, sequence_scale=0.5)
check("legacy render_sequence: unchanged content hands on the same frames", leg2.result[7] is leg1.result[7]
      and "sequence unchanged" in leg2.result[6])

# ------------------------------------------------------------------ export: written once per content + settings
root = os.path.join(TMP, "output", "kubakub_director")
st = director(doc).result[11]
e1 = sequence(st, frames=False, export="png8", export_scale=0.5, export_name="cachetest")
time.sleep(1.1)                                             # export folders are named by the second
e2 = sequence(director(ui_only).result[11], frames=False, export="png8", export_scale=0.5, export_name="cachetest")
dirs = glob.glob(os.path.join(root, "cachetest_*"))
check("export: an unchanged composition is not written again", len(dirs) == 1 and "not written again" in e2.result[3], e2.result[3])
e3 = sequence(st, frames=False, export="png8", export_scale=0.25, export_name="cachetest")
check("export: other settings write a new export", len(glob.glob(os.path.join(root, "cachetest_*"))) == 2, e3.result[3])
shutil.rmtree(dirs[0])
time.sleep(1.1)
e4 = sequence(st, frames=False, export="png8", export_scale=0.5, export_name="cachetest")
check("export: written again when the earlier export was deleted", "not written again" not in e4.result[3]
      and len(glob.glob(os.path.join(root, "cachetest_*"))) == 2)

# ------------------------------------------------------------------ M1: skip_h3_frames only with keyframe clips (h3)
schema = nd.KUBA_DirectorSequence.define_schema()
check("skip_h3_frames defaults to off", next(i for i in schema.inputs if i.id == "skip_h3_frames").default is False)
graph_plain = {"3": {"class_type": "KUBA_Director", "inputs": {}},
               "9": {"class_type": "KUBA_DirectorSequence", "inputs": {"director": ["3", 11]}},
               "10": {"class_type": "CreateVideo", "inputs": {"images": ["9", 0]}}}
graph_h3 = {**graph_plain, "11": {"class_type": "KUBA_KeyframeClips", "inputs": {"frames": ["9", 0]}}}
check("h3_after: finds keyframe clips after the node, not before / elsewhere",
      nd.h3_after(graph_h3, "9") is True and nd.h3_after(graph_plain, "9") is False and nd.h3_after(None, "9") is None)
long_doc = variant(duration=4.0, clips=[{"id": "c1", "t_start": 0.0, "t_end": 4.0}])   # 40 frames: 13-26 hidden
st_l = director(long_doc).result[11]
full = sequence(st_l, frames=True, sequence_scale=0.5, skip_h3_frames=False)
no_h3 = sequence(st_l, prompt=graph_plain, frames=True, sequence_scale=0.5, skip_h3_frames=True)
with_h3 = sequence(st_l, prompt=graph_h3, frames=True, sequence_scale=0.5, skip_h3_frames=True)
check("skip on without a keyframe clips node: every frame rendered", torch.equal(no_h3.result[0], full.result[0])
      and "no kubakub keyframe clips (h3)" in no_h3.result[3])
check("skip on with a keyframe clips node: placeholders under the clip", not torch.equal(with_h3.result[0], full.result[0])
      and "placeholders" in with_h3.result[3])
no_clip = sequence(director(variant(clips=[])).result[11], prompt=graph_h3, frames=True, sequence_scale=0.5, skip_h3_frames=True)
check("no H3 clips: nothing skipped, no warning", "placeholders" not in no_clip.result[3] and "keyframe clips" not in no_clip.result[3])

# ------------------------------------------------------------------ M3: preview state per node + workflow
check("state_key: node id alone without a workflow id, else node + workflow",
      nd.state_key("3") == "3" and nd.state_key("3", {"workflow": {"id": "a"}}) != nd.state_key("3", {"workflow": {"id": "b"}})
      and nd.state_key("12:3").startswith("12_3"))
ra = director(doc, uid="3", wf="wf-a")
rb = director(variant(time=1.0), uid="3", wf="wf-b")
ka, kb = ra.ui["kuba_director"][0]["node"], rb.ui["kuba_director"][0]["node"]
check("two workflows with director node 3: two keys, two folders, two preview states",
      ka != kb and ka in nd._LAST and kb in nd._LAST and os.path.isdir(os.path.join(TMP, "temp", "kuba_director", ka))
      and os.path.isdir(os.path.join(TMP, "temp", "kuba_director", kb)))
old = os.path.join(TMP, "temp", "kuba_director", ka, "in_0_1.png")
open(old, "wb").write(b"x")
os.utime(old, (time.time() - 7200, time.time() - 7200))
director(doc, uid="3", wf="wf-b")
check("the hourly purge of one workflow leaves the other's files alone", os.path.isfile(old))
director(doc, uid="3", wf="wf-a")
check("... and cleans its own", not os.path.isfile(old))
for i in range(nd.LAST_MAX + 3):
    nd._remember(f"x{i}", {})
check("preview states are bounded", len(nd._LAST) <= nd.LAST_MAX)

# ------------------------------------------------------------------ imports decoded once, proxies written once
a1 = nd._load_input_file("kuba_director/imp.png")
a2 = nd._load_input_file("kuba_director/imp.png")
check("an import is decoded once per file version (shared, read only)", a1 is a2 and not a1.flags.writeable)
cv2.imwrite(os.path.join(UP, "imp.png"), 255 - imp)
os.utime(os.path.join(UP, "imp.png"), (time.time() + 20, time.time() + 20))
a3 = nd._load_input_file("kuba_director/imp.png")
check("an edited import is read again", a3 is not a1 and not np.array_equal(a3, a1))
lay = torch.from_numpy(rng.random((1, 30, 40, 3)).astype(np.float32))
d_in = director(doc, uid="p1", layers=lay)
folder = os.path.join(TMP, "temp", "kuba_director", "p1")
before = sorted(os.listdir(folder))
time.sleep(0.01)
director(doc, uid="p1", layers=lay)
check("unchanged inputs / result: no new proxy files", sorted(os.listdir(folder)) == before, str(sorted(set(os.listdir(folder)) - set(before))))
check("director state tells ComfyUI's cache its size", sum(t_.untyped_storage().nbytes() for t_ in d_in.result[11]._comfy_cache_tensors())
      >= base.numel() * 4)

# ------------------------------------------------------------------ video windows decoded ahead = decoded per frame
try:
    exe = md.ffmpeg_exe()
except Exception as e:  # noqa: BLE001
    exe = None
    print(f"skip video checks (no ffmpeg: {e})")
if exe:
    N = 70
    fr = np.zeros((N, 32, 48, 4), np.uint8)
    for k in range(N):
        fr[k, ..., :3] = [(k * 7) % 256, (k * 3 + 40) % 256, 255 - k]
        fr[k, ..., 3] = 255
    subprocess.run([exe, "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgba", "-s", "48x32", "-r", "10", "-i", "-",
                    "-c:v", "png", "-pix_fmt", "rgba", os.path.join(UP, "v.mov")], input=fr.tobytes(), check=True)
    subprocess.run([exe, "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgba", "-s", "48x32", "-r", "10", "-i", "-",
                    "-c:v", "png", "-pix_fmt", "rgba", os.path.join(UP, "w.mov")], input=fr[::-1].copy().tobytes(), check=True)
    vdoc = {"version": 1, "canvas": [W, H], "timeline": {"duration": 6.5, "fps": 10, "time": 0},
            "layers": [{"id": "v", "name": "v", "kind": "image", "source": "file:kuba_director/v.mov", "x": 10, "y": 10, "w": 96, "h": 64},
                       {"id": "w", "name": "w", "kind": "image", "source": "file:kuba_director/w.mov", "x": 120, "y": 60, "w": 96, "h": 64,
                        "blend": "screen"},
                       {"id": "base", "kind": "base", "name": "clay"}]}
    vs = director(vdoc).result[11]
    seq = sequence(vs, frames=True, sequence_scale=1.0).result[0]
    n = rd.timeline(vdoc)["frames"]
    docs = {i: (rd.animate(vdoc, i / 10), i / 10) for i in range(n)}
    mf = nd.MediaFrames(docs, 1.0, [])
    ok = True
    for i in (0, 25, 49, 50, 51, 64):                      # both windows and their border
        mf.load([i])                                        # one frame at a time, decoded in this thread
        ref = rd.render(docs[i][0], vs.base, mf.sources(i, vs.sources), None, None, image_only=True, frame_seed=i)["image"]
        ok = ok and np.array_equal(ref[:seq.shape[1], :seq.shape[2]], seq[i].numpy())
    check("video layers: frames with the next window decoded ahead = each frame decoded on its own", ok and seq.shape[0] == n)
    two = mf.decode([3, 4])
    check("two videos decode in parallel into one window", len(two) == 2 and all(len(v) == 2 for v in two.values()))

shutil.rmtree(TMP, ignore_errors=True)
print()
print("all director cache tests passed" if not failures else f"{len(failures)} FAILED: " + ", ".join(failures))
sys.exit(1 if failures else 0)
