"""
Model free test of the master timeline: the file listing and upload names of its routes (kubakub/master.py), and the
clip model of the panel (web/kubakub_master.js) against the director's (kubakub/director/media.py), run with node.

Run from the ComfyUI_windows_portable folder:
    python_embeded\\python.exe ComfyUI\\custom_nodes\\ComfyUI_KubaNodes\\tests\\test_master.py

Without node the JavaScript part is skipped and said so.
"""

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from kubakub import master as ms  # noqa: E402
from kubakub.director import media as md  # noqa: E402

failures = []


def check(name, cond, detail=""):
    print(("ok   " if cond else "FAIL ") + name + (f"  ({detail})" if detail else ""))
    if not cond:
        failures.append(name)


# names of uploads
check("upload name: folders and odd characters", ms.safe_name("clips\\my:clip?.mp4") == "my_clip_.mp4",
      ms.safe_name("clips\\my:clip?.mp4"))
check("upload name: not a video or picture", ms.safe_name("notes.txt") is None and ms.safe_name("") is None)
check("upload name: types", ms.kind_of("a.MOV") == "video" and ms.kind_of("a.png") == "image" and ms.kind_of("a.txt") is None)

with tempfile.TemporaryDirectory() as d:
    os.makedirs(os.path.join(d, "sub"))
    os.makedirs(os.path.join(d, ".hidden"))
    for i, name in enumerate(["old.mp4", "sub/new.png", ".hidden/proxy.mp4", "notes.txt", "sub/.part.mp4"]):
        p = os.path.join(d, name)
        with open(p, "wb") as f:
            f.write(b"x")
        os.utime(p, (time.time() - 100 + i, time.time() - 100 + i))
    names = [e["name"] for e in ms.list_media(d)]
    check("listing: newest first, sub folders with '/', hidden ones and other types left out",
          names == ["sub/new.png", "old.mp4"], names)
    check("free name", ms.free_name(d, "old.mp4") == "old (1).mp4" and ms.free_name(d, "fresh.mp4") == "fresh.mp4")

    # sources: input/ and output/ only, never outside
    inp, outp = os.path.join(d, "sub"), d
    check("source: file: in input", ms.resolve("file:new.png", inp, outp) == os.path.realpath(os.path.join(inp, "new.png")))
    check("source: output: in output", ms.resolve("output:old.mp4", inp, outp) == os.path.realpath(os.path.join(d, "old.mp4")))
    check("source: no escaping, no other kinds", ms.resolve("file:../old.mp4", inp, outp) is None
          and ms.resolve("path:old.mp4", inp, outp) is None and ms.resolve("output:missing.mp4", inp, outp) is None)

# sounds and their waveform
check("sound files are taken", ms.kind_of("song.MP3") == "audio" and ms.safe_name("a/b/voice.wav") == "voice.wav")
pk = ms.peaks([0.0] * 40 + [0.5, -0.9] + [0.0] * 38, 2000)
check("waveform: the loudest sample of every 1/50 s", pk == [0.0, 0.9], pk)
check("waveform: nothing for silence of no length", ms.peaks([], 2000) == [])

# the panel's clip model = the director's (one segment), probed at many times
segs = [("plain", {}, 8), ("in / out", {"in": 1.25, "out": 4.5}, 8), ("speed", {"speed": 2.5, "start": 1}, 8),
        ("slow", {"speed": 0.3, "in": 2}, 8), ("hold", {"length": 12, "fill": "hold"}, 8),
        ("loop", {"in": 1, "out": 3, "length": 9, "fill": "loop", "start": 0.5}, 8),
        ("backwards", {"reverse": True, "in": 0.5, "out": 6}, 8),
        ("backwards loop", {"reverse": True, "fill": "loop", "length": 20}, 8),
        ("out past the file", {"out": 99}, 5), ("in past the file", {"in": 50}, 5),
        ("odd speed", {"speed": 1.37, "in": 0.33}, 7.92)]
cases = []
for name, seg, dur in segs:
    s = md.segments({"segments": [seg]}, dur)
    ts = [round(-0.5 + k * 0.137, 4) for k in range(170)]
    want = []
    for t in ts:
        r = md.source_time(s, t)
        want.append(None if r is None else r[1])
    cases.append({"name": name, "seg": seg, "dur": dur, "t": ts, "want": want})

node = shutil.which("node")
if node is None:
    print("skip the JavaScript checks: node is not installed")
else:
    with open(os.path.join(HERE, "..", "web", "kubakub_master.js"), encoding="utf-8") as f:
        src = f.read()
    src = re.sub(r"(?m)^import \{ app \} from .*$", "const app = { registerExtension() {} };", src)
    src = re.sub(r"(?m)^import \{ api \} from .*$", "const api = { addEventListener() {} };", src)
    src += "\nconst CASES = " + json.dumps(cases) + ";\n"
    with open(os.path.join(HERE, "js", "master_checks.mjs"), encoding="utf-8") as f:
        src += f.read()
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "master.mjs")
        with open(path, "w", encoding="utf-8") as f:
            f.write(src)
        run = subprocess.run([node, path], capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60)
    print(run.stdout.rstrip())
    check("the panel's own checks (node)", run.returncode == 0, run.stderr.strip()[-400:])

print()
print("all master timeline tests passed" if not failures else f"{len(failures)} FAILED: " + ", ".join(failures))
sys.exit(1 if failures else 0)
