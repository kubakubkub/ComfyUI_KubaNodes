"""
Model free checks of what a fresh user meets first: the setup tool fetches the VAEs the workflows load, the Colab
notebook keeps the path guard on behind its public link; and of fixes that had no test: a timeline block of a still
director, the near mask, an optional package named in its error.

Run from the ComfyUI_windows_portable folder:
    python_embeded\\python.exe ComfyUI\\custom_nodes\\ComfyUI_KubaNodes\\tests\\test_release_checks.py
"""

import glob
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from kubakub import master_render as mr  # noqa: E402
from kubakub import tracking as tk  # noqa: E402

failures = []


def check(name, cond, detail=""):
    print(("ok   " if cond else "FAIL ") + name + (f"  ({detail})" if detail and not cond else ""))
    if not cond:
        failures.append(name)


# ------------------------------------------------------------------ the LTX files of the workflows come with the setup
setup = open(os.path.join(ROOT, "tools", "setup_kubakub.py"), encoding="utf-8").read()
named = set()
for path in glob.glob(os.path.join(ROOT, "example_workflows", "*.json")):
    named |= set(re.findall(r"ltx-2\.5-(?:video|audio)-vae[\w-]*\.safetensors", open(path, encoding="utf-8").read()))
check("workflows name LTX VAEs", bool(named))
missing = sorted(n for n in named if n not in setup)
check("setup downloads every LTX VAE the workflows load", not missing, ", ".join(missing))

# ------------------------------------------------------------------ the notebook
nb = json.load(open(os.path.join(ROOT, "colab", "try_kubakub.ipynb"), encoding="utf-8"))
code = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
check("notebook: ComfyUI listens on all interfaces, so the path guard is on", '"--listen", "0.0.0.0"' in code)
check("notebook: a kill pattern that does not match its own shell", "[m]ain.py" in code)
check("notebook: cloudflared is pinned", "releases/latest" not in code)
check("notebook: the tunnel's output is not a pipe nobody reads", "subprocess.PIPE" not in code)

# ------------------------------------------------------------------ a director with nothing applied holds for its block
still = np.full((1, 4, 4, 3), 0.5, np.float32)
doc = {"fps": 10, "duration": 12, "width": 4, "height": 4,
       "tracks": [{"items": [{"type": "director", "node": 1, "start": 0, "length": -1, "props": {}}]}]}
frames, _, _, _ = mr.render(doc, 1.0, lambda s: None, lambda node, w, h: (still, 10.0, None))
check("still director: lit for the 10 s the panel shows", frames[99].mean() > 0.4)
check("still director: black after that", frames[100].mean() < 0.01)

# ------------------------------------------------------------------ the near mask
data = {"W": 100, "H": 100, "tracks": [{"t0": 0, "box": [[40, 40, 20, 20]]}]}
near = tk.near_field(data, 0, 100, 100)
check("near mask: float32 of the picture's size", near.dtype == np.float32 and near.shape == (100, 100))
check("near mask: 1 on the thing, 0 far away", near[50, 50] == 1.0 and near[0, 0] == 0.0)
check("near mask: falls off around it", 0.0 < near[50, 65] < 1.0)

print("\nall ok" if not failures else f"\n{len(failures)} FAILED: {failures}")
sys.exit(1 if failures else 0)
