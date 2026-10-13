"""
The text logic of the 'pick regions' list (web/kubakub_which.js: a choice <-> the selector text), run with node when
it is installed, and the texts it writes read by the plan's own selector (kubakub/plan.py pick_ids).

    python tests/test_which_js.py
Without node the JavaScript part is skipped and said so.
"""

import os
import re
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import _pack  # noqa: E402,F401
from kubapack.kubakub import plan as rp  # noqa: E402

failures = []


def check(name, cond, detail=""):
    print(("ok   " if cond else "FAIL ") + name + (f"  ({detail})" if detail and not cond else ""))
    if not cond:
        failures.append(name)


table = {"regions": [
    {"region_id": 0, "name": "windows_01", "group_id": "windows", "tags": ["floor_1", "left"]},
    {"region_id": 1, "name": "windows_02", "group_id": "windows", "tags": ["floor_2", "left"]},
    {"region_id": 2, "name": "columns_01", "group_id": "columns", "tags": ["floor_1"]},
    {"region_id": 3, "name": "the door", "group_id": "my group", "tags": ["floor 0"]},
]}
for text, want in (("group:windows tag:floor_1, group:columns tag:floor_1", [0, 2]), ("tag:floor_1 tag:left, tag:floor_2 tag:left", [0, 1]),
                   ("group:my?group", [3]), ("name:the?door, windows_02", [1, 3]), ("tag:floor?0", [3])):
    check(f"plan selector reads what the list writes: {text}", rp.pick_ids(table, text) == want, str(rp.pick_ids(table, text)))

node = shutil.which("node")
if node is None:
    print("skip the JavaScript checks: node is not installed")
else:
    with open(os.path.join(HERE, "..", "web", "kubakub_which.js"), encoding="utf-8") as f:
        src = f.read()
    src = re.sub(r"(?m)^import \{ app \} from .*$", "const app = { registerExtension() {} };", src)
    src = re.sub(r"(?m)^import \{ api \} from .*$", "const api = {};", src)
    with open(os.path.join(HERE, "js", "which_checks.mjs"), encoding="utf-8") as f:
        src += "\n" + f.read()
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "which.mjs")
        with open(path, "w", encoding="utf-8") as f:
            f.write(src)
        run = subprocess.run([node, path], capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60)
    print(run.stdout.rstrip())
    check("the list's own checks (node)", run.returncode == 0, run.stderr.strip()[-400:])

print()
print("all which regions tests passed" if not failures else f"{len(failures)} FAILED: " + ", ".join(failures))
sys.exit(1 if failures else 0)
