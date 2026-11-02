"""
The list logic of the kubakub shelf (web/kubakub_shelf.js): adding, the same thing again, the cap that keeps pins,
text vs colour, node previews -> sources, the values the core loaders take. Run with node when it is installed.

    python tests/test_shelf_js.py
Without node the checks are skipped and said so.
"""

import os
import re
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
failures = []


def check(name, cond, detail=""):
    print(("ok   " if cond else "FAIL ") + name + (f"  ({detail})" if detail and not cond else ""))
    if not cond:
        failures.append(name)


CHECKS = r"""
let fails = 0;
const check = (name, ok, d) => { console.log((ok ? "ok   " : "FAIL ") + name + (ok ? "" : "  " + JSON.stringify(d))); if (!ok) fails++; };

// adding, again = to the top (no copy), the cap removes the oldest unpinned only
let L = [];
addItem(L, { kind: "text", text: "a", name: "a" }); addItem(L, { kind: "image", source: "file:x/b.png", name: "b.png" });
addItem(L, { kind: "text", text: "a", name: "a" });
check("the same text again moves to the top, no copy", L.length === 2 && L[0].text === "a", L.map(i => i.name));
L = []; addItem(L, { kind: "text", text: "keep", name: "keep" }, 3); L[0].pinned = true;
for (const t of ["1", "2", "3", "4"]) addItem(L, { kind: "text", text: t, name: t }, 3);
check("the cap keeps the pinned one", L.length === 3 && L.some(i => i.text === "keep") && L[0].text === "4", L.map(i => i.text));

// text or colour
check("a hex colour is a colour", textItem(" #F18A58 ").kind === "colour" && textItem("#abc").kind === "colour");
check("rgb() is a colour", textItem("rgb(10, 20, 30)").kind === "colour");
check("a prompt is text, named by its first line", textItem("warm windows\nwet stone").kind === "text" && textItem("warm windows\nwet stone").name === "warm windows");
check("empty text is nothing", textItem("   ") === null);

// node previews
const o = sourceFromView("http://h/api/view?filename=a.png&subfolder=kubakub_show&type=output");
check("an output preview -> output:sub/name", o.source === "output:kubakub_show/a.png", o);
check("an input preview -> file:name", sourceFromView("/view?filename=b.mp4&type=input").source === "file:b.mp4");
check("a temp preview is copied in", !!sourceFromView("/view?filename=c.png&type=temp").temp);
check("not a picture / video / sound -> null", sourceFromView("/view?filename=c.json&type=output") === null && sourceFromView("/other?filename=a.png") === null);

// what the core loaders take
check("loader value for input/", loaderValue("file:kuba_director/a.png") === "kuba_director/a.png");
check("loader value for output/", loaderValue("output:kubakub_show/a.mp4") === "kubakub_show/a.mp4 [output]");
check("viewURL round trip", sourceFromView(viewURL("output:kubakub_show/a.png")).source === "output:kubakub_show/a.png");
process.exit(fails ? 1 : 0);
"""

node = shutil.which("node")
if node is None:
    print("skip the shelf checks: node is not installed")
else:
    with open(os.path.join(HERE, "..", "web", "kubakub_shelf.js"), encoding="utf-8") as f:
        src = f.read()
    src = re.sub(r"(?m)^import \{ app \} from .*$", "const app = { registerExtension() {} };", src)
    src = re.sub(r"(?m)^import \{ api \} from .*$", 'const api = { apiURL: p => "/api" + p };', src)
    src = "globalThis.location = { href: 'http://127.0.0.1:8188/' };\n" + src + "\n" + CHECKS
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "shelf.mjs")
        with open(path, "w", encoding="utf-8") as f:
            f.write(src)
        run = subprocess.run([node, path], capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60)
    print(run.stdout.rstrip())
    check("the shelf's own checks (node)", run.returncode == 0, run.stderr.strip()[-400:])

print()
print("all shelf tests passed" if not failures else f"{len(failures)} FAILED: " + ", ".join(failures))
sys.exit(1 if failures else 0)
