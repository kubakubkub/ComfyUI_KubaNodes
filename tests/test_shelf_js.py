"""
The list logic of the kubakub shelf (web/kubakub_shelf.js): adding, the same thing again, the cap that keeps pins,
text vs colour, node previews -> sources, the values the core loaders take. Run with node when it is installed.

    python tests/test_shelf_js.py
Without node the checks are skipped and said so.
"""

import os
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
const o = sourceFromView("http://127.0.0.1:8188/api/view?filename=a.png&subfolder=kubakub_show&type=output");
check("an output preview -> output:sub/name", o.source === "output:kubakub_show/a.png", o);
check("an input preview -> file:name", sourceFromView("/view?filename=b.mp4&type=input").source === "file:b.mp4");
check("a temp preview is copied in", !!sourceFromView("/view?filename=c.png&type=temp").temp);
check("a link of another server -> null (the shelf never fetches from the network)",
  sourceFromView("https://other.host/x/view?filename=a.png&type=temp") === null && sourceFromView("http://h/api/view?filename=a.png&type=output") === null);
check("names without file: / output:", fileName("file:b.png") === "b.png" && fileName("output:kubakub_show/x.mp4") === "x.mp4");
check("not a picture / video / sound -> null", sourceFromView("/view?filename=c.json&type=output") === null && sourceFromView("/other?filename=a.png") === null);

// what the core loaders take
check("loader value for input/", loaderValue("file:kuba_director/a.png") === "kuba_director/a.png");
check("loader value for output/", loaderValue("output:kubakub_show/a.mp4") === "kubakub_show/a.mp4 [output]");
check("sourceURL round trip", sourceFromView(sourceURL("output:kubakub_show/a.png")).source === "output:kubakub_show/a.png");

// folders dropped from the desktop: fake file system entries (readEntries hands out a few at a time, as browsers do)
const fileE = name => ({ isFile: true, isDirectory: false, name, file: (ok) => ok({ name, size: 3, text: async () => "warm windows" }) });
const dirE = (name, kids) => ({ isFile: false, isDirectory: true, name, createReader() { let i = 0; return { readEntries(ok) { const b = kids.slice(i, i + 2); i += b.length; setTimeout(() => ok(b)); } }; } });
(async () => {
  const drop = [dirE("facade night", [fileE("b.png"), fileE("a10.png"), fileE("a2.png"), fileE("notes.txt"), fileE("x.psd"),
    dirE("sub", [fileE("c.mp4"), fileE(".hidden.png")]), dirE(".cache", [fileE("d.png")])]), fileE("loose.wav"), fileE("readme.pdf")];
  const w = await walkEntries(drop);
  check("a folder walk takes pictures, videos, sounds and text, deep", w.files.length === 6, w.files.map(f => f.entry.name));
  check("the walk counts what it skips (psd, pdf), hidden files and folders are left out", w.skipped === 2, w.skipped);
  check("files keep the dropped folder's name, loose ones none", w.files.find(f => f.entry.name === "c.mp4").top === "facade night" && w.files.find(f => f.entry.name === "loose.wav").top === "");
  check("numbered names in number order", w.files.map(f => f.entry.name).slice(0, 3).join() === "a2.png,a10.png,b.png", w.files.map(f => f.entry.name));
  const capped = await walkEntries(drop, 3);
  check("the cap stops the walk and says so", capped.files.length === 3 && capped.capped);
  const plan = dropPlan(w);
  check("drop plan: one set per folder, loose files, texts", plan.sets.length === 1 && plan.sets[0].name === "facade night" && plan.sets[0].files.length === 4 && plan.loose.length === 1 && plan.texts.length === 1, plan);
  check("entryFile reads the file", (await entryFile(fileE("q.png"))).name === "q.png");
  let ticks = 0, live = 0, most = 0;
  const res = await inBatches([1, 2, 3, 4, 5, 6, 7], 3, async x => { live++; most = Math.max(most, live); await new Promise(r => setTimeout(r, 5)); live--; if (x === 4) throw new Error("bad"); return x * 2; }, () => ticks++);
  check("batches: order kept, at most n at once, a failure does not stop the rest", res[0] === 2 && res[6] === 14 && res[3].error && most === 3 && ticks === 7, { res, most, ticks });
  const set = { kind: "set", text: "facade night", name: "facade night", count: 4, cover: "file:kuba_director/facade night/a2.png" };
  L = []; addItem(L, set); addItem(L, Object.assign({}, set));
  check("the same folder dropped again stays one card", L.length === 1);

  // this session's runs (ComfyUI /history)
  const hist = {
    p1: { prompt: [1, "p1"], outputs: { "9": { images: [{ filename: "old.png", subfolder: "", type: "output" }] } } },
    p2: { prompt: [2, "p2"], outputs: { "9": { images: [{ filename: "new.png", subfolder: "show", type: "output" }, { filename: "prev.png", subfolder: "", type: "temp" }] },
      "12": { gifs: [{ filename: "clip.mp4", subfolder: "", type: "output", format: "video/h264-mp4" }], text: ["not a file"] } } },
    p3: { prompt: [3, "p3"], status: {}, outputs: { "4": { audio: [{ filename: "song.flac", subfolder: "", type: "output" }, { filename: "old.png", subfolder: "", type: "output" }] } } },
    bad: null };
  const hf = historyFiles(hist);
  check("history: saved files newest run first, previews (temp) left out, each file once",
    hf.map(f => f.filename).join() === "song.flac,old.png,new.png,clip.mp4", hf.map(f => f.filename));
  check("history file -> output source", outputSource(hf[2]) === "output:show/new.png" && outputSource(hf[0]) === "output:song.flac");
  process.exit(fails ? 1 : 0);
})().catch(e => { console.log("FAIL async checks " + e.stack); process.exit(1); });
"""

node = shutil.which("node")
if node is None:
    print("skip the shelf checks: node is not installed")
else:
    sys.path.insert(0, HERE)
    from _js import module_source
    src = module_source("kubakub_shelf.js") + "\n" + CHECKS
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
