"""
Model free test of kubakub crop / uncrop / compare (kubakub/crop.py, nodes/utils/nodes_crop.py, nodes_compare.py),
the director's layer crop (kubakub/director/render.py), and that the crop widget (web/kubakub_crop.js) finds the same
pixels as the node (run with node when it is installed; without node that part is skipped and said so).

Run from the ComfyUI_windows_portable folder:
    python_embeded\\python.exe ComfyUI\\custom_nodes\\ComfyUI_KubaNodes\\tests\\test_crop.py
"""

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from fractions import Fraction
from types import SimpleNamespace

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(HERE))))     # ComfyUI (folder_paths, comfy_api)
sys.path.insert(0, os.path.dirname(HERE))
import folder_paths  # noqa: E402

TMP = tempfile.mkdtemp(prefix="kub_crop_")
folder_paths.set_temp_directory(os.path.join(TMP, "temp"))

sys.path.insert(0, HERE)
import _pack  # noqa: E402,F401
from kubapack.kubakub import crop as cr  # noqa: E402
from kubapack.kubakub.director import render as rd  # noqa: E402
from kubapack.nodes.utils import nodes_compare as ncm  # noqa: E402
from kubapack.nodes.utils import nodes_crop as ncr  # noqa: E402

failures = []


def check(name, cond, detail=""):
    print(("ok   " if cond else "FAIL ") + name + (f"  ({detail})" if detail and not cond else ""))
    if not cond:
        failures.append(name)


try:
    # --- the box ---------------------------------------------------------------------------------
    check("empty / broken box = the whole picture", cr.parse_box("") == cr.FULL and cr.parse_box("a b c d") == cr.FULL
          and cr.parse_box("0 0 0 0.5") == cr.FULL)
    check("a box past the picture is kept (padding)", cr.parse_box("-0.25 0 1.5 1") == (-0.25, 0.0, 1.5, 1.0))
    check("ratio pills", cr.ratio_value("16:9", 100, 100) == 16 / 9 and cr.ratio_value("original", 300, 100) == 3
          and cr.ratio_value("picture", 100, 100, (3840, 2160)) == 3840 / 2160 and cr.ratio_value("free", 1, 1) is None)
    check("all of a 1920x1080 picture", cr.pixel_box(cr.FULL, 1920, 1080) == (0, 0, 1920, 1080))
    b = cr.pixel_box(cr.FULL, 1000, 700, None, 16)
    check("16 px steps: sizes are multiples of 16, centred, inside", b[2] % 16 == 0 and b[3] % 16 == 0 and b[2] <= 1000
          and b[3] <= 700 and abs(b[0] - (1000 - b[2]) / 2) <= 1, str(b))
    b = cr.pixel_box(cr.FULL, 1920, 1080, 1.0, 16)
    check("1:1 on a wide picture: the largest square (16 px steps), centred", b == (424, 4, 1072, 1072), str(b))
    b = cr.pixel_box((0.1, 0.1, 0.5, 0.5), 1920, 1080, 16 / 9, 16)
    check("16:9 stays 16:9 within a step", abs(b[2] / b[3] - 16 / 9) < 16 / b[3] + 1e-6, str(b))
    b = cr.pixel_box((-0.25, -0.1, 1.5, 1.2), 400, 200, None, 1)
    check("past the edge: the box reaches outside", b == (-100, -20, 600, 240), str(b))
    check("long side", cr.long_side(1920, 1080, "1024", 16) == (1024, 576) and cr.long_side(500, 300, "as cut") == (500, 300))

    # --- cut and padding -------------------------------------------------------------------------
    rng = np.random.default_rng(1)
    pic = rng.random((40, 60, 3)).astype(np.float32)
    info = cr.crop_info((10, 5, 20, 10), 60, 40)
    check("cut inside = the slice", np.array_equal(cr.cut(pic, info), pic[5:15, 10:30]))
    info = cr.crop_info((-10, -5, 80, 50), 60, 40)
    for fill in cr.FILLS:
        out = cr.cut(pic, info, fill)
        check(f"padding '{fill}': size and the picture in the middle", out.shape == (50, 80, 3)
              and np.allclose(out[5:45, 10:70], pic), str(out.shape))
    check("padding black / white / grey values", np.allclose(cr.cut(pic, info, "white")[0, 0], 1)
          and np.allclose(cr.cut(pic, info, "grey")[0, 0], 0.5) and np.allclose(cr.cut(pic, info, "black")[0, 0], 0))
    check("padding edge = the border pixel", np.allclose(cr.cut(pic, info, "edge")[20, 0], pic[15, 0]))
    check("padding mirror = mirrored", np.allclose(cr.cut(pic, info, "mirror")[20, 9], pic[15, 1]))
    big = cr.crop_info((-150, -100, 360, 240), 60, 40)
    check("padding more than the picture (mirror repeats)", cr.cut(pic, big, "mirror").shape == (240, 360, 3))
    na = cr.new_area(info)
    check("new area mask: white on the padding only", na.shape == (50, 80) and na[0, 0] == 1 and na[25, 40] == 0
          and int(na.sum()) == 50 * 80 - 40 * 60)
    check("box mask on the picture", int(cr.box_mask(cr.crop_info((10, 5, 20, 10), 60, 40)).sum()) == 200
          and int(cr.box_mask(info).sum()) == 60 * 40)

    # --- paste back ------------------------------------------------------------------------------
    info = cr.crop_info((10, 5, 20, 10), 60, 40)
    white = np.ones((10, 20, 3), np.float32)
    out = cr.paste(pic, white, info, 0)
    check("paste hard: the box is replaced, the rest untouched", np.allclose(out[5:15, 10:30], 1)
          and np.allclose(out[:5], pic[:5]) and np.allclose(out[15:], pic[15:]))
    out = cr.paste(pic, np.ones((40, 80, 3), np.float32), info, 4)
    check("paste with a soft edge: full in the middle, blended at the edge", np.allclose(out[10, 20], 1)
          and 0 < (out[5, 20] - pic[5, 20]).max() < (1 - pic[5, 20]).max() + 1e-6)
    edge = cr.crop_info((0, 0, 20, 10), 60, 40)
    out = cr.paste(pic, white, edge, 6)
    check("no soft edge on the picture's border", np.allclose(out[0, 0], 1))
    out = cr.paste(np.repeat(np.repeat(pic, 2, 0), 2, 1), white, info, 0)
    check("an original at twice the size: the same spot", np.allclose(out[10:30, 20:60], 1) and not np.allclose(out[0:10], 1))
    pad = cr.crop_info((-10, -5, 80, 50), 60, 40)
    out = cr.paste(pic, np.ones((50, 80, 3), np.float32), pad, 0)
    check("paste of a padded crop: only the picture's part comes back", out.shape == pic.shape and np.allclose(out, 1))
    m = np.zeros((10, 20), np.float32); m[:, :10] = 1
    out = cr.paste(pic, white, info, 0, m)
    check("paste mask (crop size)", np.allclose(out[5:15, 10:20], 1) and np.allclose(out[5:15, 20:30], pic[5:15, 20:30]))

    # --- the nodes -------------------------------------------------------------------------------
    frames = torch.from_numpy(rng.random((3, 108, 192, 3)).astype(np.float32))
    res = ncr.KUBA_Crop.execute(frames, "0.25 0.25 0.5 0.5", "free", "as cut", True, "edge")
    img, bm, new, info = res.result
    check("crop node: every frame, 16 px steps", tuple(img.shape) == (3, 48, 96, 3) and img.shape[1] % 16 == 0, str(tuple(img.shape)))
    check("crop node: the pixels of the box", torch.allclose(img[1], frames[1, info["y"]:info["y"] + 48, info["x"]:info["x"] + 96]))
    check("crop node: box mask at the picture size, new area black", tuple(bm.shape) == (1, 108, 192) and float(new.max()) == 0)
    shown = res.ui["kubakub_crop"][0]
    check("crop node: the picture for the widget in temp", os.path.isfile(os.path.join(folder_paths.get_temp_directory(), shown["subfolder"], shown["filename"]))
          and shown["W"] == 192 and shown["px"] == [info["x"], info["y"], info["w"], info["h"]])
    img, bm, new, info = ncr.KUBA_Crop.execute(frames, "-0.25 0 1.5 1", "free", "1024", True, "grey").result
    check("crop node: padded + long side 1024", img.shape[2] == 1024 and img.shape[2] % 16 == 0 and img.shape[1] % 16 == 0, str(tuple(img.shape)))
    check("crop node: new area white at the sides", float(new[0, img.shape[1] // 2, 2]) > 0.99 and float(new[0, img.shape[1] // 2, 512]) < 0.01)
    matrix = torch.zeros((1, 2160, 3840, 3))
    img = ncr.KUBA_Crop.execute(frames, "0 0 1 1", "free", "as cut", False, "edge", ratio_of=matrix).result[0]
    check("crop node: a connected 'ratio of' shapes the box (16:9)", abs(img.shape[2] / img.shape[1] - 16 / 9) < 0.02, str(tuple(img.shape)))
    crop, _, _, info = ncr.KUBA_Crop.execute(frames, "0.25 0.25 0.5 0.5", "free", "1024", True, "edge").result
    back = ncr.KUBA_Uncrop.execute(frames, torch.ones_like(crop), info, 0).result[0]
    check("uncrop: the edited crop goes back into its box", tuple(back.shape) == tuple(frames.shape)
          and torch.allclose(back[:, info["y"]:info["y"] + info["h"], info["x"]:info["x"] + info["w"]], torch.ones(1))
          and torch.allclose(back[:, :info["y"]], frames[:, :info["y"]]))
    back = ncr.KUBA_Uncrop.execute(frames[:1], torch.ones((5,) + tuple(crop.shape[1:])), info, 8).result[0]
    check("uncrop: one original, five crops = five frames", back.shape[0] == 5)
    try:
        ncr.KUBA_Uncrop.execute(frames, crop, {"x": 1}, 0)
        check("uncrop: a box not from kubakub crop is refused", False)
    except ValueError:
        check("uncrop: a box not from kubakub crop is refused", True)
    ins = [i.id for i in ncr.KUBA_Crop.define_schema().inputs]
    check("crop node: inputs", ins == ["image", "box", "ratio", "size", "model_safe", "fill", "ratio_of"], str(ins))

    # --- compare ---------------------------------------------------------------------------------
    still = torch.from_numpy(rng.random((1, 90, 160, 3)).astype(np.float32))
    video = SimpleNamespace(get_components=lambda: SimpleNamespace(images=torch.rand((12, 180, 320, 3)), frame_rate=Fraction(24, 1)))
    out = ncm.KUBA_Compare.execute(still, video, "swipe", "1280").ui["kubakub_compare"][0]
    check("compare: a still against a video", out["a"]["n"] == 1 and out["b"]["n"] == 12 and out["b"]["kind"] == "video"
          and out["fps"] == 24.0, json.dumps(out)[:200])
    folder = os.path.join(folder_paths.get_temp_directory(), out["b"]["subfolder"].replace("/", os.sep))
    check("compare: one JPEG per frame", sorted(os.listdir(folder)) == [f"{i:05d}.jpg" for i in range(12)])
    out = ncm.KUBA_Compare.execute(frames, None, "difference", "full", 12.0).ui["kubakub_compare"][0]
    check("compare: a batch of frames alone, its fps", out["a"]["n"] == 3 and "b" not in out and out["fps"] == 12.0)
    try:
        ncm.KUBA_Compare.execute(None, None)
        check("compare: nothing connected says so", False)
    except ValueError:
        check("compare: nothing connected says so", True)

    # --- the director's layer crop ---------------------------------------------------------------
    W, H = 120, 80
    base = np.full((H, W, 3), 0.5, np.float32)
    quad = np.zeros((20, 40, 4), np.float32); quad[..., 3] = 1
    quad[:10, :20, 0] = 1; quad[:10, 20:, 1] = 1; quad[10:, :20, 2] = 1; quad[10:, 20:] = [1, 1, 1, 1]   # red green / blue white

    def doc(L):
        return {"version": 1, "canvas": [W, H], "layers": [L, {"id": "base", "name": "facade", "kind": "base"}]}
    L = {"id": "a", "name": "q", "kind": "image", "source": "input:0", "x": 10, "y": 10, "w": 40, "h": 20, "crop": [0.5, 0, 0.5, 0.5]}
    im = rd.render(doc(L), base, {"input:0": quad})["image"]
    check("director crop: only the green quarter, over the whole box", np.allclose(im[12:28, 12:48], [0, 1, 0], atol=1e-3))
    check("director crop: parse keeps it, drops a full or broken one", rd.parse(doc(L))["layers"][0]["crop"] == [0.5, 0, 0.5, 0.5]
          and rd.parse(doc(dict(L, crop=[0, 0, 1, 1])))["layers"][0]["crop"] is None
          and rd.parse(doc(dict(L, crop="x")))["layers"][0]["crop"] is None)
    im = rd.render(doc(dict(L, crop=[0.5, 0, 0.5, 0.5], flip_h=True)), base, {"input:0": quad})["image"]
    check("director crop + flip: the same quarter, mirrored", np.allclose(im[15, 20], [0, 1, 0], atol=1e-3))
    check("director crop_px = the widget's rounding", rd.crop_px([0.5, 0, 0.5, 0.5], 40, 20) == (20, 0, 40, 10)
          and rd.crop_px([0.333, 0.25, 0.334, 0.5], 99, 7) == (33, 2, 66, 5))

    # --- the widget finds the same pixels (node) -------------------------------------------------
    node = shutil.which("node")
    if node is None:
        print("skip the JavaScript checks: node is not installed")
    else:
        with open(os.path.join(HERE, "..", "web", "kubakub_crop.js"), encoding="utf-8") as f:
            src = f.read()
        src = re.sub(r"(?m)^import \{ app \} from .*$", "const app = { registerExtension() {} };", src)
        src = re.sub(r"(?m)^import \{ api \} from .*$", "const api = {};", src)
        src = src.replace('const HERE = new URL(".", import.meta.url).href;', 'const HERE = "";')
        cases = []
        for i in range(400):
            Wc, Hc = int(rng.integers(17, 4000)), int(rng.integers(17, 3000))
            x, y = float(rng.uniform(-0.4, 0.9)), float(rng.uniform(-0.4, 0.9))
            box = [x, y, float(rng.uniform(0.02, 1.4)), float(rng.uniform(0.02, 1.4))]
            if i % 3 == 0:
                box = [abs(x) * 0.5, abs(y) * 0.5, float(rng.uniform(0.02, 0.5)), float(rng.uniform(0.02, 0.5))]   # inside
            ratio = [None, 1.0, 16 / 9, 4 / 5, 21 / 9][i % 5]
            cases.append([[float(f"{v:.5g}") for v in box], Wc, Hc, ratio, [1, 16][i % 2]])
        src += "\nconst cases = " + json.dumps(cases) + ";\n"
        src += ("console.log(JSON.stringify(cases.map(([b, W, H, r, s]) => pixelBox(b, W, H, r, s))));\n"
                "const ok = []; const d = dragBox([10, 10, 100, 50], 'se', 210, 60, 2, false, 8); ok.push(d[2] === 200 && d[3] === 100);\n"
                "const c = dragBox([10, 10, 100, 50], 'w', 30, 0, null, true, 8); ok.push(c[0] === 30 && c[2] === 60);\n"
                "ok.push(ratioName(1920, 1080) === '16:9' && ratioName(1080, 1350) === '4:5');\n"
                "console.log(JSON.stringify(ok));\n")
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "crop.mjs")
            with open(path, "w", encoding="utf-8") as f:
                f.write(src)
            run = subprocess.run([node, path], capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60)
        if run.returncode != 0:
            check("the widget's box math runs (node)", False, run.stderr.strip()[-400:])
        else:
            lines = run.stdout.strip().splitlines()
            js, ok = json.loads(lines[0]), json.loads(lines[1])
            py = [list(cr.pixel_box(tuple(b), Wc, Hc, r, s)) for b, Wc, Hc, r, s in cases]
            bad = [(c, a, b) for c, a, b in zip(cases, js, py) if a != b]
            check("the widget shows the pixels the node cuts (400 boxes)", not bad, str(bad[:3]))
            check("the widget's drag: corner with a ratio, edge from the centre, ratio names", all(ok), str(ok))
finally:
    shutil.rmtree(TMP, ignore_errors=True)

print()
print("all crop tests passed" if not failures else f"{len(failures)} FAILED: " + ", ".join(failures))
sys.exit(1 if failures else 0)
