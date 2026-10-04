"""
Model free test for kubakub versions (kubakub/versions.py and the style
reference of kubakub/strategies.py).

Checks the versions sheet (lists, labels, modes, pick, LoRA names), that every
version's rules resolve with the region plan, the contact sheet, and that a
style image becomes the model's reference with a fake adapter.
Does not start ComfyUI and loads no model.

Run from the ComfyUI_windows_portable folder:
    python_embeded\\python.exe ComfyUI\\custom_nodes\\ComfyUI_KubaNodes\\tests\\test_versions.py
"""

import os
import sys
import tempfile

import numpy as np
import torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
from kubakub import plan as rp, strategies as st, versions as vs  # noqa: E402
from kubakub.types import RegionPlan  # noqa: E402
import test_sequential as seq  # noqa: E402

failures = []


def check(name, cond, detail=""):
    print(("ok   " if cond else "FAIL ") + name + (f"  ({detail})" if detail and not cond else ""))
    if not cond:
        failures.append(name)


def raises(fn, part=""):
    try:
        fn()
    except vs.SheetError as e:
        return part in str(e)
    return False


SHEET = """// a comment
[look]
copper = oxidised copper plates, verdigris   // trailing comment
folded white paper, soft daylight
none

[region group:Windows]
warm rooms
aquariums

[lora]
none
Kubakub_FluxKlein4B_v1 : 0.8 : kubakub style

[seed]
1
2
"""

BASE = """[default]
prompt = facade at night
denoise = 0.8
[W_*]
prompt += glass
"""


def test_parse():
    axes = vs.parse_sheet(SHEET)
    check("four lists", [a.kind for a in axes] == ["look", "region", "lora", "seed"])
    look = axes[0].options
    check("named option", look[0].label == "copper" and look[0].text == "oxidised copper plates, verdigris")
    check("unnamed option gets a short label", look[1].label.startswith("folded white paper") and len(look[1].label) <= 23)
    check("none adds nothing", look[2].label == "plain" and look[2].rules == "")
    check("look appends to the default prompt", look[0].rules == "[default]\nprompt += oxidised copper plates, verdigris\n")
    check("region list uses its selector", axes[1].options[1].rules == "[group:Windows]\nprompt += aquariums\n")
    lora = axes[2].options
    check("lora none", lora[0].lora == "" and lora[0].label == "no lora")
    check("lora name, strength, words", lora[1].lora == "Kubakub_FluxKlein4B_v1" and lora[1].strength == 0.8
          and "kubakub style" in lora[1].rules)
    check("seeds", [o.seed for o in axes[3].options] == [1, 2])
    chain = vs.parse_sheet("[lora]\nStyle_v1 : 1.2 : my style + Detail_v2 : 0.5\nStyle_v1")[0].options
    check("lora chain", chain[0].chain == (("Style_v1", 1.2), ("Detail_v2", 0.5)) and "my style" in chain[0].rules
          and chain[0].label == "Style_v1 1.2 + Detail_v2 0.5", str(chain[0]))
    named = vs.parse_sheet("[lora]\nsoft pair = Style_v1 : 1.1 + Detail_v2 : 0.65")[0].options[0]
    check("a lora line can have a short name", named.label == "soft pair"
          and named.chain == (("Style_v1", 1.1), ("Detail_v2", 0.65)), str(named))
    check("single lora is a chain of one", chain[1].chain == (("Style_v1", 1.0),) and lora[0].chain == ())
    check("version carries the chain", vs.build("[lora]\na : 2 + b : 0.5", "rotate")[0][0].chain == (("a", 2.0), ("b", 0.5)))
    check("lines without a header are looks", vs.parse_sheet("red brick\nblue tiles")[0].kind == "look")
    check("unknown list", raises(lambda: vs.parse_sheet("[colour]\nred"), "unknown list"))
    check("empty list", raises(lambda: vs.parse_sheet("[look]\n[seed]\n1"), "no options"))
    check("bad strength", raises(lambda: vs.parse_sheet("[lora]\nx : strong"), "not a number"))
    drive = "X" + ":\\l\\a.safetensors"
    check("lora path with a drive letter", vs.parse_sheet(f"[lora]\n{drive} : 0.5")[0].options[0].lora == drive)
    s = vs.parse_sheet("[set denoise]\n0.6\n0.9\n[set region_mp W_*]\n0.5")
    check("set", s[0].options[1].rules == "[default]\ndenoise = 0.9\n" and s[1].options[0].rules == "[W_*]\nregion_mp = 0.5\n")


def test_modes():
    v, axes, left = vs.build(SHEET, "rotate")
    check("rotate: as many as the longest list", len(v) == 3 and left == 0)
    check("rotate: short lists start over", [x.choice for x in v] == [(0, 0, 0, 0), (1, 1, 1, 1), (2, 0, 0, 0)])
    check("rotate: count", len(vs.build(SHEET, "rotate", count=7)[0]) == 7)
    v, _, left = vs.build(SHEET, "combine")
    check("combine: every combination", len(v) == 3 * 2 * 2 * 2 and len({x.choice for x in v}) == 24)
    v, _, left = vs.build(SHEET, "combine", max_versions=5)
    check("combine: capped", len(v) == 5 and left == 19)
    v, _, _ = vs.build(SHEET, "one by one")
    check("one by one: base + one change each", len(v) == 1 + 2 + 1 + 1 + 1
          and all(sum(1 for c in x.choice if c) <= 1 for x in v))
    check("numbers are 1-based and stable", [x.number for x in v] == list(range(1, 7))
          and [x.label for x in vs.build(SHEET, "one by one")[0]] == [x.label for x in v])
    check("label names every list that varies", v[0].label == "copper · warm rooms · no lora · seed 1", v[0].label)
    check("version carries lora and seed", (v[4].lora, v[4].strength, v[5].seed) == ("Kubakub_FluxKlein4B_v1", 0.8, 2))
    one = vs.build("[look]\nred brick", "rotate")[0]
    check("a single option still gives a version", len(one) == 1 and one[0].label == "version")


def test_rotate_list():
    sheet = "[rotate W_1 | W_2 | wall]\nglass = molten glass\nwoven textile\nmoss and ivy"
    v, axes, _ = vs.build(sheet, "rotate")
    check("rotate list: one version per idea", len(v) == 3)
    check("rotate labels say what the first region gets", [o.label for o in axes[0].options]
          == ["W_1: glass", "W_1: woven textile", "W_1: moss and ivy"], str([o.label for o in axes[0].options]))
    check("turn 0: ideas in order", v[0].rules == "[W_1]\nprompt += molten glass\n[W_2]\nprompt += woven textile\n"
                                                  "[wall]\nprompt += moss and ivy\n")
    check("turn 1: moved on by one", v[1].rules.startswith("[W_1]\nprompt += woven textile\n[W_2]\nprompt += moss and ivy\n"
                                                           "[wall]\nprompt += molten glass"))
    check("rotate needs two targets", raises(lambda: vs.parse_sheet("[rotate wall]\na\nb"), "two or more"))
    check("rotate needs two ideas", raises(lambda: vs.parse_sheet("[rotate a | b]\nx"), "two ideas"))


def test_pick_and_lora_names():
    check("pick empty = all", vs.parse_pick("", 4) == [1, 2, 3, 4])
    check("pick list and range", vs.parse_pick("3, 7-9 3", 12) == [3, 7, 8, 9])
    check("pick out of range", raises(lambda: vs.parse_pick("13", 12), "does not exist"))
    check("pick nonsense", raises(lambda: vs.parse_pick("a", 12), "not a version number"))
    files = ["Flux\\flux2\\4B\\Kubakub_FluxKlein4B_v1.safetensors",
             "Flux\\flux2\\4B\\Kubakub_FluxKlein4B_v1_000002000.safetensors",
             "Flux\\flux2\\kubakub\\Kubakub_FluxKlein9B_v4_000004500.safetensors"]
    check("lora by file name", vs.match_lora("Kubakub_FluxKlein4B_v1", files) == files[0])
    check("lora by full name", vs.match_lora("Flux/flux2/4B/Kubakub_FluxKlein4B_v1.safetensors", files) == files[0])
    check("lora by a unique part", vs.match_lora("9B_v4", files) == files[2])
    check("lora ambiguous", raises(lambda: vs.match_lora("Klein4B", files), "fits 2 files"))
    check("lora missing names close ones", raises(lambda: vs.match_lora("Kubakub_FluxKlein4B_v2", files), "close:"))


def test_rules_resolve():
    regions = seq.make_regions()
    v, _, _ = vs.build(SHEET, "combine")
    for x in v:
        plan = rp.resolve(regions.table, BASE + x.rules, seed=x.seed)
    a = rp.resolve(regions.table, BASE + v[0].rules)["regions"]
    check("look after the base prompt, then the region's parts (the plan's order: group before name)",
          a[1]["prompt"] == "facade at night, oxidised copper plates, verdigris, warm rooms, glass", a[1]["prompt"])
    check("wall gets the look only", a[0]["prompt"] == "facade at night, oxidised copper plates, verdigris")
    check("the plan keeps its rules text", rp.resolve(regions.table, BASE)["rules_text"] == BASE)
    lora = [x for x in v if x.lora][0]
    check("lora words reach every prompt",
          all("kubakub style" in e["prompt"] for e in rp.resolve(regions.table, BASE + lora.rules)["regions"]))


def test_references():
    with tempfile.TemporaryDirectory() as d:
        for name in ("b_marble.png", "a_rust.jpg"):
            Image.new("RGB", (64, 48), (200, 80, 20)).save(os.path.join(d, name))
        open(os.path.join(d, "notes.txt"), "w").close()
        one = os.path.join(d, "a_rust.jpg")
        v, axes, _ = vs.build(f"[reference]\nnone\n{d}\nmy rust = {one}", "rotate", input_references=2)
        refs = axes[0].options
        check("folder = every image, sorted; file; inputs", [o.label for o in refs]
              == ["no ref", "a_rust", "b_marble", "my rust", "ref 1", "ref 2"], str([o.label for o in refs]))
        check("input references", refs[-1].reference == "input:1")
        check("a reference switches the plan to reference = style",
              v[0].rules == "" and v[1].rules == "[default]\nreference = style\n")
        for i in range(8):
            Image.new("RGB", (8, 8)).save(os.path.join(d, f"s{i}.png"))
        some = [o.reference[-6:] for o in vs.parse_sheet(f"[reference]\n{d} | 4")[0].options]
        check("folder | 4 = four images spread over the folder", len(some) == 4 and len(set(some)) == 4
              and some == [os.path.basename(p)[-6:] for p in
                           [sorted(f for f in os.listdir(d) if f.endswith(('.png', '.jpg')))[i * 10 // 4] for i in range(4)]],
              str(some))
        check("folder | 4 | 1 shifts the pick", [o.reference for o in vs.parse_sheet(f"[reference]\n{d} | 4 | 1")[0].options]
              != [o.reference for o in vs.parse_sheet(f"[reference]\n{d} | 4")[0].options])
        check("folder | x", raises(lambda: vs.parse_sheet(f"[reference]\n{d} | many"), "how many"))
        check("missing reference", raises(lambda: vs.parse_sheet("[reference]\n" + os.path.join(d, "x.png")), "neither"))
    check("references input alone makes the list", len(vs.build("[look]\nred", "rotate", input_references=3)[0]) == 3)

    # [image]: the picture a version starts from
    with tempfile.TemporaryDirectory() as d:
        os.makedirs(os.path.join(d, "maps"))
        for name in ("wire.png", os.path.join("maps", "canny.png"), os.path.join("maps", "crypto.png")):
            Image.new("RGB", (64, 48)).save(os.path.join(d, name))
        wire = os.path.join(d, "wire.png")
        v, axes, _ = vs.build(f"[image]\ninput\nwireframe = {wire}\n{os.path.join(d, 'maps')}", "rotate",
                              input_images=3)
        check("image list: input, a named file, a folder, the rest of the batch", [o.label for o in axes[0].options]
              == ["input", "wireframe", "canny", "crypto", "image 2", "image 3"], str([o.label for o in axes[0].options]))
        check("a version starts from its image", [x.image for x in v[:2]] == ["input:0", wire]
              and v[4].image == "input:1" and v[1].rules == "" and v[1].label == "wireframe")
        check("the image is in the recipe", vs.recipe(v[1], "", 0, 0)["image"] == wire)
        check("missing image", raises(lambda: vs.parse_sheet("[image]\n" + os.path.join(d, "x.png")), "image '"))
    v, axes, _ = vs.build("[look]\nred\nblue", "rotate", input_images=1)
    check("one input image: no image list", len(axes) == 1 and v[0].image == "")
    check("an image batch alone makes the list", [x.image for x in vs.build("[look]\nred", "rotate", input_images=2)[0]]
          == ["input:0", "input:1"])


def test_style_reference():
    regions = seq.make_regions()
    canvas = seq.canvas_img()
    style = torch.rand(1, 300, 500, 3)

    def run(rules, style_img, mp=1.0):
        ad = seq.FakeAdapter()
        ad.refs = []                                   # latent size of every reference added

        def add_reference(cond, latent):
            ad.refs.append(tuple(latent.shape[-2:]))
            return [[cond[0][0], {"reference_latents": [latent]}]]
        ad.add_reference = add_reference
        plan = RegionPlan(rp.resolve(regions.table, rules), regions)
        st.run_plan(ad, canvas, plan, st.SamplerSettings(style=style_img, style_mp=mp, region_mp=0.25),
                    use_cache=False)
        return ad

    ad = run("[default]\nprompt = red\nreference = style\n", style)
    check("style: two references per region, the crop first", len(ad.refs) == 2 * len(ad.calls) == 8, str(ad.refs))
    lat = ad.refs[1]
    check("style: the same style latent for every region", set(ad.refs[1::2]) == {lat})
    check("style image near 1 MP on the grid", 0.8e6 < lat[0] * 16 * lat[1] * 16 <= 1024 * 1024, str(lat))
    small = run("[default]\nprompt = red\nreference = style\n", style, mp=0.25).refs[1]
    check("style_mp scales the style image", small[0] * small[1] < lat[0] * lat[1] / 3, str(small))
    ad = run("[default]\nprompt = red\nreference = style\n[sign]\nreference = self\n", style)
    check("a region can go back to self", len(ad.refs) == 2 * len(ad.calls) - 1)
    ad = run("[default]\nprompt = red\nreference = style\n", None)
    check("no style image: falls back to self", len(ad.refs) == len(ad.calls))
    sr = st.StyleReference(seq.FakeAdapter(), style)
    check("style digest follows the pixels", sr.digest != st.StyleReference(seq.FakeAdapter(), style * 0.5).digest)


def test_merge_same_and_resize():
    regions = seq.make_regions()
    d = rp.resolve(regions.table, "[default]\nprompt = red\n[W_*]\nprompt = blue glass\n[sign]\nstrategy = keep\n")
    plan, merged = vs.merge_same(d, regions)
    names = [e["name"] for e in plan["regions"]]
    check("same treatment = one region", len(plan["regions"]) == 3 and "W_1 +1" in names, str(names))
    w = next(e for e in plan["regions"] if e["name"] == "W_1 +1")
    check("merged box and area", w["bbox"] == [80, 60, 320, 80] and w["area"] == 12800, str(w["bbox"]))
    check("labels follow", int((merged.labels == w["region_id"]).sum()) == 12800
          and int((merged.labels >= 0).sum()) == int((regions.labels >= 0).sum()))
    check("keep stays out of the order", len(plan["order"]) == 2 and merged.count == 3)
    ad = seq.FakeAdapter()
    st.run_plan(ad, seq.canvas_img(), RegionPlan(plan, merged), st.SamplerSettings(region_mp=0.25), use_cache=False)
    check("two samples instead of three", len(ad.calls) == 2)
    same, same_r = vs.merge_same(rp.resolve(regions.table, "[default]\nprompt = {a|b|c|d|e|f|g|h} {id}\n"), regions)
    check("nothing to merge: unchanged", same_r is regions)
    fif = rp.resolve(regions.table, "[default]\nprompt = red\n[W_*]\nstrategy = frame_in_frame\n")
    check("frame in frame regions stay apart", len(vs.merge_same(fif, regions)[0]["regions"]) == 3)

    small = regions.resized(240, 135)
    check("resized: size and table", small.size == (240, 135) and small.table["width"] == 240)
    check("resized: boxes measured again", small.table["regions"][1]["bbox"] == [40, 30, 40, 40],
          str(small.table["regions"][1]["bbox"]))
    check("resized: the original is untouched", regions.table["regions"][1]["bbox"] == [80, 60, 80, 80])
    check("resized: same size returns itself", regions.resized(480, 270) is regions)


def test_refine_tiles():
    from kubakub import ops, seams
    cp = ops.CropPlan(0, 0, 64, 64, 16, 16, 64, 64)
    r = seams.tile_ramp(cp, 200, 64, 16)
    check("ramp: full at canvas edges, fading towards the next tile",
          float(r[0, 0, 0]) == 1.0 and float(r[0, 32, 0]) == 1.0 and 0 < float(r[0, 0, 63]) < 0.2)
    mid = seams.tile_ramp(ops.CropPlan(64, 0, 64, 64, 16, 16, 64, 64), 200, 64, 16)
    check("ramp: both sides inside the canvas", float(mid[0, 0, 0]) < 0.2 and float(mid[0, 0, 63]) < 0.2
          and float(mid[0, 0, 32]) == 1.0)

    regions = seq.make_regions()
    plan = RegionPlan(rp.resolve(regions.table, "[default]\nprompt = red, stone\n[W_*]\nprompt = red, blue glass\n"
                                                "[sign]\nstrategy = keep\n"), regions)
    entries = plan.plan["regions"]
    lab = regions.labels[:1]
    p, n, names = seams.tile_prompt(entries, lab[:, 40:160, 60:180], torch.ones(1, 120, 120))
    check("tile prompt: the regions that fill the tile, no repeats", p == "red, stone, blue glass"
          and names == ["wall", "W_1"], f"{p} {names}")
    p, _, names = seams.tile_prompt(entries, lab[:, :40, :40], torch.ones(1, 40, 40))
    check("tile prompt: one region", p == "red, stone" and names == ["wall"])

    ad = seq.FakeAdapter()
    canvas = seq.canvas_img()
    out, lines = seams.refine_tiles(ad, canvas, plan, denoise=0.4, tile_px=128)
    check("refine: tiles cover the canvas", len(ad.calls) >= 6 and "tile(s)" in lines[0], lines[0])
    check("refine: every tile at 1:1 with itself as reference", all(c["ref"] and c["size"] == (128, 128)
                                                                    for c in ad.calls))
    sign = regions.labels[0] == 3
    check("refine: keep regions untouched", torch.equal(out[0][sign], canvas[0][sign]))
    check("refine: the rest is resampled", float((out[0][~sign] - canvas[0][~sign]).abs().mean()) > 0.05)
    check("refine: size kept", out.shape == canvas.shape)


def test_contact_sheet():
    imgs = [np.full((90, 160, 3), i / 6, np.float32) for i in range(5)]
    sheet = vs.contact_sheet(imgs, [f"{i + 1:02d}  look {i}" for i in range(5)])
    check("sheet is an image", sheet.dtype == np.float32 and sheet.ndim == 3 and sheet.shape[2] == 3)
    cols, rows, scale = vs.sheet_layout(5, 160, 90)
    check("layout holds every version", cols * rows >= 5 and scale == 1.0)
    check("labels are drawn", float((sheet[..., 0] > 0.8).mean()) > 0.0005)
    big = vs.contact_sheet([np.zeros((1080, 1600, 3), np.float32)] * 12, ["x"] * 12, max_width=4096)
    check("sheet width capped", big.shape[1] <= 4096, str(big.shape))
    check("columns as asked", vs.sheet_layout(12, 1600, 1080, columns=6)[:2] == (6, 2))


def test_picks_folder():
    import json
    import shutil
    from PIL.PngImagePlugin import PngInfo

    check("file name from a label", vs.slug("isometric · soft pair · ComfyUI_00163_")
          == "isometric_soft-pair_ComfyUI_00163")
    check("run number goes on", vs.next_run(["r01_v01_a.png", "r07_v12_b.png", "sheet_00001_.png"]) == 8
          and vs.next_run([]) == 1)

    sheet = "[look]\ncopper = copper plates\npaper = folded paper\nmoss = moss\n[lora]\nA_v1 : 0.8 + B_v2 : 0.5\n"
    versions, _, _ = vs.build(sheet, "rotate")
    small = Image.new("RGB", (8, 8))

    def save(path, **chunks):
        meta = PngInfo()
        for k, v in chunks.items():
            meta.add_text(k, v)
        small.save(path, pnginfo=meta)

    with tempfile.TemporaryDirectory() as out:
        picks = os.path.join(out, "picks")
        os.makedirs(os.path.join(picks, "final"))
        check("no images: says so", raises(lambda: vs.pick_files(picks), "no images"))
        check("not a folder: says so", raises(lambda: vs.pick_files(picks + "x"), "not a folder"))

        # a draft the node saved: the recipe is in the file, wherever it is copied to
        r = vs.recipe(versions[1], "[default]\nprompt = base\n", 5, 3)
        save(os.path.join(picks, "r01_v02_paper.png"), **{vs.RECIPE_KEY: json.dumps(r)})
        back = vs.read_recipe(os.path.join(picks, "r01_v02_paper.png"))
        check("recipe survives the file", back == r and back["chain"] == [["A_v1", 0.8], ["B_v2", 0.5]]
              and "folded paper" in back["rules"], str(back))

        # drafts written by Save Image: two runs after each other in output/drafts, counters 3-5 and 6-8
        def graph(seed, pick=""):
            return json.dumps({
                "4": {"class_type": "KUBA_Versions", "inputs": {"versions": sheet, "mode": "rotate", "pick": pick,
                                                                 "max_versions": 12, "seed": seed, "count": 0}},
                "6": {"class_type": "SaveImage", "inputs": {"filename_prefix": "drafts/version", "images": ["4", 0]}},
                "5": {"class_type": "SaveImage", "inputs": {"filename_prefix": "drafts/sheet", "images": ["4", 1]}}})

        os.makedirs(os.path.join(out, "drafts"))
        for k in range(3, 6):
            save(os.path.join(out, "drafts", f"version_{k:05d}_.png"), prompt=graph(0))
        for k in range(6, 8):
            save(os.path.join(out, "drafts", f"version_{k:05d}_.png"), prompt=graph(7, "1, 3"))
        for k in (4, 7):
            shutil.copy(os.path.join(out, "drafts", f"version_{k:05d}_.png"), picks)
        a = vs.read_recipe(os.path.join(picks, "version_00004_.png"), out)
        b = vs.read_recipe(os.path.join(picks, "version_00007_.png"), out)
        check("Save Image draft: second file of its run = version 2", a is not None and a["number"] == 2
              and a["label"] == "paper" and a["base_rules"] is None and a["offset"] == 0, str(a))
        check("Save Image draft of a picked run: second file = the second picked number", b is not None
              and b["number"] == 3 and b["offset"] == 7, str(b))
        shutil.copy(os.path.join(picks, "version_00004_.png"), os.path.join(picks, "renamed.png"))
        check("renamed or foreign file: unknown", vs.read_recipe(os.path.join(picks, "renamed.png"), out) is None)
        small.save(os.path.join(picks, "photo.jpg"))
        check("no workflow in the file: unknown", vs.read_recipe(os.path.join(picks, "photo.jpg"), out) is None)
        check("picks: the images of the folder, by name", [os.path.basename(p) for p in vs.pick_files(picks)]
              == ["photo.jpg", "r01_v02_paper.png", "renamed.png", "version_00004_.png", "version_00007_.png"])


if __name__ == "__main__":
    for t in (test_parse, test_modes, test_rotate_list, test_pick_and_lora_names, test_rules_resolve,
              test_references, test_style_reference, test_merge_same_and_resize, test_refine_tiles,
              test_contact_sheet, test_picks_folder):
        t()
    print()
    if failures:
        print(f"{len(failures)} FAILED: {', '.join(failures)}")
        sys.exit(1)
    print("all versions tests passed")
