"""
Model free test for the nodes of 2d / generate (nodes/generate): the region sampler's style input and its report
notes, the seam pass's advanced inputs, the region plan's preview on an image of another size, the regions output
of frame in frame, and kubakub versions (only, cfg, the save_path and names outputs) with the fake adapter of
test_sequential.py. Does not start ComfyUI, loads no model and stays on the CPU.

Run from the ComfyUI_windows_portable folder:
    python_embeded\\python.exe ComfyUI\\custom_nodes\\ComfyUI_KubaNodes\\tests\\test_generate_nodes.py
"""

import os
import shutil
import sys
import tempfile

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(HERE))))     # ComfyUI (comfy_api)
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)     # embedded Python leaves the script folder off the path

sys.argv = sys.argv[:1] + ["--cpu"]         # core never touches the GPU here
import comfy.options  # noqa: E402
comfy.options.enable_args_parsing()
import folder_paths  # noqa: E402

TMP = tempfile.mkdtemp(prefix="kkd_generate_")
folder_paths.set_temp_directory(os.path.join(TMP, "temp"))
folder_paths.set_output_directory(os.path.join(TMP, "output"))
os.makedirs(os.path.join(TMP, "output"), exist_ok=True)

import _pack  # noqa: E402,F401  (the pack as a package, see tests/_pack.py)
from kubapack.nodes.generate import nodes_frames as nf  # noqa: E402
from kubapack.nodes.generate import nodes_plan as npl  # noqa: E402
from kubapack.nodes.generate import nodes_sampler as nsm  # noqa: E402
from kubapack.nodes.generate import nodes_versions as nv  # noqa: E402
from kubakub import adapters, plan as rp, seams, versions as vs, viewer as vw  # noqa: E402
from kubakub.types import RegionPlan, Regions  # noqa: E402
import test_sequential as seq  # noqa: E402

failures = []


def check(name, cond, detail=""):
    print(("ok   " if cond else "FAIL ") + name + (f"  ({detail})" if detail and not cond else ""))
    if not cond:
        failures.append(name)


class NodeAdapter(seq.FakeAdapter):
    """The fake adapter with what the nodes ask of a real one; every one made is kept in `made`."""
    family, schedule = "fake", "auto"
    made = []

    def __init__(self, *a):
        super().__init__()
        self.refs, self.cfgs = [], []
        NodeAdapter.made.append(self)

    def set_schedule(self, name):
        self.schedule = name

    def add_reference(self, cond, latent):
        self.refs.append(tuple(latent.shape[-2:]))
        return super().add_reference(cond, latent)

    def sample(self, latent, pos, neg, **kw):
        self.cfgs.append(kw.get("cfg"))
        return super().sample(latent, pos, neg, **kw)


adapters.make_adapter = lambda model, clip, vae, family="auto": NodeAdapter()


def ids(items):
    return [i.id for i in items]


def plan_of(rules, regions=None):
    regions = regions or seq.make_regions()
    d = rp.resolve(regions.table, rules)
    d["rules_text"] = rules                 # as kubakub region plan keeps it for kubakub versions
    return RegionPlan(d, regions)


def test_region_sampler():
    schema = nsm.KUBA_RegionSampler.define_schema()
    check("sampler: style, material_at, material and change are the last inputs, optional",
          ids(schema.inputs)[-4:] == ["style", "material_at", "material", "change"]
          and all(i.optional and bool(i.tooltip) for i in schema.inputs[-4:]))
    check("sampler: the old inputs keep their order", ids(schema.inputs)[:-4] == [
        "model", "clip", "vae", "plan", "image", "steps", "cfg", "sampler_name", "scheduler", "region_mp",
        "max_upscale", "schedule", "only", "seed_offset", "adapter"])
    check("sampler: outputs unchanged", ids(schema.outputs) == ["image", "changed", "report"])

    def run(rules, image=None, **kw):
        NodeAdapter.made.clear()
        image = seq.canvas_img() if image is None else image
        out = nsm.KUBA_RegionSampler.execute(None, None, None, plan_of(rules), image, 0, 0.0, "euler", "simple",
                                             0.25, 4.0, **kw).args
        return NodeAdapter.made[0], out[2]

    style = torch.rand(2, 300, 500, 3)
    rules = "[default]\nprompt = red\nreference = style\n"
    ad, report = run(rules, style=style)
    check("sampler: the style image is the second reference of every region",
          len(ad.calls) == 4 and len(ad.refs) == 8 and len(set(ad.refs[1::2])) == 1, str(ad.refs))
    check("sampler: the report says the style image is used", "style image used by 4 region(s)" in report, report)
    ad, report = run(rules)
    check("sampler: reference = style without an image falls back to self, and says so",
          len(ad.refs) == 4 and "4 region(s) have reference = style but no style image is connected" in report,
          report)
    ad, report = run(rules, only="W_*")
    check("sampler: the note counts the regions of this run", "2 region(s) have reference = style" in report, report)
    ad, report = run("[default]\nprompt = red\n", style=style)
    check("sampler: a style image no region asks for is named", len(ad.refs) == 4
          and "no region of this run has reference = style" in report, report)
    ad, report = run("[default]\nprompt = red\n")
    check("sampler: nothing to say = no note", "note:" not in report, report)
    ad, report = run("[default]\nprompt = red\n", image=seq.canvas_img().repeat(3, 1, 1, 1))
    check("sampler: an image batch is named in the report",
          "batch of 3 images: only the first one is used" in report, report)


def test_material():
    """A picture and a mask from a kubakub shader as the material of the regions (kubakub/materials.py)."""
    from kubakub import materials as ma

    def run(rules, **kw):
        NodeAdapter.made.clear()
        out = nsm.KUBA_RegionSampler.execute(None, None, None, plan_of(rules), seq.canvas_img(), 0, 0.0, "euler",
                                             "simple", 0.25, 4.0, **kw).args
        return NodeAdapter.made[0], out

    lab = seq.make_regions().labels[0].numpy()
    w1, w2, wall = torch.from_numpy(lab == 1), torch.from_numpy(lab == 2), torch.from_numpy(lab == 0)
    green = torch.zeros(1, 270, 480, 3)
    green[..., 1] = 1.0
    rules = "[default]\nstrategy = keep\n[W_*]\nstrategy = inpaint\nprompt = red\n"
    pre = ma.before(plan_of(rules), seq.canvas_img(), green)
    check("material: it is the start picture of the painted regions, keep regions stay as they are",
          float(pre.canvas[0][w1][:, 1].min()) == 1.0 and float(pre.canvas[0][w2][:, 0].max()) == 0.0
          and torch.equal(pre.canvas[0][wall], seq.canvas_img()[0][wall]) and pre.reference is None and pre.change is None,
          str(pre.lines))
    one = ma.before(plan_of(rules), seq.canvas_img(), green, pick=lambda e: e["name"] == "W_1")
    check("material: only the regions of this run", float(one.canvas[0][w1][:, 1].min()) == 1.0
          and torch.equal(one.canvas[0][w2], seq.canvas_img()[0][w2]))
    ref = ma.before(plan_of(rules + "reference = material\n"), seq.canvas_img(), green)
    check("material: reference = material keeps the region's own start and shows the material instead",
          torch.equal(ref.canvas, seq.canvas_img()[..., :3]) and torch.equal(ref.reference, green)
          and "shown as the reference to 2 regions" in "\n".join(ref.lines), str(ref.lines))
    clip = torch.zeros(5, 135, 240, 3)
    clip[3, ..., 2] = 1.0
    moved = ma.before(plan_of(rules), seq.canvas_img(), clip, at=0.75)
    check("material: of a clip the frame at material_at, resized to the picture",
          float(moved.canvas[0][w1][:, 2].min()) == 1.0 and "frame 4 is used" in "\n".join(moved.lines)
          and "resized to the picture's 480x270" in "\n".join(moved.lines), str(moved.lines))
    check("material: which frame a still takes", [ma.frame_at(n, a) for n, a in ((1, 0.3), (5, 0.0), (5, 1.0), (5, 0.5), (101, 0.3))] == [0, 0, 4, 2, 30])

    grey = torch.zeros(1, 270, 480)
    grey[0, 60:100] = 0.25
    grey[0, 100:140] = 0.5
    chg = ma.before(plan_of(rules), seq.canvas_img(), None, grey)
    check("change: its brightest place on the painted regions counts as white",
          abs(float(chg.change[0, 120, 100]) - 1.0) < 1e-6 and abs(float(chg.change[0, 80, 100]) - 0.5) < 1e-6
          and "counted as white" in "\n".join(chg.lines), str(chg.lines))
    none = ma.before(plan_of(rules), seq.canvas_img(), None, torch.zeros(1, 270, 480))
    check("change: a black mask is said", "nothing changes" in "\n".join(none.lines))
    kept = ma.before(plan_of("[default]\nstrategy = keep\n"), seq.canvas_img(), green, grey)
    check("material and change with nothing painted: not used, and said", torch.equal(kept.canvas, seq.canvas_img()[..., :3])
          and kept.change is None and "not used" in "\n".join(kept.lines))

    ad, out = run(rules, material=green)
    check("sampler: the regions are painted on the material (their crops start from it)", len(ad.calls) == 2
          and "material: the start picture of 2 regions" in out[2], out[2])
    ad, out = run(rules + "reference = material\n", material=green)
    check("sampler: reference = material is a second reference of the region", len(ad.refs) == 4, str(ad.refs))
    ad, out = run(rules + "reference = material\n")
    check("sampler: reference = material with nothing connected falls back, and says so",
          len(ad.refs) == 2 and "nothing is connected to 'material'" in out[2], out[2])
    seen = []
    keep = NodeAdapter.sample

    def spy(self, latent, pos, neg, **kw):
        seen.append((bool(kw.get("soft")), kw.get("noise_mask")))
        return keep(self, latent, pos, neg, **kw)
    NodeAdapter.sample = spy
    try:
        run(rules, change=grey)
        check("sampler: a change map is sampled softly, with its grey values in the mask",
              len(seen) == 2 and all(s for s, _ in seen) and 0.0 < float(seen[0][1].unique()[1]) < 1.0, str([s for s, _ in seen]))
        seen.clear()
        run(rules)
        check("sampler: without one the mask is plain", not any(s for s, _ in seen))
    finally:
        NodeAdapter.sample = keep
    ys = torch.arange(270)[:, None].expand(270, 480)
    top = w1 & (ys < int(ys[w1].float().median()))
    half = torch.zeros(1, 270, 480)
    half[0][top] = 1.0                                # W_1: its upper half may change; W_2: nothing
    ad, out = run(rules, change=half)
    low = w1 & ~top
    check("change: black places of a region stay pixel exact and are not in 'changed'",
          torch.equal(out[0][0][low], seq.canvas_img()[0][low][..., :3]) and float(out[1][0][low].max()) == 0.0
          and float(out[1][0][top].max()) > 0.0)
    check("change: a region the map leaves black is not sampled", len(ad.calls) == 1, str(len(ad.calls)))
    for key in ("material", "change", "finish"):
        try:
            rp.resolve(seq.make_regions().table, f"[default]\n{key} = plasma\n")
            check(f"plan: '{key}' is not a plan key", False)
        except rp.PlanError as e:
            check(f"plan: '{key}' typed into the plan says where it goes", "kubakub shader" in str(e) and "line 2" in str(e), str(e))
    check("plan: reference is read whatever its capitals", rp.resolve(seq.make_regions().table, "[default]\nprompt = x\nreference = Material\n")["regions"][0]["reference"] == "material")

    pixels = np.zeros((4, 8, 8, 3), np.float32)
    mat = [np.full((8, 8, 3), 0.1 * (i + 1), np.float32) for i in range(4)]
    region = np.zeros((8, 8), bool)
    region[2:5, 2:5] = True
    g = ma.guide(pixels, mat, region, np.array([False, True, True, True]))
    check("guide: the material inside the region on the frames of the time range, the crop everywhere else",
          g is not None and float(g[0].max()) == 0.0 and abs(float(g[2, 3, 3, 0]) - 0.3) < 1e-6 and float(g[2, 0, 0, 0]) == 0.0
          and float(pixels.max()) == 0.0)
    check("guide: a material that is the picture guides nothing", ma.guide(pixels, [pixels[0]] * 4, region, np.ones(4, bool)) is None)


def test_seam_pass_schema():
    schema = nsm.KUBA_SeamPass.define_schema()
    check("seam pass: inputs keep their order", ids(schema.inputs) == [
        "model", "clip", "vae", "plan", "image", "changed", "seam_px", "denoise", "steps", "cfg", "sampler_name",
        "scheduler", "differential", "schedule", "seed", "prompt", "tile_px", "adapter"])
    adv = {i.id for i in schema.inputs if getattr(i, "advanced", None)}
    check("seam pass: the sampling inputs are advanced", adv == {"steps", "cfg", "sampler_name", "scheduler",
                                                               "tile_px", "adapter"}, str(adv))
    by = {i.id: i for i in schema.inputs}
    check("seam pass: defaults unchanged", (by["steps"].default, by["cfg"].default, by["sampler_name"].default,
                                           by["scheduler"].default) == (0, 0.0, "euler", "simple"))


def test_plan_preview():
    regions = seq.make_regions()
    rules = "[default]\nprompt = red\n[W_*]\nprompt = blue glass\n"
    run = lambda image: npl.KUBA_RegionPlan.execute(regions, rules, 0, image=image).args  # noqa: E731
    same = run(seq.canvas_img())
    grey = run(None)
    big = run(seq.canvas_img(540, 960))
    check("plan: outputs unchanged", ids(npl.KUBA_RegionPlan.define_schema().outputs)
          == ["plan", "plan_json", "report", "preview"])
    check("plan: preview at the regions' size", tuple(big[3].shape) == tuple(same[3].shape) == (1, 270, 480, 3))
    d_same = float((big[3] - same[3]).abs().mean())
    d_grey = float((big[3] - grey[3]).abs().mean())
    check("plan: an image of another size is scaled for the preview, not dropped", d_same < 0.02 < d_grey,
          f"{d_same:.4f} {d_grey:.4f}")
    check("plan: the report says so", "the image is 960x540, the regions are 480x270" in big[2]
          and "note:" not in same[2] and "note:" not in grey[2], big[2][-200:])
    odd = run(torch.rand(1, 100, 100, 4))
    check("plan: another shape and an alpha channel still give a preview", tuple(odd[3].shape) == (1, 270, 480, 3))


def test_region_prompt():
    """kubakub region prompt: the plan from fields (which regions, prompt, strategy, denoise), chained by cable."""
    node = npl.KUBA_RegionPrompt
    schema = node.define_schema()
    check("region prompt: the fields in the order they are filled in",
          ids(schema.inputs) == ["regions", "plan", "which", "prompt", "strategy", "denoise", "seed", "more", "image"]
          and ids(schema.outputs) == ["plan", "which_out", "preview", "report"]
          and all(i.tooltip for i in schema.inputs) and all(o.tooltip for o in schema.outputs))
    regions = seq.make_regions()
    names = [r["name"] for r in regions.table["regions"]]
    wins = [n for n in names if n.startswith("W_")]
    first = node.execute("W_*", "stained\nglass", "inpaint", 0.8, 3, regions=regions).args
    by = {e["name"]: e for e in first[0].plan["regions"]}
    check("region prompt: the chosen regions get prompt, strategy and denoise; the others are kept",
          all(by[n]["prompt"] == "stained glass" and by[n]["denoise"] == 0.8 and by[n]["strategy"] == "inpaint" for n in wins)
          and all(by[n]["strategy"] == "keep" for n in names if n not in wins) and first[0].order == [by[n]["region_id"] for n in wins],
          str([(e["name"], e["strategy"]) for e in first[0].plan["regions"]]))
    check("region prompt: 'which regions' goes out as it is, the report counts the regions",
          first[1] == "W_*" and first[3].startswith(f"this prompt: {len(wins)} regions") and tuple(first[2].shape) == (1, 270, 480, 3))
    wide = node.execute("", "gold", "inpaint", 0.5, 3, plan=first[0]).args
    by2 = {e["name"]: e for e in wide[0].plan["regions"]}
    check("region prompt: a second node on top wins where both name a region, also with a wider choice",
          all(e["prompt"] == "gold" and e["strategy"] == "inpaint" for e in by2.values()) and "(every region)" in wide[3])
    out = node.execute(wins[0], "", "keep", 0.5, 3, plan=first[0], more="feather_px = 20").args
    by3 = {e["name"]: e for e in out[0].plan["regions"]}
    check("region prompt: keep takes a region out again, 'more' sets the other keys",
          by3[wins[0]]["strategy"] == "keep" and by3[wins[0]]["feather_px"] == 20 and by3[wins[1]]["strategy"] == "inpaint"
          and "note:" not in out[3])
    again = rp.resolve(regions.table, out[0].plan["rules_text"], seed=3)
    check("region prompt: the plan's rule text gives the same plan again (kubakub versions builds on it)",
          [(e["strategy"], e["prompt"], e["denoise"]) for e in again["regions"]]
          == [(e["strategy"], e["prompt"], e["denoise"]) for e in out[0].plan["regions"]])
    text = npl.KUBA_RegionPlan.execute(regions, "[default]\nprompt = base\ndenoise = 0.3\n", 0).args[0]
    on_text = {e["name"]: e for e in node.execute("W_*", "glass", "inpaint", 0.7, 0, plan=text).args[0].plan["regions"]}
    check("region prompt: on top of a kubakub region plan", on_text[wins[0]]["prompt"] == "glass"
          and all(on_text[n]["prompt"] == "base" and on_text[n]["denoise"] == 0.3 for n in names if n not in wins))
    for what, kw, word in (("nothing connected", dict(), "connect 'regions'"),
                           ("a choice that matches nothing", dict(regions=regions, which="nothing_*"), "matches no region"),
                           ("a section typed into 'more'", dict(regions=regions, which="W_*", more="[x]\nprompt = y"), "chain a second"),
                           ("an unknown key in 'more'", dict(regions=regions, which="W_*", more="denoize = 1"), "denoise")):
        try:
            node.execute(kw.pop("which", ""), "x", "inpaint", 0.5, 0, **kw)
            said = "no error"
        except ValueError as e:
            said = str(e)
        check(f"region prompt: {what} is said in plain words", word in said and "line " not in said, said)
    import copy
    odd = copy.deepcopy(regions.table)
    odd["regions"][1]["name"], odd["regions"][2]["name"] = "my window", "then"
    odd_r = type(regions)(regions.labels, odd, regions.scope)
    for which, text, want in (("my window", "a 50//50 mix // of red", "a 50//50 mix// of red"), ("then", '"neon" and "glass"', '"neon" and "glass"')):
        got = node.execute(which, text, "inpaint", 0.5, 0, regions=odd_r).args
        painted = [(e["name"], e["prompt"]) for e in got[0].plan["regions"] if e["strategy"] != "keep"]
        check(f"region prompt: the region named '{which}' is the one painted, its prompt arrives whole",
              painted == [(which, want)] and got[1] == which and got[3].startswith("this prompt: 1 region"), str(painted))
    # kubakub versions joins its rules to the plan's text: on a region prompt plan they go on top of every node
    base = first[0].plan["rules_text"].rstrip("\n")
    joined = base + ("\n[then]\n" if "[then]" in base else "\n") + "[default]\nprompt += copper\ndenoise = 0.33\n"
    look = {e["name"]: e for e in rp.resolve(regions.table, joined)["regions"]}
    check("region prompt: a version's look and settings reach the regions it paints",
          "[then]" in base and all(look[n]["prompt"] == "stained glass, copper" and look[n]["denoise"] == 0.33 for n in wins)
          and all(look[n]["strategy"] == "keep" for n in names if n not in wins))
    empty = node.execute("W_*", " ", "inpaint", 0.5, 0, regions=regions).args[3]
    check("region prompt: an empty prompt is noted", "note: the prompt is empty" in empty)


def test_frame_in_frame_regions():
    schema = nf.KUBA_FrameCompose.define_schema()
    check("frame in frame: regions is a new last output", ids(schema.outputs) == [
        "image", "plan", "area", "generate", "back", "walls", "shadow", "frames_json", "report", "regions"])
    H, W = 400, 800
    lab = np.full((H, W), -1, np.int32)
    lab[150:190, 380:420] = 0
    lab[300:340, 100:140] = 1
    table = {"width": W, "height": H, "groups": {"Windows": [0, 1]}, "regions": [
        {"region_id": 0, "name": "W_A", "group_id": "Windows", "bbox": [380, 150, 40, 40], "area": 1600, "tags": []},
        {"region_id": 1, "name": "W_B", "group_id": "Windows", "bbox": [100, 300, 40, 40], "area": 1600, "tags": []}]}
    regions = Regions.from_numpy(lab, table)
    viewer = vw.Viewer(W, H, 40.0, 0.0, None, 1.7, 30.0)
    matrix = torch.full((1, H, W, 3), 0.8)

    out = nf.KUBA_FrameCompose.execute(plan_of("[W_A]\nprompt = a box\nfif_depth_m = -2\n", regions), viewer,
                                       matrix).args
    plan2, regs = out[1], out[9]
    check("frame in frame: the regions output holds the parasite's region", isinstance(regs, Regions)
          and regs.count == 3 and regs.table["regions"][2]["name"] == "W_A_out", str(regs.table["regions"][-1]))
    check("frame in frame: they are the regions of the plan output", regs is plan2.regions
          and int((regs.labels == 2).sum()) > 1600 and regions.count == 2)
    plain = plan_of("[W_A]\nprompt = a room\nfif_depth_m = 3\n", regions)
    out = nf.KUBA_FrameCompose.execute(plain, viewer, matrix).args
    check("frame in frame: without parasites the plan's own regions", out[9] is regions and out[1] is plain)


def test_versions():
    schema = nv.KUBA_Versions.define_schema()
    check("versions: new outputs at the end", ids(schema.outputs) == ["images", "sheet", "report", "save_path",
                                                                      "names"])
    check("versions: only and cfg are the last inputs, optional and advanced", ids(schema.inputs)[-2:]
          == ["only", "cfg"] and all(i.optional and i.advanced and i.tooltip for i in schema.inputs[-2:])
          and ids(schema.inputs)[-3] == "generate")
    by = {i.id: i for i in schema.inputs}
    check("versions: their defaults change nothing", by["only"].default == "" and by["cfg"].default == 0.0)

    rules = ("[default]\nprompt = red\ndenoise = 1\ncolor_match = none\nfeather_px = 0\ndilate_px = 0\n"
             "[W_*]\nprompt += glass\n[sign]\nprompt += sign\n")
    sheet = "[look]\ncopper = copper plates\npaper = white paper\n"
    canvas = seq.canvas_img()

    def run(quality="draft", rules_text=None, **kw):
        NodeAdapter.made.clear()
        out = nv.KUBA_Versions.execute(None, None, None, plan_of(rules_text or rules), canvas, sheet, "rotate", quality, "", 12,
                                       0, 0.25, 0.4, 480, **kw).args
        samples = [c for ad in NodeAdapter.made for c in ad.calls]
        cfgs = [c for ad in NodeAdapter.made for c in ad.cfgs]
        return out, samples, cfgs

    out, samples, cfgs = run(save_folder="kubakub/versions")
    images, _, report, save_path, names = out
    folder = os.path.join(TMP, "output", "kubakub", "versions")
    check("versions: save_path is the folder of the drafts", os.path.normcase(save_path) == os.path.normcase(
        os.path.realpath(folder)) and len(os.listdir(save_path)) == 2, save_path)
    check("versions: names follow the images", images.shape[0] == 2 and names.split("\n") == ["01  copper",
                                                                                            "02  paper"], names)
    check("versions: cfg 0 as before", len(samples) == 6 and set(cfgs) == {0.0}, f"{len(samples)} {set(cfgs)}")
    wall = seq.make_regions().labels[0] == 0
    check("versions: every region is painted", float((images[0][wall] - canvas[0][wall]).abs().mean()) > 0.05)

    out, samples, cfgs = run(save_folder="")
    check("versions: nothing saved = empty save_path", out[3] == "" and "saved" not in out[2])

    out, samples, cfgs = run(save_folder="", only="W_*", cfg=3.5)
    images, report = out[0], out[2]
    win = seq.make_regions().labels[0] == 1
    check("versions: only = one sample per version (the two windows together)", len(samples) == 2,
          str(len(samples)))
    check("versions: only leaves the other regions as they are", torch.equal(images[0][wall], canvas[0][wall])
          and float((images[0][win] - canvas[0][win]).abs().mean()) > 0.05)
    check("versions: the report names only", "only = W_*: 2 region(s) painted" in report, report)
    check("versions: cfg reaches the sampler", set(cfgs) == {3.5}, str(set(cfgs)))

    out, samples, cfgs = run("final", save_folder="kubakub/versions", only="W_*", cfg=2.0)
    images = out[0]
    check("versions final: save_path is the subfolder final", os.path.basename(out[3]) == "final"
          and len(os.listdir(out[3])) == 2, out[3])
    check("versions final: regions left out by only are the input image", torch.equal(images[0][wall],
                                                                                    canvas[0][wall]))
    check("versions final: cfg reaches the detail pass too", len(cfgs) > 2 and set(cfgs) == {2.0}, str(set(cfgs)))

    out, samples, cfgs = run("final", rules_text=rules + "[region:0]\nstrategy = keep\n", save_folder="")
    images = out[0]
    check("versions final: a 'keep' region of the plan is the input image, not its scaled-up draft",
          torch.equal(images[0][wall], canvas[0][wall]) and float((images[0][win] - canvas[0][win]).abs().mean()) > 0.05)

    out, samples, cfgs = run(save_folder="", only="nothing_*")
    check("versions: only that matches nothing paints nothing and says so", not samples
          and "no region matches" in out[2] and torch.equal(out[0][0], canvas[0]))
    try:
        run(save_folder="", only="floor:1")
        check("versions: a wrong selector in only is an error", False)
    except ValueError as e:
        check("versions: a wrong selector in only is an error", "kubakub versions: only:" in str(e), str(e))


def test_only_regions_and_cfg():
    regions = seq.make_regions()
    d = rp.resolve(regions.table, "[default]\nprompt = red\n[sign]\nstrategy = keep\n")
    same, n = vs.only_regions(d, "  ")
    check("only_regions: empty = the plan itself", same is d and n == 3)
    p, n = vs.only_regions(d, "group:Windows, sign")
    check("only_regions: the others are kept, the order follows", n == 2 and p["order"] == [1, 2]
          and [e["strategy"] for e in p["regions"]] == ["keep", "inpaint", "inpaint", "keep"], str(p["order"]))
    check("only_regions: the input is not changed", d["regions"][0]["strategy"] == "inpaint" and len(d["order"]) == 3)
    p, n = vs.only_regions(d, "!W_*")
    check("only_regions: NOT works", n == 1 and p["order"] == [0], str(p["order"]))

    ad = NodeAdapter()
    seams.refine_tiles(ad, seq.canvas_img(), RegionPlan(d, regions), tile_px=128, cfg=4.0)
    check("refine_tiles: cfg is passed on", ad.cfgs and set(ad.cfgs) == {4.0})
    ad = NodeAdapter()
    seams.refine_tiles(ad, seq.canvas_img(), RegionPlan(d, regions), tile_px=128)
    check("refine_tiles: cfg 0 without it", set(ad.cfgs) == {0.0})


if __name__ == "__main__":
    try:
        for t in (test_region_sampler, test_material, test_seam_pass_schema, test_plan_preview, test_region_prompt,
                  test_frame_in_frame_regions,
                  test_versions, test_only_regions_and_cfg):
            t()
    finally:
        shutil.rmtree(TMP, ignore_errors=True)
    print()
    if failures:
        print(f"{len(failures)} FAILED: {', '.join(failures)}")
        sys.exit(1)
    print("all generate node tests passed")
