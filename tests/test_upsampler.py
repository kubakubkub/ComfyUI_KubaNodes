"""
Model free test of the region sampler's prompt upsampler and first pass (kubakub/upsampler.py, strategies.run_plan):
with the fake adapter of test_sequential.py (paints red when the prompt says red, else blue) and a fake upsampler
that writes 'INSTR: <prompt>' and records what pictures it was shown. Does not start ComfyUI and loads no model.

Run from the ComfyUI_windows_portable folder:
    python_embeded\\python.exe ComfyUI\\custom_nodes\\ComfyUI_KubaNodes\\tests\\test_upsampler.py
"""

import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
from kubakub import plan as rp, strategies as st, upsampler as up  # noqa: E402
from kubakub.types import RegionPlan  # noqa: E402
import test_sequential as seq  # noqa: E402

failures = []


def check(name, cond, detail=""):
    print(("ok   " if cond else "FAIL ") + name + (f"  ({detail})" if detail and not cond else ""))
    if not cond:
        failures.append(name)


class FakeUpsampler:
    """tokenize / generate / decode as a core CLIP that generates text."""

    def __init__(self, think=True):
        self.asked, self.think = [], think

    def tokenize(self, text, images=(), system_prompt="", **kw):
        self.asked.append({"prompt": text, "sizes": [tuple(i.shape[1:3]) for i in images], "system": system_prompt})
        return text

    def generate(self, tokens, **kw):
        self.kw = kw
        return tokens

    def decode(self, ids):
        return ("<think>planning</think>" if self.think else "") + f'"INSTR: {ids}"'


RULES = """
[default]
prompt = wall
strategy = keep
[group:Windows]
strategy = inpaint
prompt = red glass
feather_px = 0
dilate_px = 0
color_match = none
"""


def plan_of(rules=RULES):
    regions = seq.make_regions()
    return RegionPlan(rp.resolve(regions.table, rules, seed=5), regions)


def test_expand():
    up.CACHE = up.Instructions()
    fu = FakeUpsampler()
    pic = seq.canvas_img()
    text, hit = up.expand(fu, "red glass", [pic])
    check("the instruction comes back without its reasoning block and quotes", text == "INSTR: red glass", text)
    check("... not from the cache the first time", not hit)
    check("the edit system prompt of the Flux 3 templates is used", "prompt upsampler of an image-editing model"
          in fu.asked[0]["system"])
    check("sampled as the templates do: the most likely text (top_k 1), temperature 1",
          fu.kw.get("top_k") == 1 and fu.kw.get("temperature") == 1.0 and fu.kw.get("do_sample") is True)
    text2, hit2 = up.expand(fu, "red glass", [pic])
    check("the same prompt and picture: from the cache, the upsampler is not asked again", hit2 and len(fu.asked) == 1)
    up.expand(fu, "red glass", [pic * 0.5])
    check("another picture: asked again", len(fu.asked) == 2)
    big = torch.rand(1, 2160, 3840, 3)
    up.expand(fu, "x", [big])
    check("a large picture is shown at most 1024 px on its long side", max(fu.asked[-1]["sizes"][0]) == 1024,
          str(fu.asked[-1]["sizes"]))
    check("an empty prompt stays empty, nothing is asked", up.expand(fu, "  ", [pic]) == ("", True)
          and len(fu.asked) == 3)


def test_regions():
    up.CACHE = up.Instructions()
    plan = plan_of()
    fu = FakeUpsampler(think=False)
    fa = seq.FakeAdapter()
    out, changed, results = st.run_plan(fa, seq.canvas_img(), plan, st.SamplerSettings(), upsampler=fu)
    check("every painted region gets its own instruction", sorted(c["prompt"] for c in fa.calls)
          == ["INSTR: red glass", "INSTR: red glass"], str([c["prompt"] for c in fa.calls]))
    check("... and the model encodes those up front", "INSTR: red glass" in fa.encoded)
    check("the upsampler sees each region's own crop", len(fu.asked) == 2 and all(len(a["sizes"]) == 1 for a in fu.asked))
    check("the plan itself is not changed", plan.plan["regions"][1]["prompt"] == "red glass")
    check("keep regions are not asked about", all(a["prompt"] == "red glass" for a in fu.asked))
    rep = st.results_report(results)
    check("the report shows what the upsampler wrote", "instructions" in rep and "W_1: INSTR: red glass" in rep, rep)
    check("the result is painted as before", torch.allclose(out[0, 60:140, 80:160], torch.tensor([1.0, 0, 0]).expand(80, 80, 3)))
    n = len(fu.asked)
    st.run_plan(seq.FakeAdapter(), seq.canvas_img(), plan, st.SamplerSettings(), upsampler=fu)
    check("a second queue: the instructions come from the cache", len(fu.asked) == n)
    # reference = style: the style picture is the second image
    up.CACHE = up.Instructions()
    fu2 = FakeUpsampler()
    style = torch.rand(1, 64, 96, 3)
    st.run_plan(seq.FakeAdapter(), seq.canvas_img(), plan_of(RULES + "reference = style\n"),
                st.SamplerSettings(style=style), upsampler=fu2)
    check("reference = style: the upsampler sees the crop and then the style picture",
          all(len(a["sizes"]) == 2 and a["sizes"][1] == (64, 96) for a in fu2.asked), str([a["sizes"] for a in fu2.asked]))


def test_first_pass():
    up.CACHE = up.Instructions()
    entries = plan_of().plan["regions"]
    t = st.first_pass_instruction(entries, [1, 2], "flux3")
    check("first pass instruction: Flux 3 is told the picture by its reference token", t.startswith("In <ref_image_0>"))
    check("... each distinct prompt once, then what stays", t.count("red glass") == 1 and "Keep the rest unchanged" in t, t)
    check("other models get it without the token", not st.first_pass_instruction(entries, [1, 2], "flux2").startswith("In <"))

    plan = plan_of()
    fa = seq.FakeAdapter()
    img = seq.canvas_img()
    out, changed, results = st.run_plan(fa, img, plan, st.SamplerSettings(), first_pass=1.0)
    first = fa.calls[0]
    check("the first sample is the whole picture from the plan's prompts", "red glass" in first["prompt"]
          and "Keep the rest unchanged" in first["prompt"] and first["mask"] is None, str(first))
    check("... with the picture as reference", first["ref"])
    check("... at about region_mp, on the grid", first["size"][0] % 16 == 0 and first["size"][1] % 16 == 0
          and abs(first["size"][0] / first["size"][1] - 480 / 270) < 0.1, str(first["size"]))
    check("then the regions refine it", len(fa.calls) == 3)
    win = torch.ones(270, 480, dtype=torch.bool)
    win[60:140, 80:160] = False
    win[60:140, 320:400] = False
    check("keep regions stay pixel exact (the first pass is pasted where the painted regions are)",
          torch.equal(out[0][win], img[0][win]))
    rep = st.results_report(results)
    check("the report has a first pass line and does not count it as a region", "first pass" in rep
          and rep.startswith("2 regions done"), rep)
    st.run_plan(fa, img, plan, st.SamplerSettings(), first_pass=1.0)       # fake adapter: no model, no cache
    fu = FakeUpsampler()
    fa2 = seq.FakeAdapter()
    st.run_plan(fa2, img, plan, st.SamplerSettings(), first_pass=0.8, upsampler=fu)
    check("first pass + upsampler: it is asked about the whole picture first", fu.asked[0]["sizes"] == [(270, 480)]
          and "Keep the rest unchanged" in fu.asked[0]["prompt"])
    check("... and the first sample uses what it wrote", fa2.calls[0]["prompt"].startswith("INSTR: "))
    none = plan_of("[default]\nstrategy = keep\nprompt = wall\n")
    fa3 = seq.FakeAdapter()
    _, _, r3 = st.run_plan(fa3, img, none, st.SamplerSettings(), first_pass=1.0)
    check("nothing painted: no first pass", not fa3.calls)


test_expand()
test_regions()
test_first_pass()
print()
print("FAILED: " + ", ".join(failures) if failures else "all upsampler tests passed")
sys.exit(1 if failures else 0)
