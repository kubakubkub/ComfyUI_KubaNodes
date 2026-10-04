"""
Model free test for S1 sequential inpaint (kubakub/strategies.py, ops.py).

A fake adapter stands in for the model: encode = 16x average pool, sample =
paint the latent a solid colour, decode = 16x nearest upscale. That is enough to
check crop planning, exact scaling, masks, scope, pasting and registration.
Does not start ComfyUI and loads no model.

Run from the ComfyUI_windows_portable folder:
    python_embeded\\python.exe ComfyUI\\custom_nodes\\ComfyUI_KubaNodes\\tests\\test_sequential.py
"""

import os
import sys
from fractions import Fraction

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from kubakub import ops, plan as rp, strategies as st  # noqa: E402
from kubakub.types import RegionPlan, Regions  # noqa: E402

failures = []


def check(name, cond, detail=""):
    print(("ok   " if cond else "FAIL ") + name + (f"  ({detail})" if detail and not cond else ""))
    if not cond:
        failures.append(name)


class FakeAdapter:
    grid = 16
    default_region_mp = 0.25

    def __init__(self):
        self.calls = []
        self.encoded = []

    def encode_prompts(self, texts):
        self.encoded += list(texts)

    def cond(self, text):
        return [[text, {}]]

    def add_reference(self, cond, latent):
        return [[cond[0][0], {"reference_latents": [latent]}]]

    def conds_for(self, prompt, negative, pixels=None, latent=None):
        pos = self.cond(prompt)
        if latent is not None:
            pos = self.add_reference(pos, latent)
        return pos, self.cond(negative)

    def encode(self, px):
        assert px.shape[1] % 16 == 0 and px.shape[2] % 16 == 0, px.shape
        return F.avg_pool2d(px.movedim(-1, 1), 16)

    def empty_latent(self, w, h):
        return torch.zeros(1, 3, h // 16, w // 16)

    def decode(self, lat):
        return F.interpolate(lat, scale_factor=16, mode="nearest").movedim(1, -1)

    def sample(self, latent, pos, neg, noise_mask=None, seed=0, **kw):
        self.calls.append({"prompt": pos[0][0], "ref": "reference_latents" in pos[0][1], "seed": seed,
                           "size": kw["pixel_size"], "mask": None if noise_mask is None else tuple(noise_mask.shape)})
        colour = torch.tensor([1.0, 0.0, 0.0]) if "red" in pos[0][0] else torch.tensor([0.0, 0.0, 1.0])
        return colour.view(1, 3, 1, 1).expand_as(latent).clone()


def make_regions(scope_rows=None):
    """480x270 canvas: wall (0), two windows (1, 2), a small sign (3)."""
    H, W = 270, 480
    lab = np.zeros((H, W), np.int32)
    lab[60:140, 80:160] = 1
    lab[60:140, 320:400] = 2
    lab[200:215, 220:260] = 3
    table = {"width": W, "height": H, "regions": [
        {"region_id": 0, "name": "wall", "group_id": "wall", "bbox": [0, 0, W, H], "area": int((lab == 0).sum()), "tags": []},
        {"region_id": 1, "name": "W_1", "group_id": "Windows", "bbox": [80, 60, 80, 80], "area": 6400, "tags": []},
        {"region_id": 2, "name": "W_2", "group_id": "Windows", "bbox": [320, 60, 80, 80], "area": 6400, "tags": []},
        {"region_id": 3, "name": "sign", "group_id": "sign", "bbox": [220, 200, 40, 15], "area": 600, "tags": []},
    ]}
    scope = None
    if scope_rows is not None:
        scope = np.zeros((H, W), bool)
        scope[scope_rows[0]:scope_rows[1]] = True
        lab[~scope] = -1
    return Regions.from_numpy(lab, table, scope)


def canvas_img(H=270, W=480):
    y, x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
    return torch.stack([x / W, y / H, torch.full_like(x, 0.5, dtype=torch.float32)], -1)[None].float()


def test_plan_crop():
    cp = ops.plan_crop([80, 60, 80, 80], 32, 480, 270, grid=16, budget_px=512 * 512, max_upscale=4.0)
    check("crop scale is grid/block", cp.scale == Fraction(16, cp.block))
    check("work = box * scale exactly", cp.work_w == cp.w * cp.scale and cp.work_h == cp.h * cp.scale)
    check("work on grid", cp.work_w % 16 == 0 and cp.work_h % 16 == 0)
    check("box holds region + context", cp.x0 <= 48 and cp.y0 <= 28 and cp.x0 + cp.w >= 192 and cp.y0 + cp.h >= 172,
          str(cp))
    check("upscale capped at 4", cp.scale <= 4)
    big = ops.plan_crop([0, 0, 3840, 2160], 64, 3840, 2160, budget_px=1024 * 1024)
    check("wall crop downscales", big.scale < 1 and big.work_w * big.work_h <= 1024 * 1024 * 1.01, str(big))
    check("oversize box centred", big.x0 < 0 and big.y0 < 0)
    edge = ops.plan_crop([0, 0, 40, 40], 64, 3840, 2160)
    check("box shifted onto canvas", edge.x0 == 0 and edge.y0 == 0)


def test_crop_paste():
    img = canvas_img()
    cp = ops.CropPlan(-10, -10, 64, 64, 16, 16, 64, 64)
    c = ops.crop(img, cp)
    check("crop pads outside", c.shape == (1, 64, 64, 3) and torch.equal(c[0, 0, 0], img[0, 0, 0]))
    out = ops.paste(img, torch.zeros_like(c), torch.ones(1, 64, 64), cp)
    check("paste clips to canvas", out[0, :54, :54].abs().sum() == 0 and torch.equal(out[0, 54:], img[0, 54:]))


def test_sequential():
    regions = make_regions()
    rules = """
[default]
prompt = wall
strategy = keep
[group:Windows]
strategy = inpaint
prompt = red glass
feather_px = 0
dilate_px = 0
color_match = none
[sign]
strategy = inpaint
prompt = blue sign
feather_px = 4
color_match = none
"""
    plan = RegionPlan(rp.resolve(regions.table, rules, seed=5), regions)
    fa = FakeAdapter()
    img = canvas_img()
    out, changed, results = st.sequential_inpaint(fa, img, plan, st.SamplerSettings())
    check("three regions sampled", len(fa.calls) == 3, str(fa.calls))
    check("reference attached", all(c["ref"] for c in fa.calls))
    check("seeds from plan", sorted(c["seed"] for c in fa.calls) == [6, 7, 8])
    check("prompts encoded up front", "red glass" in fa.encoded and "blue sign" in fa.encoded)
    win = out[0, 60:140, 80:160]
    check("window 1 painted exactly", torch.allclose(win, torch.tensor([1.0, 0, 0]).expand_as(win)))
    check("window 2 painted", torch.allclose(out[0, 60:140, 320:400], torch.tensor([1.0, 0, 0]).expand(80, 80, 3)))
    ring = torch.ones(270, 480, dtype=torch.bool)
    ring[60:140, 80:160] = False
    ring[60:140, 320:400] = False
    ring[190:225, 210:270] = False          # the sign with its feather
    check("nothing else changed", torch.equal(out[0][ring], img[0][ring]))
    check("sign interior blue", torch.allclose(out[0, 205:210, 230:250], torch.tensor([0, 0, 1.0]).expand(5, 20, 3)))
    check("changed mask covers windows", changed[0, 60:140, 80:160].min() == 1 and changed[0, 0, 0] == 0)
    check("report", "3 regions done" in st.results_report(results), st.results_report(results))


def test_scope():
    regions = make_regions(scope_rows=(0, 100))    # the lower part of the windows is out of scope
    rules = "[default]\nstrategy = keep\n[W_1]\nstrategy = inpaint\nprompt = red\ndilate_px = 8\nfeather_px = 8\ncolor_match = none"
    plan = RegionPlan(rp.resolve(regions.table, rules), regions)
    img = canvas_img()
    out, _, _ = st.sequential_inpaint(FakeAdapter(), img, plan, st.SamplerSettings())
    check("out of scope untouched", torch.equal(out[0, 100:], img[0, 100:]))
    check("in scope painted", torch.allclose(out[0, 70:90, 90:150], torch.tensor([1.0, 0, 0]).expand(20, 60, 3)))


def test_only_and_mismatch():
    regions = make_regions()
    plan = RegionPlan(rp.resolve(regions.table, "[*]\nprompt = red\ncolor_match = none"), regions)
    fa = FakeAdapter()
    only = lambda e: e["name"] == "W_2"
    st.sequential_inpaint(fa, canvas_img(), plan, st.SamplerSettings(), only=only)
    check("only filter", len(fa.calls) == 1)
    try:
        st.sequential_inpaint(fa, canvas_img(200, 300), plan, st.SamplerSettings())
        check("size mismatch refused", False)
    except ValueError:
        check("size mismatch refused", True)


def test_frame_in_frame():
    regions = make_regions()
    rules = """
[default]
strategy = keep
[W_1]
strategy = frame_in_frame
prompt = blue underwater room
feather_px = 0
fif_harmonize = 0
[W_2]
strategy = frame_in_frame
prompt = red lit room
feather_px = 4
fif_harmonize = 0.3
fif_border_px = 12
"""
    plan = RegionPlan(rp.resolve(regions.table, rules, seed=1), regions)
    fa = FakeAdapter()
    img = canvas_img()
    out, changed, results = st.run_plan(fa, img, plan, st.SamplerSettings(region_mp=0.25))
    check("fif sampled without mask + one harmonize pass", len(fa.calls) == 3, str(fa.calls))
    check("scene size keeps the box aspect", fa.calls[0]["size"][0] == fa.calls[0]["size"][1], str(fa.calls[0]))
    w1 = out[0, 60:140, 80:160]
    check("W_1 scene fills the region exactly", torch.allclose(w1, torch.tensor([0, 0, 1.0]).expand_as(w1)))
    ring = torch.ones(270, 480, dtype=torch.bool)
    ring[60:140, 80:160] = False
    ring[40:160, 300:420] = False            # W_2 with its harmonized border
    check("outside untouched", torch.equal(out[0][ring], img[0][ring]))
    check("W_2 centre red", torch.allclose(out[0, 90:110, 350:370], torch.tensor([1.0, 0, 0]).expand(20, 20, 3)))
    outer = out[0, 52:58, 330:390]
    check("harmonize ring reached outside the window", not torch.equal(outer, img[0, 52:58, 330:390]))
    check("results note fif", any("fif scene" in r.note for r in results), st.results_report(results))
    check("fif_size aspect", st.fif_size([0, 0, 300, 100], 1024 * 1024, 16) == (1776, 592))
    c = st.cover(torch.rand(1, 64, 128, 3), 50, 50)
    check("cover crops to exact size", c.shape == (1, 50, 50, 3))


def test_color_match():
    torch.manual_seed(0)
    ref = torch.rand(1, 32, 32, 3) * 0.5 + 0.25
    img = (ref * 0.6 + 0.3).clamp(0, 1)
    m = torch.ones(1, 32, 32)
    for method in ("mean_std", "mkl"):
        out = ops.color_match(img, ref, m, method)
        err_before = (img - ref).abs().mean()
        err_after = (out - ref).abs().mean()
        check(f"color_match {method} moves towards ref", err_after < err_before * 0.5,
              f"{err_before:.4f} -> {err_after:.4f}")


class PrefetchAdapter(FakeAdapter):
    """Like Qwen 2.1: encodes the crop image with the text, so run_plan encodes all regions first."""
    prefetches = True
    default_cfg = 1.0

    def __init__(self):
        super().__init__()
        self.pre, self.used, self.log = {}, [], []

    def prefetch(self, key, prompt, negative, pixels, need_negative=True):
        self.log.append(("prefetch", key, need_negative))
        self.pre[key] = ([[prompt + "+img", {}]], [[negative + ("+img" if need_negative else ""), {}]])

    def conds_for(self, prompt, negative, pixels=None, latent=None, key=None):
        if key is not None and key in self.pre:
            self.used.append(key)
            pos, neg = self.pre.pop(key)
            return self.add_reference(pos, latent), neg
        return super().conds_for(prompt, negative, pixels, latent)

    def sample(self, *a, **k):
        self.log.append(("sample",))
        return super().sample(*a, **k)


def test_prefetch():
    regions = make_regions()
    rules = """
[default]
prompt = wall
strategy = keep
[group:Windows]
strategy = inpaint
prompt = red glass
color_match = none
"""
    plan = RegionPlan(rp.resolve(regions.table, rules, seed=5), regions)
    pa = PrefetchAdapter()
    out, changed, results = st.sequential_inpaint(pa, canvas_img(), plan, st.SamplerSettings())
    done = [r for r in results if r.status == "done"]
    kinds = [x[0] for x in pa.log]
    check("prefetch: every inpaint region encoded before any sampling",
          len(done) > 0 and kinds.count("prefetch") == len(done) and kinds.index("sample") == len(done), str(kinds))
    check("prefetch: sampling used exactly the prefetched conditioning", sorted(pa.used) == sorted(r.region_id for r in done) and not pa.pre)
    check("prefetch: cfg 1 skips the image negative", all(x[2] is False for x in pa.log if x[0] == "prefetch"))


if __name__ == "__main__":
    for t in (test_plan_crop, test_crop_paste, test_sequential, test_scope, test_only_and_mismatch, test_frame_in_frame,
              test_color_match, test_prefetch):
        t()
    print()
    if failures:
        print(f"{len(failures)} FAILED: {', '.join(failures)}")
        sys.exit(1)
    print("all sequential tests passed")
