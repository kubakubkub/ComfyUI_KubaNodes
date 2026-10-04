"""
Model free test for the Region Sampler caches (kubakub/region_cache.py, strategies.py, adapters.py)
and the per-region CPU work (crop-first masks, in-place paste).

Stub model / clip / vae objects stand in for ComfyUI's; the adapters module is imported with stub
comfy modules. Checks: the per-region result cache (hit, miss, invalidation by prompt, seed, crop
pixels, model object and model patches, eviction by bytes), that cached output equals a fresh run
exactly, the Qwen-style prefetch only encodes regions that will be sampled, the text / vision
encoding cache, and the output-node flag of kubakub regions to vector. Does not start ComfyUI and
loads no model.

Run from the ComfyUI_windows_portable folder:
    python_embeded\\python.exe ComfyUI\\custom_nodes\\ComfyUI_KubaNodes\\tests\\test_region_cache.py
"""

import gc
import os
import sys
import types
import uuid
import zlib

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
PACK = os.path.dirname(HERE)
sys.path.insert(0, PACK)
sys.path.insert(0, HERE)
from kubakub import ops, plan as rp, region_cache as rc, strategies as st  # noqa: E402
from kubakub.types import RegionPlan  # noqa: E402
from test_sequential import FakeAdapter, canvas_img, make_regions  # noqa: E402

failures = []


def check(name, cond, detail=""):
    print(("ok   " if cond else "FAIL ") + name + (f"  ({detail})" if detail and not cond else ""))
    if not cond:
        failures.append(name)


class Thing:
    """A stand-in model patcher / clip / vae: weakly referencable, with a patch fingerprint."""

    def __init__(self):
        self.patches_uuid = uuid.uuid4()


class StubAdapter(FakeAdapter):
    """FakeAdapter with model identities; the sample depends on the seed, the prompt and the latent."""
    family = "stub"
    schedule = "auto"

    def __init__(self, model, clip, vae):
        super().__init__()
        self.model, self.clip, self.vae = model, clip, vae

    def sample(self, latent, pos, neg, noise_mask=None, seed=0, **kw):
        self.calls.append({"prompt": pos[0][0], "seed": seed})
        g = torch.Generator().manual_seed(int(seed) * 7919 + zlib.crc32(pos[0][0].encode()))
        colour = torch.rand(3, generator=g).view(1, 3, 1, 1)
        return latent * 0.5 + colour * 0.5


class StubPrefetch(StubAdapter):
    """Like Qwen 2.1: the conditioning holds the crop image it was encoded from."""
    prefetches = True
    default_cfg = 1.0

    def __init__(self, *a):
        super().__init__(*a)
        self.pre, self.prefetch_log = {}, []

    def prefetch(self, key, prompt, negative, pixels, need_negative=True):
        self.prefetch_log.append(key)
        self.pre[key] = ([[f"{prompt}|{float(pixels.mean()):.6f}", {}]], [[negative, {}]])

    def conds_for(self, prompt, negative, pixels=None, latent=None, key=None):
        if key is not None and key in self.pre:
            pos, neg = self.pre.pop(key)
            return self.add_reference(pos, latent), neg
        return super().conds_for(prompt, negative, pixels, latent)


RULES = """
[default]
prompt = wall
strategy = keep
[group:Windows]
strategy = inpaint
prompt = red glass
context_px = 96
[sign]
strategy = inpaint
prompt = blue sign
context_px = 24
"""


def make_plan(rules=RULES, seed=5):
    regions = make_regions()
    return RegionPlan(rp.resolve(regions.table, rules, seed=seed), regions)


def run(adapter, plan, img=None, use_cache=True, **settings):
    img = canvas_img() if img is None else img
    return st.run_plan(adapter, img, plan, st.SamplerSettings(**settings), use_cache=use_cache)


# --------------------------------------------------------------------------
# the LRU
# --------------------------------------------------------------------------

def test_lru():
    lru = rc.ByteLRU("t", 3 * (4000 + 256))
    owner = Thing()
    for i in range(3):
        lru.put(i, torch.zeros(1000), (owner,))
    check("lru holds entries up to its byte budget", len(lru) == 3 and lru.get(0) is not None)
    lru.put(3, torch.zeros(1000), (owner,))
    check("lru evicts the least recently used", lru.get(1) is None and lru.get(0) is not None and len(lru) == 3)
    check("lru refuses a value larger than the budget", not lru.put(9, torch.zeros(100_000), (owner,)))
    t = torch.arange(12.0)
    lru.put("x", t, (owner,))
    check("lru returns the exact stored tensor", lru.get("x") is t)
    other = Thing()
    lru.put("y", torch.zeros(10), (other,))
    del other
    gc.collect()
    check("an entry dies with its owner", lru.get("y") is None)
    check("owners that cannot be weakly referenced are not cached", not lru.put("z", torch.zeros(1), (1,)))
    check("digest sees every value", rc.digest(torch.zeros(4)) != rc.digest(torch.tensor([0, 0, 0, 1e-7])))
    check("digest sees shape and dtype", rc.digest(torch.zeros(4)) != rc.digest(torch.zeros(2, 2))
          and rc.digest(torch.zeros(4)) != rc.digest(torch.zeros(4, dtype=torch.float16)))
    check("digest of a strided view equals its copy", rc.digest(torch.arange(20.0).view(4, 5)[:, 1:3])
          == rc.digest(torch.arange(20.0).view(4, 5)[:, 1:3].clone()))
    check("nbytes walks conditioning lists", rc.nbytes([[torch.zeros(10), {"a": torch.zeros(5)}]]) == 60 + 256)


# --------------------------------------------------------------------------
# the result cache
# --------------------------------------------------------------------------

def test_result_cache():
    rc.clear_all()
    model, clip, vae = Thing(), Thing(), Thing()
    plan = make_plan()

    ref_out, ref_changed, _ = run(StubAdapter(model, clip, vae), plan, use_cache=False)
    a1 = StubAdapter(model, clip, vae)
    out1, ch1, res1 = run(a1, plan)
    n_all = len(a1.calls)
    check("first run samples every region", n_all == 3, str(a1.calls))
    check("first run equals an uncached run", torch.equal(out1, ref_out) and torch.equal(ch1, ref_changed))

    a2 = StubAdapter(model, clip, vae)                 # a new queue: a new adapter, the same model objects
    out2, ch2, res2 = run(a2, plan)
    check("unchanged plan: nothing sampled", len(a2.calls) == 0, str(a2.calls))
    check("cache hit output is identical", torch.equal(out2, out1) and torch.equal(ch2, ch1))
    check("report counts the cached regions", "3 from the cache" in st.results_report(res2), st.results_report(res2))

    # one prompt changes: only that region (and later regions whose crops overlap it) re-samples
    plan_b = make_plan(RULES.replace("prompt = blue sign", "prompt = green sign"))
    a3 = StubAdapter(model, clip, vae)
    out3, ch3, _ = run(a3, plan_b)
    fresh3, fresh_ch3, _ = run(StubAdapter(model, clip, vae), plan_b, use_cache=False)
    prompts = [c["prompt"] for c in a3.calls]
    check("one changed prompt: that region re-samples", "green sign" in prompts, str(prompts))
    check("one changed prompt: the others come from the cache", len(a3.calls) < n_all, str(prompts))
    check("one changed prompt: output equals a fresh run", torch.equal(out3, fresh3) and torch.equal(ch3, fresh_ch3))

    a4 = StubAdapter(model, clip, vae)
    run(a4, plan, seed_offset=1)
    check("seed change: every region re-samples", len(a4.calls) == n_all)

    img = canvas_img()
    img[0, 100, 120] = 0.0                            # one pixel inside window 1's crop
    a5 = StubAdapter(model, clip, vae)
    out5, _, _ = run(a5, plan, img=img)
    fresh5, _, _ = run(StubAdapter(model, clip, vae), plan, img=img, use_cache=False)
    check("changed crop pixels: that region re-samples", 1 <= len(a5.calls) < n_all, str(a5.calls))
    check("changed crop pixels: output equals a fresh run", torch.equal(out5, fresh5))

    a6 = StubAdapter(Thing(), clip, vae)              # e.g. a LoRA: a new model patcher object
    run(a6, plan)
    check("new model object: every region re-samples", len(a6.calls) == n_all)
    model.patches_uuid = uuid.uuid4()                 # patches added to the same patcher
    a7 = StubAdapter(model, clip, vae)
    run(a7, plan)
    check("model patches changed: every region re-samples", len(a7.calls) == n_all)
    a8 = StubAdapter(model, clip, Thing())
    run(a8, plan)
    check("new vae: every region re-samples", len(a8.calls) == n_all)

    a9 = StubAdapter(model, clip, vae)
    run(a9, plan, use_cache=False)
    check("use_cache off: every region samples", len(a9.calls) == n_all)
    check("adapters without a model are never cached", rc.run_cache(FakeAdapter()) is None)

    old = rc.RESULTS.max_bytes
    try:
        rc.RESULTS.clear()
        rc.RESULTS.max_bytes = 1
        a10, a11 = StubAdapter(model, clip, vae), StubAdapter(model, clip, vae)
        run(a10, plan)
        run(a11, plan)
        check("budget too small: nothing kept, everything samples", len(a11.calls) == n_all and len(rc.RESULTS) == 0)
    finally:
        rc.RESULTS.max_bytes = old


def test_fif_cache():
    rc.clear_all()
    model, clip, vae = Thing(), Thing(), Thing()
    rules = ("[default]\nstrategy = keep\n[W_1]\nstrategy = frame_in_frame\nprompt = red room\nfif_harmonize = 0.3\n"
             "fif_border_px = 12\n[W_2]\nstrategy = frame_in_frame\nprompt = blue room\nfif_harmonize = 0\n")
    plan = make_plan(rules)
    a1 = StubAdapter(model, clip, vae)
    out1, _, _ = run(a1, plan, region_mp=0.25)
    a2 = StubAdapter(model, clip, vae)
    out2, _, res2 = run(a2, plan, region_mp=0.25)
    check("frame_in_frame: scenes and harmonize come from the cache", len(a1.calls) == 3 and len(a2.calls) == 0,
          f"{len(a1.calls)} {len(a2.calls)}")
    check("frame_in_frame: cached output identical", torch.equal(out1, out2))
    check("frame_in_frame: report marks cached regions", all("cached" in r.note for r in res2 if r.status == "done"))


def test_prefetch_cache():
    rc.clear_all()
    model, clip, vae = Thing(), Thing(), Thing()
    plan = make_plan()
    ref, _, _ = run(StubPrefetch(model, clip, vae), plan, use_cache=False)
    p1 = StubPrefetch(model, clip, vae)
    out1, _, _ = run(p1, plan)
    check("prefetch: first run encodes every region", len(p1.prefetch_log) == 3 and len(p1.calls) == 3)
    check("prefetch: first run equals an uncached run", torch.equal(out1, ref))

    p2 = StubPrefetch(model, clip, vae)
    out2, _, _ = run(p2, plan)
    check("prefetch: unchanged plan encodes and samples nothing", not p2.prefetch_log and not p2.calls,
          f"{p2.prefetch_log} {len(p2.calls)}")
    check("prefetch: cached output identical", torch.equal(out2, out1))

    plan_b = make_plan(RULES.replace("prompt = blue sign", "prompt = green sign"))
    p3 = StubPrefetch(model, clip, vae)
    out3, _, _ = run(p3, plan_b)
    fresh, _, _ = run(StubPrefetch(model, clip, vae), plan_b, use_cache=False)
    check("prefetch: one changed prompt encodes only what is sampled",
          sorted(p3.prefetch_log) == sorted(set(p3.prefetch_log)) and len(p3.prefetch_log) == len(p3.calls) < 3,
          f"{p3.prefetch_log} {len(p3.calls)}")
    check("prefetch: output equals a fresh run", torch.equal(out3, fresh))

    # predicted hits that miss after all (evicted in between): encoded from the same pre-paint image
    orig = st._predicted_misses
    try:
        st._predicted_misses = lambda *a, **k: []
        rc.RESULTS.clear()
        p4 = StubPrefetch(model, clip, vae)
        out4, _, _ = run(p4, plan)
        check("prefetch fallback: every miss encoded on demand", len(p4.prefetch_log) == 3 and len(p4.calls) == 3)
        check("prefetch fallback: output equals a fresh run", torch.equal(out4, ref))
    finally:
        st._predicted_misses = orig


# --------------------------------------------------------------------------
# identical per-region CPU work
# --------------------------------------------------------------------------

def _old_region_masks(plan, rid, cp):
    lab = plan.regions.labels[:1]
    m = ops.crop((lab == rid).float(), cp)
    if plan.regions.scope is not None:
        scope = ops.crop(plan.regions.scope[:1].float(), cp)
    else:
        scope = ops.crop(torch.ones_like(lab, dtype=torch.float32), cp)
    return m, scope


def test_cpu_identical():
    boxes = [ops.CropPlan(60, 40, 128, 128, 1, 16, 128, 128), ops.CropPlan(-30, -20, 160, 96, 1, 16, 160, 96),
             ops.CropPlan(400, 200, 160, 112, 1, 16, 160, 112), ops.CropPlan(-100, -100, 700, 500, 1, 16, 700, 500)]
    ok_m = ok_p = True
    for scope_rows in (None, (0, 100)):
        regions = make_regions(scope_rows)
        plan = RegionPlan(rp.resolve(regions.table, "[*]\nprompt = x"), regions)
        for cp in boxes:
            for rid in (0, 1, 3):
                a, b = st.region_masks(plan, rid, cp), _old_region_masks(plan, rid, cp)
                ok_m &= torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
    torch.manual_seed(0)
    for cp in boxes:
        canvas, changed = torch.rand(1, 270, 480, 3), torch.rand(1, 270, 480)
        patch, alpha = torch.rand(1, cp.h, cp.w, 3), torch.rand(1, cp.h, cp.w)
        ref_c = ops.paste(canvas, patch, alpha, cp)
        ref_m = ops.paste(changed[..., None], torch.ones_like(patch[..., :1]), alpha, cp)[..., 0]
        c, m = canvas.clone(), changed.clone()
        st._paste_(c, patch, alpha, cp)
        st._paste_changed_(m, alpha, cp)
        ok_p &= torch.equal(c, ref_c) and torch.equal(m, ref_m)
    check("crop-first region masks equal the full-map ones (scope, off-canvas boxes)", ok_m)
    check("in-place paste equals ops.paste exactly", ok_p)


# --------------------------------------------------------------------------
# the text / vision encoding cache in adapters.py (stub comfy modules)
# --------------------------------------------------------------------------

def _import_adapters():
    names = ["comfy", "comfy.model_base", "comfy.model_management", "comfy.sample", "comfy.samplers", "node_helpers"]
    saved = {n: sys.modules.get(n) for n in names}
    mods = {n: types.ModuleType(n) for n in names}
    mods["comfy.model_base"].Flux2 = type("Flux2", (), {})
    mods["node_helpers"].conditioning_set_values = \
        lambda cond, values, append=False: [[c[0], {**c[1], **values}] for c in cond]
    for n, m in mods.items():
        sys.modules[n] = m
        if "." in n:
            setattr(mods["comfy"], n.split(".")[1], m)
    try:
        sys.modules.pop("kubakub.adapters", None)
        from kubakub import adapters
        return adapters
    finally:
        for n, m in saved.items():
            if m is None:
                sys.modules.pop(n, None)
            else:
                sys.modules[n] = m


class StubClip:
    def __init__(self):
        self.encodes = []

    def tokenize(self, text, images=None, keep_vision=False, prevent_empty_text=False):
        return (text, [] if images is None else list(images))

    def encode_from_tokens_scheduled(self, tokens):
        text, images = tokens
        self.encodes.append((text, len(images)))
        v = float(len(text)) + sum(float(i.mean()) for i in images)
        return [[torch.full((1, 4, 8), v), {}]]


def test_cond_cache():
    rc.clear_all()
    ad = _import_adapters()
    model = Thing()
    model.model_options = {"transformer_options": {"qwen_image21_cache": {"device": "gpu"}}}
    clip, vae = StubClip(), Thing()

    a1 = ad.Adapter(model, clip, vae)
    a1.encode_prompts(["a", "b", "a"])
    check("text: each distinct prompt encoded once", clip.encodes == [("a", 0), ("b", 0)], str(clip.encodes))
    a2 = ad.Adapter(model, clip, vae)                  # the next queue
    a2.encode_prompts(["a", "b", "c"])
    check("text: the next queue encodes only the new prompt", clip.encodes[2:] == [("c", 0)], str(clip.encodes))
    check("text: cached conditioning is the same object", a2.cond("a") is a1.cond("a"))
    clip2 = StubClip()
    ad.Adapter(model, clip2, vae).encode_prompts(["a"])
    check("text: another text encoder encodes again", clip2.encodes == [("a", 0)])

    q1 = ad.QwenImage21Adapter(model, clip, vae)
    img = torch.rand(1, 64, 64, 3)
    n0 = len(clip.encodes)
    q1.encode_prompts(["a"])
    check("text: cached per adapter kind (Qwen encodes its own text form)", len(clip.encodes) == n0 + 1)
    n0 = len(clip.encodes)
    q1.prefetch(1, "red", "a", img, need_negative=False)
    check("vision: prefetch encodes the prompt with the crop", clip.encodes[n0:] == [("red", 1)], str(clip.encodes[n0:]))
    q2 = ad.QwenImage21Adapter(model, clip, vae)
    n0 = len(clip.encodes)
    q2.prefetch(1, "red", "a", img.clone(), need_negative=False)
    check("vision: same text + same crop pixels is a cache hit", len(clip.encodes) == n0)
    img2 = img.clone()
    img2[0, 5, 5, 0] += 0.01
    q2.prefetch(2, "red", "a", img2, need_negative=True)
    check("vision: a changed crop (and the image negative) encode again",
          clip.encodes[n0:] == [("a", 1), ("red", 1)], str(clip.encodes[n0:]))
    n0 = len(clip.encodes)
    pos, neg = ad.QwenImage21Adapter(model, clip, vae).conds_for("red", "a", img2, torch.zeros(1, 4, 2, 2))
    check("vision: conds_for without prefetch uses the cache too", len(clip.encodes) == n0
          and "reference_latents" in pos[0][1])
    key2 = (rc.identity(clip2), "text:Adapter", "a", "-")
    alive = rc.CONDS.get(key2) is not None
    del clip2
    gc.collect()
    check("conditionings of a freed text encoder are dropped", alive and rc.CONDS.get(key2) is None)


# --------------------------------------------------------------------------
# kubakub regions to vector is an output node
# --------------------------------------------------------------------------

def test_vector_output_node():
    comfy_root = os.path.abspath(os.path.join(PACK, "..", ".."))
    sys.path.insert(0, comfy_root)
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))     # embedded Python leaves the script folder off the path
        import _pack  # noqa: E402,F401  (the pack as a package, see tests/_pack.py)
        from kubapack.nodes.regions import nodes_vector
        schema = nodes_vector.KUBA_RegionsToVector.define_schema()
    except Exception as e:  # noqa: BLE001  (ComfyUI's comfy_api not importable here)
        import ast
        src = open(os.path.join(PACK, "nodes", "regions", "nodes_vector.py"), encoding="utf-8").read()
        flag = any(isinstance(n, ast.keyword) and n.arg == "is_output_node" and getattr(n.value, "value", None) is True
                   for n in ast.walk(ast.parse(src)))
        check(f"regions to vector: is_output_node (source check, {type(e).__name__})", flag)
        return
    finally:
        sys.path.remove(comfy_root)
    check("regions to vector: is_output_node", schema.is_output_node is True)
    check("regions to vector: still has its outputs for use as an intermediate node",
          [getattr(o, "io_type", None) for o in schema.outputs] == ["IMAGE", "STRING", "STRING"],
          str([getattr(o, "io_type", None) for o in schema.outputs]))


if __name__ == "__main__":
    for t in (test_lru, test_result_cache, test_fif_cache, test_prefetch_cache, test_cpu_identical, test_cond_cache,
              test_vector_output_node):
        t()
    print()
    if failures:
        print(f"{len(failures)} FAILED: {', '.join(failures)}")
        sys.exit(1)
    print("all region cache tests passed")
