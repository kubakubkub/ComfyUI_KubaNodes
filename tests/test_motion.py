"""
Model free test for the director's behaviours (kubakub/director/motion.py) and their twin in the window
(web/kubakub_director.js, the "behaviours" section): the same cases run through both (node.js, when installed) and
must agree, so the window's preview is what the node renders.

Run from the ComfyUI_windows_portable folder:
    python_embeded\\python.exe ComfyUI\\custom_nodes\\ComfyUI_KubaNodes\\tests\\test_motion.py
"""

import json
import os
import re
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from kubakub.director import motion as mo  # noqa: E402
from kubakub.director import render as rd  # noqa: E402

failures = []


def check(name, cond, detail=""):
    print(("ok   " if cond else "FAIL ") + name + (f"  ({detail})" if detail and not cond else ""))
    if not cond:
        failures.append(name)


# ---- the engine on its own
check("hash01 in [0, 1) and spread", all(0 <= mo.hash01(i, 5) < 1 for i in range(1000))
      and 0.45 < sum(mo.hash01(i, 5) for i in range(1000)) / 1000 < 0.55)
ns = [mo.noise(i * 0.013, 3) for i in range(4000)]
check("noise bounded and smooth", max(abs(v) for v in ns) <= 1 and max(abs(a - b) for a, b in zip(ns, ns[1:])) < 0.05)
check("waves", abs(mo.wave("sine", 0.25) - 1) < 1e-12 and abs(mo.wave("triangle", 0)) < 1e-12 and mo.wave("triangle", 0.25) == 1
      and mo.wave("square", 0.2) == 1 and mo.wave("square", 0.7) == -1 and mo.wave("saw", 0.5) == 0)
b = {"type": "oscillate", "path": "rotation", "amount": 10, "period": 2, "t_start": 1, "t_end": 3, "fade": 0.5}
check("range: nothing outside, eased fade", mo.offsets([b], 0.5, 10) == {} and mo.offsets([b], 3.5, 10) == {}
      and abs(mo.envelope(b, 1.25, 10) - 0.5) < 1e-12 and mo.envelope(b, 2, 10) == 1)
d = {"type": "drift", "path": "position", "rate": 10, "angle": 90, "t_start": 1, "t_end": 2}
o = mo.offsets([d], 5, 10)
check("drift: holds after its end, angle 90 = down", abs(o["x"]) < 1e-9 and abs(o["y"] - 10) < 1e-9 and mo.offsets([d], 0.5, 10) == {})
c = mo.offsets([{"type": "oscillate", "path": "position", "amount": 40, "period": 4, "wave": "circle"}], 1, 10)
check("orbit: a circle", abs((c["x"] ** 2 + c["y"] ** 2) ** 0.5 - 40) < 1e-9)
two = mo.offsets([{"type": "drift", "path": "rotation", "rate": 30}, {"type": "drift", "path": "rotation", "rate": 30, "on": False},
                  {"type": "drift", "path": "rotation", "rate": -10}], 2, 10)
check("behaviours add up, off ones do not", abs(two["rotation"] - 40) < 1e-9)
r = mo.offsets([{"type": "random", "path": "opacity", "amount": 0.5, "every": 0.5, "seed": 3}], 0.1, 10)["opacity"]
check("random holds within a step", r == mo.offsets([{"type": "random", "path": "opacity", "amount": 0.5, "every": 0.5, "seed": 3}], 0.45, 10)["opacity"])
comp = mo.compose({"scale": 100, "x": 5}, {"x": 100, "y": 100, "w": 50, "h": 20}.get)
check("scale about the (moved) centre", comp == {"x": 80.0, "y": 90.0, "w": 100.0, "h": 40.0}, str(comp))
check("clamps: opacity, sizes, effects", mo.compose({"opacity": 5, "fx.blur": -9, "w": -500}, {"opacity": 0.5, "fx.blur": 2, "w": 10}.get)
      == {"opacity": 1.0, "fx.blur": 0.0, "w": 1.0})
keys = [{"t": 0, "v": 0, "e": "linear"}, {"t": 1, "v": 10, "e": "linear"}]
check("loop cycle / pingpong / continue", abs(mo.keyed(keys, 2.25, "cycle", rd.key_value) - 2.5) < 1e-9
      and abs(mo.keyed(keys, 1.25, "pingpong", rd.key_value) - 7.5) < 1e-9 and abs(mo.keyed(keys, 3, "continue", rd.key_value) - 30) < 1e-9
      and mo.keyed(keys, 3, None, rd.key_value) == 10)

# ---- pulse / audio
import math  # noqa: E402
bt = mo.beat_times({"bpm": 120, "phase": 0.1}, 0.3, 4)
check("beat times: 120 bpm, phase minus offset wrapped", abs(bt[0] - 0.3) < 1e-9 and abs(bt[1] - 0.8) < 1e-9 and len(bt) == 8, str(bt[:3]))
ctx = {"beats": bt, "markers": [1.0, 2.2], "level": None, "offset": 0}
pb = {"type": "pulse", "path": "scale", "amount": 10, "trigger": "beats", "attack": 0.02, "decay": 0.2}
check("pulse: 0 before the first beat, peak after attack, decays", mo.offsets([pb], 0.2, 4, ctx) == {}
      or abs(mo.offsets([pb], 0.2, 4, ctx).get("scale", 0)) < 1e-12)
check("pulse: full at beat + attack", abs(mo.offsets([pb], 0.32, 4, ctx)["scale"] - 10) < 1e-9
      and abs(mo.offsets([pb], 0.52, 4, ctx)["scale"] - 10 * math.exp(-1)) < 1e-9)
bars = dict(pb, trigger="bars")
check("pulse on bars: every 4th beat", abs(mo.offsets([bars], 0.82, 4, ctx)["scale"] - 10 * math.exp(-2.5)) < 1e-9)
sr = 8000
sig = [0.0] * sr + [0.5 * math.sin(i * 0.3) for i in range(sr)]          # 1 s silence, 1 s tone
lv = mo.level_curve(sig, sr)
check("level curve: 100 / s, silence 0, tone ~1", len(lv) == 200 and max(lv[:99]) == 0 and 0.95 < lv[150] <= 1.5, str(lv[150]))
actx = dict(ctx, level=lv)
ab = {"type": "audio", "path": "opacity", "amount": 1, "smooth": 0.2}
check("audio: follows the level, releases after", mo.offsets([ab], 0.5, 4, actx)["opacity"] == 0
      and 0.95 < mo.offsets([ab], 1.5, 4, actx)["opacity"] and 0.2 < mo.offsets([ab], 2.1, 4, actx)["opacity"] < 0.9)

# ---- stagger over regions
REGS = [(i + 1, [(i % 6) * 300 + 40, (i // 6) * 400 + 60, 200 + (i % 3) * 20, 300]) for i in range(18)] + [(40, [1500, 900, 200, 300])]
seq = {"type": "stagger", "mode": "sequence", "order": "left", "step": 0.1, "fade": 0}
w0 = mo.stagger_weights([seq], 0.05, 5, REGS, 3200, 2160)
check("stagger sequence: one region per step, leftmost (top first) on first", sorted(k for k, v in w0.items() if v == 1) == [1], str(w0))
check("stagger: all on at the end, off before the start", all(v == 1 for v in mo.stagger_weights([seq], 4, 5, REGS, 3200, 2160).values())
      and all(v == 0 for v in mo.stagger_weights([dict(seq, t_start=1)], 0.5, 5, REGS, 3200, 2160).values()))
check("stagger out: the reverse", all(v == 0 for v in mo.stagger_weights([dict(seq, direction="out")], 4, 5, REGS, 3200, 2160).values()))
ch = mo.stagger_weights([{"type": "stagger", "mode": "chase", "order": "left", "step": 0.1, "fade": 0, "hold": 0.1}], 0.35, 5, REGS, 3200, 2160)
check("stagger chase: one step lit at a time", sum(1 for v in ch.values() if v == 1) in (1, 2, 3), str(ch))
check("stagger: no stagger behaviour -> None", mo.stagger_weights([{"type": "wiggle"}], 1, 5, REGS, 3200, 2160) is None)
# (swap masks: tests/test_soundswap.py; the window twin below runs its cases too)

# ---- through animate (the node's per frame document)
doc = {"timeline": {"duration": 4, "fps": 25}, "layers": [
    {"id": "a", "kind": "shape", "x": 100, "y": 100, "w": 50, "h": 50, "rotation": 0, "opacity": 1,
     "anim": {"x": keys}, "motion": [{"type": "loop", "path": "position", "mode": "cycle"},
                                     {"type": "oscillate", "path": "scale", "amount": 100, "period": 4, "phase": 0.25}]}]}
L = rd.animate(doc, 4.0)["layers"][0]            # the keys wrap to 0, the scale peaks (+100 %)
check("animate: looped key + scale behaviour", abs(L["w"] - 100) < 1e-9 and abs(L["x"] - (0 - 25)) < 1e-9, str(L))
rdoc = {"timeline": {"duration": 4, "fps": 25}, "layers": [
    {"id": "r", "kind": "shape", "x": 100, "y": 100, "w": 50, "h": 50, "rotation": 0, "opacity": 1, "anim": {"x": keys},
     "motion": [{"type": "repeat", "count": 4, "layout": "line", "dx": 100, "dy": 0, "opacity": -0.2, "delay": 0.5}]},
    {"id": "base", "kind": "base"}]}
R = rd.animate(rdoc, 1.0)["layers"]
check("repeat: copies after the layer, ids, opacity steps", [q["id"] for q in R] == ["r", "r~1", "r~2", "r~3", "base"]
      and abs(R[3]["opacity"] - 0.4) < 1e-9, str([q["id"] for q in R]))
check("repeat: copy k shows the layer k * delay earlier, shifted by k * dx", abs(R[1]["x"] - (5 + 100)) < 1e-9 and abs(R[2]["x"] - (0 + 200)) < 1e-9,
      str([q["x"] for q in R[:4]]))
check("repeat: copies carry no repeat of their own, render parses them", not any(b.get("type") == "repeat" for b in R[1]["motion"])
      and len(rd.parse(rd.animate(rdoc, 1.0))["layers"]) == 5)
ring = [mo.repeat_offsets({"layout": "radial", "count": 4, "radius": 100, "orient": True}, k) for k in range(4)]
check("repeat ring: a quarter turn per copy", abs(ring[1][0] + 100) < 1e-9 and abs(ring[1][1] - 100) < 1e-9 and abs(ring[2][0] + 200) < 1e-9 and ring[1][2] == 90)
check("animate leaves the document alone", doc["layers"][0]["x"] == 100 and doc["layers"][0]["w"] == 50)

# ---- the same cases in the window's code
CASES = []
for t in (0, 0.3, 1.0, 1.7, 2.49, 3.9, 7.3):
    for bb in ([{"type": "wiggle", "path": "position", "amount": 20, "freq": 1.3, "seed": 4}],
               [{"type": "wiggle", "path": "rotation", "amount": 7, "freq": 0.4, "seed": 11, "t_start": 0.5, "t_end": 3, "fade": 0.4}],
               [{"type": "oscillate", "path": "opacity", "amount": 0.4, "period": 0.7, "phase": 0.1, "wave": w} for w in ("sine", "triangle", "square", "saw")],
               [{"type": "oscillate", "path": "position", "amount": 30, "period": 3, "wave": "triangle", "angle": 33}],
               [{"type": "oscillate", "path": "position", "amount": 30, "period": 3, "wave": "circle"}],
               [{"type": "drift", "path": "rotation", "rate": 45, "t_start": 1, "t_end": 2.5}],
               [{"type": "random", "path": "light.lights.#p1.power", "amount": 300, "every": 0.12, "seed": 9}],
               [{"type": "random", "path": "position", "amount": 15, "every": 0.2, "seed": 2}, {"type": "oscillate", "path": "scale", "amount": 20, "period": 1.5}]):
        CASES.append({"t": t, "motion": bb})
PCASES = []
for t in (0, 0.31, 0.55, 1.02, 1.5, 2.24, 3.3):
    for bb in ([{"type": "pulse", "path": "scale", "amount": 8, "trigger": tr, "nth": n, "attack": 0.03, "decay": 0.2} for tr in ("beats", "bars", "markers") for n in (1, 2)],
               [{"type": "pulse", "path": "position", "amount": 30, "angle": -90, "trigger": "every", "every": 0.37, "t_start": 0.2}],
               [{"type": "audio", "path": "opacity", "amount": 1, "smooth": sm} for sm in (0, 0.125, 0.15, 0.6)],
               [{"type": "audio", "path": "position", "amount": 20, "angle": 45, "smooth": 0.1, "t_start": 1, "fade": 0.3}]):
        PCASES.append({"t": t, "motion": bb})
SCASES = [{"t": t, "motion": [{"type": "stagger", "mode": m, "order": o, "step": 0.07, "fade": f, "hold": 0.2, "period": 1.3,
                               "density": 0.4, "seed": 5, "direction": d, "t_start": 0.2}]}
          for t in (0, 0.33, 0.9, 2.5) for m in ("sequence", "chase", "wave", "random") for o in ("left", "right", "top", "bottom", "centre", "outside", "size", "random")
          for f, d in ((0.15, "in"), (0, "out"))]
SCASES.append({"t": 0.7, "motion": [{"type": "stagger", "mode": "wave", "order": "top", "step": 0.1}, {"type": "stagger", "mode": "sequence", "order": "right", "step": 0.05}]})
RCASES = [({"type": "repeat", "count": 7, "layout": lay, "dx": 120, "dy": -40, "cols": 3, "radius": 300, "start": st, "orient": o,
            "rotate": 7, "scale": sc, "opacity": -0.1}, k) for lay in ("line", "grid", "radial") for st in (0, 33) for o in (True, False)
          for sc in (0, -12) for k in range(1, 7)]
LOOPS = [{"keys": keys, "t": t, "mode": m} for t in (0.5, 1.3, 2.7, 5.01) for m in ("cycle", "pingpong", "continue", None)]
# swap masks: steps on beats / bars / markers / a fixed time / hits per band, every order, with and without a fade
HITS = {"low": [[0.0, 1.0], [0.5, 0.9], [0.62, 0.2], [1.0, 1.0], [1.04, 0.8], [1.5, 0.95], [2.31, 0.4], [3.0, 1.0]],
        "mid": [[0.25, 0.5], [1.25, 0.31], [2.75, 0.29]], "high": [[0.25, 1.0], [0.3, 0.6], [0.75, 0.9], [1.25, 0.2], [1.75, 1.0], [2.2, 0.7]]}
SCTX = dict(actx, hits=HITS, bands={"low": lv, "high": [v * 0.5 for v in lv]}, offset=0.1)
SWCASES = [{"t": t, "n": n, "b": {"type": "swap", "order": o, "trigger": tr, "nth": nth, "every": 0.37, "threshold": th, "gap": gp,
                                  "fade": fd, "seed": 7, "t_start": 0.2, "t_end": te, "transition": "dip" if fd and n == 3 else "cross"}}
           for t in (0, 0.26, 0.55, 1.02, 1.31, 2.0, 3.3, 5.5) for n in (2, 3, 5) for o in ("loop", "pingpong", "random")
           for tr, nth, th, gp in (("beats", 1, 0.3, 0.1), ("bars", 1, 0.3, 0.1), ("beats", 2, 0.3, 0.1), ("markers", 1, 0.3, 0.1),
                                   ("every", 1, 0.3, 0.1), ("low", 1, 0.3, 0.1), ("low", 2, 0.0, 0.0), ("mid", 1, 0.3, 0.1), ("high", 1, 0.65, 0.3))
           for fd, te in ((0, -1), (0.2, 4.0))]
SWLAYERS = [{"id": "a", "kind": "image", "clip": "W_F1_*", "motion": [{"type": "wiggle"}, {"type": "swap", "order": "random", "trigger": "every", "every": 0.3, "fade": 0.1, "seed": 3, "with": ["d", "b", "zz", "c", "a"]}]},
            {"id": "b", "kind": "shape", "clip": "M_Door"}, {"id": "c", "kind": "image", "clip": " "},
            {"id": "d", "kind": "adjust", "clip": "group:Windows", "motion": [{"type": "swap", "trigger": "every", "with": ["e"]}]},
            {"id": "e", "kind": "image", "clip": "M_Attic", "motion": [{"type": "swap", "order": "pingpong", "trigger": "low", "threshold": 0.5, "with": ["f", "a"]}]},
            {"id": "f", "kind": "image", "clip": "M_FLOOR_F0"}, {"id": "base", "kind": "base", "clip": "x"}]
SWTIMES = (0, 0.31, 0.65, 0.95, 1.5, 2.48, 4.9)
BCASES = [{"t": t, "motion": [{"type": "audio", "path": "opacity", "amount": 1, "smooth": sm, "band": bd}]}
          for t in (0.4, 1.02, 1.5, 2.24) for sm in (0, 0.15) for bd in ("all", "low", "mid", "high", "nothing")]
BASE = {"x": 100, "y": 60, "w": 50, "h": 30, "rotation": 3, "opacity": 0.8, "light.lights.#p1.power": 500}


# ---- field: the clip along a grey ramp (motion.field_*, kubakub/maskfields.py field / along)
FB = {"type": "field", "field": "direction", "effect": "reveal", "cycle": 1.3, "from": 0.1, "to": 0.9}
FCASES = [{"t": t, "b": dict(FB, curve=cv, easing=es, trail=tr, steps=st, nth=nth, t_start=ts, t_end=te, phase=ph, seed=4, duty=0.3,
                             threshold=0.3, gap=0.1, band=bd, amount=1.4, smooth=0.1)}
          for t in (0, 0.2, 0.51, 1.02, 1.9, 2.6, 4.4, 5.9) for cv in mo.FIELD_CURVES
          for es, tr, st, nth, ts, te, ph, bd in (("linear", 0, 1, 1, 0, -1, 0, "all"), ("ease in out", 0.3, 3, 1, 0.4, 5.0, 0.25, "low"),
                                               ("ease out", 1.5, 1, 2, 0, -1, 0, "high"))]
FMAP_W, FMAP_H = 60, 40
FMAP = [[-1] * FMAP_W for _ in range(FMAP_H)]
for y_ in range(FMAP_H):
    for x_ in range(FMAP_W):
        if 3 <= x_ < 20 and 4 <= y_ < 30 and not (9 <= x_ < 20 and 12 <= y_ < 30):
            FMAP[y_][x_] = 5                             # an L
        elif (x_ - 38) ** 2 + (y_ - 14) ** 2 <= 81:
            FMAP[y_][x_] = 2                             # a disc
        elif 24 <= x_ < 57 and 31 <= y_ < 37:
            FMAP[y_][x_] = 9                             # a bar
        elif 52 <= x_ < 58 and 3 <= y_ < 9:
            FMAP[y_][x_] = 4                             # a small square
FIDS = [5, 2, 9, 4]
FRAMPS = [dict(field=k, per=pr, angle=an, cx=0.3, cy=0.7, order=od, seed=sd, noise_px=9.0, invert=inv)
          for k in mo.FIELD_KINDS for pr in mo.FIELD_PER
          for an, od, sd, inv in ((0, "left", 1, False), (37, "centre", 3, True), (90, "random", 7, False), (200, "size", 2, False))]
FALONG = [(f / 20, p / 10, ef, sf, wd, rg) for f in range(21) for p in range(-1, 12) for ef in mo.FIELD_EFFECTS
          for sf, wd, rg in ((0.05, 0.2, 3), (0.01, 0.5, 1), (0.4, 0.05, 6))]


def py_fields():
    import numpy as np
    from kubakub import maskfields as mf
    lab, n = mf.compact(np.array(FMAP, np.int32), FIDS)
    ramps = [mf.field(lab, n, c["field"], c["per"], c["angle"], (c["cx"], c["cy"]), c["order"], c["seed"], c["noise_px"], c["invert"],
                      px_scale=2.5, ids=FIDS)[0].ravel().tolist() for c in FRAMPS]
    along = [float(mf.along(np.array([[f]], np.float32), p, ef, sf, wd, rg)[0, 0]) for f, p, ef, sf, wd, rg in FALONG]
    return ramps, along


def py_side():
    out = {"offsets": [], "compose": [], "loops": [], "hash": [mo.hash01(i, s) for i in (0, 1, 7, 99999, 2 ** 31 + 5) for s in (0, 1, 123456)]}
    for c_ in CASES:
        of = mo.offsets(c_["motion"], c_["t"], 5.0)
        out["offsets"].append(of)
        out["compose"].append(mo.compose(of, BASE.get))
    for lp in LOOPS:
        out["loops"].append(mo.keyed(lp["keys"], lp["t"], lp["mode"], rd.key_value))
    out["pulse"] = [mo.offsets(c_["motion"], c_["t"], 4.0, actx) for c_ in PCASES]
    out["beats"] = mo.beat_times({"bpm": 97.3, "phase": 0.41}, 1.7, 30)
    out["level"] = mo.level_curve(sig, sr)
    out["repeat"] = [mo.repeat_apply(dict(BASE), b, k) for b, k in RCASES]
    out["stagger"] = [{str(k): v for k, v in (mo.stagger_weights(c_["motion"], c_["t"], 5.0, REGS, 3200, 2160) or {}).items()} for c_ in SCASES]
    sctx = dict(SCTX)                                  # one context for all: the step times and shuffles are kept in it
    out["swap"] = json.loads(json.dumps([mo.swap_mix(c_["b"], c_["t"], 6.0, c_["n"], sctx) for c_ in SWCASES]))
    out["swaptimes"] = [mo.swap_times(c_["b"], sctx, 6.0) for c_ in SWCASES[:54]]
    out["swapclips"] = [mo.swap_clips(SWLAYERS, t, 6.0, sctx) for t in SWTIMES]
    out["band"] = [mo.offsets(c_["motion"], c_["t"], 4.0, sctx) for c_ in BCASES]
    out["field"] = [mo.field_values(c_["b"], c_["t"], 6.0, sctx) for c_ in FCASES]
    from kubakub import maskfields as mf
    out["pruned"] = [[list(v) for v in mf.pruned(vals, ef)] for vals in out["field"] for ef in mo.FIELD_EFFECTS]
    out["ramps"], out["along"] = py_fields()
    return out


PY = py_side()
import tempfile  # noqa: E402
VEC = os.path.join(tempfile.gettempdir(), "kuba_motion_vectors.json")        # the cases for the window twin
json.dump({"cases": CASES, "loops": LOOPS, "base": BASE, "pcases": PCASES, "scases": SCASES, "rcases": RCASES, "regs": REGS, "ctx": actx, "sig": sig, "sr": sr, "expect": PY,
           "sctx": SCTX, "swcases": SWCASES, "swlayers": SWLAYERS, "swtimes": SWTIMES, "bcases": BCASES,
           "fcases": FCASES, "fmap": FMAP, "framps": FRAMPS, "falong": FALONG}, open(VEC, "w"))

node = shutil.which("node")
if not node:
    print("skip the window twin: node.js not found")
else:
    js = open(os.path.join(HERE, "..", "web", "kubakub_director.js"), encoding="utf-8").read()
    a, b_ = js.index("const isHex = "), js.index("// ---- audio: beats from a mono signal")
    engine = js[a:b_] + re.search(r"function beatTimes[\s\S]*?\n}\n", js).group(0)
    runner = engine + """
const V = JSON.parse(require("fs").readFileSync(process.argv[2], "utf8"));
const out = { offsets: [], compose: [], loops: [], hash: [] };
for (const i of [0, 1, 7, 99999, 2 ** 31 + 5]) for (const s of [0, 1, 123456]) out.hash.push(hash01(i, s));
for (const c of V.cases) { const of = motionOffsets(c.motion, c.t, 5.0); out.offsets.push(of); out.compose.push(composeMotion(of, p => V.base[p])); }
for (const l of V.loops) out.loops.push(keyedValue(l.keys, l.t, l.mode || undefined));
out.pulse = V.pcases.map(c => motionOffsets(c.motion, c.t, 4.0, V.ctx));
out.beats = beatTimes({ bpm: 97.3, phase: 0.41 }, 1.7, 30);
out.level = levelCurve(Float64Array.from(V.sig), V.sr);
out.repeat = V.rcases.map(([b, k]) => repeatApply({ ...V.base }, b, k));
out.stagger = V.scases.map(c => staggerWeights(c.motion, c.t, 5.0, V.regs, 3200, 2160) || {});
out.swap = V.swcases.map(c => swapMix(c.b, c.t, 6.0, c.n, V.sctx));
out.swaptimes = V.swcases.slice(0, 54).map(c => swapTimes(c.b, V.sctx, 6.0));
out.swapclips = V.swtimes.map(t => swapClips(V.swlayers, t, 6.0, V.sctx));
out.band = V.bcases.map(c => motionOffsets(c.motion, c.t, 4.0, V.sctx));
out.field = V.fcases.map(c => fieldValues(c.b, c.t, 6.0, V.sctx));
out.pruned = out.field.flatMap(vals => FIELD_EFFECTS.map(ef => fieldPruned(vals, ef)));
const LW = V.fmap[0].length, LH = V.fmap.length, byId = new Map();
V.fmap.forEach((row, y) => row.forEach((id, x) => { if (id < 0) return; if (!byId.has(id)) byId.set(id, { id, pos: [], box: [LW, LH, 0, 0] });
  const r = byId.get(id); r.pos.push(y * LW + x); r.box[0] = Math.min(r.box[0], x); r.box[1] = Math.min(r.box[1], y); r.box[2] = Math.max(r.box[2], x + 1); r.box[3] = Math.max(r.box[3], y + 1); }));
const rp = [...byId.values()].map(r => ({ ...r, pos: Uint32Array.from(r.pos) }));
out.ramps = V.framps.map(c => Array.from(fieldRamp(rp, LW, LH, c, 2.5)));
out.along = V.falong.map(([f, p, ef, sf, wd, rg]) => fieldAlong(f, p, ef, Math.max(1e-6, sf), Math.max(1e-6, wd), Math.max(1, rg)));
console.log(JSON.stringify(out));
"""
    tmp = os.path.join(HERE, "_motion_twin.cjs")
    open(tmp, "w", encoding="utf-8").write(runner)
    try:
        res = subprocess.run([node, tmp, VEC], capture_output=True, text=True, timeout=60)
    finally:
        os.remove(tmp)
    check("window twin runs", res.returncode == 0, res.stderr[-400:])
    if res.returncode == 0:
        JS = json.loads(res.stdout)

        def close(x, y):
            if isinstance(x, dict):
                return isinstance(y, dict) and set(x) == set(y) and all(close(x[k], y[k]) for k in x)
            if isinstance(x, list):
                return isinstance(y, list) and len(x) == len(y) and all(close(p, q) for p, q in zip(x, y))
            if isinstance(x, (int, float)) and isinstance(y, (int, float)):
                return abs(x - y) < 1e-9
            return x == y
        check("window twin: hash bit-identical", JS["hash"] == PY["hash"])
        bad = [i for i, (p, q) in enumerate(zip(PY["offsets"], JS["offsets"])) if not close(p, q)]
        check(f"window twin: {len(CASES)} behaviour cases agree", not bad, f"first bad {bad[:1]}: {PY['offsets'][bad[0]] if bad else ''} vs {JS['offsets'][bad[0]] if bad else ''}")
        check("window twin: compose (position / scale / clamps) agrees", close(PY["compose"], JS["compose"]))
        badp = [i for i, (p, q) in enumerate(zip(PY["pulse"], JS["pulse"])) if not close(p, q)]
        check(f"window twin: {len(PCASES)} pulse / audio cases agree", not badp, f"first bad {badp[:1]}: {PY['pulse'][badp[0]] if badp else ''} vs {JS['pulse'][badp[0]] if badp else ''}")
        bads = [i for i, (p, q) in enumerate(zip(PY["stagger"], JS["stagger"])) if not close(p, q)]
        check(f"window twin: {len(SCASES)} stagger cases agree", not bads, f"first bad {bads[:1]}: {SCASES[bads[0]] if bads else ''}")
        check(f"window twin: {len(RCASES)} repeat placements agree", close(PY["repeat"], JS["repeat"]))
        badw = [i for i, (p, q) in enumerate(zip(PY["swap"], JS["swap"])) if not close(p, q)]
        check(f"window twin: {len(SWCASES)} swap cases agree", not badw and len(JS["swap"]) == len(SWCASES),
              f"first bad {badw[:1]}: {SWCASES[badw[0]] if badw else ''} {PY['swap'][badw[0]] if badw else ''} vs {JS['swap'][badw[0]] if badw else ''}")
        check("window twin: swap step times agree", close(PY["swaptimes"], JS["swaptimes"]))
        check("window twin: swap groups of a document agree", close(PY["swapclips"], JS["swapclips"]), f"{PY['swapclips'][2]} vs {JS['swapclips'][2]}")
        check(f"window twin: {len(BCASES)} sound level per band cases agree", close(PY["band"], JS["band"]))
        badf = [i for i, (p, q) in enumerate(zip(PY["field"], JS["field"])) if not close(p, q)]
        check(f"window twin: {len(FCASES)} field cases agree (curve, triggers, sound level, trail)", not badf and len(JS["field"]) == len(FCASES),
              f"first bad {badf[:1]}: {FCASES[badf[0]] if badf else ''} {PY['field'][badf[0]] if badf else ''} vs {JS['field'][badf[0]] if badf else ''}")
        check("window twin: the trail is thinned the same way (what cannot win the maximum is left out)",
              close(PY["pruned"], JS["pruned"]) and any(len(v) == 1 for v in PY["pruned"]) and any(len(v) > 3 for v in PY["pruned"]))
        worst = [max(abs(a - b) for a, b in zip(p, q)) for p, q in zip(PY["ramps"], JS["ramps"])]
        check(f"window twin: {len(FRAMPS)} field ramps agree per pixel (edge distance, direction, radial, order, noise)",
              len(worst) == len(FRAMPS) and max(worst) < 2e-4, f"worst {max(worst):.5f} in {FRAMPS[worst.index(max(worst))]}")
        check(f"window twin: {len(FALONG)} reveal / hide / band / rings values agree",
              len(JS["along"]) == len(FALONG) and max(abs(a - b) for a, b in zip(PY["along"], JS["along"])) < 1e-4)
        check("window twin: beat times agree", close(PY["beats"], JS["beats"]))
        check("window twin: loudness curve agrees", close(PY["level"], JS["level"]))
        check("window twin: looped keyframes agree", close(PY["loops"], JS["loops"]), f"{PY['loops']} vs {JS['loops']}")

# ---- field through the renderer: animate writes clip_field, the clip follows the move
import numpy as np  # noqa: E402
fl = np.full((40, 60), -1, np.int32)
fl[10:30, 5:25] = 0
fl[10:30, 35:55] = 1
ftab = {"regions": [{"region_id": 0, "name": "a_01", "group_id": "a", "bbox": [5, 10, 20, 20]}, {"region_id": 1, "name": "a_02", "group_id": "a", "bbox": [35, 10, 20, 20]}]}
fdoc = {"canvas": {"w": 60, "h": 40}, "timeline": {"duration": 4, "fps": 10},
        "layers": [{"id": "s", "kind": "shape", "shape": {"type": "solid"}, "color": "#ffffff", "x": 0, "y": 0, "w": 60, "h": 40, "clip": "a_*",
                    "motion": [{"type": "field", "field": "direction", "per": "all together", "effect": "reveal", "curve": "ramp", "cycle": 2, "soft": 0.01}]},
                   {"id": "base", "kind": "base"}]}


def fframe(t):
    d = rd.animate(fdoc, t)
    return d, rd.render(d, np.zeros((40, 60, 3), np.float32), {}, fl, ftab)


try:
    d0, r0 = fframe(0.0)
    d1, r1 = fframe(1.0)
    d2, r2 = fframe(2.0)
    img = lambda r: r["image"][..., 0]      # noqa: E731
    check("animate writes the values of the field for the clip", d1["layers"][0].get("clip_field", {}).get("values") == [[0.5, 1.0]])
    check("renderer: a wipe across both regions: nothing, the left half, everything",
          img(r0).max() < 0.01 and img(r1)[20, 15] > 0.99 and img(r1)[20, 45] < 0.01 and img(r2)[20, 15] > 0.99 and img(r2)[20, 45] > 0.99
          and img(r2)[5, 15] < 0.01, f"{img(r0).max()} {img(r1)[20, 15]} {img(r1)[20, 45]} {img(r2)[20, 45]}")
except Exception as e:  # noqa: BLE001
    check("renderer: a field behaviour renders", False, repr(e))

bctx = {"beats": [k * 0.5 for k in range(12)], "markers": [], "offset": 0.0, "hits": {}}
fbeat = {"type": "field", "curve": "beats", "cycle": 0.4, "from": 0, "to": 1}
check("field on beats: the beat on the start of the range counts, every nth from the first",
      abs(mo.field_progress(fbeat, 0.2, 6.0, bctx) - 0.5) < 1e-9 and mo.field_progress(dict(fbeat, nth=4), 0.2, 6.0, bctx) == 0.5
      and mo.field_progress(dict(fbeat, nth=4), 0.7, 6.0, bctx) == 1.0 and abs(mo.field_progress(dict(fbeat, nth=4), 2.1, 6.0, bctx) - 0.25) < 1e-9
      and mo.field_progress(dict(fbeat, t_start=0.75), 0.9, 6.0, bctx) == 0.0)

# ---- the thinned trail gives exactly the same mask
import itertools  # noqa: E402
from kubakub import maskfields as mf  # noqa: E402
ramp = np.linspace(0, 1, 97, dtype=np.float32)[None]
same = True
for vals, ef in itertools.product([c_ for c_ in PY["field"] if len(c_) > 1][::7], mo.FIELD_EFFECTS):
    full = None
    for p_, w_ in vals:
        m_ = mf.along(ramp, p_, ef, 0.05, 0.2, 3) * w_
        full = m_ if full is None else np.maximum(full, m_)
    same = same and float(np.abs(full - mf.mask_at(ramp, vals, ef, 0.05, 0.2, 3)).max()) < 1e-6
check("a thinned trail gives the same mask as every one of its moments", same)

print()
if failures:
    print(f"{len(failures)} FAILED: {', '.join(failures)}")
    sys.exit(1)
print("all motion tests passed")
