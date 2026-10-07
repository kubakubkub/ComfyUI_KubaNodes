"""
Model free test for the sound side of the swap: what is heard in a sound (kubakub/sound.py: tempo, bands, hits),
the swap engine (kubakub/director/motion.py swap_*) and kubakub sound mask swap (kubakub/soundswap.py).

Run from the ComfyUI_windows_portable folder:
    python_embeded\\python.exe ComfyUI\\custom_nodes\\ComfyUI_KubaNodes\\tests\\test_soundswap.py
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from kubakub import sound as so  # noqa: E402
from kubakub import soundswap as sw  # noqa: E402
from kubakub.director import motion as mo  # noqa: E402

failures = []


def check(name, cond, detail=""):
    print(("ok   " if cond else "FAIL ") + name + (f"  ({detail})" if detail and not cond else ""))
    if not cond:
        failures.append(name)


def near(found, expect, tol=0.021):
    return len(found) == len(expect) and all(abs(a - b) <= tol for a, b in zip(found, expect))


# ---- the sound: tempo, bands, hits
SR = 44100
x = so.sample_sound(8.0, 120.0, SR)
beat = so.analyze_beats(x, SR)
check("tempo of the built-in beat: 120 bpm, a beat on 0", beat and abs(beat["bpm"] - 120) < 0.6
      and min(beat["phase"] % 0.5, 0.5 - beat["phase"] % 0.5) < 0.02, str(beat))
check("tempo: less than 2 s -> None", so.analyze_beats(x[:SR], SR) is None)
check("phase for a tempo you set", abs(so.beat_phase(np.concatenate([np.zeros(int(0.2 * SR), np.float32), x]), SR, 120) - 0.2) < 0.02)
an = so.analyze(x, SR)
low = [h[0] for h in an["hits"]["low"] if h[1] >= 0.3]
high = [h[0] for h in an["hits"]["high"] if h[1] >= 0.3]
check("low hits = the kicks (every beat, the one at 0 too)", near(low, [k * 0.5 for k in range(16)]), str(low[:6]))
check("high hits = the hats (the off-beats)", near(high, [k * 0.5 + 0.25 for k in range(16)]), str(high[:6]))
check("curves: 100 / s per band, 0..1.5", all(len(c) == 800 and 0 <= min(c) and max(c) <= 1.5 for c in an["curves"].values()))
check("low curve up on a kick, down between", an["curves"]["low"][52] > 0.5 > an["curves"]["low"][45], str(an["curves"]["low"][40:56]))
quiet = so.analyze(np.zeros(SR, np.float32), SR)
check("silence: no hits, flat curves", all(not h for h in quiet["hits"].values()) and max(quiet["curves"]["low"]) == 0)
x8 = so.analyze(so.sample_sound(4.0, 120.0, 8000), 8000)
check("8 kHz sound: the high band is above it, low still found", x8["hits"]["high"] == [] and len(x8["hits"]["low"]) >= 7)
check("mono from a core AUDIO waveform", so.to_mono(np.ones((1, 2, 10), np.float32)).shape == (10,))

# ---- the swap engine
ctx = {"beats": [k * 0.5 for k in range(16)], "markers": [1.1, 2.2], "offset": 0.0, "hits": an["hits"]}
b = {"type": "swap", "order": "loop", "trigger": "beats"}
check("before the first step every layer has its own mask (a beat on the start does not count)",
      mo.swap_mix(b, 0.0, 8, 3, ctx) == [[(0, 1.0)], [(1, 1.0)], [(2, 1.0)]])
check("loop: one mask on per beat, back after n", mo.swap_mix(b, 0.5, 8, 3, ctx) == [[(1, 1.0)], [(2, 1.0)], [(0, 1.0)]]
      and mo.swap_mix(b, 1.6, 8, 3, ctx) == mo.swap_mix(b, 0.2, 8, 3, ctx))
check("every step is a permutation", all(sorted(m for (m, _), in mo.swap_mix(dict(b, order=o), t, 8, 5, ctx)) == [0, 1, 2, 3, 4]
                                         for o in mo.SWAP_ORDERS for t in (0.3, 1.2, 2.9, 7.7)))
pp = [mo.swap_perm(4, k, "pingpong")[0] for k in range(8)]
check("pingpong: there and back", pp == [0, 1, 2, 3, 2, 1, 0, 1], str(pp))
rnd = [mo.swap_perm(n, k, "random", 3, {}) for n in (2, 3, 6) for k in range(40)]
check("random: never the same arrangement twice in a row", all(a != c for a, c in zip(rnd, rnd[1:]) if len(a) == len(c)))
check("random: the same with and without the memo, another seed another shuffle",
      mo.swap_perm(5, 17, "random", 3, {}) == mo.swap_perm(5, 17, "random", 3) != mo.swap_perm(5, 17, "random", 4))
check("bars: every 4th beat; nth: every other one", mo.swap_times(dict(b, trigger="bars"), ctx, 8) == [2.0, 4.0, 6.0]
      and mo.swap_times(dict(b, nth=2), ctx, 8)[:3] == [1.0, 2.0, 3.0])
check("every: a fixed time from the start of the range", mo.swap_times({"trigger": "every", "every": 1.5, "t_start": 1}, ctx, 8) == [2.5, 4.0, 5.5, 7.0])
check("markers", mo.swap_times({"trigger": "markers"}, ctx, 8) == [1.1, 2.2])
check("low: the kicks after the start", near(mo.swap_times({"trigger": "low"}, ctx, 8), [k * 0.5 for k in range(1, 16)]))
check("threshold above every hit: nothing; gap thins them out", mo.swap_times({"trigger": "low", "threshold": 1.01}, ctx, 8) == []
      and near(mo.swap_times({"trigger": "low", "gap": 0.9, "t_start": 0.25}, ctx, 8), [0.5 + k for k in range(8)]))
check("the sound's offset moves the hits", near(mo.swap_times({"trigger": "low"}, dict(ctx, offset=0.2), 2)[:2], [0.3, 0.8]))
check("the range: nothing before, the last state holds after",
      mo.swap_state(dict(b, t_start=2, t_end=3), 1.9, 8, ctx)[0] == 0 and mo.swap_state(dict(b, t_start=2, t_end=3), 7, 8, ctx)[0] == 2)
f = mo.swap_mix(dict(b, fade=0.2), 0.6, 8, 2, ctx)
check("fade: the upper layer fades out of its old mask and into the new one, the lower layer is whole underneath",
      [m for m, _ in f[0]] == [0, 1] and abs(f[0][0][1] - 0.5) < 1e-9 and abs(f[0][1][1] - 0.5) < 1e-9 and f[1] == [(1, 1.0), (0, 1.0)]
      and mo.swap_mix(dict(b, fade=0.2), 0.75, 8, 2, ctx) == [[(1, 1.0)], [(0, 1.0)]], str(f))
dp = [mo.swap_mix(dict(b, fade=0.4, transition="dip"), t, 8, 2, ctx)[1] for t in (0.6, 0.7, 0.8)]
check("fade style dip: out of the old mask first, then into the new one", abs(dp[0][0][1] - 0.5) < 1e-9 and dp[0][1][1] == 0
      and dp[1] == [(1, 0.0), (0, 0.0)] and dp[2][0][1] == 0 and abs(dp[2][1][1] - 0.5) < 1e-9, str(dp))
L = [{"id": "a", "kind": "image", "clip": "W_*", "motion": [dict(b, **{"with": ["b", "c", "gone", "d"]})]},
     {"id": "b", "kind": "image", "clip": "M_Door"}, {"id": "c", "kind": "shape", "clip": ""},
     {"id": "d", "kind": "shape", "clip": "M_Attic", "motion": [dict(b, **{"with": ["a"]})]}, {"id": "base", "kind": "base"}]
sc = mo.swap_clips(L, 0.7, 8, ctx)
check("layers: the owner and the layers it names trade clips; no clip / missing = not in; one group per layer",
      sc == {"a": [["M_Door", 1.0]], "b": [["M_Attic", 1.0]], "d": [["W_*", 1.0]]}, str(sc))
check("layers: a swap that is off, or alone, changes nothing",
      mo.swap_clips([dict(L[0], motion=[dict(b, on=False, **{"with": ["b"]})]), L[1]], 0.7, 8, ctx) == {}
      and mo.swap_clips([dict(L[0], motion=[b]), L[1]], 0.7, 8, ctx) == {})
ac = dict(ctx, bands=an["curves"], level=[0.0] * 800)
check("sound level per band: low follows the kick, high does not", mo.offsets([{"type": "audio", "path": "opacity", "amount": 1, "smooth": 0, "band": "low"}], 0.52, 8, ac)["opacity"] > 0.5
      > mo.offsets([{"type": "audio", "path": "opacity", "amount": 1, "smooth": 0, "band": "high"}], 0.52, 8, ac)["opacity"])

# ---- kubakub sound mask swap
bg, masks = sw.sample_layers(320, 180)
check("sample: the facade and four window layers", bg.shape == (180, 320, 3) and masks.shape == (4, 180, 320) and all(m.max() == 1 for m in masks))
st = sw.Settings(order="loop", step_on="low", fps=20, seconds=2.0, scale=0.5)
frames, seq, lines = sw.run(bg, masks, None, x, SR, st, mask_of=1, sample=True)
check("frames and the mask of layer 1: 2 s at 20 fps, half size", frames.shape == (40, 90, 160, 3) and seq.shape == (40, 90, 160)
      and frames.dtype == np.float32 and 0 <= frames.min() and frames.max() <= 1, f"{frames.shape} {seq.shape}")
m0, m1 = np.clip(sw._fit(masks[0], 160, 90), 0, 1), np.clip(sw._fit(masks[1], 160, 90), 0, 1)
check("layer 1 holds its own mask until the first kick, then the next one", np.array_equal(seq[0], m0) and np.array_equal(seq[9], m0)
      and np.array_equal(seq[10], m1) and np.array_equal(seq[19], m1), str([int(np.array_equal(s, m0)) for s in seq[:22]]))
col = np.array(sw.COLOURS[0], np.float32) / 255
ys, xs = np.nonzero(m0 == 1)
check("no pictures: the colour of layer 1 sits in its mask", np.allclose(frames[0][ys[0], xs[0]], col, atol=1e-6))
pics = np.stack([np.full((180, 320, 3), v, np.float32) for v in (0.2, 0.9)])
fr2, _, l2 = sw.run(bg, masks, pics, x, SR, sw.Settings(step_on="beats", fps=20, seconds=1.0, scale=0.5), sample=True)
ys1, xs1 = np.nonzero(m1 == 1)
check("pictures: layer 1's picture moves from mask 1 to mask 2 on the beat (two pictures repeat over four layers)",
      abs(fr2[0][ys[0], xs[0]][0] - 0.2) < 1e-6 and abs(fr2[10][ys1[0], xs1[0]][0] - 0.2) < 1e-6 and abs(fr2[10][ys[0], xs[0]][0] - 0.9) < 1e-6)
check("report: tempo, steps, frames", any("120.0 bpm" in s for s in l2) and any("steps" in s for s in l2) and any("20 frames at 160x90" in s for s in l2), str(l2))
frf, sqf, _ = sw.run(bg, masks, None, x, SR, sw.Settings(step_on="beats", fade=0.2, fps=20, seconds=1.0, scale=0.5), sample=True)
check("fade: halfway layer 1 is seen at half in both masks", abs(sqf[12][ys[0], xs[0]] - 0.5) < 1e-6 and abs(sqf[12][ys1[0], xs1[0]] - 0.5) < 1e-6, str(sqf[12][ys[0], xs[0]]))
c0, c1 = np.array(sw.COLOURS[0], np.float32) / 255, np.array(sw.COLOURS[1], np.float32) / 255
check("fade: a clean crossfade of the two layers, no background in between", np.allclose(frf[12][ys1[0], xs1[0]], 0.5 * c0 + 0.5 * c1, atol=1e-5))
_, sq2, _ = sw.run(bg, masks, None, x, SR, sw.Settings(step_on="beats", fade=0.2, fps=20, seconds=1.0, scale=0.5), mask_of=2, sample=True)
check("fade: what is seen of the layer below is the other half", abs(sq2[12][ys1[0], xs1[0]] - 0.5) < 1e-6, str(sq2[12][ys1[0], xs1[0]]))
_, _, l3 = sw.run(bg, masks, None, x, SR, sw.Settings(step_on="high", threshold=1.01, fps=10, seconds=1.0, scale=0.25))
check("no step: the report says what to change", any("nothing moves" in s for s in l3), str(l3))
try:
    sw.run(bg, masks[:1], None, x, SR, st)
    check("one mask is refused", False)
except ValueError as e:
    check("one mask is refused", "at least two masks" in str(e))
cv = np.zeros(SR * 4, np.float32)                 # a gate track: on at 1.0-1.2, 2.5-2.6 (weak), 3.0-3.5 with a dropout
for a_, b_, v_ in ((1.0, 1.2, 1.0), (2.5, 2.6, 0.3), (3.0, 3.2, 0.9), (3.21, 3.5, 0.9)):
    cv[int(a_ * SR):int(b_ * SR)] = v_
check("signal: a step where the control track rises through the threshold", near(so.gate_times(cv, SR, 0.5), [1.0, 3.0, 3.21], 0.006)
      and near(so.gate_times(cv, SR, 0.2), [1.0, 2.5, 3.0, 3.21], 0.006) and near(so.gate_times(cv, SR, 0.5, gap=0.5), [1.0, 3.0], 0.006)
      and so.gate_times(-cv, SR, 0.5) == so.gate_times(cv, SR, 0.5) and so.gate_times(cv * 0, SR) == [], str(so.gate_times(cv, SR, 0.5)))
lfo = np.sin(2 * np.pi * 1.0 * np.arange(SR * 3) / SR - np.pi / 2).astype(np.float32) * 0.5 + 0.5
check("signal: an LFO steps once per cycle", near(so.gate_times(lfo, SR, 0.5), [0.25, 1.25, 2.25], 0.011), str(so.gate_times(lfo, SR, 0.5)))
check("list: times from a text", so.parse_times("0.5, 1\n1.75;3  x -2 1") == [0.5, 1.0, 1.75, 3.0] and so.parse_times("[0.25, 2]") == [0.25, 2.0] and so.parse_times("") == [])
_, sqs, ls = sw.run(bg, masks, None, x, SR, sw.Settings(step_on="list", fps=20, seconds=2.0, scale=0.5), sample=True, times=[0.5, 1.75])
check("list / signal: the masks step at those times", np.array_equal(sqs[9], m0) and np.array_equal(sqs[10], m1) and np.array_equal(sqs[34], m1)
      and not np.array_equal(sqs[35], m1) and any("2 times in the list" in s_ for s_ in ls), str(ls))
_, sqd, _ = sw.run(bg, masks, None, x, SR, sw.Settings(step_on="beats", fade=0.4, transition="dip", fps=20, seconds=1.0, scale=0.5), sample=True)
check("fade style out, then in: halfway the layer is in neither mask", sqd[14][ys[0], xs[0]] == 0 and sqd[14][ys1[0], xs1[0]] == 0
      and 0 < sqd[12][ys[0], xs[0]] < 1 and 0 < sqd[16][ys1[0], xs1[0]] < 1, f"{sqd[12][ys[0], xs[0]]} {sqd[14][ys[0], xs[0]]} {sqd[16][ys1[0], xs1[0]]}")
_, sqf2, _ = sw.run(bg, masks, None, x, SR, sw.Settings(feather=8, fps=10, seconds=0.2, scale=0.5), sample=True)
edge = np.nonzero((m0[:, :-1] == 1) & (m0[:, 1:] == 0))
check("feather: soft mask edges, the middle stays whole", 0.2 < sqf2[0][edge[0][0], edge[1][0]] < 0.8 and np.unique(sqf2[0]).size > 10
      , str(sqf2[0][edge[0][0], edge[1][0]]))
n, t0, secs = sw.frame_count(len(x), SR, sw.Settings(start=7.0, seconds=5.0, fps=25))
check("start / seconds stay inside the sound", (n, t0, secs) == (25, 7.0, 1.0), str((n, t0, secs)))

# ---- through the director: animate writes the masks a layer holds, the render shows the picture through them
from kubakub.director import render as rd  # noqa: E402

W, H = 120, 80
base = np.full((H, W, 3), 0.5, np.float32)
labels = np.zeros((H, W), np.int32)
labels[10:30, 10:40], labels[10:30, 70:100], labels[50:70, 40:80] = 1, 2, 3
table = {"regions": [{"region_id": 0, "name": "wall", "group_id": "wall", "tags": []}] +
                    [{"region_id": i, "name": f"W_{i}", "group_id": "Windows", "tags": []} for i in (1, 2, 3)]}


def full(c):
    a = np.zeros((4, 4, 4), np.float32)
    a[..., :3], a[..., 3] = c, 1
    return a


src = {"input:0": full((1, 0, 0)), "input:1": full((0, 1, 0)), "input:2": full((0, 0, 1))}
swap = {"type": "swap", "path": "masks", "order": "loop", "trigger": "every", "every": 1.0, "with": ["g", "b"]}
ddoc = {"version": 1, "canvas": [W, H], "timeline": {"duration": 6, "fps": 10}, "layers": [
    {"id": "r", "name": "red", "kind": "image", "source": "input:0", "x": 0, "y": 0, "w": W, "h": H, "clip": "W_1", "motion": [swap]},
    {"id": "g", "name": "green", "kind": "image", "source": "input:1", "x": 0, "y": 0, "w": W, "h": H, "clip": "W_2"},
    {"id": "b", "name": "blue", "kind": "image", "source": "input:2", "x": 0, "y": 0, "w": W, "h": H, "clip": "W_3"},
    {"id": "base", "name": "facade", "kind": "base"}]}


def shot(t, d=ddoc):
    im = rd.render(rd.animate(d, t), base, src, labels, table, image_only=True)["image"]
    return [tuple(int(round(v)) for v in im[y, x_]) for y, x_ in ((20, 25), (20, 85), (60, 60))]      # windows 1, 2, 3


R_, G_, B_ = (1, 0, 0), (0, 1, 0), (0, 0, 1)
check("director: before the first step each picture in its own mask", shot(0.5) == [R_, G_, B_], str(shot(0.5)))
check("director: one step on, the pictures moved to the next mask", shot(1.5) == [B_, R_, G_], str(shot(1.5)))
check("director: after three steps back home, the wall untouched", shot(3.5) == [R_, G_, B_]
      and np.allclose(rd.render(rd.animate(ddoc, 1.5), base, src, labels, table, image_only=True)["image"][45, 5], 0.5))
a15 = rd.animate(ddoc, 1.5)["layers"]
check("director: animate writes clip_mix and keeps the layer's own clip; the document is left alone",
      a15[0]["clip_mix"] == [["W_2", 1.0]] and a15[0]["clip"] == "W_1" and "clip_mix" not in ddoc["layers"][0]
      and "clip_mix" not in rd.animate(ddoc, 0.5)["layers"][0])
fd = __import__("json").loads(__import__("json").dumps(ddoc))
fd["layers"][0]["motion"][0]["fade"] = 0.4
im = rd.render(rd.animate(fd, 1.2), base, src, labels, table, image_only=True)["image"]
check("director: halfway through a fade the old and the new picture share the mask, no base in between",
      np.allclose(im[20, 85], [0.5, 0.5, 0], atol=0.01), str(im[20, 85]))
rev = __import__("json").loads(__import__("json").dumps(ddoc))
rev["layers"][2]["motion"], rev["layers"][0]["motion"] = [dict(swap, **{"with": ["g", "r"]})], []
check("director: the layers trade in stack order, whichever of them carries the behaviour", shot(1.5, rev) == shot(1.5), str(shot(1.5, rev)))
stg = __import__("json").loads(__import__("json").dumps(ddoc))
stg["layers"][0]["motion"].append({"type": "stagger", "path": "regions", "mode": "sequence", "order": "left", "step": 0.1, "fade": 0, "direction": "out"})
check("director: a stagger works on the mask the layer holds now", shot(1.5, stg)[1] == (0, 0, 0) or shot(1.5, stg)[1] == G_, str(shot(1.5, stg)))
soft = __import__("json").loads(__import__("json").dumps(ddoc))
soft["layers"][0]["clip_feather"] = 8
sim = rd.render(rd.animate(soft, 0.5), base, src, labels, table, image_only=True)["image"]
check("director: clip feather = a soft edge (half at the border, whole in the middle, the other layers hard)",
      abs(sim[20, 40][0] - 0.75) < 0.08 and abs(sim[20, 36][0] - 1) < 0.12 and sim[20, 25][0] > 0.98 and 0.5 < sim[20, 44][0] < 0.62
      and np.allclose(sim[20, 69], 0.5) and np.allclose(sim[20, 70], [0, 1, 0]), f"{sim[20, 36]} {sim[20, 40]} {sim[20, 44]}")
strips = rd.render(rd.animate(soft, 0.5), base, src, labels, table)["image"]
check("director: the same with the full render (strips, layer masks)", np.allclose(strips, sim, atol=1e-5))
check("director: a smaller sequence render scales the feather", rd.scale_doc(soft, 0.5)["layers"][0]["clip_feather"] == 4)
sw15 = rd.render(rd.animate(soft, 1.5), base, src, labels, table, image_only=True)["image"]
check("director: the feather goes with the layer into the mask it holds after a swap", 0.55 < sw15[20, 70][0] < 0.95 and sw15[20, 85][0] > 0.98, str(sw15[20, 70]))
lm = rd.render(rd.animate(ddoc, 1.5), base, src, labels, table)["layer_masks"]
check("director: the layer masks output follows the swap", next(rd.mask_full(m, W, H) if hasattr(rd, "mask_full") else None for m in lm if m[0] == "r") is not None)

print()
if failures:
    print(f"{len(failures)} FAILED: {', '.join(failures)}")
    sys.exit(1)
print("all sound swap tests passed")
