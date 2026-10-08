"""
motion.py

Behaviours of the kubakub director (layer "motion": [...]): procedural animation on top of the keyframes, the way
Cavalry behaviours / After Effects wiggle and loopOut work. Evaluated exactly like web/kubakub_director.js (the
MOTION section there); tests/test_motion.py runs the same cases through both.

A behaviour: {"id", "type", "path", "on", "t_start", "t_end" (-1 = to the end), "fade" (s), ...type params}
  wiggle     amount, freq (Hz), seed          smooth noise
  oscillate  amount, period (s), phase (0-1), wave sine | triangle | square | saw | circle, angle (deg, position)
  drift      rate (units / s), angle           constant speed, held after the end (spin = drift on rotation)
  random     amount, every (s), seed           a new random value every step
  loop       mode cycle | pingpong | continue  the path's keyframes repeat after the last one
  pulse      amount, trigger beats | bars | markers | every, nth, every (s), attack (s), decay (s)
                                               a kick at each trigger, eased in, decaying out
  audio      amount, smooth (s), band all | low | mid | high
                                               the loudness of the timeline sound (0-1, a peak with a release)
path: any numeric animatable value, plus "position" (x and y, amounts in delivery px) and "scale" (% around the
centre). Offsets add to the keyframed value; several behaviours on one path add up.
"""

from __future__ import annotations

import bisect
import json
import math

TYPES = ("wiggle", "oscillate", "drift", "random", "loop", "pulse", "audio")
LEVEL_RATE = 100                                  # loudness curve samples per second
BANDS = ("low", "mid", "high")                    # the sound in three bands (kubakub/sound.py)
_M32 = 0xFFFFFFFF


def _imul(a, b):
    return (a * b) & _M32


def hash01(i, seed):
    """Integer hash -> [0, 1), bit-identical to the window's hash01 (Math.imul)."""
    h = (_imul(int(i), 374761393) + _imul(int(seed), 668265263)) & _M32
    h = _imul(h ^ (h >> 13), 1274126177)
    h = (h ^ (h >> 16)) & _M32
    return h / 4294967296.0


def _vnoise(x, seed):
    i = math.floor(x)
    f = x - i
    a = hash01(i, seed) * 2 - 1
    b = hash01(i + 1, seed) * 2 - 1
    u = f * f * (3 - 2 * f)
    return a + (b - a) * u


def noise(x, seed):
    """Smooth value noise in [-1, 1], two octaves."""
    return _vnoise(x, seed) * 0.72 + _vnoise(x * 2.03 + 17.1, seed + 7) * 0.28


def wave(kind, u):
    """One cycle per unit of u, in [-1, 1]; starts at 0 rising like a sine (square / saw start at their edge)."""
    frac = u - math.floor(u)
    if kind == "triangle":
        v = u + 0.25
        return 1 - 4 * abs((v - math.floor(v)) - 0.5)
    if kind == "square":
        return 1.0 if frac < 0.5 else -1.0
    if kind == "saw":
        return 2 * frac - 1
    return math.sin(2 * math.pi * u)


def _num(b, k, d):
    v = b.get(k, d)
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) else float(d)


def span(b, duration):
    s = max(0.0, _num(b, "t_start", 0))
    e = _num(b, "t_end", -1)
    return s, (duration if e < 0 else max(s, e))


def envelope(b, t, duration):
    """0 outside the behaviour's range, eased ramps of `fade` seconds at both ends."""
    s, e = span(b, duration)
    if t < s or t > e:
        return 0.0
    f = max(0.0, _num(b, "fade", 0))
    if f <= 0:
        return 1.0
    u = min(1.0, (t - s) / f, (e - t) / f)
    return u * u * (3 - 2 * u)


def _vec(b, v):
    a = math.radians(_num(b, "angle", 0))
    return v * math.cos(a), v * math.sin(a)


def beat_times(beat, offset, duration):
    """Beats in timeline time (= the window's beatTimes): the sound file starts `offset` s before 0."""
    bpm = _num(beat or {}, "bpm", 0) if isinstance(beat, dict) else 0
    if bpm <= 0:
        return []
    p = 60.0 / bpm
    t = math.fmod(math.fmod(_num(beat, "phase", 0) - offset, p) + p, p)
    out = []
    while t <= duration + 1e-9 and len(out) < 20000:
        out.append(t)
        t += p
    return out


def context(doc, level=None, sound=None):
    """What pulse / audio / swap behaviours need, once per document: beats, markers, the loudness curve (file time)
    and, from sound (kubakub/sound.py analyze), the loudness per band and the hits per band."""
    tl = (doc or {}).get("timeline") or {}
    dur = min(max(_num(tl, "duration", 10), 0.5), 3600.0)
    au = tl.get("audio") if isinstance(tl.get("audio"), dict) else {}
    ms = sorted(float(m["t"]) for m in tl.get("markers") or [] if isinstance(m, dict) and isinstance(m.get("t"), (int, float)))
    return {"beats": beat_times(tl.get("beat"), _num(au, "offset", 0), dur), "markers": ms,
            "level": level, "offset": _num(au, "offset", 0),
            "bands": (sound or {}).get("curves") or {}, "hits": (sound or {}).get("hits") or {}}


def level_curve(mono, sr):
    """Loudness per 1/100 s of a mono signal (RMS), normalised by its 98th percentile (= the window's levelCurve)."""
    import numpy as np
    x = np.asarray(mono, np.float64)
    hop = sr / LEVEL_RATE
    n = int(math.floor(len(x) / hop))
    if n < 1:
        return []
    out = np.empty(n)
    for i in range(n):
        a, b = int(math.floor(i * hop)), int(math.floor((i + 1) * hop))
        seg = x[a:b]
        out[i] = math.sqrt(float(np.dot(seg, seg)) / max(1, len(seg)))
    ref = float(np.sort(out)[min(n - 1, int(math.floor(0.98 * (n - 1))))])
    return [0.0] * n if ref <= 1e-9 else [min(1.5, v / ref) for v in out.tolist()]


def _level(ctx, t, smooth, band="all"):
    lv = ((ctx or {}).get("bands") or {}).get(band) if band in BANDS else (ctx or {}).get("level")
    if not lv:
        return 0.0
    ft = t + (ctx.get("offset") or 0.0)
    i = int(math.floor(ft * LEVEL_RATE))
    k = max(0, int(math.floor(smooth * LEVEL_RATE + 0.5)))       # JS Math.round, not banker's rounding
    if i < 0 or i - k >= len(lv):
        return 0.0
    best = 0.0
    for j in range(max(0, i - k), min(i, len(lv) - 1) + 1):   # past the file's end: its tail still releases          # peak with an exponential release of `smooth` seconds
        v = lv[j] * (math.exp(-(i - j) / (smooth * LEVEL_RATE)) if smooth > 0 else 1.0)
        if v > best:
            best = v
    return best


def _triggers(b, ctx, s, e):
    trig = b.get("trigger") or "beats"
    if trig == "markers":
        ts = (ctx or {}).get("markers") or []
    elif trig == "every":
        step = max(0.02, _num(b, "every", 0.5))
        return ("every", s, step)
    else:
        bs = (ctx or {}).get("beats") or []
        nth = max(1, int(_num(b, "nth", 1))) * (4 if trig == "bars" else 1)
        ts = bs[::nth]
    return ("list", ts)


def pulse_env(b, dt):
    a, d = max(0.0, _num(b, "attack", 0.02)), max(1e-3, _num(b, "decay", 0.25))
    if dt < 0:
        return 0.0
    if dt < a:
        u = dt / a
        return u * u * (3 - 2 * u)
    return math.exp(-(dt - a) / d)


def _last_trigger(b, ctx, t, s, e):
    tr = _triggers(b, ctx, s, e)
    if tr[0] == "every":
        _, s0, step = tr
        return None if t < s0 else s0 + math.floor((t - s0) / step + 1e-9) * step
    last = None
    for x in tr[1]:
        if x > t + 1e-9:
            break
        if x >= s - 1e-9:
            last = x
    return last


def _one(b, t, duration, ctx=None):
    """-> [(path, add)] of one behaviour at t ('position' split into x / y)."""
    typ, path = b.get("type"), str(b.get("path") or "")
    if not path or typ not in TYPES or typ == "loop" or b.get("on") is False:
        return []
    s, e = span(b, duration)
    if typ == "drift":
        if t < s:
            return []
        v = _num(b, "rate", 0) * (min(t, e) - s)
        return list(zip(("x", "y"), _vec(b, v))) if path == "position" else [(path, v)]
    env = envelope(b, t, duration)
    if env <= 0:
        return []
    tl, amt, seed = t - s, _num(b, "amount", 0) * env, int(_num(b, "seed", 1))
    if typ in ("pulse", "audio"):
        if typ == "pulse":
            lt = _last_trigger(b, ctx, t, s, e)
            v = 0.0 if lt is None else amt * pulse_env(b, t - lt)
        else:
            v = amt * _level(ctx, t, max(0.0, _num(b, "smooth", 0.15)), b.get("band") or "all")
        return list(zip(("x", "y"), _vec(b, v))) if path == "position" else [(path, v)]
    if typ == "wiggle":
        fr = _num(b, "freq", 1)
        if path == "position":
            return [("x", amt * noise(tl * fr, seed)), ("y", amt * noise(tl * fr, seed + 101))]
        return [(path, amt * noise(tl * fr, seed))]
    if typ == "random":
        k = math.floor(tl / max(1e-3, _num(b, "every", 0.5)))
        if path == "position":
            return [("x", amt * (hash01(k, seed) * 2 - 1)), ("y", amt * (hash01(k, seed + 101) * 2 - 1))]
        return [(path, amt * (hash01(k, seed) * 2 - 1))]
    u = tl / max(1e-3, _num(b, "period", 2)) + _num(b, "phase", 0)       # oscillate
    kind = b.get("wave") or "sine"
    if path == "position":
        if kind == "circle":
            return [("x", amt * math.cos(2 * math.pi * u)), ("y", amt * math.sin(2 * math.pi * u))]
        return list(zip(("x", "y"), _vec(b, amt * wave(kind, u))))
    return [(path, amt * wave(kind, u))]


def offsets(motion, t, duration, ctx=None):
    """{path: summed offset} of all behaviours at t; 'scale' in %. ctx from context() for pulse / audio."""
    out = {}
    for b in motion or []:
        if isinstance(b, dict):
            for p, v in _one(b, t, duration, ctx):
                out[p] = out.get(p, 0.0) + v
    return out


def loop_modes(motion):
    """{keyframed path: mode} from the loop behaviours ('position' = x and y)."""
    out = {}
    for b in motion or []:
        if isinstance(b, dict) and b.get("type") == "loop" and b.get("on") is not False and b.get("mode") in ("cycle", "pingpong", "continue"):
            for p in (("x", "y") if b.get("path") == "position" else (str(b.get("path") or ""),)):
                out[p] = b["mode"]
    return out


def loop_time(keys, t, mode):
    """t remapped for a cycle / pingpong loop of the keys (after the last key); None for continue / no loop."""
    ts = sorted(k["t"] for k in keys or [] if isinstance(k, dict) and isinstance(k.get("t"), (int, float)))
    if len(ts) < 2 or t <= ts[-1] or mode not in ("cycle", "pingpong"):
        return t
    t0, sp = ts[0], ts[-1] - ts[0]
    if sp <= 1e-6:
        return t
    n = math.floor((t - t0) / sp)
    r = (t - t0) - n * sp
    if mode == "pingpong" and n % 2 == 1:
        r = sp - r
    return t0 + r


def continue_value(keys, t, key_value):
    """'continue': past the last key the value goes on with the speed of the last segment (numbers only)."""
    ks = sorted((k for k in keys or [] if isinstance(k, dict) and isinstance(k.get("t"), (int, float))), key=lambda k: k["t"])
    v = key_value(ks, t)
    if len(ks) < 2 or t <= ks[-1]["t"]:
        return v
    a, b = ks[-2], ks[-1]
    if not all(isinstance(q.get("v"), (int, float)) and not isinstance(q.get("v"), bool) for q in (a, b)):
        return v
    return b["v"] + (b["v"] - a["v"]) / max(1e-9, b["t"] - a["t"]) * (t - b["t"])


def keyed(keys, t, mode, key_value):
    if mode == "continue":
        return continue_value(keys, t, key_value)
    return key_value(keys, loop_time(keys, t, mode) if mode else t)


def clamp(path, v):
    if path == "opacity":
        return min(1.0, max(0.0, v))
    if path in ("w", "h"):
        return max(1.0, v)
    if path.startswith("fx.") or path.endswith((".power", ".size", ".feather", "env_strength", ".brightness", ".strength", ".distance")):
        return max(0.0, v)
    return v


def compose(offs, get):
    """Offsets onto the base values (get(path) -> number or None) -> {path: new value}. 'scale' scales w / h about
    the centre of the (moved) box."""
    out = {}
    for p, a in offs.items():
        if p == "scale":
            continue
        base = get(p)
        if isinstance(base, (int, float)) and not isinstance(base, bool):
            out[p] = clamp(p, base + a)
    s = offs.get("scale")
    if s:
        vals = {k: out.get(k, get(k)) for k in ("x", "y", "w", "h")}
        if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in vals.values()):
            f = max(0.01, 1 + s / 100.0)
            w, h = vals["w"] * f, vals["h"] * f
            out.update(x=vals["x"] - (w - vals["w"]) / 2, y=vals["y"] - (h - vals["h"]) / 2, w=max(1.0, w), h=max(1.0, h))
    return out


# ---- stagger: the layer's clip regions light up one after another (layer clipped to regions, "stagger" behaviours)
#   mode sequence (in / out) | chase (loop) | wave | random;  order left | right | top | bottom | centre | outside |
#   size | random;  step (s between regions), fade (s ramp per region), hold (chase), period (wave), density (random)
STAGGER_ORDERS = ("left", "right", "top", "bottom", "centre", "outside", "size", "random")


def stagger_order(regs, order, W, H, seed=1):
    """regs [(id, [x, y, w, h]) in canvas px] -> ids in stagger order (ties broken by position, then id)."""
    def key(r):
        rid, (x, y, w, h) = r[0], r[1]
        cx, cy = x + w / 2.0, y + h / 2.0
        d2 = (cx - W / 2.0) ** 2 + (cy - H / 2.0) ** 2
        return {"right": (-cx, cy, rid), "top": (cy, cx, rid), "bottom": (-cy, cx, rid), "centre": (d2, 0, rid),
                "outside": (-d2, 0, rid), "size": (-(w * h), 0, rid), "random": (hash01(rid, seed), 0, rid)}.get(order, (cx, cy, rid))
    return [r[0] for r in sorted(regs, key=key)]


def _ramp(u):
    u = min(1.0, max(0.0, u))
    return u * u * (3 - 2 * u)


def stagger_weights(motion, t, duration, regs, W, H):
    """{region id: weight 0-1} of all active stagger behaviours at t (multiplied), or None without any."""
    out = None
    for b in motion or []:
        if not isinstance(b, dict) or b.get("type") != "stagger" or b.get("on") is False or not regs:
            continue
        s, e = span(b, duration)
        seed = int(_num(b, "seed", 1))
        ids = stagger_order(regs, b.get("order") or "left", W, H, seed)
        n, mode = len(ids), b.get("mode") or "sequence"
        step, fade = max(0.0, _num(b, "step", 0.1)), max(0.0, _num(b, "fade", 0.2))
        tl = min(t, e) - s
        w = {}
        for r, rid in enumerate(ids):
            if mode == "sequence":
                u = tl - r * step
                v = (_ramp(u / fade) if fade > 0 else (1.0 if u >= 0 else 0.0))
                w[rid] = 1.0 - v if b.get("direction") == "out" else v
            elif t < s or t > e:
                w[rid] = 1.0
            elif mode == "chase":
                hold = max(0.0, _num(b, "hold", 0.3))
                cyc = max(n * step, hold + 2 * fade, 1e-3)
                loc = tl - r * step
                if b.get("loop", True) is not False:
                    loc = loc - math.floor(loc / cyc) * cyc
                if loc < 0:
                    v = 0.0
                elif loc < fade:
                    v = _ramp(loc / fade)
                elif loc < fade + hold:
                    v = 1.0
                else:
                    v = 1.0 - _ramp((loc - fade - hold) / fade) if fade > 0 else 0.0
                w[rid] = v
            elif mode == "wave":
                w[rid] = 0.5 + 0.5 * math.sin(2 * math.pi * (tl - r * step) / max(1e-3, _num(b, "period", 2)))
            else:                                       # random: each region on / off per step
                k = math.floor(tl / max(1e-3, step if step > 0 else 0.1))
                w[rid] = 1.0 if hash01(k, seed + 7919 * rid) < _num(b, "density", 0.5) else 0.0
        out = w if out is None else {k: out.get(k, 1.0) * v for k, v in w.items()}
    return out


# ---- field: the layer's clip appears along a grey ramp over its regions, per pixel (layer clipped to regions, "field"
#   behaviours; the same engine as the node kubakub mask animate, the ramp itself is kubakub/maskfields.py field()).
#   field edge distance | direction | radial | region order | noise;  per each region | all together;  angle, order,
#   cx, cy, noise_px, seed, invert;  effect reveal | hide | band | rings;  soft, width, rings, trail (s);
#   curve ramp | saw | triangle | sine | square | random | noise | level | beats | bars | markers | low | mid | high;
#   cycle (s), from, to, easing, phase, duty, steps, nth, threshold, gap, band, amount, smooth
FIELD_KINDS = ("edge distance", "direction", "radial", "region order", "noise")
FIELD_PER = ("each region", "all together")
FIELD_EFFECTS = ("reveal", "hide", "band", "rings")
FIELD_SHAPES = ("ramp", "saw", "triangle", "sine", "square", "random", "noise")
FIELD_TRIGGERS = ("beats", "bars", "markers", "low", "mid", "high")
FIELD_CURVES = FIELD_SHAPES + ("level",) + FIELD_TRIGGERS
EASES = ("linear", "ease in", "ease out", "ease in out")
TRAIL_STEPS = 16


def ease(u, kind="linear"):
    u = min(1.0, max(0.0, float(u)))
    if kind == "ease in":
        return u * u
    if kind == "ease out":
        return 1.0 - (1.0 - u) ** 2
    if kind == "ease in out":
        return u * u * (3.0 - 2.0 * u)
    return u


def curve_shape(kind, u, easing="linear", duty=0.5, seed=1):
    """A time curve at u cycles (u >= 0) -> 0..1.

    ramp      0 to 1 once, then it stays          saw       0 to 1, again and again
    triangle  0 to 1 and back                     sine      a soft triangle
    square    1 for `duty` of the cycle, else 0   random    a new value every cycle
    noise     wanders softly, about one turn per cycle
    """
    whole = math.floor(u)
    part = u - whole
    if kind == "ramp":
        return ease(u, easing)
    if kind == "saw":
        return ease(part, easing)
    if kind == "triangle":
        return ease(1.0 - abs(2.0 * part - 1.0), easing)
    if kind == "sine":
        return 0.5 - 0.5 * math.cos(2.0 * math.pi * u)
    if kind == "square":
        return 1.0 if part < duty else 0.0
    if kind == "random":
        return hash01(int(whole), seed)
    if kind == "noise":
        return min(1.0, max(0.0, 0.5 + 0.65 * noise(u, seed)))
    return 0.0


def progress_triggers(t, triggers, cycle, easing="linear", steps=1):
    """Every trigger (sorted seconds) starts a ramp of cycle seconds. steps = 1: each trigger runs 0 to 1 again.
    steps > 1: each trigger moves on by one of that many steps, and after the last step it starts again."""
    k = bisect.bisect_right(triggers, t + 1e-9)
    if k == 0:
        return 0.0
    run = ease((t - triggers[k - 1]) / cycle, easing) if cycle > 0 else 1.0
    if steps <= 1:
        return run
    pos = (k - 1 + run) / float(steps)
    fr = pos - math.floor(pos)
    return 1.0 if fr < 1e-9 else fr


def field_of(motion):
    return next((b for b in motion or [] if isinstance(b, dict) and b.get("type") == "field" and b.get("on") is not False), None)


def field_triggers(b, curve, ctx, duration):
    """The times a field's ramp starts at: beats / bars / markers / hits inside its range. Unlike a swap's steps, one
    on the very start counts and every nth is counted from the first (as kubakub mask animate does). Kept in ctx."""
    s, e = span(b, duration)
    nth = max(1, int(_num(b, "nth", 1)))
    thr, gap = _num(b, "threshold", 0.3), max(0.0, _num(b, "gap", 0.1))
    memo = ctx.setdefault("_field_triggers", {})
    key = json.dumps([s, e, curve, nth, thr, gap])
    if key not in memo:
        if curve in ("beats", "bars"):
            ts = (ctx.get("beats") or [])[::4 if curve == "bars" else 1]
        elif curve == "markers":
            ts = ctx.get("markers") or []
        else:
            off, ts, last = ctx.get("offset") or 0.0, [], None
            for h in (ctx.get("hits") or {}).get(curve) or []:
                x = h[0] - off
                if h[1] >= thr and x >= s - 1e-9 and (last is None or x - last >= gap - 1e-9):
                    ts.append(x)
                    last = x
        memo[key] = [x for x in ts if s - 1e-9 <= x <= e + 1e-9][::nth]
    return memo[key]


def field_progress(b, t, duration, ctx=None):
    """Where the move is at t: the curve's value between `from` and `to`. Before the range it waits at the start,
    after it the last value holds."""
    s, e = span(b, duration)
    curve = b.get("curve") if b.get("curve") in FIELD_CURVES else "ramp"
    lo, hi = _num(b, "from", 0), _num(b, "to", 1)
    cyc, easing = max(0.01, _num(b, "cycle", 2)), b.get("easing") if b.get("easing") in EASES else "linear"
    tt = min(max(t, s), e)
    if curve == "level":
        v = min(1.0, max(0.0, _num(b, "amount", 1) * _level(ctx, tt, max(0.0, _num(b, "smooth", 0.15)), b.get("band") or "all")))
    elif curve in FIELD_TRIGGERS:
        ts = field_triggers(b, curve, ctx if isinstance(ctx, dict) else {}, duration)
        v = progress_triggers(tt, ts, cyc, easing, int(_num(b, "steps", 1)))
    else:
        v = curve_shape(curve, (tt - s) / cyc + max(0.0, _num(b, "phase", 0)), easing, _num(b, "duty", 0.5), int(_num(b, "seed", 1)))
    return lo + v * (hi - lo)


def field_values(b, t, duration, ctx=None):
    """[[progress, weight]] the mask is built from at t: now at weight 1 and, with a trail, earlier moments fading
    out (the brightest wins per pixel)."""
    out = [[field_progress(b, t, duration, ctx), 1.0]]
    trail = max(0.0, _num(b, "trail", 0))
    if trail > 0:
        s, _e = span(b, duration)
        dt = max(0.02, 2.5 * trail / TRAIL_STEPS)
        for i in range(1, TRAIL_STEPS + 1):
            if t - i * dt < s - 1e-9:
                break
            out.append([field_progress(b, t - i * dt, duration, ctx), math.exp(-i * dt / trail)])
    return out


# ---- swap: the layers of a group trade their masks (clips), one step per trigger. The owner layer carries the
#   behaviour, "with" = the ids of the other layers. The pictures stay where they are, only the masks move on.
#   order loop | pingpong | random;  trigger beats | bars | markers | every | low | mid | high;  nth (every nth
#   trigger), every (s), threshold (0-1: how strong a hit has to be), gap (s, the shortest time between two steps),
#   fade (s, 0 = the layer just appears in the new mask), transition cross (crossfade) | dip (out, then in), seed (random)
SWAP_ORDERS = ("loop", "pingpong", "random")
SWAP_TRIGGERS = ("beats", "bars", "markers", "every", "low", "mid", "high")


def swap_of(motion):
    return next((b for b in motion or [] if isinstance(b, dict) and b.get("type") == "swap" and b.get("on") is not False), None)


def swap_times(b, ctx, duration):
    """The times a swap steps at (sorted, after the start of its range up to its end). Kept in ctx per setting."""
    s, e = span(b, duration)
    trig = b.get("trigger") if b.get("trigger") in SWAP_TRIGGERS else "beats"
    nth = max(1, int(_num(b, "nth", 1)))
    every, thr, gap = max(0.02, _num(b, "every", 0.5)), _num(b, "threshold", 0.3), max(0.0, _num(b, "gap", 0.1))
    memo = ctx.setdefault("_swap_times", {}) if isinstance(ctx, dict) else {}
    key = json.dumps([s, e, trig, nth, every, thr, gap])
    if key in memo:
        return memo[key]
    if trig == "every":
        ts, j = [], 1
        while s + j * every <= e + 1e-9 and len(ts) < 20000:
            ts.append(s + j * every)
            j += 1
    else:
        if trig in ("beats", "bars"):
            ts = ((ctx or {}).get("beats") or [])[::4 if trig == "bars" else 1]
        elif trig == "markers":
            ts = (ctx or {}).get("markers") or []
        else:
            off, ts, last = (ctx or {}).get("offset") or 0.0, [], None
            for h in ((ctx or {}).get("hits") or {}).get(trig) or []:
                x = h[0] - off
                if h[1] >= thr and x > s + 1e-9 and (last is None or x - last >= gap - 1e-9):
                    ts.append(x)
                    last = x
        ts = [x for x in ts if s + 1e-9 < x <= e + 1e-9][nth - 1::nth]
    memo[key] = ts
    return ts


def swap_state(b, t, duration, ctx=None):
    """(steps done at t, progress 0-1 of the fade into the last step, linear). After the range the last state holds."""
    ts = swap_times(b, ctx, duration)
    k = bisect.bisect_right(ts, t + 1e-9)
    if k == 0:
        return 0, 1.0
    fade = max(0.0, _num(b, "fade", 0))
    return k, (1.0 if fade <= 0 else min(1.0, max(0.0, (t - ts[k - 1]) / fade)))


def swap_perm(n, k, order="loop", seed=1, memo=None):
    """Where the n layers sit after k steps: out[i] = the member whose mask layer i shows through. loop: one on per
    step; pingpong: there and back; random: a new shuffle per step, never the same twice in a row."""
    if n <= 1 or k <= 0:
        return list(range(n))
    if order == "random":
        chain = memo.setdefault(("perm", n, seed), [list(range(n))]) if isinstance(memo, dict) else [list(range(n))]
        while len(chain) <= k:
            j = len(chain)
            d = sorted(range(n), key=lambda i: (hash01(i, seed + 7919 * j), i))
            if d == list(range(n)):
                d = [(i + 1) % n for i in range(n)]
            chain.append([d[x] for x in chain[-1]])
        return list(chain[k])
    if order == "pingpong":
        m = k % (2 * n - 2)
        k = m if m < n else 2 * n - 2 - m
    return [(i + k) % n for i in range(n)]


def swap_mix(b, t, duration, n, ctx=None):
    """[[(member, weight), ...] per layer] at t: one mask each, two while a step crossfades. The layers are in
    stack order (the first on top), so in a mask that changes hands only the upper layer fades and the lower one
    is whole underneath: a clean crossfade without the background showing through. transition = dip: the layer
    fades out of its old mask in the first half of the fade and into the new one in the second half."""
    k, p = swap_state(b, t, duration, ctx)
    u = _ramp(p)
    order, seed = b.get("order") if b.get("order") in SWAP_ORDERS else "loop", int(_num(b, "seed", 1))
    memo = ctx.setdefault("_swap_perm", {}) if isinstance(ctx, dict) else None
    now = swap_perm(n, k, order, seed, memo)
    if k == 0 or p >= 1:
        return [[(now[i], 1.0)] for i in range(n)]
    was = swap_perm(n, k - 1, order, seed, memo)
    if b.get("transition") == "dip":
        return [[(now[i], 1.0)] if was[i] == now[i] else [(was[i], 1.0 - _ramp(2 * p)), (now[i], _ramp(2 * p - 1))] for i in range(n)]
    comes, goes = {now[i]: i for i in range(n)}, {was[i]: i for i in range(n)}
    return [[(now[i], 1.0)] if was[i] == now[i] else
            [(was[i], 1.0 - u if i < comes[was[i]] else 1.0), (now[i], u if i < goes[now[i]] else 1.0)] for i in range(n)]


def swap_clips(layers, t, duration, ctx=None):
    """{layer id: [[clip selector, weight], ...]} at t for every layer of a swap group (document layers, the first
    on top). Members need a clip and trade in stack order; a layer is in one group only (the first that names it)."""
    index = {str(L.get("id")): i for i, L in enumerate(layers or []) if isinstance(L, dict)}
    by_id = {str(L.get("id")): L for L in layers or [] if isinstance(L, dict) and L.get("kind") != "base"
             and isinstance(L.get("clip"), str) and L["clip"].strip()}
    out = {}
    for L in layers or []:
        b = swap_of(L.get("motion")) if isinstance(L, dict) else None
        lid = str(L.get("id")) if b else ""
        if not b or lid not in by_id or lid in out:
            continue
        ids = [lid]
        for i in b.get("with") if isinstance(b.get("with"), list) else []:
            if isinstance(i, str) and i in by_id and i not in ids and i not in out:
                ids.append(i)
        if len(ids) < 2:
            continue
        ids.sort(key=index.get)
        for i, mix in zip(ids, swap_mix(b, t, duration, len(ids), ctx)):
            out[i] = [[by_id[ids[m]]["clip"], w] for m, w in mix]
    return out


# ---- repeater: copies of a box layer (image / shape), a "repeat" behaviour (the first one that is on)
#   count, layout line | grid | radial, dx / dy (px, line and grid), cols (grid), radius / start (deg) / orient (radial),
#   rotate (deg per copy), scale (% per copy), opacity (per copy), delay (s per copy: copy k shows the layer k * delay earlier)
def repeat_of(motion):
    return next((b for b in motion or [] if isinstance(b, dict) and b.get("type") == "repeat" and b.get("on") is not False
                 and int(_num(b, "count", 1)) > 1), None)


def repeat_count(b):
    return max(1, min(200, int(_num(b, "count", 1))))


def repeat_offsets(b, k):
    """Copy k (1 ..): (dx, dy, drotation, scale factor, dopacity) relative to the layer."""
    lay = b.get("layout") or "line"
    if lay == "radial":
        n = repeat_count(b)
        r, a0 = _num(b, "radius", 200), math.radians(_num(b, "start", 0))
        a = a0 + 2 * math.pi * k / n
        dx, dy = r * (math.cos(a) - math.cos(a0)), r * (math.sin(a) - math.sin(a0))
        drot = math.degrees(a - a0) if b.get("orient") else 0.0
    elif lay == "grid":
        cols = max(1, int(_num(b, "cols", 4)))
        dx, dy, drot = (k % cols) * _num(b, "dx", 100), (k // cols) * _num(b, "dy", 100), 0.0
    else:
        dx, dy, drot = k * _num(b, "dx", 100), k * _num(b, "dy", 0), 0.0
    f = max(0.01, 1 + k * _num(b, "scale", 0) / 100.0)
    return dx, dy, drot + k * _num(b, "rotate", 0), f, k * _num(b, "opacity", 0)


def repeat_apply(c, b, k):
    """Places copy k: c = the layer's values (at its delayed time), changed in place."""
    dx, dy, dr, f, dop = repeat_offsets(b, k)
    w, h = c.get("w", 0) * f, c.get("h", 0) * f
    c["x"] = c.get("x", 0) + dx - (w - c.get("w", 0)) / 2
    c["y"] = c.get("y", 0) + dy - (h - c.get("h", 0)) / 2
    c["w"], c["h"] = max(1.0, w), max(1.0, h)
    c["rotation"] = c.get("rotation", 0) + dr
    c["opacity"] = min(1.0, max(0.0, c.get("opacity", 1) + dop))
    return c
