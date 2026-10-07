"""
sound.py

What the sound reactive behaviours hear in a sound: its tempo and beat grid (the window's analyzeBeats, in numpy),
the loudness of three bands (low / mid / high) and the hits in each band. Pure numpy, no ComfyUI imports
(tests/test_sound.py).
"""

from __future__ import annotations

import math

import numpy as np

RATE = 100                                         # curve samples per second (= motion.LEVEL_RATE)
BANDS = {"low": (30.0, 150.0), "mid": (150.0, 2500.0), "high": (4000.0, 16000.0)}      # Hz


def to_mono(waveform) -> np.ndarray:
    """A core AUDIO waveform ([B, C, S] / [C, S] / [S], torch or numpy) -> mono float32 [S]."""
    x = waveform.detach().float().cpu().numpy() if hasattr(waveform, "detach") else np.asarray(waveform, np.float32)
    while x.ndim > 2:
        x = x[0]
    return (x.mean(axis=0) if x.ndim == 2 else x).astype(np.float32)


# --------------------------------------------------------------------------
# tempo
# --------------------------------------------------------------------------

def _onsets(x, sr):
    """(onset strength per step, steps per second): the jumps in the energy of the pre-emphasised signal."""
    x = np.asarray(x, np.float64)
    hop = max(1, int(math.floor(sr / RATE + 0.5)))
    n = len(x) // hop
    if n < 2:
        return np.zeros(0), sr / hop
    y = x[:n * hop].copy()
    y[1:] -= 0.97 * x[:n * hop - 1]                 # pre-emphasis: kick and snare edges
    env = np.log(1e-10 + (y * y).reshape(n, hop).sum(axis=1))
    on = np.zeros(n)
    on[1:] = np.maximum(0.0, np.diff(env))
    w = 25                                          # minus the local mean (0.5 s): only real jumps stay
    cs = np.concatenate([[0.0], np.cumsum(on)])
    idx = np.arange(n)
    a, b = np.maximum(0, idx - w), np.minimum(n, idx + w + 1)
    return np.maximum(0.0, on - (cs[b] - cs[a]) / (b - a)), sr / hop


def beat_phase(x, sr, bpm):
    """Seconds of the first beat in the file for a tempo you know: the grid that meets the most onsets."""
    od, rate = _onsets(x, sr)
    per = 60.0 / max(1e-6, float(bpm)) * rate
    if len(od) < 2 or per < 2:
        return 0.0
    steps = min(200, max(1, int(math.ceil(per))))
    best, ph = -1.0, 0.0
    for p in range(steps):
        t0 = p * per / steps
        i = np.floor(t0 + np.arange(int((len(od) - 1 - t0) // per) + 1) * per + 0.5).astype(np.int64)
        i = i[i < len(od)]
        s = float(od[i].sum())
        if s > best:
            best, ph = s, t0
    return round(ph / rate, 3)


def analyze_beats(x, sr):
    """
    {"bpm", "phase" (s of the first beat in the file), "confidence"} or None for less than 2 s. The same method as
    the window (web/kubakub_director.js analyzeBeats): onset envelope at 100 Hz, tempo by autocorrelation with a
    soft preference around 120 bpm, phase by the best fit, a least squares fit of the beats to their onset peaks.
    """
    od, rate = _onsets(x, sr)
    n = len(od)
    if n < 200:
        return None

    def ac(lag):
        return float(np.dot(od[:n - lag], od[lag:])) / (n - lag) if lag < n else 0.0

    best, best_lag, score = 0.0, 50, {}
    for lag in range(33, 101):                      # 180 .. 60 bpm
        wgt = math.exp(-0.5 * math.log2((6000.0 / lag) / 120.0) ** 2)
        s = (ac(lag) + 0.5 * ac(2 * lag)) * wgt
        score[lag] = s
        if s > best:
            best, best_lag = s, lag
    s0, s2 = score.get(best_lag - 1, best), score.get(best_lag + 1, best)
    den = s0 - 2 * best + s2
    period = best_lag + (0.5 * (s0 - s2) / den if den < 0 else 0.0)

    def at(t):
        i = int(math.floor(t + 0.5))
        return max((od[j] for j in (i - 1, i, i + 1) if 0 <= j < n), default=0.0)

    def phase_of(per):
        ph, top = 0, -1.0
        for p in range(int(math.ceil(per))):
            s, t = 0.0, float(p)
            while t < n:
                s += at(t)
                t += per
            if s > top:
                top, ph = s, p
        return float(ph)

    ph = phase_of(period)
    on_b = off_b = 0.0                              # off-beats as strong as the beats: twice as fast (174 found as 87)
    t = ph
    while t + period / 2 < n:
        on_b += at(t)
        off_b += at(t + period / 2)
        t += period
    if off_b > 0.6 * on_b and 6000.0 / (period / 2) <= 200:
        period /= 2
        ph = phase_of(period)
    for _ in range(3):                              # snap each beat to its onset peak, fit phase + period
        pts, r = [], max(2, int(math.floor(period * 0.15 + 0.5)))
        k, f = 0, ph
        while f < n:
            c = int(math.floor(f + 0.5))
            lo, hi = max(0, c - r), min(n - 1, c + r)
            if hi >= lo:
                j = lo + int(np.argmax(od[lo:hi + 1]))
                if od[j] > 0:
                    pts.append((k, j, float(od[j])))
            k += 1
            f = ph + k * period
        if len(pts) < 4:
            break
        sw = sum(v for _, _, v in pts)
        sk = sum(v * k for k, _, v in pts)
        sf = sum(v * f for _, f, v in pts)
        skk = sum(v * k * k for k, _, v in pts)
        skf = sum(v * k * f for k, f, v in pts)
        d = sw * skk - sk * sk
        if abs(d) < 1e-9:
            break
        period = (sw * skf - sk * sf) / d
        ph = (sf - period * sk) / sw
    ph = math.fmod(math.fmod(ph, period) + period, period)
    a0 = ac(0)
    return {"bpm": round(60.0 * rate / period, 1), "phase": round(ph / rate, 2),
            "confidence": min(1.0, best / a0 * 4) if best > 0 and a0 > 0 else 0.0}


# --------------------------------------------------------------------------
# bands: loudness curves and hits
# --------------------------------------------------------------------------

def band_energy(x, sr) -> dict:
    """{band: energy per 1/100 s (float64 [n])}: a short-time spectrum (a ~23 ms Hann window centred on each step)."""
    x = np.asarray(x, np.float32)
    hop = sr / RATE
    n = int(math.floor(len(x) / hop))
    out = {b: np.zeros(max(n, 0)) for b in BANDS}
    if n < 1:
        return out
    n_fft = 1 << max(6, int(round(math.log2(max(64.0, sr * 0.023)))))
    win = np.hanning(n_fft).astype(np.float32)
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    sel = {b: (freqs >= lo) & (freqs < min(hi, sr / 2.0)) for b, (lo, hi) in BANDS.items()}
    xp = np.concatenate([np.zeros(n_fft // 2, np.float32), x, np.zeros(n_fft, np.float32)])
    starts = np.floor((np.arange(n) + 0.5) * hop).astype(np.int64)        # the window centred on the step
    span = np.arange(n_fft)[None]
    for c in range(0, n, 2048):                     # in blocks: a 5 min song is 30 000 windows
        spec = np.abs(np.fft.rfft(xp[starts[c:c + 2048, None] + span] * win, axis=1)) ** 2
        for b, m in sel.items():
            if m.any():
                out[b][c:c + 2048] = spec[:, m].sum(axis=1)
    return out


def _curve(e):
    """Loudness 0..1.5 of an energy curve: its square root over its 98th percentile (as motion.level_curve)."""
    n = len(e)
    if n < 1:
        return []
    v = np.sqrt(e)
    ref = float(np.sort(v)[min(n - 1, int(math.floor(0.98 * (n - 1))))])
    return [0.0] * n if ref <= 1e-9 else np.minimum(1.5, v / ref).round(5).tolist()


def _hits(e):
    """[[t (s), strength 0-1]] of an energy curve: where it jumps up, well above what goes on around it."""
    n = len(e)
    if n < 8:
        return []
    ref = float(np.sort(e)[min(n - 1, int(math.floor(0.98 * (n - 1))))])
    if ref <= 1e-12:
        return []
    lg = np.log1p(np.concatenate([[0.0, 0.0], e]) / ref * 100.0)        # silence before the file: a hit at 0 counts
    fx = np.maximum(0.0, lg[2:] - lg[:-2])         # the rise over two steps (20 ms)
    w = 15                                          # against the local mean (0.3 s)
    cs = np.concatenate([[0.0], np.cumsum(fx)])
    idx = np.arange(n)
    a, b = np.maximum(0, idx - w), np.minimum(n, idx + w + 1)
    mean = (cs[b] - cs[a]) / (b - a)
    top = float(fx.max())
    if top <= 0:
        return []
    pad = np.concatenate([np.zeros(5), fx, np.zeros(5)])
    near = np.lib.stride_tricks.sliding_window_view(pad, 11).max(axis=1)        # the peak within +-50 ms
    pk = np.nonzero((fx >= near) & (fx > 1.5 * mean + 0.05 * top))[0]
    if not len(pk):
        return []
    keep, last = [], -10
    for i in pk:                                    # a flat top gives two neighbours: the first one
        if i - last > 5:
            keep.append(int(i))
        last = int(i)
    s = fx[keep]
    norm = float(np.sort(s)[min(len(s) - 1, int(math.floor(0.95 * (len(s) - 1))))]) or top
    return [[round(i / RATE, 4), round(min(1.0, float(v) / norm), 4)] for i, v in zip(keep, s)]


def analyze(x, sr) -> dict:
    """{"rate": 100, "curves": {band: [0..1.5 per 1/100 s]}, "hits": {band: [[t, strength 0-1], ...]}} of a mono signal."""
    e = band_energy(x, sr)
    return {"rate": RATE, "curves": {b: _curve(v) for b, v in e.items()}, "hits": {b: _hits(v) for b, v in e.items()}}


# --------------------------------------------------------------------------
# other signals: a control wav (CV, gate, trigger, an LFO), a list of times
# --------------------------------------------------------------------------

def gate_times(x, sr, threshold=0.5, gap=0.0):
    """Seconds where a control signal rises through threshold (0-1 of its peak): a gate or trigger track, an LFO,
    an envelope. It has to fall below half the threshold before it can step again; gap = the shortest time between
    two steps. The size of the signal counts, not its sign (200 values per second)."""
    x = np.abs(np.asarray(x, np.float32))
    hop = max(1, int(round(sr / 200.0)))
    n = len(x) // hop
    if n < 1:
        return []
    env = x[:n * hop].reshape(n, hop).max(axis=1)
    top = float(env.max())
    if top <= 1e-9:
        return []
    env = env / top
    on, off = min(max(float(threshold), 0.01), 1.0), min(max(float(threshold), 0.01), 1.0) * 0.5
    out, armed, last = [], True, None
    for i in np.nonzero((env >= on) | (env < off))[0]:
        if env[i] < off:
            armed = True
        elif armed:
            armed = False
            t = float(i) * hop / float(sr)
            if last is None or t - last >= gap - 1e-9:
                out.append(round(t, 3))
                last = t
    return out


def parse_times(text):
    """Seconds from a text: numbers separated by commas, spaces or new lines, or a JSON list. Sorted, no doubles."""
    import re
    out = []
    for tok in re.split(r"[\s,;\[\]]+", str(text or "")):
        try:
            v = float(tok)
        except ValueError:
            continue
        if math.isfinite(v) and v >= 0:
            out.append(v)
    return sorted(set(out))


# --------------------------------------------------------------------------
# the built-in sample: a beat to try things on
# --------------------------------------------------------------------------

def sample_sound(seconds=8.0, bpm=120.0, sr=44100):
    """Mono float32: a kick on every beat (low), a clap on beats 2 and 4 (mid), hats on the off-beats (high)."""
    n = int(round(seconds * sr))
    out = np.zeros(n, np.float32)
    rng = np.random.default_rng(7)
    beat = 60.0 / bpm

    def put(t, sig):
        i = int(round(t * sr))
        if i < n:
            out[i:i + len(sig)] += sig[:n - i]

    tk = np.arange(int(0.25 * sr)) / sr
    kick = (np.sin(2 * np.pi * (48 * tk + 22 * (1 - np.exp(-tk * 30)) / 30 * 4)) * np.exp(-tk * 16)).astype(np.float32)
    th = np.arange(int(0.05 * sr)) / sr
    hat = np.diff(rng.standard_normal(len(th) + 2), n=2).astype(np.float32) * np.exp(-th * 90).astype(np.float32) * 0.12
    tc = np.arange(int(0.12 * sr)) / sr
    clap = (np.sin(2 * np.pi * 900 * tc) * 0.5 + np.sin(2 * np.pi * 1370 * tc) * 0.35).astype(np.float32) \
        * np.exp(-tc * 34).astype(np.float32) * 0.5
    k = 0
    while k * beat < seconds:
        put(k * beat, kick * 0.9)
        put((k + 0.5) * beat, hat)
        if k % 4 in (1, 3):
            put(k * beat, clap)
        k += 1
    return np.clip(out, -1.0, 1.0)
