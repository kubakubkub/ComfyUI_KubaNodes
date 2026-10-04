"""
media.py

Video layers of the kubakub director: probe, timing on the timeline, frame decoding, sound and browser proxies,
all through the ffmpeg that imageio-ffmpeg ships (decodes ProRes, HAP, DNxHD, H.264/5, VP9, PNG / QuickTime RLE
with alpha). No ComfyUI imports (tests/test_media.py).

Stream parsing, the two-step seek, the raw frame pipe and the audio read are adapted from ComfyUI-VideoHelperSuite
(https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite, Kosinkadink / AustinMroz).

A video layer is an image layer whose source is a video file ("file:kuba_director/clip.mov") plus
    "media": {"segments": [{"start": 0, "in": 0, "out": -1, "speed": 1, "length": -1, "fill": "hold", "file": ""}],
              "volume": 1.0, "blend_frames": false}
- segment: on the timeline from `start` (s) for `length` s (-1 = its natural length (out - in) / speed); it plays
  the file from `in` to `out` (s, -1 = the file's end) at `speed` (2 = twice as fast, 0.5 = slow motion). A length
  longer than the natural one is filled by `fill`: "loop" (play again from in) or "hold" (keep the last frame).
  `file` (optional) = another file for this segment ("kuba_director/x.mp4"), else the layer's source.
  `reverse`: the piece plays backwards (from out to in; a hold then keeps 'in', a loop replays backwards).
- A cut = two segments; outside every segment the layer is hidden. No "media" = one segment from 0, natural length.
- blend_frames: mix the two nearest source frames (smooth slow motion / fps conform), else the nearest one.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess

import numpy as np

VIDEO_EXT = (".mp4", ".mov", ".m4v", ".mkv", ".webm", ".avi", ".mxf", ".mpg", ".mpeg", ".wmv", ".gif")
_NO_WINDOW = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0


def is_video(name: str) -> bool:
    return str(name).lower().endswith(VIDEO_EXT)


def ffmpeg_exe() -> str:
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:  # noqa: BLE001
        exe = shutil.which("ffmpeg")
        if not exe:
            raise RuntimeError("ffmpeg not found (imageio-ffmpeg is not installed and ffmpeg is not on PATH)")
        return exe


def _run(args, **kw):
    return subprocess.run(args, capture_output=True, creationflags=_NO_WINDOW, **kw)


# ------------------------------------------------------------------------------------------------ probe
_PROBE: dict = {}


def probe(path: str) -> dict:
    """{w, h, fps, duration, frames, alpha, audio, codec, vp9} of a video file (cached per path + mtime)."""
    st = os.stat(path)
    key = (os.path.abspath(path), st.st_size, st.st_mtime_ns)
    if key in _PROBE:
        return _PROBE[key]
    exe = ffmpeg_exe()
    lines = _run([exe, "-hide_banner", "-i", path, "-c", "copy", "-frames:v", "1", "-f", "null", "-"]).stderr.decode("utf-8", "replace")
    vp9 = "Video: vp9" in lines                  # VP9 alpha needs the libvpx decoder (VHS)
    if vp9:
        lines = _run([exe, "-hide_banner", "-c:v", "libvpx-vp9", "-i", path, "-c", "copy", "-frames:v", "1",
                      "-f", "null", "-"]).stderr.decode("utf-8", "replace")
    info = None
    for line in lines.split("\n"):
        m = re.search(r"^ *Stream .* Video: (\w+).*?, ([1-9]\d*)x(\d+)", line)
        if m:
            fps = re.search(r", ([\d.]+) fps", line)
            tbr = re.search(r", ([\d.]+) tbr", line)
            f = float((fps or tbr).group(1)) if (fps or tbr) else 25.0
            n = round(f * 1.001)                     # 23.98 / 29.97 / 59.94 printed rounded: the exact NTSC rate
            if abs(f - round(f)) > 0.01 and abs(f * 1.001 - n) < 0.01:
                f = n * 1000 / 1001
            pf = re.search(r"Video: [^,]*, (\w+)(?:\(([^)]*)\))?", line)
            pix, tags = (pf.group(1), pf.group(2) or "") if pf else ("", "")
            bits = re.search(r"p(9|10|12|14|16)(le|be)|(48|64)(le|be)", pix)
            info = {"codec": m.group(1), "w": int(m.group(2)), "h": int(m.group(3)), "fps": f, "pix_fmt": pix,
                    # untagged YUV: players assume Rec.709 for HD, Rec.601 for SD (ffmpeg's scaler assumes 601)
                    "matrix_tagged": bool(re.search(r"bt709|bt470bg|smpte170m|bt2020|smpte240m|fcc|ycgco|gbr", tags)),
                    "yuv": pix.startswith(("yuv", "nv", "p01", "p21", "y210", "v210", "uyvy", "yuyv")),
                    "bits": (int(bits.group(1)) if bits.group(1) else 16) if bits else 8,
                    # variable frame rate (phones): average fps and the stream's base rate differ
                    "vfr": bool(fps and tbr and abs(float(fps.group(1)) - float(tbr.group(1))) > 0.05 * float(tbr.group(1))),
                    "alpha": re.search(r"(yuva|rgba|bgra|argb|abgr|gbrap|ya8|ya16)", line) is not None}
            break
    if info is None:
        raise ValueError(f"not a readable video: {os.path.basename(path)}\n{lines[-600:]}")
    r = re.search(r"displaymatrix: rotation of (-?[\d.]+) degrees", lines)
    info["rotation"] = int(round(float(r.group(1)))) % 360 if r else 0
    if info["rotation"] in (90, 270):                # ffmpeg autorotates: work in the displayed size
        info["w"], info["h"] = info["h"], info["w"]
    d = re.search(r"Duration: (\d+):(\d+):(\d+(?:\.\d+)?)", lines)
    info["duration"] = int(d.group(1)) * 3600 + int(d.group(2)) * 60 + float(d.group(3)) if d else 0.0
    info["audio"] = re.search(r"Stream .* Audio:", lines) is not None
    info["vp9"] = vp9
    frames = max(1, int(round(info["duration"] * info["fps"]))) if info["duration"] else 0
    if info["audio"] or not frames:
        # the container duration is the longest stream (often the sound) or missing (MediaRecorder, pipes):
        # count the video frames themselves (a stream copy, no decoding)
        cnt = _run([exe, "-hide_banner"] + (["-c:v", "libvpx-vp9"] if vp9 else []) + ["-i", path, "-map", "0:v:0", "-c", "copy",
                                                                                     "-f", "null", "-"]).stderr.decode("utf-8", "replace")
        fc = re.findall(r"frame=\s*(\d+)", cnt)
        if fc and int(fc[-1]) > 0:
            frames = int(fc[-1]) if not frames else min(frames, int(fc[-1]))
            if not info["duration"]:
                info["duration"] = frames / info["fps"]
    info["frames"] = max(1, frames)
    _PROBE[key] = info
    return info


# ------------------------------------------------------------------------------------------------ timing
def _f(d, k, default):
    v = d.get(k, default) if isinstance(d, dict) else default
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else float(default)


def segments(media, duration):
    """Normalised segments of a layer's media dict for a file of `duration` s (sorted by start)."""
    raw = (media or {}).get("segments") if isinstance(media, dict) else None
    raw = raw if isinstance(raw, list) and raw else [{}]
    out = []
    for s in raw:
        if not isinstance(s, dict):
            continue
        dur = float(duration) or 0.0
        t_in = min(max(_f(s, "in", 0.0), 0.0), max(0.0, dur - 1e-3)) if dur else max(_f(s, "in", 0.0), 0.0)
        t_out = _f(s, "out", -1.0)
        t_out = dur if (t_out <= t_in or (dur and t_out > dur)) else t_out
        speed = min(max(_f(s, "speed", 1.0), 0.01), 100.0)
        natural = max(1e-3, (t_out - t_in) / speed)
        length = _f(s, "length", -1.0)
        out.append({"start": _f(s, "start", 0.0), "in": t_in, "out": t_out, "speed": speed,
                    "length": natural if length <= 0 else length, "natural": natural,
                    "fill": "loop" if s.get("fill") == "loop" else "hold", "file": str(s.get("file") or ""),
                    "reverse": bool(s.get("reverse"))})
    return sorted(out, key=lambda s: s["start"])


def source_time(segs, t):
    """(segment index, seconds into the file) at timeline time t, or None when the layer is hidden then."""
    for k in range(len(segs) - 1, -1, -1):
        s = segs[k]
        if s["start"] - 1e-9 <= t < s["start"] + s["length"] - 1e-9:
            local = (t - s["start"]) * s["speed"]
            span = s["out"] - s["in"]
            if local >= span:
                local = (local % span) if (s["fill"] == "loop" and span > 1e-9) else span
            if s.get("reverse"):                              # backwards: the frame just before 'out' first
                return k, max(s["in"], s["out"] - local - 1e-6)
            return k, s["in"] + local
    return None


def frame_pick(ts, fps, frames, blend=False):
    """Source time -> [(frame index, weight)]: the nearest frame, or the two around it when blending."""
    x = ts * fps
    if not blend:
        return [(int(min(max(np.floor(x + 1e-6), 0), frames - 1)), 1.0)]
    i0 = int(min(max(np.floor(x + 1e-6), 0), frames - 1))
    a = float(min(max(x - i0, 0.0), 1.0))
    i1 = min(i0 + 1, frames - 1)
    return [(i0, 1.0)] if a < 1e-3 or i1 == i0 else [(i0, 1.0 - a), (i1, a)]


# ------------------------------------------------------------------------------------------------ decode
def _runs(idx, fps, gap_s=5.0):
    """Sorted frame indices -> runs decoded by one ffmpeg call each (a big gap = seek instead of decoding)."""
    idx = sorted(set(idx))
    runs, cur = [], [idx[0]]
    for i in idx[1:]:
        if i - cur[-1] > gap_s * fps:
            runs.append(cur)
            cur = [i]
        else:
            cur.append(i)
    return runs + [cur]


def decode(path, indices, size=None, info=None):
    """
    Frames `indices` of a video as {index: uint8 (h, w, 4) RGBA} at `size` (w, h; None = native). Decoded in
    ascending runs with an accurate seek (a fast seek to 4 s before, then an exact one, as VHS does). Index k is
    the frame at time k / fps; variable-frame-rate files are put on that grid with ffmpeg's fps filter. Every
    requested index comes back (past the real end: the last frame).
    """
    info = info or probe(path)
    if not indices:
        return {}
    w, h = (int(size[0]), int(size[1])) if size else (info["w"], info["h"])
    exe, fps = ffmpeg_exe(), info["fps"]
    dec = ["-c:v", "libvpx-vp9"] if info.get("vp9") else []
    scale = _scale_filter(info, w, h)
    deep = info.get("bits", 8) > 8                           # 10 / 12-bit sources keep their precision (16-bit RGBA)
    bpp, pixfmt, dt = (8, "rgba64le", np.dtype("<u2")) if deep else (4, "rgba", np.uint8)
    vfr = bool(info.get("vfr"))
    out = {}
    for run in _runs(indices, fps):
        first, n = run[0], run[-1] - run[0] + 1
        start = max(0.0, (first - 0.5) / fps)                    # the first frame at or after = `first`
        if vfr:           # the fps grid must stay anchored at the file's own time 0: no input seek
            pre, post, vf = [], (["-ss", f"{start:.6f}"] if start > 0 else []), [f"fps={fps!r}"] + scale
        else:
            pre = ["-ss", f"{start - 4:.6f}"] if start > 4 else []
            post = ["-ss", "4"] if start > 4 else (["-ss", f"{start:.6f}"] if start > 0 else [])
            vf = scale
        args = [exe, "-v", "error", "-an"] + pre + dec + ["-i", path] + post + (["-vf", ",".join(vf)] if vf else []) + \
               ["-frames:v", str(n), "-pix_fmt", pixfmt, "-f", "rawvideo", "-"]
        res = _run(args)
        if res.returncode != 0:
            raise RuntimeError(f"ffmpeg could not read {os.path.basename(path)}: " + res.stderr.decode("utf-8", "replace")[-400:])
        buf = np.frombuffer(res.stdout, np.uint8)
        got = buf.size // (w * h * bpp)
        frames = buf[:got * w * h * bpp].view(dt).reshape(got, h, w, 4)
        want = set(run)
        for j in range(got):
            if first + j in want:
                out[first + j] = frames[j]
        if got:                                                  # a short read = the end of the stream
            for i in run:
                out.setdefault(i, frames[got - 1])
    miss = [i for i in indices if i not in out]
    if miss:                                                     # runs that started past the real end
        have = sorted(out)
        tail = None
        for i in miss:
            below = [k for k in have if k < i]
            if below:
                out[i] = out[below[-1]]
                continue
            if tail is None:                                     # the true last frame: decode only the last 2 s
                res = _run([exe, "-v", "error", "-an", "-sseof", "-2"] + dec + ["-i", path] + (["-vf", ",".join(scale)] if scale else []) +
                           ["-pix_fmt", pixfmt, "-f", "rawvideo", "-"])
                b = np.frombuffer(res.stdout, np.uint8)
                g = b.size // (w * h * bpp)
                if g == 0:
                    raise RuntimeError(f"ffmpeg returned no frames for {os.path.basename(path)}")
                tail = b[(g - 1) * w * h * bpp:g * w * h * bpp].view(dt).reshape(h, w, 4)
            out[i] = tail
    return out


def _assumed_matrix(info):
    """The YUV matrix of a file: 'auto' when tagged (or RGB); untagged = Rec.709 for HD, Rec.601 for SD, as players do."""
    if not info.get("yuv") or info.get("matrix_tagged"):
        return "auto"
    return "bt709" if max(info["w"], info["h"]) >= 1280 or min(info["w"], info["h"]) >= 720 else "bt601"


def _scale_filter(info, w, h):
    """The decode's scale filter: resize when needed, the matrix of untagged YUV. [] when nothing is needed."""
    opts = []
    if (w, h) != (info["w"], info["h"]):
        opts.append(f"{w}:{h}:flags=area")
    m = _assumed_matrix(info)
    if m != "auto":
        opts.append("in_color_matrix=" + m)
    return ["scale=" + ":".join(opts)] if opts else []


def mix(frames: dict, picks):
    """Float RGBA (0..1) from decoded frames (8 or 16 bit) and [(index, weight)] picks."""
    acc = None
    for i, wgt in picks:
        f = frames[i].astype(np.float32) * (wgt / (255.0 if frames[i].dtype == np.uint8 else 65535.0))
        acc = f if acc is None else acc + f
    return acc


# ------------------------------------------------------------------------------------------------ sound
def read_audio(path, t_in, dur, sr=44100):
    """[2, S] float32 of the file from t_in for dur s (silence where the file has none)."""
    n = max(0, int(round(dur * sr)))
    res = _run([ffmpeg_exe(), "-v", "error", "-ss", f"{max(0.0, t_in):.6f}", "-i", path, "-t", f"{dur:.6f}", "-vn",
                "-ac", "2", "-ar", str(sr), "-f", "f32le", "-"])
    a = np.frombuffer(res.stdout, np.float32) if res.returncode == 0 else np.zeros(0, np.float32)
    a = a[:a.size // 2 * 2].reshape(-1, 2).T
    out = np.zeros((2, n), np.float32)
    out[:, :min(n, a.shape[1])] = a[:, :n]
    return out


def layer_audio(segs, resolve, duration_s, sr=44100, volume=1.0):
    """The sound of a video layer on a timeline of duration_s: [2, S] (segments at speed 1; others are silent)
    and notes. resolve(segment) -> file path or None."""
    n = int(round(duration_s * sr))
    track, notes = np.zeros((2, n), np.float32), []
    for s in segs:
        path = resolve(s)
        if not path:
            continue
        if not probe(path)["audio"]:
            continue
        if abs(s["speed"] - 1.0) > 1e-3:
            notes.append(f"{os.path.basename(path)}: sound left out at speed {s['speed']:g}")
            continue
        a0, a1 = int(round(s["start"] * sr)), min(n, int(round((s["start"] + s["length"]) * sr)))
        if a1 <= a0 or a0 >= n:
            continue
        span = s["out"] - s["in"]
        clip = read_audio(path, s["in"], span, sr)
        if s.get("reverse"):                                  # the sound plays backwards too
            clip = np.ascontiguousarray(clip[:, ::-1])
        need = a1 - max(a0, 0)
        if s["fill"] == "loop" and clip.shape[1]:
            reps = int(np.ceil((a1 - a0) / clip.shape[1]))
            clip = np.tile(clip, (1, reps))
        seg = clip[:, max(0, -a0):max(0, -a0) + need]
        track[:, max(a0, 0):max(a0, 0) + seg.shape[1]] += seg * float(volume)
    return track, notes


# ------------------------------------------------------------------------------------------------ proxy
def make_proxy(path, out_path, width=960):
    """A small browser-playable copy for the window: H.264 mp4 (VP9 webm when the source has alpha), a keyframe
    every 5 frames so scrubbing is instant, the sound as AAC / Opus. -> out_path (the extension is set here).
    Written to '<name>.tmp.<ext>' and renamed when complete: a half-written or failed proxy is never served."""
    info = probe(path)
    w = min(width, info["w"])
    w -= w % 2
    h = max(2, int(round(info["h"] * w / info["w"] / 2)) * 2)
    dec = ["-c:v", "libvpx-vp9"] if info.get("vp9") else []
    if info["alpha"]:
        ext = ".webm"
        codec = ["-c:v", "libvpx-vp9", "-pix_fmt", "yuva420p", "-b:v", "0", "-crf", "34", "-deadline", "realtime",
                 "-cpu-used", "8", "-g", "5", "-c:a", "libopus", "-b:a", "96k"]
    else:
        ext = ".mp4"
        codec = ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast", "-crf", "23", "-g", "5",
                 "-movflags", "+faststart", "-c:a", "aac", "-b:a", "128k"]
    final = os.path.splitext(out_path)[0] + ext
    tmp = os.path.splitext(out_path)[0] + ".tmp" + ext
    tags = ["-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709", "-color_range", "tv"]   # never the browser's guess
    vf = f"scale={w}:{h}:flags=area:in_color_matrix={_assumed_matrix(info)}:out_color_matrix=bt709:out_range=tv"
    try:
        res = _run([ffmpeg_exe(), "-v", "error", "-y"] + dec + ["-i", path, "-vf", vf] + codec + tags + [tmp])
        if res.returncode != 0:
            raise RuntimeError("proxy failed: " + res.stderr.decode("utf-8", "replace")[-400:])
        os.replace(tmp, final)
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass
    return final
