"""
export.py

Delivery export of the kubakub director: frames are written one by one as they are rendered (a long show at the
delivery size never has to fit in RAM). Colour: the frames are display-referred sRGB / Rec.709 floats; video files
are converted with the Rec.709 matrix and tagged Rec.709 explicitly, PNGs are plain sRGB. The timeline's sound is
muxed into video files and written as a .wav next to PNG sequences. No ComfyUI imports (tests/test_export.py).

Formats (FORMATS keys):
- png8 / png16         PNG sequence, RGB or RGBA (alpha export), 8 or 16 bit, + <name>.wav
- prores4444           .mov ProRes 4444 10 bit (with alpha when exporting alpha), PCM sound
- prores422hq          .mov ProRes 422 HQ 10 bit, PCM sound
- h264                 .mp4 H.264 8 bit 4:2:0, very high quality (plays everywhere), AAC
- h264_444             .mp4 H.264 10 bit 4:4:4 (sharp colour edges; media servers, not every player), AAC
- h265                 .mp4 H.265 10 bit 4:2:0, high quality, smaller, AAC
- preview              .mp4 H.264 at half size, small, AAC
"""

from __future__ import annotations

import os
import subprocess
import threading
import wave

import cv2
import numpy as np

from . import media as md

FORMATS = {
    "png8": {"label": "PNG sequence 8 bit", "png": 8},
    "png16": {"label": "PNG sequence 16 bit", "png": 16},
    "prores4444": {"label": "ProRes 4444 (.mov, alpha)", "ext": ".mov", "alpha": True,
                   "v": ["-c:v", "prores_ks", "-profile:v", "4", "-vendor", "apl0", "-qscale:v", "4", "-movflags", "+write_colr"], "pix": "yuv444p10le", "pix_a": "yuva444p10le",
                   "a": ["-c:a", "pcm_s24le"]},
    "prores422hq": {"label": "ProRes 422 HQ (.mov)", "ext": ".mov", "v": ["-c:v", "prores_ks", "-profile:v", "3", "-vendor", "apl0", "-movflags", "+write_colr"],
                    "pix": "yuv422p10le", "a": ["-c:a", "pcm_s24le"]},
    "h264": {"label": "H.264 high quality (.mp4)", "ext": ".mp4", "v": ["-c:v", "libx264", "-preset", "slow", "-crf", "12", "-movflags", "+faststart"],
             "pix": "yuv420p", "a": ["-c:a", "aac", "-b:a", "320k"]},
    "h264_444": {"label": "H.264 4:4:4 10 bit (.mp4)", "ext": ".mp4", "v": ["-c:v", "libx264", "-preset", "slow", "-crf", "10", "-movflags", "+faststart"],
                 "pix": "yuv444p10le", "a": ["-c:a", "aac", "-b:a", "320k"]},
    "h265": {"label": "H.265 10 bit (.mp4)", "ext": ".mp4", "v": ["-c:v", "libx265", "-preset", "medium", "-crf", "14", "-tag:v", "hvc1",
                                                                 "-x265-params", "log-level=error", "-movflags", "+faststart"],
             "pix": "yuv420p10le", "a": ["-c:a", "aac", "-b:a", "320k"]},
    "preview": {"label": "preview (.mp4, half size)", "ext": ".mp4", "half": True,
                "v": ["-c:v", "libx264", "-preset", "veryfast", "-crf", "21", "-movflags", "+faststart"], "pix": "yuv420p", "a": ["-c:a", "aac", "-b:a", "192k"]},
}
TAGS = ["-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709", "-color_range", "tv"]
ALPHA_FORMATS = [k for k, f in FORMATS.items() if f.get("png") or f.get("alpha")]     # the formats that carry alpha


def alpha_note(fmt, wanted=True):
    """A report line when alpha was asked for and the format cannot carry it (it names the ones that can), else ''."""
    if not wanted or fmt in ALPHA_FORMATS:
        return ""
    return (f"alpha not written: {FORMATS[fmt]['label']} cannot carry alpha (formats with alpha: "
            f"{', '.join(ALPHA_FORMATS)})")


def write_wav(path, wave_2xs, sr):
    """[C, S] float -> 24-bit PCM WAV."""
    a = np.clip(np.asarray(wave_2xs, np.float32), -1, 1)
    if a.ndim == 1:
        a = a[None]
    ints = (a.T * (2 ** 23 - 1)).astype("<i4")
    raw = np.frombuffer(ints.tobytes(), np.uint8).reshape(-1, 4)[:, :3].tobytes()      # little endian 24 bit
    with wave.open(path, "wb") as w:
        w.setnchannels(a.shape[0]); w.setsampwidth(3); w.setframerate(int(sr)); w.writeframes(raw)


def ordered(pool, fn, items, ahead):
    """fn(x) for every x on the pool, results in order; at most `ahead` frames are submitted and not yet taken, so
    finished frames never pile up behind a slow consumer."""
    from collections import deque
    futs = deque()
    try:
        for x in items:
            futs.append(pool.submit(fn, x))
            if len(futs) >= ahead:
                yield futs.popleft().result()
        while futs:
            yield futs.popleft().result()
    finally:
        for f in futs:                                   # an error: frames not started are dropped
            f.cancel()


class Writer:
    """
    Streams frames into a file (or a PNG sequence). write(rgb, alpha=None) takes float (h, w, 3) straight colour and
    an optional (h, w) alpha; frames must arrive in order. close() -> the written files. alpha_note: a report line
    when alpha was asked for and the format cannot carry it (the alpha is then left out), else ''.
    The same in two steps for a render pool: prep(i, rgb, alpha) quantises (and for PNGs encodes and writes frame i)
    in any thread, put(item) hands the results over in order; video frames go to ffmpeg from a writer thread.
    """

    QUEUE = 6                                            # video frames waiting for ffmpeg (~50 MB each at 3200x2160)

    def __init__(self, fmt, out_dir, name, w, h, fps, alpha=False, audio=None, sr=44100):
        if fmt not in FORMATS:
            raise ValueError(f"unknown export format {fmt!r}")
        self.f = FORMATS[fmt]
        self.fmt, self.name, self.dir = fmt, name, out_dir
        self.alpha = bool(alpha) and (self.f.get("png") or self.f.get("alpha"))
        self.alpha_note = alpha_note(fmt, alpha)         # alpha asked for, but this format has none: for the report
        self.w, self.h = int(w), int(h)
        self.fps = fps
        self.n = 0
        self.files = []
        self.err = []
        self.fail = None                                 # the writer thread's error, raised on the next put / close
        os.makedirs(out_dir, exist_ok=True)
        wav = None
        if audio is not None and np.asarray(audio).size and np.abs(audio).max() > 0:
            wav = os.path.join(out_dir, f"{name}.wav")
            write_wav(wav, audio, sr)
        if self.f.get("png"):
            self.proc = None
            self.seq_dir = os.path.join(out_dir, f"{name}_png")
            os.makedirs(self.seq_dir, exist_ok=True)
            self.files.append(self.seq_dir)
            if wav:
                self.files.append(wav)
            return
        # video: rawvideo in (16-bit), the Rec.709 conversion and tags explicit
        ow, oh = (self.w // 2 - (self.w // 2) % 2, self.h // 2 - (self.h // 2) % 2) if self.f.get("half") else (self.w, self.h)
        pix = self.f["pix_a"] if self.alpha and self.f.get("pix_a") else self.f["pix"]
        out = os.path.join(out_dir, f"{name}{self.f['ext']}")
        infmt = "rgba64le" if self.alpha else "rgb48le"
        vf = f"scale={ow}:{oh}:flags=area:out_color_matrix=bt709:out_range=tv,format={pix}" if (ow, oh) != (self.w, self.h) else              f"scale=out_color_matrix=bt709:out_range=tv,format={pix}"
        args = [md.ffmpeg_exe(), "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", infmt, "-s", f"{self.w}x{self.h}",
                "-r", f"{fps}", "-i", "-"]
        if wav:
            args += ["-i", wav]
        args += ["-vf", vf] + self.f["v"] + TAGS
        args += (self.f["a"] + ["-shortest"]) if wav else ["-an"]
        args += [out]
        self.proc = subprocess.Popen(args, stdin=subprocess.PIPE, stderr=subprocess.PIPE,
                                     creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        threading.Thread(target=lambda: self.err.append(self.proc.stderr.read()), daemon=True).start()
        self.files.append(out)
        self.wav = wav
        import queue
        self.q = queue.Queue(maxsize=self.QUEUE)
        self.feeder = threading.Thread(target=self._feed, daemon=True)
        self.feeder.start()

    def prep(self, i, rgb, alpha=None):
        """Frame i -> what put() takes: uint16 for video; a PNG is encoded and written here (returns None).
        Thread safe (own buffers per call); the inputs are not changed."""
        rgb = np.asarray(rgb, np.float32)
        if rgb.shape[:2] != (self.h, self.w):
            raise ValueError(f"frame {rgb.shape[1]}x{rgb.shape[0]} but the export is {self.w}x{self.h}")
        a = None if not self.alpha else (np.ones(rgb.shape[:2], np.float32) if alpha is None else np.asarray(alpha, np.float32))
        top = 255 if self.f.get("png") == 8 else 65535
        # clip(px * top + 0.5, 0, top) in one fresh buffer (the concat's, or the multiply's): same values, fewer copies
        px = np.concatenate([rgb, a[..., None]], -1) if a is not None else None
        px = np.multiply(px, top, out=px) if px is not None else np.multiply(rgb, top)
        px += 0.5
        np.clip(px, 0, top, out=px)
        if self.proc is None:                            # PNG: 8 or 16 bit, RGB or RGBA
            q = px.astype(np.uint8 if top == 255 else np.uint16)
            q = cv2.cvtColor(q, cv2.COLOR_RGBA2BGRA if a is not None else cv2.COLOR_RGB2BGR)
            ok, buf = cv2.imencode(".png", q, [cv2.IMWRITE_PNG_COMPRESSION, 3])
            if not ok:
                raise RuntimeError("PNG encoding failed")
            buf.tofile(os.path.join(self.seq_dir, f"{self.name}_{i:06d}.png"))
            return None
        return px.astype("<u2")

    def put(self, item):
        """prep()'s result, in frame order: a PNG is already on disk, a video frame is queued for ffmpeg."""
        if self.fail is not None:
            raise self.fail
        if self.proc is None:
            self.n += 1
        else:
            self.q.put(item)

    def write(self, rgb, alpha=None):
        self.put(self.prep(self.n, rgb, alpha))

    def _feed(self):
        """Writer thread: queued frames into ffmpeg (no tobytes copy). After an error it keeps taking frames (and
        drops them) so put() never blocks."""
        while True:
            q = self.q.get()
            if q is None:
                return
            if self.fail is not None:
                continue
            try:
                self.proc.stdin.write(memoryview(q).cast("B"))
                self.n += 1
            except (BrokenPipeError, OSError) as e:
                self.proc.wait()
                err = RuntimeError("the encoder stopped: " + b"".join(self.err).decode("utf-8", "replace")[-400:])
                err.__cause__ = e
                self.fail = err

    def close(self):
        if self.proc is not None:
            self.q.put(None)                             # the queued frames go in first
            self.feeder.join()
            try:
                self.proc.stdin.close()
            except OSError:
                pass
            code = self.proc.wait()
            if self.fail is not None:
                raise self.fail
            if code != 0:
                raise RuntimeError("the encoder failed: " + b"".join(self.err).decode("utf-8", "replace")[-400:])
            if getattr(self, "wav", None):
                try:
                    os.remove(self.wav)                  # muxed into the video
                except OSError:
                    pass
        return self.files
