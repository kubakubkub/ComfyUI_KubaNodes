"""
Model free test of kubakub timeline render (kubakub/master_render.py): the show's frames and sound from clips,
pictures, sound files and directors, with trim, speed, layering, opacity, fit, hidden and muted tracks.
Synthetic files (every video frame's colour encodes its index) made with ffmpeg.

Run from the ComfyUI_windows_portable folder:
    python_embeded\\python.exe ComfyUI\\custom_nodes\\ComfyUI_KubaNodes\\tests\\test_master_render.py
"""

import os
import shutil
import subprocess
import sys
import tempfile

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from kubakub import master_render as mr  # noqa: E402
from kubakub.director import media  # noqa: E402

failures = []


def check(name, cond, detail=""):
    print(("ok   " if cond else "FAIL ") + name + (f"  ({detail})" if detail and not cond else ""))
    if not cond:
        failures.append(name)


# ------------------------------------------------------------------ sizes and counts (pure)
check("whole fps: rounded, 1..120, junk = 25", (mr.whole_fps(23.976), mr.whole_fps(0), mr.whole_fps("x"), mr.whole_fps(500))
      == (24, 1, 25, 120))
check("frame count: length x fps, half up", mr.frame_count(2.0, 25) == 50 and mr.frame_count(1.02, 25) == 26)
check("size: scaled and even", mr.out_size(1920, 1080, 0.5) == (960, 540) and mr.out_size(1001, 601, 1.0) == (1000, 600))
check("layout: fit / fill / stretch / moved",
      mr.layout_rect({}, 100, 50, 200, 200) == (0, 50, 200, 100)
      and mr.layout_rect({"fit": "fill"}, 100, 50, 200, 200) == (-100, 0, 400, 200)
      and mr.layout_rect({"fit": "stretch", "scale": 0.5, "x": 10}, 100, 50, 200, 200) == (60, 50, 100, 100))

# ------------------------------------------------------------------ files
tmp = tempfile.mkdtemp(prefix="kuba_master_render_")
exe = media.ffmpeg_exe()
W, H, FPS, N = 64, 48, 10, 80


def colour(k):
    return np.array([(k * 7) % 256, (k * 3 + 40) % 256, 255 - k], np.uint8)


frames = np.zeros((N, H, W, 4), np.uint8)
for k in range(N):
    frames[k, ..., :3] = colour(k)
    frames[k, ..., 3] = 255
subprocess.run([exe, "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
                "-f", "lavfi", "-t", str(N / FPS), "-i", "sine=frequency=440:sample_rate=44100",
                "-c:v", "png", "-pix_fmt", "rgba", "-c:a", "pcm_s16le", "-shortest", os.path.join(tmp, "count.mov")],
               input=frames.tobytes(), check=True)
subprocess.run([exe, "-v", "error", "-y", "-f", "lavfi", "-t", "1", "-i", "sine=frequency=300:sample_rate=44100",
                os.path.join(tmp, "tone.wav")], check=True)
red = np.zeros((H, W, 3), np.uint8)
red[..., 2] = 255                                       # BGR
cv2.imwrite(os.path.join(tmp, "red.png"), red)
cv2.imwrite(os.path.join(tmp, "tall.png"), np.full((H, W // 2, 3), 255, np.uint8))


def resolve(source):
    return os.path.join(tmp, source.split(":", 1)[1])


def clip(media_id, **kw):
    it = {"type": "clip", "media": media_id, "start": 0, "in": 0, "out": -1, "speed": 1, "length": -1, "fill": "hold",
          "reverse": False, "muted": False, "props": {"opacity": 1, "x": 0, "y": 0, "scale": 1, "fit": "fit"}}
    props = kw.pop("props", {})
    it.update(kw)
    it["props"].update(props)
    return it


def show(*tracks, duration=4.0):
    """tracks top first, as in the panel."""
    return {"version": 1, "duration": duration, "fps": FPS, "width": W, "height": H,
            "tracks": [{"id": f"t{i}", "name": f"v{i}", "muted": False, "hidden": False, "items": list(t)} for i, t in enumerate(tracks)],
            "media": [{"id": "v", "name": "count.mov", "source": "file:count.mov", "kind": "video"},
                      {"id": "r", "name": "red.png", "source": "file:red.png", "kind": "image"},
                      {"id": "p", "name": "tall.png", "source": "file:tall.png", "kind": "image"},
                      {"id": "a", "name": "tone.wav", "source": "file:tone.wav", "kind": "audio", "info": {"duration": 1.0}},
                      {"id": "x", "name": "gone.mp4", "source": "file:gone.mp4", "kind": "video"}]}


def no_director(node, w, h):
    return None


def px(fr, f):
    return np.round(fr[f, H // 2, W // 2] * 255).astype(int)


def near(a, b, tol=2):
    return np.abs(np.asarray(a, int) - np.asarray(b, int)).max() <= tol


# ------------------------------------------------------------------ clips
fr, snd, fps, rep = mr.render(show([clip("v", start=1, **{"in": 2})]), 1.0, resolve, no_director)
check("frames: length x fps, project size, project fps", fr.shape == (40, H, W, 3) and fps == FPS, str(fr.shape))
check("before the clip: black", px(fr, 5).max() == 0)
check("clip at 1 s, in 2 s: frame at 1.5 s is source frame 25", near(px(fr, 15), colour(25)), f"{px(fr, 15)} vs {colour(25)}")
check("sound: placed at the clip's start", np.abs(snd[:, :44100 - 100]).max() == 0 and np.abs(snd[:, 44100 + 100:2 * 44100]).max() > 0.05)
check("report: what was rendered", rep[0].startswith("40 frames at 10 fps, 64x48") and "1 clips" in rep[0], rep[0])

fr, snd, _, rep = mr.render(show([clip("v", speed=2)]), 1.0, resolve, no_director)
check("speed 2: frame at 1 s is source frame 20", near(px(fr, 10), colour(20)), str(px(fr, 10)))
check("speed 2: sound left out and said so", np.abs(snd).max() == 0 and any("speed 2" in r for r in rep), str(rep))

fr, _, _, _ = mr.render(show([clip("v", reverse=True, out=2.0)]), 1.0, resolve, no_director)
check("backwards: starts with the last frame of the piece", near(px(fr, 0), colour(19)), str(px(fr, 0)))

# ------------------------------------------------------------------ layering, opacity, fit
fr, _, _, _ = mr.render(show([clip("r", length=4, props={"opacity": 0.5})], [clip("v")]), 1.0, resolve, no_director)
want = (np.array([255, 0, 0]) + colour(5).astype(int)) / 2
check("upper track over the lower one at its opacity", near(px(fr, 5), want), f"{px(fr, 5)} vs {want}")

fr, _, _, _ = mr.render(show([clip("p", length=1)]), 1.0, resolve, no_director)
check("fit: a narrow picture is centred, black beside it",
      fr[0, H // 2, 2].max() == 0 and fr[0, H // 2, W // 2].min() > 0.99 and fr[0, H // 2, W - 3].max() == 0)
fr, _, _, _ = mr.render(show([clip("p", length=1, props={"fit": "stretch"})]), 1.0, resolve, no_director)
check("stretch: the whole frame", fr[0].min() > 0.99)
check("picture length: 1 s", fr[9].min() > 0.99 and fr[10].max() == 0)

doc = show([clip("r", length=4)], [clip("v")])
doc["tracks"][0]["hidden"] = True
doc["tracks"][1]["muted"] = True
fr, snd, _, _ = mr.render(doc, 1.0, resolve, no_director)
check("hidden track: not seen", near(px(fr, 5), colour(5)))
check("muted track: not heard", np.abs(snd).max() == 0)

fr, _, _, _ = mr.render(show([clip("v")]), 0.5, resolve, no_director)
check("scale 0.5: half size", fr.shape[1:3] == (24, 32))

# ------------------------------------------------------------------ sound files
_, snd, _, _ = mr.render(show([clip("a", start=2)]), 1.0, resolve, no_director)
check("sound file: from 2 s for 1 s", np.abs(snd[:, :2 * 44100 - 100]).max() == 0 and np.abs(snd[:, 2 * 44100 + 100:3 * 44100 - 100]).max() > 0.05
      and np.abs(snd[:, 3 * 44100 + 100:]).max() == 0)
_, snd, _, _ = mr.render(show([clip("a", fill="loop", length=3)]), 1.0, resolve, no_director)
check("sound file looped: heard to 3 s", np.abs(snd[:, int(2.5 * 44100):int(2.9 * 44100)]).max() > 0.05)

# ------------------------------------------------------------------ keys (seconds from the piece's start)
fade = clip("r", start=1, length=3, anim={"opacity": [{"t": 0, "v": 0, "e": "linear"}, {"t": 2, "v": 1, "e": "smooth"}]})
check("props at a time: keyed values, the rest as set", mr.props_at(fade, 1.5)["opacity"] == 0.25 and mr.props_at(fade, 1.5)["fit"] == "fit")
fr, _, _, _ = mr.render(show([fade]), 1.0, resolve, no_director)
check("keyed opacity: linear fade from the piece's start", near(px(fr, 15), [64, 0, 0]) and near(px(fr, 30), [255, 0, 0]), f"{px(fr, 15)} {px(fr, 30)}")
slide = clip("p", length=2, props={"fit": "stretch", "scale": 0.5},
             anim={"x": [{"t": 0, "v": -16, "e": "hold"}, {"t": 1, "v": 16, "e": "smooth"}]})
fr, _, _, _ = mr.render(show([slide]), 1.0, resolve, no_director)
check("keyed x, hold: jumps at the next key", fr[5, H // 2, 10].min() > 0.99 and fr[5, H // 2, 40].max() == 0
      and fr[10, H // 2, 40].min() > 0.99 and fr[10, H // 2, 10].max() == 0)
zoom = clip("v", anim={"scale": [{"t": 0, "v": 0.25, "e": "linear"}, {"t": 1, "v": 1, "e": "linear"}]})
fr, _, _, _ = mr.render(show([zoom]), 1.0, resolve, no_director)
check("keyed size: small first, the whole frame at 1 s", fr[0, 2, 2].max() == 0 and near(px(fr, 0), colour(0))
      and near(np.round(fr[10, 2, 2] * 255).astype(int), colour(10)))

fr, _, _, _ = mr.render(show([clip("r", length=-1)]), 1.0, resolve, no_director)
check("a picture without a length: nothing (as on the timeline)", fr.max() == 0)
fr, _, _, _ = mr.render(show([clip("p", length=1, props={"scale": 0.25})]), 1.0, resolve, no_director)
check("a small picture keeps its shape (placed from its own size, not the shrunk one)",
      fr[0, 18:30, 28:36].min() > 0.99 and fr[0, :, :28].max() == 0 and fr[0, :, 36:].max() == 0
      and fr[0, :18].max() == 0 and fr[0, 30:].max() == 0)   # 32x48 at 0.25 of fit = 8x12 in the middle

# ------------------------------------------------------------------ directors and missing things
asked = []


def director(node, w, h):
    asked.append((node, w, h))
    if node != 7:
        return None
    d = np.zeros((10, 24, 32, 3), np.float32)
    for i in range(10):
        d[i] = i / 10.0
    return d, 5.0, {"waveform": np.full((1, 1, 44100), 0.25, np.float32)[None][0], "sample_rate": 44100}


items = [{"type": "director", "node": 7, "start": 1, "in": 0.4, "length": -1, "props": {}},
         {"type": "director", "node": 9, "start": 0, "in": 0, "length": 1, "props": {}}, clip("x", start=3)]
fr, snd, _, rep = mr.render(show(items), 1.0, resolve, director)
check("director: asked at the render size", asked[0][1:] == (W, H), str(asked))
check("director: in 0.4 s at 5 fps, at 1.5 s shows its frame 4 (0.4)", abs(fr[15, H // 2, W // 2, 0] - 0.4) < 1e-4, str(fr[15, H // 2, W // 2]))
check("director: ends after its frames (2 s - in)", fr[26].max() == 0 and fr[25].max() > 0)
check("director: its sound from 'in', placed at its start", abs(snd[0, int(1.5 * 44100)] - 0.25) < 1e-6 and np.abs(snd[:, :44100 - 10]).max() == 0)
asked.clear()
mr.render(show([{"type": "director", "node": 7, "start": 0, "in": 0, "length": 1, "props": {"scale": 1},
                 "anim": {"scale": [{"t": 0, "v": 1, "e": "linear"}, {"t": 1, "v": 2, "e": "linear"}]}}]), 1.0, resolve, director)
check("director: asked big enough for its largest size key", asked == [(7, 2 * W, 2 * H)], str(asked))
asked.clear()
doc = show([{"type": "director", "node": 7, "start": 0, "in": 0, "length": 1, "props": {}}])
doc["tracks"][0]["hidden"] = doc["tracks"][0]["muted"] = True
mr.render(doc, 1.0, resolve, director)
check("director on a hidden, muted track: not rendered at all", asked == [], str(asked))
check("not connected director and missing file: said, left out",
      any("director node 9" in r for r in rep) and any("missing file: gone.mp4" in r for r in rep), str(rep))

shutil.rmtree(tmp, ignore_errors=True)
print()
print("all timeline render tests passed" if not failures else f"{len(failures)} FAILED: " + ", ".join(failures))
sys.exit(1 if failures else 0)
