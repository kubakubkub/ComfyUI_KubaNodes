"""
Model free test for video layers of the kubakub director (kubakub/director/media.py): timing of segments
(trim, speed, loop / hold, cuts), frame picking and blending, decoding with exact seeks, alpha, scaling, sound
placement and browser proxies. Synthetic videos (every frame's colour encodes its index) made with ffmpeg.

Run from the ComfyUI_windows_portable folder:
    python_embeded\\python.exe ComfyUI\\custom_nodes\\ComfyUI_KubaNodes\\tests\\test_media.py
"""

import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from kubakub.director import media  # noqa: E402

failures = []


def check(name, cond, detail=""):
    print(("ok   " if cond else "FAIL ") + name + (f"  ({detail})" if detail and not cond else ""))
    if not cond:
        failures.append(name)


# ------------------------------------------------------------------ timing (pure)
segs = media.segments({}, 4.0)
check("no media: one segment from 0, natural length", len(segs) == 1 and segs[0]["start"] == 0 and abs(segs[0]["length"] - 4) < 1e-9)
check("inside: time maps 1:1", media.source_time(segs, 1.5) == (0, 1.5))
check("after the natural end: hidden", media.source_time(segs, 4.2) is None)
s = media.segments({"segments": [{"start": 2, "in": 1, "out": 3, "speed": 2, "length": 3, "fill": "loop"}]}, 10)
check("before the start: hidden", media.source_time(s, 1.9) is None)
check("speed 2: 0.5 s on the timeline = 1 s of the file", abs(media.source_time(s, 2.5)[1] - 2.0) < 1e-9)
check("loop: after in..out (1 s on the timeline) it starts again", abs(media.source_time(s, 3.25)[1] - 1.5) < 1e-9)
check("length 3 s: hidden after start + 3", media.source_time(s, 5.01) is None and media.source_time(s, 4.99) is not None)
h = media.segments({"segments": [{"start": 0, "in": 0, "out": 2, "length": 5, "fill": "hold"}]}, 10)
check("hold: the last frame after the end", abs(media.source_time(h, 4.0)[1] - 2.0) < 1e-9)
cut = media.segments({"segments": [{"start": 3, "in": 5, "out": 6}, {"start": 0, "in": 0, "out": 2}]}, 10)
check("cut: segments sorted, a gap between them is hidden",
      [x["start"] for x in cut] == [0, 3] and media.source_time(cut, 2.5) is None
      and media.source_time(cut, 3.5) == (1, 5.5) and media.source_time(cut, 1.0) == (0, 1.0))
slow = media.segments({"segments": [{"speed": 0.5}]}, 2)
check("slow motion 0.5: natural length doubles", abs(slow[0]["length"] - 4.0) < 1e-9 and abs(media.source_time(slow, 3.0)[1] - 1.5) < 1e-9)
check("frame pick: nearest frame", media.frame_pick(1.0, 10, 50) == [(10, 1.0)])
fp = media.frame_pick(1.04, 10, 50, blend=True)
check("frame pick: blend of the two around", [i for i, _ in fp] == [10, 11] and abs(fp[1][1] - 0.4) < 1e-6, str(fp))
check("frame pick: clamped to the last frame", media.frame_pick(99, 10, 50) == [(49, 1.0)])
check("30 fps source on a 25 fps timeline: frame for t = 1 s is 30", media.frame_pick(1.0, 30, 300) == [(30, 1.0)])
rv = media.segments({"segments": [{"start": 1, "in": 2, "out": 4, "reverse": True, "length": 5, "fill": "hold"}]}, 10)
check("reverse: starts just before 'out', runs back to 'in'", abs(media.source_time(rv, 1.0)[1] - 4.0) < 1e-5 and abs(media.source_time(rv, 2.5)[1] - 2.5) < 1e-5)
check("reverse + hold: keeps 'in' at the end", abs(media.source_time(rv, 5.5)[1] - 2.0) < 1e-9)
rl = media.segments({"segments": [{"in": 0, "out": 2, "reverse": True, "length": 6, "fill": "loop"}]}, 10)
check("reverse + loop: replays backwards", abs(media.source_time(rl, 2.5)[1] - 1.5) < 1e-5)
check("reverse: the first frame shown is the last one of the piece", media.frame_pick(media.source_time(rv, 1.0)[1], 10, 100) == [(39, 1.0)])
check("is_video", media.is_video("a/B.MOV") and media.is_video("x.mp4") and not media.is_video("x.png"))

# ------------------------------------------------------------------ decoding (ffmpeg)
tmp = tempfile.mkdtemp(prefix="kuba_media_")
exe = media.ffmpeg_exe()
W, H, FPS, N = 64, 48, 10, 80                          # 8 s: seeks past 4 s use the two-step seek


def colour(k):
    return np.array([(k * 7) % 256, (k * 3 + 40) % 256, 255 - k], np.uint8)


def make(path, alpha=False, codec=("-c:v", "png"), sound=True):
    frames = np.zeros((N, H, W, 4), np.uint8)
    for k in range(N):
        frames[k, ..., :3] = colour(k)
        frames[k, ..., 3] = 255 if not alpha else (40 + k)
    args = [exe, "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-"]
    if sound:
        args += ["-f", "lavfi", "-t", str(N / FPS), "-i", "sine=frequency=440:sample_rate=44100"]
    args += list(codec) + (["-pix_fmt", "rgba"] if codec[1] == "png" else []) + (["-c:a", "pcm_s16le"] if sound else []) + ["-shortest", path]
    subprocess.run(args, input=frames.tobytes(), check=True)


mov = os.path.join(tmp, "count.mov")
make(mov, alpha=True)
info = media.probe(mov)
check("probe: size, fps, frames, alpha, audio", (info["w"], info["h"], info["fps"], info["frames"], info["alpha"], info["audio"])
      == (W, H, 10.0, N, True, True), str(info))
got = media.decode(mov, [0, 5, 17, 45, 65, 79], info=info)
ok = all(np.array_equal(got[k][0, 0, :3], colour(k)) for k in (0, 5, 17, 45, 65, 79))
check("decode: exact frames incl. after a seek past 4 s", ok, str({k: got[k][0, 0].tolist() for k in got}))
check("decode: alpha kept", int(got[65][0, 0, 3]) == 40 + 65 and int(got[5][0, 0, 3]) == 45)
small = media.decode(mov, [12], size=(32, 24), info=info)
check("decode: scaled to the layer size", small[12].shape == (24, 32, 4) and np.abs(small[12][5, 5, :3].astype(int) - colour(12)).max() <= 2)
check("decode: runs split at big gaps", media._runs([0, 1, 2, 500, 501], 10) == [[0, 1, 2], [500, 501]])
f = media.mix(got, [(5, 0.5), (17, 0.5)])
check("mix: blended float frame", f.dtype == np.float32 and abs(f[0, 0, 0] - (colour(5)[0] / 2 + colour(17)[0] / 2) / 255) < 1e-5)
mp4 = os.path.join(tmp, "count.mp4")
make(mp4, codec=("-c:v", "libx264", "-pix_fmt", "yuv420p", "-g", "30", "-crf", "12"), sound=False)
g4 = media.decode(mp4, [3, 61], info=media.probe(mp4))
check("decode: H.264 (lossy) close to the right frame", all(np.abs(g4[k][10, 10, :3].astype(int) - colour(k)).max() < 16 for k in (3, 61)),
      str({k: g4[k][10, 10].tolist() for k in g4}))
check("probe: no sound", media.probe(mp4)["audio"] is False)

# ------------------------------------------------------------------ sound
segs = media.segments({"segments": [{"start": 1.0, "in": 0, "out": 2, "length": 5, "fill": "loop"}]}, info["duration"])
track, notes = media.layer_audio(segs, lambda s: mov, 8.0, sr=8000)
check("sound: silent before the segment", np.abs(track[:, :7900]).max() < 1e-4)
check("sound: plays from start, loops to fill the length", np.abs(track[:, 8100:8000 * 6 - 100]).max() > 0.05 and np.abs(track[:, 8000 * 3 + 50:8000 * 3 + 150]).max() > 0.05
      and np.abs(track[:, 8000 * 6 + 100:]).max() < 1e-4)
fast = media.segments({"segments": [{"speed": 2}]}, info["duration"])
_, notes = media.layer_audio(fast, lambda s: mov, 8.0, sr=8000)
check("sound: other speeds are left out with a note", len(notes) == 1 and "speed 2" in notes[0])

# ------------------------------------------------------------------ odd files (bug hunt 2026-09-26)
def frames_file(path, n, fps, extra_in=(), extra_out=(), vf=None):
    """n frames whose red channel encodes the index (lossless PNG in .mov / .mkv)."""
    fr = np.zeros((n, H, W, 4), np.uint8)
    for k in range(n):
        fr[k, ..., :3] = colour(k)
        fr[k, ..., 3] = 255
    args = [exe, "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{W}x{H}", "-r", str(fps), "-i", "-", *extra_in]
    args += (["-vf", vf] if vf else []) + ["-c:v", "png", "-pix_fmt", "rgba", *extra_out, path]
    subprocess.run(args, input=fr.tobytes(), check=True)


def index_of(px):                                    # colour(k) -> k (red = 7k mod 256, green = 3k + 40)
    return next((k for k in range(N * 4) if np.abs(px[:3].astype(int) - colour(k).astype(int)).max() <= 2), -1)


# rotated phone clip: probe in displayed size, decode not sheared
rot_src, rot = os.path.join(tmp, "rot_src.mov"), os.path.join(tmp, "rot.mov")
frames_file(rot_src, 10, 10)
subprocess.run([exe, "-v", "error", "-y", "-display_rotation", "90", "-i", rot_src, "-c", "copy", rot], check=True)
ri = media.probe(rot)
check("rotation: probe gives the displayed size", (ri["w"], ri["h"], ri["rotation"]) == (H, W, 90), str({k: ri[k] for k in ("w", "h", "rotation")}))
rf = media.decode(rot, [4], info=ri)[4]
check("rotation: native decode has the displayed shape, one clean colour", rf.shape == (W, H, 4) and np.ptp(rf[..., 0]) <= 2 and index_of(rf[5, 5]) == 4,
      f"{rf.shape} {rf[5, 5]}")
# 23.976 printed as 23.98: the exact rate
ntsc = os.path.join(tmp, "ntsc.mov")
frames_file(ntsc, 12, "24000/1001")
check("NTSC: 23.98 read as 24000/1001", abs(media.probe(ntsc)["fps"] - 24000 / 1001) < 1e-9, str(media.probe(ntsc)["fps"]))
five = os.path.join(tmp, "five.mov")
frames_file(five, 5, 5)
check("5 fps stays 5", media.probe(five)["fps"] == 5.0)
# variable frame rate: 30 frames at 30 fps, then 30 frames at 15 fps (3 s)
vfr = os.path.join(tmp, "vfr.mov")
frames_file(vfr, 60, 30, vf="setpts='if(lt(N,30),N/30,1+(N-30)/15)/TB'", extra_out=["-fps_mode", "vfr"])
vi = media.probe(vfr)
check("VFR: detected", vi["vfr"], str(vi))
vd = media.decode(vfr, list(range(0, int(vi["frames"]))), info=vi)
bad = []
for k, f in vd.items():
    t = k / vi["fps"]
    want = int(t * 30 + 1e-6) if t < 1 else 30 + int((t - 1) * 15 + 1e-6)
    if abs(index_of(f[5, 5]) - want) > 1:
        bad.append((k, index_of(f[5, 5]), want))
check("VFR: decoded frame k is the one shown at k / fps (within 1)", not bad and len(vd) == vi["frames"], str(bad[:5]))
# no Duration in the header (a piped WebM): real length from the stream, no ZeroDivisionError in a loop
piped = os.path.join(tmp, "piped.webm")
with open(piped, "wb") as fh:
    fr = np.zeros((20, H, W, 3), np.uint8) + 100
    r = subprocess.run([exe, "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", "10", "-i", "-",
                        "-c:v", "libvpx-vp9", "-deadline", "realtime", "-f", "webm", "-"], input=fr.tobytes(), capture_output=True, check=True)
    fh.write(r.stdout)
pi = media.probe(piped)
check("no duration: length from the stream", abs(pi["duration"] - 2.0) < 0.15 and pi["frames"] >= 19, str({k: pi[k] for k in ("duration", "frames")}))
zs = media.segments({"segments": [{"length": 3, "fill": "loop"}]}, 0.0)
try:
    media.source_time(zs, 2.0)
    check("zero span + loop: no ZeroDivisionError", True)
except ZeroDivisionError:
    check("zero span + loop: no ZeroDivisionError", False)
# sound longer than the picture: frames of the video, every requested index comes back
av = os.path.join(tmp, "av_long.mov")
fr = np.zeros((10, H, W, 4), np.uint8)
for k in range(10):
    fr[k, ..., :3] = colour(k); fr[k, ..., 3] = 255
subprocess.run([exe, "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{W}x{H}", "-r", "10", "-i", "-",
                "-f", "lavfi", "-t", "1.6", "-i", "sine=frequency=440:sample_rate=44100", "-c:v", "png", "-pix_fmt", "rgba",
                "-c:a", "pcm_s16le", av], input=fr.tobytes(), check=True)
ai = media.probe(av)
check("sound longer: frames counted from the video", ai["frames"] == 10, str(ai["frames"]))
past = media.decode(av, [9, 12, 14], info=ai)
check("sound longer: indices past the end give the last frame", set(past) == {9, 12, 14} and np.array_equal(past[12], past[9]))
tail = media.decode(av, [15], info=dict(ai, frames=16))
check("a run that starts past the end still returns a frame", 15 in tail and index_of(tail[15][5, 5]) == 9)

# ------------------------------------------------------------------ proxies
p1 = media.make_proxy(mp4, os.path.join(tmp, "proxy_a"), width=32)
p2 = media.make_proxy(mov, os.path.join(tmp, "proxy_b"), width=32)
check("proxy: H.264 mp4 for opaque video", p1.endswith(".mp4") and os.path.getsize(p1) > 0 and media.probe(p1)["w"] == 32)
check("proxy: VP9 webm with alpha for alpha video", p2.endswith(".webm") and media.probe(p2)["alpha"], str(media.probe(p2)))
check("proxy: no temp file left behind", not [f for f in os.listdir(tmp) if ".tmp." in f])

shutil.rmtree(tmp, ignore_errors=True)
print()
if failures:
    print(f"{len(failures)} FAILED: {', '.join(failures)}")
    sys.exit(1)
print("all media tests passed")
