"""
Model free test of kubakub point cloud (kubakub/pointcloud.py, nodes/motion/nodes_pointcloud.py) and that the node's
live view (web/kubakub_pointcloud_math.js) puts points, camera, morph and tracked things where the render does
(run with node when it is installed; without node that part is skipped and said so).

Run from the ComfyUI_windows_portable folder:
    python_embeded\\python.exe ComfyUI\\custom_nodes\\ComfyUI_KubaNodes\\tests\\test_pointcloud.py
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(HERE))))     # ComfyUI (folder_paths, comfy_api)
sys.path.insert(0, os.path.dirname(HERE))
import folder_paths  # noqa: E402

TMP = tempfile.mkdtemp(prefix="kub_pointcloud_")
folder_paths.set_temp_directory(os.path.join(TMP, "temp"))
folder_paths.set_output_directory(os.path.join(TMP, "output"))
sys.path.insert(0, HERE)
import _pack  # noqa: E402,F401
from kubapack.kubakub import pointcloud as pc  # noqa: E402
from kubapack.kubakub import tracking as tk  # noqa: E402
from kubapack.nodes.motion import nodes_pointcloud as npc  # noqa: E402

failures = []


def check(name, cond, detail=""):
    print(("ok   " if cond else "FAIL ") + name + (f"  ({detail})" if detail and not cond else ""))
    if not cond:
        failures.append(name)


def run(**kw):
    kw.setdefault("look", json.dumps({"density": 256}))
    return npc.KUBA_PointCloud.execute(**kw)


try:
    # --- the look ----------------------------------------------------------------------------------
    lk = pc.look("")
    check("empty look = the defaults", lk == pc.DEFAULT_LOOK)
    lk = pc.look({"density": 99999, "depth": -3, "colour": "rainbow", "blend": "ink", "tint": ["#000000", "nope"],
                  "camera": {"end": {"yaw": 500}, "ease": "linear"}, "tracks": {"lines": "tree", "colour": "#12ab34"}})
    check("look: numbers clamped, wrong words ignored", lk["density"] == 2048 and lk["depth"] == 0 and lk["colour"] == "true"
          and lk["blend"] == "ink" and lk["tint"] == ["#000000", pc.DEFAULT_LOOK["tint"][1]])
    check("look: camera and tracks", lk["camera"]["end"]["yaw"] == 180 and lk["camera"]["ease"] == "linear"
          and lk["camera"]["end"]["dolly"] == pc.DEFAULT_LOOK["camera"]["end"]["dolly"] and lk["tracks"]["lines"] == "tree"
          and lk["tracks"]["colour"] == "#12ab34")
    check("look: broken JSON = the defaults", pc.look("{oops") == pc.DEFAULT_LOOK)

    # --- the points ----------------------------------------------------------------------------------
    check("grid: the long side has `density` points", pc.grid_size(1000, 1920, 1080) == (1000, 562) and pc.grid_size(1000, 1080, 1920) == (562, 1000))
    i = torch.arange(50000, dtype=torch.int64)
    r = pc.rnd(i, 3)
    check("pcg randoms are 0..1 and spread", float(r.min()) >= 0 and float(r.max()) <= 1 and abs(float(r.mean()) - 0.5) < 0.01)
    W, H = 128, 72
    col = torch.rand(3, H, W)
    flat = torch.full((1, H, W), 1.0)
    lk = pc.look({"density": 128, "jitter": 0, "drift": 0})
    P, c, dist, alive = pc.points(lk, 0.0, col, flat, W / H)
    check("a near (d = 1) picture sits at distance 1, the picture's own size", torch.allclose(dist, torch.ones_like(dist))
          and torch.allclose(P[:, 2], -torch.ones_like(dist)) and float(P[:, 0].abs().max()) <= W / H / 2 + 1e-5)
    cam0 = pc.basis(pc.camera_at(lk, 0.0), lk["depth"])
    sx, sy, z = pc.project(P, cam0, W, H)
    u, v = pc.grid_uv(torch.arange(len(P)), 128, 72, 0.0)
    check("from the start view every point lands on its own pixel", torch.allclose(sx, u * W, atol=1e-3) and torch.allclose(sy, v * H, atol=1e-3))
    far = torch.zeros((1, H, W))
    lk2 = pc.look({"density": 128, "jitter": 0, "drift": 0, "depth": 1.0})
    P2, _c, dist2, _a = pc.points(lk2, 0.0, col, far, W / H)
    sx2, sy2, z2 = pc.project(P2, cam0, W, H)
    check("a far point is farther but lands on the same pixel", torch.allclose(dist2, torch.full_like(dist2, 3.0))
          and torch.allclose(sx2, sx, atol=1e-3) and torch.allclose(z2, torch.full_like(z2, 3.0), atol=1e-5))
    _P, _c, _d, alive = pc.points(pc.look({"density": 128, "far_cut": 0.5}), 0.0, col, far, W / H)
    check("far cut leaves out what is farther", not bool(alive.any()))
    k, s = pc.morph_at(0.5, 6.0, 2)
    k2, s2 = pc.morph_at(2.99, 6.0, 2)
    k3, s3 = pc.morph_at(5.0, 6.0, 2)
    check("morph timeline: hold, travel at the end of the picture's time, the last one stays",
          (k, s) == (0, 0.0) and k2 == 0 and s2 > 0.99 and (k3, s3) == (1, 0.0))
    perm = pc.rank_perm(torch.tensor([0.1, 0.9, 0.5]), torch.tensor([0.7, 0.2, 0.4]))
    check("morph partners: the same brightness rank", perm.tolist() == [1, 0, 2])

    # --- drawing -----------------------------------------------------------------------------------
    grey = torch.full((3, H, W), 0.5)
    lk = pc.look({"density": 128, "jitter": 0, "drift": 0, "opacity": 1.0, "size": 1.0})
    P, c, dist, alive = pc.points(lk, 0.0, grey, flat, W / H)
    gw, gh = pc.grid_size(128, W, H)
    im, al = pc.splat(P, c, dist, alive, lk, cam0, W, H, gh)
    check("glow from the start view: one point per pixel = the picture", torch.allclose(im, torch.full_like(im, 0.5), atol=1e-3)
          and float(al.min()) > 0.99, f"{float(im.min())} {float(im.max())}")
    im_ink, _ = pc.splat(P, c, dist, alive, pc.look({"density": 128, "blend": "ink", "opacity": 1.0}), cam0, W, H, gh)
    check("ink: the colour multiplied onto white", torch.allclose(im_ink, torch.full_like(im_ink, 0.5), atol=1e-3))
    lk_s = pc.look({"density": 128, "blend": "solid", "size": 4})
    im_s, al_s = pc.splat(P, c, dist, alive, lk_s, cam0, W, H, gh)
    check("solid: big round points cover the picture", float(al_s.mean()) > 0.99 and torch.allclose(im_s[al_s > 0], torch.tensor(0.5), atol=1e-3))
    turned = pc.basis({"yaw": 40, "pitch": 0, "dolly": 0, "x": 0, "y": 0}, 1.0)
    im_t, al_t = pc.splat(P, c, dist, alive, lk, turned, W, H, gh)
    check("turned 40 degrees: the picture is seen at an angle (part of the view empty)", 0.2 < float(al_t.mean()) < 0.95, str(float(al_t.mean())))

    # --- the node ------------------------------------------------------------------------------------
    out = run()
    images, alpha, fps, ply, report, audio_out = out.args if hasattr(out, "args") else out.result
    meta = out.ui["kubakub_pointcloud"][0]
    check("nothing connected: the sample facade, 4 s at 24 fps", images.shape[0] == 96 and alpha.shape == images.shape[:3]
          and fps == 24.0 and "sample facade" in report and meta["depth_from"] == "sample", report)
    check("the camera moves: first and last frame differ", float((images[0] - images[-1]).abs().mean()) > 0.01)
    check("the preview gets the frames and the depth", meta["colour"]["n"] == 1 and meta["depth"]["n"] == 1 and meta["n_out"] == 96)

    pics = torch.zeros((2, 36, 64, 3))
    pics[0, :, :32] = 1.0
    pics[1, 18:] = torch.tensor([0.9, 0.5, 0.2])
    out = run(image=pics, seconds=2.0, fps=10.0, save_ply=True)
    images, alpha, fps, ply, report, audio_out = out.args if hasattr(out, "args") else out.result
    meta = out.ui["kubakub_pointcloud"][0]
    check("two pictures morph: 2 s at 10 fps, the brightness relief noted", images.shape[0] == 20 and meta["mode"] == "pictures"
          and meta["K"] == 2 and "no depth connected" in report, report)
    check("the morph ends on the second picture", float(images[-1][..., 2].mean()) < float(images[0][..., 2].mean()) + 0.2
          and float((images[0] - images[-1]).abs().mean()) > 0.05)
    check("save_ply writes the first frame's points", ply.endswith(".ply") and os.path.isfile(ply) and pc.read_ply_count(ply) == 256 * 144
          and os.path.getsize(ply) > 256 * 144 * 15)

    clip = torch.from_numpy(tk.sample_clip(n=12, W=256, H=144))
    depth = clip.mean(-1, keepdim=True).repeat(1, 1, 1, 3)
    field = torch.zeros((12, 144, 256))
    field[:, 50:100, 100:160] = 1.0
    tracks = {"format": tk.TRACKS_FORMAT, "n": 12, "W": 256, "H": 144, "fps": 25, "curves": {},
              "tracks": [{"id": 1, "kind": "motion", "t0": 0, "box": [[30, 30, 16, 16]] * 12},
                         {"id": 2, "kind": "motion", "t0": 0, "box": [[190, 80, 16, 16]] * 12},
                         {"id": 3, "kind": "motion", "t0": 4, "box": [[128, 50, 16, 16]] * 4}]}
    lk = {"density": 256, "tracks": {"lines": "network", "colour": "#ff0000", "opacity": 1.0}, "blend": "ink"}
    out = run(image=clip, depth=depth, field=field, tracks=tracks, batch="video", look=json.dumps(lk), fps=25.0)
    images, alpha, fps, ply, report, audio_out = out.args if hasattr(out, "args") else out.result
    meta = out.ui["kubakub_pointcloud"][0]
    check("a video with depth, field and tracks: its own length", images.shape[0] == 12 and meta["mode"] == "video"
          and meta["depth_from"] == "node" and "field" in meta and "tracks" in meta, report)
    t3 = json.loads(Path(folder_paths.get_temp_directory(), meta["tracks"]["subfolder"], meta["tracks"]["filename"]).read_text())
    check("tracked things per frame, with the cloud's depth", len(t3["frames"]) == 12 and len(t3["frames"][0]) == 2 and len(t3["frames"][5]) == 3
          and all(0 <= p[3] <= 1 for p in t3["frames"][5]))
    red = (images[..., 0] > 0.6) & (images[..., 1] < 0.35) & (images[..., 2] < 0.35)
    check("the tracked things are drawn (rings and lines in their colour)", int(red[0].sum()) > 20, str(int(red[0].sum())))
    lk_off = dict(lk, push=0.0)
    out2 = run(image=clip, depth=depth, field=field, tracks=tracks, batch="video", look=json.dumps(lk_off), fps=25.0)
    check("the field pushes points out (push 0 = not)", float((out2.args[0] - images).abs().mean()) > 0.002 if hasattr(out2, "args")
          else float((out2.result[0] - images).abs().mean()) > 0.002)

    # --- forces, curl, trails, sound -------------------------------------------------------------------
    lk = pc.look({"density": 128, "jitter": 0, "drift": 0, "forces": [[0.2, 0.5, 0.8, 0.5]], "reach": 0.1})
    P0, *_ = pc.points(pc.look({"density": 128, "jitter": 0, "drift": 0}), 0.0, col, flat, W / H)
    P1, *_ = pc.points(lk, 0.0, col, flat, W / H)
    dx = (P1 - P0)[:, 0].reshape(72, 128)
    check("a drawn arrow pushes the points along it, only near it", float(dx[36, 64]) > 0.1 and float(dx[5, 64].abs()) < 1e-6
          and float((P1 - P0)[:, 1].abs().max()) < 1e-6)
    Pc, *_ = pc.points(pc.look({"density": 128, "drift": 0, "curl": 1.0}), 1.0, col, flat, W / H)
    Pc2, *_ = pc.points(pc.look({"density": 128, "drift": 0, "curl": 1.0}), 1.5, col, flat, W / H)
    check("curl moves the points on loops", 0.001 < float((Pc - Pc2).norm(dim=-1).mean()) < 0.05)
    lk_t = pc.look({"density": 128, "curl": 1.0, "trails": 1.0, "trail_share": 0.25})
    fr = dict(col_a=grey, dep_a=flat, aspect=W / H)
    lines = pc.trail_lines(lk_t, 2.0, fr, cam0, W, H)
    check("trails: fine lines from a share of the points", lines is not None and sum(len(c[0]) for c in lines) > 2000
          and max(float(c[2].max()) for c in lines) <= lk_t["trail_opacity"] * lk_t["opacity"] + 1e-6)
    keep = pc.TRAIL_CHUNK
    pc.TRAIL_CHUNK = 500                           # in small chunks: the same samples
    small = pc.trail_lines(lk_t, 2.0, fr, cam0, W, H)
    pc.TRAIL_CHUNK = keep
    check("trails in chunks = the same samples", len(small) > 3 and torch.equal(torch.cat([c[0] for c in small]), torch.cat([c[0] for c in lines])))
    long_path = [[k / 1199, k * 0.1, 0, 0, 0, 0] for k in range(1200)]
    lp = pc.look({"camera": {"path": long_path}})["camera"]["path"]
    check("a long recorded move is thinned to 600 keys, its end kept", len(lp) == 600 and lp[-1] == long_path[-1] and lp[0] == long_path[0])
    beat = {"level": 1.0, "low": 1.0, "mid": 0.0, "high": 1.0, "beat": 1.0}
    lk_s = pc.look({"density": 128, "drift": 0, "sound": {"sparkle": 1.0}})
    Ps, cs, *_ = pc.points(lk_s, 0.0, grey, flat, W / H, snd=beat)
    Pq, cq, *_ = pc.points(lk_s, 0.0, grey, flat, W / H)
    check("sound: the beat bursts, the bass swells, the highs sparkle", float((Ps - Pq).norm(dim=-1).mean()) > 0.01
          and float(cs.mean()) > float(cq.mean()))
    clip_s = torch.from_numpy(__import__("kubapack.kubakub.sound", fromlist=["x"]).sample_sound(3.0, 120, 22050))[None, None]
    out = run(audio={"waveform": clip_s, "sample_rate": 22050}, seconds=0.0)
    images, alpha, fps, ply, report, audio_out = out.args if hasattr(out, "args") else out.result
    meta = out.ui["kubakub_pointcloud"][0]
    check("a sound: a picture lasts as long as it, plays on the node, goes on to 'audio'", images.shape[0] == 72
          and len(meta["sound"]["beat"]) == 72 and max(meta["sound"]["beat"]) > 0.5 and meta["audio"]["filename"].endswith(".wav")
          and audio_out["waveform"].shape[-1] == 3 * 22050, report)

    # --- the live view computes the same (node) ------------------------------------------------------
    node = shutil.which("node")
    if node is None:
        print("skip the JavaScript checks: node is not installed")
    else:
        rng = np.random.default_rng(3)
        idx = [int(v) for v in rng.integers(0, 4_000_000, 300)]
        cams = [{"yaw": float(rng.uniform(-90, 90)), "pitch": float(rng.uniform(-60, 60)), "dolly": float(rng.uniform(-1, 0.9)),
                 "x": float(rng.uniform(-1, 1)), "y": float(rng.uniform(-1, 1))} for _ in range(20)]
        pts = [[k, float(rng.uniform(0, 1)), float(rng.uniform(0, 1)), float(rng.uniform(0, 1))] for k in range(14)]
        camera = {"start": cams[0], "end": cams[1], "ease": "smooth"}
        shaky = {"start": cams[0], "end": cams[1], "ease": "linear", "shake": 0.7,
                 "path": [[0, 0, 0, 0, 0, 0], [0.4, 20, -5, 0.3, 0.1, 0], [1, -30, 10, 0.5, 0, 0.2]]}
        math_js = Path(HERE, "..", "web", "kubakub_pointcloud_math.js").resolve().as_uri()
        src = (f"import * as M from {json.dumps(math_js)};\n"
               f"const idx = {json.dumps(idx)}, cams = {json.dumps(cams)}, pts = {json.dumps(pts)}, camera = {json.dumps(camera)}, shaky = {json.dumps(shaky)};\n"
               "const out = {};\n"
               "out.pcg = idx.map(i => M.pcg(i));\n"
               "out.uv = idx.slice(0, 50).map(i => M.gridUV(i % (1280 * 720), 1280, 720, 0.7));\n"
               "out.grid = [M.gridSize(1280, 1920, 1080), M.gridSize(777, 1080, 1920)];\n"
               "out.morph = [0, 0.7, 1.9, 2.6, 2.99, 3.5, 5.9, 8].map(t => M.morphAt(t, 9, 3));\n"
               "out.cam = [0, 0.25, 0.5, 1].map(a => M.cameraAt(camera, a));\n"
               "out.shaky = [0, 0.2, 0.4, 0.7, 1].map(a => M.cameraAt(shaky, a, a * 7.3));\n"
               "out.stride = [1, 0.5, 0.4, 0.1, 0.03, 0.01].map(M.trailStride);\n"
               "out.proj = cams.map(c => { const B = M.basis(c, 1.3); return pts.map(p => M.project(M.trackWorld(p, 16 / 9, 1.3), B, 1920, 1080)); });\n"
               "out.edges = ['within', 'off'].map(l => M.trackEdges(pts, { lines: l, distance: 0.3 }, 16 / 9, null, null));\n"
               "console.log(JSON.stringify(out));\n")
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "pc.mjs")
            with open(path, "w", encoding="utf-8") as f:
                f.write(src)
            res = subprocess.run([node, path], capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60)
        if res.returncode != 0:
            check("the live view's maths runs (node)", False, res.stderr.strip()[-400:])
        else:
            js = json.loads(res.stdout)
            check("pcg: the same numbers as the shader and the render", js["pcg"] == [int(pc.pcg(torch.tensor(i))) for i in idx])
            ii = torch.tensor([i % (1280 * 720) for i in idx[:50]])
            u, v = pc.grid_uv(ii, 1280, 720, 0.7)
            check("grid places", np.allclose(np.array(js["uv"]), torch.stack([u, v], -1).numpy(), atol=1e-6))
            check("grid sizes", js["grid"] == [list(pc.grid_size(1280, 1920, 1080)), list(pc.grid_size(777, 1080, 1920))])
            check("morph timeline", np.allclose(np.array(js["morph"]), np.array([pc.morph_at(t, 9, 3) for t in [0, 0.7, 1.9, 2.6, 2.99, 3.5, 5.9, 8]]), atol=1e-9))
            lkc = pc.look({"camera": camera})
            check("camera move", all(abs(js["cam"][n][k] - pc.camera_at(lkc, a)[k]) < 1e-9 for n, a in enumerate([0, 0.25, 0.5, 1]) for k in pc._CAM_RANGES))
            lks = pc.look({"camera": shaky})
            check("recorded path + handheld shake", len(lks["camera"]["path"]) == 3 and all(
                abs(js["shaky"][n][k] - pc.camera_at(lks, a, a * 7.3)[k]) < 1e-9 for n, a in enumerate([0, 0.2, 0.4, 0.7, 1]) for k in pc._CAM_RANGES))
            check("trail points", js["stride"] == [int(pc.trail_index(pc.look({"trail_share": s}), 1000)[1]) if s < 1 else 1
                                                     for s in [1, 0.5, 0.4, 0.1, 0.03, 0.01]])
            ok = True
            for cam, row in zip(cams, js["proj"]):
                B = pc.basis(cam, 1.3)
                P = torch.tensor([[(p[1] - 0.5) * 16 / 9 * pc.distance(p[3], 1.3), (0.5 - p[2]) * pc.distance(p[3], 1.3), -pc.distance(p[3], 1.3)] for p in pts], dtype=torch.float64)
                sx, sy, z = pc.project(P, B, 1920, 1080)
                ok &= np.allclose(np.array(row), torch.stack([sx, sy, z], -1).numpy(), atol=1e-6)
            check("camera + projection: tracked things land on the same pixels (20 views)", ok)
            lkw = pc.look({"tracks": {"lines": "within", "distance": 0.3}})
            check("tracked things joined the same way (within)", [list(e) for e in pc.track_edges(pts, lkw, 16 / 9)] == js["edges"][0] and js["edges"][1] == [])
finally:
    shutil.rmtree(TMP, ignore_errors=True)

print()
print("all point cloud tests passed" if not failures else f"{len(failures)} FAILED: " + ", ".join(failures))
sys.exit(1 if failures else 0)
