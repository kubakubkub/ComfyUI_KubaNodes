"""
Model free test of kubakub point cloud (kubakub/pointcloud.py, nodes/motion/nodes_pointcloud.py) and that the node's
live view (web/kubakub_pointcloud_math.js) puts points, camera, morph and tracked things where the render does
(run with node when it is installed; without node that part is skipped and said so).

Run from the ComfyUI_windows_portable folder:
    python_embeded\\python.exe ComfyUI\\custom_nodes\\ComfyUI_KubaNodes\\tests\\test_pointcloud.py
"""

import json
import math
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
    check("look: numbers clamped, wrong words ignored", lk["density"] == 3072 and lk["depth"] == 0 and lk["colour"] == "true"
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
    STILL = {"density": 128, "jitter": 0, "drift": 0, "flow": 0, "shiver": 0}
    lk = pc.look(STILL)
    P, c, dist, alive, _f = pc.points(lk, 0.0, col, flat, W / H)
    check("a near (d = 1) picture sits at distance 1, the picture's own size", torch.allclose(dist, torch.ones_like(dist))
          and torch.allclose(P[:, 2], -torch.ones_like(dist)) and float(P[:, 0].abs().max()) <= W / H / 2 + 1e-5)
    cam0 = pc.basis(pc.camera_at(lk, 0.0), lk["depth"])
    sx, sy, z = pc.project(P, cam0, W, H)
    u, v = pc.grid_uv(torch.arange(len(P)), 128, 72, 0.0)
    check("from the start view every point lands on its own pixel", torch.allclose(sx, u * W, atol=1e-3) and torch.allclose(sy, v * H, atol=1e-3))
    far = torch.zeros((1, H, W))
    lk2 = pc.look(dict(STILL, depth=1.0))
    P2, _c, dist2, _a, _f = pc.points(lk2, 0.0, col, far, W / H)
    sx2, sy2, z2 = pc.project(P2, cam0, W, H)
    check("a far point is farther but lands on the same pixel", torch.allclose(dist2, torch.full_like(dist2, 3.0))
          and torch.allclose(sx2, sx, atol=1e-3) and torch.allclose(z2, torch.full_like(z2, 3.0), atol=1e-5))
    _P, _c, _d, alive, _f = pc.points(pc.look({"density": 128, "far_cut": 0.5}), 0.0, col, far, W / H)
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
    lk = pc.look(dict(STILL, opacity=1.0, size=1.0))
    P, c, dist, alive, _f = pc.points(lk, 0.0, grey, flat, W / H)
    gw, gh = pc.grid_size(128, W, H)
    im, al = pc.splat(P, c, dist, alive, lk, cam0, W, H, gh)
    check("glow from the start view: one point per pixel = the picture", torch.allclose(im, torch.full_like(im, 0.5), atol=1e-3)
          and float(al.min()) > 0.99, f"{float(im.min())} {float(im.max())}")
    im_ink, _ = pc.splat(P, c, dist, alive, pc.look({"density": 128, "blend": "ink", "opacity": 1.0, "size": 1.0}), cam0, W, H, gh)
    check("ink: the colour multiplied onto white", torch.allclose(im_ink, torch.full_like(im_ink, 0.5), atol=1e-3))
    lk_s = pc.look({"density": 128, "blend": "solid", "size": 4, "flow": 0, "shiver": 0})
    im_s, al_s = pc.splat(P, c, dist, alive, lk_s, cam0, W, H, gh)
    check("solid: big round points cover the picture", float(al_s.mean()) > 0.99 and torch.allclose(im_s[al_s > 0], torch.tensor(0.5), atol=1e-3))
    turned = pc.basis({"yaw": 40, "pitch": 0, "dolly": 0, "x": 0, "y": 0}, 1.0)
    im_t, al_t = pc.splat(P, c, dist, alive, lk, turned, W, H, gh)
    check("turned 40 degrees: the picture is seen at an angle (part of the view empty)", 0.2 < float(al_t.mean()) < 0.95, str(float(al_t.mean())))
    one = pc.splat(torch.tensor([[(10.75 / W - 0.5) * W / H, 0.5 - 20.5 / H, -1.0]]), torch.ones(1, 3), torch.ones(1), torch.ones(1, dtype=torch.bool),
                   pc.look({"density": 128, "size": 0.5, "opacity": 4.0}), cam0, W, H, gh)[0][..., 0]
    check("a point smaller than a pixel shares its light with the pixels around it (no grain)", abs(float(one[20, 10]) - 0.75) < 0.02
          and abs(float(one[20, 11]) - 0.25) < 0.02 and float(one.sum()) < 1.01, str(one[19:22, 9:13]))

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
    lk = dict(lk, flow=0)
    lk_off = dict(lk, push=0.0)
    out2 = run(image=clip, depth=depth, field=field, tracks=tracks, batch="video", look=json.dumps(lk_off), fps=25.0)
    check("the field pushes points out (push 0 = not)", float((out2.args[0] - images).abs().mean()) > 0.002 if hasattr(out2, "args")
          else float((out2.result[0] - images).abs().mean()) > 0.002)

    # --- forces, curl, trails, sound -------------------------------------------------------------------
    lk = pc.look(dict(STILL, forces=[[0.2, 0.5, 0.8, 0.5]], reach=0.1, flowing=1.0))
    P0, *_ = pc.points(pc.look(STILL), 0.0, col, flat, W / H)
    P1, *_ = pc.points(lk, 1.0, col, flat, W / H, forces=pc.force_field(lk, W / H))
    dx = (P1 - P0)[:, 0].reshape(72, 128)
    check("a drawn arrow is a current: the points near it flow along it, the far ones hold", float(dx[36, 30:90].mean()) > 0.01
          and float(dx[5].abs().max()) < 1e-6 and float((P1 - P0)[:, 1].reshape(72, 128)[5].abs().max()) < 1e-6)
    lko = pc.look({"forces": [[0.2, 0.5, 0.8, 0.5]], "curves": [{"kind": "curl+", "pts": [[0.1, 0.1], [0.2, 0.2]]}, {"kind": "nope", "pts": [[0, 0], [1, 1]]},
                                                               {"kind": "wind", "pts": [[0.5, 0.5]]}], "where": "arrows", "regions": {"a": "attract", "b": "bad"}})
    check("an older look: arrows are wind curves, 'arrows' is 'forces'; wrong curves and kinds are left out",
          lko["curves"] == [{"kind": "wind", "pts": [[0.2, 0.5], [0.8, 0.5]]}, {"kind": "curl+", "pts": [[0.1, 0.1], [0.2, 0.2]]}]
          and lko["where"] == "forces" and lko["regions"] == {"a": "attract"})

    def moved_by(curves, at, t=1.0, **kw):          # the mean way of the points near `at` (u, v) under these curves, x right, y up
        lk_ = pc.look(dict(STILL, curves=curves, flowing=1.0, life=4.0, reach=0.12, speed=1.0, **kw))
        Q, *_ = pc.points(lk_, t, col, flat, W / H, forces=pc.force_field(lk_, W / H))
        g_ = pc.grid(128, 72, 0.0, "cpu")
        near = ((g_["u"] - at[0]) * W / H).abs().lt(0.03) & (g_["v"] - at[1]).abs().lt(0.03)
        return (Q - P0)[near][:, :2].mean(0)
    line = [[0.3, 0.5], [0.5, 0.5], [0.7, 0.5]]
    ma, mr = moved_by([{"kind": "attract", "pts": line}], (0.5, 0.42)), moved_by([{"kind": "repel", "pts": line}], (0.5, 0.42))
    mw = moved_by([{"kind": "wind", "pts": line[::-1]}], (0.5, 0.47))
    check("a curve: attract pulls onto it, repel pushes away, wind blows the way it was drawn",
          float(ma[1]) < -0.005 and float(mr[1]) > 0.005 and float(mw[0]) < -0.01 and abs(float(mw[1])) < 0.3 * abs(float(mw[0])), f"{ma} {mr} {mw}")
    ring = [[0.5 + 0.12 * H / W * math.cos(a), 0.5 + 0.12 * math.sin(a)] for a in np.linspace(0, 2 * math.pi, 24)]
    mp, mm = moved_by([{"kind": "curl+", "pts": ring}], (0.5 + 0.08 * H / W, 0.5), 0.6), moved_by([{"kind": "curl-", "pts": ring}], (0.5 + 0.08 * H / W, 0.5), 0.6)
    check("a loop with curl + turns anticlockwise, curl - clockwise (right of the middle: up / down)",
          float(mp[1]) > 0.005 and float(mm[1]) < -0.005, f"{mp} {mm}")
    lab = -np.ones((H, W), np.int32)
    for k_, x_ in enumerate([20, 60, 100]):
        lab[30:42, x_:x_ + 8] = k_
    lab[0:10, :] = 3
    reg = pc.region_basis(lab, {"windows": [0, 1, 2], "roof": [3]}, W / H)
    check("regions: the group with the most regions comes first and is the default", reg["names"] == ["windows", "roof"] and reg["default"] == "windows"
          and pc.region_kinds(pc.look({}), reg) == {"windows": "attract"} and pc.region_kinds(pc.look({"regions": {"roof": "repel"}}), reg) == {"roof": "repel"})
    lkr = pc.look(dict(STILL, flowing=1.0, life=4.0, reach=0.15, speed=1.0))
    Fr = pc.force_field(lkr, W / H, reg)
    check("strength 0: no forces", pc.force_field(pc.look(dict(lkr, force=0)), W / H, reg) is None)
    Qr, *_ = pc.points(lkr, 1.2, col, flat, W / H, forces=Fr)
    g = pc.grid(128, 72, 0.0, "cpu")
    side = ((g["u"] * W - 33).abs() < 1.5) & ((g["v"] * H - 36).abs() < 1.5)      # right of a window: pulled left into it
    above = ((g["u"] * W - 24).abs() < 3) & ((g["v"] * H - 24).abs() < 1.5)       # above a window: pulled down into it
    far = (g["v"] * H) > 60
    check("regions attract: the points near a window stream into it, the far ones hold",
          float((Qr - P0)[above][:, 1].mean()) < -0.01 and float((Qr - P0)[side][:, 0].mean()) < -0.01
          and float((Qr - P0)[far].norm(dim=-1).max()) < 1e-6, f"{(Qr - P0)[above][:, 1].mean()} {(Qr - P0)[side][:, 0].mean()}")
    Qw, *_ = pc.points(pc.look(dict(lkr, where="forces", flow=1.0, regions={"windows": "curl+"})), 1.2, col, flat, W / H,
                       forces=pc.force_field(pc.look(dict(lkr, regions={"windows": "curl+"})), W / H, reg))
    mw_ = ((Qw - P0).norm(dim=-1) > 1e-5).reshape(72, 128)
    check("where = forces: the flow acts only near the curves and regions", float(mw_[33:39, 60:68].float().mean()) > 0.5
          and not bool(mw_[60:].any()))
    out_r = run(regions=torch.from_numpy((lab >= 0) & (lab < 3)).float()[None], look=json.dumps({"density": 128}))
    meta_r = out_r.ui["kubakub_pointcloud"][0]
    rep_r = (out_r.args if hasattr(out_r, "args") else out_r.result)[4]
    check("the regions input: a mask's parts are regions, they attract, the node gets their basis", meta_r["regions"]["names"] == ["mask"]
          and meta_r["regions"]["counts"] == [3] and meta_r["regions"]["basis"]["filename"].endswith(".png") and "mask attract" in rep_r, rep_r)
    Pc, *_ = pc.points(pc.look(dict(STILL, curl=1.0)), 1.0, col, flat, W / H)
    Pc2, *_ = pc.points(pc.look(dict(STILL, curl=1.0)), 1.5, col, flat, W / H)
    check("curl sways the cloud on loops", 0.001 < float((Pc - Pc2).norm(dim=-1).mean()) < 0.05)

    # --- the flow: born at its place, flows smoothly, fades, is born again; where it acts -----------------------
    FL = {"density": 128, "jitter": 0, "drift": 0, "flow": 1.0, "flowing": 1.0, "life": 2.0, "speed": 1.0, "shiver": 0}
    lkf = pc.look(FL)
    g = pc.grid(128, 72, 0.0, "cpu")
    t_b = float((1.0 - g["r4"][0]) * 2.0)           # point 0 is born again at t_b (its age is 0)
    Pb, _c, _d, _a, fb = pc.points(lkf, t_b + 1e-4, col, flat, W / H)
    check("a flowing point is born at its own place, unseen", float((Pb[0] - P0[0]).norm()) < 1e-3 and float(fb[0]) < 0.05)
    steps = [pc.points(lkf, 0.7 + k / 24, col, flat, W / H) for k in range(3)]
    v1, v2 = steps[1][0] - steps[0][0], steps[2][0] - steps[1][0]
    cont = steps[0][3] & steps[1][3] & steps[2][3] & (v1.norm(dim=-1) > 1e-4) & (v1.norm(dim=-1) < 0.05) & (v2.norm(dim=-1) < 0.05)
    turn = torch.nn.functional.cosine_similarity(v1[cont], v2[cont], dim=-1)
    check("the flow is smooth: from frame to frame a point keeps its way (no jitter)", float(turn.median()) > 0.98
          and float(cont.float().mean()) > 0.8, f"{float(turn.median())} {float(cont.float().mean())}")
    nb = torch.cdist(P0[:300, :2], P0[:, :2]).topk(2, largest=False).indices[:, 1]
    both = (v1[:300].norm(dim=-1) > 1e-4) & (v1[nb].norm(dim=-1) > 1e-4)
    same = torch.nn.functional.cosine_similarity(v1[:300][both], v1[nb][both], dim=-1)
    check("neighbours flow the same way (a liquid, not noise)", float(same.median()) > 0.6, str(float(same.median())))
    _P, _c, _d, ad, _f = pc.points(lkf, 0.7, col, flat, W / H, now=0.7 + 1.0)
    check("a trail's earlier point of another life is not drawn", 0 < int(ad.sum()) < len(ad))
    lkh = pc.look(dict(FL, flowing=0.3))
    Ph, *_ = pc.points(lkh, 1.3, col, flat, W / H)
    moved = (Ph - P0).norm(dim=-1) > 1e-5
    check("'of points': only that share flows, the rest holds the form", 0.2 < float(moved.float().mean()) < 0.4, str(float(moved.float().mean())))
    fld = torch.zeros((1, H, W))
    fld[:, :, : W // 2] = 1.0
    Pz, *_ = pc.points(pc.look(dict(FL, where="field")), 1.3, col, flat, W / H, field=fld)
    mz = ((Pz - P0).norm(dim=-1) > 1e-5).reshape(72, 128)
    check("where = field: only the white part flows", float(mz[:, :50].float().mean()) > 0.5 and float(mz[:, 80:].float().mean()) < 0.02)
    lkt = pc.look(dict(FL, where="tracked", reach=0.15))
    Pt, *_ = pc.points(lkt, 1.3, col, flat, W / H, tracked=[[0.25, 0.5]])
    mt = ((Pt - P0).norm(dim=-1) > 1e-5).reshape(72, 128)
    Pn, *_ = pc.points(lkt, 1.3, col, flat, W / H)
    check("where = tracked: only near the tracked things (none = nothing flows)", float(mt[30:42, 26:38].float().mean()) > 0.5
          and not bool(mt[:, 100:].any()) and float((Pn - P0).norm(dim=-1).max()) < 1e-6)
    lk_t = pc.look({"density": 128, "flow": 1.0, "flowing": 1.0, "trails": 1.0, "trail_share": 0.25})
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
    LOUD = {"burst": 0.5, "swell": 0.3, "sparkle": 1.0, "spectrum": 0, "sweep": 0}
    lk_s = pc.look({"density": 128, "drift": 0, "flow": 0, "shiver": 0, "sound": LOUD})
    Ps, cs, *_ = pc.points(lk_s, 0.0, grey, flat, W / H, snd=beat)
    Pq, cq, *_ = pc.points(lk_s, 0.0, grey, flat, W / H)
    check("sound: the beat bursts, the bass swells, the highs sparkle", float((Ps - Pq).norm(dim=-1).mean()) > 0.01
          and float(cs.mean()) > float(cq.mean()))
    lk_b = pc.look({"density": 128, "drift": 0, "flow": 0, "shiver": 0, "where": "field", "sound": dict(LOUD, swell=0, sparkle=0)})
    Pb1, *_ = pc.points(lk_b, 0.0, grey, flat, W / H, snd=beat, field=fld)
    Pb0, *_ = pc.points(lk_b, 0.0, grey, flat, W / H, field=fld)
    hit = ((Pb1 - Pb0).norm(dim=-1) > 1e-5).reshape(72, 128)
    check("the beat hits only where the forces act (where = field)", float(hit[:, :50].float().mean()) > 0.9 and not bool(hit[:, 80:].any()))
    # --- the fine motion: every point on its own (his note: subtle, not big groups) and the song's bands ------------
    lk_v = pc.look(dict(STILL, shiver=1.0, speed=1.0))
    Pv1, *_ = pc.points(lk_v, 1.0, col, flat, W / H)
    Pv2, *_ = pc.points(lk_v, 1.05, col, flat, W / H)
    mv = Pv2 - Pv1
    ray0 = P0 / P0.norm(dim=-1, keepdim=True)
    side = (mv - ray0 * (mv * ray0).sum(-1, keepdim=True)).norm(dim=-1)
    nbv = torch.cdist(P0[:300, :2], P0[:, :2]).topk(2, largest=False).indices[:, 1]
    same_v = torch.nn.functional.cosine_similarity(mv[:300], mv[nbv], dim=-1)
    check("shiver: each point trembles along its own line of sight, small, neighbours not together",
          float(side.max()) < 1e-5 and 0 < float(mv.norm(dim=-1).max()) < 4.5 / 72 and abs(float(same_v.mean())) < 0.3,
          f"{float(side.max())} {float(mv.norm(dim=-1).max())} {float(same_v.mean())}")
    lk_d = pc.look(dict(STILL, dropout=0.4, speed=1.0))
    al_d = [pc.points(lk_d, t_, col, flat, W / H)[3] for t_ in (0.0, 0.1, 0.6)]
    check("dropout: that share blinks out, other points at other times",
          all(0.33 < 1 - float(a.float().mean()) < 0.47 for a in al_d) and bool((al_d[0] != al_d[2]).any()),
          str([1 - float(a.float().mean()) for a in al_d]))
    Pr, *_ = pc.points(pc.look(dict(STILL, rows=1.0)), 0.0, col, flat, W / H)
    check("scan lines: the points sit on rows (every ROW points)", len(torch.unique(Pr[:, 1].round(decimals=5))) == 72 // pc.ROW)
    bands = {"level": 0.0, "low": 0.0, "mid": 0.0, "high": 0.0, "beat": 0.0, "bands": [1.0, 0, 0, 0, 0, 0, 0, 0]}
    lk_sp = pc.look(dict(STILL, depth=1.0, sound={"spectrum": 1.0, "sweep": 0, "swell": 0, "sparkle": 0}))
    ramp = torch.linspace(0, 1, W).expand(1, H, W).contiguous()     # far on the left .. near on the right
    Q0, c0_, *_ = pc.points(lk_sp, 0.0, col, ramp, W / H)
    Q1, c1_, *_ = pc.points(lk_sp, 0.0, col, ramp, W / H, snd=bands)
    gF = pc.grid(128, 72, 0.0, "cpu")
    lft, rgt = gF["u"] < 0.08, gF["u"] > 0.35                       # the bass: the farthest slice only
    moved_sp = (Q1 - Q0).norm(dim=-1)
    check("spectrum: the bass moves and lights the far slice only", float(moved_sp[lft].mean()) > 1e-3 and float(moved_sp[rgt].max()) < 1e-6
          and float((c1_ - c0_)[lft].mean()) > 0.05, f"{float(moved_sp[lft].mean())} {float(moved_sp[rgt].max())}")
    hz_ = 22050
    tone = np.concatenate([np.sin(2 * np.pi * 60 * np.arange(hz_) / hz_), np.sin(2 * np.pi * 6000 * np.arange(hz_) / hz_)]).astype(np.float32)
    spc = pc.spectrum(tone, hz_, 10, 20)
    check("the song's bands: a low tone lights the lowest band, a high one a high band",
          spc[5][0] > 0.8 and spc[5][-1] < 0.2 and spc[15][0] < 0.5 and spc[15][-1] > 0.6, f"{spc[5]} {spc[15]}")
    spl = pc.spectrum(tone[:hz_ // 2], hz_, 10, 40)                 # the render goes on 3.5 s after the sound ends
    check("a render longer than its sound: the bands go quiet after its end", len(spl) == 40 and max(spl[-1]) < 0.05, str(spl[-1]))
    clip_s = torch.from_numpy(__import__("kubapack.kubakub.sound", fromlist=["x"]).sample_sound(3.0, 120, 22050))[None, None]
    out = run(audio={"waveform": clip_s, "sample_rate": 22050}, seconds=0.0)
    images, alpha, fps, ply, report, audio_out = out.args if hasattr(out, "args") else out.result
    meta = out.ui["kubakub_pointcloud"][0]
    check("a sound: a picture lasts as long as it, plays on the node, goes on to 'audio'", images.shape[0] == 72
          and len(meta["sound"]["beat"]) == 72 and max(meta["sound"]["beat"]) > 0.5 and len(meta["sound"]["bands"]) == 72
          and len(meta["sound"]["bands"][0]) == pc.BANDS and meta["audio"]["filename"].endswith(".wav")
          and audio_out["waveform"].shape[-1] == 3 * 22050, report)

    # --- the live view computes the same (node) ------------------------------------------------------
    node = shutil.which("node")
    if node is None:
        print("skip the JavaScript checks: node is not installed")
    else:
        for f in ("kubakub_pointcloud.js", "kubakub_pointcloud_math.js"):    # parsed as a module, as the browser does
            with open(os.path.join(HERE, "..", "web", f), encoding="utf-8") as fh:
                res = subprocess.run([node, "--check", "--input-type=module"], stdin=fh, capture_output=True, text=True,
                                     encoding="utf-8", errors="replace", timeout=60)
            check(f"{f} parses as a module", res.returncode == 0, res.stderr[-400:])
        rng = np.random.default_rng(3)
        idx = [int(v) for v in rng.integers(0, 4_000_000, 300)]
        cams = [{"yaw": float(rng.uniform(-90, 90)), "pitch": float(rng.uniform(-60, 60)), "dolly": float(rng.uniform(-1, 0.9)),
                 "x": float(rng.uniform(-1, 1)), "y": float(rng.uniform(-1, 1))} for _ in range(20)]
        pts = [[k, float(rng.uniform(0, 1)), float(rng.uniform(0, 1)), float(rng.uniform(0, 1))] for k in range(14)]
        camera = {"start": cams[0], "end": cams[1], "ease": "smooth"}
        shaky = {"start": cams[0], "end": cams[1], "ease": "linear", "shake": 0.7,
                 "path": [[0, 0, 0, 0, 0, 0], [0.4, 20, -5, 0.3, 0.1, 0], [1, -30, 10, 0.5, 0, 0.2]]}
        math_js = Path(HERE, "..", "web", "kubakub_pointcloud_math.js").resolve().as_uri()
        # a look with every kind of curve (a loop too) and two region groups: the node's field = the render's
        circle = [[0.6 + 0.1 * math.cos(a), 0.4 + 0.15 * math.sin(a)] for a in np.linspace(0, 2 * math.pi, 20)]
        FLK = pc.look({"force": 1.3, "reach": 0.11, "curves": [{"kind": k, "pts": [[0.1, 0.2 + 0.1 * n], [0.4, 0.25 + 0.1 * n], [0.7, 0.2 + 0.1 * n]]}
                                                              for n, k in enumerate(pc.CURVE_KINDS)] + [{"kind": "curl-", "pts": circle}],
                       "regions": {"windows": "repel", "roof": "curl+"}})
        FLK["curves"] = FLK["curves"][-pc.MAX_CURVES:]
        REG = pc.region_basis(lab, {"windows": [0, 1, 2], "roof": [3]}, 1.6)
        REGJ = {"names": REG["names"], "default": REG["default"], "q": [q.flatten().tolist() for q in REG["q"]]}
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
               "out.tp = [1, 3, 10, 33].map(s => idx.slice(0, 40).map(b => M.trailPoint(b % 90000, s, 3000000)));\n"
               "out.flow = [M.WAVES, M.FLOW_STEPS, M.FINE, M.MAX_TRACKED, M.WHERE, M.BANDS, M.ROW, M.BLINK];\n"
               f"const FLK = {json.dumps(FLK)}, REG = {json.dumps(REGJ)};\n"
               "REG.q = REG.q.map(a => Uint8Array.from(a));\n"
               "const ff = M.forceField(FLK, 1.6, REG); out.ff = [ff.fw, ff.fh, Array.from(ff.data)];\n"
               "out.forces = [M.CURVE_KINDS, M.REGION_KINDS, M.MAX_CURVES, M.MAX_CURVE_PTS, M.FIELD_H, M.CLOSED, M.PULL, M.CURL_SCALE, M.CURL_SPEED, M.BASIS];\n"
               "out.curves = M.curvesOf({ forces: [[0.1, 0.2, 0.3, 0.4]], curves: [{ kind: 'repel', pts: [[0, 0], [2, 0.5]] }, { kind: 'x', pts: [[0, 0], [1, 1]] }] });\n"
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
            check("trail points: one by a hash in every block", js["tp"] == [[int(pc.trail_points(3000000, st)[b % 90000]) for b in idx[:40]]
                                                                             for st in [1, 3, 10, 33]])
            check("the flow's numbers (waves, steps, fine, tracked, where, bands, rows, blink)", js["flow"] == [[list(w) for w in pc.WAVES], pc.FLOW_STEPS, pc.FINE,
                                                                                         pc.MAX_TRACKED, list(pc.WHERE), pc.BANDS, pc.ROW, pc.BLINK])
            ffp = pc.curve_field(FLK, 1.6)
            pc.region_field(FLK, REG, ffp)
            fw_, fh_ = pc.field_size(1.6)
            check("the force field (every kind of curve, a loop, regions) is the same on the node",
                  js["ff"][:2] == [fw_, fh_] and np.allclose(np.array(js["ff"][2]).reshape(fh_, fw_, 4), ffp.transpose(1, 2, 0), atol=1e-5),
                  str(np.abs(np.array(js["ff"][2]).reshape(fh_, fw_, 4) - ffp.transpose(1, 2, 0)).max()))
            check("the forces' numbers", js["forces"] == [list(pc.CURVE_KINDS), list(pc.REGION_KINDS), pc.MAX_CURVES, pc.MAX_CURVE_PTS, pc.FIELD_H,
                                                          pc.CLOSED, pc.PULL, pc.CURL_SCALE, pc.CURL_SPEED, [list(b) for b in pc.BASIS]])
            check("older arrows and wrong curves read the same way", js["curves"] == pc.curves_of({"forces": [[0.1, 0.2, 0.3, 0.4]],
                  "curves": [{"kind": "repel", "pts": [[0, 0], [2, 0.5]]}, {"kind": "x", "pts": [[0, 0], [1, 1]]}]}))
            check("trail strides", js["stride"] == [pc.trail_stride(pc.look({"trail_share": s}))
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
