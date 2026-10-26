"""kubakub point cloud: a picture or a video lifted into points by its depth (from any depth node), moved by formulas
and filmed with a camera move; tracked things from kubakub tracker sit in the cloud at their depth.

The node shows the cloud live (web/kubakub_pointcloud.js, WebGL2): drag to look around, set the start and end of
the camera move from the view, change the look while it plays. The render (kubakub/pointcloud.py, torch) draws the
same points with the same formulas at full size.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time

import numpy as np
import torch

import comfy.utils
import folder_paths
from comfy_api.latest import io

from ...kubakub import pointcloud as pc
from ...kubakub import tracking as tk
from ...kubakub.io_types import TracksType
from ..utils.nodes_compare import frames_of, write_side

log = logging.getLogger("KUBA.regions")

CATEGORY = "kubakub/2d/motion"
PREVIEW_SIDE = 1280
PICTURES_UP_TO = 16                                # a batch of up to this many pictures is a set to morph, more is a video
SECONDS_PER_PICTURE = 3.0


def _device():
    try:
        import comfy.model_management as mm
        return mm.get_torch_device()
    except Exception:  # noqa: BLE001
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _save_json(data, folder) -> dict:
    text = json.dumps(data, separators=(",", ":"))
    name = hashlib.sha1(text.encode()).hexdigest()[:20] + ".json"
    out = os.path.join(folder_paths.get_temp_directory(), folder)
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, name)
    if not os.path.isfile(path):
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
    return {"filename": name, "subfolder": folder, "type": "temp"}


def ply_path():
    """output/kubakub/pointcloud_00001.ply, the next free number."""
    folder = os.path.join(folder_paths.get_output_directory(), "kubakub")
    os.makedirs(folder, exist_ok=True)
    k = 1
    while os.path.exists(os.path.join(folder, f"pointcloud_{k:05d}.ply")):
        k += 1
    return os.path.join(folder, f"pointcloud_{k:05d}.ply")


def out_size(W, H, long_side):
    if long_side <= 0:
        return W, H
    k = long_side / max(W, H)
    return max(2, int(round(W * k / 2)) * 2), max(2, int(round(H * k / 2)) * 2)


def timeline(mode, n, K, seconds, fps):
    """(frames to render, their length in seconds): a video its own length (or `seconds`, looping), pictures
    3 s each (a single picture 4 s) unless `seconds` says."""
    if mode == "video":
        T = seconds if seconds > 0 else n / fps
    else:
        T = seconds if seconds > 0 else (4.0 if K == 1 else SECONDS_PER_PICTURE * K)
    n_out = max(1, int(round(T * fps)))
    return n_out, n_out / fps


class KUBA_PointCloud(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="KUBA_PointCloud",
            display_name="kubakub point cloud",
            category=CATEGORY,
            search_aliases=["point cloud", "points", "particles", "depth", "3d photo", "parallax", "gather", "ply", "splat"],
            description=("Lifts a picture or a video into up to two million points by its depth and films them with "
                         "a camera move. Connect the depth of any depth node (Depth Anything 3 / 2, Marigold ...). On "
                         "the node: drag to look around (Shift-drag moves, the wheel goes nearer), 'start = view' and "
                         "'end = view' set the camera move, and the look (colours, glow / ink / solid, depth, point "
                         "size, drift, breathe, wind, a scan through the depth) changes live while it plays. A batch of "
                         "pictures morphs: every point travels to the point of the next picture with the same "
                         "brightness. 'field' (a mask, e.g. the tracker's 'near') pushes points out; the tracker's "
                         "tracks sit in the cloud at their depth, joined by lines. Nothing connected: a sample facade."),
            is_output_node=True,                       # runs on its own, to see the cloud on the node
            inputs=[
                io.MultiType.Input("image", types=[io.Image, io.Video], optional=True,
                                   tooltip="A picture, a batch of pictures (they morph into each other) or a video. "
                                           "Nothing connected: a sample facade with its own depth."),
                io.Image.Input("depth", optional=True,
                               tooltip="The depth of the same picture(s) / video from any depth node; which side is "
                                       "near: the pill on the node. Nothing connected: a soft relief from the "
                                       "brightness (connect a depth node for real depth)."),
                io.Mask.Input("field", optional=True,
                              tooltip="Pushes the points out where it is white, frame by frame: the tracker's 'near' "
                                      "(the points burst around what moves), kubakub mask animate, a shader's mask. "
                                      "How far: 'push' on the node."),
                TracksType.Input("tracks", optional=True,
                                 tooltip="From kubakub tracker on the same video: each tracked thing gets a ring at its "
                                         "place in the cloud, joined by lines (network, tree, within a distance)."),
                io.String.Input("look", default="", tooltip="Set on the node; never typed."),
                io.Float.Input("seconds", default=0.0, min=0.0, max=600.0, step=0.5,
                               tooltip="How long the render is. 0 = a video's own length, 3 s per picture (one picture: 4 s). "
                                       "Longer than the video: it loops."),
                io.Float.Input("fps", default=24.0, min=1.0, max=120.0, step=1.0, advanced=True,
                               tooltip="Frames per second of the render for pictures (a video brings its own)."),
                io.Int.Input("long_side", default=0, min=0, max=8192, step=16, advanced=True,
                             tooltip="The render's long side in pixels; 0 = the picture's own size."),
                io.Combo.Input("batch", options=["auto", "video", "pictures"], default="auto", advanced=True,
                               tooltip=f"auto: a video input is a video, an image batch of up to {PICTURES_UP_TO} is pictures "
                                       "to morph, more is a video."),
                io.Boolean.Input("save_ply", default=False,
                                 tooltip="Also save the first frame's points as a .ply (output/kubakub/), for Blender, "
                                         "Houdini, MeshLab, TouchDesigner ..."),
            ],
            outputs=[
                io.Image.Output("images", tooltip="The filmed cloud, every frame."),
                io.Mask.Output("alpha", tooltip="Where points are (for putting the cloud over something)."),
                io.Float.Output("fps", tooltip="For Create Video."),
                io.String.Output("ply", tooltip="The saved .ply's path ('save_ply' on), else empty."),
                io.String.Output("report"),
            ],
        )

    @classmethod
    def execute(cls, image=None, depth=None, field=None, tracks=None, look="", seconds=0.0, fps=24.0, long_side=0,
                batch="auto", save_ply=False) -> io.NodeOutput:
        t0 = time.perf_counter()
        lk = pc.look(look)
        notes = []
        if image is None:
            img, d = pc.sample_picture()
            frames, kind, rate = torch.from_numpy(img)[None], "image", None
            dz = d[None]
            depth_from = "sample"
            notes.append("nothing connected: the sample facade and its own depth")
        else:
            frames, rate, kind = frames_of(image)
            frames = frames[..., :3].float()
            dz, depth_from = None, "light"
        n, H, W = int(frames.shape[0]), int(frames.shape[1]), int(frames.shape[2])
        mode = batch if batch in ("video", "pictures") else ("video" if kind == "video" or n > PICTURES_UP_TO else "pictures")
        if mode == "pictures" and n > 64:
            raise ValueError(f"kubakub point cloud: {n} pictures to morph is too many (64 at most); for a video set 'batch' to video.")
        fps = float(rate or fps)
        if depth is not None:
            dz = tk.depth_frames(depth, n, H, W, lk["near"])
            depth_from = "node"
            if int(depth.shape[0]) < n:
                notes.append(f"depth has {int(depth.shape[0])} frame(s) for {n}: the last one holds")
        elif dz is None:
            dz = pc.depth_from_light(frames.cpu().numpy())
            notes.append("no depth connected: a soft relief from the brightness (connect a depth node for real depth)")
        K = n if mode == "pictures" else 1
        n_out, T = timeline(mode, n, K, float(seconds), fps)
        Wo, Ho = out_size(W, H, int(long_side))
        aspect = W / H
        dev = _device()
        gw, gh = pc.grid_size(lk["density"], W, H)

        def col(k):
            return frames[k].to(dev).permute(2, 0, 1).contiguous()

        def dep(k):
            return torch.from_numpy(np.ascontiguousarray(dz[min(k, len(dz) - 1)])).to(dev)[None]
        fld = None
        if field is not None:
            fld = field.cpu().float()
            fld = fld[None] if fld.ndim == 2 else fld
            if fld.ndim == 4:
                fld = fld[..., 0]
        perms = []
        if mode == "pictures" and K > 1:
            i = torch.arange(gw * gh, device=dev, dtype=torch.int64)
            u, v = pc.grid_uv(i, gw, gh, lk["jitter"])
            luma = torch.tensor(pc.LUMA, device=dev)
            lums = [(pc.sample(col(k), u, v)[:, :3] * luma).sum(-1) for k in range(K)]
            perms = [pc.rank_perm(lums[k], lums[k + 1]) for k in range(K - 1)]
        tracks_ok = isinstance(tracks, dict) and tracks.get("format") == tk.TRACKS_FORMAT
        if tracks is not None and not tracks_ok:
            raise ValueError("kubakub point cloud: 'tracks' must come from kubakub tracker.")
        t3 = None
        if tracks_ok:
            tn = int(tracks["n"])
            if tn != n and mode == "video":
                notes.append(f"the tracks are of a video with {tn} frames, this one has {n}: they are matched frame by frame")

            def dep_at(fr):
                a = dz[min(fr, len(dz) - 1)]
                return lambda uu, vv: a[min(H - 1, max(0, int(vv * H))), min(W - 1, max(0, int(uu * W)))]
            t3 = [pc.track_points(tracks, f, dep_at(min(f, n - 1) if mode == "video" else 0)) for f in range(tn)]
        pbar = comfy.utils.ProgressBar(n_out)
        out = torch.empty((n_out, Ho, Wo, 3), dtype=torch.float32)
        alpha = torch.empty((n_out, Ho, Wo), dtype=torch.float32)
        ply = ""
        cache = {}
        try:
            for f in range(n_out):
                t = f / fps
                a = f / (n_out - 1) if n_out > 1 else 0.0
                if mode == "video":
                    src = f % n
                    ca, da, cb, db, perm, s = col(src), dep(src), None, None, None, 0.0
                else:
                    k, s = pc.morph_at(t, T, K)
                    if k not in cache:
                        cache.clear()
                        cache[k] = (col(k), dep(k))
                    ca, da = cache[k]
                    cb, db, perm = (col(k + 1), dep(k + 1), perms[k]) if s > 0 else (None, None, None)
                ff = fld[min(f, len(fld) - 1)].to(dev)[None] if fld is not None else None
                P, c, dist, alive = pc.points(lk, t, ca, da, aspect, cb, db, perm, s, ff, device=dev)
                if f == 0 and save_ply:
                    ply = pc.write_ply(ply_path(), P[alive], c[alive])
                cam = pc.basis(pc.camera_at(lk, a), lk["depth"])
                im, al = pc.splat(P, c, dist, alive, lk, cam, Wo, Ho, gh)
                if t3 is not None:
                    tf = f % len(t3) if mode == "pictures" else min(f % n, len(t3) - 1)
                    im = pc.draw_tracks(im, t3[tf], lk, cam, aspect, lk["depth"])
                out[f].copy_(im)
                alpha[f].copy_(al)
                pbar.update(1)
        finally:
            try:
                import comfy.model_management as mm
                mm.soft_empty_cache()
            except Exception:  # noqa: BLE001
                pass
        meta = {"mode": mode, "n": n, "K": K, "W": W, "H": H, "fps": fps, "n_out": n_out, "T": T,
                "depth_from": depth_from, "near": lk["near"],
                "colour": write_side(frames, "c", PREVIEW_SIDE),
                "depth": write_side(torch.from_numpy(np.asarray(dz, np.float32))[..., None].expand(-1, -1, -1, 3), "d", PREVIEW_SIDE)}
        if fld is not None:
            meta["field"] = write_side(fld[..., None].expand(-1, -1, -1, 3), "f", PREVIEW_SIDE)
        if t3 is not None:
            meta["tracks"] = _save_json({"frames": t3}, "kubakub_pointcloud")
        report = "\n".join([f"{gw * gh:,} points ({gw} x {gh}), {mode}: {n} {'frame(s)' if mode == 'video' else 'picture(s)'} "
                            f"{W}x{H}; rendered {n_out} frames {Wo}x{Ho} at {fps:g} fps in {time.perf_counter() - t0:.1f} s",
                            f"look: {lk['colour']} colours, {lk['blend']}, depth {lk['depth']:g}"
                            + (f", {len(t3[0]) if t3 and t3[0] else 0} tracked things in the first frame" if t3 is not None else ""),
                            *notes, *([f"saved {ply}"] if ply else [])])
        log.info("[KUBA point cloud] %s", report.replace("\n", " | "))
        return io.NodeOutput(out, alpha, fps, ply, report, ui={"kubakub_pointcloud": [meta]})


def _register_routes():
    """GET /kubakub/pointcloud/defaults -> the default look (the node's live view starts from it)."""
    try:
        from aiohttp import web
        from server import PromptServer
    except ImportError:
        return

    if getattr(PromptServer, "instance", None) is None:          # imported without a running server (tests)
        return

    async def defaults(request):
        return web.json_response(pc.DEFAULT_LOOK)
    try:
        PromptServer.instance.routes.get("/kubakub/pointcloud/defaults")(defaults)
    except (AttributeError, RuntimeError) as e:
        log.warning("[KUBA point cloud] route not registered: %s", e)


_register_routes()

NODE_CLASS_MAPPINGS = {"KUBA_PointCloud": KUBA_PointCloud}
NODE_DISPLAY_NAME_MAPPINGS = {"KUBA_PointCloud": "kubakub point cloud"}
