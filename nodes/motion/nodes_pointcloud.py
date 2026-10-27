"""kubakub point cloud: a picture or a video lifted into points by its depth (from any depth node), moved by formulas
and filmed with a camera move; tracked things from kubakub tracker sit in the cloud at their depth.

The node shows the cloud live (web/kubakub_pointcloud.js, WebGL2): drag to look around, set the start and end of
the camera move from the view, change the look while it plays. The render (kubakub/pointcloud.py, torch) draws the
same points with the same formulas at full size.
"""

from __future__ import annotations

import hashlib
import logging
import os
import time

import numpy as np
import torch

import comfy.model_management as mm
import comfy.utils
import folder_paths
from comfy_api.latest import io

from ...kubakub import maskfields as mf
from ...kubakub import pointcloud as pc
from ...kubakub import shader_render as sr
from ...kubakub import sound as so
from ...kubakub import tracking as tk
from ...kubakub.io_types import RegionsType, SoundType, TracksType
from ..shader.nodes_shader import _ram_check
from ..utils.nodes_compare import frames_of, free_vram, save_json, save_wav, temp_file, write_side

log = logging.getLogger("KUBA.regions")

CATEGORY = "kubakub/2d/motion"
PREVIEW_SIDE = 1280
PICTURES_UP_TO = 16                                # a batch of up to this many pictures is a set to morph, more is a video
SECONDS_PER_PICTURE = 3.0


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


def regions_of(regions, notes):
    """The regions input -> pointcloud.region_basis(): a kubakub regions node (its groups), or a mask / mask batch
    (every separate white part is a region, all in one group 'mask'; specks are left out)."""
    if isinstance(regions, torch.Tensor):
        m = regions.float().cpu()
        lab, n = mf.from_masks((m if m.ndim == 2 else m.amax(0)).numpy())
        return lab, {"mask": list(range(n))}
    if int(regions.labels.shape[0]) > 1:
        notes.append("regions: the first frame's regions are used")
    return regions.labels[0].cpu().numpy(), regions.table.get("groups") or {}


def basis_file(reg):
    """The regions' basis as one PNG in temp for the node's live view: the groups one under the other, each two cells
    wide (channels 0-2 left, 3-5 right)."""
    import cv2
    q = reg["q"]
    G, fh, fw, _ = q.shape
    img = np.concatenate([q[..., :3], q[..., 3:]], 2).reshape(G * fh, 2 * fw, 3)

    def write(path):                                 # encoded here: imwrite fails silently on a path with non-ASCII letters
        ok, buf = cv2.imencode(".png", np.ascontiguousarray(img[..., ::-1]))
        if not ok:
            raise RuntimeError("the regions' basis could not be written as a PNG")
        buf.tofile(path)
    return temp_file("kubakub_pointcloud", hashlib.sha1(img.tobytes()).hexdigest()[:20] + ".png", write)


def timeline(mode, n, K, seconds, fps, sound_secs=0.0):
    """(frames to render, their length in seconds): a video its own length (or `seconds`, looping), pictures
    as long as the sound, else 3 s each (a single picture 4 s), unless `seconds` says."""
    if mode == "video":
        T = seconds if seconds > 0 else n / fps
    else:
        T = seconds if seconds > 0 else (min(sound_secs, 600.0) if sound_secs > 0 else 4.0 if K == 1 else SECONDS_PER_PICTURE * K)
    n_out = max(1, int(round(T * fps)))
    return n_out, n_out / fps


class KUBA_PointCloud(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="KUBA_PointCloud",
            display_name="kubakub pointclouder",
            category=CATEGORY,
            search_aliases=["point cloud", "pointcloud", "points", "particles", "depth", "3d photo", "parallax", "gather",
                            "ply", "splat", "trails", "audio reactive"],
            description=("Lifts a picture or a video into up to two million points by its depth and films them with "
                         "a camera move. Connect the depth of any depth node (Depth Anything 3 / 2, Marigold ...). On "
                         "the node: drag to look around (Shift-drag moves, the wheel goes nearer), set the move from "
                         "the view, pick a camera (push in, orbit, handheld ...) or record your own; the look (colours, "
                         "glow / ink / solid, depth, fine points, a flow that streams parts of the cloud along smooth "
                         "swirls: everywhere, in a mask, near tracked things or near your forces; trails, breathe, "
                         "wind, curl, a scan through the depth) changes live while it plays. Forces: draw curves with "
                         "the mouse (wind along them, attract, repel, curl either way) and connect regions (all windows "
                         "attract ...). A batch of pictures "
                         "morphs: every point travels to the point of the next picture with the same brightness. "
                         "'field' (a mask, e.g. the tracker's 'near') pushes points out; the tracker's tracks sit in the "
                         "cloud at their depth; a sound makes the flowing parts burst on the beat, the cloud swell with "
                         "the bass. Nothing connected: a sample facade."),
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
                                      "How far: 'push' on the node. With 'where' = field the flow and the beat act "
                                      "only there."),
                io.MultiType.Input("regions", types=[RegionsType, io.Mask], optional=True,
                                   tooltip="From any kubakub regions node (regions from matrix / masks / sam3 ...), or a "
                                           "mask: its groups become forces, set on the node per group (attract, repel, "
                                           "curl + / -). Connected as it is, the group with the most regions (the windows) "
                                           "attracts the flowing points. A mask: every white part is a region."),
                TracksType.Input("tracks", optional=True,
                                 tooltip="From kubakub tracker on the same video: each tracked thing gets a ring at its "
                                         "place in the cloud, joined by lines (network, tree, within a distance). "
                                         "With 'where' = tracked the flow and the beat act only near them."),
                io.Audio.Input("audio", optional=True,
                               tooltip="A sound the cloud follows: the points burst out on the beat, swell with the "
                                       "bass, move more when it is loud (how much: 'sound' on the node). It plays with "
                                       "the cloud on the node and goes on to 'audio'. Pictures last as long as it."),
                SoundType.Input("sound", optional=True,
                                tooltip="From kubakub sound: the sound and how it is heard (tempo, gain, release), set "
                                        "once for every node that listens. Connected, it comes before 'audio'."),
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
                io.Audio.Output("audio", tooltip="The sound, cut to the render (silence without one): for Create Video."),
            ],
        )

    @classmethod
    def execute(cls, image=None, depth=None, field=None, tracks=None, audio=None, sound=None, look="", seconds=0.0,
                fps=24.0, long_side=0, batch="auto", save_ply=False, regions=None) -> io.NodeOutput:
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
        wave, hz, mono, heard = so.take_wave(sound, audio)
        n_out, T = timeline(mode, n, K, float(seconds), fps, len(mono) / hz if mono is not None else 0.0)
        per_frame = None
        if mono is not None:
            per_frame, lines = sr.frame_sound(mono, hz, heard, fps, n_out)
            per_frame["bands"] = pc.spectrum(mono, hz, fps, n_out)
            notes.append("sound: " + "; ".join(lines))
        Wo, Ho = out_size(W, H, int(long_side))
        aspect = W / H
        dev = mm.get_torch_device()
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
            g = pc.grid(gw, gh, lk["jitter"], dev)
            u, v = g["u"], g["v"]
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
        _ram_check("kubakub pointclouder", n_out, Wo, Ho, 16)   # images + alpha, float32
        pbar = comfy.utils.ProgressBar(n_out)
        out = torch.empty((n_out, Ho, Wo, 3), dtype=torch.float32)
        alpha = torch.empty((n_out, Ho, Wo), dtype=torch.float32)
        ply = ""
        pics = {}                                  # pictures on the card: the one shown and the next one it morphs into

        def pic(j):
            if j not in pics:
                for old in [x for x in pics if x < j - 1]:
                    del pics[old]
                pics[j] = (col(j), dep(j))
            return pics[j]
        fld_one = fld[0].to(dev)[None] if fld is not None and len(fld) == 1 else None
        reg = None
        if regions is not None:
            reg = pc.region_basis(*regions_of(regions, notes), aspect)
            kinds = pc.region_kinds(lk, reg)
            notes.append(f"regions: {len(reg['names'])} group(s); " + (", ".join(f"{k} {v}" for k, v in kinds.items()) or "none acts (set them on the node)"))
        forces = pc.force_field(lk, aspect, reg, dev)
        try:
            for f in range(n_out):
                t = f / fps
                a = f / (n_out - 1) if n_out > 1 else 0.0
                if mode == "video":
                    src = f % n
                    ca, da, cb, db, perm, s = col(src), dep(src), None, None, None, 0.0
                else:
                    k, s = pc.morph_at(t, T, K)
                    ca, da = pic(k)
                    cb, db, perm = (*pic(k + 1), perms[k]) if s > 0 else (None, None, None)
                ff = fld_one if fld_one is not None else fld[min(f, len(fld) - 1)].to(dev)[None] if fld is not None else None
                snd = {k: v[f] for k, v in per_frame.items()} if per_frame else None
                tf = None if t3 is None else f % len(t3) if mode == "pictures" else min(f % n, len(t3) - 1)
                trk = [p[1:3] for p in t3[tf]] if tf is not None and t3[tf] else None
                fr = dict(col_a=ca, dep_a=da, aspect=aspect, col_b=cb, dep_b=db, perm=perm, s=s, field=ff, device=dev, snd=snd,
                          tracked=trk, forces=forces)
                P, c, dist, alive, fade = pc.points(lk, t, **fr)
                if f == 0 and save_ply:
                    ply = pc.write_ply(ply_path(), P[alive], c[alive])
                cam = pc.basis(pc.camera_at(lk, a, t), lk["depth"])
                lines = pc.trail_lines(lk, t, fr, cam, Wo, Ho) if lk["trails"] > 0 else None
                size = lk["size"] * (1 + lk["sound"]["size"] * snd["level"]) if snd else None
                im, al = pc.splat(P, c, dist, alive, lk, cam, Wo, Ho, gh, lines, size, fade)
                if t3 is not None:
                    im = pc.draw_tracks(im, t3[tf], lk, cam, aspect)
                out[f].copy_(im)
                alpha[f].copy_(al)
                pbar.update(1)
        finally:
            pc.forget()
            free_vram()
        meta = {"mode": mode, "n": n, "K": K, "W": W, "H": H, "fps": fps, "n_out": n_out, "T": T,
                "depth_from": depth_from, "near": lk["near"],
                "colour": write_side(frames, "c", PREVIEW_SIDE),
                "depth": write_side(torch.from_numpy(np.asarray(dz, np.float32))[..., None].expand(-1, -1, -1, 3), "d", PREVIEW_SIDE)}
        if fld is not None:
            meta["field"] = write_side(fld[..., None].expand(-1, -1, -1, 3), "f", PREVIEW_SIDE)
        if t3 is not None:
            meta["tracks"] = save_json({"frames": t3}, "kubakub_pointcloud")
        if reg is not None and reg["names"]:
            fw, fh = pc.field_size(aspect)
            meta["regions"] = {"basis": basis_file(reg), "names": reg["names"], "counts": reg["counts"], "default": reg["default"],
                               "fw": fw, "fh": fh}
        audio_out = so.sound_out(wave, hz, n_out / fps)
        if per_frame:
            meta["sound"] = per_frame
            meta["audio"] = save_wav(audio_out, "kubakub_pointcloud")
        report = "\n".join([f"{gw * gh:,} points ({gw} x {gh}), {mode}: {n} {'frame(s)' if mode == 'video' else 'picture(s)'} "
                            f"{W}x{H}; rendered {n_out} frames {Wo}x{Ho} at {fps:g} fps in {time.perf_counter() - t0:.1f} s",
                            f"look: {lk['colour']} colours, {lk['blend']}, depth {lk['depth']:g}"
                            + (f", {len(t3[0]) if t3 and t3[0] else 0} tracked things in the first frame" if t3 is not None else ""),
                            *notes, *([f"saved {ply}"] if ply else [])])
        log.info("[KUBA point cloud] %s", report.replace("\n", " | "))
        return io.NodeOutput(out, alpha, fps, ply, report, audio_out, ui={"kubakub_pointcloud": [meta]})


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
