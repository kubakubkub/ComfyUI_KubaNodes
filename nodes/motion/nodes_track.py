"""kubakub tracker and kubakub track draw: what moves in a video, kept as tracks, drawn as a diagram.

The tracker (kubakub/tracking.py) runs the finders once and hands on the tracks; its node shows the video with what
was found, and 'try this frame' (POST /kubakub/track/probe) runs the finders on one frame with the settings as they
are on the node. Track draw (kubakub/trackdraw.py) renders the look at full size; its node plays the same look live
in the browser (web/kubakub_track.js) from the tracker's frames and tracks, before anything is queued.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

import comfy.utils
import folder_paths
from comfy_api.latest import io

from ...kubakub import region_cache as rc
from ...kubakub import trackdraw as td
from ...kubakub import shader_render as sr
from ...kubakub import sound as so
from ...kubakub import tracking as tk
from ...kubakub.io_types import SoundType, TracksType
from ..utils.nodes_compare import frames_of, free_vram, save_json, save_wav, write_side

log = logging.getLogger("KUBA.regions")

CATEGORY = "kubakub/2d/motion"
PREVIEW_SIDE = 960
_LAST: dict[str, tuple] = {}                       # node id -> (frames, fps): the frames 'try this frame' looks at
CACHE_VERSION = 2                                  # raise when the analysis itself changes (old cached tracks are then not used)


def _code_print():
    """A fingerprint of the tracking code: tracks made by older code are never taken from the cache."""
    try:
        with open(tk.__file__, "rb") as f:
            return hashlib.sha1(f.read()).hexdigest()[:12]
    except OSError:
        return "?"


CODE = _code_print()


def cache_dir():
    """Tracks of earlier runs, kept across restarts: kuba_tracks in the pack's cache folder (setting cache_folder,
    else user/kubakub_cache), at most cache_gb together with the other caches (least used first)."""
    kst = rc._pack_settings()
    env = (kst.get("cache_folder", "") if kst else "").strip()
    if env.lower() in ("temp", "off", "0"):
        base = folder_paths.get_temp_directory()
    else:
        base = os.path.expandvars(os.path.expanduser(env.strip('"'))) if env else os.path.join(folder_paths.get_user_directory(), "kubakub_cache")
    return os.path.join(base, "kuba_tracks")


def video_key(frames, s, fps):
    """What decides the tracks: every frame (a 32 x 18 sample of each, so a change anywhere counts), the size, the
    fps and the finders' settings (the depth is not in it: it is added after, it is cheap)."""
    n, H, W = int(frames.shape[0]), int(frames.shape[1]), int(frames.shape[2])
    sy, sx = max(1, H // 18), max(1, W // 32)
    h = hashlib.sha256(f"v{CACHE_VERSION}|{CODE}|{n}|{W}|{H}|{float(fps):.4f}|{json.dumps(s, sort_keys=True)}".encode())
    for i in range(0, n, 64):
        part = frames[i:i + 64, ::sy, ::sx, :3]
        part = part.cpu() if hasattr(part, "cpu") else part
        h.update((np.asarray(part, np.float32) * 255).round().astype(np.uint8).tobytes())
    return h.hexdigest()[:32]


def cached_tracks(key):
    path = os.path.join(cache_dir(), key + ".json")
    if not os.path.isfile(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        os.utime(path)                             # used now: kept longest
        return data if data.get("format") == tk.TRACKS_FORMAT else None
    except (OSError, ValueError):
        return None


def keep_tracks(key, data):
    d = cache_dir()
    try:
        os.makedirs(d, exist_ok=True)
        tmp = os.path.join(d, key + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, separators=(",", ":"))
        os.replace(tmp, os.path.join(d, key + ".json"))
        from ...kubakub.scene3d import bridge
        kst = rc._pack_settings()
        cap = int(float(kst.number("cache_gb", 20) if kst else 20) * 2 ** 30)
        bridge.maybe_purge(os.path.dirname(d), cap)
    except Exception as e:  # noqa: BLE001
        log.warning("[KUBA tracker] tracks not kept in the cache: %s", e)


def chain(before, data):
    """The tracks of a tracker before this one + this one's: one set, ids go on after the last one, a kind already
    there gets a number (colour -> colour 2), so track draw can treat them apart and join them with lines."""
    if not isinstance(before, dict) or before.get("format") != tk.TRACKS_FORMAT:
        raise ValueError("kubakub tracker: 'tracks' must come from another kubakub tracker.")
    if (before["n"], before["W"], before["H"]) != (data["n"], data["W"], data["H"]):
        raise ValueError(f"kubakub tracker: the tracker before tracked another video ({before['n']} frames "
                         f"{before['W']}x{before['H']}, this one {data['n']} frames {data['W']}x{data['H']}): chain "
                         "trackers on the same video.")
    out = json.loads(json.dumps(before))
    have = {tr["kind"] for tr in out["tracks"]}
    rename = {}
    for k in sorted({tr["kind"] for tr in data["tracks"]}):
        if k in have:
            i = 2
            while f"{k} {i}" in have:
                i += 1
            rename[k] = f"{k} {i}"
            have.add(rename[k])
    start = max([tr["id"] for tr in out["tracks"]] + [0])
    for tr in data["tracks"]:
        t2 = dict(tr, id=tr["id"] + start, kind=rename.get(tr["kind"], tr["kind"]))
        out["tracks"].append(t2)
    for k, v in data["curves"].items():
        if k == "count":
            out["curves"]["count"] = [a + b for a, b in zip(out["curves"].get("count", [0] * len(v)), v)]
        else:
            out["curves"].setdefault(k, v)
    return out, rename


def _union_masks(data, n, H, W):
    """(n, H, W) float: white where something was found (its outline, else its box)."""
    import cv2
    out = np.zeros((n, H, W), np.float32)
    for tr in data["tracks"]:
        for k, b in enumerate(tr["box"]):
            t = tr["t0"] + k
            if not tr.get("seen", [1])[min(k, len(tr.get("seen", [1])) - 1)]:
                continue
            o = (tr.get("outline") or {}).get(str(t))
            if o and len(o) >= 3:
                m = np.zeros((H, W), np.uint8)
                cv2.fillPoly(m, [np.round(np.asarray(o) * 4).astype(np.int32)], 255, cv2.LINE_AA, shift=2)
                np.maximum(out[t], m.astype(np.float32) / 255, out=out[t])
            else:
                x0, y0 = max(0, int(b[0])), max(0, int(b[1]))
                x1, y1 = min(W, int(np.ceil(b[0] + b[2]))), min(H, int(np.ceil(b[1] + b[3])))
                if x1 > x0 and y1 > y0:
                    out[t, y0:y1, x0:x1] = 1.0
    return out


class KUBA_Tracker(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="KUBA_Tracker",
            display_name="kubakub tracker",
            category=CATEGORY,
            search_aliases=["track", "tracking", "blob", "blob track", "motion", "yolo", "pose", "mediapipe", "sam2", "grounding dino", "dino",
                            "follow", "touchdesigner", "people", "count"],
            description=("Finds what moves in a video and follows it, each thing with its own id. Pick the finders on "
                         "the node (several at once): motion, a colour (eyedropper on the video), lights, people and "
                         "things (YOLO; a pose file adds body points, a seg file outlines), anything you name in words "
                         "(Grounding DINO), body / hands / face points (MediaPipe), a motif you click on a frame (SAM 2), "
                         "and exact outlines inside the boxes (SAM 2). Everything comes from models already on your "
                         "disk; nothing is downloaded. After a run the node shows the video with what was found; "
                         "'try this frame' checks a change on one frame in a second. Nothing connected: a built-in "
                         "sample clip. Draw the result with kubakub track draw."),
            is_output_node=True,                       # runs on its own, to see what it finds
            inputs=[
                io.MultiType.Input("video", types=[io.Image, io.Video], optional=True,
                                   tooltip="The video (or a batch of frames). Nothing connected: a sample clip."),
                io.String.Input("settings", default="", tooltip="Set on the node's pills and sliders; never typed."),
                io.Float.Input("fps", default=25.0, min=1.0, max=120.0, step=1.0, advanced=True,
                               tooltip="Speed of a batch of frames (a video brings its own)."),
                io.Image.Input("depth", optional=True,
                               tooltip="The depth of the same video from any depth node (Depth Anything 3 / DA3 "
                                       "Streaming, Depth Anything 2, Marigold, Video Depth Anything ...): every tracked "
                                       "thing gets its depth (near / far) and track draw can cut the picture into depth "
                                       "layers. Which side is near: the pill on the node."),
                TracksType.Input("tracks", optional=True,
                                 tooltip="The tracks of another kubakub tracker on the same video: this tracker adds "
                                         "its finds to them (other finders, another colour, another sensitivity), "
                                         "so track draw gets one set and can join everything with lines. A kind "
                                         "found twice gets a number (colour, colour 2)."),
                io.Audio.Input("audio", optional=True,
                               tooltip="The video's sound (or music): the tracks carry how loud it is and the beats, "
                                       "frame by frame, so track draw's marks pulse and its lines flash on the beat "
                                       "(how much: 'sound' in track draw's look). It plays with the video on the node "
                                       "and goes on to 'audio'."),
                SoundType.Input("sound", optional=True,
                                tooltip="From kubakub sound: the sound and how it is heard (tempo, gain, release), set "
                                        "once for every node that listens. Connected, it comes before 'audio'."),
            ],
            outputs=[
                TracksType.Output("tracks", tooltip="Everything found, with ids: for kubakub track draw."),
                io.Mask.Output("found", tooltip="White on what was found, every frame (outline if there is one, else the box)."),
                io.Mask.Output("near", tooltip="A soft glow around everything found, every frame (1 on it, 0 further than 1.5 "
                                               "times its size): the 'field' of kubakub shader, a mask for mask animate, the "
                                               "'change' of the region sampler."),
                io.Image.Output("frames", tooltip="The video's frames (for track draw's video input)."),
                io.Float.Output("fps", tooltip="The video's speed (for Create Video after track draw)."),
                io.String.Output("report"),
                io.Image.Output("depth", tooltip="The depth on 'depth', one scale for the whole clip, near = white (for "
                                                 "track draw's depth). Black without one."),
                io.Audio.Output("audio", tooltip="The sound, cut to the video (silence without one): for Create Video."),
            ],
            hidden=[io.Hidden.unique_id],
        )

    @classmethod
    def execute(cls, video=None, settings="", fps=25.0, depth=None, tracks=None, audio=None, sound=None) -> io.NodeOutput:
        t0 = time.perf_counter()
        notes = []
        if video is None:
            frames = torch.from_numpy(tk.sample_clip())
            rate = None
            notes.append("no video connected: the sample clip")
        else:
            frames, rate, _ = frames_of(video)
        fps = float(rate or fps)
        n, H, W = int(frames.shape[0]), int(frames.shape[1]), int(frames.shape[2])
        pbar = comfy.utils.ProgressBar(n)
        work = os.path.join(folder_paths.get_temp_directory(), "kubakub_track", "motif_" + hashlib.sha1(
            f"{n}|{W}|{H}|{float(frames[0].float().mean())}".encode()).hexdigest()[:12])
        s = tk.settings(settings)
        key = video_key(frames, s, fps)
        data = cached_tracks(key)
        if data is not None:
            pbar.update_absolute(n, n)
            notes.append("tracked before with these settings: the tracks come from the cache (change a finder to "
                         "track again)")
        else:
            try:
                T = tk.analyse(frames, fps, s, work_dir=work, progress=lambda k, total: pbar.update_absolute(k, total))
            finally:
                free_vram()
            data = T.to_dict()
            keep_tracks(key, data)
        dz = None
        if depth is not None:
            dz = tk.depth_frames(depth, n, H, W, s.get("depth_near", "bright"))
            tk.add_depth(data, dz)
            notes.append(f"depth: {int(depth.shape[0])} frame(s), near = {'bright' if s.get('depth_near') != 'dark' else 'dark'} "
                         f"on the depth node; every track has its depth")
        if tracks is not None:
            own = len(data["tracks"])
            data, renamed = chain(tracks, data)
            notes.append(f"chained: {len(data['tracks']) - own} tracks from the tracker before + {own} of this one"
                         + (" (" + ", ".join(f"{a} -> {b}" for a, b in renamed.items()) + ")" if renamed else ""))
        wave, hz, mono, heard = so.take_wave(sound, audio)
        audio_out = so.sound_out(wave, hz, n / fps)
        if mono is not None:                       # the tracks carry what is heard, frame by frame (track draw reacts)
            per_frame, lines = sr.frame_sound(mono, hz, heard, fps, n)
            data = {**data, "curves": {**(data.get("curves") or {}), **per_frame}}
            notes.append("sound: " + "; ".join(lines))
        uid = str(getattr(cls.hidden, "unique_id", "") or "")
        if uid:
            _LAST[uid] = (frames, fps)
        side = write_side(frames, "v", PREVIEW_SIDE)
        meta = {"frames": side, "tracks": save_json(data, "kubakub_track"), "W": W, "H": H, "n": n, "fps": fps}
        if mono is not None:
            meta["audio"] = save_wav(audio_out, "kubakub_track")
        if dz is not None:
            meta["depth"] = write_side(torch.from_numpy(np.repeat(dz[..., None], 3, -1)), "d", PREVIEW_SIDE)
        on = [k for k in tk.FINDERS if s[k]["on"]] + (["outline"] if s["outline"]["on"] else [])
        kinds = {}
        for tr in data["tracks"]:
            kinds[tr["kind"]] = kinds.get(tr["kind"], 0) + 1
        report = "\n".join([f"{n} frames {W}x{H} at {fps:g} fps, {time.perf_counter() - t0:.1f} s; finders: {', '.join(on) or 'none'}",
                            f"{len(data['tracks'])} tracks: " + (", ".join(f"{v} {k}" for k, v in sorted(kinds.items())) or "nothing found"),
                            *notes])
        log.info("[KUBA tracker] %s", report.replace("\n", " | "))
        found = torch.from_numpy(_union_masks(data, n, H, W))
        near = torch.from_numpy(np.stack([tk.near_field(data, t, W, H) for t in range(n)]))
        depth_out = torch.from_numpy(np.repeat(dz[..., None], 3, -1)) if dz is not None else torch.zeros((1, H, W, 3))
        return io.NodeOutput(data, found, near, frames, fps, report, depth_out, audio_out, ui={"kubakub_track": [meta]})


class KUBA_TrackDraw(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="KUBA_TrackDraw",
            display_name="kubakub track draw",
            category=CATEGORY,
            search_aliases=["track draw", "blob track", "diagram", "plexus", "network", "stamp", "labels", "hud", "overlay"],
            description=("Draws the tracks of kubakub tracker as a diagram, set on the node and seen live while it "
                         "plays: marks on every tracked thing (box, corner brackets, exact outline, circle, cross, dot, "
                         "labels with live values or words for its state, skeleton), lines between them (nearest, within "
                         "a distance, a network, a tree, the hull, a hub, a chain, trails, to the corners of the biggest), "
                         "what happens inside them (the video revealed, pixel blocks, frozen, inverted) and a stamp "
                         "trail that leaves their pixels along their path. Starts from a preset: diagnostic, "
                         "constellation, stamp trail. Over the video, on black, or on nothing (alpha). 'render the "
                         "look' on the node renders it (and what saves it) without running anything else."),
            is_output_node=True,                       # 'render the look' queues it (and its savers) alone
            inputs=[
                TracksType.Input("tracks", tooltip="From kubakub tracker."),
                io.Image.Input("video", optional=True,
                               tooltip="The same video (the tracker's 'frames' output): under the drawing, revealed, stamped."),
                io.String.Input("look", default="", tooltip="Set on the node; never typed."),
                io.Image.Input("depth", optional=True,
                               tooltip="The tracker's 'depth' output: the look's depth part (layers, contour lines, "
                                       "far = earlier frames, fog, a depth slice) and marks that grow when near."),
            ],
            outputs=[io.Image.Output("images"), io.Mask.Output("alpha", tooltip="Where something is drawn (or the video).")],
        )

    @classmethod
    def execute(cls, tracks, video=None, look="", depth=None) -> io.NodeOutput:
        if not isinstance(tracks, dict) or tracks.get("format") != tk.TRACKS_FORMAT:
            raise ValueError("kubakub track draw: 'tracks' must come from kubakub tracker.")
        lk = td.look(look)
        n, W, H = int(tracks["n"]), int(tracks["W"]), int(tracks["H"])
        frames = None
        if video is not None:
            frames = video.cpu().float().numpy()
            if frames.shape[1:3] != (H, W):
                import cv2
                frames = np.stack([cv2.resize(f, (W, H), interpolation=cv2.INTER_AREA) for f in frames])
        at = (lambda i: frames[min(max(i, 0), len(frames) - 1)]) if frames is not None else None
        dz = None
        if depth is not None:
            dz = depth.cpu().float().numpy()
            dz = dz[..., :3].mean(-1) if dz.ndim == 4 else dz
            if dz.shape[1:3] != (H, W):
                import cv2
                dz = np.stack([cv2.resize(f, (W, H), interpolation=cv2.INTER_LINEAR) for f in dz])
        dat = (lambda i: dz[min(max(i, 0), len(dz) - 1)]) if dz is not None else None
        pbar = comfy.utils.ProgressBar(n)
        out = np.empty((n, H, W, 3), np.float32)
        alpha = np.empty((n, H, W), np.float32)

        def one(t):
            rgb, a = td.render_frame(tracks, lk, t, at(t) if at else None, at, dat)
            out[t], alpha[t] = rgb, a
            pbar.update(1)
        with ThreadPoolExecutor(max_workers=max(1, min(8, (os.cpu_count() or 4) - 2))) as pool:
            list(pool.map(one, range(n)))
        return io.NodeOutput(torch.from_numpy(out), torch.from_numpy(alpha))


def _register_routes():
    """POST /kubakub/track/probe {node, frame, settings} -> what the finders see on that one frame (as tracks)."""
    try:
        from aiohttp import web
        from server import PromptServer
    except ImportError:
        return

    async def probe(request):
        import asyncio
        try:
            j = await request.json()
            got = _LAST.get(str(j.get("node", "")))
            if got is None:
                return web.json_response({"error": "run the tracker once, then try a frame"}, status=404)
            frames, fps = got
            t = int(j.get("frame", 0))

            def work():
                try:
                    return tk.analyse(frames, fps, j.get("settings") or "", only=[t]).to_dict()
                finally:
                    free_vram()
            data = await asyncio.get_running_loop().run_in_executor(None, work)
            return web.json_response({"tracks": data})
        except Exception as e:  # noqa: BLE001
            log.exception("[KUBA tracker] probe failed")
            return web.json_response({"error": (str(e).splitlines() or [""])[0]}, status=500)

    async def models(request):
        return web.json_response({"yolo": tk.list_models("yolo"), "sam2": tk.list_models("sam2"), "dino": tk.list_models("dino"), "presets": list(td.PRESETS),
                                  "default_look": td.DEFAULT_LOOK, "looks": td.PRESETS, "defaults": tk.DEFAULTS})

    try:
        PromptServer.instance.routes.post("/kubakub/track/probe")(probe)
        PromptServer.instance.routes.get("/kubakub/track/models")(models)
    except (AttributeError, RuntimeError) as e:
        log.warning("[KUBA tracker] routes not registered: %s", e)


_register_routes()

NODE_CLASS_MAPPINGS = {"KUBA_Tracker": KUBA_Tracker, "KUBA_TrackDraw": KUBA_TrackDraw}
NODE_DISPLAY_NAME_MAPPINGS = {"KUBA_Tracker": "kubakub tracker", "KUBA_TrackDraw": "kubakub track draw"}
