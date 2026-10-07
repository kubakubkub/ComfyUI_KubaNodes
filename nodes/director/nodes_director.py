"""
nodes_director.py

kubakub director (category KUBAKUB/director): the layer / compositing window of the Kuba nodes.
KUBA_Director takes the base image (matrix or render), optional layer images + masks, regions and a
3D scene. Each run it writes preview-size copies of everything into temp and sends a manifest to the
window (web/kubakub_director.js); the window edits a JSON document that lives in the node's
'document' widget. The run renders that document at full resolution (director/render.py) and returns
the composite, per-layer masks, the changed mask, the regions with placed layers and plan rules.

V3 node schema (comfy_api.latest), registered through the pack's NODE_CLASS_MAPPINGS loader.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from dataclasses import dataclass

import cv2
import numpy as np
import torch

import folder_paths
from comfy_api.latest import io, ui
from comfy_execution.graph_utils import ExecutionBlocker

from ...kubakub.director import media as md
from ...kubakub.director import colour
from ...kubakub.director import export as exm
from ...kubakub.director import projmask as pmk
from ...kubakub.director import render as rd
from ...kubakub.director import motion as mo
from ...kubakub import keyframes as kf
from ...kubakub import sound as snd
from ...kubakub.director.blend import MODES
from ...kubakub.io_types import DirectorType, RegionsType, SceneType
from ...kubakub.types import Regions
from ...kubakub import imio

log = logging.getLogger("KUBA.regions")

CATEGORY = "kubakub/2d/director"
PROXY_W = 2048                     # longest side of the preview copies the window works on (10k matrices: ~1:5)
UPLOAD_SUB = "kuba_director"       # imported images and paint layers live in input/kuba_director
PROXY_VERSION = 3                  # bump when make_proxy changes what it writes
COLOUR_NOTES: dict = {}            # input file -> how its colour profile was handled (for the report)


def _png(path, rgba):
    _png8(path, np.clip(rgba * 255 + 0.5, 0, 255).astype(np.uint8))


def _png8(path, a):
    if a.shape[-1] == 4:
        imio.imwrite(path, cv2.cvtColor(a, cv2.COLOR_RGBA2BGRA))
    else:
        imio.imwrite(path, cv2.cvtColor(a, cv2.COLOR_RGB2BGR))


def _proxy(img, k):
    h, w = img.shape[:2]
    if k >= 1:
        return img
    return cv2.resize(img, (max(1, round(w * k)), max(1, round(h * k))), interpolation=cv2.INTER_AREA)


_FILES = {}                        # (path, size, mtime_ns) -> (read-only RGBA float, colour note): imports decoded once
FILE_CACHE_BYTES = 2 << 30         # at most ~2 GB of decoded imports kept between runs


def _load_input_file(name):
    """'kuba_director/x.png' from ComfyUI's input folder -> RGBA float (read only, shared between runs), or None.
    Decoded once per file version (path + size + mtime); an edited or replaced file is read again."""
    base = os.path.realpath(folder_paths.get_input_directory())
    path = os.path.realpath(os.path.join(base, name))
    if not path.startswith(base + os.sep) or not os.path.isfile(path) or md.is_video(path):
        return None
    st = os.stat(path)
    key = (path, st.st_size, st.st_mtime_ns)
    hit = _FILES.pop(key, None)
    if hit is None:
        img, note = colour.read_image(path)              # an embedded ICC profile is converted to sRGB
        if img is not None:
            img.flags.writeable = False                  # shared by every run that uses the file: never changed
        hit = (img, note)
    _FILES[key] = hit                                    # most recently used last
    total = sum(v[0].nbytes for v in _FILES.values() if v[0] is not None)
    while len(_FILES) > 1 and total > FILE_CACHE_BYTES:
        old = _FILES.pop(next(iter(_FILES)))
        total -= old[0].nbytes if old[0] is not None else 0
    img, note = hit
    if note:
        COLOUR_NOTES[name] = note
    return img


def _label_png(labels):
    """Region label map as an RGB image the window decodes for picking: id + 1 in 24 bit, 0 = none."""
    v = (labels.astype(np.int64) + 1).clip(0, 0xFFFFFF)
    return np.stack([(v >> 16) & 255, (v >> 8) & 255, v & 255], -1).astype(np.float32) / 255


def _depth_vis(depth):
    from ..scene3d.nodes_scene3d import _depth_vis as vis        # one colouring for scene and director
    return np.asarray(vis(depth), np.float32)


_VIEWS = {}                        # (out dir, scene files + mtimes, W, H, k) -> {view: file}: scene views made once
_PROXIES = {}                      # (out dir, kind) -> (hash of the pixels, file): base / labels rewritten only when changed


def _png_once(out_dir, kind, stamp, rgb):
    """<kind>_<stamp>.png in out_dir, or the file from an earlier run with the same pixels -> the file name."""
    a = np.clip(rgb * 255 + 0.5, 0, 255).astype(np.uint8)
    h = hashlib.sha1(a.tobytes())
    h.update(str(a.shape).encode())
    hit = _PROXIES.get((out_dir, kind))
    if hit and hit[0] == h.hexdigest() and os.path.isfile(os.path.join(out_dir, hit[1])):
        return hit[1]
    fn = f"{kind}_{stamp}.png"
    _png8(os.path.join(out_dir, fn), a)
    _PROXIES[(out_dir, kind)] = (h.hexdigest(), fn)
    return fn


def _scene_views_key(scene, out_dir, W, H, k):
    """What the scene views depend on: the files they are made of (name, mtime, size), the canvas and proxy size."""
    ids = sorted(f for f in os.listdir(scene["ids"]) if f.startswith("ids_") and f.endswith(".png"))
    pick = next((f for f in ids if "shelves" in f), ids[0] if ids else None)
    files = [os.path.join(scene["folder"], "faceid.npy")] + [os.path.join(scene["ids"], f) for f in ("clay.png", "depth_m.npy", pick) if f]
    st = tuple((f, s.st_mtime_ns, s.st_size) for f in files for s in [os.stat(f)] if os.path.isfile(f))
    return (out_dir, scene["folder"], scene["ids"], st, W, H, k)


def _scene_views(scene, out_dir, stamp, W, H, k, made):
    """The scene's clay / depth / ids / silhouette views for the window at proxy size -> made {view: file name}."""
    for name, fname in (("clay", "clay.png"),):
        p = os.path.join(scene["ids"], fname)
        if os.path.isfile(p):
            img = cv2.cvtColor(imio.imread(p), cv2.COLOR_BGR2RGB).astype(np.float32) / 255
            fn = f"{name}_{stamp}.png"
            _png(os.path.join(out_dir, fn), _proxy(cv2.resize(img, (W, H)), k))
            made[name] = fn
    dp = os.path.join(scene["ids"], "depth_m.npy")
    if os.path.isfile(dp):
        fn = f"depth_{stamp}.png"
        _png(os.path.join(out_dir, fn), _proxy(cv2.resize(_depth_vis(np.load(dp)), (W, H)), k))
        made["depth"] = fn
    ids = sorted(f for f in os.listdir(scene["ids"]) if f.startswith("ids_") and f.endswith(".png"))
    pick = next((f for f in ids if "shelves" in f), ids[0] if ids else None)
    if pick:
        img = cv2.cvtColor(imio.imread(os.path.join(scene["ids"], pick)), cv2.COLOR_BGR2RGB).astype(np.float32) / 255
        fn = f"ids_{stamp}.png"
        _png(os.path.join(out_dir, fn), _proxy(cv2.resize(img, (W, H), interpolation=cv2.INTER_NEAREST), k))
        made["ids"] = fn
    from ..scene3d.nodes_scene3d import load_scene_cached
    sil = (load_scene_cached(scene)[0]["faceid"] > 0).astype(np.float32)   # for the projection mask 'scene'
    fn = f"silhouette_{stamp}.png"
    _png(os.path.join(out_dir, fn), _proxy(cv2.resize(sil, (W, H), interpolation=cv2.INTER_AREA), k))
    made["_silhouette"] = fn
    return made


_LAST = {}                         # state key -> what the relight preview needs from the last run (newest last)
LAST_MAX = 8                       # directors (node + workflow) whose preview state is kept


def state_key(unique_id, extra_pnginfo=None):
    """'<node id>' or '<node id>_w<hash of the workflow id>': the director's preview state, temp folder and purge
    belong to one node in one workflow (director node 3 in two open workflows no longer share them)."""
    uid = re.sub(r"[^\w.\-]+", "_", str(unique_id or "0"))         # subgraph ids contain ':'
    wf = (extra_pnginfo or {}).get("workflow") if isinstance(extra_pnginfo, dict) else None
    wid = str(wf.get("id") or "") if isinstance(wf, dict) else ""
    return f"{uid}_w{hashlib.sha1(wid.encode('utf-8', 'replace')).hexdigest()[:10]}" if wid else uid


def _remember(key, value):
    """_LAST[key] = value, at most LAST_MAX entries (the oldest go)."""
    _LAST.pop(key, None)
    _LAST[key] = value
    while len(_LAST) > LAST_MAX:
        _LAST.pop(next(iter(_LAST)))
try:
    import asyncio as _asyncio
    _RELIGHT_LOCK = _asyncio.Lock()
    _PROXY_LOCK = _asyncio.Lock()
except Exception:  # noqa: BLE001
    _RELIGHT_LOCK = _PROXY_LOCK = None


def _doc_sources(doc, known):
    """Sources a document refers to: the run's inputs plus imported / painted files from input/."""
    sources = dict(known)
    for L in doc.get("layers", []) if isinstance(doc, dict) else []:
        src = str(L.get("source") or "")
        if src.startswith("file:") and src not in sources:
            img = _load_input_file(src[5:])
            if img is not None:
                sources[src] = img
    return sources


def _projector_image(doc, layer_id, source, base, sources, labels, table, mask_cache=None):
    """What a light layer's projector casts, at full size: the layers below it, the base, or one layer (even hidden)."""
    layers = [L for L in doc.get("layers", []) if isinstance(L, dict)]
    if source == "base":
        return base[..., :3]
    if source.startswith("layer:"):
        pick = next((L for L in layers if str(L.get("id")) == source[6:]), None)
        if pick is None:
            raise ValueError(f"projector: no layer '{source[6:]}'")
        if pick.get("kind") == "base":
            return base[..., :3]
        sub = [dict(pick, visible=True, opacity=1.0, blend="normal"), {"id": "__base", "kind": "base", "visible": False}]
    else:
        i = next(k for k, L in enumerate(layers) if str(L.get("id")) == layer_id)
        sub = [dict(L) for L in layers[i + 1:]]
        if not any(L.get("kind") == "base" for L in sub):
            sub.append({"id": "__base", "kind": "base", "visible": False})
    return rd.render({"version": 1, "layers": sub, "timeline": doc.get("timeline") or {}}, base, sources, labels, table,
                     image_only=True, mask_cache=mask_cache)["image"]


def _input_path(name):
    """A file in ComfyUI's input folder (no escaping it), or None."""
    base = os.path.realpath(folder_paths.get_input_directory())
    path = os.path.realpath(os.path.join(base, name))
    return path if path.startswith(base + os.sep) and os.path.isfile(path) else None


def _source_video(ref):
    """'file:<name in input/>' (uploaded) or 'path:<absolute path>' (linked, e.g. a big ProRes on disk) -> the video
    file, or None."""
    ref = str(ref or "")
    if ref.startswith("path:"):
        p = _local_file(ref[5:])                                 # local drive letters only (mapped drives work)
        return p if p and md.is_video(p) else None
    name = ref[5:] if ref.startswith("file:") else ref
    return _input_path(name) if name and md.is_video(name) else None


def _reachable_from_outside():
    """ComfyUI was started with --listen on more than this computer: its routes answer other machines too."""
    try:
        from comfy.cli_args import args
        hosts = [h.strip().lower() for h in str(getattr(args, "listen", "") or "").split(",")]
    except Exception:  # noqa: BLE001
        return False
    return any(h not in ("", "127.0.0.1", "localhost", "::1") for h in hosts)


PATHS_OFF = ("files by path are off while ComfyUI listens on the network (--listen). Put the file into ComfyUI's input "
             "folder, or allow it with  remote_paths = on  in kubakub.ini [settings]")


def _route_paths_ok():
    """May a request to the window's routes name a file by its path ('path:...', hdri_file)? Yes on this computer
    only; with --listen only when kubakub.ini says remote_paths = on. A workflow that runs is not affected."""
    return not _reachable_from_outside() or kst.switch("remote_paths", False)


def _doc_names_paths(doc):
    """Does a document sent to a route read files by path (video / image sources 'path:...', a light's HDRI)?"""
    doc = doc if isinstance(doc, dict) else {}
    for L in doc.get("layers") or []:
        if not isinstance(L, dict):
            continue
        lt = L.get("light") if isinstance(L.get("light"), dict) else {}
        if str(L.get("source") or "").startswith("path:") or str(lt.get("hdri_file") or "").strip():
            return True
    pm = doc.get("projection_mask") if isinstance(doc.get("projection_mask"), dict) else {}
    return str(pm.get("source") or "").startswith("path:")


def _local_file(p):
    """An absolute path on a local drive (no UNC / device paths: isfile would already open SMB), or None."""
    p = os.path.expandvars(str(p or "").strip().strip('"'))
    if not p or p.replace("/", "\\").startswith("\\\\"):
        return None
    drive = os.path.splitdrive(p)[0]
    return p if len(drive) == 2 and drive[1] == ":" and os.path.isabs(p) and os.path.isfile(p) else None


def _read_image_any(ref):
    """'file:<name in input/>' or 'path:<file>' -> RGBA / RGB / grey uint8 or uint16 array (unicode paths ok), or None."""
    ref = str(ref or "")
    path = _local_file(ref[5:]) if ref.startswith("path:") else _input_path(ref[5:] if ref.startswith("file:") else ref)
    if not path:
        return None
    img = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA if img.shape[2] == 4 else cv2.COLOR_BGR2RGB)
    return img


def projection_mask(doc, scene, W, H, notes):
    """The document's projection mask as float32 H x W (1 = building), or None (off / no source)."""
    st = pmk.settings(doc)
    if st is None:
        return None
    if st["source"] == "scene":
        if scene is None:
            notes.append("projection mask from the scene: connect a scene")
            return None
        from ..scene3d.nodes_scene3d import load_scene_cached
        m, how = (load_scene_cached(scene)[0]["faceid"] > 0).astype(np.float32), "the scene's silhouette"
    else:
        img = _read_image_any(st["source"])
        if img is None:
            notes.append(f"projection mask file not found: {st['source'][5:]}")
            return None
        m, how = pmk.from_image(img)
    m, note = pmk.fit(m, W, H)
    if note:
        notes.append(note)
    m = pmk.refine(m, st["grow"], st["feather"], st["invert"])
    notes.append(f"projection mask: {how}, the building is {float(m.mean()) * 100:.0f} % of the frame")
    return m


def _video_path(L, seg=None):
    """The file a video layer (or one of its segments) plays, or None when it is not a video layer. A segment's
    'file' is a name in input/ or a 'path:' like the layer source."""
    if not isinstance(L, dict) or L.get("kind", "image") != "image":
        return None
    src = str(L.get("source") or "")
    if not (src.startswith(("file:", "path:")) and md.is_video(src)):
        return None
    return _source_video((seg or {}).get("file") or src)


def _lights_used(doc):
    """Ids of light layers that other layers react to ('light_react'): rendered even when hidden."""
    layers = [L for L in (doc or {}).get("layers", []) if isinstance(L, dict)]
    lights = [str(L.get("id")) for L in layers if L.get("kind") == "light"]
    used = set()
    for L in layers:
        rr = L.get("light_react")
        if isinstance(rr, dict) and rr.get("on") is not False and float(rr.get("amount", 1) or 0) > 0 and lights:
            used.add(str(rr.get("by")) if rr.get("by") else lights[0])
    return used


def _layer_refs(doc):
    """Ids of the layers other layers use as a picture: mask by (incl. 'below'), a projector's 'layer:<id>' and a
    glow's 'layer:<id>'. A hidden video layer in this set still needs its frames (a track matte)."""
    layers = [L for L in (doc or {}).get("layers", []) if isinstance(L, dict)]
    ref = set()
    for i, L in enumerate(layers):
        by = str((L.get("mask") or {}).get("by") or "")
        if by == "below":
            if i + 1 < len(layers):
                ref.add(str(layers[i + 1].get("id")))
        elif by:
            ref.add(by)
        light = L.get("light") if isinstance(L.get("light"), dict) else {}
        pj = str((light.get("projector") or {}).get("source") or "")
        if pj.startswith("layer:"):
            ref.add(pj[6:])
        for g in light.get("glow") or []:
            gb = str((g or {}).get("by") or "") if isinstance(g, dict) else ""
            if gb.startswith("layer:"):
                ref.add(gb[6:])
    return ref


class MediaFrames:
    """
    The frames video layers show in a set of animated documents: {frame i: (doc_i, t_i)} with placements at
    `scale` x the canvas. Planned once (which file, which frames, at which size: the layer's largest box, at most
    the native size); decoded with load(frames) one window at a time, so long sequences at delivery size stay
    in bounded memory. Per frame: sources["media:<layer id>"] = float RGBA and the layer in doc_i points to it.
    Outside its segments a layer is hidden, or, when another layer uses it (a track matte), an empty picture
    ('media:none'), as the window shows it. Docs are changed in place.
    """
    EMPTY = np.zeros((2, 2, 4), np.float32)

    def __init__(self, docs_times: dict, scale: float, notes: list):
        self.frames, self.use, self.hidden, self.size = {}, {}, set(), {}
        plan, empty = {}, set()
        for i, (d, t) in docs_times.items():
            refs = _layer_refs(d)
            for L in d.get("layers", []) if isinstance(d, dict) else []:
                if _video_path(L) is None:
                    continue
                lid = str(L.get("id"))
                if L.get("visible") is False and lid not in refs:
                    continue
                m = L.get("media") if isinstance(L.get("media"), dict) else {}
                try:
                    if lid not in plan:
                        segs = md.segments(m, md.probe(_video_path(L)).get("duration", 0.0))
                        plan[lid] = {"segs": segs, "blend": bool(m.get("blend_frames")), "bw": 0.0, "bh": 0.0}
                    p = plan[lid]
                    at = md.source_time(p["segs"], t)
                    path = None if at is None else _video_path(L, p["segs"][at[0]])
                    if at is not None and path is None:
                        notes.append(f"{L.get('name') or lid}: video file of segment {at[0] + 1} not found")
                    if path is None:
                        (empty if lid in refs else self.hidden).add((i, lid))
                        continue
                    info = md.probe(path)
                    if not L.get("w") or not L.get("h"):     # native size, like images without a box
                        L["w"], L["h"] = info["w"] * scale, info["h"] * scale
                    p["bw"], p["bh"] = max(p["bw"], abs(float(L["w"]))), max(p["bh"], abs(float(L["h"])))
                    self.use[(i, lid)] = (path, md.frame_pick(at[1], info["fps"], info["frames"], p["blend"]))
                except (OSError, ValueError, RuntimeError, ArithmeticError) as e:
                    notes.append(f"{L.get('name') or lid}: {e}".split("\n")[0])
                    (empty if lid in refs else self.hidden).add((i, lid))
        for (i, lid), (path, picks) in self.use.items():
            info = md.probe(path)
            p = plan[lid]
            self.size[(i, lid)] = (max(2, min(info["w"], int(np.ceil(p["bw"])))), max(2, min(info["h"], int(np.ceil(p["bh"])))))
        for i, (d, _) in docs_times.items():
            for L in d.get("layers", []) if isinstance(d, dict) else []:
                lid = str(L.get("id")) if isinstance(L, dict) else ""
                if (i, lid) in self.hidden:
                    L["visible"] = False
                elif (i, lid) in empty:
                    L["source"] = "media:none"
                    L.pop("remove_bg", None)
                elif (i, lid) in self.use:
                    L["source"] = f"media:{lid}"
                    L.pop("remove_bg", None)                   # not per frame (too slow); cut the file beforehand
        self.loaded = set()

    def decode(self, frames):
        """What these timeline frames need, decoded -> {(path, size): {index: frame}} (changes nothing: safe in a
        helper thread while the previous window composes). Several files decode in parallel (one ffmpeg each)."""
        want = set(frames)
        need = {}
        for (i, lid), (path, picks) in self.use.items():
            if i in want:
                need.setdefault((path, self.size[(i, lid)]), set()).update(ix for ix, _ in picks)
        if len(need) < 2:
            return {(path, size): md.decode(path, sorted(idx), size) for (path, size), idx in need.items()}
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=min(4, len(need))) as pool:
            futs = {k_: pool.submit(md.decode, k_[0], sorted(idx), k_[1]) for k_, idx in need.items()}
            return {k_: f.result() for k_, f in futs.items()}

    def use_window(self, frames, decoded):
        """Make a decode(frames) result the current window (the previous one is freed)."""
        self.frames = decoded
        self.loaded = set(frames)
        return self

    def load(self, frames):
        """Decode what these timeline frames need; the previous window is freed."""
        self.frames = {}                               # free the old window before decoding the new one
        return self.use_window(frames, self.decode(frames))

    def sources(self, i, base: dict) -> dict:
        """base sources + this frame's video frames (load() the frame first; the still loads itself)."""
        if i not in self.loaded:
            self.load([i])
        out = dict(base)
        out["media:none"] = self.EMPTY
        for (j, lid), (path, picks) in self.use.items():
            if j == i:
                out[f"media:{lid}"] = md.mix(self.frames[(path, self.size[(j, lid)])], picks)
        return out


def media_audio(doc, duration_s, sr):
    """The sound of all visible video layers ([2, S]) and notes."""
    track, notes = np.zeros((2, int(round(duration_s * sr))), np.float32), []
    for L in (doc or {}).get("layers", []):
        if _video_path(L) is None or L.get("visible") is False:
            continue
        m = L.get("media") if isinstance(L.get("media"), dict) else {}
        vol = float(m.get("volume", 1.0)) if isinstance(m.get("volume", 1.0), (int, float)) else 1.0
        if vol <= 0:
            continue
        try:
            segs = md.segments(m, md.probe(_video_path(L))["duration"])
            t, n_ = md.layer_audio(segs, lambda s, L=L: _video_path(L, s), duration_s, sr, vol)
            track += t[:, :track.shape[1]]
            notes += n_
        except (OSError, ValueError, RuntimeError, ArithmeticError) as e:
            notes.append(f"{L.get('name')}: sound not read ({str(e).splitlines()[0]})")
    return track, notes


_LEVEL_CACHE = {}


def level_of(name):
    """The loudness curve (motion.level_curve, 100 / s) of a sound file in input/, cached per file + mtime; None when
    missing. The window fetches the same curve (/kubakub/director/level), so both animate from identical numbers."""
    path = _input_path(str(name or "")) if name else None
    if not path:
        return None
    key = (path, os.path.getmtime(path))
    if key not in _LEVEL_CACHE:
        from comfy_extras.nodes_audio import load as load_audio
        wave, sr = load_audio(path)
        if len(_LEVEL_CACHE) > 8:
            _LEVEL_CACHE.clear()
        _LEVEL_CACHE[key] = mo.level_curve(wave.float().mean(dim=0).numpy(), int(sr))
    return _LEVEL_CACHE[key]


_SOUND_CACHE = {}


def sound_of(name):
    """The bands of a sound file in input/ (kubakub/sound.py analyze: loudness and hits in low / mid / high), cached
    per file + mtime; None when missing. The window fetches the same numbers (/kubakub/director/level, bands)."""
    path = _input_path(str(name or "")) if name else None
    if not path:
        return None
    key = (path, os.path.getmtime(path))
    if key not in _SOUND_CACHE:
        from comfy_extras.nodes_audio import load as load_audio
        wave, sr = load_audio(path)
        if len(_SOUND_CACHE) > 4:
            _SOUND_CACHE.clear()
        _SOUND_CACHE[key] = snd.analyze(wave.float().mean(dim=0).numpy(), int(sr))
    return _SOUND_CACHE[key]


def motion_needs(doc):
    """(loudness curve wanted, bands wanted) by the behaviours of a document: sound level, sound level of a band,
    swap masks on low / mid / high."""
    level = bands = False
    for L in (doc or {}).get("layers", []):
        for b in (L.get("motion") or []) if isinstance(L, dict) else []:
            if not isinstance(b, dict) or b.get("on") is False:
                continue
            if b.get("type") == "audio":
                if b.get("band") in mo.BANDS:
                    bands = True
                else:
                    level = True
            elif b.get("type") == "swap" and b.get("trigger") in mo.BANDS:
                bands = True
    return level, bands


def motion_context(doc):
    """Beats / markers for pulse and swap behaviours and, when a behaviour listens to the sound, the loudness curve
    and the bands of the timeline's sound file."""
    want_level, want_bands = motion_needs(doc)
    name = (((doc or {}).get("timeline") or {}).get("audio") or {}).get("file")
    return mo.context(doc, level_of(name) if want_level else None, sound_of(name) if want_bands else None)


def timeline_audio(doc, sample_rate=44100):
    """
    The timeline's sound as core AUDIO ({"waveform": [1, C, S], "sample_rate"}): timeline.audio = {"file": name in
    input/, "offset": seconds into the file where the timeline starts, "gain"}; trimmed / padded to the timeline
    length. Silence when there is no file (Create Video still gets a track of the right length).
    """
    tl = rd.timeline(doc) if doc else {"duration": 10.0}
    a = ((doc or {}).get("timeline") or {}).get("audio") or {}
    name = str(a.get("file") or "")
    wave, sr, note = None, sample_rate, ""
    if name:
        base = os.path.realpath(folder_paths.get_input_directory())
        path = os.path.realpath(os.path.join(base, name))
        if path.startswith(base + os.sep) and os.path.isfile(path):
            from comfy_extras.nodes_audio import load as load_audio
            wave, sr = load_audio(path)
            wave = wave.float()
        else:
            note = f"audio file not found: {name}"
    n = int(round(tl["duration"] * sr))
    out = torch.zeros((1, 2 if wave is None else wave.shape[0], n))
    if wave is not None:
        off = int(round(float(a.get("offset") or 0) * sr))
        src0, dst0 = max(0, off), max(0, -off)
        m = max(0, min(wave.shape[-1] - src0, n - dst0))
        if m:
            out[0, :, dst0:dst0 + m] = wave[:, src0:src0 + m] * float(a.get("gain", 1.0) or 1.0)
    if doc:                                           # + the sound of the video layers
        va, vnotes = media_audio(doc, tl["duration"], int(sr))
        if vnotes:
            note = "; ".join(x for x in [note, *vnotes] if x)
        if va.size and np.abs(va).max() > 0:
            if out.shape[1] == 1:
                out = out.repeat(1, 2, 1)
            out[0, :2, :va.shape[1]] += torch.from_numpy(va[:, :n])
    return {"waveform": out, "sample_rate": int(sr)}, note


def _render_sequence(doc, base, sources, labels, table, scene, W, H, scale, notes, skip_h3=False, pmask=None, sink=None, want_alpha=False,
                     light_scale=1.0):
    """
    Every frame of the document's timeline at scale x the matrix size -> (frames float32 [N, h, w, 3], fps, report).
    Keyframes are evaluated like the window (render.animate); light layers render in one Blender session per
    layer (frames with the same rig once), bottom up so a projector can cast a lower light layer.
    skip_h3: frames that H3 keyframe clips replace completely are not rendered; a dissolve between the nearest
    rendered frames stands in for them.
    sink(i, rgb, alpha): the export - every frame is handed over in order as soon as it is composed, nothing is kept
    (frames is then None). alpha: the base is left out and alpha = what the layers cover (x projection mask).
    A sink with prep(i, rgb, alpha) / put(item) (export.Writer): prep runs in the render pool (quantise, PNG encode),
    put gets the results in order, so rendering and encoding overlap.
    """
    t0 = time.perf_counter()
    tl = rd.timeline(doc)
    n, fps = tl["frames"], tl["fps"]
    # the exact scaled size, then the last row / column cropped to even sizes (H.264 / yuv420): at scale 1 the
    # frames are pixel-identical to the still, nothing is resampled
    wt, ht = max(16, round(W * scale)), max(16, round(H * scale))
    ws, hs = wt - wt % 2, ht - ht % 2
    sc = wt / W
    base_s = (cv2.resize(base, (wt, ht), interpolation=cv2.INTER_AREA) if (wt, ht) != (W, H) else base)[:hs, :ws]
    labels_s = None if labels is None else (cv2.resize(labels, (wt, ht), interpolation=cv2.INTER_NEAREST) if (wt, ht) != (W, H) else labels)[:hs, :ws]
    mctx = motion_context(doc)
    docs = [rd.scale_doc(rd.animate(doc, i / fps, mctx), sc) for i in range(n)]
    hidden = set(kf.hidden_frames(doc, n, fps)) if skip_h3 and sink is None else set()
    shown = [i for i in range(n) if i not in hidden]
    if want_alpha:                                    # an export with alpha: the layers without the base
        for d in docs:
            for q in d.get("layers", []):
                if isinstance(q, dict) and q.get("kind") == "base":
                    q["visible"] = False
    mf = MediaFrames({i: (docs[i], i / fps) for i in shown}, sc, notes)         # video layers: planned once
    WIN = 50                                          # frames decoded at a time (~2 s): bounded memory at any size
    windows = [shown[k:k + WIN] for k in range(0, len(shown), WIN)]
    mask_cache = {}                                   # region masks: labels and selectors are the same for every frame
    pm_s = None if pmask is None else (cv2.resize(pmask, (wt, ht), interpolation=cv2.INTER_AREA) if (wt, ht) != (W, H) else pmask)[:hs, :ws]
    lights = {}                                       # layer id -> [reader or None per frame]
    reports = []
    from concurrent.futures import ThreadPoolExecutor   # numpy / opencv release the GIL: frames in parallel
    workers = max(1, min(8, (os.cpu_count() or 4) - 2))
    pool = ThreadPoolExecutor(max_workers=workers)
    helper = ThreadPoolExecutor(max_workers=1)        # decodes the next window of video frames in the background
    try:
        return _sequence_body(doc, docs, n, fps, windows, shown, hidden, mf, sources, base_s, labels_s, table, scene,
                              wt, ht, ws, hs, pm_s, mask_cache, lights, reports, notes, sink, want_alpha, pool, helper,
                              workers, t0, light_scale)
    finally:
        helper.shutdown(wait=True, cancel_futures=True)
        pool.shutdown(wait=True, cancel_futures=True)


def _light_needs(d, lid):
    """The light layer of frame doc d when that frame renders it (visible, or another layer reacts to it), else None."""
    Li = next((q for q in d["layers"] if str(q.get("id")) == lid), None)
    if Li is None or (Li.get("visible") is False and lid not in _lights_used(d)):
        return None
    return Li


def _sequence_body(doc, docs, n, fps, windows, shown, hidden, mf, sources, base_s, labels_s, table, scene, wt, ht, ws,
                   hs, pm_s, mask_cache, lights, reports, notes, sink, want_alpha, pool, helper, workers, t0, light_scale=1.0):
    """_render_sequence after the planning: light layers (inputs prepared in the pool), then the frames."""
    from ..scene3d.nodes_scene3d import relight_sequence
    from ...kubakub.scene3d import scene_view as sv
    for L in reversed(doc.get("layers", [])):
        if not isinstance(L, dict) or L.get("kind") != "light":
            continue
        lid = str(L.get("id"))
        if scene is None:
            notes.append(f"light '{L.get('name') or lid}': connect a scene to render it")
            continue
        todo, work = [], []

        def light_input(i, lid=lid):
            """(rig, emission, projector picture) of frame i: glow masks and the cast image, prepared in the pool."""
            d = docs[i]
            rig = sv.rig_from_doc(_light_needs(d, lid).get("light"))
            src_i = mf.sources(i, sources)
            for other, readers in lights.items():
                if readers[i] is not None:
                    rgb, alpha = readers[i]()
                    src_i[f"light:{other}"] = np.concatenate([rgb, alpha[..., None]], -1)
            emission = rd.glow_masks(d, rig, src_i, labels_s, table, ws, hs)
            cast = _projector_image(d, lid, rig["projector"]["source"], base_s, src_i, labels_s, table, mask_cache) if rig["projector"]["on"] else None
            return rig, emission, cast

        for win in windows:
            need = [i for i in win if _light_needs(docs[i], lid) is not None]
            if not need:
                continue
            if not set(need) <= mf.loaded:            # this window of video frames (glow / projector may use them)
                mf.load(win)
            todo += need
            work += list(exm.ordered(pool, light_input, need, 2 * workers))
        readers = [None] * n
        if work:
            got, rep = relight_sequence(scene, work, size=(wt, ht), scale=float(light_scale))    # the still's framing, cropped below
            reports.append(f"light '{L.get('name') or lid}': {rep}")
            transparent = {i: sv.rig_from_doc(next(q for q in docs[i]["layers"] if str(q.get("id")) == lid).get("light"))["background"] == "transparent" for i in todo}
            for i, r in zip(todo, got):
                def read(r=r, tr=transparent[i]):
                    rgb, alpha = r()
                    rgb, alpha = rgb[:hs, :ws], alpha[:hs, :ws]
                    return rgb, (alpha if tr else np.ones_like(alpha))
                readers[i] = read
        lights[lid] = readers
    frames = None if sink is not None else np.zeros((n, hs, ws, 3), np.float32)
    prep = getattr(sink, "prep", None)

    def compose(i):
        src_i = mf.sources(i, sources)
        for lid, readers in lights.items():
            if readers[i] is not None:
                rgb, alpha = readers[i]()
                src_i[f"light:{lid}"] = np.concatenate([rgb, alpha[..., None]], -1)
        rr = rd.render(docs[i], base_s, src_i, labels_s, table, image_only=True, mask_cache=mask_cache, frame_seed=i, with_alpha=want_alpha)
        img, a = rr["image"], rr.get("alpha")
        if pm_s is not None:
            img = pmk.apply(img, pm_s)
            a = None if a is None else a * pm_s
        if sink is not None:
            if a is not None:                          # composited over black = premultiplied: straight colour for the file
                img = np.where(a[..., None] > 1e-6, img / np.maximum(a[..., None], 1e-6), 0).astype(np.float32)
            return prep(i, img, a) if prep else (img, a)
        frames[i] = img

    nxt = helper.submit(mf.decode, windows[0]) if windows else None
    for k_, win in enumerate(windows):                # a window of video frames: compose it, free it
        decoded = nxt.result()
        mf.frames = {}                                # the previous window goes before the next one is decoded
        mf.use_window(win, decoded)
        decoded = None
        # the next window decodes (ffmpeg) while this one composes: at most two windows in RAM
        nxt = helper.submit(mf.decode, windows[k_ + 1]) if k_ + 1 < len(windows) else None
        # streamed in order: frame k goes to the file while the next ones render (no window of frames in RAM)
        for i, got in zip(win, exm.ordered(pool, compose, win, 2 * workers)):
            if prep:
                sink.put(got)
            elif sink is not None:
                sink(i, *got)
        got = None
    if sink is not None:
        return None, fps, f"export: {n} frames at {ws}x{hs}, {fps} fps, {time.perf_counter() - t0:.1f} s" + "".join("\n" + r for r in reports)
    for i in sorted(hidden):                          # H3 replaces these: a dissolve between the rendered neighbours
        a = max(j for j in shown if j < i)
        b = min(j for j in shown if j > i)
        frames[i] = frames[a] + (frames[b] - frames[a]) * ((i - a) / (b - a))
    report = (f"sequence: {n} frames at {ws}x{hs}, {fps} fps, {time.perf_counter() - t0:.1f} s"
              + (f"\n{len(hidden)} frames under H3 keyframe clips are placeholders (a dissolve), not rendered: kubakub "
                 f"keyframe clips (h3) replaces them (skip_h3_frames; off = every frame rendered)" if hidden else "")
              + "".join("\n" + r for r in reports))
    return frames, fps, report


def _export_dir(name):
    safe = re.sub(r"[^\w.\-]+", "_", name).strip("_") or "director"
    return os.path.join(folder_paths.get_output_directory(), "kubakub_director", f"{safe}_{time.strftime('%Y%m%d-%H%M%S')}"), safe


def _export(fmt, scale, alpha, name, doc, base, sources, labels, table, scene, W, H, pmask, audio, written=None):
    """The whole timeline into a file, streamed (see director/export.py) -> a report line. written: a list that gets
    the export folder and the written files."""
    t0 = time.perf_counter()
    out_dir, safe = _export_dir(name)
    wt, ht = max(16, round(W * scale)), max(16, round(H * scale))
    ws, hs = wt - wt % 2, ht - ht % 2
    fps = rd.timeline(doc)["fps"]
    wave = audio["waveform"][0].float().cpu().numpy() if audio is not None else None
    writer = exm.Writer(fmt, out_dir, safe, ws, hs, fps, alpha=alpha, audio=wave, sr=int(audio["sample_rate"]) if audio else 44100)
    notes = []
    try:
        _, _, rep = _render_sequence(json.loads(json.dumps(doc)), base, sources, labels, table, scene, W, H, scale, notes,
                                     skip_h3=False, pmask=pmask, sink=writer, want_alpha=alpha)
    finally:
        files = writer.close()
    if written is not None:
        written += [out_dir, *files]
    size = sum(os.path.getsize(os.path.join(dp, f)) for p_ in files for dp, _, fs in (os.walk(p_) if os.path.isdir(p_) else [(os.path.dirname(p_), None, [os.path.basename(p_)])]) for f in fs)
    return (f"export {exm.FORMATS[fmt]['label']}{' with alpha' if alpha and (exm.FORMATS[fmt].get('png') or exm.FORMATS[fmt].get('alpha')) else ''}: "
            f"{writer.n} frames {ws}x{hs} -> {out_dir} ({size / 2 ** 20:.0f} MB, {time.perf_counter() - t0:.0f} s)"
            + "".join("; " + x for x in dict.fromkeys(notes)))


# ------------------------------------------------------------------------------------------------ content caches
# Every apply in the window rewrites the whole document (playhead, snap, diffusion settings ...), so ComfyUI re-runs
# everything after the director. The sequence and the export only depend on the document's content: they are kept
# per content key and handed out again when nothing that changes pixels, sound or timing has changed.
UI_DOC_KEYS = ("diffusion", "dismissed")          # window state saved with the document, never rendered
UI_TIMELINE_KEYS = ("time", "snap", "h3")         # the playhead (sequences animate every frame themselves), snap,
                                                  # the H3 prompt settings (read by keyframe clips (h3) only)
CLIP_TIMING_KEYS = ("id", "on", "t_start", "t_end", "mode")     # what hidden_frames reads of a clip
def _pack_settings():
    """The pack's settings.py (kubakub.ini [settings]); loaded by path when this module is imported on its own."""
    try:
        from ... import settings
        return settings
    except ImportError:
        import importlib.util
        p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "settings.py")
        spec = importlib.util.spec_from_file_location("kubakub_pack_settings", p)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod


kst = _pack_settings()


SEQ_CACHE_ITEMS = max(0, int(kst.number("director_seq_cache", 2)))   # 0 = off (kubakub.ini [settings])
_SEQ_CACHE = {}                                   # key -> (frames tensor, report), newest last
_EXPORTS = {}                                     # key -> (folder, files, report) of exports written this session
_DIGESTS = {}                                     # id(read-only array) -> (weakref, digest)


def _stamp(label, path):
    st = os.stat(path)
    return f"{label}|{st.st_size}|{st.st_mtime_ns}"


def file_stamps(doc):
    """Size + mtime of every file the document reads that can change on disk without the document changing: the
    timeline sound, the projection mask file, images / videos of layers and video segments, HDRIs of light layers."""
    out = []
    if not isinstance(doc, dict):
        return out
    au = str(((doc.get("timeline") or {}).get("audio") or {}).get("file") or "")
    if au and os.path.isfile(os.path.join(folder_paths.get_input_directory(), au)):
        out.append(_stamp(au, os.path.join(folder_paths.get_input_directory(), au)))
    pmsrc = str(((doc.get("projection_mask") or {}) if isinstance(doc.get("projection_mask"), dict) else {}).get("source") or "")
    pmp = _local_file(pmsrc[5:]) if pmsrc.startswith("path:") else _input_path(pmsrc[5:]) if pmsrc.startswith("file:") else None
    if pmp:                                       # the mask file can change on disk without the document changing
        out.append(_stamp(f"pm|{pmsrc}", pmp))
    for L in doc.get("layers", []):
        src = str(L.get("source") or "")
        refs = [src] + [str(s.get("file") or "") for s in ((L.get("media") or {}).get("segments") or []) if isinstance(s, dict)]
        for ref in refs:                          # images in input/, videos in input/ or linked by path
            if md.is_video(ref):
                p = _source_video(ref)
            else:
                p = os.path.join(folder_paths.get_input_directory(), ref[5:]) if ref.startswith("file:") else None
            if p and os.path.isfile(p):
                out.append(_stamp(ref, p))
    return out


def _hdri_stamps(doc):
    out = []
    for L in (doc or {}).get("layers", []) if isinstance(doc, dict) else []:
        f = str(((L.get("light") if isinstance(L, dict) and isinstance(L.get("light"), dict) else {}) or {}).get("hdri_file") or "")
        p = _local_file(f) or (_input_path(f) if f else None)
        if p:
            out.append(_stamp(f"hdri|{f}", p))
    return out


def pixel_doc(doc, skip_h3=False):
    """The document without what only the window uses (playhead, snap, diffusion settings, dismissed hints, H3
    prompt texts): two documents with the same pixel_doc give the same frames and export. Clips count only when
    skip_h3 hides frames under them, and then only their timing."""
    d = json.loads(json.dumps(doc or {}))
    if not isinstance(d, dict):
        return d
    for k in UI_DOC_KEYS:
        d.pop(k, None)
    tl = d.get("timeline")
    if isinstance(tl, dict):
        for k in UI_TIMELINE_KEYS:
            tl.pop(k, None)
        clips = tl.pop("clips", None)
        if skip_h3 and isinstance(clips, list):
            tl["clips"] = [{k: c.get(k) for k in CLIP_TIMING_KEYS if k in c} if isinstance(c, dict) else c for c in clips]
    return d


def array_digest(a):
    """Content hash of an array / tensor; read-only arrays (decoded imports) are hashed once."""
    if a is None:
        return "-"
    if isinstance(a, torch.Tensor):
        a = a.detach().cpu().numpy()
    a = np.asarray(a)
    ro = not a.flags.writeable
    if ro:
        hit = _DIGESTS.get(id(a))
        if hit is not None and hit[0]() is a:
            return hit[1]
    h = hashlib.blake2b(digest_size=16)
    h.update(f"{a.dtype}|{a.shape}|".encode())
    h.update(memoryview(np.ascontiguousarray(a)).cast("B"))
    d = h.hexdigest()
    if ro:
        import weakref
        try:
            _DIGESTS[id(a)] = (weakref.ref(a), d)
        except TypeError:
            pass
        if len(_DIGESTS) > 256:
            for k_ in [k_ for k_, v in _DIGESTS.items() if v[0]() is None]:
                _DIGESTS.pop(k_, None)
    return d


def _scene_ident(scene):
    if scene is None:
        return "-"
    parts = [json.dumps(scene, sort_keys=True, default=str)]
    if isinstance(scene, dict):
        for p in (str(scene.get("file") or ""), os.path.join(str(scene.get("folder") or ""), "faceid.npy")):
            if p and os.path.isfile(p):
                parts.append(_stamp(p, p))
    return "|".join(parts)


def content_key(doc, base, sources, labels, table, scene, W, H, pmask, skip_h3=False):
    """What a sequence / an export of the document is made of -> hex digest. Sources that a sequence never reads are
    left out: video frames at the playhead ('media:'; every frame decodes its own), background cutouts ('#nobg'; made
    from their source, which counts) and the still's light renders ('light:'; every frame relights) unless another
    layer uses that light layer as a picture (a mask / projector / glow source reads the still's render where the
    light layer is hidden)."""
    h = hashlib.blake2b(digest_size=20)
    h.update(json.dumps(pixel_doc(doc, skip_h3), sort_keys=True, default=str).encode("utf-8", "replace"))
    for s in file_stamps(doc) + _hdri_stamps(doc):
        h.update(("\n" + s).encode("utf-8", "replace"))
    refs = _layer_refs(doc)
    for k_ in sorted(sources or {}):
        if k_.startswith("media:") or k_.endswith("#nobg") or (k_.startswith("light:") and k_[6:] not in refs):
            continue
        h.update(f"\nsrc|{k_}|{array_digest(sources[k_])}".encode("utf-8", "replace"))
    h.update(f"\nbase|{array_digest(base)}|labels|{array_digest(labels)}|pmask|{array_digest(pmask)}|{W}x{H}".encode())
    h.update(("\ntable|" + json.dumps(table, sort_keys=True, default=str)).encode("utf-8", "replace"))
    h.update(("\nscene|" + _scene_ident(scene)).encode("utf-8", "replace"))
    return h.hexdigest()


def _cache_budget():
    """Bytes the sequence cache may hold: a quarter of the RAM (8 GB when unknown)."""
    try:
        import psutil
        return int(psutil.virtual_memory().total * 0.25)
    except Exception:  # noqa: BLE001
        return 8 << 30


def cached_sequence(key, make):
    """make() -> (frames tensor, report), or the result of an earlier run with the same key -> (frames, report, hit).
    Keeps the last SEQ_CACHE_ITEMS results (and at most a quarter of the RAM)."""
    hit = _SEQ_CACHE.pop(key, None)
    if hit is not None:
        _SEQ_CACHE[key] = hit
        return hit[0], hit[1], True
    frames, report = make()
    if SEQ_CACHE_ITEMS > 0:
        budget = _cache_budget()
        size = lambda t: t.element_size() * t.nelement()     # noqa: E731
        if size(frames) <= budget:
            _SEQ_CACHE[key] = (frames, report)
            while len(_SEQ_CACHE) > SEQ_CACHE_ITEMS or sum(size(v[0]) for v in _SEQ_CACHE.values()) > budget:
                _SEQ_CACHE.pop(next(iter(_SEQ_CACHE)))
    return frames, report, False


def cached_export(key, write):
    """write(written) -> report line (written: a list it fills with the folder and files), unless an export with the same content and settings was written this session and its
    files are all still there: then a report pointing to it (delete that folder, or change a setting, to write again)."""
    hit = _EXPORTS.get(key)
    if hit is not None and hit[1] and all(os.path.exists(f) for f in hit[1]):
        return (f"export unchanged since the last one, not written again: {hit[0]} (delete that folder or change a "
                f"setting to write it again)")
    written = []
    rep = write(written)
    if written:
        _EXPORTS[key] = (written[0], written, rep)
        while len(_EXPORTS) > 16:
            _EXPORTS.pop(next(iter(_EXPORTS)))
    return rep


def h3_after(prompt, node_id):
    """True / False: a kubakub keyframe clips (h3) node takes this node's outputs (directly or further down the
    graph); None when the prompt is not known (then skip_h3_frames is taken as it is)."""
    if not isinstance(prompt, dict) or node_id is None:
        return None
    users = {}
    for nid, n in prompt.items():
        for v in ((n or {}).get("inputs") or {}).values() if isinstance(n, dict) else []:
            if isinstance(v, list) and len(v) == 2 and isinstance(v[1], int):
                users.setdefault(str(v[0]), set()).add(str(nid))
    seen, todo = set(), [str(node_id)]
    while todo:
        for u in users.get(todo.pop(), ()):
            if u not in seen:
                seen.add(u)
                todo.append(u)
    return any((prompt.get(u) or {}).get("class_type") == "KUBA_KeyframeClips" for u in seen)


def sequence_frames(doc, base, sources, labels, table, scene, W, H, scale, skip_h3, pmask, h3_node=None, light_scale=1.0):
    """The document's frames through the content cache -> (frames tensor, report). skip_h3 only when a keyframe
    clips (h3) node fills the frames (h3_node True) or when that is not known (None)."""
    lines = []
    has_hidden = bool(skip_h3) and bool(kf.hidden_frames(doc, rd.timeline(doc)["frames"], rd.timeline(doc)["fps"]))
    skip = has_hidden and h3_node is not False
    if has_hidden and not skip:
        lines.append("skip_h3_frames is on, but no kubakub keyframe clips (h3) node takes these frames: every frame "
                     "rendered (nothing would replace the frames under the H3 clips)")
    t0 = time.perf_counter()
    key = ("seq", content_key(doc, base, sources, labels, table, scene, W, H, pmask, skip), float(scale), skip,
           float(light_scale))

    def make():
        notes = []
        arr, _, rep_ = _render_sequence(json.loads(json.dumps(doc)), base, dict(sources), labels, table, scene,
                                        W, H, float(scale), notes, skip_h3=skip, pmask=pmask, light_scale=light_scale)
        return torch.from_numpy(arr), "\n".join([rep_, *notes])

    frames, rep, hit = cached_sequence(key, make)
    if hit:
        rep = (f"sequence unchanged (only window state changed): the frames of the last run, "
               f"{time.perf_counter() - t0:.2f} s\n" + rep)
    return frames, "\n".join(lines + [rep])


def export_cached(fmt, scale, alpha, name, doc, base, sources, labels, table, scene, W, H, pmask, audio):
    """_export through the export cache (see cached_export)."""
    key = ("export", content_key(doc, base, sources, labels, table, scene, W, H, pmask), fmt, float(scale), bool(alpha),
           str(name), array_digest(audio["waveform"]) if audio is not None else "-",
           int(audio["sample_rate"]) if audio is not None else 0)
    return cached_export(key, lambda written: _export(fmt, scale, alpha, name, doc, base, sources, labels, table, scene, W, H,
                                                     pmask, audio, written))


class KUBA_Export(io.ComfyNode):
    """Any frames (e.g. after 'kubakub keyframe clips (h3)') into a delivery file, the same formats as the director."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="KUBA_Export",
            display_name="kubakub export video / frames",
            category="kubakub/2d/post",
            search_aliases=['prores', 'png sequence', 'h264', 'delivery'],
            is_output_node=True,
            description="Writes frames as a delivery file: PNG sequence 8/16 bit, ProRes 4444 / 422 HQ, H.264, H.265 or a "
                        "preview, Rec.709 tagged, with sound. Into output/kubakub_director/<name>_<time>/.",
            inputs=[
                io.Image.Input("images", tooltip="The frames to write (a batch; one image = a one-frame file)."),
                io.Float.Input("fps", default=25.0, min=1.0, max=120.0, step=0.001,
                               tooltip="Frames per second of the file (e.g. 25, 30 or 29.97)."),
                io.Combo.Input("format", options=list(exm.FORMATS), default="prores422hq",
                               tooltip="png8 / png16 = PNG sequence + .wav; prores4444 (with alpha) / prores422hq = .mov; "
                                       "h264 / h264_444 / h265 = .mp4; preview = a small half-size .mp4."),
                io.String.Input("name", default="export",
                                tooltip="Name of the export folder and files (a date and time is added to the folder)."),
                io.Audio.Input("audio", optional=True,
                               tooltip="Sound for the file (PNG sequences get it as a .wav next to the frames)."),
                io.Mask.Input("alpha", optional=True, tooltip="Alpha for PNG / ProRes 4444 (one mask or one per frame)."),
            ],
            outputs=[io.String.Output("report", tooltip="Format, frame count, size, the output folder and the written files.")],
        )

    @classmethod
    def execute(cls, images, fps, format, name, audio=None, alpha=None) -> io.NodeOutput:
        t0 = time.perf_counter()
        out_dir, safe = _export_dir(name)
        n, h, w = images.shape[0], images.shape[1], images.shape[2]
        wave = audio["waveform"][0].float().cpu().numpy() if audio is not None else None
        writer = exm.Writer(format, out_dir, safe, w - w % 2, h - h % 2, fps, alpha=alpha is not None, audio=wave,
                            sr=int(audio["sample_rate"]) if audio is not None else 44100)

        def prep(i):
            a = None
            if alpha is not None:
                m = alpha if alpha.ndim == 2 else alpha[min(i, alpha.shape[0] - 1)]
                a = m.float().cpu().numpy()[:h - h % 2, :w - w % 2]
            return writer.prep(i, images[i, :h - h % 2, :w - w % 2, :3].float().cpu().numpy(), a)

        from concurrent.futures import ThreadPoolExecutor   # quantise / PNG encode in parallel, written in order
        workers = max(1, min(8, (os.cpu_count() or 4) - 2))
        try:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                for got in exm.ordered(pool, prep, range(n), 2 * workers):
                    writer.put(got)
        finally:
            files = writer.close()
        rep = f"{exm.FORMATS[format]['label']}: {n} frames {w - w % 2}x{h - h % 2} -> {out_dir} ({time.perf_counter() - t0:.0f} s)"
        log.info("[KUBA export] %s", rep)
        return io.NodeOutput(rep + "\n" + "\n".join(files))


def _light_layer(doc, layer_id, sources, labels, table, W, H, scene, scale=None, samples=None, out_size=None,
                 projector=None):
    """Relight one light layer of the document -> (rgba at out_size or W x H, report). Glow masks always at W x H."""
    from ..scene3d.nodes_scene3d import relight_scene
    from ...kubakub.scene3d import scene_view as sv
    raw = next((L for L in doc.get("layers", []) if isinstance(L, dict) and str(L.get("id")) == layer_id), None)
    if raw is None or raw.get("kind") != "light":
        raise ValueError(f"no light layer '{layer_id}' in the document")
    rig = sv.rig_from_doc(raw.get("light"))
    emission = rd.glow_masks(doc, rig, sources, labels, table, W, H)
    rgb, alpha, report = relight_scene(scene, rig, emission=emission, size=out_size or (W, H), scale=scale, samples=samples,
                                       projector=projector)
    a = alpha if rig["background"] == "transparent" else np.ones_like(alpha)
    return np.concatenate([rgb, a[..., None]], -1).astype(np.float32), report


def _preview_light(uid, last, doc, layer_id, px, samples, projector=None):
    W, H = last["W"], last["H"]
    k = min(1.0, px / max(W, H))
    pw, ph = max(16, round(W * k)), max(16, round(H * k))
    sources = _doc_sources(doc, last["sources"])
    doc = json.loads(json.dumps(doc))
    if any(isinstance(L, dict) and L.get("motion") for L in doc.get("layers", [])):
        doc = rd.animate(doc, rd.timeline(doc)["time"], motion_context(doc))   # behaviours at the playhead, as the node renders
    media_notes = []                                    # video layers at the playhead (glow / projector 'layer:<video>')
    sources = MediaFrames({0: (doc, rd.timeline(doc)["time"])}, 1.0, media_notes).sources(0, sources)
    for L in doc.get("layers", []):                     # background removed on apply: glow with the cutout
        cut = str(L.get("source") or "") + "#nobg"
        if L.get("remove_bg") and cut in sources:
            L["source"] = cut
    # glow masks at full size (as on apply: the same faces glow), the render itself at preview size
    rgba, report = _light_layer(doc, layer_id, sources, last["labels"], last["table"], W, H, last["scene"], scale=1.0,
                                samples=samples, out_size=(pw, ph), projector=projector)
    out_dir = os.path.join(folder_paths.get_temp_directory(), "kuba_director", uid)
    os.makedirs(out_dir, exist_ok=True)
    safe = re.sub(r"[^\w]+", "_", layer_id)
    fn = f"light_{safe}_{hashlib.sha1(rgba.tobytes()).hexdigest()[:12]}.png"
    _png(os.path.join(out_dir, fn), rgba)
    return {"filename": fn, "subfolder": f"kuba_director/{uid}", "type": "temp"}, report


def _register_routes():
    """POST /kubakub/director/remove_bg {image: {filename, subfolder, type}} -> the foreground mask (temp PNG)."""
    try:
        from aiohttp import web
        from server import PromptServer
    except ImportError:
        return

    async def remove_bg(request):
        import asyncio
        try:
            ref = (await request.json()).get("image") or {}
            root = folder_paths.get_input_directory() if ref.get("type") == "input" else folder_paths.get_temp_directory()
            root = os.path.realpath(root)
            path = os.path.realpath(os.path.join(root, ref.get("subfolder", ""), ref.get("filename", "")))
            if not path.startswith(root + os.sep) or not os.path.isfile(path):
                return web.json_response({"error": "image not found"}, status=404)
            img = imio.imread(path, cv2.IMREAD_COLOR)
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255
            out_dir = os.path.join(folder_paths.get_temp_directory(), "kuba_director", "masks")
            os.makedirs(out_dir, exist_ok=True)
            fn = hashlib.sha1(rgb.tobytes()).hexdigest()[:16] + ".png"
            if not os.path.isfile(os.path.join(out_dir, fn)):      # same pixels: the mask made last time
                from ...kubakub.director import bg
                mask = await asyncio.get_running_loop().run_in_executor(None, bg.mask_for, rgb)
                imio.imwrite(os.path.join(out_dir, fn), np.clip(mask * 255 + 0.5, 0, 255).astype(np.uint8))
            return web.json_response({"mask": {"filename": fn, "subfolder": "kuba_director/masks", "type": "temp"}})
        except Exception as e:  # noqa: BLE001
            log.exception("[KUBA director] remove_bg failed")
            return web.json_response({"error": (str(e).splitlines() or [""])[0]}, status=500)

    async def relight(request):
        """{node, layer: light layer id, doc, px, samples} -> a quick Cycles preview of that light layer (temp PNG)."""
        import asyncio
        try:
            body = await request.json()
            uid = re.sub(r"[^\w.\-]+", "_", str(body.get("node") or ""))
            last = _LAST.get(uid)
            if last is None or last.get("scene") is None:
                return web.json_response({"error": "run the workflow once with a scene connected to the director"}, status=409)
            doc = body.get("doc") or {}
            if _doc_names_paths(doc) and not _route_paths_ok():
                return web.json_response({"error": PATHS_OFF}, status=403)
            for L in doc.get("layers") or []:          # a light's HDRI: a file on this computer, never a network share
                lt = L.get("light") if isinstance(L, dict) and isinstance(L.get("light"), dict) else None
                if lt and str(lt.get("hdri_file") or "").strip() and not _local_file(lt["hdri_file"]):
                    return web.json_response({"error": "hdri_file: not a file on a local drive"}, status=400)
            px = int(min(max(float(body.get("px") or 800), 128), 4096))
            samples = int(min(max(float(body.get("samples") or 16), 1), 1024))
            projector = None
            data = str(body.get("projector") or "")            # the window's own composite of what is cast (PNG data URL)
            if data:
                import base64
                raw = np.frombuffer(base64.b64decode(data.split(",", 1)[-1]), np.uint8)
                img = cv2.imdecode(raw, cv2.IMREAD_COLOR)
                if img is None:
                    return web.json_response({"error": "projector image could not be read"}, status=400)
                projector = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255
            loop = asyncio.get_running_loop()
            async with _RELIGHT_LOCK:                 # one Blender at a time; the window keeps only the newest answer
                ref, report = await loop.run_in_executor(None, _preview_light, uid, last, doc, str(body.get("layer") or ""), px, samples, projector)
            return web.json_response({"image": ref, "report": report})
        except Exception as e:  # noqa: BLE001
            log.exception("[KUBA director] relight preview failed")
            return web.json_response({"error": (str(e).splitlines() or [""])[0]}, status=500)

    async def media(request):
        """{source: 'file:kuba_director/x.mov' or 'path:D:/.../x.mov'} -> {info, proxy}: a small browser-playable copy (made once
        per file version, kept in temp/kuba_director/proxies)."""
        import asyncio
        try:
            name = str((await request.json()).get("source") or "")        # 'file:<name in input/>' or 'path:<file>'
            if name.startswith("path:") and not _route_paths_ok():
                return web.json_response({"error": PATHS_OFF}, status=403)
            path = _source_video(name)
            if not path:
                return web.json_response({"error": "video not found: " + name[5:]}, status=404)
            info = md.probe(path)
            st = os.stat(path)
            # PROXY_VERSION: proxies made by an older media.py (e.g. before rotated phone clips) are made again
            key = hashlib.sha1(f"{PROXY_VERSION}|{path}|{st.st_size}|{st.st_mtime_ns}".encode("utf-8", "replace")).hexdigest()[:16]
            out_dir = os.path.join(folder_paths.get_temp_directory(), "kuba_director", "proxies")
            os.makedirs(out_dir, exist_ok=True)
            ready = lambda: [f for f in (key + ".mp4", key + ".webm") if os.path.isfile(os.path.join(out_dir, f))]  # noqa: E731
            done = ready()                              # finished proxies only (never a <key>.tmp.* being written)
            if not done:
                async with _PROXY_LOCK:                 # one transcode at a time; a waiting request reuses the result
                    done = ready()
                    if not done:
                        done = [os.path.basename(await asyncio.get_running_loop().run_in_executor(
                            None, md.make_proxy, path, os.path.join(out_dir, key + ".part")))]
            fn = done[0]
            return web.json_response({"info": {k: info[k] for k in ("w", "h", "fps", "duration", "frames", "alpha", "audio", "codec")},
                                      "proxy": {"filename": fn, "subfolder": "kuba_director/proxies", "type": "temp"}})
        except Exception as e:  # noqa: BLE001
            log.exception("[KUBA director] media proxy failed")
            return web.json_response({"error": str(e).splitlines()[0]}, status=500)

    async def pmask(request):
        """{node, pm: {source, invert, grow, feather}, pw, ph} -> the projection mask as the render reads it, at the
        window's preview size (temp PNG, white = building)."""
        import asyncio
        try:
            body = await request.json()
            uid = str(body.get("node") or "")
            last = _LAST.get(uid)
            if last is None:
                return web.json_response({"error": "run the workflow once, then set the projection mask"}, status=409)
            notes = []
            doc = {"projection_mask": dict(body.get("pm") or {}, on=True)}
            if _doc_names_paths(doc) and not _route_paths_ok():
                return web.json_response({"error": PATHS_OFF}, status=403)
            m = await asyncio.get_running_loop().run_in_executor(None, projection_mask, doc, last["scene"], last["W"], last["H"], notes)
            if m is None:
                return web.json_response({"error": "; ".join(notes) or "no mask"}, status=404)
            pw, ph = (min(max(1, int(body.get(k) or last[d])), 8192) for k, d in (("pw", "W"), ("ph", "H")))   # a preview, never huge
            small = cv2.resize(m, (pw, ph), interpolation=cv2.INTER_AREA)
            out_dir = os.path.join(folder_paths.get_temp_directory(), "kuba_director", uid)
            os.makedirs(out_dir, exist_ok=True)
            png = np.clip(small * 255 + 0.5, 0, 255).astype(np.uint8)
            fn = f"pmask_{hashlib.sha1(png.tobytes()).hexdigest()[:12]}.png"
            imio.imwrite(os.path.join(out_dir, fn), png)
            return web.json_response({"image": {"filename": fn, "subfolder": f"kuba_director/{uid}", "type": "temp"},
                                      "note": "; ".join(notes)})
        except Exception as e:  # noqa: BLE001
            log.exception("[KUBA director] projection mask preview failed")
            return web.json_response({"error": str(e).splitlines()[0]}, status=500)

    async def level(request):
        """{file} (in input/) -> {"rate": 100, "level": [...]}: the loudness curve the node animates audio behaviours with.
        {file, bands: true} -> also "bands" {low / mid / high: [...]} and "hits" {low / mid / high: [[t, strength]]}."""
        import asyncio
        try:
            body = await request.json()
            lv = await asyncio.get_running_loop().run_in_executor(None, level_of, body.get("file"))
            if lv is None:
                return web.json_response({"error": "sound not found"}, status=404)
            out = {"rate": mo.LEVEL_RATE, "level": [round(v, 5) for v in lv]}
            if body.get("bands"):
                s = await asyncio.get_running_loop().run_in_executor(None, sound_of, body.get("file"))
                out.update(bands=s["curves"], hits=s["hits"])
            return web.json_response(out)
        except Exception as e:  # noqa: BLE001
            log.exception("[KUBA director] level curve failed")
            return web.json_response({"error": str(e).splitlines()[0]}, status=500)

    try:
        PromptServer.instance.routes.post("/kubakub/director/level")(level)
        PromptServer.instance.routes.post("/kubakub/director/pmask")(pmask)
        PromptServer.instance.routes.post("/kubakub/director/remove_bg")(remove_bg)
        PromptServer.instance.routes.post("/kubakub/director/relight")(relight)
        PromptServer.instance.routes.post("/kubakub/director/media")(media)
    except (AttributeError, RuntimeError) as e:
        log.warning("[KUBA director] routes not registered: %s", e)


_register_routes()


class KUBA_Director(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="KUBA_Director",
            display_name="kubakub director",
            category="kubakub/2d/director",
            search_aliases=['projection mapping', 'layers', 'compositor', 'timeline', 'lights'],
            is_output_node=True,
            description=(
                "Compose the projection by hand in a layer window: place, scale, rotate and stack images over the facade, "
                "decide what is in front of what, mask layers with other layers, paint, colour grade, import "
                "images. Run once, then 'open director'. Apply saves the composition and queues the workflow; "
                "the node renders it at full size and returns the image, per-layer masks, a changed mask, the "
                "regions with the placed layers and plan rules for the next diffusion pass."),
            inputs=[
                io.Image.Input("image", tooltip="The base layer: the matrix, a render or a photo of the facade."),
                io.Image.Input("layers", optional=True,
                               tooltip="Images to place (a batch = several layers; different sizes: use several "
                                       "director inputs later or import in the window)."),
                io.Mask.Input("layer_masks", optional=True,
                              tooltip="Masks for the layer images (1 = keep). Masks without images become "
                                      "white layers you can use as masks for other layers."),
                io.Boolean.Input("invert_masks", default=False, optional=True,
                                 tooltip="On for Load Image masks (they are 1 where the picture is transparent)."),
                RegionsType.Input("regions", optional=True,
                                  tooltip="Regions to pick, clip to and cut holes with; placed layers are added."),
                SceneType.Input("scene", optional=True, tooltip="3D scene: clay, depth and ID views in the window."),
                io.String.Input("layer_names", default="", optional=True,
                                tooltip="Names for the layer images, one per line or comma separated."),
                io.Int.Input("seam_px", default=24, min=1, max=512, optional=True,
                             tooltip="Width of the edge band in the changed mask."),
                io.String.Input("document", multiline=True, default="", optional=True,
                                tooltip="The composition, written by the window (JSON). Leave it to the window."),
                io.Boolean.Input("render_sequence", default=False, optional=True, advanced=True,
                                 tooltip="Also render every frame of the timeline (frames / fps outputs, e.g. into "
                                         "Create Video). Light layers render in Cycles per frame where they change. Kept for older workflows: the kubakub director sequence node does this now."),
                io.Boolean.Input("skip_h3_frames", default=False, optional=True, advanced=True,
                                 tooltip="Do not render the frames that H3 keyframe clips replace anyway (a dissolve "
                                         "stands in for them). Off for the full sequence without H3. Kept for older workflows: the kubakub director sequence node does this now."),
                io.Combo.Input("export", options=["none", *exm.FORMATS], default="none", optional=True, advanced=True,
                               tooltip="Write the whole timeline as a file at the delivery size, frame by frame (no RAM limit): "
                                       "PNG sequence 8/16 bit, ProRes 4444 / 422 HQ, H.264 (4:2:0 8 bit or 4:4:4 10 bit), "
                                       "H.265 10 bit, or a small preview. Rec.709 tagged, with the timeline sound. Into "
                                       "output/kubakub_director/<name>_<time>/. The window's export button uses this."),
                io.Float.Input("export_scale", default=1.0, min=0.1, max=1.0, step=0.05, optional=True, advanced=True,
                               tooltip="1 = the delivery size (the matrix)."),
                io.Boolean.Input("export_alpha", default=False, optional=True, advanced=True,
                                 tooltip="The layers without the base, transparent where they do not cover (and outside "
                                         "the projection mask): PNG sequences and ProRes 4444."),
                io.String.Input("export_name", default="director", optional=True, advanced=True, tooltip="Name of the export folder and files."),
                io.Float.Input("sequence_scale", default=0.5, min=0.1, max=1.0, step=0.05, optional=True, advanced=True,
                               tooltip="Size of the sequence frames relative to the matrix (RAM: 250 frames at "
                                       "1600x1080 = ~5 GB). Kept for older workflows: the kubakub director sequence node does this now."),
                io.Combo.Input("flow", options=['hold until apply', 'always'], default="hold until apply", optional=True,
                               tooltip="hold until apply: until a composition is applied from the window, the director only "
                                       "shows its preview (open it, compose, apply) and the nodes after it wait - no LTX / "
                                       "H3 / sequence run on an empty composition. always: the base passes on right away. "
                                       "Once a composition is applied the setting changes nothing, but switching it "
                                       "still counts as a change for ComfyUI (the nodes after the director run again)."),
            ],
            outputs=[
                io.Image.Output("image", tooltip="The composition at full size."),
                io.Mask.Output("layer_masks", tooltip="Per placed layer: where the audience sees it (batch)."),
                io.Mask.Output("changed", tooltip="Re-diffused layers + edge bands: for region sampler / seam pass."),
                RegionsType.Output("regions", tooltip="The input regions with every placed layer as a new region."),
                io.String.Output("plan_rules", tooltip="Rules for kubakub region plan (actions, prompts, denoise)."),
                io.String.Output("document", tooltip="The composition document (JSON)."),
                io.String.Output("report", tooltip="Size, layers and what was placed; sequence, sound and export notes "
                                                   "when those run; 'waiting for apply' while the flow holds."),
                io.Image.Output("frames", tooltip="Every frame of the timeline (render_sequence on), else the image. "
                                                  "For video use the kubakub director sequence node."),
                io.Float.Output("fps", tooltip="Frames per second of the timeline."),
                io.Audio.Output("audio", tooltip="The timeline's sound, trimmed to its length (silence without one): "
                                                 "frames + fps + audio into Create Video."),
                io.Mask.Output("projection_mask", tooltip="The building's silhouette from the projection mask slot "
                                                          "(1 = building; all ones when the slot is off)."),
                DirectorType.Output("director", tooltip="The composition with its sources: into kubakub director sequence "
                                                         "(frames, sound, export)."),
            ],
            hidden=[io.Hidden.unique_id, io.Hidden.prompt, io.Hidden.extra_pnginfo],
        )

    @classmethod
    def fingerprint_inputs(cls, document="", **kwargs):
        # uploaded files (paint layers, imports) change without the document text changing
        h = hashlib.sha256((document or "").encode("utf-8", "replace"))
        # the window's preview copies live in ComfyUI/temp (emptied at every ComfyUI start): run again when gone
        from ...kubakub.scene3d.bridge import folder_token
        h.update(folder_token(os.path.join(folder_paths.get_temp_directory(), "kuba_director")))
        try:                                      # an edited HDRI file re-runs the node (the path text alone does not change)
            from ..scene3d.nodes_scene3d import hdri_stamp
            for L in (json.loads(document) if (document or "").strip() else {}).get("layers", []):
                lt = L.get("light") if isinstance(L, dict) and L.get("kind") == "light" else None
                if isinstance(lt, dict) and lt.get("environment") == "file":
                    h.update(hdri_stamp(lt.get("hdri_file", "")).encode())
        except (ValueError, AttributeError, ImportError):
            pass
        try:
            for s in file_stamps(json.loads(document) if (document or "").strip() else {}):
                h.update(s.encode())
        except (ValueError, AttributeError, OSError):
            pass
        return h.hexdigest()

    @classmethod
    def execute(cls, image, layers=None, layer_masks=None, invert_masks=False, regions=None, scene=None,
                layer_names="", seam_px=24, document="", flow="hold until apply", render_sequence=False, sequence_scale=0.5,
                skip_h3_frames=False, export="none", export_scale=1.0, export_alpha=False, export_name="director") -> io.NodeOutput:
        t0 = time.perf_counter()
        if scene is not None and '"kind": "light"' in (document or "").replace('"kind":"light"', '"kind": "light"'):
            try:                                  # light layers: Blender starts while the rest is prepared
                from ..scene3d.nodes_scene3d import prestart_relight
                prestart_relight(scene)
            except Exception as e:  # noqa: BLE001
                log.info("[KUBA director] Blender pre-start skipped: %s", e)
        base = image[0, ..., :3].cpu().float().numpy()
        H, W = base.shape[:2]
        k = min(1.0, PROXY_W / max(W, H))
        hid = getattr(cls, "hidden", None)
        # node id + workflow: two workflows with a director of the same node id (two tabs) keep their own preview
        # state and temp folder; the window sends this key back (manifest 'node') for relight / mask previews
        uid = state_key(getattr(hid, "unique_id", None), getattr(hid, "extra_pnginfo", None))
        sub = f"kuba_director/{uid}"
        out_dir = os.path.join(folder_paths.get_temp_directory(), "kuba_director", uid)
        os.makedirs(out_dir, exist_ok=True)
        stamp = str(int(time.time() * 1000))
        ref = lambda fn: {"filename": fn, "subfolder": sub, "type": "temp"}  # noqa: E731

        # sources: input:i = layer image i (+ mask as alpha); masks without images = white layers
        names = [n.strip() for n in layer_names.replace(",", "\n").splitlines() if n.strip()]
        sources, inputs = {}, []
        imgs = [] if layers is None else list(layers[..., :3].cpu().float().numpy())
        msks = [] if layer_masks is None else list((layer_masks if layer_masks.ndim == 3 else layer_masks[None]).cpu().float().numpy())
        n_in = max(len(imgs), len(msks))
        for i in range(n_in):
            rgb = imgs[i] if i < len(imgs) else (np.ones(msks[i].shape + (3,), np.float32))
            a = msks[i] if i < len(msks) else np.ones(rgb.shape[:2], np.float32)
            if invert_masks and i < len(msks):
                a = 1 - a
            if a.shape != rgb.shape[:2]:
                a = cv2.resize(a, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
            rgba = np.concatenate([rgb, a[..., None]], -1).astype(np.float32)
            key = f"input:{i}"
            sources[key] = rgba
            fn = _png_once(out_dir, f"in_{i}", stamp, _proxy(rgba, k))     # unchanged input: last run's file
            inputs.append({"key": key, "name": names[i] if i < len(names) else (f"layer {i + 1}" if i < len(imgs) else f"mask {i + 1}"),
                           "w": int(rgba.shape[1]), "h": int(rgba.shape[0]), "image": ref(fn), "is_mask": i >= len(imgs)})
        doc_obj = {}
        if (document or "").strip():
            try:
                doc_obj = json.loads(document)
            except ValueError as e:
                raise ValueError(f"kubakub director: the saved composition is not valid JSON ({e}). Undo the edit "
                                 "or clear the node's document to start fresh.") from e
        for L in doc_obj.get("layers", []) if isinstance(doc_obj, dict) else []:
            src = str(L.get("source") or "")
            if src.startswith("file:") and src not in sources:
                img = _load_input_file(src[5:])
                if img is not None:
                    sources[src] = img

        # views for the window: the base, and the scene's clay / depth / ids when connected
        views = {}
        views["matrix"] = ref(_png_once(out_dir, "base", stamp, _proxy(base, k)))    # the same pixels: last run's file
        labels = table = None
        box_scale = (1.0, 1.0)             # region bboxes are in the labels' own resolution
        if regions is not None:
            labels = regions.labels[0].cpu().numpy().astype(np.int32)
            table = regions.table
            box_scale = (W / labels.shape[1], H / labels.shape[0])
            table = {**table, "canvas_size": [W, H],      # region boxes on the canvas, rounded like the window's (stagger order)
                     "canvas_bbox": {g["region_id"]: [round(float(v) * box_scale[i % 2]) for i, v in enumerate(g.get("bbox", [0, 0, 0, 0]))]
                                     for g in table.get("regions", [])}}
            if labels.shape != (H, W):
                labels = cv2.resize(labels, (W, H), interpolation=cv2.INTER_NEAREST)
            views["_labels"] = ref(_png_once(out_dir, "labels", stamp, _label_png(cv2.resize(
                labels, (max(1, round(W * k)), max(1, round(H * k))), interpolation=cv2.INTER_NEAREST))))
        if scene is not None:
            made = {}
            try:                                  # the scene views depend on the scene files only: made once
                vkey = _scene_views_key(scene, out_dir, W, H, k)
                hit = _VIEWS.get(vkey)
                if hit is not None and all(os.path.isfile(os.path.join(out_dir, f)) for f in hit.values()):
                    made = hit
                else:
                    _scene_views(scene, out_dir, stamp, W, H, k, made)
                    while len(_VIEWS) >= 8:
                        _VIEWS.pop(next(iter(_VIEWS)))
                    _VIEWS[vkey] = made
            except (OSError, KeyError, cv2.error) as e:
                log.warning("[KUBA director] scene views skipped: %s", e)
            views.update({name: ref(f) for name, f in made.items()})
        scene_info = None
        if scene is not None:
            try:
                from ..scene3d.nodes_scene3d import load_scene_cached
                from ...kubakub.scene3d import scene_view as sv
                fr = load_scene_cached(scene)[4]
                scene_info = {"frame": {k_: float(fr[k_]) for k_ in ("width_m", "height_m", "bottom_m", "top_m")},
                              "environments": list(sv.ENVIRONMENTS), "defaults": sv.RIG_DEFAULTS}
            except Exception as e:  # noqa: BLE001
                log.warning("[KUBA director] scene frame not available, no light layers: %s", e)

        # render the document (an empty one = the base alone)
        render_doc = json.loads(json.dumps(doc_obj)) if doc_obj else ""
        for L in render_doc.get("layers", []) if render_doc else []:
            src = str(L.get("source") or "")
            if L.get("remove_bg") and L.get("kind") == "image" and src in sources:
                cut = src + "#nobg"
                if cut not in sources:
                    from ...kubakub.director import bg
                    rgba = sources[src]
                    sources[cut] = np.concatenate([rgba[..., :3], (rgba[..., 3] * bg.mask_for(rgba[..., :3]))[..., None]], -1)
                L["source"] = cut
        # video layers: the frame at the timeline's current time
        media_notes = []
        seq_doc = json.loads(json.dumps(render_doc)) if render_doc else render_doc   # the sequence decodes its own frames
        if render_doc and any(isinstance(L, dict) and L.get("motion") for L in render_doc.get("layers", [])):
            render_doc = rd.animate(render_doc, rd.timeline(render_doc)["time"], motion_context(render_doc))   # the still at the playhead
        if render_doc:
            t_now = rd.timeline(render_doc)["time"]
            sources = MediaFrames({0: (render_doc, t_now)}, 1.0, media_notes).sources(0, sources)
        # light layers: Cycles relight of the scene at full size, handed to the render as sources["light:<id>"]
        light_reports = []
        if doc_obj:
            from ...kubakub.scene3d import scene_view as sv
            for L in reversed(doc_obj.get("layers", [])):    # bottom up: a projector may cast a lower light layer
                if not isinstance(L, dict) or L.get("kind") != "light" or (L.get("visible") is False and str(L.get("id")) not in _lights_used(doc_obj)):
                    continue
                if scene is None:
                    continue                    # render() notes the layer as not rendered
                lid = str(L.get("id"))
                pj = sv.rig_from_doc(L.get("light"))["projector"]
                cast = _projector_image(render_doc or doc_obj, lid, pj["source"], base, sources, labels, table) if pj["on"] else None
                rgba, rep_ = _light_layer(render_doc or doc_obj, lid, sources, labels, table, W, H, scene, projector=cast)
                sources[f"light:{lid}"] = rgba
                light_reports.append(f"light '{L.get('name') or lid}': " + rep_.replace("\n", "; "))
                fn = _png_once(out_dir, f"light_{re.sub(r'[^A-Za-z0-9_]+', '_', lid)}", stamp, _proxy(rgba, k))
                views.setdefault("_lights", {})[lid] = {"image": ref(fn), "rig": L.get("light") or {}}
        _remember(uid, {"scene": scene, "W": W, "H": H, "labels": labels, "table": table,
                        "sources": {key: v for key, v in sources.items() if key.startswith("input:") or key.endswith("#nobg")}})

        tl0 = rd.timeline(render_doc) if render_doc else {"time": 0, "fps": 25}
        r = rd.render(render_doc, base, sources, labels, table, seam_px=int(seam_px), frame_seed=int(round(tl0["time"] * tl0["fps"])))
        pm_notes = []
        pmask = projection_mask(doc_obj, scene, W, H, pm_notes) if doc_obj else None
        if pmask is not None:
            r["image"] = pmk.apply(r["image"], pmask)
        comp = torch.from_numpy(r["image"])[None]
        masks = torch.zeros((max(1, len(r["layer_masks"])), H, W))     # the output format needs full masks
        for j, (_, _, (x0, y0, x1, y1), crop) in enumerate(r["layer_masks"]):   # not k: k is the proxy scale
            masks[j, y0:y1, x0:x1] = torch.from_numpy(crop)
        changed = torch.from_numpy(r["changed"])[None]
        out_regions = Regions.from_numpy(r["labels"], r["table"])
        if regions is not None and regions.scope is not None:           # the scope travels on: samplers clip to it
            sc = regions.scope.float().cpu()
            if tuple(sc.shape[-2:]) != tuple(out_regions.labels.shape[-2:]):
                sc = (torch.nn.functional.interpolate(sc[:, None], size=tuple(out_regions.labels.shape[-2:]), mode="area")[:, 0] > 0.5).float()
            out_regions = Regions(out_regions.labels, out_regions.table, sc)
        fn = _png_once(out_dir, "out", stamp, _proxy(r["image"], k))

        # clean older proxies of this director (node + workflow: only its own folder) after an hour
        now = time.time()
        keep = {f for key, v in _VIEWS.items() if key[0] == out_dir for f in v.values()}   # reused views / proxies
        keep |= {v[1] for key, v in _PROXIES.items() if key[0] == out_dir}
        for f in os.listdir(out_dir):
            fp = os.path.join(out_dir, f)
            try:
                if stamp not in f and f not in keep and now - os.path.getmtime(fp) > 3600:
                    os.remove(fp)
            except OSError:
                pass

        region_list = [{"id": g["region_id"], "name": g["name"], "group": g.get("group_id") or "",
                        "tags": [str(t) for t in g.get("tags") or []],
                        "bbox": [round(float(v) * box_scale[i % 2]) for i, v in enumerate(g.get("bbox", [0, 0, 0, 0]))]}
                       for g in (table or {}).get("regions", [])]
        manifest = {"canvas": [W, H], "proxy_scale": k, "views": views, "inputs": inputs, "regions": region_list,
                    "groups": sorted({g["group"] for g in region_list if g["group"]}), "blend_modes": list(MODES),
                    "upload_subfolder": UPLOAD_SUB, "node": uid, "result": ref(fn), "scene": scene_info,
                    "lights": views.pop("_lights", {}), "diffusion": diffusion_manifest()}
        placed = ", ".join(e[1] for e in r["layer_masks"]) or "none"
        report = "\n".join([f"{W}x{H}, {r['layer_count']} layers, placed: {placed}",
                            f"{len(r['table']['regions'])} regions out, changed {float(r['changed'].mean()) * 100:.1f} % of the image",
                            f"{time.perf_counter() - t0:.1f} s", *light_reports, *media_notes, *pm_notes,
                            *sorted({COLOUR_NOTES[str(L.get("source"))[5:]] for L in (doc_obj or {}).get("layers", [])
                                     if isinstance(L, dict) and str(L.get("source") or "")[5:] in COLOUR_NOTES}), *r["notes"]])
        log.info("[KUBA director] %s", report.replace("\n", "\n    "))
        ui_dict = ui.PreviewImage(comp, cls=cls).as_dict()
        ui_dict["kuba_director"] = [manifest]
        frames, fps = comp, float(rd.timeline(doc_obj)["fps"] if doc_obj else 25)
        if render_sequence and doc_obj:
            frames, seq_report = sequence_frames(seq_doc or doc_obj, base, sources, labels, table, scene, W, H, float(sequence_scale),
                                                 bool(skip_h3_frames), pmask,
                                                 h3_after(getattr(hid, "prompt", None), getattr(hid, "unique_id", None)))
            fps = float(rd.timeline(seq_doc or doc_obj)["fps"])
            report += "\n" + seq_report
            log.info("[KUBA director] %s", seq_report.replace("\n", "\n    "))
        audio, audio_note = timeline_audio(doc_obj)
        if audio_note:
            report += "\n" + audio_note
        if export and export != "none" and doc_obj:
            report += "\n" + export_cached(export, float(export_scale), bool(export_alpha), str(export_name or "director"), seq_doc or doc_obj,
                                     base, sources, labels, table, scene, W, H, pmask, audio)
            log.info("[KUBA director] %s", report.split("\n")[-1])
        if flow != "always" and not doc_obj:          # nothing applied yet: preview + report only, the rest waits
            B = ExecutionBlocker(None)
            report = ("waiting for apply: nothing is composed yet, so the nodes after the director do not run (flow: hold "
                      "until apply). Open director, compose, apply - they run then. To pass the base image on without "
                      "a composition, set flow to 'always'.\n" + report)
            log.warning("[KUBA director] node %s: nothing applied yet - the nodes after it wait (flow: hold until apply; "
                        "set flow to 'always' to pass the base on)", uid)
            return io.NodeOutput(B, B, B, B, B, B, report, B, B, B, B, B, ui=ui_dict)
        pm_out = torch.from_numpy(pmask)[None] if pmask is not None else torch.ones((1, H, W))
        state = DirectorState(doc=seq_doc or doc_obj or None, base=base, sources=dict(sources), labels=labels, table=table,
                              scene=scene, W=W, H=H, pmask=pmask, audio=audio,
                              fps=float(rd.timeline(doc_obj)["fps"] if doc_obj else 25), still=comp)
        return io.NodeOutput(comp, masks, changed, out_regions, r["rules"],
                             json.dumps(doc_obj) if doc_obj else "", report, frames, fps, audio, pm_out, state, ui=ui_dict)


@dataclass(frozen=True)
class DirectorState:
    """What kubakub director hands to kubakub director sequence: the document (keys and behaviours intact) and the
    sources the still was rendered from."""
    doc: dict | None
    base: np.ndarray
    sources: dict
    labels: np.ndarray | None
    table: dict | None
    scene: object
    W: int
    H: int
    pmask: np.ndarray | None
    audio: dict | None
    fps: float
    still: torch.Tensor

    def _comfy_cache_tensors(self):
        """What ComfyUI's RAM-pressure cache counts for this output: zero-copy tensor views of the arrays (without
        it a multi-GB state would count as nothing and be evicted last)."""
        import warnings
        out = [self.still]
        with warnings.catch_warnings():               # read-only arrays (shared imports): views are only measured
            warnings.simplefilter("ignore")
            for a in (self.base, self.labels, self.pmask, *(self.sources or {}).values()):
                if isinstance(a, np.ndarray) and a.dtype != object:
                    try:
                        out.append(torch.from_numpy(a))
                    except (TypeError, ValueError, RuntimeError):
                        pass
        if isinstance(self.audio, dict) and isinstance(self.audio.get("waveform"), torch.Tensor):
            out.append(self.audio["waveform"])
        return out


class KUBA_DirectorSequence(io.ComfyNode):
    """Every frame of the director's timeline, its sound, and the delivery export - only when this node is in the graph."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="KUBA_DirectorSequence",
            display_name="kubakub director sequence",
            category="kubakub/2d/director",
            search_aliases=['render frames', 'sequence', 'export'],
            is_output_node=True,
            description=(
                "Turns the kubakub director's timeline into video: renders every frame of the timeline (frames / fps / audio, e.g. "
                "into Create Video or kubakub keyframe clips (h3)) and / or writes it as a delivery file. Changing these "
                "settings does not re-render the director's still; bypass the node when you only need the still."),
            inputs=[
                DirectorType.Input("director", tooltip="The director output of kubakub director."),
                io.Boolean.Input("frames", default=True,
                                 tooltip="Return every frame in RAM (frames output). Off when you only export: the "
                                         "frames output is then the still."),
                io.Float.Input("sequence_scale", default=0.5, min=0.1, max=1.0, step=0.05,
                               tooltip="Size of the frames relative to the matrix (RAM: 250 frames at 1600x1080 = ~5 GB)."),
                io.Boolean.Input("skip_h3_frames", default=False,
                                 tooltip="Do not render the frames that H3 keyframe clips replace anyway (a dissolve "
                                         "stands in for them). Only applies when a kubakub keyframe clips (h3) node takes "
                                         "these frames; otherwise every frame is rendered."),
                io.Combo.Input("export", options=["none", *exm.FORMATS], default="none",
                               tooltip="Write the whole timeline as a file at the delivery size, frame by frame (no RAM limit): "
                                       "PNG sequence 8/16 bit, ProRes 4444 / 422 HQ, H.264 (4:2:0 8 bit or 4:4:4 10 bit), "
                                       "H.265 10 bit, or a small preview. Rec.709 tagged, with the timeline sound. Into "
                                       "output/kubakub_director/<name>_<time>/."),
                io.Float.Input("export_scale", default=1.0, min=0.1, max=1.0, step=0.05, tooltip="1 = the delivery size (the matrix)."),
                io.Boolean.Input("export_alpha", default=False,
                                 tooltip="The layers without the base, transparent where they do not cover (and outside "
                                         "the projection mask): PNG sequences and ProRes 4444."),
                io.String.Input("export_name", default="director", tooltip="Name of the export folder and files."),
                io.Float.Input("light_scale", default=1.0, min=0.25, max=1.0, step=0.05, optional=True, advanced=True,
                               tooltip="Render size of light layers in the frames (1 = the frame size). 0.5 is 2-3x faster "
                                       "with softer shadows; the export always uses 1."),
            ],
            outputs=[
                io.Image.Output("frames", tooltip="Every frame of the timeline (or the still with frames off)."),
                io.Float.Output("fps", tooltip="Frames per second of the timeline."),
                io.Audio.Output("audio", tooltip="The timeline's sound, trimmed to its length (silence without one)."),
                io.String.Output("report", tooltip="Frames rendered (or reused from the last run), size and time, and the "
                                                   "export folder."),
            ],
            hidden=[io.Hidden.unique_id, io.Hidden.prompt],
        )

    @classmethod
    def execute(cls, director, frames=True, sequence_scale=0.5, skip_h3_frames=False, export="none", export_scale=1.0,
                export_alpha=False, export_name="director", light_scale=1.0) -> io.NodeOutput:
        S = director
        if not S.doc:
            return io.NodeOutput(S.still, S.fps, S.audio, "no composition yet: the still only")
        out, lines = S.still, []
        if frames:
            hid = getattr(cls, "hidden", None)
            out, rep_ = sequence_frames(S.doc, S.base, S.sources, S.labels, S.table, S.scene, S.W, S.H, float(sequence_scale),
                                        bool(skip_h3_frames), S.pmask,
                                        h3_after(getattr(hid, "prompt", None), getattr(hid, "unique_id", None)),
                                        light_scale=float(light_scale))
            lines.append(rep_)
        if export and export != "none":
            lines.append(export_cached(export, float(export_scale), bool(export_alpha), str(export_name or "director"), S.doc,
                                 S.base, dict(S.sources), S.labels, S.table, S.scene, S.W, S.H, S.pmask, S.audio))
        report = "\n".join(lines) or "nothing to do: frames off and no export"
        log.info("[KUBA director sequence] %s", report.replace("\n", "\n    "))
        return io.NodeOutput(out, S.fps, S.audio, report)



# ------------------------------------------------------------------------------------------------ diffusion
# Models the window offers for "diffuse" (resolved against what is installed; first match wins). LoRAs are
# listed in full; the window puts the ones whose path mentions the preset's size first.
DIFFUSION_PRESETS = [                # the first one found is the window's default: NVFP4 = 2x faster on Blackwell
    {"name": "klein 4b nvfp4", "unet": ["klein-4b-nvfp4", "klein_4b_nvfp4"], "clip": ["qwen_3_4b_fp4", "qwen_3_4b"], "type": "flux2",
     "vae": ["flux2_full_encoder_small_decoder", "flux2-vae", "flux2_vae"], "steps": 4, "lora_hint": "4b"},
    {"name": "klein 4b", "unet": ["flux-2-klein-4b", "klein_4b", "klein-4b"], "clip": ["qwen_3_4b"], "type": "flux2",
     "vae": ["flux2_full_encoder_small_decoder", "flux2-vae", "flux2_vae"], "steps": 4, "lora_hint": "4b", "exclude": ["fp4"]},
    {"name": "klein 9b", "unet": ["flux-2-klein-9b-fp8", "klein-9b-fp8", "klein_9b"], "clip": ["qwen_3_8b"], "type": "flux2",
     "vae": ["flux2_full_encoder_small_decoder", "flux2-vae", "flux2_vae"], "steps": 4, "lora_hint": "9b", "exclude": ["fp4"]},
    {"name": "klein 9b nvfp4", "unet": ["klein-9b-nvfp4", "klein_9b_nvfp4"], "clip": ["qwen_3_8b"], "type": "flux2",
     "vae": ["flux2_full_encoder_small_decoder", "flux2-vae", "flux2_vae"], "steps": 4, "lora_hint": "9b"},
]


def _find_model(kind, keys, exclude=()):
    """First file whose name contains a key (keys in priority order), skipping names with an exclude token."""
    try:
        files = folder_paths.get_filename_list(kind)
    except Exception:  # noqa: BLE001
        return None
    for key in keys:
        hit = next((f for f in files if key in os.path.basename(f).lower()
                    and not any(x in os.path.basename(f).lower() for x in exclude)), None)
        if hit:
            return hit
    return None


def diffusion_manifest():
    presets = []
    for pr in DIFFUSION_PRESETS:
        ex = pr.get("exclude", ())
        unet, clip = _find_model("diffusion_models", pr["unet"], ex), _find_model("text_encoders", pr["clip"], ex)
        vae = _find_model("vae", pr["vae"])
        if unet and clip and vae:
            presets.append({"name": pr["name"], "unet": unet, "clip": clip, "type": pr["type"], "vae": vae,
                            "steps": pr["steps"], "lora_hint": pr["lora_hint"]})
    try:
        loras = folder_paths.get_filename_list("loras")
    except Exception:  # noqa: BLE001
        loras = []
    return {"presets": presets, "loras": loras}


_COND_CACHE = {}                   # prompt -> (weakref to the text encoder, conditioning)


def _work_size(w, h, megapixels, grid=16):
    s = (megapixels * 1e6 / max(1, w * h)) ** 0.5
    return max(grid, round(w * s / grid) * grid), max(grid, round(h * s / grid) * grid)


def diffuse_masks(shape, mode, band):
    """(area to diffuse, soft alpha) from the layer's shape: rediffuse = shape + band, edges = the band only."""
    shape = shape.astype(np.uint8)
    if band > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * band + 1, 2 * band + 1))
        grown = cv2.dilate(shape, k)
        area = grown - cv2.erode(shape, k) if mode == "edges" else grown
    else:
        area = shape
    return area.astype(np.float32)


class KUBA_DirectorDiffuse(io.ComfyNode):
    """Queued by the director window (not meant for hand-built graphs): diffuse one area, hand the result back."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="KUBA_DirectorDiffuse",
            display_name="kubakub director diffuse",
            category="kubakub/2d/director",
            is_dev_only=True,
            is_output_node=True,
            description=(
                "Repaints one area for the director window, which builds and queues this node itself (not "
                "meant for hand-built graphs). Re-diffuses a crop inside a mask (Flux 2 Klein by default, "
                "your LoRAs on the model) and writes the result into input/kuba_director, where the window picks "
                "it up as a new layer. The window builds and queues this graph itself."),
            inputs=[
                io.Model.Input("model", tooltip="The diffusion model (Flux 2 Klein by default, with your LoRAs)."),
                io.Clip.Input("clip", tooltip="The text encoder that matches the model."),
                io.Vae.Input("vae", tooltip="The VAE that matches the model."),
                io.String.Input("image", default="", tooltip="Crop to diffuse, in ComfyUI's input folder."),
                io.String.Input("mask", default="", tooltip="Grey mask of the same crop: white = the shape to change."),
                io.String.Input("prompt", multiline=True, default="",
                                tooltip="What the area should become (empty = no text guidance)."),
                io.Combo.Input("mode", options=["rediffuse", "edges"], default="rediffuse",
                               tooltip="rediffuse = the shape and its edge band, edges = only the band around it."),
                io.Float.Input("denoise", default=0.6, min=0.0, max=1.0, step=0.01,
                               tooltip="How much changes: 0 = nothing, 1 = painted from scratch inside the mask."),
                io.Int.Input("steps", default=4, min=1, max=100,
                             tooltip="Sampling steps (4 suits distilled models such as Flux 2 Klein)."),
                io.Int.Input("seed", default=0, min=0, max=0xFFFFFFFFFFFFFFFF,
                             tooltip="Random seed; another seed gives another variant."),
                io.Float.Input("megapixels", default=1.0, min=0.1, max=4.0, step=0.05,
                               tooltip="Size the crop is diffused at, in megapixels (the result keeps that size)."),
                io.Int.Input("band_px", default=16, min=0, max=512, tooltip="Edge band around the shape (crop pixels)."),
                io.Boolean.Input("reference", default=True, tooltip="The crop as reference latent (keeps layout and colours)."),
                io.String.Input("request", default="", tooltip="Id the window matches the answer with."),
            ],
            outputs=[io.Image.Output("image", tooltip="The diffused crop (also written for the window as a new layer).")],
        )

    @classmethod
    def execute(cls, model, clip, vae, image, mask, prompt, mode, denoise, steps, seed, megapixels, band_px, reference,
                request) -> io.NodeOutput:
        from ...kubakub.adapters import make_adapter
        from ...kubakub import ops
        t0 = time.perf_counter()
        src = _load_input_file(image)
        m = _load_input_file(mask)
        if src is None or m is None:
            raise FileNotFoundError("kubakub director diffuse: the window's crop or mask is missing in input/")
        h, w = src.shape[:2]
        shape = cv2.resize(m[..., 0], (w, h), interpolation=cv2.INTER_LINEAR) > 0.5
        band = max(0, int(band_px))
        area = diffuse_masks(shape, mode, band)
        if not area.any():
            raise ValueError("kubakub director diffuse: the mask is empty (nothing to diffuse)")
        adapter = make_adapter(model, clip, vae)
        ww, wh = _work_size(w, h, float(megapixels), max(16, adapter.grid))
        work = ops.resize(torch.from_numpy(np.ascontiguousarray(src[..., :3]))[None], ww, wh)
        work_mask = ops.resize_mask(torch.from_numpy(area)[None], ww, wh)
        owner = getattr(clip, "cond_stage_model", clip)   # prompt encodings kept between diffuses (same text encoder)
        for t in (prompt, ""):
            hit = _COND_CACHE.get(t)
            if hit is not None and hit[0]() is owner:
                adapter._conds[t] = hit[1]
        latent = adapter.encode(work)
        pos, neg = adapter.conds_for(prompt, "", work, latent) if reference else adapter.conds_for(prompt, "")
        import weakref
        for t in (prompt, ""):
            if t in adapter._conds:
                _COND_CACHE[t] = (weakref.ref(owner), adapter._conds[t])
        while len(_COND_CACHE) > 32:
            _COND_CACHE.pop(next(iter(_COND_CACHE)))
        out = adapter.sample(latent, pos, neg, noise_mask=work_mask[:, None], denoise=float(denoise), seed=int(seed),
                             steps=int(steps), pixel_size=(ww, wh))
        gen = adapter.decode(out)[:1, ..., :3].float().cpu().clamp(0, 1)
        a = cv2.resize(area, (ww, wh), interpolation=cv2.INTER_LINEAR)
        sigma = max(1.0, band * ww / max(1, w) / 3) if band else 1.0
        a = np.clip(cv2.GaussianBlur(a, (0, 0), sigma) * 1.5, 0, 1) * (a > 0.01)
        rgba = np.concatenate([gen[0].numpy(), a[..., None]], -1).astype(np.float32)
        fn = f"diff_{hashlib.sha1(rgba.tobytes()).hexdigest()[:16]}.png"
        out_dir = os.path.join(folder_paths.get_input_directory(), UPLOAD_SUB)
        os.makedirs(out_dir, exist_ok=True)
        _png(os.path.join(out_dir, fn), rgba)
        secs = time.perf_counter() - t0
        log.info("[KUBA director] diffused %dx%d at %dx%d (%s, denoise %.2f, %d steps) in %.1f s", w, h, ww, wh, mode,
                 denoise, steps, secs)
        return io.NodeOutput(gen, ui={"kuba_diffuse": [{"name": f"{UPLOAD_SUB}/{fn}", "request": request, "w": ww,
                                                        "h": wh, "seconds": round(secs, 1)}]})


NODE_CLASS_MAPPINGS = {
    "KUBA_Export": KUBA_Export, "KUBA_Director": KUBA_Director, "KUBA_DirectorSequence": KUBA_DirectorSequence,
    "KUBA_DirectorDiffuse": KUBA_DirectorDiffuse}
NODE_DISPLAY_NAME_MAPPINGS = {
    "KUBA_Export": "kubakub export video / frames", "KUBA_Director": "kubakub director", "KUBA_DirectorSequence": "kubakub director sequence",
    "KUBA_DirectorDiffuse": "kubakub director diffuse"}
