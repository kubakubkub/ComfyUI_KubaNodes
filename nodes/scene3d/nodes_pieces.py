"""kubakub scene pieces: motion for the pieces of the facade model (a small MOPS). Logic in scene3d/pieces.py; the
frames are rendered by the scene relight (Cycles, one Blender session, every frame cached)."""

from __future__ import annotations

import copy
import hashlib
import logging
import os
import time

import json

import cv2
import numpy as np
import torch

from comfy_api.latest import io

from . import nodes_scene3d as ns
from ...kubakub.io_types import FalloffType, PiecesType, RegionsType, SceneType
from ...kubakub.scene3d import pieces as pc
from ...kubakub.scene3d import scene_ids
from ...kubakub.scene3d import scene_view as sv

log = logging.getLogger("KUBA.regions")
CATEGORY = "kubakub/3d/pieces"
CHUNK = 40                  # frames per Blender job: each finished chunk is kept, Stop works between chunks


def _save_npy(sub, arr, prefix):
    """Content-named .npy in the scene cache (a changed array = a new name = new cache keys downstream)."""
    arr = np.ascontiguousarray(arr)
    h = hashlib.sha1(memoryview(arr).cast("B"))
    h.update((str(arr.shape) + str(arr.dtype)).encode())
    key = h.hexdigest()[:16]
    d = os.path.join(ns.cache_root(), sub)
    os.makedirs(d, exist_ok=True)
    fp = os.path.join(d, f"{prefix}_{key}.npy")
    if not os.path.isfile(fp):                                # (no touch on reuse: the frame keys stamp the mtime)
        tmp = fp + f".{os.getpid()}.tmp.npy"
        np.save(tmp, arr)
        os.replace(tmp, fp)
    return fp


def _colours(n, seed=7):
    rng = np.random.default_rng(seed)
    return (0.25 + 0.75 * rng.random((max(n, 1), 3))).astype(np.float32)


class KUBA_ScenePieces(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="KUBA_ScenePieces",
            display_name="kubakub scene pieces",
            category=CATEGORY,
            search_aliases=["pieces", "mops", "instances", "parts", "break apart", "animate 3d", "motion graphics"],
            description=("The pieces of the facade model that can move: every loose part (a stone, a window frame, a "
                         "cornice block) bigger than min_size_m. The biggest part (the wall they sit on) stays. Chain "
                         "kubakub pieces transform nodes after it and render with kubakub pieces render."),
            inputs=[
                SceneType.Input("scene", tooltip="From kubakub scene render."),
                io.Float.Input("min_size_m", default=0.2, min=0.0, max=100.0, step=0.05,
                               tooltip="Smaller parts stay still (size = the diagonal of the part's box, metres)."),
                io.Float.Input("max_size_m", default=0.0, min=0.0, max=1000.0, step=0.5,
                               tooltip="Bigger parts stay still too (0 = no limit)."),
                io.Boolean.Input("keep_largest", default=True, tooltip="The biggest part (usually the wall) stays still."),
                RegionsType.Input("regions", optional=True,
                                  tooltip="Regions of the facade (regions from id maps / masks / cryptomatte ...): every "
                                          "part belongs to the region most of it is in."),
                io.Combo.Input("source", options=["loose parts", "regions: one piece each", "regions: parts inside"],
                               default="loose parts", optional=True,
                               tooltip="regions: one piece each = every region moves as one block (a floor, a bay, all "
                                       "windows); parts inside = only the parts in the selected regions move, each on its own."),
                io.String.Input("selection", default="", optional=True,
                                tooltip="Which regions (names, 'group:windows', 'tag:left', 'a, b'). Empty = all."),
            ],
            outputs=[PiecesType.Output("pieces"), io.Image.Output("preview", tooltip="Every piece in its own colour, "
                                                                                     "still parts grey."),
                     io.String.Output("report")],
        )

    @classmethod
    def execute(cls, scene, min_size_m, max_size_m, keep_largest, regions=None, source="loose parts",
                selection="") -> io.NodeOutput:
        s, pt, nrm, ground, fr = ns.load_scene_cached(scene)
        mesh = s["mesh"]
        parts = scene_ids.loose_parts(mesh, 1e-4)
        via = ""
        if source != "loose parts":
            if regions is None:
                raise ValueError(f"source = {source} needs regions connected.")
            lab = regions.labels[0].cpu().numpy().astype(np.int32)
            fh, fw = s["faceid"].shape
            if lab.shape != (fh, fw):
                lab = cv2.resize(lab, (fw, fh), interpolation=cv2.INTER_NEAREST)
            pr = pc.part_regions(s["faceid"], lab, parts)
            wanted = None
            if (selection or "").strip():
                from ...kubakub.director.render import select_regions
                wanted = select_regions(regions.table, selection)
                if not wanted:
                    raise ValueError(f"No region matches {selection!r}.")
            labels, stats, preg = pc.pieces_from_regions(parts, pr, mesh, wanted, source.endswith("each"), keep_largest,
                                                         min_size_m, max_size_m)
            names = [str(regions.table["regions"][r].get("name", r)) for r in sorted(set(preg))[:6]]
            via = f" from {len(set(preg))} region(s) ({', '.join(names)}{' ...' if len(set(preg)) > 6 else ''})"
        else:
            labels, stats = pc.pieces_from_parts(parts, mesh, min_size_m, max_size_m, keep_largest)
        P = int(labels.max()) + 1 if (labels >= 0).any() else 0
        if P == 0:
            if source != "loose parts":
                raise ValueError("No part of the model falls into the selected regions (seen from the camera).")
            raise ValueError(f"No piece of at least {min_size_m} m: the model has {int(parts.max()) + 1} loose parts. "
                             "Lower min_size_m (or the model is one closed mesh: pieces need separate parts).")
        fp = _save_npy("pieces", labels.astype(np.int32), "faces")
        faceid = s["faceid"].astype(np.int64)
        lab_pix = np.where(faceid > 0, labels[np.maximum(faceid - 1, 0)], -2)
        col = _colours(P)
        img = np.full(faceid.shape + (3,), 0.08, np.float32)
        img[lab_pix == -1] = 0.45
        m = lab_pix >= 0
        img[m] = col[lab_pix[m]]
        u, v, _ = pc.facade_coords(stats, fr)
        report = (f"{P} pieces{via} of {int(parts.max()) + 1} loose parts (size {stats['size'].min():.2f} - "
                  f"{stats['size'].max():.2f} m), facade {fr['width_m']:.1f} x {fr['top_m']:.1f} m")
        pieces = {"scene": scene, "faces": fp, "labels": labels.astype(np.int32), "n": P, "stats": stats, "frame": fr, "ops": [],
                  "facade": [float(fr["width_m"]), float(fr["top_m"])]}
        return io.NodeOutput(pieces, torch.from_numpy(img)[None], report)


class KUBA_PiecesFalloff(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="KUBA_PiecesFalloff",
            display_name="kubakub pieces falloff",
            category=CATEGORY,
            search_aliases=["falloff", "wave", "stagger", "noise", "mops"],
            description=("How strongly a transform acts on each piece over time. wave: a front runs across the "
                         "facade and the pieces it passed stay moved. pulse: a band runs across and the pieces move "
                         "and come back. stagger: one piece after the other in an order. noise: every piece drifts "
                         "on its own. all: every piece fully."),
            inputs=[
                io.Combo.Input("type", options=["wave", "pulse", "stagger", "noise", "all"], default="wave",
                               tooltip="wave: moved pieces stay moved; pulse: they come back; stagger: one after the "
                                       "other; noise: each on its own; all: every piece fully."),
                io.Combo.Input("direction", options=list(pc.ORDERS[:5]) + ["towards the centre", "random"],
                               default="left to right", tooltip="wave / pulse: where the front runs (random = a ripple in random order); stagger: the order."),
                io.Float.Input("start", default=0.0, min=-100.0, max=1000.0, step=0.1, tooltip="Seconds."),
                io.Float.Input("speed", default=6.0, min=0.01, max=1000.0, step=0.5,
                               tooltip="wave / pulse: metres per second across the facade."),
                io.Float.Input("width", default=3.0, min=0.01, max=1000.0, step=0.5,
                               tooltip="wave: the soft edge; pulse: the band (metres)."),
                io.Float.Input("spread", default=2.0, min=0.0, max=1000.0, step=0.1,
                               tooltip="stagger: seconds from the first piece to the last."),
                io.Float.Input("duration", default=1.0, min=0.01, max=100.0, step=0.1,
                               tooltip="stagger: seconds one piece takes."),
                io.Float.Input("hold", default=-1.0, min=-1.0, max=100.0, step=0.1,
                               tooltip="stagger: seconds each piece stays, then goes back (-1 = stays)."),
                io.Float.Input("loop", default=0.0, min=0.0, max=1000.0, step=0.5,
                               tooltip="wave / pulse: repeat every this many seconds (0 = once)."),
                io.Float.Input("frequency", default=0.5, min=0.01, max=50.0, step=0.05, tooltip="noise: changes per second."),
                io.Int.Input("seed", default=1, min=0, max=100000, control_after_generate=io.ControlAfterGenerate.fixed,
                             tooltip="Another number = another random pattern (the same number = the same result)."),
                io.Float.Input("amount", default=1.0, min=-10.0, max=10.0, step=0.05,
                               tooltip="Multiplies the falloff (negative = the other way)."),
                io.Boolean.Input("invert", default=False, tooltip="Pieces the falloff reached rest, the others move."),
            ],
            outputs=[FalloffType.Output("falloff")],
        )

    @classmethod
    def execute(cls, type, direction, start, speed, width, spread, duration, hold, loop, frequency, seed, amount,
                invert) -> io.NodeOutput:
        return io.NodeOutput({"type": type, "axis": direction,          # wave / pulse with "random" = a random ripple "order": direction, "start": start, "speed": speed,
                              "width": width, "spread": spread, "duration": duration, "hold": hold, "loop": loop,
                              "freq": frequency, "seed": seed, "amount": amount, "invert": invert})


class KUBA_PiecesBeatFalloff(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="KUBA_PiecesBeatFalloff",
            display_name="kubakub pieces beat falloff",
            category=CATEGORY,
            search_aliases=["beat", "bars", "markers", "music", "timeline", "falloff", "mops"],
            description=("A kick on the beats, bars or markers of the director's timeline: every piece jumps (attack) "
                         "and settles (decay). spread > 0 lets the kick run across the facade in an order, like a "
                         "ripple. Set the tempo or load a sound in the director window, or set markers (M)."),
            inputs=[
                io.String.Input("document", force_input=True, tooltip="The 'document' output of kubakub director."),
                io.Combo.Input("trigger", options=["beats", "bars", "markers"], default="beats",
                               tooltip="beats and bars come from the tempo (or the sound) of the timeline; markers from M."),
                io.Int.Input("every_nth", default=1, min=1, max=64, tooltip="Only every nth beat / bar / marker."),
                io.Float.Input("attack", default=0.03, min=0.0, max=10.0, step=0.01, tooltip="Seconds to the peak."),
                io.Float.Input("decay", default=0.3, min=0.01, max=20.0, step=0.05, tooltip="Seconds to settle (e-fold)."),
                io.Float.Input("spread", default=0.0, min=0.0, max=20.0, step=0.05,
                               tooltip="Seconds the kick takes from the first piece to the last."),
                io.Combo.Input("order", options=list(pc.ORDERS), default="left to right",
                               tooltip="With spread: which piece the kick reaches first."),
                io.Int.Input("seed", default=1, min=0, max=100000, control_after_generate=io.ControlAfterGenerate.fixed,
                             tooltip="Another number = another random pattern (the same number = the same result)."),
                io.Float.Input("amount", default=1.0, min=-10.0, max=10.0, step=0.05,
                               tooltip="Multiplies the kick (negative = the other way)."),
            ],
            outputs=[FalloffType.Output("falloff"), io.String.Output("report")],
        )

    @classmethod
    def execute(cls, document, trigger, every_nth, attack, decay, spread, order, seed, amount) -> io.NodeOutput:
        from ...kubakub.director import motion
        try:
            doc = json.loads(document or "")
        except ValueError as e:
            raise ValueError(f"The director document is not valid JSON ({e}).") from e
        ctx = motion.context(doc)
        if trigger == "markers":
            times = list(ctx["markers"])[::every_nth]
        else:
            times = list(ctx["beats"])[::every_nth * (4 if trigger == "bars" else 1)]
        if not times:
            what = "markers (M in the timeline)" if trigger == "markers" else "a tempo (load a sound or set the bpm)"
            raise ValueError(f"The timeline has no {trigger}: set {what} in the director window, then run it.")
        f = {"type": "triggers", "times": times, "attack": attack, "decay": decay, "spread": spread, "order": order,
             "seed": seed, "amount": amount}
        return io.NodeOutput(f, f"{len(times)} {trigger} from {times[0]:.2f} s to {times[-1]:.2f} s")


class KUBA_PiecesTransform(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="KUBA_PiecesTransform",
            display_name="kubakub pieces transform",
            category=CATEGORY,
            search_aliases=["transform", "push", "explode", "rotate", "scale", "mops"],
            description=("Moves the pieces, weighted by the falloff (without one: fully, all the time). Push = out of "
                         "the wall along each piece's own face; across / up / out = along the facade; tilt / turn / "
                         "roll = rotate each piece about its centre. Chain several: they add up."),
            inputs=[
                PiecesType.Input("pieces", tooltip="From kubakub scene pieces or another pieces transform."),
                FalloffType.Input("falloff", optional=True,
                                  tooltip="How strongly and when (kubakub pieces falloff / beat falloff). None = fully."),
                io.Float.Input("push", default=0.5, min=-100.0, max=100.0, step=0.05, tooltip="Metres along the piece's normal."),
                io.Float.Input("across", default=0.0, min=-100.0, max=100.0, step=0.05, tooltip="Metres left -> right."),
                io.Float.Input("up", default=0.0, min=-100.0, max=100.0, step=0.05, tooltip="Metres up."),
                io.Float.Input("out", default=0.0, min=-100.0, max=100.0, step=0.05, tooltip="Metres out of the facade."),
                io.Float.Input("tilt", default=0.0, min=-3600.0, max=3600.0, step=1.0, tooltip="Degrees about the horizontal."),
                io.Float.Input("turn", default=0.0, min=-3600.0, max=3600.0, step=1.0, tooltip="Degrees about the vertical."),
                io.Float.Input("roll", default=0.0, min=-3600.0, max=3600.0, step=1.0, tooltip="Degrees in the facade plane."),
                io.Float.Input("random_rotate", default=0.0, min=0.0, max=3600.0, step=1.0,
                               tooltip="Up to this many degrees per axis, different for every piece."),
                io.Float.Input("scale", default=1.0, min=0.0, max=20.0, step=0.05,
                               tooltip="Size of each piece where the falloff is full (1 = unchanged)."),
                io.Float.Input("jitter", default=0.0, min=0.0, max=100.0, step=0.05,
                               tooltip="Metres in a random direction, different for every piece."),
                io.Int.Input("seed", default=1, min=0, max=100000, control_after_generate=io.ControlAfterGenerate.fixed,
                             tooltip="Another number = another random pattern (the same number = the same result)."),
            ],
            outputs=[PiecesType.Output("pieces")],
        )

    @classmethod
    def execute(cls, pieces, push, across, up, out, tilt, turn, roll, random_rotate, scale, jitter, seed,
                falloff=None) -> io.NodeOutput:
        p = dict(pieces)
        p["ops"] = list(pieces.get("ops", [])) + [{
            "push": push, "move": [across, up, out], "rotate": [tilt, turn, roll], "random_rotate": random_rotate,
            "scale": scale, "jitter": jitter, "seed": seed, "falloff": copy.deepcopy(falloff) if falloff else None}]
        return io.NodeOutput(p)


class KUBA_PiecesRender(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="KUBA_PiecesRender",
            display_name="kubakub pieces render",
            category=CATEGORY,
            search_aliases=["render", "pieces", "mops", "animation", "cycles", "projector"],
            description=("Renders the moving pieces from the projection camera in Blender Cycles, frame by frame "
                         "(one Blender session; every frame is kept, so a change renders only what changed). "
                         "clay = the model in light; projected matrix = the matrix (or its frames) cast from the "
                         "projector onto the moving pieces, the way the audience would see it."),
            inputs=[
                PiecesType.Input("pieces", tooltip="The pieces with their transforms."),
                io.Float.Input("duration", default=4.0, min=0.04, max=600.0, step=0.5, tooltip="Seconds."),
                io.Float.Input("fps", default=25.0, min=1.0, max=120.0, step=1.0, tooltip="Frames per second."),
                io.Combo.Input("look", options=["clay", "projected matrix"], default="clay",
                               tooltip="clay = the model in light (content to project); projected matrix = the matrix "
                                       "cast from the projector onto the moving pieces."),
                io.Image.Input("matrix", optional=True,
                               tooltip="projected matrix: one image or one per frame (the last one repeats)."),
                io.Float.Input("projector_brightness", default=1.0, min=0.0, max=20.0, step=0.1,
                               tooltip="projected matrix: 1 = the matrix at its own brightness on the stone."),
                io.Combo.Input("environment", options=[e for e in sv.ENVIRONMENTS if e != "file"], default="night",
                               tooltip="The light around the building (Blender's built-in HDRIs)."),
                io.Float.Input("env_strength", default=0.3, min=0.0, max=20.0, step=0.05,
                               tooltip="How bright that light is."),
                io.Float.Input("clay", default=0.7, min=0.0, max=1.0, step=0.05, tooltip="Brightness of the stone."),
                io.Combo.Input("background", options=["black", "environment"], default="black",
                               tooltip="black = only the building (alpha output); environment = the HDRI behind it."),
                io.Float.Input("resolution_scale", default=0.5, min=0.1, max=1.0, step=0.05,
                               tooltip="Of the scene's size (1 = the matrix size)."),
                io.Int.Input("samples", default=32, min=1, max=4096,
                             tooltip="Cycles samples per frame (more = cleaner, slower; denoised either way)."),
                io.Combo.Input("view", options=["projector", "audience"], default="projector", optional=True,
                               tooltip="projector = from the projection camera (what the projector sends); audience = "
                                       "from a spot in front of the building: the pieces show their depth."),
                io.Float.Input("audience_distance_m", default=15.0, min=0.5, max=500.0, step=0.5, optional=True, advanced=True,
                               tooltip="view = audience: metres in front of the facade."),
                io.Float.Input("audience_offset_m", default=0.0, min=-500.0, max=500.0, step=0.5, optional=True, advanced=True,
                               tooltip="view = audience: metres left (-) or right (+) of the facade's middle."),
                io.Float.Input("eye_height_m", default=1.7, min=0.0, max=200.0, step=0.1, optional=True, advanced=True,
                               tooltip="view = audience: eye height above the ground (metres)."),
                io.Float.Input("lens_mm", default=24.0, min=6.0, max=300.0, step=1.0, optional=True, advanced=True,
                               tooltip="view = audience: focal length (smaller = wider)."),
            ],
            outputs=[io.Image.Output("frames"), io.Mask.Output("alpha"), io.Float.Output("fps"),
                     io.String.Output("report")],
        )

    @classmethod
    def execute(cls, pieces, duration, fps, look, projector_brightness, environment, env_strength, clay, background,
                resolution_scale, samples, view="projector", audience_distance_m=15.0, audience_offset_m=0.0,
                eye_height_m=1.7, lens_mm=24.0, matrix=None) -> io.NodeOutput:
        t0 = time.perf_counter()
        scene = pieces["scene"]
        s, pt, nrm, ground, fr = ns.load_scene_cached(scene)
        import comfy.model_management as mm
        import comfy.utils
        faces = pieces["faces"]
        if not os.path.isfile(faces) and pieces.get("labels") is not None:   # removed by the cache purge
            faces = _save_npy("pieces", pieces["labels"], "faces")
        xf = pc.sequence(pieces["stats"], pieces["frame"], pieces["ops"], duration, fps)
        n = xf.shape[0]
        W, H, rw, rh = ns._render_size(s["info"], None, float(resolution_scale))
        projected = look == "projected matrix"
        if projected and matrix is None:
            raise ValueError("look = projected matrix needs a matrix image (or frames).")
        rig = {"lights": [], "environment": environment, "env_strength": env_strength, "clay": clay,
               "roughness": 0.8, "background": background, "exposure": 0.0, "view": "AgX",
               "projector": {"on": projected, "brightness": projector_brightness}}
        look_from = None
        if view == "audience":
            cam = np.asarray(s["info"]["camera"]["matrix_world"], np.float64)
            look_from = pc.audience_view(fr, cam[:3, 3], audience_distance_m, audience_offset_m, eye_height_m, lens_mm)
        notes, parts = [], []
        for k in range(n):
            img = matrix[min(k, matrix.shape[0] - 1), ..., :3].cpu().numpy() if projected else None
            part, _, _, _ = ns._rig_job(s, fr, rig, (), img, ns.cache_root(), notes if k == 0 else [])
            # one file per distinct frame: a frame's cache key depends on that frame only
            part["pieces"] = {"faces": faces, "xform": _save_npy("pieces", xf[k], "xform")}
            if look_from:
                part["look_from"] = look_from
            parts.append(part)
        # identical frames (at rest, holds) render once
        uniq, index = {}, []
        for part in parts:
            key = json.dumps(part, sort_keys=True)
            index.append(uniq.setdefault(key, len(uniq)))
        todo = [None] * len(uniq)
        for part, i in zip(parts, index):
            todo[i] = part
        bar = comfy.utils.ProgressBar(len(todo))
        got, rendered, per = [], 0, []
        for c0 in range(0, len(todo), CHUNK):
            mm.throw_exception_if_processing_interrupted()   # Stop works between chunks
            chunk = todo[c0:c0 + CHUNK]
            g, missing, rj = ns.relight_frames(scene, chunk, rw, rh, int(samples),
                                              timeout=max(900, len(chunk) * max(10, int(samples)) * 2))
            got += g
            rendered += len(missing)
            per += (rj or {}).get("seconds", {}).get("per_frame") or []
            bar.update_absolute(min(len(todo), c0 + CHUNK), len(todo))
        frames = torch.empty((n, rh, rw, 3), dtype=torch.float32)
        alphas = torch.empty((n, rh, rw), dtype=torch.float32)
        reads = {}
        for k, i in enumerate(index):
            if i not in reads:
                reads = {i: ns._read_relit(got[i][0], rw, rh, rw, rh, background)}   # keep one decoded frame
            rgb, a = reads[i]
            frames[k] = torch.from_numpy(rgb)
            alphas[k] = torch.from_numpy(a)
        report = "\n".join([
            f"{n} frames at {fps:g} fps, {pieces['n']} pieces, {len(pieces['ops'])} transform(s), {rw}x{rh}, "
            f"{samples} samples, look {look}, view {view}"
            + (f" ({audience_distance_m:g} m in front, {audience_offset_m:+g} m, eyes {eye_height_m:g} m)" if look_from else ""),
            f"{len(todo)} different frames: rendered now {rendered} ({(sum(per) / len(per)) if per else 0:.2f} s per "
            f"frame), from the cache {len(todo) - rendered}, total {time.perf_counter() - t0:.1f} s", *notes])
        log.info("[KUBA pieces] %s", report.replace("\n", " | "))
        return io.NodeOutput(frames, alphas, float(fps), report)


NODE_CLASS_MAPPINGS = {"KUBA_ScenePieces": KUBA_ScenePieces, "KUBA_PiecesFalloff": KUBA_PiecesFalloff,
                       "KUBA_PiecesBeatFalloff": KUBA_PiecesBeatFalloff,
                       "KUBA_PiecesTransform": KUBA_PiecesTransform, "KUBA_PiecesRender": KUBA_PiecesRender}
NODE_DISPLAY_NAME_MAPPINGS = {"KUBA_ScenePieces": "kubakub scene pieces", "KUBA_PiecesFalloff": "kubakub pieces falloff",
                              "KUBA_PiecesBeatFalloff": "kubakub pieces beat falloff",
                              "KUBA_PiecesTransform": "kubakub pieces transform",
                              "KUBA_PiecesRender": "kubakub pieces render"}
