"""
nodes_scene3d.py

3D scene files for projection mapping (category KUBAKUB/scene3d).
KUBA_SceneRender: a .blend / .fbx / .obj / .abc / .glb ... file -> the projection view from the file's
camera at matrix size: clay render, metric depth, normals, and ID passes written in the From ID Maps
folder format (ids_<pass>.png + .txt + clay.png), so kubakub regions from id maps turns them into regions.
Blender runs headless as a subprocess (scene3d/bridge.py, scene3d/blender_export.py); the ID passes are
computed in numpy from the face index each pixel sees (scene3d/scene_ids.py).

V3 node schema (comfy_api.latest), registered through the pack's NODE_CLASS_MAPPINGS loader.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import time

import cv2
import numpy as np
import torch

import folder_paths
from comfy_api.latest import io, ui

from ...kubakub import save_paths as sp
from ...kubakub import viewer as vw
from ...kubakub.io_types import ProjectorType, RegionsType, SceneType, ViewerType
from ...kubakub.scene3d import autocam, bridge, scene_ids, scene_view as sv, surroundings as sr
from ...kubakub.types import Regions
from ...kubakub import imio
from ...kubakub import samples

log = logging.getLogger("KUBA.regions")

CATEGORY = "kubakub/3d/scene"
DEFAULT_PASSES = "elements sections floors shelves layers facing planes parts objects materials collections"
IDS_DONE = ("ids_summary.json", "scene.json", "clay.png")          # a finished ids_<key> folder
VIEW_FILES = ("faceid.npy", "position.npy", "normal.npy", "clay.png", "scene.json")   # one walkthrough view


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


def cache_root():
    """
    Where scene exports, ID passes, relight frames and walkthrough views are kept across ComfyUI restarts (ComfyUI
    empties its temp folder at every start): setting cache_folder in kubakub.ini (a folder; 'temp' = the old place in
    ComfyUI's temp), else kubakub_cache in ComfyUI's user directory. Kept under cache_gb (default 20), least used first.
    """
    return os.path.join(cache_base(), "kuba_scene3d")


def cache_base(lasting=False):
    """The folder the caches live in (setting cache_folder). lasting: for what must survive a restart (downloaded map
    data): cache_folder = temp then still means ComfyUI's user directory."""
    env = kst.get("cache_folder", "").strip()
    if env.lower() in ("temp", "off", "0"):
        env = ""
        if not lasting:
            return folder_paths.get_temp_directory()
    return bridge.clean_path(env) if env else os.path.join(folder_paths.get_user_directory(), "kubakub_cache")


def cache_cap_bytes():
    try:
        return int(kst.number("cache_gb", 20) * 2 ** 30)
    except ValueError:
        return 20 * 2 ** 30


def _after_render():
    bridge.maybe_purge(cache_root(), cache_cap_bytes())


try:
    _after_render()                                      # at start (in the background)
except Exception as _e:  # noqa: BLE001
    log.warning("[KUBA scene3d] cache purge skipped: %s", _e)


def _img(a):
    return torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32))[None]


def _depth_vis(depth):
    fg = depth > 0
    v = np.zeros(depth.shape, np.float32)
    if fg.any():
        lo, hi = np.percentile(depth[fg], [0.5, 99.5])
        v[fg] = 1.0 - np.clip((depth[fg] - lo) / max(hi - lo, 1e-6), 0, 1)     # near = bright
    rgb = cv2.applyColorMap((v * 255).astype(np.uint8), cv2.COLORMAP_TURBO)[..., ::-1] / 255.0
    return rgb * fg[..., None]


def _depth_gray(depth):
    """Linear grey depth, near = 1, far = 0, background 0 (monotonic, unlike the colour preview)."""
    fg = depth > 0
    g = np.zeros(depth.shape, np.float32)
    if fg.any():
        lo, hi = float(depth[fg].min()), float(depth[fg].max())
        g[fg] = 1 - (depth[fg] - lo) / max(hi - lo, 1e-6)
    return np.repeat(g[..., None], 3, -1)


class KUBA_SceneRender(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="KUBA_SceneRender",
            display_name="kubakub scene render",
            category="kubakub/3d/scene",
            search_aliases=['blender', 'fbx', '3d', 'clay', 'depth'],
            description=(
                "Renders a 3D model of the building from the projector's camera and splits it into regions "
                "you can paint. Takes a 3D scene file (.blend .fbx .obj .abc .glb .gltf .stl .ply .usd) and gives "
                "the projection view from the file's camera: clay render, metric depth, normals and ID passes (shelves = the "
                "facade's depth steps, layers, facing, planes, loose parts, objects / materials / "
                "collections). The ID passes are written as a From ID Maps folder: connect 'folder' to "
                "kubakub regions from id maps (regions_pass e.g. shelves or parts). Needs Blender 4.x "
                "(runs headless). .c4d: export FBX or Alembic from Cinema 4D."),
            inputs=[
                io.String.Input("file", default="", placeholder="paste the path of your .blend / .fbx / .obj  (empty = the sample building)",
                                tooltip="Your scene file: paste its path (Explorer: right click, Copy as path; the quotes "
                                        "are fine). Empty: the built-in sample building, so the node runs as it is."),
                io.String.Input("camera", default="", optional=True,
                                tooltip="Camera object name. Empty: the file's active camera, else the first "
                                        "camera, else a front camera framing the model (it finds the facade side by "
                                        "itself). A connected projector replaces the file's camera."),
                io.Int.Input("width", default=0, min=0, max=16384, step=8,
                             tooltip="Render width; 0 = the file's render resolution (the matrix size)."),
                io.Int.Input("height", default=0, min=0, max=16384, step=8,
                             tooltip="Render height; 0 = the file's render resolution."),
                io.Int.Input("frame", default=-1, min=-1, max=100000,
                             tooltip="Scene frame (animated cameras, Alembic); -1 = the file's current frame."),
                io.String.Input("passes", default=DEFAULT_PASSES,
                                tooltip="ID passes to write: shelves layers facing planes parts objects "
                                        "materials collections (objects / materials / collections only when "
                                        "the file has more than one)."),
                io.Combo.Input("preview_pass", options=list(scene_ids.PASSES), default="shelves",
                               tooltip="Which ID pass to show on ids_preview (it is also written when 'passes' "
                                       "does not list it)."),
                io.Float.Input("shelf_min_step_m", advanced=True, default=0.05, min=0.005, max=2.0, step=0.005,
                               tooltip="shelves: depth steps closer than this (m) are one shelf."),
                io.Float.Input("plane_angle_deg", default=3.0, min=0.1, max=30.0, step=0.1, advanced=True,
                               tooltip="planes: neighbour faces within this angle are one plane."),
                io.Float.Input("plane_offset_m", default=0.02, min=0.0, max=1.0, step=0.005, advanced=True,
                               tooltip="planes: and within this offset (m)."),
                io.String.Input("layer_bands_m", default="-0.5 -0.05 0.05 0.5", advanced=True,
                                tooltip="layers: 4 limits (m, + = towards the camera) relative to the main "
                                        "facade plane -> deep_recess, recess, facade, relief, projection."),
                io.String.Input("save_folder", advanced=True, default="", optional=True,
                                tooltip="Also copy the ID maps, legends, clay and scene.json here (for "
                                        "Houdini / After Effects). Empty: only the temp cache."),
                io.String.Input("blender_path", default="", optional=True, advanced=True,
                                tooltip="blender.exe or its folder. Empty: the blender setting in kubakub.ini or the newest "
                                        "C:\\Program Files\\Blender Foundation install."),
                io.Float.Input("unit_scale", advanced=True, default=1.0, min=0.0001, max=10000.0, step=0.001, optional=True,
                               tooltip="Multiplies the model (and the camera position) into metres, e.g. 0.01 "
                                       "for a file in cm. The report warns when the model looks too small or big."),
                io.Float.Input("min_size_m", advanced=True, default=0.5, min=0.0, max=50.0, step=0.05, optional=True,
                               tooltip="Pieces smaller than this across the wall (largest of width / height, m) "
                                       "merge into the neighbour they share the longest border with - the crowd "
                                       "cannot read them anyway. 0 = keep everything."),
                ProjectorType.Input("projector", optional=True,
                                    tooltip="From kubakub projector: where the projector stands (metres from the wall, "
                                            "height, throw). Used instead of the file's camera; set width / height to "
                                            "the projector's resolution."),
            ],
            outputs=[
                io.Image.Output("clay", tooltip="Workbench clay render from the projection camera."),
                io.Image.Output("ids_preview", tooltip="The preview_pass ID map."),
                io.Image.Output("depth", tooltip="Camera depth, near = bright (turbo colours)."),
                io.Image.Output("normal", tooltip="World normals as colours."),
                io.Mask.Output("coverage", tooltip="1 where the model is (a scope for the region nodes)."),
                io.String.Output("folder", tooltip="ID-map folder -> kubakub regions from id maps."),
                io.String.Output("scene_json", tooltip="Camera, resolution, names, main plane, shelves."),
                io.String.Output("report", tooltip="Faces, camera, depth range, shelves, ID passes, cache use and the ID-map folder; warns when the "
                                                  "model scale looks wrong."),
                SceneType.Output("scene", tooltip="-> kubakub scene measure / kubakub scene preview."),
                io.Image.Output("depth_gray", tooltip="Camera depth as grey, near = white, linear over the "
                                                      "model's depth range (for Relief Field and depth nodes)."),
            ],
        )

    @classmethod
    def fingerprint_inputs(cls, file="", **kwargs):
        h = hashlib.sha256(json.dumps(kwargs, sort_keys=True, default=str).encode())
        p = bridge.clean_path(file) or samples.model(folder_paths.get_temp_directory())   # as in execute
        try:
            st = os.stat(p)
            h.update(f"{p}|{st.st_size}|{st.st_mtime_ns}".encode("utf-8", "replace"))
            # if the cached folder is gone (the size cap, a cleared cache or temp), run again instead of handing
            # downstream nodes a folder that no longer exists
            folder = bridge.cache_folder(p, cache_root(),
                                         (kwargs.get("camera") or "").strip(), int(kwargs.get("width") or 0),
                                         int(kwargs.get("height") or 0), int(kwargs.get("frame", -1)),
                                         float(kwargs.get("unit_scale") or 1.0), projector=kwargs.get("projector"))
            ok = os.path.isfile(os.path.join(folder, "scene.json")) and any(
                f.startswith("ids_") for f in os.listdir(folder))
            h.update(bridge.folder_token(folder, ok))
        except OSError:
            h.update(p.encode("utf-8", "replace"))
        return h.hexdigest()

    @classmethod
    def execute(cls, file, width, height, frame, passes, preview_pass, shelf_min_step_m, plane_angle_deg,
                plane_offset_m, layer_bands_m, camera="", save_folder="", blender_path="",
                unit_scale=1.0, min_size_m=0.5, projector=None) -> io.NodeOutput:
        t0 = time.perf_counter()
        projector = autocam.spec(projector) if projector else None
        sample = not bridge.clean_path(file)
        if sample:                                     # nothing pasted yet: the sample building
            file = samples.model(folder_paths.get_temp_directory())
        folder, cached, _ = bridge.export(file, cache_root(), camera=camera.strip(), width=width, height=height,
                                          frame=frame, blender=blender_path, unit_scale=unit_scale, projector=projector)
        t_export = time.perf_counter() - t0
        wanted = [p for p in passes.replace(",", " ").split() if p]
        bad = [p for p in wanted if p not in scene_ids.PASSES]
        if bad:
            raise ValueError(f"unknown pass(es) {', '.join(bad)}; known: {' '.join(scene_ids.PASSES)}")
        if preview_pass not in wanted:
            wanted.append(preview_pass)
        bands = tuple(float(x) for x in layer_bands_m.replace(",", " ").split())
        if len(bands) != 4 or list(bands) != sorted(bands):
            raise ValueError("layer_bands_m needs 4 increasing numbers")
        key = hashlib.sha1(json.dumps([wanted, shelf_min_step_m, plane_angle_deg, plane_offset_m, bands,
                                       min_size_m]).encode())
        ids_dir = os.path.join(folder, "ids_" + key.hexdigest()[:10])
        reused = bridge.is_done(ids_dir, IDS_DONE)
        if reused:                                     # same scene folder and settings: the passes are there
            bridge.touch(ids_dir)
            with open(os.path.join(ids_dir, "ids_summary.json"), encoding="utf-8") as f:
                summary = json.load(f)
            scene = scene_ids.load(folder)
            depth = np.load(os.path.join(ids_dir, "depth_m.npy"))
            pp = os.path.join(ids_dir, f"ids_{preview_pass}.png")
            previews = {preview_pass: imio.imread(pp)[..., ::-1]} if os.path.isfile(pp) else {}
            info = scene["info"]
            info["ids"] = summary
        else:
            stage = bridge.stage_path(ids_dir)         # written aside, renamed into place when complete
            try:
                summary, previews, scene, depth = scene_ids.build(
                    folder, passes=wanted, angle_deg=plane_angle_deg, dist_m=plane_offset_m, layer_bands=bands,
                    shelf_sep_m=shelf_min_step_m, out=stage, min_size_m=min_size_m)
                shutil.copy(os.path.join(folder, "clay.png"), os.path.join(stage, "clay.png"))
                info = scene["info"]
                info["ids"] = summary
                with open(os.path.join(stage, "scene.json"), "w", encoding="utf-8") as f:
                    json.dump(info, f, indent=1)
                bridge.publish(stage, ids_dir, IDS_DONE)
            except BaseException:
                shutil.rmtree(stage, ignore_errors=True)
                raise
            _after_render()
        dst = sp.save_folder(save_folder, folder_paths.get_output_directory())
        if dst:
            os.makedirs(dst, exist_ok=True)
            for fn in os.listdir(ids_dir):
                if fn.endswith((".png", ".txt", ".json")):
                    shutil.copy(os.path.join(ids_dir, fn), os.path.join(dst, fn))

        clay = imio.imread(os.path.join(folder, "clay.png"))[..., ::-1] / 255.0
        fid = scene["faceid"]
        fg = fid > 0
        prev = previews.get(preview_pass)
        prev = prev / 255.0 if prev is not None else np.zeros(clay.shape, np.float32)
        nrm = (scene["normal"] * 0.5 + 0.5) * fg[..., None]
        cam = info["camera"]
        passes_txt = ", ".join(f"{k} {v['ids']}" + (f" ({v['merged_small']} small pieces merged)" if v.get("merged_small") else "")
                               for k, v in summary["passes"].items())
        if min_size_m > 0:
            passes_txt += f"; pieces under {min_size_m:g} m across the wall merged into neighbours"
        export_txt = ("cached" if cached else
                      f"{t_export:.1f} s (Blender {info['blender']}, Cycles {info.get('cycles_device', '?')})")
        report = "\n".join([
            f"{os.path.basename(info['file'])}: {info['faces']} faces, {len(info['objects'])} objects, "
            f"{info['width']}x{info['height']}, model covers {summary['coverage'] * 100:.0f} %",
            f"camera {cam['name']} ({cam['how']}), {cam['type']} {cam['lens_mm']:.1f} mm, shift "
            f"{cam['shift_x']:.3f} / {cam['shift_y']:.3f}; cameras in file: {', '.join(info['cameras']) or '-'}",
            f"depth {summary['depth_m'][0]:.2f} - {summary['depth_m'][1]:.2f} m from the camera; main plane "
            f"normal {summary['main_plane']['normal']}",
            f"shelves (m from the main plane): {summary.get('shelves_m', '-')}",
            f"ID passes: {passes_txt}",
            f"export {export_txt}, ID passes {'reused' if reused else 'built'}, total {time.perf_counter() - t0:.1f} s",
            f"folder: {ids_dir}", *summary["notes"], *info.get("notes", []),
            *([samples.note("3D file", "file")] if sample else [])])
        size = np.array(info["bbox_max"]) - np.array(info["bbox_min"])
        if size.max() < 2.0 or size.max() > 2000.0:
            report += (f"\nWARNING: the model is {size.max():.3g} m at its largest - wrong units? Distances need "
                       f"metres: set unit_scale (a 30 m building at {size.max():.3g} -> {30.0 / size.max():.4g}).")
        log.info("[KUBA scene3d] %s", report.replace("\n", "\n    "))
        preview = _img(prev)
        scene_out = {"folder": folder, "ids": ids_dir, "file": info["file"], "blender": blender_path,
                     "unit_scale": float(unit_scale), "frame": int(frame), "camera": camera.strip(),
                     "projector": projector}
        return io.NodeOutput(_img(clay), preview, _img(_depth_vis(depth)), _img(nrm),
                             torch.from_numpy(fg.astype(np.float32))[None], ids_dir,
                             json.dumps(info, indent=1), report, scene_out, _img(_depth_gray(depth)),
                             ui=ui.PreviewImage(preview, cls=cls))


class KUBA_Projector(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="KUBA_Projector",
            display_name="kubakub projector",
            category="kubakub/3d/scene",
            search_aliases=['camera', 'throw ratio', 'lens shift', 'beamer'],
            description=(
                "Where the projector stands, as on site: metres from the wall, along it and above the ground, and its "
                "throw ratio. Connect it to kubakub scene render: the model is rendered from there instead of from the "
                "file's camera, so a file without a camera works, and so does trying another spot or lens. For several "
                "projectors use one projector and one scene render each, then kubakub projector blend."),
            inputs=[
                io.Float.Input("distance_m", default=0.0, min=0.0, max=2000.0, step=0.5,
                               tooltip="Metres from the wall. 0 = as far as the throw ratio needs to cover the model "
                                       "(throw 1.5 when that is 0 too)."),
                io.Float.Input("offset_m", default=0.0, min=-1000.0, max=1000.0, step=0.5,
                               tooltip="Metres along the wall from the middle of the facade, + = to the right as the "
                                       "audience sees it."),
                io.Float.Input("height_m", default=1.5, min=-100.0, max=500.0, step=0.1,
                               tooltip="Metres above the ground (the lowest point of the model): a tower, a balcony, a roof."),
                io.Float.Input("throw_ratio", default=0.0, min=0.0, max=20.0, step=0.05,
                               tooltip="Distance / picture width, as on the lens data sheet (0.8 = wide, 2 = long). "
                                       "0 = the picture just covers the model."),
                io.Combo.Input("aim", options=list(autocam.AIMS), default="lens shift",
                               tooltip="lens shift: square to the wall, the picture shifted onto the model (verticals stay "
                                       "vertical). tilt: turned to the middle of the model (keystone). straight: square "
                                       "to the wall, no shift."),
                io.Combo.Input("side", options=list(autocam.SIDE_OPTIONS), default="auto", advanced=True,
                               tooltip="Which side of the model is the facade. auto: the side the model is widest "
                                       "across; front = Blender's front view."),
                io.Float.Input("turn_deg", default=0.0, min=-89.0, max=89.0, step=1.0, advanced=True,
                               tooltip="The projector's direction turned around the building, for a spot at an angle "
                                       "to the facade (+ = towards the right, seen from the audience)."),
                io.Float.Input("margin", default=0.05, min=0.0, max=1.0, step=0.01, advanced=True,
                               tooltip="Free border around the model when the picture is fitted (throw ratio 0), as a "
                                       "share of the picture."),
                io.String.Input("name", default="", optional=True, advanced=True,
                                tooltip="A name for the reports (left tower, roof ...)."),
            ],
            outputs=[ProjectorType.Output("projector", tooltip="-> kubakub scene render, projector.")],
        )

    @classmethod
    def execute(cls, distance_m, offset_m, height_m, throw_ratio, aim, side="auto", turn_deg=0.0, margin=0.05,
                name="") -> io.NodeOutput:
        return io.NodeOutput(autocam.spec({"distance_m": distance_m, "offset_m": offset_m, "height_m": height_m,
                                           "throw_ratio": throw_ratio, "aim": aim, "side": side, "turn_deg": turn_deg,
                                           "margin": margin, "name": (name or "").strip()}))


class KUBA_ProjectorBlend(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="KUBA_ProjectorBlend",
            display_name="kubakub projector blend",
            category="kubakub/3d/scene",
            search_aliases=['edge blend', 'soft edge', 'overlap', 'multi projector'],
            description=(
                "Several projectors on one building: where their pictures overlap on the model, each one fades out "
                "towards the border of its own picture, so the light adds up evenly. Connect the scene of one kubakub "
                "scene render per projector (same file, each with its own kubakub projector). Gives one blend mask per "
                "projector, in that projector's picture size: multiply your frames with it, or load it as the blend "
                "mask in the media server."),
            inputs=[
                SceneType.Input("scene_1", tooltip="Projector 1: the scene of its kubakub scene render."),
                SceneType.Input("scene_2", tooltip="Projector 2."),
                SceneType.Input("scene_3", optional=True, tooltip="Projector 3."),
                SceneType.Input("scene_4", optional=True, tooltip="Projector 4."),
                io.Float.Input("ramp", default=0.1, min=0.005, max=0.5, step=0.005,
                               tooltip="How far from its border a picture reaches full strength, as a share of its "
                                       "short side. Larger = softer blends (the overlap must be at least this wide)."),
                io.Float.Input("gamma", default=2.2, min=1.0, max=3.0, step=0.05,
                               tooltip="The projectors' gamma: the masks are made so that the light adds up, not the "
                                       "pixel values. 1 = masks in linear light."),
            ],
            outputs=[
                io.Mask.Output("mask_1", tooltip="Blend mask of projector 1 (its picture size): 1 = full, less in overlaps."),
                io.Mask.Output("mask_2"), io.Mask.Output("mask_3"), io.Mask.Output("mask_4"),
                io.Image.Output("preview", tooltip="One picture per projector, at the size of the first: the clay with "
                                                   "what it lights alone and what it shares (orange)."),
                io.String.Output("report", tooltip="Per projector: how much of the model it lights alone and shares."),
            ],
        )

    @classmethod
    def execute(cls, scene_1, scene_2, ramp, gamma, scene_3=None, scene_4=None) -> io.NodeOutput:
        t0 = time.perf_counter()
        given = [s for s in (scene_1, scene_2, scene_3, scene_4) if s is not None]
        files = {os.path.normcase(bridge.clean_path(s["file"])) for s in given}
        if len(files) > 1:
            raise ValueError("kubakub projector blend: the scenes come from different files; every projector needs "
                             "the same model (one kubakub scene render per projector on the same file).")
        loaded = [scene_ids.load(s["folder"]) for s in given]
        res = sv.projector_blend(loaded, ramp=float(ramp), gamma=float(gamma))
        H0, W0 = loaded[0]["faceid"].shape
        masks, previews, lines = [], [], []
        for k, (s, sc, r) in enumerate(zip(given, loaded, res)):
            masks.append(torch.from_numpy(r["mask"])[None])
            clay = imio.imread(os.path.join(s["folder"], "clay.png"))[..., ::-1] / 255.0
            fg = sc["faceid"] > 0
            pic = clay * 0.25
            pic[fg & ~r["shared"]] = clay[fg & ~r["shared"]] * 0.9
            pic[r["shared"]] = clay[r["shared"]] * np.array([0.95, 0.54, 0.35])           # shared: orange
            pic = pic * (0.35 + 0.65 * r["mask"][..., None])
            if pic.shape[:2] != (H0, W0):
                pic = cv2.resize(pic, (W0, H0), interpolation=cv2.INTER_AREA)
            previews.append(pic)
            c = r["counts"]
            name = (s.get("projector") or {}).get("name") or f"projector {k + 1}"
            with_whom = ", ".join(f"{n * 100 // max(c['model'], 1)} % with {j + 1}" for j, n in enumerate(r["partners"]) if n)
            lines.append(f"{name}: {sc['info']['width']}x{sc['info']['height']}, lights {c['model'] * 100 // max(sc['faceid'].size, 1)} % "
                         f"of its picture with the model; {c['alone'] * 100 // max(c['model'], 1)} % of that alone, "
                         f"{c['shared'] * 100 // max(c['model'], 1)} % shared" + (f" ({with_whom})" if with_whom else ""))
        if not any(r["counts"]["shared"] for r in res):
            lines.append("NOTE: the pictures do not overlap on the model: nothing to blend (every mask is 1).")
        empty = torch.zeros((1, 64, 64), dtype=torch.float32)
        masks += [empty] * (4 - len(masks))
        lines.append(f"ramp {ramp:g} of the short side, gamma {gamma:g}, {time.perf_counter() - t0:.1f} s")
        report = "\n".join(lines)
        log.info("[KUBA scene3d] projector blend: %s", report.replace("\n", " | "))
        preview = torch.from_numpy(np.ascontiguousarray(np.stack(previews), dtype=np.float32))
        return io.NodeOutput(*masks, preview, report, ui=ui.PreviewImage(preview, cls=cls))


MAPS = {  # output map -> (key, invert colours, label)
    "brightness": ("brightness", False, "relative projector brightness (1 = typical facade)"),
    "distance": ("distance_m", True, "distance to the projector (m)"),
    "incidence": ("incidence_deg", False, "angle of the projector ray to the surface normal (deg)"),
    "pixel_mm": ("pixel_mm", False, "size of one matrix pixel on the building (mm)"),
    "offset": ("offset_m", False, "offset from the main wall (m, + = towards the audience)"),
}


def _load_scene(scene):
    s = scene_ids.load(scene["folder"])
    pt, nrm, ground = sv.main_plane_and_ground(s)
    return s, pt, nrm, ground


class KUBA_SceneMeasure(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="KUBA_SceneMeasure",
            display_name="kubakub scene measure",
            category="kubakub/3d/scene",
            search_aliases=['distance', 'metres', 'mm per pixel'],
            description=(
                "Real distances from the 3D scene: per pixel of the projection view the distance to the "
                "projector, the angle it hits the surface, the size of one matrix pixel on the building, the "
                "relative brightness and the offset from the wall; per region the same as medians plus area "
                "(written into the regions, report as a table). Also a viewer (for frame in frame) with the real facade width "
                "and your audience spot for frame in frame."),
            inputs=[
                SceneType.Input("scene", tooltip="From kubakub scene render."),
                RegionsType.Input("regions", optional=True,
                                  tooltip="Regions at the projection view's size (e.g. regions from id renders with the "
                                          "clay as matrix). They get the measurements and a 'grazing' tag."),
                io.Float.Input("viewer_x_m", default=0.0, min=-10000.0, max=10000.0, step=0.1,
                               tooltip="Audience spot along the wall, from the centre of the projector frame (m)."),
                io.Float.Input("viewer_distance_m", default=15.0, min=0.5, max=10000.0, step=0.5,
                               tooltip="Audience distance from the main wall (m)."),
                io.Float.Input("eye_height_m", default=1.7, min=0.0, max=1000.0, step=0.05,
                               tooltip="Eye height of the audience above the ground (m)."),
                io.Combo.Input("map", options=list(MAPS), default="brightness",
                               tooltip="Which measurement the map output shows in colour: brightness (1 = a "
                                       "typical facade), distance to the projector, incidence angle, mm per matrix "
                                       "pixel, or offset from the main wall."),
                io.Float.Input("grazing_deg", default=60.0, min=0.0, max=90.0, step=1.0,
                               tooltip="Regions hit at a steeper angle than this get the tag 'grazing' "
                                       "(stretched, dim projection)."),
            ],
            outputs=[
                RegionsType.Output("regions", tooltip="The regions with a 'scene' entry each (+ grazing tag)."),
                ViewerType.Output("viewer", tooltip="Facade width / bottom from the projector frame on the wall, "
                                                    "viewer at the audience spot."),
                io.Image.Output("map", tooltip="The chosen measurement as a colour map of the projection view."),
                io.String.Output("table", tooltip="Per region: distance, offset, incidence, mm per px, "
                                                  "brightness, area, viewer distance (tab separated)."),
                io.String.Output("report", tooltip="Projector frame on the wall, median measurements, the viewer and the map's colour range."),
            ],
        )

    @classmethod
    def execute(cls, scene, viewer_x_m, viewer_distance_m, eye_height_m, map, grazing_deg,
                regions=None) -> io.NodeOutput:
        t0 = time.perf_counter()
        s, pt, nrm, ground = _load_scene(scene)
        info = s["info"]
        fr = sv.wall_frame(info, pt, nrm, ground)
        maps = sv.measure_maps(s, pt, nrm)
        fg = s["faceid"] > 0
        H, W = fg.shape
        spot = sv.spot_position(fr, viewer_x_m, viewer_distance_m, eye_height_m)
        viewer = vw.Viewer(W, H, fr["width_m"], fr["bottom_m"], fr["width_m"] / 2 + viewer_x_m, eye_height_m,
                           viewer_distance_m)
        key, inv, label = MAPS[map]
        rgb, lo, hi = sv.colorize(maps[key], fg, invert=inv)

        out_regions, table = regions, ""
        if regions is not None:
            lab = regions.labels[0].cpu().numpy()
            if lab.shape != (H, W):
                lab = cv2.resize(lab.astype(np.int32), (W, H), interpolation=cv2.INTER_NEAREST)
                note = f"regions {regions.size[0]}x{regions.size[1]} resized to the projection view {W}x{H}"
            else:
                note = ""
            stats = sv.region_stats(lab, maps, spot, s["position"])
            tab = json.loads(json.dumps(regions.table))
            rows = ["region\tdistance_m\toffset_m\tincidence_deg\tpixel_mm\tbrightness\tarea_m2\tviewer_m"]
            n_graz = 0
            for rid, e in stats.items():
                if rid >= len(tab["regions"]):
                    continue
                r = tab["regions"][rid]
                r["scene"] = e
                tags = list(r.get("tags") or [])
                if e["incidence_deg"] > grazing_deg and "grazing" not in tags:
                    tags.append("grazing")
                    n_graz += 1
                r["tags"] = tags
                rows.append(f"{r.get('name', rid)}\t{e['distance_m']}\t{e['offset_m']}\t{e['incidence_deg']}\t"
                            f"{e['pixel_mm']}\t{e['brightness']}\t{e['area_m2']}\t{e.get('viewer_distance_m', '')}")
            if note:
                tab.setdefault("notes", []).append(note)
            out_regions = Regions(regions.labels, tab, regions.scope)
            table = "\n".join(rows)
            region_line = f"{len(stats)} regions measured, {n_graz} tagged grazing (> {grazing_deg:g} deg)"
        else:
            region_line = "no regions connected"
        med = {k: float(np.median(v[fg])) for k, v in maps.items()} if fg.any() else {}
        report = "\n".join([
            f"projector frame on the wall: {fr['width_m']:.2f} x {fr['height_m']:.2f} m, bottom edge "
            f"{fr['bottom_m']:.2f} m above the ground, projector {fr['projector_distance_m']:.1f} m from the wall",
            f"medians: distance {med.get('distance_m', 0):.1f} m, incidence {med.get('incidence_deg', 0):.0f} deg, "
            f"{med.get('pixel_mm', 0):.1f} mm per matrix pixel, brightness {med.get('brightness', 0):.2f}",
            f"viewer: {viewer_distance_m:g} m from the wall, {viewer_x_m:+g} m from the frame centre, eye "
            f"{eye_height_m:g} m -> viewer: facade {fr['width_m']:.2f} m wide, bottom {fr['bottom_m']:.2f} m",
            f"map '{map}': {label}, colours {lo:.3g} .. {hi:.3g}",
            region_line, f"{time.perf_counter() - t0:.1f} s"])
        log.info("[KUBA scene3d] measure: %s", report.replace("\n", "\n    "))
        img = _img(rgb)
        return io.NodeOutput(out_regions, viewer, img, table, report, ui=ui.PreviewImage(img, cls=cls))


class KUBA_ScenePreview(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="KUBA_ScenePreview",
            display_name="kubakub scene preview",
            category="kubakub/3d/scene",
            search_aliases=['previz', 'audience'],
            description=(
                "Shows how the projection will look to the audience: the matrix (image or video frames) projected from the file's camera onto the model and "
                "seen from audience spots. Where the projector does not reach a surface the audience sees, it "
                "stays dark (projection shadow, also as a mask). The mapping is computed once per spot, so "
                "video frames are cheap. One Blender render per spot (cached)."),
            inputs=[
                SceneType.Input("scene", tooltip="From kubakub scene render."),
                io.Image.Input("matrix", tooltip="Matrix image or frames (resized to the projection view if needed)."),
                io.String.Input("spots", multiline=True, default="0, 20, 1.7\n-15, 12, 1.7\n12, 8, 1.7",
                                tooltip="One audience spot per line: x along the wall from the frame centre, "
                                        "distance from the wall, eye height (m)."),
                io.Float.Input("lens_mm", default=20.0, min=6.0, max=300.0, step=1.0,
                               tooltip="Audience camera lens (36 mm sensor); 20 = wide, 35 = natural."),
                io.Int.Input("width", default=1280, min=64, max=8192, step=16,
                             tooltip="Width of each audience view, in pixels."),
                io.Int.Input("height", default=720, min=64, max=8192, step=16,
                             tooltip="Height of each audience view, in pixels."),
                io.Float.Input("ambient", default=0.15, min=0.0, max=1.0, step=0.01,
                               tooltip="Brightness of the unlit building (clay) around the projection."),
                io.Float.Input("gain", default=1.0, min=0.0, max=4.0, step=0.05,
                               tooltip="Brightness of the projected image; 1 = as it is, 2 = twice as bright."),
                io.Float.Input("physical", default=1.0, min=0.0, max=1.0, step=0.05,
                               tooltip="1 = surfaces hit at a steep angle or far away get darker, as with a "
                                       "real projector; 0 = flat."),
                io.Image.Input("background", optional=True,
                               tooltip="Behind the building (still or frames, looping), scaled to cover the view."),
                io.Color.Input("clay_color", default=sv.CLAY_COLOR, optional=True,
                               tooltip="Colour of the clay: the building and its surroundings where no picture lands. "
                                       "Almost white by default; pick the stone's colour for a closer look."),
                io.Combo.Input("shader", options=list(sv.SHADERS), default="clay", optional=True,
                               tooltip="How the model is drawn where no picture lands: clay, wireframe (its edges as "
                                       "lines in the clay colour, not dimmed by ambient) or clay with the edges on it."),
            ],
            outputs=[
                io.Image.Output("preview", tooltip="Per spot all frames (spot-major batch)."),
                io.Mask.Output("shadow", tooltip="Per spot: building the audience sees but the projector misses."),
                io.Image.Output("views", tooltip="Per spot the clay view (no projection)."),
                io.String.Output("report", tooltip="Per spot: how much of the view is building and how much of it the projector misses."),
                SceneType.Output("scene", tooltip="The scene with these viewpoints (spots, lens, size) -> kubakub scene "
                                                  "relight with camera = previz: the lit picture from the same spot."),
            ],
        )

    @classmethod
    def execute(cls, scene, matrix, spots, lens_mm, width, height, ambient, gain, physical,
                background=None, clay_color=sv.CLAY_COLOR, shader="clay") -> io.NodeOutput:
        t0 = time.perf_counter()
        s, pt, nrm, ground = _load_scene(scene)
        info = s["info"]
        fr = sv.wall_frame(info, pt, nrm, ground)
        H, W = int(info["height"]), int(info["width"])
        frames = matrix[..., :3].cpu().float().numpy()
        if frames.shape[1:3] != (H, W):
            frames = np.stack([cv2.resize(f, (W, H), interpolation=cv2.INTER_AREA) for f in frames])
            resized = f"matrix {matrix.shape[2]}x{matrix.shape[1]} resized to the projection view {W}x{H}"
        else:
            resized = ""
        bgs = None
        if background is not None:
            bgs = [sv.fit_background(b, width, height) for b in background[..., :3].cpu().float().numpy()]
        root = cache_root()
        previews, shadows, views, lines, viewpoints = [], [], [], [], []
        for i, (x, d, eye) in enumerate(sv.parse_spots(spots)):
            loc = sv.spot_position(fr, x, d, eye)
            view = {"location": [round(float(v), 4) for v in loc], "look_at": [round(float(v), 4) for v in fr["centre"]],
                    "lens": float(lens_mm), "name": f"spot_{i + 1}"}
            viewpoints.append(dict(view, size=[int(width), int(height)]))
            vf, cached, _ = bridge.export(scene["file"], root, width=width, height=height, view=view,
                                          blender=scene.get("blender", ""), **bridge.scene_opts(scene))
            vs = scene_ids.load(vf)
            rp = sv.reprojection(vs, s)
            clay, full = sv.look_picture(vs, imio.imread(os.path.join(vf, "clay.png"))[..., ::-1].astype(np.float32) / 255,
                                         shader, clay_color)
            for fi, f in enumerate(frames):
                previews.append(sv.render_preview(f, rp, clay, ambient=1.0 if full else ambient, gain=gain, physical=physical,
                                                  background=None if bgs is None else bgs[fi % len(bgs)]))
            shadows.append(rp["shadow"].astype(np.float32))
            views.append(clay)
            model = max(int(rp["building"].sum()), 1)
            lines.append(f"spot {i + 1} (x {x:+g}, {d:g} m, eye {eye:g}): building {rp['building'].mean() * 100:.0f} % "
                         f"of the view, lit {rp['lit'].sum() / model * 100:.0f} %, projection shadow "
                         f"{rp['shadow'].sum() / model * 100:.1f} %{' (cached)' if cached else ''}")
        _after_render()
        report = "\n".join([f"{len(lines)} spot(s) x {len(frames)} frame(s), {time.perf_counter() - t0:.1f} s",
                            *lines, *([resized] if resized else []),
                            *(["with surroundings"] if scene.get("surroundings") else [])])
        log.info("[KUBA scene3d] preview: %s", report.replace("\n", "\n    "))
        out = torch.from_numpy(np.stack(previews))
        return io.NodeOutput(out, torch.from_numpy(np.stack(shadows)), torch.from_numpy(np.stack(views)), report,
                             dict(scene, viewpoints=viewpoints), ui=ui.PreviewImage(out, cls=cls))


def _frames_at(matrix, W, H):
    """Matrix batch as numpy frames at the projection view size (+ a note if resized)."""
    frames = matrix[..., :3].cpu().float().numpy()
    if frames.shape[1:3] != (H, W):
        frames = np.stack([cv2.resize(f, (W, H), interpolation=cv2.INTER_AREA) for f in frames])
        return frames, f"matrix {matrix.shape[2]}x{matrix.shape[1]} resized to the projection view {W}x{H}"
    return frames, ""


class KUBA_SceneWalkthrough(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="KUBA_SceneWalkthrough",
            display_name="kubakub scene walkthrough",
            category="kubakub/3d/scene",
            search_aliases=['previz video', 'camera path'],
            description=(
                "Makes a previz video of the projection as someone walking past would see it: a camera walks through the audience along keyframes while the matrix (a still or "
                "video frames, frame i of the walk shows matrix frame i, looping) is projected onto the model. "
                "Optional background image / frames behind the building. Output: IMAGE frames (-> Create Video / "
                "Save Video). The camera views are kept on disk: a changed path or matrix renders only new "
                "views (25 per Blender job)."),
            inputs=[
                SceneType.Input("scene", tooltip="From kubakub scene render."),
                io.Image.Input("matrix", tooltip="Matrix image or frames."),
                io.String.Input("path", multiline=True, default="-15, 25, 1.7\n0, 15, 1.7\n15, 10, 1.7",
                                tooltip="Keyframes, one per line: x along the wall from the frame centre, distance "
                                        "from the wall, eye height[, look x, look height] (m). Spread evenly over "
                                        "the walk, smooth spline between them. No look = the frame centre."),
                io.Float.Input("seconds", default=8.0, min=0.1, max=600.0, step=0.5,
                               tooltip="Length of the walk in seconds (frames = seconds x fps)."),
                io.Int.Input("fps", default=25, min=1, max=120,
                             tooltip="Frames per second of the video."),
                io.Float.Input("lens_mm", default=24.0, min=6.0, max=300.0, step=1.0,
                               tooltip="Walking camera lens (36 mm sensor); 20 = wide, 35 = natural."),
                io.Int.Input("width", default=960, min=64, max=8192, step=16,
                             tooltip="Video width in pixels."),
                io.Int.Input("height", default=540, min=64, max=8192, step=16,
                             tooltip="Video height in pixels."),
                io.Float.Input("ambient", default=0.15, min=0.0, max=1.0, step=0.01,
                               tooltip="Brightness of the unlit building (clay) around the projection."),
                io.Float.Input("gain", default=1.0, min=0.0, max=4.0, step=0.05,
                               tooltip="Brightness of the projected image; 1 = as it is, 2 = twice as bright."),
                io.Float.Input("physical", default=1.0, min=0.0, max=1.0, step=0.05,
                               tooltip="1 = surfaces hit at a steep angle or far away get darker, as with a "
                                       "real projector; 0 = flat."),
                io.Image.Input("background", optional=True,
                               tooltip="Behind the building: a still or frames (looping), scaled to cover the view. "
                                       "Default black."),
                io.Color.Input("clay_color", default=sv.CLAY_COLOR, optional=True,
                               tooltip="Colour of the clay: the building and its surroundings where no picture lands. "
                                       "Almost white by default; pick the stone's colour for a closer look."),
                io.Combo.Input("shader", options=list(sv.SHADERS), default="clay", optional=True,
                               tooltip="How the model is drawn where no picture lands: clay, wireframe (its edges as "
                                       "lines in the clay colour, not dimmed by ambient) or clay with the edges on it."),
            ],
            outputs=[
                io.Image.Output("frames", tooltip="The walk as video frames (-> Create Video / Save Video)."),
                io.String.Output("report", tooltip="Frame count, time, views rendered new or from the cache, and the projection shadow share."),
            ],
        )

    @classmethod
    def execute(cls, scene, matrix, path, seconds, fps, lens_mm, width, height, ambient, gain, physical,
                background=None, clay_color=sv.CLAY_COLOR, shader="clay") -> io.NodeOutput:
        import comfy.utils
        t0 = time.perf_counter()
        s, pt, nrm, ground = _load_scene(scene)
        info = s["info"]
        fr = sv.wall_frame(info, pt, nrm, ground)
        frames, resized = _frames_at(matrix, int(info["width"]), int(info["height"]))
        n = max(1, int(round(seconds * fps)))
        cams = sv.camera_path(fr, sv.parse_path(path), n)
        bgs = None
        if background is not None:
            bgs = [sv.fit_background(b, width, height) for b in background[..., :3].cpu().float().numpy()]
        root = cache_root()
        walk = os.path.join(root, "walk")                   # one folder per camera: a changed path renders only new views
        opts = dict(bridge.scene_opts(scene), width=width, height=height)
        pbar = comfy.utils.ProgressBar(n)
        out, shadow_share, chunk, rendered = [], [], 25, 0
        for c0 in range(0, n, chunk):
            views = [{"location": [round(float(v), 4) for v in loc], "look_at": [round(float(v), 4) for v in look],
                      "lens": float(lens_mm)} for loc, look in cams[c0:c0 + chunk]]
            dirs = [bridge.cache_folder(scene["file"], walk, views=[v], **opts) for v in views]
            need = [k for k, d in enumerate(dirs) if not bridge.is_done(d, VIEW_FILES)]
            if need:                                       # the missing views in one Blender job
                vf, _, _ = bridge.export(scene["file"], root, blender=scene.get("blender", ""), transient=True,
                                         views=[views[k] for k in need], **opts)
                try:
                    for m, k in enumerate(need):
                        src = os.path.join(vf, f"v_{m:04d}")
                        shutil.copy(os.path.join(vf, "scene.json"), os.path.join(src, "scene.json"))
                        bridge.publish(src, dirs[k], VIEW_FILES)
                finally:
                    shutil.rmtree(vf, ignore_errors=True)
                rendered += len(need)
            for k, d in enumerate(dirs):
                if k not in need:
                    bridge.touch(d)
                with open(os.path.join(d, "scene.json"), encoding="utf-8") as f:
                    vinfo = json.load(f)
                vs = {"info": vinfo, "faceid": np.load(os.path.join(d, "faceid.npy")),
                      "position": np.load(os.path.join(d, "position.npy")),
                      "normal": np.load(os.path.join(d, "normal.npy")).astype(np.float32)}
                rp = sv.reprojection(vs, s)
                clay, full = sv.look_picture(vs, imio.imread(os.path.join(d, "clay.png"))[..., ::-1].astype(np.float32) / 255,
                                             shader, clay_color)
                i = c0 + k
                out.append(sv.render_preview(frames[i % len(frames)], rp, clay, ambient=1.0 if full else ambient, gain=gain,
                                             physical=physical, background=None if bgs is None else bgs[i % len(bgs)]))
                shadow_share.append(rp["shadow"].sum() / max(int(rp["building"].sum()), 1))
                pbar.update(1)
        if rendered:
            _after_render()
        report = "\n".join([
            f"{n} frames ({seconds:g} s at {fps} fps), {width}x{height}, {time.perf_counter() - t0:.1f} s "
            f"({(time.perf_counter() - t0) / n:.2f} s per frame), {rendered} view(s) rendered, "
            f"{n - rendered} from the cache",
            f"path: {len(sv.parse_path(path))} keys; projection shadow {min(shadow_share) * 100:.1f} - "
            f"{max(shadow_share) * 100:.1f} % of the visible building",
            f"matrix: {len(frames)} frame(s){' (looping)' if len(frames) < n else ''}; background: "
            + ("none" if bgs is None else f"{len(bgs)} frame(s)"), *([resized] if resized else [])])
        log.info("[KUBA scene3d] walkthrough: %s", report.replace("\n", "\n    "))
        return io.NodeOutput(torch.from_numpy(np.stack(out)), report)


DEFAULT_LIGHTS = """// type  x  height  distance  watts  #colour  size / angle   (metres from the frame centre, the ground, the wall)
area   -9  6   6   900  #ffd2a0  4
area    9  6   6   900  #ffd2a0  4
point   0  1   1   150  #ff9a50  0.3
"""


class KUBA_SceneRelight(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="KUBA_SceneRelight",
            display_name="kubakub scene relight",
            category="kubakub/3d/scene",
            search_aliases=['blender', 'cycles', 'hdri', 'lamps'],
            description=(
                "Shows the building lit at night or by day, with lamps, glowing areas or the projection itself as "
                "light. Relights the model in Blender Cycles from the projection camera: an HDRI environment (built in or "
                "your own), point / area / spot / sun lights placed in facade metres, and masks that make the faces "
                "they cover glow (emissive material that lights the stone around it). Shadows and light bounces "
                "included. The image is at the matrix size, ready as a base or layer for the director."),
            inputs=[
                SceneType.Input("scene", tooltip="From kubakub scene render."),
                io.Combo.Input("environment", options=list(sv.ENVIRONMENTS), default="night",
                               tooltip="Blender's built-in HDRIs, 'file' = hdri_file, 'none' = only your lights."),
                io.Float.Input("env_strength", default=0.3, min=0.0, max=50.0, step=0.05,
                               tooltip="Brightness of the HDRI environment light; 0 = only your lights."),
                io.Float.Input("env_rotation_deg", advanced=True, default=0.0, min=-360.0, max=360.0, step=5.0,
                               tooltip="Turns the HDRI around the vertical axis (degrees), to move the sun or sky glow."),
                io.String.Input("lights", multiline=True, default=DEFAULT_LIGHTS,
                                tooltip="One light per line: 'point|area|spot x height distance watts [#colour] [size|angle]' "
                                        "in metres (x from the frame centre along the wall, height above the ground, "
                                        "distance from the wall), or 'sun azimuth elevation strength [#colour]'."),
                io.Float.Input("clay", default=0.7, min=0.0, max=1.0, step=0.01, tooltip="Brightness of the clay material."),
                io.Float.Input("roughness", advanced=True, default=0.8, min=0.0, max=1.0, step=0.01,
                               tooltip="Roughness of the clay material: 1 = matte stone, lower = shinier."),
                io.Int.Input("samples", advanced=True, default=64, min=1, max=4096, tooltip="Cycles samples (denoised)."),
                io.Float.Input("exposure", default=0.0, min=-10.0, max=10.0, step=0.1,
                               tooltip="Overall exposure in stops: +1 = twice as bright, -1 = half."),
                io.Combo.Input("background", options=["black", "environment", "transparent"], default="black",
                               tooltip="What shows where there is no building."),
                io.Float.Input("resolution_scale", advanced=True, default=1.0, min=0.1, max=1.0, step=0.05,
                               tooltip="Render smaller for quick looks; the output is scaled back to the matrix size."),
                io.Mask.Input("emission_masks", optional=True,
                              tooltip="Masks at matrix size (director layer masks, regions mask ...): faces covered by "
                                      "a mask glow. A batch = several emitters."),
                io.String.Input("emission_colors", advanced=True, default="#ffb060", optional=True,
                                tooltip="One colour per mask, comma separated (the last one repeats)."),
                io.Float.Input("emission_strength", advanced=True, default=20.0, min=0.0, max=10000.0, step=1.0, optional=True,
                               tooltip="Emission strength of the glowing faces."),
                io.String.Input("hdri_file", advanced=True, default="", optional=True, tooltip="An .hdr / .exr for environment = file."),
                io.Image.Input("projector", display_name="matrix", optional=True,
                               tooltip="Your projection picture (the matrix, as on scene preview): it is cast from the "
                                       "projection camera like a real projector, with falloff, surface angle and the "
                                       "shadows other lamps would not show. With camera = audience you see the "
                                       "building with your picture on it, in its street, under your lamps."),
                io.Float.Input("projector_brightness", advanced=True, default=1.0, min=0.0, max=100.0, step=0.05, optional=True,
                               tooltip="1 = the image lands at about its own brightness on a frontal wall."),
                io.Combo.Input("projector_mode", options=["light", "paint"], default="light", optional=True,
                               tooltip="light = cast from the camera like a real projector; paint = the image becomes "
                                       "the model's colour (textures on the building), lit by the lamps and HDRI."),
                io.Combo.Input("view", options=["AgX", "Standard"], default="AgX", optional=True,
                               tooltip="Tone mapping: AgX = soft highlights (lamps), Standard = the projected image's "
                                       "colours as they are."),
                io.Color.Input("clay_color", default=sv.RELIGHT_CLAY_COLOR, optional=True,
                               tooltip="Colour of the clay material (the building and its surroundings); 'clay' sets "
                                       "how bright it is. White = neutral grey clay, as before."),
                io.Combo.Input("camera", options=["projector", "audience", "previz"], default="projector", optional=True,
                               tooltip="projector = the matrix view, the building alone (content, a base for the "
                                       "director). audience = from a spot in front of the building, with its "
                                       "surroundings (kubakub scene surroundings): the lit place as a visitor sees "
                                       "it; drag the ring on the plan to move it. previz = the very viewpoint of "
                                       "kubakub scene preview (connect its scene output): same spot, lens and size."),
                io.Float.Input("audience_distance_m", default=30.0, min=0.5, max=2000.0, step=0.5, optional=True,
                               tooltip="camera = audience: metres in front of the wall (the 'distance' of a scene "
                                       "preview spot)."),
                io.Float.Input("audience_offset_m", default=0.0, min=-1000.0, max=1000.0, step=0.5, optional=True,
                               tooltip="camera = audience: metres left (-) or right (+) along the wall (the 'x' of a "
                                       "scene preview spot)."),
                io.Float.Input("eye_height_m", default=1.7, min=0.0, max=500.0, step=0.1, optional=True, advanced=True,
                               tooltip="camera = audience: eye height above the ground; 20-60 for a view from above."),
                io.Float.Input("lens_mm", default=24.0, min=6.0, max=300.0, step=1.0, optional=True, advanced=True,
                               tooltip="camera = audience: focal length (smaller = wider)."),
                io.Float.Input("plan_width_m", default=0.0, min=0.0, max=3000.0, step=10.0, optional=True, advanced=True,
                               tooltip="How many metres the light plan shows across. 0 = fitted to the lamps, the "
                                       "projector and the audience; larger to see more of the surroundings."),
                io.Int.Input("previz_spot", default=1, min=1, max=64, optional=True, advanced=True,
                             tooltip="camera = previz: which spot of scene preview (its line in 'spots')."),
                io.Float.Input("audience_turn_deg", default=0.0, min=-180.0, max=180.0, step=1.0, optional=True,
                               tooltip="camera = audience: turns the camera away from the middle of the facade, in "
                                       "degrees (+ = to the right). On the plan: drag the small dot in front of the "
                                       "ring."),
            ],
            outputs=[
                io.Image.Output("image", tooltip="The relit clay from the projection camera, or from the audience."),
                io.Mask.Output("alpha", tooltip="1 where the building is (camera = audience: and its surroundings)."),
                io.String.Output("report", tooltip="Size, lights, environment, samples, glowing faces and render time."),
                io.Image.Output("plan", tooltip="Top view of the light plan, the audience at the bottom: every lamp "
                                                "in its colour with its number (the line in 'lights') and height, "
                                                "the model (orange), the projector (dot), the audience camera "
                                                "(ring) and the surroundings (grey)."),
            ],
        )

    @classmethod
    def fingerprint_inputs(cls, environment="night", hdri_file="", **kwargs):
        # hdri_file is a text widget: an HDRI saved again under the same name has to render again
        return hdri_stamp(hdri_file) if environment == "file" else ""

    @classmethod
    def execute(cls, scene, environment, env_strength, env_rotation_deg, lights, clay, roughness, samples, exposure,
                background, resolution_scale, emission_masks=None, emission_colors="#ffb060", emission_strength=20.0,
                hdri_file="", projector=None, projector_brightness=1.0, projector_mode="light", view="AgX",
                clay_color=sv.RELIGHT_CLAY_COLOR, camera="projector", audience_distance_m=30.0, audience_offset_m=0.0,
                eye_height_m=1.7, lens_mm=24.0, plan_width_m=0.0, previz_spot=1, audience_turn_deg=0.0) -> io.NodeOutput:
        rig = {"view": view, "clay_color": clay_color, "projector": {"on": projector is not None, "brightness": projector_brightness,
                                           "mode": projector_mode},"environment": environment, "env_strength": env_strength, "env_rotation": env_rotation_deg,
               "exposure": exposure, "clay": clay, "roughness": roughness, "samples": samples, "background": background,
               "resolution_scale": resolution_scale, "hdri_file": hdri_file, "lights": sv.parse_lights(lights)}
        emission = []
        if emission_masks is not None:
            masks = emission_masks if emission_masks.ndim == 3 else emission_masks[None]
            cols = [sv._hex_rgb(c) for c in emission_colors.split(",") if c.strip()] or [(1.0, 0.69, 0.38)]
            for k, m in enumerate(masks.cpu().float().numpy()):
                emission.append((m, cols[min(k, len(cols) - 1)], float(emission_strength), f"emission mask {k + 1}"))
        pimg = None if projector is None else projector[0, ..., :3].cpu().float().numpy()
        s, pt, nrm, ground, fr = load_scene_cached(scene)
        cam = np.asarray(s["info"]["camera"]["matrix_world"], np.float64)[:3, 3]
        look_from = None
        if camera == "audience":                              # as a scene preview spot: x, distance, eye; looks at the frame's middle
            loc = sv.spot_position(fr, audience_offset_m, audience_distance_m, eye_height_m)
            look = np.asarray(fr["centre"], float) - loc
            a = -np.radians(float(audience_turn_deg))         # + = to the right = clockwise seen from above
            look[:2] = [np.cos(a) * look[0] - np.sin(a) * look[1], np.sin(a) * look[0] + np.cos(a) * look[1]]
            look_from = {"location": [round(float(v), 4) for v in loc], "lens": float(lens_mm),
                         "look_at": [round(float(v), 4) for v in loc + look]}
        elif camera == "previz":
            spots = scene.get("viewpoints") or []
            if not spots:
                raise ValueError("camera = previz needs the viewpoints of kubakub scene preview: connect its 'scene' "
                                 "output to this node's scene")
            look_from = dict(spots[min(int(previz_spot), len(spots)) - 1])
        rgb, alpha, report = relight_scene(scene, rig, emission=emission, projector=pimg, look_from=look_from)
        top = rgb.max(-1)
        white = (top > 0.98) & (alpha > 0.5)
        burnt = float(white.mean())                           # share of the whole picture that is clipped
        if burnt > 0.03:
            report += (f"\nNOTE: {burnt * 100:.0f} % of the picture is burnt out to white. Lower exposure (try -1.5), the "
                       "lamps' watts, env_strength or projector_brightness; a projection only shows where the lamps "
                       "leave the wall dark.")
        sur = scene.get("surroundings") or {}
        buildings = sr.read(sur.get("file", "")) if sur else []
        if look_from and sur:                                 # a camera inside a neighbour sees a black room
            if sr.stands_inside(buildings, sur["matrix"], look_from["location"]):
                report += ("\nWARNING: the audience camera stands inside a neighbouring building (the ring on the "
                           "plan): change audience_distance_m / audience_offset_m.")
        elif look_from:
            report += "\nno surroundings on this scene: connect the scene of kubakub scene surroundings to see the street"
        out = _img(rgb)
        plan, meta = light_plan(s, pt, nrm, ground, fr, rig["lights"], cam, buildings, sur.get("matrix") or np.eye(4),
                                look_from, across_m=float(plan_width_m))
        meta["camera"] = camera
        # the node shows its light plan (web/kubakub_light_plan.js) where it used to show the render: connect a
        # preview to 'image'
        return io.NodeOutput(out, torch.from_numpy(alpha)[None], report, _img(plan), ui={"kubakub_light_plan": [meta]})


def light_plan(s, pt, nrm, ground, fr, lights, projector, buildings, matrix, look_from=None, size=768, across_m=0.0):
    """
    The top view of a light rig: lamps (facade metres -> world), model, projector, audience camera and the
    buildings around (surroundings.read, with their placement matrix).
    -> (picture with the lamps, meta for the plan on the node: the same picture without lamps as a temp file, and
    where a lamp at x / distance lands on it: origin + x * ax + distance * ad, in pixels).
    """
    V = s["mesh"]["vert"]
    anchor, nh, ex = sr.facade_anchor(V, pt, nrm, ground)
    world = sv.lights_to_world(lights, fr)
    viewer = np.asarray(look_from["location"], float) if look_from else None
    reach = [np.linalg.norm((np.asarray(q, float) - anchor)[:2]) for q in
             [projector] + [L["location"] for L in world if L["type"] != "sun"] + ([viewer] if viewer is not None else [])]
    width = float((V @ ex).max() - (V @ ex).min())
    # fitted in steps of 20 m, so the plan does not change its scale with every small move of a lamp or the camera
    radius = across_m / 2 if across_m > 0 else np.ceil(max(40.0, 1.2 * width, 1.25 * max(reach)) / 20.0) * 20.0
    bare = sr.plan(buildings, matrix, anchor, nh, ex, V[::max(1, len(V) // 4000)], radius, size=size,
                   projector=projector)                       # no audience ring: the plan on the node draws and drags it
    png = (np.clip(bare[..., ::-1], 0, 1) * 255 + 0.5).astype(np.uint8)
    temp = folder_paths.get_temp_directory()
    name = "kubakub_lightplan_" + hashlib.sha1(png.tobytes()).hexdigest()[:16] + ".png"
    if not os.path.isfile(os.path.join(temp, name)):
        os.makedirs(temp, exist_ok=True)
        imio.imwrite(os.path.join(temp, name), png)
    k = size / (2.0 * radius)
    axis = lambda v: [round(float(np.asarray(v, float) @ ex) * k, 4), round(float(np.asarray(v, float) @ nh) * k, 4)]  # noqa: E731
    origin = axis(np.asarray(fr["centre"], float) - anchor)
    meta = {"filename": name, "subfolder": "", "type": "temp",
            "origin": [round(size / 2 + origin[0], 2), round(size / 2 + origin[1], 2)], "ax": axis(fr["ex"]),
            "ad": axis(fr["normal"])}
    if viewer is not None:                                  # where the camera stands, for the plan on the node
        meta["viewer"] = [round(size / 2 + float((viewer - anchor) @ ex) * k, 2), round(size / 2 + float((viewer - anchor) @ nh) * k, 2)]
    return sr.plan_lights(bare, world, anchor, nh, ex, radius, viewer=viewer), meta


_SCENE_CACHE = {}


def load_scene_cached(scene):
    """_load_scene() plus the wall frame, kept for the last few scene folders (director previews reuse it)."""
    folder = scene["folder"]
    try:
        stamp = os.path.getmtime(os.path.join(folder, "faceid.npy"))
    except OSError:
        stamp = 0
    hit = _SCENE_CACHE.get(folder)
    if hit and hit[0] == stamp:
        return hit[1]
    s, pt, nrm, ground = _load_scene(scene)
    got = (s, pt, nrm, ground, sv.wall_frame(s["info"], pt, nrm, ground))
    while len(_SCENE_CACHE) >= 3:
        _SCENE_CACHE.pop(next(iter(_SCENE_CACHE)))
    _SCENE_CACHE[folder] = (stamp, got)
    return got


def _rig_job(s, fr, rig, emission, projector, root, notes):
    """One rig -> the Blender job part for it (world lights, environment, emissive faces, projector image)."""
    rig = dict(rig)
    rig.setdefault("lights", [])
    world = sv.lights_to_world(rig["lights"], fr)
    environment = rig.get("environment", "night")
    hdri_file = str(rig.get("hdri_file") or "")
    env = "" if environment == "none" else (bridge.clean_path(hdri_file) if environment == "file" else environment)
    if environment == "file" and not os.path.isfile(env):
        raise FileNotFoundError(f"hdri_file not found: {hdri_file}")
    emit = []
    if emission:
        emit_dir = os.path.join(root, "emit")
        os.makedirs(emit_dir, exist_ok=True)
        for m, col, strength, label in emission:
            faces = sv.emissive_faces(s["faceid"], np.asarray(m, np.float32))
            if not len(faces):
                notes.append(f"{label} covers no face")
                continue
            key = hashlib.sha1(faces.tobytes()).hexdigest()[:16]
            fp = os.path.join(emit_dir, f"faces_{key}.npy")
            if not os.path.isfile(fp):
                np.save(fp, faces)
            emit.append({"faces": fp, "color": [float(c) for c in col], "strength": float(strength), "count": int(len(faces))})
    part = {"exposure": float(rig.get("exposure", 0.0)), "background": rig.get("background", "black"),
            "env": env, "env_strength": float(rig.get("env_strength", 0.3)), "env_rotation": float(rig.get("env_rotation", 0.0)),
            "albedo": sv.clay_albedo(rig.get("clay", 0.7), rig.get("clay_color")), "roughness": float(rig.get("roughness", 0.8)), "lights": world, "emit": emit,
            "view": rig.get("view", "AgX")}
    pj = rig.get("projector") or {}
    if projector is not None and pj.get("on"):
        a = np.clip(np.asarray(projector, np.float32)[..., :3] * 255 + 0.5, 0, 255).astype(np.uint8)
        pdir = os.path.join(root, "projector")
        os.makedirs(pdir, exist_ok=True)
        fp = os.path.join(pdir, hashlib.sha1(a.tobytes() + str(a.shape).encode()).hexdigest()[:16] + ".png")
        if not os.path.isfile(fp):
            imio.imwrite(fp, cv2.cvtColor(a, cv2.COLOR_RGB2BGR))
        if pj.get("mode") == "paint":
            part["projector"] = {"image": fp, "mode": "paint", "size": [int(s["info"]["width"]), int(s["info"]["height"])]}
            notes.append(f"painted: {a.shape[1]}x{a.shape[0]} image as the model's colour from the camera")
        else:
            power = sv.projector_power(pj.get("brightness", 1.0), fr["projector_distance_m"], rig.get("clay", 0.7))
            part["projector"] = {"image": fp, "power": power, "size": [int(s["info"]["width"]), int(s["info"]["height"])]}
            notes.append(f"projector: {a.shape[1]}x{a.shape[0]} image from the camera, {power / 1000:.0f} kW spot")
    return part, world, environment, env


def _render_size(info, size, scale):
    W, H = size or (int(info["width"]), int(info["height"]))
    sw, sh = int(info["width"]), int(info["height"])          # render in the scene's camera aspect
    k = scale * max(W / sw, H / sh)
    return W, H, max(16, round(sw * k)), max(16, round(sh * k))


def _read_relit(path, W, H, rw, rh, background):
    img = imio.imread(path, cv2.IMREAD_UNCHANGED)
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA).astype(np.float32) / (65535.0 if img.dtype == np.uint16 else 255.0)
    if (rw, rh) != (W, H):
        img = cv2.resize(img, (W, H), interpolation=cv2.INTER_CUBIC if rw < W else cv2.INTER_AREA)
    rgb, alpha = np.clip(img[..., :3], 0, 1), np.clip(img[..., 3], 0, 1)
    if background == "black":
        rgb = rgb * alpha[..., None]
    return rgb, alpha


def hdri_stamp(hdri_file):
    """Path + size + mtime of an HDRI file (for fingerprints: an edited file with the same name is new)."""
    p = bridge.clean_path(hdri_file)
    if not p:
        return ""
    try:
        st = os.stat(p)
        return f"{p}|{st.st_size}|{st.st_mtime_ns}"
    except OSError:
        return p


def relight_frame_key(scene, rw, rh, samples, part):
    """
    Content key of one relight render: the scene file (path, size, mtime), camera, frame, scale, render size,
    samples, the rig part (lights, world, glow faces, projector image) and the files it names (HDRI ...).
    """
    return bridge.cache_key(bridge.clean_path(scene["file"]), scene.get("camera", ""), int(rw), int(rh),
                            int(scene.get("frame", -1)), float(scene.get("unit_scale", 1.0)),
                            relight={"samples": int(samples), "denoise": True, **part}, projector=scene.get("projector"),
                            surroundings=scene.get("surroundings"))


def relight_frames(scene, parts, rw, rh, samples, timeout=900):
    """
    Rig parts (from _rig_job, no 'name') -> ([(png path, metadata)] per part, indices rendered now, report of the
    render). Every render is kept in <cache>/relight_frames under its content key, so a timeline edit renders only
    the frames whose rig or inputs changed; the missing ones go to Blender in one job (the scene loads once).
    The director can call this per frame too: relight_frames(scene, [part], rw, rh, samples)[0][0].
    """
    root = cache_root()
    store = os.path.join(root, "relight_frames")
    keys = [relight_frame_key(scene, rw, rh, samples, p) for p in parts]
    got = [bridge.store_get(store, k) for k in keys]
    missing = [j for j, g in enumerate(got) if g is None]
    rj = None
    if missing:
        job = {"samples": int(samples), "denoise": True,
               "frames": [dict(parts[j], name=f"relit_{n:04d}.png") for n, j in enumerate(missing)]}
        folder, _, _ = bridge.export(scene["file"], root, width=rw, height=rh, blender=scene.get("blender", ""),
                                     relight=job, timeout=timeout, transient=True, projector=scene.get("projector"),
                                     **bridge.scene_opts(scene))
        try:
            with open(os.path.join(folder, "relight.json"), encoding="utf-8") as f:
                rj = json.load(f)
            per = rj["seconds"].get("per_frame") or []
            lits = rj.get("emissive_per_frame") or []
            for n, j in enumerate(missing):
                meta = {"device": rj.get("device"), "env": rj.get("env", ""), "notes": rj.get("notes", []),
                        "emissive_faces": lits[n] if n < len(lits) else rj.get("emissive_faces", 0),
                        "render_s": per[n] if n < len(per) else 0.0, "size": [int(rw), int(rh)]}
                got[j] = (bridge.store_put(store, keys[j], os.path.join(folder, f"relit_{n:04d}.png"), meta), meta)
        finally:
            shutil.rmtree(folder, ignore_errors=True)
        _after_render()
    return got, missing, rj


def relight_scene(scene, rig, emission=(), size=None, scale=None, samples=None, projector=None, look_from=None):
    """
    Cycles relight of a KUBA_SCENE. rig: a light rig (scene_view.rig_from_doc form; 'lights' as parse_lights()
    dicts in facade metres). emission: [(mask HxW float, (r, g, b), strength, label)], faces covered glow.
    size: (W, H) of the output (default the scene's matrix size); scale / samples override the rig's
    resolution_scale / samples (previews). projector: an RGB image (float 0..1, display values) cast from the
    projection camera when rig["projector"]["on"] (the matrix as light). look_from: an audience camera
    ({"location", "look_at", "lens"}, pieces.audience_view) instead of the projection camera; the surroundings of
    the scene are in the picture then. -> (rgb HxWx3, alpha HxW, report text).
    The render is kept in the frame store, so a sequence frame with the same rig and size reuses it.
    """
    t0 = time.perf_counter()
    s, pt, nrm, ground, fr = load_scene_cached(scene)
    info = s["info"]
    notes = []
    part, world, environment, env = _rig_job(s, fr, rig, emission, projector, cache_root(), notes)
    if look_from:
        part["look_from"] = look_from
        notes.append("seen from the audience" + (", with surroundings" if scene.get("surroundings") else ""))
    samples = int(samples or rig.get("samples", 64))
    k = float(scale if scale is not None else rig.get("resolution_scale", 1.0))
    if look_from and look_from.get("size") and not size:      # a viewpoint with its own picture size (scene preview)
        W, H = (int(v) for v in look_from["size"])
        rw, rh = max(16, round(W * k)), max(16, round(H * k))
    else:
        W, H, rw, rh = _render_size(info, size, k)
    got, missing, _ = relight_frames(scene, [part], rw, rh, samples)
    fp, meta = got[0]
    cached = not missing
    rgb, alpha = _read_relit(fp, W, H, rw, rh, part["background"])
    report = "\n".join([
        f"{W}x{H} (rendered {rw}x{rh}), {len(world)} lights, environment {environment}"
        f"{' x ' + format(float(rig.get('env_strength', 0.3)), 'g') if env else ''}, {samples} samples on {meta.get('device')}",
        f"emissive faces: {meta.get('emissive_faces', 0)} from {len(part['emit'])} glow source(s)",
        f"render {'cached' if cached else format(meta.get('render_s', 0.0), '.1f') + ' s'}, total {time.perf_counter() - t0:.1f} s",
        *notes, *meta.get("notes", [])])
    log.info("[KUBA scene3d] relight: %s", report.replace("\n", "\n    "))
    return rgb, alpha, report


def relight_sequence(scene, frames, size=None, scale=None, samples=None):
    """
    frames: [(rig, emission, projector)] per output frame -> ([reader() -> (rgb, alpha)] per frame, report). Frames with the
    same Blender job are rendered once; frames rendered before (any earlier sequence or still at this size) come from
    the frame store; the rest in one Blender session (the scene loads once). scale None = the first rig's
    resolution_scale (smaller light renders, scaled up), a number overrides it.
    """
    t0 = time.perf_counter()
    s, pt, nrm, ground, fr = load_scene_cached(scene)
    info = s["info"]
    root = cache_root()
    notes, parts, index = [], [], []
    seen = {}
    for rig, emission, projector in frames:
        part = _rig_job(s, fr, rig, emission, projector, root, notes)[0]
        key = json.dumps(part, sort_keys=True)
        if key not in seen:
            seen[key] = len(parts)
            parts.append(part)
        index.append(seen[key])
    first = frames[0][0]
    samples = int(samples or first.get("samples", 64))
    W, H, rw, rh = _render_size(info, size, float(scale if scale is not None else first.get("resolution_scale", 1.0)))
    got, missing, rj = relight_frames(scene, parts, rw, rh, samples, timeout=6 * 3600)
    import functools

    @functools.lru_cache(maxsize=8)                       # frames load when used (long sequences stay out of RAM);
    def reader(j):                                        # thread safe for the parallel compositing
        return _read_relit(got[j][0], W, H, rw, rh, parts[j]["background"])

    distinct = [(lambda j=j: reader(j)) for j in range(len(parts))]
    device = (rj or {}).get("device") or got[0][1].get("device")
    if rj is None:
        timing = "cached"
    else:
        per = rj["seconds"].get("per_frame") or [0]
        timing = (f"{rj['seconds']['render']:.1f} s ({sum(per) / max(1, len(per)):.1f} s / render)"
                  + (f", {len(parts) - len(missing)} from the cache" if len(missing) < len(parts) else ""))
    report = (f"{len(frames)} frames, {len(parts)} distinct light render(s) at {rw}x{rh}, {samples} samples on {device}, "
              + timing + f", total {time.perf_counter() - t0:.1f} s" + ("; " + "; ".join(dict.fromkeys(notes)) if notes else ""))
    log.info("[KUBA scene3d] relight sequence: %s", report)
    return [distinct[i] for i in index], report


def prestart_relight(scene):
    """
    Boot the Blender worker and load this scene for relight jobs in the background (Blender starts while ComfyUI
    does its numpy work; the next relight job finds the scene loaded). Never raises; -> the thread or None.
    """
    try:
        with open(os.path.join(scene["folder"], "scene.json"), encoding="utf-8") as f:
            info = json.load(f)
        job = bridge.job_dict(scene["file"], "", scene.get("camera", ""), info["width"], info["height"],
                              scene.get("frame", -1), scene.get("unit_scale", 1.0), relight={"warm": True},
                              projector=scene.get("projector"), surroundings=scene.get("surroundings"))
        return bridge.worker_prestart(job, os.path.join(cache_root(), "warm"), blender=scene.get("blender", ""))
    except Exception as e:  # noqa: BLE001
        log.info("[KUBA scene3d] Blender pre-start skipped: %s", e)
        return None


NODE_CLASS_MAPPINGS = {
    "KUBA_SceneRender": KUBA_SceneRender,
    "KUBA_SceneMeasure": KUBA_SceneMeasure,
    "KUBA_ScenePreview": KUBA_ScenePreview,
    "KUBA_SceneWalkthrough": KUBA_SceneWalkthrough,
    "KUBA_SceneRelight": KUBA_SceneRelight,
    "KUBA_Projector": KUBA_Projector,
    "KUBA_ProjectorBlend": KUBA_ProjectorBlend,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "KUBA_SceneRender": "kubakub scene render",
    "KUBA_SceneMeasure": "kubakub scene measure",
    "KUBA_ScenePreview": "kubakub scene preview",
    "KUBA_SceneWalkthrough": "kubakub scene walkthrough",
    "KUBA_SceneRelight": "kubakub scene relight",
    "KUBA_Projector": "kubakub projector",
    "KUBA_ProjectorBlend": "kubakub projector blend",
}
