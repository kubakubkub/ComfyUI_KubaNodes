"""kubakub scene surroundings: the street around the building, from a place on the map or your own file (logic in
scene3d/surroundings.py; the geometry is built inside Blender by scene3d/blender_export.py)."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time

import numpy as np

from comfy_api.latest import io, ui

from ...kubakub import imio
from ...kubakub.io_types import SceneType
from ...kubakub.scene3d import bridge, scene_view as sv, surroundings as sr
from .nodes_scene3d import _after_render, _img, cache_base, cache_root, hdri_stamp, kst, load_scene_cached

log = logging.getLogger("KUBA.regions")

CATEGORY = "kubakub/3d/scene"
VIEWS = ("above", "street", "off")
SAMPLE_NOTE = ("no location given: a sample square is used. Paste 'latitude, longitude' from a map into 'location' "
               "(or the path of your own surroundings model into 'file').")


TERRAINS = ("flat", "real heights")
CHECK_VIEWS = {"above": (24.0, 0.3), "street": (18.0, 0.5)}      # view -> (lens mm, look height as a share of the model's)


def surroundings_root():
    """Where downloaded map data and the footprint files are kept: next to the scene cache, outside its size cap (a
    place is downloaded once). cache_folder = temp keeps them in ComfyUI's user directory all the same."""
    return os.path.join(cache_base(lasting=True), "surroundings")


_STREET = {}                 # the last map read: turning or moving the street does not read and sort the map again


def _street(root, place, radius_m, default_height_m, own_building, width, real_ground=False, level_m=0.0):
    """
    The buildings of a place on the map (or of the sample square when place is None), ready to be put against the
    model, written as a footprint file. -> {"buildings", "own_ring", "point": the place on the facade wall,
    "facing": compass direction of that wall or None, "path": the footprint file, "ground": the file of the real
    ground (real_ground, a place on the map) or None, "lines": report}.
    """
    real_ground = bool(real_ground and place)
    key = (root, place, float(radius_m), float(default_height_m), own_building, None if place else round(width, 2),
           real_ground, float(level_m), None if not place or sr.covered(root, *place, radius_m, terrain=real_ground) else time.time())  # not whole yet: ask again
    if _STREET.get("key") == key and os.path.isfile(_STREET["got"]["path"]):
        return _STREET["got"]
    got = {"own_ring": None, "point": (0.0, 0.0), "facing": None, "ground": None, "lines": []}
    if place is None:                                      # nothing pasted yet: the sample square
        buildings, got["facing"] = sr.sample(width)
        got["lines"].append(SAMPLE_NOTE)
    else:
        lat, lon = place
        data, fresh, late = sr.fetch(root, lat, lon, radius_m, allow_download=kst.switch("surroundings_download", False))
        buildings = sr.within(sr.buildings_from_osm(data, lat, lon), float(radius_m))
        if not buildings:
            raise ValueError(f"no buildings on the map within {radius_m:g} m of {lat:.5f}, {lon:.5f}: check "
                             "the location (latitude first) or raise radius_m")
        h = sr.fill_heights(buildings, default_height_m)
        own = sr.own_building(buildings)
        got["lines"] += [f"{len(buildings)} buildings within {radius_m:g} m of {lat:.5f}, {lon:.5f} "
                         f"({'downloaded now' if fresh else 'from this computer'}; {sr.CREDIT})",
                         f"heights: {h['height']} from the map, {h['levels']} from the level count, {h['guessed']} "
                         f"unknown and set to {h['default_m']:g} m", *([late] if late else [])]
        if own:
            got.update(point=own["point"], own_ring=buildings[own["index"]]["rings"][0], facing=own["facing_deg"])
            got["lines"].append(f"the building itself: map footprint {buildings[own['index']]['id']}, "
                                f"{own['distance_m']:.1f} m from the location; its wall there faces {own['facing_deg']:.0f} "
                                f"deg ({sr.compass(own['facing_deg'])}); " + ("left out" if own_building == "remove" else "kept"))
            if own_building == "remove":
                buildings = sr.without_own(buildings, own)
    if real_ground:
        allow = kst.switch("surroundings_download", False)
        t, fresh, late = sr.fetch_terrain(root, *place, radius_m, allow_download=allow)
        t, zero = sr.level(t, got["point"], float(level_m))           # the ground at the facade = the model's feet
        sr.drape(buildings, t, zero)
        grid = sr.ground_grid(t, zero, float(radius_m))
        got["ground"] = os.path.join(root, "ground_" + hashlib.sha1(
            b"".join(grid[k].tobytes() for k in "xyz")).hexdigest()[:16] + ".npz")
        if not os.path.isfile(got["ground"]):
            np.savez_compressed(got["ground"][:-4] + f".part-{os.getpid()}.npz", **grid)
            os.replace(got["ground"][:-4] + f".part-{os.getpid()}.npz", got["ground"])
        got["lines"] += [f"real ground: {zero:.0f} m above sea level at the facade, from {grid['z'].min():+.0f} to "
                         f"{grid['z'].max():+.0f} m around it, level within {level_m:g} m of the facade, a sample every "
                         f"{t['step']:g} m "
                         f"({'downloaded now' if fresh else 'from this computer'}; {sr.TERRAIN_CREDIT})",
                         *([late] if late else [])]
    doc = sr.street_doc(buildings)
    got["path"] = os.path.join(root, f"street_{hashlib.sha1(json.dumps(doc).encode()).hexdigest()[:16]}.json")
    if not os.path.isfile(got["path"]):
        sr.write(got["path"], buildings, {"source": "openstreetmap" if place else "sample", "credit": sr.CREDIT if place else ""})
    got["buildings"] = buildings
    _STREET.update(key=key, got=got)
    return got


def _own_file(file, file_scale):
    """The path of an own surroundings model (checked) and its report line."""
    path = bridge.clean_path(file)
    ext = os.path.splitext(path)[1].lower()
    if not os.path.isfile(path):
        raise FileNotFoundError(f"surroundings file not found: {path}")
    if ext == ".blend" or ext not in bridge.FORMATS:
        raise ValueError(f"surroundings file: {ext or 'no extension'} is not supported here; export "
                         ".glb, .fbx or .obj (supported: " + " ".join(f for f in bridge.FORMATS if f != ".blend") + ")")
    return path, (f"own file {os.path.basename(path)}, in the model's space"
                  + (f", scaled by {file_scale:g}" if file_scale != 1.0 else ""))


def _check_view(scene, sur, view, buildings, V, anchor, nh, ex, width, ground_z):
    """The clay model in its surroundings from above or from the street -> (picture, report line)."""
    up = np.array([0.0, 0.0, 1.0])
    lens, look = CHECK_VIEWS[view]
    if view == "above":
        d = max(40.0, 2.2 * width)
        loc = anchor + nh * d + up * d * 1.1
    else:                                                  # a spot on the street that is not inside a neighbour
        loc = sr.free_spot(buildings, sur["matrix"], [anchor + nh * max(12.0, k * width) + ex * x * width + up * 1.7
                                                      for k in (1.4, 1.0, 0.7, 2.0, 0.45) for x in (0.6, -0.6, 0.3, -0.3, 0.0)])
    v = {"location": [round(float(x), 3) for x in loc], "lens": lens, "name": f"surroundings_{view}",
         "look_at": [round(float(x), 3) for x in anchor + up * float(V[:, 2].max() - ground_z) * look]}
    vf, cached, _ = bridge.export(scene["file"], cache_root(), width=1280, height=720, view=v,
                                  blender=scene.get("blender", ""), **dict(bridge.scene_opts(scene), surroundings=sur))
    with open(os.path.join(vf, "scene.json"), encoding="utf-8") as f:
        faces = json.load(f).get("context_faces", 0)
    if not cached:
        _after_render()
    return (imio.imread(os.path.join(vf, "clay.png"))[..., ::-1].astype(np.float32) / 255,
            f"view '{view}': {faces} faces of surroundings{' (cached)' if cached else ''}")


class KUBA_SceneSurroundings(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="KUBA_SceneSurroundings",
            display_name="kubakub scene surroundings",
            category=CATEGORY,
            search_aliases=['city', 'street', 'map', 'openstreetmap', 'osm', 'neighbours', 'context', 'environment'],
            description=(
                "Puts the street around your building: the neighbouring buildings as plain boxes, so scene preview, "
                "scene walkthrough, pieces render (audience view) and scene relight show the place instead of a "
                "building in the void. Paste the coordinates of the facade from a map: the buildings around are "
                "taken from OpenStreetMap (downloaded once per place, then kept on this computer) and set against "
                "your model's facade. Or give your own surroundings model. The surroundings never become regions "
                "and are never painted: they are only there to look at. Put the node between scene render and the "
                "preview nodes."),
            inputs=[
                SceneType.Input("scene", tooltip="From kubakub scene render."),
                io.String.Input("location", default="",
                                placeholder="paste 'latitude, longitude' of the facade  (empty = a sample square)",
                                tooltip="The middle of your facade on a map, as 'latitude, longitude' (e.g. 48.85837, "
                                        "2.29448). Google Maps: right click on the wall, click the numbers to copy "
                                        "them. A map link with @lat,lon works too. Downloads are off until you write "
                                        "surroundings_download = on into kubakub.ini: then the node sends these "
                                        "numbers and the radius to the OpenStreetMap server (overpass-api.de) once "
                                        "and keeps the answer. Empty: a sample square, nothing is downloaded."),
                io.Float.Input("radius_m", default=250.0, min=30.0, max=1500.0, step=10.0,
                               tooltip="How far around the place buildings are taken (metres). 150-300 is a square "
                                       "and its streets; larger only for views from far away."),
                io.Combo.Input("own_building", options=["remove", "keep"], default="remove",
                               tooltip="remove: the map's footprint of the building itself is left out, your model "
                                       "stands there. keep: for a model that is only the facade, so the rest of the "
                                       "building stays behind it."),
                io.Combo.Input("view", options=list(VIEWS), default="above",
                               tooltip="The check picture: from above, from the street, or off (no Blender render)."),
                io.String.Input("file", default="", optional=True,
                                placeholder="or: the path of your own surroundings model (.glb .fbx .obj .abc .usd)",
                                tooltip="Your own surroundings instead of the map: a model in the same space as the "
                                        "building (same origin and axes), e.g. exported from Houdini, Blender or a "
                                        "city model. Nothing is downloaded then. Not .blend: export .glb, .fbx or .obj."),
                io.Boolean.Input("ground", default=True, optional=True,
                                 tooltip="A flat ground under the street, so buildings do not float and light has "
                                         "something to land on. Off when your own file brings its ground."),
                io.Float.Input("facing_deg", default=-1.0, min=-1.0, max=360.0, step=1.0, optional=True, advanced=True,
                               tooltip="Compass direction your facade faces (0 = north, 90 = east, 180 = south). -1 = "
                                       "read from the map: the wall of the footprint nearest to the location."),
                io.Float.Input("default_height_m", default=0.0, min=0.0, max=300.0, step=1.0, optional=True, advanced=True,
                               tooltip="Height for buildings the map has no height or level count for. 0 = the "
                                       "median of the ones around that have one."),
                io.Float.Input("turn_deg", default=0.0, min=-180.0, max=180.0, step=0.5, optional=True, advanced=True,
                               tooltip="Turns the surroundings around the middle of the facade (seen from above, + = "
                                       "counter-clockwise). For when the fit is a little off."),
                io.Float.Input("along_m", default=0.0, min=-500.0, max=500.0, step=0.25, optional=True, advanced=True,
                               tooltip="Moves the surroundings along the wall (+ = to the right as the audience sees it)."),
                io.Float.Input("out_m", default=0.0, min=-500.0, max=500.0, step=0.25, optional=True, advanced=True,
                               tooltip="Moves the surroundings towards the audience (+) or behind the wall (-)."),
                io.Float.Input("lift_m", default=0.0, min=-100.0, max=100.0, step=0.1, optional=True, advanced=True,
                               tooltip="Moves the surroundings up (+) or down, e.g. when the model's lowest point is "
                                       "a cellar."),
                io.Float.Input("file_scale", default=1.0, min=0.0001, max=10000.0, step=0.001, optional=True, advanced=True,
                               tooltip="Multiplies your own surroundings file into metres (0.01 for a file in cm)."),
                io.Combo.Input("terrain", options=list(TERRAINS), default="flat", optional=True,
                               tooltip="flat: everything stands on one level. real heights: the ground of the place "
                                       "(hills, slopes, river banks; one more download per place, from the open "
                                       "terrain tiles on AWS), the buildings set down on it and the audience "
                                       "cameras standing on it. It meets your model at the facade. Coarse: about "
                                       "10-30 m between real measurements, so no kerbs or steps. Needs a location."),
                io.Float.Input("level_m", default=40.0, min=0.0, max=1000.0, step=5.0, optional=True, advanced=True,
                               tooltip="terrain = real heights: the ground stays level this far around the facade "
                                       "and blends into the real heights beyond, so your model stands on even "
                                       "ground (the height data is too coarse for a square). 0 = the raw heights."),
            ],
            outputs=[
                SceneType.Output("scene", tooltip="The scene with its surroundings -> kubakub scene preview / scene "
                                                  "walkthrough / pieces render / scene relight."),
                io.Image.Output("plan", tooltip="Top view for checking the fit, the audience at the bottom: the "
                                                "surroundings (grey), your model (orange), the footprint that was "
                                                "left out (violet line), the projector (dot)."),
                io.Image.Output("view", tooltip="The clay model in its surroundings (the 'view' input)."),
                io.String.Output("report", tooltip="How many buildings, where their heights come from, which way the "
                                                   "facade faces, and warnings (e.g. the projector inside a building)."),
            ],
        )

    @classmethod
    def fingerprint_inputs(cls, file="", location="", radius_m=250.0, terrain="flat", **kwargs):
        if bridge.clean_path(file):
            return hdri_stamp(file)                           # an own file saved again under the same name is new
        try:
            place = sr.parse_location(location)
        except ValueError:
            return ""
        if place is None or sr.covered(surroundings_root(), *place, radius_m, terrain=terrain != "flat"):
            return ""
        return float("nan")                                   # not all on disk yet (a busy map server): ask again each run

    @classmethod
    def execute(cls, scene, location, radius_m, own_building, view, file="", ground=True, facing_deg=-1.0,
                default_height_m=0.0, turn_deg=0.0, along_m=0.0, out_m=0.0, lift_m=0.0, file_scale=1.0,
                terrain="flat", level_m=40.0) -> io.NodeOutput:
        t0 = time.perf_counter()
        s, pt, nrm, ground_z, _ = load_scene_cached(scene)
        V = s["mesh"]["vert"]
        anchor, nh, ex = sr.facade_anchor(V, pt, nrm, ground_z)
        width = float((V @ ex).max() - (V @ ex).min())
        nudges = dict(turn_deg=float(turn_deg), along_m=float(along_m), out_m=float(out_m), lift_m=float(lift_m))
        buildings, own_ring, real = [], None, None
        if bridge.clean_path(file):                            # your own model, already in the building's space
            path, line = _own_file(file, file_scale)
            lines = [line]
            M = sr.nudge(anchor, nh, ex, scale=float(file_scale), **nudges)
            centre = anchor
        else:
            st = _street(surroundings_root(), sr.parse_location(location), radius_m, default_height_m, own_building, width,
                         real_ground=terrain != "flat", level_m=level_m)
            buildings, own_ring, path, lines, real = st["buildings"], st["own_ring"], st["path"], list(st["lines"]), st["ground"]
            if terrain != "flat" and not real:
                lines.append("terrain = real heights needs a location on the map: the ground stays flat here")
            facing = st["facing"]
            if facing_deg >= 0:
                facing = float(facing_deg)
                lines.append(f"facade direction set by hand: {facing:.0f} deg ({sr.compass(facing)})")
            elif facing is None:
                raise ValueError("the location is more than 30 m from any building on the map, so the node cannot "
                                 "tell which way your facade faces: move the location onto the wall, or set facing_deg")
            M = sr.placement(anchor, nh, ex, facing, st["point"], **nudges)
            centre = (np.asarray(M) @ np.array([st["point"][0], st["point"][1], 0.0, 1.0]))[:3]
        if any(nudges.values()):
            lines.append(f"moved by hand: turn {turn_deg:g} deg, along {along_m:g} m, out {out_m:g} m, up {lift_m:g} m")
        sur = {"file": path, "matrix": np.round(np.asarray(M, float), 5).tolist(),
               "ground": {"centre": [round(float(v), 3) for v in centre], "radius": float(radius_m)} if ground else None}
        if real and ground:                                    # the real ground instead of the flat disc
            sur["terrain"] = {"file": real, "radius": float(radius_m)}

        cam = sv.Camera(s["info"])
        if not cam.ortho and sr.stands_inside(buildings, sur["matrix"], cam.C):    # a projector in a neighbour lights nothing
            lines.append("WARNING: the projector stands inside a neighbouring building (see the plan). Move it "
                         "(kubakub projector: distance_m / offset_m / height_m) or it lights nothing in the relight.")
        plan = sr.plan(buildings, sur["matrix"], anchor, nh, ex, V[::max(1, len(V) // 4000)], max(float(radius_m), width),
                       projector=cam.C, own_ring=own_ring)
        pic = None
        if view != "off":
            pic, line = _check_view(scene, sur, view, buildings, V, anchor, nh, ex, width, ground_z)
            lines.append(line)
        lines.append(f"{time.perf_counter() - t0:.1f} s")
        report = "\n".join(lines)
        log.info("[KUBA scene3d] surroundings: %s", report.replace("\n", "\n    "))
        shown = _img(pic if pic is not None else plan)
        return io.NodeOutput(dict(scene, surroundings=sur), _img(plan), shown, report, ui=ui.PreviewImage(shown, cls=cls))


NODE_CLASS_MAPPINGS = {"KUBA_SceneSurroundings": KUBA_SceneSurroundings}
NODE_DISPLAY_NAME_MAPPINGS = {"KUBA_SceneSurroundings": "kubakub scene surroundings"}
