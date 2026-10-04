"""
blender_export.py - runs INSIDE Blender (blender -b --factory-startup --python blender_export.py -- job.json).

Opens a scene file read-only (never saves it), flattens everything the renderer would show into
world-space meshes in a fresh scene, and writes to job["out"]:
    faceid.npy     uint32 (H, W): 1 + global face index seen by each pixel centre, 0 = background
    position.npy   float32 (H, W, 3): world position (m) of that surface point
    normal.npy     float16 (H, W, 3): world normal
    clay.png       Workbench clay render (studio light, cavity)
    mesh.npz       per face: normal, centre, area, object / material / collection index; polygons
    scene.json     camera (matrix_world, projection, lens ...), resolution, names, bbox, notes
Only bpy + numpy; the ComfyUI side (scene_ids.py) turns this into ID passes.
"""

import json
import math
import os
import sys
import time

import bpy
import numpy as np
from mathutils import Matrix, Vector

IMPORTERS = {
    ".fbx": lambda p: bpy.ops.import_scene.fbx(filepath=p),
    ".obj": lambda p: bpy.ops.wm.obj_import(filepath=p),
    ".abc": lambda p: bpy.ops.wm.alembic_import(filepath=p, as_background_job=False),
    ".glb": lambda p: bpy.ops.import_scene.gltf(filepath=p),
    ".gltf": lambda p: bpy.ops.import_scene.gltf(filepath=p),
    ".stl": lambda p: bpy.ops.wm.stl_import(filepath=p),
    ".ply": lambda p: bpy.ops.wm.ply_import(filepath=p),
    ".usd": lambda p: bpy.ops.wm.usd_import(filepath=p),
    ".usda": lambda p: bpy.ops.wm.usd_import(filepath=p),
    ".usdc": lambda p: bpy.ops.wm.usd_import(filepath=p),
    ".usdz": lambda p: bpy.ops.wm.usd_import(filepath=p),
}
GEOMETRY = {"MESH", "CURVE", "SURFACE", "META", "FONT", "CURVES"}
notes = []


def log(msg):
    print("[KUBA scene]", msg, flush=True)


def open_file(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".blend":
        bpy.ops.wm.open_mainfile(filepath=path, load_ui=False)
    elif ext in IMPORTERS:
        bpy.ops.wm.read_factory_settings(use_empty=True)
        IMPORTERS[ext](path)
    elif ext == ".c4d":
        raise RuntimeError("Blender cannot read .c4d; export FBX or Alembic from Cinema 4D")
    else:
        raise RuntimeError(f"unsupported file type {ext}")


def pick_camera(scene, name):
    cams = [o for o in scene.objects if o.type == "CAMERA"]
    if name:
        for o in cams:
            if o.name == name:
                return o, "by name"
        raise RuntimeError(f"camera '{name}' not found; cameras: {', '.join(o.name for o in cams) or 'none'}")
    if scene.camera is not None and scene.camera.type == "CAMERA":
        return scene.camera, "active"
    if cams:
        return cams[0], "first camera"
    return None, "none"


def flatten(scene, dg):
    """Render-visible geometry as new world-space mesh objects (modifiers applied, instances real)."""
    out = []
    for inst in dg.object_instances:
        ob = inst.object
        if ob.type not in GEOMETRY:
            continue
        orig = ob.original
        if orig.hide_render or (inst.parent is not None and inst.parent.original.hide_render):
            continue
        try:
            me = bpy.data.meshes.new_from_object(ob, preserve_all_data_layers=False, depsgraph=dg)
        except RuntimeError:
            continue
        if me is None or len(me.polygons) == 0:
            continue
        me.transform(inst.matrix_world)
        if inst.matrix_world.determinant() < 0:
            me.flip_normals()
        # flat shading and no custom normals: FBX / OBJ imports often carry smooth normals that blur the
        # clay render (window frames vanish) and would make the normal pass disagree with the faces
        if "custom_normal" in me.attributes:
            me.attributes.remove(me.attributes["custom_normal"])
        try:
            me.shade_flat()
        except AttributeError:
            me.polygons.foreach_set("use_smooth", [False] * len(me.polygons))
        col = orig.users_collection[0].name if orig.users_collection else ""
        out.append((orig.name, col, me))
    return out


def frame_share(cob, scene, pts):
    """Share of the points (world) that land inside the camera frame, in front of the camera."""
    from bpy_extras.object_utils import world_to_camera_view
    inside = 0
    for p in pts:
        v = world_to_camera_view(scene, cob, Vector(p))
        inside += (0 <= v.x <= 1) and (0 <= v.y <= 1) and v.z > 0
    return inside / max(len(pts), 1)


def size_from_name(name):
    """'RENDER_CAM_3200_2160PX' / 'cam 3840x2160' -> (3200, 2160) or None."""
    import re
    m = re.search(r"(\d{3,5})\s*[x_X\-]\s*(\d{3,5})", name or "")
    if m:
        w, h = int(m.group(1)), int(m.group(2))
        if 256 <= w <= 16384 and 256 <= h <= 16384:
            return w, h
    return None


def set_gpu():
    try:
        prefs = bpy.context.preferences.addons["cycles"].preferences
        for kind in ("OPTIX", "CUDA"):
            try:
                prefs.compute_device_type = kind
            except TypeError:
                continue
            prefs.get_devices()
            devs = [d for d in prefs.devices if d.type == kind]
            if devs:
                for d in prefs.devices:
                    d.use = d.type == kind
                return kind
    except Exception as e:  # noqa: BLE001
        notes.append(f"GPU setup failed ({e}), Cycles on CPU")
    return "CPU"


def image_to_numpy(path, channels):
    img = bpy.data.images.load(path, check_existing=False)
    w, h = img.size
    buf = np.empty(w * h * 4, np.float32)
    img.pixels.foreach_get(buf)
    bpy.data.images.remove(img)
    return buf.reshape(h, w, 4)[::-1, :, :channels]   # Blender images are bottom-up



def projector(sc, cob, P):
    """
    The matrix as light: a spot light at the projection camera whose colour per direction is the image, mapped
    onto the camera's frame (lens shift and sensor fit included via view_frame). Light shaders see their own
    direction as the Normal texture coordinate (light space, the spot shines along -Z).
    """
    frame = cob.data.view_frame(scene=sc)                       # 4 corners in camera space, z < 0
    tx = [v.x / -v.z for v in frame]
    ty = [v.y / -v.z for v in frame]
    x0, x1, y0, y1 = min(tx), max(tx), min(ty), max(ty)
    ld = bpy.data.lights.new("kuba_projector", type="SPOT")
    ld.energy = float(P.get("power", 1000.0))
    ld.color = (1.0, 1.0, 1.0)
    ld.shadow_soft_size = 0.0                                  # radius 0: Normal = direction to the shading point
    ld.spot_blend = 0.0
    reach = max(math.hypot(x, y) for x in (x0, x1) for y in (y0, y1))
    ld.spot_size = min(math.pi, 2 * math.atan(reach) * 1.02)
    if hasattr(ld, "use_soft_falloff"):
        ld.use_soft_falloff = False                             # plain inverse square, like a real projector
    ld.use_nodes = True
    nt = ld.node_tree
    em = next(n for n in nt.nodes if n.type == "EMISSION")
    tc = nt.nodes.new("ShaderNodeTexCoord")
    sp = nt.nodes.new("ShaderNodeSeparateXYZ")
    nt.links.new(tc.outputs["Normal"], sp.inputs["Vector"])

    def axis(out_name, lo, hi):                                 # (a / -z - lo) / (hi - lo)
        dv = nt.nodes.new("ShaderNodeMath"); dv.operation = "DIVIDE"
        nt.links.new(sp.outputs[out_name], dv.inputs[0]); nt.links.new(sp.outputs["Z"], dv.inputs[1])
        ma = nt.nodes.new("ShaderNodeMath"); ma.operation = "MULTIPLY_ADD"
        nt.links.new(dv.outputs[0], ma.inputs[0])
        ma.inputs[1].default_value = -1.0 / (hi - lo)
        ma.inputs[2].default_value = -lo / (hi - lo)
        return ma

    u, v = axis("X", x0, x1), axis("Y", y0, y1)
    cb = nt.nodes.new("ShaderNodeCombineXYZ")
    nt.links.new(u.outputs[0], cb.inputs["X"]); nt.links.new(v.outputs[0], cb.inputs["Y"])
    tex = nt.nodes.new("ShaderNodeTexImage")
    tex.image = bpy.data.images.load(P["image"], check_existing=True)
    tex.extension = "CLIP"
    tex.interpolation = "Linear"
    nt.links.new(cb.outputs[0], tex.inputs["Vector"])
    nt.links.new(tex.outputs["Color"], em.inputs["Color"])
    lo = bpy.data.objects.new("kuba_projector", ld)
    sc.collection.objects.link(lo)
    lo.matrix_world = cob.matrix_world.copy()


def apply_rig(sc, cob, R, placed_objects):
    """World, materials, lights and projector for one rig; called again per frame (the last frame's objects go)."""
    for o in [o for o in bpy.data.objects if o.name.startswith(("kuba_light_", "kuba_projector"))]:
        bpy.data.objects.remove(o, do_unlink=True)
    try:
        sc.view_settings.view_transform = R.get("view", "AgX")
    except TypeError:
        sc.view_settings.view_transform = "Filmic"
    sc.view_settings.exposure = float(R.get("exposure", 0.0))
    sc.render.film_transparent = R.get("background", "black") != "environment"

    # world: HDRI or darkness
    world = sc.world
    world.use_nodes = True
    nt = world.node_tree
    nt.nodes.clear()
    bg = nt.nodes.new("ShaderNodeBackground")
    wo = nt.nodes.new("ShaderNodeOutputWorld")
    nt.links.new(bg.outputs["Background"], wo.inputs["Surface"])
    env = R.get("env", "")
    if env:
        path = env if os.path.isfile(env) else os.path.join(bpy.utils.system_resource("DATAFILES", path="studiolights/world"), env + ".exr")
        if not os.path.isfile(path):
            raise RuntimeError(f"environment not found: {env}")
        tex = nt.nodes.new("ShaderNodeTexEnvironment")
        tex.image = bpy.data.images.load(path, check_existing=True)
        tc = nt.nodes.new("ShaderNodeTexCoord")
        mp = nt.nodes.new("ShaderNodeMapping")
        mp.inputs["Rotation"].default_value[2] = math.radians(float(R.get("env_rotation", 0.0)))
        nt.links.new(tc.outputs["Generated"], mp.inputs["Vector"])
        nt.links.new(mp.outputs["Vector"], tex.inputs["Vector"])
        nt.links.new(tex.outputs["Color"], bg.inputs["Color"])
        bg.inputs["Strength"].default_value = float(R.get("env_strength", 1.0))
    else:
        bg.inputs["Color"].default_value = (0, 0, 0, 1)
        bg.inputs["Strength"].default_value = 0.0

    # materials: one clay for everything, one emissive material per emission entry
    def principled(name, rgb, rough, emit_rgb=None, emit_strength=0.0):
        m = bpy.data.materials.new(name)
        m.use_nodes = True
        b = next(n for n in m.node_tree.nodes if n.type == "BSDF_PRINCIPLED")
        b.inputs["Base Color"].default_value = (*rgb, 1)
        b.inputs["Roughness"].default_value = rough
        if emit_rgb is not None:
            key = "Emission Color" if "Emission Color" in b.inputs else "Emission"
            b.inputs[key].default_value = (*emit_rgb, 1)
            b.inputs["Emission Strength"].default_value = emit_strength
        return m

    old = [m for m in bpy.data.materials if m.name.startswith(("kuba_clay", "kuba_emit"))]
    clay = principled("kuba_clay", tuple(R.get("albedo", [0.7, 0.7, 0.7])), float(R.get("roughness", 0.8)))
    emits = []
    for k, e in enumerate(R.get("emit", [])):
        faces = np.load(e["faces"]).astype(np.int64)
        emits.append((principled(f"kuba_emit_{k}", tuple(R.get("albedo", [0.7, 0.7, 0.7])), 0.6,
                                 tuple(e["color"]), float(e["strength"])), faces))
    lit = 0
    for ob, first, count in placed_objects:
        me = ob.data
        me.materials.clear()
        me.materials.append(clay)
        idx = np.zeros(count, np.int32)
        for k, (m, faces) in enumerate(emits):
            me.materials.append(m)
            local = faces[(faces >= first) & (faces < first + count)] - first
            idx[local] = k + 1
            lit += len(local)
        me.polygons.foreach_set("material_index", idx)
        me.update()
    for m in old:
        bpy.data.materials.remove(m)

    # lights (world positions computed on the ComfyUI side from facade coordinates)
    for k, L in enumerate(R.get("lights", [])):
        kind = L["type"].upper()
        ld = bpy.data.lights.new(f"kuba_light_{k}", type=kind)
        ld.color = tuple(L.get("color", [1, 1, 1]))
        ld.energy = float(L.get("power", 100.0))
        if kind == "AREA":
            ld.shape = "RECTANGLE"
            ld.size = ld.size_y = float(L.get("size", 1.0))
        elif kind in ("POINT", "SPOT"):
            ld.shadow_soft_size = float(L.get("size", 0.1))
        if kind == "SPOT":
            ld.spot_size = math.radians(float(L.get("angle", 45.0)))
            ld.spot_blend = 0.3
        lo = bpy.data.objects.new(f"kuba_light_{k}", ld)
        sc.collection.objects.link(lo)
        loc = Vector(L.get("location", (0, 0, 0)))
        lo.location = loc
        if kind == "SUN":
            d = Vector(L["direction"])
            lo.rotation_euler = d.to_track_quat("-Z", "Y").to_euler()
        elif "target" in L:
            lo.rotation_euler = (Vector(L["target"]) - loc).to_track_quat("-Z", "Y").to_euler()

    P = R.get("projector") or {}
    if P.get("image") and P.get("mode") == "paint":
        # the image as the model's colour, seen from the render camera (window coordinates = the camera frame),
        # then lit by the lamps / HDRI like the clay: layers painted on the building
        cnt = clay.node_tree
        b = next(n for n in cnt.nodes if n.type == "BSDF_PRINCIPLED")
        tc = cnt.nodes.new("ShaderNodeTexCoord")
        tex = cnt.nodes.new("ShaderNodeTexImage")
        tex.image = bpy.data.images.load(P["image"], check_existing=True)
        tex.extension = "EXTEND"
        cnt.links.new(tc.outputs["Window"], tex.inputs["Vector"])
        mul = cnt.nodes.new("ShaderNodeVectorMath"); mul.operation = "MULTIPLY"      # colour x clay brightness
        cnt.links.new(tex.outputs["Color"], mul.inputs[0])
        mul.inputs[1].default_value = tuple(R.get("albedo", [0.7, 0.7, 0.7]))
        cnt.links.new(mul.outputs["Vector"], b.inputs["Base Color"])
    elif P.get("image"):
        projector(sc, cob, P)
    return lit, env


PIECES = {}          # moving pieces of the loaded scene: key, per object (mesh, rest positions, piece per vertex)


def pieces_setup(P, placed_objects):
    """P = {"faces": npy of the piece per global face (-1 = still), ...}: per object the rest vertex positions and the
    piece of every vertex (loose parts share no vertex, so each vertex belongs to one piece)."""
    key = (P["faces"], os.path.getmtime(P["faces"]))
    if PIECES.get("key") == key:
        return PIECES["objs"]
    pieces_restore()
    labels = np.load(P["faces"])
    objs = []
    for ob, first, count in placed_objects:
        lab = labels[first:first + count]
        if not (lab >= 0).any():
            continue
        me = ob.data
        nv, npoly, nloop = len(me.vertices), len(me.polygons), len(me.loops)
        rest = np.empty(nv * 3, np.float32)
        me.vertices.foreach_get("co", rest)
        lt = np.empty(npoly, np.int32)
        me.polygons.foreach_get("loop_total", lt)
        lv = np.empty(nloop, np.int32)
        me.loops.foreach_get("vertex_index", lv)
        pv = np.full(nv, -1, np.int64)
        pv[lv] = np.repeat(lab, lt)
        objs.append((me, rest.reshape(-1, 3), pv))
    PIECES.update(key=key, objs=objs)
    return objs


def pieces_frame(P, placed_objects):
    """Moves every piece's vertices by its matrix for this frame: P["xform"] = a .npy of pieces x 4 x 4 (one file per
    distinct frame, so a frame's cache key depends only on that frame)."""
    objs = pieces_setup(P, placed_objects)
    M = np.load(P["xform"]).astype(np.float64)
    for me, rest, pv in objs:
        co = rest.astype(np.float64)
        m = pv >= 0
        Mk = M[pv[m]]
        co[m] = np.einsum("nij,nj->ni", Mk[:, :3, :3], co[m]) + Mk[:, :3, 3]
        me.vertices.foreach_set("co", co.astype(np.float32).ravel())
        me.update()


def look_camera(sc, cob, L):
    """Render from somewhere else (an audience spot) while the projector stays at the projection camera cob.
    L = {"location": [x, y, z], "look_at": [x, y, z], "lens": mm} in world metres. None = back to cob."""
    if not L:
        sc.camera = cob
        return
    ob = bpy.data.objects.get("kuba_look")
    if ob is None:
        cd = bpy.data.cameras.new("kuba_look")
        cd.sensor_width, cd.sensor_fit = 36.0, "AUTO"
        cd.clip_start, cd.clip_end = 0.05, 100000.0
        ob = bpy.data.objects.new("kuba_look", cd)
        sc.collection.objects.link(ob)
    ob.data.lens = float(L.get("lens", 24.0))
    loc, look = Vector(L["location"]), Vector(L["look_at"])
    ob.matrix_world = Matrix.Translation(loc) @ (look - loc).to_track_quat("-Z", "Y").to_matrix().to_4x4()
    sc.camera = ob


def pieces_restore():
    """The loaded scene back at rest (the worker keeps it for later relight jobs)."""
    for me, rest, pv in PIECES.get("objs") or []:
        try:
            me.vertices.foreach_set("co", rest.ravel())
            me.update()
        except ReferenceError:
            pass
    PIECES.clear()


def relight(job, sc, cob, W, H, placed_objects, out, t0):
    """
    Cycles render of the flattened model from the job's camera: clay material, HDRI / lights, emissive faces,
    projector. relight["frames"] = [{...rig..., "name": "relit_0000.png"}, ...] renders a sequence in this one
    Blender session (the scene loads once; each frame only swaps world, materials, lights and projector).
    """
    R = job["relight"]
    r = sc.render
    r.engine = "CYCLES"
    device = set_gpu()
    sc.cycles.device = "GPU" if device != "CPU" else "CPU"
    if R.get("warm"):                                         # pre-start: scene loaded and GPU set up, no render
        log(f"warm: scene loaded, {device}, {time.time() - t0:.1f}s")
        return
    sc.cycles.samples = int(R.get("samples", 64))
    sc.cycles.use_adaptive_sampling = True
    sc.cycles.use_denoising = bool(R.get("denoise", True))
    try:
        sc.cycles.denoiser = "OPENIMAGEDENOISE"
    except TypeError:
        pass
    sc.cycles.max_bounces = int(R.get("bounces", 4))
    sc.cycles.filter_width = 1.5
    sc.cycles.seed = 0                                        # the same noise every frame: no flicker from sampling
    r.use_persistent_data = True                              # frames of a sequence reuse the scene (BVH, textures)
    if hasattr(sc.cycles, "denoising_use_gpu"):
        sc.cycles.denoising_use_gpu = True                     # OIDN on the GPU (Blender 4.1+)
    sc.use_nodes = False
    r.use_compositing = False
    r.image_settings.file_format, r.image_settings.color_mode, r.image_settings.color_depth = "PNG", "RGBA", "16"
    r.image_settings.compression = 0                          # temp files: writing uncompressed saves ~0.07 s / frame
    frames = R.get("frames") or [dict(name="relit.png")]
    t1 = time.time()
    per, lits = [], []
    lit = 0
    env = ""
    try:
        lit, env = _relight_frames(sc, cob, R, frames, placed_objects, out, r, per, lits)
    finally:                                                  # the kept scene back at rest, even after an error
        pieces_restore()
        look_camera(sc, cob, None)
    info = {"file": job["file"], "width": W, "height": H, "device": device, "emissive_faces": int(lit),
            "lights": len(R.get("lights", [])), "env": env, "notes": notes, "frames": len(frames),
            "emissive_per_frame": lits,
            "seconds": {"prepare": t1 - t0, "render": time.time() - t1, "per_frame": per}}
    json.dump(info, open(os.path.join(out, "relight.json"), "w", encoding="utf-8"), indent=1)
    log(f"relit: {W}x{H}, {len(frames)} frame(s), {int(lit)} emissive faces, {time.time() - t1:.1f}s render")


def _relight_frames(sc, cob, R, frames, placed_objects, out, r, per, lits):
    lit, env = 0, ""
    for F in frames:
        rig = {k: v for k, v in R.items() if k != "frames"}
        rig.update(F)
        lit, env = apply_rig(sc, cob, rig, placed_objects)
        look_camera(sc, cob, rig.get("look_from"))            # audience view (the projector stays at cob)
        if rig.get("pieces"):                                 # moving pieces (scene3d/pieces.py): this frame's matrices
            pieces_frame(rig["pieces"], placed_objects)
        elif PIECES:
            pieces_restore()
        r.filepath = os.path.join(out, F.get("name", "relit.png"))
        tf = time.time()
        bpy.ops.render.render(write_still=True)
        per.append(time.time() - tf)
        lits.append(int(lit))
    return lit, env


def main(job=None):
    if job is None:
        job = json.load(open(sys.argv[sys.argv.index("--") + 1], encoding="utf-8"))
    out = job["out"]
    os.makedirs(out, exist_ok=True)
    t0 = time.time()
    open_file(job["file"])
    src = bpy.context.scene
    if job.get("frame", -1) >= 0:
        src.frame_set(int(job["frame"]))
    cam, how = pick_camera(src, job.get("camera", ""))
    cam_name = cam.name if cam is not None else "kuba_auto"
    is_blend = job["file"].lower().endswith(".blend")
    if job.get("width") and job.get("height"):
        W, H = int(job["width"]), int(job["height"])
    elif is_blend:
        W = int(src.render.resolution_x * src.render.resolution_percentage / 100)
        H = int(src.render.resolution_y * src.render.resolution_percentage / 100)
    else:
        named = size_from_name(cam_name)
        W, H = named or (1920, 1080)
        notes.append(f"{os.path.splitext(job['file'])[1]} carries no render size: "
                     + (f"{W}x{H} taken from the camera name" if named else "1920x1080 used, set width / height"))
    dg = bpy.context.evaluated_depsgraph_get()
    parts = flatten(src, dg)
    if not parts:
        raise RuntimeError("no renderable geometry in the file")
    unit_scale = float(job.get("unit_scale") or 1.0)
    S = Matrix.Scale(unit_scale, 4)
    if unit_scale != 1.0:
        for _, _, me in parts:
            me.transform(S)
        notes.append(f"model scaled by {unit_scale:g}")
    cam_all = [o.name for o in src.objects if o.type == "CAMERA"]
    unit = {"system": src.unit_settings.system, "scale_length": src.unit_settings.scale_length}

    # the file's own scene, emptied: only the flattened meshes (the file is never saved)
    cam_copy = (cam.data.copy(), cam.matrix_world.copy()) if cam is not None else None
    for o in list(bpy.data.objects):
        bpy.data.objects.remove(o, do_unlink=True)
    for other in list(bpy.data.scenes):
        if other != src:
            bpy.data.scenes.remove(other)
    sc = src
    for lay in sc.view_layers[1:]:
        sc.view_layers.remove(lay)
    for child in list(sc.collection.children):
        sc.collection.children.unlink(child)
    mats, cols, objs = [], [], []
    face_obj, face_mat, face_col, normals, centres, areas, loop_tot, verts_idx, verts = [], [], [], [], [], [], [], [], []
    voff = foff = 0
    placed_objects = []                       # (object, first global face index, face count)
    for oi, (name, col, me) in enumerate(parts):
        ob = bpy.data.objects.new(f"kuba_{oi}", me)
        sc.collection.objects.link(ob)
        placed_objects.append((ob, foff, len(me.polygons)))
        objs.append(name)
        if col not in cols:
            cols.append(col)
        n = len(me.polygons)
        idx = np.arange(foff + 1, foff + 1 + n, dtype=np.int64)
        attr = me.color_attributes.new("kuba_face", "FLOAT_COLOR", "FACE")
        rgba = np.zeros((n, 4), np.float32)
        rgba[:, 0] = (idx % 4096) / 4096.0
        rgba[:, 1] = (idx // 4096 % 4096) / 4096.0
        rgba[:, 2] = (idx // 16777216) / 4096.0
        rgba[:, 3] = 1.0
        attr.data.foreach_set("color", rgba.ravel())
        slot_mats = [(s.material.name if s.material else "") for s in ob.material_slots] or [""]
        mi = np.zeros(n, np.int32)
        me.polygons.foreach_get("material_index", mi)
        gl = []
        for m in slot_mats:
            if m not in mats:
                mats.append(m)
            gl.append(mats.index(m))
        face_mat.append(np.array(gl, np.int32)[np.clip(mi, 0, len(gl) - 1)])
        face_obj.append(np.full(n, oi, np.int32))
        face_col.append(np.full(n, cols.index(col), np.int32))
        a = np.zeros(n * 3, np.float32); me.polygons.foreach_get("normal", a); normals.append(a.reshape(n, 3))
        a = np.zeros(n * 3, np.float32); me.polygons.foreach_get("center", a); centres.append(a.reshape(n, 3))
        a = np.zeros(n, np.float32); me.polygons.foreach_get("area", a); areas.append(a)
        lt = np.zeros(n, np.int32); me.polygons.foreach_get("loop_total", lt); loop_tot.append(lt)
        lv = np.zeros(len(me.loops), np.int32); me.loops.foreach_get("vertex_index", lv); verts_idx.append(lv + voff)
        v = np.zeros(len(me.vertices) * 3, np.float32); me.vertices.foreach_get("co", v); verts.append(v.reshape(-1, 3))
        voff += len(me.vertices)
        foff += n
    V = np.concatenate(verts)
    np.savez_compressed(os.path.join(out, "mesh.npz"), normal=np.concatenate(normals), centre=np.concatenate(centres),
                        area=np.concatenate(areas), obj=np.concatenate(face_obj), mat=np.concatenate(face_mat),
                        col=np.concatenate(face_col), loop_total=np.concatenate(loop_tot),
                        loop_vert=np.concatenate(verts_idx), vert=V)
    lo, hi = V.min(0), V.max(0)

    # camera: the file's, or a front camera framing the bbox (3D-2 brings a smarter one)
    view = job.get("view")
    if view:
        how = "view"
        cam_name = view.get("name", "view")
        cdata = bpy.data.cameras.new("kuba_view")
        cdata.lens, cdata.sensor_width, cdata.sensor_fit = float(view.get("lens", 24.0)), 36.0, "AUTO"
        cdata.clip_start, cdata.clip_end = 0.05, 100000.0
        loc, look = Vector(view["location"]), Vector(view["look_at"])
        rot = (look - loc).to_track_quat("-Z", "Y").to_matrix().to_4x4()
        cmat = Matrix.Translation(loc) @ rot
    elif cam_copy is not None:
        cdata, cmat = cam_copy
        cmat = S @ cmat
        cdata.clip_end = max(cdata.clip_end, cdata.clip_end * unit_scale)
    else:
        how = "auto front"
        cdata = bpy.data.cameras.new("kuba_auto")
        cdata.lens, cdata.sensor_width, cdata.sensor_fit = 36.0, 36.0, "AUTO"
        size = hi - lo
        centre = (lo + hi) / 2
        tan_h = 18.0 / 36.0 if W >= H else 18.0 / 36.0 * W / H      # sensor fit AUTO: the long side
        tan_v = tan_h * H / W
        dist = 1.1 * max(size[0] / 2 / tan_h, size[2] / 2 / tan_v) + size[1] / 2
        cmat = Matrix.Translation(Vector((centre[0], centre[1] - dist, centre[2]))) @ Matrix.Rotation(math.pi / 2, 4, "X")
        notes.append(f"no camera in the file: front camera {dist:.1f} m in front of the bbox")
    # a tiny near clip (FBX cameras often come with 1e-5) wrecks Workbench's depth buffer: recessed details
    # vanish behind the wall in the clay render. Cycles is unaffected, the clay is not.
    far = max(float(np.linalg.norm(c - np.array(cmat.translation))) for c in (lo, hi)) * 1.5 + 1.0
    if cdata.clip_start < 0.01:
        notes.append(f"camera near clip {cdata.clip_start:g} raised to 0.01 m for the clay render")
        cdata.clip_start = 0.01
    if cdata.clip_end < far:
        cdata.clip_end = far
    cob = bpy.data.objects.new("kuba_cam", cdata)
    cob.matrix_world = cmat
    sc.collection.objects.link(cob)
    sc.camera = cob
    r = sc.render
    r.resolution_x, r.resolution_y, r.resolution_percentage = W, H, 100
    if how != "view" and cam_copy is not None and not is_blend and (cdata.shift_x or cdata.shift_y):
        # FBX round trips can flip the lens shift; if the camera misses the model, try the flipped shift
        step = max(1, len(V) // 2000)
        pts = [tuple(v) for v in V[::step]]
        bpy.context.view_layer.update()
        base = frame_share(cob, sc, pts)
        if base < 0.05:
            best = (base, cdata.shift_x, cdata.shift_y)
            for sx, sy in ((cdata.shift_x, -cdata.shift_y), (-cdata.shift_x, cdata.shift_y),
                           (-cdata.shift_x, -cdata.shift_y)):
                cdata.shift_x, cdata.shift_y = sx, sy
                share = frame_share(cob, sc, pts)
                if share > best[0]:
                    best = (share, sx, sy)
            cdata.shift_x, cdata.shift_y = best[1], best[2]
            if best[0] > base:
                notes.append(f"the imported camera saw {base * 100:.1f} % of the model; lens shift flipped to "
                             f"{best[1]:.3f} / {best[2]:.3f} ({best[0] * 100:.0f} % in frame), check it")
    r.pixel_aspect_x = r.pixel_aspect_y = 1.0
    world = bpy.data.worlds.new("kuba_world")
    sc.world = world
    if job.get("views"):
        v0 = job["views"][0]
        how, cam_name = "views", "walkthrough"
        cdata = bpy.data.cameras.new("kuba_views") if cdata is None else cdata
        cdata.sensor_width, cdata.sensor_fit, cdata.shift_x, cdata.shift_y = 36.0, "AUTO", 0.0, 0.0
        cdata.clip_start, cdata.clip_end = 0.05, 100000.0
        cob.data = cdata
    proj = cob.calc_matrix_camera(bpy.context.evaluated_depsgraph_get(), x=W, y=H, scale_x=1.0, scale_y=1.0)

    t_load = time.time() - t0
    if job.get("relight"):
        LOADED.clear()                                # a serving Blender keeps this scene for the next relight job
        LOADED.update(key=_scene_key(job), sc=sc, cob=cob, placed=placed_objects, notes=list(notes))
        relight(job, sc, cob, W, H, placed_objects, out, t0)
        return

    # renders: clay (Workbench), then face id + position + normal in one Cycles sample (no AA, pixel centres)
    device = set_gpu()
    mat = bpy.data.materials.new("kuba_faceid")
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    at = nt.nodes.new("ShaderNodeAttribute"); at.attribute_type = "GEOMETRY"; at.attribute_name = "kuba_face"
    em = nt.nodes.new("ShaderNodeEmission")
    mo = nt.nodes.new("ShaderNodeOutputMaterial")
    nt.links.new(at.outputs["Color"], em.inputs["Color"])
    nt.links.new(em.outputs["Emission"], mo.inputs["Surface"])
    vl = sc.view_layers[0]
    vl.use_pass_position = True
    vl.use_pass_normal = True
    sh = sc.display.shading
    sh.light, sh.color_type, sh.single_color = "STUDIO", "SINGLE", (0.8, 0.8, 0.8)
    sh.show_cavity, sh.cavity_type = True, "BOTH"
    sh.show_shadows = False
    sc.display.render_aa = "8"
    world.color = (0.05, 0.05, 0.05)
    sc.view_settings.view_transform = "Standard"
    sc.cycles.samples = 1
    sc.cycles.use_adaptive_sampling = False
    sc.cycles.use_denoising = False
    sc.cycles.max_bounces = 0
    sc.cycles.filter_width = 0.01
    sc.cycles.pixel_filter_type = "BOX"
    sc.cycles.device = "GPU" if device != "CPU" else "CPU"
    r.use_compositing = True                 # a .blend may have compositing / the sequencer switched
    r.use_sequencer = False                  # differently; the File Output node needs the compositor
    sc.use_nodes = True
    tree = sc.node_tree
    tree.nodes.clear()
    rl = tree.nodes.new("CompositorNodeRLayers"); rl.scene = sc
    fo = tree.nodes.new("CompositorNodeOutputFile")
    fo.format.file_format, fo.format.color_depth, fo.format.color_mode = "OPEN_EXR", "32", "RGB"
    fo.format.exr_codec = "ZIP"
    fo.file_slots.clear()
    for slot, sock in (("pass_faceid_", "Image"), ("pass_position_", "Position"), ("pass_normal_", "Normal")):
        fo.file_slots.new(slot)
        tree.links.new(rl.outputs[sock], fo.inputs[slot])
    timing = {"clay": 0.0, "ids": 0.0}

    def render_to(dst):
        os.makedirs(dst, exist_ok=True)
        t1 = time.time()
        r.engine = "BLENDER_WORKBENCH"
        vl.material_override = None
        sc.use_nodes = False
        r.film_transparent = False
        r.image_settings.file_format, r.image_settings.color_mode, r.image_settings.color_depth = "PNG", "RGB", "8"
        r.filepath = os.path.join(dst, "clay.png")
        bpy.ops.render.render(write_still=True)
        timing["clay"] += time.time() - t1
        t1 = time.time()
        r.engine = "CYCLES"
        vl.material_override = mat
        sc.use_nodes = True
        r.film_transparent = True
        fo.base_path = dst
        r.filepath = os.path.join(dst, "_cycles_unused.png")
        bpy.ops.render.render(write_still=False)
        timing["ids"] += time.time() - t1

        def slot_file(prefix):
            c = sorted(f for f in os.listdir(dst) if f.startswith(prefix) and f.endswith(".exr"))
            if not c:
                raise RuntimeError(f"Cycles pass {prefix} was not written")
            return os.path.join(dst, c[-1])

        fid = image_to_numpy(slot_file("pass_faceid_"), 3).astype(np.float64)
        q = np.rint(fid * 4096.0).astype(np.int64)
        faceid = (q[..., 0] + 4096 * q[..., 1] + 16777216 * q[..., 2]).astype(np.uint32)
        faceid[faceid > foff] = 0
        np.save(os.path.join(dst, "faceid.npy"), np.ascontiguousarray(faceid))
        np.save(os.path.join(dst, "position.npy"), np.ascontiguousarray(image_to_numpy(slot_file("pass_position_"), 3)))
        np.save(os.path.join(dst, "normal.npy"),
                np.ascontiguousarray(image_to_numpy(slot_file("pass_normal_"), 3)).astype(np.float16))
        for f in os.listdir(dst):
            if f.startswith("pass_") and f.endswith(".exr"):
                os.remove(os.path.join(dst, f))

    views = job.get("views") or []
    if views:
        # walkthrough: many audience cameras, one folder each (v_0000 ...), same flattened scene
        for k, v in enumerate(views):
            loc, look = Vector(v["location"]), Vector(v["look_at"])
            cob.matrix_world = Matrix.Translation(loc) @ (look - loc).to_track_quat("-Z", "Y").to_matrix().to_4x4()
            cdata.lens = float(v.get("lens", cdata.lens))
            render_to(os.path.join(out, f"v_{k:04d}"))
        cmat = cob.matrix_world.copy()
        proj = cob.calc_matrix_camera(bpy.context.evaluated_depsgraph_get(), x=W, y=H, scale_x=1.0, scale_y=1.0)
    else:
        render_to(out)
    t_clay, t_cycles = timing["clay"], timing["ids"]

    c = cdata
    info = {
        "file": job["file"], "blender": bpy.app.version_string, "width": W, "height": H,
        "unit": unit, "unit_scale": unit_scale, "frame": src.frame_current,
        "camera": {"name": cam_name, "how": how, "type": c.type,
                   "lens_mm": c.lens, "sensor_width_mm": c.sensor_width, "sensor_height_mm": c.sensor_height,
                   "sensor_fit": c.sensor_fit, "shift_x": c.shift_x, "shift_y": c.shift_y,
                   "ortho_scale": c.ortho_scale, "clip_start": c.clip_start, "clip_end": c.clip_end,
                   "matrix_world": [list(row) for row in cmat], "projection": [list(row) for row in proj]},
        "cameras": cam_all, "objects": objs, "materials": mats, "collections": cols,
        "faces": int(foff), "bbox_min": lo.tolist(), "bbox_max": hi.tolist(),
        "views": len(views), "cycles_device": device, "seconds": {"load_flatten": t_load, "clay": t_clay,
                                             "ids": t_cycles, "total": time.time() - t0},
        "notes": notes,
    }
    json.dump(info, open(os.path.join(out, "scene.json"), "w", encoding="utf-8"), indent=1)
    log(f"done: {foff} faces, {len(objs)} objects, camera {cam_name} ({how}), "
        f"{W}x{H}, {time.time() - t0:.1f}s")


LOADED = {}


def _scene_key(job):
    return json.dumps([job.get("file"), job.get("camera"), job.get("frame"), job.get("unit_scale"), job.get("view")])


def serve(idle_s=600.0):
    """
    Stay open for jobs: one job-file path per line on stdin, 'KUBA_DONE' / 'KUBA_FAIL <msg>' on stdout.
    For relight jobs the loaded scene is kept while the file stays the same (only lights, world and materials
    change), so a preview skips Blender's start, the file load and the GPU setup; other jobs (views, scene
    exports) load their file but skip Blender's start. Quits after idle_s without a job (never during one),
    which also gives the GPU memory back to ComfyUI.
    """
    import threading
    import traceback
    last = [time.time()]
    busy = [False]

    def watchdog():
        while True:
            time.sleep(5)
            if not busy[0] and time.time() - last[0] > idle_s:
                print("[KUBA scene] worker idle, quitting", flush=True)
                os._exit(0)

    threading.Thread(target=watchdog, daemon=True).start()
    print("KUBA_READY", flush=True)
    for line in sys.stdin:
        path = line.strip()
        if not path:
            continue
        last[0] = time.time()
        busy[0] = True
        try:
            job = json.load(open(path, encoding="utf-8"))
            if job.get("relight") and LOADED.get("key") == _scene_key(job):
                notes[:] = list(LOADED["notes"])
                sc = LOADED["sc"]
                W, H = int(job["width"]), int(job["height"])
                sc.render.resolution_x, sc.render.resolution_y, sc.render.resolution_percentage = W, H, 100
                os.makedirs(job["out"], exist_ok=True)
                relight(job, sc, LOADED["cob"], W, H, LOADED["placed"], job["out"], time.time())
            else:
                notes.clear()
                if not job.get("relight"):
                    LOADED.clear()                    # main() replaces the scene the relight jobs kept
                main(job)
            print("KUBA_DONE", flush=True)
        except Exception as e:  # noqa: BLE001
            LOADED.clear()
            PIECES.clear()
            traceback.print_exc()
            print(f"KUBA_FAIL {e}".replace("\n", " "), flush=True)
        busy[0] = False
        last[0] = time.time()


try:
    if "--serve" in sys.argv:
        _a = sys.argv[sys.argv.index("--serve"):]
        serve(float(_a[_a.index("--idle") + 1]) if "--idle" in _a else 600.0)
    else:
        main()
except Exception as e:  # noqa: BLE001
    import traceback
    traceback.print_exc()
    print(f"[KUBA scene] ERROR: {e}", flush=True)
    sys.exit(1)
sys.exit(0)
