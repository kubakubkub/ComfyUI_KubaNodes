"""
kubakub live: link this Blender to kubakub scene render in ComfyUI (source = live blender).

Install: Blender > Edit > Preferences > Add-ons > Install from Disk > this file, tick it. Then press refresh on the
scene render node in ComfyUI: this Blender saves a copy of the scene as it is now (your file is not touched) and
your viewport view as the camera. Nothing changed since the last refresh = the same copy again (ComfyUI's cache).

Talks through files in one folder on this computer (the same folder as live_folder in kubakub.ini, default
~/kubakub_live); no network. The protocol is described in kubakub/scene3d/live.py of the ComfyUI_KubaNodes pack.
"""

bl_info = {
    "name": "kubakub live",
    "author": "kubakub",
    "version": (1, 0, 0),
    "blender": (4, 2, 0),
    "location": "3D View > Sidebar > kubakub",
    "description": "Sends the open scene and the viewport view to kubakub scene render in ComfyUI",
    "category": "Import-Export",
}

import glob
import json
import os
import threading
import time

import bpy
from mathutils import Matrix, Vector

VIEW_CAM = "kubakub_viewport"
KEEP = 3                    # snapshots kept in the folder
TICK = 0.5                  # seconds between looks at request.json
ALIVE_EVERY = 2.0

_state = {"dirty": True, "sig": None, "file": None, "last_id": None, "alive_t": 0.0, "status": "waiting for ComfyUI"}


def _prefs():
    try:
        return bpy.context.preferences.addons[__name__].preferences
    except (KeyError, AttributeError):
        return None


def live_folder():
    p = _prefs()
    raw = (p.folder if p else "") or "~/kubakub_live"
    return os.path.abspath(os.path.expandvars(os.path.expanduser(raw.strip().strip('"'))))


def _write_json(path, obj):
    # on Windows a rename onto a file the other side is reading right now is refused: try again for a moment
    tmp = f"{path}.{os.getpid()}.{threading.get_ident()}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f)
    for i in range(20):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if i == 19:
                os.remove(tmp)
                raise
            time.sleep(0.01)


def _read_json(path):
    # a file being replaced right now can be refused or (on some disks) read half: try again for a moment
    for i in range(5):
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            return None
        except (OSError, ValueError):
            if i < 4:
                time.sleep(0.01)
    return None


def main_view():
    """(window, area, space, region_3d) of the biggest 3D viewport, or None (no window: Blender in background)."""
    best = None
    wm = bpy.context.window_manager
    for win in (wm.windows if wm else ()):
        for area in win.screen.areas:
            if area.type == "VIEW_3D":
                size = area.width * area.height
                if best is None or size > best[0]:
                    sp = area.spaces.active
                    best = (size, win, area, sp, sp.region_3d)
    return best[1:] if best else None


def render_size(scene):
    r = scene.render
    return (r.resolution_x * r.resolution_percentage // 100, r.resolution_y * r.resolution_percentage // 100)


def scene_extent(scene):
    """Diagonal of everything visible (m), for placing an orthographic camera in front of it all."""
    lo, hi = [1e18] * 3, [-1e18] * 3
    for ob in scene.objects:
        if ob.type in {"MESH", "CURVE", "SURFACE", "META", "FONT"} and ob.visible_get():
            for c in ob.bound_box:
                w = ob.matrix_world @ Vector(c)
                lo = [min(a, b) for a, b in zip(lo, w)]
                hi = [max(a, b) for a, b in zip(hi, w)]
    if lo[0] > hi[0]:
        return 100.0
    return sum((b - a) ** 2 for a, b in zip(lo, hi)) ** 0.5


def viewport_camera(cam, view_matrix, window_matrix, perspective, size, back=0.0):
    """
    Set camera data `cam` to see what the viewport sees -> its world matrix. The field of view comes from the
    viewport's projection (window_matrix), along the render's long side; an orthographic view is moved `back`
    along its view axis (Blender's viewport also sees behind its pivot, a camera does not).
    """
    W, H = size
    wm = window_matrix
    cam.sensor_width = cam.sensor_height = 36.0
    cam.sensor_fit = "HORIZONTAL" if W >= H else "VERTICAL"
    cam.shift_x = cam.shift_y = 0.0
    if perspective:
        cam.type = "PERSP"
        cam.lens = 18.0 * (wm[0][0] if W >= H else wm[1][1])
    else:
        cam.type = "ORTHO"
        cam.ortho_scale = 2.0 / (wm[0][0] if W >= H else wm[1][1])
    cam.clip_start, cam.clip_end = 0.01, 100000.0
    m = view_matrix.inverted()
    if not perspective and back:
        m = m @ Matrix.Translation((0.0, 0.0, back))
    return m


def _signature(scene, view):
    v = None
    if view:
        rv3d = view[3]
        v = (rv3d.view_perspective, tuple(tuple(r) for r in rv3d.view_matrix),
             tuple(tuple(r) for r in rv3d.window_matrix))
    return (bpy.data.filepath, scene.name, scene.frame_current, render_size(scene),
            scene.camera.name if scene.camera else "", v)


def cameras(scene):
    return [o for o in scene.objects if o.type == "CAMERA" and not o.name.startswith(VIEW_CAM)]


def send(rid, want=""):
    """Answer request `rid`, seen through camera `want` ("" = what you look through): a new snapshot when
    something changed, else the last one again."""
    root = live_folder()
    os.makedirs(root, exist_ok=True)
    scene = bpy.context.scene
    want = (want or "").strip()
    picked = next((o for o in cameras(scene) if o.name == want), None) if want else None
    missing = want if want and picked is None else ""
    view = main_view()
    sig = _signature(scene, None if picked else view) + (picked.name if picked else "",)
    last = _state["file"]
    if not _state["dirty"] and sig == _state["sig"] and last and os.path.isfile(last["file"]):
        _write_json(os.path.join(root, "live_blender.json"),
                    dict(last, file=os.path.basename(last["file"]), missing=missing, id=rid, time=time.time()))
        _state["status"] = f"sent (unchanged) {time.strftime('%H:%M:%S')}"
        return
    t0 = time.time()
    snap = os.path.join(root, f"blender_snap_{time.strftime('%Y%m%d_%H%M%S')}_{int(t0 * 1000) % 1000:03d}.blend")
    looking_through = view is not None and view[3].view_perspective == "CAMERA" and scene.camera is not None
    ob = data = None
    if picked is not None:
        camera, how = picked.name, "camera"
    elif looking_through:
        camera, how = scene.camera.name, "camera"
    elif view is not None:
        data = bpy.data.cameras.new(VIEW_CAM)
        ob = bpy.data.objects.new(VIEW_CAM, data)
        rv3d = view[3]
        ob.matrix_world = viewport_camera(data, rv3d.view_matrix, rv3d.window_matrix, rv3d.is_perspective,
                                          render_size(scene), back=rv3d.view_distance + scene_extent(scene))
        scene.collection.objects.link(ob)
        camera, how = VIEW_CAM, "viewport"
    else:
        camera, how = (scene.camera.name if scene.camera else ""), "camera"
    try:
        ctx = {"window": view[0]} if view else {}
        with bpy.context.temp_override(**ctx):
            bpy.ops.wm.save_as_mainfile(filepath=snap, copy=True, check_existing=False, compress=False,
                                        relative_remap=False)
    finally:
        if ob is not None:
            bpy.data.objects.remove(ob, do_unlink=True)
            bpy.data.cameras.remove(data)
    ans = {"app": "blender", "version": bpy.app.version_string, "file": os.path.basename(snap), "camera": camera,
           "how": how, "missing": missing, "source": os.path.basename(bpy.data.filepath) or "unsaved scene",
           "scene": scene.name,
           "save_s": round(time.time() - t0, 2)}
    _write_json(os.path.join(root, "live_blender.json"), dict(ans, id=rid, time=time.time()))
    _state.update(dirty=False, sig=sig, file=dict(ans, file=snap))
    _state["status"] = f"sent {time.strftime('%H:%M:%S')} ({how})"
    for old in sorted(glob.glob(os.path.join(root, "blender_snap_*.blend")))[:-KEEP]:
        try:
            os.remove(old)
        except OSError:
            pass                                  # ComfyUI may still be reading it: next time


def _tick():
    p = _prefs()
    if p is not None and not p.answer:
        return TICK
    try:
        root = live_folder()
        now = time.time()
        if now - _state["alive_t"] > ALIVE_EVERY:
            os.makedirs(root, exist_ok=True)
            _write_json(os.path.join(root, "blender.alive"), {"version": bpy.app.version_string, "pid": os.getpid(),
                                                                     "cameras": [o.name for o in cameras(bpy.context.scene)]})
            _state["alive_t"] = now
        req = _read_json(os.path.join(root, "request_blender.json"))
        if req and req.get("id") and req["id"] != _state["last_id"]:
            _state["last_id"] = req["id"]
            try:
                send(req["id"], str(req.get("camera") or ""))
            except Exception as e:  # noqa: BLE001  (ComfyUI shows it)
                last = _state["file"]               # the last good snapshot stays usable
                _write_json(os.path.join(root, "live_blender.json"),
                            dict(last or {}, file=os.path.basename(last["file"]) if last else "", id=req["id"],
                                 error=str(e), time=time.time()))
                _state["status"] = f"error: {e}"
    except Exception as e:  # noqa: BLE001  (a timer that raises is removed by Blender)
        _state["status"] = f"error: {e}"
    return TICK


def _on_update(scene, depsgraph):
    for u in depsgraph.updates:
        idb = u.id
        if isinstance(idb, (bpy.types.Scene, bpy.types.Collection)) or idb.name.startswith(VIEW_CAM):
            continue                              # selection, our own camera
        _state["dirty"] = True
        return


def _on_load(*_):
    _state["dirty"] = True


class KUBAKUB_OT_live_send(bpy.types.Operator):
    bl_idname = "kubakub.live_send"
    bl_label = "Send to ComfyUI now"
    bl_description = "Save the scene and the viewport view for kubakub scene render (the next refresh uses it)"

    def execute(self, context):
        _state["dirty"] = True
        try:
            send("manual")
        except Exception as e:  # noqa: BLE001
            self.report({"ERROR"}, str(e))
            return {"CANCELLED"}
        self.report({"INFO"}, _state["status"])
        return {"FINISHED"}


class KUBAKUB_PT_live(bpy.types.Panel):
    bl_space_type, bl_region_type, bl_category, bl_label = "VIEW_3D", "UI", "kubakub", "kubakub live"

    def draw(self, context):
        p = _prefs()
        col = self.layout.column()
        if p is not None:
            col.prop(p, "answer", text="answer ComfyUI refresh")
        col.label(text=_state["status"])
        col.label(text=f"folder: {live_folder()}")
        col.operator(KUBAKUB_OT_live_send.bl_idname, icon="EXPORT")


class KUBAKUB_AP_live(bpy.types.AddonPreferences):
    bl_idname = __name__
    folder: bpy.props.StringProperty(name="Live folder", subtype="DIR_PATH", default="~/kubakub_live",
                                     description="The same folder as live_folder in kubakub.ini (ComfyUI)")
    answer: bpy.props.BoolProperty(name="Answer ComfyUI", default=True,
                                   description="Send the scene when the scene render node in ComfyUI refreshes")

    def draw(self, context):
        self.layout.prop(self, "folder")
        self.layout.prop(self, "answer")


CLASSES = (KUBAKUB_AP_live, KUBAKUB_OT_live_send, KUBAKUB_PT_live)


def register():
    for c in CLASSES:
        bpy.utils.register_class(c)
    bpy.app.handlers.depsgraph_update_post.append(_on_update)
    bpy.app.handlers.load_post.append(_on_load)
    bpy.app.timers.register(_tick, first_interval=TICK, persistent=True)


def unregister():
    if bpy.app.timers.is_registered(_tick):
        bpy.app.timers.unregister(_tick)
    for h, f in ((bpy.app.handlers.depsgraph_update_post, _on_update), (bpy.app.handlers.load_post, _on_load)):
        if f in h:
            h.remove(f)
    try:
        os.remove(os.path.join(live_folder(), "blender.alive"))
    except OSError:
        pass
    for c in reversed(CLASSES):
        bpy.utils.unregister_class(c)
