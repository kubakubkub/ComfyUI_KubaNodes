"""
kubakub live for Houdini: link this Houdini to kubakub scene render in ComfyUI (source = live houdini).

Start it with the 'comfy link' tool on the kubakub shelf (installed by tools/houdini/install_houdini_link.py). Then
press refresh on the scene render node in ComfyUI: this Houdini writes what is displayed in /obj as Alembic (your
.hip is not saved) and sends your viewport view (or the camera you look through) as the camera. Nothing changed
since the last refresh = the same Alembic again (ComfyUI's cache).

Talks through files in one folder on this computer (live_folder in kubakub.ini, default ~/kubakub_live; here the
KUBAKUB_LIVE_FOLDER variable of the package, empty = the default); no network. The protocol is described in
kubakub/scene3d/live.py of the ComfyUI_KubaNodes pack.
"""

import glob
import json
import os
import threading
import time

import hou

APP = "houdini"
KEEP = 3                    # snapshots kept in the folder
TICK = 0.5                  # seconds between looks at the request
ALIVE_EVERY = 2.0
LONG_SIDE = 1920            # frame size for a viewport view (the node's width / height win)

_state = {"on": False, "sig": None, "file": None, "last_id": None, "alive_t": 0.0, "tick_t": 0.0}


def live_folder():
    raw = os.environ.get("KUBAKUB_LIVE_FOLDER", "").strip().strip('"') or "~/kubakub_live"
    return os.path.abspath(os.path.expandvars(os.path.expanduser(raw)))


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


def _status(msg):
    try:
        hou.ui.setStatusMessage("kubakub live: " + msg)
    except Exception:  # noqa: BLE001  (no UI: hython)
        print("kubakub live:", msg)


def displayed_objects():
    return [n for n in hou.node("/obj").allSubChildren()
            if isinstance(n, hou.ObjNode) and n.type().name() not in ("cam", "camera") and n.isObjectDisplayed()
            and getattr(n, "displayNode", lambda: None)() is not None]


def cameras():
    return [n for n in hou.node("/obj").allSubChildren() if isinstance(n, hou.ObjNode) and n.type().name() == "cam"]


def viewport():
    """The current viewport of the first scene viewer, or None (no UI)."""
    try:
        viewer = hou.ui.paneTabOfType(hou.paneTabType.SceneViewer)
    except Exception:  # noqa: BLE001
        return None
    return viewer.curViewport() if viewer is not None else None


def _rows(m):
    """hou.Matrix4 (row vectors, translation in the last row) -> rows of the usual column-vector matrix."""
    return [list(r) for r in m.transposed().asTupleOfTuples()]


def scene_extent(objs):
    bb = hou.BoundingBox()
    for ob in objs:
        try:
            b = ob.displayNode().geometry().boundingBox()
        except Exception:  # noqa: BLE001
            continue
        for c in ((b.minvec()[0], b.minvec()[1], b.minvec()[2]), (b.maxvec()[0], b.maxvec()[1], b.maxvec()[2])):
            bb.enlargeToContain(hou.Vector3(c) * ob.worldTransform())
    return bb.sizevec().length() if bb.isValid() else 100.0


def camera_view(cam):
    """A camera object -> (view dict for live.py, frame size)."""
    resx, resy = cam.evalParm("resx"), cam.evalParm("resy")
    focal, aperture = cam.evalParm("focal"), cam.evalParm("aperture")
    tan_h = aperture / (2.0 * focal)
    ortho = cam.evalParm("projection") == 1
    ow = cam.evalParm("orthowidth")
    view = {"matrix": _rows(cam.worldTransform()), "up": "y", "type": "ORTHO" if ortho else "PERSP",
            "tan_h": tan_h, "tan_v": tan_h * resy / resx, "ortho_w": ow, "ortho_h": ow * resy / resx,
            "shift": [cam.evalParm("winx"), cam.evalParm("winy")], "name": cam.name()}
    return view, [int(resx), int(resy)]


def viewport_view(vp, objs):
    """The viewport's own camera -> (view dict, frame size: the viewport's shape, LONG_SIDE across)."""
    dc = vp.defaultCamera()
    w, h = vp.size()[-2:]
    zoom = dc.windowSize()[0] or 1.0
    tan_h = dc.aperture() * zoom / (2.0 * dc.focalLength())
    m = vp.cameraToModelTransform()
    persp = dc.isPerspective()
    ow = dc.orthoWidth() * zoom
    if not persp:                                     # a camera does not see behind itself: step back
        m = hou.hmath.buildTranslate(0.0, 0.0, scene_extent(objs)) * m
    view = {"matrix": _rows(m), "up": "y", "type": "PERSP" if persp else "ORTHO", "tan_h": tan_h,
            "tan_v": tan_h * h / w, "ortho_w": ow, "ortho_h": ow * h / w, "shift": [0.0, 0.0], "name": "viewport"}
    if w >= h:
        size = [LONG_SIDE, max(8, int(round(LONG_SIDE * h / w / 8.0)) * 8)]
    else:
        size = [max(8, int(round(LONG_SIDE * w / h / 8.0)) * 8), LONG_SIDE]
    return view, size


def _signature(objs):
    """What the Alembic holds: a changed view or camera needs no new export, only a new answer."""
    geo = tuple((n.path(), n.worldTransform().asTuple(), n.displayNode().path(), n.displayNode().cookCount(),
                 n.displayNode().needsToCook())          # an edit the viewport has not cooked yet is a change too
                for n in objs)
    return (hou.hipFile.path(), hou.frame(), geo)


def write_alembic(path):
    """What /obj displays at the current frame -> path (a temporary Alembic ROP in /out, removed again)."""
    out = hou.node("/out") or hou.node("/").createNode("ropnet", "out")
    with hou.undos.disabler():
        rop = out.createNode("alembic", "kubakub_live_tmp")
        try:
            rop.parm("filename").set(path.replace("\\", "/"))
            rop.parm("trange").set(0)                 # the current frame
            rop.parm("root").set("/obj")
            rop.parm("objects").set("*")
            rop.parm("save_hidden").set(0)            # only what is displayed
            rop.parm("displaysop").set(1)             # the display SOP (what the viewport shows), not the render one
            rop.render()
        finally:
            rop.destroy()
    if not os.path.isfile(path):
        raise RuntimeError("the Alembic export wrote nothing (is anything displayed in /obj?)")


def send(rid, want=""):
    """Answer request `rid`, seen through camera `want` ("" = what you look through): a new Alembic when the
    scene changed, else the last one again (with the view as it is now)."""
    root = live_folder()
    os.makedirs(root, exist_ok=True)
    objs = displayed_objects()
    vp = viewport()
    want = (want or "").strip()
    picked = next((c for c in cameras() if want in (c.name(), c.path())), None) if want else None
    cam = picked or (vp.camera() if vp is not None else None)
    if cam is not None:
        view, size = camera_view(cam)
        how = "camera"
    elif vp is not None:
        view, size = viewport_view(vp, objs)
        how = "viewport"
    else:                                             # no UI: the first camera, else the node frames the model
        cams = cameras()
        view, size = camera_view(cams[0]) if cams else (None, [0, 0])
        how = "camera"
    seen = {"how": how, "size": size, "view": view, "missing": want if want and picked is None else ""}
    sig = _signature(objs)
    last = _state["file"]
    if sig == _state["sig"] and last and os.path.isfile(last["file"]):
        _state["file"] = dict(last, **seen)
        _write_json(os.path.join(root, f"live_{APP}.json"),
                    dict(last, **seen, file=os.path.basename(last["file"]), id=rid, time=time.time()))
        _status(f"sent (scene unchanged) {time.strftime('%H:%M:%S')} ({how})")
        return
    t0 = time.time()
    snap = os.path.join(root, f"houdini_snap_{time.strftime('%Y%m%d_%H%M%S')}_{int(t0 * 1000) % 1000:03d}.abc")
    write_alembic(snap)
    ans = {"app": APP, "version": hou.applicationVersionString(), "file": os.path.basename(snap), "camera": "",
           "source": hou.hipFile.basename(), **seen, "save_s": round(time.time() - t0, 2)}
    _write_json(os.path.join(root, f"live_{APP}.json"), dict(ans, id=rid, time=time.time()))
    _state.update(sig=_signature(objs), file=dict(ans, file=snap))   # after the export: it may cook
    _status(f"sent {time.strftime('%H:%M:%S')} ({how}, {len(objs)} objects)")
    for old in sorted(glob.glob(os.path.join(root, "houdini_snap_*.abc")))[:-KEEP]:
        try:
            os.remove(old)
        except OSError:
            pass                                      # ComfyUI may still be reading it: next time


def _tick():
    now = time.time()
    if now - _state["tick_t"] < TICK:                 # the event loop calls this very often
        return
    _state["tick_t"] = now
    root = live_folder()
    try:
        if now - _state["alive_t"] > ALIVE_EVERY:
            os.makedirs(root, exist_ok=True)
            _write_json(os.path.join(root, f"{APP}.alive"),
                        {"version": hou.applicationVersionString(), "pid": os.getpid(),
                         "cameras": [c.name() for c in cameras()]})
            _state["alive_t"] = now
        req = _read_json(os.path.join(root, f"request_{APP}.json"))
        if req and req.get("id") and req["id"] != _state["last_id"]:
            _state["last_id"] = req["id"]
            try:
                send(req["id"], str(req.get("camera") or ""))
            except Exception as e:  # noqa: BLE001  (ComfyUI shows it)
                last = _state["file"]                 # the last good snapshot stays usable
                _write_json(os.path.join(root, f"live_{APP}.json"),
                            dict(last or {}, file=os.path.basename(last["file"]) if last else "", id=req["id"],
                                 error=str(e), time=time.time()))
                _status(f"error: {e}")
    except Exception as e:  # noqa: BLE001  (never break Houdini's event loop)
        _status(f"error: {e}")


def start():
    if not _state["on"]:
        hou.ui.addEventLoopCallback(_tick)
        _state["on"] = True
    _status(f"linked to ComfyUI through {live_folder()}")


def stop():
    if _state["on"]:
        try:
            hou.ui.removeEventLoopCallback(_tick)
        except hou.OperationFailed:
            pass
        _state["on"] = False
    try:
        os.remove(os.path.join(live_folder(), f"{APP}.alive"))
    except OSError:
        pass
    _status("link off")


def toggle():
    stop() if _state["on"] else start()


def send_now():
    send("manual")
