"""
live.py

The live link between a running 3D app and kubakub scene render (source = live blender / live houdini). Files in one
local folder, no ports: setting live_folder (default ~/kubakub_live). Per app (blender, houdini):

    <app>.alive          the app's add-on touches it every few seconds while it is open (json: version, pid)
    request_<app>.json   ComfyUI asks for the scene as it is now: {"id": ..., "time": ...}
    live_<app>.json      the app's answer, written last: {"id": request id, "file": snapshot in this folder,
                         "camera": a camera name in the snapshot (or ""), "how": "viewport" / "camera",
                         "source": the open file's name, "version", "time", optional "size": [w, h] and
                         "view": the camera seen from the app (below); on failure also "error"}
    <snapshots>          the copies the app writes (.blend from Blender, .abc from Houdini); it keeps the last few

"view" (apps whose snapshot cannot carry the viewport as a camera): {"matrix": 4x4 camera-to-world, rows,
"up": "y" or "z", "type": "PERSP" / "ORTHO", "tan_h" / "tan_v": tangent of half the field of view across / up,
"ortho_w" / "ortho_h": the orthographic frame (m), "shift": [x, y] as fractions of the frame's width / height,
"name"}; blender_camera() turns it into Blender's camera.

The app answers with a new snapshot only when something changed since its last answer (else it repeats the last
one), so a refresh with nothing changed is a cache hit downstream. No ComfyUI imports (tests/test_live.py).
"""

from __future__ import annotations

import json
import os
import time
import uuid

ALIVE_S = 10.0          # an .alive file older than this: the app is closed or its add-on is off
APPS = {"live blender": "blender", "live houdini": "houdini"}
NAMES = {"blender": "Blender", "houdini": "Houdini"}
HOW_TO = {"blender": "install tools/blender/kubakub_live.py in Blender (Preferences > Add-ons > Install from Disk)",
          "houdini": "install the package tools/houdini (see docs/scene3d.md) and press 'comfy link' on its shelf"}
_MEMO: dict = {}        # one answer per refresh: fingerprint and execute of the same run see the same snapshot
_LAST: dict = {}        # the last good answer per folder and app (an error answer replaces live_<app>.json)


def folder(setting: str = "") -> str:
    return os.path.abspath(os.path.expandvars(os.path.expanduser(setting.strip().strip('"') or "~/kubakub_live")))


def _read(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _write(path, obj):
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f)
    os.replace(tmp, path)


def alive(root, app, now=None):
    """-> the info of the app if its add-on is running ({'version', ...}), or None."""
    p = os.path.join(root, f"{app}.alive")
    try:
        age = (time.time() if now is None else now) - os.path.getmtime(p)
    except OSError:
        return None
    return (_read(p) or {}) if age <= ALIVE_S else None


def _usable(root, ans):
    if not ans or not ans.get("file"):
        return None
    snap = os.path.abspath(os.path.join(root, os.path.basename(str(ans["file"]))))   # only a file of this folder
    return dict(ans, file=snap) if os.path.isfile(snap) else None


def current(root, app):
    """-> the app's last answer with a snapshot that still exists inside the live folder, or None."""
    ans = _usable(root, _read(os.path.join(root, f"live_{app}.json")))
    if ans:
        _LAST[(root, app)] = ans
        return ans
    return _usable(root, _LAST.get((root, app)))


def ask(root, app, wait_s=60.0, poll_s=0.1):
    """
    Ask the open app for the scene as it is now and wait for its answer.
    -> (answer dict or None, note). With the app closed: the last snapshot (or None) and a note saying so.
    """
    os.makedirs(root, exist_ok=True)
    name = NAMES.get(app, app)
    if alive(root, app) is None:
        last = current(root, app)
        return last, (f"{name} is not linked right now (closed, or the kubakub live link is off): "
                      + ("the scene as it was last sent" if last else "nothing was sent yet"))
    rid = uuid.uuid4().hex[:12]
    _write(os.path.join(root, f"request_{app}.json"), {"id": rid, "time": time.time()})
    end = time.time() + wait_s
    while time.time() < end:
        raw = _read(os.path.join(root, f"live_{app}.json")) or {}
        if raw.get("id") == rid:
            if raw.get("error"):
                raise RuntimeError(f"{name} could not send the scene: {raw['error']}")
            ans = current(root, app)
            if ans:
                return ans, ""
        time.sleep(poll_s)
    last = current(root, app)
    return last, (f"{name} did not answer in {wait_s:g} s (busy rendering?): "
                  + ("the scene as it was last sent" if last else "nothing was sent yet"))


def resolve(root, app, wait_s=60.0, memo_s=5.0):
    """ask(), remembered for a few seconds, so the checks before a run and the run itself see one snapshot."""
    hit = _MEMO.get((root, app))
    if hit and time.time() - hit[0] < memo_s:
        return hit[1], hit[2]
    ans, note = ask(root, app, wait_s)
    _MEMO[(root, app)] = (time.time(), ans, note)
    return ans, note


def forget():
    _MEMO.clear()


Y_UP_TO_Z_UP = ((1.0, 0.0, 0.0, 0.0), (0.0, 0.0, -1.0, 0.0), (0.0, 1.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))


def _matmul(a, b):
    return [[sum(a[i][k] * b[k][j] for k in range(4)) for j in range(4)] for i in range(4)]


def blender_camera(view, width, height):
    """
    The 'view' of an answer -> the live_camera of a job (blender_export.py): Blender's axes (a Y-up app's scene comes
    in rotated, as Blender's Alembic import does), a 36 mm sensor fitted to the render's long side, so that side sees
    what the app's view saw across (or up, for a portrait render).
    """
    m = [list(map(float, r)) for r in view["matrix"]]
    if str(view.get("up", "z")).lower() == "y":
        m = _matmul([list(r) for r in Y_UP_TO_Z_UP], m)
    wide = width >= height
    sx, sy = (list(view.get("shift") or [0.0, 0.0]) + [0.0, 0.0])[:2]
    cam = {"matrix": m, "name": view.get("name") or "live view", "sensor_fit": "HORIZONTAL" if wide else "VERTICAL",
           "shift_x": float(sx) if wide else float(sx) * width / max(height, 1),
           "shift_y": float(sy) * height / max(width, 1) if wide else float(sy)}
    if view.get("type") == "ORTHO":
        cam.update(type="ORTHO", ortho_scale=float(view["ortho_w"] if wide else view["ortho_h"]))
    else:
        cam.update(type="PERSP", lens=18.0 / float(view["tan_h"] if wide else view["tan_v"]))
    return cam
