"""
Model free test for the live link to an open 3D app (kubakub/scene3d/live.py, scene render source = live blender /
live houdini): fake apps answer the refresh requests through the live folder as tools/blender/kubakub_live.py and
tools/houdini/python/kubakub_live.py do.
Checks: the answer is used (snapshot + viewport camera), nothing changed = the same fingerprint, a change = a new one,
the camera picked on the node is asked from the app (its list in .alive, a missing name, an older link), Houdini's view becomes a Blender camera (Y up -> Z up, field of view on the long side, lens
shift) with its frame size, a projector wins over it, two apps open do not answer each other, the app closed = the
last snapshot with a note, nothing sent yet = the sample building, an error from the app is shown, a snapshot path
outside the live folder is never opened. Blender is replaced by stubs; nothing renders.

Run from the ComfyUI_windows_portable folder:
    python_embeded\\python.exe ComfyUI\\custom_nodes\\ComfyUI_KubaNodes\\tests\\test_live.py
"""

import json
import math
import os
import shutil
import sys
import tempfile
import threading
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PACK = os.path.dirname(HERE)
sys.path.insert(0, PACK)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(HERE))))     # ComfyUI (folder_paths, comfy_api)
import folder_paths  # noqa: E402

TMP = tempfile.mkdtemp(prefix="kuba_live_")
for name, setter in (("temp", folder_paths.set_temp_directory), ("output", folder_paths.set_output_directory),
                     ("input", folder_paths.set_input_directory), ("user", folder_paths.set_user_directory)):
    setter(os.path.join(TMP, name))
LIVE = os.path.join(TMP, "live")
os.environ["KUBA_LIVE_FOLDER"] = LIVE
os.environ["KUBA_LIVE_WAIT"] = "3"
os.environ["KUBA_BLENDER_WORKER"] = "0"
sys.path.insert(0, HERE)
import _pack  # noqa: E402,F401
import kubapack.nodes.scene3d.nodes_scene3d as ns  # noqa: E402
from kubakub.scene3d import live  # noqa: E402

failures = []


def check(name, cond, detail=""):
    print(("ok   " if cond else "FAIL ") + name + (f"  ({detail})" if not cond else ""))
    if not cond:
        failures.append(name)


# Houdini camera at (0, 5, 20) (Y up), looking down -Z, 50 mm on a 41.4214 mm aperture, 1920x1080, a small window shift
H_VIEW = {"matrix": [[1, 0, 0, 0], [0, 1, 0, 5], [0, 0, 1, 20], [0, 0, 0, 1]], "up": "y", "type": "PERSP",
          "tan_h": 41.4214 / 100.0, "tan_v": 41.4214 / 100.0 * 1080 / 1920, "ortho_w": 10.0, "ortho_h": 5.625,
          "shift": [0.1, 0.2], "name": "projcam"}


class FakeApp:
    """Answers request_<app>.json like the add-ons: a new snapshot only when `changed` was set."""

    def __init__(self, root, app, ext, extra, cams=(), old=False):
        self.root, self.app, self.ext, self.extra = root, app, ext, extra
        self.cams, self.old, self.asked = list(cams), old, None
        self.changed, self.n, self.last, self.error, self.stop = True, 0, None, "", False
        os.makedirs(root, exist_ok=True)
        self.touch()                                            # linked from the start, as an app that is open
        self.t = threading.Thread(target=self.run, daemon=True)
        self.t.start()

    def touch(self):                                            # as the add-ons: never half written
        live._write(os.path.join(self.root, f"{self.app}.alive"),
                    {"version": "1.0", **({} if self.old else {"cameras": self.cams})})

    def run(self):
        handled = None
        while not self.stop:
            self.touch()
            req = live._read(os.path.join(self.root, f"request_{self.app}.json"))
            if req and req["id"] != handled:
                handled = req["id"]
                self.asked = want = req.get("camera", "")
                out = os.path.join(self.root, f"live_{self.app}.json")
                if self.error:                                  # as the add-ons: the last file stays
                    live._write(out, {"id": handled, "error": self.error, "file": self.last or ""})
                else:
                    if self.changed or self.last is None:
                        self.n += 1
                        self.last = f"{self.app}_snap_{self.n:03d}{self.ext}"
                        with open(os.path.join(self.root, self.last), "wb") as f:
                            f.write(b"SNAP" + bytes([self.n]))
                        self.changed = False
                    seen = {} if self.old else {"missing": want if want and want not in self.cams else ""}
                    if want in self.cams and not self.old:   # as the add-ons: the picked camera
                        seen.update(camera=want, how="camera", view=None) if self.app == "blender" else                             seen.update(how="camera", view=dict(self.extra.get("view") or {}, name=want))
                    live._write(out, dict(self.extra, **seen, id=handled, app=self.app, version="1.0",
                                          file=self.last, time=time.time()))
            time.sleep(0.05)

    def close(self):
        self.stop = True
        self.t.join()
        os.remove(os.path.join(self.root, f"{self.app}.alive"))


try:
    check("live folder from the setting", ns.live_root() == os.path.abspath(LIVE), ns.live_root())
    check("live folder default ~/kubakub_live", live.folder("") == os.path.abspath(os.path.expanduser("~/kubakub_live")))
    check("sources: file, live blender, live houdini", ns.LIVE_SOURCES == ("file", "live blender", "live houdini"))

    # nothing linked, nothing sent: the sample building and a note how to link
    live.forget()
    L = ns._live_scene("", "", "live blender")
    check("nothing sent yet: the sample building", L["file"] == "" and "nothing was sent yet" in L["line"]
          and "kubakub_live.py" in L["line"], L["line"])
    L = ns._live_scene("", "", "live houdini")
    check("houdini: how to link", "install_houdini_link" in L["line"] or "tools/houdini" in L["line"], L["line"])
    check("source = file: inputs untouched", ns._live_scene("x.blend", "Cam", "file", 10, 20) ==
          {"file": "x.blend", "camera": "Cam", "width": 10, "height": 20, "live_camera": None, "line": ""})

    fb = FakeApp(LIVE, "blender", ".blend", {"camera": "kubakub_viewport", "how": "viewport", "source": "facade.blend"},
                 cams=["Camera", "Camera.001"])
    live.forget()
    L = ns._live_scene("", "", "live blender")
    check("open Blender answers: its snapshot", os.path.basename(L["file"]) == "blender_snap_001.blend"
          and os.path.isfile(L["file"]), L)
    check("empty camera = the viewport view", L["camera"] == "kubakub_viewport" and "your viewport view" in L["line"]
          and L["live_camera"] is None, L["line"])
    check("the app's cameras for the pick list", live.cameras(LIVE, "blender") == ["Camera", "Camera.001"]
          and live.cameras(LIVE, "houdini") is None)
    live.forget()
    L = ns._live_scene("", "Camera.001", "live blender")
    check("the picked camera is asked from the app", fb.asked == "Camera.001" and L["camera"] == "Camera.001"
          and "camera Camera.001" in L["line"] and "no camera" not in L["line"], L["line"])
    live.forget()
    L = ns._live_scene("", "Gone", "live blender")
    check("a camera the scene no longer has: what you look through, said", fb.asked == "Gone"
          and L["camera"] == "kubakub_viewport" and "no camera 'Gone'" in L["line"], L["line"])
    live.forget()
    L = ns._live_scene("", "", "live blender")
    check("no pick = what you look through", fb.asked == "" and L["camera"] == "kubakub_viewport", L["line"])

    kw = dict(camera="", width=0, height=0, frame=-1, passes="shelves", preview_pass="shelves", unit_scale=1.0,
              source="live blender")
    live.forget()
    fp1 = ns.KUBA_SceneRender.fingerprint_inputs(file="", **kw)
    live.forget()
    fp2 = ns.KUBA_SceneRender.fingerprint_inputs(file="", **kw)
    check("refresh with nothing changed: same fingerprint (cached)", fp1 == fp2 and fb.n == 1)
    fb.changed = True
    live.forget()
    fp3 = ns.KUBA_SceneRender.fingerprint_inputs(file="", **kw)
    check("refresh after a change: new fingerprint", fp3 != fp1 and fb.n == 2)

    # Houdini at the same time: each app answers only its own request
    fh = FakeApp(LIVE, "houdini", ".abc", {"camera": "", "how": "camera", "source": "facade.hip",
                                           "size": [1920, 1080], "view": H_VIEW})
    live.forget()
    L = ns._live_scene("", "", "live houdini")
    check("houdini answers its own request", os.path.basename(L["file"]) == "houdini_snap_001.abc" and fb.n == 2, L)
    check("houdini's frame size is used", (L["width"], L["height"]) == (1920, 1080) and "frame 1920x1080" in L["line"],
          L["line"])
    c = L["live_camera"]
    check("houdini's view becomes the camera", c is not None and c["type"] == "PERSP" and c["name"] == "projcam"
          and "camera projcam" in L["line"], L)
    m = np.array(c["matrix"])
    check("Y up -> Z up: the camera stands at (0, -20, 5)", np.allclose(m[:3, 3], [0, -20, 5]), m[:3, 3])
    check("... and looks along +Y (Blender's front)", np.allclose(m[:3, :3] @ [0, 0, -1], [0, 1, 0]), m[:3, :3])
    fov_h = 2 * math.atan(41.4214 / 100.0)
    check("same field of view across (36 mm sensor)", abs(2 * math.atan(18.0 / c["lens"]) - fov_h) < 1e-9, c["lens"])
    check("lens shift: x as is, y as a share of the width", abs(c["shift_x"] - 0.1) < 1e-9
          and abs(c["shift_y"] - 0.2 * 1080 / 1920) < 1e-9, c)
    fh.cams = ["projcam", "side_cam"]
    live.forget()
    L = ns._live_scene("", "side_cam", "live houdini")
    check("houdini: the picked camera's view, the same snapshot", fh.asked == "side_cam" and fh.n == 1
          and L["live_camera"]["name"] == "side_cam" and "camera side_cam" in L["line"], L["line"])
    fh.old = True
    live.forget()
    L = ns._live_scene("", "side_cam", "live houdini")
    check("an older link that cannot pick: said", "cannot pick a camera" in L["line"] and "restart Houdini" in L["line"],
          L["line"])
    fh.old = False
    live.forget()
    live.forget()
    L = ns._live_scene("", "", "live houdini", 1080, 1920)
    c = L["live_camera"]
    check("portrait frame on the node: the vertical field of view, node size wins",
          (L["width"], L["height"]) == (1080, 1920) and c["sensor_fit"] == "VERTICAL"
          and abs(2 * math.atan(18.0 / c["lens"]) - 2 * math.atan(H_VIEW["tan_v"])) < 1e-9, c)
    ortho = live.blender_camera(dict(H_VIEW, type="ORTHO"), 1920, 1080)
    check("orthographic: the frame width as ortho scale", ortho["type"] == "ORTHO" and ortho["ortho_scale"] == 10.0)

    # execute: the snapshot, the camera and the frame size go to the export; a projector wins over the view
    seen = {}
    W = H = 16
    sfolder = os.path.join(TMP, "scenefolder")
    os.makedirs(sfolder, exist_ok=True)
    cv2.imwrite(os.path.join(sfolder, "clay.png"), np.full((H, W, 3), 128, np.uint8))
    fid = np.ones((H, W), np.int32)
    info = {"file": "snap", "faces": 1, "objects": ["a"], "width": W, "height": H, "cameras": [],
            "blender": "stub", "camera": {"name": "live", "how": "live", "type": "PERSP",
                                           "lens_mm": 35.0, "shift_x": 0.0, "shift_y": 0.0},
            "bbox_min": [0, 0, 0], "bbox_max": [30, 1, 20], "notes": []}

    def stub_export(path, cache_root, camera="", width=0, height=0, live_camera=None, projector=None, **k):
        seen.update(path=path, camera=camera, width=width, height=height, live_camera=live_camera, projector=projector)
        return sfolder, False, ""

    def stub_build(folder, passes=(), out=None, **k):
        os.makedirs(out, exist_ok=True)
        depth = np.ones((H, W), np.float32)
        np.save(os.path.join(out, "depth_m.npy"), depth)
        summary = {"passes": {p: {"ids": 1} for p in passes}, "notes": [], "coverage": 1.0, "depth_m": [1.0, 1.0],
                   "main_plane": {"normal": [0, -1, 0]}}
        json.dump(summary, open(os.path.join(out, "ids_summary.json"), "w"))
        return summary, {}, {"info": dict(info), "faceid": fid, "normal": np.zeros((H, W, 3), np.float32)}, depth

    real_export, real_build = ns.bridge.export, ns.scene_ids.build
    ns.bridge.export, ns.scene_ids.build = stub_export, stub_build
    real_load = ns.scene_ids.load
    ns.scene_ids.load = lambda folder: {"info": dict(info), "faceid": fid, "normal": np.zeros((H, W, 3), np.float32)}
    rkw = dict(file="", width=0, height=0, frame=-1, passes="shelves", preview_pass="shelves", shelf_min_step_m=0.05,
               plane_angle_deg=3.0, plane_offset_m=0.02, layer_bands_m="-0.5 -0.05 0.05 0.5", source="live blender")
    live.forget()
    out = ns.KUBA_SceneRender.execute(**rkw)
    report = out.args[7]
    check("execute renders the snapshot", os.path.basename(seen.get("path", "")) == "blender_snap_002.blend", seen)
    check("execute uses the viewport camera", seen.get("camera") == "kubakub_viewport" and seen["live_camera"] is None,
          seen)
    check("report says live from Blender", report.startswith("live from Blender 1.0: facade.blend, your viewport view"),
          report[:200])
    check("report has no 'sample' note when live", "sample" not in report.lower(), report)
    live.forget()
    out = ns.KUBA_SceneRender.execute(**dict(rkw, source="live houdini"))
    check("houdini: the view and the frame size go to the export",
          seen["live_camera"] and seen["live_camera"]["name"] == "projcam" and (seen["width"], seen["height"]) == (1920, 1080),
          seen)
    check("the scene cable carries the live camera (preview / relight see the same view)",
          out.args[8]["live_camera"] == seen["live_camera"])
    live.forget()
    proj = {"distance_m": 30.0}
    ns.autocam.spec = lambda p: p
    ns.KUBA_SceneRender.execute(**dict(rkw, source="live houdini", projector=proj))
    check("a connected projector wins over the app's view", seen["live_camera"] is None and seen["projector"] == proj, seen)
    fh.close()

    # the app reports an error
    fb.error = "disk full"
    live.forget()
    try:
        ns._live_scene("", "", "live blender")
        check("an error from the app is raised", False)
    except RuntimeError as e:
        check("an error from the app is raised", "disk full" in str(e) and "Blender" in str(e), str(e))
    fb.error = ""

    # Blender closed: the last snapshot, said in the report, no waiting (also after a restart of ComfyUI)
    fb.close()
    live._LAST.clear()
    live.forget()
    t0 = time.time()
    L = ns._live_scene("", "", "live blender")
    check("Blender closed: the last snapshot, no wait", os.path.basename(L["file"]) == "blender_snap_002.blend"
          and "as it was last sent" in L["line"] and time.time() - t0 < 1.0, L["line"])

    # a live_<app>.json naming a file elsewhere: only a file of the live folder is opened
    outside = os.path.join(TMP, "elsewhere.blend")
    open(outside, "wb").write(b"x")
    for bad in (outside, "..\\elsewhere.blend", "../elsewhere.blend"):
        live._write(os.path.join(LIVE, "live_blender.json"), {"id": "x", "file": bad})
        got = live.current(LIVE, "blender")
        check(f"snapshot outside the live folder is not opened ({bad[-20:]})",
              got is None or os.path.dirname(got["file"]) == os.path.abspath(LIVE), got)
    live._LAST.clear()
    check("... and without an earlier good answer: nothing", live.current(LIVE, "blender") is None)

    # an answer that does not come: the last snapshot after live_wait
    with open(os.path.join(LIVE, "blender.alive"), "w") as f_:
        f_.write("{}")
    live._write(os.path.join(LIVE, "live_blender.json"), {"id": "old", "file": "blender_snap_002.blend", "time": time.time()})
    t0 = time.time()
    ans, note = live.ask(LIVE, "blender", wait_s=0.5)
    check("no answer in time: the last snapshot and a note",
          ans and os.path.basename(ans["file"]) == "blender_snap_002.blend" and "did not answer" in note
          and time.time() - t0 < 2, note)
    old = time.time() - 60
    os.utime(os.path.join(LIVE, "blender.alive"), (old, old))
    check("an old .alive file = not linked", live.alive(LIVE, "blender") is None)
    # unit_scale 0 = auto: a model in cm (a scanned facade: 2470 units) is exported again at 0.01, said in the report
    g = ns.guess_unit_scale
    check("auto scale: metres stay, cm and mm are found", (g(25.0), g(300.0), g(2470.0), g(6000.0), g(25000.0),
                                                            g(60000.0), g(0.0)) == (1.0, 1.0, 0.01, 0.01, 0.001, 0.001, 1.0))
    scales = []
    big = os.path.join(TMP, "scene_cm")
    os.makedirs(big, exist_ok=True)
    json.dump({"bbox_min": [0, 0, 0], "bbox_max": [1200, 300, 2470]}, open(os.path.join(big, "scene.json"), "w"))
    shutil.copy(os.path.join(sfolder, "clay.png"), os.path.join(big, "clay.png"))

    def stub_export_cm(path, cache_root, unit_scale=1.0, **k):
        scales.append(unit_scale)
        return (big if unit_scale == 1.0 else sfolder), False, ""
    ns.bridge.export = stub_export_cm
    live.forget()
    out = ns.KUBA_SceneRender.execute(**dict(rkw, source="live houdini"))
    check("auto: export at 1, measured, again at 0.01", scales == [1.0, 0.01], scales)
    check("auto: the report says it", "unit_scale auto: the model is 2470 units across, read as cm -> 24.7 m"
          in out.args[7] and "wrong units" not in out.args[7], out.args[7])
    check("auto: the scene cable carries the found scale", out.args[8]["unit_scale"] == 0.01)
    scales.clear()
    live.forget()
    ns.KUBA_SceneRender.execute(**dict(rkw, source="live houdini", unit_scale=1.0))
    check("a typed unit_scale is used as it is", scales == [1.0], scales)
    ns.bridge.export = stub_export

    ns.bridge.export, ns.scene_ids.build, ns.scene_ids.load = real_export, real_build, real_load

    # the bridge: the live camera is part of the cache key and of what a scene hands on
    f = os.path.join(LIVE, "blender_snap_002.blend")
    k1 = ns.bridge.cache_key(f, "", 1920, 1080, -1)
    k2 = ns.bridge.cache_key(f, "", 1920, 1080, -1, live_camera={"lens": 30})
    check("another live view = another cache folder", k1 != k2)
    check("scene_opts hands the live camera on", ns.bridge.scene_opts({"live_camera": {"lens": 30}})["live_camera"]
          == {"lens": 30})

    # the schema: source is the last input (old workflows keep their widget values), default file
    names = [i.id for i in ns.KUBA_SceneRender.define_schema().inputs]
    check("source is the last input, default file", names[-1] == "source")
finally:
    shutil.rmtree(TMP, ignore_errors=True)

print("\nALL PASSED" if not failures else f"\n{len(failures)} FAILED: {failures}")
sys.exit(1 if failures else 0)
