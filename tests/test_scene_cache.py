"""
Model free test for the scene3d disk cache (kubakub/scene3d/bridge.py, nodes/scene3d/nodes_scene3d.py): content
keys (scene file, HDRI, %VAR% paths), the size cap (least recently used first), restart survival (the cache is not
in ComfyUI's temp), the relight frame store (only changed frames render again), walkthrough views on disk and the
reuse of a finished ids_<key> folder. Blender is replaced by stubs; nothing renders.

Run from the ComfyUI_windows_portable folder:
    python_embeded\\python.exe ComfyUI\\custom_nodes\\ComfyUI_KubaNodes\\tests\\test_scene_cache.py
"""

import json
import os
import shutil
import sys
import tempfile
import time

import cv2
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
PACK = os.path.dirname(HERE)
sys.path.insert(0, PACK)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(HERE))))     # ComfyUI (folder_paths, comfy_api)
import folder_paths  # noqa: E402

TMP = tempfile.mkdtemp(prefix="kuba_scache_")
folder_paths.set_temp_directory(os.path.join(TMP, "temp"))
folder_paths.set_output_directory(os.path.join(TMP, "output"))
folder_paths.set_input_directory(os.path.join(TMP, "input"))
folder_paths.set_user_directory(os.path.join(TMP, "user"))
os.environ.pop("KUBA_CACHE", None)
os.environ["KUBA_BLENDER_WORKER"] = "0"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))     # embedded Python leaves the script folder off the path
import _pack  # noqa: E402,F401  (the pack as a package, see tests/_pack.py)
import kubapack.nodes.scene3d.nodes_scene3d as ns  # noqa: E402
from kubakub.scene3d import bridge  # noqa: E402

failures = []


def check(name, cond, detail=""):
    print(("ok   " if cond else "FAIL ") + name + (f"  ({detail})" if detail and not cond else ""))
    if not cond:
        failures.append(name)


def write(path, data=b"x"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(data)


def bump(path, dt=2.0):
    st = os.stat(path)
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + int(dt * 1e9)))


try:
    # --- where the cache lives ------------------------------------------------------------------------------
    root = ns.cache_root()
    check("default cache is in ComfyUI's user directory, not temp",
          root.startswith(folder_paths.get_user_directory()) and not root.startswith(folder_paths.get_temp_directory()))
    os.environ["KUBA_CACHE"] = os.path.join(TMP, "elsewhere")
    check("env KUBA_CACHE moves it", ns.cache_root().startswith(os.path.join(TMP, "elsewhere")))
    os.environ["KUBA_CACHE"] = "temp"
    check("KUBA_CACHE=temp keeps the old place", ns.cache_root().startswith(folder_paths.get_temp_directory()))
    os.environ.pop("KUBA_CACHE")
    os.environ["KUBA_CACHE_GB"] = "1.5"
    check("KUBA_CACHE_GB sets the cap", ns.cache_cap_bytes() == int(1.5 * 2 ** 30))
    os.environ.pop("KUBA_CACHE_GB")

    # --- content keys ------------------------------------------------------------------------------------------
    scene_file = os.path.join(TMP, "proj", "facade.obj")
    write(scene_file, b"v 0 0 0\n")
    hdri = os.path.join(TMP, "proj", "sky.hdr")
    write(hdri, b"HDR1")
    k0 = bridge.cache_key(scene_file, "", 64, 32, -1)
    check("same file, same key", k0 == bridge.cache_key(scene_file, "", 64, 32, -1))
    write(scene_file, b"v 0 0 0\nv 1 0 0\n")
    bump(scene_file)
    check("an edited scene file is a new key", bridge.cache_key(scene_file, "", 64, 32, -1) != k0)
    rel = {"samples": 16, "env": hdri, "lights": []}
    k1 = bridge.cache_key(scene_file, "", 64, 32, -1, relight=rel)
    write(hdri, b"HDR2-longer")
    bump(hdri)
    check("an edited HDRI (same name) is a new relight key", bridge.cache_key(scene_file, "", 64, 32, -1, relight=rel) != k1)
    check("hdri_stamp follows the HDRI's size / mtime", str(os.path.getsize(hdri)) in ns.hdri_stamp(hdri))
    check("SceneRelight fingerprint uses the HDRI stamp only for environment = file",
          ns.KUBA_SceneRelight.fingerprint_inputs(environment="file", hdri_file=hdri) == ns.hdri_stamp(hdri)
          and ns.KUBA_SceneRelight.fingerprint_inputs(environment="night", hdri_file=hdri) == "")

    # L1: a %VAR% path gives the same fingerprint as the expanded path, and it follows the file
    os.environ["KUBA_TEST_PROJ"] = os.path.join(TMP, "proj")
    var_path = '"%KUBA_TEST_PROJ%' + os.sep + 'facade.obj"' if os.name == "nt" else '"$KUBA_TEST_PROJ/facade.obj"'
    check("clean_path expands variables and quotes", bridge.clean_path(var_path) == scene_file, bridge.clean_path(var_path))
    kw = dict(camera="", width=64, height=32, frame=-1, unit_scale=1.0)
    f_var = ns.KUBA_SceneRender.fingerprint_inputs(file=var_path, **kw)
    f_abs = ns.KUBA_SceneRender.fingerprint_inputs(file=scene_file, **kw)
    write(scene_file, b"v 0 0 0\nv 1 0 0\nv 2 0 0\n")
    bump(scene_file, 4)
    check("scene render fingerprint: %VAR% path sees file changes",
          ns.KUBA_SceneRender.fingerprint_inputs(file=var_path, **kw) != f_var and f_var != "" and f_abs)

    # --- staging + publish -----------------------------------------------------------------------------------
    final = os.path.join(TMP, "pub", "scene_abc")
    st1, st2 = bridge.stage_path(final), bridge.stage_path(final)
    for st, tag in ((st1, b"one"), (st2, b"two")):
        write(os.path.join(st, "scene.json"), tag)
        write(os.path.join(st, "faceid.npy"), tag)
    bridge.publish(st1, final, ("scene.json", "faceid.npy"))
    bridge.publish(st2, final, ("scene.json", "faceid.npy"))       # a second process finished the same job
    check("publish: the first finished folder wins, the second is dropped",
          open(os.path.join(final, "scene.json"), "rb").read() == b"one" and not os.path.exists(st2))

    # --- size cap -----------------------------------------------------------------------------------------------
    croot = os.path.join(TMP, "cap")
    now = time.time()
    for i, age_h in enumerate((10, 8, 6, 4)):                       # four scene folders of 1000 bytes
        d = os.path.join(croot, f"scene_{i}")
        write(os.path.join(d, "faceid.npy"), b"0" * 1000)
        for p in (os.path.join(d, "faceid.npy"), d):
            os.utime(p, (now - age_h * 3600, now - age_h * 3600))
    fr_store = os.path.join(croot, "relight_frames")
    for i, age_h in enumerate((9, 0.1)):                             # two frames of 500 + metadata
        write(os.path.join(fr_store, f"k{i}.png"), b"1" * 500)
        write(os.path.join(fr_store, f"k{i}.json"), b"{}")
        for p in (os.path.join(fr_store, f"k{i}.png"), os.path.join(fr_store, f"k{i}.json")):
            os.utime(p, (now - age_h * 3600, now - age_h * 3600))
    stale = os.path.join(croot, "scene_x" + bridge.STAGE + "1-abc")
    write(os.path.join(stale, "a"), b"2" * 10)
    os.utime(stale, (now - 24 * 3600, now - 24 * 3600))
    os.utime(os.path.join(stale, "a"), (now - 24 * 3600, now - 24 * 3600))
    total0 = sum(s for _, s, _ in bridge.entries(croot))
    check("entries: scene folders, store files (with their metadata), staging leftovers",
          len(bridge.entries(croot)) == 7 and total0 == 4000 + 2 * 502 + 10, str(total0))
    n, freed, left = bridge.purge(croot, cap_bytes=2600, now=now)
    kept = sorted(os.listdir(croot))
    check("purge: oldest first until under the cap",
          not os.path.exists(os.path.join(croot, "scene_0")) and not os.path.exists(os.path.join(fr_store, "k0.png"))
          and not os.path.exists(os.path.join(croot, "scene_1")) and os.path.isdir(os.path.join(croot, "scene_3")),
          str(kept))
    check("purge: a frame's metadata goes with it", not os.path.exists(os.path.join(fr_store, "k0.json")))
    check("purge: stale staging leftovers go", not os.path.exists(stale))
    check("purge: under the cap afterwards", left <= 2600, str(left))
    bridge.touch(os.path.join(croot, "scene_3"))                   # used again (a cache hit)
    n2, _, _ = bridge.purge(croot, cap_bytes=1, now=time.time())
    check("purge: recently used entries stay even over the cap", os.path.isfile(os.path.join(fr_store, "k1.png")))
    check("touch: a hit counts as a use (not removed)", os.path.isdir(os.path.join(croot, "scene_3"))
          and not os.path.isdir(os.path.join(croot, "scene_2")))

    # --- relight frame store ---------------------------------------------------------------------------------
    calls = []
    real_export = bridge.export

    def fake_export(path, cache_root, camera="", width=0, height=0, frame=-1, blender="", timeout=900, force=False,
                    unit_scale=1.0, view=None, views=None, relight=None, transient=False, projector=None):
        """Writes what Blender would: relit_XXXX.png per frame + relight.json, or v_XXXX view folders."""
        calls.append({"relight": relight, "views": views, "view": view, "transient": transient})
        out = bridge.stage_path(bridge.cache_folder(path, cache_root, camera, width, height, frame, unit_scale,
                                                    view, views, relight))
        os.makedirs(out)
        if relight:
            for F in relight["frames"]:
                v = int(F["exposure"] * 1000) % 65535
                img = np.full((height, width, 4), v, np.uint16)
                img[..., 3] = 65535
                cv2.imwrite(os.path.join(out, F["name"]), img)
            json.dump({"device": "STUB", "env": "", "notes": [], "emissive_per_frame": [0] * len(relight["frames"]),
                       "seconds": {"render": 0.1, "per_frame": [0.1] * len(relight["frames"])}},
                      open(os.path.join(out, "relight.json"), "w"))
        if views:
            for k, vw_ in enumerate(views):
                d = os.path.join(out, f"v_{k:04d}")
                os.makedirs(d)
                np.save(os.path.join(d, "faceid.npy"), np.ones((height, width), np.uint32))
                np.save(os.path.join(d, "position.npy"), np.zeros((height, width, 3), np.float32))
                np.save(os.path.join(d, "normal.npy"), np.zeros((height, width, 3), np.float16))
                cv2.imwrite(os.path.join(d, "clay.png"), np.full((height, width, 3), int(vw_["location"][0]) % 255, np.uint8))
            json.dump({"camera": {}, "width": width, "height": height}, open(os.path.join(out, "scene.json"), "w"))
        return out, False, ""

    bridge.export = fake_export
    W, H = 40, 20
    fid = np.ones((H, W), np.uint32)
    s_stub = {"info": {"width": W, "height": H}, "faceid": fid}
    fr_stub = {"centre": np.zeros(3), "ex": np.array([1.0, 0, 0]), "normal": np.array([0, -1.0, 0]), "ground_z": 0.0,
               "projector_distance_m": 20.0}
    ns.load_scene_cached = lambda scene: (s_stub, None, None, None, fr_stub)
    scene = {"file": scene_file, "folder": os.path.join(TMP, "nofolder"), "camera": "", "frame": -1, "unit_scale": 1.0}

    def rig(exposure):
        return {"environment": "none", "exposure": exposure, "samples": 8, "lights": [], "background": "black"}

    seq = [(rig(0.1 * (i % 4)), [], None) for i in range(12)]      # 12 frames, 4 distinct rigs
    readers, rep = ns.relight_sequence(scene, seq)
    job1 = calls[-1]["relight"]
    check("sequence: distinct rigs render once, in one transient job",
          len(calls) == 1 and len(job1["frames"]) == 4 and calls[-1]["transient"], rep)
    a0 = readers[0]()[0]
    check("sequence: readers give frames at the output size", a0.shape == (H, W, 3))
    store = os.path.join(ns.cache_root(), "relight_frames")
    check("sequence: frames kept in the store, the job folder removed",
          len([f for f in os.listdir(store) if f.endswith(".png")]) == 4
          and not any(bridge.STAGE in f for f in os.listdir(ns.cache_root())))
    readers2, rep2 = ns.relight_sequence(scene, seq)
    check("sequence again: nothing renders", len(calls) == 1 and "cached" in rep2, rep2)
    check("sequence again: same pixels", np.array_equal(readers2[5]()[0], readers[5]()[0]))
    seq2 = list(seq) + [(rig(0.7), [], None)]                      # the timeline got longer: one new rig
    seq2[3] = (rig(0.9), [], None)                                 # and one key moved
    ns.relight_sequence(scene, seq2)
    check("timeline edit: only the changed / new frames render",
          len(calls) == 2 and sorted(F["exposure"] for F in calls[-1]["relight"]["frames"]) == [0.7, 0.9],
          str(calls[-1]["relight"]["frames"] if len(calls) == 2 else calls))
    rgb, alpha, rep3 = ns.relight_scene(scene, rig(0.2))
    check("still with a rig the sequence had at the same size: from the store",
          len(calls) == 2 and "render cached" in rep3 and rgb.shape == (H, W, 3), rep3)
    ns.relight_scene(scene, rig(0.2), size=(W * 2, H * 2))
    check("still at another size renders", len(calls) == 3)
    hrig = dict(rig(0.2), environment="file", hdri_file=hdri)
    ns.relight_sequence(scene, [(hrig, [], None)])
    n_before = len(calls)
    ns.relight_sequence(scene, [(hrig, [], None)])
    write(hdri, b"HDR3-edited-again")
    bump(hdri, 6)
    ns.relight_sequence(scene, [(hrig, [], None)])
    check("HDRI edited under the same name: its frames render again",
          n_before == 4 and len(calls) == 5, str(len(calls)))
    k_frame = ns.relight_frame_key(scene, W, H, 8, {"exposure": 0.0})
    check("frame key ignores nothing that renders differently",
          k_frame != ns.relight_frame_key(scene, W, H, 16, {"exposure": 0.0})
          and k_frame != ns.relight_frame_key(scene, W + 2, H, 8, {"exposure": 0.0})
          and k_frame == ns.relight_frame_key(scene, W, H, 8, {"exposure": 0.0}))

    # --- restart survival ------------------------------------------------------------------------------------
    shutil.rmtree(folder_paths.get_temp_directory(), ignore_errors=True)        # what every ComfyUI start does
    import importlib
    importlib.reload(bridge)                                                    # a new process: no memory state
    bridge.export = fake_export
    ns.bridge = bridge
    n_before = len(calls)
    ns.relight_sequence(scene, seq)
    check("after a restart (temp emptied, modules fresh): frames still cached", len(calls) == n_before)

    # --- walkthrough views on disk -----------------------------------------------------------------------------
    import kubakub.scene3d.scene_view as svm
    ns._load_scene = lambda scene: ({"info": {"width": W, "height": H}}, None, None, None)
    svm.wall_frame = lambda info, pt, nrm, ground: fr_stub
    svm.reprojection = lambda vs, s: {"shadow": np.zeros(vs["faceid"].shape, bool), "model": vs["faceid"] > 0}
    svm.render_preview = lambda f, rp, clay, **k: clay
    matrix = torch.rand(1, H, W, 3)
    walk_kw = dict(path="-15, 25, 1.7\n0, 15, 1.7\n15, 10, 1.7", seconds=1.2, fps=25, lens_mm=24.0, width=32,
                   height=16, ambient=0.1, gain=1.0, physical=1.0)
    n0 = len(calls)
    out1 = ns.KUBA_SceneWalkthrough.execute(scene, matrix, **walk_kw)
    vcalls = [c for c in calls[n0:] if c["views"]]
    check("walkthrough: 30 views rendered in 25-view jobs", [len(c["views"]) for c in vcalls] == [25, 5],
          str([len(c["views"]) for c in vcalls]))
    n1 = len(calls)
    out2 = ns.KUBA_SceneWalkthrough.execute(scene, matrix, **walk_kw)
    check("walkthrough again: no Blender job, same frames", len(calls) == n1
          and torch.equal(out2.args[0], out1.args[0]) and "0 view(s) rendered" in out2.args[1], out2.args[1])
    ns.KUBA_SceneWalkthrough.execute(scene, matrix, **dict(walk_kw, seconds=1.6))
    check("walkthrough longer: only views not seen before render", len(calls) > n1
          and sum(len(c["views"]) for c in calls[n1:]) < 40)

    # --- ids_<key> folder reuse ----------------------------------------------------------------------------------
    builds = []
    sfolder = os.path.join(TMP, "scenefolder")
    os.makedirs(sfolder, exist_ok=True)
    cv2.imwrite(os.path.join(sfolder, "clay.png"), np.full((H, W, 3), 128, np.uint8))
    info = {"file": scene_file, "faces": 1, "objects": ["a"], "width": W, "height": H, "cameras": [], "blender": "stub",
            "camera": {"name": "c", "how": "file", "type": "PERSP", "lens_mm": 35.0, "shift_x": 0.0, "shift_y": 0.0},
            "bbox_min": [0, 0, 0], "bbox_max": [30, 1, 20], "notes": []}
    stub_scene = {"info": info, "faceid": fid, "normal": np.zeros((H, W, 3), np.float32)}

    def stub_build(folder, passes=(), out=None, **k):
        builds.append(out)
        os.makedirs(out, exist_ok=True)
        img = (np.random.default_rng(1).random((H, W, 3)) * 255).astype(np.uint8)
        for p in passes:
            cv2.imwrite(os.path.join(out, f"ids_{p}.png"), img[..., ::-1])
        depth = np.linspace(1, 2, H * W, dtype=np.float32).reshape(H, W)
        np.save(os.path.join(out, "depth_m.npy"), depth)
        summary = {"passes": {p: {"ids": 3, "largest": [["a", 5]], "merged_small": 0} for p in passes}, "notes": [],
                   "main_plane": {"point": [0, 0, 0], "normal": [0, -1, 0]}, "depth_m": [1.0, 2.0], "coverage": 1.0}
        json.dump(summary, open(os.path.join(out, "ids_summary.json"), "w"))
        return summary, {p: img for p in passes}, {"info": dict(info), "faceid": fid, "normal": stub_scene["normal"]}, depth

    ns.scene_ids.build = stub_build
    ns.scene_ids.load = lambda folder: {"info": dict(info), "faceid": fid, "normal": stub_scene["normal"]}
    bridge.export = lambda *a, **k: (sfolder, True, "")
    ns.bridge = bridge
    rkw = dict(file=scene_file, width=W, height=H, frame=-1, passes="shelves layers", preview_pass="shelves",
               shelf_min_step_m=0.05, plane_angle_deg=3.0, plane_offset_m=0.02, layer_bands_m="-0.5 -0.05 0.05 0.5")
    r1 = ns.KUBA_SceneRender.execute(**rkw)
    r2 = ns.KUBA_SceneRender.execute(**rkw)
    check("scene render: the finished ids folder is reused", len(builds) == 1 and "reused" in r2.args[7], r2.args[7][-200:])
    check("scene render: reused outputs are identical",
          all(torch.equal(r1.args[i], r2.args[i]) for i in (0, 1, 2, 3, 4, 9)) and r1.args[5] == r2.args[5]
          and r1.args[6] == r2.args[6])
    check("scene render: built into a staging folder, renamed into place",
          bridge.STAGE in builds[0] and os.path.isfile(os.path.join(r1.args[5], "ids_summary.json")))
    ns.KUBA_SceneRender.execute(**dict(rkw, preview_pass="layers"))
    check("scene render: another preview pass inside the passes reuses too", len(builds) == 1)
    ns.KUBA_SceneRender.execute(**dict(rkw, shelf_min_step_m=0.1))
    check("scene render: another setting builds", len(builds) == 2)
    bridge.export = real_export

    # --- worker settings ----------------------------------------------------------------------------------------
    os.environ["KUBA_BLENDER_IDLE"] = "900"
    check("worker idle from env", bridge.worker_idle_s() == 900.0)
    os.environ["KUBA_BLENDER_IDLE"] = "x"
    check("worker idle default 600 s", bridge.worker_idle_s() == 600.0)
    os.environ.pop("KUBA_BLENDER_IDLE")
    check("pre-start does nothing with the worker off", ns.prestart_relight(scene) is None)
    check("views jobs finish with the last view's faceid and scene.json",
          bridge.done_files(views=[{}, {}]) == (os.path.join("v_0001", "faceid.npy"), "scene.json"))
finally:
    shutil.rmtree(TMP, ignore_errors=True)

print(f"\n{len(failures)} failure(s)" + (": " + ", ".join(failures) if failures else ""))
sys.exit(1 if failures else 0)
