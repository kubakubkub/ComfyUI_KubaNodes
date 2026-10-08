"""
Model free test for the 3D scene nodes themselves (nodes/scene3d): what they hand on. A synthetic facade (a wall,
a ledge, a window block and a kiosk, as tests/test_scene3d.py builds it) is written in the format Blender exports,
and the nodes run on it: the pieces falloff keeps start / speed / order, scene pieces gives its mask, scene measure
its raw map, brightness compensation its gain mask, projector blend white masks for projectors that are not there,
scene preview runs without a matrix, and one connected viewer moves the audience camera of scene preview, scene
relight and pieces render. Blender is replaced by stubs; nothing renders and nothing touches the GPU.

Run from the ComfyUI_windows_portable folder:
    python_embeded\\python.exe ComfyUI\\custom_nodes\\ComfyUI_KubaNodes\\tests\\test_scene3d_nodes.py
"""

import json
import os
import shutil
import sys
import tempfile
import types

import cv2
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
PACK = os.path.dirname(HERE)
sys.path.insert(0, PACK)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(HERE))))     # ComfyUI (folder_paths, comfy_api)
import folder_paths  # noqa: E402

TMP = tempfile.mkdtemp(prefix="kuba_s3dnodes_")
folder_paths.set_temp_directory(os.path.join(TMP, "temp"))
folder_paths.set_output_directory(os.path.join(TMP, "output"))
folder_paths.set_input_directory(os.path.join(TMP, "input"))
folder_paths.set_user_directory(os.path.join(TMP, "user"))
os.environ.pop("KUBA_CACHE", None)
os.environ["KUBA_BLENDER_WORKER"] = "0"
sys.path.insert(0, HERE)                              # embedded Python leaves the script folder off the path
import _pack  # noqa: E402,F401  (the pack as a package, see tests/_pack.py)
import kubapack.nodes.scene3d.nodes_scene3d as ns  # noqa: E402
import kubapack.nodes.scene3d.nodes_pieces as npc  # noqa: E402
import kubapack.nodes.scene3d.nodes_compensate as ncp  # noqa: E402
from kubakub import viewer as vw  # noqa: E402
from kubakub.scene3d import bridge, pieces as pc, scene_ids, scene_view as sv  # noqa: E402

# pieces render asks ComfyUI whether Stop was pressed: a stand-in, so the test never loads the GPU side of ComfyUI
import comfy  # noqa: E402
_mm = types.ModuleType("comfy.model_management")
_mm.throw_exception_if_processing_interrupted = lambda: None
sys.modules["comfy.model_management"] = _mm
comfy.model_management = _mm

failures = []


def check(name, cond, detail=""):
    print(("ok   " if cond else "FAIL ") + name + (f"  ({detail})" if detail and not cond else ""))
    if not cond:
        failures.append(name)


def box_faces(x0, x1, y0, y1, z0, z1, verts):
    base = len(verts)
    for x in (x0, x1):
        for y in (y0, y1):
            for z in (z0, z1):
                verts.append((x, y, z))
    v = lambda i, j, k: base + i * 4 + j * 2 + k  # noqa: E731
    return [((v(0, 0, 0), v(1, 0, 0), v(1, 0, 1), v(0, 0, 1)), (0, -1, 0)), ((v(0, 1, 0), v(0, 1, 1), v(1, 1, 1), v(1, 1, 0)), (0, 1, 0)),
            ((v(0, 0, 0), v(0, 0, 1), v(0, 1, 1), v(0, 1, 0)), (-1, 0, 0)), ((v(1, 0, 0), v(1, 1, 0), v(1, 1, 1), v(1, 0, 1)), (1, 0, 0)),
            ((v(0, 0, 1), v(1, 0, 1), v(1, 1, 1), v(0, 1, 1)), (0, 0, 1)), ((v(0, 0, 0), v(0, 1, 0), v(1, 1, 0), v(1, 0, 0)), (0, 0, -1))]


BOXES = [(0, 10, 0.0, 0.5, 0, 6), (3, 5, -0.4, 0.0, 3.0, 3.3), (7, 9, 0.3, 0.5, 2, 4), (11, 12, -1.0, 0.0, 0, 1)]   # wall, ledge, window, kiosk


def make_scene(folder, W=130, H=70, tilt_left=False):
    """The export of an ortho front camera 13 x 7 m, 30 m away. tilt_left: the left half of the wall is written as
    turned 37 degrees away (0.8 of the light), for the brightness compensation."""
    verts, faces = [], []
    for b in BOXES:
        faces += box_faces(*b, verts)
    V = np.array(verts, np.float32)
    X, Z = np.meshgrid(-0.5 + (np.arange(W) + 0.5) * 13.0 / W, 6.5 - (np.arange(H) + 0.5) * 7.0 / H)
    faceid, best = np.zeros((H, W), np.uint32), np.full((H, W), np.inf)
    pos, nrm = np.zeros((H, W, 3), np.float32), np.zeros((H, W, 3), np.float32)
    for bi, (x0, x1, y0, y1, z0, z1) in enumerate(BOXES):
        m = (X >= x0) & (X <= x1) & (Z >= z0) & (Z <= z1) & (y0 < best)
        if bi == 0:
            m &= ~((X > 7) & (X < 9) & (Z > 2) & (Z < 4))
        best[m] = y0
        faceid[m] = bi * 6 + 1                        # the box's front face
        pos[m] = np.stack([X[m], np.full(m.sum(), y0), Z[m]], 1)
        nrm[m] = (0, -1, 0)
    if tilt_left:
        nrm[(faceid == 1) & (X < 5)] = (0.6, -0.8, 0)
    os.makedirs(folder, exist_ok=True)
    np.save(os.path.join(folder, "faceid.npy"), faceid)
    np.save(os.path.join(folder, "position.npy"), pos)
    np.save(os.path.join(folder, "normal.npy"), nrm.astype(np.float16))
    np.savez_compressed(os.path.join(folder, "mesh.npz"), normal=np.array([f[1] for f in faces], np.float32),
                        centre=np.array([V[list(f[0])].mean(0) for f in faces], np.float32),
                        area=np.ones(len(faces), np.float32), obj=np.zeros(len(faces), np.int32),
                        mat=np.zeros(len(faces), np.int32), col=np.zeros(len(faces), np.int32),
                        loop_total=np.full(len(faces), 4, np.int32),
                        loop_vert=np.array([i for f in faces for i in f[0]], np.int32), vert=V)
    cam = np.eye(4)
    cam[:3, :3] = [[1, 0, 0], [0, 0, -1], [0, 1, 0]]  # looks along +Y
    cam[:3, 3] = (6, -30, 3)
    info = {"file": "synthetic", "blender": "-", "width": W, "height": H, "faces": len(faces),
            "camera": {"name": "cam", "how": "test", "type": "ORTHO", "lens_mm": 50, "shift_x": 0, "shift_y": 0,
                       "matrix_world": cam.tolist(), "projection": [[2 / 13, 0, 0, 0], [0, 2 / 7, 0, 0],
                                                                    [0, 0, -2 / 99.9, -100.1 / 99.9], [0, 0, 0, 1]]},
            "cameras": ["cam"], "objects": ["facade"], "materials": [""], "collections": [""],
            "bbox_min": V.min(0).tolist(), "bbox_max": V.max(0).tolist(), "notes": []}
    json.dump(info, open(os.path.join(folder, "scene.json"), "w"))
    cv2.imwrite(os.path.join(folder, "clay.png"), np.where(faceid[..., None] > 0, 204, 0).astype(np.uint8).repeat(3, -1))
    return faceid


def ins(cls):
    return {i.id: i for i in cls.define_schema().inputs}


def outs(cls):
    return cls.define_schema().outputs


def described(items, *names):
    by = {i.id: i for i in items}
    return all(len((by[n].tooltip or "").strip()) > 10 for n in names)


try:
    W, H = 130, 70
    folder = os.path.join(TMP, "export")
    faceid = make_scene(folder, W, H)
    scene_file = os.path.join(TMP, "facade.obj")
    open(scene_file, "w").write("v 0 0 0\n")
    scene = {"folder": folder, "ids": folder, "file": scene_file, "blender": "", "unit_scale": 1.0, "frame": -1,
             "camera": "", "projector": None}
    s, pt, nrm, ground, fr = ns.load_scene_cached(scene)
    viewer = vw.Viewer(W, H, 20.0, 0.0, 14.0, 2.5, 9.0)       # 4 m right of the middle, 9 m away, eyes at 2.5 m
    check("viewer -> audience spot: x from the middle, distance, eye height", ns.viewer_spot(viewer) == (4.0, 9.0, 2.5))
    check("viewer without an x stands in the middle", ns.viewer_spot(vw.Viewer(W, H, 20.0))[0] == 0.0)

    # --- pieces falloff: start, speed and the order reach the falloff ---------------------------------------------
    fkw = dict(type="wave", direction="left to right", start=2.0, speed=6.0, width=3.0, spread=2.0, duration=1.0,
               hold=-1.0, loop=0.0, frequency=0.5, seed=1, amount=1.0, invert=False)
    f = npc.KUBA_PiecesFalloff.execute(**fkw).args[0]
    check("falloff: start, speed and order are in the falloff",
          f.get("start") == 2.0 and f.get("speed") == 6.0 and f.get("order") == "left to right" and f.get("axis") == "left to right",
          str(f))
    u, v = np.array([1.0, 5.0, 9.0]), np.array([1.0, 1.0, 1.0])
    check("falloff: nothing moves before start", not pc.falloff(f, 1.0, u, v, 10.0, 6.0).any()
          and pc.falloff(npc.KUBA_PiecesFalloff.execute(**dict(fkw, start=0.0)).args[0], 1.0, u, v, 10.0, 6.0).any())
    slow = pc.falloff(npc.KUBA_PiecesFalloff.execute(**dict(fkw, start=0.0, speed=1.0)).args[0], 1.0, u, v, 10.0, 6.0)
    fast = pc.falloff(npc.KUBA_PiecesFalloff.execute(**dict(fkw, start=0.0, speed=12.0)).args[0], 1.0, u, v, 10.0, 6.0)
    check("falloff: speed sets how far the wave got", fast.sum() > slow.sum() + 0.5, f"{slow} {fast}")
    lr = pc.falloff(npc.KUBA_PiecesFalloff.execute(**dict(fkw, type="stagger", start=0.0)).args[0], 0.5, u, v, 10.0, 6.0)
    rl = pc.falloff(npc.KUBA_PiecesFalloff.execute(**dict(fkw, type="stagger", start=0.0, direction="right to left")).args[0],
                    0.5, u, v, 10.0, 6.0)
    check("falloff: stagger follows the direction", lr[0] > lr[2] and rl[2] > rl[0], f"{lr} {rl}")
    check("falloff: every input and the output say what they do",
          all(len((i.tooltip or "").strip()) > 10 for i in npc.KUBA_PiecesFalloff.define_schema().inputs)
          and "metres" in ins(npc.KUBA_PiecesFalloff)["width"].tooltip and "seconds" in ins(npc.KUBA_PiecesFalloff)["spread"].tooltip
          and "seconds" in ins(npc.KUBA_PiecesFalloff)["start"].tooltip and described(outs(npc.KUBA_PiecesFalloff), "falloff"))

    # --- scene pieces: the mask of what can move --------------------------------------------------------------------
    sp_out = npc.KUBA_ScenePieces.execute(scene, 0.2, 0.0, True).args
    pieces, mask = sp_out[0], sp_out[3].numpy()
    movers = np.isin(faceid, (7, 13, 19))                    # the front faces of ledge, window block and kiosk
    check("scene pieces: outputs stay in place, the mask comes last",
          [o.id for o in outs(npc.KUBA_ScenePieces)] == ["pieces", "preview", "report", "mask"] and len(sp_out) == 4)
    check("scene pieces: mask = 1 on the moving pieces only", pieces["n"] == 3 and mask.shape == (1, H, W)
          and np.array_equal(mask[0] > 0.5, movers), f"{pieces['n']} pieces, {int(mask.sum())} of {int(movers.sum())} px")
    check("scene pieces: pieces, report and mask have tooltips", described(outs(npc.KUBA_ScenePieces), "pieces", "report", "mask"))

    # --- scene measure: the raw map --------------------------------------------------------------------------------
    m_out = ns.KUBA_SceneMeasure.execute(scene, 3.0, 12.0, 2.0, "distance", 60.0).args
    raw, fg = m_out[5].numpy(), faceid > 0
    check("scene measure: raw comes last, the others stay",
          [o.id for o in outs(ns.KUBA_SceneMeasure)] == ["regions", "viewer", "map", "table", "report", "raw"])
    check("scene measure: raw is a 0..1 mask of the view, 0 outside the building",
          raw.shape == (1, H, W) and raw.dtype == np.float32 and raw.min() == 0.0 and raw.max() == 1.0 and not raw[0][~fg].any())
    check("scene measure: raw is not inverted (the far window block is whiter than the near kiosk)",
          raw[0][faceid == 13].mean() > raw[0][faceid == 19].mean())
    check("scene measure: the report says what 0 and 1 of the raw mask are", "raw mask: 0 = " in m_out[4], m_out[4])
    check("scene measure: no regions in, none out, and the tooltip and report say so",
          m_out[0] is None and "regions output is empty" in m_out[4] and "empty" in outs(ns.KUBA_SceneMeasure)[0].tooltip)
    check("scene measure: its viewer is the spot typed on it", ns.viewer_spot(m_out[1]) == (3.0, 12.0, 2.0),
          str(ns.viewer_spot(m_out[1])))

    # --- brightness compensation: the gain as a mask -----------------------------------------------------------------
    tilt = os.path.join(TMP, "export_tilt")
    make_scene(tilt, W, H, tilt_left=True)
    tscene = dict(scene, folder=tilt)
    images = torch.full((2, H * 2, W * 2, 3), 0.5)
    c_out = ncp.KUBA_BrightnessCompensation.execute(images, tscene, "dim areas", 1.0, 1.0, 0.0).args
    gain = c_out[3].numpy()
    check("compensation: gain comes last, the others stay",
          [o.id for o in outs(ncp.KUBA_BrightnessCompensation)] == ["images", "gain_map", "report", "gain"]
          and described(outs(ncp.KUBA_BrightnessCompensation), "gain"))
    check("compensation: the gain mask has the frames' size, one for the batch", gain.shape == (1, H * 2, W * 2), str(gain.shape))
    check("compensation: without lift the mask is the gain itself (dim left 1.0, bright right 0.8)",
          abs(gain[0, 20, 20] - 1.0) < 0.02 and abs(gain[0, 20, 150] - 0.8) < 0.02 and gain.max() <= 1.0,
          f"{gain[0, 20, 20]:.3f} {gain[0, 20, 150]:.3f}")
    check("compensation: the report says the mask's range", "gain mask: 0 .. 1 = a gain of 0 .. 1.00" in c_out[2], c_out[2])
    c2 = ncp.KUBA_BrightnessCompensation.execute(images, tscene, "average", 1.0, 2.0, 0.0).args
    g2 = c2[3].numpy()
    check("compensation: with lift, white is the highest gain (1.25) and the report says it",
          abs(g2.max() - 1.0) < 1e-6 and abs(g2[0, 20, 150] - 0.8) < 0.02 and "gain of 0 .. 1.25" in c2[2], c2[2])

    # --- projector blend: no projector 3 / 4 --------------------------------------------------------------------------
    b_out = ns.KUBA_ProjectorBlend.execute(scene, scene, 0.1, 2.2).args
    check("projector blend: masks of projectors that are not there are white at projector 1's size",
          all(tuple(b_out[k].shape) == (1, H, W) and bool((b_out[k] == 1).all()) for k in (2, 3))
          and tuple(b_out[0].shape) == (1, H, W))
    check("projector blend: every mask output says what it is", described(outs(ns.KUBA_ProjectorBlend), "mask_2", "mask_3", "mask_4"))

    # --- scene render: the passes tooltip ----------------------------------------------------------------------------
    tip = ins(ns.KUBA_SceneRender)["passes"].tooltip
    check("scene render: the passes tooltip names every pass", all(p in tip for p in scene_ids.PASSES), tip)

    # --- scene preview: no matrix, and the viewer as the spot ------------------------------------------------------------
    real_export, views = bridge.export, []

    def fake_export(path, root, **k):                        # the audience view = the projection view itself
        views.append(k.get("view"))
        return folder, True, ""

    bridge.export = fake_export
    pkw = dict(spots="0, 20, 1.7\n-5, 12, 1.7", lens_mm=20.0, width=W, height=H, ambient=0.15, gain=1.0, physical=1.0)
    p0 = ns.KUBA_ScenePreview.execute(scene, None, **pkw).args
    check("scene preview: matrix is optional now, the viewer is new, both described",
          ins(ns.KUBA_ScenePreview)["matrix"].optional and ins(ns.KUBA_ScenePreview)["viewer"].optional
          and described(ns.KUBA_ScenePreview.define_schema().inputs, "matrix", "viewer")
          and [o.id for o in outs(ns.KUBA_ScenePreview)] == ["preview", "shadow", "views", "report", "scene"])
    check("scene preview without a matrix: the clay views, one per spot, and the report says so",
          tuple(p0[0].shape) == (2, H, W, 3) and torch.allclose(p0[0], p0[2] * torch.from_numpy(fg)[None, ..., None].float())
          and float(p0[0].max()) > 0.9 and "no matrix connected" in p0[3], p0[3])
    p1 = ns.KUBA_ScenePreview.execute(scene, torch.ones(3, H, W, 3), **pkw).args
    check("scene preview with a matrix: as before (per spot all frames, dimmed clay plus the picture)",
          tuple(p1[0].shape) == (6, H, W, 3) and "no matrix" not in p1[3])
    views.clear()
    p2 = ns.KUBA_ScenePreview.execute(scene, None, viewer=viewer, **pkw).args
    want = sv.spot_position(fr, 4.0, 9.0, 2.5)
    check("scene preview: a connected viewer replaces the typed spots",
          len(views) == 1 and np.allclose(views[0]["location"], want, atol=1e-3) and tuple(p2[0].shape) == (1, H, W, 3)
          and len(p2[4]["viewpoints"]) == 1 and "connected viewer" in p2[3], f"{views} {want}")
    bridge.export = real_export

    # --- scene walkthrough: the schema (the run is in test_scene_cache.py) -----------------------------------------------
    check("scene walkthrough: matrix optional, fps and shadow after the report, fps input still a whole number",
          ins(ns.KUBA_SceneWalkthrough)["matrix"].optional
          and [o.id for o in outs(ns.KUBA_SceneWalkthrough)] == ["frames", "report", "fps", "shadow"]
          and described(outs(ns.KUBA_SceneWalkthrough), "fps", "shadow")
          and ins(ns.KUBA_SceneWalkthrough)["fps"].io_type == "INT" and "whole number" in ins(ns.KUBA_SceneWalkthrough)["fps"].tooltip)

    # --- scene relight: empty hdri_file, the viewer ------------------------------------------------------------------------
    real_relight, rigs = ns.relight_scene, []

    def fake_relight(scene, rig, emission=(), projector=None, look_from=None, **k):
        rigs.append((rig, look_from, projector))
        return np.zeros((H, W, 3), np.float32), np.ones((H, W), np.float32), "stub render"

    ns.relight_scene = fake_relight
    rkw = dict(env_strength=0.3, env_rotation_deg=0.0, lights=ns.DEFAULT_LIGHTS, clay=0.7, roughness=0.8, samples=8,
               exposure=0.0, background="black", resolution_scale=1.0)
    r0 = ns.KUBA_SceneRelight.execute(scene, "file", hdri_file="  ", **rkw).args
    check("scene relight: environment = file with an empty hdri_file runs on the night environment and says so",
          rigs[-1][0]["environment"] == "night" and "night environment" in r0[2] and "hdri_file is empty" in r0[2], r0[2])
    ns.KUBA_SceneRelight.execute(scene, "file", hdri_file=os.path.join(TMP, "sky.hdr"), **rkw)
    check("scene relight: a typed hdri_file is still used as it is", rigs[-1][0]["environment"] == "file")
    r1 = ns.KUBA_SceneRelight.execute(scene, "night", projector=torch.rand(3, H, W, 3), **rkw).args
    check("scene relight: the matrix tooltip says only the first frame is used, and only that one is sent",
          "first frame" in ins(ns.KUBA_SceneRelight)["projector"].tooltip and rigs[-1][2].shape == (H, W, 3)
          and "night environment" not in r1[2])
    r2 = ns.KUBA_SceneRelight.execute(scene, "night", camera="audience", audience_distance_m=30.0, audience_offset_m=0.0,
                                      viewer=viewer, **rkw).args
    check("scene relight: with camera = audience the connected viewer is where the camera stands",
          np.allclose(rigs[-1][1]["location"], want, atol=1e-3) and "connected viewer" in r2[2], f"{rigs[-1][1]} {want}")
    ns.KUBA_SceneRelight.execute(scene, "night", camera="audience", audience_distance_m=30.0, audience_offset_m=1.0, **rkw)
    check("scene relight: without a viewer the typed spot counts, as before",
          np.allclose(rigs[-1][1]["location"], sv.spot_position(fr, 1.0, 30.0, 1.7), atol=1e-3))
    ns.KUBA_SceneRelight.execute(scene, "night", viewer=viewer, **rkw)
    check("scene relight: camera = projector ignores the viewer", rigs[-1][1] is None)
    check("scene relight: outputs unchanged, the viewer input optional and described",
          [o.id for o in outs(ns.KUBA_SceneRelight)] == ["image", "alpha", "report", "plan"]
          and ins(ns.KUBA_SceneRelight)["viewer"].optional and described(ns.KUBA_SceneRelight.define_schema().inputs, "viewer", "hdri_file"))
    ns.relight_scene = real_relight

    # --- pieces render: exposure and the viewer --------------------------------------------------------------------------
    real_frames, jobs = ns.relight_frames, []

    def fake_frames(scene, parts, rw, rh, samples, timeout=900):
        jobs.append(parts)
        got = []
        for k, _ in enumerate(parts):
            fp = os.path.join(TMP, f"relit_{len(jobs)}_{k}.png")
            cv2.imwrite(fp, np.full((rh, rw, 4), 65535, np.uint16))
            got.append((fp, {}))
        return got, list(range(len(parts))), {"seconds": {"per_frame": [0.1] * len(parts)}}

    ns.relight_frames = fake_frames
    moved = npc.KUBA_PiecesTransform.execute(pieces, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1, falloff=f).args[0]
    qkw = dict(duration=0.2, fps=10.0, look="clay", projector_brightness=1.0, environment="night", env_strength=0.3,
               clay=0.7, background="black", resolution_scale=1.0, samples=4)
    q0 = npc.KUBA_PiecesRender.execute(moved, **qkw).args
    check("pieces render: frames, alpha, fps, report as before; exposure 0 when not set",
          tuple(q0[0].shape) == (2, H, W, 3) and tuple(q0[1].shape) == (2, H, W) and q0[2] == 10.0
          and all(p["exposure"] == 0.0 and "look_from" not in p for p in jobs[-1])
          and [o.id for o in outs(npc.KUBA_PiecesRender)] == ["frames", "alpha", "fps", "report"], q0[3])
    q1 = npc.KUBA_PiecesRender.execute(moved, exposure=1.5, **qkw).args
    check("pieces render: exposure reaches the render and the report", all(p["exposure"] == 1.5 for p in jobs[-1])
          and "exposure +1.5" in q1[3], q1[3])
    q2 = npc.KUBA_PiecesRender.execute(moved, view="audience", viewer=viewer, **qkw).args
    cam_pos = np.asarray(s["info"]["camera"]["matrix_world"], np.float64)[:3, 3]
    spot = pc.audience_view(fr, cam_pos, 9.0, 4.0, 2.5, 24.0)
    check("pieces render: with view = audience the connected viewer is where the camera stands",
          np.allclose(jobs[-1][0]["look_from"]["location"], spot["location"]) and "connected viewer" in q2[3], q2[3])
    npc.KUBA_PiecesRender.execute(moved, viewer=viewer, **qkw)
    check("pieces render: view = projector ignores the viewer", "look_from" not in jobs[-1][0])
    pin = ins(npc.KUBA_PiecesRender)
    check("pieces render: exposure (advanced, default 0) and viewer are optional and described",
          pin["exposure"].optional and pin["exposure"].advanced and pin["exposure"].default == 0.0 and pin["viewer"].optional
          and described(npc.KUBA_PiecesRender.define_schema().inputs, "exposure", "viewer"))
    ns.relight_frames = real_frames
finally:
    shutil.rmtree(TMP, ignore_errors=True)

print(f"\n{len(failures)} failure(s)" + (": " + ", ".join(failures) if failures else ""))
sys.exit(1 if failures else 0)
