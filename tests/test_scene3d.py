"""
Model free test for the 3D scene tools (kubakub/scene3d).

A synthetic facade (wall with a recessed window and a projecting ledge, plus a separate box) is
rasterized in numpy into the same export format blender_export.py writes, then run through the ID
passes (scene_ids.py) and read back by the From ID Maps logic (idmaps.py). If Blender is installed, a
tiny OBJ is also exported through the real Blender bridge (a few seconds).

Run from the ComfyUI_windows_portable folder:
    python_embeded\\python.exe ComfyUI\\custom_nodes\\ComfyUI_KubaNodes\\tests\\test_scene3d.py
"""

import json
import os
import shutil
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PACK = os.path.dirname(HERE)
sys.path.insert(0, PACK)
sys.path.insert(0, os.path.join(PACK, "kubakub"))
sys.path.insert(0, os.path.join(PACK, "kubakub", "scene3d"))
import bridge  # noqa: E402
import scene_ids as si  # noqa: E402
import scene_view as sv  # noqa: E402
import idmaps as im  # noqa: E402

failures = []


def check(name, cond, detail=""):
    print(("ok   " if cond else "FAIL ") + name + (f"  ({detail})" if detail and not cond else ""))
    if not cond:
        failures.append(name)


# --------------------------------------------------------------------------
# synthetic scene: axis-aligned boxes, facade in the XZ plane, camera looking +Y (Blender front view)
# --------------------------------------------------------------------------

def box_faces(x0, x1, y0, y1, z0, z1, verts):
    """6 quads of a box; returns [(vertex ids, normal)]; shares vertices inside the box."""
    base = len(verts)
    for x in (x0, x1):
        for y in (y0, y1):
            for z in (z0, z1):
                verts.append((x, y, z))
    v = lambda i, j, k: base + i * 4 + j * 2 + k  # noqa: E731
    return [((v(0, 0, 0), v(1, 0, 0), v(1, 0, 1), v(0, 0, 1)), (0, -1, 0)),   # front (y0, faces -Y)
            ((v(0, 1, 0), v(0, 1, 1), v(1, 1, 1), v(1, 1, 0)), (0, 1, 0)),    # back
            ((v(0, 0, 0), v(0, 0, 1), v(0, 1, 1), v(0, 1, 0)), (-1, 0, 0)),   # left
            ((v(1, 0, 0), v(1, 1, 0), v(1, 1, 1), v(1, 0, 1)), (1, 0, 0)),    # right
            ((v(0, 0, 1), v(1, 0, 1), v(1, 1, 1), v(0, 1, 1)), (0, 0, 1)),    # top
            ((v(0, 0, 0), v(0, 1, 0), v(1, 1, 0), v(1, 0, 0)), (0, 0, -1))]   # bottom


def make_scene(folder, W=260, H=140):
    verts, faces, obj = [], [], []
    boxes = [  # (x0, x1, y0, y1, z0, z1, object)
        (0, 10, 0.0, 0.5, 0, 6, 0),        # wall, front at y = 0
        (3, 5, -0.4, 0.0, 3.0, 3.3, 0),    # ledge projecting 0.4 m, glued on the wall (shares no vertices)
        (7, 9, 0.3, 0.5, 2, 4, 0),         # recessed "window" block inside the wall's depth (front at y = 0.3)
        (11, 12, -1.0, 0.0, 0, 1, 1),      # separate box, other object
    ]
    for x0, x1, y0, y1, z0, z1, o in boxes:
        for f in box_faces(x0, x1, y0, y1, z0, z1, verts):
            faces.append(f)
            obj.append(o)
    V = np.array(verts, np.float32)
    normals = np.array([f[1] for f in faces], np.float32)
    loops = np.array([i for f in faces for i in f[0]], np.int32)
    centres = np.array([V[list(f[0])].mean(0) for f in faces], np.float32)
    # orthographic raster of the front: frontmost front-facing box face per pixel; the window block is
    # seen through a hole in the wall (the wall's front is cut out over x 7..9, z 2..4)
    xs = -0.5 + (np.arange(W) + 0.5) * 13.0 / W            # pixel centres of an ortho camera 13 x 7 m
    zs = 6.5 - (np.arange(H) + 0.5) * 7.0 / H
    X, Z = np.meshgrid(xs, zs)
    faceid = np.zeros((H, W), np.uint32)
    pos = np.zeros((H, W, 3), np.float32)
    nrm = np.zeros((H, W, 3), np.float32)
    best = np.full((H, W), np.inf)
    for bi, (x0, x1, y0, y1, z0, z1, o) in enumerate(boxes):
        inside = (X >= x0) & (X <= x1) & (Z >= z0) & (Z <= z1)
        if bi == 0:
            inside &= ~((X > 7) & (X < 9) & (Z > 2) & (Z < 4))
        fi = bi * 6                           # the box's front face
        m = inside & (y0 < best)
        best[m] = y0
        faceid[m] = fi + 1
        pos[m] = np.stack([X[m], np.full(m.sum(), y0), Z[m]], 1)
        nrm[m] = (0, -1, 0)
    os.makedirs(folder, exist_ok=True)
    np.save(os.path.join(folder, "faceid.npy"), faceid)
    np.save(os.path.join(folder, "position.npy"), pos)
    np.save(os.path.join(folder, "normal.npy"), nrm.astype(np.float16))
    np.savez_compressed(os.path.join(folder, "mesh.npz"), normal=normals, centre=centres,
                        area=np.ones(len(faces), np.float32), obj=np.array(obj, np.int32),
                        mat=np.zeros(len(faces), np.int32), col=np.zeros(len(faces), np.int32),
                        loop_total=np.full(len(faces), 4, np.int32), loop_vert=loops, vert=V)
    cam = np.eye(4)
    cam[:3, :3] = [[1, 0, 0], [0, 0, -1], [0, 1, 0]]      # rotation X +90: looks along +Y
    cam[:3, 3] = (6, -30, 3)
    info = {"file": "synthetic", "blender": "-", "width": W, "height": H, "faces": len(faces),
            "camera": {"name": "cam", "how": "test", "type": "ORTHO", "lens_mm": 50, "shift_x": 0, "shift_y": 0,
                       "matrix_world": cam.tolist(), "projection": [[2 / 13, 0, 0, 0], [0, 2 / 7, 0, 0],
                                                                    [0, 0, -2 / 99.9, -100.1 / 99.9], [0, 0, 0, 1]]},
            "cameras": ["cam"], "objects": ["facade", "kiosk"], "materials": [""], "collections": [""],
            "bbox_min": V.min(0).tolist(), "bbox_max": V.max(0).tolist(), "notes": []}
    json.dump(info, open(os.path.join(folder, "scene.json"), "w"))
    return len(faces)


tmp = tempfile.mkdtemp(prefix="kuba_scene3d_test_")
try:
    src = os.path.join(tmp, "export")
    F = make_scene(src)
    scene = si.load(src)
    mesh = scene["mesh"]

    pairs = si.adjacent_pairs(mesh, 1e-4)
    check("adjacency: each box face has 4 neighbours", len(pairs) == 4 * 12 * 4 // 2 // 1 or len(pairs) == 48,
          f"{len(pairs)} pairs")
    parts = si.loose_parts(mesh, 1e-4)
    check("loose parts: 4 boxes", len(np.unique(parts)) == 4, str(np.unique(parts)))
    planes = si.coplanar_patches(mesh, pairs)
    check("planes: box faces are not merged across 90 degrees", len(np.unique(planes)) == F)

    pt, nrm = si.main_plane(scene)
    check("main plane normal faces the camera (-Y)", np.allclose(nrm, [0, -1, 0], atol=1e-4), str(nrm))
    check("main plane is the wall front (y = 0)", abs(float(pt @ nrm)) < 0.02, str(pt))

    signed = (scene["position"] - pt) @ nrm
    lab, offs = si.shelf_labels(signed, scene["faceid"] > 0, min_share=0.001)
    check("shelves: wall 0, window -0.3, ledge +0.4, kiosk +1.0",
          np.allclose(sorted(offs), [-0.3, 0.0, 0.4, 1.0], atol=0.01), str(offs))

    m0, n0 = si.merge_small(lab, scene["position"], nrm, 0.5)
    check("min size 0.5 m: the 2 m ledge and the 2 m window stay", n0 == 0 and len(np.unique(m0[m0 >= 0])) == 4)
    m1, n1 = si.merge_small(lab, scene["position"], nrm, 2.5)
    left = sorted(offs[np.unique(m1[m1 >= 0])])
    check("min size 2.5 m: ledge and window merge into the wall, the lone kiosk stays (no neighbour)",
          np.allclose(left, [0.0, 1.0], atol=0.01), f"{left}, merged {n1}")

    fl = si.facing_labels(mesh, nrm)
    check("facing: front / back / top / underside / side per box", list(fl[:6]) == [0, 4, 3, 3, 1, 2], str(fl[:6]))

    depth, dist = si.depth_maps(scene)
    fg = scene["faceid"] > 0
    check("depth: wall at 30 m, ledge 29.6, kiosk 29", abs(depth[fg].max() - 30.3) < 0.01 and
          abs(depth[fg].min() - 29.0) < 0.01, f"{depth[fg].min()} {depth[fg].max()}")

    out = os.path.join(tmp, "ids")
    summary, previews, _, _ = si.build(src, out=out, passes=si.PASSES)
    check("build writes the passes", all(os.path.isfile(os.path.join(out, f"ids_{p}.png"))
                                         for p in ("shelves", "layers", "facing", "planes", "parts", "objects")))
    check("materials skipped (only one)", not os.path.isfile(os.path.join(out, "ids_materials.png")))
    leg = open(os.path.join(out, "ids_shelves.txt"), encoding="utf-8").read()
    check("shelf names carry the offset", "shelf_+0.00m" in leg and "shelf_-0.30m" in leg, leg[:200])

    # read back through the From ID Maps logic
    passes, ref = im.discover(out)
    check("From ID Maps finds the passes", {"shelves", "parts", "objects"} <= set(passes), str(list(passes)))
    for pname, expect in (("objects", 2), ("parts", 4)):
        masks, labels, size, notes = im.pass_masks(passes[pname])
        n = sum(1 for _, m in masks if m.sum() > 0)
        check(f"{pname}: {expect} masks decoded exactly", n == expect and labels is not None
              and (labels >= 0).sum() == fg.sum(), f"{n} masks, {notes}")
    r = im.build(out, regions_pass="shelves", tag_passes="facing objects", min_region_area=0)
    names = [g["name"] for g in r["atlas"]["regions"]]
    check("regions from the shelves pass", len(names) >= 4, str(names))

    # measure (3D-2)
    cam = sv.Camera(scene["info"])
    u, v, _ = cam.project(scene["position"])
    H, W = fg.shape
    uu, vv = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
    check("camera: world -> pixel lands on the pixel centres", np.abs(u[fg] - uu[fg]).max() < 1e-3
          and np.abs(v[fg] - vv[fg]).max() < 1e-3, f"{np.abs(u[fg] - uu[fg]).max()}")
    o, d = cam.ray(np.array([W / 2]), np.array([H / 2]))
    check("camera: centre ray looks along +Y", np.allclose(d[0], [0, 1, 0], atol=1e-6), str(d))
    fr = sv.wall_frame(scene["info"], pt, nrm, 0.0)
    check("frame on the wall: 13 x 7 m, bottom edge -0.5 m", abs(fr["width_m"] - 13) < 1e-6 and
          abs(fr["height_m"] - 7) < 1e-6 and abs(fr["bottom_m"] + 0.5) < 1e-6, str({k: fr[k] for k in ("width_m", "height_m", "bottom_m")}))
    maps = sv.measure_maps(scene, pt, nrm)
    check("measure: 50 mm per matrix pixel (13 m / 260 px)", abs(np.median(maps["pixel_mm"][fg]) - 50) < 1e-3,
          str(np.median(maps["pixel_mm"][fg])))
    check("measure: fronts hit square on, brightness 1", np.abs(maps["incidence_deg"][fg]).max() < 1e-3
          and abs(np.median(maps["brightness"][fg]) - 1) < 0.05)
    check("measure: offset map = shelves", np.allclose(sorted(np.unique(np.round(maps["offset_m"][fg], 2))),
                                                       [-0.3, 0.0, 0.4, 1.0]))
    stats = sv.region_stats(lab, maps, viewer_pos=(6, -15, 1.7), position=scene["position"])
    offs_by = sorted(e["offset_m"] for e in stats.values())
    check("region stats: one entry per shelf with its offset", np.allclose(offs_by, [-0.3, 0, 0.4, 1.0], atol=1e-3), str(offs_by))
    wall = [e for e in stats.values() if abs(e["offset_m"]) < 1e-3][0]
    check("region stats: wall area ~ its visible m2", 50 < wall["area_m2"] < 60, str(wall["area_m2"]))
    check("spots parse", sv.parse_spots("0, 20; -5 10 1.6") == [(0.0, 20.0, 1.7), (-5.0, 10.0, 1.6)])

    # preview (3D-4): seen from the projector itself = the matrix; an occluder casts a projection shadow
    rng = np.random.default_rng(0)
    mat = rng.random((H, W, 3)).astype(np.float32)
    rp = sv.reprojection(scene, scene)
    check("preview: from the projector every model pixel is lit", rp["lit"].sum() == fg.sum() and not rp["shadow"].any())
    img = sv.render_preview(mat, rp, np.zeros_like(mat), ambient=0.0, physical=0.0)
    check("preview: from the projector the matrix comes back", np.abs(img[fg] - mat[fg]).max() < 1e-4,
          f"{np.abs(img[fg] - mat[fg]).max()}")
    occl = dict(scene)
    occl["position"] = scene["position"].copy()
    block = (slice(20, 40), slice(20, 60))
    occl["position"][block] -= np.array([0, 2.0, 0], np.float32)       # something 2 m in front of the wall there
    rp2 = sv.reprojection(scene, occl)
    blk = np.zeros_like(fg); blk[block] = True
    check("preview: an occluder in front leaves a projection shadow", (~rp2["lit"][blk & fg]).all()
          and rp2["lit"][fg & ~blk].all() and rp2["shadow"][blk & fg].all())

    # walkthrough path and background
    keys = sv.parse_path("-5, 20; 0, 10, 1.6, 2, 3; 5, 20")
    check("path keys parse (eye default 1.7, look default nan)", keys.shape == (3, 5) and keys[0, 2] == 1.7
          and np.isnan(keys[0, 3]) and keys[1, 3] == 2.0)
    cams = sv.camera_path(fr, keys, 21)
    locs = np.array([c[0] for c in cams])
    check("path passes through its keys", np.allclose(locs[0], sv.spot_position(fr, -5, 20, 1.7)) and
          np.allclose(locs[10], sv.spot_position(fr, 0, 10, 1.6)) and np.allclose(locs[-1], sv.spot_position(fr, 5, 20, 1.7)))
    check("path is smooth (no jumps)", np.abs(np.diff(locs, axis=0)).max() < 2.0, str(np.abs(np.diff(locs, axis=0)).max()))
    check("look at the frame centre unless given", np.allclose(cams[0][1], fr["centre"]) and
          np.allclose(cams[10][1][2], 3.0))
    bg = sv.fit_background(np.ones((50, 200, 3), np.float32) * 0.5, 64, 64)
    check("background covers the view", bg.shape == (64, 64, 3) and np.allclose(bg, 0.5))
    img = sv.render_preview(mat, rp, np.zeros_like(mat), ambient=0.0, physical=0.0,
                            background=np.full_like(mat, 0.25))
    check("background only behind the building", np.allclose(img[~fg], 0.25) and np.abs(img[fg] - mat[fg]).max() < 1e-4)

    # relight helpers: light lines, facade -> world, mask -> emissive faces
    ls = sv.parse_lights(chr(10).join(["area -2 3 5 900 #ff8000 2 // warm", "point 1 1 1 100", "sun 30 20 3 #ffffff", "spot 0 2 4 500 #00ff00 30"]))
    check("light lines parse", [q["type"] for q in ls] == ["area", "point", "sun", "spot"] and ls[0]["size"] == 2
          and ls[3]["angle"] == 30 and np.allclose(ls[0]["color"], [1, 128 / 255, 0]))
    wl = sv.lights_to_world(ls, fr)
    exp = fr["centre"] + fr["ex"] * -2 + fr["normal"] * 5
    check("area light sits 5 m in front of the wall, 3 m up", np.allclose(wl[0]["location"][:2], exp[:2]) and abs(wl[0]["location"][2] - 3) < 1e-6
          and np.allclose(np.array(wl[0]["target"]) - np.array(wl[0]["location"]), -fr["normal"] * 5))
    check("sun shines towards the wall from the audience side", np.dot(wl[2]["direction"], fr["normal"]) < 0 and wl[2]["direction"][2] < 0)
    try:
        sv.parse_lights("lamp 1 2 3 4")
        check("unknown light type is refused", False)
    except ValueError:
        check("unknown light type is refused", True)
    fid = scene["faceid"]
    wallface = int(np.bincount(fid[fid > 0]).argmax())
    em = sv.emissive_faces(fid, (fid == wallface).astype(np.float32))
    check("a mask over one face makes exactly that face emissive", list(em) == [wallface - 1], str(em))
    check("an empty mask makes nothing emissive", len(sv.emissive_faces(fid, np.zeros(fid.shape, np.float32))) == 0)

    # the bridge
    check("cache key changes with the camera",
          bridge.cache_key(os.path.join(src, "scene.json"), "a", 0, 0, -1)
          != bridge.cache_key(os.path.join(src, "scene.json"), "b", 0, 0, -1))
    try:
        bridge.export(os.path.join(tmp, "x.c4d"), tmp)
        check("c4d is refused with a hint", False)
    except (ValueError, FileNotFoundError) as e:
        check("c4d is refused with a hint", "c4d" in str(e) or "not found" in str(e), str(e))

    try:
        exe = bridge.find_blender()
    except FileNotFoundError:
        exe = None
        print("skip Blender end-to-end (Blender not installed)")
    if exe:
        obj = os.path.join(tmp, "wall.obj")
        with open(obj, "w") as f:           # a wall 4 x 3 m with a box 0.5 m in front; OBJ Y-up -> Blender Z-up
            f.write("v 0 0 0\nv 4 0 0\nv 4 3 0\nv 0 3 0\nf 1 2 3 4\n"
                    "v 1 1 0.5\nv 2 1 0.5\nv 2 2 0.5\nv 1 2 0.5\nf 5 6 7 8\n")
        folder, cached, _ = bridge.export(obj, os.path.join(tmp, "cache"), width=160, height=120, blender=exe)
        info = json.load(open(os.path.join(folder, "scene.json")))
        fid = np.load(os.path.join(folder, "faceid.npy"))
        check("Blender: OBJ exported with an auto camera", info["camera"]["how"] == "auto front"
              and info["faces"] == 2 and fid.shape == (120, 160), json.dumps(info["camera"])[:200])
        check("Blender: both faces visible", set(np.unique(fid)) == {0, 1, 2}, str(np.unique(fid)))
        s2, _, _, d2 = si.build(folder, passes=("shelves",), out=os.path.join(tmp, "ids2"))
        check("Blender: shelves 0 and +0.5 m", np.allclose(sorted(s2["shelves_m"]), [0.0, 0.5], atol=0.02),
              str(s2.get("shelves_m")))
        folder2, cached2, _ = bridge.export(obj, os.path.join(tmp, "cache"), width=160, height=120, blender=exe)
        check("Blender: second call is cached", cached2 and folder2 == folder)
finally:
    shutil.rmtree(tmp, ignore_errors=True)

# architecture: a synthetic facade seen straight on (x right, z up, the wall faces -y towards the camera)
import architecture as ar  # noqa: E402
import cv2  # noqa: E402
px = 0.02                                          # 2 cm per pixel: 30 x 14 m facade = 1500 x 700 px
Wf, Hf = 1500, 700
xs = (np.arange(Wf) + 0.5) * px
zs = (Hf - np.arange(Hf) - 0.5) * px              # image row 0 = the top
X, Z = np.meshgrid(xs, zs)
off = np.zeros((Hf, Wf))                           # distance in front of the main wall (m)
off[:, (xs >= 10) & (xs < 20)] = 0.4               # a centre section standing 40 cm forward
belt = (Z >= 5.9) & (Z < 6.3)
off[belt] += 0.3                                   # a cornice belt across the whole facade at 6 m
for x0 in (2, 6, 12, 16, 22, 26):                  # two floors of windows, 1.2 x 2 m, recessed 30 cm
    for z0 in (2.0, 8.5):
        m = (X >= x0) & (X < x0 + 1.2) & (Z >= z0) & (Z < z0 + 2)
        off[m] -= 0.3
blind = (X >= 26) & (X < 27.2) & (Z >= 11.5) & (Z < 13.1)       # a blind window: a 10 cm frame, flat inside
frame = (X >= 25.9) & (X < 27.3) & (Z >= 11.4) & (Z < 13.2) & ~blind
off[frame] += 0.1
pos = np.stack([X, -off, Z], -1).astype(np.float32)
nrm_img = np.zeros_like(pos); nrm_img[..., 1] = -1
scene = {"position": pos, "normal": nrm_img, "faceid": np.ones((Hf, Wf), np.int32)}
r = ar.analyse(scene, np.zeros(3), np.array([0.0, -1.0, 0.0]), 0.0)
el, names = r["elements"]
win = (el == names.index("windows")).astype(np.uint8)
nwin = cv2.connectedComponents(win, connectivity=8)[0] - 1
check("architecture: 12 recessed + 1 blind window", nwin == 13, f"{nwin} {r['info']}")
check("architecture: left / centre / right at the wall steps", r["sections"][1] == ["left", "centre", "right"]
      and abs(r["info"]["sections_px"][1][0] - 500) <= 30 and abs(r["info"]["sections_px"][1][1] - 1000) <= 30, str(r["info"]))
check("architecture: cornice belt found", el[int((14 - 6.1) / px), 300] == names.index("cornices"))
fl, fn = r["floors"]
check("architecture: floors from the window rows", fn[:2] == ["ground_floor", "floor_1"]
      and fl[int((14 - 3) / px), 100] == 0 and fl[int((14 - 9.5) / px), 100] == 1, f"{fn} {r['info']}")
check("architecture: the wall stays wall", el[int((14 - 1) / px), 100] == names.index("wall"))

# fingerprint token of temp result folders: stable when made, changes once when it vanished
tmpd = os.path.join(tempfile.mkdtemp(), "made_later")
t_first = bridge.folder_token(tmpd)
os.makedirs(tmpd)
check("folder token: 'not made yet' -> 'made' keeps the cache", bridge.folder_token(tmpd) == t_first)
shutil.rmtree(tmpd)
t_gone = bridge.folder_token(tmpd)
check("folder token: a vanished folder runs the node again", t_gone != t_first)
check("folder token: ... once, not on every queue", bridge.folder_token(tmpd) == t_gone)
os.makedirs(tmpd)
check("folder token: made again keeps the new token", bridge.folder_token(tmpd) == t_gone)
shutil.rmtree(os.path.dirname(tmpd))

print()
if failures:
    print(f"{len(failures)} FAILED: {', '.join(failures)}")
    sys.exit(1)
print("all scene3d tests passed")
