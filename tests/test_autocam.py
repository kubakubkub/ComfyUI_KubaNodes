"""
Model free test of the projector placement (kubakub/scene3d/autocam.py) and the blend masks of several projectors
(scene_view.projector_blend): which side of a model is the facade, where the camera stands for a file without one,
a projector placed in metres from the wall (lens shift / tilt / throw ratio), and soft-edge masks that add up to 1.
With Blender installed the sample model is also rendered from two projectors through the real bridge.

Run from the ComfyUI_windows_portable folder:
    python_embeded\\python.exe ComfyUI\\custom_nodes\\ComfyUI_KubaNodes\\tests\\test_autocam.py
"""

import json
import math
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
import autocam as ac  # noqa: E402
import bridge  # noqa: E402
import scene_ids as si  # noqa: E402
import scene_view as sv  # noqa: E402
from kubakub import sample_model as sm  # noqa: E402

failures = []


def check(name, cond, detail=""):
    print(("ok   " if cond else "FAIL ") + name + (f"  ({detail})" if detail and not cond else ""))
    if not cond:
        failures.append(name)


def box(x0, x1, y0, y1, z0, z1):
    """A box as (verts, normals, areas, centres) of its 6 faces, world axes (z up)."""
    v = np.array([[x, y, z] for x in (x0, x1) for y in (y0, y1) for z in (z0, z1)], float)
    dx, dy, dz = x1 - x0, y1 - y0, z1 - z0
    n = np.array([[-1, 0, 0], [1, 0, 0], [0, -1, 0], [0, 1, 0], [0, 0, -1], [0, 0, 1]], float)
    a = np.array([dy * dz, dy * dz, dx * dz, dx * dz, dx * dy, dx * dy], float)
    c = np.array([[x0, (y0 + y1) / 2, (z0 + z1) / 2], [x1, (y0 + y1) / 2, (z0 + z1) / 2],
                  [(x0 + x1) / 2, y0, (z0 + z1) / 2], [(x0 + x1) / 2, y1, (z0 + z1) / 2],
                  [(x0 + x1) / 2, (y0 + y1) / 2, z0], [(x0 + x1) / 2, (y0 + y1) / 2, z1]], float)
    return v, n, a, c


def rot_z(deg):
    t = math.radians(deg)
    return np.array([[math.cos(t), -math.sin(t), 0], [math.sin(t), math.cos(t), 0], [0, 0, 1]])


def join(parts, R=None):
    v, n, a, c = (np.concatenate([p[k] for p in parts]) for k in range(4))
    if R is not None:
        v, n, c = v @ R.T, n @ R.T, c @ R.T
    return v, n, a, c


# a facade slab 30 x 12 m, 0.6 m thick, front at y = 0, made of 40 stones (their sides add a lot of side area)
stones = [box(x, x + 3, 0.0, 0.6, z, z + 3) for x in range(0, 30, 3) for z in range(0, 12, 3)]
slab = join(stones)
n = ac.facade_direction(slab[1], slab[2], "auto", slab[0])
check("a facade slab: the wide side is the facade, the front (-y) wins over the back", np.allclose(n, [0, -1, 0], atol=1e-6), str(n))
turned = join(stones, rot_z(90))                        # the same slab facing -x / +x
n = ac.facade_direction(turned[1], turned[2], "auto", turned[0])
check("the slab turned 90 degrees: the facade is on the x axis", abs(abs(n[0]) - 1) < 1e-6 and abs(n[1]) < 1e-6, str(n))
skew = join(stones, rot_z(30))
n = ac.facade_direction(skew[1], skew[2], "auto", skew[0])
want = rot_z(30) @ np.array([0, -1.0, 0])
check("the slab turned 30 degrees: the direction follows the wall, not an axis", np.allclose(n, want, atol=0.02), f"{n} vs {want}")
check("side = back (+y) is taken as said", np.allclose(ac.facade_direction(slab[1], slab[2], "back (+y)"), [0, 1, 0]))

f = ac.facade_frame(*slab)
check("frame: wall at y = 0, 30 m across, 12 m up, x to the right seen from the audience",
      abs(f["wall"]) < 1e-6 and abs(f["u1"] - f["u0"] - 30) < 1e-6 and abs(f["z1"] - 12) < 1e-6 and np.allclose(f["ex"], [1, 0, 0]),
      str(f))

W, H = 1920, 1080


def frame_on_wall(cam, wall=0.0):
    """The camera's picture on the plane y = wall: (centre x, centre z, width, height). Camera looks along -back."""
    m = np.asarray(cam["matrix"], float)
    right, up, back, loc = m[:3, 0], m[:3, 1], m[:3, 2], m[:3, 3]
    long_side = 36.0
    half_w = (18.0 if W >= H else 18.0 * W / H) / cam["lens"]
    half_h = half_w * H / W
    pts = []
    for sx, sy in ((-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)):
        d = -back + right * (sx * half_w + cam["shift_x"] * long_side / cam["lens"]) \
            + up * (sy * half_h + cam["shift_y"] * long_side / cam["lens"])
        t = (wall - loc[1]) / d[1]
        pts.append(loc + d * t)
    left, rgt, bot, top, mid = pts
    return mid[0], mid[2], float(np.linalg.norm(rgt - left)), float(np.linalg.norm(top - bot))


c = ac.front_camera(*slab, W, H)
cx, cz, fw, fh = frame_on_wall(c)
check("no camera in the file: a level camera in front of the middle, the whole model in the picture",
      abs(cx - 15) < 1e-6 and abs(cz - 6) < 1e-6 and fw >= 30 and fh >= 12 and c["shift_x"] == 0 and abs(c["matrix"][2, 3] - 6) < 1e-6,
      f"{cx:.2f} {cz:.2f} {fw:.2f} x {fh:.2f}")
old = 1.1 * max(30 / 2 / 0.5, 12 / 2 / (0.5 * H / W)) + 0.3            # the placement before autocam, for a model facing -y
check("... at the same distance as before for a model that faces -y", abs(-c["matrix"][1, 3] + 0.3 - old) < 1e-6,
      f"{-c['matrix'][1, 3] + 0.3:.3f} vs {old:.3f}")

p = ac.projector_camera(*slab, W, H, {"distance_m": 25, "height_m": 1.5})
cx, cz, fw, fh = frame_on_wall(p)
loc = p["matrix"][:3, 3]
check("projector 25 m from the wall, 1.5 m up, in front of the middle", np.allclose(loc, [15, -25, 1.5], atol=1e-6), str(loc))
check("lens shift: square to the wall, the picture centred on the model and just covering it (5 % margin)",
      np.allclose(p["matrix"][:3, 2], [0, -1, 0]) and abs(cx - 15) < 1e-6 and abs(cz - 6) < 1e-6 and abs(fw - 33) < 1e-6,
      f"{cx:.2f} {cz:.2f} {fw:.2f} x {fh:.2f}, shift {p['shift_x']:.3f} {p['shift_y']:.3f}")
p = ac.projector_camera(*slab, W, H, {"distance_m": 25, "height_m": 20, "offset_m": -10, "throw_ratio": 1.0})
cx, cz, fw, fh = frame_on_wall(p)
check("throw ratio 1 at 25 m = a 25 m picture, still centred from a tower off to the left",
      abs(fw - 25) < 1e-6 and abs(cx - 15) < 1e-6 and abs(cz - 6) < 1e-6 and np.allclose(p["matrix"][:3, 3], [5, -25, 20]),
      f"{cx:.2f} {cz:.2f} {fw:.2f}")
check("... and the report says the picture does not cover the model", "covers about" in p["note"], p["note"])
p = ac.projector_camera(*slab, W, H, {"throw_ratio": 2.0})
check("distance 0: as far as the throw ratio needs (2 x the 33 m picture)", abs(p["distance_m"] - 66) < 1e-6, str(p["distance_m"]))
p = ac.projector_camera(*slab, W, H, {"distance_m": 25, "height_m": 1.5, "offset_m": 12, "aim": "tilt"})
m = p["matrix"]
aim_at = np.array([15, 0, 6.0]) - m[:3, 3]
check("tilt: the projector looks at the middle of the model, no shift", p["shift_x"] == 0 and p["shift_y"] == 0
      and np.allclose(-m[:3, 2], aim_at / np.linalg.norm(aim_at), atol=1e-6) and abs(np.linalg.det(m[:3, :3]) - 1) < 1e-6)
p = ac.projector_camera(*slab, W, H, {"distance_m": 25, "aim": "straight", "height_m": 3})
cx, cz, fw, fh = frame_on_wall(p)
check("straight: the picture sits in front of the projector", abs(cx - 15) < 1e-6 and abs(cz - 3) < 1e-6, f"{cx} {cz}")
p = ac.projector_camera(*slab, W, H, {"distance_m": 25, "turn_deg": 30})
loc = p["matrix"][:3, 3]
check("turn 30 degrees: the projector stands to the right, 25 m from the middle of the wall",
      abs(np.linalg.norm(loc[:2] - [15, 0]) - 25) < 1e-6 and loc[0] > 15 and loc[1] < 0, str(loc))
p = ac.projector_camera(*turned, W, H, {"distance_m": 25, "height_m": 1.5})
check("a model facing another way: the projector stands in front of that side",
      abs(abs(p["matrix"][0, 3]) - 25) < 0.7 and abs(p["matrix"][1, 3] - 15) < 1e-6, str(p["matrix"][:3, 3]))
for bad in ({"aim": "sideways"}, {"side": "up"}, {"distance_m": -1}):
    try:
        ac.spec(bad)
        check(f"a wrong setting is refused: {bad}", False)
    except ValueError:
        check(f"a wrong setting is refused: {bad}", True)
check("unknown keys are dropped, defaults filled", ac.spec({"zoom": 3}) == ac.DEFAULTS)


# --------------------------------------------------------------------------
# blend masks: two ortho projectors on a flat wall, their pictures overlap by 4 m
# --------------------------------------------------------------------------

def wall_view(x0, x1, px=20):
    """A flat wall at y = 0 seen by an ortho camera covering x0..x1, z 0..6 (px pixels per metre)."""
    Wp, Hp = int((x1 - x0) * px), 6 * px
    xs = x0 + (np.arange(Wp) + 0.5) / px
    zs = 6 - (np.arange(Hp) + 0.5) / px
    X, Z = np.meshgrid(xs, zs)
    pos = np.stack([X, np.zeros_like(X), Z], -1).astype(np.float32)
    nrm = np.zeros_like(pos)
    nrm[..., 1] = -1
    cam = np.eye(4)
    cam[:3, :3] = [[1, 0, 0], [0, 0, -1], [0, 1, 0]]
    cam[:3, 3] = ((x0 + x1) / 2, -30, 3)
    info = {"width": Wp, "height": Hp, "camera": {"type": "ORTHO", "matrix_world": cam.tolist(),
                                                  "projection": [[2 / (x1 - x0), 0, 0, 0], [0, 2 / 6, 0, 0],
                                                                 [0, 0, -2 / 99.9, -100.1 / 99.9], [0, 0, 0, 1]]}}
    return {"info": info, "position": pos, "normal": nrm, "faceid": np.ones((Hp, Wp), np.uint32)}


A, B = wall_view(0, 12), wall_view(8, 20)
ra, rb = sv.projector_blend([A, B], ramp=0.5, gamma=1.0)       # ramp = 3 m of the 6 m short side
row = 60                                                        # the middle row: far from the top and bottom borders
xa = (np.arange(240) + 0.5) / 20                                # wall x of A's pixels
check("alone: the full picture, up to its own border", np.allclose(ra["mask"][row, xa < 8], 1.0)
      and not ra["shared"][row, xa < 8].any(), str(ra["mask"][row, :5]))
check("shared: the 4 m where both pictures are", ra["shared"][row, xa > 8].all() and rb["shared"][row, :80].all()
      and ra["counts"]["shared"] == 80 * 120)
sum_lin = ra["mask"][row, 160:] + rb["mask"][row, :80]
check("in the overlap the two shares add up to 1 (linear light)", np.allclose(sum_lin, 1.0, atol=1e-5), str(sum_lin[:4]))
check("... and each one falls to 0 at its own border", ra["mask"][row, -1] < 0.02 and rb["mask"][row, 0] < 0.02
      and ra["mask"][row, 160] > 0.98, f"{ra['mask'][row, -1]:.3f} {ra['mask'][row, 160]:.3f}")
ga, gb = sv.projector_blend([A, B], ramp=0.5, gamma=2.2)
check("gamma 2.2: the light still adds up to 1", np.allclose(ga["mask"][row, 160:] ** 2.2 + gb["mask"][row, :80] ** 2.2, 1.0, atol=1e-4))
far = wall_view(30, 40)
r1, r2 = sv.projector_blend([A, far], ramp=0.2)
check("pictures that do not meet: every mask is 1, nothing shared", np.allclose(r1["mask"], 1) and r1["counts"]["shared"] == 0)
r3 = sv.projector_blend([A, B, wall_view(4, 16)], ramp=0.5, gamma=1.0)
s3 = r3[0]["mask"][row, 170] + r3[1]["mask"][row, 10] + r3[2]["mask"][row, 90]       # wall x = 8.525 in all three
check("three projectors on one spot add up to 1 as well", abs(s3 - 1) < 1e-4, str(s3))

# --------------------------------------------------------------------------
# the real thing: the sample model from two projectors through Blender
# --------------------------------------------------------------------------
check("cache key: a projector is another render of the same file",
      bridge.cache_key(__file__, "", 0, 0, -1) != bridge.cache_key(__file__, "", 0, 0, -1, projector={"distance_m": 20})
      and bridge.cache_key(__file__, "", 0, 0, -1) == bridge.cache_key(__file__, "", 0, 0, -1, projector=None))
try:
    exe = bridge.find_blender()
except FileNotFoundError:
    exe = None
    print("skip Blender end-to-end (Blender not installed)")
if exe:
    tmp = tempfile.mkdtemp(prefix="kuba_autocam_test_")
    try:
        obj = os.path.join(tmp, "sample.obj")
        info = sm.write_obj(obj, floors=2, bays=4, width_m=16.0, height_m=12.0)
        cache = os.path.join(tmp, "cache")
        folder, _, _ = bridge.export(obj, cache, width=480, height=270, blender=exe)
        sj = json.load(open(os.path.join(folder, "scene.json")))
        fid = np.load(os.path.join(folder, "faceid.npy"))
        cols = np.where((fid > 0).any(0))[0]
        check("Blender: the sample model without a camera is framed from the front",
              sj["camera"]["how"] == "auto front" and (fid > 0).mean() > 0.3 and cols[0] > 5 and cols[-1] < 474,
              f"{sj['camera']['how']} {(fid > 0).mean():.2f} {cols[0]} {cols[-1]}")
        left = ac.spec({"distance_m": 20, "offset_m": -4, "height_m": 2, "throw_ratio": 1.6, "name": "left"})
        right = ac.spec({"distance_m": 20, "offset_m": 4, "height_m": 2, "throw_ratio": 1.6, "name": "right", "aim": "straight"})
        f1, _, _ = bridge.export(obj, cache, width=480, height=270, blender=exe, projector=left)
        s1 = si.load(f1)
        pt, nrm = si.main_plane(s1)
        fr = sv.wall_frame(s1["info"], pt, nrm, 0.0)
        check("Blender: the projector stands 20 m from the wall, its picture is 12.5 m wide (throw 1.6)",
              s1["info"]["camera"]["how"] == "projector" and abs(fr["projector_distance_m"] - 20) < 0.1
              and abs(fr["width_m"] - 12.5) < 0.1, f"{fr['projector_distance_m']:.2f} m, {fr['width_m']:.2f} m")
        check("Blender: lens shift centres the picture on the model", abs(float(fr["centre"][0])) < 0.1
              and abs(float(fr["centre"][2]) - info["height_m"] / 2) < 0.1, str(fr["centre"]))
        left["aim"] = "straight"
        f1, _, _ = bridge.export(obj, cache, width=480, height=270, blender=exe, projector=left)
        f2, _, _ = bridge.export(obj, cache, width=480, height=270, blender=exe, projector=right)
        check("Blender: each projector has its own cache folder", len({folder, f1, f2}) == 3)
        res = sv.projector_blend([si.load(f1), si.load(f2)], ramp=0.1, gamma=1.0)
        sh = [r["counts"]["shared"] / max(r["counts"]["model"], 1) for r in res]
        check("Blender: two projectors 8 m apart share about a third of their pictures (4.5 of 12.5 m)",
              all(0.2 < x < 0.5 for x in sh), str(sh))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

print()
if failures:
    print(f"{len(failures)} FAILED: {', '.join(failures)}")
    sys.exit(1)
print("all autocam tests passed")
