"""
Model free test for kubakub scene surroundings (kubakub/scene3d/surroundings.py). No download: the map answer is
written by hand in the form the OpenStreetMap server gives (a way with a height, one with levels, one with nothing,
a courtyard building as a multipolygon in two pieces, a building part). If Blender is installed, the sample square
is also built next to a tiny OBJ through the real bridge (a few seconds).

Run from the ComfyUI_windows_portable folder:
    python_embeded\\python.exe ComfyUI\\custom_nodes\\ComfyUI_KubaNodes\\tests\\test_surroundings.py
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
import surroundings as sr  # noqa: E402

failures = []


def check(name, cond, detail=""):
    print(("ok   " if cond else "FAIL ") + name + (f"  ({detail})" if detail and not cond else ""))
    if not cond:
        failures.append(name)


# --------------------------------------------------------------------------
# the location field
# --------------------------------------------------------------------------
check("location: 'lat, lon'", sr.parse_location(" 48.85837, 2.29448 ") == (48.85837, 2.29448))
check("location: a map link", sr.parse_location("https://maps.example/place/x/@48.85837,2.29448,17z") == (48.85837, 2.29448))
check("location: empty is the sample", sr.parse_location("  ") is None)
for bad in ("somewhere", "120.5, 2.2"):
    try:
        sr.parse_location(bad)
        check(f"location: '{bad}' is refused", False)
    except ValueError:
        check(f"location: '{bad}' is refused", True)

# --------------------------------------------------------------------------
# a hand-written map answer around (50, 10)
# --------------------------------------------------------------------------
LAT, LON = 50.0, 10.0
MY, MX = sr.metres_per_degree(LAT)
check("metres per degree at 50 N", abs(MY - 111229) < 5 and abs(MX - 71696) < 5, f"{MY:.0f} {MX:.0f}")


def geom(pts):
    return [{"lat": LAT + y / MY, "lon": LON + x / MX} for x, y in pts]


def way(i, pts, tags, closed=True):
    return {"type": "way", "id": i, "geometry": geom(pts + pts[:1] if closed else pts), "tags": tags}


OWN = [(-12, 0.4), (-2, 0), (2, 0), (12, -0.4), (12, 14), (-12, 14)]       # the facade in three pieces, facing south
data = {"elements": [
    way(1, OWN, {"building": "yes", "height": "21 m"}),
    way(2, [(14, 0), (30, 0), (30, 12), (14, 12)], {"building": "apartments", "building:levels": "5"}),
    way(3, [(-40, -60), (-20, -60), (-20, -45), (-40, -45)], {"building": "yes"}),
    way(4, [(-6, 4), (0, 4), (0, 10), (-6, 10)], {"building:part": "yes", "height": "30", "min_height": "21"}),
    way(5, [(60, 60), (61, 60), (61, 61)], {"highway": "service"}),
    {"type": "relation", "id": 9, "tags": {"type": "multipolygon", "building": "yes", "height": "40'"}, "members": [
        {"type": "way", "role": "outer", "geometry": geom([(40, -60), (70, -60), (70, -30)])},
        {"type": "way", "role": "outer", "geometry": geom([(40, -60), (40, -30), (70, -30)])},
        {"type": "way", "role": "inner", "geometry": geom([(50, -50), (60, -50), (60, -40), (50, -40), (50, -50)])}]},
]}
B = sr.buildings_from_osm(data, LAT, LON)
by = {b["id"]: b for b in B}
check("map: 5 buildings, the road is not one", len(B) == 5 and "way/5" not in by, str(sorted(by)))
check("map: metres around the place", np.allclose(by["way/2"]["rings"][0], [(14, 0), (30, 0), (30, 12), (14, 12)], atol=0.01))
check("map: height in metres and in feet", by["way/1"]["height"] == 21 and abs(by["relation/9"]["height"] - 12.192) < 1e-6)
check("map: levels x 3 m", by["way/2"]["height"] == 15 and by["way/2"]["how"] == "levels")
check("map: a part starts at its min_height", by["way/4"]["part"] and by["way/4"]["base"] == 21)
r9 = by["relation/9"]["rings"]
check("map: the multipolygon's two pieces are one ring with its courtyard",
      len(r9) == 2 and len(r9[0]) == 4 and abs(abs(sr._area(r9[0])) - 900) < 1 and abs(abs(sr._area(r9[1])) - 100) < 1,
      str([len(r) for r in r9]))
stats = sr.fill_heights(B)
check("heights: the unknown one gets the median of the tagged", by["way/3"]["height"] == 15.0 and stats["guessed"] == 1
      and stats["height"] == 3 and stats["levels"] == 1, str(stats))

own = sr.own_building(B, (1.0, -3.0))                    # the place: 3 m in front of the wall
check("own building: the footprint at the place", B[own["index"]]["id"] == "way/1" and abs(own["distance_m"] - 3) < 0.05)
check("own building: its wall faces south, from all three pieces", abs(own["facing_deg"] - 180) < 1.0, f"{own['facing_deg']:.2f}")
check("own building: the place moved onto the wall", np.allclose(own["point"], (1.0, 0.0), atol=0.01), str(own["point"]))
inside_own = sr.own_building(B, (0.0, 13.0))             # a place inside, near the back wall
check("own building: a place inside it faces out of the nearest wall", abs(inside_own["facing_deg"]) < 1.0
      or abs(inside_own["facing_deg"] - 360) < 1.0, f"{inside_own['facing_deg']:.2f}")
check("own building: nothing within 30 m is none", sr.own_building(B, (0.0, 200.0)) is None)
rest = sr.without_own(B, own)
check("own building: it leaves with the part on it", sorted(b["id"] for b in rest) == ["relation/9", "way/2", "way/3"])
check("compass words", sr.compass(180) == "south" and sr.compass(350) == "north" and sr.compass(100) == "east")

# --------------------------------------------------------------------------
# onto the model: a facade in the plane x = 5 that faces +x, 20 m wide along y, its feet at z = 2
# --------------------------------------------------------------------------
V = np.array([[5, -10, 2], [5, 10, 2], [5, 10, 18], [5, -10, 18], [-3, -10, 2], [-3, 10, 2]], float)
anchor, nh, ex = sr.facade_anchor(V, np.array([5.0, 0, 0]), np.array([1.0, 0, 0]), 2.0)
check("anchor: the middle of the wall on the ground", np.allclose(anchor, (5, 0, 2)) and np.allclose(nh, (1, 0, 0))
      and np.allclose(ex, (0, 1, 0)), f"{anchor} {nh} {ex}")
M = np.array(sr.placement(anchor, nh, ex, own["facing_deg"], own["point"]))
to = lambda x, y, z=0.0: (M @ np.array([x, y, z, 1.0]))[:3]  # noqa: E731
check("placement: the place on the wall lands on the anchor", np.allclose(to(1.0, 0.0), anchor, atol=0.02), str(to(1.0, 0.0)))
check("placement: south of the wall is in front of the facade", to(1.0, -10.0)[0] > anchor[0] + 9.9, str(to(1.0, -10.0)))
check("placement: east of a south wall is to the audience's right",
      np.allclose(to(11.0, 0.0), anchor + ex * 10, atol=0.3), str(to(11.0, 0.0)))
check("placement: no mirror, heights stay", abs(np.linalg.det(M[:3, :3]) - 1) < 1e-9 and abs(to(0, 0, 7)[2] - 9) < 1e-9)
M2 = np.array(sr.placement(anchor, nh, ex, own["facing_deg"], own["point"], along_m=2, out_m=3, lift_m=1))
check("placement: along / out / up", np.allclose((M2 - M)[:3, 3], ex * 2 + nh * 3 + [0, 0, 1]))
N = np.array(sr.nudge(anchor, nh, ex, turn_deg=90))
check("own file: a turn keeps the anchor", np.allclose((N @ np.append(anchor, 1))[:3], anchor))
check("own file: no nudge is no change", np.allclose(sr.nudge(anchor, nh, ex), np.eye(4)))
try:
    sr.facade_anchor(V, np.zeros(3), np.array([0.0, 0, 1]), 0.0)
    check("anchor: a floor is refused", False)
except ValueError:
    check("anchor: a floor is refused", True)

# --------------------------------------------------------------------------
# real ground: a slope that rises 1 m every 10 m to the north, 100 m above the sea at the place
# --------------------------------------------------------------------------
tx0, ty0 = sr._tile(0.0, 0.0, 1)
check("terrain tiles: the equator and Greenwich are the middle of the map", abs(tx0 - 1) < 1e-9 and abs(ty0 - 1) < 1e-9)
tx1, ty1 = sr._tile(48.85837, 2.29448, 15)
check("terrain tiles: a known place at zoom 15", (int(tx1), int(ty1)) == (16592, 11272), f"{tx1:.2f} {ty1:.2f}")
ax = (np.arange(41) - 20) * 5.0
T = {"heights": (100.0 + ax[:, None] / 10.0 + 0 * ax[None, :]).astype(np.float32), "step": 5.0, "radius": 100.0}
check("terrain: read between the samples", np.allclose(sr.terrain_at(T, [(0, 0), (7.5, 12.5), (-33.0, -60.0)]), [100.0, 101.25, 94.0], atol=1e-4),
      str(sr.terrain_at(T, [(0, 0), (7.5, 12.5), (-33.0, -60.0)])))
check("terrain: a grid kept for a spot 10 m east answers the same ground",
      np.allclose(sr.terrain_at(dict(T, offset=(10.0, 0.0)), [(7.5, 12.5)]), [101.25], atol=1e-4))
Bt = sr.drape([{"rings": [np.array([[0, 20], [10, 20], [10, 40], [0, 40]], float)], "height": 12.0, "base": 0.0}], T, 100.0)
check("terrain: a house on the slope stands on its mean ground, its foot under its lowest corner",
      abs(Bt[0]["z"] - 3.0) < 1e-4 and abs(Bt[0]["foot"] - 1.7) < 1e-4, f"{Bt[0]['z']} {Bt[0]['foot']}")
check("terrain: inside means under its roof on the slope", sr.stands_inside(Bt, np.eye(4), (5.0, 30.0, 14.0))
      and not sr.stands_inside(Bt, np.eye(4), (5.0, 30.0, 16.0)))
G = sr.ground_grid(T, 100.0, 50.0)
check("terrain: the ground grid is cut to the radius and counted from the facade's level",
      len(G["x"]) == 23 and abs(G["x"][0] + 55) < 1e-6 and G["z"].shape == (23, 23) and abs(float(G["z"][11, 11])) < 1e-5
      and abs(float(G["z"][-1, 0]) - 5.5) < 1e-4, f"{len(G['x'])} {G['x'][0]} {G['z'][-1, 0]}")
Tl, z0 = sr.level(T, (0.0, 0.0), 20.0)
hl = sr.terrain_at(Tl, [(0, 0), (0, 15), (0, 35), (0, 60), (0, 90)])
check("terrain: level around the facade, the real slope beyond, a ramp between", z0 == 100.0 and np.allclose(hl[:2], 100.0, atol=1e-4)
      and 100.0 < hl[2] < 103.5 and np.allclose(hl[3:], [106.0, 109.0], atol=1e-3) and sr.level(T, (0, 0), 0.0)[0] is T, str(hl))
doc_t = sr.street_doc(Bt)
check("terrain: the footprint file carries the ground under each house", doc_t[0]["z"] == 3.0 and doc_t[0]["foot"] == 1.7
      and "z" not in sr.street_doc([{"rings": Bt[0]["rings"], "height": 5.0}])[0])

# --------------------------------------------------------------------------
# the sample square, the footprint file, the plan
# --------------------------------------------------------------------------
S, facing = sr.sample(24.0)
check("sample: buildings with heights, one courtyard", len(S) >= 10 and all(b["height"] > 5 for b in S)
      and sum(len(b["rings"]) > 1 for b in S) == 1 and facing == 180.0)
front = [(x, y) for x in np.linspace(-12, 12, 13) for y in np.linspace(-65, -1, 33)]
check("sample: the square in front of the facade is free", not any(sr.inside(p, b["rings"][0]) for p in front for b in S))
Mi = np.eye(4)
spot = sr.free_spot(S, Mi, [(-20.0, 5.0, 1.7), (-71.0, -5.0, 1.7), (0.0, -30.0, 1.7)])
check("street view: a spot inside a building is passed over, a courtyard is free", np.allclose(spot, (-71.0, -5.0, 1.7)), str(spot))
check("inside a building: within the walls and under the roof only", sr.stands_inside(S, Mi, (-20.0, 5.0, 1.7))
      and not sr.stands_inside(S, Mi, (-20.0, 5.0, 40.0)) and not sr.stands_inside(S, Mi, (0.0, -30.0, 1.7))
      and not sr.stands_inside(S, Mi, (-71.0, -5.0, 1.7)) and sr.stands_inside(S, Mi, (-79.0, -5.0, 1.7)))
tmp = tempfile.mkdtemp(prefix="kuba_sur_")
try:
    fp = sr.write(os.path.join(tmp, "s", "street.json"), S, {"source": "sample"})
    doc = json.load(open(fp, encoding="utf-8"))
    check("file: footprints as lists", len(doc["buildings"]) == len(S) and doc["buildings"][0]["height"] == 15.0
          and doc["source"] == "sample")
    with open(os.path.join(tmp, "osm_50.00000_10.00000_250.json"), "w", encoding="utf-8") as f:
        json.dump(data, f)
    got, fresh, late = sr.fetch(tmp, LAT, LON, 250, allow_download=False)
    check("fetch: a place on disk is not downloaded again", got == data and fresh is False and late == "")
    check("fetch: a smaller radius, or a spot 10 m away, comes from the same file",
          sr.fetch(tmp, LAT, LON, 120, allow_download=False)[0] == data
          and sr.fetch(tmp, LAT + 10 / MY, LON, 230, allow_download=False)[0] == data)
    try:
        sr.fetch(tmp, LAT, LON, 400, allow_download=False)
        check("fetch: a larger radius needs a download", False)
    except ConnectionError:
        check("fetch: a larger radius needs a download", True)
    real_download = sr.download

    def busy(*a, **k):
        raise ConnectionError("OpenStreetMap download failed (HTTP Error 504: Gateway Timeout). Queue again.")

    sr.download = busy
    try:
        got, fresh, late = sr.fetch(tmp, LAT, LON, 400)
        check("fetch: a busy map server falls back to what is on disk, and says so", got == data and not fresh
              and "250 m" in late and "400 m" in late and "504" in late, late)
        try:
            sr.fetch(tmp, LAT + 1, LON, 400)
            check("fetch: a busy server and nothing on disk is the error", False)
        except ConnectionError:
            check("fetch: a busy server and nothing on disk is the error", True)
    finally:
        sr.download = real_download
    check("within: only buildings with a corner inside the radius", sorted(b["id"] for b in sr.within(B, 35.0)) == ["way/1", "way/2", "way/4"],
          str(sorted(b["id"] for b in sr.within(B, 35.0))))
    try:
        sr.fetch(tmp, LAT, LON + 1, 250, allow_download=False)
        check("fetch: downloads switched off is an error for a new place", False)
    except ConnectionError:
        check("fetch: downloads switched off is an error for a new place", True)
    Ms = sr.placement(anchor, nh, ex, facing)
    img = sr.plan(S, Ms, anchor, nh, ex, V, 100.0, size=256, projector=anchor + nh * 40)
    check("plan: buildings and the model drawn", img.shape == (256, 256, 3) and (img[..., 0] > 0.9).any()
          and (np.abs(img[..., 0] - img[..., 2]) < 0.01).mean() > 0.5 and (img.mean(-1) > 0.2).mean() > 0.03)
    check("plan: the projector is below the model (audience at the bottom)", img[128 + int(40 * 1.28), 128, 0] > 0.9)

    back = sr.read(fp)
    check("file: read back as buildings", len(back) == len(S) and np.allclose(back[0]["rings"][0], S[0]["rings"][0])
          and sr.read(os.path.join(tmp, "x.obj")) == [])
    fr_t = {"centre": anchor + [0, 0, 8], "ex": ex, "normal": nh, "ground_z": 2.0}
    world = sv.lights_to_world(sv.parse_lights("point -20 6 30 500 #ff0000\narea 20 6 10 500 #00ff00 4\nsun 30 40 3"), fr_t)
    lp = sr.plan_lights(sr.plan(S, Ms, anchor, nh, ex, V, 100.0, size=256, viewer=anchor + nh * 60), world, anchor, nh, ex, 100.0)
    red = (lp[..., 0] > 0.8) & (lp[..., 1] < 0.3)
    green = (lp[..., 1] > 0.8) & (lp[..., 0] < 0.3)
    ry, rx = np.argwhere(red).mean(0)
    gy, gx = np.argwhere(green).mean(0)
    check("light plan: a lamp 20 m left and 30 m out is drawn left of and below the model",
          abs(rx - (128 - 20 * 1.28)) < 3 and abs(ry - (128 + 30 * 1.28)) < 3, f"{rx:.0f} {ry:.0f}")
    check("light plan: the area lamp on the other side, nearer the wall", abs(gx - (128 + 20 * 1.28)) < 3 and abs(gy - (128 + 10 * 1.28)) < 3,
          f"{gx:.0f} {gy:.0f}")

    bare = sr.plan([], Ms, anchor, nh, ex, V, 100.0, size=256)
    cam_at = anchor + nh * 60
    pov = sr.plan_lights(bare, [], anchor, nh, ex, 100.0, viewer=cam_at, look_at=anchor, lens_mm=24.0)
    ring = sr.plan_lights(bare, [], anchor, nh, ex, 100.0, viewer=cam_at)
    drawn = np.abs(pov - ring).sum(-1) > 0.05
    ys, xs = np.nonzero(drawn)
    check("light plan: the camera's direction and angle of view are drawn, towards the facade",
          drawn.sum() > 150 and ys.max() <= 128 + 60 * 1.28 + 2 and ys.min() < 128 + 60 * 1.28 - 60 and (xs.min() < 128 - 30) and (xs.max() > 128 + 30),
          f"{drawn.sum()} y {ys.min()}..{ys.max()} x {xs.min()}..{xs.max()}")

    k0 = bridge.cache_key(fp, "", 64, 64, -1)
    sur = {"file": fp, "matrix": np.eye(4).tolist(), "ground": None}
    check("cache: no surroundings keeps the old key", k0 == bridge.cache_key(fp, "", 64, 64, -1, surroundings=None))
    check("cache: surroundings are a new key", k0 != bridge.cache_key(fp, "", 64, 64, -1, surroundings=sur))

    # reprojection: surroundings pixels are drawn but never lit or counted
    info = {"width": 4, "height": 2, "faces": 10, "camera": {
        "matrix_world": np.eye(4).tolist(), "projection": np.diag([1.0, 2.0, -1.0, 1.0]).tolist(), "type": "ORTHO"}}
    pos = np.zeros((2, 4, 3), np.float32)
    pos[..., 0], pos[..., 1] = np.meshgrid((np.arange(4) + 0.5) / 2 - 1, 0.5 - (np.arange(2) + 0.5) / 2)
    pos[..., 2] = -5
    nor = np.zeros((2, 4, 3), np.float32)
    nor[..., 2] = 1
    proj = {"info": info, "faceid": np.full((2, 4), 3, np.uint32), "position": pos, "normal": nor}
    fid = np.array([[3, 3, 11, 11], [0, 3, 11, 0]], np.uint32)
    rp = sv.reprojection({"info": info, "faceid": fid, "position": pos, "normal": nor}, proj)
    check("preview: surroundings are solid, not building, not lit, not shadow",
          rp["model"].sum() == 6 and rp["building"].sum() == 3 and rp["lit"].sum() == 3 and not rp["shadow"].any(),
          f"{rp['model'].sum()} {rp['building'].sum()} {rp['lit'].sum()} {rp['shadow'].sum()}")

    # clay colour and shaders of the previz
    grey = np.full((2, 4, 3), sv.CLAY_GREY, np.float32)
    check("clay colour: the clay's grey becomes the colour", np.allclose(sv.tint_clay(grey, "#ff8040"), (1.0, 128 / 255, 64 / 255), atol=1e-3))
    check("clay colour: white x brightness is the old albedo", sv.clay_albedo(0.7, "#ffffff") == [0.7, 0.7, 0.7]
          and sv.clay_albedo(0.7, None) == [0.7, 0.7, 0.7] and sv.clay_albedo(1.0, "#808080")[0] < 0.25)
    wfid = np.zeros((6, 8), np.uint32)
    wfid[1:5, 1:4], wfid[1:5, 4:7] = 1, 2
    wpos = np.zeros((6, 8, 3), np.float32)
    wnor = np.zeros((6, 8, 3), np.float32)
    wnor[..., 2] = 1
    ws = {"info": info, "faceid": wfid, "position": wpos, "normal": wnor}
    ln = sv.wire_lines(ws, soft=0)
    check("wireframe: a line between two faces and around them, none inside or outside",
          ln[2, 3] == 1 and ln[1, 1] == 1 and ln[4, 6] == 1 and ln[2, 2] == 0 and ln[2, 5] == 0 and ln[0].sum() == 0, str(ln))
    wfid2 = np.where(wfid > 0, 5, 0).astype(np.uint32)            # one id (surroundings): a corner and a depth step
    wnor2 = wnor.copy()
    wnor2[:, 4:] = (1, 0, 0)
    c1 = sv.wire_lines({"info": info, "faceid": wfid2, "position": wpos, "normal": wnor2}, soft=0)
    wpos2 = wpos.copy()
    wpos2[:, 4:, 2] = -3
    c2 = sv.wire_lines({"info": info, "faceid": wfid2, "position": wpos2, "normal": wnor}, soft=0)
    flat = sv.wire_lines({"info": info, "faceid": wfid2, "position": wpos, "normal": wnor}, soft=0)
    check("wireframe: surroundings draw at a corner and at a depth step, a flat wall stays clean",
          c1[2, 3] == 1 and c2[2, 3] == 1 and flat[2, 3] == 0 and flat[2, 1] == 1)
    bfid = np.zeros((20, 30), np.uint32)                           # two faces wide enough to have an inside
    bfid[2:18, 2:15], bfid[2:18, 15:28] = 1, 2
    bnor = np.zeros((20, 30, 3), np.float32)
    bnor[..., 2] = 1
    bs = {"info": info, "faceid": bfid, "position": np.zeros((20, 30, 3), np.float32), "normal": bnor}
    pic, full = sv.look_picture(bs, np.full((20, 30, 3), 0.8, np.float32), "wireframe", "#ff0000")
    check("shader wireframe: lines in the clay colour on black, not dimmed", full and pic[10, 8].sum() == 0
          and pic[10, 14, 0] > 0.9 and pic[..., 1].max() == 0 and pic[0, 0].sum() == 0, f"{pic[10, 8]} {pic[10, 14]}")
    pic2, full2 = sv.look_picture(bs, np.full((20, 30, 3), 0.8, np.float32), "clay + wireframe", "#f2f2f2")
    check("shader clay + wireframe: clay with darker edges", not full2 and pic2[10, 8, 0] > 0.9 and pic2[10, 14, 0] < 0.5,
          f"{pic2[10, 8]} {pic2[10, 14]}")

    try:
        exe = bridge.find_blender()
    except FileNotFoundError:
        exe = None
        print("skip Blender end-to-end (Blender not installed)")
    if exe:
        obj = os.path.join(tmp, "wall.obj")
        with open(obj, "w") as f:           # a wall 24 x 16 m facing -Y (OBJ Y-up -> Blender Z-up)
            f.write("v -12 0 0\nv 12 0 0\nv 12 16 0\nv -12 16 0\nf 1 2 3 4\n")
        cache = os.path.join(tmp, "cache")
        folder, _, _ = bridge.export(obj, cache, width=160, height=120, blender=exe)
        s = si.load(folder)
        pt, nrm, ground_z = sv.main_plane_and_ground(s)
        a, n_h, e_x = sr.facade_anchor(s["mesh"]["vert"], pt, nrm, ground_z)
        sur = {"file": fp, "matrix": sr.placement(a, n_h, e_x, facing), "ground": {"centre": a.tolist(), "radius": 150.0}}
        view = {"location": (a + n_h * 2 + [0, 0, 220]).tolist(), "look_at": a.tolist(), "lens": 20.0}   # from above
        vf, _, _ = bridge.export(obj, cache, width=480, height=270, blender=exe, view=view, surroundings=sur)
        vi = json.load(open(os.path.join(vf, "scene.json"), encoding="utf-8"))
        fid = np.load(os.path.join(vf, "faceid.npy"))
        pos = np.load(os.path.join(vf, "position.npy"))
        ctx = fid == vi["faces"] + 1
        check("Blender: the model keeps its face ids, the surroundings get the next one",
              vi["faces"] == 1 and set(np.unique(fid)) <= {0, 1, 2} and ctx.mean() > 0.5, f"{np.unique(fid)} {ctx.mean():.2f}")
        check("Blender: walls, roofs and the ground were built", vi["context_faces"] > 60, str(vi["context_faces"]))

        def height_at(x, y):                 # map metres -> height above the model's feet seen from above
            q = np.array(sur["matrix"]) @ np.array([x, y, 0.0, 1.0])
            d = np.linalg.norm(pos[..., :2] - q[:2], axis=-1)
            d[~ctx] = 1e9
            j = np.unravel_index(np.argmin(d), d.shape)
            return float(pos[j][2] - ground_z) if d[j] < 1.5 else None

        check("Blender: a neighbour's roof at its height", height_at(-21.0, 7.0) is not None and abs(height_at(-21.0, 7.0) - 15) < 0.05,
              str(height_at(-21.0, 7.0)))
        check("Blender: the ground at the model's feet", height_at(0.0, -30.0) is not None and abs(height_at(0.0, -30.0) + 0.02) < 0.01,
              str(height_at(0.0, -30.0)))
        check("Blender: the courtyard block has its roof", height_at(-79.0, -5.0) is not None and abs(height_at(-79.0, -5.0) - 11) < 0.05,
              str(height_at(-79.0, -5.0)))
        check("Blender: its courtyard is not roofed over", height_at(-71.0, -5.0) is not None and height_at(-71.0, -5.0) < 0.1,
              str(height_at(-71.0, -5.0)))
        # real ground: the slope under the square, the neighbours and the camera on it
        St = [dict(b, rings=[r.copy() for r in b["rings"]]) for b in S]
        Ts = {"heights": (50.0 - ax[:, None] / 10.0 + 0 * ax[None, :]).astype(np.float32), "step": 5.0, "radius": 100.0}  # falls to the north
        sr.drape(St, Ts, 50.0)
        fpt = sr.write(os.path.join(tmp, "s", "street_t.json"), St)
        gp = os.path.join(tmp, "s", "ground.npz")
        np.savez_compressed(gp, **sr.ground_grid(Ts, 50.0, 95.0))
        surt = {"file": fpt, "matrix": sur["matrix"], "ground": sur["ground"], "terrain": {"file": gp, "radius": 95.0}}
        vt, _, _ = bridge.export(obj, cache, width=480, height=270, blender=exe, view=view, surroundings=surt)
        fid, pos = np.load(os.path.join(vt, "faceid.npy")), np.load(os.path.join(vt, "position.npy"))
        ctx = fid == 2
        check("Blender: the ground follows the slope (map south of the facade is 3 m up)", abs(height_at(0.0, -30.0) - 3.0) < 0.15,
              str(height_at(0.0, -30.0)))
        check("Blender: a neighbour's roof is its height above its own ground", abs(height_at(-21.0, 7.0) - (15 - 0.7)) < 0.15,
              str(height_at(-21.0, 7.0)))
        check("Blender: the ground ends at its radius (no flat disc under it)", height_at(0.0, -120.0) is None, str(height_at(0.0, -120.0)))
        vi2 = json.load(open(os.path.join(vt, "scene.json"), encoding="utf-8"))
        cz = vi2["camera"]["matrix_world"][2][3]
        check("Blender: the camera stands on the ground under it", abs(cz - (view["location"][2] + 0.2)) < 0.05 and vi2["context_faces"] > 800,
              f"{cz:.2f} {vi2['context_faces']}")
        pf, _, _ = bridge.export(obj, cache, width=160, height=120, blender=exe, surroundings=None)
        check("Blender: without surroundings the old cache entry is used", pf == folder)
finally:
    shutil.rmtree(tmp, ignore_errors=True)

print()
print("FAILED: " + ", ".join(failures) if failures else "all surroundings tests passed")
sys.exit(1 if failures else 0)
