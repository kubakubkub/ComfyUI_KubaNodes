r"""Model-free test of the facade pieces motion (scene3d/pieces.py): a wall with three separate boxes on it.

Run: python_embeded\python.exe custom_nodes\ComfyUI_KubaNodes\tests\test_pieces.py
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PACK = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(PACK, "kubakub", "scene3d"))

import pieces as pc  # noqa: E402
import scene_ids  # noqa: E402

# facade in the XZ plane at y = 0, the audience at -y: normal (0, -1, 0); ground at z = 0
FRAME = {"ex": [1.0, 0.0, 0.0], "normal": [0.0, -1.0, 0.0], "left": [0.0, 0.0, 0.0], "centre": [10.0, 0.0, 5.0],
         "ground_z": 0.0, "width_m": 20.0, "top_m": 10.0}


def box(x0, z0, s, y0=-0.5):
    """An axis-aligned box as 6 quads (own vertices)."""
    v = np.array([[x0 + dx * s, y0 + dy * s, z0 + dz * s] for dx in (0, 1) for dy in (0, 1) for dz in (0, 1)], float)
    q = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    return v, q


def mesh_of(objs):
    verts, loops, lt = [], [], []
    for v, quads in objs:
        base = sum(len(x) for x in verts)
        verts.append(v)
        for f in quads:
            loops += [base + i for i in f]
            lt.append(len(f))
    V = np.concatenate(verts)
    loop_vert = np.array(loops)
    lt = np.array(lt)
    starts = np.concatenate([[0], np.cumsum(lt)[:-1]])
    cen, nrm, area = [], [], []
    for s0, n in zip(starts, lt):
        p = V[loop_vert[s0:s0 + n]]
        c = np.cross(p[1] - p[0], p[2] - p[0])
        cen.append(p.mean(0))
        nrm.append(c / np.linalg.norm(c))
        area.append(np.linalg.norm(c))
    return {"vert": V, "loop_vert": loop_vert, "loop_total": lt, "centre": np.array(cen), "normal": np.array(nrm),
            "area": np.array(area)}


def main():
    wall = (np.array([[0, 0, 0], [20, 0, 0], [20, 0, 10], [0, 0, 10]], float), [(0, 3, 2, 1)])   # faces -y
    objs = [wall, box(2, 2, 1), box(9, 5, 1), box(17, 8, 1), box(5, 5, 0.05)]            # the last one is tiny
    mesh = mesh_of(objs)
    parts = scene_ids.loose_parts(mesh, 1e-4)
    assert int(parts.max()) + 1 == 5
    labels, st = pc.pieces_from_parts(parts, mesh, min_size_m=0.2)
    P = int(labels.max()) + 1
    assert P == 3, P                                            # wall (largest) and the tiny box stay still
    assert (labels[0] == -1) and (labels[-6:] == -1).all()
    u, v, d = pc.facade_coords(st, FRAME)
    assert np.allclose(sorted(u), [2.5, 9.5, 17.5]) and np.allclose(sorted(v), [2.5, 5.5, 8.5])
    assert np.allclose(d, 0.0)                                  # box centres at y = 0 -> in the wall plane

    left = int(np.argmin(u)); right = int(np.argmax(u))
    wave = {"type": "wave", "axis": "left to right", "speed": 5.0, "width": 1.0, "start": 0.0}
    w1 = pc.falloff(wave, 1.0, u, v, 20.0, 10.0)                # front at 5 m: the left box passed, the right not
    assert w1[left] > 0.99 and w1[right] < 0.01, w1
    assert pc.falloff(wave, -0.5, u, v, 20.0, 10.0).max() == 0
    pulse = dict(wave, type="pulse", width=2.0)
    wp = pc.falloff(pulse, 9.5 / 5.0, u, v, 20.0, 10.0)        # the band centred on the middle box
    mid = [i for i in range(3) if i not in (left, right)][0]
    assert wp[mid] > 0.99 and wp[left] < 0.01 and wp[right] < 0.01, wp
    st_f = {"type": "stagger", "order": "bottom to top", "start": 0.0, "spread": 2.0, "duration": 0.5}
    ws = pc.falloff(st_f, 0.6, u, v, 20.0, 10.0)
    low = int(np.argmin(v))
    assert ws[low] > 0.99 and ws[int(np.argmax(v))] < 0.01, ws
    hold = dict(st_f, hold=0.2)
    assert pc.falloff(hold, 3.5, u, v, 20.0, 10.0).max() < 0.01    # everything went back
    nz = pc.falloff({"type": "noise", "freq": 1.0, "seed": 3}, 0.37, u, v, 20.0, 10.0)
    assert np.all(np.abs(nz) <= 1.0) and len(set(np.round(nz, 6))) == 3
    assert np.allclose(pc.falloff({"type": "noise", "freq": 1.0, "seed": 3}, 0.37, u, v, 20.0, 10.0), nz)   # deterministic

    # push along the piece normal: the boxes' area-weighted normal is ~0 (closed boxes) -> use move out instead
    ops = [{"move": [0, 0, 2.0], "falloff": wave}]
    M = pc.evaluate(st, FRAME, ops, 1.0)
    moved = M[:, :3, 3]
    assert np.allclose(M[left, :3, 3], [0, -2.0, 0]) and np.allclose(M[right, :3, 3], 0, atol=1e-6)
    # rotation about the piece centre keeps the centre in place
    ops = [{"rotate": [0, 90, 0]}]
    M = pc.evaluate(st, FRAME, ops, 0.0)
    C = st["centre"]
    for k in range(3):
        assert np.allclose(M[k, :3, :3] @ C[k] + M[k, :3, 3], C[k])
    # a flat panel pushed along its own normal
    panel = mesh_of([wall, (np.array([[4, -0.2, 4], [6, -0.2, 4], [6, -0.2, 6], [4, -0.2, 6]], float), [(0, 1, 2, 3)])])
    lp, sp = pc.pieces_from_parts(scene_ids.loose_parts(panel, 1e-4), panel, 0.2)
    Mp = pc.evaluate(sp, FRAME, [{"push": 1.5}], 0.0)
    assert np.allclose(Mp[0, :3, 3], [0, -1.5, 0]), Mp[0, :3, 3]
    # a closed box has no facing of its own: push goes out of the facade
    Mb = pc.evaluate(st, FRAME, [{"push": 1.0}], 0.0)
    assert np.allclose(Mb[:, :3, 3], [[0, -1.0, 0]] * 3), Mb[:, :3, 3]
    # scale + the Blender-side vertex move
    M = pc.evaluate(st, FRAME, [{"scale": 2.0}], 0.0)
    lt = mesh["loop_total"]
    face_of_loop = np.repeat(np.arange(len(lt)), lt)
    piece_of_vert = np.full(len(mesh["vert"]), -1)
    piece_of_vert[mesh["loop_vert"]] = labels[face_of_loop]
    out = pc.apply_to_points(M, piece_of_vert, mesh["vert"])
    assert np.allclose(out[:4], mesh["vert"][:4])                    # the wall did not move
    vb = piece_of_vert == labels[1]
    ext0 = np.ptp(mesh["vert"][vb], axis=0)
    ext1 = np.ptp(out[vb], axis=0)
    assert np.allclose(ext1, 2 * ext0)
    # a sequence of frames
    X = pc.sequence(st, FRAME, [{"push": 0.5, "rotate": [10, 0, 0], "falloff": wave}], 2.0, 25)
    assert X.shape == (50, 3, 4, 4) and X.dtype == np.float32
    assert np.allclose(X[0, left], np.eye(4), atol=1e-6)            # t = 0: the front has not reached anything

    # --- regions: every part votes for the region most of its visible pixels are in
    fid = np.zeros((4, 6), np.int64)
    fid[0, :3] = 1                                              # face 0 (part 0) in 3 pixels
    fid[1, :2] = 3                                              # face 2 (part 1) in 2 pixels
    fid[1, 2] = 4                                               # face 3 (part 1) in 1 pixel
    fid[2, :] = 6                                               # face 5 (part 2)
    reg = np.full((4, 6), -1)
    reg[0, :2] = 7; reg[0, 2] = 5                               # part 0: 2 x region 7, 1 x region 5 -> 7
    reg[1, :3] = 5                                              # part 1: region 5
    parts_toy = np.array([0, 0, 1, 1, 2, 2, 3])                 # part 3 is never seen
    pr = pc.part_regions(fid, reg, parts_toy)
    assert pr.tolist() == [7, 5, -1, -1], pr
    # on the box model: the wall + boxes, regions "low" (two lower boxes) and "high" (top box)
    pr2 = np.full(int(parts.max()) + 1, -1)
    cen_part = {}
    for p_ in range(int(parts.max()) + 1):
        faces = np.flatnonzero(parts == p_)
        cen_part[p_] = mesh["centre"][faces].mean(0)
    for p_, c_ in cen_part.items():                              # region 0 below 7 m, region 1 above
        pr2[p_] = 0 if c_[2] < 7 else 1
    lab_g, st_g, preg = pc.pieces_from_regions(parts, pr2, mesh, wanted=None, group=True)
    assert int(lab_g.max()) + 1 == 2 and sorted(preg) == [0, 1]          # one rigid piece per region
    lab_s, st_s, preg_s = pc.pieces_from_regions(parts, pr2, mesh, wanted=[0], group=False)
    assert preg_s == [0] * (int(lab_s.max()) + 1) and int(lab_s.max()) + 1 >= 2   # only region 0's parts, each alone
    Mg = pc.evaluate(st_g, FRAME, [{"move": [0, 0, 1.0]}], 0.0)
    assert np.allclose(Mg[:, :3, 3], [[0, -1.0, 0]] * 2)

    # --- audience camera: in front of the wall on the projector's side, eyes 1.7 m up, looking at the middle
    av = pc.audience_view(FRAME, [10.0, -30.0, 5.0], distance_m=15.0, offset_m=2.0, eye_m=1.7, lens_mm=24)
    assert np.allclose(av["location"], [12.0, -15.0, 1.7]), av
    assert abs(av["look_at"][1]) < 1e-9 and abs(av["look_at"][2] - 5.0) < 1e-9 and av["lens"] == 24
    av_back = pc.audience_view(FRAME, [10.0, 30.0, 5.0], distance_m=15.0)   # projector behind: the other side
    assert av_back["location"][1] > 0

    # --- beat triggers: a kick at every beat, rippling left to right with spread
    trig = {"type": "triggers", "times": [0.0, 1.0, 2.0], "attack": 0.05, "decay": 0.2, "spread": 0.5,
            "order": "left to right"}
    w0 = pc.falloff(trig, 0.06, u, v, 20.0, 10.0)
    assert w0[left] > 0.7 and w0[right] < 0.05, w0              # the kick has not reached the right yet
    w1 = pc.falloff(trig, 0.56, u, v, 20.0, 10.0)
    assert w1[right] > 0.7 and w1[left] < 0.2, w1               # now it has, the left settled
    assert pc.falloff(dict(trig, times=[]), 0.5, u, v, 20.0, 10.0).max() == 0
    wb = pc.falloff(dict(trig, spread=0.0), 1.02, u, v, 20.0, 10.0)
    assert np.allclose(wb, wb[0]) and wb[0] > 0                 # no spread: every piece together (still rising)

    # --- review fixes: radial fronts at the set speed, nothing before a looped start, random ripple, region sizes
    uu, vv = np.array([15.0]), np.array([5.0])                  # 5 m right of the centre of a 20 x 10 facade
    rad = {"type": "wave", "axis": "from the centre", "speed": 5.0, "width": 0.2}
    assert pc.falloff(rad, 0.9, uu, vv, 20.0, 10.0)[0] < 0.01 and pc.falloff(rad, 1.1, uu, vv, 20.0, 10.0)[0] > 0.99
    lp = {"type": "pulse", "axis": "left to right", "speed": 5.0, "width": 2.0, "start": 3.0, "loop": 4.0}
    assert pc.falloff(lp, 0.4, np.array([7.0]), np.array([1.0]), 20.0, 10.0)[0] == 0.0
    rnd = {"type": "wave", "axis": "random", "speed": 5.0, "width": 0.5, "seed": 2}
    wr = pc.falloff(rnd, 2.0, u, v, 20.0, 10.0)
    assert np.all((wr >= 0) & (wr <= 1))
    lab_small, _, _ = pc.pieces_from_regions(parts, pr2, mesh, wanted=[0], group=False, min_size_m=0.2)
    assert int(lab_small.max()) + 1 == 2                        # the 5 cm box no longer moves
    X2 = pc.sequence(st, FRAME, [{"push": 0.5}], 0.2, 25)
    assert X2.dtype == np.float32 and X2.shape == (5, 3, 4, 4)
    print("ALL OK")


if __name__ == "__main__":
    main()
