"""
A 3D model of the sample facade, to try the 3d nodes without a file of your own: the elevation of sample_facade.py
built in metres as an OBJ (stones in courses, arches with voussoirs on the ground floor, framed windows with glass
behind the wall, pilasters, sill bands, cornices, a door).

Every stone, frame, voussoir and cornice is a loose part of its own (kubakub scene pieces moves them), objects are
named by element (wall_F1, window_F2_C03, pilaster_04 ...) and carry a material (stone, trim, glass, wood). The OBJ
has no camera: kubakub projector places one, or kubakub scene render frames the model by itself.

numpy only (tests/test_sample_model.py). OBJ axes: x right, y up, z towards the audience; the wall front is z = 0.
"""

from __future__ import annotations

import math
import os

import numpy as np

from . import sample_facade as sf

MATERIALS = {"stone": (0.72, 0.66, 0.56), "trim": (0.84, 0.80, 0.72), "glass": (0.10, 0.14, 0.18),
             "wood": (0.30, 0.20, 0.13)}
VERSION = 1                 # raise when the geometry changes (the node keeps the file it wrote)
WALL_DEPTH = 0.5           # m, thickness of the wall
COURSE_M, STONE_M = 0.55, 1.15


class _Mesh:
    """Collects the model: objects of pieces, a piece = prisms that share their vertices (one loose part)."""

    def __init__(self):
        self.verts, self.objects, self.pieces = [], [], 0
        self._faces = self._weld = None
        self._taken = set()

    def free_z(self, pts, z, step=0.0003):
        """z moved by a few tenths of a millimetre until no vertex of another piece sits at (x, y, z) for these
        points: pieces that touch must not share a vertex, or they would count as one loose part."""
        for k in range(200):
            zz = round(z + step * k, 5)
            if not any((round(x, 5), round(y, 5), zz) in self._taken for x, y in pts):
                return zz
        return z

    def object(self, name, material):
        self._faces = []
        self.objects.append((name, material, self._faces))

    def piece(self):
        self._weld = {}
        self.pieces += 1

    def _v(self, x, y, z):
        key = (round(x, 5), round(y, 5), round(z, 5))
        i = self._weld.get(key)
        if i is None:
            i = self._weld[key] = len(self.verts)
            self.verts.append(key)
            self._taken.add(key)
        return i

    def prism(self, poly, z0, z1):
        """A convex polygon (counter-clockwise seen from the front, x / y in metres) pulled from z0 back to z1 front."""
        pts = [p for k, p in enumerate(poly) if math.dist(p, poly[k - 1]) > 1e-6]
        if len(pts) < 3:
            return
        front = [self._v(x, y, z1) for x, y in pts]
        back = [self._v(x, y, z0) for x, y in pts]
        self._faces.append(front)
        self._faces.append(back[::-1])
        for k in range(len(pts)):
            j = (k + 1) % len(pts)
            self._faces.append([front[k], back[k], back[j], front[j]])

    def box(self, x0, x1, y0, y1, z0, z1):
        self.prism([(x0, y0), (x1, y0), (x1, y1), (x0, y1)], z0, z1)


def _courses(x0, x1, y0, y1, row0=0):
    """A wall field as stones in courses (every second course shifted by half a stone) -> [(x0, x1, y0, y1)]."""
    out = []
    if x1 - x0 < 0.02 or y1 - y0 < 0.02:
        return out
    rows = max(1, round((y1 - y0) / COURSE_M))
    for r in range(rows):
        a, b = y0 + (y1 - y0) * r / rows, y0 + (y1 - y0) * (r + 1) / rows
        n = max(1, round((x1 - x0) / STONE_M))
        cuts = [x0 + (x1 - x0) * k / n for k in range(n + 1)]
        if (r + row0) % 2 and n > 1:
            cuts = [x0] + [c - (x1 - x0) / n / 2 for c in cuts[1:]] + [x1]
        out += [(cuts[k], cuts[k + 1], a, b) for k in range(len(cuts) - 1) if cuts[k + 1] - cuts[k] > 0.02]
    return out


def build(floors=3, bays=7, width_m=24.0, height_m=16.0, relief_m=0.015, seed=0):
    """-> (_Mesh, info dict). The facade is centred on x = 0 and stands on y = 0."""
    floors, bays = int(floors), int(bays)
    W = 2000
    margin = int(W * 0.06)
    s = float(width_m) / (W - 2 * margin)                    # metres per layout pixel
    H = int(round(float(height_m) / s / 0.84))
    parts = sf.layout(W, H, floors, bays)
    ground_y = int(H * 0.94)
    X = lambda px: (px - W / 2) * s                          # noqa: E731
    Y = lambda px: (ground_y - px) * s                       # noqa: E731
    box = {n: (X(x0), X(x1), Y(y1), Y(y0)) for n, _g, (x0, y0, x1, y1), _k in parts}   # x0, x1, y bottom, y top
    rng = np.random.default_rng(int(seed))
    m = _Mesh()

    def block(polys, front):
        """One piece of the wall, from the back of the wall to its own front."""
        pts = [p for poly in polys for p in poly]
        z0, z1 = m.free_z(pts, -WALL_DEPTH, -0.0003), m.free_z(pts, front)
        m.piece()
        for poly in polys:
            m.prism(poly, z0, z1)

    def stone(x0, x1, y0, y1):
        out = float(rng.choice([0.0, 0.0, 0.0, 0.0, 0.0, 1 / 3, 2 / 3, 1.0]))     # most stones are flush: the wall plane
        block([[(x0, y0), (x1, y0), (x1, y1), (x0, y1)]], float(relief_m) * out)

    def openings(prefix):
        return sorted((box[n] for n in box if n.startswith(prefix)), key=lambda b: b[0])

    bands = [("wall_attic", box["M_Attic"], [], False)]
    bands += [(f"wall_F{f}", box[f"M_FLOOR_F{f}"], openings(f"W_F{f}_"), False) for f in range(floors, 0, -1)]
    ground = openings("W_F0_") + [box["M_Door"]]
    bands.append(("wall_F0", box["M_FLOOR_F0"], sorted(ground, key=lambda b: b[0]), True))
    # the cornices lie in front of the wall; behind them the wall runs through. Top to bottom, and every band ends
    # where the next one starts (the layout rounds each edge to a pixel on its own)
    bands[1:1] = [("wall_cornice_top", box["M_Cornice_Top"], [], False)]
    bands[-1:-1] = [("wall_cornice_ground", box["M_Cornice_Ground"], [], False)]
    wx0, wx1 = box["M_Attic"][0], box["M_Attic"][1]
    tops = [b[1][3] for b in bands] + [0.0]
    bands = [(n, (wx0, wx1, tops[k + 1], tops[k]), holes, arched) for k, (n, _b, holes, arched) in enumerate(bands)]
    for name, (x0, x1, y0, y1), holes, arched in bands:
        m.object(name, "stone")
        edge, row = x0, 0
        for (a, b, c, d) in holes:
            for st in _courses(edge, a, y0, y1, row):                      # the pier left of the opening
                stone(*st)
            top = d - (b - a) / 2 if arched else d                         # arches: the spring line
            for st in _courses(a, b, y0, c):                               # below the opening
                stone(*st)
            for st in _courses(a, b, d, y1, 1):                            # above it
                stone(*st)
            if arched:                                                     # between the arch and its box: one piece
                r, cx = (b - a) / 2, (a + b) / 2
                ang = [math.pi * k / 8 for k in range(9)]
                arc = [(cx + r * math.cos(t), top + r * math.sin(t)) for t in ang]
                rim = [(cx + r * math.cos(t) / max(abs(math.cos(t)), abs(math.sin(t))),
                        top + r * math.sin(t) / max(abs(math.cos(t)), abs(math.sin(t)))) for t in ang]
                block([[arc[k], rim[k], rim[k + 1], arc[k + 1]] for k in range(8)], 0.0)
            edge, row = b, row + 1
        for st in _courses(edge, x1, y0, y1, row):
            stone(*st)

    # what sits in front of the wall or behind it; every element is one piece
    def element(name, material, boxes):
        m.object(name, material)
        m.piece()
        for b in boxes:
            m.box(*b)

    n_el = [0]

    def lift():                                              # a few tenths of a millimetre, so elements share no vertex
        n_el[0] += 1
        return 0.0007 * (n_el[0] % 53)

    for f in range(floors, 0, -1):
        for c, (a, b, y0, y1) in enumerate(openings(f"W_F{f}_")):
            fw = 0.14
            z = 0.07 + lift()
            element(f"frame_F{f}_C{c + 1:02d}", "trim",
                    [(a - fw, b + fw, y1, y1 + fw, 0.004, z), (a - fw, b + fw, y0 - fw, y0, 0.004, z),
                     (a - fw, a, y0, y1, 0.004, z), (b, b + fw, y0, y1, 0.004, z)])
            element(f"window_F{f}_C{c + 1:02d}", "glass", [(a - 0.05, b + 0.05, y0 - 0.05, y1 + 0.05, -0.34 - lift(), -0.3)])
    arches = [(n, box[n]) for n in sorted(box) if n.startswith("W_F0_")] + [("M_Door", box["M_Door"])]
    for name, (a, b, y0, y1) in arches:
        tag = "door" if name == "M_Door" else "arch_" + name[2:]
        r, cx, top = (b - a) / 2, (a + b) / 2, y1 - (b - a) / 2
        m.object(tag, "trim")                                # voussoirs: a ring of stones around the arch
        n = 9
        for k in range(n):
            t0, t1 = math.pi * (k + 0.03) / n, math.pi * (k + 0.97) / n
            m.piece()
            ro = r + (0.42 if k == n // 2 else 0.32)         # the keystone is taller
            m.prism([(cx + r * math.cos(t0), top + r * math.sin(t0)), (cx + ro * math.cos(t0), top + ro * math.sin(t0)),
                     (cx + ro * math.cos(t1), top + ro * math.sin(t1)), (cx + r * math.cos(t1), top + r * math.sin(t1))],
                    -0.1, (0.1 if k == n // 2 else 0.06) + lift())
        pane = "wood" if name == "M_Door" else "glass"
        depth = -0.22 if name == "M_Door" else -0.3
        element(("door_leaf" if name == "M_Door" else "window_" + name[2:]), pane,
                [(a - 0.05, b + 0.05, y0, y1 + 0.05, depth - 0.04 - lift(), depth)])
    for n in sorted(box):
        x0, x1, y0, y1 = box[n]
        if n.startswith("M_Pilaster_"):
            z = 0.18 + lift()
            element("pilaster_" + n[-2:], "trim", [(x0, x1, y0 + 0.25, y1 - 0.3, 0.004, z),
                                                   (x0, x1, y0, y0 + 0.25, 0.004, z + 0.06),      # base
                                                   (x0, x1, y1 - 0.3, y1, 0.004, z + 0.06)])     # capital
        elif n.startswith("M_Sill_"):
            element("sill_" + n[7:], "trim", [(x0, x1, y0, y1, 0.004, 0.12 + lift())])
        elif n.startswith("M_Cornice_"):
            out = 0.4 if n.endswith("Top") else 0.26
            mid = (y0 + y1) / 2
            element("cornice_" + n[10:].lower(), "trim", [(x0, x1, y0, mid, 0.004, out * 0.6 + lift()),
                                                          (x0, x1, mid, y1, 0.004, out + lift())])
    fx0, fx1 = box["M_FLOOR_F0"][0], box["M_FLOOR_F0"][1]
    top_y = box["M_Attic"][3]
    info = {"width_m": fx1 - fx0, "height_m": top_y, "floors": floors, "bays": bays, "objects": len(m.objects),
            "pieces": m.pieces, "vertices": len(m.verts), "faces": sum(len(f) for _, _, f in m.objects)}
    return m, info


def write_obj(path, floors=3, bays=7, width_m=24.0, height_m=16.0, relief_m=0.015, seed=0):
    """Writes <path> (.obj) and its .mtl next to it. -> info dict (size in metres, objects, pieces, faces)."""
    m, info = build(floors, bays, width_m, height_m, relief_m, seed)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    mtl = os.path.splitext(path)[0] + ".mtl"
    with open(mtl, "w", encoding="utf-8", newline="\n") as f:
        for name, (r, g, b) in MATERIALS.items():
            f.write(f"newmtl {name}\nKd {r:.3f} {g:.3f} {b:.3f}\nKa 0 0 0\nKs 0.05 0.05 0.05\nd 1\nillum 2\n\n")
    lines = ["# kubakub sample model: a facade in metres (x right, y up, z towards the audience)",
             f"mtllib {os.path.basename(mtl)}"]
    lines += [f"v {x:.5f} {y:.5f} {z:.5f}" for x, y, z in m.verts]
    for name, material, faces in m.objects:
        lines += [f"o {name}", f"usemtl {material}"]
        lines += ["f " + " ".join(str(i + 1) for i in face) for face in faces]
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")
    info["file"] = path
    return info
