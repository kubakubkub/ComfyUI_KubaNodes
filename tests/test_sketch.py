"""
Model free test of the sketch tools (kubakub/sketch.py): a synthetic pencil facade sketch with gaps between
strokes and red marker dots, photographed in perspective on a table with uneven light -> scan -> cells -> overlay.

Run from the ComfyUI_windows_portable folder:
    python_embeded\\python.exe ComfyUI\\custom_nodes\\ComfyUI_KubaNodes\\tests\\test_sketch.py
"""

import os
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from kubakub import sketch as sk  # noqa: E402

failures = []


def check(name, cond, detail=""):
    print(("ok   " if cond else "FAIL ") + name + (f"  ({detail})" if detail else ""))
    if not cond:
        failures.append(name)


def make_sketch(W=1600, H=900, seed=0):
    """Pencil lines (grey, wobbly, gaps of ~10 px at corners), 3 x 4 windows, red dots in two windows."""
    rng = np.random.default_rng(seed)
    img = np.ones((H, W, 3), np.float32)
    truth = np.zeros((H, W), np.uint8)

    def stroke(p0, p1, gap=10):
        p0, p1 = np.array(p0, np.float32), np.array(p1, np.float32)
        d = (p1 - p0) / max(np.linalg.norm(p1 - p0), 1)
        a, b = p0 + d * gap * rng.random(), p1 - d * gap * rng.random()   # strokes stop short of the corner
        pts = np.linspace(a, b, 40) + rng.normal(0, 0.8, (40, 2))
        g = 0.35 + 0.2 * rng.random()
        cv2.polylines(img, [pts.astype(np.int32)], False, (g, g, g), 3, cv2.LINE_AA)
        cv2.polylines(truth, [np.array([p0, p1]).astype(np.int32)], False, 1, 3)

    def rect(x0, y0, x1, y1):
        stroke((x0, y0), (x1, y0)); stroke((x1, y0), (x1, y1)); stroke((x1, y1), (x0, y1)); stroke((x0, y1), (x0, y0))

    rect(100, 80, 1500, 820)                                           # the facade
    wins = []
    for r in range(3):
        for c in range(4):
            x0, y0 = 220 + c * 320, 160 + r * 220
            rect(x0, y0, x0 + 160, y0 + 140)
            wins.append((x0 + 80, y0 + 70))
    pts = np.c_[np.linspace(120, 1480, 60), 850 + 6 * np.sin(np.linspace(0, 9, 60))]   # a green pencil line (a stroke)
    cv2.polylines(img, [pts.astype(np.int32)], False, (0.1, 0.6, 0.2), 3, cv2.LINE_AA)
    cv2.polylines(truth, [pts.astype(np.int32)], False, 1, 3)
    for (x, y) in (wins[0], wins[5]):                                  # marker dots name two windows
        cv2.circle(img, (x, y), 14, (0.85, 0.12, 0.1), -1, cv2.LINE_AA)
    img *= np.array([0.97, 0.94, 0.86], np.float32)                   # cream paper
    img *= 1 - 0.05 * rng.random((H, W, 1)).astype(np.float32)         # grain
    return np.clip(img, 0, 1), truth, wins


def photograph(paper, seed=1):
    """The paper lying on a dark table, seen at an angle, lamp from the left."""
    rng = np.random.default_rng(seed)
    H, W = paper.shape[:2]
    PW, PH = 2200, 1600
    corners = np.array([[260, 190], [1930, 260], [2020, 1380], [170, 1300]], np.float32)
    m = cv2.getPerspectiveTransform(np.array([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]], np.float32), corners)
    table = np.full((PH, PW, 3), (0.22, 0.15, 0.1), np.float32) + rng.normal(0, 0.02, (PH, PW, 3)).astype(np.float32)
    warped = cv2.warpPerspective(paper, m, (PW, PH), flags=cv2.INTER_LINEAR, borderValue=(-1, -1, -1))
    photo = np.where(warped[..., :1] >= 0, warped, table)
    light = np.linspace(1.0, 0.72, PW, dtype=np.float32)[None, :, None]   # uneven lamp
    return np.clip(photo * light, 0, 1), corners


paper, truth, wins = make_sketch()
photo, true_corners = photograph(paper)

# --- scan
found, ok = sk.find_paper(photo)
err = float(np.abs(found - true_corners).max())
check("paper found in the photo", ok and err < 25, f"max corner error {err:.1f} px")
check("parse corners text", np.allclose(sk.parse_corners("10,20 30,40; 50 60 70,80"),
                                        [[10, 20], [30, 40], [50, 60], [70, 80]]))
check("order corners", np.allclose(sk.order_corners([[9, 9], [0, 0], [0, 9], [9, 0]]), [[0, 0], [9, 0], [9, 9], [0, 9]]))

t = time.perf_counter()
r = sk.scan(photo, 1600, 900, "", "pencil", 0.5, 1.0)
dt = time.perf_counter() - t
s = r["strength"]
check("scan output at the asked size", s.shape == (900, 1600) and r["drawing"].shape == (900, 1600, 3))
pred = s > 0.3
near_truth = cv2.dilate(truth, np.ones((9, 9), np.uint8)).astype(bool)
precision = float((pred & near_truth).sum() / max(pred.sum(), 1))
recall = float((cv2.dilate(pred.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool) & truth.astype(bool)).sum()
               / truth.sum())
check("lines found where drawn (precision)", precision > 0.9, f"{precision:.3f}")
check("drawn lines recovered (recall)", recall > 0.8, f"{recall:.3f}")
paper_px = r["drawing"][~near_truth]
check("paper is white, no lamp gradient left", float(paper_px.min(axis=-1).mean()) > 0.95
      and abs(float(r["drawing"][450, 60].mean()) - float(r["drawing"][450, 1540].mean())) < 0.05,
      f"mean {paper_px.mean():.3f}")
check("marker stays red in the drawing", r["drawing"][wins[0][1], wins[0][0], 0] > 0.6
      and r["drawing"][wins[0][1], wins[0][0], 1] < 0.4, str(r["drawing"][wins[0][1], wins[0][0]]))
check("scan speed", dt < 3.0, f"{dt:.2f} s")
check("marker dots left out of the line", float(s[wins[0][1], wins[0][0]]) == 0.0)
check("a green pencil line stays a line", float(s[850:858, 700:900].max()) > 0.5, f"{s[850:858, 700:900].max():.2f}")
check("dots_in_line keeps them", float(sk.scan(photo, 1600, 900, dots_in_line=True)["strength"][wins[0][1], wins[0][0]]) > 0.3)
w0, _ = sk.paper_size(true_corners)
r_auto = sk.scan(photo, 0, 0, sk.corners_text(true_corners))
check("size 0 = the paper's own size, given corners used", abs(r_auto["strength"].shape[1] - w0) < 2
      and "given" in r_auto["note"], r_auto["note"])
ink = sk.scan(photo, 1600, 900, "", "ink")["strength"]
check("ink mode is black / white", set(np.unique(ink).tolist()) <= {0.0, 1.0})

# --- thinning / gaps / cells
bar = np.zeros((60, 200), bool)
bar[25:32, 20:180] = True
th = sk.thin(bar)
check("thinning: a 7 px bar becomes 1 px", th.sum() > 100 and th[:, 100].sum() == 1, str(th[:, 100].sum()))

cells0, _, n0, _ = sk.sketch_cells(s, r["drawing"], gap_px=0)
cells, tags, n_br, lines = sk.sketch_cells(s, r["drawing"], gap_px=30)


def window_cells(c):
    ids = {int(c[y, x]) for (x, y) in wins}
    return len(ids - {-1})


check("without gap bridging the windows leak into the wall", window_cells(cells0) < 12, str(window_cells(cells0)))
check("with gap bridging every window is its own cell", window_cells(cells) == 12, f"{window_cells(cells)}, {n_br} bridges")
wall = int(cells[120, 150])
check("the wall is one cell apart from the windows", wall >= 0 and wall not in {int(cells[y, x]) for (x, y) in wins})
red = {int(cells[y, x]) for (x, y) in (wins[0], wins[5])}
check("red dots tag their windows", all(tags.get(c) == ["red"] for c in red) and len(tags) == 2, str(tags))
check("a red dot is not a line", not lines[wins[0][1], wins[0][0]])

# --- overlay
render = np.full((900, 1600, 3), 0.8, np.float32)
o = sk.overlay(render, s, 1.0, "multiply", "#000000")
check("multiply: dark on the line, untouched on paper", o[pred].mean() < 0.4 and np.allclose(o[s == 0], 0.8, atol=1e-3))
o2 = sk.overlay(np.zeros((900, 1600, 3), np.float32), s, 1.0, "screen", "#ffffff")
check("screen: light line on dark (= the line strength)", np.allclose(o2[..., 0], s, atol=1e-5))
o3 = sk.overlay(render, s, 0.0)
check("amount 0 = unchanged", np.allclose(o3, render))
check("colour mode paints the line", np.allclose(sk.overlay(render, (truth > 0).astype(np.float32), 1.0, "colour", "#f18a58")[truth > 0],
                                                  [241 / 255, 138 / 255, 88 / 255], atol=1e-3))

# --- the sample facade's sketch photo end to end (what starter 05 does)
from kubakub import sample_facade as sf  # noqa: E402
ph = sf.sketch_photo(1920, 1080, 3, 7, 0)
check("sketch photo 4:3", ph.shape == (1440, 1920, 3))
rs = sk.scan(ph, 1920, 1080)
cs, tg, nb, _ = sk.sketch_cells(rs["strength"], rs["drawing"], gap_px=24)
lay = {n: box for n, _g, box, _k in sf.layout(1920, 1080, 3, 7)}
def cell_of(n):
    x0, y0, x1, y1 = lay[n]
    return int(cs[(y0 + y1) // 2 + (y1 - y0) // 3, (x0 + x1) // 2])
wins_s = [n for n in lay if n.startswith("W_F")]
ids = [cell_of(n) for n in wins_s]
check("every sample window its own cell", len(set(ids)) == len(ids) and -1 not in ids, f"{len(set(ids))} of {len(ids)}")
check("door tagged red", tg.get(cell_of("M_Door")) == ["red"], str(tg.get(cell_of("M_Door"))))
blue = [tg.get(cell_of(n)) for n in wins_s if n.startswith("W_F1_")]
check("first floor windows tagged blue", all(b == ["blue"] for b in blue), str(blue))
from kubakub import facade_core as fc  # noqa: E402
for mra in (400, 4000):
    lab, at = fc._finish_atlas(fc.fill_unassigned(cs), [{} for _ in range(int(cs.max()) + 1)], [], [], None,
                               "line_drawing", mra, True)
    print(f"     min_region_area {mra}: {len(at['regions'])} regions")

if "--save" in sys.argv:                                              # for looking at it
    out = os.path.join(HERE, "test_output")
    os.makedirs(out, exist_ok=True)
    cv2.imwrite(os.path.join(out, "sketch_photo.png"), (photo[..., ::-1] * 255).astype(np.uint8))
    cv2.imwrite(os.path.join(out, "sketch_scan.png"), (r["drawing"][..., ::-1] * 255).astype(np.uint8))
    vis = (np.random.default_rng(3).random((cells.max() + 2, 3)) * 255).astype(np.uint8)[cells + 1]
    cv2.imwrite(os.path.join(out, "sketch_cells.png"), vis)

print("\n" + ("ALL OK" if not failures else f"{len(failures)} FAILED: {failures}"))
sys.exit(1 if failures else 0)
