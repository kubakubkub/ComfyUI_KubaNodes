"""
Model free test for the S5 seam pass (kubakub/seams.py).

Checks the seam band (borders, blend off, keep, scope, changed), the tiles and
the pass itself with the fake adapter from test_sequential.py. Does not start
ComfyUI and loads no model.

Run from the ComfyUI_windows_portable folder:
    python_embeded\\python.exe ComfyUI\\custom_nodes\\ComfyUI_KubaNodes\\tests\\test_seams.py
"""

import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
from kubakub import plan as rp, seams  # noqa: E402
from kubakub.types import RegionPlan  # noqa: E402
from test_sequential import FakeAdapter, canvas_img, make_regions  # noqa: E402

failures = []


def check(name, cond, detail=""):
    print(("ok   " if cond else "FAIL ") + name + (f"  ({detail})" if detail and not cond else ""))
    if not cond:
        failures.append(name)


def plan_for(rules, scope_rows=None):
    regions = make_regions(scope_rows)
    return RegionPlan(rp.resolve(regions.table, rules, seed=3), regions)


def test_band():
    p = plan_for("[*]\nprompt = red")
    m = seams.plan_seam_mask(p, 16)
    check("band on the window edge", float(m[0, 60, 120]) > 0.9 and float(m[0, 59, 120]) > 0.9)
    check("band fades out", float(m[0, 60 - 8, 120]) < 0.5 and float(m[0, 45, 120]) == 0)
    check("no band inside a window", float(m[0, 100, 120]) == 0)
    check("no band on the canvas edge", float(m[0, 0, 240]) == 0)

    p = plan_for("[*]\nprompt = red\n[W_1]\nblend = off")
    m = seams.plan_seam_mask(p, 16)
    check("blend off: no band around W_1", float(m[0, 60, 120]) == 0 and float(m[0, 60, 360]) > 0.9)

    p = plan_for("[*]\nprompt = red\n[wall]\nstrategy = keep")
    m = seams.plan_seam_mask(p, 16)
    check("keep: band only on the window side", float(m[0, 57, 120]) == 0 and float(m[0, 62, 120]) > 0.5)

    p = plan_for("[*]\nprompt = red", scope_rows=(0, 100))
    m = seams.plan_seam_mask(p, 16)
    check("scope: nothing below the scope", float(m[0, 100:].sum()) == 0)

    p = plan_for("[*]\nprompt = red")
    changed = torch.zeros(1, 270, 480)
    changed[0, 60:140, 320:400] = 1.0
    m = seams.plan_seam_mask(p, 16, changed=changed)
    check("changed: only around the changed window", float(m[0, 60, 120]) == 0 and float(m[0, 60, 360]) > 0.9)


def test_tiles():
    check("tile starts cover", seams.tile_starts(3840, 1024, 64) == [0, 939, 1877, 2816])
    check("tile larger than canvas", seams.tile_starts(500, 1024, 64) == [0])
    mask = torch.zeros(1, 2160, 3840)
    mask[0, 100:110, 100:500] = 1
    t = seams.seam_tiles(mask, 1024, 64)
    check("only tiles with seams", len(t) == 1 and (t[0].x0, t[0].y0) == (0, 0), str(t))
    check("tiles at 1:1 on the grid", t[0].scale == 1 and t[0].w % 16 == 0)


def test_pass():
    p = plan_for("[*]\nprompt = blue wall\n[group:Windows]\nprompt = red glass")
    fa = FakeAdapter()
    img = canvas_img()
    out, mask, lines = seams.seam_pass(fa, img, p, seams.SeamSettings(seam_px=16, tile_px=256))
    check("sampled some tiles", len(fa.calls) >= 1, "\n".join(lines))
    check("mask output shape", mask.shape == (1, 270, 480))
    off = mask[0] == 0
    check("outside the band unchanged", torch.equal(out[0][off], img[0][off]))
    check("inside the band changed", not torch.equal(out[0][mask[0] > 0.9], img[0][mask[0] > 0.9]))
    check("reference attached", all(c["ref"] for c in fa.calls))
    check("seeds from the seam seed", all(c["seed"] >= 100_000 for c in fa.calls))


if __name__ == "__main__":
    for t in (test_band, test_tiles, test_pass):
        t()
    print()
    if failures:
        print(f"{len(failures)} FAILED: {', '.join(failures)}")
        sys.exit(1)
    print("all seam tests passed")
