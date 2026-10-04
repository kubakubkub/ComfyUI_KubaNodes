"""
Model free test for the Kuba Regions geometry contract (kubakub/geometry.py).

Checks the canvas plans for the 3840x2160 matrix against the table in
the geometry contract, and that to_work -> restore puts every pixel
back where it was (registration). Does not start ComfyUI and loads no model.

Run from the ComfyUI_windows_portable folder:
    python_embeded\\python.exe ComfyUI\\custom_nodes\\ComfyUI_KubaNodes\\tests\\test_geometry.py
"""

import json
import os
import sys
from fractions import Fraction

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from kubakub import geometry as geo  # noqa: E402

TW, TH = 3840, 2160
FLUX, LTX, H3 = (geo.get_geometry(f) for f in ("flux2", "ltxav", "minimax_h3"))

failures = []


def check(name, cond, detail=""):
    print(("ok   " if cond else "FAIL ") + name + (f"  ({detail})" if detail and not cond else ""))
    if not cond:
        failures.append(name)


def test_plan_table():
    # k -> (work, padded for divisor 16, padded for divisor 32)
    table = {
        "3/2": ((2560, 1440), (2560, 1440), (2560, 1440)),
        "2": ((1920, 1080), (1920, 1088), (1920, 1088)),
        "5/2": ((1536, 864), (1536, 864), (1536, 864)),
        "3": ((1280, 720), (1280, 720), (1280, 736)),
        "15/4": ((1024, 576), (1024, 576), (1024, 576)),
        "4": ((960, 540), (960, 544), (960, 544)),
    }
    for k, (work, p16, p32) in table.items():
        for g, want in ((FLUX, p16), (LTX, p32)):
            p = geo.plan_canvas(TW, TH, g, k=k, max_pad_percent=5)
            got = ((p.work_w, p.work_h), (p.padded_w, p.padded_h))
            check(f"plan k={k} {g.family}", got == (work, want), f"{got}")
            check(f"plan k={k} {g.family} aspect exact", p.work_w * TH == p.work_h * TW)
            check(f"plan k={k} {g.family} pad sums",
                  p.pad_left + p.work_w + p.pad_right == p.padded_w
                  and p.pad_top + p.work_h + p.pad_bottom == p.padded_h)


def test_plan_errors_and_auto():
    try:
        geo.plan_canvas(TW, TH, FLUX, k=3.3)
        check("k=3.3 rejected", False)
    except ValueError as e:
        check("k=3.3 rejected with suggestions", "Nearest exact factors" in str(e), str(e))

    p = geo.plan_canvas(TW, TH, FLUX)             # budget 1 MP
    check("auto flux2 picks 1280x720", (p.padded_w, p.padded_h) == (1280, 720), p.report())
    p = geo.plan_canvas(TW, TH, LTX, integer_k=True)
    check("auto ltx integer k picks 960x544 grid", (p.padded_w, p.padded_h) == (960, 544), p.report())
    p = geo.plan_canvas(TW, TH, H3, k=3)
    check("h3 k=3 -> 1280x736", (p.padded_w, p.padded_h) == (1280, 736))
    check("h3 k=3 pad 8+8", (p.pad_top, p.pad_bottom) == (8, 8))


def test_frames():
    check("ltx frames 121 -> 121", LTX.snap_frames(121) == 121)
    check("ltx frames 125 -> 129", LTX.snap_frames(125) == 129)
    check("h3 frames 125 -> 124+17=141?", H3.snap_frames(125) == 141, str(H3.snap_frames(125)))
    check("h3 frames 124 -> 124", H3.snap_frames(124) == 124)
    p = geo.plan_canvas(TW, TH, LTX, k=4, target_frames=125)
    check("plan frames", (p.work_frames, p.target_frames) == (129, 125))


def test_json_roundtrip():
    p = geo.plan_canvas(TW, TH, FLUX, k=2)
    q = geo.CanvasPlan.from_dict(json.loads(json.dumps(p.to_dict())))
    check("json round trip", q == p)
    check("k fraction", q.k_fraction == Fraction(2))


def _pattern(w, h, b=1):
    """A test card where every pixel is unique enough to catch a one pixel shift."""
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    img = torch.stack([(x % 64) / 63.0, (y % 64) / 63.0, ((x // 64 + y // 64) % 2).float()], -1)
    return img[None].expand(b, -1, -1, -1).float().contiguous()


def test_registration():
    # Small target with the same 16:9 aspect so the test runs fast; nearest keeps pixels exact.
    tw, th = 384, 216
    near = geo.torch_resize("nearest")
    for k, g in (("2", FLUX), ("4", LTX), ("3", H3)):
        p = geo.plan_canvas(tw, th, g, k=k, max_pad_percent=50)
        img = _pattern(tw, th)
        mask = torch.zeros((1, th, tw))
        mask[:, 40:100, 60:200] = 1.0
        work, wmask, scope = geo.to_work(img, p, mask=mask, resize=near)
        check(f"to_work size k={k} {g.family}",
              tuple(work.shape[1:3]) == (p.padded_h, p.padded_w))
        check(f"scope zero on padding k={k}",
              scope[0, :p.pad_top].sum() == 0 and scope[0, p.pad_top:p.pad_top + p.work_h,
                                                        p.pad_left:p.pad_left + p.work_w].min() == 1)
        # model upscaler at an integer factor on the padded canvas, then restore
        up = int(p.k_fraction)
        big = work.repeat_interleave(up, 1).repeat_interleave(up, 2)
        bmask = wmask.repeat_interleave(up, 1).repeat_interleave(up, 2)
        out, omask = geo.restore(big, p, mask=bmask, resize=near)
        ref = _bhwc_down_up(img, p, near)
        check(f"restore exact k={k} {g.family}", torch.equal(out, ref),
              f"max diff {(out - ref).abs().max():.4f}")
        check(f"mask restored k={k}", omask.shape == mask.shape)

    # a padded canvas decoded at 1x, restored with a real resize
    p = geo.plan_canvas(tw, th, FLUX, k=2, max_pad_percent=50)
    work, _, _ = geo.to_work(_pattern(tw, th), p)
    out, _ = geo.restore(work, p)
    check("restore 1x to target size", tuple(out.shape[1:3]) == (th, tw))

    try:
        geo.restore(torch.zeros(1, 100, 190, 3), p)
        check("non-uniform size rejected", False)
    except ValueError:
        check("non-uniform size rejected", True)


def _bhwc_down_up(img, p, near):
    x = near(img.movedim(-1, 1), p.work_w, p.work_h)
    return near(x, p.target_w, p.target_h).movedim(1, -1)


def test_fit_aspect():
    img = _pattern(400, 400)
    try:
        geo.fit_aspect(img, TW, TH, "error")
        check("square vs 16:9 rejected", False)
    except ValueError:
        check("square vs 16:9 rejected", True)
    c, _, _ = geo.fit_aspect(img, TW, TH, "center_crop")
    check("center_crop aspect exact", c.shape[2] * 9 == c.shape[1] * 16, str(tuple(c.shape)))
    pd, _, valid = geo.fit_aspect(img, TW, TH, "pad")
    check("pad aspect exact", pd.shape[2] * 9 == pd.shape[1] * 16, str(tuple(pd.shape)))
    check("pad valid area", int(valid.sum()) == 400 * 400)


def test_video_trim():
    p = geo.plan_canvas(384, 216, LTX, k=2, target_frames=25, max_pad_percent=50)
    vid = torch.zeros((p.work_frames, p.padded_h, p.padded_w, 3))
    out, _ = geo.restore(vid, p)
    check("video trimmed to target frames", out.shape[0] == 25, str(out.shape[0]))


if __name__ == "__main__":
    for t in (test_plan_table, test_plan_errors_and_auto, test_frames, test_json_roundtrip,
              test_registration, test_fit_aspect, test_video_trim):
        t()
    print()
    print(geo.plan_canvas(TW, TH, FLUX, k=2).report())
    print()
    if failures:
        print(f"{len(failures)} FAILED: {', '.join(failures)}")
        sys.exit(1)
    print("all geometry tests passed")
