"""
Model free test for the frame in frame viewer geometry (kubakub/viewer.py).
Does not start ComfyUI.

Run from the ComfyUI_windows_portable folder:
    python_embeded\\python.exe ComfyUI\\custom_nodes\\ComfyUI_KubaNodes\\tests\\test_viewer.py
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from kubakub import viewer as vw  # noqa: E402

W, H = 800, 400


def facade():
    """40 x 20 m facade, 5 cm per px; viewer centred, eye 1.7 m, 20 m away."""
    return vw.Viewer(W, H, facade_width_m=40.0, bottom_m=0.0, eye_m=1.7, distance_m=20.0)


def test_scale_and_foot():
    v = facade()
    assert abs(v.m_per_px - 0.05) < 1e-12 and abs(v.facade_height_m - 20.0) < 1e-9
    fx, fy = v.foot_px()
    assert (fx, round(fy, 6)) == (400.0, round(400 - 1.7 / 0.05, 6)), (fx, fy)
    assert v.scale(0) == 1.0 and abs(v.scale(20.0) - 0.5) < 1e-12 and abs(v.scale(-10.0) - 2.0) < 1e-12
    try:
        v.scale(-20.0)
    except ValueError:
        pass
    else:
        raise AssertionError("depth at the viewer must raise")
    v2 = vw.Viewer.from_dict(v.to_dict())
    assert v2 == v
    return "s = d / (d + D), foot point, round trip"


def window(x0, y0, x1, y1, rid=0, labels=None):
    lab = np.full((H, W), -1, np.int32) if labels is None else labels
    lab[y0:y1, x0:x1] = rid
    return lab


def test_room_behind_window():
    v = facade()
    lab = window(100, 50, 200, 150)                 # upper left window; viewer below right
    r = vw.render_frames(lab, {0: 4.0}, v)          # a 4 m deep room: scale 20/24
    win = lab == 0
    assert (r["area"] == win).all(), "a room only changes its window"
    assert r["back"].any() and r["walls"].any()
    assert not (r["back"] & r["walls"]).any()
    assert (r["back"] | r["walls"])[win].mean() > 0.99, "window = back wall + side walls"
    ys, xs = np.nonzero(r["back"])
    # back wall scaled by 0.833 towards the foot point (400, 366): it moves right and down,
    # so the ceiling (top) and the left wall are visible
    assert xs.min() > 100 + 40 and ys.min() > 50 + 40, (xs.min(), ys.min())
    f = r["frames"][0]
    assert abs(f["scale"] - 20 / 24) < 1e-3 and len(f["back_quad"]) == 4
    return "back wall shifts towards the viewer, ceiling + left wall visible"


def test_parasite_sticks_out():
    v = facade()
    lab = window(380, 200, 420, 240)
    r = vw.render_frames(lab, {0: -5.0}, v)          # 5 m towards a viewer 20 m away: scale 4/3
    assert r["area"].sum() > (lab == 0).sum() * 1.3, "a parasite covers more than its frame"
    assert r["back"].sum() > (lab == 0).sum()
    ys, xs = np.nonzero(r["area"])
    assert ys.min() < 200, "above the viewer's eye it grows upwards"
    return f"front face {r['back'].sum()} px vs frame {(lab == 0).sum()} px"


def test_window_panes_hull():
    v = facade()
    lab = window(100, 50, 145, 150)
    lab = window(155, 50, 200, 150, labels=lab)      # two panes, a 10 px mullion between
    hull = vw.render_frames(lab, {0: 3.0}, v, hull=True)
    one = vw.render_frames(lab, {0: 3.0}, v, hull=False)
    mullion = np.zeros((H, W), bool)
    mullion[50:150, 145:155] = True
    assert not hull["area"][mullion].any(), "the mullion stays in front"
    assert hull["back"][:, 145:200].any() and hull["back"][:, 100:145].any(), "one room behind both panes"
    assert not one["back"][:, 155:200].any() or not one["back"][:, 100:145].any()
    return "panes share one room, mullion in front"


def test_depth_rules():
    table = {"regions": [
        {"region_id": 0, "name": "W_F1_C01", "group_id": "Windows", "tags": ["M_FLOOR_F1"]},
        {"region_id": 1, "name": "W_F0_L", "group_id": "Windows", "tags": ["M_FLOOR_F0"]},
        {"region_id": 2, "name": "wall", "group_id": "Groups", "tags": []}]}
    rules, notes = vw.parse_depth_rules("group:windows = 2\nW_F1_* = 3 // deeper\ntag:M_FLOOR_F0 = -1.5\nbad")
    assert len(rules) == 3 and len(notes) == 1
    assert vw.region_depths(table, rules) == {0: 3.0, 1: -1.5}
    assert vw.region_depths(table, rules, default_m=1.0)[2] == 1.0
    return "name / group / tag rules, later wins, default"


def test_place_media():
    canvas = np.zeros((H, W, 3), np.float32)
    media = np.zeros((100, 300, 3), np.float32)
    media[:, :150] = 1.0                             # left half white
    quad = [[100, 100], [300, 100], [300, 200], [100, 200]]
    clip = np.ones((H, W), bool)
    out = vw.place_media(canvas, media, quad, clip)
    assert out[150, 150].sum() > 2.5 and out[150, 280].sum() < 0.5 and out[50, 50].sum() == 0
    return "corner pin with cover-fit"


def test_compose_sources():
    v = facade()
    lab = window(300, 250, 360, 330, rid=0)                      # generate
    lab = window(420, 250, 480, 330, rid=1, labels=lab)          # media
    lab = window(540, 250, 600, 330, rid=2, labels=lab)          # world
    matrix = np.full((H, W, 3), 0.6, np.float32)
    red = np.zeros((50, 80, 3), np.float32)
    red[..., 0] = 1.0
    world = np.zeros((H, W, 3), np.float32)
    world[..., 2] = 1.0
    specs = {0: {"depth": 3.0, "source": "generate"}, 1: {"depth": 3.0, "source": "media", "media": "red"},
             2: {"depth": 3.0, "source": "world"}}
    out, r, notes = vw.compose(matrix, lab, specs, v, media=[red], media_names=["red"], world=world)
    assert not notes, notes
    bk = r["back_id"]
    assert np.allclose(out[bk == 1].mean(axis=0), [1, 0, 0], atol=0.05), "media on the back wall"
    assert np.allclose(out[bk == 2].mean(axis=0), [0, 0, 1], atol=0.01), "world behind the window"
    walls1 = out[r["wall_id"] == 1]
    assert walls1[:, 0].mean() > walls1[:, 1].mean() + 0.2, "media walls tinted with its colour"
    g = out[bk == 0]
    assert np.allclose(g[:, 0], g[:, 1]) and np.allclose(g[:, 1], g[:, 2]), "generate frames keep the grey guide"
    assert (out[lab == -1] == 0.6).all(), "the facade outside the frames is unchanged"
    _, _, notes = vw.compose(matrix, lab, {1: {"depth": 3.0, "source": "media", "media": "nope"}}, v,
                             media=[red], media_names=["red"])
    assert notes and "not found" in notes[0]
    return "generate / media / world in one composite, facade untouched"


def test_parasite_shadow_and_plan():
    v = facade()
    lab = window(380, 150, 420, 190, rid=0)
    lab = window(100, 300, 140, 340, rid=1, labels=lab)          # an ordinary region, untouched
    matrix = np.full((H, W, 3), 0.8, np.float32)
    specs = {0: {"depth": -2.0, "source": "generate"}}
    out, r, notes = vw.compose(matrix, lab, specs, v, shadow_angle_deg=90.0, shadow_length=1.0,
                               shadow_strength=0.5)
    sh = r["shadow"]
    assert sh.any() and not (sh & r["area"]).any()
    ys, xs = np.nonzero(sh)
    assert ys.max() > 190 + 30, "light from above: the shadow falls below, 2 m = 40 px"
    assert out[sh].mean() < 0.6 and (out[lab == 1] == 0.8).all()
    # the plan: the parasite becomes region 2 '<name>_out' covering the whole area
    table = {"regions": [{"region_id": 0, "name": "W_A", "group_id": "Windows", "tags": []},
                         {"region_id": 1, "name": "W_B", "group_id": "Windows", "tags": []}],
             "groups": {"Windows": [0, 1]}}
    plan = {"regions": [{"region_id": 0, "name": "W_A", "strategy": "inpaint", "prompt": "a box", "fif_depth_m": -2.0},
                        {"region_id": 1, "name": "W_B", "strategy": "keep", "prompt": "", "fif_depth_m": 0.0}],
            "order": [0, 1]}
    lab2, table2, plan2 = vw.extend_plan(plan, table, lab, r["area_id"], [0])
    assert len(table2["regions"]) == 3 and table2["regions"][2]["name"] == "W_A_out"
    assert (lab2 == 2).sum() == r["area"].sum() and not (lab2 == 0).any()
    assert plan2["regions"][2]["prompt"] == "a box" and plan2["regions"][2]["strategy"] == "inpaint"
    assert plan2["regions"][0]["strategy"] == "keep" and plan2["order"] == [2, 1]
    assert plan["regions"][0]["strategy"] == "inpaint" and len(table["regions"]) == 2, "inputs unchanged"
    return f"shadow {int(sh.sum())} px below the box, parasite region with the frame's settings"


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    ok = 0
    for t in tests:
        try:
            print(f"PASS {t.__name__}: {t()}")
            ok += 1
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            print(f"FAIL {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{ok}/{len(tests)} passed.")
    sys.exit(0 if ok == len(tests) else 1)
