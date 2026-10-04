"""
Model free test for kubakub regions from id maps (kubakub/idmaps.py).
Writes a small Houdini-like render folder into tests/test_output/idmaps and reads it:
- pass 'elements': wall, two windows in front of it, a clock hidden behind the wall;
  object masks (mask_<name>.png) ignore occlusion, the half-size ID map shows what is seen
- pass 'level': only an ID map (no mask files), upper / lower half
Legends are linear like a renderer writes them, the PNG ID maps sRGB encoded.

Run from the ComfyUI_windows_portable folder:
    python_embeded\\python.exe ComfyUI\\custom_nodes\\ComfyUI_KubaNodes\\tests\\test_idmaps.py
"""

import os
import shutil
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from kubakub import idmaps as im  # noqa: E402

D = os.path.join(HERE, "test_output", "idmaps")
W, H = 400, 300
ELEM = [("wall", (1.0, 0.25, 0.375)), ("windows", (0.287, 1.0, 0.25)), ("clock", (1.0, 0.25, 0.938))]
LEVEL = [("level_up", (0.5, 0.25, 1.0)), ("level_down", (0.788, 1.0, 0.25))]


def png(path, arr):
    ok, buf = cv2.imencode(".png", arr)
    buf.tofile(path)


def srgb8(c):
    return (im.to_srgb(np.asarray(c)) * 255).round().astype(np.uint8)


def make_folder():
    shutil.rmtree(D, ignore_errors=True)
    os.makedirs(D)
    wall = np.zeros((H, W), bool)
    wall[20:280, 20:380] = True
    win = np.zeros((H, W), bool)
    win[60:140, 60:140] = True
    win[60:140, 260:340] = True
    clock = np.zeros((H, W), bool)
    clock[180:220, 180:220] = True                      # behind the wall: never in the ID map
    for n, m in (("wall", wall), ("windows", win), ("clock", clock)):
        png(os.path.join(D, f"mask_{n}.png"), m.astype(np.uint8) * 255)
    # visible IDs at half size, sRGB, with one anti-aliased (blended) column at the window edge
    vis = np.full((H, W), -1)
    vis[wall] = 0
    vis[win] = 1
    idm = np.zeros((H, W, 3), np.uint8)
    for i, (_, c) in enumerate(ELEM):
        idm[vis == i] = srgb8(c)
    idm = cv2.resize(idm, (W // 2, H // 2), interpolation=cv2.INTER_NEAREST)
    idm[30:70, 29] = ((idm[30:70, 28].astype(int) + idm[30:70, 30].astype(int)) // 2)
    png(os.path.join(D, "ids_elements.png"), idm[..., ::-1])
    with open(os.path.join(D, "ids_elements.txt"), "w") as f:
        for n, c in ELEM:
            f.write(f"{n:<20} rgb {c[0]:.3f} {c[1]:.3f} {c[2]:.3f}\n")
    lv = np.zeros((H // 2, W // 2, 3), np.uint8)
    lv[10:75, 10:190] = srgb8(LEVEL[0][1])
    lv[75:140, 10:190] = srgb8(LEVEL[1][1])
    png(os.path.join(D, "ids_level.png"), lv[..., ::-1])
    with open(os.path.join(D, "ids_level.txt"), "w") as f:
        for n, c in LEVEL:
            f.write(f"{n} rgb {c[0]} {c[1]} {c[2]}\n")
    clay = np.full((H, W, 3), 128, np.uint8)
    png(os.path.join(D, "facade_clay.png"), clay)


def by_name(atlas):
    return {r["name"]: r for r in atlas["regions"]}


def test_legend_and_discover():
    make_folder()
    leg = im.parse_legend("a rgb 1 0.5 0\nb = #ff8000\nc 255, 128, 0 // comment\n\n")
    assert [n for n, _ in leg] == ["a", "b", "c"] and np.allclose(leg[2][1], [1, 128 / 255, 0])
    passes, ref = im.discover(D)
    assert set(passes) == {"elements", "level"}, set(passes)
    assert passes["level"]["masks"] == {} and len(passes["elements"]["masks"]) == 3
    assert os.path.basename(ref) == "facade_clay.png"
    return "legend forms, passes, reference"


def test_decode_srgb_auto():
    passes, _ = im.discover(D)
    lab, enc, notes = im.decode_idmap(im.read_image(passes["elements"]["idmap"]), passes["elements"]["legend"])
    assert enc == "srgb", enc
    assert (lab[30:70, 29] >= 0).all(), "blended edge pixels filled from neighbours"
    assert any("clock" in n for n in notes)
    return "sRGB detected, AA edge filled, hidden clock noted"


def test_build_id_map():
    r = im.build(D, min_region_area=50)
    names = by_name(r["atlas"])
    assert {"wall", "windows_01", "windows_02"} <= set(names), set(names)
    assert "clock" not in names, "clock is hidden in the ID map"
    assert names["windows_01"]["area"] == 80 * 80, names["windows_01"]["area"]
    assert names["windows_01"]["group_id"] == "windows"
    assert names["windows_01"]["tags"] == ["level_up"], names["windows_01"]["tags"]
    assert names["wall"]["area"] == 260 * 360 - 2 * 80 * 80
    assert any("hidden" in n and "clock" in n for n in r["atlas"]["notes"])
    assert r["reference"] is not None and r["reference"].shape == (H, W, 3)
    return f"{len(names)} regions; windows in front, clock hidden, level tags from the ID map"


def test_build_smaller_wins_and_options():
    r = im.build(D, overlap="smaller_wins", split_parts="", tag_passes="", min_region_area=50)
    names = by_name(r["atlas"])
    assert set(names) == {"wall", "windows", "clock"}, set(names)
    assert names["clock"]["area"] == 40 * 40 and "tags" not in names["clock"]
    r2 = im.build(D, width=800, height=600, min_region_area=50)
    assert r2["size"] == (800, 600) and by_name(r2["atlas"])["windows_01"]["area"] == 160 * 160
    try:
        im.build(D, regions_pass="bay")
    except ValueError as e:
        assert "passes" in str(e)
    else:
        raise AssertionError("unknown pass must raise")
    return "smaller_wins shows the clock, no split, no tags, 2x size, clear error"


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
