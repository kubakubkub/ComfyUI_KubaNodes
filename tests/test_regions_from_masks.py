"""
Model free test for kubakub regions from masks (facade_core.atlas_from_masks and
parse_mask_names). Does not start ComfyUI and loads no model.

Run from the ComfyUI_windows_portable folder:
    python_embeded\\python.exe ComfyUI\\custom_nodes\\ComfyUI_KubaNodes\\tests\\test_regions_from_masks.py
"""

import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from kubakub import facade_core as fc  # noqa: E402

W, H = 400, 300


def rect(x0, y0, x1, y1, w=W, h=H):
    m = np.zeros((h, w), dtype=bool)
    m[y0:y1, x0:x1] = True
    return m


def entries(pairs):
    return [{"source": f"mask {i}", "name": n, "group": None, "mask": m} for i, (n, m) in enumerate(pairs)]


def by_name(atlas):
    return {r["name"]: r for r in atlas["regions"]}


def test_names():
    got, notes = fc.parse_mask_names("W_F1_01\nW_F1_02 = windows // right one\n\nwall", 5)
    assert got == [("W_F1_01", None), ("W_F1_02", "windows"), ("wall", None),
                   ("mask_04", None), ("mask_05", None)], got
    assert any("without a name" in n for n in notes)
    got, _ = fc.parse_mask_names("a, b, a", 3)
    assert [n for n, _ in got] == ["a", "b", "a_2"], got
    got, notes = fc.parse_mask_names("a\nb\nc", 2)
    assert len(got) == 2 and any("more than masks" in n for n in notes)
    return "names, groups, defaults, duplicates"


def test_plain():
    wall, w1, w2 = rect(0, 0, 400, 300), rect(50, 50, 100, 120), rect(200, 50, 250, 120)
    labels, atlas = fc.atlas_from_masks(entries([("wall", wall), ("win_01", w1), ("win_02", w2)]), W, H)
    r = by_name(atlas)
    assert set(r) == {"wall", "win_01", "win_02"}
    assert r["win_01"]["group_id"] == "win" and r["wall"]["group_id"] == "wall"
    assert r["win_01"]["area"] == 50 * 70, "smaller mask must win over the wall"
    assert r["wall"]["area"] == W * H - 2 * 50 * 70
    assert atlas["unassigned_px"] == 0 and atlas["mode"] == "masks"
    assert (labels[60, 60] == r["win_01"]["region_id"])
    return f"{len(r)} regions, groups {sorted(atlas['groups'])}"


def test_resize_and_scope():
    small = rect(0, 0, 200, 150, 200, 150)            # whole half-size frame -> whole frame
    scope = rect(0, 0, 400, 200)
    labels, atlas = fc.atlas_from_masks(entries([("all", small)]), W, H, scope=scope)
    assert any("resized" in n for n in atlas["notes"])
    assert by_name(atlas)["all"]["area"] == 400 * 200
    assert (labels[250] == -1).all()
    return "resized mask clipped to scope"


def test_patch_on_top():
    # base: two floors of wall, tagged like a mask-folder atlas
    base_labels, base_atlas = fc.atlas_from_masks(
        entries([("F1_wall", rect(0, 0, 400, 150)), ("F0_wall", rect(0, 150, 400, 300)),
                 ("M_FLOOR_1", rect(0, 0, 400, 150))]), W, H, tag_only_masks="M_FLOOR_*")
    assert by_name(base_atlas)["F1_wall"]["tags"] == ["M_FLOOR_1"]
    new = entries([("W_F1_01", rect(50, 40, 110, 110)), ("door", rect(180, 200, 220, 300))])
    labels, atlas = fc.atlas_from_masks(new, W, H, base=(base_labels, base_atlas, None))
    r = by_name(atlas)
    assert set(r) == {"F1_wall", "F0_wall", "W_F1_01", "door"}, set(r)
    assert r["W_F1_01"]["area"] == 60 * 70
    assert r["F1_wall"]["area"] == 400 * 150 - 60 * 70
    assert r["W_F1_01"]["tags"] == ["M_FLOOR_1", "F1_wall"], r["W_F1_01"]["tags"]
    assert r["door"]["tags"] == ["F0_wall"]
    assert r["F1_wall"]["tags"] == ["M_FLOOR_1"], "base tags kept"
    assert r["F1_wall"]["group_id"] == "F1_wall"
    assert atlas["mode"] == "masks+masks"
    return f"window cut out of F1_wall, tags {r['W_F1_01']['tags']}"


def test_patch_underneath_and_removed():
    base_labels, base_atlas = fc.atlas_from_masks(entries([("left", rect(0, 0, 200, 300)),
                                                           ("tiny", rect(300, 0, 320, 20))]), W, H)
    # underneath: the fill only takes the right half, the base stays intact
    labels, atlas = fc.atlas_from_masks(entries([("fill", rect(0, 0, 400, 300))]), W, H,
                                        base=(base_labels, base_atlas, None), patch="underneath")
    r = by_name(atlas)
    assert r["left"]["area"] == 200 * 300 and r["tiny"]["area"] == 400
    assert r["fill"]["area"] == W * H - 200 * 300 - 400
    # on_top: a mask covering 'tiny' completely removes it
    labels, atlas = fc.atlas_from_masks(entries([("cover", rect(290, 0, 330, 30))]), W, H,
                                        base=(base_labels, base_atlas, None))
    assert "tiny" not in by_name(atlas)
    assert any("fully covered" in n and "tiny" in n for n in atlas["notes"]), atlas["notes"]
    return "underneath fills gaps only; covered base region removed with a note"


def test_patch_keeps_base_scope():
    scope = rect(0, 0, 400, 200)
    base_labels, base_atlas, base_scope = fc.atlas_from_masks(
        entries([("wall", rect(0, 0, 400, 300))]), W, H, scope=scope, with_scope=True)
    labels, atlas, sc = fc.atlas_from_masks(entries([("win", rect(10, 150, 60, 280))]), W, H,
                                            base=(base_labels, base_atlas, base_scope), with_scope=True)
    assert by_name(atlas)["win"]["area"] == 50 * 50, "new mask clipped to the base scope"
    assert sc is not None and sc.sum() == 400 * 200
    return "base scope applies to new masks"


def test_errors():
    try:
        fc.atlas_from_masks(entries([("x", np.zeros((H, W), bool))]), W, H)
    except ValueError as e:
        assert "no regions" in str(e)
    else:
        raise AssertionError("empty masks must raise")
    try:
        fc.atlas_from_masks(entries([("x", rect(0, 0, 10, 10))]), W, H,
                            base=(np.zeros((10, 10), np.int32), {"regions": []}, None))
    except ValueError as e:
        assert "expected" in str(e)
    else:
        raise AssertionError("size mismatch must raise")
    return "clear errors"


def test_full_size():
    # 60 window masks patched into a 4K two-region base
    w, h = 3840, 2160
    base_labels, base_atlas = fc.atlas_from_masks(
        [{"source": "a", "name": "top", "group": None, "mask": rect(0, 0, w, h // 2, w, h)},
         {"source": "b", "name": "bottom", "group": None, "mask": rect(0, h // 2, w, h, w, h)}], w, h)
    new = []
    for i in range(60):
        x, y = 100 + (i % 15) * 240, 200 + (i // 15) * 450
        new.append({"source": f"mask {i}", "name": f"W_{i + 1:02d}", "group": None,
                    "mask": rect(x, y, x + 120, y + 250, w, h)})
    t0 = time.perf_counter()
    _, atlas = fc.atlas_from_masks(new, w, h, base=(base_labels, base_atlas, None))
    dt = time.perf_counter() - t0
    assert len(atlas["regions"]) == 62
    return f"60 masks into 4K base in {dt:.2f}s"


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    ok = 0
    for t in tests:
        try:
            print(f"PASS {t.__name__}: {t()}")
            ok += 1
        except Exception as e:  # noqa: BLE001
            print(f"FAIL {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{ok}/{len(tests)} passed.")
    sys.exit(0 if ok == len(tests) else 1)
