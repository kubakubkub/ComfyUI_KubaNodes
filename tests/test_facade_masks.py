"""
Model free test for the Facade mask logic (kubakub/facade_core.py).

Builds a synthetic grid facade (sky, cornice, wall, 4x8 windows, ground floor,
door) and checks KUBA_FacadeMaskAtlas's three modes on it. Does not start
ComfyUI and loads no model.

Run from the ComfyUI_windows_portable folder:
    python_embeded\\python.exe ComfyUI\\custom_nodes\\ComfyUI_KubaNodes\\tests\\test_facade_masks.py
Add --full to also time color_regions at the contest size (3840x2160).
Preview PNGs are written to tests/test_output/ (git ignored).
"""

import os
import sys
import tempfile
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from kubakub import facade_core as fc  # noqa: E402

OUT = os.path.join(HERE, "test_output")

SKY = (230, 235, 240)
CORNICE = (150, 120, 100)
WALL = (200, 180, 160)
WINDOW = (40, 60, 90)
GROUND = (90, 90, 90)
ROWS, COLS = 4, 8


def layout(scale=1):
    """Rectangles (x0, y0, x1, y1, colour, name) of the synthetic facade at 1920x1080 * scale."""
    s = scale
    rects = [
        (0, 0, 1920, 80, SKY, "sky"),
        (0, 80, 1920, 140, CORNICE, "cornice"),
        (0, 140, 1920, 860, WALL, "wall"),
        (0, 860, 1920, 1080, GROUND, "ground_floor"),
    ]
    for r in range(ROWS):
        for c in range(COLS):
            x, y = 120 + c * 220, 180 + r * 170
            rects.append((x, y, x + 110, y + 150, WINDOW, f"windows_{r * COLS + c + 1:02d}"))
    rects.append((900, 900, 1020, 1080, WINDOW, "door"))
    return [(x0 * s, y0 * s, x1 * s, y1 * s, col, name) for x0, y0, x1, y1, col, name in rects]


EXPECTED_REGIONS = 4 + ROWS * COLS + 1  # sky, cornice, wall, ground, windows, door


def color_matrix(scale=1):
    img = np.zeros((1080 * scale, 1920 * scale, 3), np.uint8)
    for x0, y0, x1, y1, col, _ in layout(scale):
        img[y0:y1, x0:x1] = col
    img[500:505, 60:65] = (255, 0, 0)  # a 5x5 speck that must be merged away
    img = cv2.GaussianBlur(img, (0, 0), 0.8)  # anti-aliased edges, as exported PNGs have
    return img.astype(np.float32) / 255.0


def line_matrix(gap=False):
    img = np.full((1080, 1920, 3), 255, np.uint8)
    for x0, y0, x1, y1, _, _ in layout():
        cv2.rectangle(img, (x0, y0), (x1 - 1, y1 - 1), (0, 0, 0), 3)
    if gap:
        # a 2 px break in the left edge of the first window (x=120, y=180..330)
        img[240:242, 117:124] = 255
    return img.astype(np.float32) / 255.0


def write_mask_folder(folder):
    for x0, y0, x1, y1, _, name in layout():
        m = np.zeros((1080, 1920), np.uint8)
        m[y0:y1, x0:x1] = 255
        if name.startswith("windows"):
            # RGBA, black colour, shape only in alpha: the After Effects case
            rgba = np.zeros((1080, 1920, 4), np.uint8)
            rgba[..., 3] = m
            ok, buf = cv2.imencode(".png", rgba)
        else:
            ok, buf = cv2.imencode(".png", m)
        buf.tofile(os.path.join(folder, f"{name}.png"))


def save_preview(name, image, labels, atlas):
    os.makedirs(OUT, exist_ok=True)
    prev = fc.render_preview(image, labels, atlas)
    bgr = cv2.cvtColor((prev * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    cv2.imencode(".png", bgr)[1].tofile(os.path.join(OUT, f"{name}.png"))


# --------------------------------------------------------------------------
# checks
# --------------------------------------------------------------------------

def check_partition(labels, atlas):
    n = len(atlas["regions"])
    assert labels.min() >= 0, f"{int((labels < 0).sum())} unassigned px"
    assert labels.max() == n - 1
    for r in atlas["regions"]:
        x, y, w, h = r["bbox"]
        m = labels == r["region_id"]
        assert int(m.sum()) == r["area"]
        ys, xs = np.nonzero(m)
        assert (xs.min(), ys.min(), xs.max() + 1 - xs.min(), ys.max() + 1 - ys.min()) == (x, y, w, h)


def window_group(atlas):
    sizes = {g: len(ids) for g, ids in atlas["groups"].items()}
    return max(sizes, key=sizes.get), sizes


def check_reading_order(atlas):
    regs = atlas["regions"]
    assert regs[0]["bbox"][1] == 0, "region 0 should be the top band (sky)"
    wg, _ = window_group(atlas)
    wins = [r for r in regs if r["group_id"] == wg]
    first_row = [r for r in wins if abs(r["bbox"][1] - wins[0]["bbox"][1]) < 10]
    ids = [r["region_id"] for r in first_row]
    xs = [r["bbox"][0] for r in first_row]
    assert len(first_row) == COLS and ids == sorted(ids) and xs == sorted(xs), \
        f"first window row not in reading order: ids {ids} xs {xs}"


def test_color_regions():
    img = color_matrix()
    t = time.perf_counter()
    labels, atlas = fc.build_atlas(img, mode="color_regions", min_region_area=400)
    dt = time.perf_counter() - t
    save_preview("color_regions", img, labels, atlas)
    n = len(atlas["regions"])
    assert n == EXPECTED_REGIONS, f"expected {EXPECTED_REGIONS} regions, got {n}; notes {atlas['notes']}"
    check_partition(labels, atlas)
    wg, sizes = window_group(atlas)
    assert sizes[wg] == ROWS * COLS, f"windows should form one group of {ROWS * COLS}: {sizes}"
    assert len(atlas["groups"]) == 6, f"expected 6 groups, got {sizes}"
    check_reading_order(atlas)
    return f"{n} regions, {len(atlas['groups'])} groups, {dt:.2f}s"


def test_color_names():
    img = color_matrix()
    labels, atlas = fc.build_atlas(img, mode="color_regions",
                                   color_names="#283c5a = windows\n200, 180, 160 = wall\n#ff00ff = nothing")
    g = atlas["groups"]
    assert len(g["windows"]) == ROWS * COLS + 1, "windows + door share the named colour"
    assert len(g["wall"]) == 1
    assert any("matches no region" in s for s in atlas["notes"]), "unmatched colour should be reported"
    return f"groups {sorted(g)}"


def test_no_split():
    img = color_matrix()
    _, atlas = fc.build_atlas(img, mode="color_regions", split_disconnected=False)
    n = len(atlas["regions"])
    assert n == 5, f"one region per colour expected (5), got {n}"
    return f"{n} regions"


def test_drop_small():
    img = color_matrix()
    labels, atlas = fc.build_atlas(img, mode="color_regions", min_region_area=20000,
                                   merge_small_regions=False)
    # windows (16500 px) and the door (21600 px) straddle the limit: windows dropped, door kept
    assert atlas["unassigned_px"] > 0
    assert all(r["area"] >= 20000 for r in atlas["regions"])
    return f"{len(atlas['regions'])} regions kept, {atlas['unassigned_px']} px unassigned"


def test_line_drawing():
    img = line_matrix()
    labels, atlas = fc.build_atlas(img, mode="line_drawing", line_gap_close_px=1)
    save_preview("line_drawing", img, labels, atlas)
    n = len(atlas["regions"])
    assert n == EXPECTED_REGIONS, f"expected {EXPECTED_REGIONS}, got {n}; notes {atlas['notes']}"
    check_partition(labels, atlas)
    wg, sizes = window_group(atlas)
    assert sizes[wg] == ROWS * COLS, f"window group: {sizes}"
    check_reading_order(atlas)
    return f"{n} regions, {len(atlas['groups'])} groups"


def test_line_gap():
    img = line_matrix(gap=True)
    _, leaky = fc.build_atlas(img, mode="line_drawing", line_gap_close_px=0)
    _, closed = fc.build_atlas(img, mode="line_drawing", line_gap_close_px=2)
    assert len(leaky["regions"]) == EXPECTED_REGIONS - 1, "the gap should merge a window into the wall"
    assert len(closed["regions"]) == EXPECTED_REGIONS, "gap closing should separate them again"
    return f"gap open {len(leaky['regions'])}, closed {len(closed['regions'])}"


def test_mask_folder():
    img = color_matrix()
    with tempfile.TemporaryDirectory() as d:
        write_mask_folder(d)
        labels, atlas = fc.build_atlas(img, mode="mask_folder", mask_folder=f'"{d}"')
    save_preview("mask_folder", img, labels, atlas)
    n = len(atlas["regions"])
    assert n == EXPECTED_REGIONS, f"expected {EXPECTED_REGIONS}, got {n}"
    check_partition(labels, atlas)
    g = atlas["groups"]
    assert len(g["windows"]) == ROWS * COLS, f"groups {g.keys()}"
    wall = next(r for r in atlas["regions"] if r["name"] == "wall")
    assert wall["area"] == 1920 * 720 - ROWS * COLS * 110 * 150, "windows must be cut out of the wall"
    assert any("smaller mask won" in s for s in atlas["notes"])
    return f"{n} regions, groups {sorted(g)}"


def test_empty_folder_error():
    with tempfile.TemporaryDirectory() as d:
        try:
            fc.build_atlas(color_matrix(), mode="mask_folder", mask_folder=d)
        except ValueError as e:
            assert "no .png" in str(e)
            return "clear error"
    raise AssertionError("empty folder should raise")


def write_png(path, m):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imencode(".png", m.astype(np.uint8) * 255)[1].tofile(path)


def write_mask_folder_groups(folder):
    """Subfolders like a typical After Effects mask project: Groups/ (range, facade, floors, piers) and Windows/."""
    def rect(x0, y0, x1, y1):
        m = np.zeros((1080, 1920), bool)
        m[y0:y1, x0:x1] = True
        return m
    g, w = os.path.join(folder, "Groups"), os.path.join(folder, "Windows")
    write_png(os.path.join(g, "M_Projection_Range.png"), rect(0, 80, 1920, 1080))  # sky is out
    write_png(os.path.join(g, "M_Facade.png"), rect(0, 0, 1920, 1080))
    write_png(os.path.join(g, "M_FLOOR_F0.png"), rect(0, 860, 1920, 1080))
    write_png(os.path.join(g, "M_FLOOR_F1.png"), rect(0, 80, 1920, 860))
    piers = rect(40, 140, 80, 860) | rect(940, 140, 980, 860) | rect(1840, 140, 1880, 860)
    piers[500, 500] = True  # a speck that must not become a part
    write_png(os.path.join(g, "M_Piers.png"), piers)
    for c in range(3):
        write_png(os.path.join(w, f"W_F1_C{c + 1:02d}.png"), rect(200 + c * 300, 300, 300 + c * 300, 450))


def test_mask_folder_groups():
    img = color_matrix()
    with tempfile.TemporaryDirectory() as d:
        write_mask_folder_groups(d)
        try:
            fc.build_atlas(img, mode="mask_folder", mask_folder=d)
            raise AssertionError("without recursive the root has no PNGs and should raise")
        except ValueError as e:
            assert "no .png" in str(e)
        labels, atlas, scope = fc.build_atlas(
            img, mode="mask_folder", mask_folder=d, recursive=True, min_region_area=400,
            scope_masks="M_Projection_Range", split_masks="M_Piers", group_by="folder",
            with_scope=True)
        _, tagged = fc.build_atlas(
            img, mode="mask_folder", mask_folder=d, recursive=True,
            scope_masks="Groups/M_Projection_Range.png", split_masks="m_piers",
            tag_only_masks="M_FLOOR_*", group_by="folder")
    save_preview("mask_folder_groups", img, labels, atlas)
    regs = {r["name"]: r for r in atlas["regions"]}

    assert "M_Projection_Range" not in regs, "the scope mask must not be a region"
    assert scope is not None and not scope[:80].any() and scope[80:].all()
    assert (labels[:80] == -1).all(), "nothing above the projection range may be a region"
    assert atlas["unassigned_in_scope_px"] == 0, atlas["notes"]

    piers = sorted(n for n in regs if n.startswith("M_Piers"))
    assert piers == ["M_Piers_01", "M_Piers_02", "M_Piers_03"], piers
    assert regs["M_Piers_01"]["bbox"][0] == 40 and regs["M_Piers_03"]["bbox"][0] == 1840, "parts in reading order"
    assert set(atlas["groups"]["M_Piers"]) == {regs[p]["region_id"] for p in piers}

    assert sorted(atlas["groups"]["Windows"]) == sorted(regs[f"W_F1_C0{c}"]["region_id"] for c in (1, 2, 3))
    assert regs["W_F1_C01"]["source"] == "Windows/W_F1_C01.png"
    assert regs["W_F1_C02"]["tags"] == ["M_Facade", "M_FLOOR_F1"], regs["W_F1_C02"]["tags"]
    assert regs["M_Piers_02"]["tags"] == ["M_Facade", "M_FLOOR_F1"], regs["M_Piers_02"]["tags"]
    assert regs["M_FLOOR_F0"]["tags"] == ["M_Facade"]
    # the facade mask only keeps what nothing smaller claims: here nothing, so it vanishes
    assert "M_Facade" not in regs

    tregs = {r["name"]: r for r in tagged["regions"]}
    assert not any(n.startswith("M_FLOOR") for n in tregs), "tag-only masks must not be regions"
    assert "M_FLOOR_F1" in tregs["W_F1_C03"]["tags"]
    assert tregs["M_Facade"]["area"] > 0, "with floors tag-only, the facade keeps the wall"
    return (f"{len(regs)} regions, groups {sorted(atlas['groups'])}; tag-only: "
            f"{len(tregs)} regions")


def test_full_size():
    img = color_matrix(scale=2)
    t = time.perf_counter()
    labels, atlas = fc.build_atlas(img, mode="color_regions", min_region_area=1600)
    t1 = time.perf_counter()
    fc.render_preview(img, labels, atlas)
    t2 = time.perf_counter()
    assert len(atlas["regions"]) == EXPECTED_REGIONS, f"got {len(atlas['regions'])}"
    return f"3840x2160: atlas {t1 - t:.2f}s, preview {t2 - t1:.2f}s"


def main():
    tests = [test_color_regions, test_color_names, test_no_split, test_drop_small,
             test_line_drawing, test_line_gap, test_mask_folder, test_mask_folder_groups,
             test_empty_folder_error]
    if "--full" in sys.argv:
        tests.append(test_full_size)
    failed = 0
    for t in tests:
        try:
            print(f"PASS {t.__name__}: {t()}")
        except Exception as e:  # report every test, not just the first failure
            failed += 1
            print(f"FAIL {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed. Previews in {OUT}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
