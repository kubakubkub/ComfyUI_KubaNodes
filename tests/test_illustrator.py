"""
Model free test for kubakub regions from illustrator (kubakub/illustrator.py).
Writes a small layered PDF (like an .ai saved PDF compatible) into
tests/test_output and reads it back. Does not start ComfyUI.

Run from the ComfyUI_windows_portable folder:
    python_embeded\\python.exe ComfyUI\\custom_nodes\\ComfyUI_KubaNodes\\tests\\test_illustrator.py
"""

import os
import sys

import cv2
import fitz
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from kubakub import illustrator as il  # noqa: E402

OUT = os.path.join(HERE, "test_output")
PDF = os.path.join(OUT, "layers_test.pdf")
W, H = 400, 300      # artboard in points


def make_pdf():
    os.makedirs(OUT, exist_ok=True)
    photo = np.zeros((60, 80, 3), np.uint8)
    photo[:] = (0, 128, 255)                            # RGB test colour
    ok, png = cv2.imencode(".png", photo[..., ::-1])
    doc = fitz.open()
    page = doc.new_page(width=W, height=H)
    ocg = {n: doc.add_ocg(n, on=on) for n, on in
           (("Background", False), ("_IMG", False), ("MASK", True), ("LINES", False), ("OBST", True),
            ("TEXT", True), ("_FLOOR_UP", False))}
    # upper floor area: '_' layers are ignored unless given a role (the test sets 'tag')
    page.draw_rect(fitz.Rect(40, 30, 360, 150), fill=(0.2, 0.8, 0.2), color=None, oc=ocg["_FLOOR_UP"])
    page.draw_rect(fitz.Rect(0, 0, W, H), fill=(0, 0, 0), color=None, oc=ocg["Background"])
    page.insert_image(fitz.Rect(0, 0, W, H), stream=png.tobytes(), oc=ocg["_IMG"])
    # outside = four bars around the 40..360 x 30..270 facade
    for r in (fitz.Rect(0, 0, W, 30), fitz.Rect(0, 270, W, H), fitz.Rect(0, 30, 40, 270),
              fitz.Rect(360, 30, W, 270)):
        page.draw_rect(r, fill=(0, 0, 0), color=None, oc=ocg["MASK"])
    # line drawing: facade outline + one vertical and one horizontal line -> 4 cells
    page.draw_rect(fitz.Rect(40, 30, 360, 270), color=(1, 1, 1), width=2, oc=ocg["LINES"])
    page.draw_line((200, 30), (200, 270), color=(1, 1, 1), width=2, oc=ocg["LINES"])
    page.draw_line((40, 150), (360, 150), color=(1, 1, 1), width=2, oc=ocg["LINES"])
    # two shapes: a pink disc in the upper left cell, a purple square in the lower right one
    page.draw_circle((110, 90), 30, fill=(0.93, 0.49, 0.5), color=None, oc=ocg["OBST"])
    page.draw_rect(fitz.Rect(260, 190, 320, 250), fill=(0.5, 0.47, 0.95), color=None, oc=ocg["OBST"])
    page.insert_text((60, 290), "MATRICE 400x300", fontsize=10, oc=ocg["TEXT"])
    doc.save(PDF)
    doc.close()
    return PDF


def by_name(atlas):
    return {r["name"]: r for r in atlas["regions"]}


def test_read_layers():
    doc, page, layers = il.read_layers(make_pdf())
    assert set(layers) >= {"MASK", "LINES", "OBST", "TEXT", "_IMG", "Background"}, set(layers)
    assert len(layers["LINES"]["paths"]) == 3, "hidden layer must be read"
    assert len(layers["_IMG"]["images"]) == 1, layers["_IMG"]["images"]
    vis = il.file_visibility(PDF)
    assert vis["LINES"] is False and vis["MASK"] is True
    return f"{len(layers)} layers, hidden ones included, image on _IMG"


def test_auto_roles_and_cells():
    r = il.build(PDF, min_region_area=50)
    roles, a = r["roles"], r["atlas"]
    assert roles == {"Background": "ignore", "_IMG": "reference", "MASK": "outside", "LINES": "lines",
                     "OBST": "shapes", "TEXT": "ignore", "_FLOOR_UP": "ignore"}, roles
    names = by_name(a)
    cells = sorted(n for n in names if n.startswith("LINES_"))
    assert cells == ["LINES_001", "LINES_002", "LINES_003", "LINES_004"], cells
    assert names["LINES_001"]["tags"] == ["LINES"]
    assert names["OBST_01"]["tags"][-1] == "#ed7d80" and names["OBST_02"]["tags"][-1] == "#8078f2", \
        (names["OBST_01"]["tags"], names["OBST_02"]["tags"])
    assert "LINES_001" in names["OBST_01"]["tags"], "shape tagged with the cell it was cut from"
    assert r["scope"][10, 10] == False and r["scope"][150, 100] == True  # noqa: E712
    assert (r["labels"][r["scope"] == False] == -1).all()  # noqa: E712
    return f"{len(a['regions'])} regions: 4 cells + 2 shapes, roles auto"


def test_reference_and_lines():
    r = il.build(PDF, min_region_area=50)
    ref = r["reference"]
    assert ref is not None and abs(ref[150, 200, 1] - 128 / 255) < 0.02 and ref[150, 200, 2] > 0.95, ref[150, 200]
    assert r["lines"][150, 120] and not r["lines"][100, 120]
    return "photo placed, lines raster"


def test_scale_to_matrix_and_aspect():
    r = il.build(PDF, width=800, height=600, min_region_area=200)
    assert r["size"] == (800, 600) and r["scale"] == 2.0
    assert len([n for n in by_name(r["atlas"]) if n.startswith("LINES_")]) == 4
    try:
        il.build(PDF, width=800, height=400)
    except ValueError as e:
        assert "aspect" in str(e)
    else:
        raise AssertionError("aspect mismatch must raise")
    return "2x matrix ok, other aspect refused"


def test_roles_override():
    r = il.build(PDF, roles_text="OBST = ignore\n_FLOOR_UP = tag\nMASK = ignore\nTEXT, nothing* = ignore",
                 min_region_area=50)
    names = by_name(r["atlas"])
    assert not any(n.startswith("OBST_") for n in names)
    upper = sorted(n for n, reg in names.items() if "_FLOOR_UP" in reg.get("tags", []))
    assert upper == ["LINES_001", "LINES_002"], upper
    assert r["scope"] is None
    assert any("nothing*" in n for n in r["atlas"]["notes"]), r["atlas"]["notes"]
    return "tag role, ignore, unknown pattern noted"


def test_parse_roles():
    rules, notes = il.parse_roles("A, B = lines\nC = wrong\n// x\nD* = shapes")
    assert rules == [("a", "lines"), ("b", "lines"), ("d*", "shapes")] and len(notes) == 1
    return "roles text"


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
