"""
Model free test for kubakub regions to vector (kubakub/vector.py). Writes SVG / PDF / DXF into
tests/test_output/vector and reads them back. Does not start ComfyUI.

Run from the ComfyUI_windows_portable folder:
    python_embeded\\python.exe ComfyUI\\custom_nodes\\ComfyUI_KubaNodes\\tests\\test_vector.py
"""

import os
import shutil
import sys
import xml.etree.ElementTree as ET

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from kubakub import vector as vec  # noqa: E402

OUT = os.path.join(HERE, "test_output", "vector")
W, H = 200, 120


def scene():
    lab = np.full((H, W), -1, np.int32)
    lab[10:50, 10:50] = 0                                   # square 40 x 40
    ring = np.zeros((H, W), np.uint8)
    cv2.circle(ring, (110, 40), 30, 1, -1)
    cv2.circle(ring, (110, 40), 14, 0, -1)
    lab[ring > 0] = 1                                        # ring with a hole
    lab[80:85, 20:180] = 2                                   # thin bar
    lab[95:100, 20:120] = 2                                  # second bar, same region
    table = {"regions": [{"region_id": 0, "name": "square", "group_id": "blocks"},
                         {"region_id": 1, "name": "ring", "group_id": "blocks"},
                         {"region_id": 2, "name": "bars", "group_id": "lines"}]}
    return lab, table


def test_trace_square_exact():
    lab, _ = scene()
    shapes = vec.trace_mask(lab == 0, simplify_px=1.0)
    assert len(shapes) == 1 and not shapes[0][1]
    outer = shapes[0][0]
    assert len(outer) == 4, outer
    assert abs(abs(vec._area(outer)) - 1600) < 1.0, vec._area(outer)   # pixel edges, not centres
    segs = vec.smooth_path(outer, corner_deg=35)
    assert all(s[0] in ("M", "L") for s in segs), "straight edges stay straight"
    return f"square: 4 corners, area {abs(vec._area(outer)):.0f} px"


def test_diagonal_parts_kept():
    # three squares touching only at corner pixels: OpenCV traces them as one self-touching contour
    m = np.zeros((60, 60), bool)
    m[5:20, 5:20] = True
    m[20:35, 20:35] = True
    m[35:50, 5:20] = True
    m[10:15, 10:15] = False                                   # a hole in the first square
    shapes = vec.trace_mask(m, simplify_px=0.5)
    total = sum(abs(vec._area(o)) - sum(abs(vec._area(h)) for h in hs) for o, hs in shapes)
    assert abs(total - m.sum()) < 0.03 * m.sum(), (total, m.sum(), len(shapes))
    assert sum(len(hs) for _, hs in shapes) == 1, "the hole goes to the square that holds it"
    return f"{len(shapes)} outlines, area {total:.0f} of {int(m.sum())} px"


def test_trace_ring_curves():
    lab, _ = scene()
    shapes = vec.trace_mask(lab == 1, simplify_px=1.0)
    assert len(shapes) == 1 and len(shapes[0][1]) == 1, "outer + one hole"
    segs = vec.smooth_path(shapes[0][0], corner_deg=35)
    curves = sum(1 for s in segs if s[0] == "C")
    assert curves > 0.8 * (len(segs) - 1), (curves, len(segs))
    pts = vec.flatten(segs, 1.0)
    r = np.linalg.norm(pts - [110, 40], axis=1)
    assert 29.5 < r.mean() < 31.5 and r.std() < 0.8, (r.mean(), r.std())
    return f"ring: {curves} Bezier segments, radius {r.mean():.2f} px"


def test_centerlines_and_order():
    lab, _ = scene()
    lines = vec.centerlines(lab == 2, simplify_px=1.0)
    assert len(lines) == 2, [len(l) for l in lines]
    lengths = sorted(float(np.linalg.norm(np.diff(l, axis=0), axis=1).sum()) for l in lines)
    assert 90 < lengths[0] < 101 and 150 < lengths[1] < 161, lengths
    for l in lines:
        assert np.abs(l[:, 1] - l[:, 1].mean()).max() < 1.0, "a straight bar gives a straight line"
    order, travel = vec.order_paths(lines)
    assert sorted(i for i, _ in order) == [0, 1] and travel < 150
    return f"two bars -> lines of {lengths[0]:.0f} and {lengths[1]:.0f} px, travel {travel:.0f} px"


def test_document_and_files():
    lab, table = scene()
    shutil.rmtree(OUT, ignore_errors=True)
    doc = vec.build(lab, table, [0, 1], mode="outline", width_mm=100.0)
    assert doc["unit"] == "mm" and doc["scale"] == 0.5 and doc["width"] == 100.0
    assert list(doc["layers"]) == ["blocks"] and len(doc["layers"]["blocks"]) == 2
    files = vec.write_all(doc, OUT, "t")
    svg = ET.parse(files[0]).getroot()
    ns = {"s": "http://www.w3.org/2000/svg"}
    assert svg.get("width") == "100mm"
    ids = [p.get("id") for p in svg.findall(".//s:path", ns)]
    assert ids == ["square", "ring"], ids
    import fitz
    pdf = fitz.open(files[1])
    assert [o["name"] for o in pdf.get_ocgs().values()] == ["blocks"]
    assert abs(pdf[0].rect.width - 100 / 25.4 * 72) < 0.01
    dxf = open(files[2]).read()
    assert dxf.count("POLYLINE") == 3 and "AC1009" in dxf, "square, ring outer, ring hole"
    again = vec.write_all(doc, OUT, "t")
    assert again[0].endswith("t_00002.svg"), "earlier files are kept"
    cl = vec.build(lab, table, [2], mode="centerline", width_mm=100.0)
    assert cl["stats"]["paths"] == 2 and 120 < cl["stats"]["length"] < 132, cl["stats"]
    assert "POLYLINE" in vec.to_dxf(cl) and "<polyline" in vec.to_svg(cl)
    return f"svg / pdf (layer 'blocks') / dxf written, centerlines {cl['stats']['length']:.1f} mm"


def test_kerf():
    lab, table = scene()
    plain = vec.build(lab, table, [0], mode="outline", width_mm=100.0, corner_deg=180)
    kerf = vec.build(lab, table, [0], mode="outline", width_mm=100.0, kerf_mm=1.0)
    a0 = abs(vec._area(vec.flatten(plain["layers"]["blocks"][0]["rings"][0])))
    a1 = abs(vec._area(vec.flatten(kerf["layers"]["blocks"][0]["rings"][0])))
    # 20 x 20 mm square grown by 0.5 mm on each side -> 21 x 21
    assert abs(a0 - 400) < 1 and abs(a1 - 441) < 2, (a0, a1)
    return f"20 mm square {a0:.0f} mm2 -> kerf 1 mm {a1:.0f} mm2"


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
