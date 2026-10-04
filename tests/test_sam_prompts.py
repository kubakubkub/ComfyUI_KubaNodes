"""
Model free test for the SAM3 Masks helpers (kubakub/sam_prompts.py).
Does not start ComfyUI and loads no model.

Run from the ComfyUI_windows_portable folder:
    python_embeded\\python.exe ComfyUI\\custom_nodes\\ComfyUI_KubaNodes\\tests\\test_sam_prompts.py
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "kubakub"))
import sam_prompts as sp  # noqa: E402


def test_prompt_lines():
    got, notes = sp.parse_prompt_lines(
        "window = window : 40\nballustrade = stone balustrade, carved\ndoor // the entrance\n\nx = : 3")
    assert got[0] == ("window", "window", 40), got
    assert got[1] == ("ballustrade", "stone balustrade  carved", 50), got[1]
    assert got[2] == ("door", "door", 50), got[2]
    assert len(got) == 3 and notes, notes
    got, _ = sp.parse_prompt_lines("round window")
    assert got == [("round_window", "round window", 50)]
    return f"{len(got)} line forms"


def test_points():
    assert sp.parse_points('[{"x": 100, "y": 50}, {"x": 3, "y": 4}]', 800, 600) == [(100, 50), (3, 4)]
    assert sp.parse_points('[{"x": 0.5, "y": 0.25}]', 800, 600) == [(400, 150)]
    assert sp.parse_points("[[10, 20]]", 800, 600) == [(10, 20)]
    assert sp.parse_points("", 800, 600) == [] and sp.parse_points(None, 8, 6) == []
    assert sp.parse_points('[{"x": 1, "y": 1}]', 800, 600) == [(1, 1)], "integer 1 stays a pixel"
    return "pixels, normalized, pairs, empty"


def test_boxes():
    assert sp.parse_boxes([(10, 20, 110, 70)]) == [(10, 20, 110, 70)]
    assert sp.parse_boxes([(110, 70, 10, 20)]) == [(10, 20, 110, 70)], "flipped corners"
    assert sp.parse_boxes([{"x": 5, "y": 6, "width": 10, "height": 4}]) == [(5, 6, 15, 10)]
    assert sp.parse_boxes('[[1, 2, 3, 4]]') == [(1, 2, 3, 4)]
    assert sp.parse_boxes([(5, 5, 5, 9)]) == [], "empty box dropped"
    return "kj tuples, core dicts, json"


def test_detail_window():
    W, H = 3840, 2160
    win = sp.detail_window((1000, 500, 1100, 700), W, H)
    x0, y0, x1, y1 = win
    assert x1 - x0 == 300 and y1 - y0 == 300, win          # square, side 200 * 1.5
    assert x0 <= 1000 and x1 >= 1100 and y0 <= 500 and y1 >= 700
    x0, y0, x1, y1 = sp.detail_window((1000, 1400, 1600, 1500), W, H)
    assert x1 - x0 == y1 - y0 == 900, "long balustrade: square crop, not stretched"
    x0, y0, x1, y1 = sp.detail_window((0, 0, 50, 50), W, H)
    assert (x0, y0) == (0, 0) and x1 == 256, "clamped at the corner"
    x0, y0, x1, y1 = sp.detail_window((3800, 2100, 3840, 2160), W, H)
    assert (x1, y1) == (W, H)
    assert sp.detail_window((100, 100, 3700, 2000), W, H) is None, "big object: no detail pass"
    return f"window {win}"


def test_mask_bbox_and_numbered():
    m = np.zeros((50, 80), bool)
    m[10:20, 30:45] = True
    assert sp.mask_bbox(m) == (30, 10, 45, 20)
    assert sp.mask_bbox(np.zeros((5, 5), bool)) is None
    assert sp.numbered(["w", "w", "door", "w"]) == ["w_01", "w_02", "door", "w_03"]
    assert sp.safe_name(" W F1/01? ") == "W_F101" and sp.safe_name("", "x") == "x"
    return "bbox, numbering, safe names"


def test_parts_at():
    m = np.zeros((100, 200), bool)
    m[10:40, 10:60] = True        # clicked object
    m[70:72, 150:152] = True      # speck
    m[50:60, 100:130] = True      # another piece
    got = sp.parts_at(m, [(20, 20)])
    assert got.sum() == 30 * 50, got.sum()
    got = sp.parts_at(m, [(20, 20), (110, 55)])
    assert got.sum() == 30 * 50 + 10 * 30
    got = sp.parts_at(m, [(190, 5)])
    assert got.sum() == 30 * 50, "no hit: largest part"
    return "clicked parts kept, specks dropped"


def test_container_hits():
    def r(x0, y0, x1, y1):
        m = np.zeros((200, 400), bool)
        m[y0:y1, x0:x1] = True
        return m
    row = r(0, 100, 400, 150)
    singles = [r(10, 105, 80, 145), r(110, 105, 180, 145), r(210, 105, 280, 145)]
    surround, glass = r(300, 10, 380, 90), r(310, 20, 370, 80)
    got = sp.container_hits([row] + singles + [surround, glass])
    assert got == [0], got
    return "row dropped, surround + glass kept"


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
