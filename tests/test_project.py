"""
Model free test of kubakub project settings (kubakub/project.py) and the kubakub.ini menu switches
(menu_switches.py).

Run from the ComfyUI_windows_portable folder:
    python_embeded\\python.exe ComfyUI\\custom_nodes\\ComfyUI_KubaNodes\\tests\\test_project.py
"""

import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import menu_switches as ms  # noqa: E402
from kubakub import project as pj  # noqa: E402

failures = []


def check(name, cond, detail=""):
    print(("ok   " if cond else "FAIL ") + name + (f"  ({detail})" if detail else ""))
    if not cond:
        failures.append(name)


# size: matrix > preset > custom
check("preset size", pj.resolve_size("uhd 3840x2160", 100, 100) == (3840, 2160, "preset"))
check("custom size", pj.resolve_size("custom", 3200, 2160) == (3200, 2160, "custom"))
check("a matrix wins", pj.resolve_size("hd 1920x1080", 1, 1, (1, 2160, 4096, 3)) == (4096, 2160, "matrix"))

p = pj.build("Night / show: 1", "uhd 3840x2160", 0, 0, 25.0, 40.0, 0.0, -1.0, 1.7, 15.0)
check("safe name", p["name"] == "Night_show_1", p["name"])
check("empty name -> project", pj.safe_name("  ") == "project")
check("viewer matches the size", p["viewer"].width_px == 3840 and p["viewer"].height_px == 2160)
check("viewer centre and distance", p["viewer"].x_m is None and p["viewer"].distance_m == 15.0)
check("mm per px in the report", "10.4 mm per px" in p["report"], p["report"])
check("fps is a float", isinstance(p["fps"], float) and p["fps"] == 25.0)


# menu switches
class V3:  # stands in for an io.ComfyNode
    def __init__(self, cat):
        self.cat = cat

    def define_schema(self):
        return type("S", (), {"category": self.cat})()


class Old:
    CATEGORY = "kubakub/lab/mosaic"


with tempfile.TemporaryDirectory() as d:
    ini = os.path.join(d, "kubakub.ini")
    check("no file = nothing off", ms.read_switches(ini) == [])
    with open(ini, "w", encoding="utf-8") as f:
        f.write("[menu]\n3d = off   # no 3d\n2d/motion = on\nlab/ = OFF\n2d/regions = maybe\n")
    check("read off keys", sorted(ms.read_switches(ini)) == ["3d", "lab"], str(ms.read_switches(ini)))
    classes = {"a": V3("kubakub/3d/scene"), "b": V3("kubakub/3d/fabricate/relief"), "c": V3("kubakub/2d/motion"),
               "d": Old, "e": V3("kubakub/3dx"), "f": V3("kubakub/2d/regions")}
    names = {k: k for k in classes}
    gone = ms.apply(classes, names, ini)
    check("3d and lab hidden, prefix exact", sorted(gone) == ["a", "b", "d"], str(sorted(gone)))
    check("names removed too", sorted(names) == ["c", "e", "f"])

print("\n" + ("ALL OK" if not failures else f"{len(failures)} FAILED: {failures}"))
sys.exit(1 if failures else 0)
