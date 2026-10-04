"""
Model free test of where nodes may write (kubakub/save_paths.py) and of the blender_path check (kubakub/scene3d/bridge.py).
Run: python_embeded\\python.exe custom_nodes\\ComfyUI_KubaNodes\\tests\\test_save_paths.py
"""
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from kubakub import save_paths as sp  # noqa: E402
from kubakub.scene3d import bridge  # noqa: E402

results = []


def step(name, ok, detail=""):
    results.append(ok)
    print(("ok   " if ok else "FAIL ") + name + (f"  ({detail})" if detail else ""))


def raises(fn, *a):
    try:
        fn(*a)
    except ValueError as e:
        return str(e)
    return ""


with tempfile.TemporaryDirectory() as tmp:
    out = os.path.join(tmp, "output")
    os.makedirs(out)
    real = sp.anywhere
    sp.anywhere = lambda: False                       # the default of a new install, whatever this install's ini says
    step("empty = nothing to save", sp.save_folder("", out) == "" and sp.save_folder('  "" ', out) == "")
    got = sp.save_folder("kubakub/versions", out)
    step("a folder name lands inside output", got == os.path.realpath(os.path.join(out, "kubakub", "versions")), got)
    step("an absolute path inside output is fine", sp.save_folder(os.path.join(out, "a"), out) == os.path.realpath(os.path.join(out, "a")))
    step("an absolute path elsewhere is refused, with the way to allow it",
         "save_anywhere = on" in raises(sp.save_folder, os.path.join(tmp, "elsewhere"), out))
    step("'..' out of output is refused", bool(raises(sp.save_folder, "../elsewhere", out)))
    step("a network path is refused", bool(raises(sp.save_folder, r"\\server\share\x", out)))
    sp.anywhere = lambda: True
    step("save_anywhere = on: any folder", sp.save_folder(os.path.join(tmp, "elsewhere"), out) == os.path.realpath(os.path.join(tmp, "elsewhere"))
         and sp.save_folder(r"\\server\share\x", out) == os.path.normpath(r"\\server\share\x"))
    sp.anywhere = real
    step("file_stem keeps a plain prefix", sp.file_stem("region_video") == "region_video" and sp.file_stem("shot 01-a.v2") == "shot 01-a.v2")
    step("file_stem drops folders and '..'", sp.file_stem(r"..\..\evil") == "evil" and sp.file_stem("C:/x/y") == "y" and sp.file_stem("..", "d") == "d")

    exe = os.path.join(tmp, "blender.exe" if os.name == "nt" else "blender")
    other = os.path.join(tmp, "calc.exe")
    for f in (exe, other):
        open(f, "wb").close()
    step("blender_path: blender itself is taken", bridge.find_blender(exe) == exe)
    step("blender_path: the folder that holds it is taken", bridge.find_blender(tmp) == exe)
    step("blender_path: another program is refused", "is not blender" in raises(bridge.find_blender, other))
    step("blender_path: a network path is refused", "network path" in raises(bridge.find_blender, r"\\server\share\blender.exe"))

print(f"\n{sum(results)}/{len(results)} passed.")
sys.exit(0 if all(results) else 1)
