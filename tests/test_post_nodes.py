"""
Model free test of the post nodes themselves (nodes/post/nodes_post.py) and kubakub project settings
(nodes/project/nodes_project.py): the optional mask of colour match / apply lut / deflicker, the fps and seconds
outputs of retime, the facade width and mm per px outputs of project settings, and that no socket moved.

Run from the ComfyUI_windows_portable folder:
    python_embeded\\python.exe ComfyUI\\custom_nodes\\ComfyUI_KubaNodes\\tests\\test_post_nodes.py
"""

import os
import shutil
import sys
import tempfile

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(HERE))))     # ComfyUI (folder_paths, comfy_api)
sys.path.insert(0, os.path.dirname(HERE))
import folder_paths  # noqa: E402

TMP = tempfile.mkdtemp(prefix="kub_post_nodes_")
folder_paths.set_temp_directory(os.path.join(TMP, "temp"))
folder_paths.set_output_directory(os.path.join(TMP, "output"))
folder_paths.set_input_directory(os.path.join(TMP, "input"))

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))     # embedded Python leaves the script folder off the path
import _pack  # noqa: E402,F401  (the pack as a package, see tests/_pack.py)
from kubapack.kubakub import post as pp  # noqa: E402
from kubapack.nodes.post import nodes_post as npo  # noqa: E402
from kubapack.nodes.project import nodes_project as npr  # noqa: E402

failures = []


def check(name, cond, detail=""):
    print(("ok   " if cond else "FAIL ") + name + (f"  ({detail})" if detail and not cond else ""))
    if not cond:
        failures.append(name)


def sockets(cls):
    s = cls.define_schema()
    return [i.id for i in s.inputs], [o.display_name or o.id for o in s.outputs], s


try:
    rng = np.random.default_rng(0)
    frames = torch.from_numpy((rng.random((5, 48, 64, 3)) * 0.5 + 0.2).astype(np.float32))
    ref = torch.from_numpy((rng.random((1, 20, 20, 3)) * 0.3 + 0.6).astype(np.float32))
    left = torch.zeros((1, 48, 64))
    left[:, :, :32] = 1.0                                             # the left half

    # --- colour match
    ins, outs, sch = sockets(npo.KUBA_ColourMatch)
    check("colour match: old sockets in place, mask added last",
          ins == ["images", "reference", "method", "strength", "save_lut", "mask"] and outs == ["images", "lut_file"],
          f"{ins} {outs}")
    check("colour match: mask is optional, strength has a tooltip",
          sch.inputs[5].optional and bool(sch.inputs[3].tooltip) and bool(sch.inputs[5].tooltip))
    full = npo.KUBA_ColourMatch.execute(frames, ref, "mkl", 1.0, "").result[0]
    part = npo.KUBA_ColourMatch.execute(frames, ref, "mkl", 1.0, "", mask=left).result[0]
    check("colour match: inside the mask = the full result", torch.allclose(part[:, :, :32], full[:, :, :32], atol=1e-6))
    check("colour match: outside the mask = untouched", torch.allclose(part[:, :, 32:], frames[:, :, 32:], atol=1e-6)
          and not torch.allclose(full[:, :, 32:], frames[:, :, 32:], atol=1e-2))
    small = torch.zeros((1, 24, 32))
    small[:, :, :16] = 1.0
    res = npo.KUBA_ColourMatch.execute(frames, ref, "mkl", 1.0, "", mask=small).result[0]
    check("colour match: a mask of another size is resized", torch.allclose(res[:, :, :30], full[:, :, :30], atol=1e-6)
          and torch.allclose(res[:, :, 34:], frames[:, :, 34:], atol=1e-6))
    per = torch.zeros((5, 48, 64))
    per[2] = 1.0
    res = npo.KUBA_ColourMatch.execute(frames, ref, "mkl", 1.0, "", mask=per).result[0]
    check("colour match: a mask per frame is used per frame", torch.allclose(res[2], full[2], atol=1e-6)
          and torch.allclose(res[0], frames[0], atol=1e-6))
    grey = npo.KUBA_ColourMatch.execute(frames, ref, "mkl", 1.0, "", mask=torch.full((48, 64), 0.5)).result[0]
    check("colour match: a grey mask (H x W, no batch) blends half", torch.allclose(grey, (frames + full) / 2, atol=1e-5))

    # --- apply lut
    ins, outs, sch = sockets(npo.KUBA_ApplyLUT)
    check("apply lut: old sockets in place, mask added last", ins == ["images", "lut", "strength", "lut_path", "mask"]
          and outs == ["images"] and sch.inputs[4].optional and bool(sch.inputs[2].tooltip), f"{ins} {outs}")
    cube = os.path.join(TMP, "invert.cube")
    with open(cube, "w", encoding="utf-8") as f:
        f.write("LUT_1D_SIZE 2\n1 1 1\n0 0 0\n")
    full = npo.KUBA_ApplyLUT.execute(frames, "", 1.0, cube).result[0]
    part = npo.KUBA_ApplyLUT.execute(frames, "", 1.0, cube, mask=left).result[0]
    check("apply lut: without a mask as before", torch.allclose(full, 1 - frames, atol=1e-5))
    check("apply lut: only inside the mask", torch.allclose(part[:, :, :32], full[:, :, :32], atol=1e-6)
          and torch.allclose(part[:, :, 32:], frames[:, :, 32:], atol=1e-6))

    # --- deflicker
    ins, outs, sch = sockets(npo.KUBA_Deflicker)
    check("deflicker: old sockets in place, mask added last", ins == ["images", "window", "grid", "strength", "mask"]
          and outs == ["images"] and sch.inputs[4].optional and bool(sch.inputs[3].tooltip), f"{ins} {outs}")
    flick = torch.full((12, 48, 64, 3), 0.5) * (1 + 0.2 * torch.sin(torch.arange(12) * 2.7)).view(12, 1, 1, 1)
    full = npo.KUBA_Deflicker.execute(flick, 9, 1, 1.0).result[0]
    part = npo.KUBA_Deflicker.execute(flick, 9, 1, 1.0, mask=left).result[0]
    check("deflicker: calmer inside the mask, untouched outside", float(part[:, :, :32].mean((1, 2, 3)).std())
          < 0.5 * float(flick.mean((1, 2, 3)).std()) and torch.allclose(part[:, :, 32:], flick[:, :, 32:], atol=1e-6)
          and torch.allclose(part[:, :, :32], full[:, :, :32], atol=1e-6))
    two = npo.KUBA_Deflicker.execute(flick[:2], 9, 1, 1.0, mask=left).result[0]
    check("deflicker: fewer than 3 frames come back unchanged", two.shape[0] == 2 and torch.equal(two, flick[:2]))

    # --- retime
    ins, outs, sch = sockets(npo.KUBA_Retime)
    check("retime: old sockets in place, fps in and fps / seconds out added last",
          ins == ["images", "speed", "frames", "mode", "fps"] and outs == ["images", "frame_count", "fps", "seconds"]
          and sch.inputs[4].optional and bool(sch.inputs[0].tooltip) and all(o.tooltip for o in sch.outputs[2:]),
          f"{ins} {outs}")
    r = npo.KUBA_Retime.execute(frames, 0.5, 0, "blend").result
    check("retime: no fps given = fps 0, seconds 0", r[1] == 9 and r[2] == 0.0 and r[3] == 0.0, str(r[1:]))
    r = npo.KUBA_Retime.execute(frames, 0.5, 0, "blend", fps=25.0).result
    check("retime by speed: the fps stays, 9 frames = 0.36 s", r[0].shape[0] == 9 and r[2] == 25.0 and abs(r[3] - 0.36) < 1e-9,
          str(r[1:]))
    r = npo.KUBA_Retime.execute(frames, 0.5, 10, "nearest", fps=25.0).result
    check("retime by frame count: 5 -> 10 frames play at 50 fps, as long as before (0.2 s)",
          r[1] == 10 and abs(r[2] - 50.0) < 1e-9 and abs(r[3] - 0.2) < 1e-9, str(r[1:]))
    check("retime_fps: one frame keeps the fps", pp.retime_fps(1, 8, 25.0, True) == (25.0, 8 / 25.0))

    # --- burn in
    ins, outs, sch = sockets(npo.KUBA_BurnIn)
    check("burn in: sockets in place, every input has a tooltip",
          ins == ["images", "text", "name", "fps", "start_frame", "position", "size", "opacity"]
          and all(i.tooltip for i in sch.inputs), str([i.id for i in sch.inputs if not i.tooltip]))

    # --- project settings
    ins, outs, sch = sockets(npr.KUBA_Project)
    check("project settings: old outputs in place, facade_width_m and mm_per_px added last",
          outs == ["width", "height", "fps", "name", "viewer", "report", "facade_width_m", "mm_per_px"]
          and all(o.tooltip for o in sch.outputs), str(outs))
    r = npr.KUBA_Project.execute("show", "uhd 3840x2160", 0, 0, 25.0, 40.0, 0.0, -1.0, 1.7, 30.0).result
    check("project settings: 40 m over 3840 px = 10.4 mm per px", r[0] == 3840 and r[6] == 40.0
          and abs(r[7] - 40000 / 3840) < 1e-9 and f"{r[7]:.1f} mm per px" in r[5], str(r[5:]))
    r = npr.KUBA_Project.execute("show", "hd 1920x1080", 0, 0, 25.0, 40.0, 0.0, -1.0, 1.7, 30.0,
                                 matrix=torch.zeros((1, 100, 2000, 3))).result
    check("project settings: a connected matrix sets the width for mm per px", abs(r[7] - 20.0) < 1e-9, str(r[7]))
finally:
    shutil.rmtree(TMP, ignore_errors=True)

print()
print("all post node tests passed" if not failures else f"{len(failures)} FAILED: " + ", ".join(failures))
sys.exit(1 if failures else 0)
