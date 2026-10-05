"""
samples.py

What a node uses when its file or folder field is left empty: the sample facade written in the form that node reads
(a 3D file, a folder of render passes, a folder of ID renders, a cryptomatte EXR, a layered PDF, a mask folder).
So every node runs the moment it is dropped on the canvas, and a pasted path is the only step to your own file.

Each sample is written once into <root>/kubakub_sample/ (root = ComfyUI's temp folder) and reused.
numpy + OpenCV; the PDF needs PyMuPDF, like the Illustrator node itself (tests/test_samples.py).
"""

from __future__ import annotations

import json
import os

import cv2
import numpy as np

from . import sample_facade as sf
from . import sample_model as sm

VERSION = 1                 # raise when a sample changes: it is written again under a new folder name
NOTE = "no {what} given: the built-in sample is used. Paste the path of your own {what} into '{field}'."


def note(what, field):
    return NOTE.format(what=what, field=field)


def _folder(root, kind):
    return os.path.join(root, "kubakub_sample", f"{kind}_v{VERSION}")


def _done(folder):
    return os.path.isfile(os.path.join(folder, "_done.txt"))


def _finish(folder):
    with open(os.path.join(folder, "_done.txt"), "w", encoding="utf-8") as f:
        f.write("kubakub sample, written by kubakub/samples.py\n")


def _png(path, img):
    cv2.imencode(".png", img)[1].tofile(path)


def _bgr8(rgb01):
    return (np.clip(rgb01[..., ::-1], 0, 1) * 255 + 0.5).astype(np.uint8)


def _union(masks, pick):
    out = None
    for name, (_g, m) in masks.items():
        if pick(name):
            out = (m > 0) if out is None else (out | (m > 0))
    return out


def model(root):
    """The sample model (.obj, 24 x 16 m) -> its path. For kubakub scene render."""
    folder = _folder(root, f"model{sm.VERSION}")
    path = os.path.join(folder, "sample_facade.obj")
    if not _done(folder):
        sm.write_obj(path)
        _finish(folder)
    return path


def mask_folder(root, W, H):
    """After Effects style masks of the sample facade at W x H (Groups/, Windows/) -> folder."""
    folder = os.path.join(_folder(root, f"masks_{int(W)}x{int(H)}"), "Masks")
    if not _done(os.path.dirname(folder)):
        sf.write_mask_folder(sf.render(int(W), int(H))["masks"], folder)
        _finish(os.path.dirname(folder))
    return folder


def passes(root, W=1920, H=1080):
    """Render passes of the sample facade (sample_beauty / _depth / _windows / _door + facade_mask) -> folder."""
    folder = _folder(root, "passes")
    if _done(folder):
        return folder
    os.makedirs(folder, exist_ok=True)
    r = sf.render(W, H)
    masks = r["masks"]
    win = _union(masks, lambda n: n.startswith("W_"))
    door = masks["M_Door"][1] > 0
    trim = _union(masks, lambda n: n.startswith(("M_Cornice", "M_Sill", "M_Pilaster")))
    depth = np.zeros((H, W), np.float32)                   # near = white: trim in front, openings behind the wall
    depth[r["silhouette"] > 0] = 0.6
    depth[trim] = 0.8
    depth[win | door] = 0.35
    _png(os.path.join(folder, "sample_beauty.png"), _bgr8(r["image"]))
    _png(os.path.join(folder, "sample_depth.png"), (depth * 65535).astype(np.uint16))
    _png(os.path.join(folder, "sample_windows.png"), win.astype(np.uint8) * 255)
    _png(os.path.join(folder, "sample_door.png"), door.astype(np.uint8) * 255)
    _png(os.path.join(folder, "facade_mask.png"), (r["silhouette"] > 0).astype(np.uint8) * 255)
    _finish(folder)
    return folder


def _srgb8(c):
    c = np.asarray(c, np.float64)
    return (np.where(c <= 0.0031308, c * 12.92, 1.055 * np.power(c, 1 / 2.4) - 0.055) * 255).round().astype(np.uint8)


def _id_colour(i):
    return ((i * 0.37 + 0.25) % 1.0, (i * 0.61 + 0.55) % 1.0, (i * 0.83 + 0.15) % 1.0)


def id_renders(root, W=1920, H=1080):
    """ID renders of the sample facade as a renderer writes them (ids_elements, ids_level + legends, a clay) -> folder."""
    folder = _folder(root, "id_renders")
    if _done(folder):
        return folder
    os.makedirs(folder, exist_ok=True)
    r = sf.render(W, H)
    masks = r["masks"]
    classes = [("wall", lambda n: n.startswith(("M_FLOOR", "M_Attic"))), ("cornices", lambda n: n.startswith(("M_Cornice", "M_Sill"))),
               ("pilasters", lambda n: n.startswith("M_Pilaster")), ("windows", lambda n: n.startswith("W_")),
               ("door", lambda n: n == "M_Door")]
    levels = [(f"level_{n[9:]}" if n.startswith("M_FLOOR_F") else "level_attic", n)
              for n in masks if n.startswith(("M_FLOOR_F", "M_Attic"))]

    def write(name, entries):
        img = np.zeros((H, W, 3), np.uint8)
        lines = []
        for i, (nm, m) in enumerate(entries):
            c = _id_colour(i)
            img[m] = _srgb8(c)
            lines.append(f"{nm:<20} rgb {c[0]:.4f} {c[1]:.4f} {c[2]:.4f}")
        _png(os.path.join(folder, f"ids_{name}.png"), img[..., ::-1])
        with open(os.path.join(folder, f"ids_{name}.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

    write("elements", [(nm, _union(masks, pick)) for nm, pick in classes])       # painted in this order: openings last
    write("level", [(nm, masks[src][1] > 0) for nm, src in sorted(levels)])
    _png(os.path.join(folder, "sample_clay.png"), _bgr8(r["image"]))
    _finish(folder)
    return folder


def cryptomatte(root, W=1280, H=720):
    """A cryptomatte EXR of the sample facade: objects by name, materials, the picture -> its path."""
    from . import exr
    folder = _folder(root, "cryptomatte")
    path = os.path.join(folder, "sample_crypto.exr")
    if _done(folder):
        return path
    os.makedirs(folder, exist_ok=True)
    r = sf.render(W, H)
    order = {"M_FLOOR": 0, "M_Attic": 0, "M_Cornice": 1, "M_Sill": 1, "M_Pilaster": 2, "W_": 3, "M_Door": 3}
    names = sorted((n for n in r["masks"] if n not in ("M_Sky", "M_Ground")),
                   key=lambda n: next(v for k, v in order.items() if n.startswith(k)))
    material = lambda n: "glass" if n.startswith("W_") else "wood" if n == "M_Door" else "stone" if n.startswith(("M_FLOOR", "M_Attic")) else "trim"  # noqa: E731
    obj_name = lambda n: ("window_" + n[2:] if n.startswith("W_") else n[2:]).lower()  # noqa: E731
    ids = np.zeros((H, W), np.uint32)
    mat = np.zeros((H, W), np.uint32)
    mat_hash = {m: 0x3E000001 + 0x00010203 * k for k, m in enumerate(("stone", "trim", "glass", "wood"))}
    manifest = {}
    for k, n in enumerate(names):                         # painted back to front: the walls, the trim, the openings
        h = 0x3F000000 + 0x00010101 * (k + 1)
        manifest[obj_name(n)] = f"{h:08x}"
        sel = r["masks"][n][1] > 0
        ids[sel] = h
        mat[sel] = mat_hash[material(n)]
    cov = (ids > 0).astype(np.float32)
    zero = np.zeros((H, W), np.float32)
    lin = np.where(r["image"] <= 0.04045, r["image"] / 12.92, ((r["image"] + 0.055) / 1.055) ** 2.4).astype(np.float32)
    chan = {"View.CryptoObject00.r": ids.view(np.float32), "View.CryptoObject00.g": cov, "View.CryptoObject00.b": zero,
            "View.CryptoObject00.a": zero, "View.CryptoMaterial00.r": mat.view(np.float32), "View.CryptoMaterial00.g": cov,
            "View.CryptoMaterial00.b": zero, "View.CryptoMaterial00.a": zero,
            "View.Combined.R": np.ascontiguousarray(lin[..., 0]), "View.Combined.G": np.ascontiguousarray(lin[..., 1]),
            "View.Combined.B": np.ascontiguousarray(lin[..., 2])}
    attrs = {"cryptomatte/aaa0001/name": "View.CryptoObject", "cryptomatte/aaa0001/manifest": json.dumps(manifest),
             "cryptomatte/bbb0002/name": "View.CryptoMaterial",
             "cryptomatte/bbb0002/manifest": json.dumps({m: f"{h:08x}" for m, h in mat_hash.items()})}
    exr.write(path, chan, "zip", attrs)
    _finish(folder)
    return path


def illustrator(root, W=960, H=540):
    """A layered PDF of the sample facade (what an .ai file is inside): MASK, LINES, WINDOWS, DOOR layers -> its path."""
    from . import optional
    fitz = optional.need("fitz", "regions from illustrator")
    folder = _folder(root, "illustrator")
    path = os.path.join(folder, "sample_facade.pdf")
    if _done(folder):
        return path
    os.makedirs(folder, exist_ok=True)
    parts = sf.layout(W, H)
    box = {n: b for n, _g, b, _k in parts}
    x0, x1 = box["M_Attic"][0], box["M_Attic"][2]
    y0, y1 = box["M_Attic"][1], box["M_FLOOR_F0"][3]
    doc = fitz.open()
    page = doc.new_page(width=W, height=H)
    ocg = {n: doc.add_ocg(n, on=on) for n, on in (("MASK", True), ("LINES", False), ("WINDOWS", True), ("DOOR", True))}
    for r in (fitz.Rect(0, 0, W, y0), fitz.Rect(0, y1, W, H), fitz.Rect(0, y0, x0, y1), fitz.Rect(x1, y0, W, y1)):
        page.draw_rect(r, fill=(0, 0, 0), color=None, oc=ocg["MASK"])              # everything that is not facade
    page.draw_rect(fitz.Rect(x0, y0, x1, y1), color=(1, 1, 1), width=2, oc=ocg["LINES"])
    for n, _g, (a, b, c, d), _k in parts:                                        # floor lines across the facade
        if n.startswith(("M_FLOOR_F", "M_Attic")) and d < y1:
            page.draw_line((x0, d), (x1, d), color=(1, 1, 1), width=2, oc=ocg["LINES"])
    for n, _g, (a, b, c, d), kind in parts:
        if kind == "glass":
            page.draw_rect(fitz.Rect(a, b, c, d), fill=(0.5, 0.47, 0.95), color=None, oc=ocg["WINDOWS"])
        elif n == "M_Door":
            page.draw_rect(fitz.Rect(a, b, c, d), fill=(0.93, 0.49, 0.3), color=None, oc=ocg["DOOR"])
    doc.save(path)
    doc.close()
    _finish(folder)
    return path
