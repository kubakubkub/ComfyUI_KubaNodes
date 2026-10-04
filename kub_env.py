"""Startup self-check of the Kub nodes: what this install lacks, printed once next to the [Kub] line.

Cheap and silent when everything is there: optional packages are looked up with find_spec (never imported), Blender
and ffmpeg are only located (never started), no network. Keep OPTIONAL in step with README "Requirements".
"""

import importlib.util
import shutil

MIN_COMFYUI = (0, 37, 0)          # the V3 node API with advanced inputs, core SAM3 / background removal / H3 guide
TESTED = "ComfyUI 0.37.0, torch 2.10 cu130, Python 3.12, Blender 4.5, Windows 11, RTX 5070 Ti 12 GB"

REQUIRED = [("cv2", "opencv-python-headless"), ("scipy", "scipy"), ("numpy", "numpy"), ("PIL", "Pillow")]
OPTIONAL = [                        # (module, pip package, what needs it)
    ("shapely", "shapely", "regions to vector (outlines)"),
    ("skimage", "scikit-image", "regions to vector (centerlines)"),
    ("fitz", "PyMuPDF", "regions from illustrator, vector PDF"),
    ("imageio", "imageio", "EXR ID maps in regions from id maps"),
    ("trimesh", "trimesh", "mesh to field"),
    ("onnxruntime", "onnxruntime", "onnx style transfer"),
]


def _has(module):
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def comfyui_version():
    try:
        from comfyui_version import __version__
        return tuple(int(x) for x in __version__.split(".")[:3])
    except Exception:  # noqa: BLE001  (not run inside ComfyUI, or an unusual version string)
        return None


def blender():
    try:
        from .kubakub.scene3d.bridge import find_blender
        return find_blender(), ""
    except Exception as e:  # noqa: BLE001
        return None, str(e)


def ffmpeg():
    if _has("imageio_ffmpeg") or shutil.which("ffmpeg"):
        return True
    return False


def problems():
    """-> (missing required packages, [lines about missing optional features])"""
    req = [pip for mod, pip in REQUIRED if not _has(mod)]
    lines = []
    v = comfyui_version()
    if v is not None and v < MIN_COMFYUI:
        lines.append(f"ComfyUI {'.'.join(map(str, v))} is older than {'.'.join(map(str, MIN_COMFYUI))}: some nodes may not load")
    miss = [(pip, what) for mod, pip, what in OPTIONAL if not _has(mod)]
    if miss:
        lines.append("not installed (only these features are off): "
                     + "; ".join(f"{what} -> pip install {pip}" for pip, what in miss))
    exe, why = blender()
    if not exe:
        lines.append(f"Blender not found (scene render, light layers, relight): {why}")
    if not ffmpeg():
        lines.append("ffmpeg not found (video layers, sequence export): pip install imageio-ffmpeg or put ffmpeg on PATH")
    return req, lines


def report():
    try:
        req, lines = problems()
    except Exception as e:  # noqa: BLE001  (the check itself must never break the pack)
        print(f"[Kub] environment check skipped: {e}")
        return
    if req:
        print(f"[Kub] MISSING required packages: pip install {' '.join(req)}")
    for ln in lines:
        print(f"[Kub] {ln}")
