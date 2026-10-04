"""
optional.py

Optional packages, imported only when a feature needs them. A missing one says what to install
instead of a bare ModuleNotFoundError. No ComfyUI imports.
"""

from __future__ import annotations

import importlib

PIP = {"shapely": "shapely", "skimage": "scikit-image", "fitz": "PyMuPDF", "imageio": "imageio",
       "trimesh": "trimesh", "onnxruntime": "onnxruntime"}        # module -> pip package (kub_env.OPTIONAL)


def hint(module: str, feature: str) -> str:
    pip = PIP.get(module.split(".")[0], module.split(".")[0])
    return (f"kubakub {feature} needs the package {pip}. Install it and restart ComfyUI:\n"
            f"  python -m pip install {pip}\n"
            f"  (portable ComfyUI: python_embeded\\python.exe -m pip install {pip})")


def need(module: str, feature: str):
    """The imported module, or an ImportError that names the pip package and the command."""
    try:
        return importlib.import_module(module)
    except ImportError as e:
        raise ImportError(hint(module, feature)) from e
