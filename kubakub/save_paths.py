"""
save_paths.py

Where a node may write when a workflow names the folder (save_folder and the like). A workflow can come from anyone,
so a folder typed into a node stays inside ComfyUI's output folder unless the owner of the install allows more:

    [settings]
    save_anywhere = on      # kubakub.ini: save_folder inputs may name any folder on this computer

Pure logic (no ComfyUI import): the caller passes the output folder.
"""

from __future__ import annotations

import os
import re


def _pack_settings():
    """The pack's settings.py (kubakub.ini [settings]); loaded by path when this module is imported on its own."""
    try:
        from .. import settings
        return settings
    except ImportError:
        import importlib.util
        p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "settings.py")
        spec = importlib.util.spec_from_file_location("kubakub_pack_settings", p)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod


kst = _pack_settings()

HOW = "Type a folder name (it is created inside ComfyUI's output folder), or allow any folder with\n" \
      "    save_anywhere = on\nin kubakub.ini [settings] (next to kubakub.ini.example in the pack folder)."


def anywhere() -> bool:
    return kst.switch("save_anywhere", False)


def is_network(path: str) -> bool:
    """A UNC or device path (\\\\server\\share, //server/share, \\\\?\\...): opening it talks to another computer."""
    return (path or "").strip().strip('"').replace("/", "\\").startswith("\\\\")


def save_folder(value: str, output_dir: str, what: str = "save_folder") -> str:
    """'' -> ''. A relative folder -> inside output_dir. An absolute folder, or one that climbs out with '..',
    only with save_anywhere = on; else a ValueError that says how to allow it."""
    p = os.path.expandvars(os.path.expanduser((value or "").strip().strip('"').strip("'")))
    if not p:
        return ""
    base = os.path.realpath(output_dir)
    if is_network(p):
        full = os.path.normpath(p)                       # never resolved: that would already open the share
    elif os.path.isabs(p) or os.path.splitdrive(p)[0]:
        full = os.path.realpath(p)
    else:
        full = os.path.realpath(os.path.join(base, p))
    a, b = os.path.normcase(full), os.path.normcase(base)
    inside = a == b or a.startswith(b + os.sep)
    if not inside and not anywhere():
        raise ValueError(f"{what}: '{value.strip()}' is outside ComfyUI's output folder.\n{HOW}")
    return full


def file_stem(name: str, default: str = "frame") -> str:
    """A file name start from a typed prefix: no folders, no '..', only letters, digits, '_', '-', '.', spaces."""
    stem = re.sub(r"[^\w.\- ]+", "_", os.path.basename((name or "").replace("\\", "/")).strip()).strip(". ")
    return stem or default
