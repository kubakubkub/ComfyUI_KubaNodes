"""
menu_switches.py

kubakub.ini next to this file switches whole menu groups off, so everyone sees only the tools they use:

    [menu]
    3d = off            # everything under kubakub/3d
    3d/fabricate = off  # only the fabrication tools
    lab = off

Keys are menu paths below "kubakub/". A switched-off node is not registered at all: workflows that use it show it
as missing until it is switched on again. No file = everything on.
"""

from __future__ import annotations

import configparser
import os

INI = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kubakub.ini")
OFF = {"off", "false", "no", "0", "hide", "hidden"}


def read_switches(path: str = INI) -> list[str]:
    """Menu paths switched off, e.g. ['3d', '2d/motion'] (empty without a file)."""
    if not os.path.isfile(path):
        return []
    cp = configparser.ConfigParser(inline_comment_prefixes=("#", ";"))
    cp.optionxform = str                      # keep the case of menu paths
    with open(path, encoding="utf-8") as f:
        cp.read_string(f.read())
    if not cp.has_section("menu"):
        return []
    return [k.strip().strip("/") for k, v in cp.items("menu") if v.strip().lower() in OFF]


def category_of(cls) -> str:
    """The node's menu category (V3 schema or the old CATEGORY attribute)."""
    if hasattr(cls, "define_schema"):
        try:
            return cls.define_schema().category or ""
        except Exception:
            pass
    return getattr(cls, "CATEGORY", "") or ""


def is_off(category: str, off: list[str]) -> bool:
    for key in off:
        base = "kubakub/" + key
        if category == base or category.startswith(base + "/"):
            return True
    return False


def apply(classes: dict, names: dict, path: str = INI) -> list[str]:
    """Remove switched-off nodes from the mappings (in place); returns the removed node ids."""
    off = read_switches(path)
    if not off:
        return []
    gone = [k for k, c in classes.items() if is_off(category_of(c), off)]
    for k in gone:
        classes.pop(k, None)
        names.pop(k, None)
    print(f"[Kub] kubakub.ini: {', '.join(off)} off ({len(gone)} nodes hidden)")
    return gone
