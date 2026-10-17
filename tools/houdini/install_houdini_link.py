"""
install_houdini_link.py

Adds the kubakub live link to Houdini: writes packages/kubakub_live.json into your Houdini preference folder(s)
(Documents/houdiniXX.Y), pointing at this folder, so Houdini finds the 'kubakub' shelf (tools 'comfy link' and
'send now') and the kubakub_live Python module. Asks before writing; changes nothing else.

    python install_houdini_link.py            asks for every Houdini version found
    python install_houdini_link.py --yes      writes for all of them without asking
    python install_houdini_link.py --remove   removes the package file again

Standard library only. Restart Houdini afterwards, then: shelf tab '+' > Shelves > kubakub.
"""

from __future__ import annotations

import argparse
import configparser
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PACK = os.path.dirname(os.path.dirname(HERE))
NAME = "kubakub_live.json"


def pref_dirs():
    env = os.environ.get("HOUDINI_USER_PREF_DIR", "").replace("__HVER__", "*")
    found = glob.glob(env) if env else []
    docs = os.path.join(os.path.expanduser("~"), "Documents")
    found += glob.glob(os.path.join(docs, "houdini[0-9]*.[0-9]*"))
    return sorted({os.path.abspath(p) for p in found if os.path.isdir(p)})


def live_folder_setting():
    """live_folder from kubakub.ini [settings] (empty = the default both sides use)."""
    cp = configparser.ConfigParser(inline_comment_prefixes=("#", ";"), interpolation=None)
    try:
        cp.read(os.path.join(PACK, "kubakub.ini"), encoding="utf-8")
        return cp.get("settings", "live_folder", fallback="").strip()
    except configparser.Error:
        return ""


def package(folder_setting=""):
    env = [{"KUBAKUB_LIVE": HERE.replace("\\", "/")},
           {"PYTHONPATH": {"value": "$KUBAKUB_LIVE/python", "method": "prepend"}}]
    if folder_setting:
        env.append({"KUBAKUB_LIVE_FOLDER": folder_setting.replace("\\", "/")})
    return {"env": env, "path": "$KUBAKUB_LIVE"}


def main(argv=None):
    ap = argparse.ArgumentParser(description="kubakub live link for Houdini")
    ap.add_argument("--yes", action="store_true", help="write without asking")
    ap.add_argument("--remove", action="store_true", help="remove the package file")
    a = ap.parse_args(argv)
    dirs = pref_dirs()
    if not dirs:
        print("No Houdini preference folder found (Documents/houdiniXX.Y). Start Houdini once, then run this again.")
        return 1
    pkg = package(live_folder_setting())
    for d in dirs:
        target = os.path.join(d, "packages", NAME)
        if a.remove:
            if os.path.isfile(target):
                os.remove(target)
                print("removed", target)
            continue
        if not a.yes and input(f"Add the kubakub live link to {os.path.basename(d)} ({target})? [y/N] ").strip().lower() != "y":
            continue
        os.makedirs(os.path.dirname(target), exist_ok=True)
        with open(target, "w", encoding="utf-8") as f:
            json.dump(pkg, f, indent=2)
        print("wrote", target)
    if not a.remove:
        print("Restart Houdini, then: shelf tab '+' > Shelves > kubakub, and press 'comfy link'.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
