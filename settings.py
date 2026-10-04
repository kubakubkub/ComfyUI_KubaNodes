"""
settings.py

Pack settings with plain names, in kubakub.ini next to this file (the same file as the menu switches):

    [settings]
    region_cache_mb = 4096      # region sampler results (0 = off)
    cond_cache_mb = 2048        # prompt encodings (0 = off)
    director_seq_cache = 2      # director sequence results kept (0 = off)
    cache_folder =              # scene render / relight / walkthrough cache (empty = user/kubakub_cache)
    cache_gb = 20               # its size cap
    blender =                   # blender.exe (empty = Program Files / PATH)
    blender_worker = on         # keep one Blender process loaded between jobs
    blender_idle = 600          # seconds before that process quits

Order: the key in kubakub.ini, then the old environment variable of the same setting (KUBA_ + the key in capitals,
kept so older setups still work), then the default. No file = the defaults.
"""

from __future__ import annotations

import configparser
import os

INI = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kubakub.ini")
OLD_ENV = {"cache_folder": "KUBA_CACHE"}      # old names that are not KUBA_ + the key
_cache = {}


def _ini():
    try:
        st = os.stat(INI)
    except OSError:
        return {}
    key = (st.st_mtime_ns, st.st_size)
    if _cache.get("key") != key:
        cp = configparser.ConfigParser(inline_comment_prefixes=("#", ";"), interpolation=None)   # paths may hold %
        try:
            with open(INI, encoding="utf-8") as f:
                cp.read_string(f.read())
            vals = {k.strip().lower(): v.strip() for k, v in cp.items("settings")} if cp.has_section("settings") else {}
        except (OSError, configparser.Error):
            vals = {}
        _cache.update(key=key, vals=vals)
    return _cache["vals"]


def get(key, default=""):
    """The setting as a string: kubakub.ini [settings], else the old environment variable, else the default."""
    v = _ini().get(key, "")
    if v != "":
        return v
    v = os.environ.get(OLD_ENV.get(key, "KUBA_" + key.upper()), "").strip()
    return v if v != "" else str(default)


def number(key, default):
    try:
        return float(get(key, default))
    except ValueError:
        return float(default)


def switch(key, default=True):
    return get(key, "on" if default else "off").strip().lower() not in ("off", "false", "no", "0")


def save(key, value, path=None):
    """Writes key = value into the [settings] section of kubakub.ini, keeping every other line and comment."""
    path = path or INI
    lines = open(path, encoding="utf-8").read().splitlines() if os.path.isfile(path) else []
    start = next((i for i, ln in enumerate(lines) if ln.strip().lower() == "[settings]"), None)
    entry = f"{key} = {value}"
    if start is None:
        lines += ([""] if lines and lines[-1].strip() else []) + ["[settings]", entry]
    else:
        end = next((i for i in range(start + 1, len(lines)) if lines[i].strip().startswith("[")), len(lines))
        hit = next((i for i in range(start + 1, end)
                    if lines[i].split("=", 1)[0].strip().lower() == key and not lines[i].lstrip().startswith(("#", ";"))), None)
        if hit is None:
            lines.insert(end, entry)
        else:
            lines[hit] = entry
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    _cache.clear()
