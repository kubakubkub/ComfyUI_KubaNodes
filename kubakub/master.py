"""
master.py

The master timeline (web/kubakub_master.js): the bottom panel above the directors, where directors and rendered clips
are put one after another. Pure logic for its routes (nodes/master/nodes_master.py); no ComfyUI imports, so the tests
run without a server.

The timeline document lives in the workflow (workflow.extra.kubakub_master). A clip on it is one segment of the
director's video model (kubakub/director/media.py: start / in / out / speed / length / fill / reverse, in seconds), so
a clip behaves on the master timeline exactly as a video layer does in the director.
"""

from __future__ import annotations

import os
import re

VIDEO_EXT = {".mp4", ".mov", ".webm", ".mkv", ".avi", ".m4v", ".gif"}
IMAGE_EXT = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
AUDIO_EXT = {".wav", ".mp3", ".ogg", ".flac", ".m4a", ".aac", ".opus"}
WAVE_RATE = 50                     # waveform peaks per second
LIST_LIMIT = 400


def kind_of(name: str) -> str | None:
    """'video', 'image', 'audio' or None (not something the timeline takes)."""
    ext = os.path.splitext(str(name))[1].lower()
    return "video" if ext in VIDEO_EXT else "image" if ext in IMAGE_EXT else "audio" if ext in AUDIO_EXT else None


def list_media(base: str, limit: int = LIST_LIMIT) -> list[dict]:
    """Videos, pictures and sounds under `base`, newest first: {name (relative, '/'), kind, mtime, size}. Folders starting with
    '.' are skipped (proxies, partial uploads)."""
    out = []
    if not base or not os.path.isdir(base):
        return out
    for root, dirs, files in os.walk(base):
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        for f in files:
            kind = kind_of(f)
            if not kind or f.startswith("."):
                continue
            p = os.path.join(root, f)
            try:
                st = os.stat(p)
            except OSError:
                continue
            out.append({"name": os.path.relpath(p, base).replace(os.sep, "/"), "kind": kind,
                        "mtime": st.st_mtime, "size": st.st_size})
    out.sort(key=lambda e: e["mtime"], reverse=True)
    return out[:limit]


def safe_name(name: str) -> str | None:
    """The file name of an upload as it may be saved ('' or a type the timeline cannot use -> None)."""
    n = os.path.basename(str(name or "").replace("\\", "/"))
    n = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", n).strip(" .")
    return n if n and kind_of(n) else None


def free_name(folder: str, name: str) -> str:
    """`name`, or 'name (1).ext', 'name (2).ext' ... the first one not taken in `folder`."""
    stem, ext = os.path.splitext(name)
    cand, i = name, 1
    while os.path.exists(os.path.join(folder, cand)):
        cand = f"{stem} ({i}){ext}"
        i += 1
    return cand


def resolve(source: str, input_dir: str, output_dir: str) -> str | None:
    """'file:<name in input/>' or 'output:<name in output/>' -> the file (never outside those folders), or None."""
    source = str(source or "")
    base, name = (output_dir, source[7:]) if source.startswith("output:") else (input_dir, source[5:] if source.startswith("file:") else "")
    if not name or not base:
        return None
    base = os.path.realpath(base)
    path = os.path.realpath(os.path.join(base, name))
    return path if path.startswith(base + os.sep) and os.path.isfile(path) else None


def peaks(mono, sr: int, rate: int = WAVE_RATE) -> list[float]:
    """Loudest sample of every 1/rate s of a mono float signal, 0..1, rounded (the waveform on the timeline)."""
    import numpy as np
    a = np.abs(np.asarray(mono, np.float32))
    step = max(1, int(round(sr / rate)))
    n = -(-a.size // step) if a.size else 0
    if not n:
        return []
    a = np.pad(a, (0, n * step - a.size))
    return [round(float(v), 3) for v in np.clip(a.reshape(n, step).max(axis=1), 0, 1)]
