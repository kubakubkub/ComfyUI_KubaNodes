"""
bridge.py

Runs blender_export.py in a headless Blender (a worker that stays open, or a one-off subprocess) and caches the
result per (file, size, mtime, camera, resolution, frame, script version, files the job reads). Blender is never
imported into ComfyUI's Python. Results are written to a staging folder and renamed into place, so two ComfyUI
processes can share one cache folder. purge() keeps the cache under a size cap (least recently used first).
No ComfyUI imports (tests/test_scene3d.py, tests/test_scene_cache.py).
"""

from __future__ import annotations

import glob
import hashlib
import json
import os
import re
import shutil
import subprocess
import time
import uuid

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(HERE, "blender_export.py")
SCRIPTS = (SCRIPT, os.path.join(HERE, "autocam.py"))      # what runs inside Blender: a change is a new cache key
FORMATS = (".blend", ".fbx", ".obj", ".abc", ".glb", ".gltf", ".stl", ".ply", ".usd", ".usda", ".usdc", ".usdz")
STAGE = ".part-"                                      # marks a folder / file that is still being written


def _pack_settings():
    """The pack's settings.py (kubakub.ini [settings]); loaded by path when this module is imported on its own."""
    try:
        from ... import settings
        return settings
    except ImportError:
        import importlib.util
        p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "settings.py")
        spec = importlib.util.spec_from_file_location("kubakub_pack_settings", p)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod


kst = _pack_settings()


def find_blender(explicit: str = "") -> str:
    """Explicit path, setting blender (kubakub.ini), the newest 'Blender Foundation' install, or blender on PATH."""
    for i, cand in enumerate((explicit.strip().strip('"'), kst.get("blender", "").strip().strip('"'))):
        if cand:
            if i == 0:          # typed into a node, so it can come from a stranger's workflow: only a local blender
                name = os.path.basename(cand.replace("\\", "/").rstrip("/")).lower()
                if cand.replace("/", "\\").startswith("\\\\"):
                    raise ValueError("blender_path: a network path is not started from a node. "
                                     "Set  blender = ...  in kubakub.ini [settings] instead.")
                if not os.path.isdir(cand) and name not in ("blender.exe", "blender"):
                    raise ValueError(f"blender_path: '{cand}' is not blender.exe (or the folder that holds it). "
                                     "Another program is only started from  blender = ...  in kubakub.ini [settings].")
            if os.path.isdir(cand):
                cand = os.path.join(cand, "blender.exe" if os.name == "nt" else "blender")
            if os.path.isfile(cand):
                return cand
            raise FileNotFoundError(f"Blender not found at {cand}")
    roots = [os.environ.get("ProgramFiles", r"C:\Program Files"), os.environ.get("ProgramFiles(x86)", ""),
             os.path.expanduser("~")]
    found = []
    for r in filter(None, roots):
        for exe in glob.glob(os.path.join(r, "Blender Foundation", "Blender*", "blender.exe")):
            m = re.search(r"Blender\s*([\d.]+)", exe)
            ver = tuple(int(x) for x in m.group(1).split(".") if x) if m else (0,)
            found.append((ver, exe))
    if found:
        return max(found)[1]
    exe = shutil.which("blender")
    if exe:
        return exe
    raise FileNotFoundError("Blender not found: install Blender 4.x, or set blender_path or blender = ... in kubakub.ini [settings]")


def clean_path(path: str) -> str:
    """A path as typed (quotes from 'Copy as path', %VAR%, ~) -> the file path the export uses."""
    return os.path.expandvars(os.path.expanduser((path or "").strip().strip('"').strip("'")))


def _file_stamps(obj, out):
    """Size and mtime of every existing file a job dict names (HDRI, projector image, emissive faces)."""
    if isinstance(obj, dict):
        for k in sorted(obj):
            _file_stamps(obj[k], out)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _file_stamps(v, out)
    elif isinstance(obj, str) and len(obj) > 3 and ("/" in obj or "\\" in obj):
        try:
            st = os.stat(obj)
            if not os.path.isdir(obj):
                out.append(f"{obj}|{st.st_size}|{st.st_mtime_ns}")
        except (OSError, ValueError):
            pass
    return out


def cache_key(path: str, camera: str, width: int, height: int, frame: int, unit_scale: float = 1.0,
              view: dict | None = None, views: list | None = None, relight: dict | None = None,
              projector: dict | None = None) -> str:
    st = os.stat(path)
    h = hashlib.sha1()
    for part in (os.path.abspath(path).lower(), st.st_size, st.st_mtime_ns, camera, width, height, frame,
                 unit_scale, json.dumps(view, sort_keys=True) if view else "",
                 json.dumps(views, sort_keys=True) if views else "",
                 json.dumps(relight, sort_keys=True) if relight else "", *(os.stat(f).st_mtime_ns for f in SCRIPTS)):
        h.update(str(part).encode("utf-8", "replace") + b"|")
    if projector:                                         # kubakub projector: another camera on the same file
        h.update(json.dumps(projector, sort_keys=True).encode("utf-8", "replace") + b"|")
    for s in _file_stamps(relight, []) if relight else ():     # an edited HDRI is a new job
        h.update(s.encode("utf-8", "replace") + b"|")
    return h.hexdigest()[:16]


_FOLDERS: dict = {}                                   # folder -> [generation, seen present]


def folder_token(folder: str, ok: bool | None = None) -> bytes:
    """
    Fingerprint part for a result folder. Stays the same from 'not made yet' to 'made' (so the run after the first
    one is cached) and changes once when a folder seen before has vanished (ComfyUI emptied temp at a start, or the
    cache cap removed it): then the node runs again instead of handing on a folder that no longer exists.
    """
    ok = os.path.isdir(folder) if ok is None else bool(ok)
    st = _FOLDERS.setdefault(os.path.normcase(os.path.abspath(folder)), [0, False])
    if ok:
        st[1] = True
    elif st[1]:
        st[0], st[1] = st[0] + 1, False
    return f"gen{st[0]}".encode()


def cache_folder(path, cache_root, camera="", width=0, height=0, frame=-1, unit_scale=1.0, view=None, views=None,
                 relight=None, projector=None) -> str:
    """Where export() keeps (or finds) the result for these settings."""
    path = clean_path(path)
    cache_root = os.path.abspath(cache_root)      # Blender runs from its own working directory
    stem = re.sub(r"[^\w.\-]+", "_", os.path.splitext(os.path.basename(path))[0])[:40]
    return os.path.join(cache_root, f"{stem}_{cache_key(path, camera, width, height, frame, unit_scale, view, views, relight, projector)}")


def job_dict(path, out, camera="", width=0, height=0, frame=-1, unit_scale=1.0, view=None, views=None, relight=None,
             projector=None):
    """The job file blender_export.py reads (the worker keeps a loaded scene per file / camera / frame / scale)."""
    return {"file": clean_path(path), "out": out, "camera": camera, "width": int(width), "height": int(height),
            "frame": int(frame), "unit_scale": float(unit_scale), "view": view, "views": views, "relight": relight,
            "projector": projector or None}


def done_files(view=None, views=None, relight=None):
    """The files a finished job leaves (written last)."""
    if relight:                                   # a sequence writes its own frame names; relight.json comes last
        return ("relight.json",) if relight.get("frames") else ("relit.png", "relight.json")
    if views:                                     # walkthrough: one folder per camera, scene.json at the end
        return (os.path.join(f"v_{len(views) - 1:04d}", "faceid.npy"), "scene.json")
    return ("scene.json", "faceid.npy")


def touch(path):
    """Mark a cache entry as used (the size cap removes the least recently used first)."""
    try:
        os.utime(path, None)
    except OSError:
        pass


def is_done(folder, files) -> bool:
    return all(os.path.isfile(os.path.join(folder, f)) for f in files)


def publish(stage, final, files=()) -> str:
    """Rename a finished staging folder into place. Another process that got there first wins (same content)."""
    for attempt in range(5):
        if is_done(final, files) if files else os.path.isdir(final):
            shutil.rmtree(stage, ignore_errors=True)
            return final
        try:
            if os.path.isdir(final):              # a half-written folder from a crash
                shutil.rmtree(final, ignore_errors=True)
            os.rename(stage, final)
            return final
        except OSError:
            time.sleep(0.2 * (attempt + 1))       # Windows: a scanner or reader holds a file for a moment
    shutil.copytree(stage, final, dirs_exist_ok=True)
    shutil.rmtree(stage, ignore_errors=True)
    return final


def stage_path(final) -> str:
    return f"{final}{STAGE}{os.getpid()}-{uuid.uuid4().hex[:8]}"


def export(path: str, cache_root: str, camera: str = "", width: int = 0, height: int = 0, frame: int = -1,
           blender: str = "", timeout: int = 900, force: bool = False, unit_scale: float = 1.0,
           view: dict | None = None, views: list | None = None, relight: dict | None = None,
           transient: bool = False, projector: dict | None = None):
    """
    -> (folder with scene.json / faceid.npy / ..., cached: bool, blender log tail).
    transient: always render, into a folder of this call's own (the caller moves the files on and deletes it).
    view: {"location": [x, y, z], "look_at": [x, y, z], "lens": mm, "name": str} renders from that camera
    instead of the file's (audience spots; metres after unit_scale). views: a list of such cameras,
    rendered in one Blender session into v_0000, v_0001 ... (walkthroughs). projector: the dict of kubakub projector
    (autocam.py): the projection camera is placed in front of the facade instead of taken from the file.
    """
    path = clean_path(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"scene file not found: {path}")
    ext = os.path.splitext(path)[1].lower()
    if ext == ".c4d":
        raise ValueError("Blender cannot read .c4d files: export FBX or Alembic from Cinema 4D")
    if ext not in FORMATS:
        raise ValueError(f"unsupported file type {ext} (supported: {' '.join(FORMATS)})")
    out = cache_folder(path, cache_root, camera, width, height, frame, unit_scale, view, views, relight, projector)
    done = done_files(view, views, relight)
    if not force and not transient and is_done(out, done):
        touch(out)
        return out, True, ""
    stage = stage_path(out)
    os.makedirs(stage, exist_ok=True)
    try:
        tail = _run_job(job_dict(path, stage, camera, width, height, frame, unit_scale, view, views, relight, projector),
                        stage, done, blender, timeout)
        if transient:
            return stage, False, tail
        if force:
            shutil.rmtree(out, ignore_errors=True)
        return publish(stage, out, done), False, tail
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def _run_job(job_d, out, done, blender, timeout):
    job = os.path.join(out, "job.json")
    with open(job, "w", encoding="utf-8") as f:
        json.dump(job_d, f)
    exe = find_blender(blender)
    if worker_enabled():
        try:
            tail = _worker_run(exe, job, timeout)
            if is_done(out, done):
                return tail
        except Exception as e:  # noqa: BLE001  (a dead or stuck worker: the one-off process below)
            _worker_stop(exe)
            print(f"[KUBA scene3d] Blender worker failed ({e}); running Blender once for this job")
    cmd = [exe, "-b", "--factory-startup", "--python", SCRIPT, "--", job]
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace",
                         timeout=timeout, creationflags=flags)
    tail = "\n".join((res.stdout or "").splitlines()[-40:])
    if res.returncode != 0 or not is_done(out, done):
        err = [ln for ln in (res.stdout or "").splitlines() + (res.stderr or "").splitlines()
               if "[KUBA scene] ERROR" in ln or "Error" in ln]
        raise RuntimeError(f"Blender export failed (exit {res.returncode}): "
                           + ("; ".join(err[-3:]) or tail[-800:]))
    return tail


# --------------------------------------------------------------------------------------------------------------
# a content-addressed store of single files (relight frames): <key>.png + <key>.json (metadata, written last)
# --------------------------------------------------------------------------------------------------------------

def store_get(store, key, ext=".png"):
    """-> (file path, metadata dict) when the store has the key, else None. A hit counts as a use."""
    fp, meta = os.path.join(store, key + ext), os.path.join(store, key + ".json")
    if not (os.path.isfile(fp) and os.path.isfile(meta)):
        return None
    try:
        with open(meta, encoding="utf-8") as f:
            m = json.load(f)
    except (OSError, ValueError):
        return None
    touch(fp)
    touch(meta)
    return fp, m


def store_put(store, key, src, meta, ext=".png", move=True):
    """Put a finished file under its key (renamed when on the same drive), then its metadata. -> the stored path."""
    os.makedirs(store, exist_ok=True)
    fp = os.path.join(store, key + ext)
    tmp = stage_path(fp)
    if move:
        try:
            os.replace(src, tmp)
        except OSError:
            shutil.copyfile(src, tmp)
    else:
        shutil.copyfile(src, tmp)
    try:
        os.replace(tmp, fp)
    except OSError:                               # another process has it open (it wrote the same pixels)
        os.remove(tmp)
        if not os.path.isfile(fp):
            raise
    write_json(os.path.join(store, key + ".json"), meta)
    return fp


def write_json(path, obj):
    tmp = stage_path(path)
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1)
    try:
        os.replace(tmp, path)
    except OSError:
        os.remove(tmp)
        if not os.path.isfile(path):
            raise


# --------------------------------------------------------------------------------------------------------------
# size cap: least recently used entries go first
# --------------------------------------------------------------------------------------------------------------
STORES = ("relight_frames", "walk", "emit", "projector", "warm", "pieces")   # folders whose children are entries


def _size(path):
    if not os.path.isdir(path):
        try:
            return os.path.getsize(path)
        except OSError:
            return 0
    total = 0
    for dp, _, fns in os.walk(path):
        for fn in fns:
            try:
                total += os.path.getsize(os.path.join(dp, fn))
            except OSError:
                pass
    return total


def _stamp(path):
    """Last use of an entry: the newest mtime of the entry and its direct children."""
    try:
        t = os.path.getmtime(path)
    except OSError:
        return 0.0
    if os.path.isdir(path):
        try:
            for e in os.scandir(path):
                try:
                    t = max(t, e.stat().st_mtime)
                except OSError:
                    pass
        except OSError:
            pass
    return t


def entries(root):
    """[(path, size bytes, last use)] of a cache root: each scene folder, and each file / folder inside a store."""
    out = []
    if not os.path.isdir(root):
        return out
    for e in os.scandir(root):
        if e.name in STORES and e.is_dir():
            for c in os.scandir(e.path):
                if c.name.endswith(".json") and os.path.isfile(c.path[:-5] + ".png"):
                    continue                      # a frame's metadata goes with its image
                out.append((c.path, _size(c.path) + (_size(c.path[:-4] + ".json") if c.name.endswith(".png") else 0),
                            _stamp(c.path)))
        else:
            out.append((e.path, _size(e.path), _stamp(e.path)))
    return out


def purge(root, cap_bytes, min_age_s=7200.0, stale_stage_s=6 * 3600.0, now=None):
    """
    Remove the least recently used entries until the cache is under cap_bytes. Entries used in the last
    min_age_s stay (another queue or ComfyUI may be reading them). Staging leftovers of crashed runs go after
    stale_stage_s. -> (entries removed, bytes freed, bytes left).
    """
    now = time.time() if now is None else now
    ents = entries(root)
    total = sum(s for _, s, _ in ents)
    removed = freed = 0
    for p, s, t in sorted(ents, key=lambda e: e[2]):
        stage = STAGE in os.path.basename(p)
        if not stage and total <= cap_bytes:
            continue                              # under the cap: only staging leftovers still go
        if now - t < (stale_stage_s if stage else min_age_s):
            continue
        if os.path.isdir(p):
            shutil.rmtree(p, ignore_errors=True)
        else:
            for f in (p, p[:-4] + ".json") if p.endswith(".png") else (p,):
                try:
                    os.remove(f)
                except OSError:
                    pass
        if not os.path.exists(p):
            total -= s
            freed += s
            removed += 1
    return removed, freed, total


_PURGE = {"last": 0.0}


def maybe_purge(root, cap_bytes, every_s=600.0):
    """purge() in a background thread, at most once per every_s (after renders and at start)."""
    import threading
    now = time.time()
    if now - _PURGE["last"] < every_s or cap_bytes <= 0:
        return None
    _PURGE["last"] = now

    def run():
        try:
            n, freed, left = purge(root, cap_bytes)
            if n:
                print(f"[KUBA scene3d] cache: removed {n} old entries ({freed / 2 ** 30:.1f} GB), "
                      f"{left / 2 ** 30:.1f} GB left")
        except Exception as e:  # noqa: BLE001
            print(f"[KUBA scene3d] cache purge failed: {e}")

    t = threading.Thread(target=run, daemon=True)
    t.start()
    return t


# --------------------------------------------------------------------------------------------------------------
# a Blender that stays open (blender_export.py --serve): jobs skip Blender's start and GPU setup; relight jobs on
# the same scene also skip the file load
# --------------------------------------------------------------------------------------------------------------
import queue as _queue
import threading as _threading

_WORKERS = {}
_WORKER_LOCK = _threading.Lock()


def worker_enabled():
    return kst.switch("blender_worker", True)


def worker_idle_s():
    """Seconds an idle worker stays open (setting blender_idle, default 600). It holds some GPU memory."""
    return max(10.0, kst.number("blender_idle", 600))


class _Worker:
    def __init__(self, exe):
        flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        self.proc = subprocess.Popen([exe, "-b", "--factory-startup", "--python", SCRIPT, "--", "--serve",
                                      "--idle", str(worker_idle_s())],
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                     text=True, encoding="utf-8", errors="replace", bufsize=1, creationflags=flags)
        self.lines = _queue.Queue()
        _threading.Thread(target=self._pump, daemon=True).start()

    def _pump(self):
        for line in self.proc.stdout:
            self.lines.put(line.rstrip("\n"))
        self.lines.put(None)                              # the process ended

    def alive(self):
        return self.proc.poll() is None

    def run(self, job_path, timeout):
        self.proc.stdin.write(job_path + "\n")
        self.proc.stdin.flush()
        tail, end = [], time.time() + timeout
        while True:
            line = self.lines.get(timeout=max(0.1, end - time.time()))
            if line is None:
                raise RuntimeError("Blender worker ended: " + " | ".join(tail[-5:]))
            tail = (tail + [line])[-40:]
            if line.startswith("KUBA_DONE"):
                return "\n".join(tail)
            if line.startswith("KUBA_FAIL"):
                raise RuntimeError(line[9:].strip())


def _worker_run(exe, job_path, timeout):
    with _WORKER_LOCK:                                    # one job at a time per Blender
        w = _WORKERS.get(exe)
        if w is None or not w.alive():
            w = _WORKERS[exe] = _Worker(exe)
        return w.run(job_path, timeout)


def _worker_stop(exe):
    w = _WORKERS.pop(exe, None)
    if w is not None and w.alive():
        w.proc.kill()


def worker_prestart(job_d, folder, blender="", timeout=300):
    """
    Start the worker in the background and let it load the scene of a relight job (job_d from job_dict() with
    relight={"warm": True}), while ComfyUI does other work. The next relight job queues behind it and finds the
    scene loaded. Never raises; -> the thread, or None when the worker is off.
    """
    if not worker_enabled():
        return None

    def run():
        try:
            exe = find_blender(blender)
            os.makedirs(folder, exist_ok=True)
            job = os.path.join(folder, f"warm_{os.getpid()}.json")
            with open(job, "w", encoding="utf-8") as f:
                json.dump(dict(job_d, out=folder), f)
            _worker_run(exe, job, timeout)
        except Exception as e:  # noqa: BLE001  (the real job reports problems)
            print(f"[KUBA scene3d] Blender pre-start skipped: {e}")

    t = _threading.Thread(target=run, daemon=True)
    t.start()
    return t
