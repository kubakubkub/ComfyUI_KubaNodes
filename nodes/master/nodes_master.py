"""
nodes_master.py

Routes of the master timeline (web/kubakub_master.js, the "kubakub timeline" tab of ComfyUI's bottom panel). No nodes
yet: the master render comes with the next step. Preview copies of videos come from the director's own route
(/kubakub/director/media), which also reads 'output:<name>' sources.

    GET  /kubakub/master/outputs           videos and images in output/, newest first
    POST /kubakub/master/upload            multipart "file" -> input/kuba_director/<name> (streamed, so videos bigger
                                           than ComfyUI's upload limit work too)
    POST /kubakub/master/wave              {source} -> {duration, rate, peaks}: the sound of a clip as a waveform
"""

from __future__ import annotations

import hashlib
import logging
import os
import uuid

import folder_paths
import numpy as np

from ...kubakub import master as ms
from ...kubakub.director import media as md

log = logging.getLogger("KUBA.master")

UPLOAD_SUB = "kuba_director"       # the director's import folder: a file dropped on either lands in the same place


def _same_file(a, b):
    if os.path.getsize(a) != os.path.getsize(b):
        return False
    ha, hb = hashlib.sha1(), hashlib.sha1()
    for p, h in ((a, ha), (b, hb)):
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
    return ha.digest() == hb.digest()


_WAVES = {}                       # (path, mtime, size) -> the answer of /wave (a few files, small lists)


def _wave(path):
    st = os.stat(path)
    key = (path, st.st_mtime_ns, st.st_size)
    if key not in _WAVES:
        sr = 2000
        res = md._run([md.ffmpeg_exe(), "-v", "error", "-i", path, "-vn", "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"])
        mono = np.frombuffer(res.stdout, np.float32) if res.returncode == 0 else np.zeros(0, np.float32)
        while len(_WAVES) >= 64:
            _WAVES.pop(next(iter(_WAVES)))
        _WAVES[key] = {"duration": round(mono.size / sr, 4), "rate": ms.WAVE_RATE, "peaks": ms.peaks(mono, sr)}
    return _WAVES[key]


def _register_routes():
    try:
        from aiohttp import web
        from server import PromptServer
    except Exception:  # noqa: BLE001  (tests / no server)
        return

    async def outputs(request):
        import asyncio
        files = await asyncio.get_running_loop().run_in_executor(None, ms.list_media, folder_paths.get_output_directory())
        return web.json_response({"files": files})

    async def upload(request):
        import asyncio
        try:
            reader = await request.multipart()
            field = await reader.next()
            while field is not None and field.name != "file":
                field = await reader.next()
            name = ms.safe_name(field.filename if field is not None else "")
            if not name:
                return web.json_response({"error": "not a video, picture or sound the timeline can use"}, status=400)
            folder = os.path.join(folder_paths.get_input_directory(), UPLOAD_SUB)
            os.makedirs(folder, exist_ok=True)
            tmp = os.path.join(folder, f".upload_{uuid.uuid4().hex}.part")
            try:
                with open(tmp, "wb") as f:
                    while chunk := await field.read_chunk(1 << 20):
                        f.write(chunk)
                same = os.path.isfile(os.path.join(folder, name)) and await asyncio.get_running_loop().run_in_executor(
                    None, _same_file, tmp, os.path.join(folder, name))
                if not same:                            # the same file dropped again is reused, never copied twice
                    name = ms.free_name(folder, name)
                    os.replace(tmp, os.path.join(folder, name))
            finally:
                if os.path.exists(tmp):
                    os.remove(tmp)
            return web.json_response({"name": f"{UPLOAD_SUB}/{name}", "kind": ms.kind_of(name)})
        except Exception as e:  # noqa: BLE001
            log.exception("[KUBA master] upload failed")
            return web.json_response({"error": str(e).splitlines()[0]}, status=500)

    async def wave(request):
        import asyncio
        try:
            src = str((await request.json()).get("source") or "")
            path = ms.resolve(src, folder_paths.get_input_directory(), folder_paths.get_output_directory())
            if not path or ms.kind_of(path) not in ("video", "audio"):
                return web.json_response({"error": "not found: " + src.split(":", 1)[-1]}, status=404)
            return web.json_response(await asyncio.get_running_loop().run_in_executor(None, _wave, path))
        except Exception as e:  # noqa: BLE001
            log.exception("[KUBA master] waveform failed")
            return web.json_response({"error": str(e).splitlines()[0]}, status=500)

    try:
        PromptServer.instance.routes.post("/kubakub/master/wave")(wave)
        PromptServer.instance.routes.get("/kubakub/master/outputs")(outputs)
        PromptServer.instance.routes.post("/kubakub/master/upload")(upload)
    except (AttributeError, RuntimeError) as e:
        log.warning("[KUBA master] routes not registered: %s", e)


_register_routes()

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
