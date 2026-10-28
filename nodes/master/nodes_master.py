"""
nodes_master.py

The master timeline (web/kubakub_master.js, the "kubakub timeline" tab of ComfyUI's bottom panel): its render node and
its routes. Preview copies of videos come from the director's own route (/kubakub/director/media), which also reads
'output:<name>' sources.

kubakub timeline render: the whole show at the project size x scale and fps -> frames, fps, audio (into kubakub
export video / frames for a delivery file). The panel keeps the show in the node's hidden 'timeline' widget; the
directors on the timeline are rendered from their 'director' outputs connected to it.

    GET  /kubakub/master/outputs           videos and images in output/, newest first
    POST /kubakub/master/upload            multipart "file" -> input/kuba_director/<name> (streamed, so videos bigger
                                           than ComfyUI's upload limit work too)
    POST /kubakub/master/wave              {source} -> {duration, rate, peaks}: the sound of a clip as a waveform
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import uuid

import folder_paths
import numpy as np
import torch
from comfy_api.latest import io

from ...kubakub import master as ms
from ...kubakub import master_render as mr
from ...kubakub.director import media as md
from ...kubakub.io_types import DirectorType
from ..director.nodes_director import sequence_frames

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


def show_of(timeline, extra_pnginfo=None):
    """The show: the node's 'timeline' text (written by the panel when queued), else the one saved in the workflow."""
    try:
        d = json.loads(timeline) if timeline else None
    except ValueError:
        d = None
    if isinstance(d, dict) and d.get("tracks") is not None:
        return d
    try:
        d = (extra_pnginfo or {}).get("workflow", {}).get("extra", {}).get("kubakub_master")
    except AttributeError:
        d = None
    return d if isinstance(d, dict) else None


class KUBA_MasterRender(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="KUBA_MasterRender",
            display_name="kubakub timeline render",
            category="kubakub/2d/director",
            search_aliases=["master timeline", "edit", "sequence", "show", "montage"],
            description=(
                "Renders the whole show of the kubakub timeline (bottom panel) at the project size x scale and the "
                "project fps: every director on it, rendered clips, pictures and sounds, cut, trimmed and sped up as on "
                "the timeline. Connect the 'director' output of every director that is on the timeline. Into kubakub "
                "export video / frames for a delivery file."),
            inputs=[
                io.Autogrow.Input("directors", optional=True, template=io.Autogrow.TemplatePrefix(
                    input=DirectorType.Input("director", tooltip="The 'director' output of a kubakub director on the timeline."),
                    prefix="director_", min=0, max=16)),
                io.Float.Input("scale", default=0.5, min=0.1, max=1.0, step=0.05,
                               tooltip="Size of the frames: 1 = the project size (memory: frames x width x height x 12 bytes)."),
                io.String.Input("timeline", default="", tooltip="The show, written here by the kubakub timeline panel "
                                                                "(hidden; empty = the one saved in the workflow)."),
                io.Autogrow.Input("clips", optional=True, template=io.Autogrow.TemplatePrefix(
                    input=io.Video.Input("clip", tooltip="The video output of a node whose saved clip is on the timeline "
                                                         "(e.g. Save Video after an LTX node): the show is rendered after "
                                                         "it, so a clip made in the same run is there."),
                    prefix="clip_", min=0, max=16)),
            ],
            outputs=[
                io.Image.Output("frames", tooltip="Every frame of the show."),
                io.Float.Output("fps", tooltip="The project's frames per second."),
                io.Audio.Output("audio", tooltip="The show's sound: the clips' and the directors' sound mixed."),
                io.Int.Output("frame_count", tooltip="Frames of the show (its length x fps)."),
                io.String.Output("report", tooltip="What was rendered, and what was left out and why."),
            ],
            hidden=[io.Hidden.extra_pnginfo],
        )

    @classmethod
    def execute(cls, scale, timeline="", directors=None, clips=None) -> io.NodeOutput:
        # clips only set the order (the files are read from the timeline's sources, after the nodes that save them)
        hid = getattr(cls, "hidden", None)
        doc = show_of(timeline, getattr(hid, "extra_pnginfo", None))
        if doc is None:
            raise ValueError("kubakub timeline render: no show yet - open the kubakub timeline tab of the bottom panel, "
                             "put something on it, and queue again")
        states = {}
        for S in (directors or {}).values():
            if S is not None and getattr(S, "uid", ""):
                states[str(S.uid)] = S
        done = {}

        def director(node_id, w, h):
            S = states.get(str(node_id))
            if S is None:
                return None
            if not S.doc:                                   # nothing applied yet: its still
                return S.still, float(S.fps), S.audio
            dscale = min(1.0, max(0.05, max(w / S.W, h / S.H)))
            key = (str(node_id), round(dscale, 4))          # a director on the timeline twice at one size: rendered once
            if key not in done:
                done[key] = sequence_frames(S.doc, S.base, S.sources, S.labels, S.table, S.scene, S.W, S.H, dscale,
                                            False, S.pmask)[0]
            return done[key], float(S.fps), S.audio

        inp, outp = folder_paths.get_input_directory(), folder_paths.get_output_directory()
        frames, sound, fps, lines = mr.render(doc, float(scale), lambda src: ms.resolve(src, inp, outp), director)
        report = "\n".join(lines)
        log.info("[KUBA timeline render] %s", report.replace("\n", "\n    "))
        audio = {"waveform": torch.from_numpy(sound)[None], "sample_rate": mr.SR}
        return io.NodeOutput(torch.from_numpy(frames), float(fps), audio, int(frames.shape[0]), report)


NODE_CLASS_MAPPINGS = {"KUBA_MasterRender": KUBA_MasterRender}
NODE_DISPLAY_NAME_MAPPINGS = {"KUBA_MasterRender": "kubakub timeline render"}
