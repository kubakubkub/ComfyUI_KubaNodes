"""kubakub render passes: a folder of render passes from any 3D tool, or one multilayer EXR -> picture, depth,
normals, masks (logic in render_passes.py). The layer buttons on the node (web/kubakub_render_passes.js) ask the
route at the end of this file which layers an EXR holds."""

from __future__ import annotations

import logging
import os
import sys

import numpy as np
import torch

import folder_paths

from comfy_api.latest import io, ui

from ...kubakub import render_passes as rp
from ...kubakub import samples

log = logging.getLogger("KUBA.regions")

CATEGORY = "kubakub/3d/from renders"


def _clean(path):
    return os.path.expandvars(os.path.expanduser((path or "").strip().strip('"').strip("'")))


class KUBA_RenderPasses(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="KUBA_RenderPasses",
            display_name="kubakub render passes",
            category="kubakub/3d/from renders",
            search_aliases=['aov', 'beauty', 'depth pass', 'houdini', 'karma', 'blender', 'c4d', 'redshift', 'load passes'],
            description=("Your own render as the start: reads a folder of passes written by any 3D tool, named "
                         "<render>_<pass>.png or .exr (facade_beauty.png, facade_depth.png, facade_normal.png, facade_cut.png), "
                         "or one multilayer EXR: the node shows which layers are inside and you pick them. "
                         "Gives the picture, depth and normals, and every other pass as a mask with its name. Connect "
                         "masks and names to kubakub regions from masks: each mask becomes a region you can repaint "
                         "while the rest of the render stays as it is."),
            inputs=[
                io.String.Input("folder", default="", placeholder="paste the folder of your passes  (empty = sample passes)",
                                tooltip="The folder with your passes, or one multilayer .exr file: paste its path "
                                        "(Explorer: Copy as path; the quotes are fine). Empty: built-in sample passes, "
                                        "so the node runs as it is."),
                io.String.Input("render", default="", optional=True,
                                tooltip="Which render, when the folder holds several (facade_a for facade_a_beauty.png). "
                                        "Empty: the first one; the report lists them."),
                io.String.Input("invert", default="", optional=True, placeholder="cut",
                                tooltip="Masks where black means inside (names, wildcards, comma separated)."),
                io.String.Input("exclude", default="", optional=True, advanced=True, placeholder="alpha, wire*",
                                tooltip="Passes that are no masks (names, wildcards, comma separated)."),
                io.Combo.Input("depth", options=["as rendered", "near is white", "near is black"], default="as rendered",
                               optional=True, advanced=True,
                               tooltip="How the depth output reads. The last two stretch it over the whole range."),
                io.Image.Input("matrix", optional=True,
                               tooltip="Your matrix: a render of another size is fitted to it (picture, depth, normal "
                                       "and masks; same proportions only). Without: the render's own size."),
                io.String.Input("exr_layers", default="", multiline=True, optional=True, dynamic_prompts=False,
                                placeholder="multilayer EXR: the layers to read as masks, one per line  (empty = all)",
                                tooltip="For a multilayer EXR: which layers become masks, one name per line (wildcards "
                                        "work: wall*). Click the layer buttons on the node, or type. Empty: every layer "
                                        "that looks like a mask. The picture, depth and normal are found by their "
                                        "names; to say it yourself write 'picture = C', 'depth = Z_render', "
                                        "'normal = N_world', 'mask = diffuse' or 'skip = name'. A plain name is a mask, "
                                        "unless that layer is the picture, depth or normal. The report lists every "
                                        "layer of the file."),
            ],
            outputs=[
                io.Image.Output("beauty", tooltip="The picture: the image for the region sampler / versions."),
                io.Image.Output("depth", tooltip="The depth pass (black when the folder has none)."),
                io.Image.Output("normal", tooltip="The normal pass (black when the folder has none)."),
                io.Mask.Output("masks", tooltip="Every other pass as a mask (batch) -> kubakub regions from masks, masks."),
                io.String.Output("names", tooltip="The masks' names, one per line -> kubakub regions from masks, names."),
                io.String.Output("report", tooltip="What was found in the folder."),
            ],
        )

    @classmethod
    def fingerprint_inputs(cls, folder, **kw):
        p = _clean(folder)
        try:
            if os.path.isfile(p):                          # one multilayer EXR
                stamp = [(os.path.basename(p), os.path.getsize(p), os.stat(p).st_mtime_ns)]
            else:
                stamp = sorted((e.name, e.stat().st_size, e.stat().st_mtime_ns) for e in os.scandir(p) if e.is_file())
        except OSError:
            stamp = []
        return f"{p}|{stamp}|{sorted(kw.items())}"

    @classmethod
    def execute(cls, folder, render="", invert="", exclude="", depth="as rendered", matrix=None,
                exr_layers="") -> io.NodeOutput:
        sample = not _clean(folder)
        mw, mh = (int(matrix.shape[2]), int(matrix.shape[1])) if matrix is not None else (0, 0)
        if sample:
            path = samples.passes(folder_paths.get_temp_directory())
            try:
                r = rp.load(path, render, invert, exclude, depth, mw, mh)
            except ValueError:                             # the sample has other proportions than the matrix
                r = rp.load(path, render, invert, exclude, depth)
                r["notes"].append(f"the sample has other proportions than the matrix {mw}x{mh}: it stays at its own size")
            r["notes"].append(samples.note("folder of render passes", "folder"))
        else:
            r = rp.load(_clean(folder), render, invert, exclude, depth, mw, mh, exr_layers or "")
        H, W = r["beauty"].shape[:2]
        img = lambda a: torch.from_numpy(np.ascontiguousarray(a if a is not None else np.zeros((H, W, 3), np.float32)))[None]  # noqa: E731
        if r["masks"]:
            masks = torch.from_numpy(np.stack([m for _, m in r["masks"]]))
        else:
            masks = torch.zeros((1, H, W), dtype=torch.float32)
        names = "\n".join(n for n, _ in r["masks"])
        report = "\n".join([
            f"render '{r['name'] or 'beauty'}' {W}x{H}" + (f"; also in the folder: {', '.join(n for n in r['renders'] if n != r['name'])}"
                                                           if len(r["renders"]) > 1 else ""),
            f"depth: {'yes' if r['depth'] is not None else 'none'}, normal: {'yes' if r['normal'] is not None else 'none'}",
            "masks: " + (", ".join(f"{n} ({m.mean() * 100:.0f} %)" for n, m in r["masks"])
                         or "none (the output is one empty mask)"),
            *r["notes"],
            *([f"layers in {r['file']}:"] + [f"  {L['name']} ({' '.join(L['channels'])}): {L['used']}" for L in r["layers"]]
              if r.get("layers") is not None else [])])
        log.info("[KUBA render passes] %s", report.replace("\n", " | "))
        beauty = img(r["beauty"])
        return io.NodeOutput(beauty, img(r["depth"]), img(r["normal"]), masks, names, report,
                             ui=ui.PreviewImage(beauty, cls=cls))


def _from_outside():
    """A request may not name a path: ComfyUI listens on the network and kubakub.ini does not allow it (the one
    rule of the pack's routes, settings.route_paths_ok)."""
    from ... import settings
    return not settings.route_paths_ok()


def _routes():
    """POST /kubakub/render_passes/layers {path, render} -> {layers: [{name, channels, role, why}], file, note}: the
    layers of the EXR in the node's folder field, for the layer buttons. Reads EXR headers only, writes nothing."""
    srv = sys.modules.get("server")                        # no running ComfyUI (the tests): no route
    if srv is None or getattr(getattr(srv, "PromptServer", None), "instance", None) is None:
        return
    try:
        from aiohttp import web
    except ImportError:
        return

    async def layers(request):
        import asyncio
        try:
            body = await request.json()
            path, render = _clean(str(body.get("path") or "")), str(body.get("render") or "")
            if path and _from_outside():
                return web.json_response({"layers": [], "file": "", "note": "the layer list is off while ComfyUI "
                                          "listens on the network (--listen); allow it with  remote_paths = on  in "
                                          "kubakub.ini [settings]. Typing the layer names works as it is."})
            return web.json_response(await asyncio.get_running_loop().run_in_executor(None, rp.layers_for, path, render))
        except Exception as e:  # noqa: BLE001
            return web.json_response({"layers": [], "file": "", "note": (str(e).splitlines() or [""])[0]})

    try:
        srv.PromptServer.instance.routes.post("/kubakub/render_passes/layers")(layers)
    except (AttributeError, RuntimeError) as e:
        log.warning("[KUBA render passes] route not registered: %s", e)


_routes()

NODE_CLASS_MAPPINGS = {"KUBA_RenderPasses": KUBA_RenderPasses}
NODE_DISPLAY_NAME_MAPPINGS = {"KUBA_RenderPasses": "kubakub render passes"}
