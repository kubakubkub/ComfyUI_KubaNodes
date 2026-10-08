"""kubakub regions from cryptomatte: a render's cryptomatte EXR -> named regions (logic in cryptomatte.py, exr.py)."""

from __future__ import annotations

import json
import logging
import os
import time

import cv2
import numpy as np
import torch

import folder_paths
from comfy_api.latest import io, ui

from ...kubakub import cryptomatte as cm
from ...kubakub import exr
from ...kubakub import samples
from ...kubakub.io_types import RegionsType
from ...kubakub.types import Regions

try:
    from ...kubakub import facade_core as fc
except ImportError:
    from kubakub import facade_core as fc

log = logging.getLogger("KUBA.regions")

CATEGORY = "kubakub/3d/from renders"
_CACHE = {}                                   # path -> (mtime, exr dict): re-queues with other settings are instant


def _read_cached(path):
    st = os.path.getmtime(path)
    hit = _CACHE.get(path)
    if hit and hit[0] == st:
        return hit[1]
    data = exr.read(path)
    _CACHE.clear()
    _CACHE[path] = (st, data)
    return data


def _srgb(lin):
    lin = np.clip(lin, 0.0, 1.0)
    return np.where(lin <= 0.0031308, lin * 12.92, 1.055 * np.power(lin, 1 / 2.4) - 0.055).astype(np.float32)


class KUBA_RegionsFromCryptomatte(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="KUBA_RegionsFromCryptomatte",
            display_name="kubakub regions from cryptomatte",
            category="kubakub/3d/from renders",
            search_aliases=['cryptomatte', 'exr', 'object id', 'blender', 'houdini', 'arnold', 'redshift', 'karma'],
            description=("Render a cryptomatte pass in your 3D tool (Blender, Houdini, C4D, Maya ...) and every "
                         "object becomes a region with its real name from the scene (window_left_03). Objects "
                         "named alike (window_left_01, _02 ...) share a group; a second cryptomatte layer "
                         "(materials) adds tags for the plan rules. Also gives the render itself."),
            inputs=[
                io.String.Input("file", default="", placeholder="paste the path of your cryptomatte .exr  (empty = a sample)",
                                tooltip="Your multilayer EXR with the cryptomatte pass (32-bit): paste its path. Empty: a "
                                        "built-in sample EXR, so the node runs as it is."),
                io.String.Input("layer", default="object",
                                tooltip="Which cryptomatte becomes the regions: object, material, asset, or the "
                                        "layer's full name."),
                io.String.Input("tag_layers", default="material",
                                tooltip="Other cryptomatte layers whose names become tags (comma separated); "
                                        "e.g. material -> [tag:glass] in the plan."),
                io.String.Input("exclude", default="", placeholder="ground, sky*, helper_*",
                                tooltip="Object names (wildcards) that are no regions."),
                io.Int.Input("min_region_area", default=64, min=0, max=100_000_000, step=8,
                             tooltip="Objects smaller than this (pixels) are merged or dropped."),
                io.Boolean.Input("merge_small_regions", default=True,
                                 tooltip="On: small objects join the neighbour with the longest border. Off: dropped."),
                io.Image.Input("matrix", optional=True,
                               tooltip="Your matrix: its size is used (the render is resized) and it is the preview background."),
                io.Mask.Input("scope", optional=True,
                              tooltip="Pixels outside this mask are never part of a region (the size of the "
                                      "matrix, or of the render without one)."),
            ],
            outputs=[
                RegionsType.Output("regions", tooltip="Label map + region table for the kubakub region nodes."),
                io.Mask.Output("region_masks", tooltip="One mask per region; batch index = region_id."),
                io.String.Output("regions_json", tooltip="The region table as text."),
                io.Image.Output("preview", tooltip="Regions over the render (or the matrix)."),
                io.Image.Output("render", tooltip="The render's picture (Combined pass), sRGB."),
                io.String.Output("report", tooltip="Layers found, objects, groups and tags."),
            ],
        )

    @classmethod
    def fingerprint_inputs(cls, file, **kw):
        p = fc.clean_folder_path(file)
        try:
            return f"{p}|{os.path.getmtime(p)}"
        except OSError:
            return p

    @classmethod
    def execute(cls, file, layer, tag_layers, exclude, min_region_area, merge_small_regions,
                matrix=None, scope=None) -> io.NodeOutput:
        t0 = time.perf_counter()
        path = fc.clean_folder_path(file)                 # quotes of Explorer's 'Copy as path', ~ and %VARS%
        sample = not path
        if sample:
            path = samples.cryptomatte(folder_paths.get_temp_directory())
        if not os.path.isfile(path):
            raise ValueError(f"kubakub regions from cryptomatte: file not found: '{path}'")
        data = _read_cached(path)
        chans, lays = data["channels"], cm.layers(data["attrs"])
        if not lays:
            raise ValueError("no cryptomatte in this EXR (render the cryptomatte pass, 32-bit float, multilayer)")
        main = cm.pick_layer(list(lays), layer)
        if main is None:
            raise ValueError(f"no cryptomatte layer '{layer}'; the file has: {', '.join(lays)}")
        labels, names = cm.label_map(chans, main, lays[main]["manifest"], exclude)
        tags = [[] for _ in names]
        used_tags = []
        for tl in [t.strip() for t in (tag_layers or "").split(",") if t.strip()]:
            ln = cm.pick_layer([n for n in lays if n != main], tl)
            if ln:
                for i, t in enumerate(cm.tags_from(chans, ln, lays[ln]["manifest"], labels, len(names))):
                    tags[i] += [x for x in t if x not in tags[i]]
                used_tags.append(ln)
        H, W = labels.shape
        rgb = cm.beauty(chans)
        render = _srgb(rgb) if rgb is not None else np.full((H, W, 3), 0.5, np.float32)
        notes = []
        if matrix is not None:
            mh, mw = int(matrix.shape[1]), int(matrix.shape[2])
            if (mh, mw) != (H, W):
                labels = cv2.resize(labels, (mw, mh), interpolation=cv2.INTER_NEAREST)
                render = cv2.resize(render, (mw, mh), interpolation=cv2.INTER_LINEAR)
                notes.append(f"render {W}x{H} resized to the matrix {mw}x{mh}")
                H, W = mh, mw
        sc = None
        if scope is not None:
            sc = (scope[0] if scope.ndim == 3 else scope).cpu().numpy() > 0.5
            if sc.shape != (H, W):
                raise ValueError(f"scope mask is {sc.shape[1]}x{sc.shape[0]}, the regions {W}x{H}; they must match "
                                 "exactly (connect the same matrix to this node and to the node that made the scope).")
            notes.append(f"scope: regions only on {sc.mean() * 100:.0f} % of the picture")
        meta = [{"name": n, "group": fc.group_name_from_stem(n), "source": f"cryptomatte {main}",
                 **({"tags": t} if t else {})} for n, t in zip(names, tags)]
        labels, atlas, sc = fc._finish_atlas(labels, meta, notes, [], sc, "mask_folder", int(min_region_area),
                                             bool(merge_small_regions), with_scope=True)
        atlas["mode"] = "cryptomatte"
        regions = Regions.from_numpy(labels, atlas, sc)
        bg = matrix[0, ..., :3].cpu().float().numpy() if matrix is not None else render
        preview = torch.from_numpy(fc.render_preview(bg, labels, atlas))[None]
        report = (f"{os.path.basename(path)} ({data['compression']}): layers {', '.join(lays)}; regions from {main}: "
                  f"{len(atlas['regions'])} objects in {len(atlas['groups'])} groups"
                  + (f"; tags from {', '.join(used_tags)}" if used_tags else "")
                  + "".join(f"; {n}" for n in atlas["notes"]) + f"; {time.perf_counter() - t0:.1f} s"
                  + ("\n" + samples.note("cryptomatte EXR", "file") if sample else ""))
        log.info("[KUBA cryptomatte] %s", report)
        return io.NodeOutput(regions, regions.masks(), json.dumps(atlas, indent=1), preview,
                             torch.from_numpy(np.ascontiguousarray(render))[None], report,
                             ui=ui.PreviewImage(preview, cls=cls))


NODE_CLASS_MAPPINGS = {"KUBA_RegionsFromCryptomatte": KUBA_RegionsFromCryptomatte}
NODE_DISPLAY_NAME_MAPPINGS = {"KUBA_RegionsFromCryptomatte": "kubakub regions from cryptomatte"}
