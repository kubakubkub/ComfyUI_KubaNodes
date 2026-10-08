"""kubakub sketch tools: scan to line, regions from sketch, line overlay (logic in sketch.py)."""

from __future__ import annotations

import json
import logging
import time

import numpy as np
import torch

from comfy_api.latest import io, ui

from ...kubakub import sketch as sk
from ...kubakub.io_types import RegionsType
from ...kubakub.types import Regions

try:
    from ...kubakub import facade_core as fc
except ImportError:                     # tests import the package flat
    from kubakub import facade_core as fc

log = logging.getLogger("KUBA.regions")

CATEGORY = "kubakub/2d/sketch"


def _img(a):
    return torch.from_numpy(np.ascontiguousarray(a, np.float32))[None]


class KUBA_ScanToLine(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="KUBA_ScanToLine",
            display_name="kubakub scan to line",
            category="kubakub/2d/sketch",
            search_aliases=['sketch', 'drawing', 'paper', 'photo of drawing', 'hand drawn', 'pencil'],
            description=("A phone photo of a drawing becomes a clean line on the matrix: finds the paper (or use 4 "
                         "corners), straightens it, removes paper colour, shadows and grain. Pencil keeps the greys, "
                         "ink gives clean black. Draw on a print of the festival template and the scan lands exactly "
                         "on it. A batch of photos (drawn animation) is scanned frame by frame."),
            inputs=[
                io.Image.Input("photo", tooltip="Photo(s) of the drawing; the paper should be brighter than the table."),
                io.Int.Input("width", default=0, min=0, max=16384, step=8,
                             tooltip="Matrix width (link it from project settings); 0 = the paper's own size."),
                io.Int.Input("height", default=0, min=0, max=16384, step=8,
                             tooltip="Matrix height; 0 = from the width and the paper's proportions."),
                io.Combo.Input("line", options=["pencil", "ink"], default="pencil",
                               tooltip="pencil = keep the soft greys of the stroke; ink = clean black and white."),
                io.Float.Input("clean", default=0.5, min=0.0, max=1.0, step=0.05,
                               tooltip="How much paper grain and faint marks are removed (1 = only strong lines)."),
                io.Float.Input("boost", default=1.0, min=0.2, max=4.0, step=0.05,
                               tooltip="Makes light pencil darker (above 1) or keeps it delicate (below 1)."),
                io.Boolean.Input("dots_in_line", default=False, optional=True, advanced=True,
                                 tooltip="Off: coloured marker dots (names for regions from sketch) are left out of "
                                         "the line and projection outputs. On: they stay."),
                io.String.Input("corners", default="", optional=True, placeholder="120,80 1900,95 1880,1400 110,1380",
                                tooltip="Paper corners in the photo (tl tr br bl, pixels) when the automatic "
                                        "search picks the wrong shape. Empty = automatic."),
            ],
            outputs=[
                io.Image.Output("drawing", tooltip="The drawing on white, coloured strokes kept."),
                io.Mask.Output("line", tooltip="Line strength 0..1 (for regions from sketch and line overlay)."),
                io.Image.Output("projection", tooltip="White lines on black: ready to project as they are."),
                io.Image.Output("found", tooltip="The photo with the paper outline that was used."),
                io.String.Output("corners", tooltip="The corners used (copy them into 'corners' to fix them)."),
                io.String.Output("report", tooltip="One line per photo: how the paper was found, its size in the photo "
                                                   "(pixels), the size it was straightened to and the corners used."),
            ],
        )

    @classmethod
    def execute(cls, photo, width, height, line, clean, boost, corners="", dots_in_line=False) -> io.NodeOutput:
        t0 = time.perf_counter()
        drawings, lines, founds, notes, used = [], [], [], [], []
        for i in range(int(photo.shape[0])):
            rgb = photo[i, ..., :3].cpu().float().numpy()
            r = sk.scan(rgb, int(width), int(height), corners or "", line, float(clean), float(boost),
                        bool(dots_in_line))
            note = r["note"]
            if drawings and r["drawing"].shape != drawings[0].shape:        # batch frames must share one size
                import cv2
                h0, w0 = drawings[0].shape[:2]
                r["drawing"] = cv2.resize(r["drawing"], (w0, h0), interpolation=cv2.INTER_AREA)
                r["strength"] = cv2.resize(r["strength"], (w0, h0), interpolation=cv2.INTER_AREA)
                note += f", resized to the first photo's {w0}x{h0}"
            drawings.append(r["drawing"])
            lines.append(r["strength"])
            notes.append(note)
            used.append(sk.corners_text(r["corners"]))
            if i == 0:
                founds.append(sk.draw_quad(rgb, r["corners"]))
        d = torch.from_numpy(np.stack(drawings))
        m = torch.from_numpy(np.stack(lines))
        proj = m[..., None].expand(-1, -1, -1, 3).contiguous()
        found = _img(founds[0])
        report = "\n".join(f"photo {i}: {n}; corners {c}" for i, (n, c) in enumerate(zip(notes, used)))
        log.info("[KUBA sketch] scan: %d photo(s), %s, %.1f s", photo.shape[0], notes[0], time.perf_counter() - t0)
        for row in report.split("\n")[1:]:
            log.info("[KUBA sketch]   %s", row)
        return io.NodeOutput(d, m, proj, found, used[0], report, ui=ui.PreviewImage(d[:1], cls=cls))


class KUBA_RegionsFromSketch(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="KUBA_RegionsFromSketch",
            display_name="kubakub regions from sketch",
            category="kubakub/2d/sketch",
            search_aliases=['sketch', 'drawing', 'closed shapes', 'cells', 'hand drawn'],
            description=("Every closed shape of a drawing becomes a region. Hand-drawn strokes rarely meet: each "
                         "stroke end is extended to the next line within gap_px. A dot of coloured marker inside a "
                         "shape names it: the region gets the colour as tag and name (red_1 ...), so plan rules like "
                         "[tag:red] pick it. Shapes drawn alike (all the windows) land in one group."),
            inputs=[
                io.Image.Input("drawing", tooltip="From kubakub scan to line (or any line drawing on white)."),
                io.Mask.Input("line", optional=True, tooltip="The line output of scan to line (better than the drawing alone)."),
                io.Int.Input("gap_px", default=24, min=0, max=500,
                             tooltip="Largest gap between strokes that still closes a shape (pixels of the drawing)."),
                io.Float.Input("line_threshold", default=0.3, min=0.01, max=0.99, step=0.01,
                               tooltip="How strong a stroke must be to count as a wall between shapes."),
                io.Boolean.Input("colour_names", default=True,
                                 tooltip="Coloured marker dots name the shape they sit in (and are no wall)."),
                io.Combo.Input("outside", options=["keep", "drop"], default="keep",
                               tooltip="drop = shapes touching the edge of the paper are no regions (the sky "
                                       "around a drawn building)."),
                io.Int.Input("min_region_area", default=400, min=0, max=100_000_000, step=8,
                             tooltip="Shapes smaller than this (pixels) - hatching, little marks - are merged or dropped."),
                io.Boolean.Input("merge_small_regions", default=True,
                                 tooltip="On: small shapes join the neighbour with the longest border. Off: dropped."),
                io.Image.Input("matrix", optional=True,
                               tooltip="Your matrix: the preview background, and the size of the regions (a drawing "
                                       "of another size is fitted to it; same proportions only)."),
                io.Mask.Input("scope", optional=True,
                              tooltip="Pixels outside this mask are never part of a region (the size of the "
                                      "matrix, or of the drawing without one)."),
            ],
            outputs=[
                RegionsType.Output("regions", tooltip="Label map + region table for the kubakub region nodes."),
                io.Mask.Output("region_masks", tooltip="One mask per region; batch index = region_id."),
                io.String.Output("regions_json", tooltip="The region table as text."),
                io.Image.Output("preview", tooltip="Regions over the drawing (or the matrix), one colour per group."),
                io.String.Output("report", tooltip="Regions and groups found, gaps bridged, shapes named by colour, "
                                                   "small shapes merged or dropped."),
                io.Mask.Output("closed_lines", tooltip="The lines the shapes were cut with: your strokes plus the "
                                                       "bridges over the gaps (white = wall). Shows where gap_px closed "
                                                       "a shape, or still leaves one open."),
            ],
        )

    @classmethod
    def execute(cls, drawing, gap_px, line_threshold, colour_names, outside, min_region_area, merge_small_regions,
                line=None, matrix=None, scope=None) -> io.NodeOutput:
        import cv2
        t0 = time.perf_counter()
        d = drawing[0, ..., :3].cpu().float().numpy()
        if line is not None:
            s = (line[0] if line.ndim == 3 else line).cpu().float().numpy()
        else:
            s = np.clip((1.0 - d.min(axis=2) - 0.1) / 0.6, 0, 1).astype(np.float32)
        H, W = d.shape[:2]
        if s.shape != (H, W):
            s = cv2.resize(s, (W, H), interpolation=cv2.INTER_LINEAR)
        notes = []
        gap = float(gap_px)
        if matrix is not None and (int(matrix.shape[1]), int(matrix.shape[2])) != (H, W):
            mh, mw = int(matrix.shape[1]), int(matrix.shape[2])
            if abs(W / H - mw / mh) > 0.005 * (mw / mh):
                raise ValueError(f"the drawing is {W}x{H}, the matrix {mw}x{mh}: another aspect, one uniform scale "
                                 "cannot fit it. Set width and height of kubakub scan to line to the matrix.")
            interp = cv2.INTER_AREA if mw < W else cv2.INTER_LINEAR
            d = np.clip(cv2.resize(d, (mw, mh), interpolation=interp), 0, 1)
            s = np.clip(cv2.resize(s, (mw, mh), interpolation=interp), 0, 1)
            gap *= mw / W                                  # gap_px stays in pixels of the drawing
            notes.append(f"drawing {W}x{H} fitted to the matrix {mw}x{mh}")
            H, W = mh, mw
        given = None
        if scope is not None:
            given = (scope[0] if scope.ndim == 3 else scope).cpu().numpy() > 0.5
            if given.shape != (H, W):
                raise ValueError(f"scope mask is {given.shape[1]}x{given.shape[0]}, the regions {W}x{H}; they must "
                                 "match exactly.")
        cells, tags, n_br, closed = sk.sketch_cells(s, d, gap, float(line_threshold), use_colour=bool(colour_names))
        n = int(cells.max()) + 1
        scope = given
        notes.append(f"{n_br} gap(s) between strokes bridged (gap_px {gap_px})")
        if given is not None:
            notes.append(f"scope: regions only on {given.mean() * 100:.0f} % of the drawing")
        if outside == "drop":
            edge = np.unique(np.r_[cells[0], cells[-1], cells[:, 0], cells[:, -1]])
            edge = edge[edge >= 0]
            if len(edge):
                inner = ~np.isin(cells, edge)
                scope = inner if scope is None else (scope & inner)
                notes.append(f"{len(edge)} shape(s) touching the edge dropped")
        labels = fc.fill_unassigned(cells)
        count = {}
        meta = []
        for cid in range(n):
            t = tags.get(cid)
            if t:
                count[t[0]] = count.get(t[0], 0) + 1
                meta.append({"name": f"{t[0]}_{count[t[0]]}", "tags": list(t)})
            else:
                meta.append({})
        labels, atlas, scope_used = fc._finish_atlas(labels, meta, notes, [], scope, "line_drawing",
                                                     int(min_region_area), bool(merge_small_regions),
                                                     with_scope=True)
        regions = Regions.from_numpy(labels, atlas, scope_used)
        bg = matrix[0, ..., :3].cpu().float().numpy() if matrix is not None else d
        preview = _img(fc.render_preview(bg, labels, atlas))
        report = "\n".join([
            f"{len(atlas['regions'])} regions in {len(atlas['groups'])} groups, {W}x{H}, {time.perf_counter() - t0:.1f} s",
            "named by colour: " + (", ".join(f"{k} {v}" for k, v in count.items()) or "none"),
            *atlas["notes"]])
        log.info("[KUBA sketch] regions: %s", report.replace("\n", " | "))
        return io.NodeOutput(regions, regions.masks(), json.dumps(atlas, indent=1), preview, report,
                             torch.from_numpy(closed.astype(np.float32))[None], ui=ui.PreviewImage(preview, cls=cls))


class KUBA_LineOverlay(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="KUBA_LineOverlay",
            display_name="kubakub line overlay",
            category="kubakub/2d/sketch",
            search_aliases=['sketch', 'keep my line', 'outline', 'trace'],
            description=("Your own line on top of the render: amount 0.3 = a light trace of the hand, 1 = exactly "
                         "your stroke. multiply = dark line, screen = light line (on dark projection content), "
                         "colour = paint the line in one colour."),
            inputs=[
                io.Image.Input("image", tooltip="The render (one image or frames)."),
                io.Mask.Input("line", tooltip="From scan to line (one mask for all frames, or one per frame)."),
                io.Float.Input("amount", default=0.8, min=0.0, max=1.0, step=0.05, tooltip="How much of the line shows."),
                io.Combo.Input("mode", options=["multiply", "screen", "colour"], default="multiply",
                               tooltip="multiply = dark line, screen = light line, colour = the line in the colour below."),
                io.String.Input("colour", default="#000000", tooltip="Line colour (hex). Black for multiply, white for screen."),
                io.Int.Input("grow_px", default=0, min=0, max=64, tooltip="Makes the line bolder."),
                io.Float.Input("soften_px", default=0.0, min=0.0, max=64.0, step=0.5, tooltip="Softens the line (a glow with screen)."),
            ],
            outputs=[io.Image.Output("image", tooltip="The render with your line."),
                     io.Mask.Output("mask", tooltip="The line as it was laid on the render: after grow_px and "
                                                    "soften_px, before amount (0..1, one per frame).")],
        )

    @classmethod
    def execute(cls, image, line, amount, mode, colour, grow_px, soften_px) -> io.NodeOutput:
        lm = line if line.ndim == 3 else line[None]
        out, masks, s = [], [], None
        for i in range(int(image.shape[0])):
            if i < lm.shape[0]:                         # one line for every frame is shaped once
                s = lm[i].cpu().float().numpy()
                s = sk.line_shape(s, int(image.shape[1]), int(image.shape[2]), int(grow_px), float(soften_px))
            out.append(sk.overlay(image[i].cpu().float().numpy(), s, float(amount), mode, colour))
            masks.append(s)
        return io.NodeOutput(torch.from_numpy(np.stack(out)), torch.from_numpy(np.stack(masks)))


NODE_CLASS_MAPPINGS = {"KUBA_ScanToLine": KUBA_ScanToLine, "KUBA_RegionsFromSketch": KUBA_RegionsFromSketch,
                       "KUBA_LineOverlay": KUBA_LineOverlay}
NODE_DISPLAY_NAME_MAPPINGS = {"KUBA_ScanToLine": "kubakub scan to line",
                              "KUBA_RegionsFromSketch": "kubakub regions from sketch",
                              "KUBA_LineOverlay": "kubakub line overlay"}
