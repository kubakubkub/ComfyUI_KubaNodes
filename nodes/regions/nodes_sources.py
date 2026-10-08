"""
nodes_sources.py

Region sources (category KUBAKUB/regions): nodes that turn masks from anywhere
(SAM, Illustrator layers, ID renders, hand painted) into KUBA_REGIONS.
KUBA_RegionsFromMasks: MASK batch + names -> KUBA_REGIONS, or patched into an
existing atlas. The array work is facade_core.atlas_from_masks().
KUBA_RegionsSAM3Masks: named masks from core SAM3 Detect (text, points, boxes)
with a full resolution crop pass; helpers in sam_prompts.py.
KUBA_RegionsFromIllustrator: regions from the layers of an .ai / .pdf; logic in
illustrator.py.
KUBA_RegionsFromIDMaps: regions from 3D ID renders + legends; logic in idmaps.py.

V3 node schema (comfy_api.latest), registered through the pack's
NODE_CLASS_MAPPINGS loader in ../__init__.py.
"""

from __future__ import annotations

import json
import logging
import os
import time

import numpy as np
import torch

from comfy_api.latest import io, ui
from comfy_execution.graph_utils import ExecutionBlocker

from ...kubakub import facade_core as fc
from ...kubakub import save_paths
from ...kubakub import sam_prompts as sp
from ...kubakub.io_types import RegionsType
from ...kubakub.types import Regions

log = logging.getLogger("KUBA.regions")

CATEGORY = "kubakub/2d/regions"


class KUBA_RegionsFromMasks(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="KUBA_RegionsFromMasks",
            display_name="kubakub regions from masks",
            category="kubakub/2d/regions",
            search_aliases=['masks', 'patch'],
            description=(
                "Turn any set of masks (SAM, painted, layers, ID passes) into named regions you can "
                "prompt one by one. One region per mask, named by the names list. Where new masks overlap, the smaller one wins. With "
                "base_regions the masks are patched into that atlas: on_top cuts them out of the "
                "regions below (a window out of a wall), underneath only fills pixels without a "
                "region. New regions are tagged with the base region they were cut from."),
            inputs=[
                io.Mask.Input("masks", tooltip="One mask per region (batch). Values over 0.5 are inside."),
                io.String.Input("names", multiline=True, default="",
                                placeholder="W_F1_01\nW_F1_02 = windows\nbalcony = ironwork",
                                tooltip="One name per line in batch order, or one comma separated line. "
                                        "'name = group' sets the group; otherwise the group is the name "
                                        "without its trailing number (W_F1_02 -> W_F1). Empty: mask_01.."),
                RegionsType.Input("base_regions", optional=True,
                                  tooltip="Existing atlas to patch. Its scope, names, groups and tags "
                                          "are kept."),
                io.Combo.Input("patch", options=list(fc.PATCH_MODES), default="on_top",
                               tooltip="With base_regions. on_top: new masks cut themselves out of the "
                                       "regions below. underneath: new masks only fill pixels no base "
                                       "region has."),
                io.Image.Input("matrix", optional=True,
                               tooltip="Background for the preview; also sets the size when there are "
                                       "no base_regions (masks of another size are resized, nearest)."),
                io.Mask.Input("scope", optional=True,
                              tooltip="Pixels outside this mask are never part of a region."),
                io.Int.Input("min_region_area", default=64, min=0, max=100_000_000, step=8,
                             tooltip="Regions smaller than this (pixels) are merged or dropped."),
                io.Boolean.Input("merge_small_regions", default=True,
                                 tooltip="On: merge small regions into the neighbour with the longest "
                                         "shared border. Off: drop them."),
                io.String.Input("scope_masks", default="", optional=True, advanced=True,
                                placeholder="M_Projection_Range",
                                tooltip="Names (wildcards ok) of masks that are no regions but the area "
                                        "regions may use."),
                io.String.Input("split_masks", default="", optional=True, advanced=True,
                                placeholder="M_Pilasters",
                                tooltip="Names of masks holding several shapes: one region per shape, "
                                        "<name>_01, _02 ... in group <name>."),
                io.String.Input("tag_only_masks", default="", optional=True, advanced=True,
                                placeholder="M_FLOOR_*",
                                tooltip="Names of masks that only tag the regions they cover."),
            ],
            outputs=[
                RegionsType.Output("regions", tooltip="Label map + region table for the kubakub region nodes."),
                io.Mask.Output("region_masks", tooltip="One mask per region; batch index = region_id."),
                io.String.Output("regions_json", tooltip="The region table as text (names, groups, tags, "
                                                         "boxes, notes)."),
                io.Image.Output("preview", tooltip="One colour per group, region borders and ids over "
                                                   "a grey copy of the matrix."),
                io.String.Output("report", tooltip="How many masks became how many regions and groups, pixels "
                                                   "left without a region, and the notes (renamed, resized, "
                                                   "merged or dropped masks)."),
                io.Mask.Output("scope", tooltip="The scope that was used: 1 where regions may be (all 1 "
                                                "without a scope)."),
            ],
        )

    @classmethod
    def execute(cls, masks, names, patch, min_region_area, merge_small_regions, base_regions=None,
                matrix=None, scope=None, scope_masks="", split_masks="", tag_only_masks="") -> io.NodeOutput:
        t0 = time.perf_counter()
        if masks.ndim == 2:
            masks = masks[None]
        if base_regions is not None:
            if base_regions.labels.shape[0] > 1:
                log.warning("[KUBA regions] base_regions has %d frames; using the first.",
                            base_regions.labels.shape[0])
            width, height = base_regions.size
        elif matrix is not None:
            height, width = int(matrix.shape[1]), int(matrix.shape[2])
        else:
            height, width = int(masks.shape[1]), int(masks.shape[2])
        if matrix is not None and (int(matrix.shape[1]), int(matrix.shape[2])) != (height, width):
            raise ValueError(f"matrix is {matrix.shape[2]}x{matrix.shape[1]}, base_regions "
                             f"{width}x{height}; they must match.")

        named, name_notes = fc.parse_mask_names(names, int(masks.shape[0]))
        m_np = masks.detach().cpu().numpy() > 0.5
        entries = [{"source": f"mask {i}", "name": n, "group": g, "mask": m_np[i]}
                   for i, (n, g) in enumerate(named)]

        scope_np = None
        if scope is not None:
            sc = scope[0] if scope.ndim == 3 else scope
            if tuple(sc.shape) != (height, width):
                raise ValueError(f"scope mask is {sc.shape[1]}x{sc.shape[0]}, the regions "
                                 f"{width}x{height}; they must match exactly.")
            scope_np = sc.detach().cpu().numpy() > 0.5

        base = None
        if base_regions is not None:
            bs = None if base_regions.scope is None else base_regions.scope[0].cpu().numpy() > 0.5
            base = (base_regions.labels[0].cpu().numpy(), base_regions.table, bs)

        labels, atlas, scope_used = fc.atlas_from_masks(
            entries, width, height, base=base, patch=patch, scope=scope_np,
            min_region_area=min_region_area, merge_small_regions=merge_small_regions,
            scope_masks=scope_masks or "", tag_only_masks=tag_only_masks or "",
            split_masks=split_masks or "", with_scope=True)
        atlas["notes"] = name_notes + atlas["notes"]

        regions = Regions.from_numpy(labels, atlas, scope_used)
        bg = (matrix[0].detach().cpu().float().numpy() if matrix is not None
              else np.full((height, width, 3), 0.5, dtype=np.float32))
        preview = torch.from_numpy(fc.render_preview(bg, labels, atlas))[None]

        log.info("[KUBA regions] from masks: %d masks -> %d regions in %d groups (%s), %d px "
                 "unassigned, %.2fs", masks.shape[0], len(atlas["regions"]), len(atlas["groups"]),
                 atlas["mode"], atlas["unassigned_px"], time.perf_counter() - t0)
        for note in atlas["notes"]:
            log.info("[KUBA regions]   %s", note)
        report = "\n".join([
            f"{int(masks.shape[0])} masks -> {len(atlas['regions'])} regions in {len(atlas['groups'])} groups, "
            f"{width}x{height}, {atlas['unassigned_px']} px without a region", *atlas["notes"]])
        scope_out = regions.scope if regions.scope is not None else torch.ones((1, height, width))

        return io.NodeOutput(regions, regions.masks(), json.dumps(atlas, indent=1), preview, report, scope_out,
                             ui=ui.PreviewImage(preview, cls=cls))


# --------------------------------------------------------------------------
# SAM3 masks
# --------------------------------------------------------------------------

BBoxType = io.Custom("BBOX")   # KJNodes Points Editor box output (list of xyxy tuples)


def _core_boxes(boxes):
    """Core BOUNDING_BOX (one dict, a list of dicts, or one list per frame) -> what sam_prompts.parse_boxes reads."""
    if isinstance(boxes, (list, tuple)) and boxes and all(isinstance(b, (list, tuple)) for b in boxes) \
            and any(b and isinstance(b[0], dict) for b in boxes):
        return list(boxes[0])          # per frame: this node segments the first image
    return boxes
POINT_MODES = ("one region per point", "all points one region")


def _detect(model, image, **kwargs):
    """Core SAM3_Detect, per-object masks. Returns (masks [N, H, W] float, box dicts of frame 0)."""
    from comfy_extras.nodes_sam3 import SAM3_Detect
    out = SAM3_Detect.execute(model, image, individual_masks=True, **kwargs)
    masks, boxes = out.args[0], out.args[1]
    return masks.float().cpu(), (boxes[0] if boxes else [])


def _pts_json(pts):
    return json.dumps([{"x": int(round(x)), "y": int(round(y))} for x, y in pts]) if pts else None


def _detail(model, image, coarse, win, threshold, refine, box):
    """Segment the box (x0, y0, x1, y1) again on the crop win at full resolution; paste back."""
    x0, y0, x1, y1 = win
    crop = image[:, y0:y1, x0:x1, :]
    bx0, by0, bx1, by1 = box
    masks, _ = _detect(model, crop, threshold=threshold, refine_iterations=refine,
                       bboxes=[{"x": bx0 - x0, "y": by0 - y0, "width": bx1 - bx0, "height": by1 - by0}])
    if masks.shape[0] == 0:
        return None
    m = (masks > 0.5).any(dim=0).numpy()
    if not m.any():
        return None
    full = np.zeros_like(coarse)
    full[y0:y1, x0:x1] = m
    return full


class KUBA_RegionsSAM3Masks(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="KUBA_RegionsSAM3Masks",
            display_name="kubakub regions from sam3",
            category="kubakub/2d/regions",
            search_aliases=['sam', 'segment', 'points', 'text prompt'],
            description=(
                "Find facade parts automatically and get one named mask per object, ready for "
                "regions from masks. Uses SAM 3.1 (core SAM3 Detect) with text prompts "
                "(one line per kind of object, every hit its own mask), points from the KJNodes Points "
                "Editor (one object per positive point) and boxes. Points and boxes get a second pass "
                "on a crop at full resolution (detail), so small facade parts keep their edges. "
                "Optionally saves every mask as <name>.png into a project Masks folder."),
            inputs=[
                io.Model.Input("model", tooltip="SAM 3.1: Load Checkpoint with sam3.1_multiplex_fp16."),
                io.Image.Input("image", tooltip="The matrix or render to segment (first image of a batch)."),
                io.Clip.Input("clip", optional=True,
                              tooltip="The CLIP output of the same Load Checkpoint; needed for text prompts."),
                io.String.Input("prompts", multiline=True, default="",
                                placeholder="window = window : 40\nbalcony = wrought iron balcony\ndoor",
                                tooltip="One line per kind of object: 'name = text' or just 'text'. "
                                        "':N' caps the hits (default 50). Every hit becomes a mask "
                                        "name_01, name_02 ... in reading order."),
                io.String.Input("positive_coords", force_input=True, optional=True,
                                tooltip="Points JSON (KJNodes Points Editor positive_coords, in image "
                                        "pixels or normalized)."),
                io.String.Input("negative_coords", force_input=True, optional=True,
                                tooltip="Points that must stay outside (Points Editor negative_coords)."),
                BBoxType.Input("bboxes", optional=True,
                               tooltip="Boxes (Points Editor bbox, xyxy); one object per box."),
                io.String.Input("point_names", multiline=True, default="", optional=True,
                                placeholder="W_F1_01\nW_F1_02",
                                tooltip="Names for the point objects in click order, then the boxes "
                                        "(bboxes first, then boxes). Missing: point_01.., box_01.."),
                io.Combo.Input("point_mode", options=list(POINT_MODES), default=POINT_MODES[0],
                               tooltip="one region per point: each positive point is its own object "
                                       "(negative points apply to all). all points one region: core "
                                       "behaviour, one object from all points."),
                io.Float.Input("threshold", advanced=True, default=0.5, min=0.0, max=1.0, step=0.01,
                               tooltip="Text detection confidence."),
                io.Int.Input("refine_iterations", advanced=True, default=2, min=0, max=5,
                             tooltip="SAM decoder refinement passes (core)."),
                io.Boolean.Input("detail", default=True,
                                 tooltip="Points and boxes: segment again on a crop around the object, "
                                         "so the 1008 px model input covers the object instead of the "
                                         "whole 4K frame."),
                io.Boolean.Input("drop_groups", advanced=True, default=True,
                                 tooltip="Text: drop hits that cover two or more other hits of the same "
                                         "prompt (SAM often adds one mask for the whole row)."),
                io.Int.Input("min_area", advanced=True, default=64, min=0, max=10_000_000,
                             tooltip="Masks smaller than this (pixels) are dropped."),
                io.Mask.Input("scope", optional=True,
                              tooltip="Masks are clipped to this; objects mostly outside it are dropped."),
                io.String.Input("save_folder", advanced=True, default="", optional=True,
                                placeholder="D:\\PROJECT\\Masks\\SAM",
                                tooltip="Optional: write every mask as <name>.png (white on black, image "
                                        "size) into this folder, e.g. the project Masks folder."),
                io.Boolean.Input("overwrite", advanced=True, default=False, optional=True,
                                 tooltip="Replace existing PNGs of the same name in save_folder."),
                io.BoundingBox.Input("boxes", force_input=True, optional=True,
                                     tooltip="Boxes from core nodes (the bboxes output of SAM3 Detect, in image "
                                             "pixels); one object per box, added after the Points Editor "
                                             "boxes. Of a batch, the boxes of the first image are used."),
            ],
            outputs=[
                io.Mask.Output("masks", tooltip="One mask per object (batch), same order as names."),
                io.String.Output("names", tooltip="One name per line; plug into regions from masks."),
                io.Image.Output("preview", tooltip="The found objects coloured and numbered over the image."),
                io.String.Output("report", tooltip="Every object with its size in pixels, how it was found "
                                                   "and its score, plus notes (dropped hits, saved files)."),
            ],
        )

    @classmethod
    def execute(cls, model, image, prompts, point_mode, threshold, refine_iterations, detail, drop_groups,
                min_area, clip=None, positive_coords=None, negative_coords=None, bboxes=None, point_names="",
                scope=None, save_folder="", overwrite=False, boxes=None) -> io.NodeOutput:
        t0 = time.perf_counter()
        img = image[:1]
        H, W = int(img.shape[1]), int(img.shape[2])
        lines, notes = sp.parse_prompt_lines(prompts)
        pos = sp.parse_points(positive_coords, W, H)
        neg = sp.parse_points(negative_coords, W, H)
        boxes = sp.parse_boxes(bboxes) + sp.parse_boxes(_core_boxes(boxes))   # Points Editor boxes first, then core
        if lines and clip is None:
            raise ValueError("prompts need the clip input (CLIP output of the SAM 3.1 Load Checkpoint).")
        if not (lines or pos or boxes):
            raise ValueError("Nothing to segment: write prompts, connect points or boxes.")

        objects = []   # (name, bool mask, how, score)

        for name, text, max_det in lines:
            cond = clip.encode_from_tokens_scheduled(clip.tokenize(f"{text}:{max_det}"))
            masks, dets = _detect(model, img, conditioning=cond, threshold=threshold,
                                  refine_iterations=refine_iterations)
            ms = [(masks[i] > 0.5).numpy() for i in range(masks.shape[0])]
            keep = [i for i, m in enumerate(ms) if m.any()]
            if not keep:
                notes.append(f"'{text}': nothing above threshold {threshold}")
                continue
            if drop_groups and len(keep) > 2:
                cont = {keep[c] for c in sp.container_hits([ms[i] for i in keep])}
                if cont:
                    keep = [i for i in keep if i not in cont]
                    notes.append(f"'{text}': {len(cont)} hit(s) covering a whole group dropped")
            bbs = np.asarray([sp.mask_bbox(ms[i]) for i in keep], dtype=np.int64)
            order = [keep[j] for j in fc.reading_order(bbs)]
            names = sp.numbered([name] * len(order)) if len(order) > 1 else [name]
            for n, i in zip(names, order):
                score = dets[i]["score"] if i < len(dets) else None
                objects.append((n, ms[i], f"text '{text}'", score))
            cap = " (cap reached, raise :N)" if len(order) >= max_det else ""
            notes.append(f"'{text}': {len(order)} hit(s){cap}")

        given = [sp.safe_name(n.split("//", 1)[0]) for n in (point_names or "").splitlines()
                 if n.split("//", 1)[0].strip()]
        groups = [[p] for p in pos] if point_mode == POINT_MODES[0] else ([pos] if pos else [])
        for k, grp in enumerate(groups):
            name = given[k] if k < len(given) else f"point_{k + 1:02d}"
            masks, _ = _detect(model, img, positive_coords=_pts_json(grp), negative_coords=_pts_json(neg),
                               refine_iterations=refine_iterations, threshold=threshold)
            m = (masks > 0.5).any(dim=0).numpy() if masks.shape[0] else np.zeros((H, W), bool)
            m = sp.parts_at(m, grp) if m.any() else m
            how = "points" if len(grp) > 1 else f"point {int(grp[0][0])},{int(grp[0][1])}"
            if detail and m.any():
                # A single click is ambiguous on a zoomed crop (one baluster or the whole
                # balustrade), so the crop is prompted with the coarse mask's box instead.
                bb = sp.mask_bbox(m)
                win = sp.detail_window(bb, W, H)
                fine = _detail(model, img, m, win, threshold, refine_iterations, box=bb) if win else None
                if fine is not None:
                    fine = sp.parts_at(fine, grp)
                    ratio = fine.sum() / max(1, m.sum())
                    if 0.3 <= ratio <= 3.0:
                        m, how = fine, how + " + detail"
                    else:
                        notes.append(f"{name}: detail mask {ratio:.1f}x the coarse one, coarse kept")
                elif win:
                    notes.append(f"{name}: detail pass found nothing, coarse mask kept")
            objects.append((name, m, how, None))

        for k, box in enumerate(boxes):
            j = len(groups) + k
            name = given[j] if j < len(given) else f"box_{k + 1:02d}"
            x0, y0, x1, y1 = box
            masks, _ = _detect(model, img, bboxes=[{"x": x0, "y": y0, "width": x1 - x0, "height": y1 - y0}],
                               refine_iterations=refine_iterations, threshold=threshold)
            m = (masks > 0.5).any(dim=0).numpy() if masks.shape[0] else np.zeros((H, W), bool)
            how = "box"
            if detail:
                win = sp.detail_window(tuple(int(round(v)) for v in box), W, H)
                fine = _detail(model, img, m, win, threshold, refine_iterations, box=box) if win else None
                ratio = fine.sum() / max(1, m.sum()) if fine is not None else 0
                if fine is not None and (not m.any() or 0.3 <= ratio <= 3.0):
                    m, how = fine, "box + detail"
                elif fine is not None:
                    notes.append(f"{name}: detail mask {ratio:.1f}x the coarse one, coarse kept")
                elif win:
                    notes.append(f"{name}: detail pass found nothing, coarse mask kept")
            objects.append((name, m, how, None))

        if scope is not None:
            sc = (scope[0] if scope.ndim == 3 else scope).cpu().numpy() > 0.5
            if sc.shape != (H, W):
                raise ValueError(f"scope is {sc.shape[1]}x{sc.shape[0]}, the image {W}x{H}.")
            kept = []
            for n, m, how, s in objects:
                if m.sum() and (m & sc).sum() < 0.5 * m.sum():
                    notes.append(f"{n}: mostly outside the scope, dropped")
                    continue
                kept.append((n, m & sc, how, s))
            objects = kept
        small = {i for i, (_, m, _, _) in enumerate(objects) if m.sum() < max(1, min_area)}   # by object: names repeat
        if small:
            notes.append(f"under {min_area} px, dropped: {', '.join(objects[i][0] for i in sorted(small))}")
        objects = [o for i, o in enumerate(objects) if i not in small]
        final = sp.numbered([n for n, *_ in objects])
        objects = [(f, *o[1:]) for f, o in zip(final, objects)]
        if not objects:
            raise ValueError("SAM3 found nothing. Lower the threshold, check the prompts or points. "
                             + "; ".join(notes))

        import folder_paths
        folder = save_paths.save_folder(save_folder, folder_paths.get_output_directory())
        if folder:
            import cv2
            os.makedirs(folder, exist_ok=True)
            saved = 0
            for n, m, _, _ in objects:
                path = os.path.join(folder, f"{n}.png")
                if os.path.exists(path) and not overwrite:
                    notes.append(f"{n}.png exists, not overwritten")
                    continue
                ok, buf = cv2.imencode(".png", m.astype(np.uint8) * 255)
                if ok:
                    buf.tofile(path)   # tofile: non-ASCII paths work on Windows
                    saved += 1
            notes.append(f"saved {saved} PNG(s) to {folder}")

        out = torch.from_numpy(np.stack([m for _, m, _, _ in objects]).astype(np.float32))
        entries = [{"source": n, "name": n, "group": None, "mask": m} for n, m, _, _ in objects]
        labels, atlas = fc.atlas_from_masks(entries, W, H, min_region_area=0, merge_small_regions=False)
        preview = torch.from_numpy(fc.render_preview(img[0].cpu().float().numpy(), labels, atlas))[None]

        rows = [f"{n:<24} {int(m.sum()):>9} px  {how}" + (f"  score {s:.2f}" if s is not None else "")
                for n, m, how, s in objects]
        report = "\n".join([f"SAM3 masks: {len(objects)} object(s), {time.perf_counter() - t0:.1f} s", *rows,
                            *([""] + notes if notes else [])])
        log.info("[KUBA regions] %s", report.replace("\n", "\n    "))
        return io.NodeOutput(out, "\n".join(n for n, *_ in objects), preview, report,
                             ui=ui.PreviewImage(preview, cls=cls))


# --------------------------------------------------------------------------
# Illustrator
# --------------------------------------------------------------------------

class KUBA_RegionsFromIllustrator(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="KUBA_RegionsFromIllustrator",
            display_name="kubakub regions from illustrator",
            category="kubakub/2d/regions",
            search_aliases=['ai', 'pdf', 'layers', 'vector'],
            description=(
                "Regions straight from an Illustrator file (.ai saved PDF compatible, or .pdf). "
                "Every layer is read, hidden ones too. Line drawing layers are cut into the cells "
                "between their lines (<LAYER>_001.., grouped by shape), filled layers give one region per "
                "shape on top, a mask layer that covers the outside cuts the scope. Roles are guessed "
                "from the content and can be set per layer. The artboard maps onto the matrix with one "
                "uniform scale. Also outputs the line drawing, the placed photo and every layer as a mask."),
            inputs=[
                io.String.Input("ai_file", default="", placeholder="paste the path of your .ai / .pdf  (empty = a sample)",
                                tooltip="Path of the .ai or .pdf (Explorer 'Copy as path' quotes are fine)."),
                io.Image.Input("matrix", optional=True,
                               tooltip="The matrix: sets the output size (the artboard must have its aspect) "
                                       "and is the preview background."),
                io.String.Input("layer_roles", multiline=True, default="", optional=True,
                                placeholder="MASK = outside\nCENTRE, COUR, JAR = lines\nOBSTACLES = shapes\n"
                                            "TEXT, Background = ignore\n_IMG = reference",
                                tooltip="Optional, one line per role: '<layer names or wildcards> = role'. "
                                        "Roles: lines (cells between the lines), shapes (one region per filled "
                                        "shape, on top), outside (not projected on), scope (only here), tag "
                                        "(tags the regions it covers), reference (photo output), ignore. "
                                        "Layers not listed get a role from their content (see the report)."),
                io.Int.Input("min_region_area", default=400, min=0, max=100_000_000, step=10,
                             tooltip="Cells and shapes smaller than this (pixels) are merged into a neighbour "
                                     "or dropped."),
                io.Boolean.Input("merge_small_regions", default=True,
                                 tooltip="On: merge small cells into the neighbour with the longest border. "
                                         "Off: drop them."),
                io.Int.Input("line_gap_close_px", default=1, min=0, max=32,
                             tooltip="Thicken the lines by this much first, so small gaps in the drawing "
                                     "don't leak one cell into the next."),
                io.Boolean.Input("split_shapes", default=True,
                                 tooltip="Shape layers: one region per separate shape (on) or one region per "
                                         "layer (off)."),
                io.Float.Input("scale", default=1.0, min=0.05, max=16.0, step=0.05, advanced=True,
                               tooltip="Without matrix: pixels per point (1 = the artboard size in points)."),
                io.Mask.Input("scope", optional=True,
                              tooltip="Pixels outside this mask are never part of a region."),
            ],
            outputs=[
                RegionsType.Output("regions", tooltip="Label map + region table for the kubakub region nodes."),
                io.Image.Output("preview", tooltip="One colour per group, region borders and ids over "
                                                   "the matrix, the photo or the drawing."),
                io.Image.Output("lines", tooltip="The line layers, white on black (a control image)."),
                io.Image.Output("reference", tooltip="The placed photo (reference layers); black if none."),
                io.Mask.Output("layer_masks", tooltip="Every layer with paths as a mask (what it covers)."),
                io.String.Output("layer_names", tooltip="Names of layer_masks, one per line."),
                io.Mask.Output("scope_mask", tooltip="1 where regions may be."),
                io.String.Output("regions_json", tooltip="The region table as text, with the layer list."),
                io.String.Output("report", tooltip="Every layer with its role, paths and covered pixels, "
                                                   "and regions per layer."),
            ],
        )

    @classmethod
    def fingerprint_inputs(cls, ai_file="", **kwargs):
        # the file can change on disk without any input changing
        path = fc.clean_folder_path(ai_file)
        try:
            st = os.stat(path)
            return f"{path}|{st.st_size}|{st.st_mtime_ns}"
        except OSError:
            return path

    @classmethod
    def execute(cls, ai_file, min_region_area, merge_small_regions, line_gap_close_px, split_shapes,
                matrix=None, layer_roles="", scale=1.0, scope=None) -> io.NodeOutput:
        from ...kubakub import illustrator as il
        t0 = time.perf_counter()
        path = fc.clean_folder_path(ai_file)
        sample = not path
        if sample:                                     # nothing pasted yet: a layered PDF of the sample facade
            import folder_paths
            from ...kubakub import samples
            path = samples.illustrator(folder_paths.get_temp_directory())
        if not os.path.isfile(path):
            raise ValueError(f"ai_file does not exist: {path}")
        width = height = None
        if matrix is not None:
            height, width = int(matrix.shape[1]), int(matrix.shape[2])
        sc = None
        if scope is not None:
            sc = (scope[0] if scope.ndim == 3 else scope).cpu().numpy() > 0.5

        r = il.build(path, width=width, height=height, scale=scale, roles_text=layer_roles or "",
                     min_region_area=min_region_area, merge_small_regions=merge_small_regions,
                     line_gap_close_px=line_gap_close_px, split_shapes=split_shapes, scope=sc)
        labels, atlas = r["labels"], r["atlas"]
        W, H = r["size"]
        regions = Regions.from_numpy(labels, atlas, r["scope"])

        lines = torch.from_numpy(np.repeat(r["lines"][..., None], 3, axis=2).astype(np.float32))[None]
        ref = r["reference"]
        reference = torch.from_numpy(ref if ref is not None else np.zeros((H, W, 3), np.float32))[None]
        names = [n for n in r["layer_masks"]]
        layer_masks = (torch.from_numpy(np.stack([r["layer_masks"][n] for n in names]).astype(np.float32))
                       if names else torch.zeros((1, H, W)))
        scope_out = (torch.from_numpy(r["scope"].astype(np.float32))[None] if r["scope"] is not None
                     else torch.ones((1, H, W)))
        if matrix is not None:
            bg = matrix[0].detach().cpu().float().numpy()
        elif ref is not None:
            bg = ref
        else:
            bg = np.repeat(r["lines"][..., None], 3, axis=2).astype(np.float32) * 0.6 + 0.2
        preview = torch.from_numpy(fc.render_preview(bg, labels, atlas))[None]

        rows = [f"{x['layer']:<16} {x['role']:<9} {x['paths']:>6} paths {x['images']:>2} img  "
                f"{'visible' if x['visible_in_file'] else 'hidden ':<7} {x['covered_px']:>9} px"
                for x in atlas["layers"]]
        counts = {}
        for reg in atlas["regions"]:
            key = reg["group_id"].split("_g")[0] if "_g" in str(reg["group_id"]) else reg["group_id"]
            counts[key] = counts.get(key, 0) + 1
        report = "\n".join([
            f"{atlas['source_file']}: {len(atlas['regions'])} regions in {len(atlas['groups'])} groups, "
            f"{W}x{H}, {time.perf_counter() - t0:.1f} s",
            "layer            role       paths / images / in file / covered", *rows,
            "regions per layer: " + ", ".join(f"{k} {v}" for k, v in counts.items()),
            "", *atlas["notes"],
            *(["no Illustrator file given: the built-in sample is used. Paste the path of your own .ai / .pdf into 'ai_file'."]
              if sample else [])])
        log.info("[KUBA regions] %s", report.replace("\n", "\n    "))
        return io.NodeOutput(regions, preview, lines, reference, layer_masks, "\n".join(names), scope_out,
                             json.dumps(atlas, indent=1), report, ui=ui.PreviewImage(preview, cls=cls))


# --------------------------------------------------------------------------
# ID maps (3D renders)
# --------------------------------------------------------------------------

class KUBA_RegionsFromIDMaps(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        from ...kubakub import idmaps as im
        return io.Schema(
            node_id="KUBA_RegionsFromIDMaps",
            display_name="kubakub regions from id renders",
            category="kubakub/3d/from renders",
            search_aliases=['houdini', 'id map', 'cryptomatte', '3d'],
            description=(
                "Turn ID renders of your 3D model into named regions, so every element of the "
                "building can get its own prompt. Reads a folder (from Houdini, kubakub scene "
                "render ...) with ids_<pass>.png + "
                "ids_<pass>.txt legends ('<name> rgb r g b') and optional full-res mask_<name>.exr/png. "
                "One pass gives the regions (elements), the other passes tag them (zone, level, bay). "
                "Object masks overlap; the ID map decides what is in front. Split parts are numbered "
                "windows_01, windows_02 ... in reading order."),
            inputs=[
                io.String.Input("folder", default="", placeholder="paste the folder of your ID renders  (empty = a sample)",
                                tooltip="Folder with the ID maps, legends and masks (png/ subfolder too)."),
                io.Image.Input("matrix", optional=True,
                               tooltip="Sets the output size (masks are resized to it) and is the preview "
                                       "background. Without: the size of the mask files."),
                io.String.Input("regions_pass", default="elements",
                                tooltip="The pass whose names become regions (ids_<pass>.txt)."),
                io.String.Input("tag_passes", default="*",
                                tooltip="Passes whose names tag the regions they cover by half or more "
                                        "(names or wildcards, e.g. 'zone level bay'); empty = none."),
                io.Combo.Input("overlap", options=list(im.OVERLAPS), default="id_map",
                               tooltip="id_map: what the camera sees wins (depth order learned from the ID "
                                       "map; hidden elements vanish). smaller_wins: a smaller mask is always "
                                       "in front (brings hidden details like a recessed clock back)."),
                io.String.Input("split_parts", default="*",
                                tooltip="Names (wildcards) whose separate parts become separate regions "
                                        "(windows_01..). Empty = one region per name."),
                io.Int.Input("min_region_area", default=400, min=0, max=100_000_000, step=10,
                             tooltip="Parts smaller than this (pixels) are merged into a neighbour or dropped."),
                io.Boolean.Input("merge_small_regions", default=True,
                                 tooltip="On: merge small parts into the neighbour with the longest shared "
                                         "border. Off: drop them."),
                io.Combo.Input("legend_colors", options=["auto", "srgb", "linear"], default="auto", advanced=True,
                               tooltip="How the ID map's pixels relate to the legend (renderers write linear "
                                       "legends, 8 bit PNGs are sRGB). auto picks the better match."),
                io.Mask.Input("scope", optional=True,
                              tooltip="Pixels outside this mask are never part of a region."),
            ],
            outputs=[
                RegionsType.Output("regions", tooltip="Label map + region table for the kubakub region nodes."),
                io.Image.Output("preview", tooltip="One colour per group, region borders and ids over "
                                                   "the matrix or the reference render."),
                io.Image.Output("reference", tooltip="The clay / beauty render of the folder (tone mapped), "
                                                     "black if none."),
                io.String.Output("regions_json", tooltip="The region table as text (names, groups, tags, "
                                                         "passes, notes)."),
                io.String.Output("report", tooltip="Regions per group, the passes found, the reference "
                                                   "render used, and notes."),
            ],
        )

    @classmethod
    def fingerprint_inputs(cls, folder="", **kwargs):
        import glob
        import hashlib
        path = fc.clean_folder_path(folder)
        if not path:          # linked (e.g. from scene render): ComfyUI passes only constants here, and a change
            return "linked"   # upstream reaches this node anyway; glob('*') would hash the working directory
        h = hashlib.sha256(path.encode("utf-8", "replace"))
        for f in sorted(glob.glob(os.path.join(path, "*")) + glob.glob(os.path.join(path, "png", "*"))):
            try:
                st = os.stat(f)
                h.update(f"{f}|{st.st_size}|{st.st_mtime_ns}".encode("utf-8", "replace"))
            except OSError:
                pass
        return h.hexdigest()

    @classmethod
    def execute(cls, folder, regions_pass, tag_passes, overlap, split_parts, min_region_area,
                merge_small_regions, legend_colors="auto", matrix=None, scope=None) -> io.NodeOutput:
        from ...kubakub import idmaps as im
        t0 = time.perf_counter()
        width = height = None
        if matrix is not None:
            height, width = int(matrix.shape[1]), int(matrix.shape[2])
        sc = None
        if scope is not None:
            sc = (scope[0] if scope.ndim == 3 else scope).cpu().numpy() > 0.5
        r_note = ""
        sample = not fc.clean_folder_path(folder)
        if sample:                                     # nothing pasted or linked: ID renders of the sample facade
            import folder_paths
            from ...kubakub import samples
            folder = samples.id_renders(folder_paths.get_temp_directory())
            r_note = samples.note("folder of ID renders", "folder")
        r = im.build(folder, regions_pass=regions_pass.strip(), tag_passes=tag_passes, width=width, height=height,
                     overlap=overlap, split_parts=split_parts, min_region_area=min_region_area,
                     merge_small_regions=merge_small_regions, encoding=legend_colors, scope=sc)
        labels, atlas = r["labels"], r["atlas"]
        W, H = r["size"]
        regions = Regions.from_numpy(labels, atlas, r["scope"])
        ref = r["reference"]
        reference = torch.from_numpy(ref if ref is not None else np.zeros((H, W, 3), np.float32))[None]
        bg = matrix[0].detach().cpu().float().numpy() if matrix is not None else (
            ref if ref is not None else np.full((H, W, 3), 0.4, np.float32))
        preview = torch.from_numpy(fc.render_preview(bg, labels, atlas))[None]

        groups = ", ".join(f"{g} {len(v)}" for g, v in atlas["groups"].items())
        passes = "; ".join(f"{k}: {', '.join(v)}" for k, v in atlas["passes"].items())
        report = "\n".join([
            f"{len(atlas['regions'])} regions in {len(atlas['groups'])} groups, {W}x{H}, "
            f"{time.perf_counter() - t0:.1f} s",
            f"regions per group: {groups}", f"passes: {passes}",
            f"reference: {os.path.basename(r['reference_path']) if r['reference_path'] else '-'}",
            "", *atlas["notes"], *([r_note] if r_note else [])])
        log.info("[KUBA regions] %s", report.replace("\n", "\n    "))
        return io.NodeOutput(regions, preview, reference, json.dumps(atlas, indent=1), report,
                             ui=ui.PreviewImage(preview, cls=cls))


REGION_MASKS_MAX_GB = 4.0      # regions to mask: largest region_masks batch (float32) it will build


class KUBA_RegionsMask(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="KUBA_RegionsMask",
            display_name="kubakub regions to mask",
            category="kubakub/2d/regions",
            search_aliases=['select', 'mask'],
            description=(
                "Make a mask of chosen regions, e.g. all windows, to use with any effect or core node. "
                "The regions are picked with a selector in plan syntax: names with wildcards "
                "('W_F1_*'), 'group:Windows', 'tag:front', comma = or, space = and, '!' = not. Lets any "
                "effect or core node work on regions, or feeds the director as a layer mask."),
            inputs=[
                RegionsType.Input("regions", tooltip="From any kubakub regions node (e.g. regions from masks)."),
                io.String.Input("select", default="*", tooltip="Plan selector, e.g. 'group:Windows, W_F0_*' or 'tag:underside'."),
                io.Int.Input("grow_px", default=0, min=-256, max=256, tooltip="Grow (> 0) or shrink (< 0) the mask."),
                io.Int.Input("feather_px", default=0, min=0, max=256,
                             tooltip="Soft edge: blurs the edge both ways, inwards and outwards, over about "
                                     "this many pixels in total."),
                io.Boolean.Input("invert", default=False,
                                 tooltip="On: the mask covers everything except the selected regions."),
                io.Boolean.Input("per_region_masks", default=False, optional=True,
                                 tooltip="On: region_masks gives one mask per selected region. Takes regions x "
                                         "width x height x 4 bytes of RAM (100 regions at 3840x2160 = 3.3 GB), "
                                         "so it is off until you need it."),
            ],
            outputs=[io.Mask.Output("mask", tooltip="1 on the selected regions (after grow, feather, invert)."), io.String.Output("names", tooltip="The matching region names."),
                     io.Mask.Output("region_masks", tooltip="One mask per selected region (batch), in the order of "
                                                            "names, before grow, feather and invert. Needs "
                                                            "per_region_masks on. Above 4 GB of RAM the node "
                                                            "stops and asks for a narrower select.")],
        )

    @classmethod
    def execute(cls, regions, select, grow_px, feather_px, invert, per_region_masks=False) -> io.NodeOutput:
        import cv2
        from ...kubakub.director.render import select_regions
        all_ids = [r["region_id"] for r in regions.table.get("regions", [])]
        ids = all_ids if select.strip() in ("", "*") else select_regions(regions.table, select)
        if not ids and all_ids:
            raise ValueError(f"select '{select.strip()}' matches no region. Names, groups and tags are in the region table "
                             "(the report of the node that made the regions).")
        lab = regions.labels[0].cpu().numpy()
        m = np.isin(lab, ids).astype(np.float32)
        if grow_px:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * abs(grow_px) + 1,) * 2)
            m = (cv2.dilate if grow_px > 0 else cv2.erode)(m, k)
        if feather_px:
            m = cv2.GaussianBlur(m, (0, 0), feather_px / 2.0)
        if invert:
            m = 1 - m
        chosen = [r for r in regions.table.get("regions", []) if r["region_id"] in set(ids)]
        names = [r["name"] for r in chosen]
        gb = len(chosen) * lab.shape[0] * lab.shape[1] * 4 / 1024 ** 3
        if not per_region_masks:
            # a node reading region_masks stops with this message instead of working on an empty mask
            each = ExecutionBlocker("kubakub regions to mask: switch on 'per_region_masks' to get region_masks.")
        elif gb > REGION_MASKS_MAX_GB:
            raise ValueError(f"region_masks: {len(chosen)} regions at {lab.shape[1]}x{lab.shape[0]} would take "
                             f"{gb:.1f} GB of RAM (the limit is {REGION_MASKS_MAX_GB:g} GB). Narrow 'select' to "
                             "fewer regions, or switch per_region_masks off.")
        elif not chosen:                                # a table without regions
            each = torch.zeros((1, *lab.shape), dtype=torch.float32)
        else:
            each = torch.zeros((len(chosen), *lab.shape), dtype=torch.float32)
            for i, r in enumerate(chosen):              # one at a time: no second full batch as bool
                each[i] = torch.from_numpy(lab == r["region_id"])
        return io.NodeOutput(torch.from_numpy(np.clip(m, 0, 1))[None], chr(10).join(names), each)


NODE_CLASS_MAPPINGS = {
    "KUBA_RegionsFromMasks": KUBA_RegionsFromMasks,
    "KUBA_RegionsSAM3Masks": KUBA_RegionsSAM3Masks,
    "KUBA_RegionsFromIllustrator": KUBA_RegionsFromIllustrator,
    "KUBA_RegionsFromIDMaps": KUBA_RegionsFromIDMaps,
    "KUBA_RegionsMask": KUBA_RegionsMask,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "KUBA_RegionsFromMasks": "kubakub regions from masks",
    "KUBA_RegionsSAM3Masks": "kubakub regions from sam3",
    "KUBA_RegionsFromIllustrator": "kubakub regions from illustrator",
    "KUBA_RegionsFromIDMaps": "kubakub regions from id renders",
    "KUBA_RegionsMask": "kubakub regions to mask",
}
