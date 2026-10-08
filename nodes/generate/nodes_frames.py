"""
nodes_frames.py

Frame in frame: the viewer camera and the
perspective guide. Category KUBAKUB/regions.

KUBA_Viewer:      facade size in metres + where the audience stands -> KUBA_VIEWER.
kubakub frame guide: regions + viewer + depth rules -> guide image (tunnels behind
                  windows, parasites sticking out), masks and corner-pin quads.
Geometry in viewer.py. V3 node schema, registered via ../__init__.py.
"""

from __future__ import annotations

import json
import logging

import cv2
import numpy as np
import torch

from comfy_api.latest import io, ui

from ...kubakub import viewer as vw
from ...kubakub.io_types import PlanType, RegionsType, ViewerType
from ...kubakub.types import RegionPlan, Regions

log = logging.getLogger("KUBA.regions")

CATEGORY = "kubakub/2d/generate"


def _img(a):
    return torch.from_numpy(np.ascontiguousarray(a, np.float32))[None]


class KUBA_Viewer(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="KUBA_Viewer",
            display_name="kubakub audience viewpoint",
            category="kubakub/2d/generate",
            search_aliases=['viewer', 'perspective'],
            description=(
                "Where the audience stands in front of the facade, in metres. Frame in frame uses it for the "
                "perspective of rooms behind windows and of things sticking out of the facade. The matrix maps "
                "onto the facade with one uniform scale (facade_width_m / matrix width)."),
            inputs=[
                io.Image.Input("matrix", tooltip="The matrix (size and preview)."),
                io.Float.Input("facade_width_m", default=40.0, min=0.1, max=10000.0, step=0.1,
                               tooltip="Real width of the whole matrix on the building."),
                io.Float.Input("bottom_m", default=0.0, min=-100.0, max=1000.0, step=0.1,
                               tooltip="Height of the matrix's bottom edge above the ground where people stand."),
                io.Float.Input("viewer_x_m", default=-1.0, min=-10000.0, max=10000.0, step=0.1,
                               tooltip="Viewer position along the facade from its left edge; -1 = centre."),
                io.Float.Input("eye_height_m", default=1.7, min=0.0, max=1000.0, step=0.05,
                               tooltip="Eye height of the viewer above the ground, in metres (1.7 = a standing adult)."),
                io.Float.Input("distance_m", default=30.0, min=0.5, max=10000.0, step=0.5,
                               tooltip="Distance of the viewer from the facade."),
            ],
            outputs=[
                ViewerType.Output("viewer", tooltip="The viewpoint, for frame in frame."),
                io.Image.Output("preview", tooltip="The matrix with a red dot where the viewer's eye line meets the facade."),
                io.String.Output("report", tooltip="Facade size, millimetres per pixel and how big rooms and parasites appear."),
            ],
        )

    @classmethod
    def execute(cls, matrix, facade_width_m, bottom_m, viewer_x_m, eye_height_m, distance_m) -> io.NodeOutput:
        H, W = int(matrix.shape[1]), int(matrix.shape[2])
        v = vw.Viewer(W, H, facade_width_m, bottom_m, None if viewer_x_m < 0 else viewer_x_m, eye_height_m,
                      distance_m)
        fx, fy = v.foot_px()
        prev = (matrix[0, ..., :3].cpu().float().numpy() * 255).astype(np.uint8).copy()
        r = max(6, W // 200)
        cv2.circle(prev, (int(round(fx)), int(round(min(max(fy, r), H - r)))), r, (255, 40, 40), -1)
        report = (f"facade {facade_width_m:g} x {v.facade_height_m:.1f} m, {v.m_per_px * 1000:.1f} mm per px; "
                  f"viewer {distance_m:g} m in front, eye at pixel ({fx:.0f}, {fy:.0f})"
                  + (" (below the matrix)" if fy > H else "") + "; a room 3 m deep appears at "
                  f"{v.scale(3.0) * 100:.0f} %, something 1 m in front at {v.scale(-1.0) * 100:.0f} %")
        return io.NodeOutput(v, _img(prev.astype(np.float32) / 255), report,
                             ui=ui.PreviewImage(_img(prev.astype(np.float32) / 255), cls=cls))


class KUBA_FrameGuide(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="KUBA_FrameGuide",
            display_name="kubakub frame guide (old)",
            category="kubakub/lab",
            is_deprecated=True,
            description=(
                "Perspective guide for frame in frame. Depth rules give regions a depth in metres: > 0 = a room "
                "or world behind the frame (the facade hides everything outside the frame, a window's mullions "
                "stay in front), < 0 = a 'parasite' sticking out of the facade towards the viewer. The guide "
                "shows side walls (shaded) and back / front faces as the viewer sees them; use it as the "
                "structure for generation, and the masks and quads to place media."),
            inputs=[
                RegionsType.Input("regions", tooltip="The facade regions (the same size as the viewer's matrix)."),
                ViewerType.Input("viewer", tooltip="From kubakub audience viewpoint: where the audience stands."),
                io.String.Input("depth_rules", multiline=True, default="",
                                placeholder="W_F1_* = 3\ngroup:Windows = 2.5\nW_F2_C03 = -1.5   // sticks out",
                                tooltip="'<name wildcard> = metres', 'group:<g> = m', 'tag:<t> = m'. Later lines "
                                        "win. Negative = in front of the facade."),
                io.Float.Input("default_depth_m", default=0.0, min=-100.0, max=1000.0, step=0.1,
                               tooltip="Depth for regions no rule names; 0 = not a frame."),
                io.Image.Input("matrix", optional=True, tooltip="Background of the guide (dimmed)."),
                io.Float.Input("dim", default=0.55, min=0.0, max=1.0, step=0.05,
                               tooltip="Dim the matrix outside the frames: 0.55 for looking, 0 when the guide is "
                                       "the input of a generation (the facade stays as it is)."),
                io.Boolean.Input("window_hull", default=True, advanced=True,
                                 tooltip="On: a frame made of several parts (glass panes) is one opening, the "
                                         "gaps (mullions) stay in front. Off: the largest part only."),
            ],
            outputs=[
                io.Image.Output("guide", tooltip="Matrix dimmed, frames drawn as the viewer sees them."),
                io.Mask.Output("area", tooltip="Pixels generation may change (window for rooms; frame, walls and "
                                               "front face for parasites)."),
                io.Mask.Output("back", tooltip="Back faces (rooms) and front faces (parasites)."),
                io.Mask.Output("walls", tooltip="Side walls."),
                io.String.Output("frames_json", tooltip="Per frame: depth, scale, outline, back_quad (corner pin)."),
                io.String.Output("report", tooltip="The frames with their depth and scale."),
            ],
        )

    @classmethod
    def execute(cls, regions, viewer, depth_rules, default_depth_m, dim=0.55, matrix=None,
                window_hull=True) -> io.NodeOutput:
        labels = regions.labels[0].cpu().numpy()
        H, W = labels.shape
        if (W, H) != (viewer.width_px, viewer.height_px):
            raise ValueError(f"regions are {W}x{H}, the viewer was made for {viewer.width_px}x{viewer.height_px}")
        rules, notes = vw.parse_depth_rules(depth_rules)
        depths = vw.region_depths(regions.table, rules, default_depth_m)
        if not depths:
            raise ValueError("no frame: no depth rule matches a region and default_depth_m is 0. Region names: "
                             + ", ".join(r["name"] for r in regions.table["regions"][:30]))
        bg = matrix[0, ..., :3].cpu().float().numpy() if matrix is not None else None
        if bg is not None and bg.shape[:2] != (H, W):
            bg = cv2.resize(bg, (W, H), interpolation=cv2.INTER_AREA)
        r = vw.render_frames(labels, depths, viewer, background=bg, hull=window_hull, dim=dim)
        names = {int(x["region_id"]): x["name"] for x in regions.table["regions"]}
        for f in r["frames"]:
            f["name"] = names.get(f["region_id"])
        rooms = sum(1 for f in r["frames"] if f["depth_m"] > 0)
        report = "\n".join([f"{len(r['frames'])} frames: {rooms} behind the facade, "
                            f"{len(r['frames']) - rooms} sticking out",
                            *[f"{f['name']:<16} {f['depth_m']:+.1f} m  scale {f['scale']:.3f}" for f in r["frames"]],
                            *notes])
        log.info("[KUBA regions] frame guide: %s", report.split("\n")[0])
        mask = lambda a: torch.from_numpy(a.astype(np.float32))[None]  # noqa: E731
        return io.NodeOutput(_img(r["guide"]), mask(r["area"]), mask(r["back"]), mask(r["walls"]),
                             json.dumps({"foot_px": r["foot"], "frames": r["frames"]}, indent=1), report,
                             ui=ui.PreviewImage(_img(r["guide"]), cls=cls))


class KUBA_FrameCompose(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="KUBA_FrameCompose",
            display_name="kubakub frame in frame",
            category="kubakub/2d/generate",
            search_aliases=['window', 'rooms', 'depth'],
            description=(
                "Opens the facade's windows onto rooms, images or a shared world behind the wall, and adds "
                "'parasites' sticking out towards the audience, all in the viewer's perspective. Every region whose plan has fif_depth_m (metres, > 0 behind the "
                "facade, < 0 in front) becomes a frame seen from the viewer: fif_source = generate draws the "
                "perspective guide (feed the image to the region sampler, denoise ~0.95, reference self), "
                "media corner-pins one of your images onto the back wall, world shows the world image behind "
                "the facade (the same world through every window). Side walls of media / world frames are "
                "shaded in the content's colour. Plan example: [W_F1_*] fif_depth_m = 3 / fif_source = media."),
            inputs=[
                PlanType.Input("plan", tooltip="region plan with fif_depth_m / fif_source / fif_media per region."),
                ViewerType.Input("viewer", tooltip="From kubakub audience viewpoint: where the audience stands."),
                io.Image.Input("matrix", tooltip="The matrix (the full projection image at delivery size) the frames are drawn into."),
                io.Image.Input("media", optional=True,
                               tooltip="Images for fif_source = media (a batch; cover-fitted to each back wall)."),
                io.String.Input("media_names", multiline=True, default="", optional=True,
                                placeholder="aquarium\nforest\nlibrary",
                                tooltip="One name per media image, for fif_media = <name>. Or use indices."),
                io.Image.Input("world", optional=True,
                               tooltip="The shared world behind the facade (stretched to the matrix: through "
                                       "the viewer projection it maps 1:1 onto the facade)."),
                io.Float.Input("shadow_strength", default=0.45, min=0.0, max=1.0, step=0.05,
                               tooltip="Parasites (fif_depth_m < 0) cast a shadow on the facade; 0 = none."),
                io.Float.Input("shadow_angle_deg", default=60.0, min=-180.0, max=180.0, step=5.0,
                               tooltip="Direction the shadow falls in the facade plane: 0 right, 90 down, 180 left."),
                io.Float.Input("shadow_length", default=0.7, min=0.0, max=10.0, step=0.05,
                               tooltip="Shadow length per metre the parasite sticks out (1 = light at 45 deg)."),
                io.Float.Input("wall_level", default=0.7, min=0.0, max=2.0, step=0.05, advanced=True,
                               tooltip="Brightness of the side walls of media / world frames."),
                io.Boolean.Input("window_hull", default=True, advanced=True,
                                 tooltip="On: a frame made of several parts (glass panes) is one opening, the "
                                         "gaps (mullions) stay in front. Off: the largest part only."),
            ],
            outputs=[
                io.Image.Output("image", tooltip="The matrix with all frames; generate frames carry the guide."),
                PlanType.Output("plan", tooltip="The plan for the region sampler: every parasite is a new region "
                                                "'<name>_out' (frame + walls + front face) with the frame's settings."),
                io.Mask.Output("area", tooltip="All frame pixels."),
                io.Mask.Output("generate", tooltip="Frames the region sampler should paint (fif_source = generate)."),
                io.Mask.Output("back", tooltip="Back walls (rooms) and front faces (parasites)."),
                io.Mask.Output("walls", tooltip="Side walls."),
                io.Mask.Output("shadow", tooltip="Where parasites darken the facade."),
                io.String.Output("frames_json", tooltip="Per frame: depth, source, scale, outline, back_quad (corner pin)."),
                io.String.Output("report", tooltip="The frames with their depth, source and scale."),
                RegionsType.Output("regions", tooltip="The regions of the plan output, with the '<name>_out' "
                                                      "regions of the parasites: for masks, previews and other "
                                                      "nodes that take regions. Without parasites these are the "
                                                      "plan's own regions."),
            ],
        )

    @classmethod
    def execute(cls, plan, viewer, matrix, media=None, media_names="", world=None, shadow_strength=0.45,
                shadow_angle_deg=60.0, shadow_length=0.7, wall_level=0.7, window_hull=True) -> io.NodeOutput:
        labels = plan.regions.labels[0].cpu().numpy()
        H, W = labels.shape
        if (W, H) != (viewer.width_px, viewer.height_px):
            raise ValueError(f"regions are {W}x{H}, the viewer was made for {viewer.width_px}x{viewer.height_px}")
        specs = {}
        for e in plan.plan["regions"]:
            d = float(e.get("fif_depth_m", 0.0) or 0.0)
            if d != 0:
                specs[int(e["region_id"])] = {"depth": d, "source": e.get("fif_source", "generate"),
                                              "media": e.get("fif_media", ""), "name": e.get("name")}
        if not specs:
            raise ValueError("no frame: give regions fif_depth_m in the region plan, e.g. [W_F1_*] fif_depth_m = 3")
        mat = matrix[0, ..., :3].cpu().float().numpy()
        if mat.shape[:2] != (H, W):
            mat = cv2.resize(mat, (W, H), interpolation=cv2.INTER_AREA)
        media_list = [m[..., :3].cpu().float().numpy() for m in media] if media is not None else []
        names = [n.strip() for n in (media_names or "").splitlines() if n.strip()]
        world_np = world[0, ..., :3].cpu().float().numpy() if world is not None else None
        out, r, notes = vw.compose(mat, labels, specs, viewer, media=media_list, media_names=names,
                                   world=world_np, hull=window_hull, wall_level=wall_level,
                                   shadow_angle_deg=shadow_angle_deg, shadow_length=shadow_length,
                                   shadow_strength=shadow_strength)
        parasites = [f["region_id"] for f in r["frames"] if f["depth_m"] < 0]
        new_plan = plan
        if parasites:
            lab2, table2, plan2 = vw.extend_plan(plan.plan, plan.regions.table, labels, r["area_id"], parasites)
            sc = plan.regions.scope
            new_plan = RegionPlan(plan2, Regions(torch.from_numpy(lab2.astype(np.int32))[None], table2, sc))
            notes.append(f"{len(parasites)} parasite(s) added to the plan as <name>_out regions")
        gen = np.zeros((H, W), bool)
        for f in r["frames"]:
            f["name"] = specs[f["region_id"]]["name"]
            if f["source"] == "generate":
                gen |= (r["back_id"] == f["region_id"]) | (r["wall_id"] == f["region_id"]) | \
                       (labels == f["region_id"])
        gen &= r["area"]
        counts = {}
        for f in r["frames"]:
            counts[f["source"]] = counts.get(f["source"], 0) + 1
        report = "\n".join([f"{len(r['frames'])} frames: " + ", ".join(f"{k} {v}" for k, v in counts.items()),
                            *[f"{f['name']:<16} {f['depth_m']:+.1f} m  {f['source']:<8} scale {f['scale']:.3f}"
                              for f in r["frames"]], *notes])
        log.info("[KUBA regions] frame compose: %s", report.split("\n")[0])
        mask = lambda a: torch.from_numpy(a.astype(np.float32))[None]  # noqa: E731
        return io.NodeOutput(_img(out), new_plan, mask(r["area"]), mask(gen), mask(r["back"]), mask(r["walls"]),
                             mask(r["shadow"]), json.dumps({"foot_px": r["foot"], "frames": r["frames"]}, indent=1),
                             report, new_plan.regions,
                             ui=ui.PreviewImage(_img(out), cls=cls))


NODE_CLASS_MAPPINGS = {
    "KUBA_Viewer": KUBA_Viewer,
    "KUBA_FrameGuide": KUBA_FrameGuide,
    "KUBA_FrameCompose": KUBA_FrameCompose,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "KUBA_Viewer": "kubakub audience viewpoint",
    "KUBA_FrameGuide": "kubakub frame guide (old)",
    "KUBA_FrameCompose": "kubakub frame in frame",
}
