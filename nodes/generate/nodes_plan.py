"""
nodes_plan.py

KUBA_RegionPlan (category KUBAKUB/regions): rule text + KUBA_REGIONS -> KUBA_PLAN,
one resolved settings dict per region. The logic lives in plan.py; the rule
syntax is documented there and in docs/generate.md.

V3 node schema (comfy_api.latest), registered through the pack's
NODE_CLASS_MAPPINGS loader in ../__init__.py.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os

import numpy as np
import torch

from comfy_api.latest import io, ui

from ...kubakub import facade_core as fc
from ...kubakub import plan as rp
from ...kubakub.io_types import PlanType, RegionsType
from ...kubakub.types import RegionPlan

log = logging.getLogger("KUBA.regions")

CATEGORY = "kubakub/2d/generate"

EXAMPLE_RULES = """[plan]
order = largest_first

[default]
prompt = weathered sandstone facade at night
denoise = 0.55

[group:Windows]
prompt = dark glass, faint reflections

[W_F1_*]
strategy = frame_in_frame
prompt = an underwater room with {jellyfish|a sunken piano|koi}
"""


def _read_rules_file(path: str) -> tuple[str, str]:
    path = fc.clean_folder_path(path)
    if not path:
        return "", ""
    if not os.path.isfile(path):
        raise ValueError(f"rules_file does not exist: {path}")
    with open(path, encoding="utf-8-sig") as f:
        return path, f.read()


def plan_preview(image, labels: np.ndarray, plan: dict) -> np.ndarray:
    """Regions coloured by what will happen to them: same strategy and prompt = same colour."""
    keys, fake = {}, []
    for e in plan["regions"]:
        k = (e["strategy"], e["prompt"])
        if k not in keys:
            short = {"inpaint": "p", "frame_in_frame": "fif", "keep": "keep"}[e["strategy"]]
            keys[k] = short if short == "keep" else f"{short}{sum(1 for s, _ in keys if s == e['strategy']) + 1}"
        fake.append({**{x: e[x] for x in ("region_id", "bbox")}, "group_id": keys[k],
                     "centroid": [e["bbox"][0] + e["bbox"][2] / 2, e["bbox"][1] + e["bbox"][3] / 2]})
    groups = {}
    for r in fake:
        groups.setdefault(r["group_id"], []).append(r["region_id"])
    return fc.render_preview(image, labels, {"regions": fake, "groups": groups})


class KUBA_RegionPlan(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="KUBA_RegionPlan",
            display_name="kubakub region plan",
            category="kubakub/2d/generate",
            search_aliases=['prompt per region', 'rules'],
            description=(
                "Give every region its own prompt and settings, written as simple text rules. "
                "Sections: [default], [W_F1_*] (name "
                "wildcard), [group:Windows], [tag:M_FLOOR_F1], [region:3-7]; space = AND, comma = "
                "OR, ! = NOT. More specific rules win (region > name > group > tag > default). "
                "prompt += text appends, {a|b|c} picks one per region."),
            inputs=[
                RegionsType.Input("regions", tooltip="From kubakub regions from matrix / masks (regions output)."),
                io.String.Input("rules", multiline=True, default=EXAMPLE_RULES, dynamic_prompts=False,
                                tooltip="The plan. See docs/generate.md (region plan) for all "
                                        "settings. Applied after rules_file, so it can override it."),
                io.Int.Input("seed", default=0, min=0, max=0xFFFFFFFFFFFFFFFF, control_after_generate=False,
                             tooltip="Plan seed: region seeds (seed + region id) and {a|b} choices."),
                io.String.Input("rules_file", default="", optional=True,
                                tooltip="Optional .txt with rules, e.g. kept in the project folder. "
                                        "Read only."),
                io.Image.Input("image", optional=True,
                               tooltip="Background for the preview (the matrix)."),
            ],
            outputs=[
                PlanType.Output("plan", tooltip="Connect to kubakub region sampler and seam pass."),
                io.String.Output("plan_json", tooltip="The resolved settings of every region as text."),
                io.String.Output("report", tooltip="A table of every region with its strategy, denoise "
                                                   "and prompt, plus notes."),
                io.Image.Output("preview", tooltip="Same colour = same strategy and prompt."),
            ],
        )

    @classmethod
    def fingerprint_inputs(cls, rules_file="", **kwargs):
        path = fc.clean_folder_path(rules_file or "")
        if not path or not os.path.isfile(path):
            return ""
        st = os.stat(path)
        return hashlib.sha256(f"{path}|{st.st_size}|{st.st_mtime_ns}".encode()).hexdigest()

    @classmethod
    def execute(cls, regions, rules, seed, rules_file="", image=None) -> io.NodeOutput:
        path, file_text = _read_rules_file(rules_file or "")
        text = rules or ""
        if file_text:
            text = file_text.rstrip("\n") + "\n" + text
        try:
            plan = rp.resolve(regions.table, text, seed=int(seed))
        except rp.PlanError as e:
            where = (f" (line numbers count from the start of {os.path.basename(path)}, "
                     f"then continue into the rules widget)") if file_text else ""
            raise ValueError(f"kubakub region plan: {e}{where}") from None
        if path:
            plan["rules_file"] = path
        rep = rp.report(plan)
        log.info("[KUBA regions] plan: %d regions, %d to process, %d notes",
                 len(plan["regions"]), len(plan["order"]), len(plan["notes"]))
        for n in plan["notes"]:
            log.info("[KUBA regions]   %s", n)

        labels = regions.labels[0].cpu().numpy()
        h, w = labels.shape
        if image is not None and tuple(image.shape[1:3]) == (h, w):
            bg = image[0].detach().cpu().float().numpy()
        else:
            bg = np.full((h, w, 3), 0.5, np.float32)
        preview = torch.from_numpy(plan_preview(bg, labels, plan))[None]
        return io.NodeOutput(RegionPlan(plan, regions), json.dumps(plan, indent=1), rep, preview,
                             ui=ui.PreviewImage(preview, cls=cls))


NODE_CLASS_MAPPINGS = {"KUBA_RegionPlan": KUBA_RegionPlan}
NODE_DISPLAY_NAME_MAPPINGS = {"KUBA_RegionPlan": "kubakub region plan"}
