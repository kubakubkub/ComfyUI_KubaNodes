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
import re

import cv2
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
                               tooltip="Background for the preview (the matrix). Another size than the "
                                       "regions is scaled to fit, for the preview only."),
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
        if image is not None:
            bg = image[0].detach().cpu().float().numpy()
            if bg.shape[:2] != (h, w):
                # another size than the regions: scaled for the preview only (the samplers need the exact size)
                ih, iw = bg.shape[:2]
                bg = cv2.resize(bg, (w, h), interpolation=cv2.INTER_AREA if iw * ih > w * h else cv2.INTER_LINEAR)
                note = (f"the image is {iw}x{ih}, the regions are {w}x{h}: it was scaled for the preview only. "
                        f"The region sampler needs an image of exactly {w}x{h}.")
                rep += "\nnote: " + note
                log.info("[KUBA regions] plan: %s", note)
        else:
            bg = np.full((h, w, 3), 0.5, np.float32)
        preview = torch.from_numpy(plan_preview(bg, labels, plan))[None]
        return io.NodeOutput(RegionPlan(plan, regions), json.dumps(plan, indent=1), rep, preview,
                             ui=ui.PreviewImage(preview, cls=cls))


KEEP_REST = "[default]\nstrategy = keep\n"          # what a first kubakub region prompt starts from


def prompt_rule(which, prompt, strategy, denoise, more=""):
    """The fields of kubakub region prompt as one section of rule text -> (header, text)."""
    header = rp.header_of(which) or "default"
    if header == "*":
        header = "default"
    if "\n" in header or "[" in header or "]" in header:
        raise ValueError(f"kubakub region prompt: {rp.WHICH} is one line without brackets (got {header!r}). {rp.WHICH_TIP}")
    lines = [f"[{header}]", f"strategy = {strategy}", f"denoise = {float(denoise):g}"]
    text = " ".join((prompt or "").split())             # the rule text is read line by line
    text = text.replace(" //", "//")                    # ' //' would start a comment there
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'":
        text = f'"{text}"'                              # the rule reader takes one pair of outer quotes off
    if text:
        lines.append(f"prompt = {text}")
    for ln in (more or "").splitlines():
        if ln.strip().startswith("["):
            raise ValueError("kubakub region prompt: 'more' takes lines like  key = value  (no [section]: the regions "
                             f"are chosen in '{rp.WHICH}'). For several groups, chain a second kubakub region prompt.")
        if ln.strip():
            lines.append(ln.strip())
    return header, "\n".join(lines) + "\n"


class KUBA_RegionPrompt(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="KUBA_RegionPrompt",
            display_name="kubakub region prompt",
            category="kubakub/2d/generate",
            search_aliases=['prompt per region', 'plan', 'rules', 'region plan'],
            description=(
                "A prompt for chosen regions, with nothing to type but the prompt. Pick the regions with the "
                "button (windows, a floor, a section ...), write what they become, set how they are painted and "
                "how much they may change. Its plan goes to kubakub region sampler; the regions it does not "
                "name stay as they are. For another prompt on other regions, put a second kubakub region "
                "prompt behind it (plan -> plan). Its 'which regions' output carries the choice to other nodes, "
                "e.g. to the same field of a kubakub shader, so it is set once. The same thing as written "
                "rules, with every setting there is: kubakub region plan."),
            inputs=[
                RegionsType.Input("regions", optional=True,
                                  tooltip="From any kubakub regions node. Not needed when 'plan' is connected: "
                                          "the plan brings its regions."),
                PlanType.Input("plan", optional=True,
                               tooltip="The prompts so far: another kubakub region prompt (or a kubakub region "
                                       "plan). This one is added on top: where both name a region, this node "
                                       "wins."),
                io.String.Input("which", display_name=rp.WHICH, default="", placeholder="empty = every region",
                                tooltip="The regions this prompt is for: use the pick regions button below, or "
                                        "type. Empty = every region. " + rp.WHICH_TIP),
                io.String.Input("prompt", multiline=True, default="", dynamic_prompts=False,
                                placeholder="what these regions become",
                                tooltip="What these regions become. {a|b|c} picks one per region, so a row of "
                                        "windows can tell different stories; {name} and {group} are filled in."),
                io.Combo.Input("strategy", options=["inpaint", "keep", "frame_in_frame"], default="inpaint",
                               tooltip="inpaint: repainted in place from the prompt. keep: left untouched (to "
                                       "take regions out of an earlier, wider prompt). frame_in_frame: a scene "
                                       "of its own, set into the region like a window onto it."),
                io.Float.Input("denoise", default=0.6, min=0.0, max=1.0, step=0.05,
                               tooltip="How much the regions may change: 0.3 = a little, 0.6 = clearly, 1 = "
                                       "painted anew."),
                io.Int.Input("seed", default=0, min=0, max=0xFFFFFFFFFFFFFFFF, control_after_generate=False,
                             tooltip="Another number = another result (region seeds and {a|b} choices). With a "
                                     "'plan' connected, the seed of the last node in the chain counts."),
                io.String.Input("more", multiline=True, default="", optional=True, advanced=True, dynamic_prompts=False,
                                placeholder="lora = name:0.8\nanimate = on",
                                tooltip="Other settings for these regions, one  key = value  per line: negative, "
                                        "lora, reference, color_match, feather_px, animate, video_prompt ... (the "
                                        "list is in docs/generate.md). Empty = the usual values."),
                io.Image.Input("image", optional=True,
                               tooltip="Background for the preview (the matrix). Another size than the regions is "
                                       "scaled to fit, for the preview only."),
            ],
            outputs=[
                PlanType.Output("plan", tooltip="Into kubakub region sampler (or the next kubakub region prompt)."),
                io.String.Output("which_out", display_name=rp.WHICH,
                                 tooltip="The text of 'which regions' as it is: connect it to the same field of a "
                                         "kubakub shader, mask field, regions to mask ... so the choice is made once."),
                io.Image.Output("preview", tooltip="Same colour = same strategy and prompt; grey = kept."),
                io.String.Output("report", tooltip="How many regions this prompt takes, then every region with "
                                                   "its strategy, denoise and prompt."),
            ],
        )

    @classmethod
    def execute(cls, which="", prompt="", strategy="inpaint", denoise=0.6, seed=0, regions=None, plan=None, more="",
                image=None) -> io.NodeOutput:
        if regions is None and plan is None:
            raise ValueError("kubakub region prompt: connect 'regions' (from a kubakub regions node), or 'plan' "
                             "(from another kubakub region prompt).")
        given = which or ""
        base = plan.plan.get("rules_text") or "" if plan is not None else KEEP_REST
        regions = plan.regions if plan is not None else regions
        try:
            mine = rp.pick_or_say(regions.table, which, "kubakub region prompt")
        except rp.PlanError as e:
            raise ValueError(f"kubakub region prompt: {e}") from None
        named = [r for r in regions.table.get("regions", []) if str(r.get("name", "")).lower() == (which or "").strip().lower()]
        if named and re.search(r"[\s,\[\]]", (which or "").strip()):    # one region's own name with a space in it
            which = "name:" + re.sub(r"[\s,\[\]]", "?", which.strip())
        header, section = prompt_rule(which, prompt, strategy, denoise, more)
        # [then]: this node's section wins over whatever came before, and kubakub versions goes on top of it
        text = base.rstrip("\n") + "\n\n[then]\n" + section
        try:
            out = rp.resolve(regions.table, text, seed=int(seed))
        except rp.PlanError as e:
            raise ValueError("kubakub region prompt: " + re.sub(r"^line \d+: ", "", str(e))) from None
        lines = [f"this prompt: {len(mine)} region{'s' if len(mine) != 1 else ''}"
                 + (f" ({rp.WHICH} = {header})" if header != "default" else " (every region)")]
        if strategy != "keep" and not " ".join((prompt or "").split()):
            lines.append("note: the prompt is empty")
        rep = "\n".join(lines) + "\n\n" + rp.report(out)
        log.info("[KUBA regions] region prompt: %s; %d regions, %d to process", lines[0], len(out["regions"]), len(out["order"]))
        labels = regions.labels[0].cpu().numpy()
        h, w = labels.shape
        bg = np.full((h, w, 3), 0.5, np.float32)
        if image is not None:
            bg = image[0].detach().cpu().float().numpy()
            if bg.shape[:2] != (h, w):
                bg = cv2.resize(bg, (w, h), interpolation=cv2.INTER_AREA if bg.shape[0] * bg.shape[1] > w * h else cv2.INTER_LINEAR)
        preview = torch.from_numpy(plan_preview(bg, labels, out))[None]
        return io.NodeOutput(RegionPlan(out, regions), given, preview, rep, ui=ui.PreviewImage(preview, cls=cls))


NODE_CLASS_MAPPINGS = {"KUBA_RegionPlan": KUBA_RegionPlan, "KUBA_RegionPrompt": KUBA_RegionPrompt}
NODE_DISPLAY_NAME_MAPPINGS = {"KUBA_RegionPlan": "kubakub region plan", "KUBA_RegionPrompt": "kubakub region prompt"}
