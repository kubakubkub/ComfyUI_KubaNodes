"""
video_models.py

Which video model a node got, so the video nodes take any LTX or MiniMax H3 variant the user plugs in.
- family: from the loaded objects themselves (the model's class: LTXAV = LTX 2.x audio-video, LTXV = LTX video
  only, MiniMaxH3), and the VAE / text encoder checked against it, so a mixed-up set is said plainly.
- variant and speed: from the file names upstream of the node's model input (the UNETLoader / checkpoint and every
  LoRA loader on the way), because a dev and a distilled LTX, or H3 with and without a 4-step LoRA, have the same
  weights' layout but need other schedules. The `schedule` / `steps` widgets override what the names say.
- the schedule per case: the sigmas or scheduler, steps, sampler and cfg from core's templates and the tested
  settings of each speed LoRA.
No ComfyUI imports at module level (tests/test_video_models.py); the objects are only looked at.
"""

from __future__ import annotations

import math
import re

FAMILIES = ("ltx", "ltxv", "h3")
NAMES = {"ltx": "LTX 2.x", "ltxv": "LTX (video only)", "h3": "MiniMax H3"}
SCHEDULES = ("auto", "fast (distilled / speed LoRA)", "full steps (dev / no speed LoRA)")
MODEL_FILES = (".safetensors", ".gguf", ".sft", ".ckpt", ".pt", ".pth", ".bin")

# ---- H3 geometry (core comfy_extras/nodes_minimax_h3.py) ------------------------------------------------------------
H3_FPS = 24
H3_AUDIO_FPS = 40


def h3_frames(n: int) -> int:
    """The H3 frame count >= n (17k + 5)."""
    n = max(5, int(n))
    while n % 17 != 5:
        n += 1
    return n


def h3_latent_frames(frames: int) -> int:
    f = h3_frames(frames)
    return 2 if f <= 5 else ((f - 5) // 17) * 5 + 2


def h3_audio_frames(frames: int) -> int:
    return round(h3_frames(frames) / H3_FPS * H3_AUDIO_FPS)


def model_frames(family: str, seconds: float, fps: float) -> tuple:
    """(frames, fps) the model renders `seconds` in: LTX at the node's fps (8k + 1 frames), H3 at its own 24 fps."""
    if family == "h3":
        return h3_frames(int(math.ceil(seconds * H3_FPS - 1e-6)) + 1), float(H3_FPS)
    from .video import ltx_frames
    return ltx_frames(int(round(seconds * fps)) + 1), float(fps)


def time_index(n_out: int, fps_out: float, n_src: int, fps_src: float) -> list:
    """For each of n_out frames at fps_out, the source frame (of n_src at fps_src) at the same time."""
    return [min(n_src - 1, int(round(k / fps_out * fps_src))) for k in range(n_out)]


# ---- what was plugged in ------------------------------------------------------------------------------------------

def _cls_names(obj) -> str:
    """The modules and class names of an object's wrapped model (VAE.first_stage_model, CLIP's cond_stage_model),
    lower case, for family checks without importing ComfyUI: comfy.ldm.lightricks / comfy.text_encoders.lt = LTX,
    comfy.ldm.minimax / comfy.text_encoders.minimax = H3."""
    names = []
    for o in (getattr(obj, "first_stage_model", None), getattr(obj, "cond_stage_model", None)):
        if o is not None:
            names += [f"{c.__module__}.{c.__name__}" for c in type(o).__mro__[:3]]
    return " ".join(names).lower()


def family(model):
    """'ltx' / 'ltxv' / 'h3' for a loaded model, None when it is not one of them (a stub in a test may carry a
    `video_family` attribute)."""
    f = getattr(model, "video_family", None)
    if f:
        return f
    names = [c.__name__ for c in type(getattr(model, "model", None)).__mro__]
    if "MiniMaxH3" in names:
        return "h3"
    if "LTXAV" in names:
        return "ltx"
    if "LTXV" in names:
        return "ltxv"
    return None


def part_family(obj):
    """The family a VAE or text encoder belongs to ('h3' / 'ltx'), None when it cannot tell."""
    f = getattr(obj, "video_family", None)
    if f:
        return f
    n = _cls_names(obj)
    if "minimax_music" in n:
        return None
    if ".minimax." in n:
        return "h3"
    if "lightricks" in n or "text_encoders.lt." in n:
        return "ltx"
    return None


def check_set(model_family, **parts):
    """Raise a plain error when a connected VAE / text encoder is from the other model family."""
    if model_family is None:
        return
    want = "h3" if model_family == "h3" else "ltx"
    for name, obj in parts.items():
        if obj is None:
            continue
        got = part_family(obj)
        if got is not None and got != want:
            raise ValueError(f"the model is {NAMES[model_family]} but '{name}' is a {NAMES[got]} one: connect the "
                             f"{NAMES[model_family]} {name.replace('_', ' ')} (each model family has its own VAE, "
                             f"audio VAE and text encoder)")


def upstream_files(prompt, node_id, input_name="model"):
    """The model files feeding one input of a node, walked up the graph through every node with a model input:
    [(file name lower case, strength or None)], loader first. Works for any loader whose widget holds a file name
    (UNETLoader, checkpoint, GGUF, LoraLoaderModelOnly, a LoRA stack with {lora, on, strength} entries)."""
    out, seen = [], set()
    if not isinstance(prompt, dict) or node_id is None:
        return out
    link = (prompt.get(str(node_id)) or {}).get("inputs", {}).get(input_name)
    stack = [link]
    while stack:
        link = stack.pop()
        if not (isinstance(link, (list, tuple)) and len(link) == 2):
            continue
        nid = str(link[0])
        if nid in seen:
            continue
        seen.add(nid)
        inputs = (prompt.get(nid) or {}).get("inputs", {})
        strength = next((inputs[k] for k in ("strength_model", "strength", "lora_strength") if
                         isinstance(inputs.get(k), (int, float))), None)
        found = []
        for k, v in inputs.items():
            if isinstance(v, str) and v.lower().endswith(MODEL_FILES):
                found.append((v.replace("\\", "/").split("/")[-1].lower(), strength))
            elif isinstance(v, dict) and isinstance(v.get("lora"), str):      # LoRA stacks: {on, lora, strength}
                if v.get("on", True) and v.get("lora", "").lower() not in ("", "none"):
                    found.append((v["lora"].replace("\\", "/").split("/")[-1].lower(), v.get("strength")))
        out = found + out                                     # upstream first: the loader leads
        for k, v in inputs.items():
            if isinstance(v, (list, tuple)) and len(v) == 2 and k in ("model", "model_in", "unet"):
                stack.append(v)
    return [(f, s) for f, s in out if s is None or abs(float(s)) > 1e-6]


# ---- the schedule ------------------------------------------------------------------------------------------------

LTX_DISTILLED = (1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0)    # LTX 2.x blueprints
LTX_DISTILLED_2 = (0.85, 0.725, 0.4219, 0.0)                                               # after the x2 upscaler
# H3 speed LoRAs with their own sigmas (tested / published); any other "<n>step" name gets n steps of the simple
# schedule with res_multistep, the way core's H3 templates run their turbo LoRAs
H3_KNOWN = (
    ("flashgen", "FlashGen 4-step LoRA", "euler", (1.0, 0.965517, 0.888889, 0.679245, 0.0)),
    ("taomate", "TaoMate 3-step LoRA", "euler", (1.0, 0.961165, 0.853333, 0.0)),
)
_STEPS = re.compile(r"(\d+)[ _-]?(?:step|nfe)")


def _speed_lora(files):
    """(file, steps) of the speed LoRA / distilled model among the files, the last one wins (closest to the node)."""
    hit = None
    for f, _s in files:
        m = _STEPS.search(f)
        if m and 1 <= int(m.group(1)) <= 16:
            hit = (f, int(m.group(1)))
    return hit


def profile(fam, files=(), schedule="auto", steps=0):
    """How to sample this model: {family, task, label, sampler, sigmas (manual list) or scheduler + steps, cfg,
    sigmas_2 (second stage after the LTX upscaler), why}. `files` from upstream_files; schedule / steps override."""
    names = [f for f, _ in files]
    loader = names[0] if names else ""
    steps = int(steps or 0)
    fast = schedule == SCHEDULES[1]
    full = schedule == SCHEDULES[2]
    p = {"family": fam, "task": None, "sigmas": None, "scheduler": None, "steps": None, "cfg": 1.0,
         "sigmas_2": None, "why": ""}
    if fam == "h3":
        p["task"] = "ref2va" if re.search(r"ref2v|r2v|ref_?2", loader) else "fl2va"
        known = next(((k, label, smp, sig) for k, label, smp, sig in H3_KNOWN for f in names if k in f), None)
        speed = _speed_lora(files)
        if full or (not fast and known is None and speed is None):
            p.update(label="H3, full steps", sampler="res_multistep", scheduler="simple", steps=steps or 20)
            p["why"] = "no speed LoRA in the model's file names" if schedule == "auto" else "schedule = full steps"
        elif known is not None and (not steps or steps == len(known[3]) - 1):
            p.update(label=f"H3 + {known[1]}", sampler=known[2], sigmas=list(known[3]), steps=len(known[3]) - 1)
            p["why"] = "its own sigmas"
        else:
            n = steps or (speed[1] if speed else 4)
            p.update(label=f"H3, {n} fast steps", sampler="res_multistep", scheduler="simple", steps=n)
            p["why"] = (f"from '{speed[0]}'" if speed and not steps else "steps set on the node")
        p["label"] += f" ({p['task']})"
        return p
    distilled = any("distill" in f for f in names)
    dev = "dev" in loader and not distilled
    if full or (not fast and dev):
        n = steps or 20
        p.update(label="LTX, full steps (dev)", sampler="euler", scheduler="ltxv", steps=n, cfg=4.0,
                 sigmas_2=list(LTX_DISTILLED_2))
        p["why"] = ("a dev model without the distilled LoRA" if schedule == "auto" else "schedule = full steps")
    else:
        p.update(label="LTX, distilled", sampler="euler_ancestral", sigmas=list(LTX_DISTILLED), steps=8,
                 sigmas_2=list(LTX_DISTILLED_2))
        p["why"] = ("distilled in the file names" if distilled else
                    "schedule = fast" if fast else "no file name says dev: taken as distilled (set schedule = full "
                                                   "steps for a dev model without the distilled LoRA)")
        if steps and steps != 8:
            p.update(sigmas=None, scheduler="ltxv", steps=steps)
    if fam == "ltxv":
        p["label"] = p["label"].replace("LTX", "LTX video only")
    return p


def describe(p) -> str:
    """One report line for a profile."""
    how = (f"sigmas {', '.join(f'{s:g}' for s in p['sigmas'])}" if p.get("sigmas") else
           f"{p['steps']} steps {p['scheduler']}")
    return f"model: {p['label']}, {p['sampler']}, {how}, cfg {p['cfg']:g} ({p['why']})"
