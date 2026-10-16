"""
upsampler.py

A prompt upsampler: a language model loaded as a CLIP (CLIPLoader, e.g. qwen3.5_9b_flux3_prompt_upsampler, type
flux3) that turns a short prompt into one full instruction while it looks at the pictures it is about: the region's
crop, a style picture, the whole facade. With Flux 3 Klein an edit follows such an instruction far better than the
short prompt (2026-10-16 tests: the short "make the windows stained glass" left the windows alone, the upsampled one
changed every window).

The system prompts are those of the FLUX.3 Klein templates (Black Forest Labs, Comfy-Org workflow templates
image_flux3_klein_t2i / image_flux3_klein_image_edit, 2026-10), sampled as there: temperature 1, top_k 1 (the one
most likely text), so the same prompt and pictures always give the same instruction. Instructions are kept for the
session per (upsampler, prompt, pictures): a queue only asks for what changed.
"""

from __future__ import annotations

import logging
from collections import OrderedDict

import torch

from . import ops
from . import region_cache as rc

log = logging.getLogger("KUBA.regions")

SYSTEM_EDIT = (
    "You are the prompt upsampler of an image-editing model. You see the reference image(s), attached in order (the "
    "first image, the second image, ...), and the user's short request. Expand it into ONE detailed, self-contained "
    "instruction: the caption the editing model receives together with the same images. Output only the instruction "
    "as one continuous English prose block, with no greetings, explanations, Markdown, JSON, coordinates, ids or "
    "angle-bracket tokens. Read a vague or conversational request as the most plausible edit of the supplied images."
    "\n\nPick the voice by task shape. When one reference stays and gets bounded changes, write an imperative editing "
    "instruction on it (\"Remove the ...\", \"Change the ... to ...\", \"Zoom in on ...\"); a whole-image "
    "transformation opens with its scope (\"Convert the entire image into ...\", \"Apply a ... filter over the "
    "entire scene\", \"Horizontally flip the entire image so that ...\", \"Shift the camera viewpoint to ...\", then "
    "where the major elements should now be). When the references are drawn into a newly composed frame (combining "
    "subjects, a new scene, a garment, object or style donor), write a generation request (\"Create a photograph of "
    "...\", \"Create an anime-style illustration of ...\", \"Generate a new image featuring ...\") that says what to "
    "take from each image, by ordinal, and what is newly generated.\n\nKeep every explicit edit, constraint, count, "
    "position, style cue and piece of wording, and decide what is left open as one concrete plan. Ground everything "
    "in what is actually visible: identify each referent unambiguously by its real appearance, never mention content "
    "that is not there, and use every image the user attached. State every end state concretely (which color, how "
    "large, which pose, which material) and give new content enough appearance to be generated. Cover the knock-on "
    "effects of the change (shadows, reflections, occlusion, perspective, scale, lighting) and say what fills a "
    "removed region. A zoom-in names what stays in frame and which neighbors are cropped out; an outpainting "
    "continues the scene in the same medium. Describe the change, not the whole scene, and close an editing "
    "instruction with one compact clause naming what stays unchanged; transferred people keep their identity, face "
    "and accessories unless asked otherwise. Transcribe on-image text verbatim in its original language inside "
    "double quotes, with line breaks as \\n, and match the original font, size and color when replacing it. Name "
    "recognizable public figures, brands and landmarks you can see. Directions are frame-relative; use no evaluative "
    "or quality words. Typical length is 50 to 150 words.\n\nOutput only the instruction.")

MAX_SIDE = 1024                     # pictures are shown to the upsampler at most this large (it sees about 1 MP anyway)
MAX_TOKENS = 1024                   # an instruction is 50-150 words; this only stops a run-away


class Instructions:
    """The session's instructions, newest last; a few hundred are a few hundred KB."""

    def __init__(self, size: int = 512):
        self.size, self.items = size, OrderedDict()

    def get(self, key):
        if key in self.items:
            self.items.move_to_end(key)
            return self.items[key]
        return None

    def put(self, key, text):
        self.items[key] = text
        self.items.move_to_end(key)
        while len(self.items) > self.size:
            self.items.popitem(last=False)


CACHE = Instructions()


def _small(img: torch.Tensor) -> torch.Tensor:
    img = img[:1, ..., :3].float().cpu()
    h, w = int(img.shape[1]), int(img.shape[2])
    s = min(1.0, MAX_SIDE / max(h, w))
    if s < 1.0:
        img = ops.resize(img, max(16, round(w * s)), max(16, round(h * s)))
    return img


def _clean(text: str) -> str:
    """The instruction alone: no reasoning block, no quotes around it."""
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    text = text.replace("<think>", "").strip()
    if len(text) > 1 and text[0] == text[-1] and text[0] in "\"'":
        text = text[1:-1].strip()
    return text


def _after_generate():
    """What core does after every node (execution.py) and between its own generations (text_encoders/ace15.py):
    with dynamic VRAM (comfy-aimdo) the decode keeps CUDA graphs and buffers sized for that one text; a second
    generation in the same node run would replay them on another length (2026-10-16: 'index_copy_(): index out of
    bounds', then the server stopped)."""
    try:
        import comfy.memory_management
        import comfy.model_management
        import comfy.model_prefetch
    except Exception:  # noqa: BLE001  (tests without ComfyUI)
        return
    if getattr(comfy.memory_management, "aimdo_enabled", False):
        comfy.model_prefetch.cleanup_prefetch_queues()
        comfy.model_management.reset_cast_buffers()
        try:
            import comfy_aimdo.model_vbar
            comfy_aimdo.model_vbar.vbars_reset_watermark_limits()
        except Exception:  # noqa: BLE001
            pass


def expand(upsampler, prompt: str, pictures=(), system: str = SYSTEM_EDIT) -> tuple[str, bool]:
    """(the instruction for `prompt` about `pictures` (IMAGE tensors, in order: the first image, the second ...),
    whether it came from the cache). An empty prompt stays empty."""
    prompt = (prompt or "").strip()
    if not prompt:
        return "", True
    pics = [_small(p) for p in pictures if p is not None]
    key = (rc.identity(upsampler), prompt, system, tuple(rc.digest(p) for p in pics))
    hit = CACHE.get(key)
    if hit is not None:
        return hit, True
    tokens = upsampler.tokenize(prompt, images=pics, skip_template=False, min_length=1, thinking=False,
                                system_prompt=system)
    try:
        ids = upsampler.generate(tokens, do_sample=True, max_length=MAX_TOKENS, temperature=1.0, top_k=1, top_p=0.95,
                                 min_p=0.0, repetition_penalty=1.0, presence_penalty=0.0, seed=0, mtp=False)
    finally:
        _after_generate()
    text = _clean(upsampler.decode(ids))
    if not text:                    # nothing usable came back: the short prompt is better than none
        log.warning("[KUBA regions] upsampler gave no text for '%s'; the prompt is used as written", prompt[:60])
        text = prompt
    CACHE.put(key, text)
    return text, False
