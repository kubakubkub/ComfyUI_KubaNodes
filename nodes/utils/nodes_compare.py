"""kubakub compare: two pictures or two videos side by side on the node (web/kubakub_compare.js).

Each side is written as JPEG frames into ComfyUI's temp folder; the web file shows them frame by frame, so the two
sides always show the same frame (play, scrub, step), in swipe, side by side, difference, blink or onion skin.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import weakref
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

import folder_paths
from comfy_api.latest import io

from ...kubakub import imio

log = logging.getLogger("KUBA.regions")

CATEGORY = "kubakub/utils"
MODES = ["swipe", "side by side", "difference", "blink", "onion"]
PREVIEWS = ["1280", "1920", "full"]
MAX_FRAMES = 3000


_decoded = weakref.WeakKeyDictionary()            # VIDEO -> (frames, fps): the last one decoded, gone with its VIDEO
KEEP_DECODED_GB = 4.0                             # a bigger clip is decoded again rather than kept in RAM


def frames_of(x):
    """IMAGE (n, h, w, c) or a VIDEO -> (frames tensor, fps or None, 'image' | 'video'). A VIDEO is decoded once:
    Load Video hands every node after it (and every run while it is cached) the same object, so the tracker, the
    pointclouder and a re-run read the frames decoded the first time. The frames are read only, never changed."""
    if x is None:
        return None, None, None
    if hasattr(x, "get_components"):
        try:
            return (*_decoded[x], "video")
        except (KeyError, TypeError):
            pass
        c = x.get_components()
        fps = float(c.frame_rate) if getattr(c, "frame_rate", None) else None
        _decoded.clear()
        if c.images.numel() * c.images.element_size() <= KEEP_DECODED_GB * 2 ** 30:
            try:
                _decoded[x] = (c.images, fps)
            except TypeError:                       # not weak-referenceable: no cache
                pass
        return c.images, fps, "video"
    if hasattr(x, "shape") and len(x.shape) == 4:
        return x, None, "image"
    raise ValueError("kubakub compare: connect a picture (IMAGE) or a video (VIDEO).")


def write_side(frames, tag: str, side: int | None) -> dict:
    """The frames as JPEGs in temp/kubakub_compare/<key>/ -> what the web file reads."""
    n = min(int(frames.shape[0]), MAX_FRAMES)
    h, w = int(frames.shape[1]), int(frames.shape[2])
    k = 1.0 if not side else min(1.0, side / max(h, w))
    pw, ph = max(1, round(w * k)), max(1, round(h * k))
    digest = hashlib.sha1(f"{n}|{w}|{h}|{pw}".encode())
    sy, sx = max(1, h // 256), max(1, w // 256)
    for i in range(n):                              # two clips that differ anywhere never share a folder
        digest.update(frames[i, ::sy, ::sx].cpu().numpy().tobytes())
    key = digest.hexdigest()[:16]
    sub = f"kubakub_compare/{key}"
    out = os.path.join(folder_paths.get_temp_directory(), "kubakub_compare", key)
    os.makedirs(out, exist_ok=True)

    def one(i):
        path = os.path.join(out, f"{i:05d}.jpg")
        if os.path.isfile(path):
            return
        f = frames[i].cpu().float().numpy()[..., :3]
        if k < 1:
            f = cv2.resize(f, (pw, ph), interpolation=cv2.INTER_AREA)
        u8 = (np.clip(f, 0, 1) * 255 + 0.5).astype(np.uint8)
        if u8.shape[-1] == 1:
            u8 = np.repeat(u8, 3, -1)
        imio.imwrite(path, cv2.cvtColor(u8, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])

    with ThreadPoolExecutor(max_workers=max(1, min(8, (os.cpu_count() or 4) - 2))) as pool:
        list(pool.map(one, range(n)))
    return {"subfolder": sub, "type": "temp", "n": n, "w": w, "h": h, "cut": int(frames.shape[0]) > n, "tag": tag}


def temp_file(folder, name, write) -> dict:
    """temp/<folder>/<name>, written by write(path) unless it is there -> the reference the web files /view."""
    out = os.path.join(folder_paths.get_temp_directory(), folder)
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, name)
    if not os.path.isfile(path):
        write(path)
    return {"filename": name, "subfolder": folder, "type": "temp"}


def save_json(data, folder) -> dict:
    """data as compact JSON in temp, named by its content."""
    text = json.dumps(data, separators=(",", ":"))

    def write(path):
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
    return temp_file(folder, hashlib.sha1(text.encode()).hexdigest()[:20] + ".json", write)


def save_wav(audio, folder) -> dict:
    """A core AUDIO as a WAV in temp (for a node to play along), named by its content."""
    from ...kubakub.director.export import write_wav
    w = audio["waveform"][0].float().cpu().numpy()
    name = hashlib.sha1(w[:, ::97].tobytes() + f"{w.shape}|{int(audio['sample_rate'])}".encode()).hexdigest()[:20] + ".wav"
    return temp_file(folder, name, lambda path: write_wav(path, w[:2], int(audio["sample_rate"])))


def free_vram():
    """Let ComfyUI hand back the graphics card memory a node used."""
    try:
        import comfy.model_management as mm
        mm.soft_empty_cache()
    except Exception:  # noqa: BLE001
        pass


class KUBA_Compare(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="KUBA_Compare",
            display_name="kubakub compare",
            category=CATEGORY,
            search_aliases=["compare", "before after", "a b", "ab", "swipe", "difference", "wipe", "video compare"],
            description=("Two pictures or two videos on the node, always on the same frame: swipe (drag the line), "
                         "side by side, difference (what changed lights up), blink and onion skin. Play, scrub and step "
                         "frame by frame; the big window (the corner button) adds zoom: wheel = zoom, drag = pan, "
                         "double click = fit. Sides of another shape are each fitted, never stretched. Any mix works: a still "
                         "against a video, a batch of frames against a video file."),
            is_output_node=True,
            inputs=[
                io.MultiType.Input("a", types=[io.Image, io.Video], optional=True,
                                   tooltip="A (left / under): a picture, a batch of frames or a video."),
                io.MultiType.Input("b", types=[io.Image, io.Video], optional=True,
                                   tooltip="B (right / over): a picture, a batch of frames or a video."),
                io.Combo.Input("mode", options=MODES, default="swipe", tooltip="How the two are shown (the pills on the node)."),
                io.Combo.Input("preview", options=PREVIEWS, default="1280",
                               tooltip="The long side the frames are shown at (full = every pixel; more to load)."),
                io.Float.Input("fps", default=25.0, min=1.0, max=120.0, step=1.0, advanced=True,
                               tooltip="Speed of a batch of frames (a video plays at its own speed)."),
            ],
            outputs=[],
        )

    @classmethod
    def execute(cls, a=None, b=None, mode="swipe", preview="1280", fps=25.0) -> io.NodeOutput:
        if a is None and b is None:
            raise ValueError("kubakub compare: connect a picture or video to 'a' and / or 'b'.")
        side = None if preview == "full" else int(preview)
        shown, rates = {"mode": mode}, []
        for tag, x in (("a", a), ("b", b)):
            frames, rate, kind = frames_of(x)
            if frames is None:
                continue
            s = write_side(frames, tag, side)
            s["kind"], s["fps"] = kind, rate or float(fps)
            shown[tag] = s
            if rate:
                rates.append(rate)
        shown["fps"] = rates[0] if rates else float(fps)
        log.info("[KUBA compare] %s", ", ".join(f"{t}: {shown[t]['n']} frame(s) {shown[t]['w']}x{shown[t]['h']}"
                                                for t in ("a", "b") if t in shown))
        return io.NodeOutput(ui={"kubakub_compare": [shown]})


NODE_CLASS_MAPPINGS = {"KUBA_Compare": KUBA_Compare}
NODE_DISPLAY_NAME_MAPPINGS = {"KUBA_Compare": "kubakub compare"}
