"""
keyframes.py

Pure logic of kubakub keyframe clips (H3): the director's timeline clips (timeline.clips = [{t_start, t_end,
prompt, on}]) become MiniMax H3 first / last frame generations that replace that stretch of the sequence.
numpy only, no ComfyUI imports (tests/test_keyframes.py).

H3 runs at 24 fps with 17k + 5 frames; the sequence has its own fps: the generated clip is mapped back by time.
"""

from __future__ import annotations

import math

import numpy as np

H3_FPS = 24


def parse_clips(doc, duration):
    """Valid clips of the document, sorted, overlaps trimmed (a later clip starts where the earlier one ends)."""
    tl = (doc or {}).get("timeline") or {}
    out = []
    for i, c in enumerate(tl.get("clips") or []):
        if not isinstance(c, dict) or c.get("on") is False:
            continue
        try:
            a, b = float(c.get("t_start", 0)), float(c.get("t_end", 0))
        except (TypeError, ValueError):
            continue
        a, b = max(0.0, a), min(float(duration), b)
        if b - a < 0.2:
            continue
        out.append({"id": str(c.get("id") or f"c{i + 1}"), "t_start": a, "t_end": b,
                    "prompt": " ".join(str(c.get("prompt") or "").split())})
    out.sort(key=lambda c: c["t_start"])
    for p, c in zip(out, out[1:]):
        if c["t_start"] < p["t_end"]:
            c["t_start"] = p["t_end"]
    return [c for c in out if c["t_end"] - c["t_start"] >= 0.2]


HIDDEN_MARGIN = 12      # director frames kept at each end of a keyframe clip: covers crossfades up to 12


def hidden_frames(doc, n, fps, margin=HIDDEN_MARGIN):
    """Frame indices of an n-frame sequence that H3 keyframe clips replace completely (inside the clip, more than
    `margin` frames from both ends): the director need not render them. Reference clips use their whole stretch
    as the reference video, so they hide nothing. Same frame mapping as KUBA_KeyframeClips."""
    if n < 2 or fps <= 0:
        return []
    raw = {str(c.get("id")): c for c in (((doc or {}).get("timeline") or {}).get("clips") or []) if isinstance(c, dict)}
    out = set()
    for c in parse_clips(doc, (n - 1) / float(fps)):
        if raw.get(c["id"], {}).get("mode") == "reference":
            continue
        i0, i1 = int(round(c["t_start"] * fps)), min(n - 1, int(round(c["t_end"] * fps)))
        if i1 - i0 < 2:
            continue
        out.update(range(i0 + margin + 1, i1 - margin))
        out.difference_update(ia for ia, _ in clip_anchors((doc.get("timeline") or {}).get("markers"), fps, i0, i1, h3_length((i1 - i0) / float(fps))))
    return sorted(out)


def clip_work(doc, n, fps):
    """The clips KUBA_KeyframeClips renders on an n-frame sequence: [(k, clip, mode, i0, i1)], k = the clip's index
    in parse_clips (its seed offset), mode 'keyframes' or 'reference'; clips shorter than 3 frames are left out."""
    if n < 2 or fps <= 0:
        return []
    raw = {str(c.get("id")): c for c in (((doc or {}).get("timeline") or {}).get("clips") or []) if isinstance(c, dict)}
    out = []
    for k, c in enumerate(parse_clips(doc, (n - 1) / float(fps))):
        i0, i1 = int(round(c["t_start"] * fps)), min(n - 1, int(round(c["t_end"] * fps)))
        if i1 - i0 < 2:
            continue
        out.append((k, c, "reference" if raw.get(c["id"], {}).get("mode") == "reference" else "keyframes", i0, i1))
    return out


def models_needed(doc, n, fps, enhance=True, has_model_ref=True):
    """The model inputs KUBA_KeyframeClips needs for this document (its lazy inputs): none without a clip to render,
    'model' for keyframe clips, 'model_ref' for reference clips (they are skipped without it), 'enhance_clip' when
    enhance is on and a clip is not 'raw'."""
    raw = {str(c.get("id")): c for c in (((doc or {}).get("timeline") or {}).get("clips") or []) if isinstance(c, dict)}
    need = set()
    for _k, c, mode, _i0, _i1 in clip_work(doc, n, fps):
        if mode == "reference" and not has_model_ref:
            continue
        need.update(("clip", "vae", "audio_vae", "model" if mode == "keyframes" else "model_ref"))
        if enhance and not raw.get(c["id"], {}).get("raw"):
            need.add("enhance_clip")
    return need


def dependency_runs(spans):
    """Clips in runs that can be prepared together: spans = [(read_lo, read_hi, write_lo, write_hi)] in clip order
    (the sequence frames a clip reads as its anchors / references, and the frames it writes). A clip that reads a
    frame an earlier clip of the current run writes starts a new run, so a run's inputs never depend on its own
    results. -> [[clip indices]]."""
    runs, written = [], []
    for i, (rlo, rhi, wlo, whi) in enumerate(spans):
        if runs and not any(rlo <= b and a <= rhi for a, b in written):
            runs[-1].append(i)
        else:
            if runs:
                written = []
            runs.append([i])
        if wlo <= whi:
            written.append((wlo, whi))
    return runs


def clip_anchors(markers, fps, i0, i1, h3_frames):
    """Markers set to pin their frame ('anchor': true) inside a keyframe clip (sequence frames i0..i1), at least
    2 frames from both ends -> [(sequence frame, H3 frame)], the H3 frame by the same mapping as frame_map. H3 is
    guided through the director's picture at these frames (core MiniMaxH3AddGuide)."""
    if i1 - i0 < 4:
        return []
    used = min(h3_frames, int(round((i1 - i0) / float(fps) * H3_FPS)) + 1)
    out, seen = [], set()
    for m in markers or []:
        if not isinstance(m, dict) or not m.get("anchor"):
            continue
        try:
            ia = int(round(float(m.get("t")) * fps))
        except (TypeError, ValueError):
            continue
        if i0 + 2 <= ia <= i1 - 2:
            hf = int(round((ia - i0) / (i1 - i0) * (used - 1)))
            if hf not in seen:
                seen.add(hf)
                out.append((ia, hf))
    return sorted(out)


def h3_length(duration_s):
    """H3 frame count for a duration: at least the frames the time span needs, snapped up to 17k + 5."""
    need = max(5, int(round(duration_s * H3_FPS)) + 1)
    return 5 + 17 * max(0, math.ceil((need - 5) / 17))


def h3_size(w, h, megapixels=0.85, grid=32):
    """Work size for H3 with the sequence's aspect (multiples of the grid)."""
    s = math.sqrt(megapixels * 1e6 / max(1, w * h))
    return max(grid, int(round(w * s / grid)) * grid), max(grid, int(round(h * s / grid)) * grid)


def frame_map(n_seq, duration_s, h3_frames):
    """For each of the n_seq sequence frames of the clip (first .. last), the H3 frame index at the same time."""
    used = min(h3_frames, int(round(duration_s * H3_FPS)) + 1)
    if n_seq <= 1:
        return [0]
    return [int(round(j / (n_seq - 1) * (used - 1))) for j in range(n_seq)]


# ---- H3's prompt language (from core's H3 templates): look, anchors / tagged references, a timeline, the camera,
# an Audio line, what to avoid. Tags are numbered in the order the references are connected.
H3_DEFAULTS = {"look": "", "camera": "Static locked-off camera on a tripod, no camera movement, no zoom, no cuts.",
               "avoid": "No text, no subtitles, no logos; the building's architecture never changes shape.",
               "audio": "Quiet night ambience of a city square."}


def _t(v):
    return f"{v:.1f}".rstrip("0").rstrip(".") + "s"


def clip_beats(markers, t_start, t_end):
    """Named timeline markers inside a clip, relative to the clip start (default 'marker N' names are skipped)."""
    out = []
    for m in markers or []:
        try:
            t = float(m.get("t"))
        except (TypeError, ValueError, AttributeError):
            continue
        name = " ".join(str(m.get("name") or "").split())
        if t_start < t < t_end and name and not name.lower().startswith("marker"):
            out.append((t - t_start, name))
    return sorted(out)


def build_h3_prompt(clip, h3, duration, beats=(), refs=()):
    """
    clip: {"prompt", "audio", "mode", "raw"}; h3: timeline.h3 {"look", "camera", "avoid", "audio"};
    beats: [(seconds into the clip, text)]; refs: tagged references in connection order, e.g.
    [("<Video 1>", "the layout and motion"), ("<Picture 1>", "the kiosk"), ("<Audio 1>", "the music")].
    """
    what = " ".join(str(clip.get("prompt") or "").split())
    if clip.get("raw"):
        return what
    h3 = {**H3_DEFAULTS, **{k: v for k, v in (h3 or {}).items() if isinstance(v, str) and v.strip()}}
    lines = [(h3["look"].strip() + " " if h3["look"].strip() else "") + "The environment is constant throughout."]
    if clip.get("mode") == "reference":
        if refs:
            parts = [f"{tag} {'exactly as it is' if tag.startswith('<Audio') else 'as ' + what_}" for tag, what_ in refs]
            lines.append("Use " + ", ".join(parts) + ".")
    else:
        lines.append("The scene opens exactly on image 1 and ends exactly on image 2.")
    cuts = [0.0] + [b for b, _ in beats] + [float(duration)]
    texts = [what or "The light changes slowly."] + [t for _, t in beats]
    lines.append("Timeline:")
    for a, b, text in zip(cuts, cuts[1:], texts):
        if b - a > 0.05:
            lines.append(f"[{_t(a)}-{_t(b)}] {text}")
    lines.append(h3["camera"].strip())
    lines.append("Audio: " + (" ".join(str(clip.get("audio") or "").split()) or h3["audio"].strip()))
    lines.append(h3["avoid"].strip())
    return "\n".join(line for line in lines if line)


H3_ENHANCE_SYSTEM = """You write prompts for MiniMax H3, a video model that generates video with sound. You get a draft
prompt and the first frame of the shot (a projection-mapping view of a building facade).
Rewrite the draft into one rich H3 prompt in exactly this structure, English only:
1. One paragraph on the look: lighting, colours, materials, atmosphere, what the image shows. End it with
   "The environment is constant throughout."
2. Keep every line of the draft that anchors frames or references (a line with "image 1" / "image 2" or a tag
   like <Picture 1>, <Video 1>, <Audio 1>) word for word. Never rename these and never add tags the draft does
   not have.
3. "Timeline:" with the draft's time ranges unchanged, each one described concretely: what moves, what lights up,
   how fast, in which order, where on the facade.
4. The camera line of the draft unchanged (the camera never moves).
5. "Audio:" with concrete sounds and music that fit the timeline.
6. The draft's last line (what to avoid) unchanged.
Output only the prompt, no explanations, no markdown."""


_ANCHOR = r"<(?:Picture|Video|Audio) \d+>|image [12]"


def restore_anchors(text, draft):
    """Puts the draft's anchor / reference lines back word for word when the enhancer reworded them (models like
    to turn 'image 1' into '<Picture 1>'): a line with a foreign tag or 'opens exactly' is replaced by the draft's
    line, a missing one goes in before 'Timeline:'. Keeps the rest of the rewrite."""
    import re
    anchors = [ln for ln in draft.splitlines() if re.search(_ANCHOR, ln)]
    if not text or not anchors:
        return text
    tags = set(re.findall(_ANCHOR, draft))
    lines = [ln for ln in text.strip().splitlines()]
    out, used = [], set()
    for ln in lines:
        if ln.strip() in anchors:
            used.add(ln.strip())
            out.append(ln)
        elif (set(re.findall(_ANCHOR, ln)) - tags) or ("opens exactly" in ln and "opens exactly" in "\n".join(anchors)):
            todo = [a for a in anchors if a not in used]
            if todo:                                   # the reworded anchor line -> the draft's line
                used.add(todo[0])
                out.append(todo[0])
        else:
            out.append(ln)
    todo = [a for a in anchors if a not in used and a not in "\n".join(out)]
    if todo:
        at = next((i for i, ln in enumerate(out) if ln.strip().lower().startswith("timeline")), len(out))
        out[at:at] = todo
    return "\n".join(out)


def plausible_prompt(text, draft):
    """An enhanced prompt is used only if it reads like language and keeps the draft's structure: words with
    spaces and lower case, 'Timeline:', and every anchor / reference tag of the draft. Else the draft is kept."""
    import re
    t = (text or "").strip()
    if len(t) < 60:
        return False
    words = t.split()
    if len(words) < 12 or sum(c.islower() for c in t) < 0.5 * sum(c.isalpha() for c in t):
        return False                                   # a wall of capitals / one long token = a model that cannot write
    if "Timeline:" in draft and "timeline" not in t.lower():
        return False
    for tag in re.findall(_ANCHOR, draft):
        if tag not in t:
            return False
    return True


def resample(wave, sr_from, sr_to):
    """Linear resampling of [C, S] audio."""
    wave = np.asarray(wave, np.float32)
    if sr_from == sr_to or wave.shape[-1] == 0:
        return wave
    n = int(round(wave.shape[-1] * sr_to / sr_from))
    x_old = np.linspace(0, 1, wave.shape[-1], endpoint=False)
    x_new = np.linspace(0, 1, n, endpoint=False)
    return np.stack([np.interp(x_new, x_old, ch) for ch in wave.reshape(-1, wave.shape[-1])]).reshape(wave.shape[:-1] + (n,))


def mix_audio(base, sr, segments, gain=0.6):
    """base [C, S] at sr; segments [(start_s, wave [C', S'], sr')] are added at their start (channels matched)."""
    out = np.array(base, np.float32, copy=True)
    C, S = out.shape
    for start, wave, sr_seg in segments:
        w = resample(wave, sr_seg, sr)
        if w.shape[0] != C:
            w = np.repeat(w.mean(axis=0, keepdims=True), C, axis=0)
        a = int(round(start * sr))
        if a >= S:
            continue
        m = min(w.shape[-1], S - a)
        out[:, a:a + m] += gain * w[:, :m]
    return np.clip(out, -1.0, 1.0)
