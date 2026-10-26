"""
tracking.py

kubakub tracker: what moves in a video, found frame by frame and kept as tracks with ids.
numpy / opencv / scipy; the models (YOLO, MediaPipe, SAM 2) are loaded only when their finder is on, from files
already on disk, with no network (tests/test_tracking.py).

Finders (several at once; each found thing keeps its `kind`):
    motion   what moves against the still background (OpenCV MOG2)
    colour   things of a picked colour (distance in Lab)
    lights   bright spots
    yolo     people and things (Ultralytics weights in models/ultralytics; detection)
    body     MediaPipe pose, inside every YOLO person box when YOLO finds people, else one person in the frame
    hands    MediaPipe hands (fingertips)
    face     MediaPipe face mesh
    motif    SAM 2.1 video: points clicked on a frame, followed through the clip (exact outline)
    outline  SAM 2.1 image: the exact outline inside every box of the chosen kinds

The analysis is offline, so it looks both ways: gaps are filled between the frames a thing was seen and the paths
are smoothed with a centred window (no lag).

Tracks (`Tracks.to_dict()`, also what web/kubakub_track.js reads):
    {"format": "kubakub.tracks", "W", "H", "n" frames, "fps",
     "tracks": [{"id", "kind", "label", "t0", "box": [[x, y, w, h] per frame from t0], "seen": [0/1 ...],
                 "colour": "#rrggbb", "outline": {t: [[x, y] ...]}, "points": {t: {name: [x, y]}}}],
     "curves": {"motion": [...], "count": [...], "luma": [...]},
     "skeleton": {"body": [[a, b] ...], "hands": [...]}}
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field

import cv2
import numpy as np

log = logging.getLogger("KUBA.regions")

TRACKS_FORMAT = "kubakub.tracks"
FINDERS = ["motion", "colour", "lights", "yolo", "words", "body", "hands", "face", "motif"]
ANALYSIS = {"640": 640, "960": 960, "1280": 1280, "full": None}

DEFAULTS = {
    "motion": {"on": True, "sensitivity": 0.5, "min_size": 0.002},
    "colour": {"on": False, "hex": "#f18a58", "tolerance": 0.25, "min_size": 0.001},
    "lights": {"on": False, "level": 0.85, "min_size": 0.0002},
    "yolo": {"on": False, "model": "", "classes": ["person"], "confidence": 0.35},
    "words": {"on": False, "model": "", "words": "person", "confidence": 0.3},
    "body": {"on": False},
    "hands": {"on": False},
    "face": {"on": False},
    "motif": {"on": False, "model": "", "points": []},       # points: [[frame, x 0-1, y 0-1, 1 = this / 0 = not this, object]]
    "outline": {"on": False, "kinds": ["yolo"], "model": ""},
    "analysis": "960", "keep_lost": 8, "smooth": 0.3, "min_frames": 3, "max_tracks": 200,
    "depth_near": "bright",                              # a depth picture's near side: bright (Depth Anything) or dark (Marigold)
}

# MediaPipe pose / hand connections (as in mediapipe.solutions.*.POSE_CONNECTIONS, written out so the web file and
# the tests do not need MediaPipe)
BODY_NAMES = ["nose", "left_eye_inner", "left_eye", "left_eye_outer", "right_eye_inner", "right_eye", "right_eye_outer",
              "left_ear", "right_ear", "mouth_left", "mouth_right", "left_shoulder", "right_shoulder", "left_elbow",
              "right_elbow", "left_wrist", "right_wrist", "left_pinky", "right_pinky", "left_index", "right_index",
              "left_thumb", "right_thumb", "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle",
              "right_ankle", "left_heel", "right_heel", "left_foot_index", "right_foot_index"]
BODY_EDGES = [[0, 1], [1, 2], [2, 3], [3, 7], [0, 4], [4, 5], [5, 6], [6, 8], [9, 10], [11, 12], [11, 13], [13, 15],
              [15, 17], [15, 19], [15, 21], [17, 19], [12, 14], [14, 16], [16, 18], [16, 20], [16, 22], [18, 20],
              [11, 23], [12, 24], [23, 24], [23, 25], [24, 26], [25, 27], [26, 28], [27, 29], [28, 30], [29, 31],
              [30, 32], [27, 31], [28, 32]]
HAND_EDGES = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [5, 9], [9, 10], [10, 11], [11, 12],
              [9, 13], [13, 14], [14, 15], [15, 16], [13, 17], [0, 17], [17, 18], [18, 19], [19, 20]]
HAND_NAMES = ["wrist", "thumb_1", "thumb_2", "thumb_3", "thumb_tip", "index_1", "index_2", "index_3", "index_tip",
              "middle_1", "middle_2", "middle_3", "middle_tip", "ring_1", "ring_2", "ring_3", "ring_tip",
              "pinky_1", "pinky_2", "pinky_3", "pinky_tip"]
COCO_NAMES = ["nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder", "right_shoulder", "left_elbow",
              "right_elbow", "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle",
              "right_ankle"]
COCO_EDGES = [[0, 1], [0, 2], [1, 3], [2, 4], [5, 6], [5, 7], [7, 9], [6, 8], [8, 10], [5, 11], [6, 12], [11, 12],
              [11, 13], [13, 15], [12, 14], [14, 16]]
FACE_KEEP = [1, 33, 263, 61, 291, 199, 10, 152, 234, 454, 168, 6, 4, 13, 14, 70, 300, 105, 334, 159, 386, 145, 374]


def settings(raw) -> dict:
    """The node's hidden settings text (JSON) merged over DEFAULTS; anything unreadable = the defaults."""
    try:
        got = json.loads(raw) if isinstance(raw, str) and raw.strip() else (raw if isinstance(raw, dict) else {})
    except json.JSONDecodeError:
        got = {}
    out = json.loads(json.dumps(DEFAULTS))
    for k, v in (got or {}).items():
        if isinstance(out.get(k), dict) and isinstance(v, dict):
            out[k].update(v)
        elif k in out:
            out[k] = v
    return out


# ------------------------------------------------------------------------------------------------
# detections
# ------------------------------------------------------------------------------------------------

@dataclass
class Det:
    kind: str
    box: tuple                         # x, y, w, h in analysis pixels
    label: str = ""
    score: float = 1.0
    outline: np.ndarray | None = None  # (n, 2) analysis pixels
    points: dict | None = None         # name -> (x, y)


def _components(mask, kind, min_area, label="", limit=64):
    """Blobs of a binary mask -> Det list (largest first), outlines simplified."""
    m = mask.astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    order = np.argsort(-stats[1:, cv2.CC_STAT_AREA]) + 1 if n > 1 else []
    out = []
    for i in order[:limit]:
        x, y, w, h, a = stats[i]
        if a < min_area:
            break
        sub = (lab[y:y + h, x:x + w] == i).astype(np.uint8)
        cnt, _ = cv2.findContours(sub, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        poly = None
        if cnt:
            c = max(cnt, key=cv2.contourArea)
            c = cv2.approxPolyDP(c, max(1.0, 0.01 * cv2.arcLength(c, True)), True)
            poly = c.reshape(-1, 2).astype(np.float32) + [x, y]
        out.append(Det(kind, (float(x), float(y), float(w), float(h)), label, 1.0, poly))
    return out


class MotionFinder:
    def __init__(self, s, area):
        sens = float(s.get("sensitivity", 0.5))
        self.bg = cv2.createBackgroundSubtractorMOG2(history=150, varThreshold=float(np.interp(sens, [0, 1], [60, 6])),
                                                     detectShadows=False)
        self.min_area = max(4, float(s.get("min_size", 0.002)) * area)
        self.k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self.amount = 0.0

    def __call__(self, rgb8, t):
        m = self.bg.apply(rgb8, learningRate=-1 if t > 0 else 1.0)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, self.k)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, self.k, iterations=2)
        self.amount = float((m > 0).mean())
        return [] if t == 0 else _components(m > 0, "motion", self.min_area)


def hex_rgb(h):
    h = str(h or "").lstrip("#")
    try:
        return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4)) if len(h) == 6 else (241, 138, 88)
    except ValueError:
        return 241, 138, 88


class ColourFinder:
    def __init__(self, s, area):
        rgb = np.uint8([[hex_rgb(s.get("hex"))]])
        self.ref = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)[0, 0]
        self.tol = float(np.interp(float(s.get("tolerance", 0.25)), [0, 1], [6, 90]))
        self.min_area = max(4, float(s.get("min_size", 0.001)) * area)
        self.label = "#%02x%02x%02x" % hex_rgb(s.get("hex"))

    def __call__(self, rgb8, t):
        lab = cv2.cvtColor(rgb8, cv2.COLOR_RGB2LAB).astype(np.float32)
        d = np.sqrt(((lab - self.ref) ** 2).sum(-1))
        m = cv2.morphologyEx((d < self.tol).astype(np.uint8), cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
        k = max(5, int(round(0.009 * rgb8.shape[1])) | 1)    # a textured patch of the colour (glowing glass, a printed
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))   # pattern) as one blob
        return _components(m, "colour", self.min_area, self.label)


class LightFinder:
    def __init__(self, s, area):
        self.level = float(s.get("level", 0.85))
        self.min_area = max(2, float(s.get("min_size", 0.0002)) * area)

    def __call__(self, rgb8, t):
        y = cv2.cvtColor(rgb8, cv2.COLOR_RGB2GRAY)
        m = cv2.GaussianBlur(y, (5, 5), 0) >= int(self.level * 255)
        return _components(m, "lights", self.min_area)


# ---- model folders (never downloaded: only what is on disk)
def model_dirs(kind):
    try:
        import folder_paths
        root = folder_paths.models_dir
    except Exception:  # noqa: BLE001
        root = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "models")
    return [os.path.join(root, d) for d in {"yolo": ["ultralytics", "ultralytics/bbox", "ultralytics/segm", "yolo"],
                                            "sam2": ["sam2", "RMBG/sam2"], "dino": ["grounding-dino", "RMBG/grounding-dino"]}[kind]]


def list_models(kind):
    ext = {"yolo": (".pt",), "dino": (".safetensors", ".pth")}.get(kind, (".safetensors", ".pt"))
    out = []
    for d in model_dirs(kind):
        if os.path.isdir(d):
            out += [f for f in sorted(os.listdir(d)) if f.lower().endswith(ext) and f not in out
                    and not (kind == "yolo" and ("-cls" in f or "-obb" in f))]
    if kind == "dino":                                   # one of each (the .safetensors when both are there)
        out = sorted({o.rsplit(".", 1)[0]: o for o in sorted(out, key=lambda f: f.endswith(".safetensors"))}.values())
    return out


def model_path(kind, name):
    """The file of a model name (empty or not found = the first one on disk)."""
    have = list_models(kind)
    for pick in ([name] if name else []) + have:
        for d in model_dirs(kind):
            p = os.path.join(d, pick)
            if os.path.isfile(p):
                return p
    raise ValueError(f"no {kind} model on disk (put one in models/{os.path.basename(model_dirs(kind)[0])})")


def _quiet_ultralytics():
    """No anonymous usage data and no automatic pip installs from this process (nothing written to its settings)."""
    os.environ["YOLO_AUTOINSTALL"] = "false"
    os.environ["YOLO_OFFLINE"] = "true"                  # ultralytics probes 1.1.1.1 on import unless this is "true"
    for where in ("ultralytics.utils.events", "ultralytics.hub.utils"):     # 8.4 / 8.3
        try:
            import importlib
            importlib.import_module(where).events.enabled = False
        except Exception:  # noqa: BLE001
            pass
    try:
        from ultralytics.utils import checks
        checks.AUTOINSTALL = False
    except Exception:  # noqa: BLE001
        pass


class YoloFinder:
    def __init__(self, s, area):
        _quiet_ultralytics()
        from ultralytics import YOLO
        self.path = model_path("yolo", s.get("model", ""))
        self.model = YOLO(self.path)
        names = self.model.names
        want = [str(c).lower() for c in s.get("classes") or [] if str(c).strip()]
        self.classes = None if not want or "any" in want else [i for i, n in names.items() if n.lower() in want]
        if self.classes == []:
            raise ValueError(f"yolo: none of {want} is a kind this model knows ({', '.join(list(names.values())[:12])} …)")
        self.conf = float(s.get("confidence", 0.35))
        self.names = names
        self.task = getattr(self.model, "task", "detect")   # pose: body points for every person; segment: outlines
        try:
            import torch
            self.device = 0 if torch.cuda.is_available() else "cpu"
        except Exception:  # noqa: BLE001
            self.device = "cpu"

    def __call__(self, rgb8, t):
        r = self.model.predict(np.ascontiguousarray(rgb8[..., ::-1]), conf=self.conf, classes=self.classes,
                               device=self.device, verbose=False)[0]
        out = []
        kps = r.keypoints.data.cpu().numpy() if getattr(r, "keypoints", None) is not None else None
        polys = r.masks.xy if getattr(r, "masks", None) is not None else None
        for j, ((x0, y0, x1, y1), c, k) in enumerate(zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(), r.boxes.cls.cpu().numpy())):
            d = Det("yolo", (float(x0), float(y0), float(x1 - x0), float(y1 - y0)), self.names[int(k)], float(c))
            if kps is not None and j < len(kps):
                d.points = {n: (float(p[0]), float(p[1])) for n, p in zip(COCO_NAMES, kps[j]) if p[2] > 0.35}
            if polys is not None and j < len(polys) and len(polys[j]) >= 3:
                pl = np.asarray(polys[j], np.float32).reshape(-1, 1, 2)
                pl = cv2.approxPolyDP(pl, max(1.0, 0.004 * cv2.arcLength(pl, True)), True)
                d.outline = pl.reshape(-1, 2)
            out.append(d)
        return out

    def close(self):
        self.model = None


class PoseFinder:
    """MediaPipe pose (the full model that ships inside the package): one person per crop."""

    def __init__(self):
        import mediapipe as mp
        self.pose = mp.solutions.pose.Pose(static_image_mode=True, model_complexity=1, enable_segmentation=False)

    def run(self, rgb8, box=None):
        H, W = rgb8.shape[:2]
        if box is None:
            x0, y0, x1, y1 = 0, 0, W, H
        else:
            x, y, w, h = box
            pad = 0.12
            x0, y0 = int(max(0, x - w * pad)), int(max(0, y - h * pad))
            x1, y1 = int(min(W, x + w * (1 + pad))), int(min(H, y + h * (1 + pad)))
        if x1 - x0 < 8 or y1 - y0 < 8:
            return None
        crop = rgb8[y0:y1, x0:x1]
        up = 384 / max(crop.shape[:2])                   # a far-away person: blown up so MediaPipe sees a body
        if up > 1:
            crop = cv2.resize(crop, None, fx=up, fy=up, interpolation=cv2.INTER_CUBIC)
        res = self.pose.process(np.ascontiguousarray(crop))
        if not res.pose_landmarks:
            return None
        pts = {}
        for name, lm in zip(BODY_NAMES, res.pose_landmarks.landmark):
            if lm.visibility > 0.35:
                pts[name] = (x0 + lm.x * (x1 - x0), y0 + lm.y * (y1 - y0))
        if len(pts) < 4:
            return None
        xy = np.array(list(pts.values()))
        bx = (float(xy[:, 0].min()), float(xy[:, 1].min()), float(np.ptp(xy[:, 0]) + 1), float(np.ptp(xy[:, 1]) + 1))
        return Det("body", bx, "body", 1.0, None, pts)

    def close(self):
        self.pose.close()


class HandFinder:
    def __init__(self):
        import mediapipe as mp
        self.hands = mp.solutions.hands.Hands(static_image_mode=True, max_num_hands=8, model_complexity=1)

    def __call__(self, rgb8, t):
        H, W = rgb8.shape[:2]
        res = self.hands.process(np.ascontiguousarray(rgb8))
        out = []
        for lms, hd in zip(res.multi_hand_landmarks or [], res.multi_handedness or []):
            pts = {n: (lm.x * W, lm.y * H) for n, lm in zip(HAND_NAMES, lms.landmark)}
            xy = np.array(list(pts.values()))
            out.append(Det("hands", (float(xy[:, 0].min()), float(xy[:, 1].min()), float(np.ptp(xy[:, 0]) + 1), float(np.ptp(xy[:, 1]) + 1)),
                           hd.classification[0].label.lower(), float(hd.classification[0].score), None, pts))
        return out

    def close(self):
        self.hands.close()


class FaceFinder:
    def __init__(self):
        import mediapipe as mp
        self.mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=8, refine_landmarks=False)

    def __call__(self, rgb8, t):
        H, W = rgb8.shape[:2]
        res = self.mesh.process(np.ascontiguousarray(rgb8))
        out = []
        for lms in res.multi_face_landmarks or []:
            allxy = np.array([(lm.x * W, lm.y * H) for lm in lms.landmark], np.float32)
            pts = {f"p{i}": tuple(allxy[i]) for i in FACE_KEEP}
            hull = cv2.convexHull(allxy).reshape(-1, 2)
            out.append(Det("face", (float(allxy[:, 0].min()), float(allxy[:, 1].min()), float(np.ptp(allxy[:, 0]) + 1),
                                    float(np.ptp(allxy[:, 1]) + 1)), "face", 1.0, hull, pts))
        return out

    def close(self):
        self.mesh.close()


class DinoFinder:
    """Grounding DINO: things named by words ("umbrella . pigeon . head"), each word its own label. The text model
    (bert-base-uncased) is read from the Hugging Face cache folder on disk, never from the hub."""

    def __init__(self, s):
        import torch
        from groundingdino.models import build_model
        from groundingdino.util.slconfig import SLConfig
        from groundingdino.util.utils import clean_state_dict
        path = model_path("dino", s.get("model", ""))
        big = "swinb" in os.path.basename(path).lower()
        cfg = next((os.path.join(d, n) for d in model_dirs("dino") for n in (["GroundingDINO_SwinB.cfg.py"] if big else ["GroundingDINO_SwinT_OGC.cfg.py"])
                    if os.path.isfile(os.path.join(d, n))), None)
        if cfg is None:
            import groundingdino
            cfg = os.path.join(os.path.dirname(groundingdino.__file__), "config", "GroundingDINO_SwinB_cfg.py" if big else "GroundingDINO_SwinT_OGC.py")
        from transformers import BertModel
        if not hasattr(BertModel, "get_head_mask"):    # transformers 5 dropped it; Grounding DINO's wrapper still asks
            BertModel.get_head_mask = lambda self, head_mask, n, is_attention_chunked=False: [None] * n if head_mask is None else head_mask
        args = SLConfig.fromfile(cfg)
        args.text_encoder_type = _bert_folder()
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        args.device = self.dev
        m = build_model(args)
        if path.endswith(".safetensors"):
            from safetensors.torch import load_file
            sd = load_file(path)
        else:
            sd = torch.load(path, map_location="cpu", weights_only=False)
            sd = sd.get("model", sd)
        m.load_state_dict(clean_state_dict(sd), strict=False)
        bert = getattr(m, "bert", None)                  # transformers 5: the third argument is a dtype now, not a device
        if bert is not None and hasattr(bert, "get_extended_attention_mask"):
            orig = bert.get_extended_attention_mask
            bert.get_extended_attention_mask = lambda am, shape, *rest: orig(am, shape)
        self.model = m.to(self.dev).eval()
        words = [w.strip().lower() for w in re.split(r"[.,;\n]+", str(s.get("words", ""))) if w.strip()]
        if not words:
            raise ValueError("words: type what to find (e.g. umbrella, pigeon, head)")
        self.words = words
        self.caption = " . ".join(words) + " ."
        self.box_t = float(s.get("confidence", 0.3))
        self.text_t = 0.25

    def __call__(self, rgb8, t):
        import torch
        from groundingdino.util.inference import predict
        H, W = rgb8.shape[:2]
        k = 800 / min(H, W)
        if max(H, W) * k > 1333:
            k = 1333 / max(H, W)
        im = cv2.resize(rgb8, (max(1, round(W * k)), max(1, round(H * k))), interpolation=cv2.INTER_AREA if k < 1 else cv2.INTER_LINEAR)
        x = torch.from_numpy(im).float().div(255).permute(2, 0, 1)
        x = (x - torch.tensor([0.485, 0.456, 0.406])[:, None, None]) / torch.tensor([0.229, 0.224, 0.225])[:, None, None]
        with torch.inference_mode():
            boxes, logits, phrases = predict(self.model, x, self.caption, self.box_t, self.text_t, device=self.dev)
        out = []
        for (cx, cy, bw, bh), sc, ph in zip(boxes.cpu().numpy(), logits.cpu().numpy(), phrases):
            label = next((w for w in self.words if w in ph), ph.strip() or self.words[0])
            out.append(Det("words", (float((cx - bw / 2) * W), float((cy - bh / 2) * H), float(bw * W), float(bh * H)), label, float(sc)))
        return out

    def close(self):
        self.model = None


def _bert_folder():
    """The bert-base-uncased snapshot in the Hugging Face cache (Grounding DINO's text model), as a local folder."""
    roots = [os.environ.get("HF_HUB_CACHE"), os.path.join(os.environ.get("HF_HOME", ""), "hub") if os.environ.get("HF_HOME") else None,
             os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")]
    for d in [r for r in roots if r]:
        snap = os.path.join(d, "models--bert-base-uncased", "snapshots")
        if os.path.isdir(snap):
            for s_ in sorted(os.listdir(snap)):
                f = os.path.join(snap, s_)
                if os.path.isfile(os.path.join(f, "config.json")) and os.path.isfile(os.path.join(f, "vocab.txt")):
                    return f
    try:
        import folder_paths
        f = os.path.join(folder_paths.models_dir, "bert-base-uncased")
        if os.path.isfile(os.path.join(f, "config.json")):
            return f
    except Exception:  # noqa: BLE001
        pass
    raise ValueError("words: Grounding DINO needs bert-base-uncased on disk (Hugging Face cache or models/bert-base-uncased)")


# ---- SAM 2.1 (the sam2 package that is installed; weights from models/sam2)
SAM2_CONFIGS = {"tiny": "sam2.1_hiera_t.yaml", "small": "sam2.1_hiera_s.yaml", "base_plus": "sam2.1_hiera_b+.yaml",
                "large": "sam2.1_hiera_l.yaml"}


class _NoJit:
    """SAM 2 compiles its picture transforms with torch.jit.script, which needs the Python source of the standard
    library; the portable Python ships it zipped. The transforms run the same uncompiled."""

    def __enter__(self):
        import torch
        self.keep = torch.jit.script
        torch.jit.script = lambda obj, *a, **k: obj
        return self

    def __exit__(self, *exc):
        import torch
        torch.jit.script = self.keep


def _sam2_model(name, video):
    import torch
    from sam2.build_sam import build_sam2, build_sam2_video_predictor
    path = model_path("sam2", name)
    size = next((k for k in SAM2_CONFIGS if k in os.path.basename(path).lower()), "tiny")
    cfg = "configs/sam2.1/" + SAM2_CONFIGS[size]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    build = build_sam2_video_predictor if video else build_sam2
    with _NoJit():
        m = build(cfg, None, device=dev)
    if path.endswith(".safetensors"):
        from safetensors.torch import load_file
        sd = load_file(path)
    else:
        sd = torch.load(path, map_location="cpu", weights_only=True)
        sd = sd.get("model", sd)
    missing, unexpected = m.load_state_dict({k: v.float() for k, v in sd.items()}, strict=False)
    if len(missing) > 8:
        raise ValueError(f"SAM 2 weights {os.path.basename(path)} do not fit the {size} model ({len(missing)} missing)")
    return m.eval(), dev


def _mask_det(mask, kind, label):
    m = (mask > 0).astype(np.uint8)
    if m.sum() < 4:
        return None
    d = _components(m, kind, 1, label, limit=1)
    if not d:
        return None
    ys, xs = np.nonzero(m)
    d[0].box = (float(xs.min()), float(ys.min()), float(xs.max() - xs.min() + 1), float(ys.max() - ys.min() + 1))
    return d[0]


def sam2_motif(frames8, points, model_name, work_dir):
    """points [[t, x, y, positive, object] (x, y 0-1)] -> {t: [Det ...]} followed through all frames (both ways)."""
    import torch
    pred, dev = _sam2_model(model_name, True)
    os.makedirs(work_dir, exist_ok=True)
    for i, f in enumerate(frames8):                      # the video predictor reads a folder of numbered JPEGs
        p = os.path.join(work_dir, f"{i:05d}.jpg")
        if not os.path.isfile(p):
            cv2.imwrite(p, cv2.cvtColor(f, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
    H, W = frames8[0].shape[:2]
    out = {}
    with torch.inference_mode(), torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
        state = pred.init_state(video_path=work_dir, offload_video_to_cpu=True)
        by = {}
        for t, x, y, pos, obj in points:
            by.setdefault((int(t), int(obj)), []).append(([x * W, y * H], 1 if pos else 0))
        for (t, obj), pl in by.items():
            pred.add_new_points_or_box(state, frame_idx=min(max(t, 0), len(frames8) - 1), obj_id=obj + 1,
                                       points=np.array([p for p, _ in pl], np.float32), labels=np.array([l for _, l in pl], np.int32))
        for reverse in (False, True):
            for t, ids, logits in pred.propagate_in_video(state, reverse=reverse):
                for oid, lg in zip(ids, logits):
                    d = _mask_det((lg[0] > 0).cpu().numpy(), "motif", f"motif {oid}")
                    if d is not None:
                        out.setdefault(t, {})[oid] = d
    del pred
    return {t: list(v.values()) for t, v in out.items()}


class Sam2Outliner:
    """The exact outline inside boxes (SAM 2.1 image predictor)."""

    def __init__(self, model_name):
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        m, self.dev = _sam2_model(model_name, False)
        with _NoJit():
            self.pred = SAM2ImagePredictor(m)

    def __call__(self, rgb8, dets):
        import torch
        if not dets:
            return
        with torch.inference_mode(), torch.autocast(self.dev, dtype=torch.bfloat16, enabled=self.dev == "cuda"):
            self.pred.set_image(rgb8)
            boxes = np.array([[d.box[0], d.box[1], d.box[0] + d.box[2], d.box[1] + d.box[3]] for d in dets], np.float32)
            masks, _, _ = self.pred.predict(box=boxes, multimask_output=False)
        masks = np.asarray(masks).reshape(len(dets), -1, *rgb8.shape[:2])[:, 0]
        for d, m in zip(dets, masks):
            o = _mask_det(m, d.kind, d.label)
            if o is not None:
                d.outline = o.outline

    def close(self):
        self.pred = None


# ------------------------------------------------------------------------------------------------
# ids over time
# ------------------------------------------------------------------------------------------------

def iou(a, b):
    ax1, ay1, bx1, by1 = a[0] + a[2], a[1] + a[3], b[0] + b[2], b[1] + b[3]
    iw, ih = max(0.0, min(ax1, bx1) - max(a[0], b[0])), max(0.0, min(ay1, by1) - max(a[1], b[1]))
    inter = iw * ih
    return inter / max(1e-6, a[2] * a[3] + b[2] * b[3] - inter)


@dataclass
class _Live:
    id: int
    kind: str
    label: str
    dets: dict = field(default_factory=dict)     # t -> Det
    last: int = 0
    vel: tuple = (0.0, 0.0)


def associate(per_frame: list[list[Det]], keep_lost=8, max_tracks=200) -> list[_Live]:
    """Detections per frame -> tracks (ids kept per kind). Hungarian matching on box overlap and centre distance,
    the next position guessed from the speed; a track lost for more than keep_lost frames ends."""
    from scipy.optimize import linear_sum_assignment
    live, done, next_id = [], [], 1
    for t, dets in enumerate(per_frame):
        for kind in sorted({d.kind for d in dets} | {L.kind for L in live}):
            cand = [L for L in live if L.kind == kind]
            ds = [d for d in dets if d.kind == kind]
            used_t, used_d = set(), set()
            if cand and ds:
                C = np.full((len(cand), len(ds)), 9.0)
                for i, L in enumerate(cand):
                    p = L.dets[L.last].box
                    gap = t - L.last
                    guess = (p[0] + L.vel[0] * gap, p[1] + L.vel[1] * gap, p[2], p[3])
                    for j, d in enumerate(ds):
                        if L.label and d.label and L.label != d.label and kind != "motif":
                            continue
                        ov = iou(guess, d.box)
                        cx, cy = guess[0] + guess[2] / 2, guess[1] + guess[3] / 2
                        dx, dy = d.box[0] + d.box[2] / 2 - cx, d.box[1] + d.box[3] / 2 - cy
                        reach = max(8.0, 0.75 * max(guess[2], guess[3], d.box[2], d.box[3])) * (1 + 0.25 * (gap - 1))
                        dist = np.hypot(dx, dy) / reach
                        if ov > 0.05 or dist < 1.0:
                            C[i, j] = (1 - ov) + min(dist, 1.0)
                for i, j in zip(*linear_sum_assignment(C)):
                    if C[i, j] < 9.0:
                        L, d = cand[i], ds[j]
                        p = L.dets[L.last].box
                        gap = t - L.last
                        v = ((d.box[0] + d.box[2] / 2 - p[0] - p[2] / 2) / gap, (d.box[1] + d.box[3] / 2 - p[1] - p[3] / 2) / gap)
                        L.vel = (0.6 * L.vel[0] + 0.4 * v[0], 0.6 * L.vel[1] + 0.4 * v[1])
                        L.dets[t] = d
                        L.last = t
                        used_t.add(i)
                        used_d.add(j)
            for j, d in enumerate(ds):
                if j not in used_d and len(live) < max_tracks:
                    L = _Live(next_id, kind, d.label if kind != "motif" else d.label, {t: d}, t)
                    next_id += 1
                    live.append(L)
        for L in [q for q in live if t - q.last > keep_lost]:
            live.remove(L)
            done.append(L)
    return done + live


# ------------------------------------------------------------------------------------------------
# tracks
# ------------------------------------------------------------------------------------------------

class Tracks:
    def __init__(self, W, H, n, fps):
        self.W, self.H, self.n, self.fps = int(W), int(H), int(n), float(fps)
        self.tracks = []
        self.curves = {"motion": [0.0] * n, "count": [0] * n, "luma": [0.0] * n}

    def to_dict(self):
        return {"format": TRACKS_FORMAT, "W": self.W, "H": self.H, "n": self.n, "fps": self.fps, "tracks": self.tracks,
                "curves": self.curves, "skeleton": {"body": BODY_EDGES, "hands": HAND_EDGES, "yolo": COCO_EDGES},
                "names": {"body": BODY_NAMES, "hands": HAND_NAMES, "yolo": COCO_NAMES}}


def _smooth(a, win):
    if win <= 1 or len(a) < 3:
        return a
    k = np.ones(win) / win
    pad = np.pad(a, ((win // 2, win - 1 - win // 2), (0, 0)), mode="edge")
    return np.stack([np.convolve(pad[:, c], k, mode="valid") for c in range(a.shape[1])], 1)


def build_tracks(lives, frames8, scale, s, n, fps, W, H):
    """Tracks -> the dict web and draw read: gaps filled, paths smoothed, sizes in the video's own pixels."""
    T = Tracks(W, H, n, fps)
    win = 1 + 2 * int(round(float(s.get("smooth", 0.3)) * 6))
    min_frames = max(1, int(s.get("min_frames", 3)))
    for L in sorted(lives, key=lambda q: (min(q.dets), q.id)):
        ts = sorted(L.dets)
        if len(ts) < min_frames:
            continue
        t0, t1 = ts[0], ts[-1]
        boxes = np.array([L.dets[t].box for t in ts], np.float64)
        full = np.stack([np.interp(np.arange(t0, t1 + 1), ts, boxes[:, c]) for c in range(4)], 1)
        full = _smooth(full, win) / scale
        seen = np.zeros(t1 - t0 + 1, np.int8)
        seen[np.array(ts) - t0] = 1
        # colour: the median of the box centres over the frames it was seen
        cols = []
        for t in ts[:: max(1, len(ts) // 8)]:
            x, y, w, h = L.dets[t].box
            f = frames8[t]
            cx, cy = int(min(max(x + w / 2, 0), f.shape[1] - 1)), int(min(max(y + h / 2, 0), f.shape[0] - 1))
            r = max(1, int(min(w, h) * 0.2))
            cols.append(f[max(0, cy - r):cy + r + 1, max(0, cx - r):cx + r + 1].reshape(-1, 3).mean(0))
        c = np.median(np.array(cols), 0) if cols else np.zeros(3)
        tr = {"id": L.id, "kind": L.kind, "label": L.label or L.kind, "t0": t0,
              "box": [[round(float(v), 1) for v in b] for b in full], "seen": seen.tolist(),
              "colour": "#%02x%02x%02x" % tuple(int(v) for v in np.clip(c, 0, 255))}
        outl = {str(t): np.round(L.dets[t].outline / scale, 1).tolist() for t in ts if L.dets[t].outline is not None and len(L.dets[t].outline) >= 3}
        if outl:
            tr["outline"] = outl
        pts = {str(t): {k: [round(v[0] / scale, 1), round(v[1] / scale, 1)] for k, v in L.dets[t].points.items()}
               for t in ts if L.dets[t].points}
        if pts:
            tr["points"] = pts
        T.tracks.append(tr)
        for t in range(t0, t1 + 1):
            T.curves["count"][t] += 1
    return T


def frame_rgb8(frame) -> np.ndarray:
    a = np.asarray(frame, np.float32)[..., :3]
    return (np.clip(a, 0, 1) * 255 + 0.5).astype(np.uint8)


def depth_frames(depth, n, H, W, near="bright"):
    """A depth clip from any depth node (n2, h, w[, c], any scale) -> (n, H, W) float32 0-1, near = 1, one scale for the
    whole clip (2nd / 98th percentile of a few frames), so a thing keeps its depth from frame to frame. Fewer frames
    than the video: the last one holds."""
    d = depth.cpu().float().numpy() if hasattr(depth, "cpu") else np.asarray(depth, np.float32)
    if d.ndim == 4:
        d = d[..., :3].mean(-1)
    if d.ndim == 2:
        d = d[None]
    pick = d[np.linspace(0, len(d) - 1, min(len(d), 12)).astype(int)]
    lo, hi = np.percentile(pick, 2), np.percentile(pick, 98)
    out = np.empty((n, H, W), np.float32)
    for i in range(n):
        f = d[min(i, len(d) - 1)]
        if f.shape != (H, W):
            f = cv2.resize(f, (W, H), interpolation=cv2.INTER_LINEAR)
        f = np.clip((f - lo) / max(1e-6, hi - lo), 0, 1)
        out[i] = f if near != "dark" else 1.0 - f
    return out


def add_depth(data, dz):
    """Each track's z per frame (its median depth inside the middle of its box or its outline): 0 far .. 1 near."""
    n, H, W = dz.shape
    for tr in data["tracks"]:
        zs = []
        for k, b in enumerate(tr["box"]):
            t = tr["t0"] + k
            f = dz[min(t, n - 1)]
            o = (tr.get("outline") or {}).get(str(t))
            if o and len(o) >= 3:
                m = np.zeros((H, W), np.uint8)
                cv2.fillPoly(m, [np.round(np.asarray(o)).astype(np.int32)], 1)
                vals = f[m > 0]
            else:
                x, y, w, h = b
                x0, y0 = int(max(0, x + w * 0.25)), int(max(0, y + h * 0.25))
                x1, y1 = int(min(W, x + w * 0.75 + 1)), int(min(H, y + h * 0.75 + 1))
                vals = f[y0:y1, x0:x1].ravel() if x1 > x0 and y1 > y0 else np.array([])
            zs.append(float(np.median(vals)) if len(vals) else (zs[-1] if zs else 0.5))
        if len(zs) >= 5:                                  # the depth model flickers a little: a short centred median
            zs = [float(np.median(zs[max(0, i - 2):i + 3])) for i in range(len(zs))]
        tr["z"] = [round(z, 3) for z in zs]
    data["curves"]["depth"] = [round(float(dz[min(i, n - 1)].mean()), 4) for i in range(data["n"])]
    return data


def analyse(frames, fps=25.0, raw_settings="", work_dir=None, progress=None, only=None) -> Tracks:
    """frames: (n, H, W, 3) float 0-1 (numpy or torch) -> Tracks. only = frame indices to look at (the one-frame
    check in the node); the rest are skipped."""
    s = settings(raw_settings)
    n = int(frames.shape[0])
    H, W = int(frames.shape[1]), int(frames.shape[2])
    side = ANALYSIS.get(str(s.get("analysis", "960")), 960)
    scale = 1.0 if not side or max(W, H) <= side else side / max(W, H)
    aw, ah = max(1, int(round(W * scale))), max(1, int(round(H * scale)))
    idx = list(range(n)) if only is None else sorted({min(max(int(i), 0), n - 1) for i in only})
    need = set(range(n)) if only is None else set(idx) | {max(0, i - 2) for i in idx}   # motion compares with a frame before
    small = {}
    for i in sorted(need):
        f = frame_rgb8(frames[i] if not hasattr(frames, "cpu") else frames[i].cpu().numpy())
        small[i] = cv2.resize(f, (aw, ah), interpolation=cv2.INTER_AREA) if scale < 1 else f
    area = aw * ah
    finders, closers = [], []
    try:
        if s["motion"]["on"]:
            finders.append(MotionFinder(s["motion"], area))
        if s["colour"]["on"]:
            finders.append(ColourFinder(s["colour"], area))
        if s["lights"]["on"]:
            finders.append(LightFinder(s["lights"], area))
        yolo = YoloFinder(s["yolo"], area) if s["yolo"]["on"] else None
        if s["words"]["on"]:
            finders.append(DinoFinder(s["words"]))
        pose = PoseFinder() if s["body"]["on"] else None
        for on, make in ((s["hands"]["on"], HandFinder), (s["face"]["on"], FaceFinder)):
            if on:
                finders.append(make())
        outliner = Sam2Outliner(s["outline"].get("model", "")) if s["outline"]["on"] else None
        closers = [f for f in finders + [yolo, pose, outliner] if f is not None and hasattr(f, "close")]
        motif = {}
        if s["motif"]["on"] and s["motif"].get("points") and only is None:
            motif = sam2_motif([small[i] for i in range(n)], s["motif"]["points"], s["motif"].get("model", ""),
                               work_dir or os.path.join(os.getcwd(), "kubakub_track_frames"))
        per_frame = [[] for _ in range(n)]
        mot = next((f for f in finders if isinstance(f, MotionFinder)), None)
        for k, i in enumerate(idx):
            f = small[i]
            if mot is not None and only is not None:      # one frame alone: two frames before stand in as background
                mot.bg = cv2.createBackgroundSubtractorMOG2(history=150, varThreshold=mot.bg.getVarThreshold(), detectShadows=False)
                mot(small[max(0, i - 2)], 0)
            dets = []
            for fd in finders:
                dets += fd(f, i if only is None else (1 if isinstance(fd, MotionFinder) else i))
            if yolo is not None:
                yd = yolo(f, i)
                dets += yd
                if pose is not None:
                    people = [d for d in yd if d.label == "person" and not d.points]
                    for d in people:
                        b = pose.run(f, d.box)
                        if b is not None:
                            dets.append(b)
            if pose is not None and yolo is None:
                b = pose.run(f)
                if b is not None:
                    dets.append(b)
            dets += motif.get(i, [])
            if outliner is not None:
                outliner(f, [d for d in dets if d.kind in (s["outline"].get("kinds") or ["yolo"]) and d.outline is None])
            per_frame[i] = dets
            if progress:
                progress(k + 1, len(idx))
        lives = associate(per_frame, int(s.get("keep_lost", 8)), int(s.get("max_tracks", 200)))
        frames_for_colour = [small.get(i) if i in small else np.zeros((ah, aw, 3), np.uint8) for i in range(n)]
        T = build_tracks(lives, frames_for_colour, scale, s, n, fps, W, H)
        if only is not None:                              # one-frame check: every detection of those frames, ids by order
            T = Tracks(W, H, n, fps)
            nid = 1
            for i in idx:
                for d in per_frame[i]:
                    tr = {"id": nid, "kind": d.kind, "label": d.label or d.kind, "t0": i,
                          "box": [[round(v / scale, 1) for v in d.box]], "seen": [1], "colour": "#ffffff"}
                    if d.outline is not None and len(d.outline) >= 3:
                        tr["outline"] = {str(i): np.round(d.outline / scale, 1).tolist()}
                    if d.points:
                        tr["points"] = {str(i): {k2: [round(v[0] / scale, 1), round(v[1] / scale, 1)] for k2, v in d.points.items()}}
                    T.tracks.append(tr)
                    nid += 1
        for i in idx:
            T.curves["luma"][i] = round(float(cv2.cvtColor(small[i], cv2.COLOR_RGB2GRAY).mean() / 255), 4)
        return T
    finally:
        for c in closers:
            try:
                c.close()
            except Exception:  # noqa: BLE001
                pass


# ------------------------------------------------------------------------------------------------
# what other nodes read from tracks: a glow around what is tracked, the arrow of the main thing
# ------------------------------------------------------------------------------------------------

def _at(tr, t):
    k = t - tr["t0"]
    return tr["box"][k] if 0 <= k < len(tr["box"]) else None


def near_field(data, t, w, h, reach=1.5):
    """(h, w) float32 at frame t: 1 on every tracked thing, falling smoothly to 0 at `reach` times its size away
    (the brightest wins where two meet). The picture's size is free (scaled from the tracks' own)."""
    sx, sy = w / data["W"], h / data["H"]
    out = np.zeros((h, w), np.float32)
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    for tr in data["tracks"]:
        b = _at(tr, t)
        if b is None:
            continue
        cx, cy = (b[0] + b[2] / 2) * sx, (b[1] + b[3] / 2) * sy
        rx, ry = max(1.0, b[2] / 2 * sx), max(1.0, b[3] / 2 * sy)
        x0, x1 = int(max(0, cx - rx * (1 + reach))), int(min(w, cx + rx * (1 + reach) + 1))
        y0, y1 = int(max(0, cy - ry * (1 + reach))), int(min(h, cy + ry * (1 + reach) + 1))
        if x1 <= x0 or y1 <= y0:
            continue
        d = np.sqrt(((xs[y0:y1, x0:x1] - cx) / rx) ** 2 + ((ys[y0:y1, x0:x1] - cy) / ry) ** 2)      # 1 = the thing's edge
        u = np.clip(1 - (d - 1) / reach, 0, 1)
        np.maximum(out[y0:y1, x0:x1], u * u * (3 - 2 * u), out=out[y0:y1, x0:x1])
    return out


def main_arrow(data, t, back=6, min_len=0.06):
    """The arrow of the biggest thing at frame t, as drawn on a picture: (x0, y0, x1, y1) 0-1, y down: from where it
    was `back` frames ago to where it is, at least min_len long (a still thing points the way it last went).
    None = nothing there."""
    best = None
    for tr in data["tracks"]:
        b = _at(tr, t)
        if b is not None and (best is None or b[2] * b[3] > best[1][2] * best[1][3]):
            best = (tr, b)
    if best is None:
        return None
    tr, b = best
    W, H = data["W"], data["H"]
    k = t - tr["t0"]
    p = tr["box"][max(0, k - back)]
    x1, y1 = (b[0] + b[2] / 2) / W, (b[1] + b[3] / 2) / H
    x0, y0 = (p[0] + p[2] / 2) / W, (p[1] + p[3] / 2) / H
    dx, dy = x1 - x0, y1 - y0
    L = float(np.hypot(dx, dy))
    if L < 1e-6:
        dx, dy, L = 1.0, 0.0, 1.0
    if L < min_len:
        x0, y0 = x1 - dx / L * min_len, y1 - dy / L * min_len
    return (float(x0), float(y0), float(x1), float(y1))


# ------------------------------------------------------------------------------------------------
# the sample clip (nothing connected): shapes that move, a light, walkers
# ------------------------------------------------------------------------------------------------

def sample_clip(n=72, W=640, H=360) -> np.ndarray:
    rng = np.random.default_rng(7)
    bg = np.zeros((H, W, 3), np.float32)
    bg[:] = np.linspace(0.12, 0.3, W, dtype=np.float32)[None, :, None]
    bg += rng.normal(0, 0.01, bg.shape).astype(np.float32)
    out = np.empty((n, H, W, 3), np.float32)
    walkers = [(rng.uniform(0, W), rng.uniform(80, H - 60), rng.uniform(-4, 4), rng.uniform(-1, 1)) for _ in range(6)]
    for t in range(n):
        f = bg.copy()
        for k, (x, y, vx, vy) in enumerate(walkers):
            cx, cy = (x + vx * t) % W, y + vy * t
            cv2.ellipse(f, (int(cx), int(cy)), (9, 22), 0, 0, 360, (0.75, 0.72, 0.68), -1, cv2.LINE_AA)
            cv2.circle(f, (int(cx), int(cy - 28)), 7, (0.85, 0.75, 0.65), -1, cv2.LINE_AA)
        a = 2 * np.pi * t / n
        cv2.circle(f, (int(W / 2 + 180 * np.cos(a)), int(H / 2 + 90 * np.sin(2 * a))), 16, (0.95, 0.54, 0.35), -1, cv2.LINE_AA)
        cv2.circle(f, (int(80 + 480 * t / n), 40), 6, (1.0, 1.0, 1.0), -1, cv2.LINE_AA)
        out[t] = np.clip(f, 0, 1)
    return out
