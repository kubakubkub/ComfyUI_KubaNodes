"""
library.py

The library behind the kubakub shelf: a small sqlite database of pictures, videos and sounds with what they were made
with, so they can be found again. No ComfyUI imports (the routes are in nodes/utils/nodes_library.py), nothing leaves
this computer.

Per file: where it is, kind, size, width / height, length, date, tags (a dropped folder's name, "this session"), and
what the file itself says about how it was made:
  - ComfyUI: the "prompt" graph (PNG text chunk, WEBP / AVIF exif "prompt:{...}", video container metadata or a
    "comment" holding {prompt, workflow}) -> positive / negative prompt (followed from the sampler to the text
    encoders), seed, steps, cfg, sampler, scheduler, models and LoRAs with their strengths, size
  - A1111 / Forge "parameters" text
The embedded workflow itself is not stored: it is read from the file when asked (workflow_of).

Search: sqlite FTS5 over name, prompts, models and tags (a LIKE search where FTS5 is missing).
Only files under the allowed folders are indexed or handed out (ComfyUI input / output / temp and the folders added
with "index a folder"); see Library.allowed.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import threading
import time

VIDEO_EXT = {".mp4", ".mov", ".webm", ".mkv", ".avi", ".m4v", ".gif"}
IMAGE_EXT = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff", ".avif"}
AUDIO_EXT = {".wav", ".mp3", ".ogg", ".flac", ".m4a", ".aac", ".opus"}
FOLDER_LIMIT = 20000               # files indexed from one folder in one go
SCHEMA = 1


def kind_of(name: str) -> str | None:
    ext = os.path.splitext(str(name))[1].lower()
    return "video" if ext in VIDEO_EXT else "image" if ext in IMAGE_EXT else "audio" if ext in AUDIO_EXT else None


# ------------------------------------------------------------------------------------------------ reading a file
def _json(s):
    if isinstance(s, (dict, list)):
        return s
    try:
        return json.loads(s) if isinstance(s, (str, bytes)) and s else None
    except ValueError:
        return None


def embedded(path: str) -> dict:
    """What the file carries: {prompt: graph | None, workflow: dict | None, parameters: str | None, w, h, duration}."""
    out = {"prompt": None, "workflow": None, "parameters": None, "w": None, "h": None, "duration": None}
    kind = kind_of(path)
    if kind == "image":
        try:
            from PIL import Image
            with Image.open(path) as im:
                out["w"], out["h"] = im.size
                info = dict(im.info or {})
                try:
                    for v in im.getexif().values():          # WEBP / AVIF: "prompt:{...}", "workflow:{...}"
                        if isinstance(v, bytes):
                            v = v.decode("utf-8", "ignore")
                        if isinstance(v, str) and ":" in v:
                            k, _, rest = v.partition(":")
                            if k.lower() in ("prompt", "workflow") and k.lower() not in info:
                                info[k.lower()] = rest
                except Exception:  # noqa: BLE001  (no exif)
                    pass
                if getattr(im, "n_frames", 1) > 1 and im.info.get("duration"):
                    out["duration"] = round(im.n_frames * float(im.info["duration"]) / 1000.0, 3)
            out["prompt"] = _json(info.get("prompt"))
            out["workflow"] = _json(info.get("workflow"))
            p = info.get("parameters")
            out["parameters"] = p if isinstance(p, str) else None
        except Exception:  # noqa: BLE001  (not readable: indexed by name only)
            pass
    elif kind in ("video", "audio"):
        try:
            import av
            with av.open(path) as c:
                meta = {str(k).lower(): v for k, v in (c.metadata or {}).items()}
                if c.duration:
                    out["duration"] = round(c.duration / 1_000_000, 3)
                vs = [s for s in c.streams if s.type == "video"]
                if vs:
                    out["w"], out["h"] = vs[0].codec_context.width or None, vs[0].codec_context.height or None
            out["prompt"] = _json(meta.get("prompt"))
            out["workflow"] = _json(meta.get("workflow"))
            com = _json(meta.get("comment"))              # Video Helper Suite: comment = {"prompt": ..., "workflow": ...}
            if isinstance(com, dict):
                out["prompt"] = out["prompt"] or _json(com.get("prompt"))
                out["workflow"] = out["workflow"] or _json(com.get("workflow"))
        except Exception:  # noqa: BLE001
            pass
    if not isinstance(out["prompt"], dict):
        out["prompt"] = None
    if not isinstance(out["workflow"], dict):
        out["workflow"] = None
    return out


# ------------------------------------------------------------------------------------------------ the prompt graph
SAMPLER_HINTS = ("sampler", "ksampler")
GUIDER_KEYS = ("guider",)
COND_KEYS = ("positive", "negative", "conditioning", "conditioning_1", "conditioning_2", "conditioning_to",
             "conditioning_from", "cond", "uncond")
# links not followed when looking for the prompt text: they lead to pictures, models or latents, not words
NOT_TEXT = {"clip", "model", "vae", "image", "images", "pixels", "latent", "latent_image", "samples", "mask",
            "control_net", "clip_vision", "clip_vision_output", "style_model", "noise", "sigmas", "sampler", "seed",
            "noise_seed", "width", "height", "batch_size", "steps", "cfg", "denoise", "upscale_model", "audio", "video"}
TEXT_KEYS = ("text", "text_g", "text_l", "t5xxl", "clip_l", "prompt", "string", "value", "positive", "negative",
             "user_prompt", "system_prompt", "caption", "string_a", "string_b", "text_a", "text_b", "tags")
MODEL_KEYS = ("ckpt_name", "unet_name", "model_name", "diffusion_model", "gguf_name", "clip_name", "clip_name1",
              "clip_name2", "clip_name3", "vae_name", "control_net_name", "upscale_model_name", "style_model_name")


def _is_link(v):
    return isinstance(v, list) and len(v) == 2 and isinstance(v[0], (str, int)) and isinstance(v[1], int)


def _resolve(graph, v, depth=0):
    """A widget value that is a link to a primitive / string node -> its value."""
    while _is_link(v) and depth < 8:
        n = graph.get(str(v[0])) or {}
        ins = n.get("inputs") or {}
        for k in ("value", "string", "text", "seed", "noise_seed", "int", "float", "number"):
            if k in ins:
                v = ins[k]
                break
        else:
            return None
        depth += 1
    return None if _is_link(v) else v


def _texts(graph, link, seen, depth=0):
    """Every prompt text reached from a conditioning link (through combine / guidance / reference nodes)."""
    if not _is_link(link) or depth > 24:
        return []
    nid = str(link[0])
    if nid in seen:
        return []
    seen.add(nid)
    n = graph.get(nid) or {}
    ct = str(n.get("class_type") or "").lower()
    if "zeroout" in ct:
        return []
    out = []
    for k, v in (n.get("inputs") or {}).items():
        if k in NOT_TEXT:
            continue
        if _is_link(v):
            r = _resolve(graph, v)
            if isinstance(r, str) and k in TEXT_KEYS:
                if r.strip():
                    out.append(r.strip())
            else:
                out += _texts(graph, v, seen, depth + 1)
        elif isinstance(v, str) and k in TEXT_KEYS and v.strip():
            out.append(v.strip())
    return out


def _uniq(xs):
    seen, out = set(), []
    for x in xs:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def parse_prompt(graph: dict) -> dict:
    """A ComfyUI API graph -> {positive, negative, seed, steps, cfg, sampler, scheduler, denoise, models, loras,
    width, height}. Missing things stay None / empty."""
    info = {"positive": "", "negative": "", "seed": None, "steps": None, "cfg": None, "sampler": None,
            "scheduler": None, "denoise": None, "models": [], "loras": [], "width": None, "height": None}
    if not isinstance(graph, dict):
        return info
    graph = {str(k): v for k, v in graph.items() if isinstance(v, dict)}
    pos, neg = [], []
    order = sorted(graph, key=lambda k: (len(k), k))
    samplers = [k for k in order if any(h in str(graph[k].get("class_type", "")).lower() for h in SAMPLER_HINTS)
                and any(c in (graph[k].get("inputs") or {}) for c in ("positive", "negative", "guider"))]
    guiders = [k for k in order if "guider" in str(graph[k].get("class_type", "")).lower()]
    for k in samplers + guiders:
        ins = graph[k].get("inputs") or {}
        for key, dest in (("positive", pos), ("conditioning", pos), ("cond", pos), ("negative", neg), ("uncond", neg)):
            if key in ins:
                dest += _texts(graph, ins[key], set())
    if not pos and not neg:                         # no sampler: every text of a text-ish node
        for k in order:
            n = graph[k]
            ct = str(n.get("class_type", "")).lower()
            if any(w in ct for w in ("text", "prompt", "string", "encode")):
                for key, v in (n.get("inputs") or {}).items():
                    if key in TEXT_KEYS and isinstance(v, str) and v.strip():
                        pos.append(v.strip())
    info["positive"] = "\n\n".join(_uniq(pos))
    info["negative"] = "\n\n".join(t for t in _uniq(neg) if t not in pos)

    def first(keys, typ, nodes=None):
        for k in (nodes or []) + order:
            ins = graph[k].get("inputs") or {}
            for key in keys:
                if key in ins:
                    v = _resolve(graph, ins[key])
                    if isinstance(v, typ) and not isinstance(v, bool):
                        return v
        return None
    info["seed"] = first(("seed", "noise_seed"), int, samplers)
    info["steps"] = first(("steps",), int, samplers)
    info["cfg"] = first(("cfg",), (int, float), samplers)
    info["sampler"] = first(("sampler_name",), str, samplers)
    info["scheduler"] = first(("scheduler",), str, samplers)
    info["denoise"] = first(("denoise",), (int, float), samplers)
    info["width"] = first(("width",), int)
    info["height"] = first(("height",), int)
    models, loras = [], []
    for k in order:
        ins = graph[k].get("inputs") or {}
        for key in MODEL_KEYS:
            v = _resolve(graph, ins.get(key))
            if isinstance(v, str) and v and v.lower() != "none":
                models.append(v)
        ln = _resolve(graph, ins.get("lora_name"))
        if isinstance(ln, str) and ln and ln.lower() != "none":
            sm = _resolve(graph, ins.get("strength_model", ins.get("strength", 1.0)))
            loras.append({"name": ln, "strength": round(float(sm), 3) if isinstance(sm, (int, float)) else None})
        for key, v in ins.items():                  # stacks: lora_1 / lora_01 ... with a matching strength
            if key.startswith("lora_") and key != "lora_name" and isinstance(v, str) and v.lower() not in ("", "none"):
                if os.path.splitext(v)[1].lower() in (".safetensors", ".pt", ".ckpt", ".gguf"):
                    st = ins.get(key.replace("lora_", "strength_"), ins.get(key + "_strength"))
                    loras.append({"name": v, "strength": round(float(st), 3) if isinstance(st, (int, float)) else None})
    info["models"] = _uniq(models)
    seen = set()
    info["loras"] = [lo for lo in loras if not (lo["name"] in seen or seen.add(lo["name"]))]
    return info


_A1111_LORA = re.compile(r"<lora:([^:>]+):?([-\d.]*)>")


def parse_parameters(text: str) -> dict:
    """A1111 / Forge 'parameters' text -> the same dict as parse_prompt."""
    info = {"positive": "", "negative": "", "seed": None, "steps": None, "cfg": None, "sampler": None,
            "scheduler": None, "denoise": None, "models": [], "loras": [], "width": None, "height": None}
    if not isinstance(text, str) or not text.strip():
        return info
    lines = text.strip().split("\n")
    settings = ""
    if re.match(r"^\s*Steps:\s*\d+", lines[-1]):
        settings = lines.pop()
    body = "\n".join(lines)
    pos, _, neg = body.partition("Negative prompt:")
    info["positive"], info["negative"] = pos.strip(), neg.strip()
    kv = {}
    for m in re.finditer(r'\s*([\w ]+):\s*("(?:[^"\\]|\\.)*"|[^,]*)(?:,|$)', settings):
        kv[m.group(1).strip().lower()] = m.group(2).strip().strip('"')

    def num(k, typ):
        try:
            return typ(kv[k])
        except (KeyError, ValueError):
            return None
    info["steps"], info["cfg"], info["seed"] = num("steps", int), num("cfg scale", float), num("seed", int)
    info["sampler"] = kv.get("sampler")
    info["scheduler"] = kv.get("schedule type") or kv.get("scheduler")
    info["denoise"] = num("denoising strength", float)
    if "size" in kv and "x" in kv["size"]:
        w, _, h = kv["size"].partition("x")
        info["width"], info["height"] = (int(w), int(h)) if w.isdigit() and h.isdigit() else (None, None)
    if kv.get("model"):
        info["models"] = [kv["model"]]
    info["loras"] = [{"name": n, "strength": float(s) if re.fullmatch(r"-?\d+(\.\d+)?", s or "") else None}
                     for n, s in _A1111_LORA.findall(info["positive"])]
    return info


def describe(path: str) -> dict:
    """embedded() + the parsed settings: everything the library stores about a file."""
    e = embedded(path)
    info = parse_prompt(e["prompt"]) if e["prompt"] else parse_parameters(e["parameters"]) if e["parameters"] else \
        parse_prompt(None)
    info.update({"w": e["w"], "h": e["h"], "duration": e["duration"], "has_workflow": bool(e["workflow"] or e["prompt"]),
                 "source_app": "comfyui" if e["prompt"] or e["workflow"] else "a1111" if e["parameters"] else ""})
    return info


def workflow_of(path: str):
    """The workflow embedded in the file ({workflow} UI graph, else {prompt} API graph), or None."""
    e = embedded(path)
    if e["workflow"]:
        return {"workflow": e["workflow"]}
    if e["prompt"]:
        return {"prompt": e["prompt"]}
    return None


# ------------------------------------------------------------------------------------------------ the database
def _norm(p):
    return os.path.normcase(os.path.realpath(p))


def inside(path, root):
    path, root = _norm(path), _norm(root)
    return path == root or path.startswith(root.rstrip(os.sep) + os.sep)


class Library:
    """The database. `fixed` = {'input': dir, 'output': dir, 'temp': dir} (always allowed); added folders are kept in
    the table `roots`."""

    def __init__(self, db_path: str, fixed: dict):
        self.db_path, self.fixed = db_path, {k: v for k, v in fixed.items() if v}
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        self.lock = threading.RLock()
        self.con = sqlite3.connect(db_path, check_same_thread=False)
        self.con.row_factory = sqlite3.Row
        self.con.execute("PRAGMA journal_mode=WAL")
        self.fts = True
        with self.lock, self.con:
            self.con.execute("""CREATE TABLE IF NOT EXISTS files (id INTEGER PRIMARY KEY, path TEXT UNIQUE NOT NULL,
                name TEXT, kind TEXT, size INTEGER, mtime REAL, added REAL, w INTEGER, h INTEGER, duration REAL,
                positive TEXT, negative TEXT, seed INTEGER, steps INTEGER, cfg REAL, sampler TEXT, scheduler TEXT,
                denoise REAL, models TEXT, loras TEXT, tags TEXT, has_workflow INTEGER, source_app TEXT)""")
            self.con.execute("CREATE TABLE IF NOT EXISTS roots (path TEXT PRIMARY KEY, added REAL)")
            self.con.execute("CREATE TABLE IF NOT EXISTS meta (k TEXT PRIMARY KEY, v TEXT)")
            self.con.execute("INSERT OR IGNORE INTO meta VALUES ('schema', ?)", (str(SCHEMA),))
            try:
                self.con.execute("CREATE VIRTUAL TABLE IF NOT EXISTS files_fts USING fts5(name, positive, negative, "
                                 "models, tags, tokenize='unicode61 remove_diacritics 2')")
            except sqlite3.OperationalError:
                self.fts = False

    def close(self):
        with self.lock:
            self.con.close()

    # ---- where files may come from
    def roots(self) -> list[str]:
        with self.lock:
            return [r["path"] for r in self.con.execute("SELECT path FROM roots ORDER BY added")]

    def allowed(self, path: str) -> bool:
        return any(inside(path, r) for r in list(self.fixed.values()) + self.roots())

    def source_of(self, path: str) -> str | None:
        """'file:<in input/>' / 'output:<in output/>' (what the shelf and the loaders take), else None."""
        for key, prefix in (("output", "output:"), ("input", "file:")):
            base = self.fixed.get(key)
            if base and inside(path, base):
                return prefix + os.path.relpath(os.path.realpath(path), os.path.realpath(base)).replace(os.sep, "/")
        return None

    def add_root(self, folder: str) -> str:
        p = os.path.realpath(os.path.expanduser(str(folder or "").strip().strip('"')))
        if not folder or not os.path.isdir(p):
            raise ValueError(f"not a folder on this computer: {folder}")
        if os.path.dirname(p) == p:
            raise ValueError("a whole drive is too much: pick a folder")
        with self.lock, self.con:
            self.con.execute("INSERT OR IGNORE INTO roots VALUES (?, ?)", (p, time.time()))
        return p

    def remove_root(self, folder: str) -> int:
        p = os.path.realpath(folder)
        with self.lock, self.con:
            self.con.execute("DELETE FROM roots WHERE path = ?", (p,))
            ids = [r["id"] for r in self.con.execute("SELECT id, path FROM files") if inside(r["path"], p)
                   and not any(inside(r["path"], b) for b in self.fixed.values())]
            for i in ids:
                self._drop(i)
        return len(ids)

    # ---- adding
    def _drop(self, fid):
        self.con.execute("DELETE FROM files WHERE id = ?", (fid,))
        if self.fts:
            self.con.execute("DELETE FROM files_fts WHERE rowid = ?", (fid,))

    def add(self, path: str, tags=()) -> int | None:
        """Index one file (again if it changed); tags are added to what it has. -> its id, or None (not allowed /
        not a picture, video or sound / gone)."""
        path = os.path.realpath(path)
        if not kind_of(path) or not os.path.isfile(path) or not self.allowed(path):
            return None
        st = os.stat(path)
        with self.lock:
            row = self.con.execute("SELECT * FROM files WHERE path = ?", (path,)).fetchone()
        old_tags = [t for t in (row["tags"] or "").split("\n") if t] if row else []
        new_tags = _uniq(old_tags + [t.strip() for t in tags if t and t.strip()])
        if row and row["size"] == st.st_size and abs((row["mtime"] or 0) - st.st_mtime) < 1e-3:
            if new_tags != old_tags:
                with self.lock, self.con:
                    self.con.execute("UPDATE files SET tags = ? WHERE id = ?", ("\n".join(new_tags), row["id"]))
                    self._fts(row["id"])
            return row["id"]
        d = describe(path)
        vals = dict(path=path, name=os.path.basename(path), kind=kind_of(path), size=st.st_size, mtime=st.st_mtime,
                    added=row["added"] if row else time.time(), w=d["w"] or d["width"], h=d["h"] or d["height"],
                    duration=d["duration"], positive=d["positive"], negative=d["negative"], seed=d["seed"],
                    steps=d["steps"], cfg=d["cfg"], sampler=d["sampler"], scheduler=d["scheduler"],
                    denoise=d["denoise"], models=json.dumps(d["models"]), loras=json.dumps(d["loras"]),
                    tags="\n".join(new_tags), has_workflow=int(d["has_workflow"]), source_app=d["source_app"])
        with self.lock, self.con:
            if row:
                self.con.execute("UPDATE files SET " + ", ".join(f"{k} = :{k}" for k in vals) + " WHERE id = :id",
                                 dict(vals, id=row["id"]))
                fid = row["id"]
            else:
                fid = self.con.execute(f"INSERT INTO files ({', '.join(vals)}) VALUES ({', '.join(':' + k for k in vals)})",
                                       vals).lastrowid
            self._fts(fid)
        return fid

    def _fts(self, fid):
        if not self.fts:
            return
        r = self.con.execute("SELECT * FROM files WHERE id = ?", (fid,)).fetchone()
        models = " ".join(json.loads(r["models"] or "[]") + [lo["name"] for lo in json.loads(r["loras"] or "[]")])
        self.con.execute("DELETE FROM files_fts WHERE rowid = ?", (fid,))
        self.con.execute("INSERT INTO files_fts (rowid, name, positive, negative, models, tags) VALUES (?,?,?,?,?,?)",
                         (fid, re.sub(r"[_\-.]", " ", r["name"] or ""), r["positive"] or "", r["negative"] or "",
                          re.sub(r"[_\-./\\]", " ", models), (r["tags"] or "").replace("\n", " ")))

    def add_folder(self, folder: str, tags=(), limit=FOLDER_LIMIT, progress=None) -> dict:
        """Every picture, video and sound under an allowed folder, indexed in place. Gone files of that folder are
        dropped. -> {added, total, capped}."""
        folder = os.path.realpath(folder)
        if not self.allowed(folder):
            raise ValueError("not an allowed folder")
        seen, n, capped = set(), 0, False
        for root, dirs, files in os.walk(folder):
            dirs[:] = sorted(d for d in dirs if not d.startswith("."))
            for f in sorted(files):
                if f.startswith(".") or not kind_of(f):
                    continue
                if len(seen) >= limit:
                    capped = True
                    break
                p = os.path.realpath(os.path.join(root, f))
                seen.add(os.path.normcase(p))
                try:
                    if self.add(p, tags) is not None:
                        n += 1
                except Exception:  # noqa: BLE001  (one bad file never stops the rest)
                    pass
                if progress and n % 50 == 0:
                    progress(n)
            if capped:
                break
        if not capped:
            with self.lock, self.con:
                for r in self.con.execute("SELECT id, path FROM files").fetchall():
                    if inside(r["path"], folder) and os.path.normcase(r["path"]) not in seen:
                        self._drop(r["id"])
        return {"added": n, "total": len(seen), "capped": capped}

    def prune(self) -> int:
        """Drop rows whose file is gone."""
        with self.lock, self.con:
            gone = [r["id"] for r in self.con.execute("SELECT id, path FROM files") if not os.path.isfile(r["path"])]
            for i in gone:
                self._drop(i)
        return len(gone)

    def tag(self, fid: int, tag: str, on: bool = True):
        with self.lock, self.con:
            r = self.con.execute("SELECT tags FROM files WHERE id = ?", (fid,)).fetchone()
            if not r:
                return
            tags = [t for t in (r["tags"] or "").split("\n") if t]
            tags = _uniq(tags + [tag]) if on else [t for t in tags if t != tag]
            self.con.execute("UPDATE files SET tags = ? WHERE id = ?", ("\n".join(tags), fid))
            self._fts(fid)

    # ---- reading
    def row(self, fid: int) -> dict | None:
        with self.lock:
            r = self.con.execute("SELECT * FROM files WHERE id = ?", (fid,)).fetchone()
        return self._out(r) if r else None

    def _out(self, r) -> dict:
        d = dict(r)
        d["models"] = json.loads(d["models"] or "[]")
        d["loras"] = json.loads(d["loras"] or "[]")
        d["tags"] = [t for t in (d["tags"] or "").split("\n") if t]
        d["source"] = self.source_of(d["path"])
        return d

    def search(self, q: str = "", kind: str = "", model: str = "", lora: str = "", tag: str = "",
               sort: str = "newest", limit: int = 200, offset: int = 0) -> dict:
        where, args = [], []
        q = str(q or "").strip()
        if q:
            if self.fts:
                terms = [t for t in re.findall(r"[\w#]+", q, re.UNICODE) if t]
                if terms:
                    where.append("id IN (SELECT rowid FROM files_fts WHERE files_fts MATCH ?)")
                    args.append(" ".join('"' + t.replace('"', "") + '"*' for t in terms))
            else:
                for t in q.split():
                    where.append("(name LIKE ? OR positive LIKE ? OR negative LIKE ? OR models LIKE ? OR loras LIKE ? "
                                 "OR tags LIKE ?)")
                    args += [f"%{t}%"] * 6
        if kind:
            where.append("kind = ?")
            args.append(kind)
        if model:
            where.append("EXISTS (SELECT 1 FROM json_each(files.models) WHERE value = ?)")
            args.append(model)
        if lora:
            where.append("EXISTS (SELECT 1 FROM json_each(files.loras) WHERE json_extract(value, '$.name') = ?)")
            args.append(lora)
        if tag:
            where.append("('\n' || tags || '\n') LIKE ?")
            args.append(f"%\n{tag}\n%")
        order = {"newest": "mtime DESC", "oldest": "mtime ASC", "name": "name COLLATE NOCASE ASC"}.get(sort, "mtime DESC")
        sql = "FROM files" + (" WHERE " + " AND ".join(where) if where else "")
        with self.lock:
            total = self.con.execute("SELECT COUNT(*) " + sql, args).fetchone()[0]
            rows = self.con.execute(f"SELECT * {sql} ORDER BY {order} LIMIT ? OFFSET ?",
                                    args + [int(limit), int(offset)]).fetchall()
        return {"total": total, "items": [self._out(r) for r in rows]}

    def facets(self) -> dict:
        """The pick lists: models, LoRAs and tags with how many files use them; the added folders; the count."""
        models, loras, tags = {}, {}, {}
        with self.lock:
            rows = self.con.execute("SELECT models, loras, tags FROM files").fetchall()
        for r in rows:
            for m in json.loads(r["models"] or "[]"):
                models[m] = models.get(m, 0) + 1
            for lo in json.loads(r["loras"] or "[]"):
                loras[lo["name"]] = loras.get(lo["name"], 0) + 1
            for t in (r["tags"] or "").split("\n"):
                if t:
                    tags[t] = tags.get(t, 0) + 1
        top = lambda d: [{"name": k, "count": v} for k, v in sorted(d.items(), key=lambda kv: (-kv[1], kv[0].lower()))]
        return {"count": len(rows), "models": top(models), "loras": top(loras), "tags": top(tags), "roots": self.roots(),
                "fts": self.fts}
