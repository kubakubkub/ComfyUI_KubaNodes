"""
versions.py

The versions sheet of Kuba Regions: a short text that lists what may change
between versions of one facade (looks, ideas per region, prompt rotation,
LoRAs, style references, seeds, any plan setting), and the list of versions
that comes out of it. A version is nothing but extra rules appended to the
region plan, plus a LoRA, a style reference and a seed.

    // one option per line; 'short name = text' gives the option a label
    [look]
    copper = oxidised copper plates, verdigris, rivets
    paper = folded white paper, soft daylight

    [region windows]                  // any plan selector: name, group:, tag:, region:
    warm rooms, people moving inside
    aquariums, fish, blue light

    [rotate windows | roof | wall]    // prompt rotation: the ideas move on by one region per version
    molten glass
    woven textile
    moss and ivy

    [lora]
    none
    My_Style_v1 : 0.8 : my style                      // name : strength : words added to the prompt
    My_Style_v1 : 1.2 + Detail_v2 : 0.5               // a chain: several LoRAs on one version

    [reference]                       // style reference images: a file, or a folder (every image in it)
    none
    refs/styles

    [image]                           // the picture a version starts from: a file, a folder, or 'input'
    input                             // the image connected to the node
    renders/facade_wireframe.png
    renders/facade_canny.png

    [set denoise]                     // any plan setting; '[set denoise windows]' for some regions only
    0.7
    0.9

    [seed]
    1
    2

Modes: rotate (version n takes option n of every list, short lists start over),
combine (every combination), one by one (the first option of everything, then
one change at a time).

Pure logic, no ComfyUI imports (tests/test_versions.py).
"""

from __future__ import annotations

import itertools
import os
import re
from dataclasses import dataclass, field

MODES = ("rotate", "combine", "one by one")
IMAGE_EXT = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff")
NOTHING = ("", "-", "none", "off")

EXAMPLE_SHEET = """// one option per line; 'short name = text' gives it a label
[look]
copper = oxidised copper plates, verdigris, rivets
paper = folded white paper, soft daylight
moss = overgrown with moss and ivy, wet stone

[region windows]
warm rooms, people moving inside
aquariums, fish, blue light

[lora]
none
"""


class SheetError(ValueError):
    pass


@dataclass
class Option:
    label: str
    text: str = ""          # look / region / set: the value; rotate: unused
    rules: str = ""         # rule text this option appends to the plan
    lora: str = ""          # lora axis: name as written
    strength: float = 1.0
    chain: tuple = ()       # lora axis: ((name, strength), ...) in the order they are applied
    reference: str = ""     # reference axis: file path, or 'input:<n>' for the node's image input
    image: str = ""         # image axis: file path, or 'input:<n>' for the node's image input
    seed: int = 0


@dataclass
class Axis:
    kind: str               # look | region | rotate | lora | reference | set | seed
    title: str              # what the report calls it
    options: list[Option] = field(default_factory=list)


@dataclass
class Version:
    number: int             # 1-based, stable for a sheet + mode: drafts and finals share it
    label: str
    rules: str
    lora: str = ""
    strength: float = 1.0
    chain: tuple = ()       # ((lora name, strength), ...)
    reference: str = ""
    seed: int = 0
    choice: tuple = ()      # option index per axis
    image: str = ""         # the image the version starts from; '' = the first image of the node's input


def _strip_comment(line: str) -> str:
    return re.split(r"(?<!:)//", line, maxsplit=1)[0]


def _short(text: str, n: int = 22) -> str:
    text = " ".join(text.split())
    if len(text) <= n:
        return text
    cut = text[:n].rsplit(" ", 1)[0] if " " in text[:n] else text[:n]
    return cut.rstrip(" ,;") + "…"


_NAMED = re.compile(r"^([\w][\w \-]{0,23}?)\s+=\s+(.+)$")


def _named(line: str) -> tuple[str, str]:
    """'copper = oxidised copper plates' -> ('copper', 'oxidised copper plates'); else a short label."""
    m = _NAMED.match(line)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    return ("plain" if line.lower() in NOTHING else _short(line)), line


def _rule(selector: str, key: str, op: str, value: str) -> str:
    return f"[{selector}]\n{key} {op} {value}\n"


def _reference_options(line: str, line_no: int, field: str = "reference") -> list[Option]:
    """The options of one line of [reference] or [image] (field = which of the two): a file, or a folder."""
    label, value = _named(line)
    if field == "image" and value.lower() in NOTHING + ("input",):
        return [Option(label if _NAMED.match(line) else "input", image="input:0")]
    if value.lower() in NOTHING:
        return [Option("no ref", reference="")]
    # 'folder | 12' = 12 images spread evenly over the folder; 'folder | 12 | 5' = the same, starting 5 further
    path, *pick = [p.strip() for p in value.split("|")]
    path = path.strip('"')
    if os.path.isdir(path):
        files = sorted(f for f in os.listdir(path) if f.lower().endswith(IMAGE_EXT))
        if not files:
            raise SheetError(f"line {line_no}: no images in the folder {path}")
        if pick:
            try:
                n, shift = int(pick[0]), int(pick[1]) if len(pick) > 1 else 0
            except ValueError:
                raise SheetError(f"line {line_no}: after '|' write how many images (and a shift): "
                                 f"folder | 12 | 5") from None
            n = max(1, min(n, len(files)))
            files = [files[(shift + i * len(files) // n) % len(files)] for i in range(n)]
        return [Option(_short(os.path.splitext(f)[0], 18), **{field: os.path.join(path, f)}) for f in files]
    if not os.path.isfile(path):
        raise SheetError(f"line {line_no}: {field} '{path}' is neither a file nor a folder")
    named = _NAMED.match(line) is not None
    return [Option(label if named else _short(os.path.splitext(os.path.basename(path))[0], 18), **{field: path})]


def _lora_option(line: str, line_no: int) -> Option:
    """'name : strength : words', or a chain of them joined with ' + ' (applied one after the other)."""
    if line.lower() in NOTHING:
        return Option("no lora")
    m = _NAMED.match(line)                                # 'pair a = Style_v1 : 1.2 + Detail_v2 : 0.5'
    if m and ":" not in m.group(1):
        opt = _lora_option(m.group(2).strip(), line_no)
        opt.label = m.group(1).strip()
        return opt
    links =[_lora_link(part.strip(), line_no) for part in line.split(" + ") if part.strip()]
    if len(links) == 1:
        return links[0]
    return Option(" + ".join(o.label for o in links), lora=" + ".join(o.lora for o in links),
                  strength=links[0].strength, rules="".join(o.rules for o in links),
                  chain=tuple(c for o in links for c in o.chain))


def _lora_link(line: str, line_no: int) -> Option:
    parts = [p.strip() for p in line.split(":")]
    if len(parts) > 1 and len(parts[0]) == 1 and parts[1][:1] in "\\/":      # a path with a drive letter
        parts = [parts[0] + ":" + parts[1]] + parts[2:]
    name = parts[0]
    strength, words = 1.0, ""
    if len(parts) > 1 and parts[1]:
        try:
            strength = float(parts[1])
        except ValueError:
            raise SheetError(f"line {line_no}: lora strength '{parts[1]}' is not a number "
                             f"(name : strength : words for the prompt)") from None
    if len(parts) > 2:
        words = ":".join(parts[2:]).strip()
    base = os.path.splitext(os.path.basename(name.replace("\\", "/")))[0]
    if len(base) > 22:                                   # keep the end: checkpoints differ in their last digits
        base = base[:9] + "…" + base[-12:]
    label = f"{base} {strength:g}"
    return Option(label, lora=name, strength=strength, chain=((name, strength),),
                  rules=_rule("default", "prompt", "+=", words) if words else "")


def parse_sheet(text: str, input_references: int = 0, input_images: int = 0) -> list[Axis]:
    """The sheet as a list of axes. input_references: images on the node's references input (an extra
    reference axis, or more options of the sheet's own [reference]). input_images: the same for a batch of
    several images on the node's image input and [image]."""
    axes: list[Axis] = []
    cur: Axis | None = None
    targets: list[str] = []
    ideas: list[tuple[str, str]] = []
    set_key = set_sel = ""

    def close():
        nonlocal ideas
        if cur is not None and cur.kind == "rotate" and not cur.options:
            if len(ideas) < 2:
                raise SheetError(f"[{cur.title}] needs at least two ideas to rotate")
            for k in range(len(ideas)):
                rules = "".join(_rule(sel, "prompt", "+=", ideas[(j + k) % len(ideas)][1])
                                for j, sel in enumerate(targets) if ideas[(j + k) % len(ideas)][1].lower() not in NOTHING)
                cur.options.append(Option(f"{_short(targets[0], 12)}: {ideas[k][0]}", rules=rules))
            ideas = []

    for n, raw in enumerate((text or "").splitlines(), 1):
        line = _strip_comment(raw).strip()
        if not line:
            continue
        if line.startswith("["):
            if not line.endswith("]"):
                raise SheetError(f"line {n}: '{line}' is missing its ']'")
            close()
            head = line[1:-1].strip()
            kind, _, rest = head.partition(" ")
            kind, rest = kind.lower(), rest.strip()
            if kind in ("look", "looks"):
                cur = Axis("look", "look")
            elif kind in ("region", "regions"):
                if not rest:
                    raise SheetError(f"line {n}: [region ...] needs a selector, e.g. [region windows]")
                cur = Axis("region", rest)
                set_sel = rest
            elif kind == "rotate":
                targets = [t.strip() for t in rest.split("|") if t.strip()]
                if len(targets) < 2:
                    raise SheetError(f"line {n}: [rotate a | b | c] needs two or more region selectors")
                cur = Axis("rotate", "rotate " + " | ".join(targets))
            elif kind in ("lora", "loras"):
                cur = Axis("lora", "lora")
            elif kind in ("reference", "references", "style"):
                cur = Axis("reference", "reference")
            elif kind in ("image", "images", "start"):
                cur = Axis("image", "image")
            elif kind in ("seed", "seeds"):
                cur = Axis("seed", "seed")
            elif kind == "set":
                set_key, _, set_sel = rest.partition(" ")
                if not set_key:
                    raise SheetError(f"line {n}: [set ...] needs a plan setting, e.g. [set denoise]")
                set_sel = set_sel.strip() or "default"
                cur = Axis("set", f"{set_key} {set_sel}" if set_sel != "default" else set_key)
            else:
                raise SheetError(f"line {n}: unknown list [{head}]. Use look, region <selector>, "
                                 f"rotate a | b, lora, reference, image, set <setting>, seed.")
            axes.append(cur)
            continue
        if cur is None:                                   # lines before any header are looks
            cur = Axis("look", "look")
            axes.append(cur)
        if cur.kind == "look":
            label, value = _named(line)
            cur.options.append(Option(label, value, "" if value.lower() in NOTHING
                                      else _rule("default", "prompt", "+=", value)))
        elif cur.kind == "region":
            label, value = _named(line)
            cur.options.append(Option(label, value, "" if value.lower() in NOTHING
                                      else _rule(set_sel, "prompt", "+=", value)))
        elif cur.kind == "rotate":
            ideas.append(_named(line))
        elif cur.kind == "lora":
            cur.options.append(_lora_option(line, n))
        elif cur.kind in ("reference", "image"):
            cur.options.extend(_reference_options(line, n, cur.kind))
        elif cur.kind == "seed":
            try:
                cur.options.append(Option(f"seed {int(line)}", seed=int(line)))
            except ValueError:
                raise SheetError(f"line {n}: seed '{line}' is not a whole number") from None
        elif cur.kind == "set":
            cur.options.append(Option(f"{set_key} {line}", line, _rule(set_sel, set_key, "=", line)))
    close()

    if input_references > 0:
        ref = next((a for a in axes if a.kind == "reference"), None)
        if ref is None:
            ref = Axis("reference", "reference")
            axes.append(ref)
        ref.options.extend(Option(f"ref {i + 1}", reference=f"input:{i}") for i in range(input_references))
    if input_images > 1:
        img = next((a for a in axes if a.kind == "image"), None)
        if img is None:
            img = Axis("image", "image")
            axes.append(img)
        # 'input' in the sheet is the first one already
        have = {o.image for o in img.options}
        img.options.extend(Option(f"image {i + 1}", image=f"input:{i}") for i in range(input_images)
                           if f"input:{i}" not in have)
    for a in axes:
        if not a.options:
            raise SheetError(f"[{a.title}] has no options (one per line)")
    return axes


def _choices(axes: list[Axis], mode: str, count: int) -> list[tuple]:
    sizes = [len(a.options) for a in axes]
    if not axes:
        return [()]
    if mode == "combine":
        return list(itertools.product(*(range(s) for s in sizes)))
    if mode == "one by one":
        out = [tuple(0 for _ in sizes)]
        for i, s in enumerate(sizes):
            out += [tuple(k if j == i else 0 for j in range(len(sizes))) for k in range(1, s)]
        return out
    n = count if count > 0 else max(sizes)
    return [tuple(i % s for s in sizes) for i in range(n)]


def build(text: str, mode: str = "rotate", count: int = 0, max_versions: int = 0,
          input_references: int = 0, input_images: int = 0) -> tuple[list[Version], list[Axis], int]:
    """(versions, axes, how many were left out by max_versions)."""
    if mode not in MODES:
        raise SheetError(f"unknown mode '{mode}' (use {', '.join(MODES)})")
    axes = parse_sheet(text, input_references, input_images)
    choices = _choices(axes, mode, count)
    left_out = 0
    if max_versions > 0 and len(choices) > max_versions:
        left_out = len(choices) - max_versions
        choices = choices[:max_versions]
    varying = [i for i, a in enumerate(axes) if len(a.options) > 1]
    versions = []
    for n, ch in enumerate(choices, 1):
        opts = [axes[i].options[k] for i, k in enumerate(ch)]
        v = Version(n, " · ".join(opts[i].label for i in varying) or "version", "".join(o.rules for o in opts),
                    choice=ch)
        for a, o in zip(axes, opts):
            if a.kind == "lora":
                v.lora, v.strength, v.chain = o.lora, o.strength, o.chain
            elif a.kind == "reference":
                v.reference = o.reference
            elif a.kind == "image":
                v.image = o.image
            elif a.kind == "seed":
                v.seed = o.seed
        if v.reference:
            v.rules += _rule("default", "reference", "=", "style")
        versions.append(v)
    return versions, axes, left_out


def parse_pick(text: str, n: int) -> list[int]:
    """'3, 7-9' -> [3, 7, 8, 9] (1-based version numbers); empty = all."""
    text = (text or "").strip()
    if not text:
        return list(range(1, n + 1))
    out = []
    for tok in re.split(r"[,\s;]+", text):
        if not tok:
            continue
        m = re.fullmatch(r"(\d+)(?:-(\d+))?", tok)
        if not m:
            raise SheetError(f"pick: '{tok}' is not a version number or a range like 3-5")
        a, b = int(m.group(1)), int(m.group(2) or m.group(1))
        for k in range(min(a, b), max(a, b) + 1):
            if not 1 <= k <= n:
                raise SheetError(f"pick: version {k} does not exist (the sheet gives 1-{n})")
            if k not in out:
                out.append(k)
    return out


def only_regions(plan: dict, only: str) -> tuple[dict, int]:
    """(plan, how many regions are still painted) where every region the selectors of `only` do not match is
    set to keep ('W_F1_*, group:M_Pilasters': plan selector syntax, space = AND, comma = OR). Empty = the plan
    as it is. The input is not changed."""
    from . import plan as rp
    only = (only or "").strip()
    if not only:
        return plan, len(plan["order"])
    try:
        section = rp.parse_rules(f"[{only}]\n")[0]
    except rp.PlanError as e:
        raise SheetError(f"only: {e}") from None
    hit = {e["region_id"] for e in plan["regions"] if section.match(e) is not None}
    entries = [e if e["region_id"] in hit else {**e, "strategy": "keep"} for e in plan["regions"]]
    order = [i for i in plan["order"] if i in hit]
    return {**plan, "regions": entries, "order": order}, len(order)


def match_lora(name: str, available: list[str]) -> str:
    """The one file of `available` (names relative to the loras folders) that `name` means: the exact name,
    the file name without folder / extension, or a part of it when that is unambiguous."""
    want = name.replace("\\", "/").lower()
    norm = [(a, a.replace("\\", "/").lower()) for a in available]
    for a, low in norm:
        if low == want:
            return a
    stem = os.path.splitext(want)[0].rsplit("/", 1)[-1]
    exact = [a for a, low in norm if os.path.splitext(low)[0].rsplit("/", 1)[-1] == stem]
    if len(exact) == 1:
        return exact[0]
    part = exact or [a for a, low in norm if want in low]
    if len(part) == 1:
        return part[0]
    if not part:
        import difflib
        near = difflib.get_close_matches(stem, [os.path.splitext(low)[0].rsplit("/", 1)[-1] for _, low in norm], n=3,
                                         cutoff=0.4)
        raise SheetError(f"lora '{name}' not found" + (f"; close: {', '.join(near)}" if near else ""))
    raise SheetError(f"lora '{name}' fits {len(part)} files, write more of the name: "
                     + ", ".join(part[:6]) + (" ..." if len(part) > 6 else ""))


def report(versions: list[Version], axes: list[Axis], mode: str, left_out: int = 0, picked=None) -> str:
    lines = [f"{len(versions)} version(s), mode {mode}"
             + (f", {left_out} more left out by max_versions" if left_out else "")]
    for a in axes:
        lines.append(f"  {a.title}: {len(a.options)} option(s): " + ", ".join(o.label for o in a.options))
    lines.append("")
    for v in versions:
        mark = "" if picked is None else ("* " if v.number in picked else "  ")
        lines.append(f"{mark}{v.number:02d}  {v.label}")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# picks folder: a draft file knows which version it is
# --------------------------------------------------------------------------

RECIPE_KEY = "kubakub_version"          # text chunk of a PNG the node saved


def recipe(v: Version, base_rules: str | None, plan_seed: int | None, offset: int) -> dict:
    """What a final needs to know about a draft; stored in the draft's PNG. base_rules / plan_seed None =
    not known (a draft written by Save Image): the connected plan's are used."""
    return {"number": v.number, "label": v.label, "base_rules": base_rules, "rules": v.rules,
            "plan_seed": plan_seed, "offset": int(offset), "chain": [[n, float(s)] for n, s in v.chain],
            "reference": v.reference, "image": v.image}


def slug(label: str, n: int = 70) -> str:
    """'isometric · soft pair' -> 'isometric_soft-pair' (a file name)."""
    parts = [re.sub(r"[^\w\-]+", "-", p.strip()).strip("-") for p in label.split("·")]
    return "_".join(p for p in parts if p)[:n].rstrip("-_") or "version"


def next_run(names) -> int:
    """Drafts are saved as r<run>_v<number>_<label>.png: the run number after the highest one in the folder."""
    runs = [int(m.group(1)) for m in (re.match(r"r(\d+)_v\d+", n) for n in names) if m]
    return max(runs, default=0) + 1


def pick_files(folder: str) -> list[str]:
    """The images of a picks folder (not of its subfolders), by name."""
    if not os.path.isdir(folder):
        raise SheetError(f"picks_folder: '{folder}' is not a folder")
    files = sorted(f for f in os.listdir(folder)
                   if f.lower().endswith(IMAGE_EXT) and os.path.isfile(os.path.join(folder, f)))
    if not files:
        raise SheetError(f"picks_folder: no images in {folder}")
    return [os.path.join(folder, f) for f in files]


def _chunks(path: str) -> dict:
    from PIL import Image
    try:
        with Image.open(path) as im:
            return {k: v for k, v in im.info.items() if isinstance(v, str)}
    except Exception:
        return {}


_SAVED = re.compile(r"^(.*_)(\d{5})(_\.png)$", re.I)       # Save Image: <prefix>_00032_.png


def _saved_recipe(text: str, path: str, output_dir: str) -> dict | None:
    """A draft written by core's Save Image: every file of a run holds the same workflow, so the file's place
    among its neighbours in the folder it was saved to is its place among the versions of that run."""
    import json
    try:
        graph = json.loads(text)
        nid, node = next((k, n) for k, n in graph.items() if n.get("class_type") == "KUBA_Versions")
        i = node["inputs"]
        m = _SAVED.match(os.path.basename(path))
        prefix = next(n["inputs"]["filename_prefix"] for n in graph.values()
                      if n.get("class_type") == "SaveImage" and n["inputs"].get("images") == [nid, 0])
        # a connected reference batch changes the list of versions, and its size is not in the file
        if m is None or isinstance(i.get("references"), list) or os.path.basename(prefix) + "_" != m.group(1):
            return None
        folder = os.path.join(output_dir, os.path.dirname(prefix))

        def same_run(k):
            p = os.path.join(folder, f"{m.group(1)}{k:05d}{m.group(3)}")
            return os.path.isfile(p) and _chunks(p).get("prompt") == text

        counter = int(m.group(2))
        if not same_run(counter):
            return None
        index = 0
        while same_run(counter - index - 1):
            index += 1
        versions, _, _ = build(i["versions"], i["mode"], int(i.get("count", 0)), int(i["max_versions"]), 0)
        v = versions[sorted(parse_pick(i.get("pick", ""), len(versions)))[index] - 1]
        return recipe(v, None, None, int(i.get("seed", 0)) + v.seed)
    except (ValueError, KeyError, TypeError, IndexError, StopIteration, AttributeError):
        return None


def read_recipe(path: str, output_dir: str = "") -> dict | None:
    """The version a draft file is: from the chunk the node wrote, else worked out for a draft written by
    Save Image (needs the run's other files where they were saved). None = the file does not say."""
    import json
    info = _chunks(path)
    if RECIPE_KEY in info:
        try:
            return json.loads(info[RECIPE_KEY])
        except ValueError:
            return None
    return _saved_recipe(info["prompt"], path, output_dir) if "prompt" in info else None


# --------------------------------------------------------------------------
# drafts: regions that get the same treatment are sampled together
# --------------------------------------------------------------------------

_OWN_KEYS = ("region_id", "name", "group_id", "bbox", "area", "tags", "rules", "seed")


def merge_same(plan: dict, regions):
    """
    (plan dict, Regions) where every set of regions with the same resolved settings (prompt, denoise,
    strategy ...) is one region: one sample instead of one per region. For drafts; a final keeps the regions
    apart, so each gets its own resolution. Frame in frame regions stay on their own (each is its own scene).
    """
    import torch
    from .types import Regions

    entries = plan["regions"]
    groups: dict = {}
    for e in entries:
        key = tuple((k, e[k]) for k in sorted(e) if k not in _OWN_KEYS)
        if e["strategy"] == "frame_in_frame":
            key += (("own", e["region_id"]),)
        groups.setdefault(key, []).append(e)
    if len(groups) == len(entries):
        return plan, regions
    rank = {rid: i for i, rid in enumerate(plan["order"])}
    members = sorted(groups.values(), key=lambda m: min(rank.get(e["region_id"], len(rank)) for e in m))
    lut = torch.full((len(entries),), -1, dtype=regions.labels.dtype)
    new_entries, new_rows = [], []
    for gid, m in enumerate(members):
        for e in m:
            lut[e["region_id"]] = gid
        x0 = min(e["bbox"][0] for e in m)
        y0 = min(e["bbox"][1] for e in m)
        x1 = max(e["bbox"][0] + e["bbox"][2] for e in m)
        y1 = max(e["bbox"][1] + e["bbox"][3] for e in m)
        name = m[0]["name"] if len(m) == 1 else f"{m[0]['name']} +{len(m) - 1}"
        own = {"region_id": gid, "name": name, "group_id": m[0]["group_id"], "bbox": [x0, y0, x1 - x0, y1 - y0],
               "area": sum(e["area"] or 0 for e in m), "tags": sorted({t for e in m for t in e["tags"]})}
        new_entries.append({**m[0], **own, "rules": m[0]["rules"], "seed": min(e["seed"] for e in m)})
        new_rows.append(dict(own))
    lab = regions.labels
    merged = torch.where(lab >= 0, lut[lab.clamp_min(0).long()], lab)
    table = {**regions.table, "regions": new_rows}
    order = [e["region_id"] for e in new_entries if e["strategy"] != "keep"]
    return {**plan, "regions": new_entries, "order": order}, Regions(merged, table, regions.scope)


# --------------------------------------------------------------------------
# contact sheet
# --------------------------------------------------------------------------

def sheet_layout(n: int, cell_w: int, cell_h: int, columns: int = 0, max_width: int = 4096,
                 gap: int = 12) -> tuple[int, int, float]:
    """(columns, rows, scale of a cell) so the sheet is at most max_width wide."""
    import math
    if columns <= 0:
        columns = max(1, math.ceil(math.sqrt(n * cell_h / max(cell_w, 1) * 1.4)))
    columns = min(columns, n)
    rows = math.ceil(n / columns)
    scale = min(1.0, (max_width - gap * (columns + 1)) / (columns * cell_w))
    return columns, rows, scale


def contact_sheet(images, labels: list[str], columns: int = 0, max_width: int = 4096):
    """images: list of float32 numpy [H,W,3] in 0..1 (same size). Returns float32 [H,W,3]: a grid with the
    version number and label under every image."""
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    gap = 12
    h0, w0 = images[0].shape[:2]
    cols, rows, scale = sheet_layout(len(images), w0, h0, columns, max_width, gap)
    cw, ch = max(8, int(w0 * scale)), max(8, int(h0 * scale))
    font_px = max(14, cw // 34)
    try:
        font = ImageFont.load_default(size=font_px)
    except TypeError:                                   # older Pillow
        font = ImageFont.load_default()
    bar = int(font_px * 1.9)
    W = gap + cols * (cw + gap)
    H = gap + rows * (ch + bar + gap)
    sheet = Image.new("RGB", (W, H), (22, 22, 22))
    draw = ImageDraw.Draw(sheet)
    for i, (img, label) in enumerate(zip(images, labels)):
        x = gap + (i % cols) * (cw + gap)
        y = gap + (i // cols) * (ch + bar + gap)
        tile = Image.fromarray((np.clip(img, 0, 1) * 255 + 0.5).astype("uint8"))
        if tile.size != (cw, ch):
            tile = tile.resize((cw, ch), Image.LANCZOS)
        sheet.paste(tile, (x, y))
        text = label
        while len(text) > 4 and draw.textlength(text, font=font) > cw:
            text = text[:-2].rstrip() + "…"
        draw.text((x, y + ch + int(font_px * 0.3)), text, fill=(241, 138, 88), font=font)
    return np.asarray(sheet, dtype=np.float32) / 255.0
