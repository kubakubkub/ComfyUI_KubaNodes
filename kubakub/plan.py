"""
plan.py

The region plan of Kuba Regions: rule text
in, one fully resolved settings dict per region out. Pure logic, no ComfyUI
imports, so tests/test_plan.py runs without starting ComfyUI.

Rule text (INI-like, one section per rule):

    // comment
    [plan]
    order = largest_first

    [default]
    prompt = weathered sandstone facade at night
    denoise = 0.55

    [group:M_Pilasters]
    prompt += carved vines climbing the stone

    [W_F1_*]                          // bare selector = region name, wildcards allowed
    strategy = frame_in_frame
    prompt = an underwater room, {jellyfish|a sunken piano|koi}

    [tag:M_FLOOR_F3 group:Windows]    // space = AND
    denoise = 0.8

    [region:12, region:30-33, !W_F0_L] // comma = OR, ! = NOT (inside an AND term)
    strategy = keep

Selectors: `default` / `*` (all), `name:<wildcard>` (or a bare word), `group:`,
`tag:`, `region:<id>` or `region:<a>-<b>`. Matching is case insensitive.
Priority: a more specific rule beats a less specific one (region > name >
group > tag > default), an AND of several terms beats a single term of the same
kind, and among equals the later section wins. Settings merge key by key.

Values: `prompt = {prompt}, more` or `prompt += more` builds on the inherited
prompt. `{name}`, `{group}`, `{id}` are filled in. `{a|b|c}` picks one option
per region from the plan seed, so a row of windows can tell different stories.
"""

from __future__ import annotations

import difflib
import fnmatch
import random
import re
from dataclasses import dataclass, field

PLAN_FORMAT = "kubakub.regions.plan"
PLAN_VERSION = 1

STRATEGIES = ("inpaint", "frame_in_frame", "keep")
ORDERS = ("plan", "reading", "largest_first", "smallest_first")
COLOR_MATCH = ("none", "mean_std", "mkl")

# key -> (type, default, help)
SETTINGS: dict[str, tuple[type | tuple, object, str]] = {
    "prompt": (str, "", "positive prompt"),
    "negative": (str, "", "negative prompt (ignored by distilled cfg 1 models)"),
    "strategy": (STRATEGIES, "inpaint", "inpaint = repaint in place (S1), frame_in_frame = own "
                                        "scene set into the region (S4), keep = leave untouched"),
    "denoise": (float, 0.6, "0..1, how much the region may change"),
    "steps": (int, 0, "0 = backend default"),
    "cfg": (float, 0.0, "0 = backend default"),
    "seed": (int, -1, "-1 = plan seed + region id"),
    "order": (int, 0, "lower runs earlier (plan order)"),
    "context_px": (int, 64, "canvas around the region the model sees, at target scale"),
    "dilate_px": (int, 4, "grow the inpaint mask by this much before sampling"),
    "feather_px": (int, 8, "soft edge when pasting back"),
    "color_match": (COLOR_MATCH, "mean_std", "match the region's colours to its context"),
    "blend": (bool, True, "include this region's borders in the seam pass"),
    "reference": (str, "self", "none | self | style (the style image of kubakub versions): reference for the model"),
    "lora": (str, "", "name:strength; name:strength (hook LoRA, M2)"),
    "region_mp": (float, 0.0, "work megapixels for the region crop, 0 = backend default"),
    "z": (int, 0, "stacking for frame in frame"),
    "fif_width": (int, 0, "frame in frame: sub-generation width, 0 = from the region box"),
    "fif_height": (int, 0, "frame in frame: sub-generation height, 0 = from the region box"),
    "fif_harmonize": (float, 0.35, "frame in frame: denoise of the border harmonizing pass, 0 = off"),
    "fif_border_px": (int, 24, "frame in frame: width of the harmonized border"),
    "fif_depth_m": (float, 0.0, "frame in frame 2: metres behind (> 0, a room / world) or in front (< 0, a "
                                "parasite) of the facade, as the viewer sees it; 0 = not a frame"),
    "fif_source": (("generate", "media", "world"), "generate",
                   "frame in frame 2: generate = perspective guide for the sampler, media = an image of Frame "
                   "Compose's media input on the back wall, world = the shared world image behind the facade"),
    "fif_media": (str, "", "frame in frame 2: which media (a name from media_names or an index); empty = in turn"),
    "animate": (bool, False, "video: this region moves in the Region Video Sampler"),
    "video_prompt": (str, "", "video: what happens in the region (empty = prompt)"),
    "t_start": (float, 0.0, "video: seconds from which the region moves"),
    "t_end": (float, -1.0, "video: seconds until which it moves; -1 = to the end"),
    "motion": (float, 1.0, "video: 0..1, how much the region may change over time"),
}
ALIASES = {"negative_prompt": "negative", "neg": "negative", "positive": "prompt",
           "context": "context_px", "dilate": "dilate_px", "feather": "feather_px"}
PLAN_KEYS = {"order": (ORDERS, "plan", "processing order of the regions")}

_KIND_RANK = {"default": 0, "tag": 1, "group": 2, "name": 3, "region": 4}


class PlanError(ValueError):
    pass


# --------------------------------------------------------------------------
# parsing
# --------------------------------------------------------------------------

@dataclass
class Term:
    kind: str           # default | name | group | tag | region
    value: str          # lower case pattern, or "a-b" for region
    negate: bool = False

    def matches(self, r: dict) -> bool:
        hit = self._hit(r)
        return not hit if self.negate else hit

    def _hit(self, r: dict) -> bool:
        if self.kind == "default":
            return True
        if self.kind == "region":
            a, b = self.value.split("-") if "-" in self.value else (self.value, self.value)
            return int(a) <= r["region_id"] <= int(b)
        if self.kind == "name":
            return fnmatch.fnmatchcase(str(r.get("name", "")).lower(), self.value)
        if self.kind == "group":
            return fnmatch.fnmatchcase(str(r.get("group_id", "")).lower(), self.value)
        if self.kind == "tag":
            return any(fnmatch.fnmatchcase(t.lower(), self.value) for t in r.get("tags", []))
        return False


@dataclass
class Section:
    header: str
    line: int
    alternatives: list[list[Term]]          # OR of ANDs
    settings: list[tuple[str, str, str, int]] = field(default_factory=list)  # key, op, raw value, line
    is_plan: bool = False

    def match(self, r: dict):
        """Specificity of the best matching alternative, or None."""
        best = None
        for terms in self.alternatives:
            if all(t.matches(r) for t in terms):
                pos = [t for t in terms if not t.negate] or [Term("default", "*")]
                spec = (max(_KIND_RANK[t.kind] for t in pos), len(pos))
                best = spec if best is None or spec > best else best
        return best


def _parse_term(tok: str, line: int) -> Term:
    neg = tok.startswith("!")
    tok = tok[1:] if neg else tok
    if not tok:
        raise PlanError(f"line {line}: empty selector after '!'")
    low = tok.lower()
    if low in ("default", "*", "all"):
        return Term("default", "*", neg)
    kind, _, val = low.partition(":")
    if not _:
        return Term("name", low, neg)
    if kind in ("id", "region"):
        kind = "region"
        if not re.fullmatch(r"\d+(-\d+)?", val):
            raise PlanError(f"line {line}: region selector '{tok}' needs a number or a range like region:3-7")
    elif kind not in ("name", "group", "tag"):
        raise PlanError(f"line {line}: unknown selector kind '{kind}' in '{tok}'. "
                        f"Use name:, group:, tag:, region: or a bare name.")
    if not val:
        raise PlanError(f"line {line}: '{tok}' has no value")
    return Term(kind, val, neg)


def parse_rules(text: str) -> list[Section]:
    sections: list[Section] = []
    cur: Section | None = None
    for n, raw in enumerate((text or "").splitlines(), 1):
        line = _strip_comment(raw).strip()
        if not line:
            continue
        if line.startswith("["):
            if not line.endswith("]"):
                raise PlanError(f"line {n}: section header '{line}' is missing its ']'")
            header = line[1:-1].strip()
            if header.lower() == "plan":
                cur = Section(header, n, [], is_plan=True)
            else:
                alts = []
                for alt in header.split(","):
                    toks = alt.split()
                    if not toks:
                        raise PlanError(f"line {n}: empty selector in [{header}]")
                    alts.append([_parse_term(t, n) for t in toks])
                cur = Section(header, n, alts)
            sections.append(cur)
            continue
        m = re.match(r"^([A-Za-z_][\w]*)\s*(\+?=)\s*(.*)$", line)
        if not m:
            raise PlanError(f"line {n}: expected 'key = value' or a [selector] header, got '{line}'")
        if cur is None:
            cur = Section("default", n, [[Term("default", "*")]])
            sections.append(cur)
        key, op, val = m.group(1).lower(), m.group(2), m.group(3).strip()
        key = ALIASES.get(key, key)
        known = PLAN_KEYS if cur.is_plan else SETTINGS
        if key not in known:
            hint = difflib.get_close_matches(key, list(known), n=1)
            raise PlanError(f"line {n}: unknown setting '{key}'"
                            + (f"; did you mean '{hint[0]}'?" if hint else f". Known: {', '.join(known)}"))
        if op == "+=" and key not in ("prompt", "negative", "lora", "video_prompt"):
            raise PlanError(f"line {n}: '+=' only works for prompt, negative and lora")
        cur.settings.append((key, op, val, n))
    return sections


def _strip_comment(line: str) -> str:
    # '//' starts a comment unless it is part of a URL-like 'x://'
    return re.split(r"(?<!:)//", line, maxsplit=1)[0]


def _convert(key: str, raw: str, line: int, table=SETTINGS):
    typ = table[key][0]
    v = raw.strip()
    if len(v) >= 2 and v[0] == v[-1] and v[0] in "\"'":
        v = v[1:-1]
    try:
        if isinstance(typ, tuple):
            low = v.lower()
            if key == "strategy" and low in ("skip", "none", "off"):
                low = "keep"
            if low not in typ:
                raise ValueError(f"must be one of {', '.join(typ)}")
            return low
        if typ is bool:
            low = v.lower()
            if low in ("1", "true", "on", "yes"):
                return True
            if low in ("0", "false", "off", "no"):
                return False
            raise ValueError("must be on/off")
        if typ is int:
            return int(float(v))
        if typ is float:
            return float(v)
        return v
    except ValueError as e:
        raise PlanError(f"line {line}: {key} = {raw!r}: {e}") from None


# --------------------------------------------------------------------------
# resolving
# --------------------------------------------------------------------------

_CHOICE = re.compile(r"\{([^{}]*\|[^{}]*)\}")


def _fill(text: str, r: dict, inherited: str, rng: random.Random) -> str:
    text = text.replace("{prompt}", inherited).replace("{negative}", inherited)
    text = (text.replace("{name}", str(r.get("name", "")))
                .replace("{group}", str(r.get("group_id", "")))
                .replace("{id}", str(r["region_id"])))
    while True:
        m = _CHOICE.search(text)
        if not m:
            break
        text = text[:m.start()] + rng.choice(m.group(1).split("|")).strip() + text[m.end():]
    return text


def _join(a: str, b: str) -> str:
    a, b = a.strip().rstrip(","), b.strip()
    if not a:
        return b
    if not b:
        return a
    return f"{a}, {b}" if not b.startswith(",") else a + b


def resolve(table: dict, rules: str, seed: int = 0) -> dict:
    """Resolve the rule text against a region table (the atlas JSON). Returns the plan dict."""
    if not isinstance(table, dict) or "regions" not in table:
        raise PlanError("The region table has no 'regions' list.")
    sections = parse_rules(rules)
    regions = table["regions"]
    notes = []

    plan_settings = {k: d for k, (_, d, _) in PLAN_KEYS.items()}
    for s in sections:
        if s.is_plan:
            for key, _, val, line in s.settings:
                plan_settings[key] = _convert(key, val, line, PLAN_KEYS)

    rule_sections = [s for s in sections if not s.is_plan]
    hits = {i: 0 for i in range(len(rule_sections))}
    for s_i, s in enumerate(rule_sections):
        for alt in s.alternatives:
            for t in alt:
                if t.kind == "region" and not t.negate:
                    a = max(int(v) for v in t.value.split("-"))
                    if a >= len(regions):
                        raise PlanError(f"line {s.line}: region {a} does not exist "
                                        f"(the table has regions 0-{len(regions) - 1})")

    out = []
    for r in regions:
        rid = r["region_id"]
        matched = []
        for s_i, s in enumerate(rule_sections):
            spec = s.match(r)
            if spec is not None:
                matched.append((spec, s_i, s))
                hits[s_i] += 1
        matched.sort(key=lambda x: (x[0], x[1]))
        rng = random.Random(f"{seed}:{rid}")
        vals = {k: d for k, (_, d, _) in SETTINGS.items()}
        for _, _, s in matched:
            for key, op, raw, line in s.settings:
                if key in ("prompt", "negative", "lora", "video_prompt"):
                    v = _convert(key, raw, line)
                    v = _fill(v, r, vals[key], rng) if key != "lora" else v
                    vals[key] = _join(vals[key], v) if op == "+=" else v
                else:
                    vals[key] = _convert(key, raw, line)
        if vals["seed"] < 0:
            vals["seed"] = (int(seed) + rid) % (2 ** 63)
        vals["denoise"] = min(max(vals["denoise"], 0.0), 1.0)
        entry = {"region_id": rid, "name": r.get("name", f"r{rid}"), "group_id": r.get("group_id"),
                 "bbox": r.get("bbox"), "area": r.get("area"), "tags": r.get("tags", []),
                 "rules": [s.header for _, _, s in matched], **vals}
        if vals["strategy"] != "keep" and not vals["prompt"]:
            notes.append(f"region {rid} ({entry['name']}) has an empty prompt")
        out.append(entry)

    for s_i, s in enumerate(rule_sections):
        if hits[s_i] == 0:
            notes.append(f"line {s.line}: [{s.header}] matches no region")

    order = _order(out, plan_settings["order"])
    return {"format": PLAN_FORMAT, "version": PLAN_VERSION, "seed": int(seed),
            "width": table.get("width"), "height": table.get("height"),
            "plan": plan_settings, "order": order, "regions": out, "notes": notes,
            "rules_text": rules or ""}         # kubakub versions appends each version's rules to this


def _order(entries: list[dict], mode: str) -> list[int]:
    active = [e for e in entries if e["strategy"] != "keep"]
    if mode == "largest_first":
        key = lambda e: (e["order"], -(e["area"] or 0), e["region_id"])
    elif mode == "smallest_first":
        key = lambda e: (e["order"], e["area"] or 0, e["region_id"])
    elif mode == "reading":
        key = lambda e: e["region_id"]
    else:
        key = lambda e: (e["order"], e["region_id"])
    return [e["region_id"] for e in sorted(active, key=key)]


def report(plan: dict, max_prompt: int = 70) -> str:
    """A readable table of the resolved plan."""
    lines = [f"{len(plan['regions'])} regions, {len(plan['order'])} to process, "
             f"order {plan['plan']['order']}, seed {plan['seed']}"]
    counts = {}
    for e in plan["regions"]:
        counts[e["strategy"]] = counts.get(e["strategy"], 0) + 1
    lines.append("strategies: " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())))
    lines.append("")
    for e in plan["regions"]:
        p = e["prompt"] if len(e["prompt"]) <= max_prompt else e["prompt"][:max_prompt - 3] + "..."
        lines.append(f"{e['region_id']:>3} {e['name'][:18]:<18} {e['strategy']:<14} "
                     f"d{e['denoise']:.2f}  {p}")
    for n in plan["notes"]:
        lines.append(f"note: {n}")
    return "\n".join(lines)


def settings_help() -> str:
    rows = [f"{k} = {d!r}  // {h}" for k, (_, d, h) in SETTINGS.items()]
    return "\n".join(rows)
