"""
regions_seen.py

What the regions nodes of this session made, kept by node id, so every 'which regions' field can offer the names
there are (groups, floors, sections ...) as a list to pick from: web/kubakub_which.js asks /kubakub/regions/seen.

Kept in memory only (the last tables, not the masks): nothing is written to disk or into a workflow, and after a
restart the list is there again once the regions node has run.
"""

from __future__ import annotations

import re
from collections import Counter, OrderedDict

MAX_NODES = 64
ROW_MAX = 32                       # a row of tags with more names than this is not shown as buttons
FIRST = ("floors", "sections", "facing", "layers")

_SEEN: "OrderedDict[str, dict]" = OrderedDict()


def note(table, node_id=None) -> None:
    """Remember a region table for the node that is running now (called when a Regions is made)."""
    if not isinstance(table, dict) or not table.get("regions"):
        return
    if node_id is None:
        try:
            from comfy_execution.utils import get_executing_context
            ctx = get_executing_context()
        except Exception:          # noqa: BLE001  (a test, or a ComfyUI without it: the list just stays empty)
            ctx = None
        node_id = getattr(ctx, "node_id", None)
    if node_id is None:
        return
    key = str(node_id)
    _SEEN[key] = table
    _SEEN.move_to_end(key)
    while len(_SEEN) > MAX_NODES:
        _SEEN.popitem(last=False)


def find(node_ids):
    """(node id, table) of the first of these nodes that made regions in this session, or (None, None)."""
    for n in node_ids:
        t = _SEEN.get(str(n))
        if t is not None:
            return str(n), t
    return None, None


def _natural(s):
    return [int(p) if p.isdigit() else p.lower() for p in re.split(r"(\d+)", str(s))]


def summary(table: dict, row_max: int = ROW_MAX) -> dict:
    """
    A region table as what there is to pick:

    groups    [[name, regions], ...]
    rows      [{"label": "floors", "tags": [[name, regions], ...]}, ...]   tags sorted into the passes they come
              from (floors, sections ...); tags of no pass are the row "tags"
    many      [[label, names, an example], ...]   rows too long to show as buttons (planes, parts)
    regions   [[name, group, [its tags that are in rows]], ...]
    """
    regs = table.get("regions", [])
    groups = Counter(str(r.get("group_id")) for r in regs if r.get("group_id") not in (None, ""))
    tag_n = Counter(str(t) for r in regs for t in r.get("tags", []) or [])
    low_groups = {g.lower() for g in groups}
    rows, many, used = [], [], set()
    passes = table.get("passes") if isinstance(table.get("passes"), dict) else {}
    for label in sorted(passes, key=lambda k: (FIRST.index(k) if k in FIRST else len(FIRST), k)):
        names = [str(n) for n in passes[label] or [] if str(n) in tag_n]
        if not names or {n.lower() for n in names} <= low_groups:      # the pass the regions come from = the groups
            continue
        used.update(names)
        names.sort(key=_natural)
        if len(names) > row_max:
            many.append([str(label), len(names), names[0]])
        else:
            rows.append({"label": str(label), "tags": [[n, tag_n[n]] for n in names]})
    rest = sorted((t for t in tag_n if t not in used), key=_natural)
    if rest and len(rest) <= row_max:
        rows.append({"label": "tags", "tags": [[n, tag_n[n]] for n in rest]})
    elif rest:
        many.append(["tags", len(rest), rest[0]])
    shown = {t for row in rows for t, _ in row["tags"]}
    return {"count": len(regs),
            "groups": [[g, groups[g]] for g in sorted(groups, key=_natural)],
            "rows": rows, "many": many,
            "regions": [[str(r.get("name", "")), str(r.get("group_id", "") if r.get("group_id") is not None else ""),
                         [str(t) for t in r.get("tags", []) or [] if str(t) in shown]] for r in regs]}
