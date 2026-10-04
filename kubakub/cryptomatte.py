"""
cryptomatte.py

Cryptomatte (Psyop's open standard, written by Blender, Houdini Karma / Mantra, Arnold, Redshift, V-Ray, Octane ...)
into region label maps. The EXR header holds per layer a manifest {object name: 32-bit hash as hex}; the channels
<layer>00.r/g/b/a, <layer>01... hold (id, coverage) pairs per pixel, the id being the hash's bits stored as float32.
A pixel belongs to the id with the most coverage (rank 0). Pure numpy (tests/test_cryptomatte.py).
Spec: https://github.com/Psyop/Cryptomatte (Psyop, Jonah Friedman, Andy Jones).
"""

from __future__ import annotations

import fnmatch
import json
import re

import numpy as np


def layers(attrs):
    """{layer name: {"manifest": {name: hex}, "key": ...}} from the EXR header attributes."""
    out = {}
    for k, v in attrs.items():
        m = re.match(r"cryptomatte/([0-9a-fA-F]+)/name$", k)
        if not m:
            continue
        key = m.group(1)
        man = attrs.get(f"cryptomatte/{key}/manifest", "{}")
        try:
            manifest = json.loads(man) if isinstance(man, str) else dict(man)
        except ValueError:
            manifest = {}
        out[str(v)] = {"manifest": manifest, "key": key}
    return out


def pick_layer(names, wanted):
    """The layer whose name matches wanted ('object', 'material', 'asset' or a full name), else None."""
    wanted = (wanted or "").strip().lower()
    for n in names:
        if n.lower() == wanted:
            return n
    for n in names:
        if wanted and wanted in n.lower():
            return n
    return None


def _rank_channels(channels, layer):
    """[(id channel, coverage channel), ...] in rank order for one layer."""
    pairs = []
    k = 0
    while True:
        base = f"{layer}{k:02d}."
        found = {c[len(base):].lower(): c for c in channels if c.startswith(base)}
        if not found:
            break
        if "r" in found and "g" in found:
            pairs.append((found["r"], found["g"]))
        if "b" in found and "a" in found:
            pairs.append((found["b"], found["a"]))
        k += 1
    return pairs


def id_bits(a):
    """float32 channel -> its uint32 bit pattern (the hash)."""
    a = np.asarray(a)
    if a.dtype == np.uint32:
        return a
    return np.ascontiguousarray(a.astype(np.float32)).view(np.uint32)


def rank0(channels, layer):
    """(ids uint32 HxW, coverage float32 HxW) of the dominant object per pixel."""
    pairs = _rank_channels(channels, layer)
    if not pairs:
        raise ValueError(f"no cryptomatte channels for layer '{layer}'")
    idc, covc = pairs[0]
    return id_bits(channels[idc]), np.asarray(channels[covc], np.float32)


def names_by_hash(manifest):
    return {int(h, 16): n for n, h in manifest.items()}


def label_map(channels, layer, manifest, exclude="", min_coverage=0.0):
    """-> (labels HxW int32, -1 = none, [names per label]). Excluded names (wildcards) and uncovered pixels -> -1."""
    ids, cov = rank0(channels, layer)
    lookup = names_by_hash(manifest)
    pats = [p.strip() for p in re.split(r"[,\n]", exclude or "") if p.strip()]
    uniq, inv = np.unique(ids, return_inverse=True)
    names, remap = [], np.full(len(uniq), -1, np.int32)
    for i, h in enumerate(uniq.tolist()):
        if h == 0:
            continue
        name = lookup.get(h, f"id_{h:08x}")
        if any(fnmatch.fnmatchcase(name, p) for p in pats):
            continue
        remap[i] = len(names)
        names.append(name)
    labels = remap[inv.reshape(ids.shape)]
    labels[cov <= min_coverage] = -1
    return labels.astype(np.int32), names


def tags_from(channels, layer, manifest, labels, n):
    """Per label the name the tag layer shows most (e.g. its material) -> [[tag], ...]."""
    ids, cov = rank0(channels, layer)
    lookup = names_by_hash(manifest)
    out = [[] for _ in range(n)]
    sel = labels >= 0
    if not sel.any():
        return out
    lab, tid = labels[sel], ids[sel]
    for i in range(n):
        m = lab == i
        if not m.any():
            continue
        vals, cnt = np.unique(tid[m], return_counts=True)
        h = int(vals[np.argmax(cnt)])
        if h:
            out[i] = [lookup.get(h, f"id_{h:08x}")]
    return out


def beauty(channels):
    """The picture of the render (Combined / RGBA) as linear float32 HxWx3, or None."""
    for pre in ("ViewLayer.Combined.", "Combined.", "", "rgba.", "RGBA."):
        keys = [pre + c for c in ("R", "G", "B")]
        if all(k in channels for k in keys):
            return np.stack([np.asarray(channels[k], np.float32) for k in keys], -1)
    rgb = sorted(c for c in channels if c.endswith((".R", ".G", ".B")) and "Crypto" not in c)
    for c in rgb:
        if c.endswith(".R"):
            base = c[:-2]
            if all(f"{base}.{x}" in channels for x in "GB"):
                return np.stack([np.asarray(channels[f"{base}.{x}"], np.float32) for x in "RGB"], -1)
    return None
