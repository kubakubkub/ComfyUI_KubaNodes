"""
region_cache.py

In-memory caches of the Region Sampler that live across queues (until ComfyUI
restarts), so changing one prompt out of 300 regions re-samples only that
region (and later regions whose crops overlap it):

RESULTS  one region's generated pixels (after colour match, at crop size),
         keyed on the actual input crop pixels, the masks, prompt, seed and
         every sampling setting, plus the identity of model / clip / vae.
CONDS    text and text + vision conditioning, keyed on the text encoder's
         identity, the text and a hash of the reference image.

Model identity is the object itself (ComfyUI keeps a loader's output object
alive while its inputs do not change; a LoRA or any patch makes a new object
or a new patches_uuid) and is held by weak reference, so a cached entry never
keeps a model alive and a reused id() can never match a dead model's entry.
Both caches are LRU bounded by bytes (settings region_cache_mb, default 4096, and
cond_cache_mb, default 2048, in kubakub.ini; 0 switches a cache off).

Pure torch, no ComfyUI imports (tests/test_region_cache.py).
"""

from __future__ import annotations

import hashlib
import logging
import os
import threading
import weakref
from collections import OrderedDict

import torch

log = logging.getLogger("KUBA.regions")

def _pack_settings():
    """The pack's settings.py (kubakub.ini [settings]); loaded by path when this module is imported on its own."""
    try:
        from .. import settings
        return settings
    except ImportError:
        import importlib.util
        p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "settings.py")
        spec = importlib.util.spec_from_file_location("kubakub_pack_settings", p)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod


kst = _pack_settings()


def _env_mb(key: str, default: int) -> int:
    """A cache size in MB from kubakub.ini [settings] (or the old environment variable)."""
    return max(0, int(kst.number(key, default)))


# --------------------------------------------------------------------------
# hashing and sizes
# --------------------------------------------------------------------------

def digest(t: torch.Tensor | None) -> str:
    """Exact content hash of a tensor (dtype, shape and every byte)."""
    if t is None:
        return "-"
    t = t.detach()
    if t.device.type != "cpu":
        t = t.cpu()
    t = t.contiguous()
    h = hashlib.sha256(f"{t.dtype}|{tuple(t.shape)}|".encode())
    if t.numel():
        h.update(memoryview(t.reshape(-1).view(torch.uint8).numpy()))
    return h.hexdigest()[:32]


def nbytes(obj) -> int:
    """Bytes held by the tensors inside a value (tensors, lists, tuples, dicts)."""
    seen = set()

    def walk(o) -> int:
        if isinstance(o, torch.Tensor):
            if id(o) in seen:
                return 0
            seen.add(id(o))
            return o.numel() * o.element_size()
        if isinstance(o, (list, tuple)):
            return sum(walk(x) for x in o)
        if isinstance(o, dict):
            return sum(walk(x) for x in o.values())
        return 0

    return walk(obj) + 256


def identity(obj):
    """Key part for a model / clip / vae object: type, id and its patch fingerprint (None for None)."""
    if obj is None:
        return None
    fp = []
    for o in (obj, getattr(obj, "patcher", None)):
        u = getattr(o, "patches_uuid", None)
        if u is not None:
            fp.append(str(u))
    return (type(obj).__name__, id(obj), *fp)


# --------------------------------------------------------------------------
# LRU by bytes, entries tied to owner objects by weak reference
# --------------------------------------------------------------------------

class ByteLRU:
    def __init__(self, name: str, max_bytes: int):
        self.name = name
        self.max_bytes = int(max_bytes)
        self._d: OrderedDict = OrderedDict()     # key -> (value, owner weakrefs, bytes)
        self.bytes = 0
        self.hits = 0
        self.misses = 0
        self._lock = threading.Lock()

    def __len__(self) -> int:
        return len(self._d)

    @staticmethod
    def _alive(refs) -> bool:
        return all(r() is not None for r in refs)

    def get(self, key):
        if self.max_bytes <= 0 or key is None:
            return None
        with self._lock:
            hit = self._d.get(key)
            if hit is not None and not self._alive(hit[1]):
                self._drop(key)
                hit = None
            if hit is None:
                self.misses += 1
                return None
            self._d.move_to_end(key)
            self.hits += 1
            return hit[0]

    def put(self, key, value, owners=()) -> bool:
        if self.max_bytes <= 0 or key is None:
            return False
        try:
            refs = tuple(weakref.ref(o) for o in owners if o is not None)
        except TypeError:                         # an owner that cannot be weakly referenced: do not cache
            return False
        size = nbytes(value)
        if size > self.max_bytes:
            return False
        with self._lock:
            if key in self._d:
                self._drop(key)
            for k in [k for k, v in self._d.items() if not self._alive(v[1])]:
                self._drop(k)
            while self._d and self.bytes + size > self.max_bytes:
                self._drop(next(iter(self._d)))
            self._d[key] = (value, refs, size)
            self.bytes += size
        return True

    def _drop(self, key) -> None:
        v = self._d.pop(key, None)
        if v is not None:
            self.bytes -= v[2]

    def clear(self) -> None:
        with self._lock:
            self._d.clear()
            self.bytes = 0
            self.hits = self.misses = 0

    def stats(self) -> str:
        return f"{self.name}: {len(self._d)} entries, {self.bytes / 2 ** 20:.0f} MB"


RESULTS = ByteLRU("region results", _env_mb("region_cache_mb", 4096) * 2 ** 20)
CONDS = ByteLRU("conditionings", _env_mb("cond_cache_mb", 2048) * 2 ** 20)


# --------------------------------------------------------------------------
# per-run context
# --------------------------------------------------------------------------

class RunCache:
    """The RESULTS cache seen from one run of one adapter: a base key (family, schedule, model identities)."""

    def __init__(self, adapter, store: ByteLRU = RESULTS):
        self.store = store
        self.owners = [getattr(adapter, a, None) for a in ("model", "clip", "vae")]
        self.base = (type(adapter).__name__, getattr(adapter, "family", ""), getattr(adapter, "schedule", ""),
                     int(getattr(adapter, "grid", 0)), getattr(adapter, "core_scheduler", ""),
                     *(identity(o) for o in self.owners))
        self.hits = 0
        self.misses = 0

    def key(self, *parts):
        return (self.base, *parts)

    def get(self, key):
        v = self.store.get(key)
        if v is None:
            self.misses += 1
        else:
            self.hits += 1
        return v

    def put(self, key, value) -> None:
        self.store.put(key, value, self.owners)


def run_cache(adapter, enabled: bool = True) -> RunCache | None:
    """A RunCache for this adapter, or None when caching is off or the adapter has no model and vae to key on."""
    if not enabled or RESULTS.max_bytes <= 0:
        return None
    if getattr(adapter, "model", None) is None or getattr(adapter, "vae", None) is None:
        return None
    return RunCache(adapter)


def cached_cond(clip, kind: str, text: str, image: torch.Tensor | None, encode):
    """Conditioning for (clip identity, kind, text, image hash) from CONDS, or encode() and store it."""
    key = None
    if clip is not None and CONDS.max_bytes > 0:
        key = (identity(clip), kind, text, digest(image) if image is not None else "-")
        hit = CONDS.get(key)
        if hit is not None:
            return hit
    cond = encode()
    if key is not None:
        CONDS.put(key, cond, (clip,))
    return cond


def clear_all() -> None:
    RESULTS.clear()
    CONDS.clear()
