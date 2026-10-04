"""
clip_cache.py

In-memory caches of kubakub region video sampler (LTX) and kubakub keyframe clips (h3): rendered clips and enhanced
prompts, so that changing one clip of a timeline does not render (or enhance) the others again.
Key = a content hash of everything that shapes the result; an entry is valid only while the same model objects are
alive (weak references: a reloaded or re-patched model, e.g. another LoRA, is a new object and misses).
No ComfyUI imports (tests/test_video.py, tests/test_keyframes.py).
"""

from __future__ import annotations

import hashlib
import weakref
from collections import OrderedDict

import numpy as np
import torch


def digest(*parts):
    """A content hash of tensors, numpy arrays, dicts (audio), lists and plain values."""
    h = hashlib.blake2b(digest_size=16)
    for p in parts:
        if isinstance(p, torch.Tensor):
            t = p.detach().cpu()
            t = (t.float() if t.dtype == torch.bfloat16 else t).contiguous()
            h.update(repr((tuple(t.shape), str(t.dtype))).encode())
            h.update(t.numpy().view(np.uint8).reshape(-1))            # no bytes copy of big frame stacks
        elif isinstance(p, np.ndarray):
            a = np.ascontiguousarray(p)
            h.update(repr(("nd", a.shape, str(a.dtype))).encode())
            h.update(a.view(np.uint8).reshape(-1))
        elif isinstance(p, dict):
            h.update(digest(*[x for k in sorted(p, key=str) for x in (str(k), p[k])]).encode())
        elif isinstance(p, (list, tuple)):
            h.update(digest(*p).encode())
        else:
            h.update(repr(p).encode())
    return h.hexdigest()


def _none():
    return None


def _nbytes(value):
    if isinstance(value, (torch.Tensor,)):
        return value.element_size() * value.nelement()
    if isinstance(value, np.ndarray):
        return value.nbytes
    if isinstance(value, (list, tuple)):
        return sum(_nbytes(v) for v in value)
    if isinstance(value, dict):
        return sum(_nbytes(v) for v in value.values())
    return 0


class ClipCache:
    """An LRU of key -> value, bound to the identity of model objects (weak references), capped by count and bytes."""

    def __init__(self, max_items=6, max_bytes=None):
        self.max_items, self.max_bytes = int(max_items), max_bytes
        self._d: OrderedDict = OrderedDict()
        self.hits = self.misses = 0

    def __len__(self):
        return len(self._d)

    def clear(self):
        self._d.clear()

    def get(self, key, objs=()):
        """The cached value if the same model objects made it, else None."""
        hit = self._d.get(key)
        if hit is None or len(hit[0]) != len(objs) or any(r() is not o for r, o in zip(hit[0], objs)):
            self.misses += 1
            return None
        self._d.move_to_end(key)
        self.hits += 1
        return hit[1]

    def put(self, key, objs, value):
        try:
            refs = tuple(weakref.ref(o) if o is not None else _none for o in objs)
        except TypeError:                                 # an object without weak references: no caching
            return
        self._d[key] = (refs, value, _nbytes(value))
        self._d.move_to_end(key)
        while len(self._d) > self.max_items:
            self._d.popitem(last=False)
        if self.max_bytes:
            while len(self._d) > 1 and sum(v[2] for v in self._d.values()) > self.max_bytes:
                self._d.popitem(last=False)
