"""
types.py

Payloads passed between Kuba Regions nodes. Pure logic (torch only).

KUBA_REGIONS: one int32 label map instead of N float masks. 100 regions at 4K
cost 33 MB instead of 3.3 GB. Nodes that feed core get MASKs on demand.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

REGIONS_FORMAT = "kubakub.regions.regions"
REGIONS_VERSION = 1


@dataclass
class Regions:
    labels: torch.Tensor                # int32 [T, H, W], -1 = no region, else region_id
    table: dict                         # the atlas JSON (regions, groups, notes ...)
    scope: torch.Tensor | None = None   # float [T, H, W], 1 = may be diffused

    @property
    def count(self) -> int:
        return len(self.table.get("regions", []))

    @property
    def size(self) -> tuple[int, int]:
        return int(self.labels.shape[-1]), int(self.labels.shape[-2])

    def region(self, region_id: int) -> dict:
        return self.table["regions"][region_id]

    def mask(self, ids) -> torch.Tensor:
        """Float mask [T, H, W] of the union of the given region ids."""
        if isinstance(ids, int):
            ids = [ids]
        ids = torch.as_tensor(list(ids), dtype=self.labels.dtype, device=self.labels.device)
        return torch.isin(self.labels, ids).float()

    def masks(self) -> torch.Tensor:
        """MASK batch, one per region (only for single frame regions)."""
        lab = self.labels[0]
        out = torch.zeros((self.count, *lab.shape), dtype=torch.float32)
        for i in range(self.count):
            out[i] = (lab == i).float()
        return out

    def resized(self, w: int, h: int) -> "Regions":
        """The same regions on a smaller (or larger) canvas: labels by nearest pixel, boxes and areas
        measured again (a region too small to survive keeps its scaled box and area 0). For fast drafts."""
        import copy
        import torch.nn.functional as F
        W, H = self.size
        if (w, h) == (W, H):
            return self
        lab = F.interpolate(self.labels[:, None].float(), size=(h, w), mode="nearest")[:, 0].to(self.labels.dtype)
        scope = None
        if self.scope is not None:
            scope = (F.interpolate(self.scope[:, None].float(), size=(h, w), mode="area")[:, 0] > 0.5).float()
        table = copy.deepcopy(self.table)
        table["width"], table["height"] = w, h
        sx, sy = w / W, h / H
        first = lab[0]
        for r in table.get("regions", []):
            ys, xs = torch.nonzero(first == r["region_id"], as_tuple=True)
            if len(xs):
                x0, y0 = int(xs.min()), int(ys.min())
                r["bbox"] = [x0, y0, int(xs.max()) - x0 + 1, int(ys.max()) - y0 + 1]
            elif r.get("bbox"):
                x, y, bw, bh = r["bbox"]
                r["bbox"] = [int(x * sx), int(y * sy), max(1, round(bw * sx)), max(1, round(bh * sy))]
            r["area"] = int(len(xs))
            if r.get("centroid"):
                r["centroid"] = [r["centroid"][0] * sx, r["centroid"][1] * sy]
        return Regions(lab, table, scope)

    @classmethod
    def from_numpy(cls, labels, table: dict, scope=None) -> "Regions":
        lab = torch.from_numpy(labels.astype("int32"))
        if lab.ndim == 2:
            lab = lab[None]
        sc = None
        if scope is not None:
            sc = torch.from_numpy(scope.astype("float32"))
            if sc.ndim == 2:
                sc = sc[None]
        return cls(lab, table, sc)


@dataclass
class RegionPlan:
    """KUBA_PLAN: the resolved plan (plan.resolve dict) together with the regions it is for."""
    plan: dict
    regions: Regions

    def entry(self, region_id: int) -> dict:
        return self.plan["regions"][region_id]

    @property
    def order(self) -> list[int]:
        return self.plan["order"]
