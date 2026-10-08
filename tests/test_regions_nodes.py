"""
Model free test of the region nodes themselves (nodes/regions): the outputs added at the end of regions to mask,
regions from masks, regions from matrix / masks and canvas plan, the one selector of regions to svg / pdf / dxf,
and the core boxes of regions from sam3 (the box parsing only, SAM is not run). Does not load a model.

Run from the ComfyUI_windows_portable folder:
    python_embeded\\python.exe ComfyUI\\custom_nodes\\ComfyUI_KubaNodes\\tests\\test_regions_nodes.py
"""

import os
import shutil
import sys
import tempfile

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(HERE))))     # ComfyUI (folder_paths, comfy_api)
sys.path.insert(0, os.path.dirname(HERE))
import folder_paths  # noqa: E402

TMP = tempfile.mkdtemp(prefix="kkd_regions_")
folder_paths.set_temp_directory(os.path.join(TMP, "temp"))
folder_paths.set_output_directory(os.path.join(TMP, "output"))

sys.path.insert(0, HERE)     # embedded Python leaves the script folder off the path
import _pack  # noqa: E402,F401  (the pack as a package, see tests/_pack.py)

from comfy_execution.graph_utils import ExecutionBlocker  # noqa: E402
import kubapack.nodes.regions.facade_mask_atlas_Kub as na  # noqa: E402
import kubapack.nodes.regions.nodes_canvas as nc  # noqa: E402
import kubapack.nodes.regions.nodes_sources as ns  # noqa: E402
import kubapack.nodes.regions.nodes_vector as nv  # noqa: E402
from kubakub import sam_prompts as sp  # noqa: E402
from kubakub.director.render import select_regions  # noqa: E402
from kubakub.types import Regions  # noqa: E402

failures = []


def check(name, cond, detail=""):
    print(("ok   " if cond else "FAIL ") + name + (f"  ({detail})" if detail and not cond else ""))
    if not cond:
        failures.append(name)


def outs(node):
    return [(o.id, o.io_type) for o in node.define_schema().outputs]


def ins(node):
    return {i.id: i for i in node.define_schema().inputs}


# ---- a small atlas: a wall, two windows, a door whose name has a space in it
W, H = 200, 100
lab = np.zeros((H, W), np.int32)
lab[20:40, 20:50] = 1
lab[20:40, 120:150] = 2
lab[60:100, 80:110] = 3
table = {"regions": [
    {"region_id": 0, "name": "wall", "group_id": "wall", "tags": []},
    {"region_id": 1, "name": "W_F1_01", "group_id": "Windows", "tags": ["front", "floor 1"]},
    {"region_id": 2, "name": "W_F1_02", "group_id": "Windows", "tags": ["side"]},
    {"region_id": 3, "name": "ground door", "group_id": "doors", "tags": ["front"]},
], "groups": {"wall": [0], "Windows": [1, 2], "doors": [3]}}
regions = Regions.from_numpy(lab, table, None)

# ---- every output of before is still there in its place; the new ones follow, with a tooltip
check("regions to mask: outputs", outs(ns.KUBA_RegionsMask) == [("mask", "MASK"), ("names", "STRING"), ("region_masks", "MASK")],
      str(outs(ns.KUBA_RegionsMask)))
check("regions from masks: outputs", [o for o, _ in outs(ns.KUBA_RegionsFromMasks)] ==
      ["regions", "region_masks", "regions_json", "preview", "report", "scope"], str(outs(ns.KUBA_RegionsFromMasks)))
check("regions from matrix / masks: outputs", [o for o, _ in outs(na.KUBA_FacadeMaskAtlas)] ==
      ["masks", "regions_json", "preview", "scope", "regions", "report"], str(outs(na.KUBA_FacadeMaskAtlas)))
check("canvas plan: outputs", outs(nc.KUBA_CanvasPlan)[-3:] == [("k", "FLOAT"), ("target_width", "INT"), ("target_height", "INT")]
      and [o for o, _ in outs(nc.KUBA_CanvasPlan)][:7] == ["canvas_plan", "width", "height", "frames", "fps", "plan_json", "report"],
      str(outs(nc.KUBA_CanvasPlan)))
check("regions to svg / pdf / dxf: outputs", [o for o, _ in outs(nv.KUBA_RegionsToVector)] == ["preview", "files", "report"])
nodes = (ns.KUBA_RegionsMask, ns.KUBA_RegionsFromMasks, ns.KUBA_RegionsSAM3Masks, na.KUBA_FacadeMaskAtlas, nc.KUBA_CanvasPlan,
         nv.KUBA_RegionsToVector)
check("every input and output has a tooltip",
      all(getattr(i, "tooltip", None) for n in nodes for i in list(n.define_schema().inputs) + list(n.define_schema().outputs)))
check("the new inputs are optional", ins(ns.KUBA_RegionsMask)["per_region_masks"].optional is True
      and ins(ns.KUBA_RegionsSAM3Masks)["boxes"].optional is True)
check("input order of before is kept", list(ins(ns.KUBA_RegionsMask))[:5] == ["regions", "select", "grow_px", "feather_px", "invert"]
      and list(ins(ns.KUBA_RegionsSAM3Masks))[-3:] == ["save_folder", "overwrite", "boxes"]
      and list(ins(nv.KUBA_RegionsToVector))[:10] == ["regions", "mask", "select", "mode", "width_mm", "simplify_px", "corner_deg",
                                                    "min_area_px", "spur_px", "kerf_mm"])

# ---- regions to mask: one mask per selected region
res = ns.KUBA_RegionsMask.execute(regions, "group:Windows", 4, 6, True).result
check("regions to mask: as before without the switch, region_masks tells what to switch on",
      res[0].shape == (1, H, W) and res[1] == "W_F1_01\nW_F1_02" and isinstance(res[2], ExecutionBlocker)
      and "per_region_masks" in str(res[2].message))
res = ns.KUBA_RegionsMask.execute(regions, "group:Windows, wall", 4, 6, True, per_region_masks=True).result
each = res[2]
check("regions to mask: region_masks in the order of names", res[1].split("\n") == ["wall", "W_F1_01", "W_F1_02"]
      and each.shape == (3, H, W) and each.dtype == torch.float32)
check("regions to mask: region_masks before grow / feather / invert",
      all(torch.equal(each[i], torch.from_numpy((lab == rid).astype(np.float32))) for i, rid in enumerate((0, 1, 2))))
check("regions to mask: the mask itself is unchanged by the switch",
      torch.equal(res[0], ns.KUBA_RegionsMask.execute(regions, "group:Windows, wall", 4, 6, True).result[0]))
keep = ns.REGION_MASKS_MAX_GB
ns.REGION_MASKS_MAX_GB = 3 * H * W * 4 / 1024 ** 3 * 0.9          # as if three masks were too many
try:
    ns.KUBA_RegionsMask.execute(regions, "*", 0, 0, False, per_region_masks=True)
    check("regions to mask: too many region masks are refused", False)
except ValueError as e:
    check("regions to mask: too many region masks are refused, in plain words", "Narrow 'select'" in str(e) and "GB" in str(e), str(e))
check("regions to mask: under the limit after a narrower select",
      ns.KUBA_RegionsMask.execute(regions, "W_F1_01", 0, 0, False, per_region_masks=True).result[2].shape == (1, H, W))
check("regions to mask: over the limit but switched off still runs",
      ns.KUBA_RegionsMask.execute(regions, "*", 0, 0, False).result[0].shape == (1, H, W))
ns.REGION_MASKS_MAX_GB = keep

# ---- regions to svg / pdf / dxf: the plan selector, and what the first selector matched still matches
sel = lambda t: nv.select_ids(table, t)  # noqa: E731
check("select: empty = all", sel("") == [0, 1, 2, 3] and sel(" \n ") == [0, 1, 2, 3])
check("select: names, group:, tag:, comma and new line = or", sel("W_F1_*") == [1, 2] and sel("group:Windows, wall") == [0, 1, 2]
      and sel("tag:side\nwall") == [0, 2] and sel("GROUP:windows") == [1, 2])
check("select: same as the plan selector of regions to mask",
      all(sel(t) == sorted(select_regions(table, t)) for t in ("group:Windows", "tag:front", "W_F1_*, wall", "tag:front !ground*",
                                                               "group:Windows !W_F1_02", "*", "region:1-2")))
check("select: space = and, ! = not (new)", sel("group:Windows tag:front") == [1] and sel("tag:front !W_*") == [3])
check("select: a name with a space still matches (first selector)", sel("ground door") == [3] and sel("wall, ground door") == [0, 3]
      and sel("tag:floor 1") == [1] and sel("ground d*") == [3])
check("select: nothing matches -> empty", sel("nothing_here") == [])
for text in ("W_F1_*", "group:Windows, wall", "tag:side\nwall", "ground door", "tag:floor 1", "w_f1_01", "group:win*", "x, tag:front"):
    old = [r["region_id"] for r in table["regions"] if any(nv._old_match(r, p.strip().lower())
                                                           for p in text.replace("\n", ",").split(",") if p.strip())]
    check(f"select {text!r}: everything the first selector matched", set(old) <= set(sel(text)), f"{old} vs {sel(text)}")

common = dict(mode="outline", width_mm=0.0, simplify_px=1.0, corner_deg=35.0, min_area_px=4.0, spur_px=6.0, kerf_mm=0.0,
              svg=True, pdf=False, dxf=False, filename_prefix="t")
rep = nv.KUBA_RegionsToVector.execute(regions=regions, select="group:Windows !W_F1_02", **common).result[2]
check("to vector: select in plan syntax", rep.startswith("outline: 1 regions") and "not used" not in rep, rep)
m = torch.from_numpy((lab == 1).astype(np.float32))[None]
rep = nv.KUBA_RegionsToVector.execute(regions=regions, mask=m, select="group:Windows", **common).result[2]
check("to vector: mask and select -> the report says the mask is used", "mask is vectorised" in rep and "select are not used" in rep, rep)
rep = nv.KUBA_RegionsToVector.execute(mask=m, select="", **common).result[2]
check("to vector: a mask alone -> no such note", "not used" not in rep, rep)
iv = ins(nv.KUBA_RegionsToVector)
check("to vector: kerf_mm says it needs width_mm; the three rare inputs are advanced",
      "width_mm is 0" in iv["kerf_mm"].tooltip and all(iv[k].advanced for k in ("spur_px", "kerf_mm", "corner_deg"))
      and not iv["width_mm"].advanced and not iv["simplify_px"].advanced)

# ---- regions from sam3: core boxes next to the Points Editor ones
core = [{"x": 10, "y": 20, "width": 30, "height": 40, "score": 0.9}]
check("sam3: 'boxes' is the core bounding box socket, 'bboxes' the Points Editor one",
      ins(ns.KUBA_RegionsSAM3Masks)["boxes"].io_type == "BOUNDING_BOX" and ins(ns.KUBA_RegionsSAM3Masks)["bboxes"].io_type == "BBOX")
check("sam3: core boxes as a list, one dict, per frame", sp.parse_boxes(ns._core_boxes(core)) == [(10, 20, 40, 60)]
      and sp.parse_boxes(ns._core_boxes(core[0])) == [(10, 20, 40, 60)]
      and sp.parse_boxes(ns._core_boxes([core, [{"x": 0, "y": 0, "width": 5, "height": 5}]])) == [(10, 20, 40, 60)]
      and sp.parse_boxes(ns._core_boxes([[], core])) == [] and sp.parse_boxes(ns._core_boxes(None)) == []
      and sp.parse_boxes(ns._core_boxes([])) == [])
check("sam3: Points Editor boxes pass through", ns._core_boxes([(1, 2, 3, 4)]) == [(1, 2, 3, 4)]
      and sp.parse_boxes(ns._core_boxes([[1, 2, 3, 4], [5, 6, 9, 9]])) == [(1, 2, 3, 4), (5, 6, 9, 9)])

# ---- regions from masks: report and scope
masks = torch.zeros((3, H, W))
masks[0] = 1
masks[1, 20:40, 20:50] = 1
masks[2, 0:3, 0:3] = 1                      # 9 px: under min_region_area, merged
res = ns.KUBA_RegionsFromMasks.execute(masks, "wall\nwin_01", "on_top", 64, True).result
check("regions from masks: report counts and notes", res[4].startswith(f"3 masks -> 2 regions in 2 groups, {W}x{H}")
      and "without a name" in res[4] and "merged" in res[4], res[4])
check("regions from masks: scope is all ones without one", res[5].shape == (1, H, W) and float(res[5].min()) == 1.0)
sc = torch.zeros((1, H, W))
sc[0, :, :100] = 1
res = ns.KUBA_RegionsFromMasks.execute(masks, "wall\nwin_01\nspeck", "on_top", 64, False, scope=sc).result
check("regions from masks: scope is the one that was used", torch.equal(res[5], sc) and res[5].dtype == torch.float32
      and float(res[0].labels[0][:, 100:].max()) < 0, str(res[5].shape))
check("regions from masks: dropped masks are in the report", "dropped" in res[4], res[4])
check("regions from masks: the outputs of before", isinstance(res[0], Regions) and res[1].shape[1:] == (H, W) and res[2].startswith("{")
      and res[3].shape == (1, H, W, 3))

# ---- regions from matrix / masks: report, with the sample note
img = torch.zeros((1, 120, 160, 3))
img[0, :, :80] = torch.tensor([0.8, 0.2, 0.2])
img[0, :, 80:] = torch.tensor([0.2, 0.2, 0.8])
img[0, 0:2, 0:2] = torch.tensor([0.1, 0.9, 0.1])             # 4 px speck
res = na.KUBA_FacadeMaskAtlas.execute(img, "color_regions", 100, True, 24, True, 0.08).result
check("regions from matrix: report counts", res[5].startswith("color_regions: 2 regions in ") and "160x120" in res[5], res[5])
check("regions from matrix: the outputs of before", res[0].shape == (2, 120, 160) and res[1].startswith("{") and res[2].shape == (1, 120, 160, 3)
      and res[3].shape == (1, 120, 160) and isinstance(res[4], Regions))
res = na.KUBA_FacadeMaskAtlas.execute(img, "mask_folder", 16, True, 24, True, 0.08, mask_folder="").result
check("regions from matrix: empty mask folder -> the sample, said in the report", "sample" in res[5] and "mask_folder: " in res[5], res[5])

# ---- canvas plan: k and the target size
res = nc.KUBA_CanvasPlan.execute(3840, 2160, "flux2", "2", 0.0, False, 0).result
check("canvas plan: k, target size", res[7:] == (2.0, 3840, 2160) or list(res[7:]) == [2.0, 3840, 2160], str(res[7:]))
check("canvas plan: the outputs of before", (res[1], res[2], res[3]) == (1920, 1088, 1) and isinstance(res[4], float)
      and res[5].startswith("{") and isinstance(res[6], str))
res = nc.KUBA_CanvasPlan.execute(3840, 2160, "flux2", "15/4", 0.0, False, 0).result
check("canvas plan: a fraction k as a number", res[7] == 3.75 and isinstance(res[7], float), str(res[7]))
res = nc.KUBA_CanvasPlan.execute(64, 64, "flux2", "auto", 0.0, False, 0, size_from=torch.zeros((1, 1080, 1920, 3))).result
check("canvas plan: target size from size_from, k from auto", (res[8], res[9]) == (1920, 1080) and res[7] > 0
      and abs(res[7] - 1920 / res[0].work_w) < 1e-9, str(res[7:]))

shutil.rmtree(TMP, ignore_errors=True)
print()
print("all region node tests passed" if not failures else f"{len(failures)} FAILED: " + ", ".join(failures))
sys.exit(1 if failures else 0)
