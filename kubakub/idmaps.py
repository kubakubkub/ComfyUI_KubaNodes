"""
idmaps.py

Regions from 3D ID renders (Houdini or any renderer) for KUBA_Regions From ID
Maps. numpy + opencv, no ComfyUI imports (tests/test_idmaps.py).

Expected folder (what a Houdini ID setup writes; names are free):
    ids_<pass>.png   ID map of one pass: every object in a flat colour
    ids_<pass>.txt   legend, one line per object: '<name> rgb <r> <g> <b>' (0..1, linear)
    mask_<name>.exr  optional full-res mask per object (alpha), also mask_<name>.png or png/mask_<name>.png
    *clay*.exr/png   optional shaded render, used as the reference image

A pass is one way of slicing the facade (elements, level, zone, bay ...). One
pass gives the regions, the others become tags. Masks of a pass may overlap
(object masks ignore occlusion); the ID map shows what the camera sees, so
the depth order is learned from it: for every pair of overlapping masks, the
one the ID map shows more often is in front.
"""

from __future__ import annotations

import glob
import os
import re
from concurrent.futures import ThreadPoolExecutor

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")   # read on the first EXR decode

import cv2  # noqa: E402
import numpy as np  # noqa: E402

try:
    from . import facade_core as fc
except ImportError:          # tests: the pack folder is on sys.path
    from kubakub import facade_core as fc

OVERLAPS = ("id_map", "smaller_wins")


# --------------------------------------------------------------------------
# files
# --------------------------------------------------------------------------

def parse_legend(text: str):
    """'<name> rgb r g b' lines (also 'name r g b', 'name = #rrggbb', 0..1 or 0..255). -> [(name, rgb 0..1)]"""
    out = []
    for raw in text.splitlines():
        line = raw.split("//", 1)[0].replace("=", " ").strip()
        if not line:
            continue
        m = re.match(r"^(\S+)\s+(?:rgb\s+)?#([0-9a-fA-F]{6})\s*$", line)
        if m:
            v = int(m.group(2), 16)
            out.append((m.group(1), np.array([(v >> 16) & 255, (v >> 8) & 255, v & 255], float) / 255))
            continue
        p = line.replace(",", " ").split()
        nums = [x for x in p[1:] if re.fullmatch(r"[-+]?\d*\.?\d+", x)]
        if len(nums) >= 3:
            c = np.array([float(x) for x in nums[:3]])
            out.append((p[0], c / 255 if c.max() > 1.0 else c))
    return out


def read_image(path: str) -> np.ndarray:
    """Float32 HxWxC (0..1 for 8/16 bit, as stored for EXR), channels RGB(A)."""
    if path.lower().endswith(".exr"):
        return read_exr(path)
    else:
        img = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError(f"cannot read {path}")
        scale = 65535.0 if img.dtype == np.uint16 else 255.0
        img = img.astype(np.float32) / scale
    if img.ndim == 2:
        img = img[..., None]
    if img.shape[2] >= 3:
        img = np.concatenate([img[..., 2::-1], img[..., 3:]], axis=2)   # BGR(A) -> RGB(A)
    return img


def read_exr(path: str) -> np.ndarray:
    """
    Float32 RGB(A) of an EXR. ComfyUI imports OpenCV before this pack, so its
    EXR codec stays off (OPENCV_IO_ENABLE_OPENEXR is read at import); imageio's
    FreeImage backend is tried first (needs FreeImage.dll on the system, it never
    downloads anything at runtime), then OpenCV. Raises when neither can.
    """
    errors = []
    try:
        import warnings

        import imageio.v3 as iio
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            img = np.asarray(iio.imread(path, plugin="EXR-FI"), dtype=np.float32)
        return img[..., None] if img.ndim == 2 else img
    except Exception as e:  # noqa: BLE001
        errors.append(f"imageio/FreeImage: {type(e).__name__}")
    try:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is not None:
            img = img.astype(np.float32)
            if img.ndim == 2:
                return img[..., None]
            return np.concatenate([img[..., 2::-1], img[..., 3:]], axis=2)
        errors.append("OpenCV: no image")
    except cv2.error:
        errors.append("OpenCV: EXR codec disabled")
    raise ValueError(f"cannot read EXR {os.path.basename(path)} ({'; '.join(errors)}). EXR ID masks need the package imageio "
                     f"with its FreeImage plugin (python -m pip install imageio, then once: python -c "
                     f"\"import imageio; imageio.plugins.freeimage.download()\"), or save the ID masks as PNG.")


def mask_from_image(img: np.ndarray) -> np.ndarray:
    """Float mask: alpha when it varies, else the brightest channel."""
    if img.shape[2] in (2, 4) and img[..., -1].min() < img[..., -1].max():
        return img[..., -1]
    return img[..., :3].max(axis=2) if img.shape[2] >= 3 else img[..., 0]


def discover(folder: str):
    """
    {pass: {"legend": [(name, rgb)], "idmap": path|None, "masks": {name: path}}}, reference path|None.
    """
    folder = fc.clean_folder_path(folder)
    if not os.path.isdir(folder):
        hint = (" - it was a kubakub scene render cache folder that is gone now (the cache size cap or a cleared "
                "cache removed it); queue again and the scene render recreates it") if "kuba_scene3d" in folder else ""
        raise ValueError(f"folder does not exist: {folder}{hint}")
    files = {}
    for f in glob.glob(os.path.join(folder, "*")) + glob.glob(os.path.join(folder, "png", "*")):
        files.setdefault(os.path.basename(f).lower(), []).append(f)
    passes = {}
    for f in sorted(glob.glob(os.path.join(folder, "*.txt"))):
        stem = os.path.splitext(os.path.basename(f))[0]
        legend = parse_legend(open(f, encoding="utf-8-sig").read())
        if not legend:
            continue
        name = re.sub(r"^(ids?|id_map|idmap)[_\-]?", "", stem, flags=re.I) or stem
        idmap = next((p for ext in (".png", ".exr", ".tif", ".tiff") for p in [os.path.join(folder, stem + ext)]
                      if os.path.isfile(p)), None)
        masks = {}
        for n, _ in legend:
            found = []
            for cand in (f"mask_{n}.exr", f"mask_{n}.png", f"{n}.exr", f"{n}.png"):
                # files in the folder itself before copies in png/
                found += sorted(files.get(cand.lower(), []), key=lambda p: (os.path.dirname(p) != folder, p))
            if found:
                masks[n] = found          # tried in this order (EXR may not be readable everywhere)
        passes[name] = {"legend": legend, "idmap": idmap, "masks": masks, "legend_file": f}
    ref = next((p for p in sorted(glob.glob(os.path.join(folder, "*")))
                if re.search(r"clay|beauty|shade|render", os.path.basename(p), re.I)
                and p.lower().endswith((".exr", ".png", ".jpg", ".tif", ".tiff"))), None)
    if not passes:
        raise ValueError(f"no ID legend (<pass>.txt with '<name> rgb r g b' lines) in {folder}")
    return passes, ref


# --------------------------------------------------------------------------
# ID map decoding
# --------------------------------------------------------------------------

def to_srgb(c):
    c = np.clip(np.asarray(c, float), 0, 1)
    return np.where(c <= 0.0031308, 12.92 * c, 1.055 * np.power(c, 1 / 2.4) - 0.055)


# the 16 bit value the generic decode path gives each 8 bit value (read_image: float32 v / 255, x 65535, rounded)
_Q16_OF_8BIT = np.clip(np.rint((np.arange(256, dtype=np.float32) / 255.0).astype(np.float64) * 65535), 0, 65535).astype(np.int64)


def _as_8bit(rgb):
    """uint8 values when every pixel is exactly float32(v) / 255 (an 8 bit file through read_image), else None."""
    if rgb.dtype != np.float32:
        return None
    v = rgb * np.float32(255)
    np.rint(v, out=v)
    np.clip(v, 0, 255, out=v)
    with np.errstate(invalid="ignore"):
        q8 = v.astype(np.uint8)
    return q8 if np.array_equal(q8.astype(np.float32) / np.float32(255), rgb) else None


def decode_idmap(img: np.ndarray, legend, encoding="auto", tol=1.5 / 255):
    """
    Label map (-1 = background) of an ID map. The legend is linear (renderer
    colours); 8/16 bit files are usually sRGB encoded, EXR linear. 'auto' takes
    whichever of as-is / sRGB matches more pixels. Anti-aliased edge pixels
    (no exact match, not background) get the label of the nearest labelled pixel.
    Returns (labels, encoding used, notes).
    """
    cols = np.stack([c for _, c in legend])
    options = {"linear": cols, "srgb": to_srgb(cols)}
    h, w = img.shape[:2]
    # work on the distinct colours (a few hundred in a flat ID map) instead of every pixel per legend entry
    q8 = _as_8bit(img[..., :3])
    if q8 is not None:
        # an 8 bit file (read_image: v / 255): 24 bit keys and a bincount, the same colours in the same order
        key = ((q8[..., 0].astype(np.int32) << 16) | (q8[..., 1].astype(np.int32) << 8) | q8[..., 2]).ravel()
        cnt = np.bincount(key, minlength=1 << 24)
        ukey = np.flatnonzero(cnt)
        ucount = cnt[ukey]
        lut = np.zeros(1 << 24, np.intp)
        lut[ukey] = np.arange(len(ukey))
        inv = lut[key]
        q16 = _Q16_OF_8BIT
        ucol = np.stack([q16[ukey >> 16], q16[(ukey >> 8) & 255], q16[ukey & 255]], 1) / 65535.0
    else:
        rgb = img[..., :3].astype(np.float64)
        q = np.clip(np.rint(rgb.reshape(-1, 3) * 65535), 0, 65535).astype(np.int64)
        key = (q[:, 0] << 32) | (q[:, 1] << 16) | q[:, 2]
        ukey, inv, ucount = np.unique(key, return_inverse=True, return_counts=True)
        ucol = np.stack([(ukey >> 32) & 0xFFFF, (ukey >> 16) & 0xFFFF, ukey & 0xFFFF], 1) / 65535.0

    def nearest(cs):
        lab = np.full(len(ucol), -1, np.int32)
        dist = np.full(len(ucol), np.inf)
        for a in range(0, len(ucol), 4096):            # chunks keep (colours x legend) small
            d = np.abs(ucol[a:a + 4096, None, :] - cs[None]).max(axis=2)
            k = d.argmin(axis=1)
            dk = d[np.arange(len(k)), k]
            ok = dk < tol
            lab[a:a + 4096][ok], dist[a:a + 4096][ok] = k[ok], dk[ok]
        return lab

    per_enc = {}
    if encoding == "auto":
        counts = {}
        for k, cs in options.items():
            per_enc[k] = nearest(cs)
            counts[k] = int(ucount[per_enc[k] >= 0].sum())
        encoding = max(counts, key=counts.get)
    ulab = per_enc.get(encoding)
    if ulab is None:
        ulab = nearest(options[encoding])
    labels = ulab[inv.ravel()].reshape(h, w).astype(np.int32)
    notes = []
    background = img[..., :3].max(axis=2).astype(np.float64) < 0.02
    edge = (labels < 0) & ~background
    # background next to an object keeps the object's half: fill the AA pixels only
    if edge.any():
        filled = fc.fill_unassigned(labels.copy())
        labels[edge] = filled[edge]
    seen = np.bincount(labels[labels >= 0], minlength=len(legend))   # one pass, not one compare per name
    missing = [n for i, (n, _) in enumerate(legend) if not seen[i]]
    if missing:
        notes.append(f"not visible in the ID map: {', '.join(missing)}")
    return labels, encoding, notes


# --------------------------------------------------------------------------
# build
# --------------------------------------------------------------------------

def _resize_mask(m, W, H):
    if m.shape == (H, W):
        return m
    return cv2.resize(m.astype(np.float32), (W, H), interpolation=cv2.INTER_AREA
                      if m.shape[1] > W else cv2.INTER_LINEAR)


def pass_masks(p, W=None, H=None, encoding="auto"):
    """
    Full-res float masks of a pass [(name, mask)], the decoded ID label map
    (at ID map size) and notes. Masks come from the mask files; names without
    a file are cut from the ID map (upscaled, nearest).
    """
    notes, out = [], []
    labels = None
    if p["idmap"]:
        labels, enc, n2 = decode_idmap(read_image(p["idmap"]), p["legend"], encoding)
        notes += [f"{os.path.basename(p['idmap'])}: legend read as {enc}"] + n2
    fallback = []
    for i, (name, _) in enumerate(p["legend"]):
        m = None
        for k, path in enumerate(p["masks"].get(name, [])):
            try:
                m = mask_from_image(read_image(path))
                if k:
                    fallback.append(name)
                break
            except ValueError as e:
                last = str(e)
        if m is not None:
            pass
        elif name in p["masks"]:
            raise ValueError(f"{name}: no readable mask file ({last})")
        elif labels is not None:
            m = (labels == i).astype(np.float32)
            notes.append(f"{name}: no mask file, cut from the ID map")
        else:
            notes.append(f"{name}: no mask file and no ID map, skipped")
            continue
        if W is None:
            H, W = m.shape
        if m.shape != (H, W):
            m = _resize_mask(m, W, H)
        out.append((name, m))
    if fallback:
        notes.append(f"EXR not readable here, used the PNG copies for: {', '.join(fallback)} "
                     "(8 bit edges; FreeImage.dll makes the EXRs readable)")
    return out, labels, (W, H), notes


def _decode_pass(p, encoding="auto"):
    """(labels at ID map size, notes) of a pass's ID map."""
    labels, enc, notes = decode_idmap(read_image(p["idmap"]), p["legend"], encoding)
    return labels, [f"{os.path.basename(p['idmap'])}: legend read as {enc}, ID map only (labels used directly)"] + notes


def pass_labels(p, W=None, H=None, encoding="auto", decoded=None):
    """
    Fast path for a pass that has an ID map and no mask files (e.g. kubakub scene render): the ID map
    already shows what the camera sees, so its label map is the answer - no per-name masks, no
    depth order. decoded: a _decode_pass result made earlier. Returns (names, labels at W x H, (W, H), notes).
    """
    labels, notes = decoded if decoded is not None else _decode_pass(p, encoding)
    if W is None:
        H, W = labels.shape
    if labels.shape != (H, W):
        labels = cv2.resize(labels, (W, H), interpolation=cv2.INTER_NEAREST)
    return [n for n, _ in p["legend"]], labels, (W, H), notes


def _label_slices(labels, n):
    """Bounding-box slices per label 0..n-1 (None where absent) - avoids n full-image compares."""
    from scipy import ndimage
    sl = ndimage.find_objects(labels + 1, max_label=n)
    return list(sl) + [None] * (n - len(sl))


def _label_masks(labels, n):
    """[labels == j for j in 0..n-1], each compared inside the label's bounding box only."""
    out = []
    for j, sl in enumerate(_label_slices(labels, n)):
        m = np.zeros(labels.shape, bool)
        if sl is not None:
            m[sl] = labels[sl] == j
        out.append(m)
    return out


def depth_order(masks, idlabels, W, H):
    """
    Back-to-front order of overlapping masks, learned from the ID map: wins[a, b]
    = pixels where both cover and the ID map shows a. Masks never shown go to
    the back. Returns (order, wins).
    """
    n = len(masks)
    ids = cv2.resize(idlabels.astype(np.int32), (W, H), interpolation=cv2.INTER_NEAREST) if \
        idlabels.shape != (H, W) else idlabels
    hard = [m > 0.5 for _, m in masks]
    wins = np.zeros((n, n), np.int64)
    for b in range(n):
        # pixels mask b covers but the ID map shows another object a: a is in front of b there
        shown = ids[hard[b] & (ids >= 0) & (ids != b)]
        if shown.size:
            wins[:, b] += np.bincount(shown, minlength=n)
    score = (wins > wins.T).sum(axis=1) - (wins < wins.T).sum(axis=1)
    visible = np.bincount(ids[ids >= 0], minlength=n)
    order = sorted(range(n), key=lambda i: (visible[i] > 0, score[i], -int(hard[i].sum())))
    return order, wins


def build(folder, regions_pass="elements", tag_passes="*", width=None, height=None, overlap="id_map",
          split_parts="*", min_region_area=400, merge_small_regions=True, encoding="auto", scope=None):
    """
    Returns {"labels", "atlas", "scope", "reference", "idmaps", "passes", "size"}.
    """
    passes, ref_path = discover(folder)
    if regions_pass not in passes:
        raise ValueError(f"regions_pass '{regions_pass}' not found; passes: {', '.join(passes)}")
    if overlap not in OVERLAPS:
        raise ValueError(f"overlap must be one of {OVERLAPS}")
    # the ID-map-only passes (regions and tags) are decoded side by side: numpy / OpenCV release the GIL
    tag_pats = [p.strip() for p in (tag_passes or "").replace(",", " ").split()]
    used = [regions_pass] + [n for n in passes if n != regions_pass
                             and any(re.fullmatch(fnmatch_to_re(t), n, re.I) for t in tag_pats)]
    only_id = [n for n in used if passes[n]["idmap"] and not passes[n]["masks"]]
    with ThreadPoolExecutor(max(1, min(len(only_id), os.cpu_count() or 1, 6))) as ex:
        decoded = {n: ex.submit(_decode_pass, passes[n], encoding) for n in only_id}
        try:
            return _build(passes, ref_path, regions_pass, tag_pats, width, height, overlap, split_parts,
                          min_region_area, merge_small_regions, encoding, scope, decoded)
        finally:
            for f in decoded.values():
                f.cancel()


def _build(passes, ref_path, regions_pass, tag_pats, width, height, overlap, split_parts, min_region_area,
           merge_small_regions, encoding, scope, decoded):
    notes = []
    rp = passes[regions_pass]
    fast = bool(rp["idmap"]) and not rp["masks"]
    if fast:
        names, visible, (W, H), n1 = pass_labels(rp, width, height, encoding, decoded[regions_pass].result())
        masks, idl = [(n, None) for n in names], None
    else:
        masks, idl, (W, H), n1 = pass_masks(rp, width, height, encoding)
    notes += n1
    if width and height and (W, H) != (width, height):
        raise ValueError("internal size mismatch")
    if not masks:
        raise ValueError(f"pass '{regions_pass}' has no masks")

    # one visible label per pixel
    hard = [] if fast else [m > 0.5 for _, m in masks]
    if fast:
        pass
    elif overlap == "id_map" and idl is not None:
        order, wins = depth_order(masks, idl, W, H)
        hidden = [masks[i][0] for i in range(len(masks))
                  if not (idl == i).any() and hard[i].any()]
        if hidden:
            notes.append(f"hidden behind other elements in the ID map (painted at the back): "
                         f"{', '.join(hidden)}; overlap = smaller_wins brings them to the front")
        notes.append("depth order (back to front): " + ", ".join(masks[i][0] for i in order))
    else:
        if overlap == "id_map":
            notes.append("no ID map for the regions pass: smaller mask wins")
        order = sorted(range(len(masks)), key=lambda i: -int(hard[i].sum()))
    if not fast:
        visible = np.full((H, W), -1, np.int32)
        for i in order:
            visible[hard[i]] = i

    # regions: one label per element, or per connected part of it (split_parts patterns)
    split_pats = [t.strip().lower() for t in (split_parts or "").replace(",", " ").split()]
    labels = np.full((H, W), -1, np.int32)
    meta = []
    slices = _label_slices(visible, len(masks))
    for i, (name, _) in enumerate(masks):
        sl = slices[i]
        if sl is None:
            notes.append(f"{name}: fully covered by other elements, no region")
            continue
        vis = visible[sl] == i                      # work inside the element's bounding box
        lab_sl = labels[sl]
        if not any(re.fullmatch(fnmatch_to_re(t), name.lower()) for t in split_pats):
            lab_sl[vis] = len(meta)
            meta.append({"name": name, "group": name, "source": f"{regions_pass}:{name}", "own_mask": name})
            continue
        n, cc, stats, _ = cv2.connectedComponentsWithStats(vis.astype(np.uint8), connectivity=8)
        areas = stats[1:, cv2.CC_STAT_AREA]
        big = np.flatnonzero(areas >= max(1, min_region_area))
        boxes = np.asarray([[stats[k + 1, 0], stats[k + 1, 1], stats[k + 1, 0] + stats[k + 1, 2],
                             stats[k + 1, 1] + stats[k + 1, 3]] for k in big], np.int64).reshape(-1, 4)
        # parts big enough get numbers in reading order; slivers (anti-aliased seams, specks)
        # get their own label and are merged into a neighbour by the small-region pass
        remap = np.full(n, -1, np.int64)
        for rank, j in enumerate(fc.reading_order(boxes) if len(big) else []):
            remap[big[j] + 1] = len(meta)
            meta.append({"name": name if len(big) == 1 else f"{name}_{rank + 1:02d}", "group": name,
                         "source": f"{regions_pass}:{name}", "own_mask": name})
        for k in np.flatnonzero(areas < max(1, min_region_area)):
            remap[k + 1] = len(meta)
            meta.append({"name": f"{name}_s{k + 1:04d}", "group": name, "source": f"{regions_pass}:{name}",
                         "own_mask": name})
        lab_sl[vis] = remap[cc[vis]]

    # tag passes: every name of the pass tags the regions it covers by half or more
    tag_masks = []
    idmaps_out = {regions_pass: visible}
    for pname, p in passes.items():
        if pname == regions_pass or not any(re.fullmatch(fnmatch_to_re(t), pname, re.I) for t in tag_pats):
            continue
        if p["idmap"] and not p["masks"]:
            tnames, lab, _, n2 = pass_labels(p, W, H, encoding, decoded[pname].result())
            notes += n2
            tag_masks += zip(tnames, _label_masks(lab, len(tnames)))
        else:
            tmasks, tidl, _, n2 = pass_masks(p, W, H, encoding)
            notes += n2
            lab = np.full((H, W), -1, np.int32)
            for j, (tname, tm) in enumerate(tmasks):
                tag_masks.append((tname, tm > 0.5))
                lab[tm > 0.5] = j
        idmaps_out[pname] = lab
    sc = None if scope is None else np.asarray(scope, bool)
    labels, atlas, sc = fc._finish_atlas(labels, meta, notes, tag_masks, sc, f"id_maps:{regions_pass}",
                                         int(min_region_area), merge_small_regions, with_scope=True)
    atlas["passes"] = {k: [n for n, _ in v["legend"]] for k, v in passes.items()}

    reference = None
    if ref_path:
        img = read_image(ref_path)[..., :3]
        if ref_path.lower().endswith(".exr"):      # linear, unbounded: simple tone map for viewing
            img = img / (1.0 + img)
            img = to_srgb(img / max(float(np.percentile(img, 99.5)), 1e-6)).astype(np.float32)
        if img.shape[:2] != (H, W):
            img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        reference = np.clip(img, 0, 1).astype(np.float32)
    return {"labels": labels, "atlas": atlas, "scope": sc, "reference": reference, "idmaps": idmaps_out,
            "passes": passes, "size": (W, H), "reference_path": ref_path}


def fnmatch_to_re(pat: str) -> str:
    return re.escape(pat).replace(r"\*", ".*").replace(r"\?", ".")
