"""
scene_ids.py

ID passes for kubakub regions from id maps from a Blender export (blender_export.py): the face index each
pixel sees plus per-face data. Every pass is a labelling of faces or pixels; the result is written as
ids_<pass>.png + ids_<pass>.txt (the Houdini folder contract in idmaps.py), so no re-render is needed
when a threshold changes. numpy + scipy + opencv only, no ComfyUI imports (tests/test_scene3d.py).

Passes:
    objects / materials / collections   from the file (written when more than one is visible)
    parts     loose parts (faces connected through shared, welded vertices)
    planes    connected coplanar face patches (normal angle + plane offset tolerance)
    shelves   the depth steps of the facade found automatically (peaks of the pixel depth histogram
              along the facade normal), named by their offset from the main plane: shelf_+0.35m
    layers    depth of every pixel relative to the main facade plane in fixed bands (recess / facade ...)
    elements  architecture (architecture.py): windows, window_frames, columns, cornices, relief, wall, roof
    sections  left / centre / right ... where the wall steps forward or back (else thirds)
    floors    ground_floor, floor_1 ... from the window rows and the cornice belts; roof above the top belt
    facing    front / top / underside / side / back relative to the main plane and world up (Z)
"""

from __future__ import annotations

import json
import os
import re
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

PASSES = ("elements", "sections", "floors", "shelves", "layers", "facing", "planes", "parts", "objects", "materials", "collections")
LAYER_NAMES = ("deep_recess", "recess", "facade", "relief", "projection")


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------

def load(folder):
    info = json.load(open(os.path.join(folder, "scene.json"), encoding="utf-8"))
    m = np.load(os.path.join(folder, "mesh.npz"))
    mesh = {k: m[k] for k in m.files}
    return {"info": info, "mesh": mesh, "faceid": np.load(os.path.join(folder, "faceid.npy")),
            "position": np.load(os.path.join(folder, "position.npy")),
            "normal": np.load(os.path.join(folder, "normal.npy")).astype(np.float32)}


def camera_frame(info):
    """Camera position and unit forward / up / right in world space (the camera object may be scaled)."""
    M = np.array(info["camera"]["matrix_world"], float)
    R = M[:3, :3] / np.linalg.norm(M[:3, :3], axis=0)
    return M[:3, 3], -R[:, 2], R[:, 1], R[:, 0]


def depth_maps(scene):
    """(camera depth along the view axis, distance to the camera) in metres, 0 on background."""
    pos, fid = scene["position"], scene["faceid"]
    c, fwd, _, _ = camera_frame(scene["info"])
    d = pos - c
    depth = (d @ fwd).astype(np.float32)
    dist = np.linalg.norm(d, axis=-1).astype(np.float32)
    bg = fid == 0
    depth[bg] = 0
    dist[bg] = 0
    return depth, dist


# --------------------------------------------------------------------------
# mesh topology
# --------------------------------------------------------------------------

def weld(vert, tol):
    """Vertex -> welded vertex id (positions quantized to tol)."""
    q = np.floor(vert / tol + 0.5).astype(np.int64)
    _, inv = np.unique(q, axis=0, return_inverse=True)
    return inv.ravel()


def face_loops(mesh):
    lt = mesh["loop_total"].astype(np.int64)
    start = np.concatenate([[0], np.cumsum(lt)[:-1]])
    face_of_loop = np.repeat(np.arange(len(lt)), lt)
    k = np.arange(len(face_of_loop))
    nxt = k + 1
    last = start + lt - 1
    nxt[last] = start
    return face_of_loop, nxt


def adjacent_pairs(mesh, tol):
    """Pairs of faces sharing an edge (welded vertices), each pair once."""
    wv = weld(mesh["vert"], tol)[mesh["loop_vert"]]
    face_of_loop, nxt = face_loops(mesh)
    a, b = wv, wv[nxt]
    key = np.minimum(a, b) * (wv.max() + 1) + np.maximum(a, b)
    order = np.argsort(key, kind="stable")
    ks, fs = key[order], face_of_loop[order]
    same = ks[1:] == ks[:-1]
    p = np.stack([fs[:-1][same], fs[1:][same]], 1)
    p = p[p[:, 0] != p[:, 1]]
    return np.unique(np.sort(p, 1), axis=0) if len(p) else p.reshape(0, 2)


def components(n, pairs):
    if len(pairs) == 0:
        return np.arange(n)
    g = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n))
    return connected_components(g, directed=False)[1]


def loose_parts(mesh, tol):
    """Face label per loose part (faces sharing a welded vertex)."""
    F = len(mesh["loop_total"])
    wv = weld(mesh["vert"], tol)[mesh["loop_vert"]]
    face_of_loop, _ = face_loops(mesh)
    g = coo_matrix((np.ones(len(wv)), (face_of_loop, wv)), shape=(F, int(wv.max()) + 1))
    n = F + g.shape[1]
    big = coo_matrix((np.ones(len(wv)), (face_of_loop, F + wv)), shape=(n, n))
    return connected_components(big, directed=False)[1][:F]


def coplanar_patches(mesh, pairs, angle_deg=3.0, dist_m=0.02):
    """Face label per connected coplanar patch."""
    n, c = mesh["normal"], mesh["centre"]
    F = len(n)
    if len(pairs) == 0:
        return np.arange(F)
    i, j = pairs[:, 0], pairs[:, 1]
    cos = (n[i] * n[j]).sum(1)
    off = np.abs(((c[j] - c[i]) * n[i]).sum(1))
    keep = (cos >= np.cos(np.radians(angle_deg))) & (off <= dist_m)
    return components(F, pairs[keep])


# --------------------------------------------------------------------------
# main plane, layers, facing
# --------------------------------------------------------------------------

def _most_common_row(q):
    """
    Bool mask of the rows of the int (N, 3) array q equal to its most common row (ties: the lexicographically
    smallest, as np.unique(axis=0) + argmax). One int64 key per row and a bincount instead of the row sort.
    """
    lo = q.min(0).astype(np.int64)
    span = q.max(0).astype(np.int64) - lo + 1
    if int(span[0]) * int(span[1]) * int(span[2]) > 1 << 26:   # sparse keys: too big for a bincount, sort
        _, inv, counts = np.unique(q, axis=0, return_inverse=True, return_counts=True)
        return inv.ravel() == counts.argmax()
    d = q.astype(np.int64) - lo
    key = (d[:, 0] * span[1] + d[:, 1]) * span[2] + d[:, 2]   # same order as the rows, lexicographic
    return key == np.bincount(key).argmax()


def main_plane(scene, bin_m=0.01):
    """
    (point, unit normal towards the camera) of the facade: the normal is the dominant direction of the
    camera-facing pixels, the offset the depth shelf along it where most of those pixels lie (a
    recessed wall or one big patch does not win over the many pieces of the actual facade surface).
    """
    fid = scene["faceid"]
    fg = fid > 0
    cam, fwd, _, _ = camera_frame(scene["info"])
    n = scene["normal"][fg]
    p = scene["position"][fg]
    front = (n @ -fwd) > 0.5
    if front.sum() < 16:
        raise ValueError("no surface faces the camera")
    n, p = n[front], p[front]
    # dominant normal: the most common quantized direction, refined by the mean of its members
    q = np.round(n * 20).astype(np.int32)
    members = _most_common_row(q)
    nrm = n[members].mean(0)
    nrm /= np.linalg.norm(nrm)
    if nrm @ -fwd < 0:
        nrm = -nrm
    s = p[members] @ nrm
    lo, hi = float(s.min()), float(s.max())
    hist, edges = np.histogram(s, bins=max(1, int(np.ceil((hi - lo) / bin_m))), range=(lo, hi + 1e-9))
    k = int(np.argmax(np.convolve(hist, [1, 2, 1], mode="same")))
    peak = (edges[k] + edges[k + 1]) / 2
    near = np.abs(s - peak) <= 1.5 * bin_m
    if near.any():
        peak = float(np.median(s[near]))          # refine inside the peak bin
    return nrm * peak, nrm


def layer_labels(scene, pt, nrm, bands=(-0.5, -0.05, 0.05, 0.5)):
    """Per-pixel layer index (-1 background) from the signed distance to the main plane (m, + = towards camera)."""
    s = (scene["position"] - pt) @ nrm
    lab = np.digitize(s, np.asarray(bands, float)).astype(np.int32)
    lab[scene["faceid"] == 0] = -1
    return lab, s.astype(np.float32)


def shelf_labels(signed, fg, min_sep_m=0.05, min_share=0.002, max_shelves=16, bin_m=0.005):
    """
    Per-pixel shelf index (-1 background) and shelf offsets (m): peaks of the histogram of the signed
    distance to the main plane (at least min_sep_m apart, each holding min_share of the pixels);
    every pixel goes to the nearest shelf.
    """
    s = signed[fg]
    lo, hi = float(s.min()), float(s.max())
    nb = max(1, int(np.ceil((hi - lo) / bin_m)))
    hist, edges = np.histogram(s, bins=nb, range=(lo, hi + 1e-9))
    centres = (edges[:-1] + edges[1:]) / 2
    sm = np.convolve(hist, np.ones(3) / 3, mode="same")
    peaks = [i for i in np.argsort(-sm) if sm[i] > 0]
    chosen, sep = [], max(1, int(round(min_sep_m / bin_m)))
    for i in peaks:
        if len(chosen) >= max_shelves:
            break
        if all(abs(i - j) >= sep for j in chosen):
            chosen.append(i)
    offs = np.sort(centres[chosen])
    lab = np.full(signed.shape, -1, np.int32)
    lab[fg] = np.abs(s[:, None] - offs[None, :]).argmin(1)
    counts = np.bincount(lab[fg], minlength=len(offs))
    keep = counts >= min_share * fg.sum()
    if not keep.all():               # drop tiny shelves, reassign their pixels
        offs = offs[keep]
        lab[fg] = np.abs(s[:, None] - offs[None, :]).argmin(1)
    # report each shelf at the median of its pixels (not the histogram bin centre)
    offs = np.array([float(np.median(s[lab[fg] == k])) for k in range(len(offs))])
    return lab, offs


def facing_labels(mesh, nrm, up=(0.0, 0.0, 1.0), cos_front=0.7, cos_up=0.7):
    """Per-face: 0 front, 1 top, 2 underside, 3 side, 4 back (relative to the main plane normal and world up)."""
    n = mesh["normal"]
    up = np.asarray(up, float)
    f = n @ nrm
    u = n @ up
    lab = np.full(len(n), 3, np.int32)
    lab[u >= cos_up] = 1
    lab[u <= -cos_up] = 2
    lab[f >= cos_front] = 0
    lab[f <= -cos_front] = 4
    return lab


FACING_NAMES = ("front", "top", "underside", "side", "back")


# --------------------------------------------------------------------------
# minimum size in metres (what the crowd can still read)
# --------------------------------------------------------------------------

def _components(lab):
    """Connected pieces of every label (8-connected, inside each label's bounding box) -> comp map, comp -> label."""
    from scipy import ndimage
    comp = np.full(lab.shape, -1, np.int64)
    owner = []
    n = int(lab.max()) + 1 if (lab >= 0).any() else 0
    for li, sl in enumerate(ndimage.find_objects(lab + 1, max_label=n)):
        if sl is None:
            continue
        m = lab[sl] == li
        cc, k = ndimage.label(m, structure=np.ones((3, 3), bool))
        sub = comp[sl]
        sub[m] = cc[m] - 1 + len(owner)
        owner += [li] * k
    return comp, np.asarray(owner, np.int64)


def merge_small(lab, position, nrm, min_m, iterations=4):
    """
    Pieces whose largest extent across the wall (horizontal along the wall, vertical) is below min_m
    take the label of the neighbour they share the longest border with. Returns (labels, merged count).
    """
    from scipy import ndimage
    if min_m <= 0 or not (lab >= 0).any():
        return lab, 0
    lab = lab.copy()
    up = np.array([0.0, 0.0, 1.0])
    ex = np.cross(up, nrm)
    if np.linalg.norm(ex) < 1e-6:                   # facade normal points up: use any horizontal axis
        ex = np.array([1.0, 0.0, 0.0])
    ex /= np.linalg.norm(ex)
    ez = up - (up @ nrm) * nrm
    ez = ez / np.linalg.norm(ez) if np.linalg.norm(ez) > 1e-6 else np.cross(nrm, ex)
    a, b = position @ ex, position @ ez
    total = 0
    for _ in range(iterations):
        comp, owner = _components(lab)
        nc = len(owner)
        if nc < 2:
            break
        # extents per piece: one sort by piece id, then reduceat (every id 0..nc-1 is present)
        fgm = comp >= 0
        cflat = comp[fgm]
        order = np.argsort(cflat, kind="stable")
        starts = np.flatnonzero(np.r_[True, cflat[order][1:] != cflat[order][:-1]])
        ext = np.zeros(nc)
        for vals in (a[fgm][order], b[fgm][order]):
            ext = np.maximum(ext, np.maximum.reduceat(vals, starts) - np.minimum.reduceat(vals, starts))
        small = ext < min_m
        if not small.any():
            break
        # border lengths between neighbouring pieces (4-neighbour pixel pairs)
        pairs = []
        for p, q in ((comp[:, :-1], comp[:, 1:]), (comp[:-1, :], comp[1:, :])):
            m = (p != q) & (p >= 0) & (q >= 0)
            pairs.append(np.stack([p[m], q[m]], 1))
        pr = np.concatenate(pairs)
        if not len(pr):
            break
        pr = np.concatenate([pr, pr[:, ::-1]])
        keys, cnt = np.unique(pr[:, 0] * nc + pr[:, 1], return_counts=True)
        src, dst = keys // nc, keys % nc
        sel = small[src]
        src, dst, cnt = src[sel], dst[sel], cnt[sel]
        cnt = cnt + np.where(small[dst], 0, cnt.max() + 1)   # prefer a neighbour that is big enough
        order = np.lexsort((-cnt, src))
        src, dst = src[order], dst[order]
        first = np.r_[True, src[1:] != src[:-1]]
        target = np.arange(nc)
        target[src[first]] = dst[first]
        new_label = owner[target]
        changed = small & (new_label != owner)
        if not changed.any():
            break
        m = comp >= 0
        lab[m] = np.where(changed[comp[m]], new_label[comp[m]], lab[m])
        total += int(changed.sum())
    return lab, total


# --------------------------------------------------------------------------
# writing
# --------------------------------------------------------------------------

def safe_name(s, fallback):
    s = re.sub(r"[^\w.+\-]+", "_", str(s)).strip("_")
    return s or fallback


def distinct_colors(n, seed=0):
    """n distinct 8-bit RGB colours, none black, deterministic."""
    out, seen = [], set()
    k = seed
    while len(out) < n:
        k += 1
        v = (k * 2654435761) & 0xFFFFFF
        c = ((v >> 16) & 255, (v >> 8) & 255, v & 255)
        if sum(c) < 120 or c in seen:
            continue
        seen.add(c)
        out.append(c)
    return np.array(out, np.uint8).reshape(-1, 3)


def pixel_labels_from_faces(faceid, face_labels):
    lab = np.full(faceid.shape, -1, np.int64)
    fg = faceid > 0
    lab[fg] = face_labels[faceid[fg].astype(np.int64) - 1]
    return lab


def write_pass(folder, name, pix_labels, names=None, prefix=None):
    """Relabel by visible area (largest first), write ids_<name>.png/.txt. Returns [(name, pixels)]."""
    fg = pix_labels >= 0
    ids, counts = np.unique(pix_labels[fg], return_counts=True)
    order = np.argsort(-counts, kind="stable")
    ids, counts = ids[order], counts[order]
    lut = np.full(int(pix_labels.max()) + 2 if fg.any() else 1, -1, np.int64)
    lut[ids] = np.arange(len(ids))
    new = np.where(fg, lut[np.where(fg, pix_labels, 0)], -1)
    cols = distinct_colors(len(ids), seed=len(name))
    img = np.zeros(pix_labels.shape + (3,), np.uint8)
    img[fg] = cols[new[fg]]
    cv2.imwrite(os.path.join(folder, f"ids_{name}.png"), img[..., ::-1])
    used, lines = set(), []
    for k, (i, c) in enumerate(zip(ids, counts)):
        base = safe_name(names[i], f"{prefix or name}_{k + 1:03d}") if names is not None else f"{prefix or name}_{k + 1:03d}"
        nm, s = base, 2
        while nm in used:
            nm, s = f"{base}_{s}", s + 1
        used.add(nm)
        r, g, b = (int(v) for v in cols[k])
        lines.append((nm, int(c), f"{nm} {r} {g} {b}"))
    with open(os.path.join(folder, f"ids_{name}.txt"), "w", encoding="utf-8") as f:
        f.write(f"// {name}: {len(lines)} ids, 8 bit colours of ids_{name}.png, largest visible area first\n")
        f.write("\n".join(line for _, _, line in lines) + "\n")
    return [(nm, c) for nm, c, _ in lines], img


def build(folder, passes=PASSES, angle_deg=3.0, dist_m=0.02, layer_bands=(-0.5, -0.05, 0.05, 0.5),
          shelf_sep_m=0.05, weld_m=1e-4, out=None, min_size_m=0.0):
    """Compute and write the ID passes. Returns a summary dict (+ preview images per pass)."""
    out = out or folder
    os.makedirs(out, exist_ok=True)
    scene = load(folder)
    info, mesh, fid = scene["info"], scene["mesh"], scene["faceid"]
    summary = {"passes": {}, "notes": []}
    previews = {}
    pairs = adjacent_pairs(mesh, weld_m)
    planes = coplanar_patches(mesh, pairs, angle_deg, dist_m)
    pt, nrm = main_plane(scene)
    summary["main_plane"] = {"point": pt.round(4).tolist(), "normal": nrm.round(5).tolist()}
    arch = None
    jobs = []                                       # (pass, labels, names), in pass order
    for p in passes:
        if p in ("elements", "sections", "floors"):
            if arch is None:
                try:
                    from . import architecture
                except ImportError:                     # loaded as a plain module (tests)
                    import architecture
                arch = architecture.analyse(scene, pt, nrm, float(mesh["vert"][:, 2].min()))
                summary["architecture"] = arch["info"]
            lab, names = arch[p]
        elif p == "planes":
            lab, names = pixel_labels_from_faces(fid, planes), None
        elif p == "parts":
            lab, names = pixel_labels_from_faces(fid, loose_parts(mesh, weld_m)), None
        elif p == "layers":
            lab, _ = layer_labels(scene, pt, nrm, layer_bands)
            names = list(LAYER_NAMES) if len(layer_bands) == 4 else [f"layer_{i}" for i in range(len(layer_bands) + 1)]
        elif p == "shelves":
            signed = (scene["position"] - pt) @ nrm
            lab, offs = shelf_labels(signed, fid > 0, min_sep_m=shelf_sep_m)
            names = [f"shelf_{round(float(o), 2) + 0.0:+.2f}m" for o in offs]
            summary["shelves_m"] = [round(float(o), 3) for o in offs]
        elif p == "facing":
            lab, names = pixel_labels_from_faces(fid, facing_labels(mesh, nrm)), list(FACING_NAMES)
        elif p in ("objects", "materials", "collections"):
            key = {"objects": "obj", "materials": "mat", "collections": "col"}[p]
            lab = pixel_labels_from_faces(fid, mesh[key].astype(np.int64))
            names = [n or f"no_{p[:-1]}" for n in info[p]]
            if len(np.unique(lab[lab >= 0])) < 2:
                summary["notes"].append(f"{p}: only one visible, pass skipped")
                continue
        else:
            raise ValueError(f"unknown pass {p}")
        jobs.append((p, lab, names))

    def finish(p, lab, names):
        merged = 0
        if min_size_m > 0 and p not in ("elements", "sections", "floors"):   # those clean up themselves (and keep the gaps between arches)
            lab, merged = merge_small(lab, scene["position"], nrm, min_size_m)
        ids, img = write_pass(out, p, lab, names)
        return {"ids": len(ids), "largest": ids[:5], "merged_small": merged}, img

    # the passes are independent (merge_small copies its labels, every pass writes its own files):
    # numpy and OpenCV release the GIL, so threads run them side by side
    with ThreadPoolExecutor(max(1, min(len(jobs), os.cpu_count() or 1, 6))) as ex:
        results = [ex.submit(finish, *job) for job in jobs]
        for (p, _, _), fut in zip(jobs, results):
            summary["passes"][p], previews[p] = fut.result()
    depth, dist = depth_maps(scene)
    np.save(os.path.join(out, "depth_m.npy"), depth)
    fg = fid > 0
    summary["depth_m"] = [float(depth[fg].min()), float(depth[fg].max())] if fg.any() else [0.0, 0.0]
    summary["coverage"] = float(fg.mean())
    summary["min_size_m"] = float(min_size_m)
    with open(os.path.join(out, "ids_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1)
    return summary, previews, scene, depth
