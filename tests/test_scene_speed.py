"""
Model free test for the fast paths of the scene ID tools (kubakub/scene3d, kubakub/idmaps.py).
Each fast path is checked against a plain reference on synthetic data; the results must be identical.
- architecture._split_necks: bounding-box crops vs the full-image version
- scene_ids._most_common_row: bincount key vs np.unique(axis=0)
- scene_ids.build: passes finished in threads vs one after another (the same files, byte for byte)
- idmaps.decode_idmap: 24 bit path for 8 bit ID maps vs the generic 16 bit path; tag masks per bounding box
- idmaps.build: ID maps decoded in threads vs one after another

Run from the ComfyUI_windows_portable folder:
    python_embeded\\python.exe ComfyUI\\custom_nodes\\ComfyUI_KubaNodes\\tests\\test_scene_speed.py
"""

import filecmp
import json
import os
import shutil
import sys
import tempfile
from concurrent.futures import Future

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PACK = os.path.dirname(HERE)
sys.path.insert(0, PACK)
sys.path.insert(0, os.path.join(PACK, "kubakub"))
sys.path.insert(0, os.path.join(PACK, "kubakub", "scene3d"))
import architecture as A  # noqa: E402
import idmaps as im  # noqa: E402
import scene_ids as si  # noqa: E402

failures = []


def check(name, cond, detail=""):
    print(("ok   " if cond else "FAIL ") + name + (f"  ({detail})" if detail and not cond else ""))
    if not cond:
        failures.append(name)


class SerialExecutor:
    """Stand-in for ThreadPoolExecutor that runs every job at submit: the one-after-another reference."""

    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def submit(self, fn, *args, **kwargs):
        f = Future()
        try:
            f.set_result(fn(*args, **kwargs))
        except BaseException as e:  # noqa: BLE001
            f.set_exception(e)
        return f


def same(a, b):
    if isinstance(a, np.ndarray):
        return isinstance(b, np.ndarray) and a.dtype == b.dtype and np.array_equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and list(a) == list(b) and all(same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    return a == b


rng = np.random.default_rng(7)

# --------------------------------------------------------------------------
# architecture._split_necks
# --------------------------------------------------------------------------


def split_necks_ref(mask, r):
    """The full-image version: one full-size mask per part."""
    n, cc = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    for i in range(1, n):
        comp = (cc == i).astype(np.uint8)
        seeds_n, seeds = cv2.connectedComponents(cv2.erode(comp, A._kernel(2 * r + 1, 2 * r + 1)), connectivity=8)
        if seeds_n <= 2:
            yield comp.astype(bool)
            continue
        inv = (seeds == 0).astype(np.uint8)
        _, lab = cv2.distanceTransformWithLabels(inv, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_CCOMP)
        lut = np.zeros(int(lab.max()) + 1, np.int64)
        lut[lab[seeds > 0]] = seeds[seeds > 0]
        part = lut[lab] * comp
        for k in range(1, seeds_n):
            yield cv2.erode((part == k).astype(np.uint8), A._kernel(3, 3)).astype(bool)


def arcade_mask(H=180, W=260):
    """Arches joined by thin necks, some cut by the image border, plus loose blocks."""
    m = np.zeros((H, W), np.uint8)
    for k in range(6):
        x = -10 + k * 48
        cv2.rectangle(m, (x, 40), (x + 30, 120), 1, -1)
        cv2.circle(m, (x + 15, 40), 15, 1, -1)
        cv2.rectangle(m, (x + 30, 90), (x + 48, 90 + k % 3), 1, -1)       # necks 1..3 px
    cv2.rectangle(m, (0, 150), (40, H - 1), 1, -1)                        # touches two borders
    cv2.rectangle(m, (W - 25, 0), (W - 1, 30), 1, -1)
    cv2.ellipse(m, (200, 160), (40, 12), 10, 0, 360, 1, -1)
    return m


def blob_mask(H=200, W=240, thr=0.55):
    """Random smooth blobs: irregular parts, holes, necks of every width."""
    noise = cv2.GaussianBlur(rng.random((H, W)).astype(np.float32), (0, 0), 6)
    noise = (noise - noise.min()) / (noise.max() - noise.min())
    return (noise > thr).astype(np.uint8)


cases = [("arcade", arcade_mask())] + [(f"blobs {i}", blob_mask(thr=t)) for i, t in enumerate((0.5, 0.55, 0.6))]
for name, mask in cases:
    for r in (1, 2, 4, 7):
        ref = list(split_necks_ref(mask, r))
        got = []
        for y0, x0, crop in A._split_necks(mask, r):
            full = np.zeros(mask.shape, bool)
            full[y0:y0 + crop.shape[0], x0:x0 + crop.shape[1]] = crop
            got.append(full)
        ok = len(ref) == len(got) and all(np.array_equal(a, b) for a, b in zip(ref, got))
        check(f"split_necks crops = full image ({name}, r={r}, {len(ref)} parts)", ok,
              f"{len(ref)} vs {len(got)} parts")

# --------------------------------------------------------------------------
# scene_ids._most_common_row
# --------------------------------------------------------------------------


def most_common_ref(q):
    _, inv, counts = np.unique(q, axis=0, return_inverse=True, return_counts=True)
    return inv.ravel() == counts.argmax()


n = rng.normal(size=(50000, 3)).astype(np.float32)
n /= np.linalg.norm(n, axis=1, keepdims=True)
qs = {"normals": np.round(n * 20).astype(np.int32),
      "small ints": rng.integers(-3, 4, (20000, 3)).astype(np.int32),
      "sparse (sort fallback)": rng.integers(-10 ** 6, 10 ** 6, (5000, 3)).astype(np.int32)}
tie = np.array([[1, 0, 0]] * 5 + [[0, 5, 0]] * 5 + [[0, 0, -2]] * 5 + [[-1, 9, 9]] * 4, np.int32)
qs["three-way tie"] = tie[rng.permutation(len(tie))]
qs["one row"] = np.array([[3, -4, 5]], np.int32)
for name, q in qs.items():
    check(f"most common row = np.unique(axis=0) ({name})", np.array_equal(si._most_common_row(q), most_common_ref(q)))

# --------------------------------------------------------------------------
# a synthetic facade export (the blender_export.py format) for the build checks
# --------------------------------------------------------------------------


def box_faces(x0, x1, y0, y1, z0, z1, verts):
    base = len(verts)
    for x in (x0, x1):
        for y in (y0, y1):
            for z in (z0, z1):
                verts.append((x, y, z))
    v = lambda i, j, k: base + i * 4 + j * 2 + k  # noqa: E731
    return [((v(0, 0, 0), v(1, 0, 0), v(1, 0, 1), v(0, 0, 1)), (0, -1, 0)),
            ((v(0, 1, 0), v(0, 1, 1), v(1, 1, 1), v(1, 1, 0)), (0, 1, 0)),
            ((v(0, 0, 0), v(0, 0, 1), v(0, 1, 1), v(0, 1, 0)), (-1, 0, 0)),
            ((v(1, 0, 0), v(1, 1, 0), v(1, 1, 1), v(1, 0, 1)), (1, 0, 0)),
            ((v(0, 0, 1), v(1, 0, 1), v(1, 1, 1), v(0, 1, 1)), (0, 0, 1)),
            ((v(0, 0, 0), v(0, 1, 0), v(1, 1, 0), v(1, 0, 0)), (0, 0, -1))]


def make_scene(folder, W=320, H=180):
    """A wall with a grid of blocks of mixed size and depth (some below the minimum size) and a kiosk."""
    boxes = [(0, 10, 0.0, 0.5, 0, 6, 0)]
    for i in range(5):
        for j in range(3):
            s = (0.3, 0.45, 0.8, 1.2, 0.35)[(i + j) % 5]
            x, z = 0.6 + i * 1.9, 0.8 + j * 1.8
            boxes.append((x, x + s, -0.1 - 0.12 * ((i * 3 + j) % 4), 0.0, z, z + s * 1.3, 0))
    boxes.append((11, 12, -1.0, 0.0, 0, 1, 1))
    verts, faces, obj = [], [], []
    for x0, x1, y0, y1, z0, z1, o in boxes:
        for f in box_faces(x0, x1, y0, y1, z0, z1, verts):
            faces.append(f)
            obj.append(o)
    V = np.array(verts, np.float32)
    xs = -0.5 + (np.arange(W) + 0.5) * 13.0 / W
    zs = 6.5 - (np.arange(H) + 0.5) * 7.0 / H
    X, Z = np.meshgrid(xs, zs)
    faceid = np.zeros((H, W), np.uint32)
    pos = np.zeros((H, W, 3), np.float32)
    nrm = np.zeros((H, W, 3), np.float32)
    best = np.full((H, W), np.inf)
    for bi, (x0, x1, y0, y1, z0, z1, o) in enumerate(boxes):
        m = (X >= x0) & (X <= x1) & (Z >= z0) & (Z <= z1) & (y0 < best)
        best[m] = y0
        faceid[m] = bi * 6 + 1
        pos[m] = np.stack([X[m], np.full(m.sum(), y0), Z[m]], 1)
        nrm[m] = (0, -1, 0)
    os.makedirs(folder, exist_ok=True)
    np.save(os.path.join(folder, "faceid.npy"), faceid)
    np.save(os.path.join(folder, "position.npy"), pos)
    np.save(os.path.join(folder, "normal.npy"), nrm.astype(np.float16))
    np.savez_compressed(os.path.join(folder, "mesh.npz"), normal=np.array([f[1] for f in faces], np.float32),
                        centre=np.array([V[list(f[0])].mean(0) for f in faces], np.float32),
                        area=np.ones(len(faces), np.float32), obj=np.array(obj, np.int32),
                        mat=np.array([k % 3 for k in range(len(faces))], np.int32),
                        col=np.zeros(len(faces), np.int32), loop_total=np.full(len(faces), 4, np.int32),
                        loop_vert=np.array([i for f in faces for i in f[0]], np.int32), vert=V)
    cam = np.eye(4)
    cam[:3, :3] = [[1, 0, 0], [0, 0, -1], [0, 1, 0]]
    cam[:3, 3] = (6, -30, 3)
    info = {"file": "synthetic", "blender": "-", "width": W, "height": H, "faces": len(faces),
            "camera": {"name": "cam", "how": "test", "type": "ORTHO", "lens_mm": 50, "shift_x": 0, "shift_y": 0,
                       "matrix_world": cam.tolist(), "projection": [[2 / 13, 0, 0, 0], [0, 2 / 7, 0, 0],
                                                                    [0, 0, -2 / 99.9, -100.1 / 99.9], [0, 0, 0, 1]]},
            "cameras": ["cam"], "objects": ["facade", "kiosk"], "materials": ["stone", "glass", "metal"],
            "collections": [""], "bbox_min": V.min(0).tolist(), "bbox_max": V.max(0).tolist(), "notes": []}
    json.dump(info, open(os.path.join(folder, "scene.json"), "w"))


tmp = tempfile.mkdtemp(prefix="kuba_scene_speed_")
try:
    src = os.path.join(tmp, "export")
    make_scene(src)

    # ----------------------------------------------------------------------
    # scene_ids.build: threads vs one after another
    # ----------------------------------------------------------------------
    out_t, out_s = os.path.join(tmp, "ids_threads"), os.path.join(tmp, "ids_serial")
    sum_t = si.build(src, out=out_t, min_size_m=0.5)[0]
    pool = si.ThreadPoolExecutor
    si.ThreadPoolExecutor = SerialExecutor
    try:
        sum_s = si.build(src, out=out_s, min_size_m=0.5)[0]
    finally:
        si.ThreadPoolExecutor = pool
    files = sorted(os.listdir(out_s))
    check("scene build: same files in threads", files == sorted(os.listdir(out_t)) and len(files) > 10, str(files))
    diff = [f for f in files if not filecmp.cmp(os.path.join(out_s, f), os.path.join(out_t, f), shallow=False)]
    check("scene build: every file byte-identical in threads", not diff, str(diff))
    check("scene build: same summary in threads", json.dumps(sum_t) == json.dumps(sum_s))
    check("scene build: some small pieces were merged",
          any(v["merged_small"] for v in sum_s["passes"].values()), str(sum_s["passes"]))

    # ----------------------------------------------------------------------
    # idmaps: 8 bit decode path, tag masks, threaded build
    # ----------------------------------------------------------------------
    q16 = np.clip(np.rint((np.arange(256, dtype=np.float32) / 255.0).astype(np.float64) * 65535), 0, 65535)
    check("8 bit -> 16 bit table matches the generic quantisation and keeps the order",
          np.array_equal(im._Q16_OF_8BIT, q16.astype(np.int64)) and (np.diff(im._Q16_OF_8BIT) > 0).all())

    legend = [(f"id_{k}", c) for k, c in enumerate(rng.random((40, 3)))] + [("never_seen", np.array([0.3, 0.6, 0.9]))]
    Hh, Ww = 150, 210
    lab = rng.integers(0, 40, (Hh // 10, Ww // 10))
    lab = cv2.resize(lab.astype(np.float32), (Ww, Hh), interpolation=cv2.INTER_NEAREST).astype(np.int64)
    srgb = np.round(im.to_srgb(np.stack([c for _, c in legend[:40]])) * 255).astype(np.uint8)
    rgb8 = srgb[lab]
    rgb8[:12] = 0                                                       # background band
    soft = cv2.GaussianBlur(rgb8, (3, 3), 0)                           # anti-aliased edges: colours off the legend
    edge = rng.random((Hh, Ww)) < 0.15
    rgb8[edge] = soft[edge]
    rgb8[rng.random((Hh, Ww)) < 0.01] = (255, 255, 255)                # specks no legend entry matches
    path = os.path.join(tmp, "ids_test.png")
    cv2.imencode(".png", rgb8[..., ::-1])[1].tofile(path)
    img = im.read_image(path)
    check("8 bit ID map is recognised", im._as_8bit(img[..., :3]) is not None)
    check("float input that is not 8 bit takes the generic path",
          im._as_8bit(img[..., :3] + np.float32(1e-4)) is None and im._as_8bit(img[..., :3].astype(np.float64)) is None)
    for enc in ("auto", "linear", "srgb"):
        fast = im.decode_idmap(img, legend, enc)
        generic = im.decode_idmap(img.astype(np.float64), legend, enc)      # float64: the 16 bit path
        check(f"decode 8 bit fast path = generic path ({enc})", same(fast, generic))
    labels = fast[0]
    missing_ref = [nm for i, (nm, _) in enumerate(legend) if not (labels == i).any()]
    check("decode notes list exactly the names not in the map",
          "never_seen" in missing_ref and fast[2] == [f"not visible in the ID map: {', '.join(missing_ref)}"],
          str(fast[2]))
    p16 = os.path.join(tmp, "ids_16.png")
    cv2.imencode(".png", (rgb8.astype(np.uint16) * 257 + rng.integers(0, 3, rgb8.shape).astype(np.uint16))[..., ::-1])[1].tofile(p16)
    img16 = im.read_image(p16)
    check("16 bit ID map takes the generic path", im._as_8bit(img16[..., :3]) is None)

    lab_t = labels.astype(np.int32)
    masks = im._label_masks(lab_t, len(legend))
    check("tag masks per bounding box = labels == j",
          len(masks) == len(legend) and all(np.array_equal(m, lab_t == j) for j, m in enumerate(masks)))

    res_t = im.build(out_s, regions_pass="layers", tag_passes="*", min_region_area=20)
    pool = im.ThreadPoolExecutor
    im.ThreadPoolExecutor = SerialExecutor
    try:
        res_s = im.build(out_s, regions_pass="layers", tag_passes="*", min_region_area=20)
    finally:
        im.ThreadPoolExecutor = pool
    keys = ("labels", "atlas", "idmaps", "size", "scope")
    check("from id maps: threaded decode = one after another", all(same(res_t[k], res_s[k]) for k in keys),
          str([k for k in keys if not same(res_t[k], res_s[k])]))
    check("from id maps: tags from the other passes", len(res_s["idmaps"]) > 3, str(list(res_s["idmaps"])))
finally:
    shutil.rmtree(tmp, ignore_errors=True)

if failures:
    print(f"\n{len(failures)} FAILED: {', '.join(failures)}")
    sys.exit(1)
print("\nall scene speed checks passed")
