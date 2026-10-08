"""
Model free test of kubakub render passes (kubakub/render_passes.py): a folder of passes named <render>_<pass>.png is
sorted into picture / depth / normal / masks, shared masks reach every render, inverted and 16-bit files read right,
and the masks become regions through the regions from masks logic.

Run from the ComfyUI_windows_portable folder:
    python_embeded\\python.exe ComfyUI\\custom_nodes\\ComfyUI_KubaNodes\\tests\\test_render_passes.py
"""

import os
import sys
import tempfile

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from kubakub import facade_core as fc  # noqa: E402
from kubakub import render_passes as rp  # noqa: E402

failures = []


def check(name, cond, detail=""):
    print(("ok   " if cond else "FAIL ") + name + (f"  ({detail})" if detail and not cond else ""))
    if not cond:
        failures.append(name)


def save(folder, name, img):
    cv2.imencode(os.path.splitext(name)[1], img)[1].tofile(os.path.join(folder, name))


W, H = 320, 200
with tempfile.TemporaryDirectory() as d:
    beauty = np.full((H, W, 3), (40, 90, 200), np.uint8)                   # BGR: an orange-ish wall
    for nm in ("facade_a", "facade_b"):
        save(d, f"{nm}_beauty.png", beauty)
    cut = np.full((H, W), 255, np.uint8)                                   # black = cut open: two holes
    cut[40:90, 30:100] = 0
    cut[110:170, 180:290] = 0
    save(d, "facade_a_cut.png", cut)
    depth16 = np.tile(np.linspace(65535, 20000, H).astype(np.uint16)[:, None], (1, W))   # far (white) at the top
    save(d, "facade_a_depth.png", depth16)
    save(d, "facade_a_normal.png", np.full((H // 2, W // 2, 3), (255, 128, 128), np.uint8))   # half size
    facade = np.zeros((H, W), np.uint8)
    facade[20:190, 10:310] = 255
    save(d, "facade_mask.png", facade)
    ids = np.zeros((H, W, 3), np.uint8)
    ids[:, : W // 2] = (0, 0, 255)
    ids[:, W // 2:] = (0, 255, 0)
    save(d, "facade_a_ids.png", ids)
    save(d, "_sheet_beauty.jpg", beauty)
    open(os.path.join(d, "facade_a.exr"), "wb").write(b"not read")
    open(os.path.join(d, "facade_a_recipe.json"), "w").write("{}")

    found = rp.discover(d)
    check("two renders in the folder", sorted(found["renders"]) == ["facade_a", "facade_b"], str(sorted(found["renders"])))
    a = found["renders"]["facade_a"]
    check("picture, depth, normal and the other passes are told apart", all(a[k] for k in ("beauty", "depth", "normal"))
          and sorted(a["masks"]) == ["cut", "ids"], str(a))
    check("a file of no render is a shared mask; sheets and EXRs are skipped", list(found["shared"]) == ["facade_mask"]
          and set(found["skipped"]) == {"_sheet_beauty.jpg", "facade_a.exr"}, f"{found['shared']} {found['skipped']}")

    r = rp.load(d, "facade_a", invert="cut")
    names = [n for n, _ in r["masks"]]
    masks = dict(r["masks"])
    check("the picture comes back in RGB", r["beauty"].shape == (H, W, 3)
          and np.allclose(r["beauty"][5, 5], np.array([200, 90, 40]) / 255, atol=1e-3), str(r["beauty"][5, 5]))
    check("masks: cut and the shared facade mask; the ID map is left out with a note", names == ["cut", "facade_mask"]
          and any("ID map" in n for n in r["notes"]), f"{names} {r['notes']}")
    check("invert = cut: the holes are inside", masks["cut"][60, 60] == 1 and masks["cut"][5, 5] == 0
          and abs(masks["cut"].mean() - (50 * 70 + 60 * 110) / (W * H)) < 1e-6)
    check("16-bit depth keeps its range", abs(r["depth"][0, 0, 0] - 1.0) < 1e-4 and abs(r["depth"][-1, 0, 0] - 20000 / 65535) < 1e-3,
          f"{r['depth'][0, 0, 0]} {r['depth'][-1, 0, 0]}")
    check("a half-size normal pass is resized, with a note", r["normal"].shape == (H, W, 3) and any("normal" in n for n in r["notes"]))
    near = rp.load(d, "facade_a", depth="near is white")["depth"][..., 0]
    check("near is white: stretched to 0..1, the far top row black", near[0, 0] < 0.01 and near[-1, 0] > 0.99, f"{near[0, 0]} {near[-1, 0]}")
    b = rp.load(d, "FACADE_B")
    check("the other render (name without case) has only the shared mask and no depth", [n for n, _ in b["masks"]] == ["facade_mask"]
          and b["depth"] is None and b["normal"] is None)
    check("no name = the first render", rp.load(d)["name"] == "facade_a")
    check("exclude leaves a mask out", [n for n, _ in rp.load(d, "facade_a", exclude="facade*")["masks"]] == ["cut"])
    try:
        rp.load(d, "nothing")
        check("an unknown render name is refused and the folder's renders are named", False)
    except ValueError as e:
        check("an unknown render name is refused and the folder's renders are named", "facade_a" in str(e), str(e))

    # the masks as regions: every hole its own region, cut out of the wall (regions from masks, split_masks = cut)
    entries = [{"source": n, "name": n, "group": None, "mask": masks[n] > 0.5} for n in names]
    labels, atlas = fc.atlas_from_masks(entries, W, H, split_masks="cut", min_region_area=0)
    got = sorted(g["name"] for g in atlas["regions"])
    check("regions: cut_01, cut_02 and the wall around them", got == ["cut_01", "cut_02", "facade_mask"], str(got))
    wall = [g for g in atlas["regions"] if g["name"] == "facade_mask"][0]
    check("the wall region has the holes cut out", wall["area"] == int(facade.astype(bool).sum()) - (50 * 70 + 60 * 110), str(wall["area"]))

with tempfile.TemporaryDirectory() as d:
    save(d, "beauty.png", np.zeros((40, 60, 3), np.uint8))
    rgba = np.zeros((40, 60, 4), np.uint8)
    rgba[10:20, 10:30, 3] = 255                                            # a mask kept in the alpha channel
    save(d, "windows.png", rgba)
    r = rp.load(d)
    check("plain names (beauty.png, windows.png) and a mask in the alpha channel", r["name"] == ""
          and [n for n, _ in r["masks"]] == ["windows"] and r["masks"][0][1].sum() == 200, f"{r['name']!r} {r['masks'][0][1].sum()}")
with tempfile.TemporaryDirectory() as d:
    try:
        rp.load(d)
        check("an empty folder says how to name the picture", False)
    except ValueError as e:
        check("an empty folder says how to name the picture", "_beauty" in str(e), str(e))

# EXR passes (one pass per file, written with the pack's own exr.py) and the fit to a matrix
from kubakub import exr  # noqa: E402

with tempfile.TemporaryDirectory() as d:
    f32 = lambda v: np.full((H, W), v, np.float32)  # noqa: E731
    exr.write(os.path.join(d, "shot_beauty.exr"), {"R": f32(0.5), "G": f32(0.0), "B": f32(4.0), "A": f32(1.0)})
    z = np.tile(np.linspace(40.0, 2.0, H, dtype=np.float32)[:, None], (1, W))      # metres, far at the top
    z[:10] = 1e10                                                          # the empty sky
    exr.write(os.path.join(d, "shot_depth.exr"), {"Z": z})
    exr.write(os.path.join(d, "shot_normal.exr"), {"N.x": f32(-1.0), "N.y": f32(0.0), "N.z": f32(1.0)})
    m = np.zeros((H, W), np.float32)
    m[50:100, 40:140] = 1.0
    exr.write(os.path.join(d, "shot_windows.exr"), {"Y": m}, "none")
    exr.write(os.path.join(d, "shot_door.exr"), {"A": 1.0 - m})
    save(d, "shot_door.png", (m * 255).astype(np.uint8))                   # a PNG copy of a pass wins over its EXR
    exr.write(os.path.join(d, "shot.exr"), {"R": f32(1.0), "G": f32(1.0), "B": f32(1.0)})   # the multilayer master
    open(os.path.join(d, "shot_broken.exr"), "wb").write(b"not an exr")
    found = rp.discover(d)
    s = found["renders"]["shot"]
    check("EXR passes are found by name; the PNG copy wins, the master <render>.exr is not read",
          s["beauty"].endswith("shot_beauty.exr") and s["depth"].endswith(".exr") and s["normal"].endswith(".exr")
          and s["masks"]["door"].endswith(".png") and sorted(s["masks"]) == ["broken", "door", "windows"]
          and found["skipped"] == ["shot.exr", "shot_door.exr"], f"{s} {found['skipped']}")
    r = rp.load(d)
    masks = dict(r["masks"])
    check("EXR picture: linear to sRGB, clipped to 0..1", np.allclose(r["beauty"][5, 5], [0.7354, 0.0, 1.0], atol=1e-3)
          and r["beauty"].dtype == np.float32, str(r["beauty"][5, 5]))
    dz = r["depth"][..., 0]
    check("EXR depth in scene units: stretched to 0..1, the empty sky is far, with a note", abs(dz[-1, 0]) < 1e-6
          and abs(dz[10, 0] - 1.0) < 1e-6 and dz[0, 0] == 1.0 and any("depth EXR 2 .. 38.09" in n for n in r["notes"]),
          f"{dz[-1, 0]} {dz[10, 0]} {dz[0, 0]} {r['notes']}")
    check("EXR normals in -1..1 are packed to 0..1", np.allclose(r["normal"][3, 3], [0.0, 0.5, 1.0], atol=1e-6))
    check("EXR masks read; an EXR that cannot be read is left out with a note", list(masks) == ["door", "windows"]
          and masks["windows"].sum() == 50 * 100 and masks["door"].sum() == 50 * 100
          and any("shot_broken.exr" in n and "left out" in n for n in r["notes"]), f"{list(masks)} {r['notes']}")
    near = rp.load(d, depth="near is white")["depth"][..., 0]
    check("EXR depth, near is white", near[-1, 0] > 0.99 and near[0, 0] < 0.01, f"{near[-1, 0]} {near[0, 0]}")

    # matrix: one uniform scale, another aspect is refused
    big = rp.load(d, width=W * 2, height=H * 2)
    bm = dict(big["masks"])
    check("a matrix twice the size: picture, depth, normal and masks are fitted to it, with one note",
          big["beauty"].shape == (H * 2, W * 2, 3) and big["depth"].shape == (H * 2, W * 2, 3)
          and big["normal"].shape == (H * 2, W * 2, 3) and bm["windows"].shape == (H * 2, W * 2)
          and bm["windows"].sum() == 4 * 50 * 100 and sum("matrix" in n for n in big["notes"]) == 1
          and not any("resized to the picture" in n for n in big["notes"]), str(big["notes"]))
    small = rp.load(d, width=W // 2, height=H // 2)
    check("a smaller matrix", small["beauty"].shape == (H // 2, W // 2, 3) and small["masks"][0][1].shape == (H // 2, W // 2))
    same = rp.load(d, width=W, height=H)
    check("a matrix of the render's size changes nothing", np.array_equal(same["beauty"], r["beauty"])
          and not any("matrix" in n for n in same["notes"]))
    try:
        rp.load(d, width=W, height=W)
        check("a matrix of another aspect is refused in plain words", False)
    except ValueError as e:
        check("a matrix of another aspect is refused in plain words", "another aspect" in str(e) and f"{W}x{W}" in str(e), str(e))
with tempfile.TemporaryDirectory() as d:
    open(os.path.join(d, "shot_beauty.exr"), "wb").write(b"not an exr")
    try:
        rp.load(d)
        check("a picture EXR that cannot be read is named", False)
    except ValueError as e:
        check("a picture EXR that cannot be read is named", "shot_beauty.exr" in str(e), str(e))

# one multilayer EXR: its layers are listed with a guessed role, picked with exr_layers, and read like single passes
import struct  # noqa: E402

with tempfile.TemporaryDirectory() as d:
    f32 = lambda v: np.full((H, W), v, np.float32)  # noqa: E731
    z = np.tile(np.linspace(40.0, 2.0, H, dtype=np.float32)[:, None], (1, W))
    z[:10] = 1e10
    win = np.zeros((H, W), np.float32)
    win[50:100, 40:140] = 1.0
    wall = np.zeros((H, W), np.float32)
    wall[20:180, 10:300] = 1.0
    multi = {"R": f32(0.5), "G": f32(0.0), "B": f32(4.0), "A": f32(1.0),
             "depth.Z": z * 2, "Z_render.Z": z, "N_world.x": f32(-1.0), "N_world.y": f32(0.0), "N_world.z": f32(1.0),
             "windows.Y": win, "wall_a.R": wall, "wall_a.G": wall, "wall_a.B": wall,
             "diffuse.R": f32(0.9), "diffuse.G": f32(0.1), "diffuse.B": f32(0.2),
             "CryptoObject00.R": f32(0.3), "CryptoObject00.G": f32(0.3), "CryptoObject00.B": f32(0.3),
             "CryptoObject00.A": f32(0.3)}
    path = os.path.join(d, "shot.exr")
    exr.write(path, multi)
    lay = {L["name"]: L for L in rp.layers_of(path)}
    check("the layers of a multilayer EXR are listed, with their channels", sorted(lay) == sorted(
        ["rgba", "depth", "Z_render", "N_world", "windows", "wall_a", "diffuse", "CryptoObject00"])
        and lay["rgba"]["channels"] == ["R", "G", "B", "A"] and lay["N_world"]["channels"] == ["x", "y", "z"]
        and lay["windows"]["channels"] == ["Y"], str({k: v["channels"] for k, v in lay.items()}))
    roles = {k: v["role"] for k, v in lay.items()}
    check("roles are guessed from the names; one depth, the other is skipped and says why", roles == {
        "rgba": "picture", "depth": "depth", "Z_render": "skip", "N_world": "normal", "windows": "mask", "wall_a": "mask",
        "diffuse": "skip", "CryptoObject00": "skip"} and "depth" in lay["Z_render"]["why"]
        and "cryptomatte" in lay["CryptoObject00"]["why"], str(roles))
    names = ["C", "Cf", "ViewLayer.Combined", "beauty", "ViewLayer.Depth", "Pz", "N", "ViewLayer.Normal", "P", "albedo",
             "directdiffuse", "ViewLayer.DiffCol", "specular", "emission", "ViewLayer.IndexOB", "object_id",
             "CryptoMaterial01", "ViewLayer.Denoising Depth", "ViewLayer.AO", "mask_c", "facade"]
    got = [rp._guess(n)[0] for n in names]
    want = ["picture", "picture", "picture", "picture", "depth", "depth", "normal", "normal", "skip", "skip", "skip", "skip",
            "skip", "skip", "id", "id", "skip", "skip", "mask", "mask", "mask"]
    check("typical layer names of Karma, Blender, Arnold, Redshift, V-Ray", got == want,
          str([(n, g) for n, g, w in zip(names, got, want) if g != w]))

    r = rp.load(path)
    used = {L["name"]: L["used"] for L in r["layers"]}
    check("the EXR file itself in 'folder': picture, depth, normal, and the mask layers by name", r["name"] == "shot"
          and np.allclose(r["beauty"][5, 5], [0.7354, 0.0, 1.0], atol=1e-3) and [n for n, _ in r["masks"]] == ["wall_a", "windows"]
          and dict(r["masks"])["windows"].sum() == 50 * 100 and np.allclose(r["normal"][3, 3], [0.0, 0.5, 1.0], atol=1e-6)
          and any("depth EXR 4 .. 76.18" in n for n in r["notes"]) and r["file"] == "shot.exr", f"{r['notes']}")
    check("every layer of the file is in the result with what it was used as", used["rgba"] == "picture"
          and used["depth"] == "depth" and used["N_world"] == "normal" and used["windows"] == "mask"
          and used["diffuse"].startswith("not read") and "cryptomatte" in used["CryptoObject00"], str(used))
    p = rp.load(path, layers="w*\ndepth = Z_render\nmask = diffuse\nnothing_here")
    pm = dict(p["masks"])
    check("exr_layers: a wildcard, a forced depth, a colour layer forced to be a mask, an unknown name noted",
          list(pm) == ["diffuse", "wall_a", "windows"] and abs(float(pm["diffuse"].mean()) - 0.4) < 1e-4
          and any("depth EXR 2 .. 38.09" in n for n in p["notes"]) and any("nothing_here" in n for n in p["notes"])
          and {L["name"]: L["used"] for L in p["layers"]}["depth"].startswith("not read"), f"{list(pm)} {p['notes']}")
    check("exr_layers: one name = only that mask; a comment alone = no masks; skip takes one out of a wildcard",
          [n for n, _ in rp.load(path, layers="windows")["masks"]] == ["windows"]
          and rp.load(path, layers="# no layer as a mask")["masks"] == []
          and [n for n, _ in rp.load(path, layers="w*\nskip = wall_a")["masks"]] == ["windows"])
    check("exr_layers: a layer picked by its plain name is a mask even when it was guessed as a colour pass",
          [n for n, _ in rp.load(path, layers="diffuse")["masks"]] == ["diffuse"]
          and [n for n, _ in rp.load(path, layers="*")["masks"]] == ["wall_a", "windows"])
    check("invert and exclude work on layers", rp.load(path, invert="windows", exclude="wall*")["masks"][0][1].sum()
          == W * H - 50 * 100)
    try:
        rp.load(path, layers="colour = windows")
        check("exr_layers: an unknown role is refused in plain words", False)
    except ValueError as e:
        check("exr_layers: an unknown role is refused in plain words", "is no role" in str(e) and "picture, depth" in str(e), str(e))
    big = rp.load(path, width=W * 2, height=H * 2)
    check("a multilayer EXR is fitted to the matrix like a folder of passes", big["beauty"].shape == (H * 2, W * 2, 3)
          and big["depth"].shape == (H * 2, W * 2, 3) and dict(big["masks"])["windows"].sum() == 4 * 50 * 100)
    only = exr.read(path, only={"windows.Y"})["channels"]
    check("exr.read keeps only the channels asked for", list(only) == ["windows.Y"] and only["windows.Y"].sum() == 50 * 100)

    # the same file as the only render of a folder, next to a mask file; and next to separate pass files
    save(d, "facade_mask.png", (wall * 255).astype(np.uint8))
    found = rp.discover(d)
    check("a folder: the multilayer EXR is a render of its own, the PNG a shared mask", list(found["exrs"]) == ["shot"]
          and list(found["shared"]) == ["facade_mask"] and not found["renders"], str(found))
    fr = rp.load(d)
    check("a folder with one multilayer EXR reads its layers, plus the folder's mask files", fr["name"] == "shot"
          and [n for n, _ in fr["masks"]] == ["wall_a", "windows", "facade_mask"] and fr["layers"] is not None)
    info = rp.layers_for(d)
    check("the layer list for the buttons: from the file, and from the folder that holds it", info["file"] == "shot.exr"
          and [L["name"] for L in info["layers"]] == [L["name"] for L in rp.layers_for(path)["layers"]]
          and set(info["layers"][0]) == {"name", "channels", "role", "why"} and "8 layers" in info["note"], str(info["note"]))
    check("the layer list never raises: empty, missing, no EXR", "sample" in rp.layers_for("")["note"]
          and "not found" in rp.layers_for(os.path.join(d, "nope"))["note"]
          and "no EXR" in rp.layers_for(os.path.join(d, "facade_mask.png"))["note"]
          and all(rp.layers_for(x)["layers"] == [] for x in ("", os.path.join(d, "nope"), os.path.join(d, "facade_mask.png"))))
    save(d, "shot_beauty.png", np.full((H, W, 3), 128, np.uint8))
    save(d, "shot_windows.png", (win * 255).astype(np.uint8))
    sr = rp.load(d, layers="windows")
    check("separate pass files next to their master EXR: the files are read, as before, and a note says what is inside",
          sr["layers"] is None and [n for n, _ in sr["masks"]] == ["windows", "facade_mask"]
          and any("shot.exr holds 8 layers" in n for n in sr["notes"]) and rp.layers_for(d)["layers"] == []
          and "separate files" in rp.layers_for(d)["note"], str(sr["notes"]))

with tempfile.TemporaryDirectory() as d:
    flat = os.path.join(d, "flat.exr")
    exr.write(flat, {"R": f32(0.5), "G": f32(0.5), "B": f32(0.5)})
    one = rp.layers_of(flat)
    r = rp.load(flat)
    check("a flat EXR is one layer, the picture", [(L["name"], L["role"], L["channels"]) for L in one] == [("rgb", "picture", ["R", "G", "B"])]
          and r["beauty"].shape == (H, W, 3) and r["masks"] == [] and r["depth"] is None)
    nopic = os.path.join(d, "masks.exr")
    exr.write(nopic, {"windows.Y": win, "door.Y": 1.0 - win})
    try:
        rp.load(nopic)
        check("an EXR without a picture layer names its layers and says what to write", False)
    except ValueError as e:
        check("an EXR without a picture layer names its layers and says what to write", "door, windows" in str(e)
              and "picture = " in str(e), str(e))
    check("... and 'picture = windows' reads it", rp.load(nopic, layers="picture = windows\ndoor")["beauty"].shape == (H, W, 3))
    try:
        rp.load(os.path.join(d, "nope.png"))
        check("a path that is no folder and no file is refused", False)
    except ValueError as e:
        check("a path that is no folder and no file is refused", "not found" in str(e), str(e))

    # a multipart file: the headers of every part are listed, only the first part's layers can be read
    attr = lambda n, t, raw: n.encode() + b"\0" + t.encode() + b"\0" + struct.pack("<i", len(raw)) + raw  # noqa: E731
    chl = lambda names: b"".join(n.encode() + b"\0" + struct.pack("<iB3xii", 1, 0, 1, 1) for n in names) + b"\0"  # noqa: E731
    part = lambda name, names: attr("channels", "chlist", chl(names)) + attr("name", "string", name.encode()) + b"\0"  # noqa: E731
    mp = os.path.join(d, "multipart.exr")
    open(mp, "wb").write(struct.pack("<iI", exr.MAGIC, 0x1000 | 2) + part("rgba", ["A", "B", "G", "R"])
                         + part("masks", ["B", "G", "R", "windows.Y"]) + b"\0" + b"pixels would follow")
    parts = rp.layers_of(mp)
    check("a multipart EXR: every part's layers are listed; those of the later parts are marked as not read",
          [(L["name"], L["role"], L["part"]) for L in parts] == [("rgba", "picture", 0), ("masks.rgb", "skip", 1), ("windows", "skip", 1)]
          and "multipart" in parts[1]["why"] and len(exr.read_headers(mp)) == 2, str(parts))

print()
print("all render passes tests passed" if not failures else f"{len(failures)} FAILED: " + ", ".join(failures))
sys.exit(1 if failures else 0)
