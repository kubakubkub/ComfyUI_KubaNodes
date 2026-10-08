"""
render_passes.py

A folder of render passes from any 3D tool (Houdini, Blender, C4D, Maya ...) -> the picture, depth, normals and
every other pass as a mask. Files are named <render>_<pass>.<ext> (facade_beauty.png, facade_depth.png,
facade_cut.png) or just <pass>.<ext>; several renders can share a folder, and files that belong to no render
(facade_mask.png next to facade_a_beauty.png, facade_b_beauty.png) are masks for all of them.
EXR passes (one pass per file) are read with exr.py: the picture goes from linear to sRGB, depth in scene units is
stretched to 0..1, normals in -1..1 are packed to 0..1.
One multilayer EXR (the file itself, or the only render of a folder) is read layer by layer: layers_of() lists what
is inside with a guessed role per layer, pick_layers() reads the node's exr_layers text, and each layer then goes
the way a single-pass file of that role goes.

numpy + OpenCV only (tests/test_render_passes.py).
"""

from __future__ import annotations

import fnmatch
import os
import re

import cv2
import numpy as np

from . import imio

EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp", ".bmp", ".exr")
ROLES = {
    "beauty": ("beauty", "rgb", "rgba", "combined", "color", "colour", "clay", "render", "image", "diffuse", "c"),
    "depth": ("depth", "z", "zdepth", "depthmap", "dist"),
    "normal": ("normal", "normals", "nrm", "n"),
}
_ROLE_OF = {tok: role for role, toks in ROLES.items() for tok in toks}


def _split(stem):
    """'facade_a_beauty' -> ('facade_a', 'beauty'); 'beauty' -> ('', 'beauty')."""
    m = re.match(r"^(.*?)[_\-. ]+([^_\-. ]+)$", stem)
    return (m.group(1), m.group(2)) if m else ("", stem)


def discover(folder):
    """
    -> {"renders": {name: {"beauty": path, "depth": path, "normal": path, "masks": {pass: path}}},
        "shared": {stem: path}, "skipped": [file names], "exrs": {stem: path}}.
    A render is every name that has a beauty pass. Files starting with '_' (contact sheets) and other file types
    are skipped; so is an EXR next to a PNG / TIFF ... of the same name (the copy is read, as before) and
    <render>.exr (the multilayer master of a render: its passes are the separate files).
    exrs: every other EXR with several layers, one of them a picture: a render of its own, read layer by layer.
    """
    if not os.path.isdir(folder):
        raise ValueError(f"render passes: folder not found: '{folder}'")
    files, skipped = [], []
    for fn in sorted(os.listdir(folder), key=str.lower):
        p = os.path.join(folder, fn)
        if not os.path.isfile(p):
            continue
        stem, ext = os.path.splitext(fn)
        if ext.lower() not in EXTS or fn.startswith(("_", ".")):
            if ext.lower() in (".exr", ".hdr") or fn.startswith("_"):
                skipped.append(fn)
            continue
        files.append((stem, p))
    plain = {stem.lower() for stem, p in files if not _is_exr(p)}
    split = [(_split(stem), stem, p) for stem, p in files if not (_is_exr(p) and stem.lower() in plain)]
    names = {pre.lower(): pre for (pre, tok), _s, _p in split if _ROLE_OF.get(tok.lower()) == "beauty"}
    split = [x for x in split if not (_is_exr(x[2]) and x[1].lower() in names)]
    used = {p for _k, _s, p in split}
    skipped = sorted(skipped + [os.path.basename(p) for _s, p in files if p not in used], key=str.lower)
    renders = {n: {"beauty": None, "depth": None, "normal": None, "masks": {}} for n in names.values()}
    shared, exrs = {}, {}
    for (pre, tok), stem, p in split:
        r = renders.get(names.get(pre.lower(), None))
        if r is None:
            if _is_exr(p) and _is_multilayer(p):
                exrs[stem] = p
            else:
                shared[stem] = p
            continue
        role = _ROLE_OF.get(tok.lower())
        if role and r[role] is None:
            r[role] = p
        else:
            r["masks"][tok] = p
    return {"renders": renders, "shared": shared, "skipped": skipped, "exrs": exrs}


def _is_exr(path):
    return path.lower().endswith(".exr")


def _is_multilayer(path):
    """An EXR that is a render of its own: several layers, one of them the picture."""
    try:
        inside = layers_of(path)
    except ValueError:
        return False
    return len(inside) > 1 and any(L["role"] == "picture" for L in inside)


def _read_exr(path):
    """An EXR pass -> float32 (H, W, C) as written (linear; C = 1, 3 or 4): the first layer with R G B (A) or
    X Y Z, else one channel (Y, Z, A ...). Cryptomatte layers are no picture."""
    from . import exr
    def reading(what):
        try:
            return what()
        except Exception as e:                                         # not an EXR, a codec nothing here reads ...
            raise ValueError(f"render passes: cannot read {os.path.basename(path)}: {e}") from None

    # the names first, then only the pixels used (a beauty EXR can carry dozens of other passes)
    keys = _exr_pass_keys([c[0] for c in (reading(lambda: exr.read_headers(path))[0].get("channels") or [])], path)
    ch = reading(lambda: exr.read(path, only=set(keys)))["channels"]
    return np.stack([np.asarray(ch[k], np.float32) for k in keys], -1)


def _exr_pass_keys(names, path):
    """The channels of the pass in an EXR with these channel names (see _read_exr)."""
    layers = {}
    for name in names:
        pre, _, c = name.rpartition(".")
        if "crypto" not in pre.lower():
            layers.setdefault(pre, {})[c.upper()] = name
    order = sorted(layers, key=lambda n: (n != "", "combined" not in n.lower(), n.lower()))
    for want in ("RGB", "XYZ"):
        for pre in order:
            g = layers[pre]
            if all(c in g for c in want):
                return [g[c] for c in want] + ([g["A"]] if want == "RGB" and "A" in g else [])
    for c in ("Y", "Z", "A"):
        for pre in order:
            if c in layers[pre]:
                return [layers[pre][c]]
    if not order:
        raise ValueError(f"render passes: {os.path.basename(path)} holds only cryptomatte "
                         f"(kubakub regions from cryptomatte reads it)")
    return [sorted(layers[order[0]].values())[0]]


# ------------------------------------------------------------------------------ the layers of one multilayer EXR

LAYER_ROLES = ("picture", "depth", "normal", "mask", "id", "skip")
_PICTURE = ("combined", "beauty", "c", "cf", "rgba", "rgb", "color", "colour", "image", "render", "clay")
_DEPTH = ("depth", "z", "zdepth", "pz", "depthmap", "dist", "distance", "vrayzdepth")
_NORMAL = ("normal", "normals", "n", "nrm", "ng", "nworld", "vraynormals")
_ID = ("id", "ids", "objectid", "objid", "materialid", "matid", "idmap")
_DATA = ("p", "position", "pos", "pworld", "pref", "uv", "st", "vector", "motion", "motionvector", "velocity", "mv")
# a colour pass: its whole name is made of these words (diffuse_direct, GlossInd, lightgroup2), so a mask called
# 'transom' or 'lightwell' is not taken for one
_COLOUR = re.compile(r"^(?:denois(?:e|ed|ing)?|noisy|diff(?:use)?|gloss(?:y|iness)?|spec(?:ular)?|refl(?:ect(?:ion)?)?|"
                     r"refr(?:act(?:ion)?)?|trans(?:mission|lucent|lucency|parent|parency)?|emi(?:t|ssion|ssive)?|"
                     r"env(?:ironment)?|albedo|basecol(?:or|our)?|subsurf(?:ace)?|volume(?:tric)?|direct|indirect|coat|"
                     r"sheen|light(?:s|ing)?|illum(?:ination)?|dir|ind|col|color|colour|filter|raw|group|pass|\d+)+$")
NO_MASKS = "# no layer as a mask"          # what the layer buttons write when the last mask is clicked off
_ALIAS = {"picture": "picture", "beauty": "picture", "depth": "depth", "normal": "normal", "normals": "normal",
          "mask": "mask", "skip": "skip"}


def _guess(name, uint=False, root=False):
    """What a layer is, from its name (Karma, Blender, Arnold, Redshift, V-Ray names) -> (role, rank, why)."""
    seg = name.rpartition(".")[2].lower()                              # ViewLayer.Combined -> combined
    toks = [t for t in re.split(r"[^a-z0-9]+", seg) if t] or [seg]
    flat = "".join(toks)
    has = lambda words: flat in words or toks[0] in words or any(len(t) > 1 and t in words for t in toks)  # noqa: E731
    rank = lambda words: min([words.index(t) for t in [flat] + toks if t in words] or [len(words)])  # noqa: E731
    if "crypto" in name.lower():
        return "skip", 0, "cryptomatte: kubakub regions from cryptomatte reads it"
    if uint or has(_ID) or flat.startswith("index"):
        return "id", 0, "an ID pass: kubakub regions from id renders reads ID maps"
    if (_COLOUR.match(flat) and not has(_PICTURE)) or flat.startswith(("denois", "noisy")) or flat in ("gi", "sss") or has(_DATA):
        return "skip", 0, "a colour or data pass, no mask"
    if has(_DEPTH) or "depth" in flat:
        return "depth", rank(_DEPTH), ""
    if has(_NORMAL) or "normal" in flat:
        return "normal", rank(_NORMAL), ""
    if has(_PICTURE):
        return "picture", -1 if root else rank(_PICTURE), ""
    return "mask", 0, ""


def layers_of(path):
    """
    The layers inside one EXR, read from its header alone (no pixels, so every compression works and Blender is
    never started) -> [{"name", "channels": [R, G, B ...], "role", "why", "full": {R: channel name in the file},
    "part"}], in the file's order.
    A layer is every channel prefix (diffuse.R diffuse.G diffuse.B -> diffuse; ViewLayer.Depth.Z -> ViewLayer.Depth).
    Channels without a prefix: R G B (A) are the layer 'rgba' ('rgb' without alpha), each other one (Z, Y, A) is its
    own layer, so a flat EXR is one layer. role: a guess from the name, one of LAYER_ROLES; one layer each is the
    picture, the depth and the normal, 'why' says why a layer is skipped.
    """
    from . import exr
    try:
        parts = exr.read_headers(path)
    except (OSError, ValueError) as e:
        raise ValueError(f"render passes: cannot read {os.path.basename(path)}: {e}") from None
    order = "RGBAXYZUVW"
    out, seen = [], set()
    for pi, attrs in enumerate(parts):
        groups, root = {}, {}
        for cname, ptype, _xs, _ys in attrs.get("channels") or []:
            pre, _, c = cname.rpartition(".")
            (groups.setdefault(pre, {}) if pre else root)[c] = (cname, ptype)
        up = {c.upper(): c for c in root}
        found = []
        if all(c in up for c in "RGB"):
            found.append(("rgba" if "A" in up else "rgb", {up[c]: root.pop(up[c]) for c in "RGBA" if c in up}, True))
        found += [(c, {c: v}, True) for c, v in root.items()] + [(pre, g, False) for pre, g in groups.items()]
        for nm, g, is_root in found:
            if pi and is_root and attrs.get("name"):                   # another part: its own name comes first
                nm = f"{attrs['name']}.{nm}"
            while nm.lower() in seen:
                nm += "_2"
            seen.add(nm.lower())
            role, rank, why = _guess(nm, any(pt == 0 for _n, pt in g.values()), is_root and not pi)
            if pi:
                role, why = "skip", f"in part {pi + 1} of a multipart EXR: only the first part is read"
            out.append({"name": nm, "role": role, "why": why, "part": pi, "_rank": rank,
                        "channels": sorted(g, key=lambda c: (order.find(c.upper()) if c.upper() in order else 99, c)),
                        "full": {c.upper(): v[0] for c, v in g.items()}})
    for role in ("picture", "depth", "normal"):                        # one each: the best name, then the first
        cand = sorted((L for L in out if L["role"] == role), key=lambda L: L["_rank"])
        for L in cand[1:]:
            L["role"], L["why"] = "skip", f"another {role} layer; {cand[0]['name']} is the {role}"
    for L in out:
        del L["_rank"]
    return out


def _chooses_masks(text):
    """Does the text say which layers are masks (a layer name, 'mask = name', or the buttons' "no masks" line)?
    Lines that only correct a role ('depth = Z_render') or take one out ('skip = name') leave the guessed masks."""
    for ln in text.splitlines():
        ln = ln.strip()
        role, eq, _pat = ln.partition("=")
        if ln.lower() == NO_MASKS or (ln and not ln.startswith("#") and (not eq or _ALIAS.get(role.strip().lower()) == "mask")):
            return True
    return False


def pick_layers(layers, text=""):
    """
    Which layer is read as what -> ({layer name: 'picture' / 'depth' / 'normal' / 'mask' / ''}, names of the masks
    taken as they are, notes).
    text (the node's exr_layers), one line per layer, wildcards, '#' starts a comment:
      windows          this layer is a mask (a layer guessed as picture / depth / normal keeps that role)
      wall*            every layer of that name that was guessed to be a mask
      depth = Z_cam    picture / depth / normal / mask / skip = layer: read this layer as that
    No line that names a mask: every layer as it was guessed (role and skip lines only correct the guess). The
    picture, the depth and the normal always come from their guessed layers unless a line names another one.
    """
    use = {L["name"]: (L["role"] if L["role"] in ("picture", "depth", "normal") else "") for L in layers}
    forced, notes = set(), []
    if not _chooses_masks(text or ""):
        use.update({L["name"]: "mask" for L in layers if L["role"] == "mask"})
    for ln in (text or "").splitlines():
        ln = ln.strip()
        if not ln or ln.startswith("#"):
            continue
        role, eq, pat = ln.partition("=")
        role, pat = (role.strip().lower(), pat.strip()) if eq else ("", ln)
        if eq and role not in _ALIAS:
            raise ValueError(f"render passes: exr_layers, line '{ln}': '{role}' is no role. Write picture, depth, "
                             f"normal, mask or skip before the '='.")
        role, q = _ALIAS.get(role, ""), pat.lower()
        hits = [L for L in layers if fnmatch.fnmatch(L["name"].lower(), q)
                or fnmatch.fnmatch(L["name"].rpartition(".")[2].lower(), q)]
        if not hits:
            notes.append(f"exr_layers: no layer '{pat}' in the file")
            continue
        wild = any(c in pat for c in "*?[")
        if role in ("picture", "depth", "normal"):
            L = hits[0]
            if L["part"]:
                notes.append(f"exr_layers: {L['name']} cannot be the {role} ({L['why']})")
                continue
            use.update({k: "" for k, v in use.items() if v == role})
            use[L["name"]] = role
            forced.discard(L["name"])
            continue
        for L in hits:
            n = L["name"]
            if role == "skip":
                use[n] = ""
            elif wild and use[n] in ("picture", "depth", "normal"):
                pass
            elif L["part"] or "crypto" in n.lower():
                if not wild:
                    notes.append(f"exr_layers: {n} cannot be a mask ({L['why']})")
            elif role == "mask" or (not wild and L["role"] in ("skip", "id")):
                use[n] = "mask"
                forced.add(n)                                          # asked for by name: its brightness, colour or not
            elif L["role"] == "mask":
                use[n] = "mask"
    return use, forced, notes


def _layer_pixels(ch, L, role):
    g = L["full"]
    if all(c in g for c in "RGB"):
        keys = [g[c] for c in "RGB"] + ([g["A"]] if "A" in g else [])
    elif all(c in g for c in "XYZ") and role != "depth":
        keys = [g[c] for c in "XYZ"]
    else:
        keys = [g["Z"] if role == "depth" and "Z" in g else g[L["channels"][0].upper()]]
    if any(k not in ch for k in keys):
        raise ValueError(f"render passes: layer {L['name']} did not come out of the file")
    return np.stack([np.asarray(ch[k], np.float32) for k in keys], -1)


def _exr_passes(path, text, notes):
    """One multilayer EXR as a render -> (its passes for load(), its layers for the report). The file is read once,
    and only the channels of the layers that are used."""
    from . import exr
    base = os.path.basename(path)
    layers = layers_of(path)
    use, forced, more = pick_layers(layers, text)
    notes += more
    if "picture" not in use.values():
        raise ValueError(f"render passes: no layer in {base} looks like the picture; it holds: "
                         f"{', '.join(L['name'] for L in layers) or 'nothing'}. Write 'picture = <layer>' in exr_layers.")
    wanted = {full for L in layers if use[L["name"]] for full in L["full"].values()}
    cache = {}

    def source(L, role):
        def read():
            if "ch" not in cache:
                try:
                    cache["ch"] = exr.read(path, wanted)["channels"]
                except Exception as e:                                 # a codec nothing here reads, a broken file ...
                    raise ValueError(f"render passes: cannot read {base}: {e}") from None
            return _layer_pixels(cache["ch"], L, role)
        return (f"{base}, layer {L['name']}", True, read)

    r = {"beauty": None, "depth": None, "normal": None, "masks": []}
    for L in layers:
        u = use[L["name"]]
        L["used"] = u or ("not read" + (f" ({L['why']})" if L["why"] else ""))
        if u == "mask":
            r["masks"].append((L["name"], source(L, u), L["name"] in forced, L))
        elif u:
            r["beauty" if u == "picture" else u] = source(L, u)
    return r, layers


def _file_passes(r, shared=()):
    """A render of discover() (files) in the same form."""
    src = lambda p: (os.path.basename(p), _is_exr(p), lambda: _read(p)) if p else None  # noqa: E731
    return {"beauty": src(r["beauty"]), "depth": src(r["depth"]), "normal": src(r["normal"]),
            "masks": [(nm, src(p), False, None) for nm, p in list(r["masks"].items()) + list(shared)]}


def layers_for(path, render=""):
    """
    For the layer buttons of the node (the route /kubakub/render_passes/layers): what the node would read from this
    'folder' field -> {"layers": [{"name", "channels", "role", "why"}], "file": name, "note": plain words}.
    Reads headers only, writes nothing and never raises: an empty or missing path comes back as a note.
    """
    out = {"layers": [], "file": "", "note": ""}
    try:
        if not path:
            out["note"] = "no folder: the node runs on its sample passes (PNG files, no EXR)"
            return out
        if os.path.isdir(path):
            found = discover(path)
            key = {n.lower(): n for n in list(found["exrs"]) + list(found["renders"])}
            pick = key.get((render or "").strip().lower()) or sorted(found["renders"] or found["exrs"] or [""], key=str.lower)[0]
            if pick not in found["exrs"]:
                also = f" Also here: {', '.join(found['exrs'])}.exr (write its name in 'render')." if found["exrs"] else ""
                out["note"] = ("this folder's passes are separate files, read by their names." + also if found["renders"]
                               else "no multilayer EXR in this folder")
                return out
            path = found["exrs"][pick]
        elif not os.path.isfile(path):
            out["note"] = f"not found: {path}"
            return out
        elif not _is_exr(path):
            out["note"] = f"{os.path.basename(path)} is no EXR: paste a folder of passes or one .exr file"
            return out
        layers = layers_of(path)
        out["file"] = os.path.basename(path)
        out["layers"] = [{k: L[k] for k in ("name", "channels", "role", "why")} for L in layers]
        out["note"] = f"{out['file']}: {len(layers)} layer{'s' if len(layers) != 1 else ''}"
    except Exception as e:  # noqa: BLE001
        out["note"] = str(e).replace("render passes: ", "")
    return out


def _srgb(lin):
    lin = np.clip(np.nan_to_num(lin), 0.0, 1.0)
    return np.where(lin <= 0.0031308, lin * 12.92, 1.055 * np.power(lin, 1 / 2.4) - 0.055).astype(np.float32)


def _read(path):
    """-> float32 (H, W, C) in 0..1 (C = 1, 3 or 4, RGB order), 16-bit files keep their range. An EXR comes as
    written (load() brings it to 0..1 by what the pass is)."""
    if _is_exr(path):
        return _read_exr(path)
    img = imio.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"render passes: cannot read {os.path.basename(path)}")
    scale = 65535.0 if img.dtype == np.uint16 else 255.0 if img.dtype == np.uint8 else 1.0
    a = img.astype(np.float32) / scale
    if a.ndim == 2:
        return a[..., None]
    if a.shape[2] == 4:
        return a[..., [2, 1, 0, 3]]
    return a[..., ::-1]


def _rgb(a):
    return np.repeat(a[..., :1], 3, 2) if a.shape[2] == 1 else np.ascontiguousarray(a[..., :3])


def _is_colour(a):
    """An ID map or a picture, not a mask: its channels differ."""
    if a.shape[2] < 3:
        return False
    s = a[::4, ::4, :3]
    return float(np.abs(s - s.mean(2, keepdims=True)).max()) > 0.08


def _mask(a):
    """White (or opaque, when the picture itself is flat) = inside."""
    if a.shape[2] == 4 and float(np.ptp(a[..., 3])) > 0.5 and float(np.ptp(a[..., :3])) < 0.05:
        return a[..., 3]
    return a[..., :3].mean(2) if a.shape[2] >= 3 else a[..., 0]


def _fit(a, W, H, nearest=False):
    if a.shape[:2] == (H, W):
        return a
    out = cv2.resize(a, (W, H), interpolation=cv2.INTER_NEAREST if nearest else cv2.INTER_AREA)
    return out[..., None] if out.ndim == 2 and a.ndim == 3 else out


def load(folder, name="", invert="", exclude="", depth="as rendered", width=0, height=0, layers=""):
    """
    -> dict(name, beauty (H, W, 3), depth (H, W, 3) or None, normal (H, W, 3) or None, masks [(name, (H, W) float32)],
            notes [str], renders [names in the folder], file, layers).
    folder: a folder of passes, or one multilayer .exr file. name: which render ('' = the first). invert / exclude:
    mask names, wildcards, comma separated ('cut' = black means inside). depth: 'as rendered', 'near is white' or
    'near is black' (the last two stretch it to 0..1).
    width, height: the matrix; a render of another size is fitted to it with one uniform scale (another aspect is
    refused). 0 = the render's own size.
    layers: which layers of a multilayer EXR are read as what (pick_layers; '' = all, as guessed). The result's
    'layers' lists every layer of that file with its channels, guessed role and what it was 'used' as (None when the
    render is separate files), 'file' is the EXR's name.
    """
    notes = []
    skipped, listed, exr_file = [], None, ""
    if os.path.isfile(folder):
        if not _is_exr(folder):
            raise ValueError(f"render passes: '{os.path.basename(folder)}' is no EXR: give the folder of your passes, "
                             f"or one multilayer .exr file")
        pick, exr_file = os.path.splitext(os.path.basename(folder))[0], folder
        every = [pick]
        r, listed = _exr_passes(folder, layers, notes)
    else:
        found = discover(folder)
        renders, exrs, skipped = found["renders"], found["exrs"], found["skipped"]
        if not renders and not exrs:
            raise ValueError(f"render passes: no picture in '{folder}': name it <render>_beauty.png "
                             f"(also: {', '.join(ROLES['beauty'][1:8])}), or put one multilayer .exr here")
        every = sorted(list(renders) + list(exrs), key=str.lower)
        key = {n.lower(): n for n in every}
        name = (name or "").strip()
        if name and name.lower() not in key:
            raise ValueError(f"render passes: no render '{name}' in the folder; it has: {', '.join(every) or '-'}")
        pick = key[name.lower()] if name else sorted(renders or exrs, key=str.lower)[0]   # separate files come first
        if pick in exrs:
            exr_file = exrs[pick]
            r, listed = _exr_passes(exr_file, layers, notes)
            r["masks"] += _file_passes({"beauty": None, "depth": None, "normal": None, "masks": {}},
                                       found["shared"].items())["masks"]
        else:
            r = _file_passes(renders[pick], found["shared"].items())
            master = next((fn for fn in skipped if fn.lower() == pick.lower() + ".exr"), None)
            try:
                inside = layers_of(os.path.join(folder, master)) if master else []
            except ValueError:
                inside = []
            if len(inside) > 1:
                notes.append(f"{master} holds {len(inside)} layers ({', '.join(L['name'] for L in inside[:8])}"
                             f"{' ...' if len(inside) > 8 else ''}); the separate pass files are read. Paste the path "
                             f"of {master} itself into 'folder' to read its layers.")
            elif (layers or "").strip():
                notes.append("exr_layers is for a multilayer EXR: this render is separate files, read by their names")
    label, is_exr, read = r["beauty"]
    beauty = _rgb(read())
    if is_exr:
        beauty = _srgb(beauty)
        notes.append(f"{label}: a linear EXR, shown as sRGB")
    H0, W0 = beauty.shape[:2]                                          # the render's own size
    H, W = H0, W0
    if width and height and (int(height), int(width)) != (H0, W0):
        if abs(W0 / H0 - width / height) > 0.005 * (width / height):
            raise ValueError(f"render passes: the render is {W0}x{H0}, the matrix {int(width)}x{int(height)}: another "
                             f"aspect, one uniform scale cannot fit it. Render in the proportions of the matrix, or "
                             f"disconnect 'matrix' to work at the render's own size.")
        H, W = int(height), int(width)
        beauty = np.clip(cv2.resize(beauty, (W, H), interpolation=cv2.INTER_AREA if W < W0 else cv2.INTER_CUBIC),
                         0.0, 1.0)
        notes.append(f"render {W0}x{H0} fitted to the matrix {W}x{H} (every pass)")

    def aux(src, what):
        if not src:
            return None
        _label, is_exr, read = src
        try:
            a = _rgb(read())
        except ValueError as e:
            if not is_exr:
                raise
            notes.append(f"{str(e).replace('render passes: ', '')}: no {what}")
            return None
        if is_exr:
            a = np.nan_to_num(a, posinf=1e10, neginf=-1e10)
            if what == "depth":
                g = a[..., 0]
                seen = np.abs(g) < 1e9                                 # the empty background is written as 1e10 / inf
                lo, hi = (float(g[seen].min()), float(g[seen].max())) if seen.any() else (0.0, 1.0)
                g = np.where(seen, g, hi)
                if lo < 0.0 or hi > 1.0:
                    g = (g - lo) / max(hi - lo, 1e-9)
                    notes.append(f"depth EXR {lo:.4g} .. {hi:.4g} (scene units) stretched to 0..1, near is black")
                a = np.repeat(g[..., None], 3, 2)
            elif float(a.min()) < -0.01:
                a = a * 0.5 + 0.5
                notes.append(f"{what} EXR in -1..1 packed to 0..1")
            a = np.clip(a, 0.0, 1.0).astype(np.float32)
        if a.shape[:2] != (H0, W0):
            notes.append(f"{what} {a.shape[1]}x{a.shape[0]} resized to the picture {W}x{H}")
        return _fit(a, W, H)

    dep = aux(r["depth"], "depth")
    if dep is not None and depth != "as rendered":
        g = dep[..., 0]
        lo, hi = float(g.min()), float(g.max())
        g = (g - lo) / max(hi - lo, 1e-9)
        far_white = float(g[: max(1, H // 20)].mean()) > float(g[H // 3: 2 * H // 3].mean())   # the sky row is far
        if (depth == "near is white") == far_white:
            g = 1 - g
        dep = np.repeat(g[..., None], 3, 2).astype(np.float32)
    nrm = aux(r["normal"], "normal")
    pats = lambda t: [x.strip().lower() for x in (t or "").replace("\n", ",").split(",") if x.strip()]  # noqa: E731
    inv, exc = pats(invert), pats(exclude)
    hit = lambda nm, pp: any(fnmatch.fnmatch(nm.lower(), q) for q in pp)  # noqa: E731
    masks = []
    for nm, (label, is_exr, read), as_is, layer in r["masks"]:
        left = lambda why: layer.update(used=f"mask, left out: {why}") if layer else None  # noqa: E731
        if hit(nm, exc):
            left("named in exclude")
            continue
        try:
            a = read()
        except ValueError as e:
            if not is_exr:
                raise
            notes.append(f"{str(e).replace('render passes: ', '')}: left out")
            left("could not be read")
            continue
        if is_exr:
            a = np.clip(np.nan_to_num(a), 0.0, 1.0)
        if not as_is and _is_colour(a):
            if layer:                                                  # said in the report's list of layers
                left(f"a colour picture, not a mask ('mask = {nm}' in exr_layers takes its brightness)")
            else:
                notes.append(f"{label} is a colour picture (an ID map?), not a mask: left out "
                             f"(kubakub regions from id renders reads ID maps)")
            continue
        m = _mask(a)
        if m.shape != (H0, W0):
            notes.append(f"mask {nm} {m.shape[1]}x{m.shape[0]} resized to the picture {W}x{H}")
        if m.shape != (H, W):
            m = _fit(m, W, H, nearest=True)
        if hit(nm, inv):
            m = 1.0 - m
        masks.append((nm, np.ascontiguousarray(m, dtype=np.float32)))
    if skipped:
        notes.append(f"not read: {', '.join(skipped[:6])}" + (" ..." if len(skipped) > 6 else ""))
    if listed is not None:
        listed = [{k: L[k] for k in ("name", "channels", "role", "used")} for L in listed]
    return {"name": pick, "beauty": beauty, "depth": dep, "normal": nrm, "masks": masks, "notes": notes,
            "renders": every, "file": os.path.basename(exr_file), "layers": listed}
