"""
illustrator.py

Read an Illustrator .ai (PDF compatible, the default when saving from
Illustrator) or a PDF for kubakub regions from illustrator. PyMuPDF + numpy +
opencv, no ComfyUI imports, so tests/test_illustrator.py runs without a model.

How it works:
- All layers (optional content groups) are switched on in memory before
  anything is read (MuPDF caches the layer state on first use, and hidden
  layers are otherwise skipped). The file on disk is never written.
- get_drawings() gives every path with its layer name. Each layer is
  rasterised by replaying its paths on an empty page (MuPDF's own renderer
  does not switch layers of these files reliably), so curves, even-odd fills
  and stroke widths stay exact.
- Placed images are assigned to their layer by walking the content stream
  (/OC /MCn BDC ... /Fm Do ... EMC) and pasted at their bbox.
- The artboard (page rect) maps onto the matrix with one uniform scale; an
  aspect mismatch is refused.
"""

from __future__ import annotations

import fnmatch
import os
import re

import cv2
import numpy as np

from . import optional

try:
    from . import facade_core as fc
except ImportError:          # tests: the pack folder is on sys.path
    from kubakub import facade_core as fc

ROLES =("lines", "shapes", "outside", "scope", "tag", "reference", "ignore")


def _fitz():
    return optional.need("fitz", "regions from illustrator")       # PyMuPDF, imported late


def open_all_layers(path: str):
    """Open the file with every layer switched on (in memory only)."""
    fitz = _fitz()
    doc = fitz.open(path)
    if not doc.is_pdf:
        raise ValueError(f"{os.path.basename(path)} is not PDF compatible. In Illustrator: Save As .ai "
                         "with 'Create PDF Compatible File' on, or Save As PDF.")
    cat = doc.pdf_catalog()
    kind, ocgs = doc.xref_get_key(cat, "OCProperties/OCGs")
    if kind == "array":
        doc.xref_set_key(cat, "OCProperties/D/ON", ocgs)
        doc.xref_set_key(cat, "OCProperties/D/OFF", "[]")
    return doc


def layer_info(doc):
    """{ocg xref: (name, visible in the file)} before our in-memory switch (read from /D/OFF)."""
    cat = doc.pdf_catalog()
    kind, ocgs = doc.xref_get_key(cat, "OCProperties/OCGs")
    if kind != "array":
        return {}
    xrefs = [int(x) for x in re.findall(r"(\d+)\s+0\s+R", ocgs)]
    out = {}
    for x in xrefs:
        k, name = doc.xref_get_key(x, "Name")
        out[x] = _pdf_string(name) if k == "string" else f"layer_{x}"
    return out


def _pdf_string(s: str) -> str:
    return s.strip("()") if s.startswith("(") else s


def image_layers(doc, page):
    """[(layer name or None, image xref, bbox)] for the images drawn on the page."""
    names = layer_info(doc)
    kind, props = doc.xref_get_key(page.xref, "Resources/Properties")
    tag_to_ocg = {m.group(1): int(m.group(2))
                  for m in re.finditer(r"/(\w+)\s+(\d+)\s+0\s+R", props or "")} if kind == "dict" else {}
    stream = b"".join(doc.xref_stream(x) or b"" for x in page.get_contents())
    stack, xobj_layer = [], {}
    for m in re.finditer(rb"/OC\s*/(\w+)\s*BDC|/\w+\s*BDC|BMC|EMC|/(\w+)\s+Do", stream):
        t = m.group(0)
        if m.group(1):
            stack.append(names.get(tag_to_ocg.get(m.group(1).decode(), -1)))
        elif t.endswith(b"BDC") or t.endswith(b"BMC"):
            stack.append(stack[-1] if stack else None)
        elif t == b"EMC":
            if stack:
                stack.pop()
        elif m.group(2):
            xobj_layer.setdefault(m.group(2).decode(), next((s for s in reversed(stack) if s), None))

    def images_in(xref, depth=0):
        """Image xrefs drawn by an XObject (the image itself, or images inside a form)."""
        k, sub = doc.xref_get_key(xref, "Subtype")
        if sub == "/Image":
            return [xref]
        if sub != "/Form" or depth > 4:
            return []
        k, res = doc.xref_get_key(xref, "Resources/XObject")
        if k != "dict":
            return []
        found = []
        for mm in re.finditer(r"/\w+\s+(\d+)\s+0\s+R", res):
            found += images_in(int(mm.group(1)), depth + 1)
        return found

    k, res = doc.xref_get_key(page.xref, "Resources/XObject")
    name_to_xref = ({mm.group(1): int(mm.group(2)) for mm in re.finditer(r"/(\w+)\s+(\d+)\s+0\s+R", res)}
                    if k == "dict" else {})
    image_layer = {}
    for name, layer in xobj_layer.items():
        if name in name_to_xref and layer:
            for ix in images_in(name_to_xref[name]):
                image_layer.setdefault(ix, layer)
    # other writers (PyMuPDF, some exporters) put the layer on the XObject itself: /OC n 0 R
    for name, xref in name_to_xref.items():
        k, oc = doc.xref_get_key(xref, "OC")
        mm = re.match(r"(\d+)\s+0\s+R", oc or "")
        if k == "xref" and mm and int(mm.group(1)) in names:
            for ix in images_in(xref):
                image_layer.setdefault(ix, names[int(mm.group(1))])
    return [(image_layer.get(info["xref"]), info["xref"], info["bbox"], info.get("transform"))
            for info in page.get_image_info(xrefs=True)]


def read_layers(path: str):
    """
    Returns (doc, page, layers) where layers is an ordered dict
    name -> {"paths": [...], "visible": bool, "images": [(xref, bbox, transform)]}.
    Layers without paths or images (text only, empty) are listed too.
    """
    doc = open_all_layers(path)
    if doc.page_count < 1:
        raise ValueError("the file has no page / artboard")
    if doc.page_count > 1:
        pass  # first artboard only; the node notes it
    cat = doc.pdf_catalog()
    k, off = doc.xref_get_key(cat, "OCProperties/D/OFF")
    page = doc[0]
    layers = {}
    for x, name in layer_info(doc).items():
        layers[name] = {"paths": [], "visible": True, "images": []}
    for d in page.get_drawings():
        layers.setdefault(d.get("layer") or "(no layer)", {"paths": [], "visible": True, "images": []})
        layers[d.get("layer") or "(no layer)"]["paths"].append(d)
    for layer, xref, bbox, tr in image_layers(doc, page):
        layers.setdefault(layer or "(no layer)", {"paths": [], "visible": True, "images": []})
        layers[layer or "(no layer)"]["images"].append((xref, bbox, tr))
    return doc, page, layers


def file_visibility(path: str):
    """{layer name: visible} as saved in the file (Illustrator's eye icons)."""
    fitz = _fitz()
    doc = fitz.open(path)
    try:
        return {o["name"]: bool(o["on"]) for o in doc.get_ocgs().values()}
    finally:
        doc.close()


def target_size(page_rect, width=None, height=None, scale=1.0):
    """(W, H, s): the matrix size and the uniform scale from artboard points to pixels."""
    aw, ah = float(page_rect.width), float(page_rect.height)
    if width and height:
        if abs(aw / ah - width / height) > 0.005 * (width / height):
            raise ValueError(f"artboard {aw:g}x{ah:g} pt has another aspect than the matrix {width}x{height}; "
                             "one uniform scale cannot map it. Fix the artboard in Illustrator.")
        return int(width), int(height), width / aw
    return int(round(aw * scale)), int(round(ah * scale)), float(scale)


def rasterize(paths, page_rect, W, H, s):
    """RGBA uint8 [H, W, 4] of the paths replayed on an empty page at scale s."""
    fitz = _fitz()
    out = fitz.open()
    pg = out.new_page(width=page_rect.width, height=page_rect.height)
    shape = pg.new_shape()
    for p in paths:
        for it in p["items"]:
            if it[0] == "l":
                shape.draw_line(it[1], it[2])
            elif it[0] == "c":
                shape.draw_bezier(it[1], it[2], it[3], it[4])
            elif it[0] == "re":
                shape.draw_rect(it[1])
            elif it[0] == "qu":
                shape.draw_quad(it[1])
        # lineCap / lineJoin are left out: passing them through breaks the content stream
        shape.finish(fill=p.get("fill"), color=p.get("color"), width=p.get("width") or 0,
                     even_odd=bool(p.get("even_odd")), closePath=bool(p.get("closePath")),
                     fill_opacity=p.get("fill_opacity") or 1, stroke_opacity=p.get("stroke_opacity") or 1)
    shape.commit()
    pix = pg.get_pixmap(matrix=fitz.Matrix(s, s), alpha=True)
    a = np.frombuffer(pix.samples, np.uint8).reshape(pix.h, pix.w, pix.n)
    return _fit(a, W, H)


def _fit(a, W, H):
    """Pad or crop a render (off by a pixel from rounding) to exactly W x H."""
    h, w = a.shape[:2]
    if (h, w) == (H, W):
        return np.ascontiguousarray(a)
    out = np.zeros((H, W, a.shape[2]), a.dtype)
    out[:min(h, H), :min(w, W)] = a[:min(h, H), :min(w, W)]
    return out


def place_images(doc, images, W, H, s):
    """RGB float [H, W, 3] with the layer's images pasted at their bbox (axis aligned)."""
    fitz = _fitz()
    canvas = np.zeros((H, W, 3), np.float32)
    notes = []
    for xref, bbox, tr in images:
        if tr is not None and (abs(tr[1]) > 1e-6 or abs(tr[2]) > 1e-6):
            notes.append(f"image {xref} is rotated or skewed, skipped")
            continue
        pix = fitz.Pixmap(doc, xref)
        if pix.alpha:
            pix = fitz.Pixmap(pix, 0)
        if pix.colorspace is None or pix.colorspace.n != 3:
            pix = fitz.Pixmap(fitz.csRGB, pix)
        img = np.frombuffer(pix.samples, np.uint8).reshape(pix.h, pix.w, pix.n)[..., :3]
        x0, y0, x1, y1 = (v * s for v in bbox)
        if x1 - x0 < 1 or y1 - y0 < 1:
            continue
        flip_x = tr is not None and tr[0] < 0
        flip_y = tr is not None and tr[3] < 0
        if flip_x:
            img = img[:, ::-1]
        if flip_y:
            img = img[::-1]
        dw, dh = int(round(x1 - x0)), int(round(y1 - y0))
        big = cv2.resize(np.ascontiguousarray(img), (dw, dh), interpolation=cv2.INTER_AREA
                         if dw < img.shape[1] else cv2.INTER_CUBIC).astype(np.float32) / 255
        ox, oy = int(round(x0)), int(round(y0))
        sx0, sy0 = max(0, -ox), max(0, -oy)
        tx0, ty0 = max(0, ox), max(0, oy)
        tw, th = min(W - tx0, dw - sx0), min(H - ty0, dh - sy0)
        if tw > 0 and th > 0:
            canvas[ty0:ty0 + th, tx0:tx0 + tw] = big[sy0:sy0 + th, sx0:sx0 + tw]
    return canvas, notes


# --------------------------------------------------------------------------
# roles
# --------------------------------------------------------------------------

def parse_roles(text: str):
    """'MASK = outside', 'CENTRE, COUR, JAR = lines', 'TEXT* = ignore'. Returns ([(pattern, role)], notes)."""
    out, notes = [], []
    for n, raw in enumerate((text or "").splitlines(), 1):
        line = raw.split("//", 1)[0].strip()
        if not line:
            continue
        names, eq, role = line.rpartition("=")
        role = role.strip().lower()
        if not eq or role not in ROLES:
            notes.append(f"layer_roles line {n}: expected '<layer names> = <{'|'.join(ROLES)}>', got '{line}'")
            continue
        for pat in names.split(","):
            if pat.strip():
                out.append((pat.strip().lower(), role))
    return out, notes


def auto_role(name, layer, cover, border):
    """Role from the layer's content: see the README table."""
    paths, images = layer["paths"], layer["images"]
    low = name.lower()
    if not paths:
        return "reference" if images else "ignore"
    if cover >= 0.98:
        return "ignore"                       # a background fill
    if re.search(r"mask|range|scope|matte|cutout", low):
        return "outside" if border > 0.5 else "scope"
    if re.search(r"^_|text|label|guide|grid|info", low):
        return "ignore"
    strokes = sum(1 for p in paths if p.get("color") is not None and p.get("fill") is None)
    return "lines" if strokes >= 0.5 * len(paths) else "shapes"


def assign_roles(layers, alphas, roles_text=""):
    """{name: role}, notes. alphas: {name: bool covered [H, W]} for layers with paths."""
    rules, notes = parse_roles(roles_text)
    out = {}
    for name, layer in layers.items():
        role = None
        for pat, r in rules:
            if fnmatch.fnmatchcase(name.lower(), pat):
                role = r
        if role is None:
            a = alphas.get(name)
            cover = float(a.mean()) if a is not None else 0.0
            if a is not None:
                edge = np.concatenate([a[0], a[-1], a[:, 0], a[:, -1]])
                border = float(edge.mean())
            else:
                border = 0.0
            role = auto_role(name, layer, cover, border)
        out[name] = role
    unused = [pat for pat, _ in rules if not any(fnmatch.fnmatchcase(n.lower(), pat) for n in layers)]
    if unused:
        notes.append(f"layer_roles: no layer matches {', '.join(unused)}; layers are {', '.join(layers)}")
    return out, notes


def fill_colors(paths):
    """Distinct fill colours (hex) of a layer, most common first."""
    counts = {}
    for p in paths:
        if p.get("fill") is not None:
            h = "#" + "".join(f"{int(round(c * 255)):02x}" for c in p["fill"][:3])
            counts[h] = counts.get(h, 0) + 1
    return [h for h, _ in sorted(counts.items(), key=lambda kv: -kv[1])]


def layer_extent(covered: np.ndarray, close_px: int = 6) -> np.ndarray:
    """Area enclosed by a line drawing: close small gaps, fill the outer contours."""
    m = covered.astype(np.uint8)
    if close_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * close_px + 1, 2 * close_px + 1))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    cnt, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros_like(m)
    cv2.drawContours(out, cnt, -1, 1, thickness=cv2.FILLED)
    return out > 0


# --------------------------------------------------------------------------
# regions
# --------------------------------------------------------------------------

def _safe(name: str) -> str:
    s = re.sub(r"\s+", "_", name.strip())
    return re.sub(r"[^A-Za-z0-9_\-]", "", s) or "layer"


def build(path, width=None, height=None, scale=1.0, roles_text="", min_region_area=400,
          merge_small_regions=True, line_gap_close_px=1, split_shapes=True, scope=None):
    """
    Regions from an Illustrator file. Line layers are cut into the closed cells
    between their lines (named <LAYER>_001.. in reading order, grouped by shape
    as <LAYER>_g<n>, tagged with the layer); shape layers are patched on top, one
    region per separate shape (<LAYER>_01.., group <LAYER>, tagged with the fill
    colour); outside layers cut the scope, scope layers limit it; tag layers tag
    the regions they cover. Returns a dict, see the node.
    """
    doc, page, layers = read_layers(path)
    notes = []
    if doc.page_count > 1:
        notes.append(f"{doc.page_count} artboards / pages: only the first is read")
    rect = page.rect
    W, H, s = target_size(rect, width, height, scale)
    notes.append(f"artboard {rect.width:g}x{rect.height:g} pt -> {W}x{H} px (scale {s:g})")
    try:
        visible = file_visibility(path)
    except Exception:  # noqa: BLE001
        visible = {}

    rgba = {n: rasterize(l["paths"], rect, W, H, s) for n, l in layers.items() if l["paths"]}
    covered = {n: a[..., 3] > 127 for n, a in rgba.items()}
    roles, role_notes = assign_roles(layers, covered, roles_text)
    notes += role_notes

    sc = None
    in_scope = [n for n, r in roles.items() if r == "scope" and n in covered]
    if in_scope:
        sc = np.zeros((H, W), bool)
        for n in in_scope:
            sc |= covered[n]
    for n in (n for n, r in roles.items() if r == "outside" and n in covered):
        sc = (~covered[n]) if sc is None else (sc & ~covered[n])
    if scope is not None:
        scope = np.asarray(scope, bool)
        if scope.shape != (H, W):
            raise ValueError(f"scope is {scope.shape[1]}x{scope.shape[0]}, the regions {W}x{H}")
        sc = scope if sc is None else (sc & scope)

    line_layers = [n for n, r in roles.items() if r == "lines" and n in rgba]
    lines = np.zeros((H, W), bool)
    for n in line_layers:
        lines |= rgba[n][..., 3] > 60

    labels = atlas = None
    if line_layers:
        img = np.repeat(lines[..., None], 3, axis=2).astype(np.float32)
        labels, atlas = fc.build_atlas(img, mode="line_drawing", line_polarity="light_lines",
                                       line_gap_close_px=int(line_gap_close_px),
                                       min_region_area=int(min_region_area),
                                       merge_small_regions=merge_small_regions, scope=sc)
        n_reg = len(atlas["regions"])
        ext = {n: layer_extent(rgba[n][..., 3] > 60) for n in line_layers}
        areas = np.maximum(np.bincount(labels[labels >= 0], minlength=n_reg), 1)
        best, share = np.full(n_reg, -1), np.zeros(n_reg)
        for k, n in enumerate(line_layers):
            inside = labels[ext[n] & (labels >= 0)]
            frac = np.bincount(inside, minlength=n_reg) / areas
            better = frac > share
            best[better], share[better] = k, frac[better]
        counters = {}
        for r in atlas["regions"]:
            i = r["region_id"]
            layer = line_layers[best[i]] if share[i] >= 0.5 else "lines"
            counters[layer] = counters.get(layer, 0) + 1
            r["name"] = f"{_safe(layer)}_{counters[layer]:03d}"
            r["group_id"] = f"{_safe(layer)}_{r['group_id']}"
            r["tags"] = [layer]
            r.pop("color", None)
        groups = {}
        for r in atlas["regions"]:
            groups.setdefault(r["group_id"], []).append(r["region_id"])
        atlas["groups"] = groups
        atlas["mode"] = "illustrator_lines"

    entries, colour_of = [], {}
    for n, r in roles.items():
        if n not in covered:
            continue
        if r == "shapes":
            cols = fill_colors(layers[n]["paths"])
            parts = fc.split_components(covered[n], max(1, int(min_region_area) // 4)) if split_shapes else [covered[n]]
            for j, part in enumerate(parts):
                name = f"{_safe(n)}_{j + 1:02d}" if split_shapes else _safe(n)
                if len(cols) > 1:
                    mean = rgba[n][part][:, :3].mean(axis=0)
                    rgbs = np.asarray([[int(h[i:i + 2], 16) for i in (1, 3, 5)] for h in cols], np.float32)
                    colour_of[name] = cols[int(np.argmin(np.linalg.norm(rgbs - mean, axis=1)))]
                elif cols:
                    colour_of[name] = cols[0]
                entries.append({"source": n, "name": name, "group": _safe(n), "mask": part})
        elif r == "tag":
            entries.append({"source": n, "name": _safe(n), "group": None, "mask": covered[n]})
    tag_names = ", ".join(_safe(n) for n, r in roles.items() if r == "tag" and n in covered)

    if entries:
        base = (labels, atlas, sc) if atlas is not None else None
        labels, atlas, sc = fc.atlas_from_masks(
            entries, W, H, base=base, patch="on_top", scope=sc if base is None else None,
            min_region_area=int(min_region_area), merge_small_regions=merge_small_regions,
            tag_only_masks=tag_names, with_scope=True)
        for r in atlas["regions"]:
            if r["name"] in colour_of:
                r["tags"] = list(r.get("tags", [])) + [colour_of[r["name"]]]
    elif atlas is None:
        raise ValueError("no line or shape layer: nothing to cut into regions. Layers and roles: "
                         + ", ".join(f"{n}={r}" for n, r in roles.items()))

    reference, ref_layers = None, [n for n, r in roles.items() if r == "reference"]
    for n in ref_layers:
        if layers[n]["images"]:
            img, img_notes = place_images(doc, layers[n]["images"], W, H, s)
            notes += img_notes
            reference = img if reference is None else np.where(img.any(axis=2, keepdims=True), img, reference)
        elif n in rgba:
            a = rgba[n].astype(np.float32) / 255
            layer_img = a[..., :3] * a[..., 3:]
            reference = layer_img if reference is None else layer_img + reference * (1 - a[..., 3:])

    table = []
    for n, l in layers.items():
        table.append({"layer": n, "role": roles[n], "paths": len(l["paths"]), "images": len(l["images"]),
                      "visible_in_file": visible.get(n), "covered_px": int(covered[n].sum()) if n in covered else 0})
    atlas["source_file"] = os.path.basename(path)
    atlas["layers"] = table
    atlas["notes"] = notes + atlas.get("notes", [])
    return {"labels": labels, "atlas": atlas, "scope": sc, "lines": lines, "reference": reference,
            "layer_masks": covered, "roles": roles, "size": (W, H), "scale": s}
