"""
exr.py

A small OpenEXR reader in numpy for render passes (cryptomatte, AOVs): single-part scanline or tiled files with NONE,
RLE, ZIPS or ZIP compression - what Blender, Houdini and most renderers write for data passes - including all header
attributes (cryptomatte manifests live there). Other files (PIZ, DWA, B44, PXR24, multipart, deep) are read through
the OpenImageIO module that ships with Blender 4.x (read_via_blender), or the OpenEXR module when installed.

read(path) -> {"width", "height", "channels": {name: HxW array (float16 / float32 / uint32)}, "attrs": {...},
               "compression": name}
Float channels keep their exact bits (cryptomatte ids are float32 bit patterns).
write(path, channels, compression) writes simple scanline files (for tests and hand-offs).
"""

from __future__ import annotations

import json
import os
import struct
import subprocess
import tempfile
import zlib

import numpy as np

MAGIC = 20000630
COMPRESSIONS = ["none", "rle", "zips", "zip", "piz", "pxr24", "b44", "b44a", "dwaa", "dwab", "htj2k256", "htj2k32"]
LINES_PER_BLOCK = {"none": 1, "rle": 1, "zips": 1, "zip": 16, "piz": 32, "pxr24": 16, "b44": 32, "b44a": 32,
                   "dwaa": 32, "dwab": 256}
PIXEL = {0: (np.uint32, 4), 1: (np.float16, 2), 2: (np.float32, 4)}
OWN = {"none", "rle", "zips", "zip"}


class Unsupported(Exception):
    pass


# ------------------------------------------------------------------------------------------------------- header


def _cstr(b, i):
    j = b.index(b"\0", i)
    return b[i:j].decode("latin-1"), j + 1


def _attr_value(typ, raw):
    if typ == "chlist":
        chans, i = [], 0
        while raw[i] != 0:
            name, i = _cstr(raw, i)
            ptype, _plin, xs, ys = struct.unpack_from("<iB3xii", raw, i)
            i += 16
            chans.append((name, ptype, xs, ys))
        return chans
    if typ == "compression" or typ == "lineOrder" or typ == "envmap" or typ == "deepImageState":
        return raw[0]
    if typ == "box2i":
        return struct.unpack("<4i", raw)
    if typ == "box2f":
        return struct.unpack("<4f", raw)
    if typ in ("int",):
        return struct.unpack("<i", raw)[0]
    if typ == "float":
        return struct.unpack("<f", raw)[0]
    if typ == "double":
        return struct.unpack("<d", raw)[0]
    if typ == "v2i":
        return struct.unpack("<2i", raw)
    if typ == "v2f":
        return struct.unpack("<2f", raw)
    if typ == "string":
        return raw.decode("utf-8", "replace")
    if typ == "tiledesc":
        xs, ys, mode = struct.unpack("<IIB", raw)
        return {"x": xs, "y": ys, "mode": mode}
    if typ == "stringvector":
        out, i = [], 0
        while i + 4 <= len(raw):
            n = struct.unpack_from("<i", raw, i)[0]
            out.append(raw[i + 4:i + 4 + n].decode("utf-8", "replace"))
            i += 4 + n
        return out
    return raw


def read_header(buf):
    """(attrs, tiled, end of header) from the file's bytes."""
    if len(buf) < 8 or struct.unpack_from("<i", buf, 0)[0] != MAGIC:
        raise ValueError("not an OpenEXR file")
    flags = struct.unpack_from("<I", buf, 4)[0]
    if flags & 0x1000:
        raise Unsupported("multipart EXR")
    if flags & 0x800:
        raise Unsupported("deep EXR")
    attrs, i = {}, 8
    while buf[i] != 0:
        name, i = _cstr(buf, i)
        typ, i = _cstr(buf, i)
        size = struct.unpack_from("<i", buf, i)[0]
        i += 4
        attrs[name] = _attr_value(typ, bytes(buf[i:i + size]))
        i += size
    return attrs, bool(flags & 0x200), i + 1


# ---------------------------------------------------------------------------------------------------- decoding


def _undo_zip(data):
    t = np.frombuffer(data, np.uint8).astype(np.int32)
    if t.size > 1:
        t[1:] -= 128
        t = np.cumsum(t) & 0xFF                                       # predictor: t[i] = t[i-1] + t[i] - 128
    t = t.astype(np.uint8)
    half = (t.size + 1) // 2
    out = np.empty_like(t)
    out[0::2] = t[:half]
    out[1::2] = t[half:]
    return out.tobytes()


def _rle(data, expect):
    out = bytearray()
    i, n = 0, len(data)
    while i < n and len(out) < expect:
        c = struct.unpack_from("b", data, i)[0]
        i += 1
        if c < 0:
            out += data[i:i - c]
            i -= c
        else:
            out += data[i:i + 1] * (c + 1)
            i += 1
    return bytes(out)


def _decode(data, comp, expect):
    if len(data) == expect or comp == "none":
        return data
    if comp in ("zip", "zips"):
        return _undo_zip(zlib.decompress(data))
    if comp == "rle":
        return _undo_zip(_rle(data, expect))
    raise Unsupported(f"{comp} compression")


def read(path):
    """The whole image (see module doc). Falls back to OpenEXR / Blender's OpenImageIO for what numpy can't read."""
    try:
        return _read_own(path)
    except Unsupported as e:
        try:
            return read_via_openexr(path)
        except ImportError:
            pass
        try:
            return read_via_blender(path)
        except FileNotFoundError:
            raise Unsupported(f"{os.path.basename(path)}: {e}; save it with ZIP compression (scanline) or install "
                              "Blender 4.x (its OpenImageIO reads every EXR)") from None


def _read_own(path):
    with open(path, "rb") as f:
        buf = f.read()
    attrs, tiled, pos = read_header(buf)
    comp = COMPRESSIONS[attrs.get("compression", 0)]
    if comp not in OWN:
        raise Unsupported(f"{comp} compression")
    x0, y0, x1, y1 = attrs["dataWindow"]
    W, H = x1 - x0 + 1, y1 - y0 + 1
    chans = sorted(attrs["channels"], key=lambda c: c[0])
    if any(c[2] != 1 or c[3] != 1 for c in chans):
        raise Unsupported("subsampled channels")
    px_bytes = sum(PIXEL[c[1]][1] for c in chans)
    out = {c[0]: np.zeros((H, W), PIXEL[c[1]][0]) for c in chans}
    if not tiled:
        lpb = LINES_PER_BLOCK[comp]
        n = (H + lpb - 1) // lpb
        for off in np.frombuffer(buf, "<u8", n, pos):
            yy, size = struct.unpack_from("<ii", buf, int(off))
            r0 = yy - y0
            rows = min(lpb, H - r0)
            raw = _decode(buf[int(off) + 8:int(off) + 8 + size], comp, rows * W * px_bytes)
            _scatter(raw, chans, out, r0, rows, 0, W)
    else:
        td = attrs["tiles"]
        if td["mode"] & 0x0F != 0:
            raise Unsupported("mipmapped / ripmapped tiles")
        tx, ty = td["x"], td["y"]
        nx, ny = (W + tx - 1) // tx, (H + ty - 1) // ty
        for off in np.frombuffer(buf, "<u8", nx * ny, pos):
            ix, iy, _lx, _ly, size = struct.unpack_from("<5i", buf, int(off))
            c0, r0 = ix * tx, iy * ty
            cols, rows = min(tx, W - c0), min(ty, H - r0)
            raw = _decode(buf[int(off) + 20:int(off) + 20 + size], comp, rows * cols * px_bytes)
            _scatter(raw, chans, out, r0, rows, c0, cols)
    return {"width": W, "height": H, "channels": out, "attrs": attrs, "compression": comp}


def _scatter(raw, chans, out, r0, rows, c0, cols):
    """Block layout: for each line, for each channel (alphabetical), cols values."""
    buf = np.frombuffer(raw, np.uint8)
    line_bytes = cols * sum(PIXEL[c[1]][1] for c in chans)
    blk = buf[:rows * line_bytes].reshape(rows, line_bytes)
    pos = 0
    for name, ptype, _xs, _ys in chans:
        dt, nb = PIXEL[ptype]
        part = np.ascontiguousarray(blk[:, pos:pos + cols * nb])
        out[name][r0:r0 + rows, c0:c0 + cols] = part.view(np.dtype(dt).newbyteorder("<")).reshape(rows, cols)
        pos += cols * nb


# ------------------------------------------------------------------------------------------------- fallbacks


def read_via_openexr(path):
    import OpenEXR  # noqa: F401  (optional)
    with OpenEXR.File(path, separate_channels=True) as fl:
        part = fl.parts[0]
        chans = {k: np.asarray(v.pixels) for k, v in part.channels.items()}
        hdr = {k: (v if isinstance(v, (str, int, float, tuple, list, dict)) else str(v)) for k, v in part.header.items()}
    h, w = next(iter(chans.values())).shape[:2]
    return {"width": w, "height": h, "channels": chans, "attrs": hdr, "compression": "openexr"}


_BLENDER_SCRIPT = r'''
import sys, json, numpy as np, OpenImageIO as oiio
src, dst = sys.argv[-2], sys.argv[-1]
inp = oiio.ImageInput.open(src)
spec = inp.spec()
names = list(spec.channelnames)
fmts = [str(f) for f in spec.channelformats] if len(spec.channelformats) else [str(spec.format)] * len(names)
data = {}
for i, n in enumerate(names):
    t = oiio.UINT if fmts[i] == "uint" else oiio.FLOAT
    px = inp.read_image(0, 0, i, i + 1, t)
    data["c_" + str(i)] = np.asarray(px).reshape(spec.height, spec.width)
attrs = {}
for a in spec.extra_attribs:
    try:
        attrs[a.name] = a.value if isinstance(a.value, (str, int, float)) else str(a.value)
    except Exception:
        pass
inp.close()
data["meta"] = np.frombuffer(json.dumps({"names": names, "fmts": fmts, "attrs": attrs,
                                        "w": spec.width, "h": spec.height}).encode(), np.uint8)
np.savez(dst, **data)
'''


def blender_path():
    from .scene3d import bridge
    return bridge.find_blender("")


def read_via_blender(path):
    """Any EXR through the OpenImageIO module inside Blender 4.x (a background process, a few seconds)."""
    exe = blender_path()
    if not exe or not os.path.isfile(exe):
        raise FileNotFoundError("Blender")
    with tempfile.TemporaryDirectory() as d:
        script = os.path.join(d, "read_exr.py")
        dst = os.path.join(d, "exr.npz")
        with open(script, "w", encoding="utf-8") as fh:
            fh.write(_BLENDER_SCRIPT)
        r = subprocess.run([exe, "-b", "--factory-startup", "--python", script, "--", path, dst],
                           capture_output=True, text=True, timeout=600)
        if not os.path.isfile(dst):
            raise RuntimeError("Blender could not read the EXR: " + (r.stdout + r.stderr)[-800:])
        with np.load(dst) as z:                                      # closed before the temp folder goes (Windows)
            meta = json.loads(bytes(z["meta"]).decode())
            chans = {}
            for i, n in enumerate(meta["names"]):
                a = z[f"c_{i}"]
                chans[n] = a.astype(np.uint32) if meta["fmts"][i] == "uint" else a.astype(np.float32)
    return {"width": meta["w"], "height": meta["h"], "channels": chans, "attrs": meta["attrs"],
            "compression": "via Blender"}


# ------------------------------------------------------------------------------------------------------ writer


def write(path, channels, compression="zip", attrs=None):
    """Scanline EXR with NONE or ZIP compression. channels: {name: HxW float16 / float32 / uint32}; attrs: extra
    string attributes (e.g. a cryptomatte manifest)."""
    names = sorted(channels)
    H, W = next(iter(channels.values())).shape
    ptype = {np.dtype(np.uint32): 0, np.dtype(np.float16): 1, np.dtype(np.float32): 2}

    def attr(name, typ, raw):
        return name.encode() + b"\0" + typ.encode() + b"\0" + struct.pack("<i", len(raw)) + raw

    ch = b"".join(n.encode() + b"\0" + struct.pack("<iB3xii", ptype[channels[n].dtype], 0, 1, 1) for n in names) + b"\0"
    comp = {"none": 0, "zip": 3}[compression]
    hdr = attr("channels", "chlist", ch) + attr("compression", "compression", bytes([comp]))
    hdr += attr("dataWindow", "box2i", struct.pack("<4i", 0, 0, W - 1, H - 1))
    hdr += attr("displayWindow", "box2i", struct.pack("<4i", 0, 0, W - 1, H - 1))
    hdr += attr("lineOrder", "lineOrder", b"\0") + attr("pixelAspectRatio", "float", struct.pack("<f", 1.0))
    hdr += attr("screenWindowCenter", "v2f", struct.pack("<2f", 0, 0)) + attr("screenWindowWidth", "float", struct.pack("<f", 1.0))
    for k, v in (attrs or {}).items():
        hdr += attr(k, "string", v.encode("utf-8"))
    hdr += b"\0"
    lpb = LINES_PER_BLOCK[compression]
    chunks = []
    for y in range(0, H, lpb):
        rows = min(lpb, H - y)
        raw = b"".join(channels[n][y + r].astype(channels[n].dtype.newbyteorder("<")).tobytes()
                       for r in range(rows) for n in names)
        if compression == "zip":
            b = np.frombuffer(raw, np.uint8)
            inter = np.concatenate([b[0::2], b[1::2]]).astype(np.int32)
            d = np.diff(inter, prepend=0)
            d[0] = inter[0]
            d[1:] = (d[1:] + 128) & 0xFF
            z = zlib.compress(d.astype(np.uint8).tobytes(), 6)
            data = z if len(z) < len(raw) else raw
        else:
            data = raw
        chunks.append(struct.pack("<ii", y, len(data)) + data)
    head = struct.pack("<iI", MAGIC, 2) + hdr
    pos = len(head) + 8 * len(chunks)
    offs = []
    for c in chunks:
        offs.append(pos)
        pos += len(c)
    with open(path, "wb") as f:
        f.write(head + struct.pack(f"<{len(offs)}Q", *offs) + b"".join(chunks))
