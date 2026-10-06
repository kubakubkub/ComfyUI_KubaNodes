"""
surroundings.py

The neighbourhood of the building as plain geometry: the footprints of the buildings around a place on the map
(OpenStreetMap, fetched once per place and kept on disk), each with a height, in metres around that place
(x = east, y = north, z = up), and on request the real ground under them (terrain heights, fetched the same way). blender_export.py extrudes them next to the model, so audience views, walkthroughs
and relights show the street instead of a building in the void. The surroundings are context only: they get no
regions, no IDs and no glow.

Also: which footprint is the building itself (it is left out, the model stands there), which way its facade
faces, and the matrix that puts the map onto the model's facade.
Standard library + numpy, no ComfyUI imports (tests/test_surroundings.py).
"""

from __future__ import annotations

import json
import math
import os
import re
import urllib.parse
import urllib.request
import uuid

import numpy as np

OVERPASS = "https://overpass-api.de/api/interpreter"           # one host, as the docs promise
TERRAIN = "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"    # heights as PNG tiles, no key
TERRAIN_CREDIT = "terrain heights: Mapzen terrain tiles on AWS (SRTM, EU-DEM and other open sources)"
ORANGE, VIOLET = (0.95, 0.54, 0.35), (0.63, 0.53, 0.72)         # the plan: the model and projector, what stands apart
AGENT = "kubakub-nodes (ComfyUI projection mapping previz; one request per place)"
CREDIT = "map data (c) OpenStreetMap contributors, ODbL"
LEVEL_M = 3.0               # height of one building level when only the level count is tagged
FALLBACK_M = 12.0           # height when nothing in the area is tagged
OWN_MAX_M = 30.0            # the place must be this close to a footprint to count as that building


# --------------------------------------------------------------------------
# place on the map
# --------------------------------------------------------------------------

def parse_location(text):
    """'48.85837, 2.29448' (a map's 'copy coordinates'), or a map link with '@lat,lon' -> (lat, lon); '' -> None."""
    t = (text or "").strip()
    if not t:
        return None
    m = re.search(r"@(-?\d+\.\d+),\s*(-?\d+\.\d+)", t) or re.search(r"(-?\d+(?:\.\d+)?)\s*[,;\s]\s*(-?\d+(?:\.\d+)?)", t)
    if not m:
        raise ValueError(f"location '{t}': need 'latitude, longitude', e.g. 48.85837, 2.29448 "
                         "(a map's right click, copy coordinates)")
    lat, lon = float(m.group(1)), float(m.group(2))
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        raise ValueError(f"location '{t}': latitude is -90..90 and longitude -180..180 (latitude comes first)")
    return lat, lon


def metres_per_degree(lat):
    """(metres per degree of latitude, of longitude) at this latitude (WGS84 series)."""
    p = math.radians(lat)
    return (111132.92 - 559.82 * math.cos(2 * p) + 1.175 * math.cos(4 * p),
            111412.84 * math.cos(p) - 93.5 * math.cos(3 * p))


# --------------------------------------------------------------------------
# OpenStreetMap
# --------------------------------------------------------------------------

def overpass_query(lat, lon, radius_m):
    a = f"(around:{float(radius_m):.0f},{lat:.6f},{lon:.6f})"
    return ("[out:json][timeout:60];(" + "".join(
        f'way["{k}"]{a};relation["{k}"]["type"="multipolygon"]{a};' for k in ("building", "building:part"))
        + ");out geom tags;")


BUSY = (429, 502, 503, 504)   # the map server's "later": worth asking again


def _ask(url, data=None, timeout=45, what="OpenStreetMap", accept=None):
    """
    The bytes a server answers. A busy one (429 / 5xx or no answer in time) is asked up to three times, a few
    seconds apart; any other error fails at once. accept(bytes) may refuse an answer (-> a reason) to ask again.
    """
    import socket
    import time
    import urllib.error
    last = None
    for k in range(3):
        if k:
            time.sleep(3.0 * k)
        try:
            req = urllib.request.Request(url, data=data, headers={"User-Agent": AGENT})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                raw = r.read()
            last = accept(raw) if accept else None
            if not last:
                return raw
        except urllib.error.HTTPError as e:
            last = e
            if e.code not in BUSY:
                break
        except (urllib.error.URLError, socket.timeout, TimeoutError, ValueError) as e:
            last = e
    raise ConnectionError(f"{what} download failed ({last}). The server is busy at times: queue again in a minute, and "
                          "check the internet connection; a smaller radius_m is answered sooner; or give your own "
                          "surroundings file.")


def download(lat, lon, radius_m, timeout=45):
    """
    The buildings around a place from the Overpass API -> its JSON as a dict. An answer that says the query ran out
    of time or memory (it comes as a normal answer with a 'remark') counts as a failure, so half a map is never kept.
    """
    def whole(raw):
        data = json.loads(raw.decode("utf-8"))
        if "error" in str(data.get("remark", "")).lower() or not isinstance(data.get("elements"), list):
            return f"the map server gave up on the query: {str(data.get('remark', 'no elements'))[:120]}"

    body = urllib.parse.urlencode({"data": overpass_query(lat, lon, radius_m)}).encode()
    return json.loads(_ask(OVERPASS, body, timeout, accept=whole).decode("utf-8"))


def on_disk(folder, lat, lon, kind="osm", ext="json"):
    """Places kept in the folder that lie within 25 m of this one -> [(radius_m, metres away, path)], largest first.
    kind: 'osm' (buildings, .json) or 'terrain' (heights, .npz)."""
    out = []
    my, mx = metres_per_degree(lat)
    for fn in os.listdir(folder) if os.path.isdir(folder) else ():
        m = re.fullmatch(rf"{kind}_(-?\d+\.\d+)_(-?\d+\.\d+)_(\d+)\.{ext}", fn)
        if m:
            d = math.hypot((float(m.group(1)) - lat) * my, (float(m.group(2)) - lon) * mx)
            if d <= 25.0:
                out.append((float(m.group(3)), d, os.path.join(folder, fn)))
    return sorted(out, reverse=True)


def _covering(kept, radius_m):
    """The first place of on_disk() that holds this radius whole, or None."""
    return next((path for r, d, path in kept if r >= float(radius_m) + d - 0.5), None)


def covered(folder, lat, lon, radius_m, terrain=False):
    """True when this place and radius are on disk already (no download needed); terrain: its heights as well."""
    return (_covering(on_disk(folder, lat, lon), radius_m) is not None
            and (not terrain or _covering(on_disk(folder, lat, lon, "terrain", "npz"), radius_m) is not None))


def _dump(path, doc):
    """JSON written aside and renamed into place (another queue may write the same file)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.part-{os.getpid()}-{uuid.uuid4().hex[:8]}"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(doc, f)
    try:
        os.replace(tmp, path)
    except OSError:                                   # a reader holds it (Windows): the same content is there
        os.remove(tmp)
        if not os.path.isfile(path):
            raise


def fetch(folder, lat, lon, radius_m, allow_download=True):
    """
    -> (Overpass dict, downloaded now: bool, note). Kept as <folder>/osm_<lat>_<lon>_<radius>.json, so a place is
    downloaded once and everything after that works offline. A place already on disk with a larger radius (or a
    few metres away) serves a smaller one without a download: clip with within(). When the map server does not
    answer, the largest area on disk for this place is used and the note says so.
    """
    radius_m = float(radius_m)
    kept = on_disk(folder, lat, lon)

    def load(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    whole = _covering(kept, radius_m)
    if whole:
        return load(whole), False, ""
    if not allow_download:
        raise ConnectionError(
            "the map of this place is not on this computer yet, and downloads are off. To let the node fetch it once "
            "from OpenStreetMap (it sends the coordinates and the radius to overpass-api.de), write\n"
            "    surroundings_download = on\n"
            "under [settings] in kubakub.ini (in the pack's folder; copy kubakub.ini.example if there is none) and "
            "restart ComfyUI. Or give your own surroundings model in 'file'.")
    try:
        data = download(lat, lon, radius_m)
    except ConnectionError as e:
        if not kept:
            raise
        r, d, path = kept[0]
        return load(path), False, (f"NOTE: the map server did not answer ({e.args[0].split('. ')[0]}); the {r:g} m "
                                   f"around this place that are on this computer are used instead of {radius_m:g} m. "
                                   "Queue again later for the full radius.")
    _dump(os.path.join(folder, f"osm_{lat:.5f}_{lon:.5f}_{radius_m:.0f}.json"), data)
    return data, True, ""


# --------------------------------------------------------------------------
# the ground: real heights around the place
# --------------------------------------------------------------------------

def _tile(lat, lon, z):
    """Web-mercator tile coordinates (fractions included) of a place at zoom z."""
    n = 2.0 ** z
    return (lon + 180.0) / 360.0 * n, (1.0 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2.0 * n


def terrain_download(lat, lon, radius_m):
    """
    The ground around a place from the terrain tiles -> {"heights": (n, n) float32 metres above sea level, row 0 =
    south, column 0 = west, "step": metres between samples, "radius": half the side}. The tiles hold about 3 m per
    pixel at best (the source data is coarser: hills and slopes, not kerbs); they are read smooth onto a grid of
    2.5 m or more.
    """
    import cv2
    z = 15 if radius_m <= 900 else 14
    my, mx = metres_per_degree(lat)
    r = float(radius_m) + 20.0
    x0, y0 = _tile(lat + r / my, lon - r / mx, z)                 # north-west corner
    x1, y1 = _tile(lat - r / my, lon + r / mx, z)
    tx0, ty0, tx1, ty1 = int(x0), int(y0), int(x1), int(y1)
    mosaic = np.zeros(((ty1 - ty0 + 1) * 256, (tx1 - tx0 + 1) * 256), np.float32)
    for ty in range(ty0, ty1 + 1):
        for tx in range(tx0, tx1 + 1):
            raw = _ask(TERRAIN.format(z=z, x=tx, y=ty), timeout=30, what="terrain")
            im = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
            if im is None or im.shape[:2] != (256, 256):
                raise ConnectionError("terrain download failed (a tile came back unreadable): queue again")
            b, g, rr = (im[..., k].astype(np.float32) for k in range(3))
            mosaic[(ty - ty0) * 256:(ty - ty0 + 1) * 256, (tx - tx0) * 256:(tx - tx0 + 1) * 256] = rr * 256 + g + b / 256 - 32768
    step = max(2.5, float(radius_m) / 160.0)
    n = 2 * int(math.ceil(float(radius_m) / step)) + 1
    axis = (np.arange(n) - n // 2) * step
    lats, lons = lat + axis / my, lon + axis / mx
    px = ((lons + 180.0) / 360.0 * 2.0 ** z - tx0) * 256 - 0.5
    py = ((1.0 - np.arcsinh(np.tan(np.radians(lats))) / math.pi) / 2.0 * 2.0 ** z - ty0) * 256 - 0.5
    heights = cv2.remap(mosaic, np.tile(px.astype(np.float32), (n, 1)), np.tile(py.astype(np.float32)[:, None], (1, n)),
                        cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return {"heights": heights.astype(np.float32), "step": step, "radius": (n // 2) * step}


def fetch_terrain(folder, lat, lon, radius_m, allow_download=True):
    """
    -> (terrain, downloaded now: bool, note). terrain = {"heights", "step", "radius", "offset": where the grid's
    middle lies from this place (x east, y north, m)}. Kept as <folder>/terrain_<lat>_<lon>_<radius>.npz and reused
    like the buildings (fetch()): a larger area on disk serves a smaller one, and when the server does not answer
    the largest area on disk is used and the note says so.
    """
    radius_m = float(radius_m)
    kept = on_disk(folder, lat, lon, "terrain", "npz")
    my, mx = metres_per_degree(lat)

    def load(path):
        m = re.fullmatch(r"terrain_(-?\d+\.\d+)_(-?\d+\.\d+)_\d+\.npz", os.path.basename(path))
        with np.load(path) as f:
            return {"heights": f["heights"], "step": float(f["step"]), "radius": float(f["radius"]),
                    "offset": ((float(m.group(2)) - lon) * mx, (float(m.group(1)) - lat) * my)}

    whole = _covering(kept, radius_m)
    if whole:
        return load(whole), False, ""
    if not allow_download:
        raise ConnectionError("the terrain of this place is not on this computer yet, and downloads are off "
                              "(surroundings_download = on in kubakub.ini allows them); or set terrain = flat")
    try:
        t = terrain_download(lat, lon, radius_m)
    except ConnectionError as e:
        if not kept:
            raise
        r, d, path = kept[0]
        return load(path), False, (f"NOTE: the terrain server did not answer ({e.args[0].split('. ')[0]}); the {r:g} m "
                                   "of ground that are on this computer are used. Queue again later for the full radius.")
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"terrain_{lat:.5f}_{lon:.5f}_{radius_m:.0f}.npz")
    tmp = f"{path}.part-{os.getpid()}-{uuid.uuid4().hex[:8]}.npz"
    np.savez_compressed(tmp, heights=t["heights"], step=t["step"], radius=t["radius"])
    os.replace(tmp, path)
    return load(path), True, ""


def terrain_at(terrain, xy):
    """Ground height (m above sea level) at points (N, 2) in metres around the place, read between the samples;
    outside the grid the edge value."""
    h, step = terrain["heights"], terrain["step"]
    n = h.shape[0]
    q = (np.asarray(xy, float).reshape(-1, 2) - terrain.get("offset", (0.0, 0.0))) / step + n // 2
    q = np.clip(q, 0, n - 1 - 1e-6)
    i, j = q[:, 1].astype(int), q[:, 0].astype(int)
    fy, fx = q[:, 1] - i, q[:, 0] - j
    return (h[i, j] * (1 - fx) * (1 - fy) + h[i, j + 1] * fx * (1 - fy) + h[i + 1, j] * (1 - fx) * fy + h[i + 1, j + 1] * fx * fy)


def level(terrain, point, flat_m, blend_m=None):
    """
    The terrain with the ground around 'point' (the facade) kept level: flat within flat_m, the real heights from
    flat_m + blend_m on (default blend = 1.5 x flat), a smooth ramp between. The height data is far coarser than a
    building, so without this a nearby hill tilts the square under the model's feet. -> (terrain, height at point).
    """
    zero = float(terrain_at(terrain, [point])[0])
    if flat_m <= 0:
        return terrain, zero
    h, step = terrain["heights"], terrain["step"]
    n = h.shape[0]
    ox, oy = terrain.get("offset", (0.0, 0.0))
    axis = (np.arange(n) - n // 2) * step
    d = np.hypot(axis[None, :] + ox - point[0], axis[:, None] + oy - point[1])
    t = np.clip((d - flat_m) / max(blend_m if blend_m is not None else 1.5 * flat_m, 1e-6), 0.0, 1.0)
    return dict(terrain, heights=(zero + (h - zero) * (t * t * (3 - 2 * t))).astype(np.float32)), zero


def drape(buildings, terrain, zero):
    """Sets every building onto the ground: 'z' = the ground under it (its mean, relative to the height 'zero' at
    the facade), 'foot' = a little under its lowest corner, so a house on a slope has no gap under it."""
    for b in buildings:
        h = terrain_at(terrain, b["rings"][0]) - zero
        b["z"], b["foot"] = float(h.mean()), float(h.min()) - 0.3
    return buildings


def ground_grid(terrain, zero, radius_m):
    """The ground as blender_export.py builds it: {"x", "y": sample positions in metres around the place, "z": (n, n)
    heights relative to 'zero', row = y}, cut to the square that holds radius_m."""
    h, step = terrain["heights"], terrain["step"]
    n = h.shape[0]
    ox, oy = terrain.get("offset", (0.0, 0.0))
    axis = (np.arange(n) - n // 2) * step
    kx, ky = np.abs(axis + ox) <= radius_m + step, np.abs(axis + oy) <= radius_m + step
    return {"x": (axis + ox)[kx].astype(np.float32), "y": (axis + oy)[ky].astype(np.float32),
            "z": (h[ky][:, kx] - zero).astype(np.float32)}


def within(buildings, radius_m):
    """The buildings with a corner inside radius_m of the place (what a download of that radius would hold)."""
    return [b for b in buildings if float(np.linalg.norm(b["rings"][0], axis=1).min()) <= radius_m]


def _number(v):
    """'12', '12.5 m', '12,5', \"40'\", '40 ft' -> metres, or None."""
    m = re.match(r"\s*(-?\d+(?:[.,]\d+)?)\s*(m|ft|')?", str(v or ""))
    if not m:
        return None
    x = float(m.group(1).replace(",", "."))
    return x * 0.3048 if m.group(2) in ("ft", "'") else x


def _join(ways):
    """Open ways that share end points -> closed rings (multipolygon members come in pieces)."""
    rings, open_ = [], []
    for w in ways:
        if len(w) >= 4 and w[0] == w[-1]:
            rings.append(w)
        elif len(w) >= 2:
            open_.append(list(w))
    while open_:
        cur = open_.pop()
        grown = True
        while grown and cur[0] != cur[-1]:
            grown = False
            for k, w in enumerate(open_):
                if w[0] == cur[-1]:
                    cur += w[1:]
                elif w[-1] == cur[-1]:
                    cur += w[-2::-1]
                elif w[-1] == cur[0]:
                    cur = w[:-1] + cur
                elif w[0] == cur[0]:
                    cur = w[:0:-1] + cur
                else:
                    continue
                open_.pop(k)
                grown = True
                break
        if len(cur) >= 4 and cur[0] == cur[-1]:
            rings.append(cur)
    return rings


def inside(pt, ring):
    """Point in polygon (ring: (N, 2), not repeated at the end), even-odd."""
    x, y = pt
    a, b = ring, np.roll(ring, -1, axis=0)
    cross = (a[:, 1] > y) != (b[:, 1] > y)
    dy = np.where(cross, b[:, 1] - a[:, 1], 1.0)
    return bool((cross & (x < a[:, 0] + (y - a[:, 1]) * (b[:, 0] - a[:, 0]) / dy)).sum() % 2)


def _area(ring):
    return 0.5 * float(np.sum(ring[:, 0] * np.roll(ring[:, 1], -1) - np.roll(ring[:, 0], -1) * ring[:, 1]))


def buildings_from_osm(data, lat, lon):
    """
    Overpass JSON -> [{"rings": [outer, hole ...] as (N, 2) arrays in metres around (lat, lon), "height": m or
    None, "base": m, "how": 'height' | 'levels' | '', "part": building:part, "id": 'way/123'}].
    """
    my, mx = metres_per_degree(lat)

    def ring(geom):
        r = np.array([((g["lon"] - lon) * mx, (g["lat"] - lat) * my) for g in geom], float)
        if len(r) > 1 and np.allclose(r[0], r[-1]):
            r = r[:-1]
        keep = np.linalg.norm(r - np.roll(r, 1, axis=0), axis=1) > 0.01          # doubled nodes
        r = r[keep]
        return r if len(r) >= 3 and abs(_area(r)) > 1.0 else None

    out = []
    for el in data.get("elements", []):
        tags = el.get("tags") or {}
        part = tags.get("building:part", "no") != "no"
        if not part and tags.get("building", "no") == "no":
            continue
        if el.get("type") == "way":
            groups = [([el.get("geometry") or []], [])]
        else:
            mem = [m for m in el.get("members", []) if m.get("type") == "way" and m.get("geometry")]
            key = lambda g: [(p["lat"], p["lon"]) for p in g]  # noqa: E731
            back = lambda w: [{"lat": a, "lon": b} for a, b in w]  # noqa: E731
            outers = [back(w) for w in _join([key(m["geometry"]) for m in mem if m.get("role") != "inner"])]
            inners = [back(w) for w in _join([key(m["geometry"]) for m in mem if m.get("role") == "inner"])]
            groups = [([o], inners) for o in outers]
        height, levels = _number(tags.get("height")), _number(tags.get("building:levels"))
        base = _number(tags.get("min_height"))
        if base is None:
            base = (_number(tags.get("building:min_level")) or 0.0) * LEVEL_M
        how = "height" if height else ("levels" if levels else "")
        if not height and levels:
            height = (levels + (_number(tags.get("roof:levels")) or 0.0)) * LEVEL_M
        for (outer,), holes in groups:
            o = ring(outer)
            if o is None:
                continue
            hs = [h for h in (ring(g) for g in holes) if h is not None and inside(h.mean(0), o)]
            out.append({"rings": [o] + hs, "height": height if height and height > base else None, "base": float(base),
                        "how": how, "part": part, "id": f"{el.get('type')}/{el.get('id')}"})
    return out


def fill_heights(buildings, default_m=0.0):
    """Buildings without a height get default_m, or the median of the tagged ones around (12 m if none).
    -> {"height": n, "levels": n, "guessed": n, "default_m": m}"""
    known = [b["height"] for b in buildings if b["height"] and not b["part"]]
    d = float(default_m) if default_m and default_m > 0 else (float(np.median(known)) if known else FALLBACK_M)
    n = {"height": 0, "levels": 0, "guessed": 0}
    for b in buildings:
        if not b["height"]:
            b["height"], b["how"] = b["base"] + d, ""
        n[b["how"] or "guessed"] += 1
    return dict(n, default_m=round(d, 1))


# --------------------------------------------------------------------------
# the building itself: which footprint, which way it faces
# --------------------------------------------------------------------------

def _boxes(buildings):
    """(N, 4) x0, y0, x1, y1 of every footprint: a cheap first look before the point-in-polygon test."""
    if not buildings:
        return np.zeros((0, 4))
    return np.array([[*b["rings"][0].min(0), *b["rings"][0].max(0)] for b in buildings], float)


def _near(buildings, pt, reach_m=0.0, boxes=None):
    """Indices of the footprints whose box lies within reach_m of the point (0 = the point is in the box)."""
    bx = _boxes(buildings) if boxes is None else boxes
    if not len(bx):
        return []
    dx = np.maximum(np.maximum(bx[:, 0] - pt[0], pt[0] - bx[:, 2]), 0.0)
    dy = np.maximum(np.maximum(bx[:, 1] - pt[1], pt[1] - bx[:, 3]), 0.0)
    return np.flatnonzero(np.hypot(dx, dy) <= reach_m).tolist()


def _edge_dist(pt, ring):
    """(distance, nearest point, edge index) from a point to a ring's edges."""
    a, b = ring, np.roll(ring, -1, axis=0)
    d = b - a
    t = np.clip(((pt - a) * d).sum(1) / np.maximum((d * d).sum(1), 1e-12), 0, 1)
    near = a + d * t[:, None]
    dist = np.linalg.norm(near - pt, axis=1)
    k = int(np.argmin(dist))
    return float(dist[k]), near[k], k


def own_building(buildings, point=(0.0, 0.0)):
    """
    The footprint the place sits on (or is nearest to, within 30 m) and the wall it means.
    -> {"index", "point": the place moved onto that wall, "facing_deg": compass direction the wall faces
    (0 = north, 90 = east), "distance_m"} or None. The direction is the length-weighted one of the footprint's
    edges near the place that run roughly like the nearest edge (map walls come in short pieces).
    """
    pt = np.asarray(point, float)
    best = None
    for i in _near(buildings, pt, OWN_MAX_M):
        b = buildings[i]
        if b["part"]:
            continue
        dist, near, k = _edge_dist(pt, b["rings"][0])
        rank = 0.0 if inside(pt, b["rings"][0]) else dist
        if best is None or (rank, dist) < best[:2]:
            best = (rank, dist, i, near, k)
    if best is None or best[0] > OWN_MAX_M:
        return None
    _, dist, i, near, k = best
    ring = buildings[i]["rings"][0]
    a, b = ring, np.roll(ring, -1, axis=0)
    d = b - a
    ang = np.arctan2(d[:, 1], d[:, 0])
    mid_d = np.linalg.norm((a + b) / 2 - near, axis=1)
    diff = np.abs(np.angle(np.exp(2j * (ang - ang[k])))) / 2                    # edges are lines: fold by 180
    pick = (mid_d < 25.0) & (diff < np.radians(20))
    pick[k] = True
    z = (np.linalg.norm(d, axis=1)[pick] * np.exp(2j * ang[pick])).sum()
    wall = np.angle(z) / 2
    n = np.array([-math.sin(wall), math.cos(wall)])
    if inside(near + n * 0.5, ring) and not inside(near - n * 0.5, ring):
        n = -n                                                                    # outwards
    return {"index": i, "point": near, "distance_m": dist,
            "facing_deg": float(math.degrees(math.atan2(n[0], n[1])) % 360.0)}


def without_own(buildings, own):
    """The buildings without the footprint of the building itself and the parts that stand on it."""
    ring = buildings[own["index"]]["rings"][0]
    return [b for i, b in enumerate(buildings)
            if i != own["index"] and not (b["part"] and inside(b["rings"][0].mean(0), ring))]


COMPASS = ("north", "north-east", "east", "south-east", "south", "south-west", "west", "north-west")


def compass(deg):
    return COMPASS[int(((deg % 360) + 22.5) // 45) % 8]


# --------------------------------------------------------------------------
# map -> model
# --------------------------------------------------------------------------

def facade_anchor(vert, pt, nrm, ground_z):
    """
    Where the map's place lands on the model: the middle of the model along its wall, on the main facade plane, on
    the ground. vert: the model's vertices; pt, nrm: the main plane (scene_ids.main_plane). -> (anchor (3,), unit
    horizontal normal towards the audience, unit wall axis: to the right as the audience sees it).
    """
    nrm = np.asarray(nrm, float)
    nh = np.array([nrm[0], nrm[1], 0.0])
    if np.linalg.norm(nh) < 0.2:
        raise ValueError("the model's main plane is close to horizontal: there is no facade to put the surroundings at")
    nh /= np.linalg.norm(nh)
    ex = np.cross([0.0, 0.0, 1.0], nh)
    t = np.asarray(vert, float) @ ex
    mid = (float(t.min()) + float(t.max())) / 2
    d = (float(np.asarray(pt, float) @ nrm) - ground_z * nrm[2]) / float(nrm @ nh)
    a = ex * mid + nh * d
    a[2] = ground_z
    return a, nh, ex


def placement(anchor, nh, ex, facing_deg, point=(0.0, 0.0), turn_deg=0.0, along_m=0.0, out_m=0.0, lift_m=0.0):
    """
    4 x 4 matrix (nested lists) from map metres (x east, y north, z up) to the model: the place 'point' lands on the
    anchor and the compass direction facing_deg on the facade normal nh. turn / along / out / lift move the
    surroundings against the model afterwards (degrees around the anchor; metres along the wall, towards the
    audience, up).
    """
    f = math.radians(facing_deg)
    phi = math.atan2(nh[1], nh[0]) - math.atan2(math.cos(f), math.sin(f)) + math.radians(turn_deg)
    c, s = math.cos(phi), math.sin(phi)
    M = np.eye(4)
    M[:2, :2] = [[c, -s], [s, c]]
    p = np.array([point[0], point[1], 0.0])
    M[:3, 3] = np.asarray(anchor, float) + ex * along_m + nh * out_m + np.array([0.0, 0.0, lift_m]) - M[:3, :3] @ p
    return M.tolist()


def nudge(anchor, nh, ex, turn_deg=0.0, along_m=0.0, out_m=0.0, lift_m=0.0, scale=1.0):
    """4 x 4 matrix for a surroundings file that is already in the model's space: scale, then turn around the
    anchor and move along the wall / towards the audience / up."""
    c, s = math.cos(math.radians(turn_deg)), math.sin(math.radians(turn_deg))
    M = np.eye(4)
    M[:2, :2] = [[c, -s], [s, c]]
    M[:3, :3] *= float(scale)
    a = np.asarray(anchor, float)
    M[:3, 3] = a - np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]) @ a + ex * along_m + nh * out_m + np.array([0.0, 0.0, lift_m])
    return M.tolist()


def stands_inside(buildings, M, point, boxes=None):
    """
    True when a point of the model's space (x, y, z) stands inside one of the buildings (M: placement() / nudge()):
    within its footprint, not in a courtyard, between its base and its roof. A camera there sees a black room, a
    projector there lights nothing.
    """
    q = np.linalg.inv(np.asarray(M, float)) @ np.append(np.asarray(point, float), 1.0)
    for i in _near(buildings, q[:2], boxes=boxes):
        b = buildings[i]
        z = b.get("z", 0.0)                                 # the ground under it (terrain), 0 on flat ground
        if (z + b.get("base", 0.0) <= q[2] <= z + b["height"] and inside(q[:2], b["rings"][0])
                and not any(inside(q[:2], h) for h in b["rings"][1:])):
            return True
    return False


def free_spot(buildings, M, spots):
    """The first of the spots (model space, (N, 3)) that does not stand inside a building; the first one if all do."""
    boxes = _boxes(buildings)
    return np.asarray(next((p for p in spots if not stands_inside(buildings, M, p, boxes)), spots[0]), float)


# --------------------------------------------------------------------------
# the sample street (no map, no download)
# --------------------------------------------------------------------------

def sample(width_m=24.0):
    """A square around a facade width_m wide that faces south (facing 180): neighbours in the row, a row across the
    square (the projector stands on it) and a block with a courtyard. -> (buildings, facing_deg), in the form of
    buildings_from_osm()."""
    def box(x0, x1, y0, y1, h):
        return {"rings": [np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], float)], "height": float(h), "base": 0.0,
                "how": "height", "part": False, "id": "sample"}

    w = width_m / 2
    out = [box(-w - 18, -w - 0.5, 0, 14, 15), box(w + 0.5, w + 22, 0, 16, 19), box(-w - 42, -w - 19, 1, 13, 12),
           box(w + 23, w + 40, 0, 12, 22)]
    x, hs = -w - 44.0, (14, 17, 13, 18, 16, 21, 15)
    for k, h in enumerate(hs):                                   # the row across the square, 70 m away
        d = 13.0 + 3.0 * (k % 3)
        out.append(box(x, x + d, -86, -70, h))
        x += d + (9.0 if k == 3 else 0.6)                        # one gap: a side street
    yard = box(-w - 70, -w - 48, -20, 10, 11)                    # a block with a courtyard
    yard["rings"].append(np.array([[-w - 64, -14], [-w - 54, -14], [-w - 54, 4], [-w - 64, 4]], float))
    return out + [yard], 180.0


# --------------------------------------------------------------------------
# the file blender_export.py reads
# --------------------------------------------------------------------------

def write(path, buildings, meta=None):
    """Footprints as JSON (street_doc(), plus meta) -> path. Written aside and renamed."""
    _dump(path, dict(meta or {}, kind="kubakub surroundings", buildings=street_doc(buildings)))
    return path


def street_doc(buildings):
    """The buildings as the footprint file holds them: rings in metres (2 decimals), height, base, and on real
    ground 'z' / 'foot' (drape())."""
    return [dict({"rings": [np.round(r, 2).tolist() for r in b["rings"]], "height": round(float(b["height"]), 2),
                  "base": round(float(b.get("base", 0.0)), 2)},
                 **({"z": round(b["z"], 2), "foot": round(b["foot"], 2)} if "z" in b else {})) for b in buildings]


def read(path):
    """A footprint file written by write() -> buildings (rings as arrays); [] for anything else (an own mesh file)."""
    if not str(path).lower().endswith(".json") or not os.path.isfile(path):
        return []
    with open(path, encoding="utf-8") as f:
        doc = json.load(f)
    return [dict(b, rings=[np.array(r, float) for r in b["rings"]], height=float(b["height"]), base=float(b.get("base", 0.0)))
            for b in doc.get("buildings", [])]


def _to_px(anchor, nh, ex, radius_m, size):
    """-> px(points (N, 3) of the model's space) = (N, 2) plan pixels; pt(point) = one pixel as an int tuple."""
    k = size / (2.0 * radius_m)

    def px(p):
        q = np.asarray(p, float).reshape(-1, 3) - anchor
        return np.stack([size / 2 + (q @ ex) * k, size / 2 + (q @ nh) * k], 1).astype(np.int32)

    return px, lambda p: tuple(int(v) for v in px(p)[0])


def plan(buildings, M, anchor, nh, ex, model_xyz, radius_m, size=768, projector=None, own_ring=None, viewer=None):
    """
    Top view for checking the fit, the audience side at the bottom: the surroundings where they land around the
    model (grey, taller = lighter), the model's outline (orange), the footprint that was left out (violet line),
    the projector (orange dot) and the audience camera 'viewer' (violet ring). M: placement() / nudge(); anchor,
    nh, ex: facade_anchor(). -> (size, size, 3) float32.
    """
    import cv2
    M = np.asarray(M, float)
    px, pt = _to_px(anchor, nh, ex, radius_m, size)

    def world(r):                                            # map metres (N, 2) -> model space
        return np.concatenate([r, np.zeros((len(r), 1))], 1) @ M[:3, :3].T + M[:3, 3]

    img = np.full((size, size, 3), 0.06, np.float32)
    for b in buildings:
        shade = 0.22 + 0.5 * min(float(b["height"]), 40.0) / 40.0
        cv2.fillPoly(img, [px(world(r)) for r in b["rings"]], (shade, shade, shade), lineType=cv2.LINE_AA)
    if own_ring is not None:
        cv2.polylines(img, [px(world(own_ring))], True, VIOLET, 1, cv2.LINE_AA)
    cv2.polylines(img, [cv2.convexHull(px(model_xyz))], True, ORANGE, 2, cv2.LINE_AA)
    if projector is not None:
        cv2.circle(img, pt(projector), 5, ORANGE, -1, cv2.LINE_AA)
    if viewer is not None:
        c = pt(viewer)
        cv2.circle(img, c, 7, VIOLET, 2, cv2.LINE_AA)
        cv2.putText(img, "audience", (c[0] + 11, c[1] + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.42, VIOLET, 1, cv2.LINE_AA)
    return img


def plan_lights(img, lights, anchor, nh, ex, radius_m, viewer=None, look_at=None, lens_mm=None):
    """
    A copy of a plan() picture (same anchor, nh, ex, radius_m) with the lamps on it: lamps in world space
    (scene_view.lights_to_world), each in its colour with its number and height; point = dot, area = square, spot =
    dot with its aim, sun = arrow from where it shines. viewer: the audience camera (violet ring), for a plan()
    drawn without it; with look_at (a point it looks at) an arrow shows its direction, with lens_mm (36 mm sensor)
    two lines its angle of view.
    """
    import cv2
    img = img.copy()
    size = img.shape[0]
    _, pt = _to_px(anchor, nh, ex, radius_m, size)
    if viewer is not None:
        c = pt(viewer)
        if look_at is not None:
            d = np.asarray(pt(look_at), float) - c
            if np.linalg.norm(d) > 1e-6:
                a = math.atan2(d[1], d[0])
                tip = lambda ang, r: (int(c[0] + math.cos(ang) * r), int(c[1] + math.sin(ang) * r))  # noqa: E731
                if lens_mm:
                    half = math.atan(18.0 / float(lens_mm))
                    for sgn in (-1, 1):
                        cv2.line(img, c, tip(a + sgn * half, size * 0.34), tuple(0.55 * v for v in VIOLET), 1, cv2.LINE_AA)
                cv2.arrowedLine(img, c, tip(a, size * 0.11), VIOLET, 2, cv2.LINE_AA, tipLength=0.3)
        cv2.circle(img, c, 7, VIOLET, 2, cv2.LINE_AA)
        cv2.putText(img, "audience", (c[0] + 11, c[1] + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.42, VIOLET, 1, cv2.LINE_AA)
    for n, L in enumerate(lights or []):
        col = tuple(float(v) for v in np.clip(np.asarray(L.get("color", (1, 1, 1)), float) * 0.85 + 0.15, 0, 1))
        if L["type"] == "sun":                               # an arrow at the rim, along the way the light travels
            d = np.array([np.dot(L["direction"], ex), np.dot(L["direction"], nh)])
            d = d / max(np.linalg.norm(d), 1e-6)
            tail = np.array([size / 2, size / 2]) - d * size * 0.44
            c = tuple(int(v) for v in tail)
            cv2.arrowedLine(img, c, tuple(int(v) for v in tail + d * size * 0.09), col, 2, cv2.LINE_AA, tipLength=0.35)
            label = f"{n + 1} sun {L.get('elevation', 0):g} deg"
        else:
            c = pt(L["location"])
            if L["type"] == "spot" and "target" in L:
                cv2.line(img, c, pt(L["target"]), col, 1, cv2.LINE_AA)
            if L["type"] == "area":
                cv2.rectangle(img, (c[0] - 6, c[1] - 6), (c[0] + 6, c[1] + 6), col, -1, cv2.LINE_AA)
            else:
                cv2.circle(img, c, 6, col, -1, cv2.LINE_AA)
            label = f"{n + 1} {L['type']} {L.get('height', 0):g} m"
        cv2.putText(img, label, (c[0] + 10, c[1] + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0.92, 0.92, 0.92), 1, cv2.LINE_AA)
    return img
