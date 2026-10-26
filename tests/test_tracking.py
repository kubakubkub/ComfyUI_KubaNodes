"""
Test of kubakub tracker / track draw (kubakub/tracking.py, kubakub/trackdraw.py, nodes/motion/nodes_track.py) and of
the live preview's drawing logic (web/kubakub_track.js, run with node when it is installed): the same states, labels,
lines and stamps as the full size render.

The model finders (YOLO, MediaPipe, SAM 2) run when their files are on disk, with every network connection of the
process blocked, so a test that passes also proves they stay offline; without the files they are skipped and said so.

Run from the ComfyUI_windows_portable folder:
    python_embeded\\python.exe ComfyUI\\custom_nodes\\ComfyUI_KubaNodes\\tests\\test_tracking.py
"""

import json
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(HERE))))     # ComfyUI (folder_paths, comfy_api)
sys.path.insert(0, os.path.dirname(HERE))
import folder_paths  # noqa: E402

TMP = tempfile.mkdtemp(prefix="kub_track_")
folder_paths.set_temp_directory(os.path.join(TMP, "temp"))
folder_paths.set_user_directory(os.path.join(TMP, "user"))          # the tracks cache goes there, not into the real one

sys.path.insert(0, HERE)
import _pack  # noqa: E402,F401
from kubapack.kubakub import trackdraw as td  # noqa: E402
from kubapack.kubakub import tracking as tk  # noqa: E402
from kubapack.nodes.motion import nodes_track as nt  # noqa: E402

failures = []
connections = []


def check(name, cond, detail=""):
    print(("ok   " if cond else "FAIL ") + name + (f"  ({detail})" if detail and not cond else ""))
    if not cond:
        failures.append(name)


def no_network(self, addr):
    connections.append(addr)
    raise OSError("network blocked by the test")


socket.socket.connect = no_network

try:
    # --- finders on the sample clip --------------------------------------------------------------
    clip = tk.sample_clip()
    check("sample clip", clip.shape == (72, 360, 640, 3) and clip.dtype == np.float32)
    d = tk.analyse(clip, 25, json.dumps({"motion": {"on": False}, "lights": {"on": True, "level": 0.9}})).to_dict()
    check("lights: the moving light is one track for the whole clip", len(d["tracks"]) == 1 and len(d["tracks"][0]["box"]) == 72,
          str([len(t["box"]) for t in d["tracks"]]))
    xs = [b[0] + b[2] / 2 for b in d["tracks"][0]["box"]]
    check("lights: it moves to the right, smoothly", xs[-1] - xs[0] > 400 and max(np.diff(xs)) < 15)
    d = tk.analyse(clip, 25, json.dumps({"motion": {"on": False}, "colour": {"on": True, "hex": "#f28a59", "tolerance": 0.2}})).to_dict()
    check("colour: the orange disc, one track, its own colour", len(d["tracks"]) == 1 and d["tracks"][0]["colour"].startswith("#f"),
          str([(t["colour"], len(t["box"])) for t in d["tracks"]]))
    check("colour: an outline", len(d["tracks"][0].get("outline", {})) > 60)
    d = tk.analyse(clip, 25, "").to_dict()
    check("motion (default): the walkers, ids from 1, at least 5 long tracks", sum(len(t["box"]) > 25 for t in d["tracks"]) >= 5
          and min(t["id"] for t in d["tracks"]) >= 1, str([len(t["box"]) for t in d["tracks"]]))
    check("tracks carry what the web file reads", all(k in d for k in ("W", "H", "n", "fps", "tracks", "curves", "skeleton", "names"))
          and d["format"] == tk.TRACKS_FORMAT and d["curves"]["count"][40] > 0)
    one = tk.analyse(clip, 25, "", only=[30]).to_dict()
    check("one frame alone (try this frame) finds the walkers too", 3 <= len(one["tracks"]) <= 12 and all(t["t0"] == 30 for t in one["tracks"]),
          str(len(one["tracks"])))
    check("settings: unreadable text = the defaults", tk.settings("{not json")["motion"]["on"] is True and tk.settings("")["analysis"] == "960")

    # --- ids over time ---------------------------------------------------------------------------
    per = [[tk.Det("yolo", (10 + 5 * t, 20, 30, 60), "person"), tk.Det("yolo", (300 - 5 * t, 20, 30, 60), "person")] for t in range(30)]
    per[12] = per[12][:1]                                              # one is missed for a frame
    lives = tk.associate(per, keep_lost=4)
    check("two walkers crossing keep two ids, a missed frame does not break one",
          len(lives) == 2 and sorted(len(L.dets) for L in lives) == [29, 30], str([len(L.dets) for L in lives]))
    built = tk.build_tracks(lives, [np.zeros((100, 400, 3), np.uint8)] * 30, 1.0, tk.settings(""), 30, 25, 400, 100).to_dict()
    check("the gap is filled (30 boxes) and marked unseen", all(len(t["box"]) == 30 for t in built["tracks"])
          and any(t["seen"][12] == 0 for t in built["tracks"]))

    # --- drawing ---------------------------------------------------------------------------------
    d = tk.analyse(clip, 25, json.dumps({"motion": {"on": True}, "lights": {"on": True, "level": 0.9},
                                         "colour": {"on": True, "hex": "#f28a59", "tolerance": 0.2}})).to_dict()
    for name in td.PRESETS:
        rgb, a = td.render_frame(d, td.look(name), 50, clip[50], lambda i: clip[i])
        check(f"preset '{name}' draws", rgb.shape == (360, 640, 3) and float(np.abs(rgb - clip[50]).max()) > 0.3)
    lk = td.look(json.dumps({"background": "transparent", "marks": [{"type": "box"}], "connect": []}))
    rgb, a = td.render_frame(d, lk, 50, clip[50], lambda i: clip[i])
    check("transparent: alpha only where something is drawn", 0 < float(a.mean()) < 0.1)
    lk = td.look(json.dumps({"background": "transparent", "marks": [], "connect": [], "stamp": {"on": True, "every": 8}}))
    early = td.render_frame(d, lk, 5, clip[5], lambda i: clip[i])[1].sum()
    late = td.render_frame(d, lk, 70, clip[70], lambda i: clip[i])[1].sum()
    check("stamp trail grows over time", late > early * 3, f"{early:.0f} -> {late:.0f}")
    check("look: a preset name, a JSON text and nonsense all give a full look",
          td.look("constellation")["background"] == "black" and td.look('{"background": "black"}')["marks"] and td.look("{x")["marks"])
    P = [(0, 0), (10, 0), (0, 10), (10, 10), (5, 5)]
    check("network of a square with a centre: 8 lines", len(td.delaunay_edges(P)) == 8, str(td.delaunay_edges(P)))
    check("tree joins 5 points with 4 lines, hull = 4 corners", len(td.mst_edges(P)) == 4 and len(td.hull_idx(P)) == 4)

    # --- the nodes -------------------------------------------------------------------------------
    res = nt.KUBA_Tracker.execute(None, json.dumps({"lights": {"on": True, "level": 0.9}}), 25.0)
    tracks, found, near, frames, fps_out, report, depth_out = res.result
    meta = res.ui["kubakub_track"][0]
    check("tracker node: nothing connected = the sample clip, said in the report; its fps", "sample clip" in report and frames.shape[0] == 72 and fps_out == 25.0)
    check("tracker node: found mask per frame", tuple(found.shape) == (72, 360, 640) and 0 < float(found.mean()) < 0.2)
    check("tracker node: near = a glow, 1 on the light, softer around it", tuple(near.shape) == (72, 360, 640)
          and float(near.max()) > 0.99 and 0 < float(near.mean()) < 0.2 and float(((near > 0.05) & (near < 0.95)).sum()) > 0)
    ar = tk.main_arrow(tracks, 40)
    check("main arrow: the light moves right", ar is not None and ar[2] > ar[0] and all(0 <= v <= 1 for v in ar), str(ar))
    check("tracker node: frames and tracks for the web file in temp",
          os.path.isfile(os.path.join(folder_paths.get_temp_directory(), meta["tracks"]["subfolder"], meta["tracks"]["filename"]))
          and meta["frames"]["n"] == 72)
    imgs, alpha = nt.KUBA_TrackDraw.execute(tracks, frames, "diagnostic").result
    check("track draw node: every frame at the video's size", tuple(imgs.shape) == (72, 360, 640, 3) and tuple(alpha.shape) == (72, 360, 640))
    imgs, alpha = nt.KUBA_TrackDraw.execute(tracks, None, "constellation").result
    check("track draw node: without the video, on black", float(imgs[40].mean()) < 0.2)
    # the tracks cache: a second run with the same video and settings does not track again
    res_again = nt.KUBA_Tracker.execute(None, json.dumps({"lights": {"on": True, "level": 0.9}}), 25.0)
    check("cache: the same video and settings come from the cache (survives a restart: on disk)",
          "from the cache" in res_again.result[5] and len(res_again.result[0]["tracks"]) == len(tracks["tracks"])
          and os.path.isdir(nt.cache_dir()) and len(os.listdir(nt.cache_dir())) >= 1, res_again.result[5])
    keep_code = nt.CODE
    nt.CODE = "changed"                                      # newer tracking code: the old tracks are not used
    res_new_code = nt.KUBA_Tracker.execute(None, json.dumps({"lights": {"on": True, "level": 0.9}}), 25.0)
    nt.CODE = keep_code
    check("cache: tracks of older tracking code are not used", "from the cache" not in res_new_code.result[5])
    res_other = nt.KUBA_Tracker.execute(None, json.dumps({"lights": {"on": True, "level": 0.8}}), 25.0)
    check("cache: another setting tracks again", "from the cache" not in res_other.result[5])
    # chained trackers: one set of tracks, ids go on, a kind found twice gets a number
    first = nt.KUBA_Tracker.execute(None, json.dumps({"motion": {"on": False}, "colour": {"on": True, "hex": "#f28a59", "tolerance": 0.2}}), 25.0).result[0]
    both = nt.KUBA_Tracker.execute(None, json.dumps({"motion": {"on": False}, "colour": {"on": True, "hex": "#bfb8ad", "tolerance": 0.15},
                                                     "lights": {"on": True, "level": 0.9}}), 25.0, tracks=first)
    kinds_b = sorted({tr["kind"] for tr in both.result[0]["tracks"]})
    ids_b = [tr["id"] for tr in both.result[0]["tracks"]]
    check("chain: the tracker before + this one, ids unique, the second colour is 'colour 2'",
          "colour" in kinds_b and "colour 2" in kinds_b and "lights" in kinds_b and len(ids_b) == len(set(ids_b)) and "chained:" in both.result[5],
          f"{kinds_b} {both.result[5][-200:]}")
    try:
        short = {**first, "n": 10}
        nt.KUBA_Tracker.execute(None, "", 25.0, tracks=short)
        check("chain: another video is refused, saying why", False)
    except ValueError as e:
        check("chain: another video is refused, saying why", "same video" in str(e))
    check("track draw is an output node ('render the look' can queue it alone)", nt.KUBA_TrackDraw.define_schema().is_output_node)
    # depth: any depth clip (here a synthetic one: far at the top, near at the bottom)
    ramp = torch.linspace(0.1, 0.9, 360)[:, None].repeat(1, 640)
    depth_in = ramp[None, ..., None].repeat(72, 1, 1, 3)
    res = nt.KUBA_Tracker.execute(None, json.dumps({"lights": {"on": True, "level": 0.9}}), 25.0, depth=depth_in)
    dtracks, dout, dreport = res.result[0], res.result[6], res.result[5]
    zl = [tr for tr in dtracks["tracks"] if tr["kind"] == "lights"][0]["z"]
    check("depth: every track has its z per frame, the light high up = far", len(zl) == 72 and max(zl) < 0.35, str(zl[:5]))
    check("depth: the depth goes out normalised (near = white), said in the report",
          tuple(dout.shape) == (72, 360, 640, 3) and float(dout[0, -1, 0, 0]) > 0.9 and float(dout[0, 0, 0, 0]) < 0.1 and "depth:" in dreport)
    res2 = nt.KUBA_Tracker.execute(None, json.dumps({"lights": {"on": True, "level": 0.9}, "depth_near": "dark"}), 25.0, depth=depth_in)
    zd = [tr for tr in res2.result[0]["tracks"] if tr["kind"] == "lights"][0]["z"]
    check("depth: 'near is dark' turns it round (Marigold)", min(zd) > 0.65, str(zd[:5]))
    for kind in ("layers", "lines", "time", "fog", "slice"):
        lk_d = json.dumps({"background": "video", "marks": [], "connect": [], "depth": [{"type": kind, "count": 6}]})
        img_d = nt.KUBA_TrackDraw.execute(dtracks, frames, lk_d, dout).result[0]
        check(f"depth look '{kind}' draws", float((img_d[40] - frames[40]).abs().max()) > 0.1)
    img_nodepth = nt.KUBA_TrackDraw.execute(dtracks, frames, json.dumps({"background": "video", "marks": [], "connect": [], "depth": [{"type": "layers"}]})).result[0]
    check("depth look without a depth: the video as it is", float((img_nodepth[40] - frames[40]).abs().max()) < 1e-5)
    big = td.render_frame(dtracks, td.look(json.dumps({"background": "transparent", "connect": [], "marks": [{"type": "box", "depth_scale": 2.0, "width": 2, "kinds": ["lights"]}]})),
                          40, None, None)[1]
    small = td.render_frame(dtracks, td.look(json.dumps({"background": "transparent", "connect": [], "marks": [{"type": "box", "depth_scale": 0, "width": 2, "kinds": ["lights"]}]})),
                            40, None, None)[1]
    check("depth scale: a far thing's box is drawn smaller", 0 < float(big.sum()) < float(small.sum()), f"{big.sum():.0f} vs {small.sum():.0f}")
    # time: every element its own span (0-1 of the clip) and fade
    sp_lk = {"background": "transparent", "connect": [], "marks": [{"type": "box", "width": 3, "span": [0.5, 1.0]}]}
    early = td.render_frame(dtracks, td.look(json.dumps(sp_lk)), 10, None, None)[1].sum()
    late = td.render_frame(dtracks, td.look(json.dumps(sp_lk)), 60, None, None)[1].sum()
    check("time: a mark only in its span (second half of the clip)", early == 0 and late > 0, f"{early} {late}")
    fe = {"span": [0.2, 0.8], "span_fade": 0.4}
    ws = [td.span_weight(fe, t, dtracks) for t in (10, 15, 18, 35, 50, 56, 60)]
    check("time: lines' 'fade with distance' is not a time fade", td.span_weight({"span": [0.2, 0.8], "fade": True}, 15, dtracks) == 1.0)
    check("time: fades in and out over 'span_fade' seconds", ws[0] == 0 and 0 < ws[1] < 1 and ws[3] == 1 and 0 < ws[5] < 1 and ws[6] == 0, str(ws))
    st_lk = {"background": "transparent", "connect": [], "marks": [], "stamp": {"on": True, "every": 8, "span": [0.5, 1.0]}}
    a_mid = td.render_frame(dtracks, td.look(json.dumps(st_lk)), 40, frames[40].numpy(), lambda i: frames[i].numpy())[1].sum()
    st_lk["stamp"]["span"] = [0.0, 1.0]
    a_all = td.render_frame(dtracks, td.look(json.dumps(st_lk)), 40, frames[40].numpy(), lambda i: frames[i].numpy())[1].sum()
    check("time: the stamp trail starts at its span (no stamps from before)", 0 < a_mid < a_all, f"{a_mid:.0f} vs {a_all:.0f}")
    # blend modes, each element its own
    base_lk = {"background": "video", "connect": [], "marks": [{"type": "box", "colour": "#f18a58", "width": 6, "fill": 1.0}]}
    nrm = td.render_frame(dtracks, td.look(json.dumps(base_lk)), 40, frames[40].numpy(), None)[0]
    base_lk["marks"][0]["blend"] = "difference"
    dif = td.render_frame(dtracks, td.look(json.dumps(base_lk)), 40, frames[40].numpy(), None)[0]
    check("blend: a mark in difference differs from normal, the rest of the picture does not",
          float(np.abs(nrm - dif).max()) > 0.2 and float(np.abs(nrm[:5, :5] - dif[:5, :5]).max()) < 1e-6)
    try:
        nt.KUBA_TrackDraw.execute({"x": 1}, None, "")
        check("track draw node: tracks not from the tracker are refused", False)
    except ValueError:
        check("track draw node: tracks not from the tracker are refused", True)

    # --- the live preview draws the same (node) --------------------------------------------------
    node = shutil.which("node")
    if node is None:
        print("skip the JavaScript checks: node is not installed")
    else:
        with open(os.path.join(HERE, "..", "web", "kubakub_track.js"), encoding="utf-8") as f:
            src = f.read()
        src = re.sub(r"(?m)^import \{ app \} from .*$", "const app = { registerExtension() {} };", src)
        src = re.sub(r"(?m)^import \{ api \} from .*$", "const api = {};", src)
        src = src.replace('const HERE = new URL(".", import.meta.url).href;', 'const HERE = "";')
        rng = np.random.default_rng(3)
        data = {"W": 1280, "H": 720, "n": 40, "fps": 25, "tracks": [], "names": tk.Tracks(1, 1, 1, 1).to_dict()["names"],
                "skeleton": tk.Tracks(1, 1, 1, 1).to_dict()["skeleton"]}
        for i in range(14):
            t0, n = int(rng.integers(0, 20)), int(rng.integers(5, 30))
            x, y = rng.uniform(0, 1200), rng.uniform(0, 650)
            vx, vy = rng.uniform(-6, 6), rng.uniform(-4, 4)
            boxes = [[round(x + vx * k, 1), round(y + vy * k * (k % 3 != 0), 1), round(20 + 3 * i, 1), round(30 + 2 * i, 1)] for k in range(n)]
            data["tracks"].append({"id": i + 1, "kind": ["yolo", "motion", "lights"][i % 3], "label": "x", "t0": t0, "box": boxes,
                                   "seen": [1 if k % 5 else 0 for k in range(n)], "colour": "#aabbcc",
                                   **({"z": [round(float(v), 3) for v in rng.uniform(0, 1, n)]} if i % 4 else {})})
        lk = td.look("")
        modes = [{"mode": m, "k": 2, "distance": 0.2, "hub": h, "length": 6, "same_kind": s}
                 for m, h, s in [("nearest", "", False), ("within", "", True), ("network", "", False), ("tree", "", False),
                                 ("hull", "", False), ("hub", "biggest", False), ("hub", "top right", False), ("chain", "", True),
                                 ("trail", "", False), ("corners", "", False), ("network", "", True)]]
        modes.append({"mode": "within", "distance": 0.4, "hub": "", "same_kind": False, "same_depth": 0.2})
        py = {"states": [], "labels": [], "edges": [], "stamps": []}
        for t in range(0, 40, 3):
            items = td.visible(data, t, lk)
            py["states"].append([[e[0]["id"], e[1]] for e in items])
            py["labels"].append([td.label_text("{id} {id_hex} {id_long} {state} {x} {y} {z} {speed} {size} {data} {frame} {t}", e[0], e[1], e[2], t, data, lk) for e in items])
            py["edges"].append([[[round(v, 3) for v in s[0]] + [round(v, 3) for v in s[1]] for s in td.connect_edges(c, items, data, lk, t)] for c in modes])
        py["stamps"] = [[list(map(lambda v: round(v, 3), s)) for s in td.stamp_points(tr, 9, 720 / 1080)] for tr in data["tracks"]]
        py["spans"] = [[round(td.span_weight({"span": [a, b], "span_fade": f}, t, data), 3) for t in range(40)] for a, b, f in ((0.1, 0.6, 0), (0.25, 0.9, 0.3), (0, 1, 1))]
        src += f"\nconst data = {json.dumps(data)}, lk = lookOf('', {{}}, {json.dumps(td.DEFAULT_LOOK)}), modes = {json.dumps(modes)};\n"
        src += ("const out = {states: [], labels: [], edges: [], stamps: []}; const r3 = v => Math.round(v * 1000) / 1000;\n"
                "for (let t = 0; t < 40; t += 3) { const items = visible(data, t, lk);\n"
                "  out.states.push(items.map(e => [e[0].id, e[1]]));\n"
                "  out.labels.push(items.map(e => labelText('{id} {id_hex} {id_long} {state} {x} {y} {z} {speed} {size} {data} {frame} {t}', e[0], e[1], e[2], t, data, lk)));\n"
                "  out.edges.push(modes.map(c => connectEdges(c, items, data, lk, t).map(s => [...s[0].map(r3), ...s[1].map(r3)]))); }\n"
                "out.stamps = data.tracks.map(tr => stampPoints(tr, 9, 720 / 1080).map(s => s.map(r3)));\n"
                "out.spans = [[0.1, 0.6, 0], [0.25, 0.9, 0.3], [0, 1, 1]].map(([a, b, f]) => Array.from({length: 40}, (_, t) => r3(spanWeight({span: [a, b], span_fade: f}, t, data))));\n"
                "console.log(JSON.stringify(out));\n")
        with tempfile.TemporaryDirectory() as dd:
            path = os.path.join(dd, "track.mjs")
            with open(path, "w", encoding="utf-8") as f:
                f.write(src)
            run = subprocess.run([node, path], capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60)
        if run.returncode != 0:
            check("the preview's drawing logic runs (node)", False, run.stderr.strip()[-500:])
        else:
            js = json.loads(run.stdout.strip().splitlines()[-1])
            for key in ("states", "labels", "stamps", "spans"):
                diff = next(((x, y) for fj, fp in zip(js[key], py[key]) for x, y in zip(fj if isinstance(fj, list) else [fj], fp if isinstance(fp, list) else [fp]) if x != y), None)
                check(f"preview = render: {key}", js[key] == py[key], f"first difference: {diff}")
            for mi, c in enumerate(modes):
                a = [sorted(map(tuple, f[mi])) for f in js["edges"]]
                b = [sorted(map(tuple, f[mi])) for f in py["edges"]]
                check(f"preview = render: lines '{c['mode']}'{' ' + c['hub'] if c['hub'] else ''}{' same kind' if c['same_kind'] else ''}{' same depth' if c.get('same_depth') else ''}", a == b,
                      next((f"frame {i}: {x[:3]} vs {y[:3]}" for i, (x, y) in enumerate(zip(a, b)) if x != y), ""))

    # --- tracks drive other nodes: the shader, the director's 'follow' ---------------------------
    from kubapack.kubakub.director import motion as mo  # noqa: E402
    from kubapack.kubakub.director import render as rd  # noqa: E402
    light = tk.analyse(clip, 25, json.dumps({"motion": {"on": False}, "lights": {"on": True, "level": 0.9}})).to_dict()
    try:
        from kubapack.nodes.shader import nodes_shader as ns
        r = ns.KUBA_Shader.execute("rings", width=320, height=180, tracks=light)
        check("shader with tracks: as long as the tracks, says it follows them",
              r.result[0].shape[0] == 72 and "tracks:" in r.result[5], r.result[5][-300:])
        a, b = r.result[0][5].numpy(), r.result[0][65].numpy()
        check("shader with tracks: the rings move with the light", float(np.abs(a - b).mean()) > 0.02)
    except Exception as e:  # noqa: BLE001
        print(f"skip the shader with tracks ({type(e).__name__}: {str(e).splitlines()[0][:120]})")
    doc = {"version": 1, "canvas": [1280, 720], "timeline": {"duration": 2.88, "fps": 25}, "tracks": mo.compact_tracks(light),
           "layers": [{"id": "L1", "name": "dot", "kind": "shape", "x": 100, "y": 100, "w": 50, "h": 50, "shape": {"type": "ellipse"},
                       "motion": [{"type": "follow", "path": "position", "track": "biggest", "amount": 1}]},
                      {"id": "base", "name": "base", "kind": "base"}]}
    ctx = mo.context(doc)
    xs = [rd.animate(doc, t, ctx)["layers"][0]["x"] for t in (0.0, 1.0, 2.0)]
    check("director follow: the layer moves right with the light, scaled to the canvas (x2)",
          xs[0] == 100 and xs[2] - xs[0] > 600, str(xs))
    check("director follow: nothing tracked = no move", rd.animate(dict(doc, tracks=None), 2.0, mo.context(dict(doc, tracks=None)))["layers"][0]["x"] == 100)
    if node:
        dsrc = open(os.path.join(HERE, "..", "web", "kubakub_director.js"), encoding="utf-8").read()
        body = dsrc[dsrc.index("function followPick"):dsrc.index("\n}\n", dsrc.index("function followOne")) + 3]
        js = (body + "\nconst mNum = (b, k, d) => { const v = b[k]; return typeof v === 'number' && Number.isFinite(v) ? v : d; };\n"
              f"const tracks = {json.dumps(mo.compact_tracks(light))}, ctx = {{ tracks, canvas: [1280, 720] }};\n"
              "const out = [];\nfor (const b of [{type: 'follow', path: 'position', track: 'biggest', amount: 1}, {type: 'follow', path: 'scale', track: '1', amount: 0.5, t_start: 0.4}])\n"
              "  for (const t of [0, 0.37, 1.5, 2.8]) out.push(followOne(b, t, b.t_start || 0, 2.88, ctx).map(([p, v]) => [p, Math.round(v * 1000) / 1000]));\n"
              "console.log(JSON.stringify(out));\n")
        with tempfile.TemporaryDirectory() as dd:
            path = os.path.join(dd, "follow.mjs")
            open(path, "w", encoding="utf-8").write(js)
            run = subprocess.run([node, path], capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60)
        pyv = [[[p, round(v, 3)] for p, v in mo._one(b, t, 2.88, ctx)]
               for b in ({"type": "follow", "path": "position", "track": "biggest", "amount": 1}, {"type": "follow", "path": "scale", "track": "1", "amount": 0.5, "t_start": 0.4})
               for t in (0, 0.37, 1.5, 2.8)]
        check("director follow: the window moves it as the render does (node)", run.returncode == 0 and json.loads(run.stdout) == pyv,
              (run.stderr or run.stdout)[-300:] + " vs " + str(pyv)[:200])

    # --- the model finders, offline --------------------------------------------------------------
    yolo = tk.list_models("yolo")
    sam2 = tk.list_models("sam2")
    if not yolo:
        print("skip YOLO / MediaPipe: no weights in models/ultralytics")
    else:
        dd = tk.analyse(clip[:4], 25, json.dumps({"motion": {"on": False}, "yolo": {"on": True, "model": yolo[0]}, "body": {"on": True}})).to_dict()
        check("YOLO + MediaPipe body run without the network", isinstance(dd["tracks"], list) and not connections, str(connections))
    connections.clear()
    for kind, word in (("pose", "body points"), ("seg", "outlines")):
        f = next((m for m in yolo if f"-{kind}" in m), None)
        if f is None:
            print(f"skip YOLO {kind}: no *-{kind}.pt in models/ultralytics")
            continue
        try:
            yd = tk.YoloFinder({"model": f, "classes": ["person"], "confidence": 0.05}, 1)
            check(f"YOLO {kind} loads with the installed ultralytics ({word})", yd.task in ("pose", "segment"), yd.task)
            yd.close()
        except Exception as e:  # noqa: BLE001
            check(f"YOLO {kind} loads with the installed ultralytics ({word})", False, str(e).splitlines()[0][:200])
    dino = tk.list_models("dino")
    if not dino:
        print("skip Grounding DINO: no weights in models/grounding-dino")
    else:
        try:
            df = tk.DinoFinder({"model": next((m for m in dino if "swint" in m.lower()), dino[0]), "words": "orange disc, white dot", "confidence": 0.2})
            dets = df(tk.frame_rgb8(clip[20]), 20)
            df.close()
            check("Grounding DINO finds named things offline", isinstance(dets, list) and all(d.kind == "words" for d in dets) and not connections,
                  f"{[(d.label, round(d.score, 2)) for d in dets]} {connections}")
        except Exception as e:  # noqa: BLE001
            check("Grounding DINO finds named things offline", False, str(e).splitlines()[0][:200])
    if not sam2:
        print("skip SAM 2: no weights in models/sam2")
    else:
        pick = next((m for m in sam2 if "tiny" in m), sam2[0])
        dd = tk.analyse(clip[:12], 25, json.dumps({"motion": {"on": False}, "motif": {"on": True, "model": pick, "points": [[0, 0.78, 0.5, 1, 0]]}}),
                        work_dir=os.path.join(TMP, "motif")).to_dict()
        check("SAM 2 follows a clicked motif through the clip, offline", len(dd["tracks"]) >= 1 and len(dd["tracks"][0]["box"]) >= 10
              and dd["tracks"][0].get("outline") and not connections, str([len(t["box"]) for t in dd["tracks"]]))
finally:
    shutil.rmtree(TMP, ignore_errors=True)

print()
print("all tracking tests passed" if not failures else f"{len(failures)} FAILED: " + ", ".join(failures))
sys.exit(1 if failures else 0)
