// kubakub master timeline: the "kubakub timeline" tab of ComfyUI's bottom panel, above the directors.
// The directors of the workflow are blocks on it, rendered clips (output/) and dropped files (input/kuba_director) are
// clips. A clip is one segment of the director's video model (= kubakub/director/media.py: start / in / out / speed /
// length / fill / reverse, seconds), so trim, cut and speed work as on a video layer in the director.
// The document lives in the workflow (workflow.extra.kubakub_master); fps and size come from kubakub project settings
// when the workflow has one. Plain ES module, no build step. Look: kubakub.art (Hanken Grotesk, #f18a58 actions,
// #a187b7 selection), like the director window.
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const HERE = new URL(".", import.meta.url).href;
const ORANGE = "#f18a58", LAV = "#a187b7";
const KEY = "kubakub_master";
const DRAG_MIME = "application/x-kubakub-master";
const PRESETS = { "hd 1920x1080": [1920, 1080], "uhd 3840x2160": [3840, 2160], "dci 4k 4096x2160": [4096, 2160], "8k 7680x4320": [7680, 4320] };
const VIDEO_RE = /\.(mp4|mov|webm|mkv|avi|m4v|gif)$/i, IMAGE_RE = /\.(png|jpe?g|webp|bmp|tiff?)$/i, AUDIO_RE = /\.(wav|mp3|ogg|flac|m4a|aac|opus)$/i;
const kindOf = name => (VIDEO_RE.test(name) ? "video" : IMAGE_RE.test(name) ? "image" : AUDIO_RE.test(name) ? "audio" : null);
const COLS = "kubakub.timeline.columns";            // the column widths, per browser
const COL_DEF = { preview: 240, list: 210, insp: 220 };
const HEADER_W = 112, RULER_H = 24, TRACK_H = 42, EDGE = 7, SNAP_PX = 8, UNDO_MAX = 200;

// ------------------------------------------------------------------------------------------------ model
const uid = p => p + Math.random().toString(36).slice(2, 9);
const r4 = v => +(+v).toFixed(4);
const clamp = (v, a, b) => Math.min(b, Math.max(a, v));
export const wholeFps = v => { const f = Math.round(+v); return Number.isFinite(f) ? clamp(f, 1, 120) : 25; };   // = project.whole_fps
const defProps = () => ({ opacity: 1, x: 0, y: 0, scale: 1, fit: "fit" });
export const ANIM = ["opacity", "x", "y", "scale"];   // the picture settings that take keys
export const EASES = ["smooth", "linear", "hold"];
const newTrack = name => ({ id: uid("t"), name, muted: false, hidden: false, items: [] });
export function newDoc() {
  return { version: 1, duration: 10, fps: 25, width: 3840, height: 2160, tracks: [newTrack("v2"), newTrack("v1")], media: [] };
}
export function migrate(d) {                         // an older / partial document -> the current one (in place)
  if (!d || typeof d !== "object") return newDoc();
  const n = newDoc();
  d.version = 1;
  d.duration = clamp(+d.duration || n.duration, 0.5, 3600);
  d.fps = wholeFps(d.fps || n.fps);
  d.width = Math.round(+d.width || n.width); d.height = Math.round(+d.height || n.height);
  d.media = Array.isArray(d.media) ? d.media.filter(m => m && m.id && m.source) : [];
  d.tracks = Array.isArray(d.tracks) && d.tracks.length ? d.tracks : n.tracks;
  for (const t of d.tracks) {
    t.id = t.id || uid("t"); t.name = String(t.name || "v");
    t.items = (Array.isArray(t.items) ? t.items : []).filter(i => i && (i.type === "clip" || i.type === "director"));
    for (const i of t.items) {
      i.id = i.id || uid("i"); i.props = Object.assign(defProps(), i.props || {});
      const anim = {};
      for (const p of ANIM) {
        const ks = (Array.isArray(i.anim?.[p]) ? i.anim[p] : []).filter(k => k && Number.isFinite(k.t) && Number.isFinite(k.v));
        if (ks.length) anim[p] = ks.map(k => ({ t: k.t, v: k.v, e: EASES.includes(k.e) ? k.e : "smooth" })).sort((a, b) => a.t - b.t);
      }
      if (Object.keys(anim).length) i.anim = anim; else delete i.anim;
    }
  }
  return d;
}

// one segment, normalised exactly like mediaSegs() in kubakub_director.js / media.segments() in Python
export function segOf(c, dur) {
  const num = (k, d) => (typeof c?.[k] === "number" && Number.isFinite(c[k]) ? c[k] : d);
  const D = +dur || 0, tin = D ? Math.min(Math.max(num("in", 0), 0), Math.max(0, D - 1e-3)) : Math.max(num("in", 0), 0);
  let tout = num("out", -1); if (tout <= tin || (D && tout > D)) tout = D;
  const speed = clamp(num("speed", 1), 0.01, 100), natural = Math.max(1e-3, (tout - tin) / speed), len = num("length", -1);
  return { start: num("start", 0), in: tin, out: tout, speed, natural, length: len <= 0 ? natural : len,
    fill: c?.fill === "loop" ? "loop" : "hold", reverse: !!c?.reverse };
}
// seconds into the file at timeline time t (= media.source_time for one segment), or null outside it
export function srcTime(s, t) {
  if (!(s.start - 1e-9 <= t && t < s.start + s.length - 1e-9)) return null;
  let local = (t - s.start) * s.speed; const span = s.out - s.in;
  if (local >= span) local = s.fill === "loop" && span > 1e-9 ? local % span : span;
  return s.reverse ? Math.max(s.in, s.out - local - 1e-6) : s.in + local;
}
// a director block: `in` seconds into the director's own timeline, `length` (-1 = to its end)
export function dirSpan(it, dirDur, fps) {
  const tin = Math.max(0, +it.in || 0), natural = Math.max(1 / fps, (+dirDur || 10) - tin);
  return { start: +it.start || 0, in: tin, length: +it.length > 0 ? +it.length : natural, natural };
}
// [start, end) of an item on the master timeline
export function spanOf(it, env) {
  if (it.type === "director") { const s = dirSpan(it, env.dirDur(it.node), env.fps); return [s.start, s.start + s.length]; }
  const s = segOf(it, env.mediaDur(it.media)); return [s.start, s.start + s.length];
}
// cut an item at t: [left, right] (left = the item itself), or null when t is not inside it. Clips like Ctrl K in
// the director (splitVideo): the frame shown at every time stays the same.
export function splitItem(it, t, env) {
  const [a, b] = spanOf(it, env); if (t - a < 1 / env.fps - 1e-9 || b - t < 1 / env.fps - 1e-9) return null;
  const B = { ...it, id: uid("i"), props: { ...it.props }, start: r4(t) };
  if (it.anim) { B.anim = JSON.parse(JSON.stringify(it.anim)); shiftKeys(B, a - t); }
  if (it.type === "director") {
    const s = dirSpan(it, env.dirDur(it.node), env.fps);
    B.in = r4(s.in + (t - s.start)); B.length = +it.length > 0 ? r4(b - t) : -1; it.length = r4(t - a);
    return [it, B];
  }
  const s = segOf(it, env.mediaDur(it.media)), ts = srcTime(s, t), inFill = (t - s.start) >= s.natural - 1e-9;
  const explicit = +it.length > 0;
  if (!inFill && s.reverse) { it.in = r4(ts); it.length = -1; B.out = r4(ts); B.length = explicit ? r4(b - t) : -1; }
  else if (!inFill) { it.out = r4(ts); it.length = -1; B.in = r4(ts); B.length = explicit ? r4(b - t) : -1; }
  else {
    it.length = r4(t - s.start); B.length = r4(b - t);
    if (s.fill === "hold") { const f = 1 / (env.mediaFps(it.media) || env.fps); B.in = r4(Math.max(s.in, s.out - f)); B.out = r4(s.out); B.speed = 1; }
  }
  return [it, B];
}
// move an item's left edge to t (trim in), like the left grip in the director; its keys stay where they are in time
export function trimLeftTo(it, t, env) {
  const s0 = +it.start || 0; trimLeft(it, t, env); shiftKeys(it, s0 - it.start);
}
function trimLeft(it, t, env) {
  const fps = env.fps;
  if (it.type === "director") {
    const s = dirSpan(it, env.dirDur(it.node), fps), nin = Math.max(0, s.in + (t - s.start)), start = s.start + (nin - s.in);
    if (+it.length > 0) it.length = r4(Math.max(1 / fps, s.length - (start - s.start)));
    it.in = r4(nin); it.start = r4(Math.max(0, start)); return;
  }
  const dur = env.mediaDur(it.media), s = segOf(it, dur);
  if (!dur) {                                        // a picture (no time inside it): only its start moves, the end stays
    const end = s.start + s.length, start = Math.min(Math.max(0, t), end - 1 / fps);
    it.start = r4(start); it.length = r4(end - start); return;
  }
  const nin = Math.min(Math.max(0, s.in + (t - s.start) * s.speed), s.out - 1 / fps), start = s.start + (nin - s.in) / s.speed;
  if (+it.length > 0) it.length = r4(Math.max(1 / fps, s.length - (start - s.start)));
  it.in = r4(nin); it.start = r4(Math.max(0, start));
}
// move an item's right edge to t: trim out (or a shorter length when it is held / looped / a director block)
export function trimRightTo(it, t, env) {
  const len = Math.max(1 / env.fps, t - (+it.start || 0));
  if (it.type === "director") { it.length = r4(len); return; }
  const dur = env.mediaDur(it.media), s = segOf(it, dur);
  if (+it.length > 0 || !dur) it.length = r4(len);
  else { it.out = r4(Math.min(dur, Math.max(s.in + 1 / env.fps, s.in + len * s.speed))); it.length = -1; }
}
// make room for [a, b) on a track: what overlaps it (except `keep`) is trimmed, cut or removed (overwrite)
export function clearRange(track, a, b, keep, env) {
  const out = [];
  for (const it of track.items) {
    const [s, e] = spanOf(it, env);
    if (keep.has(it.id) || e <= a + 1e-9 || s >= b - 1e-9) { out.push(it); continue; }
    if (s < a - 1e-9 && e > b + 1e-9) {
      const parts = splitItem(it, b, env);
      if (parts) { trimRightTo(parts[0], a, env); out.push(parts[0], parts[1]); } else { trimRightTo(it, a, env); out.push(it); }
    } else if (s < a - 1e-9) { trimRightTo(it, a, env); out.push(it); }
    else if (e > b + 1e-9) { trimLeftTo(it, b, env); out.push(it); }
    // else: covered completely -> removed
  }
  track.items = out.sort((p, q) => p.start - q.start);
}
// ---- keys: item.anim = { opacity | x | y | scale: [{ t, v, e }] }, t in seconds from the item's start (they move with it)
// keys sorted by t -> the value at t (= keyValue in kubakub_director.js / key_value in kubakub/director/render.py, numbers)
export function keyValue(keys, t) {
  if (t <= keys[0].t) return keys[0].v;
  const n = keys.length; if (t >= keys[n - 1].t) return keys[n - 1].v;
  let i = 0; while (i < n - 2 && keys[i + 1].t <= t) i++;
  const a = keys[i], b = keys[i + 1];
  if (a.e === "hold") return a.v;
  let u = (t - a.t) / Math.max(1e-9, b.t - a.t); if (a.e !== "linear") u = u * u * (3 - 2 * u);
  return a.v + (b.v - a.v) * u;
}
// the picture settings of an item at timeline time t: its props, the keyed ones at t
export function propsAt(it, t) {
  const p = { ...it.props };
  for (const k of ANIM) if (it.anim?.[k]?.length) p[k] = keyValue(it.anim[k], t - (+it.start || 0));
  return p;
}
export const offSpeed = speed => Math.abs(speed - 1) > 1e-3;   // no sound at another speed (as in the director and the render)
const keyAt = (keys, lt, fps) => keys?.find(k => Math.abs(k.t - lt) < 0.5 / fps);
const shiftKeys = (it, d) => { for (const ks of Object.values(it.anim || {})) for (const k of ks) k.t = r4(k.t + d); };
// a picture setting edited at timeline time t: a key there when it has keys (like the director), else the plain value
export function setProp(it, p, v, t, fps) {
  const ks = it.anim?.[p];
  if (!ks?.length) { it.props[p] = v; return; }
  const lt = r4(t - (+it.start || 0)), k = keyAt(ks, lt, fps);
  if (k) k.v = v; else { ks.push({ t: lt, v, e: "smooth" }); ks.sort((a, b) => a.t - b.t); }
}
// the key button: a key of p at t (with the value shown there), or that key removed (the last one leaves its value)
export function toggleKey(it, p, t, fps) {
  const lt = r4(t - (+it.start || 0)), ks = it.anim?.[p], k = keyAt(ks, lt, fps);
  if (k) {
    ks.splice(ks.indexOf(k), 1);
    if (!ks.length) { it.props[p] = k.v; delete it.anim[p]; if (!Object.keys(it.anim).length) delete it.anim; }
    return;
  }
  const v = propsAt(it, t)[p];
  it.anim = it.anim || {}; (it.anim[p] = ks || []).push({ t: lt, v, e: "smooth" }); it.anim[p].sort((a, b) => a.t - b.t);
}
// the times of an item's keys (seconds from its start), each once
export const keyTimes = it => [...new Set(Object.values(it.anim || {}).flatMap(ks => ks.map(k => k.t)))].sort((a, b) => a - b);
// the easing of every key of an item at t (seconds on the timeline)
export function easeAt(it, t, fps, e) {
  const lt = t - (+it.start || 0);
  for (const ks of Object.values(it.anim || {})) { const k = keyAt(ks, lt, fps); if (k) k.e = e; }
}

// where a picture of sw x sh goes in the W x H frame (centre-anchored, x / y in project pixels)
export function layoutRect(p, sw, sh, W, H) {
  let w = W, h = H;
  if (p.fit !== "stretch") { const k = p.fit === "fill" ? Math.max(W / sw, H / sh) : Math.min(W / sw, H / sh); w = sw * k; h = sh * k; }
  w *= p.scale; h *= p.scale;
  return { x: (W - w) / 2 + p.x, y: (H - h) / 2 + p.y, w, h };
}

// ------------------------------------------------------------------------------------------------ graph
const rootGraph = () => app.rootGraph || app.graph;
const nodesOf = type => (rootGraph()?._nodes || []).filter(n => (n.comfyClass || n.type) === type);
const widgetOf = (n, name) => n?.widgets?.find(w => w.name === name);
function directorDoc(node) { try { return JSON.parse(widgetOf(node, "document")?.value || "{}") || {}; } catch { return {}; } }
function directorInfo(node) {                       // = loadDocument() in kubakub_director.js
  const T = directorDoc(node).timeline || {};
  return { duration: clamp(+T.duration || 10, 0.5, 3600), fps: clamp(Math.round(+T.fps || 25), 1, 120) };
}
const directorOpen = node => { try { return window.kubakubDirector?.()?.node === node; } catch { return false; } };
function setDirectorFps(node, fps) {                 // false when its window is open (it saves its own copy on close)
  if (directorOpen(node)) return false;
  const w = widgetOf(node, "document"); if (!w) return false;
  const doc = directorDoc(node); doc.timeline = { ...(doc.timeline || {}), fps };
  w.value = JSON.stringify(doc); return true;
}
function projectNode() { return nodesOf("KUBA_Project")[0] || null; }
function projectValues(n) {                          // what kubakub project settings outputs (a connected matrix: unknown)
  const fps = wholeFps(widgetOf(n, "fps")?.value ?? 25), preset = widgetOf(n, "preset")?.value;
  const wh = PRESETS[preset] || [Math.round(+widgetOf(n, "width")?.value || 3840), Math.round(+widgetOf(n, "height")?.value || 2160)];
  return { fps, width: wh[0], height: wh[1], matrix: !!n.inputs?.find(i => i.name === "matrix" && i.link != null) };
}
function setWidget(n, name, v) { const w = widgetOf(n, name); if (!w) return; w.value = v; try { w.callback?.(v); } catch { /* the node's own handler */ } }
function nodeStill(node) {                            // the director's last picture (its preview after a run), or null
  const out = app.nodeOutputs?.[String(node.id)] || app.nodeOutputs?.[node.id];
  const im = out?.images?.[0];
  if (im) return api.apiURL(`/view?${new URLSearchParams({ filename: im.filename, subfolder: im.subfolder || "", type: im.type || "temp" })}`);
  return node.imgs?.[0]?.src || null;
}
function markChanged() { try { app.extensionManager?.workflow?.activeWorkflow?.changeTracker?.captureCanvasState?.(); } catch { /* older frontends */ } }
function sourceURL(source) {                          // 'file:<input/>' | 'output:<output/>' -> /view
  const [type, rest] = source.startsWith("output:") ? ["output", source.slice(7)] : ["input", source.replace(/^file:/, "")];
  const i = rest.lastIndexOf("/");
  return api.apiURL(`/view?${new URLSearchParams({ filename: rest.slice(i + 1), subfolder: i < 0 ? "" : rest.slice(0, i), type })}`);
}
async function postJSON(path, body) {
  const r = await api.fetchApi(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  const j = await r.json().catch(() => ({})); if (!r.ok) throw new Error(j.error || r.statusText); return j;
}

// ------------------------------------------------------------------------------------------------ store
class Store {
  constructor() { this.doc = newDoc(); this.extra = null; this.undoS = []; this.redoS = []; this.before = null; this.fns = new Set(); this.time = 0; this.sel = new Set(); }
  on(fn) { this.fns.add(fn); }
  emit(why) { for (const fn of this.fns) fn(why); }
  // the workflow changed (another tab, a load, ComfyUI's own undo): bind to its document
  bind(graph) {
    const extra = graph?.extra; if (!extra) return false;
    const saved = extra[KEY];
    if (extra === this.extra && (!saved || saved === this.doc)) return false;
    this.extra = extra; this.doc = saved ? migrate(saved) : newDoc(); if (saved) extra[KEY] = this.doc;
    if (!saved) {                                   // a first look: the directors one after another on v1
      let t = 0; const v1 = this.doc.tracks[this.doc.tracks.length - 1];
      for (const n of nodesOf("KUBA_Director")) {
        const d = directorInfo(n).duration; v1.items.push({ id: uid("i"), type: "director", node: n.id, start: r4(t), in: 0, length: -1, props: defProps() }); t += d;
      }
      if (t > 0) this.doc.duration = r4(Math.max(this.doc.duration, t));
    }
    this.undoS = []; this.redoS = []; this.sel.clear(); this.time = clamp(this.time, 0, this.doc.duration);
    this.emit("bind"); return true;
  }
  persist() { if (this.extra) this.extra[KEY] = this.doc; markChanged(); }
  edit(fn) { const b = JSON.stringify(this.doc); fn(this.doc); this.commit(b); }
  begin() { this.before = JSON.stringify(this.doc); }
  moved() { this.emit("drag"); }
  end() { const b = this.before; this.before = null; if (b != null) this.commit(b); }
  cancel() { if (this.before == null) return; this.restore(this.before); this.before = null; this.emit("edit"); }
  commit(b) {
    if (b === JSON.stringify(this.doc)) { this.emit("edit"); return; }
    this.undoS.push(b); if (this.undoS.length > UNDO_MAX) this.undoS.shift(); this.redoS = [];
    this.persist(); this.emit("edit");
  }
  restore(json) { const n = migrate(JSON.parse(json)); for (const k of Object.keys(this.doc)) delete this.doc[k]; Object.assign(this.doc, n); }
  undo() { if (!this.undoS.length) return; this.redoS.push(JSON.stringify(this.doc)); this.restore(this.undoS.pop()); this.prune(); this.persist(); this.emit("edit"); }
  redo() { if (!this.redoS.length) return; this.undoS.push(JSON.stringify(this.doc)); this.restore(this.redoS.pop()); this.prune(); this.persist(); this.emit("edit"); }
  *items() { for (const track of this.doc.tracks) for (const it of track.items) yield { track, it }; }
  find(id) { for (const x of this.items()) if (x.it.id === id) return x; return null; }
  selected() { return [...this.items()].filter(x => this.sel.has(x.it.id)); }
  prune() { for (const id of [...this.sel]) if (!this.find(id)) this.sel.delete(id); }
  media(id) { return this.doc.media.find(m => m.id === id) || null; }
  q(t) { return Math.round(t * this.doc.fps) / this.doc.fps; }      // snapped to a frame
  setTime(t) { t = clamp(this.q(t), 0, 36000); if (t === this.time) return; this.time = t; this.emit("time"); }
  env() {
    const fps = this.doc.fps;
    return { fps, mediaDur: id => this.media(id)?.info?.duration || 0, mediaFps: id => this.media(id)?.info?.fps || 0,
      dirDur: nid => { const n = rootGraph()?.getNodeById?.(nid); return n ? directorInfo(n).duration : 10; } };
  }
}

// ------------------------------------------------------------------------------------------------ preview
class Preview {
  constructor(store, cache) {
    this.store = store; this.cache = cache;
    this.cv = document.createElement("canvas"); this.ctx = this.cv.getContext("2d");
    this.videos = new Map(); this.images = new Map();
    this.playing = false; this.loop = true; this.from = 0; this.t0 = 0; this.dirty = true;
  }
  play() { if (this.playing) return; const s = this.store; if (s.time >= s.doc.duration - 1e-6) s.setTime(0); this.playing = true; this.resync(); }
  pause() { if (!this.playing) return; this.playing = false; for (const v of this.videos.values()) v.el.pause(); this.dirty = true; this.store.emit("time"); }
  toggle() { this.playing ? this.pause() : this.play(); }
  resync() { this.from = this.store.time; this.t0 = performance.now(); }
  active(t, ahead = 0) {                             // bottom track first (it is drawn first); hidden tracks still sound
    const s = this.store, env = s.env(), out = [];
    for (let k = s.doc.tracks.length - 1; k >= 0; k--) {
      const tr = s.doc.tracks[k];
      for (const it of tr.items) { const [a, b] = spanOf(it, env); if (t < b - 1e-9 && t + ahead >= a - 1e-9) out.push({ tr, it, on: t >= a - 1e-9 }); }
    }
    return out;
  }
  tick(now) {
    const s = this.store, D = s.doc.duration;
    if (this.playing) {
      let t = this.from + (now - this.t0) / 1000;
      if (t >= D) { if (this.loop) { this.from = 0; this.t0 = now; t = 0; } else { t = D; this.pause(); } }
      s.setTime(Math.floor(t * s.doc.fps + 1e-6) / s.doc.fps); this.dirty = true;
    }
    this.syncVideos();
    if (this.dirty) { this.dirty = false; this.draw(); }
  }
  syncVideos() {
    const s = this.store, t = s.time, need = new Set();
    for (const { tr, it, on } of this.active(t, 1.5)) {
      if (it.type !== "clip") continue;
      const m = s.media(it.media); if (!m || (m.kind !== "video" && m.kind !== "audio")) continue;
      if (tr.hidden && (tr.muted || it.muted || m.kind === "video" && m.info?.audio === false)) continue;   // nothing to see or hear
      const url = this.cache.playURL(m); if (!url) continue;
      need.add(it.id);
      let v = this.videos.get(it.id);
      if (v && v.url !== url) { this.drop(it.id); v = null; }
      if (!v) {
        const el = document.createElement(m.kind === "audio" ? "audio" : "video"); el.preload = "auto"; el.playsInline = true; el.src = url;
        const redraw = () => (this.dirty = true); el.addEventListener("seeked", redraw); el.addEventListener("loadeddata", redraw);
        v = { el, url }; this.videos.set(it.id, v);
      }
      const seg = segOf(it, m.info?.duration || 0), el = v.el;
      el.muted = tr.muted || !!it.muted || offSpeed(seg.speed);
      const ts = srcTime(seg, on ? t : seg.start) ?? seg.in;
      const fps = m.info?.fps || s.doc.fps, centre = Math.min(seg.out - 0.5 / fps, Math.floor(ts * fps + 1e-6) / fps + 0.5 / fps);
      const held = on && (t - seg.start) >= seg.natural - 1e-9 && seg.fill === "hold";
      if (this.playing && on && !held && !seg.reverse) {   // backwards: seek frame by frame (browsers cannot play reverse)
        const rate = clamp(seg.speed, 0.0625, 16); if (el.playbackRate !== rate) el.playbackRate = rate;
        if (el.readyState >= 1 && Math.abs(el.currentTime - ts) > 0.2) el.currentTime = ts;
        if (el.paused) el.play().catch(() => {});
      } else {
        if (!el.paused) el.pause();
        if (el.readyState >= 1 && Math.abs(el.currentTime - centre) > 0.5 / fps) el.currentTime = Math.max(0, centre);
      }
    }
    for (const id of [...this.videos.keys()]) if (!need.has(id)) this.drop(id);
  }
  drop(id) { const v = this.videos.get(id); if (!v) return; v.el.pause(); v.el.removeAttribute("src"); v.el.load(); this.videos.delete(id); }
  image(url) {
    let im = this.images.get(url);
    if (!im) { im = new Image(); im.onload = () => (this.dirty = true); im.src = url; this.images.set(url, im); }
    return im;
  }
  draw() {
    const s = this.store, { width: W, height: H } = s.doc, k = Math.min(1, 960 / W);
    const cw = Math.max(16, Math.round(W * k)), ch = Math.max(16, Math.round(H * k));
    if (this.cv.width !== cw || this.cv.height !== ch) { this.cv.width = cw; this.cv.height = ch; }
    const ctx = this.ctx; ctx.save(); ctx.globalAlpha = 1; ctx.fillStyle = "#000"; ctx.fillRect(0, 0, cw, ch); ctx.scale(k, k);
    for (const { tr, it, on } of this.active(s.time)) {
      if (!on || tr.hidden) continue;
      let src = null, sw = 0, sh = 0;
      if (it.type === "director") {
        const n = rootGraph()?.getNodeById?.(it.node), url = n && nodeStill(n);
        if (url) { const im = this.image(url); if (im.complete && im.naturalWidth) { src = im; sw = im.naturalWidth; sh = im.naturalHeight; } }
        if (!src) { this.card(ctx, W, H, n ? (n.title || "kubakub director") : "director removed", n ? "run the workflow once to see it here" : ""); continue; }
      } else {
        const m = s.media(it.media); if (!m || m.kind === "audio") continue;
        if (m.kind === "video") { const v = this.videos.get(it.id); if (!v || v.el.readyState < 2) continue; src = v.el; sw = m.info?.w || v.el.videoWidth; sh = m.info?.h || v.el.videoHeight; }
        else { const im = this.image(sourceURL(m.source)); if (!im.complete || !im.naturalWidth) continue; src = im; sw = im.naturalWidth; sh = im.naturalHeight; }
      }
      if (!sw || !sh) continue;
      const p = propsAt(it, s.time), r = layoutRect(p, sw, sh, W, H);
      ctx.globalAlpha = clamp(+p.opacity, 0, 1); ctx.drawImage(src, r.x, r.y, r.w, r.h);
    }
    ctx.restore();
  }
  card(ctx, W, H, title, sub) {                      // a director that has no picture yet
    ctx.globalAlpha = 1; ctx.fillStyle = "#2c2c2e"; ctx.fillRect(0, 0, W, H);
    ctx.fillStyle = ORANGE; ctx.font = `600 ${Math.round(H / 14)}px "Hanken Grotesk KKD", system-ui, sans-serif`; ctx.textAlign = "center"; ctx.textBaseline = "middle";
    ctx.fillText(title, W / 2, H / 2 - H / 20); ctx.fillStyle = "rgba(255,255,255,.55)"; ctx.font = `400 ${Math.round(H / 24)}px "Hanken Grotesk KKD", system-ui, sans-serif`;
    ctx.fillText(sub, W / 2, H / 2 + H / 20);
  }
}

// preview copies of videos: the director's own route makes them (once per file version); waveforms of the sound
class MediaCache {
  constructor(onReady) { this.urls = new Map(); this.pending = new Map(); this.waves = new Map(); this.onReady = onReady; }
  playURL(m) { if (m.kind !== "video") return sourceURL(m.source); this.ensure(m); return this.urls.get(m.source) || null; }
  // {duration, rate, peaks} of a clip's sound; null while it loads (then onReady redraws), false without one
  wave(m) {
    if (m.kind !== "audio" && !(m.kind === "video" && m.info?.audio)) return false;
    const w = this.waves.get(m.source); if (w !== undefined) return w;
    this.waves.set(m.source, null);
    this.waveOf(m.source).then(r => { this.waves.set(m.source, r); this.onReady(); }).catch(() => this.waves.set(m.source, false));
    return null;
  }
  waveOf(source) { return postJSON("/kubakub/master/wave", { source }); }
  ensure(m) {
    if (m.kind !== "video" || this.urls.has(m.source) || this.pending.has(m.source)) return this.pending.get(m.source);
    const p = postJSON("/kubakub/director/media", { source: m.source }).then(r => {
      const pr = r.proxy; this.urls.set(m.source, api.apiURL(`/view?${new URLSearchParams({ filename: pr.filename, subfolder: pr.subfolder, type: pr.type })}`));
      this.onReady(); return r.info;
    }).finally(() => this.pending.delete(m.source));
    this.pending.set(m.source, p); return p;
  }
  async info(source) {                               // {w, h, fps, duration, frames, audio} of a video
    const r = await postJSON("/kubakub/director/media", { source });
    const pr = r.proxy; this.urls.set(source, api.apiURL(`/view?${new URLSearchParams({ filename: pr.filename, subfolder: pr.subfolder, type: pr.type })}`));
    return r.info;
  }
}

// ------------------------------------------------------------------------------------------------ timeline canvas
class Timeline {
  constructor(store, panel) {
    this.store = store; this.panel = panel;
    this.cv = document.createElement("canvas"); this.cv.className = "kkt-tl"; this.ctx = this.cv.getContext("2d");
    this.pps = 60; this.scrollX = -0.2; this.scrollY = 0; this.snap = true; this.drag = null; this.ghost = null; this.snapAt = null; this.hover = null;
    this.w = 0; this.h = 0; this.dirty = true;
    const c = this.cv;
    c.addEventListener("pointerdown", e => this.down(e)); c.addEventListener("pointermove", e => this.move(e));
    c.addEventListener("pointerup", e => this.up(e)); c.addEventListener("pointercancel", () => this.cancel());
    c.addEventListener("wheel", e => this.wheel(e), { passive: false }); c.addEventListener("dblclick", e => this.dbl(e));
    c.addEventListener("contextmenu", e => this.context(e));
    new ResizeObserver(() => this.resize()).observe(c);
  }
  resize() {
    const r = this.cv.getBoundingClientRect(), dpr = window.devicePixelRatio || 1;
    this.w = r.width; this.h = r.height; this.cv.width = Math.max(1, Math.round(r.width * dpr)); this.cv.height = Math.max(1, Math.round(r.height * dpr)); this.dirty = true;
  }
  tx(t) { return HEADER_W + (t - this.scrollX) * this.pps; }
  xt(x) { return this.scrollX + (x - HEADER_W) / this.pps; }
  ty(i) { return RULER_H + i * TRACK_H - this.scrollY; }
  trackAt(y) { if (y < RULER_H) return -1; const i = Math.floor((y - RULER_H + this.scrollY) / TRACK_H); return i >= 0 && i < this.store.doc.tracks.length ? i : -1; }
  local(e) { const r = this.cv.getBoundingClientRect(); return { x: e.clientX - r.left, y: e.clientY - r.top }; }
  locate(cx, cy) {                                    // a drop: track (one past the last = a new track) and time
    const r = this.cv.getBoundingClientRect(), x = cx - r.left, y = cy - r.top;
    let ti = this.trackAt(y); if (ti < 0) ti = y >= RULER_H ? this.store.doc.tracks.length : 0;
    return { track: ti, t: Math.max(0, this.store.q(this.xt(Math.max(HEADER_W, x)))) };
  }
  hit(x, y) {
    const ti = this.trackAt(y); if (ti < 0 || x < HEADER_W) return null;
    const tr = this.store.doc.tracks[ti], env = this.store.env();
    for (let k = tr.items.length - 1; k >= 0; k--) {
      const it = tr.items[k], [a, b] = spanOf(it, env), x0 = this.tx(a), x1 = this.tx(b);
      if (x < x0 || x > x1) continue;
      const e = Math.min(EDGE, (x1 - x0) / 3);
      if (x - x0 <= e || x1 - x <= e) return { tr, ti, it, zone: x - x0 <= e ? "l" : "r" };   // the grips first
      if (it.anim && y >= this.ty(ti) + TRACK_H - 15) {
        const kt = keyTimes(it).find(k => Math.abs(this.tx(a + k) - x) <= 5);
        if (kt !== undefined) return { tr, ti, it, zone: "key", t: a + kt };
      }
      return { tr, ti, it, zone: "body" };
    }
    return null;
  }
  buttons(i) { const y = this.ty(i) + (TRACK_H - 18) / 2; return { mute: { x: HEADER_W - 46, y, w: 18, h: 18 }, hide: { x: HEADER_W - 24, y, w: 18, h: 18 } }; }
  // snap: the times of other items' edges, the playhead, 0 and the end; Ctrl = free (as in the director)
  snapD(edges, skip, e) {
    this.snapAt = null; if (!this.snap || e.ctrlKey) return 0;
    const s = this.store, env = s.env(), cands = [0, s.time, s.doc.duration];
    for (const { it } of s.items()) if (!skip.has(it.id)) cands.push(...spanOf(it, env));
    let best = null; const tol = SNAP_PX / this.pps;
    for (const c of cands) for (const g of edges) { const d = c - g; if (Math.abs(d) <= tol && (!best || Math.abs(d) < Math.abs(best.d))) best = { d, c }; }
    if (!best) return 0; this.snapAt = best.c; return best.d;
  }
  down(e) {
    const { x, y } = this.local(e), s = this.store; this.cv.setPointerCapture(e.pointerId);
    if (e.button === 1) { e.preventDefault(); this.drag = { type: "pan", x0: x, s0: this.scrollX }; return; }
    if (e.button !== 0) return;
    if (y < RULER_H) { if (x >= HEADER_W) { this.drag = { type: "scrub" }; this.seek(x); } return; }
    if (x < HEADER_W) {
      const ti = this.trackAt(y); if (ti < 0) return;
      const b = this.buttons(ti), inR = r => x >= r.x && x <= r.x + r.w && y >= r.y && y <= r.y + r.h;
      if (inR(b.mute)) s.edit(d => (d.tracks[ti].muted = !d.tracks[ti].muted));
      else if (inR(b.hide)) s.edit(d => (d.tracks[ti].hidden = !d.tracks[ti].hidden));
      return;
    }
    const h = this.hit(x, y);
    if (!h) { if (!e.shiftKey) s.sel.clear(); s.emit("sel"); this.drag = { type: "scrub" }; this.seek(x); return; }
    if (h.zone === "key") { s.sel.clear(); s.sel.add(h.it.id); s.emit("sel"); s.setTime(h.t); this.panel.seeked(); return; }
    const id = h.it.id; let click = false;
    if (e.shiftKey) s.sel.has(id) ? s.sel.delete(id) : s.sel.add(id);
    else if (!s.sel.has(id)) { s.sel.clear(); s.sel.add(id); } else click = true;
    s.emit("sel"); if (!s.sel.has(id)) return;
    const type = h.zone === "body" ? "move" : h.zone === "l" ? "l" : "r";
    const list = type === "move" ? s.selected() : [{ track: h.tr, it: h.it }];
    this.drag = { type, x0: x, y0: y, started: false, click, alt: e.altKey, shift: e.shiftKey && type === "r", main: h.it,
      list: list.map(({ track, it }) => ({ it, track, o: JSON.parse(JSON.stringify(it)) })) };
  }
  move(e) {
    const { x, y } = this.local(e), d = this.drag, s = this.store;
    if (!d) { this.cursor(x, y); return; }
    if (d.type === "pan") { this.scrollX = d.s0 - (x - d.x0) / this.pps; this.dirty = true; return; }
    if (d.type === "scrub") { this.seek(x); return; }
    if (!d.started) { if (Math.hypot(x - d.x0, y - d.y0) < 3) return; d.started = true; s.begin(); }
    const env = s.env(), dt = (x - d.x0) / this.pps, skip = new Set(d.list.map(l => l.it.id));
    if (d.type === "move") {
      const minS = Math.min(...d.list.map(l => l.o.start)), main = d.list.find(l => l.it === d.main);
      let dd = Math.max(s.q(dt), -minS);
      const [a, b] = spanOf(main.o, env); dd += this.snapD([a + dd, b + dd], skip, e); dd = Math.max(s.q(dd), -minS);
      for (const l of d.list) l.it.start = r4(l.o.start + dd);
      if (d.list.length === 1) {                      // one item: it can change track too
        const ti = this.trackAt(y), l = d.list[0], cur = s.doc.tracks.indexOf(l.track);
        if (ti >= 0 && ti !== cur) { l.track.items.splice(l.track.items.indexOf(l.it), 1); l.track = s.doc.tracks[ti]; l.track.items.push(l.it); }
      }
    } else {
      const l = d.list[0]; Object.assign(l.it, JSON.parse(JSON.stringify(l.o)));
      const [a, b] = spanOf(l.o, env);
      if (d.type === "l") { let t = s.q(a + dt); t += this.snapD([t], skip, e); trimLeftTo(l.it, Math.max(0, s.q(t)), env); }
      else {
        let t = s.q(b + dt); t += this.snapD([t], skip, e); t = s.q(t);
        const len = Math.max(1 / env.fps, t - l.o.start);
        if (l.it.type === "clip" && d.alt && env.mediaDur(l.o.media)) {           // Alt: speed (the same piece of the file plays in the new length)
          const sg = segOf(l.o, env.mediaDur(l.o.media)); l.it.speed = r4(clamp((sg.out - sg.in) / len, 0.05, 20)); l.it.length = -1;
        } else if (l.it.type === "clip" && d.shift) l.it.length = r4(len);   // Shift: length (then hold / loop)
        else trimRightTo(l.it, t, env);
      }
    }
    this.dirty = true; s.moved();
  }
  up(e) {
    const d = this.drag; this.drag = null; this.snapAt = null; if (!d || d.type === "pan" || d.type === "scrub") return;
    const s = this.store;
    if (!d.started) { if (d.click && !e.shiftKey) { s.sel.clear(); s.sel.add(d.main.id); s.emit("sel"); } return; }
    const env = s.env(), keep = new Set(d.list.map(l => l.it.id));
    for (const l of d.list) { const [a, b] = spanOf(l.it, env); clearRange(l.track, a, b, keep, env); }
    s.end();
  }
  cancel() { const d = this.drag; this.drag = null; this.snapAt = null; if (d?.started) this.store.cancel(); this.dirty = true; }
  seek(x) { this.store.setTime(Math.max(0, this.xt(x))); this.panel.seeked(); }
  cursor(x, y) {
    const h = y >= RULER_H ? this.hit(x, y) : null;
    this.cv.style.cursor = y < RULER_H && x >= HEADER_W ? "ew-resize" : h ? (h.zone === "body" ? "grab" : "col-resize") : "default";
    const hv = h && h.zone !== "body" ? h.it.id + h.zone : null; if (hv !== this.hover) { this.hover = hv; this.dirty = true; }
  }
  wheel(e) {
    e.preventDefault(); const { x } = this.local(e), xx = Math.max(HEADER_W, x);
    if (e.ctrlKey || e.altKey || e.metaKey) { const t = this.xt(xx); this.pps = clamp(this.pps * Math.exp(-e.deltaY * 0.0015), 0.5, 2000); this.scrollX = t - (xx - HEADER_W) / this.pps; }
    else if (e.shiftKey) this.scrollY = clamp(this.scrollY + e.deltaY, 0, Math.max(0, this.store.doc.tracks.length * TRACK_H - (this.h - RULER_H) + 8));
    else this.scrollX += (e.deltaX + e.deltaY) / this.pps;
    this.scrollX = Math.max(-20 / this.pps, this.scrollX); this.dirty = true;
  }
  dbl(e) {
    const { x, y } = this.local(e), ti = this.trackAt(y); if (ti < 0) return;
    if (x < HEADER_W) {
      const name = prompt("track name", this.store.doc.tracks[ti].name);
      if (name != null && name.trim()) this.store.edit(d => (d.tracks[ti].name = name.trim()));
      return;
    }
    const h = this.hit(x, y); if (h?.it.type === "director") this.panel.openDirector(h.it.node);
  }
  context(e) { e.preventDefault(); const { x, y } = this.local(e), ti = this.trackAt(y); if (x < HEADER_W && ti >= 0) this.panel.removeTrack(ti); }
  fit() {
    const s = this.store, env = s.env(); let end = s.doc.duration;
    for (const { it } of s.items()) end = Math.max(end, spanOf(it, env)[1]);
    this.pps = clamp((this.w - HEADER_W - 24) / Math.max(0.5, end), 0.5, 2000); this.scrollX = -8 / this.pps; this.dirty = true;
  }
  follow() { const x = this.tx(this.store.time); if (x > this.w - 10 || x < HEADER_W) { this.scrollX = this.store.time - 10 / this.pps; this.dirty = true; } }
  css() {
    const cs = getComputedStyle(this.cv), v = n => cs.getPropertyValue(n).trim();
    return { paper: v("--paper"), paper2: v("--paper2"), ink: v("--ink"), muted: v("--muted"), faint: v("--faint"), line: v("--line"), stage: v("--stage") };
  }
  draw() {
    if (!this.w || !this.h) return;
    const ctx = this.ctx, dpr = window.devicePixelRatio || 1, C = this.css(), s = this.store, doc = s.doc, env = s.env();
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.fillStyle = C.paper; ctx.fillRect(0, 0, this.w, this.h);
    ctx.font = '500 11px "Hanken Grotesk KKD", system-ui, sans-serif'; ctx.textBaseline = "middle";
    doc.tracks.forEach((tr, i) => {
      const y = this.ty(i); if (y + TRACK_H < RULER_H || y > this.h) return;
      ctx.fillStyle = i % 2 ? C.paper : C.paper2; ctx.fillRect(HEADER_W, y, this.w - HEADER_W, TRACK_H);
      ctx.fillStyle = C.line; ctx.fillRect(0, y + TRACK_H - 1, this.w, 1);
    });
    ctx.save(); ctx.beginPath(); ctx.rect(HEADER_W, RULER_H, this.w - HEADER_W, this.h - RULER_H); ctx.clip();
    const xe = this.tx(doc.duration); if (xe < this.w) { ctx.fillStyle = "rgba(0,0,0,.28)"; ctx.fillRect(Math.max(HEADER_W, xe), RULER_H, this.w, this.h); }
    doc.tracks.forEach((tr, i) => { const y = this.ty(i); if (y + TRACK_H < RULER_H || y > this.h) return; for (const it of tr.items) this.item(ctx, C, tr, it, y, env); });
    if (this.ghost) {
      const g = this.ghost, y = this.ty(Math.min(g.track, doc.tracks.length - 1)) + (g.track >= doc.tracks.length ? TRACK_H : 0);
      ctx.setLineDash([4, 3]); ctx.strokeStyle = ORANGE; ctx.lineWidth = 1.5; ctx.strokeRect(this.tx(g.t) + 0.5, y + 3.5, Math.max(6, 2 * this.pps), TRACK_H - 7); ctx.setLineDash([]);
    }
    if (this.snapAt != null) { ctx.fillStyle = LAV; ctx.fillRect(Math.round(this.tx(this.snapAt)), RULER_H, 1, this.h); }
    ctx.restore();
    ctx.fillStyle = C.paper2; ctx.fillRect(0, RULER_H, HEADER_W, this.h);
    doc.tracks.forEach((tr, i) => {
      const y = this.ty(i); if (y + TRACK_H < RULER_H || y > this.h) return;
      ctx.fillStyle = C.line; ctx.fillRect(0, y + TRACK_H - 1, HEADER_W, 1);
      ctx.fillStyle = tr.hidden ? C.faint : C.ink; ctx.textAlign = "left"; ctx.fillText(this.fitText(ctx, tr.name, HEADER_W - 60), 10, y + TRACK_H / 2);
      const b = this.buttons(i); this.toggle(ctx, C, b.mute, "m", tr.muted, ORANGE); this.toggle(ctx, C, b.hide, tr.hidden ? "–" : "◉", tr.hidden, LAV);
    });
    ctx.fillStyle = C.line; ctx.fillRect(HEADER_W - 1, RULER_H, 1, this.h);
    this.ruler(ctx, C, doc);
    const px = Math.round(this.tx(s.time)) + 0.5;
    if (px >= HEADER_W) {
      ctx.strokeStyle = ORANGE; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(px, RULER_H - 4); ctx.lineTo(px, this.h); ctx.stroke();
      ctx.fillStyle = ORANGE; ctx.beginPath(); ctx.arc(px, RULER_H - 7, 5, 0, Math.PI * 2); ctx.fill();
    }
    this.dirty = false;
  }
  item(ctx, C, tr, it, y, env) {
    const [a, b] = spanOf(it, env), x0 = this.tx(a), x1 = this.tx(b); if (x1 < HEADER_W || x0 > this.w) return;
    const s = this.store, sel = s.sel.has(it.id), w = Math.max(2, x1 - x0), y0 = y + 4, hh = TRACK_H - 9;
    const isDir = it.type === "director", n = isDir ? rootGraph()?.getNodeById?.(it.node) : null, m = isDir ? null : s.media(it.media);
    ctx.globalAlpha = tr.hidden ? 0.4 : 1;
    ctx.fillStyle = isDir ? (n ? C.stage : "#7a3a3a") : m ? ({ video: LAV, audio: "#5f7f9e", image: "#7d9a8a" }[m.kind] || LAV) : "#7a3a3a";
    this.round(ctx, x0, y0, w, hh, 6); ctx.fill();
    if (isDir) { ctx.fillStyle = ORANGE; ctx.fillRect(x0, y0 + 3, 3, hh - 6); }
    if (m) this.wave(ctx, it, m, x0, x1, y0, hh, tr.muted || it.muted || offSpeed(+it.speed || 1));
    if (sel) { ctx.strokeStyle = ORANGE; ctx.lineWidth = 2; this.round(ctx, x0 + 1, y0 + 1, w - 2, hh - 2, 5); ctx.stroke(); }
    if (w > 26) {
      ctx.save(); ctx.beginPath(); ctx.rect(x0 + 6, y0, w - 12, hh); ctx.clip(); ctx.fillStyle = "#fff"; ctx.textAlign = "left";
      const name = isDir ? (n ? (n.title || "kubakub director") : "director removed") : m ? m.name : "missing file";
      ctx.fillText(name, x0 + 9, y0 + 11);
      const tags = [];
      if (isDir && n) { const di = directorInfo(n); if (di.fps !== s.doc.fps) tags.push(`${di.fps} fps ≠ ${s.doc.fps}`); }
      if (!isDir && Math.abs((+it.speed || 1) - 1) > 1e-4) tags.push(`${+(+it.speed).toFixed(3)}×`);
      if (!isDir && it.reverse) tags.push("backwards");
      if (+it.props.opacity < 1) tags.push(`${Math.round(it.props.opacity * 100)} %`);
      if (tags.length) { ctx.fillStyle = "rgba(255,255,255,.75)"; ctx.fillText(tags.join(" · "), x0 + 9, y0 + 24); }
      ctx.restore();
    }
    if (it.anim) {
      ctx.fillStyle = sel ? ORANGE : "#fff";
      for (const kt of keyTimes(it)) {
        const kx = this.tx(a + kt); if (kx < x0 - 1 || kx > x1 + 1) continue;
        ctx.beginPath(); ctx.moveTo(kx, y0 + hh - 9); ctx.lineTo(kx + 4, y0 + hh - 5); ctx.lineTo(kx, y0 + hh - 1); ctx.lineTo(kx - 4, y0 + hh - 5); ctx.fill();
      }
    }
    for (const z of ["l", "r"]) if (this.hover === it.id + z) { ctx.fillStyle = ORANGE; ctx.fillRect(z === "l" ? x0 : x1 - 4, y0, 4, hh); }
    ctx.globalAlpha = 1;
  }
  // the clip's sound under its name, through the clip's own timing (in / speed / hold / loop / backwards)
  wave(ctx, it, m, x0, x1, y0, hh, muted) {
    const w = this.panel.cache.wave(m); if (!w?.peaks?.length) return;
    const seg = segOf(it, m.info?.duration || w.duration), band = m.kind === "audio" ? hh - 16 : 12, mid = y0 + hh - 3 - band / 2;
    const a = Math.max(HEADER_W, Math.floor(x0) + 1), b = Math.min(this.w, Math.ceil(x1) - 1);
    ctx.fillStyle = muted ? "rgba(0,0,0,.25)" : "rgba(0,0,0,.42)";
    for (let x = a; x < b; x++) {
      const ts = srcTime(seg, this.xt(x + 0.5)); if (ts === null) continue;
      const k = Math.min(w.peaks.length - 1, Math.max(0, Math.floor(ts * w.rate))), v = w.peaks[k] * (seg.fill === "hold" && ts >= seg.out - 1e-6 ? 0 : 1);
      const hgt = Math.max(1, v * band); ctx.fillRect(x, mid - hgt / 2, 1, hgt);
    }
  }
  round(ctx, x, y, w, h, r) { r = Math.min(r, w / 2, h / 2); ctx.beginPath(); ctx.roundRect ? ctx.roundRect(x, y, w, h, r) : ctx.rect(x, y, w, h); }
  toggle(ctx, C, r, label, on, col) {
    ctx.fillStyle = on ? col : "transparent"; ctx.strokeStyle = on ? col : C.line; ctx.lineWidth = 1;
    this.round(ctx, r.x + 0.5, r.y + 0.5, r.w - 1, r.h - 1, 9); ctx.fill(); ctx.stroke();
    ctx.fillStyle = on ? "#1d0f07" : C.muted; ctx.textAlign = "center"; ctx.fillText(label, r.x + r.w / 2, r.y + r.h / 2 + 0.5);
  }
  ruler(ctx, C, doc) {
    ctx.fillStyle = C.paper; ctx.fillRect(0, 0, this.w, RULER_H);
    const steps = [1 / doc.fps, 2 / doc.fps, 5 / doc.fps, 0.5, 1, 2, 5, 10, 15, 30, 60, 120, 300, 600];
    const major = steps.find(v => v >= 0.5 && v * this.pps >= 64) || 600, minor = steps.find(v => v * this.pps >= 7) || major;
    const t0 = Math.floor(this.xt(HEADER_W) / minor) * minor, t1 = this.xt(this.w);
    ctx.textAlign = "left";
    for (let i = 0, t = t0; t <= t1 && i < 4000; i++, t = t0 + i * minor) {
      const x = Math.round(this.tx(t)) + 0.5; if (x < HEADER_W) continue;
      const isMaj = Math.abs(t / major - Math.round(t / major)) < 1e-6;
      ctx.fillStyle = isMaj ? C.muted : C.line; ctx.fillRect(x, isMaj ? 11 : 17, 1, RULER_H - (isMaj ? 11 : 17));
      if (isMaj) { ctx.fillStyle = C.muted; ctx.fillText(t >= 60 ? `${Math.floor(t / 60)}:${String(Math.round(t % 60)).padStart(2, "0")}` : `${+t.toFixed(2)} s`, x + 3, 8); }
    }
    ctx.fillStyle = C.line; ctx.fillRect(0, RULER_H - 1, this.w, 1);
    ctx.fillStyle = C.paper2; ctx.fillRect(0, 0, HEADER_W, RULER_H);
    ctx.fillStyle = C.muted; ctx.fillText(`${doc.fps} fps`, 10, RULER_H / 2);
  }
  fitText(ctx, t, w) { if (ctx.measureText(t).width <= w) return t; while (t.length > 1 && ctx.measureText(t + "…").width > w) t = t.slice(0, -1); return t + "…"; }
}

// ------------------------------------------------------------------------------------------------ panel
const esc = s => String(s).replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
function h(tag, attrs = {}, ...kids) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k.startsWith("on")) e.addEventListener(k.slice(2), v); else if (k === "class") e.className = v; else if (v != null && v !== false) e.setAttribute(k, v === true ? "" : v);
  }
  for (const c of kids.flat()) if (c != null) e.append(c);
  return e;
}
const fmtT = (t, fps) => `${t.toFixed(2)} s · frame ${Math.round(t * fps)}`;

class Panel {
  constructor() {
    injectStyle();
    this.store = new Store(); this.cache = new MediaCache(() => { this.preview.dirty = true; this.tl.dirty = true; });
    this.preview = new Preview(this.store, this.cache); this.tl = new Timeline(this.store, this);
    this.tab = "directors"; this.outputs = null; this.lastProj = null; this.lastDirs = "";
    this.build();
    this.store.on(why => {
      this.tl.dirty = true; this.preview.dirty = true;
      if (why === "bind") { this.preview.pause(); for (const m of this.store.doc.media) this.cache.ensure(m)?.catch?.(() => {}); requestAnimationFrame(() => this.tl.fit()); }
      if (why !== "time") this.side(why);
      else if (!this.preview.playing) { const sel = this.store.selected(); if (sel.length === 1 && sel[0].it.anim) this.inspector(); }   // keyed values at the playhead
      this.bar();
    });
    this.frame = this.frame.bind(this); requestAnimationFrame(this.frame);
    setInterval(() => this.watch(), 400);
  }
  build() {
    const num = (key, tip, step = 1) => h("input", { type: "number", step, "data-tip": tip, title: tip, onchange: e => this.setProject(key, e.target.value) });
    this.f = { duration: num("duration", "length of the show, in seconds", 0.04), fps: num("fps", "frames per second (whole), the same in every director"),
      width: num("width", "width in pixels", 8), height: num("height", "height in pixels", 8) };
    this.playB = h("button", { class: "kkd-pill accent", title: "play / pause (P or space)", onclick: () => this.preview.toggle() }, "play");
    this.timeL = h("span", { class: "kkt-time" });
    this.projL = h("span", { class: "kkt-faint" });
    this.statusL = h("span", { class: "kkt-status" });
    this.snapB = h("button", { class: "kkd-pill", "aria-pressed": "true", title: "snap to edges and the playhead (hold Ctrl to move freely)", onclick: () => this.toggleSnap() }, "snap");
    this.loopB = h("button", { class: "kkd-pill", "aria-pressed": "true", title: "loop playback", onclick: () => this.toggleLoop() }, "loop");
    const bar = h("div", { class: "kkt-bar" },
      h("span", { class: "kkd-logo kkt-logo" }, h("b", {}, "kubakub"), " ", h("span", {}, "timeline"), h("i")),
      this.playB, this.loopB, this.timeL,
      h("label", { class: "kkt-field" }, "length", this.f.duration), h("label", { class: "kkt-field" }, "fps", this.f.fps),
      h("label", { class: "kkt-field" }, "size", this.f.width, "×", this.f.height), this.projL,
      h("span", { class: "kkd-grow" }), this.statusL,
      h("button", { class: "kkd-pill", title: "cut at the playhead (S or Ctrl K)", onclick: () => this.cut() }, "cut"),
      h("button", { class: "kkd-pill", title: "remove the selected (Del)", onclick: () => this.remove() }, "remove"),
      h("button", { class: "kkd-pill", title: "undo (Ctrl Z)", onclick: () => this.store.undo() }, "undo"),
      h("button", { class: "kkd-pill", title: "redo (Ctrl Y)", onclick: () => this.store.redo() }, "redo"),
      this.snapB,
      h("button", { class: "kkd-pill", title: "show everything (F)", onclick: () => this.tl.fit() }, "fit"),
      h("button", { class: "kkd-pill", title: "a new track on top", onclick: () => this.addTrack() }, "+ track"));
    this.list = h("div", { class: "kkt-list" });
    this.tabs = h("div", { class: "kkd-pills kkt-tabs" },
      ...[["directors", "directors"], ["renders", "renders"], ["files", "in this show"]].map(([k, l]) => h("button", { class: "kkd-pill", "data-tab": k, onclick: () => this.setTab(k) }, l)),
      h("button", { class: "kkd-pill ico", title: "import videos or pictures", onclick: () => this.file.click() }, "+"));
    this.file = h("input", { type: "file", multiple: true, accept: "video/*,image/*,audio/*", hidden: true, onchange: e => { this.importFiles([...e.target.files]); e.target.value = ""; } });
    this.insp = h("div", { class: "kkt-insp" });
    this.previewBox = h("div", { class: "kkt-preview kkd-stage" }, this.preview.cv);
    this.tlBox = h("div", { class: "kkt-tlbox" }, this.tl.cv);
    this.root = h("div", { class: "kkt-root", tabindex: "0" }, bar,
      this.body = h("div", { class: "kkt-body" }, this.previewBox, this.splitter("preview", 1), h("div", { class: "kkt-bin" }, this.tabs, this.list, this.file),
        this.splitter("list", 1), this.tlBox, this.splitter("insp", -1), this.insp));
    this.cols = { ...COL_DEF };
    try { Object.assign(this.cols, JSON.parse(localStorage.getItem(COLS) || "{}")); } catch { /* private window: the defaults */ }
    this.applyCols();
    this.root.addEventListener("keydown", e => this.key(e));
    this.root.addEventListener("pointerdown", e => { if (!e.target.closest("input,select,textarea,button")) this.root.focus({ preventScroll: true }); });
    this.dropZone(this.tlBox, true); this.dropZone(this.list, false);
    // ComfyUI opens a file dropped anywhere on the page as a workflow (a rendered mp4 carries one): keep ours here
    for (const ev of ["dragover", "drop"]) this.root.addEventListener(ev, e => { e.preventDefault(); e.stopPropagation(); });
    this.side("bind"); this.bar();
  }
  // column widths: drag a separator, double click = back to the default; remembered in this browser
  splitter(col, sign) {
    const lim = { preview: [120, 1200], list: [140, 520], insp: [160, 560] }[col];
    const sp = h("div", { class: "kkt-split", title: "drag to resize · double click = default" });
    sp.addEventListener("pointerdown", e => {
      if (e.button !== 0) return; e.preventDefault(); sp.setPointerCapture(e.pointerId); sp.classList.add("drag");
      const x0 = e.clientX, w0 = this.cols[col];
      const move = ev => { this.cols[col] = clamp(Math.round(w0 + sign * (ev.clientX - x0)), lim[0], lim[1]); this.applyCols(); };
      const up = () => { sp.classList.remove("drag"); sp.removeEventListener("pointermove", move); sp.removeEventListener("pointerup", up); sp.removeEventListener("pointercancel", up); this.saveCols(); };
      sp.addEventListener("pointermove", move); sp.addEventListener("pointerup", up); sp.addEventListener("pointercancel", up);
    });
    sp.addEventListener("dblclick", () => { this.cols[col] = COL_DEF[col]; this.applyCols(); this.saveCols(); });
    return sp;
  }
  applyCols() {
    for (const [k, v] of Object.entries(this.cols)) if (COL_DEF[k]) this.body.style.setProperty(`--c-${k}`, `${clamp(+v || COL_DEF[k], 100, 1200)}px`);
    this.preview.dirty = true;
  }
  saveCols() { try { localStorage.setItem(COLS, JSON.stringify(this.cols)); } catch { /* not kept: fine */ } }
  mount(el) { el.append(this.root); this.theme(); this.watch(); this.tl.resize(); this.tl.dirty = true; }
  theme() {                                          // follow ComfyUI's light / dark palette, not the system's
    const dark = document.documentElement.classList.contains("dark-theme") || document.body.classList.contains("dark-theme")
      || !document.documentElement.classList.contains("light-theme");
    this.root.dataset.theme = dark ? "dark" : "light";
  }
  frame(now) {
    requestAnimationFrame(this.frame);
    if (!this.root.isConnected) { if (this.preview.playing) this.preview.pause(); return; }
    this.preview.tick(now); if (this.preview.playing) { this.tl.follow(); this.bar(); }
    if (this.tl.dirty) this.tl.draw();
  }
  // the workflow around us: another workflow, the project settings node, the directors (their names, length, fps)
  watch() {
    const s = this.store; s.bind(rootGraph());
    const pn = projectNode();
    if (pn) {
      const p = projectValues(pn), k = JSON.stringify(p);
      if (k !== this.lastProj) {
        const first = this.lastProj === null; this.lastProj = k;
        const d = s.doc, fpsChanged = d.fps !== p.fps;
        if (fpsChanged || d.width !== p.width || d.height !== p.height) {
          Object.assign(d, { fps: p.fps, width: p.width, height: p.height });
          if (s.extra?.[KEY] === d) s.persist();     // a show not saved yet stays unsaved (opening the panel changes nothing)
          if (fpsChanged && !first) this.pushFps(p.fps);
          s.emit("project");
        }
      }
    } else this.lastProj = null;
    const dirs = nodesOf("KUBA_Director").map(n => `${n.id}|${n.title}|${widgetOf(n, "document")?.value?.length || 0}`).join(";");
    if (dirs !== this.lastDirs) { this.lastDirs = dirs; this.tl.dirty = true; this.preview.dirty = true; if (this.tab === "directors") this.side("dirs"); }
    this.theme();
  }
  pushFps(fps) {                                     // every director follows the project's fps
    let n = 0; const open = [];
    for (const node of nodesOf("KUBA_Director")) {
      if (directorInfo(node).fps === fps) continue;
      if (setDirectorFps(node, fps)) n++; else open.push(node.title || "director");
    }
    if (n) markChanged();
    this.status(n || open.length ? `${n ? `${n} director${n > 1 ? "s" : ""} now at ${fps} fps` : ""}${open.length ? `${n ? " · " : ""}close ${open.join(", ")} and match it there` : ""}` : "");
    this.tl.dirty = true;
  }
  status(t) { this.statusL.textContent = t || ""; clearTimeout(this.stT); if (t) this.stT = setTimeout(() => (this.statusL.textContent = ""), 6000); }
  bar() {
    const d = this.store.doc, pn = projectNode(), pv = pn && projectValues(pn);
    this.playB.textContent = this.preview.playing ? "pause" : "play";
    this.timeL.textContent = fmtT(this.store.time, d.fps);
    for (const [k, inp] of Object.entries(this.f)) if (document.activeElement !== inp) inp.value = d[k];
    const sizeLocked = !!pv?.matrix;
    this.f.width.disabled = this.f.height.disabled = sizeLocked;
    this.projL.textContent = pn ? (sizeLocked ? "size from the matrix · project settings" : "from project settings") : "";
  }
  seeked() { if (this.preview.playing) this.preview.resync(); this.preview.dirty = true; this.bar(); }
  setProject(key, v) {
    v = Number(v); const s = this.store, pn = projectNode();
    if (!Number.isFinite(v)) { this.bar(); return; }
    if (key === "duration") { s.edit(d => (d.duration = r4(clamp(s.q(v), 1 / d.fps, 3600)))); return; }
    if (key === "fps") {
      v = wholeFps(v); if (pn) setWidget(pn, "fps", v);
      s.edit(d => (d.fps = v)); this.pushFps(v); if (pn) this.lastProj = JSON.stringify(projectValues(pn)); return;
    }
    v = clamp(Math.round(v / 8) * 8, 64, 16384);
    if (pn) { if (PRESETS[widgetOf(pn, "preset")?.value]) { setWidget(pn, "width", s.doc.width); setWidget(pn, "height", s.doc.height); setWidget(pn, "preset", "custom"); } setWidget(pn, key, v); }
    s.edit(d => (d[key] = v)); if (pn) this.lastProj = JSON.stringify(projectValues(pn));
  }
  toggleSnap() { this.tl.snap = !this.tl.snap; this.snapB.setAttribute("aria-pressed", String(this.tl.snap)); }
  toggleLoop() { this.preview.loop = !this.preview.loop; this.loopB.setAttribute("aria-pressed", String(this.preview.loop)); }
  openDirector(nid) {
    const n = rootGraph()?.getNodeById?.(nid); if (!n) { this.status("that director is no longer in the workflow"); return; }
    const b = widgetOf(n, "open director"); if (b?.callback) b.callback(); else this.status("open it from its node");
  }

  // ---- editing
  cut() {
    const s = this.store, env = s.env(), t = s.time; let n = 0;
    s.edit(d => {
      for (const tr of d.tracks) {
        const add = [];
        for (const it of tr.items) {
          if (s.sel.size && !s.sel.has(it.id)) continue;
          const parts = splitItem(it, t, env); if (parts) { add.push(parts[1]); n++; if (s.sel.has(it.id)) s.sel.add(parts[1].id); }
        }
        tr.items.push(...add); tr.items.sort((a, b) => a.start - b.start);
      }
    });
    if (!n) this.status("nothing to cut under the playhead");
  }
  remove() {
    const s = this.store; if (!s.sel.size) return;
    s.edit(d => { for (const tr of d.tracks) tr.items = tr.items.filter(it => !s.sel.has(it.id)); }); s.sel.clear(); s.emit("sel");
  }
  addTrack() { this.store.edit(d => d.tracks.unshift(newTrack(`v${d.tracks.length + 1}`))); }
  removeTrack(i) {
    const d = this.store.doc, tr = d.tracks[i]; if (d.tracks.length <= 1) return;
    if (!tr.items.length || confirm(`remove track ${tr.name} and its ${tr.items.length} piece${tr.items.length > 1 ? "s" : ""}?`)) this.store.edit(x => x.tracks.splice(i, 1));
  }
  place(item, track, isNewMedia) {                   // put an item down (it overwrites what it lands on)
    const s = this.store, env0 = s.env();
    s.edit(d => {
      if (isNewMedia) d.media.push(isNewMedia);
      while (track >= d.tracks.length) d.tracks.push(newTrack(`v${d.tracks.length + 1}`));
      const env = { ...env0, mediaDur: id => (isNewMedia && id === isNewMedia.id ? isNewMedia.info?.duration || 0 : env0.mediaDur(id)) };
      const [a, b] = spanOf(item, env); clearRange(d.tracks[track], a, b, new Set(), env);
      d.tracks[track].items.push(item); d.tracks[track].items.sort((p, q) => p.start - q.start);
      if (b > d.duration) d.duration = r4(s.q(b));
    });
    s.sel.clear(); s.sel.add(item.id); s.emit("sel");
  }
  async mediaFor(source) {                           // the show's entry for a file, made (probed) when new
    const old = this.store.doc.media.find(m => m.source === source); if (old) return [old, null];
    const kind = kindOf(source) || "image";
    let info = null;
    if (kind === "video") { this.status(`reading ${source.split("/").pop()}…`); info = await this.cache.info(source); }
    else if (kind === "audio") { this.status(`reading ${source.split("/").pop()}…`); const w = await this.cache.waveOf(source); this.cache.waves.set(source, w); info = { duration: w.duration, audio: true }; }
    else info = await new Promise(res => { const im = new Image(); im.onload = () => res({ w: im.naturalWidth, h: im.naturalHeight }); im.onerror = () => res({}); im.src = sourceURL(source); });
    const m = { id: uid("m"), name: source.split("/").pop(), source, kind, info };
    return [m, m];
  }
  async addSource(source, at) {                      // at: {track, t} (onto the timeline) or null (the show's files only)
    const [m, fresh] = await this.mediaFor(source);
    if (!at) { if (fresh) this.store.edit(d => d.media.push(m)); this.status(""); return 0; }
    const item = { id: uid("i"), type: "clip", media: m.id, start: at.t, in: 0, out: -1, speed: 1, length: m.kind === "image" ? 2 : -1, fill: "hold", reverse: false, muted: false, props: defProps() };
    this.place(item, at.track, fresh); this.status("");
    return at.t + (m.kind === "image" ? 2 : Math.max(1 / this.store.doc.fps, m.info?.duration || 2));
  }
  async importFiles(files, at) {
    files = files.filter(f => kindOf(f.name));
    if (!files.length) { this.status("drop videos, pictures or sounds"); return; }
    let t = at?.t;
    for (const f of files) {
      try {
        this.status(`importing ${f.name}…`);
        const fd = new FormData(); fd.append("file", f, f.name);
        const r = await api.fetchApi("/kubakub/master/upload", { method: "POST", body: fd }), j = await r.json();
        if (!r.ok) throw new Error(j.error || r.statusText);
        t = await this.addSource("file:" + j.name, at ? { track: at.track, t } : null);
      } catch (e) { this.status(`${f.name}: ${e.message}`); return; }
    }
    this.status("");
  }
  dropZone(zone, onTimeline) {
    zone.addEventListener("dragover", e => {
      e.preventDefault(); e.stopPropagation(); e.dataTransfer.dropEffect = "copy"; zone.classList.add("kkt-drop");
      if (onTimeline) { this.tl.ghost = this.tl.locate(e.clientX, e.clientY); this.tl.dirty = true; }
    });
    zone.addEventListener("dragleave", e => { if (zone.contains(e.relatedTarget)) return; zone.classList.remove("kkt-drop"); this.tl.ghost = null; this.tl.dirty = true; });
    zone.addEventListener("drop", async e => {
      e.preventDefault(); e.stopPropagation(); zone.classList.remove("kkt-drop"); this.tl.ghost = null; this.tl.dirty = true;
      const at = onTimeline ? this.tl.locate(e.clientX, e.clientY) : null, dt = e.dataTransfer;
      try {
        if (dt.files?.length) { await this.importFiles([...dt.files], at); return; }
        const own = dt.getData(DRAG_MIME);
        if (own) {
          const p = JSON.parse(own);
          if (p.director != null && at) this.place({ id: uid("i"), type: "director", node: p.director, start: at.t, in: 0, length: -1, props: defProps() }, at.track, null);
          else if (p.source) await this.addSource(p.source, at);
          return;
        }
        const url = (dt.getData("text/uri-list") || dt.getData("text/plain") || "").split("\n")[0].trim();
        const src = this.sourceFromView(url); if (src) await this.addSource(src, at); else this.status("drop a video or picture, a director or a render from the list");
      } catch (err) { this.status(err.message); }
    });
  }
  sourceFromView(url) {                              // a preview dragged out of a node: /view?filename=&subfolder=&type=
    try {
      const u = new URL(url, location.href); if (!u.pathname.endsWith("/view")) return null;
      const f = u.searchParams.get("filename"), sub = u.searchParams.get("subfolder") || "", type = u.searchParams.get("type");
      if (!f || !kindOf(f) || (type !== "output" && type !== "input")) return null;
      return `${type === "output" ? "output:" : "file:"}${sub ? sub + "/" : ""}${f}`;
    } catch { return null; }
  }

  // ---- side: list (directors / renders / the show's files) and the inspector
  setTab(t) { this.tab = t; if (t === "renders") this.outputs = null; this.side("tab"); }
  async loadOutputs() {
    try { const r = await api.fetchApi("/kubakub/master/outputs"); this.outputs = (await r.json()).files || []; }
    catch (e) { this.outputs = []; this.status(e.message); }
    if (this.tab === "renders") this.side("tab");
  }
  rendersChanged() {                                // a run finished: new renders, new director pictures
    this.preview.dirty = true; this.tl.dirty = true;
    if (this.tab === "renders" && this.root.isConnected) this.loadOutputs(); else this.outputs = null;
    if (this.tab === "directors") this.listView();
  }
  side(why) { if (why !== "drag") this.listView(); this.inspector(); }
  row(name, sub, payload, thumb, extra) {
    const r = h("div", { class: "kkt-row", draggable: "true", title: `${name}\ndrag it onto the timeline` },
      thumb || h("span", { class: "kkt-thumb" }), h("div", { class: "kkt-rowtext" }, h("div", { class: "kkt-name" }, name), h("div", { class: "kkt-faint" }, sub)), extra || null);
    r.addEventListener("dragstart", e => { e.dataTransfer.setData(DRAG_MIME, JSON.stringify(payload)); e.dataTransfer.effectAllowed = "copy"; });
    return r;
  }
  listView() {
    for (const b of this.tabs.querySelectorAll("[data-tab]")) b.setAttribute("aria-pressed", String(b.dataset.tab === this.tab));
    const L = this.list, s = this.store, fps = s.doc.fps; L.replaceChildren();
    const empty = t => L.append(h("div", { class: "kkt-empty" }, t));
    if (this.tab === "directors") {
      const dirs = nodesOf("KUBA_Director");
      if (!dirs.length) empty("no kubakub director in this workflow");
      for (const n of dirs) {
        const di = directorInfo(n), off = di.fps !== fps, still = nodeStill(n);
        const extra = h("span", { class: "kkd-pills" },
          off ? h("button", { class: "kkd-pill", title: `this director runs at ${di.fps} fps, the show at ${fps}`, onclick: () => this.pushFps(fps) }, `${di.fps} → ${fps} fps`) : null,
          h("button", { class: "kkd-pill", title: "open the director", onclick: () => this.openDirector(n.id) }, "open"));
        L.append(this.row(n.title || "kubakub director", `${di.duration.toFixed(2)} s · ${di.fps} fps`, { director: n.id }, still ? h("img", { class: "kkt-thumb", src: still }) : null, extra));
      }
    } else if (this.tab === "renders") {
      if (this.outputs === null) { empty("…"); this.loadOutputs(); return; }
      if (!this.outputs.length) empty("no videos or pictures in the output folder yet");
      for (const f of this.outputs.slice(0, 200)) {
        const source = "output:" + f.name, when = new Date(f.mtime * 1000).toLocaleString([], { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" });
        const thumb = this.thumb(f.kind, source);
        L.append(this.row(f.name.split("/").pop(), `${f.name.includes("/") ? f.name.slice(0, f.name.lastIndexOf("/")) + " · " : ""}${when}`, { source }, thumb));
      }
    } else {
      if (!s.doc.media.length) empty("drop videos or pictures here or onto the timeline");
      const used = new Set([...s.items()].map(x => x.it.media));
      for (const m of s.doc.media) {
        const i = m.info || {}, sub = m.kind === "video" ? `${i.w}×${i.h} · ${+(+i.fps || 0).toFixed(2)} fps · ${(+i.duration || 0).toFixed(2)} s${i.audio ? " · sound" : ""}`
          : m.kind === "audio" ? `sound · ${(+i.duration || 0).toFixed(2)} s` : `${i.w}×${i.h} picture`;
        const del = used.has(m.id) ? null : h("button", { class: "kkd-pill ico", title: "remove from the show", onclick: () => this.store.edit(d => (d.media = d.media.filter(x => x.id !== m.id))) }, "×");
        const thumb = this.thumb(m.kind, m.source);
        L.append(this.row(m.name, sub, { source: m.source }, thumb, del));
      }
    }
  }
  thumb(kind, source) {
    if (kind === "video") return h("video", { class: "kkt-thumb", src: sourceURL(source) + "#t=0.1", preload: "metadata", muted: true });
    if (kind === "audio") return h("span", { class: "kkt-thumb kkt-sound" }, "♪");
    return h("img", { class: "kkt-thumb", src: sourceURL(source), loading: "lazy" });
  }
  inspector() {
    const box = this.insp, s = this.store, sel = s.selected();
    if (sel.length !== 1) { box.replaceChildren(h("div", { class: "kkt-empty" }, sel.length ? `${sel.length} selected` : "select a piece on the timeline")); delete box.dataset.id; return; }
    const { track, it } = sel[0], env = s.env(), [a, b] = spanOf(it, env);
    if (box.dataset.id === it.id && box.contains(document.activeElement) && document.activeElement.tagName === "INPUT") return;
    box.dataset.id = it.id;
    const cur = () => s.find(it.id)?.it;
    const num = (label, v, set, o = {}) => {
      const inp = h("input", { type: "number", step: o.step ?? 0.04, min: o.min, max: o.max, value: o.empty && v < 0 ? "" : +(+v).toFixed(4), placeholder: o.empty || null, title: o.tip || null });
      inp.addEventListener("change", () => { const x = inp.value === "" && o.empty ? -1 : Number(inp.value); if (Number.isFinite(x)) s.edit(() => { const c = cur(); if (c) set(c, x); }); });
      return h("label", { class: "kkt-row2" }, h("span", {}, label), inp);
    };
    const fps = s.doc.fps, now = propsAt(it, s.time);
    const keyBtn = key => h("button", { class: "kkd-pill ico kkt-key", "aria-pressed": String(!!keyAt(it.anim?.[key], s.time - it.start, fps)),
      title: it.anim?.[key] ? "a key here / remove the key here" : "animate: a key here", onclick: () => s.edit(() => { const c = cur(); if (c) toggleKey(c, key, s.time, fps); }) }, "◆");
    const slider = (label, key, lo, hi, st) => {
      const show = v => String(+(+v).toFixed(2)), out = h("span", { class: "kkt-faint" }, show(now[key]));
      const inp = h("input", { type: "range", min: lo, max: hi, step: st, value: now[key] });
      inp.addEventListener("pointerdown", () => s.begin());
      inp.addEventListener("input", () => { const c = cur(); if (!c) return; setProp(c, key, Number(inp.value), s.time, fps); out.textContent = show(inp.value); s.moved(); });
      inp.addEventListener("change", () => s.end());
      return h("label", { class: "kkt-row2" }, h("span", {}, label), inp, out, keyBtn(key));
    };
    const keyed = (row, key) => { row.append(keyBtn(key)); return row; };
    const pills = (label, opts, val, set) => h("div", { class: "kkt-row2" }, h("span", {}, label), h("span", { class: "kkd-pills" },
      ...opts.map(o => h("button", { class: "kkd-pill", "aria-pressed": String(val === o), onclick: () => s.edit(() => { const c = cur(); if (c) set(c, o); }) }, String(o)))));
    const isDir = it.type === "director", n = isDir ? rootGraph()?.getNodeById?.(it.node) : null, m = isDir ? null : s.media(it.media);
    const rows = [h("h2", { class: "kkt-h2" }, isDir ? (n?.title || "kubakub director") : (m?.name || "missing file")),
      h("div", { class: "kkt-faint" }, `${track.name} · ${a.toFixed(2)}–${b.toFixed(2)} s`),
      num("at", it.start, (c, x) => (c.start = r4(Math.max(0, s.q(x)))), { tip: "where it starts on the timeline (s)" }),
      num("in", it.in, (c, x) => (c.in = r4(Math.max(0, x))), { tip: isDir ? "seconds into the director's timeline" : "seconds into the file" })];
    if (isDir) {
      rows.push(num("length", it.length, (c, x) => (c.length = x > 0 ? r4(s.q(x)) : -1), { empty: "to its end", tip: "empty = to the end of the director's timeline" }),
        h("button", { class: "kkd-pill accent", onclick: () => this.openDirector(it.node) }, "open director"));
    } else {
      rows.push(num("out", it.out, (c, x) => (c.out = x > 0 ? r4(x) : -1), { empty: "end", tip: "seconds into the file where it stops (empty = the end)" }),
        num("speed", it.speed, (c, x) => { c.speed = r4(clamp(x, 0.05, 20)); }, { step: 0.05, min: 0.05, max: 20 }),
        num("length", it.length, (c, x) => (c.length = x > 0 ? r4(s.q(x)) : -1), { empty: "as in → out plays", tip: "time on the timeline; longer than the piece = hold or loop" }),
        pills("then", ["hold", "loop"], it.fill, (c, o) => (c.fill = o)),
        m?.kind === "audio" ? null : pills("backwards", ["off", "on"], it.reverse ? "on" : "off", (c, o) => (c.reverse = o === "on")),
        m?.kind === "image" ? null : pills("sound", ["on", "off"], it.muted ? "off" : "on", (c, o) => (c.muted = o === "off")));
    }
    if (m?.kind === "audio") { box.replaceChildren(...rows.filter(Boolean)); return; }
    const kts = keyTimes(it).map(k => it.start + k).filter(k => k >= a - 1e-9 && k <= b + 1e-9);   // a cut keeps the keys past its end
    const go = t => { s.setTime(t); this.seeked(); };
    const prev = kts.filter(k => k < s.time - 0.5 / fps).pop(), next = kts.find(k => k > s.time + 0.5 / fps);
    const ease = Object.values(it.anim || {}).map(ks => keyAt(ks, s.time - it.start, fps)).find(Boolean)?.e;
    rows.push(h("h2", { class: "kkt-h2" }, "picture"), slider("opacity", "opacity", 0, 1, 0.01),
      keyed(num("x", now.x, (c, x) => setProp(c, "x", x, s.time, fps), { step: 1, tip: "pixels of the show size" }), "x"),
      keyed(num("y", now.y, (c, x) => setProp(c, "y", x, s.time, fps), { step: 1 }), "y"),
      keyed(num("size", now.scale, (c, x) => setProp(c, "scale", clamp(x, 0.01, 20), s.time, fps), { step: 0.05 }), "scale"),
      pills("fit", ["fit", "fill", "stretch"], it.props.fit, (c, o) => (c.props.fit = o)));
    if (kts.length) rows.push(h("div", { class: "kkt-row2" }, h("span", {}, "keys"), h("span", { class: "kkd-pills" },
        h("button", { class: "kkd-pill", disabled: prev === undefined, title: "the key before", onclick: () => go(prev) }, "‹"),
        h("button", { class: "kkd-pill", disabled: next === undefined, title: "the key after", onclick: () => go(next) }, "›"))),
      ease ? pills("ease", EASES, ease, (c, o) => easeAt(c, s.time, fps, o)) : null);
    box.replaceChildren(...rows.filter(Boolean));
  }

  // ---- keys (only while the panel has focus; ComfyUI's save / queue shortcuts pass)
  key(e) {
    const ctrl = e.ctrlKey || e.metaKey, k = e.key, s = this.store, step = 1 / s.doc.fps;
    if (ctrl && /^(s|enter)$/i.test(k)) return;
    if (e.target.closest("input,select,textarea")) { e.stopPropagation(); return; }
    let done = true;
    if (k === " " || k === "p" || k === "P") this.preview.toggle();
    else if (ctrl && /^z$/i.test(k)) e.shiftKey ? s.redo() : s.undo();
    else if (ctrl && /^y$/i.test(k)) s.redo();
    else if (ctrl && /^k$/i.test(k)) this.cut();
    else if (ctrl && /^a$/i.test(k)) { for (const { it } of s.items()) s.sel.add(it.id); s.emit("sel"); }
    else if (k === "Escape") { if (this.tl.drag) this.tl.cancel(); else { s.sel.clear(); s.emit("sel"); } }
    else if (k === "Delete" || k === "Backspace") this.remove();
    else if (!ctrl && /^s$/i.test(k)) this.cut();
    else if (!ctrl && /^f$/i.test(k)) this.tl.fit();
    else if (k === "," || k === "ArrowLeft") { s.setTime(s.time - (e.shiftKey ? 10 : 1) * step); this.seeked(); }
    else if (k === "." || k === "ArrowRight") { s.setTime(s.time + (e.shiftKey ? 10 : 1) * step); this.seeked(); }
    else if (k === "Home") { s.setTime(0); this.seeked(); }
    else if (k === "End") { s.setTime(s.doc.duration); this.seeked(); }
    else done = false;
    if (done) { e.preventDefault(); e.stopPropagation(); }
  }
}

// ------------------------------------------------------------------------------------------------ style
let styled = false;
function injectStyle() {
  if (styled) return; styled = true;
  const st = document.createElement("style");
  st.textContent = `
@font-face{font-family:"Hanken Grotesk KKD";src:url("${HERE}fonts/HankenGrotesk-latin-var.woff2") format("woff2");font-weight:200 800;font-display:swap}
.kkt-root{--paper:#141414;--paper2:#1c1c1d;--ink:rgba(255,255,255,.88);--muted:rgba(255,255,255,.55);--faint:rgba(255,255,255,.25);--line:rgba(255,255,255,.12);
  --orange:${ORANGE};--orange-ink:#1d0f07;--lav:${LAV};--lav-soft:rgba(161,135,183,.22);--stage:#2c2c2e;--stage-edge:#222224;color-scheme:dark;
  position:absolute;inset:0;display:flex;flex-direction:column;gap:8px;padding:8px 12px;box-sizing:border-box;background:var(--paper);color:var(--ink);
  font:400 13px/1.35 "Hanken Grotesk KKD","Segoe UI",system-ui,sans-serif;outline:none}
.kkt-root[data-theme=light]{--paper:#ffffff;--paper2:#f6f4f3;--ink:rgba(0,0,0,.85);--muted:rgba(0,0,0,.55);--faint:rgba(0,0,0,.25);--line:rgba(0,0,0,.12);
  --lav-soft:rgba(161,135,183,.18);--stage:#3b3b3e;--stage-edge:#303033;color-scheme:light}
.kkt-root *{box-sizing:border-box}
.kkt-root button,.kkt-root input,.kkt-root select{font:inherit;color:inherit}
.kkt-root [hidden]{display:none!important}
.kkt-bar{display:flex;flex-wrap:wrap;gap:6px;align-items:center;text-transform:lowercase}
.kkt-root .kkd-logo{font-size:17px;letter-spacing:-.01em;margin-right:6px;white-space:nowrap;text-transform:lowercase}
.kkt-root .kkd-logo b{font-weight:800}.kkt-root .kkd-logo span{font-weight:200;color:var(--lav)}
.kkt-root .kkd-logo i{display:inline-block;width:7px;height:7px;border-radius:50%;background:var(--orange);margin-left:3px;vertical-align:2px}
.kkt-root .kkd-grow{flex:1}
.kkt-root .kkd-pill{border:1px solid var(--line);background:transparent;border-radius:10em;padding:3px 11px;cursor:pointer;text-transform:lowercase;white-space:nowrap}
.kkt-root .kkd-pill:hover{border-color:var(--faint)}
.kkt-root .kkd-pill[aria-pressed="true"]{background:var(--lav-soft);border-color:var(--lav)}
.kkt-root .kkd-pill.accent{background:var(--orange);border-color:var(--orange);color:var(--orange-ink);font-weight:600;min-width:62px}
.kkt-root .kkd-pill.ico{padding:3px 9px}
.kkt-root .kkd-pills{display:inline-flex;gap:4px;flex-wrap:wrap}
.kkt-root :focus-visible{outline:2px solid var(--orange);outline-offset:2px}
.kkt-root input[type=number]{width:64px;background:var(--paper2);border:1px solid var(--line);border-radius:8px;padding:2px 6px}
.kkt-root input[type=number]:disabled{opacity:.45}
.kkt-root input[type=range]{accent-color:var(--orange);min-width:0;flex:1}
.kkt-field{display:inline-flex;gap:5px;align-items:center;color:var(--muted)}
.kkt-time{font-variant-numeric:tabular-nums;color:var(--ink);min-width:150px;padding:0 4px}
.kkt-faint{color:var(--faint);font-size:11px}
.kkt-status{color:var(--lav);font-size:12px;max-width:40ch;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.kkt-body{flex:1;min-height:0;display:grid;grid-template-columns:var(--c-preview,240px) 10px var(--c-list,210px) 10px minmax(120px,1fr) 10px var(--c-insp,220px)}
.kkt-split{position:relative;cursor:col-resize;touch-action:none}
.kkt-split::after{content:"";position:absolute;top:0;bottom:0;left:4px;width:2px;border-radius:2px;background:transparent;transition:background .12s}
.kkt-split:hover::after,.kkt-split.drag::after{background:var(--lav)}
.kkt-root .kkd-stage{background:var(--stage);border:1px solid var(--stage-edge);border-radius:6px}
.kkt-preview{display:grid;place-items:center;overflow:hidden;min-height:0}
.kkt-preview canvas{max-width:100%;max-height:100%;display:block}
.kkt-bin{display:flex;flex-direction:column;gap:6px;min-height:0}
.kkt-list{flex:1;min-height:0;overflow:auto;display:flex;flex-direction:column;gap:2px;border-radius:8px;padding:2px}
.kkt-list.kkt-drop,.kkt-tlbox.kkt-drop{outline:2px dashed var(--orange);outline-offset:-2px}
.kkt-row{display:grid;grid-template-columns:44px minmax(0,1fr) auto;gap:7px;align-items:center;padding:4px 6px;border-radius:8px;border:1px solid transparent;cursor:grab;user-select:none}
.kkt-row:hover{border-color:var(--line);background:var(--paper2)}
.kkt-thumb{width:44px;height:26px;object-fit:cover;border-radius:4px;background:var(--stage);display:block}
.kkt-sound{display:grid;place-items:center;color:#fff;background:#5f7f9e;font-size:14px}
.kkt-rowtext{min-width:0}.kkt-name{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.kkt-empty{color:var(--faint);padding:8px 4px;font-size:12px;text-transform:lowercase}
.kkt-tlbox{min-width:0;min-height:0;position:relative;border:1px solid var(--line);border-radius:6px;overflow:hidden}
.kkt-tl{position:absolute;inset:0;width:100%;height:100%;display:block;touch-action:none}
.kkt-insp{min-height:0;overflow:auto;display:flex;flex-direction:column;gap:5px;padding-right:2px;text-transform:lowercase}
.kkt-h2{font-size:12px;font-weight:600;letter-spacing:.06em;color:var(--muted);margin:6px 0 2px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;text-transform:none}
.kkt-row2{display:flex;gap:6px;align-items:center;min-height:24px}.kkt-row2>span:first-child{width:62px;flex:none;color:var(--muted)}
.kkt-root .kkt-key{margin-left:auto;flex:none;font-size:10px;padding:2px 6px}.kkt-root .kkt-key[aria-pressed=true]{background:var(--orange);color:var(--orange-ink);border-color:var(--orange)}
.kkt-root .kkt-tabs .kkd-pill{padding:3px 9px}`;
  document.head.append(st);
}

// ------------------------------------------------------------------------------------------------ register
let panel = null;
app.registerExtension({
  name: "kubakub.master",
  bottomPanelTabs: [{ id: "kubakub-timeline", title: "kubakub timeline", type: "custom",
    render: el => { panel = panel || new Panel(); el.style.position = "relative"; el.style.height = "100%"; panel.mount(el); } }],
  setup() {
    api.addEventListener("executed", () => panel?.rendersChanged());
  },
  // the render node's hidden 'timeline' input gets the show when queued (so an edit re-renders); not saved in the node
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== "KUBA_MasterRender") return;
    const created = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = created?.apply(this, arguments);
      const w = this.widgets?.find(w => w.name === "timeline");
      if (w) {
        w.hidden = true; w.computeSize = () => [0, -4];
        if (w.element) w.element.style.display = "none";
        if (w.inputEl) w.inputEl.style.display = "none";
        w.serializeValue = () => { const d = rootGraph()?.extra?.[KEY]; return d ? JSON.stringify(d) : ""; };
      }
      return r;
    };
  },
});
