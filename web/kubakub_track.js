// kubakub tracker / kubakub track draw on the node.
// tracker: the video with what was found (scrub, play); finder pills and sliders; the colour picked with an
//   eyedropper on the video; a motif clicked on a frame (Shift = not this); 'try this frame' runs the finders on the
//   frame shown with the settings as they are now (POST /kubakub/track/probe). Settings live in the hidden
//   'settings' widget (JSON, = kubakub/tracking.py settings()).
// track draw: the look played live from the tracker's frames and tracks, before anything is queued; preset pills on
//   the node, everything else in the editor window (big preview + marks, lines, inside, stamp trail, words).
//   The look lives in the hidden 'look' widget (JSON, = kubakub/trackdraw.py look()).
// What decides what is drawn (states, labels, which points connect, where stamps land) is the same as in
// kubakub/trackdraw.py and checked against it by tests/test_tracking.py.
// Plain ES module, no build step, no network. Look: kubakub.art (Hanken Grotesk, #f18a58 actions, #a187b7 selection).
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const HERE = new URL(".", import.meta.url).href;
export const ORANGE = "#f18a58", LAV = "#a187b7";
export const STATE_WORDS = ["INPUT", "PROCESS", "STABLE", "ARCHIVED"];
// the director's blend modes (kubakub/director/blend.py) as the browser names them
export const OPS = { normal: "source-over", multiply: "multiply", screen: "screen", overlay: "overlay", darken: "darken",
  lighten: "lighten", "color dodge": "color-dodge", "color burn": "color-burn", "hard light": "hard-light",
  "soft light": "soft-light", difference: "difference", exclusion: "exclusion", hue: "hue", saturation: "saturation",
  color: "color", luminosity: "luminosity" };
const blendOp = e => OPS[e?.blend] || "source-over";

// ------------------------------------------------------------------------------------------ shared with trackdraw.py
export const centre = b => [b[0] + b[2] / 2, b[1] + b[3] / 2];
const dist = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1]);
export function stateOf(tr, t, st, unit) {
  const n = tr.box.length, k = t - tr.t0, lost = parseInt(st.lost ?? 24);
  if (k < 0 || k >= n + lost) return null;
  if (k >= n) return [3, tr.box[n - 1]];
  const b = tr.box[k];
  if (k < parseInt(st.new ?? 8)) return [0, b];
  const c = centre(b), q = centre(tr.box[k - 1]), speed = Math.hypot(c[0] - q[0], c[1] - q[1]) / unit;
  return [speed > parseFloat(st.moving ?? 1.2) ? 1 : 2, b];
}
export function speedAt(tr, t, unit) {
  const k = t - tr.t0; if (k < 1 || k >= tr.box.length) return 0;
  const c = centre(tr.box[k]), q = centre(tr.box[k - 1]); return Math.hypot(c[0] - q[0], c[1] - q[1]) / unit;
}
export function visible(data, t, lk, kinds) {
  const unit = data.H / 1080, out = [];
  for (const tr of data.tracks) {
    if (kinds?.length && !kinds.includes(tr.kind)) continue;
    const s = stateOf(tr, t, lk.states, unit); if (s) out.push([tr, s[0], s[1]]);
  }
  return out.sort((a, b) => a[0].id - b[0].id);
}
export const pad = (v, n) => String(v).padStart(n, "0");
const pmod = (v, m) => ((v % m) + m) % m;           // Python's % (never negative), so labels match the render
export function spanWeight(e, t, data) {          // = trackdraw.span_weight: 0 outside the element's time, 1 in, ramps over its fade
  const sp = e?.span; if (!sp || sp.length !== 2) return 1;
  const last = Math.max(1, data.n - 1), a = sp[0] * last, b = sp[1] * last;
  if (t < a - 1e-9 || t > b + 1e-9) return 0;
  const f = (parseFloat(e.span_fade || 0) || 0) * (data.fps || 25); if (f <= 0) return 1;
  return Math.max(0, Math.min(1, (t - a) / f, (b - t) / f));
}
const spanStart = (e, data) => e?.span?.length === 2 ? e.span[0] * Math.max(1, data.n - 1) : -1;
export function soundAt(data, t) {                 // = trackdraw.sound_at: what the tracker heard at frame t, or null without a sound
  const c = data.curves; if (!c?.beat?.length) return null; const i = Math.min(Math.max(t, 0), c.beat.length - 1);
  return { level: c.level?.[i] || 0, beat: c.beat[i] || 0 };
}
export function zAt(tr, t) { const z = tr.z; if (!z?.length) return null; return z[Math.min(Math.max(t - tr.t0, 0), z.length - 1)]; }
export function labelText(text, tr, state, b, t, data, lk) {
  const W = data.W, H = data.H, fps = data.fps || 25, unit = H / 1080;
  const words = [...(lk.states.words?.length ? lk.states.words : STATE_WORDS), ...STATE_WORDS], [cx, cy] = centre(b);
  const rep = { "{id}": String(tr.id), "{id_hex}": (tr.id * 2654435761 % 65536).toString(16).toUpperCase().padStart(4, "0"),
    "{id_long}": pad(1006000 + tr.id * 7919 % 1000, 7), "{kind}": tr.kind, "{label}": tr.label || "", "{state}": String(words[state]),
    "{x}": (cx / W).toFixed(3), "{y}": (cy / H).toFixed(3), "{speed}": speedAt(tr, t, unit).toFixed(1),
    "{size}": (b[2] * b[3] / (W * H) * 100).toFixed(2), "{frame}": String(t), "{t}": (t / fps).toFixed(2),
    "{data}": pad(pmod(Math.trunc(cx), 10000), 4) + "." + pad(pmod(Math.trunc(cy * 10000 / H), 10000), 4), "{z}": zAt(tr, t) == null ? "" : zAt(tr, t).toFixed(3) };
  let out = String(text); for (const [k, v] of Object.entries(rep)) out = out.split(k).join(v); return out;
}
export function hullIdx(P) {
  const idx = P.map((_, i) => i).sort((a, b) => P[a][0] - P[b][0] || P[a][1] - P[b][1]);
  if (idx.length < 3) return idx;
  const cross = (o, a, b) => (P[a][0] - P[o][0]) * (P[b][1] - P[o][1]) - (P[a][1] - P[o][1]) * (P[b][0] - P[o][0]);
  const lower = [], upper = [];
  for (const i of idx) { while (lower.length >= 2 && cross(lower[lower.length - 2], lower[lower.length - 1], i) <= 0) lower.pop(); lower.push(i); }
  for (const i of [...idx].reverse()) { while (upper.length >= 2 && cross(upper[upper.length - 2], upper[upper.length - 1], i) <= 0) upper.pop(); upper.push(i); }
  return [...lower.slice(0, -1), ...upper.slice(0, -1)];
}
export function delaunayEdges(P) {                  // Bowyer-Watson, the same steps as trackdraw.delaunay_edges
  const n = P.length; if (n < 2) return []; if (n === 2) return [[0, 1]];
  const xs = P.map(p => p[0]), ys = P.map(p => p[1]), mx = Math.min(...xs), my = Math.min(...ys);
  const dm = Math.max(Math.max(...xs) - mx, Math.max(...ys) - my, 1) * 20;
  const pts = [...P, [mx - dm, my - dm], [mx + dm * 2, my - dm], [mx - dm, my + dm * 2]];
  const circ = tri => { const [ax, ay] = pts[tri[0]], [bx, by] = pts[tri[1]], [cx, cy] = pts[tri[2]];
    const d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by)); if (Math.abs(d) < 1e-12) return [0, 0, Infinity];
    const ux = ((ax * ax + ay * ay) * (by - cy) + (bx * bx + by * by) * (cy - ay) + (cx * cx + cy * cy) * (ay - by)) / d;
    const uy = ((ax * ax + ay * ay) * (cx - bx) + (bx * bx + by * by) * (ax - cx) + (cx * cx + cy * cy) * (bx - ax)) / d;
    return [ux, uy, (ax - ux) ** 2 + (ay - uy) ** 2]; };
  let tris = [[n, n + 1, n + 2]]; tris[0].c = circ(tris[0]);
  for (let i = 0; i < n; i++) {
    const [px, py] = pts[i], bad = tris.filter(tr => (px - tr.c[0]) ** 2 + (py - tr.c[1]) ** 2 < tr.c[2] - 1e-9), edges = new Map();
    for (const tr of bad) for (const [a, b] of [[tr[0], tr[1]], [tr[1], tr[2]], [tr[2], tr[0]]]) { const k = Math.min(a, b) + "," + Math.max(a, b); edges.set(k, (edges.get(k) || 0) + 1); }
    tris = tris.filter(tr => !bad.includes(tr));
    for (const [k, c] of edges) if (c === 1) { const [a, b] = k.split(",").map(Number), tri = [a, b, i]; tri.c = circ(tri); tris.push(tri); }
  }
  const out = new Set();
  for (const tr of tris) for (const [a, b] of [[tr[0], tr[1]], [tr[1], tr[2]], [tr[2], tr[0]]]) if (a < n && b < n) out.add(Math.min(a, b) * 100000 + Math.max(a, b));
  return [...out].sort((a, b) => a - b).map(v => [Math.floor(v / 100000), v % 100000]);
}
export function mstEdges(P) {
  const n = P.length; if (n < 2) return [];
  const inside = Array(n).fill(false), best = Array(n).fill(Infinity), from = Array(n).fill(-1), out = []; best[0] = 0;
  for (let it = 0; it < n; it++) {
    let u = -1; for (let i = 0; i < n; i++) if (!inside[i] && (u < 0 || best[i] < best[u])) u = i;
    inside[u] = true; if (from[u] >= 0) out.push([Math.min(u, from[u]), Math.max(u, from[u])]);
    for (let v = 0; v < n; v++) if (!inside[v]) { const d = dist(P[u], P[v]); if (d < best[v] - 1e-9) { best[v] = d; from[v] = u; } }
  }
  return out.sort((a, b) => a[0] - b[0] || a[1] - b[1]);
}
export function connectEdges(c, items, data, lk, t) {
  const W = data.W, H = data.H, mode = c.mode || "within", kinds = c.kinds || [];
  const its = items.filter(e => !kinds.length || kinds.includes(e[0].kind)), P = its.map(e => centre(e[2]));
  let groups = [P.map((_, i) => i)];
  if (c.same_kind) { const gs = {}; its.forEach((e, i) => (gs[e[0].kind] ||= []).push(i)); groups = Object.keys(gs).sort().map(k => gs[k]); }
  const segs = [], area = e => e[2][2] * e[2][3];
  for (const g of groups) {
    const Q = g.map(i => P[i]); let pairs = [];
    if (mode === "nearest") {
      const k = Math.max(1, parseInt(c.k ?? 1)), s = new Set();
      for (let a = 0; a < Q.length; a++) Q.map((_, b) => b).filter(b => b !== a).sort((x, y) => dist(Q[a], Q[x]) - dist(Q[a], Q[y]) || x - y).slice(0, k)
        .forEach(b => s.add(Math.min(a, b) * 100000 + Math.max(a, b)));
      pairs = [...s].sort((a, b) => a - b).map(v => [Math.floor(v / 100000), v % 100000]);
    } else if (mode === "within") { const r = parseFloat(c.distance ?? 0.18) * W;
      for (let a = 0; a < Q.length; a++) for (let b = a + 1; b < Q.length; b++) if (dist(Q[a], Q[b]) < r) pairs.push([a, b]);
    } else if (mode === "network") pairs = delaunayEdges(Q);
    else if (mode === "tree") pairs = mstEdges(Q);
    else if (mode === "hull") { const h = hullIdx(Q); pairs = h.length > 2 ? h.map((v, i) => [v, h[(i + 1) % h.length]]) : h.length === 2 ? [[h[0], h[1]]] : []; }
    else if (mode === "chain") for (let a = 0; a < Q.length - 1; a++) pairs.push([a, a + 1]);
    else if (mode === "hub" && Q.length) {
      const how = c.hub || "biggest";
      if (how === "biggest" || how === "first") {
        let hi = 0; if (how === "biggest") for (let i = 1; i < Q.length; i++) if (area(its[g[i]]) > area(its[g[hi]])) hi = i;
        Q.forEach((q, b) => { if (b !== hi) segs.push([Q[hi], q]); });
      } else { const hp = { centre: [W / 2, H / 2], "top left": [0, 0], "top right": [W, 0], "bottom left": [0, H], "bottom right": [W, H] }[how] || [W / 2, H / 2];
        Q.forEach(q => segs.push([hp, q])); }
    } else if (mode === "corners" && Q.length > 1) {
      let big = 0; for (let i = 1; i < Q.length; i++) if (area(its[g[i]]) > area(its[g[big]])) big = i;
      const [bx, by, bw, bh] = its[g[big]][2], cs = [[bx, by], [bx + bw, by], [bx, by + bh], [bx + bw, by + bh]];
      for (let a = 0; a < Q.length; a++) if (a !== big) segs.push([Q[a], cs.reduce((m, q) => dist(q, Q[a]) < dist(m, Q[a]) - 1e-12 ? q : m)]);
    } else if (mode === "trail") {
      const n = Math.max(2, parseInt(c.length ?? 20));
      for (const i of g) { const tr = its[i][0], k = t - tr.t0, path = [];
        for (let j = Math.max(0, k - n); j <= Math.min(k, tr.box.length - 1); j++) path.push(centre(tr.box[j]));
        for (let j = 0; j < path.length - 1; j++) segs.push([path[j], path[j + 1], (j + 1) / path.length]); }
    }
    if (c.same_depth !== undefined && c.same_depth !== null && c.same_depth !== false) {
      const tol = c.same_depth === true ? 0.15 : parseFloat(c.same_depth), zs = g.map(i => zAt(its[i][0], t));
      pairs = pairs.filter(([a, b]) => zs[a] == null || zs[b] == null || Math.abs(zs[a] - zs[b]) <= tol);
    }
    for (const [a, b] of pairs) segs.push([Q[a], Q[b]]);
  }
  return segs;
}
export function stampPoints(tr, every, unit) {
  const step = Math.max(1, parseFloat(every)) * unit, out = [], seen = tr.seen || tr.box.map(() => 1);
  let walked = 0, last = null;
  tr.box.forEach((b, k) => { const c = centre(b); if (last) walked += Math.hypot(c[0] - last[0], c[1] - last[1]); last = c;
    if ((!out.length || walked >= step) && seen[k]) { out.push([k, c[0], c[1]]); walked = 0; } });
  return out;
}
export function styled(a, b, style) {
  if (style === "angle") return [a, [b[0], a[1]], b];
  if (style === "curved") { const mx = (a[0] + b[0]) / 2, my = (a[1] + b[1]) / 2, cx = mx - (b[1] - a[1]) * 0.2, cy = my + (b[0] - a[0]) * 0.2, out = [];
    for (let i = 0; i <= 12; i++) { const u = i / 12; out.push([(1 - u) ** 2 * a[0] + 2 * (1 - u) * u * cx + u * u * b[0], (1 - u) ** 2 * a[1] + 2 * (1 - u) * u * cy + u * u * b[1]]); }
    return out; }
  return [a, b];
}
export function lookOf(raw, presets, def) {        // = trackdraw.look()
  if (typeof raw === "string" && presets?.[raw.trim()]) raw = presets[raw.trim()];
  let got = {}; try { got = typeof raw === "string" ? (raw.trim() ? JSON.parse(raw) : {}) : raw || {}; } catch { got = {}; }
  const out = JSON.parse(JSON.stringify(def || {}));
  for (const [k, v] of Object.entries(got || {})) {
    if (out[k] && typeof out[k] === "object" && !Array.isArray(out[k]) && v && typeof v === "object" && !Array.isArray(v)) Object.assign(out[k], v);
    else if (k in out) out[k] = v;
  }
  out.states = { ...(def?.states || {}), ...(out.states || {}) }; out.stamp = { ...(def?.stamp || {}), ...(out.stamp || {}) };
  return out;
}

// ------------------------------------------------------------------------------------------ depth (= trackdraw depth_under / depth_lines)
const DEPTH_GRID = 480;
const hexRGB = h => { h = String(h || "#ffffff").replace("#", ""); return [0, 2, 4].map(i => parseInt(h.slice(i, i + 2), 16) || 0); };
function depthGrid(img, W, H, blur) {              // the depth picture on the work grid: Float32Array (0-1, near = 1), gw, gh
  const gw = Math.min(W, DEPTH_GRID), gh = Math.max(2, Math.round(gw * H / W));
  const c = document.createElement("canvas"); c.width = gw; c.height = gh;
  const q = c.getContext("2d", { willReadFrequently: true }); if (blur) q.filter = "blur(1px)"; q.drawImage(img, 0, 0, gw, gh);
  const px = q.getImageData(0, 0, gw, gh).data, d = new Float32Array(gw * gh);
  for (let i = 0; i < d.length; i++) d[i] = px[i * 4] / 255;
  return { d, gw, gh };
}
function gridCanvas(gw, gh, fill) {                // an RGBA canvas of the grid's size, filled per cell by fill(depth) -> [r, g, b, a]
  const c = document.createElement("canvas"); c.width = gw; c.height = gh;
  const q = c.getContext("2d"), im = q.createImageData(gw, gh);
  return { c, q, im, put() { q.putImageData(im, 0, 0); return c; } };
}
export function marchingSquares(d, gw, gh, level) {     // contour segments of one level on the grid (cell centres at +0.5)
  const segs = [], v = (x, y) => d[y * gw + x];
  for (let y = 0; y < gh - 1; y++) for (let x = 0; x < gw - 1; x++) {
    const a = v(x, y), b = v(x + 1, y), c = v(x + 1, y + 1), e = v(x, y + 1);
    const k = (a >= level ? 8 : 0) | (b >= level ? 4 : 0) | (c >= level ? 2 : 0) | (e >= level ? 1 : 0);
    if (k === 0 || k === 15) continue;
    const lerp = (p, q) => (level - p) / ((q - p) || 1e-9);
    const T = [x + lerp(a, b), y], R = [x + 1, y + lerp(b, c)], B = [x + lerp(e, c), y + 1], L = [x, y + lerp(a, e)];
    const add = (p, q) => segs.push([p[0] + 0.5, p[1] + 0.5, q[0] + 0.5, q[1] + 0.5]);
    switch (k) { case 1: case 14: add(L, B); break; case 2: case 13: add(B, R); break; case 3: case 12: add(L, R); break; case 4: case 11: add(T, R); break;
      case 6: case 9: add(T, B); break; case 7: case 8: add(L, T); break; case 5: add(L, T); add(B, R); break; case 10: add(T, R); add(L, B); break; }
  }
  return segs;
}
function depthUnder(g, lk, t, depthAt, frameAt, W, H, data) {
  const entries = (lk.depth || []).filter(e => ["time", "slice", "fog", "layers"].includes(e.type) && (!data || spanWeight(e, t, data) > 0)), img = depthAt?.(t);
  if (!entries.length || !img) return;
  const { d, gw, gh } = depthGrid(img, W, H, false), cv = g.canvas, M = g.getTransform();
  const temp = () => { const c = document.createElement("canvas"); c.width = cv.width; c.height = cv.height; const q = c.getContext("2d"); q.setTransform(M); return [c, q]; };
  const toOut = c => { g.save(); g.setTransform(1, 0, 0, 1, 0, 0); g.drawImage(c, 0, 0); g.restore(); };
  for (const e of entries) {
    const sw = data ? spanWeight(e, t, data) : 1, cnt = Math.max(2, parseInt(e.count ?? 6)), band = i => Math.min(cnt - 1, Math.floor((1 - d[i]) * cnt));
    if (e.type === "time") {
      const delay = Math.max(0, parseInt(e.delay ?? 4));
      for (let k = 1; k < cnt; k++) {
        const f = frameAt(Math.max(0, t - k * delay)); if (!f) continue;
        const m = gridCanvas(gw, gh); for (let i = 0; i < d.length; i++) m.im.data[i * 4 + 3] = band(i) === k ? 255 : 0;
        const [c, q] = temp(); q.drawImage(f, 0, 0, W, H); q.globalCompositeOperation = "destination-in"; q.imageSmoothingEnabled = false; q.drawImage(m.put(), 0, 0, W, H); toOut(c);
      }
    } else if (e.type === "slice") {
      const lo = parseFloat(e.from ?? 0.5), hi = parseFloat(e.to ?? 1), soft = 0.03, m = gridCanvas(gw, gh);
      for (let i = 0; i < d.length; i++) m.im.data[i * 4 + 3] = 255 * (1 - sw * (1 - Math.min(1, Math.max(0, (d[i] - lo) / soft + 0.5)) * Math.min(1, Math.max(0, (hi - d[i]) / soft + 0.5))));
      g.save(); g.globalCompositeOperation = "destination-in"; g.drawImage(m.put(), 0, 0, W, H); g.restore();
      g.save(); g.globalCompositeOperation = "destination-over"; g.fillStyle = lk.background === "transparent" ? "rgba(0,0,0,0)" : "#000"; g.fillRect(0, 0, W, H); g.restore();
    } else if (e.type === "fog") {
      const [r, gg, b] = hexRGB(e.colour ?? "#000000"), amt = parseFloat(e.amount ?? 0.7) * sw, m = gridCanvas(gw, gh);
      for (let i = 0; i < d.length; i++) { const o = i * 4; m.im.data[o] = r; m.im.data[o + 1] = gg; m.im.data[o + 2] = b; m.im.data[o + 3] = 255 * Math.min(1, Math.max(0, amt * (1 - d[i]))); }
      g.save(); g.globalCompositeOperation = blendOp(e); g.drawImage(m.put(), 0, 0, W, H); g.restore();
    } else if (e.type === "layers") {
      const n0 = hexRGB(e.near ?? "#f18a58"), f0 = hexRGB(e.far ?? "#a187b7"), op = parseFloat(e.opacity ?? 0.6) * sw, m = gridCanvas(gw, gh);
      for (let i = 0; i < d.length; i++) { const k = band(i) / (cnt - 1), o = i * 4; for (let ch = 0; ch < 3; ch++) m.im.data[o + ch] = n0[ch] + (f0[ch] - n0[ch]) * k; m.im.data[o + 3] = 255 * op; }
      g.save(); g.imageSmoothingEnabled = false; g.globalCompositeOperation = blendOp(e); g.drawImage(m.put(), 0, 0, W, H); g.restore();
    }
  }
}
function depthLines(g, lk, t, depthAt, W, H, unit, data) {
  const entries = (lk.depth || []).filter(e => e.type === "lines"), img = depthAt?.(t);
  if (!entries.length || !img) return;
  const { d, gw, gh } = depthGrid(img, W, H, true), sx = W / gw, sy = H / gh;
  for (const e of entries) {
    const cnt = Math.max(2, parseInt(e.count ?? 12));
    g.save(); g.strokeStyle = e.colour || "#fff"; g.lineWidth = parseFloat(e.width ?? 0.8) * unit; g.globalAlpha = parseFloat(e.opacity ?? 0.7) * (data ? spanWeight(e, t, data) : 1); g.globalCompositeOperation = blendOp(e); g.beginPath();
    for (let k = 1; k < cnt; k++) for (const [x0, y0, x1, y1] of marchingSquares(d, gw, gh, k / cnt)) { g.moveTo(x0 * sx, y0 * sy); g.lineTo(x1 * sx, y1 * sy); }
    g.stroke(); g.restore();
  }
}

// ------------------------------------------------------------------------------------------ the drawing (browser)
// g: a 2D context whose transform maps data pixels; frameAt(i) -> an image of frame i (any size) or null.
export function renderFrame(g, data, lk, t, frameAt, depthAt) {
  const W = data.W, H = data.H, unit = H / 1080, frame = frameAt(t);
  const put = (im, sx, sy, sw, sh, dx, dy, dw, dh) => { if (!im) return; const fx = (im.naturalWidth || im.width) / W, fy = (im.naturalHeight || im.height) / H;
    g.drawImage(im, sx * fx, sy * fy, sw * fx, sh * fy, dx, dy, dw, dh); };
  g.save(); g.clearRect(0, 0, W, H);
  if (lk.background === "video" && frame) put(frame, 0, 0, W, H, 0, 0, W, H);
  else if (lk.background !== "transparent") { g.fillStyle = "#000"; g.fillRect(0, 0, W, H); }
  const items = visible(data, t, lk), col = (h, tr) => h === "track" ? tr?.colour || "#fff" : h || "#fff";
  depthUnder(g, lk, t, depthAt, frameAt, W, H, data);
  // stamps
  const st = lk.stamp || {};
  const swSt = spanWeight(st, t, data), stFrom = spanStart(st, data);
  if (st.on && swSt > 0) {
    const stay = parseFloat(st.stay || 0) * (data.fps || 25);
    for (const tr of data.tracks) {
      if (st.kinds?.length && !st.kinds.includes(tr.kind)) continue;
      for (const [k, x, y] of stampPoints(tr, st.every ?? 14, unit)) {
        const ts = tr.t0 + k; if (ts > t || ts < stFrom - 1e-9 || (stay > 0 && t - ts > stay)) continue;
        const first = st.patch === "first", b = tr.box[first ? 0 : k];
        const s = st.size === "fixed" ? parseFloat(st.fixed ?? 40) * unit : Math.max(4, Math.min(b[2], b[3]));
        const [sx, sy] = first ? centre(b) : [x, y], src = frameAt(first ? tr.t0 : ts); if (!src) continue;
        const fade = (stay <= 0 ? 1 : Math.max(0, 1 - (t - ts) / stay)) * swSt;
        const x0 = Math.round(sx - s / 2), y0 = Math.round(sy - s / 2), x1 = Math.round(sx + s / 2), y1 = Math.round(sy + s / 2);
        const cx0 = Math.max(0, x0), cy0 = Math.max(0, y0), cx1 = Math.min(W, x1), cy1 = Math.min(H, y1); if (cx1 <= cx0 || cy1 <= cy0) continue;
        const dx = Math.round(x - s / 2) + (cx0 - x0), dy = Math.round(y - s / 2) + (cy0 - y0);
        g.globalAlpha = fade; g.globalCompositeOperation = blendOp(st); put(src, cx0, cy0, cx1 - cx0, cy1 - cy0, dx, dy, cx1 - cx0, cy1 - cy0);
        g.globalCompositeOperation = "source-over";
        if (st.outline) { g.globalAlpha = 0.6 * fade; g.strokeStyle = "#fff"; g.lineWidth = 0.7 * unit; g.strokeRect(dx, dy, cx1 - cx0, cy1 - cy0); }
        g.globalAlpha = 1;
      }
    }
  }
  // inside
  const regionPath = (tr, b, shape) => { const o = shape === "outline" ? tr.outline?.[String(t)] : null; g.beginPath();
    if (o && o.length >= 3) o.forEach(([x, y], i) => i ? g.lineTo(x, y) : g.moveTo(x, y)); else g.rect(b[0], b[1], b[2], b[3]); g.closePath(); };
  for (const e of lk.inside || []) for (const [tr, s, b] of items) {
    const sw = spanWeight(e, t, data);
    if ((e.kinds?.length && !e.kinds.includes(tr.kind)) || s === 3 || sw <= 0) continue;
    g.save(); regionPath(tr, b, e.shape || "box"); g.clip(); g.globalCompositeOperation = blendOp(e); g.globalAlpha = sw;
    if (e.type === "reveal" && frame) put(frame, 0, 0, W, H, 0, 0, W, H);
    else if (e.type === "pixels" && frame) { const blk = Math.max(2, Math.floor(parseFloat(e.block ?? 12) * unit)), sw = Math.max(1, Math.floor(W / blk)), sh = Math.max(1, Math.floor(H / blk));
      const c = (renderFrame._px ||= document.createElement("canvas")); if (c.width !== sw || c.height !== sh) { c.width = sw; c.height = sh; }
      const q = c.getContext("2d"); q.imageSmoothingEnabled = true; q.drawImage(frame, 0, 0, sw, sh); g.imageSmoothingEnabled = false; g.drawImage(c, 0, 0, W, H); g.imageSmoothingEnabled = true; }
    else if (e.type === "freeze") { const f0 = frameAt(tr.t0), b0 = tr.box[0]; put(f0, b0[0], b0[1], b0[2], b0[3], Math.trunc(b[0]), Math.trunc(b[1]), b0[2], b0[3]); }
    else if (e.type === "invert") { g.globalCompositeOperation = "difference"; g.fillStyle = "#fff"; g.fillRect(0, 0, W, H); }      // invert is a difference with white
    g.restore();
  }
  depthLines(g, lk, t, depthAt, W, H, unit, data);
  const sn = soundAt(data, t), so = lk.sound || {};
  // lines
  for (const c0 of lk.connect || []) {
    const c = sn && c0.mode === "within" ? { ...c0, distance: parseFloat(c0.distance ?? 0.18) * (1 + (so.reach || 0) * sn.level) } : c0;
    const segs = connectEdges(c, items, data, lk, t), maxd = Math.max(1, ...segs.map(s => dist(s[0], s[1])));
    const op = parseFloat(c.opacity ?? 1) * spanWeight(c, t, data) * (sn ? 1 - (so.flash || 0) * 0.7 * (1 - sn.beat) : 1);
    if (op <= 0) continue;
    g.strokeStyle = col(c.colour); g.lineWidth = parseFloat(c.width ?? 1) * unit * (sn ? 1 + (so.width || 0) * sn.level : 1); g.setLineDash(c.dash ? [6 * unit, 4 * unit] : []); g.globalCompositeOperation = blendOp(c);
    for (const s of segs) { let o = op * (s.length > 2 ? s[2] : 1); if (c.fade && s.length === 2) o *= Math.max(0.15, 1 - dist(s[0], s[1]) / (maxd * 1.05));
      g.globalAlpha = o; g.beginPath(); styled(s[0], s[1], c.style || "straight").forEach(([x, y], i) => i ? g.lineTo(x, y) : g.moveTo(x, y)); g.stroke(); }
    g.setLineDash([]); g.globalAlpha = 1; g.globalCompositeOperation = "source-over";
  }
  // marks
  for (const mk of lk.marks || []) {
    g.globalCompositeOperation = blendOp(mk);
    const kinds = mk.kinds || [], typ = mk.type || "box", width = parseFloat(mk.width ?? 1) * unit, op = parseFloat(mk.opacity ?? 1) * spanWeight(mk, t, data), sz0 = parseFloat(mk.size ?? 1), ds = parseFloat(mk.depth_scale || 0);
    if (op <= 0) continue;
    for (const [tr, s, b] of items) {
      if (kinds.length && !kinds.includes(tr.kind)) continue;
      const zz = ds ? zAt(tr, t) : null, sz = (zz != null ? sz0 * Math.max(0.1, 1 + ds * (zz - 0.5) * 2) : sz0) * (sn ? 1 + (so.pulse || 0) * (0.3 * sn.level + 0.5 * sn.beat) : 1);
      const c = col(mk.colour, tr); g.strokeStyle = c; g.fillStyle = c; g.lineWidth = width; g.globalAlpha = op * (s === 3 ? 0.45 : 1);
      let [x, y, w, h] = b;
      if ((typ === "box" || typ === "brackets") && sz !== 1) { const [cx, cy] = centre(b); w *= sz; h *= sz; x = cx - w / 2; y = cy - h / 2; }
      if (typ === "box") { g.strokeRect(x, y, w, h); if (mk.fill) { g.globalAlpha *= parseFloat(mk.fill); g.fillRect(x, y, w, h); } }
      else if (typ === "brackets") { const L = Math.min(w, h) * 0.22; g.beginPath();
        for (const p of [[[x, y + L], [x, y], [x + L, y]], [[x + w - L, y], [x + w, y], [x + w, y + L]], [[x + w, y + h - L], [x + w, y + h], [x + w - L, y + h]], [[x + L, y + h], [x, y + h], [x, y + h - L]]])
          p.forEach(([px, py], i) => i ? g.lineTo(px, py) : g.moveTo(px, py)); g.stroke(); }
      else if (typ === "outline") { const o = tr.outline?.[String(t)]; if (o && o.length >= 3) { g.beginPath(); o.forEach(([px, py], i) => i ? g.lineTo(px, py) : g.moveTo(px, py)); g.closePath(); g.stroke();
        if (mk.fill) { g.globalAlpha *= parseFloat(mk.fill); g.fill(); } } }
      else if (typ === "circle") { const [cx, cy] = centre(b); g.beginPath(); g.arc(cx, cy, Math.max(2, Math.hypot(w, h) / 2 * sz), 0, 7);
        if (mk.fill) { const a = g.globalAlpha; g.globalAlpha = a * parseFloat(mk.fill); g.fill(); g.globalAlpha = a; } g.stroke(); }
      else if (typ === "cross") { const [cx, cy] = centre(b), L = Math.max(4, 7 * sz * unit); g.beginPath(); g.moveTo(cx - L, cy); g.lineTo(cx + L, cy); g.moveTo(cx, cy - L); g.lineTo(cx, cy + L); g.stroke(); }
      else if (typ === "dot") { const [cx, cy] = centre(b); g.beginPath(); g.arc(cx, cy, Math.max(1, sz * unit), 0, 7); g.fill(); }
      else if (typ === "label") { const txt = labelText(mk.text ?? "{id}", tr, s, b, t, data, lk), size = parseFloat(mk.size ?? 14) * unit * (zz != null ? sz / sz0 : 1), at = mk.at || "top left";
        g.font = `600 ${size}px "Hanken Grotesk KKD", system-ui, sans-serif`; g.textBaseline = "alphabetic"; g.textAlign = at === "centre" ? "center" : "left";
        const o = { "top left": [x, y - 4 * unit], "top right": [x + w + 4 * unit, y + size * 0.7], below: [x, y + h + size + 2 * unit], "bottom left": [x, y + h - 4 * unit], centre: [x + w / 2, y + h / 2] }[at] || [x, y - 4 * unit];
        g.fillText(txt, o[0], o[1]); g.textAlign = "left"; }
      else if (typ === "skeleton" || typ === "points") { const pts = tr.points?.[String(t)]; if (!pts) continue;
        if (typ === "points") for (const [px, py] of Object.values(pts)) { g.beginPath(); g.arc(px, py, Math.max(1, 2 * sz * unit), 0, 7); g.fill(); }
        else { const names = data.names?.[tr.kind] || [], edges = data.skeleton?.[tr.kind] || []; g.beginPath();
          for (const [a, bb] of edges) { const p = pts[names[a]], q = pts[names[bb]]; if (p && q) { g.moveTo(p[0], p[1]); g.lineTo(q[0], q[1]); } } g.stroke(); } }
    }
    g.globalAlpha = 1; g.globalCompositeOperation = "source-over";
  }
  g.restore();
}

// ------------------------------------------------------------------------------------------ UI helpers
let styled_ = false;
function style() {
  if (styled_) return; styled_ = true;
  const s = document.createElement("style");
  s.textContent = `
@font-face{font-family:"Hanken Grotesk KKD";src:url("${HERE}fonts/HankenGrotesk-latin-var.woff2") format("woff2");font-weight:200 800;font-display:swap}
.kkt{display:flex;flex-direction:column;gap:5px;width:100%;height:100%;font-family:"Hanken Grotesk KKD",system-ui,sans-serif;font-size:11px;color:rgba(255,255,255,.85);text-transform:lowercase;box-sizing:border-box}
.kkt-stage{flex:1;min-height:120px;position:relative;background:#101010;border-radius:6px;overflow:hidden}
.kkt-stage canvas{position:absolute;inset:0;width:100%;height:100%;display:block;touch-action:none;outline:none}
.kkt-row{display:flex;flex-wrap:wrap;gap:4px;align-items:center}
.kkt-row:has(>.kv){flex-wrap:nowrap;align-items:flex-start}
.kkt-row>.k{color:rgba(255,255,255,.45);min-width:54px;flex:none;line-height:18px}
.kkt-row>.kv{display:flex;flex-wrap:wrap;gap:4px;align-items:center;flex:1;min-width:0}
.kkt-row input[type=range]{flex:1;min-width:60px;max-width:160px;accent-color:${ORANGE}}
.kkt-row input[type=text]{background:#1d1b20;border:1px solid rgba(255,255,255,.14);color:#fff;border-radius:6px;padding:1px 6px;font:inherit;text-transform:none;min-width:90px;flex:1}
.kkt-row input[type=color]{width:22px;height:18px;border:none;padding:0;background:none;cursor:pointer}
.kkt-row .v{min-width:34px;color:rgba(255,255,255,.6);font-variant-numeric:tabular-nums}
.kkt-row select{background:#1d1b20;color:#fff;border:1px solid rgba(255,255,255,.14);border-radius:6px;font:inherit}
.kkt button{font:inherit;text-transform:lowercase;border:1px solid rgba(255,255,255,.14);background:#1d1b20;color:rgba(255,255,255,.8);border-radius:999px;padding:1px 8px;cursor:pointer;line-height:16px}
.kkt button:hover{border-color:${ORANGE};color:#fff}
.kkt button[aria-pressed=true]{background:${LAV};border-color:${LAV};color:#141414}
.kkt button.act{border-color:${ORANGE};color:${ORANGE}}
.kkt button.sw{width:16px;height:16px;padding:0;border-radius:50%}
.kkt-hint{color:rgba(255,255,255,.5);min-height:14px}
.kkt-card{border:1px solid rgba(255,255,255,.1);border-radius:8px;padding:5px 6px;display:flex;flex-direction:column;gap:3px}
.kkt-card .kkt-row:first-child{justify-content:space-between}
.kkt-span{position:relative;height:22px;flex:1;min-width:140px;cursor:pointer;touch-action:none}
.kkt-span .tr{position:absolute;left:6px;right:6px;top:10px;height:2px;background:rgba(255,255,255,.18);border-radius:2px}
.kkt-span .on{position:absolute;top:9px;height:4px;background:${ORANGE};border-radius:2px}
.kkt-span .dot{position:absolute;top:4px;width:12px;height:12px;margin-left:-6px;border-radius:50%;background:#fff;border:2px solid ${ORANGE};box-sizing:border-box}
.kkt-span .now{position:absolute;top:2px;width:2px;height:18px;margin-left:-1px;background:${LAV}}
.kkt-big{position:fixed;inset:0;z-index:10000;background:rgba(8,8,10,.97);display:grid;grid-template-columns:1fr 420px;gap:12px;padding:14px;box-sizing:border-box}
.kkt-big .kkt{height:100%}
.kkt-side{overflow:auto;display:flex;flex-direction:column;gap:8px;padding-right:4px}
.kkt-side h4{margin:6px 0 0;font-size:11px;font-weight:700;color:#fff;letter-spacing:.04em;display:flex;justify-content:space-between;align-items:center}`;
  document.head.appendChild(s);
}
export function hideWidget(w) { if (!w) return; w.hidden = true; w.computeSize = () => [0, -4]; if (w.element) w.element.style.display = "none"; if (w.inputEl) w.inputEl.style.display = "none"; }
export const viewURL = m => api.apiURL(`/view?filename=${encodeURIComponent(m.filename)}&subfolder=${encodeURIComponent(m.subfolder || "")}&type=${encodeURIComponent(m.type || "temp")}`);
export const frameURL = (side, i) => api.apiURL(`/view?filename=${pad(i, 5)}.jpg&subfolder=${encodeURIComponent(side.subfolder)}&type=${encodeURIComponent(side.type || "temp")}`);
export const esc = s => String(s ?? "").replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
// a row with a label: everything after the label goes into its own wrapping box (.kv), so a long row wraps under
// its pills and not under the label; kept up to date while the rows are rebuilt (also for kubakub_pointcloud.js)
export function tidyRows(root) {
  const run = () => { for (const r of root.querySelectorAll(".kkp-row, .kkt-row")) { const k = r.firstElementChild;
    if (!k?.classList.contains("k") || r.children.length < 2 || r.lastElementChild.classList.contains("kv")) continue;
    const kv = document.createElement("div"); kv.className = "kv"; while (k.nextSibling) kv.appendChild(k.nextSibling); r.appendChild(kv); } };
  new MutationObserver(run).observe(root, { childList: true, subtree: true }); run();
}
export const stop = el => { for (const t of ["pointerdown", "pointermove", "pointerup", "wheel", "keydown"]) el.addEventListener(t, e => { if (!e.target.closest?.("canvas")) e.stopPropagation(); }, { passive: true }); };
let SERVER = null;                                  // presets, default look, finder defaults, model lists (from the node file)
async function server() {
  if (SERVER) return SERVER;
  try { const r = await api.fetchApi("/kubakub/track/models"); SERVER = await r.json(); } catch { SERVER = { yolo: [], sam2: [], presets: [], looks: {}, default_look: {}, defaults: {} }; }
  return SERVER;
}
const KIND_NAMES = { motion: "motion", colour: "colour", lights: "lights", yolo: "people & things", words: "words", body: "body", hands: "hands", face: "face", motif: "motif" };
const KIND_COL = { motion: "#ffffff", colour: "#6ae4ff", lights: "#ffe27a", yolo: LAV, words: "#9dff8a", body: "#39ff6a", hands: "#6ae4ff", face: "#ff8ad8", motif: ORANGE };
const yoloLabel = m => m + (/-pose/i.test(m) ? "  (+ body points)" : /-seg/i.test(m) ? "  (+ outlines)" : "");
const YOLO_CLASSES = ["person", "car", "bicycle", "motorcycle", "bus", "truck", "bird", "dog", "cat", "horse", "umbrella", "backpack", "handbag", "boat", "any"];
export const PALETTE = ["#ffffff", ORANGE, LAV, "#39ff6a", "#e8b04a", "#ff3b30", "#6ae4ff"];
// a node + the output nodes after it, nothing else (ComfyUI's partial run) -> how many nodes after it
export function renderWithSavers(node) {
  const g = node.graph, ids = new Set([String(node.id)]), seen = new Set([node.id]);
  const link = id => g.getLink?.(id) ?? g.links?.get?.(id) ?? g.links?.[id];
  const walk = n => { for (const o of n.outputs || []) for (const id of o.links || []) { const l = link(id), t = l && g.getNodeById(l.target_id);
    if (!t || seen.has(t.id)) continue; seen.add(t.id); if (t.constructor?.nodeData?.output_node) ids.add(String(t.id)); walk(t); } };
  walk(node); app.queuePrompt(0, 1, [...ids]);
  return ids.size - 1;
}

// The player: frames of a tracker run (JPEGs in temp), cached, one shared clock.
class Player {                                       // the tracker's frames, and its sound when it has one (then the sound is the clock)
  constructor(onFrame) { this.onFrame = onFrame; this.cache = new Map(); this.t = 0; this.playing = false; this.side = null; this.audio = null; this.audioKey = ""; }
  set(side, sound) {
    if (this.side?.subfolder !== side?.subfolder) { this.cache.clear(); } this.side = side; this.t = Math.min(this.t, Math.max(0, (side?.n || 1) - 1));
    const key = sound?.filename || ""; if (key !== this.audioKey) { this.audio?.pause(); this.audioKey = key; this.audio = sound ? new Audio(viewURL(sound)) : null; }
  }
  img(i) {
    const s = this.side; if (!s) return null; i = Math.min(Math.max(0, i), s.n - 1);
    let im = this.cache.get(i);
    if (!im) { im = new Image(); im.crossOrigin = "anonymous"; im.onload = () => this.onFrame(); im.src = frameURL(s, i); this.cache.set(i, im); if (this.cache.size > 900) this.cache.delete(this.cache.keys().next().value); }
    return im.complete && im.naturalWidth ? im : null;
  }
  preload(k = 20) { for (let i = 1; i <= k; i++) this.img(this.t + i); }
  toggle(on = !this.playing, fps = 25) {
    const run = this.run = (this.run || 0) + 1;      // a new toggle ends the loop before it (no two loops at once)
    this.playing = on && (this.side?.n || 0) > 1; if (!this.playing) { this.audio?.pause(); return; }
    const a = this.audio; if (a) { a.currentTime = this.t / fps; a.play().catch(() => {}); }
    let prev = performance.now(), acc = 0;
    const step = now => { if (run !== this.run) return; if (!this.playing) { a?.pause(); return; }
      if (a && !a.paused) { const f = Math.floor(a.currentTime * fps);
        if (f >= this.side.n) { this.t = 0; a.currentTime = 0; this.onFrame(); }
        else if (f !== this.t) { this.t = f; this.preload(); this.onFrame(); } }
      else { acc += (now - prev) / 1000 * fps;
        if (acc >= 1 && this.img(this.t)) { this.t = (this.t + Math.floor(acc)) % this.side.n; acc %= 1; this.preload(); this.onFrame(); } else if (acc > 3) acc = 1;
        if (a?.ended) { a.currentTime = this.t / fps; a.play().catch(() => {}); } }
      prev = now; requestAnimationFrame(step); };
    requestAnimationFrame(step);
  }
}
function transport(el, player, fps) {                // play button + scrub bar + frame counter in a row
  const row = document.createElement("div"); row.className = "kkt-row";
  row.innerHTML = `<button data-play class="act">play</button><input type="range" min="0" max="0" step="1" value="0"><span class="v"></span>`;
  const r = row.querySelector("input"), v = row.querySelector(".v"), b = row.querySelector("button");
  b.onclick = e => { e.stopPropagation(); player.toggle(!player.playing, fps()); b.textContent = player.playing ? "pause" : "play"; };
  r.oninput = () => { player.toggle(false); b.textContent = "play"; player.t = +r.value; player.preload(); player.onFrame(); };
  row.update = () => { const n = player.side?.n || 1; r.max = String(n - 1); r.value = String(player.t); v.textContent = `${player.t + 1} / ${n}`; row.style.display = n > 1 ? "" : "none"; };
  stop(row); return row;
}

// ------------------------------------------------------------------------------------------ the tracker node
function trackerUI(node) {
  style();
  const W_ = n => node.widgets?.find(w => w.name === n);
  const el = document.createElement("div"); el.className = "kkt"; tidyRows(el);
  el.innerHTML = `<div class="kkt-stage" style="min-height:230px"><canvas tabindex="0"></canvas></div><div data-tr></div><div data-set></div><div class="kkt-hint"></div>`;
  const cv = el.querySelector("canvas"), ctx = cv.getContext("2d"), stage = el.querySelector(".kkt-stage"), hint = el.querySelector(".kkt-hint");
  const st = { data: null, dataKey: "", probe: null, pick: "", busy: false };
  const meta = () => node.properties?.kubakub_track || null;
  const player = new Player(() => draw());
  const fps = () => meta()?.fps || 25;
  const tr = transport(el, player, fps); el.querySelector("[data-tr]").appendChild(tr);
  const getS = () => { let s = {}; try { s = JSON.parse(W_("settings")?.value || "{}") || {}; } catch { s = {}; }
    const d = SERVER?.defaults || {}; const out = JSON.parse(JSON.stringify(d));
    for (const [k, v] of Object.entries(s)) { if (out[k] && typeof out[k] === "object" && !Array.isArray(out[k]) && v && typeof v === "object") Object.assign(out[k], v); else out[k] = v; }
    return out; };
  const setS = s => { const w = W_("settings"); if (!w) return; const v = JSON.stringify(s); if (w.value === v) return; w.value = v; w.callback?.(v); node.graph?.setDirtyCanvas(true, true); st.probe = null; render(); draw(); };
  new ResizeObserver(() => { const d = Math.min(devicePixelRatio || 1, 2); cv.width = Math.max(10, stage.clientWidth * d); cv.height = Math.max(10, stage.clientHeight * d); draw(); }).observe(stage);
  async function load() {
    const m = meta(); if (!m) return; player.set(m.frames, m.audio);
    const key = m.tracks?.filename || ""; if (key === st.dataKey) return; st.dataKey = key;
    try { const r = await fetch(viewURL(m.tracks)); st.data = await r.json(); } catch { st.data = null; }
    draw();
  }
  const fit = () => { const m = meta(); if (!m) return null; const k = Math.min(cv.width / m.W, cv.height / m.H) * 0.98; return { k, ox: (cv.width - m.W * k) / 2, oy: (cv.height - m.H * k) / 2 }; };
  function overlay(data, t, dashed) {
    const m = meta(), f = fit(); if (!data || !f) return;
    ctx.save(); ctx.setTransform(f.k, 0, 0, f.k, f.ox, f.oy); const u = 1 / f.k;
    for (const trk of data.tracks) {
      const k = t - trk.t0; if (k < 0 || k >= trk.box.length) continue;
      const [x, y, w, h] = trk.box[k], c = KIND_COL[trk.kind] || "#fff";
      ctx.strokeStyle = c; ctx.fillStyle = c; ctx.lineWidth = 1.5 * u; ctx.setLineDash(dashed ? [5 * u, 3 * u] : []);
      ctx.globalAlpha = trk.seen?.[k] === 0 ? 0.45 : 1; ctx.strokeRect(x, y, w, h);
      const o = trk.outline?.[String(t)]; if (o?.length >= 3) { ctx.beginPath(); o.forEach(([px, py], i) => i ? ctx.lineTo(px, py) : ctx.moveTo(px, py)); ctx.closePath(); ctx.globalAlpha *= 0.85; ctx.stroke(); }
      const pts = trk.points?.[String(t)]; if (pts) for (const [px, py] of Object.values(pts)) { ctx.beginPath(); ctx.arc(px, py, 2.2 * u, 0, 7); ctx.fill(); }
      ctx.globalAlpha = 1; ctx.font = `700 ${11 * u}px "Hanken Grotesk KKD", sans-serif`; ctx.fillText(`${trk.id} ${trk.label || trk.kind}`, x, y - 3 * u);
    }
    ctx.setLineDash([]); ctx.restore();
    void m;
  }
  function draw() {
    const m = meta(), D = Math.min(devicePixelRatio || 1, 2);
    ctx.setTransform(1, 0, 0, 1, 0, 0); ctx.fillStyle = "#101010"; ctx.fillRect(0, 0, cv.width, cv.height); tr.update();
    if (!m) { ctx.fillStyle = "rgba(255,255,255,.5)"; ctx.font = `${13 * D}px "Hanken Grotesk KKD", sans-serif`; ctx.textAlign = "center";
      ctx.fillText("pick the finders below, then run once: the video and what was found show here", cv.width / 2, cv.height / 2); ctx.textAlign = "left"; hint.textContent = ""; return; }
    const f = fit(), im = player.img(player.t); if (im) ctx.drawImage(im, f.ox, f.oy, m.W * f.k, m.H * f.k);
    if (st.probe && st.probe.t === player.t) overlay(st.probe.data, player.t, true); else overlay(st.data, player.t, false);
    const s = getS();
    hint.textContent = st.pick === "colour" ? "click the colour on the video" : st.pick === "motif" ? "click what to follow · shift click = not this · as many clicks as you like" :
      st.probe ? (st.probe.t === player.t ? `this frame with the settings now: ${st.probe.data.tracks.length} found (dashed) · queue to run the whole clip` : "") :
      `${st.data?.tracks?.length ?? 0} tracks · ${m.n} frames ${m.W}×${m.H}`;
    if (s.motif?.on) {                                // the clicked motif points on this frame
      ctx.save(); ctx.setTransform(1, 0, 0, 1, 0, 0);
      for (const [pt, x, y, pos] of s.motif.points || []) if (pt === player.t) { ctx.fillStyle = pos ? ORANGE : "#ff3b30"; ctx.beginPath(); ctx.arc(f.ox + x * m.W * f.k, f.oy + y * m.H * f.k, 5 * D, 0, 7); ctx.fill(); }
      ctx.restore(); }
  }
  // finder settings (pills and sliders)
  const set = el.querySelector("[data-set]"); stop(set);
  const slider = (path, label, min, max, step, val, fmt = v => v) => `<div class="kkt-row"><span class="k">${label}</span><input type="range" data-p="${path}" min="${min}" max="${max}" step="${step}" value="${val}"><span class="v">${fmt(val)}</span></div>`;
  function render() {
    const s = getS(), S = SERVER || {};
    const on = k => s[k]?.on;
    let h = `<div class="kkt-row"><span class="k">find</span>${["motion", "colour", "lights", "yolo", "words", "body", "hands", "face", "motif"].map(k => `<button data-f="${k}" aria-pressed="${!!on(k)}">${KIND_NAMES[k]}</button>`).join("")}<button data-f="outline" aria-pressed="${!!on("outline")}" title="the exact outline inside the boxes (SAM 2)">outlines</button></div>`;
    if (on("motion")) h += `<div class="kkt-card"><div class="kkt-row"><b>motion</b></div>${slider("motion.sensitivity", "sensitivity", 0, 1, 0.05, s.motion.sensitivity)}${slider("motion.min_size", "smallest", 0.0002, 0.05, 0.0002, s.motion.min_size, v => (v * 100).toFixed(2) + "%")}</div>`;
    if (on("colour")) h += `<div class="kkt-card"><div class="kkt-row"><b>colour</b></div><div class="kkt-row"><span class="k">colour</span><input type="color" data-p="colour.hex" value="${esc(s.colour.hex)}"><button data-pick="colour" class="act">pick on the video</button></div>${slider("colour.tolerance", "how close", 0, 1, 0.02, s.colour.tolerance)}${slider("colour.min_size", "smallest", 0.0001, 0.05, 0.0001, s.colour.min_size, v => (v * 100).toFixed(2) + "%")}</div>`;
    if (on("lights")) h += `<div class="kkt-card"><div class="kkt-row"><b>lights</b></div>${slider("lights.level", "brighter than", 0.3, 1, 0.01, s.lights.level)}${slider("lights.min_size", "smallest", 0.00005, 0.01, 0.00005, s.lights.min_size, v => (v * 100).toFixed(3) + "%")}</div>`;
    if (on("yolo") || on("body")) h += `<div class="kkt-card"><div class="kkt-row"><b>people & things (yolo)</b>${on("body") && !on("yolo") ? '<span class="k">body: one person per frame (turn on people & things for everybody)</span>' : ""}</div>
      <div class="kkt-row"><span class="k">model</span><select data-p="yolo.model">${(S.yolo || []).map(m => `<option value="${esc(m)}" ${m === s.yolo.model ? "selected" : ""}>${esc(yoloLabel(m))}</option>`).join("") || "<option>no model in models/ultralytics</option>"}</select></div>
      <div class="kkt-row"><span class="k">what</span>${YOLO_CLASSES.map(c => `<button data-cls="${c}" aria-pressed="${(s.yolo.classes || []).includes(c)}">${c}</button>`).join("")}</div>
      ${slider("yolo.confidence", "sure", 0.05, 0.95, 0.05, s.yolo.confidence)}</div>`;
    if (on("words")) h += `<div class="kkt-card"><div class="kkt-row"><b>words (grounding dino)</b></div>
      <div class="kkt-row"><span class="k">find</span><input type="text" data-p="words.words" value="${esc(s.words.words)}" placeholder="umbrella, pigeon, head"></div>
      <div class="kkt-row"><span class="k">model</span><select data-p="words.model">${(S.dino || []).map(m => `<option ${m === s.words.model ? "selected" : ""}>${esc(m)}</option>`).join("") || "<option>no model in models/grounding-dino</option>"}</select></div>
      ${slider("words.confidence", "sure", 0.05, 0.9, 0.05, s.words.confidence)}</div>`;
    if (on("motif")) h += `<div class="kkt-card"><div class="kkt-row"><b>motif (sam 2)</b><span><button data-pick="motif" class="act">click on the video</button> <button data-clear>clear ${(s.motif.points || []).length} points</button></span></div>
      <div class="kkt-row"><span class="k">object</span>${[0, 1, 2, 3].map(o => `<button data-obj="${o}" aria-pressed="${(st.obj || 0) === o}">${o + 1}</button>`).join("")}<span class="k">model</span><select data-p="motif.model">${(S.sam2 || []).map(m => `<option ${m === s.motif.model ? "selected" : ""}>${esc(m)}</option>`).join("")}</select></div></div>`;
    if (on("outline")) h += `<div class="kkt-card"><div class="kkt-row"><b>outlines (sam 2)</b></div><div class="kkt-row"><span class="k">around</span>${["yolo", "words", "motion", "colour", "lights"].map(k => `<button data-ok="${k}" aria-pressed="${(s.outline.kinds || []).includes(k)}">${KIND_NAMES[k]}</button>`).join("")}</div>
      <div class="kkt-row"><span class="k">model</span><select data-p="outline.model">${(S.sam2 || []).map(m => `<option ${m === s.outline.model ? "selected" : ""}>${esc(m)}</option>`).join("")}</select></div></div>`;
    const di = node.inputs?.findIndex(x => x.name === "depth");
    if (di >= 0 && node.inputs[di].link != null) h += `<div class="kkt-row"><span class="k">depth</span><span class="k" style="min-width:0">near is</span>${["bright", "dark"].map(v => `<button data-near="${v}" aria-pressed="${(s.depth_near || "bright") === v}" title="${v === "bright" ? "Depth Anything 2 / 3, Video Depth Anything" : "Marigold, most metric depth"}">${v}</button>`).join("")}</div>`;
    h += `<div class="kkt-row"><span class="k">look at</span>${["640", "960", "1280", "full"].map(a => `<button data-an="${a}" aria-pressed="${String(s.analysis) === a}">${a}</button>`).join("")}<span style="flex:1"></span><button data-probe class="act" ${meta() ? "" : "disabled"} title="the finders on the frame shown, with the settings as they are now">${st.busy ? "looking…" : "try this frame"}</button></div>`;
    h += slider("smooth", "smooth", 0, 1, 0.05, s.smooth) + slider("keep_lost", "keep lost", 0, 60, 1, s.keep_lost, v => v + " fr");
    set.innerHTML = h;
  }
  set.addEventListener("input", e => {
    const p = e.target.dataset?.p; if (!p) return; const s = getS(), [a, b] = p.split(".");
    let v = e.target.type === "range" ? +e.target.value : e.target.value;
    if (b) (s[a] ||= {})[b] = v; else s[a] = v;
    if (e.target.type === "range") { const sp = e.target.nextElementSibling; if (sp) sp.textContent = String(v); }
    const w = W_("settings"); const txt = JSON.stringify(s); w.value = txt; w.callback?.(txt); st.probe = null;
    if (e.type === "input" && e.target.type !== "range" && e.target.type !== "text") render();
  });
  set.addEventListener("change", e => { if (e.target.dataset?.p) { st.probe = null; render(); draw(); } });
  set.addEventListener("click", async e => {
    const t = e.target.closest("button"); if (!t) return; e.stopPropagation(); const s = getS();
    if (t.dataset.f) { s[t.dataset.f].on = !s[t.dataset.f].on; if (t.dataset.f === "body" && s.body.on && !s.yolo.on) s.yolo.on = true; setS(s); }
    else if (t.dataset.cls) { const c = t.dataset.cls, l = new Set(s.yolo.classes || []); l.has(c) ? l.delete(c) : l.add(c); s.yolo.classes = [...l]; setS(s); }
    else if (t.dataset.ok) { const c = t.dataset.ok, l = new Set(s.outline.kinds || []); l.has(c) ? l.delete(c) : l.add(c); s.outline.kinds = [...l]; setS(s); }
    else if (t.dataset.an) { s.analysis = t.dataset.an; setS(s); }
    else if (t.dataset.near) { s.depth_near = t.dataset.near; setS(s); }
    else if (t.dataset.pick) { st.pick = st.pick === t.dataset.pick ? "" : t.dataset.pick; cv.style.cursor = st.pick ? "crosshair" : "default"; draw(); }
    else if ("clear" in t.dataset) { s.motif.points = []; setS(s); }
    else if (t.dataset.obj) { st.obj = +t.dataset.obj; render(); }
    else if ("probe" in t.dataset) probe();
  });
  async function probe() {
    if (st.busy || !meta()) return; st.busy = true; render();
    try { const r = await api.fetchApi("/kubakub/track/probe", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ node: String(node.id), frame: player.t, settings: W_("settings")?.value || "" }) });
      const j = await r.json(); if (!r.ok || j.error) throw new Error(j.error || r.status); st.probe = { t: player.t, data: j.tracks }; }
    catch (err) { hint.textContent = "try this frame: " + err.message; st.probe = null; }
    st.busy = false; render(); draw();
  }
  cv.addEventListener("pointerdown", e => {
    if (!st.pick || !meta()) return; e.preventDefault(); e.stopPropagation();
    const m = meta(), f = fit(), r = cv.getBoundingClientRect(), D = cv.width / r.width, x = ((e.clientX - r.left) * D - f.ox) / (m.W * f.k), y = ((e.clientY - r.top) * D - f.oy) / (m.H * f.k);
    if (x < 0 || y < 0 || x > 1 || y > 1) return; const s = getS();
    if (st.pick === "colour") { const p = ctx.getImageData(Math.round((e.clientX - r.left) * D), Math.round((e.clientY - r.top) * D), 1, 1).data;
      s.colour.hex = "#" + [p[0], p[1], p[2]].map(v => pad(v.toString(16), 2)).join(""); st.pick = ""; cv.style.cursor = "default"; setS(s); }
    else { s.motif.points = [...(s.motif.points || []), [player.t, +x.toFixed(4), +y.toFixed(4), e.shiftKey ? 0 : 1, st.obj || 0]]; setS(s); }
  });
  cv.addEventListener("wheel", e => e.stopPropagation(), { passive: true });
  el.kktLoad = () => { load(); render(); draw(); };
  server().then(() => { render(); draw(); });
  return el;
}

// ------------------------------------------------------------------------------------------ the track draw node
const DEPTH_FIELDS = { layers: ["count", "near", "far", "opacity"], lines: ["count", "colour", "width", "opacity"], time: ["count", "delay"], fog: ["colour", "amount"], slice: ["from", "to"] };
const MARK_FIELDS = { box: ["colour", "width", "size", "fill"], brackets: ["colour", "width", "size"], outline: ["colour", "width", "fill"],
  circle: ["colour", "width", "size", "fill"], cross: ["colour", "width", "size"], dot: ["colour", "size"], label: ["colour", "text", "tsize", "at"],
  skeleton: ["colour", "width"], points: ["colour", "size"] };
const LABEL_TOKENS = ["{id}", "{id_hex}", "{id_long}", "{state}", "{label}", "{kind}", "{x}", "{y}", "{z}", "{speed}", "{size}", "{data}", "{frame}", "{t}"];

function drawUI(node) {
  style();
  const W_ = n => node.widgets?.find(w => w.name === n);
  const el = document.createElement("div"); el.className = "kkt"; tidyRows(el);
  el.innerHTML = `<div class="kkt-stage"><canvas></canvas></div><div data-tr></div><div class="kkt-row" data-pre></div><div class="kkt-hint"></div>`;
  const cv = el.querySelector("canvas"), ctx = cv.getContext("2d"), stage = el.querySelector(".kkt-stage"), hint = el.querySelector(".kkt-hint");
  const st = { data: null, key: "" };
  const src = () => { const i = node.inputs?.findIndex(s => s.name === "tracks"); if (i < 0 || node.inputs[i].link == null) return null;
    const up = node.getInputNode?.(i); return up?.properties?.kubakub_track || null; };
  const player = new Player(() => views.forEach(v => v()));
  const views = new Set();
  const dcache = new Map();                          // the tracker's depth frames (JPEG, near = white)
  const depthImg = i => { const side = src()?.depth; if (!side) return null; i = Math.min(Math.max(0, i), side.n - 1); const key = side.subfolder + "|" + i;
    let im = dcache.get(key); if (!im) { im = new Image(); im.crossOrigin = "anonymous"; im.onload = () => views.forEach(v => v()); im.src = frameURL(side, i); dcache.set(key, im); if (dcache.size > 600) dcache.delete(dcache.keys().next().value); }
    return im.complete && im.naturalWidth ? im : null; };
  const fps = () => src()?.fps || 25;
  const tr = transport(el, player, fps); el.querySelector("[data-tr]").appendChild(tr);
  const getLook = () => lookOf(W_("look")?.value || "", SERVER?.looks, SERVER?.default_look);
  const setLook = lk => { const w = W_("look"); const v = JSON.stringify(lk); if (w.value !== v) { w.value = v; w.callback?.(v); node.graph?.setDirtyCanvas(true, true); } views.forEach(f => f()); renderPresets(); };
  async function load() {
    const m = src(); if (!m) { st.data = null; st.key = ""; return; } player.set(m.frames, m.audio);
    const key = m.tracks?.filename || ""; if (key === st.key) return; st.key = key;
    try { const r = await fetch(viewURL(m.tracks)); st.data = await r.json(); } catch { st.data = null; }
    views.forEach(f => f());
  }
  const off = document.createElement("canvas");
  function paint(canvas, c2) {                     // the look at frame t, fitted into a canvas
    const m = src(), D = Math.min(devicePixelRatio || 1, 2);
    c2.setTransform(1, 0, 0, 1, 0, 0); c2.fillStyle = "#101010"; c2.fillRect(0, 0, canvas.width, canvas.height); tr.update();
    if (!m || !st.data) { c2.fillStyle = "rgba(255,255,255,.5)"; c2.font = `${13 * D}px "Hanken Grotesk KKD", sans-serif`; c2.textAlign = "center";
      c2.fillText(m ? "loading…" : "connect kubakub tracker and run it once: the look plays here live", canvas.width / 2, canvas.height / 2); c2.textAlign = "left"; return; }
    const data = st.data, k = Math.min(canvas.width / data.W, canvas.height / data.H) * 0.98, w = Math.max(1, Math.round(data.W * k)), h = Math.max(1, Math.round(data.H * k));
    if (off.width !== w || off.height !== h) { off.width = w; off.height = h; }
    const g = off.getContext("2d"); g.setTransform(w / data.W, 0, 0, h / data.H, 0, 0);
    renderFrame(g, data, getLook(), player.t, i => player.img(i), depthImg);
    const ox = (canvas.width - w) / 2, oy = (canvas.height - h) / 2;
    if (getLook().background === "transparent") { c2.fillStyle = "#2a2a2a"; for (let y = 0; y < h; y += 12) for (let x = (y / 12) % 2 ? 12 : 0; x < w; x += 24) c2.fillRect(ox + x, oy + y, 12, 12); }
    c2.drawImage(off, ox, oy);
  }
  new ResizeObserver(() => { const d = Math.min(devicePixelRatio || 1, 2); cv.width = Math.max(10, stage.clientWidth * d); cv.height = Math.max(10, stage.clientHeight * d); paint(cv, ctx); }).observe(stage);
  const own = () => { if (!el.isConnected) { views.delete(own); return; } paint(cv, ctx);
    hint.textContent = st.data ? `${st.data.tracks.length} tracks · ${getLook().marks?.length || 0} marks · ${getLook().connect?.length || 0} kinds of lines${getLook().stamp?.on ? " · stamp trail" : ""}` : ""; };
  views.add(own);
  const pre = el.querySelector("[data-pre]"); stop(pre);
  function renderPresets() {
    const names = SERVER?.presets || [];
    pre.innerHTML = `<span class="k" style="color:rgba(255,255,255,.45);min-width:44px">start</span>` + names.map(n => `<button data-preset="${esc(n)}">${esc(n)}</button>`).join("") +
      `<span style="flex:1"></span><button data-edit class="act">edit the look</button><button data-render class="act" title="Render the look at full size, and whatever saves it after this node (Create Video, Save ...); nothing else in the workflow runs. The tracking comes from the cache.">render the look</button>`;
  }
  pre.addEventListener("click", e => { const t = e.target.closest("button"); if (!t) return; e.stopPropagation();
    if (t.dataset.preset) setLook(lookOf(t.dataset.preset, SERVER?.looks, SERVER?.default_look)); else if ("edit" in t.dataset) openEditor(); else if ("render" in t.dataset) renderLook(); });
  function renderLook() {
    const after = renderWithSavers(node);
    hint.textContent = `rendering the look${after ? ` and ${after} node(s) after it` : ""}…`;
  }
  // ---- the editor window: big live preview + everything of the look
  function openEditor() {
    const wrap = document.createElement("div"); wrap.className = "kkt-big"; tidyRows(wrap);
    wrap.innerHTML = `<div class="kkt"><div class="kkt-stage"><canvas></canvas></div><div data-tr2></div></div><div class="kkt kkt-side"></div>`;
    const bcv = wrap.querySelector("canvas"), bctx = bcv.getContext("2d"), bstage = wrap.querySelector(".kkt-stage"), side = wrap.querySelector(".kkt-side");
    const tr2 = transport(wrap, player, fps); wrap.querySelector("[data-tr2]").appendChild(tr2);
    const view = () => { if (!wrap.isConnected) { views.delete(view); return; } paint(bcv, bctx); tr2.update(); };
    views.add(view);
    new ResizeObserver(() => { const d = Math.min(devicePixelRatio || 1, 2); bcv.width = Math.max(10, bstage.clientWidth * d); bcv.height = Math.max(10, bstage.clientHeight * d); view(); }).observe(bstage);
    const kindsHere = () => [...new Set((st.data?.tracks || []).map(t => t.kind))];
    const hasZ = () => (st.data?.tracks || []).some(t => t.z?.length);
    const colourRow = (path, v) => `<div class="kkt-row"><span class="k">colour</span>${PALETTE.map(c => `<button class="sw" data-set="${path}" data-val="${c}" style="background:${c}" aria-pressed="${v === c}"></button>`).join("")}<button data-set="${path}" data-val="track" aria-pressed="${v === "track"}" title="each thing in its own colour">own</button><input type="color" data-in="${path}" value="${/^#[0-9a-f]{6}$/i.test(v || "") ? v : "#ffffff"}"></div>`;
    const range = (path, label, min, max, step, v) => `<div class="kkt-row"><span class="k">${label}</span><input type="range" data-in="${path}" min="${min}" max="${max}" step="${step}" value="${v}"><span class="v">${v}</span></div>`;
    const pills = (path, label, opts, v) => `<div class="kkt-row"><span class="k">${label}</span>${opts.map(o => `<button data-set="${path}" data-val="${esc(o)}" aria-pressed="${String(v) === String(o)}">${esc(o)}</button>`).join("")}</div>`;
    const toggle = (path, label, v, tip = "") => `<button data-tog="${path}" aria-pressed="${!!v}" title="${esc(tip)}">${label}</button>`;
    const secs = u => ((st.data?.n || 1) - 1) * u / (st.data?.fps || 25);
    const spanRow = (path, e) => { const sp = e?.span?.length === 2 ? e.span : [0, 1], nowU = (player.t || 0) / Math.max(1, (st.data?.n || 1) - 1);
      return `<div class="kkt-row"><span class="k">time</span><div class="kkt-span" data-span="${path}"><div class="tr"></div><div class="on" style="left:calc(6px + (100% - 12px) * ${sp[0]});width:calc((100% - 12px) * ${sp[1] - sp[0]})"></div>
        <div class="now" style="left:calc(6px + (100% - 12px) * ${nowU})"></div><div class="dot" data-end="0" style="left:calc(6px + (100% - 12px) * ${sp[0]})"></div><div class="dot" data-end="1" style="left:calc(6px + (100% - 12px) * ${sp[1]})"></div></div>
        <span class="v" style="min-width:76px">${secs(sp[0]).toFixed(1)}–${secs(sp[1]).toFixed(1)} s</span></div>`
        + `<div class="kkt-row"><span class="k">fade</span><input type="range" data-in="${path.replace(/\.span$/, ".span_fade")}" min="0" max="3" step="0.1" value="${e?.span_fade ?? 0}"><span class="v">${e?.span_fade ?? 0}</span></div>`; };
    const blendRow = (path, v) => `<div class="kkt-row"><span class="k">blend</span><select data-in="${path}">${Object.keys(OPS).map(m => `<option ${m === (v || "normal") ? "selected" : ""}>${m}</option>`).join("")}</select></div>`;
    const kinds = (path, v) => { const ks = kindsHere(); if (ks.length < 2) return ""; return `<div class="kkt-row"><span class="k">on</span><button data-kinds="${path}" data-val="" aria-pressed="${!(v || []).length}">all</button>${ks.map(k => `<button data-kinds="${path}" data-val="${k}" aria-pressed="${(v || []).includes(k)}">${KIND_NAMES[k] || k}</button>`).join("")}</div>`; };
    function renderSide() {
      const lk = getLook(), S = SERVER || {};
      let h = `<div class="kkt-row" style="justify-content:space-between"><b style="font-size:13px">the look</b><span><button data-render class="act" title="Render it at full size (and what saves it); nothing else runs">render the look</button> <button data-close class="act">close</button></span></div>`;
      h += `<div class="kkt-row"><span class="k">start</span>${(S.presets || []).map(n => `<button data-preset="${esc(n)}">${esc(n)}</button>`).join("")}</div>`;
      h += pills("background", "under it", ["video", "black", "transparent"], lk.background);
      h += `<h4>marks <select data-add="marks"><option value="">+ mark</option>${Object.keys(MARK_FIELDS).map(m => `<option>${m}</option>`).join("")}</select></h4>`;
      (lk.marks || []).forEach((m, i) => {
        const p = `marks.${i}`, F = MARK_FIELDS[m.type] || [];
        h += `<div class="kkt-card"><div class="kkt-row"><b>${esc(m.type)}</b><span>${i ? `<button data-up="${p}">↑</button>` : ""}<button data-del="${p}">×</button></span></div>`;
        if (F.includes("colour")) h += colourRow(`${p}.colour`, m.colour ?? "#ffffff");
        if (F.includes("text")) h += `<div class="kkt-row"><span class="k">text</span><input type="text" data-in="${p}.text" value="${esc(m.text ?? "{id}")}"></div><div class="kkt-row"><span class="k"></span>${LABEL_TOKENS.map(tk => `<button data-tok="${p}.text" data-val="${tk}">${tk.slice(1, -1)}</button>`).join("")}</div>`;
        if (F.includes("tsize")) h += range(`${p}.size`, "size", 6, 48, 1, m.size ?? 14) + pills(`${p}.at`, "where", ["top left", "top right", "below", "bottom left", "centre"], m.at || "top left");
        if (F.includes("width")) h += range(`${p}.width`, "line", 0.2, 6, 0.1, m.width ?? 1);
        if (F.includes("size")) h += range(`${p}.size`, m.type === "dot" || m.type === "points" ? "size" : "scale", 0.2, m.type === "dot" ? 12 : 3, 0.05, m.size ?? 1);
        if (F.includes("fill")) h += range(`${p}.fill`, "fill", 0, 1, 0.05, m.fill ?? 0);
        h += spanRow(`${p}.span`, m) + blendRow(`${p}.blend`, m.blend) + range(`${p}.opacity`, "opacity", 0, 1, 0.05, m.opacity ?? 1) + (hasZ() ? range(`${p}.depth_scale`, "near = bigger", 0, 2, 0.05, m.depth_scale ?? 0) : "") + kinds(`${p}.kinds`, m.kinds) + `</div>`;
      });
      h += `<h4>lines <select data-add="connect"><option value="">+ lines</option>${["nearest", "within", "network", "tree", "hull", "hub", "chain", "trail", "corners"].map(m => `<option>${m}</option>`).join("")}</select></h4>`;
      (lk.connect || []).forEach((c, i) => {
        const p = `connect.${i}`;
        h += `<div class="kkt-card"><div class="kkt-row"><b>${esc(c.mode)}</b><span><button data-del="${p}">×</button></span></div>`;
        h += pills(`${p}.mode`, "join", ["nearest", "within", "network", "tree", "hull", "hub", "chain", "trail", "corners"], c.mode);
        if (c.mode === "nearest") h += range(`${p}.k`, "how many", 1, 6, 1, c.k ?? 1);
        if (c.mode === "within") h += range(`${p}.distance`, "closer than", 0.02, 0.8, 0.01, c.distance ?? 0.18);
        if (c.mode === "hub") h += pills(`${p}.hub`, "to", ["biggest", "first", "centre", "top left", "top right", "bottom left", "bottom right"], c.hub || "biggest");
        if (c.mode === "trail") h += range(`${p}.length`, "length", 2, 120, 1, c.length ?? 20);
        h += pills(`${p}.style`, "line", ["straight", "curved", "angle"], c.style || "straight") + colourRow(`${p}.colour`, c.colour ?? "#ffffff");
        h += range(`${p}.width`, "width", 0.2, 6, 0.1, c.width ?? 1) + range(`${p}.opacity`, "opacity", 0, 1, 0.05, c.opacity ?? 1) + blendRow(`${p}.blend`, c.blend) + spanRow(`${p}.span`, c);
        h += `<div class="kkt-row"><span class="k"></span>${toggle(`${p}.dash`, "dashed", c.dash)}${toggle(`${p}.fade`, "fade with distance", c.fade)}${toggle(`${p}.same_kind`, "same kind only", c.same_kind, "only things found the same way are joined")}${hasZ() ? toggle(`${p}.same_depth`, "same depth only", c.same_depth, "only things at a similar depth are joined") : ""}</div>`;
        h += kinds(`${p}.kinds`, c.kinds) + `</div>`;
      });
      h += `<h4>inside <select data-add="inside"><option value="">+ inside</option>${["reveal", "pixels", "freeze", "invert"].map(m => `<option>${m}</option>`).join("")}</select></h4>`;
      (lk.inside || []).forEach((e, i) => {
        const p = `inside.${i}`;
        h += `<div class="kkt-card"><div class="kkt-row"><b>${esc(e.type)}</b><span><button data-del="${p}">×</button></span></div>` + pills(`${p}.type`, "what", ["reveal", "pixels", "freeze", "invert"], e.type) +
          pills(`${p}.shape`, "shape", ["box", "outline"], e.shape || "box") + (e.type === "pixels" ? range(`${p}.block`, "blocks", 2, 80, 1, e.block ?? 12) : "") + (e.type === "invert" ? "" : blendRow(`${p}.blend`, e.blend)) + spanRow(`${p}.span`, e) + kinds(`${p}.kinds`, e.kinds) + `</div>`;
      });
      h += `<h4>depth <select data-add="depth"><option value="">+ depth</option>${Object.keys(DEPTH_FIELDS).map(m => `<option>${m}</option>`).join("")}</select></h4>`;
      if (!src()?.depth && (lk.depth || []).length) h += `<div class="kkt-hint">connect a depth node to the tracker's 'depth' and run it: the depth part shows then</div>`;
      else if (!src()?.depth) h += `<div class="kkt-hint">layers in depth and time: connect a depth node (Depth Anything 3, DA3 Streaming, Marigold …) to the tracker's 'depth'</div>`;
      (lk.depth || []).forEach((e, i) => {
        const p = `depth.${i}`, F = DEPTH_FIELDS[e.type] || [];
        h += `<div class="kkt-card"><div class="kkt-row"><b>${esc({ layers: "depth layers", lines: "depth lines", time: "far = earlier", fog: "fog", slice: "depth slice" }[e.type] || e.type)}</b><span><button data-del="${p}">×</button></span></div>`;
        if (F.includes("count")) h += range(`${p}.count`, e.type === "lines" ? "lines" : "layers", 2, e.type === "lines" ? 40 : 16, 1, e.count ?? (e.type === "lines" ? 12 : 6));
        if (F.includes("near")) h += colourRow(`${p}.near`, e.near ?? "#f18a58").replace(">colour<", ">near<") + colourRow(`${p}.far`, e.far ?? "#a187b7").replace(">colour<", ">far<");
        if (F.includes("colour")) h += colourRow(`${p}.colour`, e.colour ?? (e.type === "fog" ? "#000000" : "#ffffff"));
        if (F.includes("width")) h += range(`${p}.width`, "line", 0.2, 6, 0.1, e.width ?? 0.8);
        if (F.includes("delay")) h += range(`${p}.delay`, "frames later", 0, 60, 1, e.delay ?? 4);
        if (F.includes("amount")) h += range(`${p}.amount`, "amount", 0, 1, 0.05, e.amount ?? 0.7);
        if (F.includes("from")) h += range(`${p}.from`, "from (far)", 0, 1, 0.01, e.from ?? 0.5) + range(`${p}.to`, "to (near)", 0, 1, 0.01, e.to ?? 1);
        if (F.includes("opacity")) h += range(`${p}.opacity`, "opacity", 0, 1, 0.05, e.opacity ?? (e.type === "lines" ? 0.7 : 0.6));
        if (["layers", "lines", "fog"].includes(e.type)) h += blendRow(`${p}.blend`, e.blend);
        h += spanRow(`${p}.span`, e) + `</div>`;
      });
      const sp = lk.stamp || {};
      h += `<h4>stamp trail ${toggle("stamp.on", sp.on ? "on" : "off", sp.on)}</h4>`;
      if (sp.on) h += `<div class="kkt-card">${range("stamp.every", "every", 2, 120, 1, sp.every ?? 14)}${pills("stamp.size", "size", ["blob", "fixed"], sp.size || "blob")}${sp.size === "fixed" ? range("stamp.fixed", "px", 4, 300, 1, sp.fixed ?? 40) : ""}
        ${pills("stamp.patch", "pixels", ["now", "first"], sp.patch || "now")}${blendRow("stamp.blend", sp.blend)}${spanRow("stamp.span", sp)}${range("stamp.stay", "stays (s)", 0, 30, 0.5, sp.stay ?? 0)}<div class="kkt-row"><span class="k"></span>${toggle("stamp.outline", "thin frame", sp.outline)}</div>${kinds("stamp.kinds", sp.kinds)}</div>`;
      const so = lk.sound || {};
      h += `<h4>sound</h4>`;
      if (st.data?.curves?.beat?.length) h += `<div class="kkt-card">${range("sound.pulse", "beat → marks pulse", 0, 2, 0.05, so.pulse ?? 0.5)}${range("sound.flash", "beat → lines flash", 0, 1, 0.05, so.flash ?? 0.4)}
        ${range("sound.reach", "loud → more lines", 0, 2, 0.05, so.reach ?? 0.5)}${range("sound.width", "loud → thicker", 0, 3, 0.05, so.width ?? 0)}</div>`;
      else h += `<div class="kkt-hint">connect a sound (Load Audio, the video's sound, kubakub sound) to the tracker's 'audio' and run it: the marks pulse and the lines flash on the beat</div>`;
      const ss = lk.states || {}, words = ss.words?.length ? ss.words : STATE_WORDS;
      h += `<h4>state words</h4><div class="kkt-card">${["new", "moving", "still", "lost"].map((n, i) => `<div class="kkt-row"><span class="k">${n}</span><input type="text" data-word="${i}" value="${esc(words[i] ?? "")}"></div>`).join("")}
        ${range("states.moving", "moving above", 0.1, 10, 0.1, ss.moving ?? 1.2)}${range("states.new", "new for", 0, 60, 1, ss.new ?? 8)}${range("states.lost", "lost for", 0, 120, 1, ss.lost ?? 24)}</div>`;
      side.innerHTML = h;
    }
    const pathSet = (o, p, v) => { const ks = p.split("."); let q = o; for (let i = 0; i < ks.length - 1; i++) q = q[ks[i]] ??= {}; q[ks[ks.length - 1]] = v; };
    const pathGet = (o, p) => p.split(".").reduce((q, k) => q?.[k], o);
    const num = v => (v !== "" && !isNaN(+v) && !/^#/.test(v) ? +v : v);
    side.addEventListener("click", e => {
      const t = e.target.closest("button"); if (!t) return; e.stopPropagation(); const lk = getLook();
      if ("close" in t.dataset) { close(); return; }
      if ("render" in t.dataset) { renderLook(); return; }
      if (t.dataset.preset) { setLook(lookOf(t.dataset.preset, SERVER?.looks, SERVER?.default_look)); renderSide(); return; }
      if (t.dataset.set) pathSet(lk, t.dataset.set, num(t.dataset.val));
      else if (t.dataset.tog) pathSet(lk, t.dataset.tog, !pathGet(lk, t.dataset.tog));
      else if (t.dataset.kinds) { const v = t.dataset.val, cur = new Set(pathGet(lk, t.dataset.kinds) || []); if (!v) cur.clear(); else cur.has(v) ? cur.delete(v) : cur.add(v); pathSet(lk, t.dataset.kinds, [...cur]); }
      else if (t.dataset.del) { const [a, i] = t.dataset.del.split("."); lk[a].splice(+i, 1); }
      else if (t.dataset.up) { const [a, i] = t.dataset.up.split("."); const L = lk[a]; [L[+i - 1], L[+i]] = [L[+i], L[+i - 1]]; }
      else if (t.dataset.tok) { pathSet(lk, t.dataset.tok, (pathGet(lk, t.dataset.tok) || "") + (pathGet(lk, t.dataset.tok) ? " " : "") + t.dataset.val); }
      setLook(lk); renderSide();
    });
    let spanDrag = null;
    side.addEventListener("pointerdown", e => {
      const bar = e.target.closest(".kkt-span"); if (!bar) return; e.preventDefault(); e.stopPropagation();
      const lk = getLook(), path = bar.dataset.span, sp = [...(pathGet(lk, path) || [0, 1])], r = bar.getBoundingClientRect();
      const u = Math.min(1, Math.max(0, (e.clientX - r.left - 6) / Math.max(1, r.width - 12)));
      const end = e.target.dataset.end != null ? +e.target.dataset.end : (Math.abs(u - sp[0]) <= Math.abs(u - sp[1]) ? 0 : 1);
      spanDrag = { bar, path, end, r }; bar.setPointerCapture(e.pointerId); moveSpan(e);
    });
    const moveSpan = e => {
      if (!spanDrag) return; const { path, end, r } = spanDrag, lk = getLook(), sp = [...(pathGet(lk, path) || [0, 1])];
      const u = Math.round(Math.min(1, Math.max(0, (e.clientX - r.left - 6) / Math.max(1, r.width - 12))) * 1000) / 1000;
      sp[end] = end === 0 ? Math.min(u, sp[1]) : Math.max(u, sp[0]);
      pathSet(lk, path, sp[0] <= 0 && sp[1] >= 1 ? undefined : sp); if (sp[0] <= 0 && sp[1] >= 1) { const ks = path.split("."); let q = lk; for (const k of ks.slice(0, -1)) q = q[k]; delete q[ks[ks.length - 1]]; }
      player.toggle(false); player.t = Math.round(sp[end] * Math.max(1, (st.data?.n || 1) - 1)); player.preload();   // the preview jumps to the dot
      setLook(lk); renderSide();
      spanDrag.bar = side.querySelector(`[data-span="${path}"]`); spanDrag.r = spanDrag.bar?.getBoundingClientRect() || r;
    };
    side.addEventListener("pointermove", moveSpan);
    side.addEventListener("pointerup", () => { spanDrag = null; });
    side.addEventListener("input", e => {
      const lk = getLook();
      if (e.target.dataset.in) { pathSet(lk, e.target.dataset.in, e.target.type === "range" ? +e.target.value : e.target.value);
        if (e.target.type === "range") e.target.nextElementSibling.textContent = e.target.value; }
      else if (e.target.dataset.word) { const ws = [...(lk.states.words?.length ? lk.states.words : STATE_WORDS)]; ws[+e.target.dataset.word] = e.target.value; lk.states.words = ws; }
      else return;
      setLook(lk);
    });
    side.addEventListener("change", e => {
      const a = e.target.dataset.add; if (!a || !e.target.value) { if (e.target.type === "color") renderSide(); return; }
      const lk = getLook(), v = e.target.value;
      const fresh = a === "marks" ? { type: v, colour: "#ffffff", width: 1, ...(v === "label" ? { text: "{id}", size: 14 } : {}) } : a === "connect" ? { mode: v, colour: "#ffffff", width: 0.8, opacity: 0.7 }
        : a === "depth" ? { type: v, ...{ layers: { count: 6, near: "#f18a58", far: "#a187b7", opacity: 0.6 }, lines: { count: 12, colour: "#ffffff", width: 0.8, opacity: 0.7 }, time: { count: 6, delay: 4 }, fog: { colour: "#000000", amount: 0.7 }, slice: { from: 0.5, to: 1 } }[v] } : { type: v, shape: "box" };
      (lk[a] ||= []).push(fresh); setLook(lk); renderSide();
    });
    const keys = e => { if (!wrap.isConnected) return; if (e.target.matches?.("input[type=text]")) { e.stopPropagation(); return; }
      if (e.key === "Escape") { close(); e.stopPropagation(); } else if (e.key === " ") { player.toggle(!player.playing, fps()); e.preventDefault(); e.stopPropagation(); }
      else if (e.key === "," || e.key === ".") { player.toggle(false); player.t = Math.max(0, Math.min((player.side?.n || 1) - 1, player.t + (e.key === "." ? 1 : -1))); views.forEach(f => f()); e.stopPropagation(); } };
    window.addEventListener("keydown", keys, true);
    function close() { views.delete(view); wrap.remove(); window.removeEventListener("keydown", keys, true); views.forEach(f => f()); }
    document.body.appendChild(wrap); renderSide(); view();
  }
  el.kktLoad = () => { load().then(() => views.forEach(f => f())); renderPresets(); };
  el.kktOpen = openEditor;
  server().then(() => { renderPresets(); views.forEach(f => f()); });
  let seen = "";                                   // the tracker before ran again: load its new frames and tracks
  const watch = setInterval(() => { if (!el.isConnected) { clearInterval(watch); return; } const m = src(), k = (m?.tracks?.filename || "") + "|" + (W_("look")?.value || "");
    if (k !== seen) { seen = k; load(); views.forEach(f => f()); } }, 400);
  return el;
}

app.registerExtension({
  name: "kubakub.track",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    const kind = { KUBA_Tracker: "tracker", KUBA_TrackDraw: "draw" }[nodeData.name]; if (!kind) return;
    const hidden = kind === "tracker" ? ["settings"] : ["look"];
    const created = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = created?.apply(this, arguments);
      for (const n of hidden) hideWidget(this.widgets?.find(w => w.name === n));
      const el = kind === "tracker" ? trackerUI(this) : drawUI(this);
      const w = this.addDOMWidget("track_view", "kubakub_track", el, { serialize: false, hideOnZoom: false, getMinHeight: () => (kind === "tracker" ? 560 : 320) });
      w.serialize = false; this.kktView = el;
      const want = kind === "tracker" ? [480, 760] : [440, 440];
      if ((this.size?.[1] || 0) < want[1]) this.setSize?.([Math.max(this.size?.[0] || 0, want[0]), want[1]]);
      return r;
    };
    const configured = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function () {
      const r = configured?.apply(this, arguments);
      for (const n of hidden) hideWidget(this.widgets?.find(w => w.name === n));
      requestAnimationFrame(() => this.kktView?.kktLoad?.());
      return r;
    };
    if (kind === "tracker") {
      const executed = nodeType.prototype.onExecuted;
      nodeType.prototype.onExecuted = function (msg) {
        const r = executed?.apply(this, arguments);
        const m = msg?.kubakub_track?.[0]; if (m) { (this.properties ||= {}).kubakub_track = m; this.kktView?.kktLoad?.(); }
        return r;
      };
    }
  },
});
