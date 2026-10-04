// kubakub director: the layer window of the Kuba nodes (KUBA_Director).
// The window edits a JSON document stored in the node's "document" widget; the node renders it at full
// size in Python (kubakub/director/render.py). Plain ES module, no build step, no network: the font
// ships with the pack (web/fonts, OFL). Look: kubakub.art (Hanken Grotesk, #f18a58 actions, #a187b7 selection).
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const HERE = new URL(".", import.meta.url).href;
const OPS = { normal: "source-over", multiply: "multiply", screen: "screen", overlay: "overlay", darken: "darken",
  lighten: "lighten", "color dodge": "color-dodge", "color burn": "color-burn", "hard light": "hard-light",
  "soft light": "soft-light", difference: "difference", exclusion: "exclusion", hue: "hue", saturation: "saturation",
  color: "color", luminosity: "luminosity" };
const ORANGE = "#f18a58", LAV = "#a187b7";
const QUALITY = { fast: { px: 800, samples: 16 }, fine: { px: 1600, samples: 48 } };    // light previews
const LIGHT_DEF = {                                   // new lights: facade metres (x from the frame centre)
  point: { power: 400, color: "#ffb070", size: 0.3, distance: 1.5 },
  area: { power: 1500, color: "#ffd2a0", size: 4, distance: 6 },
  spot: { power: 3000, color: "#fff1dc", size: 0.2, angle: 35, distance: 8 },
  sun: { power: 3, color: "#fff4e0", azimuth: 30, elevation: 35 } };
// ---- timeline: keyframes per value path ("x", "adjust.hue", "light.lights.#p1.power" - '#id' picks from a list)
const ANIM_LAYER = ["x", "y", "w", "h", "rotation", "opacity", "visible"];
const ANIM_ADJUST = ["adjust.brightness", "adjust.contrast", "adjust.saturate", "adjust.hue", "adjust.sepia"];
const ANIM_RIG = ["light.env_strength", "light.env_rotation", "light.exposure", "light.projector.brightness"];
const ANIM_LAMP = ["x", "height", "distance", "power", "color", "size", "angle", "azimuth", "elevation", "on"];
const ANIM_GLOW = ["strength", "color", "on"];
// ---- layer effects (= kubakub/director/fx.py): blur -> sharpen -> glow -> grain, distances in delivery px
const FX_DEF = { blur: 0, sharpen: 0, sharpen_radius: 1.5, glow: 0, glow_radius: 20, glow_threshold: 0.6, grain: 0, grain_size: 1.5 };
const FX_UI = [["blur", "amount", 0, 100, 0.5, " px", "Soft focus (px of the delivery size)"],
  ["sharpen", "amount", 0, 3, 0.05, "", "Unsharp mask strength"], ["sharpen_radius", "radius", 0.3, 10, 0.1, " px", "Sharpen radius"],
  ["glow", "strength", 0, 3, 0.05, "", "The bright parts bloom outward"], ["glow_radius", "radius", 1, 200, 1, " px", "Glow size"],
  ["glow_threshold", "threshold", 0, 0.95, 0.01, "", "Only what is brighter than this glows"],
  ["grain", "amount", 0, 1, 0.01, "", "Film grain (new every frame)"], ["grain_size", "size", 0.5, 10, 0.1, " px", "Grain size"]];
const FX_GROUPS = [["blur", "Soft focus", ["blur"]], ["sharpen", "Unsharp mask", ["sharpen", "sharpen_radius"]],
  ["glow", "The bright parts bloom outward", ["glow", "glow_radius", "glow_threshold"]], ["grain", "Film grain, new every frame", ["grain", "grain_size"]]];
const fxOf = L => Object.assign({}, FX_DEF, L?.fx || {});
const fxActive = f => f.blur > 0.05 || f.sharpen > 0.001 || (f.glow > 0.001 && f.glow_radius > 0) || f.grain > 0.001;
function pathGet(o, path) {
  for (const s of path.split(".")) {
    if (o == null) return undefined;
    o = s[0] === "#" && Array.isArray(o) ? o.find(e => e && e.id === s.slice(1)) : o[s];
  }
  return o;
}
function pathSet(o, path, v) {
  const seg = path.split("."), last = seg.pop();
  for (const s of seg) { o = s[0] === "#" && Array.isArray(o) ? o.find(e => e && e.id === s.slice(1)) : o?.[s]; if (o == null) return false; }
  o[last] = v; return true;
}
const isHex = v => typeof v === "string" && /^#[0-9a-f]{6}$/i.test(v);
function mixHex(a, b, u) {
  const p = h => [1, 3, 5].map(i => parseInt(h.slice(i, i + 2), 16));
  const A = p(a), B = p(b); return "#" + A.map((x, i) => Math.round(x + (B[i] - x) * u).toString(16).padStart(2, "0")).join("");
}
function keyValue(keys, t) {                          // keys sorted by t: [{t, v, e: "smooth" | "linear" | "hold"}]
  if (!keys?.length) return undefined;
  if (t <= keys[0].t) return keys[0].v;
  const n = keys.length; if (t >= keys[n - 1].t) return keys[n - 1].v;
  let i = 0; while (i < n - 2 && keys[i + 1].t <= t) i++;
  const a = keys[i], b = keys[i + 1];
  if (a.e === "hold" || !(typeof a.v === "number" && typeof b.v === "number" || isHex(a.v) && isHex(b.v))) return a.v;
  let u = (t - a.t) / Math.max(1e-9, b.t - a.t); if (a.e !== "linear") u = u * u * (3 - 2 * u);
  return typeof a.v === "number" ? a.v + (b.v - a.v) * u : mixHex(a.v, b.v, u);
}
// ---- behaviours (= kubakub/director/motion.py, checked by tests/test_motion.py): procedural motion on top of the keyframes
const M_TYPES = ["wiggle", "oscillate", "drift", "random", "loop", "pulse", "audio", "stagger", "repeat"];
const LEVEL_RATE = 100;                               // loudness curve samples per second
function levelCurve(mono, sr) {                       // RMS per 1/100 s, / its 98th percentile (= motion.level_curve)
  const hop = sr / LEVEL_RATE, n = Math.floor(mono.length / hop); if (n < 1) return [];
  const out = new Float64Array(n);
  for (let i = 0; i < n; i++) { const a = Math.floor(i * hop), b = Math.floor((i + 1) * hop); let q = 0; for (let j = a; j < b; j++) q += mono[j] * mono[j]; out[i] = Math.sqrt(q / Math.max(1, b - a)); }
  const ref = Float64Array.from(out).sort()[Math.min(n - 1, Math.floor(0.98 * (n - 1)))];
  return ref <= 1e-9 ? new Array(n).fill(0) : Array.from(out, v => Math.min(1.5, v / ref));
}
function mLevel(ctx, t, smooth) {
  const lv = ctx?.level; if (!lv?.length) return 0;
  const i = Math.floor((t + (ctx.offset || 0)) * LEVEL_RATE), k = Math.max(0, Math.round(smooth * LEVEL_RATE));
  if (i < 0 || i - k >= lv.length) return 0;
  let best = 0;
  for (let j = Math.max(0, i - k), e = Math.min(i, lv.length - 1); j <= e; j++) { const v = lv[j] * (smooth > 0 ? Math.exp(-(i - j) / (smooth * LEVEL_RATE)) : 1); if (v > best) best = v; }
  return best;
}
function pulseEnv(b, dt) {
  const a = Math.max(0, mNum(b, "attack", 0.02)), d = Math.max(1e-3, mNum(b, "decay", 0.25));
  if (dt < 0) return 0;
  if (dt < a) { const u = dt / a; return u * u * (3 - 2 * u); }
  return Math.exp(-(dt - a) / d);
}
function mTriggers(b, ctx) {                          // trigger times of a pulse (sorted), or null for "every"
  const trig = b.trigger || "beats";
  if (trig === "every") return null;
  if (trig === "markers") return ctx?.markers || [];
  const nth = Math.max(1, Math.trunc(mNum(b, "nth", 1))) * (trig === "bars" ? 4 : 1);
  return (ctx?.beats || []).filter((_, i) => i % nth === 0);
}
function lastTrigger(b, ctx, t, s) {
  const ts = mTriggers(b, ctx);
  if (!ts) { const step = Math.max(0.02, mNum(b, "every", 0.5)); return t < s ? null : s + Math.floor((t - s) / step + 1e-9) * step; }
  let last = null;
  for (const x of ts) { if (x > t + 1e-9) break; if (x >= s - 1e-9) last = x; }
  return last;
}
function hash01(i, seed) {                            // integer hash -> [0, 1), bit-identical to motion.py
  let h = (Math.imul(i | 0, 374761393) + Math.imul(seed | 0, 668265263)) >>> 0;
  h = Math.imul(h ^ (h >>> 13), 1274126177) >>> 0; h = (h ^ (h >>> 16)) >>> 0; return h / 4294967296;
}
function vnoise(x, seed) { const i = Math.floor(x), f = x - i, a = hash01(i, seed) * 2 - 1, b = hash01(i + 1, seed) * 2 - 1, u = f * f * (3 - 2 * f); return a + (b - a) * u; }
const mNoise = (x, seed) => vnoise(x, seed) * 0.72 + vnoise(x * 2.03 + 17.1, seed + 7) * 0.28;
function mWave(kind, u) {
  const frac = u - Math.floor(u);
  if (kind === "triangle") { const v = u + 0.25; return 1 - 4 * Math.abs((v - Math.floor(v)) - 0.5); }
  if (kind === "square") return frac < 0.5 ? 1 : -1;
  if (kind === "saw") return 2 * frac - 1;
  return Math.sin(2 * Math.PI * u);
}
const M_DEF = { freq: 1, seed: 1, period: 2, every: 0.5, attack: 0.02, decay: 0.25, smooth: 0.15, nth: 1, step: 0.1, hold: 0.3,
  density: 0.5, count: 1, dx: 100, cols: 4, radius: 200 };                           // = the defaults the engine reads with
const mDefault = (b, k) => k === "fade" ? (b.type === "stagger" ? 0.2 : 0) : k === "dy" ? ((b.layout || "line") === "grid" ? 100 : 0) : (M_DEF[k] ?? 0);
const mNum = (b, k, d) => { const v = b[k]; return typeof v === "number" && Number.isFinite(v) ? v : d; };
function mSpan(b, dur) { const s = Math.max(0, mNum(b, "t_start", 0)), e = mNum(b, "t_end", -1); return [s, e < 0 ? dur : Math.max(s, e)]; }
function mEnvelope(b, t, dur) {
  const [s, e] = mSpan(b, dur); if (t < s || t > e) return 0;
  const f = Math.max(0, mNum(b, "fade", 0)); if (f <= 0) return 1;
  const u = Math.min(1, (t - s) / f, (e - t) / f); return u * u * (3 - 2 * u);
}
function mVec(b, v) { const a = mNum(b, "angle", 0) * Math.PI / 180; return [v * Math.cos(a), v * Math.sin(a)]; }
function mOne(b, t, dur, ctx) {                            // one behaviour at t -> [[path, add]] ("position" split into x / y)
  const typ = b.type, path = String(b.path || "");
  if (!path || !M_TYPES.includes(typ) || typ === "loop" || b.on === false) return [];
  const [s, e] = mSpan(b, dur);
  if (typ === "drift") {
    if (t < s) return [];
    const v = mNum(b, "rate", 0) * (Math.min(t, e) - s);
    if (path === "position") { const [x, y] = mVec(b, v); return [["x", x], ["y", y]]; }
    return [[path, v]];
  }
  const env = mEnvelope(b, t, dur); if (env <= 0) return [];
  const tl = t - s, amt = mNum(b, "amount", 0) * env, seed = Math.trunc(mNum(b, "seed", 1));
  if (typ === "pulse" || typ === "audio") {
    let v;
    if (typ === "pulse") { const lt = lastTrigger(b, ctx, t, s); v = lt === null ? 0 : amt * pulseEnv(b, t - lt); }
    else v = amt * mLevel(ctx, t, Math.max(0, mNum(b, "smooth", 0.15)));
    if (path === "position") { const [x, y] = mVec(b, v); return [["x", x], ["y", y]]; }
    return [[path, v]];
  }
  if (typ === "wiggle") {
    const fr = mNum(b, "freq", 1);
    return path === "position" ? [["x", amt * mNoise(tl * fr, seed)], ["y", amt * mNoise(tl * fr, seed + 101)]] : [[path, amt * mNoise(tl * fr, seed)]];
  }
  if (typ === "random") {
    const k = Math.floor(tl / Math.max(1e-3, mNum(b, "every", 0.5)));
    return path === "position" ? [["x", amt * (hash01(k, seed) * 2 - 1)], ["y", amt * (hash01(k, seed + 101) * 2 - 1)]] : [[path, amt * (hash01(k, seed) * 2 - 1)]];
  }
  const u = tl / Math.max(1e-3, mNum(b, "period", 2)) + mNum(b, "phase", 0), kind = b.wave || "sine";
  if (path === "position") {
    if (kind === "circle") return [["x", amt * Math.cos(2 * Math.PI * u)], ["y", amt * Math.sin(2 * Math.PI * u)]];
    const [x, y] = mVec(b, amt * mWave(kind, u)); return [["x", x], ["y", y]];
  }
  return [[path, amt * mWave(kind, u)]];
}
function motionOffsets(motion, t, dur, ctx) {
  const out = {};
  for (const b of motion || []) if (b && typeof b === "object") for (const [p, v] of mOne(b, t, dur, ctx)) out[p] = (out[p] || 0) + v;
  return out;
}
function loopModes(motion) {
  const out = {};
  for (const b of motion || []) if (b && b.type === "loop" && b.on !== false && ["cycle", "pingpong", "continue"].includes(b.mode))
    for (const p of b.path === "position" ? ["x", "y"] : [String(b.path || "")]) out[p] = b.mode;
  return out;
}
function loopTime(keys, t, mode) {
  const ts = (keys || []).filter(k => k && typeof k.t === "number").map(k => k.t).sort((a, b) => a - b);
  if (ts.length < 2 || t <= ts[ts.length - 1] || !(mode === "cycle" || mode === "pingpong")) return t;
  const t0 = ts[0], sp = ts[ts.length - 1] - t0; if (sp <= 1e-6) return t;
  const n = Math.floor((t - t0) / sp); let r = (t - t0) - n * sp;
  if (mode === "pingpong" && n % 2 === 1) r = sp - r;
  return t0 + r;
}
function keyedValue(keys, t, mode) {                  // keys sorted by t
  if (mode === "continue" && keys?.length >= 2 && t > keys[keys.length - 1].t) {
    const a = keys[keys.length - 2], b = keys[keys.length - 1];
    if (typeof a.v === "number" && typeof b.v === "number") return b.v + (b.v - a.v) / Math.max(1e-9, b.t - a.t) * (t - b.t);
  }
  return keyValue(keys, mode ? loopTime(keys, t, mode) : t);
}
// stagger (= motion.stagger_order / stagger_weights): the clip regions of a layer light up one after another
const STAGGER_ORDERS = ["left", "right", "top", "bottom", "centre", "outside", "size", "random"];
function staggerOrder(regs, order, W, H, seed) {      // regs [[id, [x, y, w, h]]] canvas px -> ids in order
  const key = ([rid, [x, y, w, h]]) => {
    const cx = x + w / 2, cy = y + h / 2, d2 = (cx - W / 2) ** 2 + (cy - H / 2) ** 2;
    return ({ right: [-cx, cy, rid], top: [cy, cx, rid], bottom: [-cy, cx, rid], centre: [d2, 0, rid], outside: [-d2, 0, rid],
      size: [-(w * h), 0, rid], random: [hash01(rid, seed), 0, rid] })[order] || [cx, cy, rid];
  };
  const ks = regs.map(r => [key(r), r[0]]);
  ks.sort((a, b) => { for (let i = 0; i < 3; i++) if (a[0][i] !== b[0][i]) return a[0][i] < b[0][i] ? -1 : 1; return 0; });
  return ks.map(k => k[1]);
}
const mRamp = u => { u = Math.min(1, Math.max(0, u)); return u * u * (3 - 2 * u); };
function staggerWeights(motion, t, dur, regs, W, H) {  // {id: 0-1} of the active stagger behaviours (multiplied), or null
  let out = null;
  for (const b of motion || []) {
    if (!b || b.type !== "stagger" || b.on === false || !regs.length) continue;
    const [s, e] = mSpan(b, dur), seed = Math.trunc(mNum(b, "seed", 1));
    const ids = staggerOrder(regs, b.order || "left", W, H, seed), n = ids.length, mode = b.mode || "sequence";
    const step = Math.max(0, mNum(b, "step", 0.1)), fade = Math.max(0, mNum(b, "fade", 0.2)), tl = Math.min(t, e) - s, w = {};
    ids.forEach((rid, r) => {
      let v;
      if (mode === "sequence") { const u = tl - r * step; v = fade > 0 ? mRamp(u / fade) : (u >= 0 ? 1 : 0); if (b.direction === "out") v = 1 - v; }
      else if (t < s || t > e) v = 1;
      else if (mode === "chase") {
        const hold = Math.max(0, mNum(b, "hold", 0.3)), cyc = Math.max(n * step, hold + 2 * fade, 1e-3); let loc = tl - r * step;
        if (b.loop !== false) loc = loc - Math.floor(loc / cyc) * cyc;
        v = loc < 0 ? 0 : loc < fade ? mRamp(loc / fade) : loc < fade + hold ? 1 : fade > 0 ? 1 - mRamp((loc - fade - hold) / fade) : 0;
      } else if (mode === "wave") v = 0.5 + 0.5 * Math.sin(2 * Math.PI * (tl - r * step) / Math.max(1e-3, mNum(b, "period", 2)));
      else { const k = Math.floor(tl / Math.max(1e-3, step > 0 ? step : 0.1)); v = hash01(k, seed + 7919 * rid) < mNum(b, "density", 0.5) ? 1 : 0; }
      w[rid] = v;
    });
    if (out) for (const k of Object.keys(w)) w[k] *= out[k] ?? 1;
    out = w;
  }
  return out;
}
// repeater (= motion.repeat_of / repeat_offsets / repeat_apply): copies of a box layer
const repeatOf = motion => (motion || []).find(b => b && b.type === "repeat" && b.on !== false && Math.trunc(mNum(b, "count", 1)) > 1) || null;
const repeatCount = b => Math.max(1, Math.min(200, Math.trunc(mNum(b, "count", 1))));
function repeatOffsets(b, k) {                        // copy k -> [dx, dy, drotation, scale factor, dopacity]
  const lay = b.layout || "line"; let dx, dy, drot;
  if (lay === "radial") {
    const n = repeatCount(b), r = mNum(b, "radius", 200), a0 = mNum(b, "start", 0) * Math.PI / 180, a = a0 + 2 * Math.PI * k / n;
    dx = r * (Math.cos(a) - Math.cos(a0)); dy = r * (Math.sin(a) - Math.sin(a0)); drot = b.orient ? (a - a0) * 180 / Math.PI : 0;
  } else if (lay === "grid") { const cols = Math.max(1, Math.trunc(mNum(b, "cols", 4))); dx = (k % cols) * mNum(b, "dx", 100); dy = Math.floor(k / cols) * mNum(b, "dy", 100); drot = 0; }
  else { dx = k * mNum(b, "dx", 100); dy = k * mNum(b, "dy", 0); drot = 0; }
  return [dx, dy, drot + k * mNum(b, "rotate", 0), Math.max(0.01, 1 + k * mNum(b, "scale", 0) / 100), k * mNum(b, "opacity", 0)];
}
function repeatApply(c, b, k) {
  const [dx, dy, dr, f, dop] = repeatOffsets(b, k), w0 = c.w || 0, h0 = c.h || 0, w = w0 * f, h = h0 * f;
  c.x = (c.x || 0) + dx - (w - w0) / 2; c.y = (c.y || 0) + dy - (h - h0) / 2; c.w = Math.max(1, w); c.h = Math.max(1, h);
  c.rotation = (c.rotation || 0) + dr; c.opacity = Math.min(1, Math.max(0, (c.opacity ?? 1) + dop));
  return c;
}
function mClamp(p, v) {
  if (p === "opacity") return Math.min(1, Math.max(0, v));
  if (p === "w" || p === "h") return Math.max(1, v);
  if (p.startsWith("fx.") || [".power", ".size", ".feather", "env_strength", ".brightness", ".strength", ".distance"].some(e => p.endsWith(e))) return Math.max(0, v);
  return v;
}
function composeMotion(offs, get) {                   // offsets onto base values -> {path: value}; scale about the box centre
  const out = {}, num = v => typeof v === "number";
  for (const [p, a] of Object.entries(offs)) { if (p === "scale") continue; const b = get(p); if (num(b)) out[p] = mClamp(p, b + a); }
  const sc = offs.scale;
  if (sc) {
    const v = Object.fromEntries(["x", "y", "w", "h"].map(k => [k, k in out ? out[k] : get(k)]));
    if (Object.values(v).every(num)) {
      const f = Math.max(0.01, 1 + sc / 100), w = v.w * f, h = v.h * f;
      Object.assign(out, { x: v.x - (w - v.w) / 2, y: v.y - (h - v.h) / 2, w: Math.max(1, w), h: Math.max(1, h) });
    }
  }
  return out;
}
const M_PRESETS = [                                   // [label, needs a box, behaviour]
  ["wiggle position", true, { type: "wiggle", path: "position", amount: 20, freq: 1, seed: 1 }],
  ["wiggle rotation", true, { type: "wiggle", path: "rotation", amount: 5, freq: 1, seed: 2 }],
  ["breathe (scale)", true, { type: "oscillate", path: "scale", amount: 5, period: 2, phase: 0, wave: "sine" }],
  ["orbit", true, { type: "oscillate", path: "position", amount: 40, period: 4, phase: 0, wave: "circle", angle: 0 }],
  ["sway", true, { type: "oscillate", path: "rotation", amount: 6, period: 3, phase: 0, wave: "sine" }],
  ["spin", true, { type: "drift", path: "rotation", rate: 30 }],
  ["drift", true, { type: "drift", path: "position", rate: 50, angle: 0 }],
  ["flicker (opacity)", false, { type: "random", path: "opacity", amount: 0.6, every: 0.08, seed: 3 }],
  ["pulse (opacity)", false, { type: "oscillate", path: "opacity", amount: 0.5, period: 1, phase: 0, wave: "square" }],
  ["wiggle …", false, { type: "wiggle", amount: 1, freq: 1, seed: 1 }],
  ["oscillate …", false, { type: "oscillate", amount: 1, period: 2, phase: 0, wave: "sine" }],
  ["drift …", false, { type: "drift", rate: 1 }],
  ["random steps …", false, { type: "random", amount: 1, every: 0.5, seed: 1 }],
  ["pulse on beats (scale)", true, { type: "pulse", path: "scale", amount: 8, trigger: "beats", nth: 1, attack: 0.02, decay: 0.2 }],
  ["kick on markers (up)", true, { type: "pulse", path: "position", amount: 30, angle: -90, trigger: "markers", nth: 1, attack: 0.03, decay: 0.3 }],
  ["flash on bars (glow)", false, { type: "pulse", path: "fx.glow", amount: 1.5, trigger: "bars", nth: 1, attack: 0.01, decay: 0.35 }],
  ["sound → scale", true, { type: "audio", path: "scale", amount: 15, smooth: 0.15 }],
  ["sound → opacity", false, { type: "audio", path: "opacity", amount: 1, smooth: 0.1 }],
  ["pulse …", false, { type: "pulse", amount: 1, trigger: "beats", nth: 1, attack: 0.02, decay: 0.25 }],
  ["sound level …", false, { type: "audio", amount: 1, smooth: 0.15 }],
  ["loop keyframes …", false, { type: "loop", mode: "cycle" }],
  ["repeat in a row", true, { type: "repeat", path: "copies", count: 5, layout: "line", dx: 250, dy: 0, rotate: 0, scale: 0, opacity: 0, delay: 0 }],
  ["repeat as a grid", true, { type: "repeat", path: "copies", count: 12, layout: "grid", cols: 4, dx: 250, dy: 250, rotate: 0, scale: 0, opacity: 0, delay: 0 }],
  ["repeat in a ring", true, { type: "repeat", path: "copies", count: 8, layout: "radial", radius: 400, start: 0, orient: true, rotate: 0, scale: 0, opacity: 0, delay: 0 }],
  ["echo trail (delayed copies)", true, { type: "repeat", path: "copies", count: 6, layout: "line", dx: 0, dy: 0, rotate: 0, scale: 0, opacity: -0.15, delay: 0.08 }],
  ["stagger: regions on in order", false, { type: "stagger", path: "regions", mode: "sequence", order: "left", step: 0.08, fade: 0.25, direction: "in" }],
  ["stagger: chase (running light)", false, { type: "stagger", path: "regions", mode: "chase", order: "left", step: 0.1, fade: 0.08, hold: 0.3 }],
  ["stagger: wave across", false, { type: "stagger", path: "regions", mode: "wave", order: "centre", step: 0.05, period: 2 }],
  ["stagger: random flicker", false, { type: "stagger", path: "regions", mode: "random", order: "left", step: 0.12, density: 0.5, seed: 1 }]];
const M_PARAMS = { wiggle: [["amount", "amount"], ["freq", "per second"], ["seed", "seed"]], oscillate: [["amount", "amount"], ["period", "period s"], ["phase", "phase 0-1"]],
  drift: [["rate", "per second"]], random: [["amount", "amount"], ["every", "every s"], ["seed", "seed"]], loop: [],
  pulse: [["amount", "amount"], ["attack", "attack s"], ["decay", "decay s"]], audio: [["amount", "amount"], ["smooth", "release s"]],
  stagger: [["step", "step s"], ["fade", "fade s"]],
  repeat: [["count", "copies"], ["rotate", "rotate °/copy"], ["scale", "scale %/copy"], ["opacity", "opacity/copy"], ["delay", "delay s/copy"]] };
// ---- audio: beats from a mono signal (onset envelope at 100 Hz, tempo by autocorrelation, phase by the best fit)
function analyzeBeats(x, sr) {
  const hop = Math.max(1, Math.round(sr / 100)), n = Math.floor(x.length / hop);
  if (n < 200) return null;
  const env = new Float32Array(n);
  let prev = 0;
  for (let i = 0; i < n; i++) {                       // energy of the pre-emphasised signal (kick and snare edges)
    let e = 0;
    for (let j = i * hop, end = j + hop; j < end; j++) { const y = x[j] - 0.97 * prev; prev = x[j]; e += y * y; }
    env[i] = Math.log(1e-10 + e);
  }
  const on = new Float32Array(n);
  for (let i = 1; i < n; i++) on[i] = Math.max(0, env[i] - env[i - 1]);
  const w = 25, cs = new Float64Array(n + 1);        // minus the local mean (0.5 s): only real jumps stay
  for (let i = 0; i < n; i++) cs[i + 1] = cs[i] + on[i];
  const od = new Float32Array(n);
  for (let i = 0; i < n; i++) { const a = Math.max(0, i - w), b = Math.min(n, i + w + 1); od[i] = Math.max(0, on[i] - (cs[b] - cs[a]) / (b - a)); }
  const ac = lag => { let s = 0; for (let i = 0; i + lag < n; i++) s += od[i] * od[i + lag]; return s / (n - lag); };
  let best = 0, bestLag = 50;
  const score = new Map();
  for (let lag = 33; lag <= 100; lag++) {            // 180 .. 60 BPM, a soft preference around 120
    const bpm = 6000 / lag, wgt = Math.exp(-0.5 * (Math.log2(bpm / 120) / 1.0) ** 2);
    const s = (ac(lag) + 0.5 * ac(2 * lag)) * wgt; score.set(lag, s);
    if (s > best) { best = s; bestLag = lag; }
  }
  const s0 = score.get(bestLag - 1) ?? best, s2 = score.get(bestLag + 1) ?? best, den = s0 - 2 * best + s2;
  let period = bestLag + (den < 0 ? 0.5 * (s0 - s2) / den : 0);                // parabolic refinement
  const at = t => { const i = Math.round(t); return Math.max(od[i - 1] || 0, od[i] || 0, od[i + 1] || 0); };   // ±1 frame
  const phaseOf = per => { let ph = 0, b = -1;
    for (let p = 0; p < per; p++) { let s = 0; for (let t = p; t < n; t += per) s += at(t); if (s > b) { b = s; ph = p; } } return ph; };
  let ph = phaseOf(period);
  // off-beats as strong as the beats: the real tempo is twice as fast (174 found as 87)
  let onB = 0, offB = 0; for (let t = ph; t + period / 2 < n; t += period) { onB += at(t); offB += at(t + period / 2); }
  if (offB > 0.6 * onB && 6000 / (period / 2) <= 200) { period /= 2; ph = phaseOf(period); }
  for (let it = 0; it < 3; it++) {                   // snap each beat to its onset peak, fit phase + period (least squares)
    const pts = [], r = Math.max(2, Math.round(period * 0.15));
    for (let k = 0, f = ph; f < n; k++, f = ph + k * period) {
      let bi = -1, bv = 0; for (let j = Math.max(0, Math.round(f) - r); j <= Math.min(n - 1, Math.round(f) + r); j++) if (od[j] > bv) { bv = od[j]; bi = j; }
      if (bi >= 0) pts.push([k, bi, bv]);
    }
    if (pts.length < 4) break;
    let sw = 0, sk = 0, sf = 0, skk = 0, skf = 0;
    for (const [k, f, v] of pts) { sw += v; sk += v * k; sf += v * f; skk += v * k * k; skf += v * k * f; }
    const d = sw * skk - sk * sk; if (Math.abs(d) < 1e-9) break;
    period = (sw * skf - sk * sf) / d; ph = (sf - period * sk) / sw;
  }
  ph = ((ph % period) + period) % period;
  return { bpm: Math.round(60000 / period) / 10, phase: Math.round(ph) / 100, confidence: best > 0 ? Math.min(1, best / (ac(0) || 1) * 4) : 0 };
}
function beatTimes(beat, offset, duration) {         // beats in timeline time (the file starts `offset` s before 0)
  if (!beat?.bpm) return [];
  const p = 60 / beat.bpm; let t = ((beat.phase - offset) % p + p) % p; const out = [];
  for (let k = 0; t <= duration + 1e-9 && out.length < 20000; k++, t += p) out.push(t);
  return out;
}
const stable = v => Array.isArray(v) ? `[${v.map(stable).join(",")}]` : v && typeof v === "object"
  ? `{${Object.keys(v).sort().map(k => JSON.stringify(k) + ":" + stable(v[k])).join(",")}}` : JSON.stringify(v ?? null);

// ------------------------------------------------------------------------------------------------ style
const CSS = `
@font-face{font-family:"Hanken Grotesk KKD";src:url("${HERE}fonts/HankenGrotesk-latin-var.woff2") format("woff2");font-weight:200 800;font-display:swap}
.kkd-root,.kkd-menu,.kkd-pop,.kkd-tip,.kkd-modal,.kkd-dropzone,.kkd-note{--paper:#ffffff;--paper2:#f6f4f3;--ink:rgba(0,0,0,.85);--muted:rgba(0,0,0,.55);--faint:rgba(0,0,0,.25);--line:rgba(0,0,0,.12);
  --orange:${ORANGE};--orange-ink:#1d0f07;--lav:${LAV};--lav-soft:rgba(161,135,183,.18);--stage:#3b3b3e;--stage-edge:#303033;--tip:#111;--tip-ink:#fff;
  color:var(--ink);font:400 14px/1.35 "Hanken Grotesk KKD","Segoe UI",system-ui,sans-serif}
.kkd-root{position:fixed;inset:0;z-index:10000;background:var(--paper);
  display:grid;grid-template-columns:48px minmax(0,1fr) var(--kkd-sw,clamp(300px,24vw,420px));grid-template-rows:auto minmax(0,1fr) auto;gap:10px;padding:12px 16px;box-sizing:border-box}
.kkd-tl{grid-column:2/3;grid-row:3;display:flex;flex-direction:column;gap:4px;min-width:0}
.kkd-tlbar{display:flex;gap:8px;align-items:center;flex-wrap:wrap;font-size:12px;color:var(--muted)}
.kkd-tlbar input{width:52px!important;padding:2px 6px!important}
.kkd-tlc{width:100%;display:block;cursor:ew-resize}
.kkd-tlwrap{max-height:var(--kkd-th,140px);overflow-y:auto;overflow-x:hidden}
.kkd-split{position:relative;flex:0 0 auto;touch-action:none}.kkd-split::after{content:"";position:absolute;border-radius:2px;background:var(--line);transition:background .12s}
.kkd-split:hover::after,.kkd-split.drag::after{background:var(--lav)}
.kkd-split.h{height:10px;cursor:row-resize}.kkd-split.h::after{left:calc(50% - 18px);width:36px;top:4px;height:2px}
.kkd-split.v{position:absolute;left:-10px;top:0;bottom:0;width:10px;cursor:col-resize}.kkd-split.v::after{top:0;bottom:0;left:4px;width:2px;background:transparent}
.kkd-tl>.kkd-split.h{margin:-8px 0 -2px}
.kkd-rec{width:26px;height:26px;border-radius:50%;border:1px solid var(--line);background:transparent;cursor:pointer;display:grid;place-items:center;padding:0}
.kkd-rec::after{content:"";width:10px;height:10px;border-radius:50%;background:var(--faint)}
.kkd-rec[aria-pressed="true"]{border-color:var(--orange)}.kkd-rec[aria-pressed="true"]::after{background:var(--orange)}
@media (prefers-color-scheme:dark){.kkd-root,.kkd-menu,.kkd-pop,.kkd-tip,.kkd-modal,.kkd-dropzone,.kkd-note{color-scheme:dark;--paper:#141414;--paper2:#1c1c1d;--ink:rgba(255,255,255,.88);--muted:rgba(255,255,255,.55);
  --faint:rgba(255,255,255,.25);--line:rgba(255,255,255,.12);--lav-soft:rgba(161,135,183,.22);--stage:#2c2c2e;--stage-edge:#222224;--tip:#f4f1ee;--tip-ink:#141414}}
.kkd-root *{box-sizing:border-box}
.kkd-root [hidden],.kkd-menu[hidden],.kkd-pop[hidden],.kkd-dropzone[hidden],.kkd-note[hidden]{display:none!important}
.kkd-root button,.kkd-root input,.kkd-root select,.kkd-root textarea,.kkd-menu button,.kkd-pop button,.kkd-pop input,.kkd-pop select,.kkd-pop textarea{font:inherit;color:inherit}
.kkd-bar{grid-column:1/-1;display:flex;flex-wrap:wrap;gap:8px;align-items:center}
.kkd-logo{font-size:22px;letter-spacing:-.01em;margin-right:10px;white-space:nowrap;text-transform:lowercase}
.kkd-logo b{font-weight:800}.kkd-logo span{font-weight:200;color:var(--lav)}
.kkd-logo i{display:inline-block;width:8px;height:8px;border-radius:50%;background:var(--orange);margin-left:3px;vertical-align:2px}
.kkd-grow{flex:1}
.kkd-pill.ico{padding:4px 9px;display:inline-grid;place-items:center}.kkd-pill.ico svg{width:16px;height:16px;stroke:currentColor;fill:none;stroke-width:1.6;stroke-linecap:round;stroke-linejoin:round}
@media (max-width:1440px){.kkd-stamp{display:none}.kkd-bar{gap:6px}.kkd-logo{margin-right:4px}}
.kkd-pill{border:1px solid var(--line);background:transparent;border-radius:10em;padding:5px 12px;cursor:pointer;text-transform:lowercase;white-space:nowrap}
.kkd-pill:hover{border-color:var(--faint)}
.kkd-pill[aria-pressed="true"]{background:var(--lav-soft);border-color:var(--lav)}
.kkd-pill.accent{background:var(--orange);border-color:var(--orange);color:var(--orange-ink);font-weight:600}
.kkd-pill.lav{background:var(--lav);border-color:var(--lav);color:#fff;font-weight:600}
.kkd-pills{display:inline-flex;gap:4px;flex-wrap:wrap}
.kkd-root :focus-visible{outline:2px solid var(--orange);outline-offset:2px}
.kkd-tools{grid-column:1;grid-row:2;display:flex;flex-direction:column;gap:4px;align-items:center}
.kkd-tool{width:40px;height:40px;border-radius:10em;border:1px solid transparent;background:transparent;cursor:pointer;display:grid;place-items:center;color:var(--muted)}
.kkd-tool svg{width:19px;height:19px;stroke:currentColor;fill:none;stroke-width:1.5;stroke-linecap:round;stroke-linejoin:round}
.kkd-tool:hover{color:var(--ink);border-color:var(--line)}
.kkd-tool[aria-pressed="true"]{background:var(--orange);color:var(--orange-ink);border-color:var(--orange)}
.kkd-tool:disabled{opacity:.35;cursor:not-allowed}
.kkd-sep{width:22px;height:1px;background:var(--line);margin:4px 0}
.kkd-stage{grid-column:2;grid-row:2;position:relative;background:var(--stage);border:1px solid var(--stage-edge);overflow:hidden;min-height:0}
.kkd-stage canvas{width:100%;height:100%;display:block;touch-action:none}
.kkd-hud{position:absolute;left:10px;bottom:10px;font-size:12px;color:#e8e6e3;background:rgba(0,0,0,.45);border:1px solid rgba(161,135,183,.6);padding:3px 9px;border-radius:10em;font-variant-numeric:tabular-nums;pointer-events:none}
.kkd-brush{position:absolute;border:1px solid rgba(255,255,255,.9);box-shadow:0 0 0 1px rgba(0,0,0,.5);border-radius:50%;pointer-events:none;transform:translate(-50%,-50%)}
.kkd-busy{position:absolute;inset:0;display:grid;place-items:center;pointer-events:none;color:#eee;font-size:18px;font-weight:200;background:rgba(0,0,0,.35);text-transform:lowercase}
.kkd-side{display:flex;flex-direction:column;gap:6px;min-height:0;position:relative;grid-column:3;grid-row:2/4}
.kkd-side>[data-id=insp]{flex:1 1 auto;min-height:0;overflow:auto;padding-right:4px}
.kkd-side .kkd-layers{max-height:var(--kkd-lh,34vh);overflow:auto;padding-right:2px}
.kkd-pmhead{cursor:pointer}.kkd-pmhead .kkd-chev{display:inline-block;transition:transform .12s;color:var(--faint)}.kkd-pmhead[aria-expanded="true"] .kkd-chev{transform:rotate(90deg)}
.kkd-pmhead small{font-weight:400;letter-spacing:0;color:var(--faint);overflow:hidden;text-overflow:ellipsis;white-space:nowrap;min-width:0}
.kkd-sec h2{font-size:12px;font-weight:600;letter-spacing:.06em;text-transform:lowercase;color:var(--muted);margin:0 0 6px;display:flex;align-items:center;gap:6px}
.kkd-layers{display:flex;flex-direction:column;gap:2px;position:relative}
.kkd-layer{display:grid;grid-template-columns:14px 20px 34px minmax(0,1fr) auto;gap:7px;align-items:center;padding:5px 6px;border-radius:8px;border:1px solid transparent;cursor:pointer;user-select:none}
.kkd-layer:hover{background:var(--paper2)}
.kkd-layer[aria-selected="true"]{background:var(--lav-soft);border-color:var(--lav)}
.kkd-layer.dragging{opacity:.4}
.kkd-grip{cursor:grab;color:var(--faint);font-size:12px;letter-spacing:-2px}
.kkd-eye{width:18px;height:18px;border-radius:50%;border:1px solid var(--line);display:grid;place-items:center;background:transparent;cursor:pointer;padding:0}
.kkd-eye.on::after{content:"";width:8px;height:8px;border-radius:50%;background:var(--ink)}
.kkd-thumb{width:34px;height:23px;background:var(--stage);border:1px solid var(--line);display:block;object-fit:contain}
.kkd-lname{min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.kkd-lname small{display:block;color:var(--muted);font-size:11px;text-transform:lowercase}
.kkd-tag{font-size:11px;border-radius:10em;padding:1px 7px;white-space:nowrap;text-transform:lowercase}
.kkd-tag.front{background:rgba(241,138,88,.18)}.kkd-tag.behind{background:var(--lav);color:#fff}.kkd-tag.wall{border:1px solid var(--line);color:var(--muted)}
.kkd-drop{position:absolute;left:4px;right:4px;height:2px;background:var(--orange);border-radius:2px;pointer-events:none}
.kkd-field{display:grid;grid-template-columns:78px minmax(0,1fr) auto;gap:8px;align-items:center;margin:5px 0}
.kkd-field>span:first-child{color:var(--muted);font-size:12px;text-transform:lowercase}
.kkd-field .v{font-size:12px;min-width:34px;text-align:right;font-variant-numeric:tabular-nums;color:var(--muted)}
.kkd-root input[type=text],.kkd-root input[type=number],.kkd-root select,.kkd-root textarea,.kkd-pop input[type=text],.kkd-pop select{width:100%;background:var(--paper);border:1px solid var(--line);border-radius:8px;padding:5px 8px}
.kkd-root textarea{resize:vertical;min-height:44px;grid-column:1/-1}
.kkd-root input[type=checkbox],.kkd-pop input[type=checkbox]{accent-color:var(--lav);width:15px;height:15px}
.kkd-root input[type=range],.kkd-pop input[type=range]{-webkit-appearance:none;appearance:none;width:100%;height:18px;background:transparent;margin:0;--p:0%}
.kkd-root input[type=range]::-webkit-slider-runnable-track,.kkd-pop input[type=range]::-webkit-slider-runnable-track{height:3px;border-radius:3px;background:linear-gradient(to right,var(--orange) var(--p),var(--faint) var(--p))}
.kkd-root input[type=range]::-webkit-slider-thumb,.kkd-pop input[type=range]::-webkit-slider-thumb{-webkit-appearance:none;width:14px;height:14px;margin-top:-5.5px;border-radius:50%;background:var(--orange);border:2px solid var(--paper);box-shadow:0 0 0 1px var(--orange)}
.kkd-root input[type=range]::-moz-range-track,.kkd-pop input[type=range]::-moz-range-track{height:3px;border-radius:3px;background:var(--faint)}
.kkd-root input[type=range]::-moz-range-progress,.kkd-pop input[type=range]::-moz-range-progress{height:3px;border-radius:3px;background:var(--orange)}
.kkd-root input[type=range]::-moz-range-thumb,.kkd-pop input[type=range]::-moz-range-thumb{width:12px;height:12px;border-radius:50%;background:var(--orange);border:2px solid var(--paper)}
.kkd-root input[type=range]:disabled{opacity:.4}
.kkd-root input[type=color]{width:34px;height:24px;border:1px solid var(--line);border-radius:10em;padding:0;background:none}
.kkd-seg{display:inline-flex;gap:3px;flex-wrap:wrap}.kkd-seg .kkd-pill{padding:3px 10px;font-size:13px}
.kkd-menu,.kkd-pop{position:fixed;z-index:10002;background:var(--paper);border:1px solid var(--line);border-radius:12px;padding:6px;box-shadow:0 12px 30px rgba(0,0,0,.18)}
.kkd-menu{min-width:220px}.kkd-pop{padding:14px;width:min(430px,calc(100vw - 32px));max-height:calc(100vh - 72px);overflow:auto}
.kkd-menu button{display:flex;justify-content:space-between;gap:16px;width:100%;text-align:left;background:transparent;border:0;padding:6px 9px;border-radius:8px;cursor:pointer;text-transform:lowercase}
.kkd-menu button:hover{background:var(--paper2)}
.kkd-menu button:disabled{opacity:.4;cursor:not-allowed}
.kkd-regpick{max-height:min(360px,50vh);overflow:auto;min-width:200px}
.kkd-regpick button{text-transform:none;font-size:12px;padding:4px 9px}
.kkd-regpick button.on{background:var(--lav-soft);font-weight:700}
.kkd-regpick button.hi{outline:1px solid var(--lav)}
.kkd-regpick .none{color:var(--muted);font-style:italic}
.kkd-pop h3{margin:0 0 8px;font-weight:800;font-size:15px;text-transform:lowercase}
.kkd-root kbd,.kkd-menu kbd,.kkd-pop kbd{font:600 11px inherit;border:1px solid var(--line);border-bottom-width:2px;border-radius:5px;padding:0 5px;color:var(--muted)}
.kkd-tip{position:fixed;z-index:10003;background:var(--tip);color:var(--tip-ink);padding:6px 10px;border-radius:10px;font-size:12px;max-width:280px;pointer-events:none;opacity:0;transition:opacity .12s}
.kkd-tip kbd{border-color:rgba(127,127,127,.5);color:inherit;margin-left:6px}
.kkd-keys{display:grid;grid-template-columns:auto 1fr;gap:5px 14px;font-size:13px}.kkd-keys span:nth-child(odd){text-align:right;white-space:nowrap}
.kkd-outs{display:flex;flex-wrap:wrap;gap:5px}.kkd-outs span{font-size:12px;border-radius:10em;padding:3px 9px;background:var(--paper2)}
.kkd-dropzone{position:fixed;inset:12px;z-index:10001;border:2px dashed var(--lav);border-radius:18px;background:rgba(161,135,183,.16);display:grid;place-items:center;font-size:20px;font-weight:200;text-transform:lowercase;pointer-events:none}
.kkd-modal{position:fixed;inset:0;z-index:10005;background:rgba(0,0,0,.35);display:grid;place-items:center}
.kkd-modal>div{background:var(--paper,#fff);color:var(--ink,#111);border-radius:16px;padding:18px 20px;box-shadow:0 20px 50px rgba(0,0,0,.3);width:min(440px,calc(100vw - 32px));font:400 14px "Hanken Grotesk KKD",system-ui,sans-serif}
.kkd-modal h3{margin:0 0 6px;font-weight:800;font-size:16px;text-transform:lowercase}
.kkd-modal p{margin:0 0 12px;color:var(--muted,#666)}
.kkd-modal .row{display:flex;gap:8px;flex-wrap:wrap;justify-content:flex-end;margin-top:12px}
.kkd-modal ul{list-style:none;margin:0;padding:0;max-height:300px;overflow:auto}
.kkd-modal li{display:flex;justify-content:space-between;gap:10px;padding:7px 9px;border-radius:8px;cursor:pointer}
.kkd-modal li:hover{background:var(--paper2,#f3f3f3)}
.kkd-modal li small{color:var(--muted,#888)}
.kkd-modal input{width:100%;border:1px solid var(--line,#ddd);border-radius:8px;padding:6px 9px;font:inherit;background:transparent;color:inherit}
.kkd-stamp{font-size:12px;color:var(--muted);white-space:nowrap}
.kkd-tag.light{background:rgba(241,138,88,.9);color:var(--orange-ink)}
.kkd-lights{display:flex;flex-direction:column;gap:2px;margin:4px 0 6px}
.kkd-lrow{display:grid;grid-template-columns:14px minmax(0,1fr) auto auto;gap:7px;align-items:center;padding:4px 6px;border-radius:8px;border:1px solid transparent;cursor:pointer;font-size:12px}
.kkd-lrow:hover{background:var(--paper2)}
.kkd-lrow[aria-selected="true"]{background:var(--lav-soft);border-color:var(--lav)}
.kkd-lrow.off{opacity:.45}
.kkd-dot{width:12px;height:12px;border-radius:50%;border:1px solid var(--line)}
.kkd-x{border:0;background:transparent;color:var(--muted);cursor:pointer;padding:0 4px;font-size:14px}
.kkd-x:hover{color:var(--ink)}
.kkd-mcard{border:1px solid var(--line);border-radius:10px;padding:6px 8px;margin:6px 0}.kkd-mcard.off{opacity:.55}
.kkd-mhead{display:flex;gap:6px;align-items:center}.kkd-mhead select{flex:1;min-width:0;font-size:12px}
.kkd-mgrid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:4px 8px;margin-top:6px}
.kkd-mf{display:flex;flex-direction:column;font-size:11px;color:var(--muted);gap:2px}.kkd-mf input,.kkd-mf select{width:100%;min-width:0;font-size:12px;color:var(--ink,inherit);padding:3px 6px!important}
.kkd-x{background:transparent;border:0;cursor:pointer;font-size:16px;line-height:1;color:var(--muted);padding:0 4px}.kkd-x:hover{color:#c0392b}
.kkd-sub{font-size:11px;font-weight:600;letter-spacing:.06em;color:var(--muted);text-transform:lowercase;margin:10px 0 2px;display:flex;gap:6px;align-items:center}
.kkd-lstatus{font-size:12px;color:var(--muted);min-height:16px}
.kkd-lstatus.err{color:#c0392b}
.kkd-fxh{display:flex;align-items:center;gap:6px;font-size:12px;font-weight:600;margin:8px 0 0;text-transform:lowercase}
.kkd-fxh+.kkd-field{margin-top:2px}
.kkd-glow{display:grid;grid-template-columns:minmax(0,1fr) 34px auto;gap:6px;align-items:center;margin:4px 0}
.kkd-glow input[type=range]{grid-column:1/-1}
.kkd-note{position:fixed;left:50%;top:40%;transform:translate(-50%,-50%);z-index:10004;background:#fff;color:#111;border-radius:14px;padding:16px 20px;box-shadow:0 16px 40px rgba(0,0,0,.25);font:400 15px "Hanken Grotesk KKD",system-ui,sans-serif;max-width:420px}
`;
function injectStyle() {
  if (document.getElementById("kkd-style")) return;
  const s = document.createElement("style"); s.id = "kkd-style"; s.textContent = CSS; document.head.appendChild(s);
}

// ------------------------------------------------------------------------------------------------ helpers
const viewURL = r => api.apiURL(`/view?filename=${encodeURIComponent(r.filename)}&subfolder=${encodeURIComponent(r.subfolder || "")}&type=${r.type || "temp"}&t=${Date.now()}`);
const stableURL = r => api.apiURL(`/view?filename=${encodeURIComponent(r.filename)}&subfolder=${encodeURIComponent(r.subfolder || "")}&type=${r.type || "temp"}`);
const inputURL = name => { const i = name.lastIndexOf("/"); return viewURL({ filename: name.slice(i + 1), subfolder: i >= 0 ? name.slice(0, i) : "", type: "input" }); };
const loadImage = url => new Promise((res, rej) => { const im = new Image(); im.onload = () => res(im); im.onerror = () => rej(new Error("image " + url)); im.src = url; });
const mkCanvas = (w, h) => Object.assign(document.createElement("canvas"), { width: Math.max(1, Math.round(w)), height: Math.max(1, Math.round(h)) });
const iw = im => im.videoWidth || im.naturalWidth || im.width, ih = im => im.videoHeight || im.naturalHeight || im.height;
// ---- video layers: an image layer whose source is a video file, timed by media.segments (kubakub/director/media.py)
const VIDEO_RE = /\.(mp4|mov|m4v|mkv|webm|avi|mxf|mpg|mpeg|wmv|gif)$/i;
const isVideoName = n => VIDEO_RE.test(String(n || ""));
const isVideo = L => L?.kind === "image" && /^(file|path):/.test(L.source || "") && isVideoName(L.source);   // path: = linked from disk
function mediaSegs(media, dur) {                    // = media.segments(): normalised, sorted by start
  const num = (s, k, d) => (typeof s?.[k] === "number" && Number.isFinite(s[k]) ? s[k] : d);
  const raw = Array.isArray(media?.segments) && media.segments.length ? media.segments : [{}];
  return raw.filter(s => s && typeof s === "object").map((s, i) => {
    const D = +dur || 0, tin = D ? Math.min(Math.max(num(s, "in", 0), 0), Math.max(0, D - 1e-3)) : Math.max(num(s, "in", 0), 0);
    let tout = num(s, "out", -1); if (tout <= tin || (D && tout > D)) tout = D;
    const speed = Math.min(Math.max(num(s, "speed", 1), 0.01), 100), natural = Math.max(1e-3, (tout - tin) / speed), len = num(s, "length", -1);
    return { i, start: num(s, "start", 0), in: tin, out: tout, speed, natural, length: len <= 0 ? natural : len, fill: s.fill === "loop" ? "loop" : "hold", reverse: !!s.reverse };
  }).sort((a, b) => a.start - b.start);
}
function mediaTime(segs, t) {                       // = media.source_time(): [segment, seconds into the file, held] or null
  for (let k = segs.length - 1; k >= 0; k--) {
    const s = segs[k];
    if (s.start - 1e-9 <= t && t < s.start + s.length - 1e-9) {
      let local = (t - s.start) * s.speed; const span = s.out - s.in, over = local >= span, loop = s.fill === "loop" && span > 1e-9;
      if (over) local = loop ? local % span : span;
      return [s, s.reverse ? Math.max(s.in, s.out - local - 1e-6) : s.in + local, over && !loop];     // = media.source_time
    }
  }
  return null;
}
const esc = s => String(s).replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const slug = s => String(s).replace(/[^\w.+\-]+/g, "_").replace(/^_+|_+$/g, "") || "layer";
function note(text) {
  injectStyle();
  const d = document.createElement("div"); d.className = "kkd-note"; d.textContent = text; document.body.appendChild(d);
  setTimeout(() => d.remove(), 3200);
}
async function contentName(blob, prefix) {       // same pixels -> same name; different workflows never clash
  const buf = new Uint8Array(await blob.arrayBuffer());
  let hex;
  if (crypto?.subtle) hex = [...new Uint8Array(await crypto.subtle.digest("SHA-1", buf))].slice(0, 10).map(b => b.toString(16).padStart(2, "0")).join("");
  else { let h = 0x811c9dc5 >>> 0; for (let i = 0; i < buf.length; i++) h = Math.imul(h ^ buf[i], 16777619) >>> 0; hex = h.toString(16) + buf.length.toString(16); }
  return `${prefix}_${hex}.png`;
}
async function upload(blob, name, overwrite) {
  const fd = new FormData();
  fd.append("image", blob, name); fd.append("type", "input"); fd.append("subfolder", "kuba_director");
  if (overwrite) fd.append("overwrite", "true");
  const r = await api.fetchApi("/upload/image", { method: "POST", body: fd });
  if (r.status !== 200) throw new Error("upload failed: " + r.status);
  const j = await r.json();
  return (j.subfolder ? j.subfolder + "/" : "") + j.name;
}

const ICON = {
  move: '<path d="M5 3l6 16 2-7 7-2z"/>',
  pick: '<rect x="4" y="4" width="7" height="7"/><rect x="13" y="13" width="7" height="7"/><rect x="13" y="4" width="7" height="7" stroke-dasharray="2 2"/>',
  brush: '<path d="M4 20c3 0 4-2 4-4l9-9 3 3-9 9c-2 0-4 1-4 4"/>',
  erase: '<path d="M8 20h12M5 15l8-8 6 6-5 5H9z"/>',
  hand: '<path d="M8 12V6a1.5 1.5 0 013 0v5M11 11V4.5a1.5 1.5 0 013 0V11M14 11V6a1.5 1.5 0 013 0v7c0 4-2.5 7-6 7s-5-2-7-5l-1-2a1.5 1.5 0 012.6-1.4L8 14"/>',
  snap: '<path d="M6 3v7a6 6 0 0012 0V3"/><path d="M6 3h4v7a2 2 0 004 0V3h4"/><path d="M6 7h4M14 7h4"/>',
  pin: '<path d="M5 6l14-2 1 15-16 1z"/><circle cx="5" cy="6" r="1.4"/><circle cx="19" cy="4" r="1.4"/><circle cx="20" cy="19" r="1.4"/><circle cx="4" cy="20" r="1.4"/>',
};
const EXPORT_FORMATS = [["png8", "PNG sequence 8 bit"], ["png16", "PNG sequence 16 bit"], ["prores4444", "ProRes 4444 (.mov, alpha)"],
  ["prores422hq", "ProRes 422 HQ (.mov)"], ["h264", "H.264 high quality (.mp4)"], ["h264_444", "H.264 4:4:4 10 bit (.mp4)"],
  ["h265", "H.265 10 bit (.mp4)"], ["preview", "preview (.mp4, half size)"]];     // director/export.py FORMATS
const KEYS = [["V", "move / scale / rotate"], ["S", "snapping on / off (hold Ctrl: off while dragging)"], ["N", "region names on / off"],
  ["Shift + corner", "scale freely"], ["Alt + handle", "scale from the centre"], ["R", "clip to region"], ["B / E", "brush / eraser"], ["[ / ]", "brush size"], ["H or Space", "pan"],
  ["Ctrl scroll", "zoom"], ["0", "fit view"], ["1 – 4", "switch view"], ["Arrows", "nudge (Shift = 10×)"], ["Ctrl ] / Ctrl [", "forward / back"],
  ["Ctrl Shift ] / [", "to front / to back"], ["Ctrl O", "import image or video (or drop / paste)"], ["Shift N", "new paint layer"], ["Shift A", "new colour grade"],
  ["Shift L", "new light layer (relight the scene)"], ["D", "diffuse the selected layer (Klein + your LoRAs)"], ["double click", "light layer: add a point light there"],
  ["Ctrl Alt G", "clip to layer below"], ["Alt I", "invert mask (cut out)"], ["Ctrl J", "duplicate"], ["Alt drag", "drag a copy"], ["Ctrl K", "cut the video at the playhead"], ["Del", "delete layer"], ["Alt H", "show / hide layer"],
  ["P", "play / pause (with the sound)"], [", / .", "previous / next frame"], ["< / >", "previous / next keyframe or marker"], ["K", "keyframe the selection now"],
  ["Shift K", "record: changes become keyframes"], ["M", "marker at the playhead (double click it to rename)"], ["Ctrl + drag", "on the timeline: no snapping"], ["Home", "back to the start"], ["Ctrl Z / Ctrl Shift Z", "undo / redo"], ["Ctrl Enter", "apply"], ["Ctrl S", "save the composition under a name"], ["Esc", "close (without apply)"], ["?", "this sheet"]];

// ------------------------------------------------------------------------------------------------ the window
class Director {
  constructor(node, manifest) {
    this.node = node; this.m = manifest;
    [this.W, this.H] = manifest.canvas; this.k = manifest.proxy_scale || 1;
    this.PW = Math.max(1, Math.round(this.W * this.k)); this.PH = Math.max(1, Math.round(this.H * this.k));
    this.view = "matrix"; this.tool = "move"; this.zoom = 1; this.panX = 0; this.panY = 0; this.sel = 0; this.space = false;
    this.undo = []; this.redo = []; this.maskCache = new Map(); this.uid = 1;
    this.comp = mkCanvas(this.PW, this.PH); this.cc = this.comp.getContext("2d");
    this.lyr = mkCanvas(this.PW, this.PH); this.lc = this.lyr.getContext("2d");
    this.msk = mkCanvas(this.PW, this.PH); this.mc = this.msk.getContext("2d");
    this.mtmp = mkCanvas(this.PW, this.PH); this.mtc = this.mtmp.getContext("2d");
    this.fxtmp = mkCanvas(this.PW, this.PH); this.fxc = this.fxtmp.getContext("2d");
    this.views = {}; this.labels = null; this.snapOn = true; this.guides = []; this.names = false;
    this.lq = "fast"; this.lsel = 0; this.dismissed = []; this.lightTimers = new Map();
    this.diff = this.diffDefaults(null); this.jobs = new Map();
    this.tl = { duration: 10, fps: 25, time: 0, markers: [], audio: null, beat: null, snap: true, clips: [], h3: { look: '', camera: '', avoid: '', audio: '' } }; this.autokey = false; this.playing = false;
    this.pm = { on: false, source: "", invert: false, grow: 0, feather: 0, view: "black" };                   // the projection mask slot (always on top of everything)
  }
  newId(prefix, list) { let i = list.length + 1; while (list.some(e => e.id === prefix + i)) i++; return prefix + i; }

  // ---- timeline (T1): keyframes on layer / light values; the layers always hold the values at this.tl.time
  animPaths(L) {
    const out = [...ANIM_LAYER.filter(p => L.kind === "image" || L.kind === "shape" || p === "opacity" || p === "visible")];
    if (L.kind === "adjust") out.push(...ANIM_ADJUST);
    if (L.kind === "shape") out.push("color", "shape.feather");
    if (["image", "paint", "shape", "base", "adjust"].includes(L.kind)) out.push(...Object.keys(FX_DEF).map(k => "fx." + k));
    if (L.light_react) out.push("light_react.amount");
    if (L.kind === "light" && L.light) {
      out.push(...ANIM_RIG);
      for (const q of L.light.lights) out.push(...ANIM_LAMP.filter(f => q[f] !== undefined || f === "on").map(f => `light.lights.#${q.id}.${f}`));
      for (const g of L.light.glow) out.push(...ANIM_GLOW.map(f => `light.glow.#${g.id}.${f}`));
    }
    return out;
  }
  pathLabel(L, p) {
    const m = p.match(/^light\.(lights|glow)\.#([^.]+)\.(.+)$/);
    if (m) { const list = L.light?.[m[1]] || [], i = list.findIndex(e => e.id === m[2]), e = list[i];
      return `${m[1] === "lights" ? (e?.type || "lamp") + " " + (i + 1) : "glow " + (i + 1)} ${m[3]}`; }
    return p.replace(/^light\./, "").replace(/^adjust\./, "").replace("projector.brightness", "projector").replace(/_/g, " ");
  }
  syncEval() {                                       // remember what every value is now (edits are found against it)
    this.evalCache = new Map();
    for (const L of this.layers || []) { const c = {}; for (const p of this.animPaths(L)) c[p] = pathGet(L, p); this.evalCache.set(L, c); }
  }
  recordKeys() {                                     // edits since the last look: keyframes where the value animates (or auto-key)
    if (!this.layers || !this.evalCache) return;
    const t = this.tl.time;
    for (const L of this.layers) {
      let c = this.evalCache.get(L); if (!c) { c = {}; this.evalCache.set(L, c); }
      for (const p of this.animPaths(L)) {
        const v = pathGet(L, p);
        if (!(p in c)) { c[p] = v; continue; }
        if (v === c[p] || v === undefined) continue;
        const keys = L.anim?.[p];
        if (keys?.length || this.autokey) {                                  // keys hold values without the behaviours' offsets
          L.anim = L.anim || {};
          if (!keys?.length && t > 0 && c[p] !== undefined) L.anim[p] = [{ t: 0, v: this.unmoved(L, p, c[p]), e: "smooth" }];   // animate from the old value
          this.putKey(L, p, t, this.unmoved(L, p, v));
        }
        c[p] = v;
      }
    }
  }
  putKey(L, p, t, v) {
    const keys = (L.anim = L.anim || {})[p] = L.anim[p] || [];
    const k = keys.find(q => Math.abs(q.t - t) < 0.5 / this.tl.fps);
    if (k) k.v = v; else { keys.push({ t, v, e: "smooth" }); keys.sort((a, b) => a.t - b.t); }
    this.tlDirty = true;
  }
  applyTime(t) {                                     // write every animated value at time t into the layers
    this.recordKeys();
    this.tl.time = Math.max(0, Math.min(this.tl.duration, t));
    for (const L of this.layers || []) {
      const loops = L.motion?.length ? loopModes(L.motion) : {};
      for (const [p, keys] of Object.entries(L.anim || {})) {
        if (!keys.length) { delete L.anim[p]; continue; }
        const v = keyedValue(keys, this.tl.time, loops[p]); if (v !== undefined) pathSet(L, p, v);
      }
      if (L.motion?.length || L._rest || L._off) this.applyMotion(L);
    }
    this.syncEval();                                 // region masks stay cached: selectors are not animated
    this.syncVideos(this.playing);
  }
  restOf(L, p) {                                     // a behaviour's base value: moved by what the user changed since it was applied
    const cur = pathGet(L, p), r = L._rest?.[p], a = L._applied?.[p];
    if (r === undefined || a === undefined) return this.unmoved(L, p, cur);
    return typeof cur === "number" && typeof r === "number" && typeof a === "number" ? r + (cur - a) : cur === a ? r : cur;
  }
  unmoved(L, p, v) {                                 // a shown value without what the behaviours added at the last frame
    const o = L._off?.[p]; return typeof v === "number" && typeof o === "number" ? v - o : v;
  }
  motionCtx() {                                     // beats / markers / loudness for pulse and audio behaviours (cached)
    const off = this.tl.audio?.offset || 0, key = [this.tl.beat?.bpm, this.tl.beat?.phase, off, this.tl.duration, (this.tl.markers || []).map(m => m.t).join(",")].join("|");
    if (this._mctx?.key !== key) this._mctx = { key, beats: beatTimes(this.tl.beat, off, this.tl.duration), markers: (this.tl.markers || []).map(m => m.t).filter(Number.isFinite).sort((a, b) => a - b), offset: off };
    const file = this.tl.audio?.file || "";
    const wantLevel = file && this.layers?.some(L => L.motion?.some(b => b.type === "audio" && b.on !== false));
    if (wantLevel && this._levelFile !== file) {       // the node's own curve (motion.level_curve), so preview = render
      this._levelFile = file; this._level = null;
      api.fetchApi("/kubakub/director/level", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ file }) })
        .then(r => r.json()).then(j => { if (this._levelFile !== file || !Array.isArray(j.level) || this.closed) return;
          this._level = j.level; this.applyTime(this.tl.time); this.draw(); })
        .catch(e => console.warn("[kubakub director] level", e));
    }
    this._mctx.level = this._levelFile === file ? this._level : null;
    return this._mctx;
  }
  applyMotion(L) {                                   // behaviours on top of the keyed / rest values; the rest values are what is saved
    const offs = motionOffsets(L.motion || [], this.tl.time, this.tl.duration, this.motionCtx());
    const touched = new Set(Object.keys(offs).flatMap(p => p === "scale" ? ["x", "y", "w", "h"] : [p]));
    const keyed = p => !!L.anim?.[p]?.length;
    L._rest = L._rest || {}; L._applied = L._applied || {};
    for (const p of Object.keys(L._rest)) if (!touched.has(p) || keyed(p)) {         // no longer moved: back to rest
      if (!keyed(p)) pathSet(L, p, this.restOf(L, p));
      delete L._rest[p]; delete L._applied[p];
    }
    for (const p of Object.keys(L._off || {})) if (!touched.has(p) && !keyed(p)) pathSet(L, p, this.unmoved(L, p, pathGet(L, p)));
    for (const p of touched) if (!keyed(p)) { const r = this.restOf(L, p); if (typeof r === "number") { L._rest[p] = r; pathSet(L, p, r); } }
    const base = {}, off = {};
    const vals = composeMotion(offs, p => { const v = pathGet(L, p); if (!(p in base)) base[p] = v; return v; });
    for (const [p, v] of Object.entries(vals)) {
      pathSet(L, p, v); if (!keyed(p) && p in L._rest) L._applied[p] = v;
      if (typeof base[p] === "number") off[p] = v - base[p];
    }
    for (const p of Object.keys(L._rest)) if (!(p in L._applied) || !(p in vals)) { L._applied[p] = pathGet(L, p); }
    if (Object.keys(off).length) L._off = off; else delete L._off;
    if (!Object.keys(L._rest).length) { delete L._rest; delete L._applied; }
  }
  fxShown(L) {                                     // effect groups in the inspector: in use, animated, or just added
    const f = fxOf(L), added = (this._fxAdded ||= new WeakMap()).get(L) || [], mv = (L.motion || []).map(b => String(b.path || ""));
    return FX_GROUPS.filter(([name, , keys]) => f[keys[0]] > 0 || added.includes(name) || keys.some(k => mv.includes("fx." + k))).map(g => g[0]);
  }
  fillRanges() {                                   // the orange part of every slider track
    for (const el of document.querySelectorAll(".kkd-root input[type=range], .kkd-pop input[type=range]")) {
      const lo = el.min === "" ? 0 : +el.min, hi = el.max === "" ? 100 : +el.max;
      el.style.setProperty("--p", Math.max(0, Math.min(100, (+el.value - lo) / ((hi - lo) || 1) * 100)) + "%");
    }
  }
  renderMotion(L) {
    const el = this.$("g-motion"), paths = this.motionPaths(L);
    el.hidden = !paths.length; if (el.hidden) return;
    const box = L.kind === "image" || L.kind === "shape", mv = L.motion || [];
    const opt = (v, cur, label) => `<option value="${esc(v)}" ${v === cur ? "selected" : ""}>${esc(label ?? v)}</option>`;
    const lbl = p => p === "position" ? "position (x y)" : p === "scale" ? "scale %" : this.pathLabel(L, p);
    const num = (b, i, k, label) => `<label class="kkd-mf"><span>${label}</span><input type="text" inputmode="decimal" data-num data-mi="${i}" data-mk="${k}" value="${+(b[k] ?? mDefault(b, k))}"></label>`;
    el.kkdHTML = `<div class="kkd-sub" data-tip="Procedural motion on top of the keyframes: wiggle, oscillate, drift, random steps, looped keyframes. Several add up; the node renders them per frame">behaviours <span class="kkd-grow"></span>
      <select data-id="m-add" style="font-size:12px;max-width:150px"><option value="">+ behaviour</option>${M_PRESETS.map(([l, needBox, pr], i) => needBox && !box ? ""
        : pr.type === "stagger" && !L.clip ? `<option value="${i}" disabled>${l} (clip to regions first)</option>` : `<option value="${i}">${l}</option>`).join("")}</select></div>` +
      mv.map((b, i) => `<div class="kkd-mcard ${b.on === false ? "off" : ""}">
        <div class="kkd-mhead"><select data-mi="${i}" data-mk="type">${M_TYPES.map(t => opt(t, b.type)).join("")}</select>
          ${b.type === "repeat" ? `<select disabled><option>copies of this layer</option></select>` : b.type === "stagger" ? `<select disabled><option>regions (${esc(L.clip || "no clip")})</option></select>` : `<select data-mi="${i}" data-mk="path">${(pl => (pl.includes(b.path) ? "" : `<option value="" selected>— pick a value —</option>`) + pl.map(p => opt(p, b.path, lbl(p))).join(""))(b.type === "loop" ? this.loopPaths(L) : paths)}</select>`}
          <input type="checkbox" data-mi="${i}" data-mk="on" ${b.on === false ? "" : "checked"} data-tip="On / off"><button class="kkd-x" data-mdel="${i}" data-tip="Remove">×</button></div>
        <div class="kkd-mgrid">${(M_PARAMS[b.type] || []).map(([k, l]) => num(b, i, k, l)).join("")}
          ${b.type === "oscillate" ? `<label class="kkd-mf"><span>wave</span><select data-mi="${i}" data-mk="wave">${["sine", "triangle", "square", "saw", ...(b.path === "position" ? ["circle"] : [])].map(w => opt(w, b.wave || "sine")).join("")}</select></label>` : ""}
          ${b.type === "repeat" ? `<label class="kkd-mf"><span>layout</span><select data-mi="${i}" data-mk="layout">${[["line", "line"], ["grid", "grid"], ["radial", "ring"]].map(([v, l]) => opt(v, b.layout || "line", l)).join("")}</select></label>`
            + ((b.layout || "line") === "radial" ? num(b, i, "radius", "radius px") + num(b, i, "start", "start °") + `<label class="kkd-mf"><span>orient</span><input type="checkbox" data-mi="${i}" data-mk="orient" ${b.orient ? "checked" : ""}></label>`
              : num(b, i, "dx", "step x px") + num(b, i, "dy", "step y px") + (b.layout === "grid" ? num(b, i, "cols", "columns") : "")) : ""}
          ${b.type === "stagger" ? `<label class="kkd-mf"><span>mode</span><select data-mi="${i}" data-mk="mode">${[["sequence", "sequence"], ["chase", "chase"], ["wave", "wave"], ["random", "random"]].map(([v, l]) => opt(v, b.mode || "sequence", l)).join("")}</select></label>
            <label class="kkd-mf"><span>order</span><select data-mi="${i}" data-mk="order">${[["left", "left → right"], ["right", "right → left"], ["top", "top → bottom"], ["bottom", "bottom → top"], ["centre", "centre → out"], ["outside", "outside → in"], ["size", "biggest first"], ["random", "random"]].map(([v, l]) => opt(v, b.order || "left", l)).join("")}</select></label>`
            + ((b.mode || "sequence") === "sequence" ? `<label class="kkd-mf"><span>direction</span><select data-mi="${i}" data-mk="direction">${[["in", "turn on"], ["out", "turn off"]].map(([v, l]) => opt(v, b.direction || "in", l)).join("")}</select></label>` : "")
            + (b.mode === "chase" ? num(b, i, "hold", "hold s") : b.mode === "wave" ? num(b, i, "period", "period s") : b.mode === "random" ? num(b, i, "density", "on 0-1") + num(b, i, "seed", "seed") : "")
            + (b.order === "random" && b.mode !== "random" ? num(b, i, "seed", "seed") : "") : ""}
          ${b.type === "pulse" ? `<label class="kkd-mf"><span>on</span><select data-mi="${i}" data-mk="trigger">${[["beats", "beats"], ["bars", "bars"], ["markers", "markers"], ["every", "every … s"]].map(([v, l]) => opt(v, b.trigger || "beats", l)).join("")}</select></label>`
            + (b.trigger === "every" ? num(b, i, "every", "every s") : b.trigger === "markers" ? "" : num(b, i, "nth", "every nth")) : ""}
          ${b.path === "position" && (b.type === "drift" || b.type === "pulse" || b.type === "audio" || b.type === "oscillate" && b.wave !== "circle") ? num(b, i, "angle", "angle °") : ""}
          ${b.type === "repeat" ? "" : b.type === "loop" ? `<label class="kkd-mf"><span>mode</span><select data-mi="${i}" data-mk="mode">${["cycle", "pingpong", "continue"].map(m => opt(m, b.mode)).join("")}</select></label>`
            : `${num(b, i, "t_start", "from s")}<label class="kkd-mf"><span>to s</span><input type="text" inputmode="decimal" data-num data-mi="${i}" data-mk="t_end" value="${mNum(b, "t_end", -1) < 0 ? "" : b.t_end}" placeholder="end"></label>${b.type === "drift" || b.type === "stagger" ? "" : num(b, i, "fade", "fade s")}`}</div>
        ${b.type === "loop" && !b.path ? `<div class="kkd-lstatus">key a value at two times or more first, then loop it</div>` : ""}
        ${b.type === "audio" && !this.tl.audio?.file ? `<div class="kkd-lstatus">load a sound first (♪ audio in the timeline)</div>` : ""}
        ${b.type === "pulse" && (b.trigger || "beats") !== "every" && (b.trigger === "markers" ? !this.tl.markers?.length : !this.tl.beat?.bpm) ? `<div class="kkd-lstatus">${b.trigger === "markers" ? "no markers yet (M in the timeline)" : "no beats yet: load a sound (♪ audio) or set the tempo"}</div>` : ""}</div>`).join("");
  }
  layerAt(L, t) {                                    // a copy of the layer with its keys and behaviours at time t (for delayed repeats)
    const c = { ...L, fx: L.fx ? { ...L.fx } : undefined, shape: L.shape ? { ...L.shape } : undefined, adjust: L.adjust ? { ...L.adjust } : undefined,
      light_react: L.light_react ? { ...L.light_react } : undefined, mask: L.mask ? { ...L.mask } : undefined };
    const loops = loopModes(L.motion || []), keyed = new Set();
    for (const [p, keys] of Object.entries(L.anim || {})) if (keys.length) { const v = keyedValue(keys, t, loops[p]); if (v !== undefined) { pathSet(c, p, v); keyed.add(p); } }
    for (const p of Object.keys(L._rest || {})) if (!keyed.has(p)) pathSet(c, p, this.restOf(L, p));
    const offs = motionOffsets(L.motion || [], t, this.tl.duration, this.motionCtx());
    for (const [p, v] of Object.entries(composeMotion(offs, p => pathGet(c, p)))) pathSet(c, p, v);
    return c;
  }
  repeatCopies(L) {                                  // [the layer, copy 1, copy 2 …] (copies drawn below it)
    const rb = (L.kind === "image" || L.kind === "shape") ? repeatOf(L.motion) : null; if (!rb) return [L];
    const out = [L], delay = mNum(rb, "delay", 0), motion = (L.motion || []).filter(b => b.type !== "repeat");
    for (let k = 1; k < repeatCount(rb); k++) {
      const c = delay && !L.media ? this.layerAt(L, this.tl.time - k * delay) : { ...L, fx: L.fx ? { ...L.fx } : undefined };
      c.id = L.id + "~" + k; c.motion = motion; c._copyOf = L;
      out.push(repeatApply(c, rb, k));
    }
    return out;
  }
  motionPaths(L) {                                   // numeric values a behaviour can move (+ position / scale for boxes)
    const box = L.kind === "image" || L.kind === "shape";
    return [...(box ? ["position", "scale"] : []), ...this.animPaths(L).filter(p => p !== "visible" && (p.startsWith("fx.") || typeof pathGet(L, p) === "number"))];
  }
  loopPaths(L) {                                     // keyed values a loop can repeat (position = x / y)
    const k2 = p => (L.anim?.[p] || []).length > 1, box = L.kind === "image" || L.kind === "shape";
    return [...(box && (k2("x") || k2("y")) ? ["position"] : []), ...Object.keys(L.anim || {}).filter(k2)];
  }
  setTime(t, quiet) {
    this.applyTime(Math.round(t * this.tl.fps) / this.tl.fps);
    if (!quiet) this._panels = true;                 // rendered with the next frame (key repeat / scrubbing: once per frame)
    this.draw();
  }
  keyNow() {                                         // K: keyframe every value of the selected layer (or lamp) as it is now
    const L = this.cur(); if (!L) return; this.snap();
    const lamp = L.kind === "light" ? L.light.lights[this.lsel] : null;
    for (const p of this.animPaths(L)) {
      if (lamp && p.startsWith("light.lights.") && !p.startsWith(`light.lights.#${lamp.id}.`)) continue;
      if (L.kind === "light" && !p.startsWith("light.") && !L.anim?.[p]) continue;
      const v = this.unmoved(L, p, pathGet(L, p)); if (v !== undefined) this.putKey(L, p, this.tl.time, v);
    }
    this.syncEval(); this.drawTimeline(); this.scheduleDraft();
  }
  // ---- H3 clips (rendered by the 'kubakub keyframe clips (h3)' node after the director)
  addClip() {
    const t = this.tl.time, ms = this.tl.markers.map(m => m.t).sort((a, b) => a - b);
    let a = [...ms].reverse().find(m => m <= t + 1e-6), b = ms.find(m => m > t + 1e-6);
    if (a === undefined || b === undefined || b - a < 0.4) { a = t; b = Math.min(this.tl.duration, t + 3); }
    const id = this.newId("c", this.tl.clips);
    this.tl.clips.push({ id, t_start: +a.toFixed(2), t_end: +b.toFixed(2), mode: "keyframes", prompt: "", audio: "", refs: [], raw: false, on: true });
    this.tl.clips.sort((x, y) => x.t_start - y.t_start);
    this.scheduleDraft(); this.drawTimeline(); return id;
  }
  openClips(focus) {
    if (!this.clipPop) {
      this.clipPop = document.createElement("div"); this.clipPop.className = "kkd-pop"; this.clipPop.hidden = true;
      this.clipPop.style.maxHeight = "70vh"; this.clipPop.style.overflow = "auto"; this.clipPop.style.width = "min(520px, calc(100vw - 32px))";
      this.root.appendChild(this.clipPop);
      const P = this.clipPop, C = () => this.tl.clips;
      const clipOf = el => C().find(c => c.id === el.closest("[data-cid]")?.dataset.cid);
      P.addEventListener("input", e => {
        const t = e.target, k = t.dataset.h3;
        if (k) { this.tl.h3[k] = t.value; this.scheduleDraft(); return; }
        const c = clipOf(t), f = t.dataset.cf; if (!c || !f) return;
        if (f === "prompt" || f === "audio") { c[f] = t.value; this.scheduleDraft(); }
      });
      P.addEventListener("change", e => {
        const t = e.target;
        if (t.dataset.pin !== undefined) {           // a marker pins its frame: H3 passes through the director's picture there
          const m = this.tl.markers[+t.dataset.pin]; if (!m) return;
          this.snap(); if (t.checked) m.anchor = true; else delete m.anchor;
          this.scheduleDraft(); this.renderClips(); this.drawTimeline(); return;
        }
        const c = clipOf(t), f = t.dataset.cf; if (!c || !f) return;
        if (f === "t_start" || f === "t_end") { const v = parseFloat(t.value); if (Number.isFinite(v)) c[f] = Math.max(0, Math.min(this.tl.duration, v));
          if (c.t_end < c.t_start + 0.2) c.t_end = Math.min(this.tl.duration, c.t_start + 0.2); this.tl.clips.sort((x, y) => x.t_start - y.t_start); }
        if (f === "mode") c.mode = t.value;
        if (f === "ref") { const v = t.value; c.refs = t.checked ? [...new Set([...c.refs, v])] : c.refs.filter(r => r !== v); }
        this.scheduleDraft(); this.renderClips(); this.drawTimeline();
      });
      P.addEventListener("click", e => {
        const t = e.target;
        if (t.closest("[data-cadd]")) { const id = this.addClip(); this.renderClips(id); return; }
        const c = clipOf(t); if (!c) return;
        const act = t.closest("[data-cact]")?.dataset.cact; if (!act) return;
        if (act === "start") c.t_start = +this.tl.time.toFixed(2);
        if (act === "end") c.t_end = +this.tl.time.toFixed(2);
        if (act === "raw") c.raw = !c.raw;
        if (act === "on") c.on = !c.on;
        if (act === "go") this.setTime(c.t_start);
        if (act === "del") this.tl.clips = this.tl.clips.filter(q => q !== c);
        if (c.t_end < c.t_start + 0.2) c.t_end = Math.min(this.tl.duration, c.t_start + 0.2);
        this.scheduleDraft(); this.renderClips(); this.drawTimeline();
      });
    }
    this.renderClips(typeof focus === "number" ? this.tl.clips[focus]?.id : null);
    const r = this.$("tl-h3").getBoundingClientRect(); this.clipPop.hidden = false;
    this.clipPop.style.left = Math.max(16, Math.min(r.left, innerWidth - this.clipPop.offsetWidth - 16)) + "px";
    this.clipPop.style.top = Math.max(16, r.top - this.clipPop.offsetHeight - 8) + "px";
  }
  renderClips(focusId) {
    const P = this.clipPop, H = this.tl.h3;
    const layers = this.layers.filter(q => q.kind === "image" || q.kind === "paint");
    const refBox = (c, v, label) => `<label style="white-space:nowrap;font-size:12px"><input type="checkbox" data-cf="ref" value="${esc(v)}" ${c.refs.includes(v) ? "checked" : ""}> ${esc(label)}</label>`;
    P.innerHTML = `<h3>h3 clips</h3>
      <div class="kkd-lstatus">Rendered by the <b>kubakub keyframe clips (h3)</b> node after <b>kubakub director sequence</b>. Keyframes: the frames at start and end, H3 animates between them (fl2va). Reference: layers / music / the sequence as references (ref2va). Named markers inside a clip become its timeline beats.</div>
      <div class="kkd-sub">for all clips</div>
      <div class="kkd-field"><span>look</span><textarea data-h3="look" placeholder="the look: light, colours, materials, mood">${esc(H.look)}</textarea></div>
      <div class="kkd-field"><span>camera</span><input type="text" data-h3="camera" value="${esc(H.camera)}" placeholder="locked-off, no camera movement (default)"><span></span></div>
      <div class="kkd-field"><span>sound</span><input type="text" data-h3="audio" value="${esc(H.audio)}" placeholder="default sound (each clip can say its own)"><span></span></div>
      <div class="kkd-field"><span>avoid</span><input type="text" data-h3="avoid" value="${esc(H.avoid)}" placeholder="no text, the architecture never changes (default)"><span></span></div>
      <div class="kkd-sub">clips <span class="kkd-grow"></span><button class="kkd-pill" style="padding:2px 8px;font-size:12px" data-cadd data-tip="A clip between the markers around the playhead, or the next 3 s">+ clip</button></div>
      ${this.tl.clips.map(c => `<div data-cid="${esc(c.id)}" style="border:1px solid ${c.id === focusId ? "var(--orange)" : "var(--line)"};border-radius:10px;padding:8px;margin:6px 0;${c.on === false ? "opacity:.5" : ""}">
        <div class="kkd-field" style="grid-template-columns:auto auto auto auto auto auto 1fr auto"><span>from</span><input type="text" data-cf="t_start" value="${c.t_start}" style="width:56px">
          <button class="kkd-pill" style="padding:1px 7px;font-size:12px" data-cact="start" data-tip="Start at the playhead">here</button>
          <span>to</span><input type="text" data-cf="t_end" value="${c.t_end}" style="width:56px">
          <button class="kkd-pill" style="padding:1px 7px;font-size:12px" data-cact="end" data-tip="End at the playhead">here</button><span></span>
          <button class="kkd-x" data-cact="del" data-tip="Remove the clip">×</button></div>
        <div class="kkd-field"><span>type</span><select data-cf="mode"><option value="keyframes" ${c.mode === "keyframes" ? "selected" : ""}>keyframes (start + end frame)</option><option value="reference" ${c.mode === "reference" ? "selected" : ""}>reference (layers, music, sequence)</option></select><span></span></div>
        <div class="kkd-field"><textarea data-cf="prompt" placeholder="what happens in this clip">${esc(c.prompt)}</textarea></div>
        <div class="kkd-field"><span>sound</span><input type="text" data-cf="audio" value="${esc(c.audio)}" placeholder="the sound of this clip (optional)"><span></span></div>
        ${c.mode !== "reference" ? (() => { const ms = this.tl.markers.map((m, i) => [m, i]).filter(([m]) => m.t > c.t_start && m.t < c.t_end);
          return ms.length ? `<div style="display:flex;flex-wrap:wrap;gap:4px 12px;margin:4px 0;font-size:12px" data-tip="A pinned marker makes H3 pass through the director's picture at that moment (a keyframe in the middle of the clip); unpinned named markers are beats in the prompt">
            <span style="color:var(--muted)">pin frames:</span>${ms.map(([m, i]) => `<label style="white-space:nowrap"><input type="checkbox" data-pin="${i}" ${m.anchor ? "checked" : ""}> ${esc(m.name || "marker")} · ${m.t.toFixed(2)} s</label>`).join("")}</div>`
            : `<div class="kkd-lstatus">add markers (M) inside the clip to pin frames in its middle</div>`; })() : ""}
        ${c.mode === "reference" ? `<div style="display:flex;flex-wrap:wrap;gap:4px 12px;margin:4px 0">${refBox(c, "sequence", "the sequence itself")}${refBox(c, "music", "the timeline music")}${layers.map(q => refBox(c, "layer:" + q.id, q.name)).join("")}</div>` : ""}
        <div style="display:flex;gap:6px;flex-wrap:wrap">
          <button class="kkd-pill" style="padding:1px 8px;font-size:12px" data-cact="go">go to</button>
          <button class="kkd-pill" style="padding:1px 8px;font-size:12px" data-cact="raw" aria-pressed="${!!c.raw}" data-tip="Send the prompt exactly as written (no H3 structure, no enhancing)">raw</button>
          <button class="kkd-pill" style="padding:1px 8px;font-size:12px" data-cact="on" aria-pressed="${c.on !== false}">${c.on === false ? "off" : "on"}</button></div>
      </div>`).join("") || `<div class="kkd-lstatus">no clips yet</div>`}`;
    if (focusId) P.querySelector(`[data-cid="${focusId}"]`)?.scrollIntoView({ block: "nearest" });
  }
  togglePlay() {
    if (!this.layers) return;                        // still loading
    this.playing = !this.playing; this.$("tl-play").textContent = this.playing ? "pause" : "play";
    if (this._playRaf) { cancelAnimationFrame(this._playRaf); this._playRaf = 0; }
    const gen = this._playGen = (this._playGen || 0) + 1;     // any toggle retires the running loop
    if (!this.playing) { this.stopAudio(); this.syncVideos(false); this.renderInspector(); this.renderLayers(); this.draw(); return; }
    let last = performance.now(), clock = this.tl.time, shown = -1;
    if (this.au) this.startAudio(this.tl.time);
    const tick = now => {
      if (gen !== this._playGen || !this.playing || !this.root?.isConnected) { this._playRaf = 0; return; }
      clock = this.src ? this.audT0 + (this.actx.currentTime - this.audC0) : clock + (now - last) / 1000;   // the sound is the clock
      last = now;
      if (clock > this.tl.duration) { clock = 0; shown = -1; if (this.au) this.startAudio(0); }
      const fi = Math.floor(clock * this.tl.fps + 1e-6);                // one picture per timeline frame (= the export)
      if (fi === shown) { this._playRaf = requestAnimationFrame(tick); return; }
      shown = fi;
      try { this.applyTime(fi / this.tl.fps); this.drawNow(); this.drawTimeline(); }
      catch (e) { console.error("[kubakub director] play", e); this._playRaf = 0; this.togglePlay(); return; }
      this._playRaf = requestAnimationFrame(tick);
    };
    this._playRaf = requestAnimationFrame(tick);
  }
  // ---- sound: waveform, beats, playback in sync (Web Audio; the file lives in input/kuba_director)
  async loadAudio() {
    const a = this.tl.audio, tok = this._audTok = {}; this.au = null; this.stopAudio();
    const stale = () => this._audTok !== tok || this.tl.audio !== a || this.closed;
    if (a?.file) {
      try {
        const r = await fetch(inputURL(a.file)); if (stale()) return; if (!r.ok) throw new Error("file not found (" + r.status + ")");
        this.actx = this.actx || new (window.AudioContext || window.webkitAudioContext)();
        const ab = await this.actx.decodeAudioData(await r.arrayBuffer());
        if (stale()) return;
        const c0 = ab.getChannelData(0), c1 = ab.numberOfChannels > 1 ? ab.getChannelData(1) : c0;
        const N = 6000, per = ab.length / N, mn = new Float32Array(N), mx = new Float32Array(N);
        for (let i = 0; i < N; i++) { let lo = 0, hi = 0; for (let j = Math.floor(i * per), e = Math.min(ab.length, Math.floor((i + 1) * per)); j < e; j += 4) { const v = (c0[j] + c1[j]) / 2; if (v < lo) lo = v; if (v > hi) hi = v; } mn[i] = lo; mx[i] = hi; }
        this.au = { buffer: ab, mn, mx, N, dur: ab.duration };
        if (!this.tl.beat) {                         // tempo + grid, once per sound (saved with the composition)
          const step = Math.max(1, Math.round(ab.sampleRate / 11025)), x = new Float32Array(Math.floor(ab.length / step));
          for (let i = 0; i < x.length; i++) x[i] = (c0[i * step] + c1[i * step]) / 2;
          const r2 = analyzeBeats(x, ab.sampleRate / step); if (r2) this.tl.beat = { bpm: r2.bpm, phase: r2.phase };
        }
      } catch (e) { if (stale()) return; console.error("[kubakub director] audio", e); note("could not read the sound: " + e.message); }
    }
    this.fillTl(); this.drawTimeline();
  }
  async importAudio(file) {
    this.busy(true, "loading the sound");
    try {
      const name = await upload(file, (file.name || "sound").replace(/[^\w.\-]+/g, "_"), false);
      const mine = this.tl.audio = { file: name, offset: 0, gain: 1 }; this.tl.beat = null;
      await this.loadAudio();
      if (this.tl.audio !== mine || this.closed) { this.busy(false); return; }     // replaced or removed meanwhile
      if (this.au) {
        const d = Math.min(600, Math.round(this.au.dur * 10) / 10);
        if (Math.abs(d - this.tl.duration) > 0.05) { this.tl.duration = d; note(`timeline length set to the sound: ${d} s${this.tl.beat ? " · " + this.tl.beat.bpm + " bpm" : ""}`); }
        this.setTime(Math.min(this.tl.time, d));
      }
      this.fillTl(); this.scheduleDraft();
    } catch (e) { note("sound import failed: " + e.message); }
    this.busy(false);
  }
  startAudio(t) {
    this.stopAudio(); if (!this.au || !this.actx) return;
    this.actx.resume?.();
    const src = this.actx.createBufferSource(), g = this.actx.createGain();
    src.buffer = this.au.buffer; g.gain.value = this.tl.audio?.gain ?? 1; src.connect(g).connect(this.actx.destination);
    const off = (this.tl.audio?.offset || 0) + t, now = this.actx.currentTime;
    if (off < this.au.dur) src.start(off < 0 ? now - off : now, Math.max(0, off));
    this.src = src; this.audT0 = t; this.audC0 = now;
  }
  stopAudio() { try { this.src?.stop(); } catch { /* not started */ } this.src = null; }
  beats() { return this.au || this.tl.beat ? beatTimes(this.tl.beat, this.tl.audio?.offset || 0, this.tl.duration) : []; }
  snapT(t, e, skip) {                                // nearest beat / marker / keyframe within 6 px (Ctrl = free)
    if (!this.tl.snap || e?.ctrlKey || e?.metaKey) return t;
    const g = this.tlGeom(), thr = 6 * g.dpr / Math.max(1, g.x1 - g.x0) * this.tl.duration;
    const cands = [0, this.tl.duration, ...this.beats(), ...this.tl.markers.filter(m => m !== skip).map(m => m.t),
      ...this.tlRows().flatMap(r => r.keys.filter(k => k !== skip).map(k => k.t))];
    let best = t, bd = thr;
    for (const c of cands) { const d = Math.abs(c - t); if (d < bd) { bd = d; best = c; } }
    return best;
  }
  tlRows() {                                         // the selected layer's animated values (keyframes, behaviours)
    const L = this.cur(); if (!L) return [];
    const rows = Object.keys(L.anim || {}).filter(p => L.anim[p]?.length && pathGet(L, p) !== undefined).map(p => ({ p, keys: L.anim[p] }));
    for (const b of L.motion || []) {
      const p = String(b.path || ""); if (!p || b.type === "loop" || b.type === "repeat") continue;
      if (!rows.some(r => r.p === p)) rows.push({ p, keys: [] });
    }
    return rows;
  }
  tlGeom() {
    const c = this.tlc, dpr = Math.min(devicePixelRatio || 1, 2), lab = 130 * dpr, pad = 8 * dpr, ruler = 22 * dpr, wave = this.au ? 30 * dpr : 0;
    const cur = this.cur(), mrow = cur?.vinfo ? 22 * dpr : 0;        // the selected video layer's clips
    return { dpr, lab, x0: lab + pad, x1: c.width - pad, ruler, wave, mrow, mtop: ruler + wave, top: ruler + wave + mrow, row: 16 * dpr };
  }
  tX(t) { const g = this.tlGeom(); return g.x0 + (g.x1 - g.x0) * t / Math.max(1e-6, this.tl.duration); }
  xT(x) { const g = this.tlGeom(); return Math.max(0, Math.min(this.tl.duration, (x - g.x0) / Math.max(1, g.x1 - g.x0) * this.tl.duration)); }
  drawTimeline() {
    const c = this.tlc; if (!c) return;
    const r = c.getBoundingClientRect(), g0 = this.tlGeom(), rows = this.tlRows();
    const h = Math.round(g0.top + (Math.max(1, rows.length) * 16 + 6) * g0.dpr);
    if (c.width !== Math.round(r.width * g0.dpr) || c.height !== h) { c.width = Math.max(10, Math.round(r.width * g0.dpr)); c.height = h; c.style.height = h / g0.dpr + "px"; }
    const g = this.tlGeom(), ctx = c.getContext("2d"), css = getComputedStyle(this.root);
    const ink = css.getPropertyValue("--muted").trim() || "#888", line = css.getPropertyValue("--line").trim() || "#ddd";
    ctx.clearRect(0, 0, c.width, c.height);
    ctx.font = `${Math.round(11 * g.dpr)}px "Hanken Grotesk KKD", system-ui, sans-serif`; ctx.textBaseline = "middle";
    const D = this.tl.duration, step = D <= 5 ? 0.5 : D <= 20 ? 1 : D <= 60 ? 5 : D <= 300 ? 15 : 60;
    ctx.strokeStyle = line; ctx.fillStyle = ink; ctx.lineWidth = 1;
    for (let t = 0; t <= D + 1e-6; t += step) {
      const x = Math.round(this.tX(t)) + .5; ctx.beginPath(); ctx.moveTo(x, g.ruler - 6 * g.dpr); ctx.lineTo(x, c.height); ctx.stroke();
      ctx.fillText(`${+t.toFixed(1)}s`, x + 3 * g.dpr, 7 * g.dpr);
    }
    // beats (bars every 4th) on the ruler, thinned out when they get too dense
    const beats = this.beats(), px = beats.length > 1 ? this.tX(beats[1]) - this.tX(beats[0]) : 0;
    if (beats.length && px > 2 * g.dpr) {
      ctx.strokeStyle = LAV;
      beats.forEach((t, i) => { const bar = i % 4 === 0; if (!bar && px < 5 * g.dpr) return;
        const x = Math.round(this.tX(t)) + .5; ctx.lineWidth = bar ? 1.5 * g.dpr : 1; ctx.beginPath(); ctx.moveTo(x, g.ruler - (bar ? 9 : 5) * g.dpr); ctx.lineTo(x, g.ruler); ctx.stroke(); });
    }
    // waveform of the part of the sound the timeline plays
    if (this.au) {
      const a = this.au, off = this.tl.audio?.offset || 0, mid = g.ruler + g.wave / 2, amp = g.wave / 2 - 2 * g.dpr;
      ctx.fillStyle = "rgba(161,135,183,.55)";
      for (let x = Math.floor(g.x0); x < g.x1; x++) {
        const t0 = off + this.xT(x), t1 = off + this.xT(x + 1); if (t1 < 0 || t0 > a.dur) continue;
        const i0 = Math.max(0, Math.floor(t0 / a.dur * a.N)), i1 = Math.min(a.N - 1, Math.max(i0, Math.floor(t1 / a.dur * a.N)));
        let lo = 0, hi = 0; for (let i = i0; i <= i1; i++) { if (a.mn[i] < lo) lo = a.mn[i]; if (a.mx[i] > hi) hi = a.mx[i]; }
        ctx.fillRect(x, mid - hi * amp, 1, Math.max(1, (hi - lo) * amp));
      }
    }
    // H3 clips: bars across the top of the ruler
    (this.tl.clips || []).forEach(c => {
      const a = this.tX(c.t_start), b = this.tX(Math.min(c.t_end, D));
      ctx.fillStyle = c.on === false ? 'rgba(128,128,128,.35)' : c.mode === 'reference' ? 'rgba(161,135,183,.75)' : 'rgba(241,138,88,.75)';
      ctx.fillRect(a, 12 * g.dpr, Math.max(3, b - a), 5 * g.dpr);
    });
    // markers: orange flags with their names
    this.tl.markers.forEach((m, i) => {
      const x = Math.round(this.tX(m.t)) + .5, sel = this.tlMark === i;
      ctx.strokeStyle = ORANGE; ctx.lineWidth = sel ? 2 * g.dpr : 1; ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, c.height); ctx.stroke();
      ctx.fillStyle = ORANGE; ctx.beginPath(); ctx.moveTo(x, g.ruler - 12 * g.dpr); ctx.lineTo(x + 7 * g.dpr, g.ruler - 8 * g.dpr); ctx.lineTo(x, g.ruler - 4 * g.dpr); ctx.fill();
      if (m.name) { ctx.fillStyle = sel ? ORANGE : ink; ctx.fillText(m.name.slice(0, 18), x + 9 * g.dpr, g.ruler - 8 * g.dpr); }
      if (m.anchor) { const s_ = 4 * g.dpr; ctx.save(); ctx.translate(x, g.ruler - 1 * g.dpr); ctx.rotate(Math.PI / 4); ctx.fillStyle = ORANGE;   // a pinned frame
        ctx.fillRect(-s_ / 2, -s_ / 2, s_, s_); ctx.restore(); }
    });
    const L = this.cur();
    if (g.mrow) this.drawClipRow(ctx, g, L, ink);
    if (L?.video?.on) {
      const a = this.tX(Math.max(0, L.video.t_start || 0)), b = this.tX(L.video.t_end >= 0 ? L.video.t_end : this.tl.duration);
      ctx.fillStyle = "rgba(241,138,88,.35)"; ctx.fillRect(a, g.ruler - 3 * g.dpr, Math.max(2, b - a), 3 * g.dpr);
    }
    ctx.fillStyle = ink;
    if (this.au) ctx.fillText((this.tl.audio?.file || "").split("/").pop().slice(0, 20), 6 * g.dpr, g.ruler + g.wave / 2);
    if (!rows.length) { ctx.fillText(this.autokey ? "recording: changes become keyframes" : "no keyframes on this layer · K keys it, ● records changes", 6 * g.dpr, g.top + g.row / 2 + 2 * g.dpr); }
    rows.forEach(({ p, keys }, i) => {                 // (a stagger row: steps over its regions)
      const y = g.top + i * g.row + g.row / 2 + 2 * g.dpr;
      const mv = (L.motion || []).filter(b => b.type !== "loop" && b.type !== "repeat" && b.path === p);
      ctx.fillStyle = ink; ctx.fillText((mv.length ? "~ " : "") + this.pathLabel(L, p).slice(0, 22), 6 * g.dpr, y);
      ctx.strokeStyle = line; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(g.x0, y); ctx.lineTo(g.x1, y); ctx.stroke();
      mv.forEach(b => {                              // behaviours: a wave over their range (a straight line for drift)
        const [t0, t1] = mSpan(b, D), a = this.tX(t0), e = this.tX(t1), amp = 3.5 * g.dpr;
        ctx.strokeStyle = b.on === false ? "rgba(128,128,128,.5)" : LAV; ctx.lineWidth = 1.5 * g.dpr; ctx.beginPath();
        if (b.type === "drift") { ctx.moveTo(a, y - amp); ctx.lineTo(e, y - amp); }
        else if (b.type === "stagger") { const n = Math.max(1, L._stg?.regs?.length || this.selectIds(L.clip || "").size || 1), st = Math.max(0, mNum(b, "step", 0.1));
          const k = Math.min(n, 60), e2 = Math.min(e, this.tX(t0 + n * st)); ctx.moveTo(a, y + amp);
          for (let j = 0; j < k; j++) { const x0 = a + (e2 - a) * j / k, x1 = a + (e2 - a) * (j + 1) / k, yy = y + amp - 2 * amp * (j + 1) / k; ctx.lineTo(x0, yy); ctx.lineTo(x1, yy); }
          if ((b.mode || "sequence") !== "sequence") ctx.lineTo(e, y + amp); }
        else if (b.type === "pulse") { const mc = this.motionCtx(), ts = mTriggers(b, mc) || (() => { const st = Math.max(0.02, mNum(b, "every", 0.5)), o = []; for (let q = t0; q <= t1 + 1e-9 && o.length < 5000; q += st) o.push(q); return o; })();
          for (const q of ts) if (q >= t0 - 1e-9 && q <= t1) { const x = this.tX(q); ctx.moveTo(x, y + amp); ctx.lineTo(x, y - 2 * amp); } }
        else if (b.type === "audio") { const mc = this.motionCtx(), sm = Math.max(0, mNum(b, "smooth", 0.15));
          for (let x = a; x <= e; x += 2) { const v = Math.min(1.5, mLevel(mc, this.xT(x), sm)); x === a ? ctx.moveTo(x, y + amp - v * 2 * amp) : ctx.lineTo(x, y + amp - v * 2 * amp); } }
        else for (let x = a; x <= e; x += 2) { const q = (x - a) / (7 * g.dpr), v = b.type === "random" ? (Math.floor(q) % 2 ? 1 : -1) : Math.sin(q);
          x === a ? ctx.moveTo(x, y + v * amp) : ctx.lineTo(x, y + v * amp); }
        ctx.stroke();
      });
      const loop = loopModes(L.motion || [])[p];
      if (loop && keys.length > 1) {                 // looped keys: dashed after the last one
        const a = this.tX(keys[keys.length - 1].t); ctx.save(); ctx.setLineDash([4 * g.dpr, 3 * g.dpr]); ctx.strokeStyle = LAV; ctx.lineWidth = 1.5 * g.dpr;
        ctx.beginPath(); ctx.moveTo(a, y); ctx.lineTo(g.x1, y); ctx.stroke(); ctx.restore();
        ctx.fillStyle = LAV; ctx.fillText(loop === "continue" ? "→" : loop === "pingpong" ? "⇄" : "↻", Math.min(g.x1 - 12 * g.dpr, a + 6 * g.dpr), y - 5 * g.dpr);
      }
      keys.forEach((k, j) => {
        const x = this.tX(k.t), s = 6 * g.dpr, sel = this.tlSel && this.tlSel.p === p && this.tlSel.i === j;
        ctx.save(); ctx.translate(x, y); ctx.fillStyle = sel ? ORANGE : LAV;
        if (k.e === "hold") ctx.fillRect(-s / 2, -s / 2, s, s);                       // square = hold
        else { ctx.rotate(Math.PI / 4); ctx.fillRect(-s / 2, -s / 2, s, s); if (k.e === "linear") { ctx.fillStyle = "#fff"; ctx.fillRect(-s / 5, -s / 5, s / 2.5, s / 2.5); } }
        ctx.restore();
      });
    });
    const x = Math.round(this.tX(this.tl.time)) + .5;
    ctx.strokeStyle = ORANGE; ctx.lineWidth = 2 * g.dpr; ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, c.height); ctx.stroke();
    const beatNo = beats.length ? beats.filter(b => b <= this.tl.time + 1e-6).length : 0;
    const t = this.$("tl-time"); if (t) t.textContent = `${this.tl.time.toFixed(2)} s · frame ${Math.round(this.tl.time * this.tl.fps)}` +
      (beatNo ? ` · bar ${Math.floor((beatNo - 1) / 4) + 1}.${(beatNo - 1) % 4 + 1}` : "");
  }
  drawClipRow(ctx, g, L, ink) {                      // video segments: solid = the file plays, faint = loop / hold fill
    const y0 = g.mtop + 3 * g.dpr, hh = g.mrow - 6 * g.dpr, D = this.tl.duration, fmt = v => (+v).toFixed(v < 10 ? 2 : 1);
    ctx.fillStyle = ink; ctx.fillText("video", 6 * g.dpr, y0 + hh / 2);
    for (const s of this.segsOf(L)) {
      const a = this.tX(Math.min(D, s.start)), n = this.tX(Math.min(D, s.start + Math.min(s.length, s.natural))), b = this.tX(Math.min(D, s.start + s.length));
      ctx.fillStyle = "rgba(161,135,183,.9)"; ctx.fillRect(a, y0, Math.max(2, n - a), hh);
      if (b > n + 0.5) { ctx.fillStyle = s.fill === "loop" ? "rgba(161,135,183,.5)" : "rgba(161,135,183,.25)"; ctx.fillRect(n, y0, b - n, hh);
        if (s.fill === "loop") { ctx.strokeStyle = "rgba(255,255,255,.5)"; ctx.lineWidth = 1;
          for (let x = n + s.natural / D * (g.x1 - g.x0); x < b - 1; x += Math.max(4, s.natural / D * (g.x1 - g.x0))) { ctx.beginPath(); ctx.moveTo(x, y0); ctx.lineTo(x, y0 + hh); ctx.stroke(); } } }
      ctx.fillStyle = "rgba(0,0,0,.35)"; ctx.fillRect(a, y0, 3 * g.dpr, hh); ctx.fillRect(Math.max(a, b - 3 * g.dpr), y0, 3 * g.dpr, hh);   // grips
      const label = `${s.reverse ? "◀ " : ""}${s.speed !== 1 ? s.speed.toFixed(2).replace(/\.?0+$/, "") + "× · " : ""}${fmt(s.in)}–${fmt(s.out)} s${b > n + 0.5 ? " · " + s.fill : ""}`;
      if (b - a > ctx.measureText(label).width + 10 * g.dpr) { ctx.fillStyle = "#fff"; ctx.fillText(label, a + 6 * g.dpr, y0 + hh / 2); }
      if (this.segSel === s.i) { ctx.strokeStyle = ORANGE; ctx.lineWidth = 2 * g.dpr; ctx.strokeRect(a + g.dpr, y0 + g.dpr, Math.max(2, b - a) - 2 * g.dpr, hh - 2 * g.dpr); }
    }
  }
  dragSeg(td, rawT, e) {                             // move / trim in / trim out (Alt: speed, Shift: length with loop / hold)
    const { raw: r, s0, edge } = td, dur = td.L.vinfo.duration || s0.out, fps = this.tl.fps, dt = rawT - td.t0;
    const q = v => Math.round(v * fps) / fps, r4 = v => +(+v).toFixed(4);
    if (!edge) { r.start = r4(Math.max(0, this.snapT(q(s0.start + dt), e))); }
    else if (edge === "l") {
      const ns = this.snapT(q(s0.start > this.tl.duration ? rawT : s0.start + dt), e);
      const nin = Math.min(Math.max(0, s0.in + (ns - s0.start) * s0.speed), s0.out - 1 / fps);
      const start = s0.start + (nin - s0.in) / s0.speed;
      r.in = r4(nin); r.start = r4(Math.max(0, start));
      if (+r.length > 0) r.length = r4(Math.max(1 / fps, s0.length - (start - s0.start)));
    } else {
      const e0 = s0.start + s0.length, end = this.snapT(q(e0 > this.tl.duration ? rawT : e0 + dt), e), len = Math.max(1 / fps, end - s0.start);
      if (td.alt) { r.speed = r4(Math.min(20, Math.max(0.05, (s0.out - s0.in) / len))); r.length = -1; }
      else if (td.shift) r.length = r4(len);
      else { r.out = r4(Math.min(dur, Math.max(s0.in + 1 / fps, s0.in + len * s0.speed))); r.length = -1; }
    }
    this.applyTime(this.tl.time); this.drawNow(); this.drawTimeline();
  }
  splitVideo() {                                     // Ctrl K: cut the selected video layer at the playhead
    const L = this.cur(), t = this.tl.time; if (!L?.vinfo) { note("select a video layer to cut it"); return; }
    const at = mediaTime(this.segsOf(L), t); if (!at) { note("no video under the playhead on this layer"); return; }
    const [s, ts] = at; if (t - s.start < 1 / this.tl.fps || s.start + s.length - t < 1 / this.tl.fps) return;
    this.snap();
    const raw = L.media.segments[s.i], inFill = (t - s.start) >= s.natural - 1e-9;
    const A = { ...raw }, B = { ...raw, start: +t.toFixed(4) };
    if (!inFill && s.reverse) { A.in = +ts.toFixed(4); A.length = -1; B.out = +ts.toFixed(4); B.length = raw.length > 0 ? +(s.start + s.length - t).toFixed(4) : -1; }
    else if (!inFill) { A.out = +ts.toFixed(4); A.length = -1; B.in = +ts.toFixed(4); B.length = raw.length > 0 ? +(s.start + s.length - t).toFixed(4) : -1; }
    else {
      A.length = +(t - s.start).toFixed(4); B.length = +(s.start + s.length - t).toFixed(4);
      if (s.fill === "hold") {                       // the frozen last frame stays frozen
        const f = 1 / (L.vinfo?.fps || this.tl.fps || 25);
        B.in = +Math.max(s.in, s.out - f).toFixed(4); B.out = +s.out.toFixed(4); B.speed = 1;
      }                                              // loop: the second piece starts the loop again from 'in'
    }
    L.media.segments.splice(s.i, 1, A, B); this.segSel = s.i + 1;
    this.applyTime(t); this.renderInspector(); this.draw(); this.drawTimeline(); this.scheduleDraft();
  }
  diffDefaults(d) {                                  // the diffusion settings travel with the composition
    const pre = this.m.diffusion?.presets || [];
    const out = Object.assign({ preset: pre[0]?.name || "klein 4b", steps: pre[0]?.steps || 4, megapixels: 1, seed: -1, band: 12,
      context: 0.2, reference: true }, d || {});
    out.loras = [0, 1, 2].map(i => Object.assign({ name: "", strength: 1, on: true }, (d?.loras || [])[i] || {}));
    return out;
  }

  async open() {
    injectStyle();
    this.buildDom();
    this.busy(true, "loading");
    try {
      for (const [k, r] of Object.entries(this.m.views || {})) {
        const im = await loadImage(viewURL(r)); if (this.closed) return;
        if (k === "_labels") this.decodeLabels(im); else this.views[k] = im;
      }
      this.inputImgs = {};
      for (const inp of this.m.inputs || []) { this.inputImgs[inp.key] = await loadImage(viewURL(inp.image)); if (this.closed) return; }
      const start = await this.pickStart(); if (this.closed) return;
      await this.loadDocument(start); if (this.closed) return;
    } catch (e) { if (this.closed) return; console.error("[kubakub director]", e); note("the director could not load its images: run the workflow once more"); this.close(); return; }
    this.busy(false); this.ready = true;
    if (this.labels) this.views.regions = this.regionsView();
    this.fillRegionList();
    this.renderViews(); this.refresh(); this.fit(); this.fillTl(); if (this.tl.audio) this.loadAudio();
  }

  // ---- document <-> layers
  widget() { return this.node.widgets?.find(w => w.name === "document"); }
  async loadDocument(override) {
    let doc = {};
    if (override) doc = override;
    else { try { doc = JSON.parse(this.widget()?.value || "{}") || {}; } catch { doc = {}; } }
    let layers = (Array.isArray(doc.layers) ? doc.layers : []).filter(l => l && typeof l === "object" && !Array.isArray(l));
    if (!layers.some(l => l.kind === "base")) layers.push({ id: "base", name: "base", kind: "base" });
    // inputs that are not in the document yet land on top, at their own size (deleted ones stay deleted)
    this.dismissed = Array.isArray(doc.dismissed) ? doc.dismissed.filter(k => typeof k === "string") : [];
    this.diff = this.diffDefaults(doc.diffusion);
    const T = doc.timeline || {};
    this.tl = { duration: Math.max(0.5, Math.min(3600, +T.duration || 10)), fps: Math.max(1, Math.min(120, Math.round(+T.fps || 25))), time: 0 };
    this.tl.time = Math.max(0, Math.min(this.tl.duration, +T.time || 0));
    this.tl.markers = (Array.isArray(T.markers) ? T.markers : []).filter(q => q && Number.isFinite(+q.t)).map(q => ({ t: +q.t, name: String(q.name || ""), ...(q.anchor ? { anchor: true } : {}) }));
    this.tl.audio = T.audio?.file ? { file: String(T.audio.file), offset: +T.audio.offset || 0, gain: Number.isFinite(+T.audio.gain) ? +T.audio.gain : 1 } : null;
    this.tl.beat = T.beat?.bpm > 0 ? { bpm: +T.beat.bpm, phase: +T.beat.phase || 0 } : null;
    this.tl.snap = T.snap !== false;
    this.tl.clips = (Array.isArray(T.clips) ? T.clips : []).filter(c => c && Number.isFinite(+c.t_start) && Number.isFinite(+c.t_end))
      .map((c, i) => ({ id: String(c.id || 'c' + (i + 1)), t_start: +c.t_start, t_end: +c.t_end, mode: c.mode === 'reference' ? 'reference' : 'keyframes',
        prompt: String(c.prompt || ''), audio: String(c.audio || ''), refs: Array.isArray(c.refs) ? c.refs.map(String) : [], raw: !!c.raw, on: c.on !== false }));
    this.tl.h3 = Object.assign({ look: '', camera: '', avoid: '', audio: '' }, T.h3 && typeof T.h3 === 'object' ? T.h3 : {});
    this.pm = Object.assign({ on: false, source: "", invert: false, grow: 0, feather: 0, view: "black" }, doc.projection_mask && typeof doc.projection_mask === "object" ? doc.projection_mask : {});
    this.loadPm();
    const have = new Set([...layers.map(l => l.source), ...this.dismissed]);
    const fresh = (this.m.inputs || []).filter(i => !have.has(i.key)).map(i => this.inputLayer(i));
    layers = [...fresh, ...layers];
    const before = this.layers;
    this.layers = await Promise.all(layers.map(d => this.hydrate(d)));          // in parallel, order kept
    if (before) this.quietVideos(before);
    this.dropDangling();
    for (const L of this.layers) if (L.fullRender) { L.fullKey = this.lightKey(L, true); delete L.fullRender; }  // glow layers exist now
    this.syncEval();
    if (this.layers.some(L => L.motion?.length)) this.applyTime(this.tl.time);   // behaviours at the restored playhead
    this.syncVideos(false);                          // every video at its frame for the restored playhead
    this.uid = 1 + Math.max(0, ...this.layers.map(l => +(String(l.id).match(/\d+$/) || [0])[0]));
    this.sel = Math.max(0, this.layers.findIndex(l => l.kind !== "base"));
  }
  dropDangling() {
    const ids = new Set(this.layers.map(q => q.id)), ok = r => ids.has(String(r).slice(6));
    for (const q of this.layers) {
      if (q.mask?.by && q.mask.by !== "below" && !ids.has(q.mask.by)) q.mask.by = "";
      if (q.light) {
        if (Array.isArray(q.light.glow)) q.light.glow = q.light.glow.filter(g => !String(g.by).startsWith("layer:") || ok(g.by));
        if (q.light.projector && String(q.light.projector.source).startsWith("layer:") && !ok(q.light.projector.source)) q.light.projector.source = "below";
      }
    }
    for (const c of this.tl.clips || []) c.refs = (c.refs || []).filter(r => !String(r).startsWith("layer:") || ok(r));
  }
  quietVideos(old) {                                 // pause elements no current layer owns (undo may bring them back)
    const live = new Set((this.layers || []).map(q => q.img));
    for (const L of old || []) { const v = L.img; if (v instanceof HTMLVideoElement && !live.has(v)) { if (!v.paused) v.pause(); v.muted = true; } }
  }
  inputLayer(inp) {
    const full = inp.w === this.W && inp.h === this.H;
    const s = full ? 1 : Math.min(1, (this.W * 0.4) / inp.w, (this.H * 0.5) / inp.h);
    const w = inp.w * s, h = inp.h * s;
    return { id: "in" + inp.key.split(":")[1], name: inp.name, kind: "image", source: inp.key, x: full ? 0 : (this.W - w) / 2,
      y: full ? 0 : (this.H - h) / 2, w, h, action: inp.is_mask ? "keep" : "rediffuse", visible: !inp.is_mask };
  }
  defaults(d) {
    return Object.assign({ source: "", visible: true, opacity: 1, blend: "normal", x: 0, y: 0, w: 0, h: 0, rotation: 0, flip_h: false, flip_v: false,
      clip: "", holes: "", mask: { by: "", mode: "shape", invert: false }, action: d.kind === "image" || d.kind === "paint" ? "rediffuse" : "keep",
      prompt: "", denoise: 0.5, adjust: { brightness: 1, contrast: 1, saturate: 1, hue: 0, sepia: 0 } }, d,
      { mask: Object.assign({ by: "", mode: "shape", invert: false }, d.mask || {}),
        adjust: Object.assign({ brightness: 1, contrast: 1, saturate: 1, hue: 0, sepia: 0 }, d.adjust || {}) });
  }
  async attachVideo(L) {                             // a small proxy of the file plays in a <video>; the node renders the file itself
    const r = await api.fetchApi("/kubakub/director/media", { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ source: L.source }) });
    const j = await r.json().catch(() => ({ error: "no answer (" + r.status + ")" }));
    if (!r.ok || j.error) throw new Error(j.error || r.status);
    const v = document.createElement("video");
    Object.assign(v, { muted: true, playsInline: true, preload: "auto", crossOrigin: "anonymous", loop: false });
    (this.videos ||= new Set()).add(v);              // every element this window made: paused when orphaned, released on close
    v.addEventListener("seeked", () => {             // undo / redo rebuild layer objects: find the owner by its element
      const q = (this.layers || []).find(z => z.img === v);
      if (q?.vpend != null) { const p = q.vpend; q.vpend = null; if (Math.abs(v.currentTime - p) > 1e-4) { v.currentTime = p; return; } }
      if (q && !this.playing) this.draw();           // while playing the tick draws every frame anyway
    });
    await new Promise((res, rej) => { v.addEventListener("loadeddata", res, { once: true }); v.addEventListener("error", () => rej(new Error("the preview copy does not play")), { once: true });
      v.src = stableURL(j.proxy); });
    L.img = v; L.vinfo = j.info; L.media = this.mediaDefaults(L.media);
    return L;
  }
  mediaDefaults(m) {
    const out = Object.assign({ volume: 1, blend_frames: false }, m && typeof m === "object" ? m : {});
    out.segments = (Array.isArray(m?.segments) && m.segments.length ? m.segments : [{ start: 0 }]).map(s => ({ start: 0, in: 0, out: -1, speed: 1, length: -1, fill: "hold", reverse: false, ...s }));
    return out;
  }
  segsOf(L) { return mediaSegs(L.media, L.vinfo?.duration || 0); }
  usedPictures() {                                   // layers other layers use as a picture: masks, projector, glow
    const layers = this.layers || [], used = new Set(), byId = id => layers.find(x => x.id === id);
    layers.forEach((q, i) => {
      if (!q.visible) return;
      const m = this.maskSource(i); if (m) used.add(m);
      if (q.kind === "light" && q.light) {
        const P = q.light.projector; if (P?.on && String(P.source).startsWith("layer:")) { const z = byId(P.source.slice(6)); if (z) used.add(z); }
        for (const g of q.light.glow || []) if (g.on !== false && String(g.by).startsWith("layer:")) { const z = byId(g.by.slice(6)); if (z) used.add(z); }
      }
    });
    return used;
  }
  syncVideos(playing) {                              // every video layer shows its frame for the timeline time
    const t = this.tl.time, layers = this.layers || [], used = this.usedPictures(), live = new Set();
    for (const L of layers) {
      const v = L.img; if (!L.vinfo || !(v instanceof HTMLVideoElement)) continue;
      live.add(v);
      const at = (L.visible || used.has(L)) ? mediaTime(this.segsOf(L), t) : null;     // a hidden matte still plays
      L.vhidden = !at || !Number.isFinite(at[1]);
      if (L.vhidden) { if (!v.paused) v.pause(); continue; }
      const [s, ts, held] = at, fps = L.vinfo.fps || 25;
      if (playing && !held && s.speed <= 16 && !s.reverse) {       // the browser plays it, nudged towards the playhead
        const drift = ts - v.currentTime;
        if (Math.abs(drift) > 0.15 * Math.max(1, s.speed) || v.currentTime > s.out + 0.02 || v.currentTime < s.in - 0.05) v.currentTime = ts;
        const rate = Math.min(16, Math.max(0.0625, s.speed * (1 + Math.max(-0.2, Math.min(0.2, drift)))));
        if (Math.abs(v.playbackRate - rate) > 0.005) v.playbackRate = rate;
        this.videoSound(v, L.visible ? Math.max(0, +(L.media?.volume ?? 1)) : 0, Math.abs(s.speed - 1) > 1e-3);
        if (v.paused) v.play().catch(() => {});
      } else {                                       // paused, held, backwards or too fast for the browser: exact frames
        if (!v.paused) v.pause(); v.muted = true;
        const fi = Math.max(0, Math.min(Math.floor(ts * fps + 1e-6), (L.vinfo.frames || 1e9) - 1));   // = media.frame_pick
        const want = (fi + 0.5) / fps;               // the middle of that frame: never lands on the one before
        if (Math.abs(v.currentTime - want) > 0.3 / fps) { if (v.seeking) L.vpend = want; else v.currentTime = want; }
      }
    }
    for (const v of this.videos || []) if (!live.has(v) && !v.paused) { v.pause(); v.muted = true; }   // deleted / undone layers stay quiet
  }
  videoSound(v, vol, silent) {                       // through a gain node: 0-200 %, as the render mixes it
    if (silent || !(vol > 0)) { v.muted = true; return; }
    try {
      this.actx = this.actx || new (window.AudioContext || window.webkitAudioContext)();
      if (!v._gain) { const src = this.actx.createMediaElementSource(v); v._gain = this.actx.createGain(); src.connect(v._gain).connect(this.actx.destination); }
      v._gain.gain.value = vol; v.volume = 1; v.muted = false; this.actx.resume?.();
    } catch { v.volume = Math.min(1, vol); v.muted = false; }
  }
  async hydrate(d) {
    const L = this.defaults(d);
    L.source = String(L.source || "");
    if (isVideo(L)) {
      try { await this.attachVideo(L); } catch (e) { console.error("[kubakub director] video", e); L.img = null; L.vinfo = null; L.verr = e.message; }
      if (L.img && !L.w) { L.w = L.vinfo.w; L.h = L.vinfo.h; }
    } else if (L.kind === "image") {
      if (L.source.startsWith("input:")) L.img = this.inputImgs[L.source] || null;
      else if (L.source.startsWith("file:")) { try { L.img = await loadImage(inputURL(L.source.slice(5))); } catch { L.img = null; } }
      else L.img = null;
      // input previews are proxy-sized, imported files full size
      if (L.img && !L.w) { const d = L.source.startsWith("input:") ? this.k : 1; L.w = iw(L.img) / d; L.h = ih(L.img) / d; }
      if (L.img && L.remove_bg) this.cutout(L, true);
    } else if (L.kind === "paint") {
      L.canvas = mkCanvas(this.PW, this.PH); L.ver = 0;
      if (L.source.startsWith("file:")) { try { L.canvas.getContext("2d").drawImage(await loadImage(inputURL(L.source.slice(5))), 0, 0, this.PW, this.PH); } catch { /* new */ } }
      L.color = L.color || ORANGE; L.size = L.size || 60;
    } else if (L.kind === "light") {
      L.light = this.rigDefaults(L.light);
      const done = this.m.lights?.[L.id];            // the node's full-size render, if the rig has not changed since
      if (done && stable(this.rigDefaults(done.rig)) === stable(L.light)) {
        try { L.img = await loadImage(viewURL(done.image)); L.fullRender = true; L.status = "full size render from the last run"; } catch { L.img = null; }
      }
    }
    return L;
  }
  serialize() {
    const keys = ["id", "name", "kind", "source", "visible", "opacity", "blend", "x", "y", "w", "h", "rotation", "flip_h", "flip_v",
      "clip", "holes", "mask", "adjust", "action", "prompt", "denoise", "color", "size", "remove_bg", "light", "anim", "video", "media", "shape", "fx", "light_react", "motion"];
    const doc = { version: 1, canvas: [this.W, this.H],
      layers: this.layers.map(L => { const o = Object.fromEntries(keys.filter(k => L[k] !== undefined).map(k => [k, L[k]]));
        for (const p of Object.keys(L._rest || {})) {                       // saved without the behaviours (the node adds them per frame)
          const top = p.split(".")[0]; if (o[top] && typeof o[top] === "object" && o[top] === L[top]) o[top] = JSON.parse(JSON.stringify(o[top]));
          pathSet(o, p, this.restOf(L, p));
        }
        return o; }) };
    if (this.dismissed.length) doc.dismissed = [...this.dismissed];
    doc.timeline = { ...this.tl };
    doc.diffusion = JSON.parse(JSON.stringify(this.diff));
    if (this.pm.on || this.pm.source) doc.projection_mask = { ...this.pm };
    return doc;
  }

  // ---- light layers: a Cycles relight of the connected scene, rig in facade metres
  rigDefaults(r) {
    const d = this.m.scene?.defaults || {};
    const rig = Object.assign({ environment: "night", env_strength: 0.3, env_rotation: 0, exposure: 0, clay: 0.7, roughness: 0.8,
      samples: 64, background: "black", resolution_scale: 1, hdri_file: "" }, d, r || {});
    rig.lights = (Array.isArray(r?.lights) ? r.lights : [
      { type: "area", x: -this.facadeW() * 0.28, height: this.facadeTop() * 0.3, ...LIGHT_DEF.area },
      { type: "area", x: this.facadeW() * 0.28, height: this.facadeTop() * 0.3, ...LIGHT_DEF.area }])
      .filter(q => q && typeof q === "object").map((q, i) => ({ on: true, ...q, id: q.id || "p" + (i + 1) }));
    rig.glow = (Array.isArray(r?.glow) ? r.glow : []).filter(g => g && typeof g === "object").map((g, i) => ({ by: "", color: "#ffb060", strength: 20, on: true, ...g, id: g.id || "g" + (i + 1) }));
    for (const list of [rig.lights, rig.glow]) {       // ids stay unique (keyframes address lamps by id)
      const seen = new Set(); for (const e of list) { while (seen.has(e.id)) e.id += "x"; seen.add(e.id); }
    }
    rig.projector = Object.assign({ on: false, source: "below", brightness: 1, mode: "light" }, r?.projector || {});
    if (!["AgX", "Standard"].includes(rig.view)) rig.view = "AgX";
    return rig;
  }
  frame() { return this.m.scene?.frame || null; }
  facadeW() { return this.frame()?.width_m || 30; }
  facadeTop() { return this.frame()?.top_m || 20; }
  toFacade(px, py) {                                  // canvas pixel -> x along the wall (m), height above the ground (m)
    const f = this.frame(); if (!f) return [0, 0];
    return [(px / this.W - 0.5) * f.width_m, f.top_m - py / this.H * (f.top_m - f.bottom_m)];
  }
  fromFacade(x, h) {
    const f = this.frame(); if (!f) return [this.W / 2, this.H / 2];
    return [(x / f.width_m + 0.5) * this.W, (f.top_m - h) / Math.max(1e-6, f.top_m - f.bottom_m) * this.H];
  }
  lightKey(L, ignoreQuality) {                        // what the preview depends on: the rig and the glowing layers
    const deps = (L.light.glow || []).filter(g => g.on !== false && String(g.by).startsWith("layer:")).map(g => {
      const q = this.layers?.find(z => z.id === g.by.slice(6));
      return q ? [q.id, q.source, q.x, q.y, q.w, q.h, q.rotation, q.flip_h, q.flip_v, q.clip, q.ver || 0] : null;
    });
    return stable({ light: L.light, deps, cast: this.castDeps(L), q: ignoreQuality ? "" : this.lq });
  }
  castLayers(L) {                                    // indices of what the projector casts (null = the matrix alone)
    const P = L.light.projector; if (!P?.on) return [];
    if (P.source === "base") return null;
    if (String(P.source).startsWith("layer:")) return this.layers.map((q, i) => q.id === P.source.slice(6) ? i : -1).filter(i => i >= 0);
    const i = this.layers.indexOf(L); return this.layers.map((q, j) => j > i && q.visible ? j : -1).filter(j => j >= 0);
  }
  castDeps(L) {
    const idx = this.castLayers(L); if (idx === null) return "matrix"; if (!idx.length) return null;
    return idx.map(i => { const q = this.layers[i];
      return [q.id, q.kind, q.source, q.visible, q.opacity, q.blend, q.x, q.y, q.w, q.h, q.rotation, q.flip_h, q.flip_v, q.clip, q.holes,
        stable(q.mask), stable(q.adjust), q.ver || 0, q.kind === "light" ? q.imgKey || "" : "", !!q.remove_bg, q.img ? 1 : 0,
        q.motion ? stable(q.motion) : "", q.motion?.some(b => b.on !== false && (b.type === "stagger" || b.type === "repeat" && mNum(b, "delay", 0))) ? this.tl.time : ""]; });
  }
  castImage(L) {                                     // the window's own composite of what is cast, as a PNG data URL
    const idx = this.castLayers(L), P = L.light.projector;
    if (!P?.on) return null;
    const c = mkCanvas(this.PW, this.PH);
    this._forceMatrix = true;
    try {
      if (idx === null) c.getContext("2d").drawImage(this.views.matrix, 0, 0, this.PW, this.PH);
      else this.composite(c, idx, P.source !== "below");
    } finally { this._forceMatrix = false; }
    return c.toDataURL("image/png");
  }
  checkLights() {                                    // re-render light layers whose rig or glow sources changed
    if (!this.m.scene || !this.layers || this.playing) return;
    for (const L of this.layers) {
      if (L.kind !== "light" || (!L.visible && !this.reactLight(L))) continue;
      const key = this.lightKey(L);
      if (key === L.imgKey || key === L.pendingKey) continue;
      if (L.fullKey && L.fullKey === this.lightKey(L, true)) continue;      // the node's full-size render still fits
      clearTimeout(this.lightTimers.get(L.id));
      this.lightTimers.set(L.id, setTimeout(() => this.renderLight(L), 450));
    }
  }
  async renderLight(L) {
    if (!this.layers?.includes(L) || !this.root?.isConnected) return;
    const key = this.lightKey(L); if (key === L.imgKey) return;
    L.pendingKey = key; L.status = "rendering…"; L.err = false; this.updateLightStatus(L);
    const t0 = performance.now();
    try {
      if ((L.light.glow || []).some(g => String(g.by).startsWith("layer:") && this.layers.find(q => q.id === g.by.slice(6))?.kind === "paint")) await this.uploadPaint();
      const qa = QUALITY[this.lq] || QUALITY.fast;
      const r = await api.fetchApi("/kubakub/director/relight", { method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ node: this.m.node, layer: L.id, doc: this.serialize(), px: qa.px, samples: qa.samples, projector: this.castImage(L) }) });
      const j = await r.json().catch(() => ({ error: "no answer (" + r.status + ")" }));
      if (!r.ok || j.error) throw new Error(j.error || r.status);
      const im = await loadImage(viewURL(j.image));
      if (L.pendingKey !== key) return;                // a newer edit is on its way
      L.img = im; L.imgKey = key; L.pendingKey = null; L.fullKey = null;
      L.status = `${this.lq} preview · ${((performance.now() - t0) / 1000).toFixed(1)} s`;
    } catch (e) {
      if (L.pendingKey !== key) return;
      console.error("[kubakub director] relight", e);
      L.imgKey = key; L.pendingKey = null; L.err = true; L.status = "relight failed: " + e.message;
    }
    this.updateLightStatus(L); this.renderLayers(); this.draw();
  }
  // ---- diffusion from the window: the layer's area goes through Klein (+ LoRAs) and comes back as a layer
  diffusable(L) { return L && (L.kind === "image" && L.img || L.kind === "paint"); }
  shapeCanvas(L) {                                   // the layer alone (placement, clip), proxy size
    const c = mkCanvas(this.PW, this.PH), g = c.getContext("2d"); g.save(); this.drawContent(g, L); g.restore(); return c;
  }
  diffuseBox(L, shape) {                             // full-size box around the layer's pixels + context
    let x0, y0, x1, y1;
    if (L.kind === "image") [x0, y0, x1, y1] = this.aabb(L);
    else {
      const a = shape.getContext("2d", { willReadFrequently: true }).getImageData(0, 0, this.PW, this.PH).data;
      x0 = this.PW; y0 = this.PH; x1 = -1; y1 = -1;
      for (let y = 0, p = 3; y < this.PH; y++) for (let x = 0; x < this.PW; x++, p += 4) if (a[p] > 8) { if (x < x0) x0 = x; if (x > x1) x1 = x; if (y < y0) y0 = y; if (y > y1) y1 = y; }
      if (x1 < 0) return null;
      [x0, y0, x1, y1] = [x0 / this.k, y0 / this.k, (x1 + 1) / this.k, (y1 + 1) / this.k];
    }
    const pad = Math.max(32, this.diff.context * Math.max(x1 - x0, y1 - y0));
    x0 = Math.max(0, Math.floor(x0 - pad)); y0 = Math.max(0, Math.floor(y0 - pad));
    x1 = Math.min(this.W, Math.ceil(x1 + pad)); y1 = Math.min(this.H, Math.ceil(y1 + pad));
    return x1 - x0 >= 16 && y1 - y0 >= 16 ? { x: x0, y: y0, w: x1 - x0, h: y1 - y0 } : null;
  }
  diffGraph(image, mask, L, request) {
    const pre = (this.m.diffusion?.presets || []).find(p => p.name === this.diff.preset) || this.m.diffusion?.presets?.[0];
    if (!pre) throw new Error("no Klein model found (flux-2-klein-4b / 9b with qwen_3 text encoder and flux2 vae)");
    const g = { u: { class_type: "UNETLoader", inputs: { unet_name: pre.unet, weight_dtype: "default" } },
      c: { class_type: "CLIPLoader", inputs: { clip_name: pre.clip, type: pre.type, device: "default" } },
      v: { class_type: "VAELoader", inputs: { vae_name: pre.vae } } };
    let model = ["u", 0];
    this.diff.loras.filter(l => l.on !== false && l.name && +l.strength !== 0).forEach((l, i) => {
      g["l" + i] = { class_type: "LoraLoaderModelOnly", inputs: { model, lora_name: l.name, strength_model: +l.strength } }; model = ["l" + i, 0]; });
    // comfy kitchen INT8 attention (core falls back to pytorch attention where it is not available)
    g.ab = { class_type: "ModelAttentionBackend", inputs: { model, attention: "comfy kitchen attention" } }; model = ["ab", 0];
    const seed = this.diff.seed >= 0 ? this.diff.seed : Math.floor(Math.random() * 2 ** 32);
    g.d = { class_type: "KUBA_DirectorDiffuse", inputs: { model, clip: ["c", 0], vae: ["v", 0], image, mask, prompt: L.prompt || "",
      mode: L.action === "edges" ? "edges" : "rediffuse", denoise: +L.denoise, steps: +this.diff.steps || pre.steps, seed,
      megapixels: +this.diff.megapixels, band_px: Math.round(+this.diff.band), reference: !!this.diff.reference, request } };
    return { graph: g, seed };
  }
  async diffuse(L) {
    L = L || this.cur();
    if (!this.diffusable(L)) { note("select an image or paint layer to diffuse"); return; }
    if (L.diffusing) return;
    const shape = this.shapeCanvas(L), box = this.diffuseBox(L, shape);
    if (!box) { note("the layer is empty or outside the canvas"); return; }
    L.diffusing = true; L.dstatus = "preparing…"; this.updateDiffStatus(L);
    try {
      const k = this.k, sx = Math.round(box.x * k), sy = Math.round(box.y * k), sw = Math.max(16, Math.round(box.w * k)), sh = Math.max(16, Math.round(box.h * k));
      const comp = mkCanvas(this.PW, this.PH);
      this._forceMatrix = true; try { this.composite(comp); } finally { this._forceMatrix = false; }
      const crop = mkCanvas(sw, sh); crop.getContext("2d").drawImage(comp, sx, sy, sw, sh, 0, 0, sw, sh);
      const mk = mkCanvas(sw, sh), mg = mk.getContext("2d", { willReadFrequently: true });
      mg.drawImage(shape, sx, sy, sw, sh, 0, 0, sw, sh);
      const d = mg.getImageData(0, 0, sw, sh);                          // alpha -> white on black
      for (let p = 0; p < d.data.length; p += 4) { const a = d.data[p + 3]; d.data[p] = d.data[p + 1] = d.data[p + 2] = a; d.data[p + 3] = 255; }
      mg.putImageData(d, 0, 0);
      const blob = c => new Promise(r => c.toBlob(r, "image/png"));
      const cb = await blob(crop), mb = await blob(mk);
      const image = await upload(cb, await contentName(cb, "dcrop"), true), mask = await upload(mb, await contentName(mb, "dmask"), true);
      const request = (crypto.randomUUID?.() || String(Date.now()) + Math.random()).slice(0, 18);
      const { graph, seed } = this.diffGraph(image, mask, L, request);
      const r = await api.fetchApi("/prompt", { method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt: graph, client_id: api.clientId ?? api.initialClientId ?? undefined }) });
      const j = await r.json().catch(() => ({}));
      if (!r.ok || !j.prompt_id) throw new Error(j.error?.message || j.node_errors && Object.values(j.node_errors)[0]?.errors?.[0]?.details || "queue refused (" + r.status + ")");
      this.jobs.set(j.prompt_id, { id: L.id, name: L.name, prompt: L.prompt, denoise: L.denoise, box, request, seed, t0: performance.now() });
      L.dstatus = "queued"; this.updateDiffStatus(L);
    } catch (e) {
      console.error("[kubakub director] diffuse", e); L.diffusing = false; L.dstatus = "diffuse failed: " + e.message; L.derr = true; this.updateDiffStatus(L);
    }
  }
  jobLayer(j) { return this.layers?.find(q => q.id === j.id) || null; }
  async diffuseDone(job, out) {
    const { box, seed } = job; let L = this.jobLayer(job); if (L) L.diffusing = false;
    try {
      const img = await loadImage(inputURL(out.name));
      if (this.closed) return;
      L = this.jobLayer(job);                        // it may have changed while the image loaded
      this.commitReg?.(); this.snap();
      const N = this.defaults({ id: "L" + (this.uid++), name: `${L?.name ?? job.name} · diffused`, kind: "image", source: "file:" + out.name,
        x: box.x, y: box.y, w: box.w, h: box.h, action: "keep", prompt: L?.prompt ?? job.prompt, denoise: L?.denoise ?? job.denoise, seed });
      N.img = img;
      const i = L ? this.layers.indexOf(L) : -1;
      if (i >= 0) { this.layers.splice(i, 0, N); this.sel = i; } else this.insert(N);
      if (L) { L.dstatus = `done in ${out.seconds} s · seed ${seed}`; L.derr = false; }
      this.refresh();
    } catch (e) { if (L) { L.dstatus = "could not load the result: " + e.message; L.derr = true; this.updateDiffStatus(L); } }
  }
  updateDiffStatus(L) {
    if (this.cur() !== L) return;
    const s = this.$("d-status"); if (s) { s.textContent = L.dstatus || ""; s.classList.toggle("err", !!L.derr); }
    const b = this.$("d-go"); if (b) { b.disabled = !!L.diffusing; b.textContent = L.diffusing ? "diffusing…" : "diffuse"; }
  }
  renderDiffPop() {
    const D = this.diff, dm = this.m.diffusion || { presets: [], loras: [] }, pre = dm.presets.find(p => p.name === D.preset) || dm.presets[0];
    const hint = (pre?.lora_hint || "").toLowerCase();
    const fits = n => { const s = n.toLowerCase(); return s.includes(hint) && (s.includes("klein") || s.includes("flux2")); };
    const sorted = [...dm.loras.filter(fits), ...dm.loras.filter(n => !fits(n))];
    const opt = (n, cur) => `<option value="${esc(n)}" ${n === cur ? "selected" : ""}>${esc(n.replace(/\\/g, "/"))}</option>`;
    this.diffpop.innerHTML = `<h3>diffusion</h3>
      ${dm.presets.length ? "" : `<p class="kkd-lstatus err">no Klein model found in diffusion_models</p>`}
      <div class="kkd-field"><span>model</span><select data-df="preset">${dm.presets.map(p => `<option ${p.name === D.preset ? "selected" : ""}>${esc(p.name)}</option>`).join("")}</select><span></span></div>
      ${D.loras.map((l, i) => `<div class="kkd-field"><span>lora ${i + 1}</span><select data-dl="${i}"><option value="">none</option>
        ${sorted.filter(fits).length ? `<optgroup label="for ${esc(pre?.name || "")}">${sorted.filter(fits).map(n => opt(n, l.name)).join("")}</optgroup>` : ""}
        <optgroup label="all loras">${sorted.filter(n => !fits(n)).map(n => opt(n, l.name)).join("")}</optgroup></select>
        <input type="range" data-dls="${i}" min="-1" max="2" step="0.05" value="${l.strength}" style="width:70px" data-tip="LoRA strength"><span class="v" data-dlv="${i}">${(+l.strength).toFixed(2)}</span></div>`).join("")}
      <div class="kkd-field"><span>steps</span><input type="range" data-df="steps" min="1" max="12" step="1" value="${D.steps}"><span class="v">${D.steps}</span></div>
      <div class="kkd-field" data-tip="Working size of the model: bigger = more detail, slower"><span>size</span><input type="range" data-df="megapixels" min="0.25" max="2" step="0.05" value="${D.megapixels}"><span class="v">${(+D.megapixels).toFixed(2)} MP</span></div>
      <div class="kkd-field" data-tip="Extra room around the layer the model sees, as a share of its size"><span>context</span><input type="range" data-df="context" min="0" max="1" step="0.05" value="${D.context}"><span class="v">${Math.round(D.context * 100)}%</span></div>
      <div class="kkd-field" data-tip="Width of the blended edge around the shape (preview pixels)"><span>edge band</span><input type="range" data-df="band" min="0" max="64" step="1" value="${D.band}"><span class="v">${D.band} px</span></div>
      <div class="kkd-field" data-tip="-1 = a new seed every time"><span>seed</span><input type="text" data-df="seed" value="${D.seed}"><span></span></div>
      <div class="kkd-field"><span>reference</span><button class="kkd-pill" data-dref aria-pressed="${!!D.reference}" data-tip="The crop as reference latent: keeps layout, perspective and colours">${D.reference ? "on" : "off"}</button><span></span></div>`;
  }
  updateLightStatus(L) {
    if (this.cur() !== L) return;
    const s = this.$("l-status"); if (s) { s.textContent = L.status || ""; s.classList.toggle("err", !!L.err); }
  }

  // ---- regions
  decodeLabels(im) {
    const c = mkCanvas(im.naturalWidth, im.naturalHeight), g = c.getContext("2d", { willReadFrequently: true });
    g.drawImage(im, 0, 0); const d = g.getImageData(0, 0, c.width, c.height).data;
    const ids = new Int32Array(c.width * c.height);
    for (let p = 0, q = 0; q < ids.length; p += 4, q++) ids[q] = ((d[p] << 16) | (d[p + 1] << 8) | d[p + 2]) - 1;
    this.labels = { w: c.width, h: c.height, ids }; this.maskCache.clear();
  }
  selectIds(sel) {
    const regs = this.m.regions || []; const out = new Set();
    const exact = regs.filter(r => String(r.name).toLowerCase() === String(sel || "").trim().toLowerCase());
    if (exact.length) return new Set(exact.map(r => r.id));          // picked names may contain spaces or ':'
    // the plan's selector syntax (plan.py parse_rules): ',' = or, space = and, '!' negates one term
    const hit = (term, r) => {
      const low = term.toLowerCase(); if (["default", "*", "all"].includes(low)) return true;
      const i = low.indexOf(":"), kind = i < 0 ? "name" : low.slice(0, i), val = i < 0 ? low : low.slice(i + 1);
      if (kind === "id" || kind === "region") { const [a, b = a] = val.split("-").map(Number); return r.id >= a && r.id <= b; }
      const re = new RegExp("^" + val.replace(/[.+^${}()|\\]/g, "\\$&").replace(/\*/g, ".*").replace(/\?/g, ".") + "$");
      if (kind === "group") return re.test(String(r.group).toLowerCase());
      if (kind === "tag") return (r.tags || []).some(t => re.test(String(t).toLowerCase()));
      return kind === "name" && re.test(String(r.name).toLowerCase());
    };
    const alts = String(sel || "").split(",").map(a => a.trim().split(/\s+/).filter(Boolean)).filter(a => a.length);
    try {
      for (const r of regs) if (alts.some(a => a.every(t => t.startsWith("!") ? !hit(t.slice(1), r) : hit(t, r)))) out.add(r.id);
    } catch { return new Set(); }                      // unparsable pattern: nothing, like the render
    return out;
  }
  regionMask(sel, scratch) {                          // proxy-size canvas, white where the selector matches
    if (!scratch && this.maskCache.has(sel)) { const m = this.maskCache.get(sel); this.maskCache.delete(sel); this.maskCache.set(sel, m); return m; }
    let c;
    if (scratch) {                                    // hover highlight: one reused canvas, never cached
      if (this._hoverSel === sel && this._hoverMask) return this._hoverMask;
      c = this._hoverMask = this._hoverMask || mkCanvas(this.PW, this.PH); c.getContext("2d").clearRect(0, 0, this.PW, this.PH);
      this._hoverSel = sel;
    } else c = mkCanvas(this.PW, this.PH);
    if (this.labels) {
      const ids = this.selectIds(sel), t = mkCanvas(this.labels.w, this.labels.h), g = t.getContext("2d");
      const im = g.createImageData(t.width, t.height), a = im.data, L = this.labels.ids;
      for (let q = 0, p = 0; q < L.length; q++, p += 4) if (ids.has(L[q])) { a[p] = a[p + 1] = a[p + 2] = a[p + 3] = 255; }
      g.putImageData(im, 0, 0); c.getContext("2d").drawImage(t, 0, 0, this.PW, this.PH);
    }
    if (!scratch) { this.maskCache.set(sel, c); if (this.maskCache.size > 24) this.maskCache.delete(this.maskCache.keys().next().value); }
    return c;
  }
  clipMask(L) {                                      // the clip regions; with stagger behaviours each region weighted
    if (!L.motion?.some(b => b.type === "stagger" && b.on !== false) || !this.labels) return this.regionMask(L.clip);
    const lab = this.labels, owner = L._copyOf || L;  // the cache lives on the layer, its copies share it
    let c = owner._stg;
    if (!c || c.sel !== L.clip || c.lab !== lab || c.PW !== this.PW) {  // the clipped pixels once per selection, grouped by region
      const ids = this.selectIds(L.clip), LW = lab.w, LH = lab.h, src = lab.ids;
      let maxId = 0; for (const id of ids) if (id > maxId) maxId = id;
      const slot = new Int32Array(maxId + 2).fill(-1), order = [], count = [];
      for (let q = 0; q < src.length; q++) { const id = src[q]; if (id <= maxId && ids.has(id)) { let k = slot[id]; if (k < 0) { k = slot[id] = order.length; order.push(id); count.push(0); } count[k]++; } }
      const start = new Uint32Array(order.length + 1); for (let k = 0; k < order.length; k++) start[k + 1] = start[k] + count[k];
      const buf = new Uint32Array(start[order.length]), fill = Uint32Array.from(start.subarray(0, order.length)), box = order.map(() => [LW, LH, 0, 0]);
      const t = mkCanvas(LW, LH), g = t.getContext("2d"), im = g.createImageData(LW, LH), D = im.data;
      for (let y = 0, q = 0; y < LH; y++) for (let x = 0; x < LW; x++, q++) {
        const id = src[q]; if (id < 0 || id > maxId) continue; const k = slot[id]; if (k < 0) continue;
        buf[fill[k]++] = q; D[q * 4] = D[q * 4 + 1] = D[q * 4 + 2] = 255;
        const b = box[k]; if (x < b[0]) b[0] = x; if (x > b[2]) b[2] = x; if (y < b[1]) b[1] = y; if (y > b[3]) b[3] = y;
      }
      const rp = order.map((id, k) => ({ id, pos: buf.subarray(start[k], start[k + 1]), box: [box[k][0], box[k][1], box[k][2] + 1, box[k][3] + 1], a: -1 }));
      const regs = (this.m.regions || []).filter(r => ids.has(r.id)).map(r => [r.id, r.bbox]);
      c = owner._stg = { sel: L.clip, lab, PW: this.PW, rp, t, g, im, regs, out: mkCanvas(this.PW, this.PH) };
    }
    const stg = L.motion.filter(b => b.type === "stagger" && b.on !== false), key = `${this.tl.time}|${this.tl.duration}|${this.W}x${this.H}|${stable(stg)}`;
    if (c.key === key) return c.out;                 // the same weights: this frame's mask is ready
    c.key = key;
    const w = staggerWeights(stg, this.tl.time, this.tl.duration, c.regs, this.W, this.H) || {}, A = c.im.data;
    let X0 = Infinity, Y0 = Infinity, X1 = -1, Y1 = -1;
    for (const r of c.rp) {                          // only regions whose weight changed are rewritten
      const v = w[r.id], a = v === undefined ? 255 : Math.round(255 * v); if (a === r.a) continue;
      r.a = a; for (const q of r.pos) A[q * 4 + 3] = a;
      if (r.box[0] < X0) X0 = r.box[0]; if (r.box[1] < Y0) Y0 = r.box[1]; if (r.box[2] > X1) X1 = r.box[2]; if (r.box[3] > Y1) Y1 = r.box[3];
    }
    if (X1 >= 0) {                                   // upload the changed rectangle, rescale the mask
      c.g.putImageData(c.im, 0, 0, X0, Y0, X1 - X0, Y1 - Y0);
      const og = c.out.getContext("2d"); og.clearRect(0, 0, this.PW, this.PH); og.drawImage(c.t, 0, 0, this.PW, this.PH);
    }
    return c.out;
  }
  regionsView() {                                     // every region in its own colour, like the names drawn on it
    const { w, h, ids } = this.labels, c = mkCanvas(w, h), g = c.getContext("2d"), im = g.createImageData(w, h), a = im.data;
    const col = id => { const t = (id * 0.618034) % 1, k = (x => Math.round(255 * x));
      const hsv = (hh, ss, vv) => { const i = Math.floor(hh * 6), f = hh * 6 - i, p = vv * (1 - ss), q = vv * (1 - f * ss), u = vv * (1 - (1 - f) * ss);
        return [[vv, u, p], [q, vv, p], [p, vv, u], [p, q, vv], [u, p, vv], [vv, p, q]][i % 6]; };
      return hsv(t, 0.45, 0.62 + 0.25 * ((id * 7) % 3) / 2).map(k); };
    const cache = new Map();
    for (let q = 0, p = 0; q < ids.length; q++, p += 4) {
      const id = ids[q]; if (id < 0) { a[p] = a[p + 1] = a[p + 2] = 20; a[p + 3] = 255; continue; }
      let cc = cache.get(id); if (!cc) { cc = col(id); cache.set(id, cc); }
      a[p] = cc[0]; a[p + 1] = cc[1]; a[p + 2] = cc[2]; a[p + 3] = 255;
    }
    g.putImageData(im, 0, 0); return c;
  }
  drawNames() {
    const show = this.names || this.view === "regions" || this.view === "ids";
    if (!show || !(this.m.regions || []).length) return;
    const ctx = this.ctx, S = this.S(), px = Math.max(10, Math.round(11 * (devicePixelRatio || 1)));
    ctx.save(); ctx.font = `600 ${px}px "Hanken Grotesk KKD", system-ui, sans-serif`; ctx.textAlign = "center"; ctx.textBaseline = "middle";
    ctx.lineJoin = "round"; ctx.lineWidth = Math.max(2, px / 4);
    for (const r of this.m.regions) {
      const b = this.rbox(r); if (!b) continue;
      const w = (b[2] - b[0]) * S, h = (b[3] - b[1]) * S;
      const tw = ctx.measureText(r.name).width;
      if (w < Math.min(tw * 0.9, 160) || h < px * 1.4) continue;     // too small at this zoom: hover or zoom in
      const X = this.OX() + (b[0] + b[2]) / 2 * S, Y = this.OY() + (b[1] + b[3]) / 2 * S;
      ctx.strokeStyle = "rgba(0,0,0,.75)"; ctx.strokeText(r.name, X, Y); ctx.fillStyle = "#fff"; ctx.fillText(r.name, X, Y);
    }
    ctx.restore();
  }
  rbox(r) { const b = r?.bbox; return b && b[2] > 0 && b[3] > 0 ? [b[0], b[1], b[0] + b[2], b[1] + b[3]] : null; }    // -> x0, y0, x1, y1
  regionAt(x, y) {
    if (!this.labels) return null;
    const lx = Math.floor(x / this.W * this.labels.w), ly = Math.floor(y / this.H * this.labels.h);
    if (lx < 0 || ly < 0 || lx >= this.labels.w || ly >= this.labels.h) return null;
    const id = this.labels.ids[ly * this.labels.w + lx];
    return (this.m.regions || []).find(r => r.id === id) || null;
  }

  // ---- rendering
  filterOf(a) { return `brightness(${a.brightness}) contrast(${a.contrast}) saturate(${a.saturate}) hue-rotate(${a.hue}deg) sepia(${a.sepia})`; }
  baseImage() { return (this._forceMatrix ? this.views.matrix : this.views[this.view]) || this.views.matrix; }
  drawContent(g, L) {
    const k = this.k;
    if (L.kind === "base") {
      g.drawImage(this.baseImage(), 0, 0, this.PW, this.PH);
      if (L.holes) { g.save(); g.globalCompositeOperation = "destination-out"; g.drawImage(this.regionMask(L.holes), 0, 0); g.restore(); }
    } else if (L.kind === "paint") g.drawImage(L.canvas, 0, 0);
    else if (L.kind === "light") { if (L.img) g.drawImage(L.img, 0, 0, this.PW, this.PH); }
    else if (L.kind === "image" && L.img && !L.vhidden) {
      g.save(); g.translate((L.x + L.w / 2) * k, (L.y + L.h / 2) * k); g.rotate(L.rotation * Math.PI / 180);
      g.scale(L.flip_h ? -1 : 1, L.flip_v ? -1 : 1);
      g.drawImage(L.img, -L.w / 2 * k, -L.h / 2 * k, L.w * k, L.h * k); g.restore();
    } else if (L.kind === "shape" && L.shape) {       // = render.shape_source: the path in the box, blurred by the feather
      const w = L.w * k, h = L.h * k, sh = L.shape, f = (+sh.feather || 0) * k;
      g.save(); g.translate((L.x + L.w / 2) * k, (L.y + L.h / 2) * k); g.rotate(L.rotation * Math.PI / 180);
      g.scale(L.flip_h ? -1 : 1, L.flip_v ? -1 : 1);
      if (f > 0.5 * k) g.filter = `blur(${f}px)`;
      g.fillStyle = isHex(L.color) ? L.color : "#ffffff"; g.beginPath();
      if (sh.type === "ellipse") g.ellipse(0, 0, Math.abs(w) / 2, Math.abs(h) / 2, 0, 0, Math.PI * 2);
      else if (sh.type === "poly" && (sh.points || []).length >= 3) sh.points.forEach(([u, v], i) => (i ? g.lineTo : g.moveTo).call(g, -w / 2 + u * w, -h / 2 + v * h));
      else g.rect(-w / 2, -h / 2, w, h);
      g.fill(); g.restore();
    }
    if (L.clip) { g.save(); g.globalCompositeOperation = "destination-in"; g.drawImage(this.clipMask(L), 0, 0); g.restore(); }
  }
  maskSource(i) {
    const L = this.layers[i]; if (!L || !L.mask.by) return null;
    const s = L.mask.by === "below" ? this.layers[i + 1] : this.layers.find(q => q.id === L.mask.by);
    return s && s !== L && s.kind !== "adjust" ? s : null;
  }
  fxFilter(L) {                                      // one SVG filter per layer (blur, sharpen, glow), rebuilt when they change
    const f = fxOf(L); if (!fxActive({ ...f, grain: 0 })) return "";
    const k = this.k, id = "kkdfx_" + String(L.id).replace(/[^\w-]/g, "_"), key = stable({ ...f, grain: 0, grain_size: 0 }) + "|" + k + "|" + this.PW;
    if (!this.fxSvg) {
      this.fxSvg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
      this.fxSvg.setAttribute("width", "0"); this.fxSvg.setAttribute("height", "0"); this.fxSvg.style.position = "absolute";
      document.body.appendChild(this.fxSvg); this.fxEls = new Map();
    }
    let el = this.fxEls.get(id);
    if (!el || el._key !== key) {
      if (!el) { el = document.createElementNS("http://www.w3.org/2000/svg", "filter"); el.id = id; this.fxSvg.appendChild(el); this.fxEls.set(id, el); }
      const n = v => (+v).toFixed(4), parts = [`<feOffset in="SourceGraphic" dx="0" dy="0" result="s0"/>`];
      let cur = "s0";
      if (f.blur > 0.05) { parts.push(`<feGaussianBlur in="${cur}" stdDeviation="${n(f.blur * k)}" result="s1"/>`); cur = "s1"; }
      if (f.sharpen > 0.001) { parts.push(`<feGaussianBlur in="${cur}" stdDeviation="${n(f.sharpen_radius * k)}" result="sb"/>`,
        `<feComposite in="${cur}" in2="sb" operator="arithmetic" k1="0" k2="${n(1 + f.sharpen)}" k3="${n(-f.sharpen)}" k4="0" result="s2"/>`); cur = "s2"; }
      if (f.glow > 0.001 && f.glow_radius > 0) { const th = Math.min(0.99, f.glow_threshold), sl = n(1 / (1 - th)), ic = n(-th / (1 - th));
        parts.push(`<feComponentTransfer in="${cur}" result="g0">${["R", "G", "B"].map(c => `<feFunc${c} type="linear" slope="${sl}" intercept="${ic}"/>`).join("")}</feComponentTransfer>`,
          `<feGaussianBlur in="g0" stdDeviation="${n(f.glow_radius * k)}" result="g1"/>`,
          `<feComposite in="${cur}" in2="g1" operator="arithmetic" k1="0" k2="1" k3="${n(f.glow)}" k4="0" result="s3"/>`); cur = "s3"; }
      parts.push(`<feOffset in="${cur}" dx="0" dy="0"/>`);
      el.setAttribute("filterUnits", "userSpaceOnUse"); el.setAttribute("x", "0"); el.setAttribute("y", "0");
      el.setAttribute("width", String(this.PW)); el.setAttribute("height", String(this.PH));
      el.setAttribute("color-interpolation-filters", "sRGB"); el.innerHTML = parts.join(""); el._key = key;
    }
    return `url(#${id})`;
  }
  lightFor(L) {                                      // the light layer a layer reacts to ("" = the first one)
    const by = L.light_react?.by || ""; return this.layers.find(q => q.kind === "light" && (by ? q.id === by : true)) || null;
  }
  reactLight(Lt) { return this.layers.some(q => q.light_react && +(q.light_react.amount ?? 1) > 0 && this.lightFor(q) === Lt); }
  applyShade(lc, L) {                                 // x (light on the clay / its albedo), mixed by amount (= render.py light_react)
    const amt = Math.max(0, Math.min(1, +(L.light_react.amount ?? 1))), Lt = this.lightFor(L);
    if (!(amt > 0) || !Lt?.img) return;
    const PW = this.PW, PH = this.PH, orig = this.mtc, t = this.fxc, albedo = Math.max(0.05, +(Lt.light?.clay ?? 0.7));
    orig.clearRect(0, 0, PW, PH); orig.drawImage(this.lyr, 0, 0);
    t.clearRect(0, 0, PW, PH); t.drawImage(this.lyr, 0, 0);
    t.save(); t.globalCompositeOperation = "multiply"; t.drawImage(Lt.img, 0, 0, PW, PH); t.restore();
    t.save(); t.globalCompositeOperation = "destination-in"; t.drawImage(this.lyr, 0, 0); t.restore();
    lc.save(); lc.clearRect(0, 0, PW, PH); lc.drawImage(this.mtmp, 0, 0);
    lc.globalAlpha = amt; lc.filter = `brightness(${(1 / albedo).toFixed(4)})`; lc.drawImage(this.fxtmp, 0, 0); lc.restore();
  }
  noiseFrame(size) {                                 // grey noise (mean 0.5, sd 0.15) at the grain size: 8 frames, cycled
    const key = `${size}|${this.k}|${this.PW}`, i = ((Math.round(this.tl.time * this.tl.fps) % 8) + 8) % 8;
    return this.noiseMake(key, size, i);
  }
  noiseMake(key, size, i) {
    if (!this._noiseSets || this._noisePW !== this.PW) { this._noiseSets = new Map(); this._noisePW = this.PW; }
    let set = this._noiseSets.get(key);
    if (!set) { set = []; this._noiseSets.set(key, set); if (this._noiseSets.size > 2) this._noiseSets.delete(this._noiseSets.keys().next().value);
      const idle = window.requestIdleCallback || (f => setTimeout(f, 50)), fill = j => { if (j < 8 && this._noiseSets?.get(key) === set && !this.closed) idle(() => { this.noiseMake(key, size, j); fill(j + 1); }); };
      fill(0); }                                     // the other 7 frames in idle time: no stutter when play starts
    this._noise = set;
    if (!this._noise[i]) {
      const cell = Math.max(1, size * this.k), w = Math.max(2, Math.round(this.PW / cell)), h = Math.max(2, Math.round(this.PH / cell));   // never above the preview size
      const small = mkCanvas(w, h), sg = small.getContext("2d"), im = sg.createImageData(w, h), a = im.data;
      for (let p = 0; p < a.length; p += 4) {        // sum of 4 uniforms ~ normal (sd 0.15 like the render), cheap
        const g = (Math.random() + Math.random() + Math.random() + Math.random() - 2) * 1.7320508;
        a[p] = a[p + 1] = a[p + 2] = Math.max(0, Math.min(255, 127.5 + 38 * g)); a[p + 3] = 255;
      }
      sg.putImageData(im, 0, 0);
      const c = mkCanvas(this.PW, this.PH), cg = c.getContext("2d"); cg.imageSmoothingQuality = "high"; cg.drawImage(small, 0, 0, this.PW, this.PH);
      this._noise[i] = c;
    }
    return this._noise[i];
  }
  applyGrain(lc, f) {                                 // overlay the noise by the amount, keep the layer's own alpha (= fx.py grain)
    const t = this.fxc, PW = this.PW, PH = this.PH;
    t.clearRect(0, 0, PW, PH); t.drawImage(this.lyr, 0, 0);
    t.save(); t.globalCompositeOperation = "overlay"; t.globalAlpha = f.grain; t.drawImage(this.noiseFrame(f.grain_size), 0, 0); t.restore();
    t.save(); t.globalCompositeOperation = "destination-in"; t.drawImage(this.lyr, 0, 0); t.restore();
    lc.save(); lc.clearRect(0, 0, PW, PH); lc.drawImage(this.fxtmp, 0, 0); lc.restore();
  }
  async loadPm() {                                    // the projection mask at preview size (the node reads the file / scene)
    const P = this.pm, key = stable({ s: P.source, i: !!P.invert, g: +P.grow || 0, f: +P.feather || 0 });
    if (!P.on || !P.source) { this.pmOut = null; this.pmNote = P.source ? "off" : ""; this.renderPm?.(); this.draw(); return; }
    if (key === this.pmKey && this.pmOut) { this.renderPm?.(); this.draw(); return; }
    const tok = this._pmTok = {}; this.pmKey = key; this.pmNote = "reading the mask…"; this.renderPm?.();
    try {
      const r = await api.fetchApi("/kubakub/director/pmask", { method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ node: this.m.node, pm: { source: P.source, invert: !!P.invert, grow: +P.grow || 0, feather: +P.feather || 0 }, pw: this.PW, ph: this.PH }) });
      const j = await r.json().catch(() => ({ error: "no answer (" + r.status + ")" }));
      if (!r.ok || j.error) throw new Error(j.error || r.status);
      const im = await loadImage(viewURL(j.image));
      if (tok !== this._pmTok || this.closed) return;
      const bld = mkCanvas(this.PW, this.PH), bg = bld.getContext("2d");           // white = building -> alpha
      bg.filter = "url(#kkdLumA)"; bg.drawImage(im, 0, 0, this.PW, this.PH); bg.filter = "none";
      const out = mkCanvas(this.PW, this.PH), og = out.getContext("2d");          // black everywhere but the building
      og.fillStyle = "#000"; og.fillRect(0, 0, this.PW, this.PH); og.globalCompositeOperation = "destination-out"; og.drawImage(bld, 0, 0);
      this.pmOut = out; this.pmNote = (j.note || "").replace(/^projection mask: /, "");
    } catch (e) { if (tok !== this._pmTok) return; this.pmOut = null; this.pmKey = null; this.pmNote = "mask: " + e.message; }
    this.renderPm?.(); this.draw();
  }
  composite(target = this.comp, only = null, solo = false) {
    // only: the layer indices to draw (the projector's source); solo: one layer as it is, even hidden
    const cc = target.getContext("2d"), lc = this.lc, mc = this.mc, PW = this.PW, PH = this.PH;
    cc.globalCompositeOperation = "source-over"; cc.globalAlpha = 1; cc.filter = "none"; cc.fillStyle = "#000"; cc.fillRect(0, 0, PW, PH);
    for (let i = this.layers.length - 1; i >= 0; i--) {
      const L0 = this.layers[i]; if (only ? !only.includes(i) : !L0.visible) continue;
      const copies = solo ? [L0] : this.repeatCopies(L0);
      for (let ci = copies.length - 1; ci >= 0; ci--) {
      const L = copies[ci], R = this.layerRect(L);
      if (R && (R[2] <= 0 || R[3] <= 0)) continue;   // entirely off the canvas
      const fxf = this.fxFilter(L), clipped = [];
      const clipTo = g => { if (!R || clipped.includes(g)) return; g.save(); g.beginPath(); g.rect(R[0], R[1], R[2], R[3]); g.clip(); clipped.push(g); };
      // blurs (feather, fx filter) stay full size: the GPU blur depends on its bounds, so the picture stays identical
      const soft = R && (fxf || (+(L.shape?.feather) || 0) > 0.5);
      if (!soft) clipTo(lc);
      lc.save(); lc.clearRect(0, 0, PW, PH);
      if (L.kind === "adjust") {
        lc.filter = this.filterOf(L.adjust); lc.drawImage(target, 0, 0); lc.filter = "none";
        if (L.clip) { lc.globalCompositeOperation = "destination-in"; lc.drawImage(this.clipMask(L), 0, 0); }
      } else this.drawContent(lc, L);
      lc.restore();
      if (fxf) {                                     // the layer's effects, before masks by other layers
        const t = this.fxc; t.clearRect(0, 0, PW, PH); t.filter = fxf; t.drawImage(this.lyr, 0, 0); t.filter = "none";
        lc.save(); lc.clearRect(0, 0, PW, PH); lc.drawImage(this.fxtmp, 0, 0); lc.restore(); }
      for (const g of [lc, this.fxc, this.mtc, mc]) clipTo(g);   // grain, shade, mask and the draw: only the box
      const fg = fxOf(L).grain; if (fg > 0.001 && L.kind !== "light") this.applyGrain(lc, fxOf(L));
      if (L.light_react && ["image", "paint", "shape"].includes(L.kind)) this.applyShade(lc, L);
      const src = this.maskSource(i);
      if (src) {
        if (L.mask.mode === "brightness") {           // luminance x alpha as alpha, in one GPU filter pass
          const t = this.mtc; t.save(); t.clearRect(0, 0, PW, PH); this.drawContent(t, src); t.restore();
          mc.save(); mc.clearRect(0, 0, PW, PH); mc.filter = "url(#kkdLumA)"; mc.drawImage(this.mtmp, 0, 0); mc.restore();
        } else { mc.save(); mc.clearRect(0, 0, PW, PH); this.drawContent(mc, src); mc.restore(); }
        lc.save(); lc.globalCompositeOperation = L.mask.invert ? "destination-out" : "destination-in"; lc.drawImage(this.msk, 0, 0); lc.restore();
      }
      cc.save(); cc.globalAlpha = solo ? 1 : L.opacity; cc.globalCompositeOperation = solo ? "source-over" : OPS[L.blend] || "source-over";
      if (R) cc.drawImage(this.lyr, R[0], R[1], R[2], R[3], R[0], R[1], R[2], R[3]); else cc.drawImage(this.lyr, 0, 0);
      cc.restore();
      for (const g of clipped) g.restore();
      }
    }
    if (target === this.comp && !only && this.pmOut && this.pm.on && this.pm.view !== "off" && !this._forceMatrix) {
      cc.save(); cc.globalAlpha = this.pm.view === "dim" ? 0.7 : 1; cc.drawImage(this.pmOut, 0, 0); cc.restore();   // the projection mask on top
    }
  }
  layerRect(L) {                                     // preview px box a shape / image layer can touch (+3 sigma of its blurs), null = all
    if (!(L.kind === "shape" || L.kind === "image")) return null;
    if (L.kind === "shape" && L.shape?.type === "poly" && (L.shape.points || []).some(([u, v]) => !(u >= 0 && u <= 1 && v >= 0 && v <= 1))) return null;
    const f = fxOf(L), k = this.k, c = this.handles(L).corners, xs = c.map(q => q[0] * k), ys = c.map(q => q[1] * k);
    const sg = (+(L.shape?.feather) || 0) + (f.blur > 0.05 ? f.blur : 0) + (f.sharpen > 0.001 ? f.sharpen_radius : 0) + (f.glow > 0.001 && f.glow_radius > 0 ? f.glow_radius : 0);
    const m = Math.ceil(3 * sg * k) + 2;
    const x0 = Math.max(0, Math.floor(Math.min(...xs)) - m), y0 = Math.max(0, Math.floor(Math.min(...ys)) - m);
    const x1 = Math.min(this.PW, Math.ceil(Math.max(...xs)) + m), y1 = Math.min(this.PH, Math.ceil(Math.max(...ys)) + m);
    if (![x0, y0, x1, y1].every(Number.isFinite)) return null;
    return [x0, y0, Math.max(0, x1 - x0), Math.max(0, y1 - y0)];
  }
  guardHTML(el) {                                    // el.kkdHTML = markup: skipped when identical to what the panel shows
    if (!el || el._kkdGuard) return;
    const g = el._kkdGuard = { html: null }, obs = new MutationObserver(() => { g.html = null; });
    obs.observe(el, { childList: true, subtree: true, characterData: true, attributes: true });
    Object.defineProperty(el, "kkdHTML", { configurable: true, set: html => {
      if (obs.takeRecords().length) g.html = null;   // changed from outside since the last render
      if (g.html === html) return;
      el.innerHTML = html; obs.takeRecords(); g.html = html;
    } });
  }
  S() { return this.fitScale * this.zoom; }
  OX() { return (this.cv.width - this.W * this.S()) / 2 + this.panX; }
  OY() { return (this.cv.height - this.H * this.S()) / 2 + this.panY; }
  fit() {
    const r = this.cv.getBoundingClientRect(), dpr = Math.min(devicePixelRatio || 1, 2);
    this.cv.width = Math.max(10, r.width * dpr); this.cv.height = Math.max(10, r.height * dpr);
    this.fitScale = Math.min(this.cv.width / this.W, this.cv.height / this.H) * 0.94; this.draw();
  }
  draw() {                                           // at most once per frame (drags, paint strokes, sliders)
    if (this._raf) return;
    this._raf = requestAnimationFrame(() => { this._raf = 0; if (!this.root?.isConnected) return;
      if (this._panels) { this._panels = false; this.renderInspector(); this.renderLayers(); }
      this.drawNow(); this.drawTimeline(); });
  }
  drawNow() {
    if (!this.layers) return;
    this.recordKeys();
    this.checkLights();
    this.composite();
    const ctx = this.ctx; ctx.setTransform(1, 0, 0, 1, 0, 0); ctx.clearRect(0, 0, this.cv.width, this.cv.height);
    ctx.imageSmoothingQuality = "high"; ctx.drawImage(this.comp, this.OX(), this.OY(), this.W * this.S(), this.H * this.S());
    ctx.strokeStyle = "rgba(161,135,183,.8)"; ctx.lineWidth = 1; ctx.strokeRect(this.OX() + .5, this.OY() + .5, this.W * this.S(), this.H * this.S());
    this.drawNames();
    if (this.tool === "pick" && this.hoverRegion) {
      const m = this.regionMask(`region:${this.hoverRegion.id}`, true);
      ctx.save(); ctx.globalAlpha = 0.45; ctx.drawImage(this.tint(m), this.OX(), this.OY(), this.W * this.S(), this.H * this.S()); ctx.restore();
    }
    this.drawSelection();
    if (this.tool === "poly" && this.polyDraft?.length) {       // the polygon being clicked
      const ctx = this.ctx, P = this.polyDraft.map(([x, y]) => [this.OX() + x * this.S(), this.OY() + y * this.S()]);
      ctx.save(); ctx.strokeStyle = ORANGE; ctx.lineWidth = 2; ctx.setLineDash([6, 4]); ctx.beginPath();
      P.forEach(([x, y], i) => i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)); ctx.stroke(); ctx.setLineDash([]);
      ctx.fillStyle = ORANGE; for (const [x, y] of P) { ctx.beginPath(); ctx.arc(x, y, 4, 0, 7); ctx.fill(); } ctx.restore();
    }
    this.drawLights();
    if (this.guides.length) {
      ctx.save(); ctx.strokeStyle = LAV; ctx.lineWidth = 1; ctx.setLineDash([6, 4]);
      for (const g of this.guides) {
        ctx.beginPath();
        if (g.x !== undefined) { const X = this.OX() + g.x * this.S(); ctx.moveTo(X, 0); ctx.lineTo(X, this.cv.height); }
        else { const Y = this.OY() + g.y * this.S(); ctx.moveTo(0, Y); ctx.lineTo(this.cv.width, Y); }
        ctx.stroke();
      }
      ctx.restore();
    }
  }
  lightSpots() {                                     // screen positions of the selected light layer's lamps
    const L = this.cur(); if (!L || L.kind !== "light" || !this.frame()) return [];
    return L.light.lights.map((q, i) => {
      if (q.type === "sun") return null;
      const [px, py] = this.fromFacade(+q.x || 0, +q.height || 0);
      return { i, q, X: this.OX() + px * this.S(), Y: this.OY() + py * this.S() };
    }).filter(Boolean);
  }
  drawLights() {
    const ctx = this.ctx, dpr = Math.min(devicePixelRatio || 1, 2);
    for (const { i, q, X, Y } of this.lightSpots()) {
      const r = 9 * dpr, sel = i === this.lsel;
      ctx.save();
      if (q.type === "area") {                       // the lamp's size on the wall, as a guide
        const s = (+q.size || 1) / this.facadeW() * this.W * this.S();
        ctx.strokeStyle = "rgba(255,255,255,.6)"; ctx.setLineDash([4, 4]); ctx.strokeRect(X - s / 2, Y - s / 2, s, s); ctx.setLineDash([]);
      }
      ctx.beginPath(); ctx.arc(X, Y, r, 0, 7);
      ctx.fillStyle = q.on === false ? "rgba(0,0,0,.35)" : q.color || "#fff"; ctx.fill();
      ctx.lineWidth = sel ? 3 : 1.5; ctx.strokeStyle = sel ? ORANGE : "#fff"; ctx.stroke();
      ctx.fillStyle = "rgba(0,0,0,.75)"; ctx.font = `700 ${Math.round(10 * dpr)}px "Hanken Grotesk KKD", system-ui, sans-serif`;
      ctx.textAlign = "center"; ctx.textBaseline = "middle"; ctx.fillText(q.type[0], X, Y + .5);
      ctx.font = `600 ${Math.round(11 * dpr)}px "Hanken Grotesk KKD", system-ui, sans-serif`; ctx.textAlign = "left";
      ctx.lineWidth = 3; ctx.strokeStyle = "rgba(0,0,0,.7)"; const t = `${(+q.distance || 0).toFixed(1)} m out`;
      ctx.strokeText(t, X + r + 4, Y); ctx.fillStyle = "#fff"; ctx.fillText(t, X + r + 4, Y);
      ctx.restore();
    }
  }
  aabb(L) {
    const h = this.handles(L), xs = h.corners.map(p => p[0]), ys = h.corners.map(p => p[1]);
    return [Math.min(...xs), Math.min(...ys), Math.max(...xs), Math.max(...ys)];
  }
  snapTargets(except) {
    const tx = [0, this.W / 2, this.W], ty = [0, this.H / 2, this.H];
    for (const q of this.layers) {
      if (q === except || !q.visible || !this.transformable(q)) continue;
      const [x0, y0, x1, y1] = this.aabb(q); tx.push(x0, (x0 + x1) / 2, x1); ty.push(y0, (y0 + y1) / 2, y1);
    }
    for (const r of this.m.regions || []) {
      const b = this.rbox(r); if (!b) continue;
      tx.push(b[0], (b[0] + b[2]) / 2, b[2]); ty.push(b[1], (b[1] + b[3]) / 2, b[3]);
    }
    return { tx, ty };
  }
  nearest(vals, targets) {                           // best (delta, target) moving any of vals onto a target
    const thr = 8 / this.S(); let best = null;
    for (const v of vals) for (const t of targets) { const d = t - v; if (Math.abs(d) <= thr && (!best || Math.abs(d) < Math.abs(best[0]))) best = [d, t]; }
    return best;
  }
  tint(mask) {
    if (!this._tint) this._tint = mkCanvas(this.PW, this.PH);
    const g = this._tint.getContext("2d"); g.clearRect(0, 0, this.PW, this.PH); g.drawImage(mask, 0, 0);
    g.globalCompositeOperation = "source-in"; g.fillStyle = LAV; g.fillRect(0, 0, this.PW, this.PH); g.globalCompositeOperation = "source-over";
    return this._tint;
  }
  transformable(L) { return L && (L.kind === "image" && L.img || L.kind === "shape"); }
  handles(L) {
    const a = L.rotation * Math.PI / 180, c = Math.cos(a), s = Math.sin(a), cx = L.x + L.w / 2, cy = L.y + L.h / 2;
    const P = (lx, ly) => [cx + lx * c - ly * s, cy + lx * s + ly * c];
    return { corners: [P(-L.w / 2, -L.h / 2), P(L.w / 2, -L.h / 2), P(L.w / 2, L.h / 2), P(-L.w / 2, L.h / 2)],
      edges: [P(0, -L.h / 2), P(L.w / 2, 0), P(0, L.h / 2), P(-L.w / 2, 0)],
      rot: P(0, -L.h / 2 - 90 / this.S()), top: P(0, -L.h / 2), c: [cx, cy], cos: c, sin: s };
  }
  drawSelection() {
    const L = this.layers[this.sel]; if (!L || !L.visible || !this.transformable(L)) return;
    const ctx = this.ctx, h = this.handles(L), sx = p => this.OX() + p[0] * this.S(), sy = p => this.OY() + p[1] * this.S();
    ctx.strokeStyle = LAV; ctx.lineWidth = 2; ctx.beginPath(); h.corners.forEach((p, i) => i ? ctx.lineTo(sx(p), sy(p)) : ctx.moveTo(sx(p), sy(p))); ctx.closePath(); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(sx(h.top), sy(h.top)); ctx.lineTo(sx(h.rot), sy(h.rot)); ctx.stroke();
    ctx.fillStyle = LAV; h.corners.forEach(p => ctx.fillRect(sx(p) - 4, sy(p) - 4, 8, 8));
    ctx.fillStyle = "#fff"; ctx.strokeStyle = LAV; ctx.lineWidth = 1.5;
    h.edges.forEach(p => { ctx.fillRect(sx(p) - 3, sy(p) - 3, 6, 6); ctx.strokeRect(sx(p) - 3, sy(p) - 3, 6, 6); });
    ctx.beginPath(); ctx.arc(sx(h.rot), sy(h.rot), 5, 0, 7); ctx.fillStyle = ORANGE; ctx.fill();
  }

  // ---- DOM
  $(id) { return this.root.querySelector(`[data-id="${id}"]`); }
  buildDom() {
    const tool = (id, tip, dis) => `<button class="kkd-tool" data-id="t-${id}" aria-pressed="${id === "move"}" ${dis ? "disabled" : ""} data-tip="${tip}"><svg viewBox="0 0 24 24">${ICON[id]}</svg></button>`;
    this.root = document.createElement("div"); this.root.className = "kkd-root"; this.root.tabIndex = -1;
    this.root.innerHTML = `
      <div class="kkd-bar">
        <div class="kkd-logo"><b>kubakub</b><span> director</span><i></i></div>
        <div class="kkd-pills" data-id="views" role="group" aria-label="view"></div>
        <span class="kkd-grow"></span>
        <span class="kkd-stamp" data-id="stamp"></span>
        <button class="kkd-pill" data-id="file" data-tip="Save the composition under a name, open a saved one, export / import as a file">file</button>
        <button class="kkd-pill lav" data-id="add" data-tip="Import an image, or a new paint / colour-grade layer · drop or paste images too|Ctrl O / Shift N / Shift A">+ layer</button>
        <button class="kkd-pill ico" data-id="undo" aria-label="undo" data-tip="Undo|Ctrl Z"><svg viewBox="0 0 24 24"><path d="M9 14 4 9l5-5"/><path d="M4 9h10.5a5.5 5.5 0 0 1 0 11H11"/></svg></button>
        <button class="kkd-pill ico" data-id="redo" aria-label="redo" data-tip="Redo|Ctrl Shift Z"><svg viewBox="0 0 24 24"><path d="m15 14 5-5-5-5"/><path d="M20 9H9.5a5.5 5.5 0 0 0 0 11H13"/></svg></button>
        <button class="kkd-pill" data-id="diffbtn" data-tip="Diffusion settings: model (Klein), your LoRAs, steps, size, seed">diffusion</button>
        <button class="kkd-pill" data-id="outputs" data-tip="What goes back into the workflow">outputs</button>
        <button class="kkd-pill" data-id="keysbtn" data-tip="All keyboard shortcuts|?">?</button>
        <button class="kkd-pill" data-id="exportbtn" data-tip="Render the whole timeline at the delivery size into a file: PNG with alpha, ProRes, high quality H.264 / H.265 (only this node runs)">export</button>
        <button class="kkd-pill" data-id="close" data-tip="Close without applying|Esc">close</button>
        <button class="kkd-pill" data-id="flow" aria-pressed="false" data-tip="What the nodes after the director do: hold = wait until you apply a composition (nothing runs on an empty one) · auto = the base passes on at every queue">hold</button>
        <button class="kkd-pill accent" data-id="apply" data-tip="Save the composition into the node and queue the workflow|Ctrl Enter">apply</button>
      </div>
      <div class="kkd-tools" role="toolbar" aria-label="tools">
        ${tool("move", "Move · drag corners to scale, the dot to rotate · scroll scales, Shift+scroll rotates|V")}
        ${tool("pick", "Clip to region: click a window or element to keep the selected layer inside it, click again to free it|R")}
        ${tool("brush", "Brush on a paint layer (made for you if needed) · [ and ] size|B")}
        ${tool("erase", "Eraser on the paint layer|E")}
        ${tool("hand", "Pan · hold Space in any tool · Ctrl+scroll zooms · 0 fits|H")}
        <span class="kkd-sep"></span>
        <button class="kkd-tool" data-id="snap" aria-pressed="true" data-tip="Snapping to the canvas, other layers and region borders (windows, cornices) · hold Ctrl to drag without|S"><svg viewBox="0 0 24 24">${ICON.snap}</svg></button>
        ${tool("pin", "Corner pin for perspective (coming in step W3)|C", true)}
      </div>
      <div class="kkd-stage" data-id="stage"><canvas data-id="cv"></canvas><div class="kkd-brush" data-id="brush" hidden></div>
        <div class="kkd-hud" data-id="hud">x 0 · y 0</div><div class="kkd-busy" data-id="busy" hidden></div></div>
      <div class="kkd-tl" data-id="tl">
        <div class="kkd-split h" data-split="th" data-tip="Drag to make the timeline taller or shorter · double click resets"></div>
        <div class="kkd-tlbar">
          <button class="kkd-pill" data-id="tl-play" data-tip="Play / pause (lights re-render when you stop)|P">play</button>
          <button class="kkd-rec" data-id="tl-rec" aria-pressed="false" data-tip="Record: every change at the current time becomes a keyframe|Shift K"></button>
          <button class="kkd-pill" data-id="tl-key" data-tip="Keyframe the selected layer (or lamp) as it is now|K">◆ key</button>
          <span data-id="tl-time">0.00 s</span>
          <button class="kkd-pill" data-id="tl-mark" data-tip="A marker at the playhead (double click it on the ruler to rename, drag to move)|M">+ marker</button>
          <button class="kkd-pill" data-id="tl-h3" data-tip="H3 clips: let MiniMax H3 animate a stretch of the timeline between its start and end frame (keyframes) or from references (layers, music, the sequence) · needs the 'keyframe clips (h3)' node after the director">h3 clips</button>
          <button class="kkd-pill" data-id="tl-snap" aria-pressed="true" data-tip="Scrubbing and dragged keyframes / markers snap to beats, markers and keyframes · hold Ctrl to move freely">snap</button>
          <span class="kkd-grow"></span>
          <button class="kkd-pill" data-id="tl-audio" data-tip="Load a sound (mp3, wav, ogg …) · or drop it onto the window · beats are found automatically">♪ audio</button>
          <span data-id="tl-ainfo" hidden><span data-id="tl-aname"></span> <button class="kkd-x" data-id="tl-arm" data-tip="Remove the sound">×</button>
            <span>starts at</span><input type="text" data-id="tl-aoff" data-tip="Seconds into the sound file where the timeline starts"><span>s · bpm</span><input type="text" data-id="tl-bpm" data-tip="Tempo: found in the sound, type to correct">
            <button class="kkd-pill" data-id="tl-half" style="padding:2px 7px" data-tip="Half the tempo">÷2</button><button class="kkd-pill" data-id="tl-dbl" style="padding:2px 7px" data-tip="Double the tempo">×2</button>
            <button class="kkd-pill" data-id="tl-beat1" style="padding:2px 7px" data-tip="The playhead is on a beat: move the beat grid here">beat here</button></span>
          <span>length</span><input type="text" data-id="tl-dur" data-tip="Length in seconds"><span>s · fps</span><input type="text" data-id="tl-fps">
        </div>
        <div class="kkd-tlwrap"><canvas class="kkd-tlc" data-id="tlc" data-tip="Click / drag to scrub · drag a ◆ to move it in time · Alt click = smooth / linear / hold · Del removes it|, / ."></canvas></div>
      </div>
      <div class="kkd-side">
        <div class="kkd-split v" data-split="sw" data-tip="Drag to make the panel wider or narrower · double click resets"></div>
        <div class="kkd-sec" data-id="pm"></div>
        <div class="kkd-sec"><h2>layers <span class="kkd-grow"></span><span data-id="count"></span></h2><div class="kkd-layers" data-id="layers" role="listbox"></div></div>
        <div class="kkd-split h" data-split="lh" data-tip="Drag to show more or fewer layers · double click resets"></div>
        <div class="kkd-sec" data-id="insp">
          <h2 data-id="insp-title">layer</h2>
          <div class="kkd-field"><span>name</span><input type="text" data-id="i-name"><span></span></div>
          <div class="kkd-field"><span>blend</span><select data-id="i-blend"></select><span></span></div>
          <div class="kkd-field"><span>opacity</span><input type="range" data-id="i-op" min="0" max="1" step="0.01"><span class="v" data-id="i-op-v"></span></div>
          <div class="kkd-field" data-id="g-holes"><span>holes</span><input type="text" autocomplete="off" data-id="i-holes" placeholder="none" data-tip="Cut regions out of the base: layers below it show through · a name, group:NAME or W_F1_* (see the names in the regions view)"><span></span></div>
          <div class="kkd-field" data-id="g-clip"><span>clip to</span><input type="text" autocomplete="off" data-id="i-clip" placeholder="anywhere" data-tip="Keep the layer inside these regions · type a name, group:NAME or W_F1_*, or click a region with the R tool|R"><span></span></div>
          <div class="kkd-field" data-id="g-mask"><span>mask by</span><select data-id="i-mask" data-tip="Use another layer to keep or cut this one · the mask layer can stay hidden|Ctrl Alt G"></select><span></span></div>
          <div class="kkd-field" data-id="g-maskopt"><span></span><div class="kkd-seg" data-id="i-mmode">
            <button class="kkd-pill" data-m="shape" data-tip="Keep where the mask layer has pixels (its alpha)">shape</button>
            <button class="kkd-pill" data-m="brightness" data-tip="Keep where the mask layer is bright, like a black / white matte">brightness</button>
            <button class="kkd-pill" data-id="i-minv" data-tip="Cut the mask out of this layer instead of keeping it|Alt I">cut out</button></div><span></span></div>
          <div data-id="g-shape"><div class="kkd-sub">shape</div>
            <div class="kkd-field"><span>colour</span><input type="color" data-id="sh-col"><span></span></div>
            <div class="kkd-field" data-tip="Soft edge (px of the delivery size); it spreads outside the shape"><span>feather</span><input type="range" data-id="sh-f" min="0" max="200" step="1"><span class="v" data-id="sh-f-v"></span></div>
          </div>
          <div data-id="g-paint"><div class="kkd-sub">brush</div>
            <div class="kkd-field"><span>colour</span><input type="color" data-id="i-col"><span></span></div>
            <div class="kkd-field"><span>size</span><input type="range" data-id="i-size" min="4" max="600" step="1"><span class="v" data-id="i-size-v"></span></div>
            <div class="kkd-field"><span></span><button class="kkd-pill" data-id="i-clear" data-tip="Clear everything painted on this layer">clear</button><span></span></div>
          </div>
          <div data-id="g-adj"><div class="kkd-sub">grade</div>
            ${[["brightness", "exposure", .3, 2, .01], ["contrast", "contrast", .3, 2, .01], ["saturate", "saturation", 0, 2.5, .01], ["hue", "hue", -180, 180, 1], ["sepia", "warmth", 0, 1, .01]]
              .map(([k, t, a, b, st]) => `<div class="kkd-field"><span>${t}</span><input type="range" data-id="a-${k}" min="${a}" max="${b}" step="${st}"><span class="v" data-id="a-${k}-v"></span></div>`).join("")}
          </div>
          <div data-id="g-light"></div>
          <div data-id="g-media"></div>
          <div class="kkd-field" data-id="g-bg"><span>background</span><button class="kkd-pill" data-id="i-bg" aria-pressed="false" data-tip="Cut the subject out with ComfyUI's background removal model (BiRefNet) · the full-size cutout is made on apply">remove</button><span></span></div>
          <div data-id="g-diff">
            <div class="kkd-field"><span>after apply</span><div class="kkd-seg" data-id="i-action">
              <button class="kkd-pill" data-a="rediffuse" data-tip="Region Sampler pass on this layer with its prompt">re-diffuse</button>
              <button class="kkd-pill" data-a="edges" data-tip="Only blend the outline into the facade (seam pass)">edges</button>
              <button class="kkd-pill" data-a="keep" data-tip="Leave the pixels exactly as placed">keep</button></div><span></span></div>
            <div class="kkd-field"><textarea data-id="i-prompt" placeholder="prompt for this layer"></textarea></div>
            <div class="kkd-field"><span>denoise</span><input type="range" data-id="i-den" min="0" max="1" step="0.05"><span class="v" data-id="i-den-v"></span></div>
            <div class="kkd-field"><span></span><button class="kkd-pill accent" data-id="d-go" data-tip="Diffuse this layer now with its prompt and denoise (re-diffuse = the shape, edges = only its outline) · the result comes back as a new layer above|D">diffuse</button><span></span></div>
            <div class="kkd-lstatus" data-id="d-status"></div>
            <div class="kkd-sub">animate (ltx) <span class="kkd-grow"></span><button class="kkd-pill" style="padding:2px 8px;font-size:12px" data-id="v-on" aria-pressed="false" data-tip="Let LTX animate this layer's region in the Region Video Sampler (plan: animate, video_prompt, t_start, t_end, motion)">off</button></div>
            <div data-id="v-box">
              <div class="kkd-field"><textarea data-id="v-prompt" placeholder="what happens here over time (empty = the layer prompt)"></textarea></div>
              <div class="kkd-field"><span>from</span><input type="text" data-id="v-start" data-tip="Seconds on the timeline"><button class="kkd-pill" style="padding:2px 8px;font-size:12px" data-id="v-start-here" data-tip="Start at the playhead">here</button></div>
              <div class="kkd-field"><span>to</span><input type="text" data-id="v-end" placeholder="end" data-tip="Seconds on the timeline · empty = to the end"><button class="kkd-pill" style="padding:2px 8px;font-size:12px" data-id="v-end-here" data-tip="End at the playhead">here</button></div>
              <div class="kkd-field"><span>motion</span><input type="range" data-id="v-motion" min="0" max="1" step="0.05"><span class="v" data-id="v-motion-v"></span></div>
            </div>
          </div>
          <div data-id="g-react"></div>
          <div data-id="g-fx"></div>
          <div data-id="g-motion"></div>
        </div>
      </div>`;
    document.body.appendChild(this.root);
    if (!document.getElementById("kkdLumA")) {        // the brightness-mask filter (display:none would disable url() filters)
      const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
      svg.setAttribute("width", "0"); svg.setAttribute("height", "0"); svg.style.position = "absolute";
      svg.innerHTML = '<filter id="kkdLumA" color-interpolation-filters="sRGB"><feColorMatrix type="matrix" values="0 0 0 0 1  0 0 0 0 1  0 0 0 0 1  0.2126 0.7152 0.0722 0 0"/><feComposite in2="SourceGraphic" operator="in"/></filter>';
      document.body.appendChild(svg);
    }
    for (const id of ["layers", "g-motion", "g-fx", "g-react", "i-blend", "i-mask"]) this.guardHTML(this.$(id));
    const unguard = e => { for (let n = e.target; n && n !== this.root; n = n.parentElement) if (n._kkdGuard) n._kkdGuard.html = null; };
    this.root.addEventListener("input", unguard, true); this.root.addEventListener("change", unguard, true);
    this.menu = document.createElement("div"); this.menu.className = "kkd-menu"; this.menu.hidden = true;
    this.menu.innerHTML = `<button data-new="import">import image or video… <kbd>Ctrl O</kbd></button><button data-new="linkvideo" data-tip="Big files (over 100 MB): the video stays where it is on disk">video from a path…</button><button data-new="paint">paint layer <kbd>Shift N</kbd></button>
      <button data-new="adjust">colour grade <kbd>Shift A</kbd></button>
      <button data-new="solid" data-tip="A layer of one colour over the whole canvas (clip it to regions, mask it, blend it)">solid colour</button>
      <button data-new="rect">rectangle</button><button data-new="ellipse">ellipse</button>
      <button data-new="poly" data-tip="Click the corners on the canvas · Enter or double click finishes · Esc cancels">polygon (click points)</button>
      <button data-new="light" ${this.m.scene ? "" : "disabled"} data-tip="${this.m.scene ? "Relight the 3D scene: HDRI, lamps placed on the facade, glowing layers" : "Connect a scene (kubakub scene render) to the director and run once"}">light (relight the scene) <kbd>Shift L</kbd></button>
      <button data-new="dup">duplicate selection <kbd>Ctrl J</kbd></button>
      <button data-new="inputs" data-id="m-inputs" hidden>bring back deleted node inputs</button>`;
    this.filemenu = document.createElement("div"); this.filemenu.className = "kkd-menu"; this.filemenu.hidden = true;
    this.filemenu.innerHTML = `<button data-f="save">save as… <kbd>Ctrl S</kbd></button><button data-f="open">open…</button>
      <button data-f="export">export as file</button><button data-f="import">import a file…</button>`;
    this.docfile = Object.assign(document.createElement("input"), { type: "file", accept: ".json,application/json", hidden: true });
    this.keypop = document.createElement("div"); this.keypop.className = "kkd-pop"; this.keypop.hidden = true;
    this.keypop.innerHTML = `<h3>shortcuts</h3><div class="kkd-keys">${KEYS.map(([k, d]) => `<span>${k.split(" / ").map(x => x.split(" ").map(y => `<kbd>${y}</kbd>`).join(" ")).join(" / ")}</span><span>${d}</span>`).join("")}</div>`;
    this.outpop = document.createElement("div"); this.outpop.className = "kkd-pop"; this.outpop.hidden = true;
    this.diffpop = document.createElement("div"); this.diffpop.className = "kkd-pop"; this.diffpop.hidden = true;
    this.exportpop = document.createElement("div"); this.exportpop.className = "kkd-pop"; this.exportpop.hidden = true;
    this.regpick = document.createElement("div"); this.regpick.className = "kkd-menu kkd-regpick"; this.regpick.hidden = true;
    this.tipEl = document.createElement("div"); this.tipEl.className = "kkd-tip";
    this.dropzone = document.createElement("div"); this.dropzone.className = "kkd-dropzone"; this.dropzone.hidden = true; this.dropzone.textContent = "drop images, videos or a sound";
    this.file = Object.assign(document.createElement("input"), { type: "file", accept: "image/*,video/*,.mov,.mxf,.mkv,.avi", multiple: true, hidden: true });
    for (const el of [this.menu, this.filemenu, this.docfile, this.keypop, this.outpop, this.exportpop, this.regpick, this.tipEl, this.dropzone, this.file]) document.body.appendChild(el);
    this.root.appendChild(this.diffpop);                // inside the root: its fields use the window's styles
    this.cv = this.$("cv"); this.ctx = this.cv.getContext("2d"); this.tlc = this.$("tlc");
    this.bind();
    this.ro = new ResizeObserver(() => this.fit()); this.ro.observe(this.$("stage"));
    this.root.focus();
  }
  busy(on, text) { const b = this.$("busy"); b.hidden = !on; if (text) b.textContent = text; }
  renderViews() {
    const names = Object.keys(this.views).filter(k => !k.startsWith("_"));
    const tips = { matrix: "The base image", clay: "Clay render from the projector camera", depth: "Depth: warm = close to the audience",
      ids: "ID shelves of the 3D scene, with the region names", regions: "Every region in its own colour with its name · use the names in clip to / holes" };
    this.$("views").innerHTML = names.map((v, i) => `<button class="kkd-pill" data-view="${v}" aria-pressed="${v === this.view}" data-tip="${tips[v] || v}|${i + 1}">${v}</button>`).join("");
  }
  setView(v) { if (!this.views[v]) return; this.view = v; this.renderViews(); this.draw(); this.renderLayers(); }
  setTool(t) {
    this.tool = t; for (const k of ["move", "pick", "brush", "erase", "hand"]) this.$("t-" + k).setAttribute("aria-pressed", String(k === t));
    this.cv.style.cursor = t === "hand" ? "grab" : t === "pick" || t === "poly" ? "crosshair" : (t === "brush" || t === "erase") ? "none" : "default";
    if (t !== "poly") this.polyDraft = null;
    this.$("brush").hidden = !(t === "brush" || t === "erase"); this.draw();
  }
  thumb(L) {                                         // cached until the layer's pixels change
    const key = L.kind === "base" ? this.baseImage() : L.kind === "paint" ? "v" + (L.ver || 0) : L.kind === "adjust" ? "adj" : L.kind === "shape" ? "s" + L.color + L.shape?.type : L.img;
    if (L._thumb && L._thumbKey === key) return L._thumb;
    const c = mkCanvas(68, 46), g = c.getContext("2d"); g.fillStyle = "#3b3b3e"; g.fillRect(0, 0, 68, 46);
    if (L.kind === "base") g.drawImage(this.baseImage(), 0, 0, 68, 46);
    else if (L.kind === "paint") g.drawImage(L.canvas, 0, 0, 68, 46);
    else if (L.kind === "light") { if (L.img) g.drawImage(L.img, 0, 0, 68, 46); else { g.fillStyle = ORANGE; g.beginPath(); g.arc(34, 23, 7, 0, 7); g.fill(); } }
    else if (L.kind === "shape") { g.fillStyle = isHex(L.color) ? L.color : "#fff"; g.beginPath();
      if (L.shape?.type === "ellipse") g.ellipse(34, 23, 22, 15, 0, 0, 7); else if (L.shape?.type === "poly") { g.moveTo(14, 38); g.lineTo(54, 38); g.lineTo(34, 8); } else g.rect(12, 8, 44, 30); g.fill(); }
    else if (L.kind === "adjust") { const gr = g.createLinearGradient(0, 0, 68, 0); gr.addColorStop(0, "#222"); gr.addColorStop(1, ORANGE); g.fillStyle = gr; g.fillRect(8, 16, 52, 14); }
    else if (L.img) { const k = Math.min(60 / iw(L.img), 40 / ih(L.img)); g.drawImage(L.img, 34 - iw(L.img) * k / 2, 23 - ih(L.img) * k / 2, iw(L.img) * k, ih(L.img) * k); }
    L._thumbKey = key; return (L._thumb = c.toDataURL());
  }
  maskLabel(i) {
    const L = this.layers[i], s = this.maskSource(i); if (!s) return "";
    return ` · ${L.mask.invert ? "cut by" : "masked by"} ${L.mask.by === "below" ? "layer below" : esc(s.name)}`;
  }
  relTag(i) {
    const b = this.layers.findIndex(q => q.kind === "base"), L = this.layers[i];
    if (L.kind === "base") return `<span class="kkd-tag wall">base</span>`;
    if (L.kind === "adjust") return `<span class="kkd-tag wall">grade</span>`;
    if (L.kind === "light") return `<span class="kkd-tag light">light</span>`;
    return i < b ? "" : `<span class="kkd-tag behind" data-tip="Under the base: shows only through its holes">behind</span>`;
  }
  renderLayers() {
    this.$("layers").kkdHTML = this.layers.map((L, i) => `
      <div class="kkd-layer" role="option" tabindex="0" aria-selected="${i === this.sel}" data-i="${i}">
        <span class="kkd-grip" data-tip="Drag to change what is in front|Ctrl ] / Ctrl [">⋮⋮</span>
        <button class="kkd-eye ${L.visible ? "on" : ""}" data-eye="${i}" aria-label="show or hide" data-tip="Show / hide|Alt H"></button>
        <img class="kkd-thumb" alt="" src="${this.thumb(L)}">
        <span class="kkd-lname">${L.mask.by === "below" ? "↳ " : ""}${esc(L.name)}<small>${L.kind === "image" ? (L.blend !== "normal" ? L.blend : isVideo(L) ? "video" : "image") : L.kind === "shape" ? (L.w >= this.W * 0.98 && L.h >= this.H * 0.98 && L.shape?.type === "rect" ? "solid" : L.shape?.type || "shape") : L.kind}${L.clip ? " · in " + esc(L.clip) : ""}${this.maskLabel(i)}${L.kind === "image" && !L.img ? (isVideo(L) ? " · video not loaded" : " · missing file") : ""}${L.video?.on ? " · ltx" : ""}${L.kind === "light" ? " · " + esc(L.pendingKey ? "rendering…" : L.err ? "failed" : L.img ? L.light.environment + ", " + L.light.lights.filter(q => q.on !== false).length + " lights" : "not rendered") : ""}</small></span>
        ${this.relTag(i)}
      </div>`).join("");
    this.$("count").textContent = this.layers.length;
    if (this._selShown !== this.sel) { this._selShown = this.sel; this.$("layers").querySelector('[aria-selected="true"]')?.scrollIntoView({ block: "nearest" }); }
  }
  fillRegionList() {                                  // entries for the clip to / holes picker
    // groups (windows, columns …), tags (left / centre / right, floors, other tag passes), handy combinations
    // ('tag:left group:windows' = the windows of the left section), then every single region
    const regs = this.m.regions || [], groups = this.m.groups || [];
    const tags = [...new Set(regs.flatMap(r => r.tags || []))].sort();
    const isSection = t => /^(left|right)(_\d+)?$|^centre$/.test(t), isFloor = t => /^(ground_floor|floor_\d+|roof)$/.test(t);
    const has = (t, g) => regs.some(r => r.group === g && (r.tags || []).includes(t));
    const combos = [...tags.filter(isSection).flatMap(t => groups.filter(g => has(t, g)).map(g => `tag:${t} group:${g}`)),
      ...tags.filter(isFloor).flatMap(t => groups.filter(g => g === "windows" && has(t, g)).map(g => `tag:${t} group:${g}`))];
    this.regEntries = [...groups.map(g => "group:" + g), ...tags.map(t => "tag:" + t), ...combos, ...regs.map(r => r.name)];
  }
  openRegPick(input, filter) {                        // every region, filtered only by what was typed since opening
    const cur = input.value.trim(), f = (filter || "").toLowerCase(), all = this.regEntries || [];
    const hits = f ? all.filter(n => n.toLowerCase().includes(f)) : all;
    const none = input.dataset.id === "i-holes" ? "none" : "anywhere";
    this.regpick.innerHTML = (f ? "" : `<button data-v="" class="none${cur ? "" : " on"}">${none}</button>`) +
      hits.slice(0, 400).map(n => `<button data-v="${esc(n)}"${n === cur ? ' class="on"' : ""}>${esc(n)}</button>`).join("") +
      (hits.length > 400 ? `<button disabled class="none">${hits.length - 400} more · type to filter</button>` : "") +
      (!hits.length && f ? `<button disabled class="none">no region matches · Enter keeps "${esc(filter)}" as a pattern</button>` : "");
    const r = input.getBoundingClientRect();
    Object.assign(this.regpick.style, { top: (r.bottom + 4) + "px", left: r.left + "px", minWidth: r.width + "px" });
    this.regpick.hidden = false; this.regpick.dataset.for = input.dataset.id; this.regHi = -1;
    this.regpick.querySelector(".on")?.scrollIntoView({ block: "nearest" });
  }
  renderInspector() {
    const L = this.layers[this.sel]; if (!L) return; const $ = id => this.$(id);
    $("insp-title").textContent = L.name;
    $("i-name").value = L.name;
    $("i-blend").kkdHTML = (this.m.blend_modes || Object.keys(OPS)).map(b => `<option ${b === L.blend ? "selected" : ""}>${b}</option>`).join("");
    $("i-op").value = L.opacity; $("i-op-v").textContent = Math.round(L.opacity * 100) + "%";
    const typing = el => el === document.activeElement && this.regLayer === L;      // do not overwrite what is being typed
    $("g-holes").hidden = L.kind !== "base"; if (!typing($("i-holes"))) $("i-holes").value = L.holes || "";
    $("g-clip").hidden = L.kind === "base"; if (!typing($("i-clip"))) $("i-clip").value = L.clip || "";
    const opts = [["", "none"], ["below", "layer below (clipping)"], ...this.layers.filter(q => q !== L && q.kind !== "adjust").map(q => [q.id, q.name + (q.visible ? "" : " (hidden)")])];
    $("i-mask").kkdHTML = opts.map(([v, t]) => `<option value="${esc(v)}" ${L.mask.by === v ? "selected" : ""}>${esc(t)}</option>`).join("");
    $("g-mask").hidden = L.kind === "base"; $("g-maskopt").hidden = !L.mask.by || L.kind === "base";
    this.root.querySelectorAll("[data-m]").forEach(b => b.setAttribute("aria-pressed", String(b.dataset.m === L.mask.mode)));
    $("i-minv").setAttribute("aria-pressed", String(!!L.mask.invert));
    $("g-diff").hidden = !(L.kind === "image" || L.kind === "paint") || isVideo(L);
    $("g-media").hidden = !isVideo(L); if (isVideo(L)) this.renderMediaPanel(); else $("g-media").innerHTML = "";
    if (!$("g-diff").hidden) {
      const V = L.video || {}; $("v-on").setAttribute("aria-pressed", String(!!V.on)); $("v-on").textContent = V.on ? "on" : "off"; $("v-box").hidden = !V.on;
      $("v-prompt").value = V.prompt || ""; $("v-start").value = +(V.t_start || 0).toFixed(2); $("v-end").value = V.t_end >= 0 ? +(+V.t_end).toFixed(2) : "";
      $("v-motion").value = V.motion ?? 1; $("v-motion-v").textContent = (+(V.motion ?? 1)).toFixed(2);
    }
    if (!$("g-diff").hidden) { $("d-status").textContent = L.dstatus || ""; $("d-status").classList.toggle("err", !!L.derr);
      $("d-go").disabled = !!L.diffusing; $("d-go").textContent = L.diffusing ? "diffusing…" : "diffuse"; }
    $("g-bg").hidden = L.kind !== "image" || isVideo(L); $("i-bg").setAttribute("aria-pressed", String(!!L.remove_bg));
    $("i-bg").textContent = L.remove_bg ? "removed" : "remove";
    this.root.querySelectorAll("[data-a]").forEach(b => b.setAttribute("aria-pressed", String(b.dataset.a === L.action)));
    $("i-prompt").value = L.prompt; $("i-den").value = L.denoise; $("i-den-v").textContent = (+L.denoise).toFixed(2);
    const lights = this.layers.filter(q => q.kind === "light");
    $("g-react").hidden = !["image", "paint", "shape"].includes(L.kind) || !lights.length;
    if (!$("g-react").hidden) { const R = L.light_react;
      $("g-react").kkdHTML = `<div class="kkd-field" data-tip="Shade this layer with the lamps of a light layer: brighter where they hit, darker in shadow, tinted by coloured lamps (the light layer can stay hidden)"><span>light</span>
        <select data-id="r-by"><option value="off" ${R ? "" : "selected"}>no reaction</option>${lights.map(q => `<option value="${esc(q.id)}" ${R && (R.by || lights[0].id) === q.id ? "selected" : ""}>shaded by ${esc(q.name)}</option>`).join("")}</select><span></span></div>
        ${R ? `<div class="kkd-field"><span>amount</span><input type="range" data-id="r-amt" min="0" max="1" step="0.01" value="${+(R.amount ?? 1)}"><span class="v" data-id="r-amt-v">${Math.round(+(R.amount ?? 1) * 100)}%</span></div>` : ""}`; }
    $("g-fx").hidden = !["image", "paint", "shape", "base", "adjust"].includes(L.kind);
    if (!$("g-fx").hidden) { const f = fxOf(L), shown = this.fxShown(L), rest = FX_GROUPS.filter(g => !shown.includes(g[0]));
      $("g-fx").kkdHTML = `<div class="kkd-sub">effects <span class="kkd-grow"></span>${rest.length ? `<select data-id="fx-add" style="font-size:12px;max-width:130px"><option value="">+ effect</option>${rest.map(g => `<option value="${g[0]}">${g[0]}</option>`).join("")}</select>` : ""}${shown.length ? `<button class="kkd-pill" style="padding:2px 8px;font-size:12px" data-fxreset data-tip="All effects off">reset</button>` : ""}</div>` +
        FX_GROUPS.filter(g => shown.includes(g[0])).map(([name, tip, keys]) => `<div class="kkd-fxh" data-tip="${esc(tip)}">${name}<span class="kkd-grow"></span><button class="kkd-x" data-fxdel="${name}" data-tip="Remove ${name}">×</button></div>` +
          keys.map(key => { const [, label, lo, hi, st, unit, t] = FX_UI.find(x => x[0] === key);
            return `<div class="kkd-field" data-tip="${esc(t)}"><span>${label}</span><input type="range" data-fx="${key}" min="${lo}" max="${hi}" step="${st}" value="${f[key]}"><span class="v" data-fxv="${key}">${+(+f[key]).toFixed(2)}${unit}</span></div>`; }).join("")).join(""); }
    this.renderMotion(L);
    $("g-shape").hidden = L.kind !== "shape";
    if (L.kind === "shape") { $("sh-col").value = isHex(L.color) ? L.color : "#ffffff"; $("sh-f").value = +L.shape?.feather || 0; $("sh-f-v").textContent = (+L.shape?.feather || 0) + " px"; }
    $("g-paint").hidden = L.kind !== "paint"; if (L.kind === "paint") { $("i-col").value = L.color; $("i-size").value = L.size; $("i-size-v").textContent = L.size + " px"; }
    $("g-adj").hidden = L.kind !== "adjust";
    if (L.kind === "adjust") for (const k of Object.keys(L.adjust)) { $("a-" + k).value = L.adjust[k]; $("a-" + k + "-v").textContent = k === "hue" ? L.adjust[k] + "°" : (+L.adjust[k]).toFixed(2); }
    $("g-light").hidden = L.kind !== "light";
    if (L.kind === "light") this.renderLightPanel(); else $("g-light").innerHTML = "";
    this.fillRanges();
  }

  // ---- the projection mask slot: the building's silhouette at the delivery size, always on top
  renderPm() {
    const box = this.$("pm"); if (!box) return;
    const P = this.pm, srcName = P.source === "scene" ? "the 3D scene" : P.source ? P.source.replace(/^(file|path):/, "").split(/[\\/]/).pop() : "none";
    const open = !!this.pmOpen;
    box.innerHTML = `<h2 class="kkd-pmhead" data-pm="toggle" aria-expanded="${open}" data-tip="Show / hide the projection mask settings"><span class="kkd-chev">›</span>projection mask ${open ? "" : `<small>${esc(P.source ? srcName : "none")}</small>`}<span class="kkd-grow"></span><button class="kkd-pill" data-pm="on" aria-pressed="${!!P.on}" style="padding:2px 10px;font-size:12px"
        data-tip="Everything outside the building is black: in the window, the render, the sequence and the export (transparent with alpha)">${P.on ? "on" : "off"}</button></h2>
      ${open ? `<div class="kkd-field"><span>source</span><div class="kkd-seg">
        <button class="kkd-pill" data-pm="scene" aria-pressed="${P.source === "scene"}" ${this.views._silhouette ? "" : "disabled"} data-tip="${this.views._silhouette ? "The silhouette of the connected 3D scene" : "Connect a scene (kubakub scene render) and run once"}">scene</button>
        <button class="kkd-pill" data-pm="file" aria-pressed="${String(P.source).startsWith("file:")}" data-tip="The template's mask image (PNG with alpha, or white building on black)">file…</button>
        <button class="kkd-pill" data-pm="path" aria-pressed="${String(P.source).startsWith("path:")}" data-tip="A mask image on disk, read where it is">path…</button></div><span></span></div>
      <div class="kkd-lstatus${String(this.pmNote || "").startsWith("mask:") ? " err" : ""}">${esc(srcName)}${this.pmNote ? " · " + esc(this.pmNote) : ""}</div>
      ${P.source ? `<div class="kkd-field"><span>view</span><div class="kkd-seg">${[["black", "As projected: black outside"], ["dim", "Outside darkened: see what you place over the edge"], ["off", "Not shown while working (still applied on apply and export)"]]
        .map(([v, t]) => `<button class="kkd-pill" data-pmv="${v}" aria-pressed="${P.view === v}" data-tip="${t}">${v}</button>`).join("")}</div><span></span></div>
      <div class="kkd-field"><span>invert</span><button class="kkd-pill" data-pm="invert" aria-pressed="${!!P.invert}" data-tip="Swap building and outside">${P.invert ? "inverted" : "normal"}</button><span></span></div>
      <div class="kkd-field" data-tip="Grow (+) or shrink (-) the building by pixels of the delivery size"><span>grow</span><input type="range" data-pmr="grow" min="-50" max="50" step="1" value="${+P.grow || 0}"><span class="v" data-pmv-v="grow">${+P.grow || 0} px</span></div>
      <div class="kkd-field" data-tip="Soft edge width in pixels of the delivery size"><span>feather</span><input type="range" data-pmr="feather" min="0" max="100" step="1" value="${+P.feather || 0}"><span class="v" data-pmv-v="feather">${+P.feather || 0} px</span></div>` : ""}` : ""}`;
    this.fillRanges();
  }
  async setPmSource(kind) {
    if (kind === "scene") { this.snap(); Object.assign(this.pm, { source: "scene", on: true }); this.loadPm(); return; }
    if (kind === "path") {
      const p = await this.ask("mask from a path", "The full path of the template's mask image (PNG with alpha, or white building on black). It is read where it is.",
        [["cancel", "cancel"], ["ok", "use", "accent"]], String(this.pm.source).startsWith("path:") ? this.pm.source.slice(5) : "");
      if (!p) return; this.snap(); Object.assign(this.pm, { source: "path:" + p.replace(/^"|"$/g, ""), on: true }); this.loadPm(); return;
    }
    this.pmFile.click();                             // file: upload into input/kuba_director
  }

  // ---- the video panel (inspector of a video layer)
  renderMediaPanel() {
    const L = this.cur(), box = this.$("g-media"), I = L.vinfo;
    if (!I) { box.innerHTML = `<div class="kkd-lstatus err">video not loaded: ${esc(L.verr || "run the workflow once")}</div>`; return; }
    // L.media is never replaced here: a timeline drag holds on to its segments
    const M = L.media || (L.media = this.mediaDefaults(null)), fmt = v => (+v).toFixed(3).replace(/\.?0+$/, "") || "0";
    const num = (i, k, v, tip) => `<input type="text" data-ms="${i}" data-mk="${k}" value="${v}" data-tip="${esc(tip)}" style="width:100%;padding:2px 6px">`;
    box.innerHTML = `
      <div class="kkd-sub">video <span class="kkd-grow"></span><span style="font-weight:400;letter-spacing:0;text-transform:none">${I.w}×${I.h} · ${fmt((+I.fps).toFixed(2))} fps · ${(+I.duration).toFixed(2)} s${I.alpha ? " · alpha" : ""}${I.audio ? " · sound" : ""}</span></div>
      ${M.segments.map((q, i) => `<div data-msi="${i}" style="border:1px solid ${i === this.segSel ? "var(--orange)" : "var(--line)"};border-radius:10px;padding:6px 8px;margin:4px 0;font-size:12px;display:grid;grid-template-columns:auto minmax(0,1fr) auto minmax(0,1fr);gap:4px 6px;align-items:center">
        <span>at</span>${num(i, "start", fmt(q.start), "Where it starts on the timeline (s) · drag the clip on the timeline")}
        <span>speed</span>${num(i, "speed", fmt(q.speed), "2 = twice as fast, 0.5 = slow motion · Alt-drag the clip end on the timeline")}
        <span>in</span>${num(i, "in", fmt(q.in), "From this second of the file · drag the clip start on the timeline")}
        <span>out</span>${num(i, "out", q.out < 0 ? "" : fmt(q.out), "To this second of the file (empty = its end) · drag the clip end on the timeline")}
        <span>length</span>${num(i, "length", q.length < 0 ? "" : fmt(q.length), "Time on the timeline (empty = as long as in → out plays) · Shift-drag the clip end")}
        <select data-ms="${i}" data-mk="fill" style="padding:2px 6px" data-tip="After out, until the length is over"><option value="hold" ${q.fill !== "loop" ? "selected" : ""}>then hold</option><option value="loop" ${q.fill === "loop" ? "selected" : ""}>then loop</option></select>
        <button class="kkd-x" data-mdel="${i}" data-tip="Remove this piece (Del on the timeline)">×</button>
        <span></span><button class="kkd-pill" style="padding:1px 8px;font-size:12px;grid-column:span 3;justify-self:start" data-mrev="${i}" aria-pressed="${!!q.reverse}" data-tip="Play this piece backwards (from out to in, the sound too)">${q.reverse ? "◀ backwards" : "forwards"}</button></div>`).join("")}
      <div class="kkd-field"><span>cut</span><div class="kkd-seg"><button class="kkd-pill" data-id="m-split" data-tip="Cut the clip at the playhead|Ctrl K">at playhead</button>
        <button class="kkd-pill" data-id="m-tl" data-tip="Timeline length = the end of this video">timeline = video</button></div><span></span></div>
      <div class="kkd-field"><span>fit</span><div class="kkd-seg">${[["canvas", "The whole matrix"], ["regions", "The box around the regions of 'clip to' (windows, a floor …)"], ["native", "The video's own pixel size"]]
        .map(([v, tip]) => `<button class="kkd-pill" data-mfit="${v}" data-tip="${tip}">${v}</button>`).join("")}</div><span></span></div>
      <div class="kkd-field"><span>sound</span><input type="range" data-id="m-vol" min="0" max="2" step="0.05" value="${M.volume}"><span class="v" data-id="m-vol-v">${Math.round(M.volume * 100)}%</span></div>
      <div class="kkd-field"><span>frames</span><button class="kkd-pill" data-id="m-blend" aria-pressed="${!!M.blend_frames}" data-tip="In the render: blend neighbouring frames (smooth slow motion, other fps) or take the nearest one · the window shows the nearest">${M.blend_frames ? "blended" : "nearest"}</button><span></span></div>`;
  }
  fitVideo(how) {
    const L = this.cur(); if (!L?.vinfo) return;
    let box;
    if (how === "canvas") box = [0, 0, this.W, this.H];
    else if (how === "native") { const cx = L.x + L.w / 2, cy = L.y + L.h / 2; box = [cx - L.vinfo.w / 2, cy - L.vinfo.h / 2, cx + L.vinfo.w / 2, cy + L.vinfo.h / 2]; }
    else {
      if (!L.clip) { note("choose regions in 'clip to' first (a floor, the windows …)"); return; }
      const ids = this.selectIds(L.clip), bb = (this.m.regions || []).filter(r => ids.has(r.id)).map(r => this.rbox(r)).filter(Boolean);
      if (!bb.length) { note("no region matches '" + L.clip + "'"); return; }
      box = [Math.min(...bb.map(b => b[0])), Math.min(...bb.map(b => b[1])), Math.max(...bb.map(b => b[2])), Math.max(...bb.map(b => b[3]))];
    }
    this.snap(); Object.assign(L, { x: box[0], y: box[1], w: box[2] - box[0], h: box[3] - box[1], rotation: 0 }); this.refresh();
  }

  // ---- the light panel (inspector of a light layer)
  renderLightPanel() {
    const L = this.cur(), R = L.light, box = this.$("g-light"), f = this.frame() || { width_m: 30, top_m: 20, bottom_m: 0 };
    this.lsel = Math.min(this.lsel, R.lights.length - 1);
    const q = R.lights[this.lsel];
    const range = (key, label, min, max, step, val, unit, tip) => `<div class="kkd-field"${tip ? ` data-tip="${esc(tip)}"` : ""}><span>${label}</span>
      <input type="range" data-l="${key}" min="${min}" max="${max}" step="${step}" value="${val}"><span class="v" data-v="${key}">${this.fmtL(val, unit)}</span></div>`;
    const envs = (this.m.scene?.environments || ["none", "night"]).map(e => `<option ${e === R.environment ? "selected" : ""}>${e}</option>`).join("");
    const reach = Math.ceil(f.width_m / 2 + 10);
    let lamp = "";
    if (q) {
      lamp = `<div class="kkd-field"><span>colour</span><input type="color" data-l="q.color" value="${esc(q.color || "#ffffff")}"><span></span></div>`;
      if (q.type === "sun") {
        lamp += range("q.power", "strength", 0, 20, 0.1, +q.power || 0, "", "Sun strength (W/m²; 3 = soft daylight)")
          + range("q.azimuth", "direction", -180, 180, 1, +q.azimuth || 0, "°", "0 = from the audience side, + = from the right")
          + range("q.elevation", "elevation", -10, 90, 1, +q.elevation || 0, "°", "Height of the sun above the horizon");
      } else {
        const pmax = { point: 5000, area: 20000, spot: 30000 }[q.type];
        lamp += range("q.power", "power", 0, pmax, 10, +q.power || 0, " W")
          + range("q.x", "across", -reach, reach, 0.1, +q.x || 0, " m", "Along the wall from the centre of the projector frame · or drag the dot on the facade")
          + range("q.height", "height", -2, Math.ceil(f.top_m + 10), 0.1, +q.height || 0, " m", "Above the ground · or drag the dot on the facade")
          + range("q.distance", "out", -2, 60, 0.1, +q.distance || 0, " m", "Distance in front of the wall, towards the audience (negative = inside the building)")
          + (q.type === "spot" ? range("q.angle", "cone", 5, 160, 1, +q.angle || 45, "°") : "")
          + range("q.size", q.type === "area" ? "size" : "softness", q.type === "area" ? 0.1 : 0.01, q.type === "area" ? 30 : 3, 0.01, +q.size || 0.3, " m",
            q.type === "area" ? "Side of the square lamp: bigger = softer shadows" : "Radius of the lamp: bigger = softer shadows");
      }
    }
    const layerOpts = this.layers.filter(z => z !== L && (z.kind === "image" || z.kind === "paint" || z.kind === "base")).map(z => ["layer:" + z.id, z.name]);
    const glow = R.glow.map((g, i) => {
      const isLayer = String(g.by).startsWith("layer:");
      const opts = [...layerOpts, ["", "regions…"]].map(([v, t]) => `<option value="${esc(v)}" ${(isLayer ? g.by === v : v === "") ? "selected" : ""}>${esc(t)}</option>`).join("");
      return `<div class="kkd-glow" data-g="${i}"><select data-l="g.src" data-tip="What glows: a layer's shape or regions">${opts}</select>
        <input type="color" data-l="g.color" value="${esc(g.color)}"><button class="kkd-x" data-lx="glow" data-tip="Remove this glow">×</button>
        ${isLayer ? "" : `<input type="text" data-l="g.by" value="${esc(g.by)}" placeholder="group:Windows or W_F1_*" style="grid-column:1/-1">`}
        <input type="range" data-l="g.strength" min="0" max="300" step="1" value="${+g.strength}" data-tip="Glow strength (the faces light the stone around them)"></div>`;
    }).join("");
    box.innerHTML = `
      <div class="kkd-sub">render <span class="kkd-grow"></span><div class="kkd-seg">
        ${Object.keys(QUALITY).map(k => `<button class="kkd-pill" data-lq="${k}" aria-pressed="${k === this.lq}" data-tip="${k === "fast" ? "Quick previews (800 px, 16 samples)" : "Sharper previews (1600 px, 48 samples), slower"}">${k}</button>`).join("")}</div></div>
      <div class="kkd-lstatus${L.err ? " err" : ""}" data-id="l-status">${esc(L.status || "")}</div>
      <div class="kkd-sub">environment</div>
      <div class="kkd-field"><span>hdri</span><select data-l="environment" data-tip="Blender's built-in HDRIs · file = your own .hdr / .exr">${envs}</select><span></span></div>
      ${R.environment === "file" ? `<div class="kkd-field"><span>file</span><input type="text" data-l="hdri_file" value="${esc(R.hdri_file)}" placeholder="path of an .hdr / .exr file"><span></span></div>` : ""}
      ${range("env_strength", "strength", 0, 5, 0.05, R.env_strength, "")}
      ${range("env_rotation", "rotation", -180, 180, 1, R.env_rotation, "°")}
      ${range("exposure", "exposure", -5, 5, 0.1, R.exposure, "")}
      <div class="kkd-sub">lamps <span class="kkd-grow"></span>${["point", "area", "spot", "sun"].map(t => `<button class="kkd-pill" style="padding:2px 8px;font-size:12px" data-ladd="${t}" data-tip="Add a ${t} light${t === "sun" ? "" : " · or double click the facade for a point light"}">+ ${t}</button>`).join("")}</div>
      <div class="kkd-lights">${R.lights.map((z, i) => `<div class="kkd-lrow${z.on === false ? " off" : ""}" data-li="${i}" aria-selected="${i === this.lsel}">
        <span class="kkd-dot" style="background:${esc(z.color || "#fff")}"></span>
        <span>${z.type} <small style="color:var(--muted)">${z.type === "sun" ? `${Math.round(+z.azimuth || 0)}° / ${Math.round(+z.elevation || 0)}°` : `${(+z.x || 0).toFixed(1)} · ${(+z.height || 0).toFixed(1)} · ${(+z.distance || 0).toFixed(1)} m`}</small></span>
        <button class="kkd-eye ${z.on === false ? "" : "on"}" data-lon="${i}" data-tip="Lamp on / off"></button>
        <button class="kkd-x" data-lx="lamp" data-tip="Remove this lamp">×</button></div>`).join("") || `<div class="kkd-lstatus">no lamps: environment light only</div>`}</div>
      ${lamp}
      <div class="kkd-sub">glow <span class="kkd-grow"></span><button class="kkd-pill" style="padding:2px 8px;font-size:12px" data-ladd="glow" data-tip="Make a layer or regions glow (emissive faces)">+ glow</button></div>
      ${glow}
      <div class="kkd-sub">projector <span class="kkd-grow"></span><button class="kkd-pill" style="padding:2px 8px;font-size:12px" data-lpj aria-pressed="${!!R.projector.on}" data-tip="The matrix as light: cast from the projection camera onto the model, with falloff, surface angle and bounce light">${R.projector.on ? "on" : "off"}</button></div>
      ${R.projector.on ? `<div class="kkd-field"><span>casts</span><select data-l="projector.source" data-tip="What the projector shows · a single layer is cast even when it is hidden">
          ${[["below", "layers below this one"], ["base", "the matrix (base)"], ...this.layers.filter(z => z !== L && z.kind !== "adjust").map(z => ["layer:" + z.id, z.name])]
            .map(([v, t]) => `<option value="${esc(v)}" ${R.projector.source === v ? "selected" : ""}>${esc(t)}</option>`).join("")}</select><span></span></div>
        <div class="kkd-field"><span>as</span><div class="kkd-seg">${[["light", "light", "Cast from the projection camera like the real projector"],
          ["paint", "paint", "Your layers become the building's colour (textures, masks), lit by the lamps and the HDRI with their shadows"]]
          .map(([v, t, tip]) => `<button class="kkd-pill" data-lpm="${v}" aria-pressed="${R.projector.mode === v}" data-tip="${tip}">${t}</button>`).join("")}</div><span></span></div>
        ${R.projector.mode === "paint" ? "" : range("projector.brightness", "brightness", 0, 4, 0.05, R.projector.brightness, "×", "1 = the image lands at about its own brightness on a frontal wall")}` : ""}
      <div class="kkd-sub">material</div>
      ${range("clay", "clay", 0, 1, 0.01, R.clay, "", "Brightness of the model's clay material")}
      ${range("roughness", "roughness", 0, 1, 0.01, R.roughness, "")}
      <div class="kkd-field"><span>outside</span><div class="kkd-seg">${["black", "environment", "transparent"].map(b => `<button class="kkd-pill" data-lbg="${b}" aria-pressed="${b === R.background}" data-tip="${{ black: "Black where there is no building", environment: "The HDRI where there is no building", transparent: "Layers below show where there is no building" }[b]}">${b}</button>`).join("")}</div><span></span></div>
      <div class="kkd-field"><span>tone</span><div class="kkd-seg">${["AgX", "Standard"].map(b => `<button class="kkd-pill" data-lview="${b}" aria-pressed="${b === R.view}" data-tip="${b === "AgX" ? "Soft highlights: lamps and glow look natural" : "Colours as they are: the projected matrix keeps its colours"}">${b.toLowerCase()}</button>`).join("")}</div><span></span></div>
      <div class="kkd-sub">on apply (full size)</div>
      ${range("samples", "samples", 8, 512, 8, R.samples, "", "Cycles samples for the full-size render when you apply (denoised)")}
      ${range("resolution_scale", "resolution", 0.25, 1, 0.05, R.resolution_scale, "×", "Render smaller on apply and scale up: faster for 10k matrices")}`;
  }
  fmtL(v, unit) { v = +v; return (Math.abs(v) >= 100 ? Math.round(v) : Math.abs(v) >= 10 ? v.toFixed(1) : v.toFixed(2).replace(/\.?0+$/, "") || "0") + (unit || ""); }
  renderOutputs() {
    const placed = this.layers.filter(q => q.visible && (q.kind === "image" || q.kind === "paint"));
    this.outpop.innerHTML = `<h3>back into the workflow</h3><div class="kkd-outs">${[["image", `${this.W}×${this.H}`], ["layer masks", placed.length],
      ["changed", "re-diffused + edge bands"], ["regions", `${(this.m.regions || []).length} + ${placed.length}`], ["plan rules", "text"], ["document", "json"]]
      .map(([a, b]) => `<span><b>${a}</b> ${b}</span>`).join("")}</div>`;
  }
  refresh() {
    this.renderPm();
    this.renderLayers(); this.renderInspector(); this.renderOutputs(); this.draw();     // region masks stay cached
    const mi = this.menu?.querySelector("[data-id=m-inputs]"); if (mi) mi.hidden = !(this.m.inputs || []).some(i => this.dismissed.includes(i.key));
  }

  // ---- history and layer operations
  paintPix(L) {                                      // copy-on-write: snapshots share pixels until the layer is painted on
    if (!L._pix || L._pixVer !== (L.ver || 0)) { L._pix = L.canvas.getContext("2d").getImageData(0, 0, this.PW, this.PH); L._pixVer = L.ver || 0; }
    return L._pix;
  }
  state() {
    return { layers: this.layers.map(L => ({ ...L, mask: { ...L.mask }, adjust: { ...L.adjust }, light: L.light ? JSON.parse(JSON.stringify(L.light)) : undefined,
      shape: L.shape ? JSON.parse(JSON.stringify(L.shape)) : undefined, fx: L.fx ? { ...L.fx } : undefined, light_react: L.light_react ? { ...L.light_react } : undefined,
      anim: L.anim ? JSON.parse(JSON.stringify(L.anim)) : undefined, video: L.video ? { ...L.video } : undefined,
      media: L.media ? JSON.parse(JSON.stringify(L.media)) : undefined, motion: L.motion ? JSON.parse(JSON.stringify(L.motion)) : undefined,
      _rest: L._rest ? { ...L._rest } : undefined, _applied: L._applied ? { ...L._applied } : undefined, _off: L._off ? { ...L._off } : undefined,
      pix: L.kind === "paint" ? this.paintPix(L) : null })), dismissed: [...this.dismissed],
      tl: { duration: this.tl.duration, fps: this.tl.fps, markers: JSON.parse(JSON.stringify(this.tl.markers || [])) }, pm: { ...this.pm } };
  }
  restore(st) {
    const old = this.layers;
    this.dismissed = [...st.dismissed];
    if (st.pm) { const same = stable(st.pm) === stable(this.pm); this.pm = { ...st.pm }; if (!same) this.loadPm(); }
    if (st.tl) { Object.assign(this.tl, { duration: st.tl.duration, fps: st.tl.fps, markers: JSON.parse(JSON.stringify(st.tl.markers)) });
      this.tlMark = null; this.tl.time = Math.min(this.tl.time, this.tl.duration); this.fillTl?.(); }
    this.layers = st.layers.map(s => { const L = { ...s, mask: { ...s.mask }, adjust: { ...s.adjust }, light: s.light ? JSON.parse(JSON.stringify(s.light)) : undefined,
      shape: s.shape ? JSON.parse(JSON.stringify(s.shape)) : undefined, fx: s.fx ? { ...s.fx } : undefined, light_react: s.light_react ? { ...s.light_react } : undefined,
      anim: s.anim ? JSON.parse(JSON.stringify(s.anim)) : undefined, video: s.video ? { ...s.video } : undefined,
      media: s.media ? JSON.parse(JSON.stringify(s.media)) : undefined, motion: s.motion ? JSON.parse(JSON.stringify(s.motion)) : undefined,
      _rest: s._rest ? { ...s._rest } : undefined, _applied: s._applied ? { ...s._applied } : undefined, _off: s._off ? { ...s._off } : undefined };
      for (const k of ["motion", "_rest", "_applied", "_off"]) if (L[k] === undefined) delete L[k];
      const prev = old?.find(q => q.id === L.id); if (prev?._stg) L._stg = prev._stg; else delete L._stg;   // still checked per selection
      if (L.media === undefined) delete L.media;
      if (L.anim === undefined) delete L.anim;
      if (L.video === undefined) delete L.video;
      if (s.kind === "paint") { L.canvas = mkCanvas(this.PW, this.PH); L.canvas.getContext("2d").putImageData(s.pix, 0, 0); L.dirty = !!s.dirty || !s.source; L._pix = s.pix; L._pixVer = L.ver || 0; }
      if (L.light === undefined) delete L.light;
      delete L.pix; delete L.pendingKey; delete L.dstatus; delete L.derr; delete L._cutTok;
      L.diffusing = [...this.jobs.values()].some(j => j.id === L.id) || undefined;       // a run still going keeps its state
      return L; });
    this.quietVideos(old);
    for (const L of this.layers) if (isVideo(L) && !(L.img instanceof HTMLVideoElement) && !L.verr)   // e.g. a copy undone before it loaded
      this.attachVideo(L).then(() => { if (this.layers.includes(L)) { this.syncVideos(false); this.refresh(); } })
        .catch(e => { L.verr = e.message; if (this.layers.includes(L)) this.refresh(); });
    this.sel = Math.min(this.sel, this.layers.length - 1); this.syncEval(); this.applyTime(this.tl.time); this.refresh();
  }
  snap() { this.undo.push(this.state()); if (this.undo.length > 30) this.undo.shift(); this.redo.length = 0; this.scheduleDraft(); }
  commit(st) {                                       // push a state taken before a drag, once the drag really changed something
    if (!st) return null;
    this.undo.push(st); if (this.undo.length > 30) this.undo.shift(); this.redo.length = 0; this.scheduleDraft(); return null;
  }
  once() { if (!this._once) { this.snap(); this._once = true; setTimeout(() => this._once = false, 600); } }
  doUndo() { if (!this.undo.length) return; this.redo.push(this.state()); this.restore(this.undo.pop()); }
  doRedo() { if (!this.redo.length) return; this.undo.push(this.state()); this.restore(this.redo.pop()); }
  cur() { return this.layers?.[this.sel]; }         // no layers yet while the window loads
  insert(L) { const b = this.layers.findIndex(q => q.kind === "base"); const at = Math.max(0, Math.min(this.sel, b)); this.layers.splice(at, 0, L); this.sel = at; }
  addShape(type) {                                   // solid = a rect over the whole canvas; others in the middle
    this.snap();
    const id = "L" + (this.uid++), full = type === "solid";
    const w = full ? this.W : this.W * 0.3, h = full ? this.H : this.H * 0.3;
    const L = this.defaults({ id, name: `${full ? "solid" : type} ${id}`, kind: "shape", action: "keep", color: full ? "#202020" : ORANGE,
      shape: { type: full ? "rect" : type, points: [], feather: 0 }, x: (this.W - w) / 2, y: (this.H - h) / 2, w, h });
    this.insert(L); this.setTool("move"); this.refresh(); return L;
  }
  finishPoly() {                                     // the clicked points -> a polygon shape layer in their box
    const pts = (this.polyDraft || []).filter((p, i, a) => !i || Math.hypot(p[0] - a[i - 1][0], p[1] - a[i - 1][1]) > 2 / this.S());
    this.polyDraft = null; this.setTool("move");
    if (pts.length < 3) { note("a polygon needs at least 3 points"); this.draw(); return; }
    const xs = pts.map(p => p[0]), ys = pts.map(p => p[1]), x0 = Math.min(...xs), y0 = Math.min(...ys);
    const w = Math.max(1, Math.max(...xs) - x0), h = Math.max(1, Math.max(...ys) - y0);
    this.snap();
    const id = "L" + (this.uid++);
    const L = this.defaults({ id, name: `polygon ${id}`, kind: "shape", action: "keep", color: ORANGE, x: x0, y: y0, w, h,
      shape: { type: "poly", points: pts.map(([x, y]) => [+((x - x0) / w).toFixed(5), +((y - y0) / h).toFixed(5)]), feather: 0 } });
    this.insert(L); this.refresh();
  }
  addLayer(kind) {
    this.snap();
    const id = "L" + (this.uid++);
    if (kind === "light") {                          // on top: it is the look of the whole building
      if (!this.m.scene) { note("connect a scene (kubakub scene render) to the director and run once"); this.undo.pop(); return null; }
      const L = this.defaults({ id, name: `light ${id}`, kind: "light", action: "keep" });
      L.light = this.rigDefaults(null); this.layers.unshift(L); this.sel = 0; this.lsel = 0; this.setTool("move"); this.refresh(); return L;
    }
    const L = this.defaults(kind === "paint" ? { id, name: `paint ${id}`, kind: "paint", source: "", action: "keep" } : { id, name: `colour grade ${id}`, kind: "adjust" });
    if (kind === "paint") { L.canvas = mkCanvas(this.PW, this.PH); L.color = ORANGE; L.size = 60; L.dirty = true; L.ver = 0; }
    this.insert(L); this.refresh(); return L;
  }
  duplicate(offset = true) {                         // offset = false: the copy lands on the original (Alt drag)
    const L = this.cur(); if (!L || L.kind === "base") return null; this.snap();
    const c = { ...L, id: "L" + (this.uid++), name: L.name + " copy", mask: { ...L.mask }, adjust: { ...L.adjust } };
    if (L.shape) c.shape = JSON.parse(JSON.stringify(L.shape));
    if (L.fx) c.fx = { ...L.fx };
    if (L.light_react) c.light_react = { ...L.light_react };
    if (L.motion) c.motion = JSON.parse(JSON.stringify(L.motion));
    if (L._rest) { c._rest = { ...L._rest }; c._applied = { ...L._applied }; }
    if (L._off) c._off = { ...L._off };
    delete c._stg;
    delete c._pix; delete c._thumb; delete c.pendingKey; delete c.diffusing; delete c.dstatus; delete c.derr; delete c._cutTok;
    if (!c.remove_bg) delete c.imgOrig;
    if (L.light) c.light = JSON.parse(JSON.stringify(L.light));
    if (L.anim) c.anim = JSON.parse(JSON.stringify(L.anim));
    if (L.video) c.video = { ...L.video };
    if (L.media) c.media = JSON.parse(JSON.stringify(L.media));
    if (L.vinfo) { c.img = null; this.attachVideo(c).then(() => {
      if (this.closed) { c.img?.pause?.(); return; }
      if (this.layers.includes(c)) { this.syncVideos(this.playing); this.refresh(); } }).catch(e => { c.verr = e.message; note("video copy failed: " + e.message); }); }
    if (L.kind === "paint") { c.canvas = mkCanvas(this.PW, this.PH); c.canvas.getContext("2d").drawImage(L.canvas, 0, 0); c.dirty = true; c.source = ""; c.ver = 0; }
    if (c.kind === "image" && offset) { c.x += 60; c.y += 60; }
    this.layers.splice(this.sel, 0, c); this.refresh();
    return c;
  }
  removeSel() {
    const L = this.cur(); if (!L || L.kind === "base") return; this.snap();
    if (L.source?.startsWith("input:") && !this.layers.some(q => q !== L && q.source === L.source)) this.dismissed.push(L.source);
    this.layers.splice(this.sel, 1); this.sel = Math.min(this.sel, this.layers.length - 1);
    this.dropDangling(); this.quietVideos([L]); this.refresh();
  }
  restoreInputs() {                                  // the node's layer inputs that were deleted in the window
    const back = (this.m.inputs || []).filter(i => this.dismissed.includes(i.key)); if (!back.length) return;
    this.snap(); this.dismissed = this.dismissed.filter(k => !back.some(i => i.key === k));
    for (const i of back) { const L = this.defaults(this.inputLayer(i)); L.img = this.inputImgs[i.key] || null; this.insert(L); }
    this.refresh();
  }
  moveSel(to) { to = Math.max(0, Math.min(this.layers.length - 1, to)); if (to === this.sel) return; this.snap(); const [L] = this.layers.splice(this.sel, 1); this.layers.splice(to, 0, L); this.sel = to; this.refresh(); }
  async importFile(file, at) {
    if (file?.type?.startsWith("audio/") || /\.(mp3|wav|ogg|flac|m4a|aac)$/i.test(file?.name || "")) return this.importAudio(file);
    if (file?.type?.startsWith("video/") || isVideoName(file?.name)) return this.importVideo(file, null, at);
    if (!file || !file.type.startsWith("image/")) return;
    this.busy(true, "importing");
    try {
      const name = await upload(file, (file.name || "pasted.png").replace(/[^\w.\-]+/g, "_"), false);
      const img = await loadImage(inputURL(name));
      if (this.closed) return;
      const w0 = img.naturalWidth, h0 = img.naturalHeight, s = Math.min(1, (this.W * 0.4) / w0, (this.H * 0.5) / h0);
      const [cx, cy] = at || [this.W / 2, this.H / 2];
      this.snap();
      const L = this.defaults({ id: "L" + (this.uid++), name: (file.name || "pasted image").replace(/\.[^.]+$/, "").toLowerCase(), kind: "image",
        source: "file:" + name, x: cx - w0 * s / 2, y: cy - h0 * s / 2, w: w0 * s, h: h0 * s });
      L.img = img; this.insert(L); this.setTool("move"); this.refresh();
    } catch (e) { console.error("[kubakub director]", e); note("import failed: " + e.message); }
    this.busy(false);
  }

  async importVideo(file, path, at) {                // an uploaded file (input/kuba_director) or one linked by its path
    this.busy(true, "importing the video · making a preview copy");
    try {
      let source;
      if (path) source = "path:" + path.trim().replace(/^"|"$/g, "");
      else {
        if (file.size > 95 * 1024 * 1024) throw new Error(`${(file.size / 2 ** 20).toFixed(0)} MB is over ComfyUI's upload limit: use + layer → video from a path`);
        source = "file:" + await upload(file, (file.name || "video.mp4").replace(/[^\w.\-]+/g, "_"), false);
      }
      const base = source.split(/[\\/:]/).pop().replace(/\.[^.]+$/, "").toLowerCase();
      const L = this.defaults({ id: "L" + (this.uid++), name: base || "video", kind: "image", source, action: "keep",
        media: { segments: [{ start: +this.tl.time.toFixed(3), in: 0, out: -1, speed: 1, length: -1, fill: "hold" }], volume: 1, blend_frames: false } });
      await this.attachVideo(L);
      if (this.closed) { L.img?.pause?.(); return; }
      const { w: w0, h: h0, duration } = L.vinfo;
      const same = Math.abs(w0 / h0 - this.W / this.H) < 0.02;         // made for this building: fill the canvas
      const s = same ? this.W / w0 : Math.min((this.W * 0.8) / w0, (this.H * 0.8) / h0);
      const [cx, cy] = same || !at ? [this.W / 2, this.H / 2] : at;
      Object.assign(L, { x: cx - w0 * s / 2, y: cy - h0 * s / 2, w: w0 * s, h: h0 * s });
      this.snap();
      const end = L.media.segments[0].start + duration;
      if (end > this.tl.duration + 0.05 && !this.tl.audio) { this.tl.duration = Math.min(3600, Math.ceil(end * 100) / 100); this.fillTl(); note(`timeline length set to ${this.tl.duration} s for the video`); }
      this.insert(L); this.setTool("move"); this.applyTime(this.tl.time); this.refresh();
    } catch (e) { console.error("[kubakub director] video", e); note("video import failed: " + e.message); }
    this.busy(false);
  }
  async linkVideo() {
    const p = await this.ask("video from a path", "The full path of a video on this computer (ProRes, HAP, DNxHD, H.264 …). It is read where it is, never copied; the window plays a small preview copy.",
      [["cancel", "cancel"], ["ok", "add", "accent"]], this.lastVideoPath ?? "");
    if (!p) return; this.lastVideoPath = p; await this.importVideo(null, p);
  }

  // ---- background removal (preview; the node makes the full-size cutout)
  imageRef(L) {
    if (L.source.startsWith("input:")) return (this.m.inputs || []).find(i => i.key === L.source)?.image;
    if (L.source.startsWith("file:")) { const n = L.source.slice(5), i = n.lastIndexOf("/"); return { filename: n.slice(i + 1), subfolder: i >= 0 ? n.slice(0, i) : "", type: "input" }; }
    return null;
  }
  async cutout(L, quiet) {
    const ref = this.imageRef(L); if (!ref || !L.img) return;
    const tok = L._cutTok = {};                       // a newer toggle / undo makes this answer stale
    const stale = () => L._cutTok !== tok || !this.layers?.includes(L) || this.closed;
    if (!quiet) this.busy(true, "removing background");
    try {
      const r = await api.fetchApi("/kubakub/director/remove_bg", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ image: ref }) });
      const j = await r.json(); if (!r.ok || j.error) throw new Error(j.error || r.status);
      const m = await loadImage(viewURL(j.mask)), src = L.imgOrig || L.img;
      if (stale()) { if (!quiet) this.busy(false); return; }
      const c = mkCanvas(iw(src), ih(src)), g = c.getContext("2d");
      const mc = mkCanvas(iw(src), ih(src)), mg = mc.getContext("2d", { willReadFrequently: true });
      mg.drawImage(m, 0, 0, mc.width, mc.height);                    // grey mask -> alpha
      const d = mg.getImageData(0, 0, mc.width, mc.height); for (let p = 0; p < d.data.length; p += 4) { d.data[p + 3] = d.data[p]; }
      mg.putImageData(d, 0, 0);
      g.drawImage(src, 0, 0); g.globalCompositeOperation = "destination-in"; g.drawImage(mc, 0, 0);
      L.imgOrig = src; L.img = c; L.remove_bg = true;
    } catch (e) { console.error("[kubakub director]", e); if (!stale()) { note("background removal failed: " + e.message); L.remove_bg = false; } }
    if (!quiet) { this.busy(false); this.refresh(); } else if (this.layers) { this.renderLayers(); this.draw(); }
  }
  toggleSnap() { this.snapOn = !this.snapOn; this.$("snap").setAttribute("aria-pressed", String(this.snapOn)); }

  // ---- drafts (autosave) and saved compositions: ComfyUI user data (user/default/kubakub_director/)
  draftFile() {
    const p = this.node.properties = this.node.properties || {};
    // properties travel with copy / paste and clones: a draft id that belongs to another node starts fresh
    if (!p.kkd_draft_id || (p.kkd_draft_node != null && p.kkd_draft_node !== this.node.id)) {
      p.kkd_draft_id = (crypto.randomUUID?.() || String(Date.now()) + Math.random().toString(16).slice(2)).slice(0, 18);
      p.kkd_draft = null;
    }
    p.kkd_draft_node = this.node.id;
    return `kubakub_director/drafts/${p.kkd_draft_id}.json`;
  }
  scheduleDraft() {
    if (!this.layers || this.closed) return;
    clearTimeout(this.draftTimer); this.draftPending = true;
    this.draftTimer = setTimeout(() => { this.draftPending = false; this.saveDraft().catch(e => console.warn("[kubakub director] draft", e)); }, 2500);
  }
  async uploadPaint() {                              // paint layers get their content-named file before any save
    for (const L of this.layers) {
      if (L.kind !== "paint" || (!L.dirty && L.source)) continue;
      const ver = L.ver || 0, blob = await new Promise(r => L.canvas.toBlob(r, "image/png"));
      L.source = "file:" + await upload(blob, await contentName(blob, "paint"), true); if ((L.ver || 0) === ver) L.dirty = false;
    }
  }
  async saveDraft() {
    await this.uploadPaint();
    const doc = this.serialize(), payload = { doc, time: Date.now(), applied: this.widget()?.value || "" };
    this.node.properties.kkd_draft = JSON.stringify(payload);          // also travels with the workflow
    await api.storeUserData(this.draftFile(), payload, { overwrite: true, stringify: true, throwOnError: false });
    this.$("stamp").textContent = "draft saved " + new Date(payload.time).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  }
  async readDraft() {
    let best = null;
    try { const r = await api.getUserData(this.draftFile()); if (r.ok) best = await r.json(); } catch { /* none */ }
    try { const p = JSON.parse(this.node.properties?.kkd_draft || "null"); if (p && (!best || p.time > best.time)) best = p; } catch { /* none */ }
    return best;
  }
  async dropDraft() {
    if (this.node.properties) this.node.properties.kkd_draft = null;
    try { await api.deleteUserData(this.draftFile()); } catch { /* none */ }
  }
  async pickStart() {                                // null = the applied document from the node
    const d = await this.readDraft();
    const applied = this.widget()?.value || "";
    if (!d || !d.doc || JSON.stringify(d.doc) === applied) return null;
    const when = new Date(d.time).toLocaleString([], { day: "2-digit", month: "2-digit", hour: "2-digit", minute: "2-digit" });
    const n = (d.doc.layers || []).length;
    const choice = await this.ask("continue where you left off?",
      `There is a draft from ${when} with ${n} layers that was not applied yet.`,
      [["applied", "use the applied composition"], ["draft", "continue the draft", "accent"]]);
    if (choice === "draft") return d.doc;
    return null;
  }
  ask(title, text, buttons, withInput) {
    const hasInput = typeof withInput === "string";
    return new Promise(resolve => {
      const m = document.createElement("div"); m.className = "kkd-modal";
      m.innerHTML = `<div><h3>${esc(title)}</h3><p>${esc(text)}</p>${hasInput ? `<input type="text" value="${esc(withInput)}">` : ""}
        <div class="row">${buttons.map(([v, t, cls]) => `<button class="kkd-pill ${cls || ""}" data-v="${esc(v)}">${esc(t)}</button>`).join("")}</div></div>`;
      document.body.appendChild(m);
      const inp = m.querySelector("input"); m.tabIndex = -1;
      if (inp) { inp.focus(); inp.select(); } else (m.querySelector(".kkd-pill.accent") || m).focus();   // keys go to the dialog
      const done = v => { m.remove(); this.root?.focus(); resolve(hasInput ? (v ? inp.value.trim() : null) : v); };
      m.addEventListener("click", e => { const b = e.target.closest("[data-v]"); if (b) done(b.dataset.v === "cancel" ? null : b.dataset.v); else if (e.target === m) done(null); });
      m.addEventListener("keydown", e => { e.stopPropagation(); if (e.key === "Enter") done(buttons[buttons.length - 1][0]); if (e.key === "Escape") done(null); });
    });
  }
  async saveAs() {
    const name = await this.ask("save composition", "A name for this composition (saved in ComfyUI's user folder):",
      [["cancel", "cancel"], ["save", "save", "accent"]], this.savedName || this.node.title || "composition");
    if (!name) return;
    this.busy(true, "saving");
    try {
      await this.uploadPaint();
      const file = `kubakub_director/compositions/${slug(name)}.json`;
      await api.storeUserData(file, { name, time: Date.now(), canvas: [this.W, this.H], doc: this.serialize() }, { overwrite: true, stringify: true });
      this.savedName = name; note(`saved as ${name}`);
    } catch (e) { note("saving failed: " + e.message); }
    this.busy(false);
  }
  async openSaved() {
    let files = [];
    try { files = (await api.listUserDataFullInfo("kubakub_director/compositions")) || []; } catch { files = []; }
    files = files.filter(f => String(f.path || f).endsWith(".json")).sort((a, b) => (b.modified || 0) - (a.modified || 0));
    if (!files.length) { note("no saved compositions yet (file → save as…)"); return; }
    const m = document.createElement("div"); m.className = "kkd-modal";
    m.innerHTML = `<div><h3>open composition</h3><ul>${files.map((f, i) => `<li data-i="${i}"><span>${esc(String(f.path || f).replace(/\.json$/, ""))}</span>
      <small>${f.modified ? new Date(f.modified * (f.modified < 1e12 ? 1000 : 1)).toLocaleString([], { day: "2-digit", month: "2-digit", hour: "2-digit", minute: "2-digit" }) : ""}</small></li>`).join("")}</ul>
      <div class="row"><button class="kkd-pill" data-v="cancel">cancel</button></div></div>`;
    document.body.appendChild(m); m.tabIndex = -1; m.focus();
    m.addEventListener("keydown", e => { e.stopPropagation(); if (e.key === "Escape") { m.remove(); this.root.focus(); } });
    m.addEventListener("click", async e => {
      if (e.target === m || e.target.closest("[data-v=cancel]")) { m.remove(); return; }
      const li = e.target.closest("li"); if (!li) return;
      m.remove();
      const f = files[+li.dataset.i];
      try {
        const r = await api.getUserData(`kubakub_director/compositions/${String(f.path || f)}`);
        const d = await r.json(); await this.useDocument(d.doc); this.savedName = d.name; note(`opened ${d.name}`);
      } catch (err) { note("could not open it: " + err.message); }
    });
  }
  exportFile() {
    const blob = new Blob([JSON.stringify({ name: this.savedName || "composition", time: Date.now(), canvas: [this.W, this.H], doc: this.serialize() }, null, 1)], { type: "application/json" });
    const a = Object.assign(document.createElement("a"), { href: URL.createObjectURL(blob), download: `${slug(this.savedName || "composition")}.kubakub.json` });
    document.body.appendChild(a); a.click(); a.remove(); setTimeout(() => URL.revokeObjectURL(a.href), 2000);
  }
  async useDocument(doc) {
    if (!doc || !Array.isArray(doc.layers)) throw new Error("no layers in it");
    doc = { ...doc, layers: doc.layers.filter(l => l && typeof l === "object" && !Array.isArray(l)) };
    this.snap(); this.busy(true, "loading");
    try { await this.loadDocument(doc); }
    catch (e) { const st = this.undo.pop(); if (st) this.restore(st); throw e; }
    finally { this.busy(false); }
    this.refresh(); this.fillTl(); this.loadAudio();
  }

  // ---- apply / close
  async apply() {
    if (this.applying) return;
    this.applying = true; this.$("apply").disabled = true;
    this.busy(true, "saving");
    try {
      await this.saveToNode();
      // a second Ctrl Enter right after would reach ComfyUI's own shortcut once the window is gone: swallow it briefly
      const swallow = e => { if ((e.ctrlKey || e.metaKey) && e.key === "Enter") { e.stopPropagation(); e.preventDefault(); } };
      window.addEventListener("keydown", swallow, true); setTimeout(() => window.removeEventListener("keydown", swallow, true), 800);
      this.close();
      app.queuePrompt(0, 1);
    } catch (e) { console.error("[kubakub director]", e); note("saving failed: " + e.message); this.busy(false); this.applying = false; this.$("apply").disabled = false; }
  }
  async saveToNode() {                               // paint layers uploaded, the composition into the node's document widget
    for (const L of this.layers) {
      if (L.kind !== "paint" || (!L.dirty && L.source)) continue;
      const ver = L.ver || 0, blob = await new Promise(r => L.canvas.toBlob(r, "image/png"));
      const name = await upload(blob, await contentName(blob, "paint"), true);
      L.source = "file:" + name; if ((L.ver || 0) === ver) L.dirty = false;
    }
    const w = this.widget();
    if (w) { w.value = JSON.stringify(this.serialize()); w.callback?.(w.value); }
    clearTimeout(this.draftTimer); this.draftPending = false; await this.dropDraft();
    this.node.setDirtyCanvas?.(true, true);
  }
  renderExportPop() {
    const wv = n => this.node.widgets?.find(w => w.name === n)?.value;
    const x = this.exportSet ||= { format: wv("export") && wv("export") !== "none" ? wv("export") : "prores422hq",
      scale: +wv("export_scale") || 1, alpha: !!wv("export_alpha"), name: wv("export_name") || "director" };
    const withAlpha = ["png8", "png16", "prores4444"].includes(x.format);
    const W = Math.round(this.W * x.scale / 2) * 2, H = Math.round(this.H * x.scale / 2) * 2;
    const dur = this.tl?.duration || 0, fps = this.tl?.fps || 25;
    this.exportpop.innerHTML = `<h3>export</h3>
      <div class="kkd-field"><span>format</span><select data-ex="format">${EXPORT_FORMATS.map(([k, l]) => `<option value="${k}" ${k === x.format ? "selected" : ""}>${l}</option>`).join("")}</select><span></span></div>
      <div class="kkd-field"><span>size</span><select data-ex="scale">${[1, 0.75, 0.5, 0.25].map(v => `<option value="${v}" ${v === x.scale ? "selected" : ""}>${v === 1 ? "delivery (matrix)" : Math.round(v * 100) + " %"}</option>`).join("")}</select><span style="color:var(--muted);font-size:12px">${x.format === "preview" ? "half of it" : W + " × " + H}</span></div>
      <div class="kkd-field"><span>alpha</span><label style="font-size:13px"><input type="checkbox" data-ex="alpha" ${x.alpha ? "checked" : ""} ${withAlpha ? "" : "disabled"}> layers only, transparent outside ${withAlpha ? "" : "(png / prores 4444)"}</label><span></span></div>
      <div class="kkd-field"><span>name</span><input type="text" data-ex="name" value="${esc(x.name)}"><span></span></div>
      <div class="kkd-lstatus">${dur.toFixed(2)} s at ${fps} fps = ${Math.max(1, Math.round(dur * fps) + 1)} frames, Rec.709, with the timeline sound. Only this node runs (not the H3 / LTX nodes after it); the composition is saved into the node first. Into output/kubakub_director/&lt;name&gt;_&lt;time&gt;/.</div>
      <div style="display:flex;gap:8px;align-items:center;margin-top:10px"><button class="kkd-pill accent" data-ex-go ${this.exportJob ? "disabled" : ""}>render and export</button>
      <span data-id="ex-status" class="kkd-lstatus" style="margin:0">${esc(this.exportMsg || "")}</span></div>`;
  }
  exportStatus(msg, err) {
    this.exportMsg = msg; note(msg);
    const el = this.exportpop?.querySelector("[data-id=ex-status]"); if (el) { el.textContent = msg; el.style.color = err ? "#c0392b" : ""; }
    const go = this.exportpop?.querySelector("[data-ex-go]"); if (go) go.disabled = !!this.exportJob;
  }
  async exportNow() {
    if (this.exportJob) return;
    const x = this.exportSet, id = String(this.node.id);
    try {
      this.exportStatus("saving and queueing…", false);
      await this.saveToNode();
      const p = await app.graphToPrompt();
      const n = p.output?.[id];
      if (!n) throw new Error("the director node is bypassed or muted");
      Object.assign(n.inputs, { export: x.format, export_scale: x.scale, export_alpha: x.alpha && ["png8", "png16", "prores4444"].includes(x.format), export_name: x.name || "director" });
      const r = await api.fetchApi("/prompt", { method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt: p.output, client_id: api.clientId ?? api.initialClientId ?? undefined,
          extra_data: { extra_pnginfo: { workflow: p.workflow } }, partial_execution_targets: [id] }) });
      const j = await r.json().catch(() => ({}));
      if (!r.ok || !j.prompt_id) throw new Error(j.error?.message || j.node_errors && Object.values(j.node_errors)[0]?.errors?.[0]?.details || "queue refused (" + r.status + ")");
      this.exportJob = j.prompt_id;
      this.exportStatus("exporting… (the window can stay open)", false);
    } catch (e) { console.error("[kubakub director] export", e); this.exportJob = null; this.exportStatus("export failed: " + e.message, true); }
  }
  close() {
    clearTimeout(this.draftTimer);                   // save a pending draft now: a reopened window reads it at once
    if (this.draftPending) { this.draftPending = false; this.saveDraft().catch(e => console.warn("[kubakub director] draft", e)); }
    this.closed = true;                              // async work finishing later must not snapshot, save or insert
    if (this._playRaf) { cancelAnimationFrame(this._playRaf); this._playRaf = 0; }
    window.removeEventListener("blur", this.onBlur);
    for (const t of this.lightTimers.values()) clearTimeout(t);
    if (this._raf) { cancelAnimationFrame(this._raf); this._raf = 0; }
    this.ro?.disconnect();
    window.removeEventListener("keydown", this.onKey, true); window.removeEventListener("keyup", this.onKeyUp, true);
    for (const t of ["dragenter", "dragleave", "dragover", "drop", "paste"]) document.removeEventListener(t, this[`on_${t}`], true);
    document.removeEventListener("pointerover", this.onTip, true);
    for (const el of [this.root, this.menu, this.filemenu, this.docfile, this.keypop, this.outpop, this.diffpop, this.exportpop, this.regpick, this.tipEl, this.dropzone, this.file, this.pmFile, this.fxSvg]) el?.remove();
    for (const [t, f] of this.apiEvents || []) api.removeEventListener(t, f);
    this.playing = false; this.stopAudio(); this.actx?.close?.().catch(() => {}); this.afile?.remove();
    for (const v of this.videos || []) { v.pause(); v.removeAttribute("src"); v.load(); }     // incl. deleted / undone layers
    this.videos?.clear(); this.undo = []; this.redo = [];
    if ([...this.jobs.values()].length) note("diffusion still running: the result lands in input/kuba_director; reopen and import it");
    document.querySelectorAll(".kkd-modal").forEach(m => m.remove());
    Director.current = null;
  }

  // ---- events
  bind() {
    const $ = id => this.$(id);
    const popAt = (pop, btn) => { const r = btn.getBoundingClientRect(); pop.hidden = !pop.hidden; pop.style.top = (r.bottom + 6) + "px";
      pop.style.left = Math.max(16, Math.min(r.right - pop.offsetWidth, innerWidth - pop.offsetWidth - 16)) + "px"; this.fillRanges(); };
    this.root.addEventListener("click", e => {
      const v = e.target.closest("[data-view]"); if (v) return this.setView(v.dataset.view);
      if (e.target.closest("[data-id=snap]")) return this.toggleSnap();
      const t = e.target.closest(".kkd-tool"); if (t && !t.disabled) return this.setTool(t.dataset.id.slice(2));
    });
    $("undo").onclick = () => this.doUndo(); $("redo").onclick = () => this.doRedo();
    const LAY = "kubakub-director-layout";
    let lay = {}; try { lay = JSON.parse(localStorage.getItem(LAY) || "{}") || {}; } catch (e) {}
    const applyLay = () => { for (const k of ["sw", "lh", "th"]) { if (lay[k]) this.root.style.setProperty("--kkd-" + k, lay[k] + "px"); else this.root.style.removeProperty("--kkd-" + k); } };
    this.saveLayout = patch => { Object.assign(lay, patch || {}); try { localStorage.setItem(LAY, JSON.stringify(lay)); } catch (e) {} };
    this.pmOpen = !!lay.pm; applyLay();
    const SPLIT = {
      sw: { start: () => this.root.querySelector(".kkd-side").getBoundingClientRect().width, at: (a, dx) => Math.max(260, Math.min(Math.min(720, innerWidth * 0.45), a - dx)) },
      lh: { start: () => $("layers").getBoundingClientRect().height, at: (a, dx, dy) => Math.max(48, Math.min(innerHeight * 0.7, a + dy)) },
      th: { start: () => parseFloat(getComputedStyle(this.root.querySelector(".kkd-tlwrap")).maxHeight) || 140, at: (a, dx, dy) => Math.max(40, Math.min(innerHeight * 0.6, a - dy)) } };
    for (const el of this.root.querySelectorAll("[data-split]")) {
      const S = SPLIT[el.dataset.split], k = el.dataset.split;
      el.addEventListener("pointerdown", e => { if (e.button) return; e.preventDefault(); e.stopPropagation(); el.setPointerCapture(e.pointerId); el.classList.add("drag");
        const x0 = e.clientX, y0 = e.clientY, a = S.start();
        const mv = ev => { lay[k] = Math.round(S.at(a, ev.clientX - x0, ev.clientY - y0)); applyLay(); if (k === "th") this.drawTimeline(); };
        const up = () => { el.removeEventListener("pointermove", mv); el.removeEventListener("pointerup", up); el.removeEventListener("pointercancel", up); el.classList.remove("drag"); this.saveLayout(); };
        el.addEventListener("pointermove", mv); el.addEventListener("pointerup", up); el.addEventListener("pointercancel", up); });
      el.addEventListener("dblclick", () => { delete lay[k]; applyLay(); this.saveLayout(); if (k === "th") this.drawTimeline(); });
    }
    this.root.addEventListener("input", e => { if (e.target.type === "range") this.fillRanges(); }, true);
    $("apply").onclick = () => this.apply(); $("close").onclick = () => this.close();
    const flowW = () => this.node.widgets?.find(w => w.name === "flow");
    const showFlow = () => { const auto = flowW()?.value === "always", b = $("flow"); b.textContent = auto ? "auto" : "hold";
      b.setAttribute("aria-pressed", String(auto)); b.hidden = !flowW(); };
    $("flow").onclick = () => { const w = flowW(); if (!w) return; w.value = w.value === "always" ? "hold until apply" : "always";
      w.callback?.(w.value); this.node.setDirtyCanvas?.(true, true); showFlow(); };
    showFlow();
    $("add").onclick = e => { e.stopPropagation(); popAt(this.menu, $("add")); };
    $("outputs").onclick = e => { e.stopPropagation(); this.keypop.hidden = true; popAt(this.outpop, $("outputs")); };
    $("keysbtn").onclick = e => { e.stopPropagation(); this.outpop.hidden = true; popAt(this.keypop, $("keysbtn")); };
    this.menu.onclick = e => { const b = e.target.closest("button"); if (!b) return; this.menu.hidden = true;
      const k = b.dataset.new;
      if (k === "solid" || k === "rect" || k === "ellipse") { this.addShape(k); return; }
      if (k === "poly") { this.polyDraft = []; this.setTool("poly"); note("click the corners · Enter or double click to finish · Esc to cancel"); return; }
      k === "dup" ? this.duplicate() : k === "import" ? this.file.click() : k === "linkvideo" ? this.linkVideo() : k === "inputs" ? this.restoreInputs() : this.addLayer(k); };
    this.file.onchange = e => { [...e.target.files].forEach(f => this.importFile(f)); e.target.value = ""; };
    $("file").onclick = e => { e.stopPropagation(); popAt(this.filemenu, $("file")); };
    this.filemenu.onclick = e => { const b = e.target.closest("button"); if (!b) return; this.filemenu.hidden = true;
      ({ save: () => this.saveAs(), open: () => this.openSaved(), export: () => this.exportFile(), import: () => this.docfile.click() })[b.dataset.f](); };
    this.docfile.onchange = async e => { const f = e.target.files[0]; e.target.value = ""; if (!f) return;
      try { const d = JSON.parse(await f.text()); await this.useDocument(d.doc || d); note("composition imported"); } catch (err) { note("not a composition file: " + err.message); } };
    this.root.addEventListener("pointerdown", e => { for (const p of [this.menu, this.filemenu, this.keypop, this.outpop, this.diffpop, this.exportpop, this.clipPop].filter(Boolean)) if (!p.hidden && !p.contains(e.target) && !e.target.closest("[data-id=add],[data-id=outputs],[data-id=keysbtn],[data-id=file],[data-id=diffbtn],[data-id=exportbtn],[data-id=tl-h3],[data-id=tlc]")) p.hidden = true; });
    // delivery export: only this node runs, with the export settings in the queued prompt (the node's widgets stay as they are)
    $("exportbtn").onclick = e => { e.stopPropagation(); this.keypop.hidden = this.outpop.hidden = this.diffpop.hidden = true; this.renderExportPop(); popAt(this.exportpop, $("exportbtn")); };
    this.exportpop.addEventListener("change", e => { const t = e.target, x = this.exportSet;
      if (t.dataset.ex === "alpha") x.alpha = t.checked; else if (t.dataset.ex === "scale") x.scale = +t.value; else if (t.dataset.ex) x[t.dataset.ex] = t.value;
      if (t.dataset.ex === "format" || t.dataset.ex === "scale") this.renderExportPop(); });
    this.exportpop.addEventListener("click", e => { if (e.target.closest("[data-ex-go]")) this.exportNow(); });
    // diffusion settings and runs
    $("diffbtn").onclick = e => { e.stopPropagation(); this.keypop.hidden = this.outpop.hidden = true; this.renderDiffPop(); popAt(this.diffpop, $("diffbtn")); };
    this.diffpop.addEventListener("input", e => {
      const t = e.target, D = this.diff;
      if (t.dataset.df && t.type === "range") { D[t.dataset.df] = +t.value; t.nextElementSibling.textContent = t.dataset.df === "megapixels" ? (+t.value).toFixed(2) + " MP" : t.dataset.df === "context" ? Math.round(t.value * 100) + "%" : t.dataset.df === "band" ? t.value + " px" : t.value; }
      if (t.dataset.dls !== undefined) { D.loras[+t.dataset.dls].strength = +t.value; this.diffpop.querySelector(`[data-dlv="${t.dataset.dls}"]`).textContent = (+t.value).toFixed(2); }
      this.scheduleDraft();
    });
    this.diffpop.addEventListener("change", e => {
      const t = e.target, D = this.diff;
      if (t.dataset.df === "preset") { D.preset = t.value; D.steps = (this.m.diffusion?.presets || []).find(p => p.name === t.value)?.steps || D.steps; this.renderDiffPop(); }
      else if (t.dataset.df === "seed") { const v = parseInt(t.value, 10); D.seed = Number.isFinite(v) ? Math.max(-1, v) : -1; t.value = D.seed; }
      else if (t.dataset.dl !== undefined) D.loras[+t.dataset.dl].name = t.value;
      this.scheduleDraft();
    });
    this.diffpop.addEventListener("click", e => { if (e.target.closest("[data-dref]")) { this.diff.reference = !this.diff.reference; this.renderDiffPop(); this.scheduleDraft(); } });
    $("d-go").onclick = () => this.diffuse();
    const V = () => { const q = L(); q.video = q.video || { on: false, prompt: "", t_start: 0, t_end: -1, motion: 1 }; return q.video; };
    $("v-on").onclick = () => { this.snap(); V().on = !V().on; this.renderInspector(); this.renderLayers(); this.drawTimeline(); };
    $("v-prompt").oninput = e => { this.once(); V().prompt = e.target.value; };
    $("v-start").onchange = e => { const v = parseFloat(e.target.value); this.snap(); V().t_start = Number.isFinite(v) ? Math.max(0, v) : 0; this.renderInspector(); this.drawTimeline(); };
    $("v-end").onchange = e => { const v = parseFloat(e.target.value); this.snap(); V().t_end = Number.isFinite(v) && v >= 0 ? v : -1; this.renderInspector(); this.drawTimeline(); };
    $("v-start-here").onclick = () => { this.snap(); V().t_start = this.tl.time; this.renderInspector(); this.drawTimeline(); };
    $("v-end-here").onclick = () => { this.snap(); V().t_end = this.tl.time; this.renderInspector(); this.drawTimeline(); };
    $("v-motion").oninput = e => { this.once(); V().motion = +e.target.value; $("v-motion-v").textContent = (+e.target.value).toFixed(2); };
    const job = e => this.jobs.get(e.detail?.prompt_id);
    this.apiEvents = [
      ["executing", e => { const j = job(e), q = j && this.jobLayer(j); if (q && q.dstatus === "queued") { q.dstatus = "loading the model…"; this.updateDiffStatus(q); } }],
      ["progress", e => { const j = job(e), q = j && this.jobLayer(j); if (q) { q.dstatus = `diffusing ${e.detail.value} / ${e.detail.max}`; this.updateDiffStatus(q); } }],
      ["executed", e => { const j = job(e), out = e.detail?.output?.kuba_diffuse?.[0];
        if (j && out && out.request === j.request) { this.jobs.delete(e.detail.prompt_id); this.diffuseDone(j, out); } }],
      ["execution_error", e => { const j = job(e); if (j) { this.jobs.delete(e.detail.prompt_id); const q = this.jobLayer(j);
        if (q) { q.diffusing = false; q.derr = true; q.dstatus = "diffuse failed: " + (e.detail.exception_message || "error"); this.updateDiffStatus(q); } } }],
      ["execution_interrupted", e => { const j = job(e); if (j) { this.jobs.delete(e.detail.prompt_id); const q = this.jobLayer(j);
        if (q) { q.diffusing = false; q.dstatus = "interrupted"; this.updateDiffStatus(q); } } }],
    ];
    const exp = e => this.exportJob && e.detail?.prompt_id === this.exportJob;
    this.apiEvents.push(
      ["execution_success", e => { if (exp(e)) { this.exportJob = null; this.exportStatus("written: output/kubakub_director/" + this.exportSet.name + "_…", false); } }],
      ["execution_error", e => { if (exp(e)) { this.exportJob = null; this.exportStatus("export failed: " + (e.detail.exception_message || "error"), true); } }],
      ["execution_interrupted", e => { if (exp(e)) { this.exportJob = null; this.exportStatus("export interrupted", true); } }]);
    for (const [t, f] of this.apiEvents) api.addEventListener(t, f);
    // inspector
    const L = () => this.cur();
    $("i-name").oninput = e => { this.once(); L().name = e.target.value; this.renderLayers(); $("insp-title").textContent = L().name; };
    $("i-blend").onchange = e => { this.snap(); L().blend = e.target.value; this.refresh(); };
    $("i-op").oninput = e => { this.once(); L().opacity = +e.target.value; $("i-op-v").textContent = Math.round(L().opacity * 100) + "%"; this.draw(); };
    const setReg = (input, v) => { v = v.trim(); input.value = v; const key = input.dataset.id === "i-holes" ? "holes" : "clip";
      const q = this.regLayer && this.layers.includes(this.regLayer) ? this.regLayer : L();
      if ((q[key] || "") !== v) { this.snap(); q[key] = v; this.refresh(); } };
    this.commitReg = () => {                         // the selection changes while typing: commit to the old layer, leave the field
      const el = document.activeElement; if (el !== $("i-clip") && el !== $("i-holes")) return;
      setReg(el, el.value); this.regLayer = null; this.regpick.hidden = true; el.blur();
    };
    for (const input of [$("i-holes"), $("i-clip")]) {
      input.onfocus = () => { this.regLayer = L(); input.select(); this.openRegPick(input, ""); };
      input.onclick = () => { if (this.regpick.hidden) this.openRegPick(input, ""); };
      input.oninput = () => this.openRegPick(input, input.value.trim());
      input.onchange = () => setReg(input, input.value);
      input.onblur = () => setTimeout(() => { if (document.activeElement !== input) { this.regpick.hidden = true; this.regLayer = null; } }, 150);
    }
    this.regKey = (input, e) => {                     // called from the captured key handler (fields never see keydown)
      if (e.key === "ArrowDown" || e.key === "ArrowUp") {
        e.preventDefault(); if (this.regpick.hidden) this.openRegPick(input, "");
        const bs = [...this.regpick.querySelectorAll("button[data-v]")];
        this.regHi = Math.max(0, Math.min(bs.length - 1, (this.regHi ?? -1) + (e.key === "ArrowDown" ? 1 : -1)));
        bs.forEach((b, i) => b.classList.toggle("hi", i === this.regHi)); bs[this.regHi]?.scrollIntoView({ block: "nearest" });
        return true;
      }
      if (e.key === "Enter") {
        e.preventDefault(); const b = this.regpick.hidden ? null : this.regpick.querySelectorAll("button[data-v]")[this.regHi];
        setReg(input, b ? b.dataset.v : input.value); this.regpick.hidden = true; return true;
      }
      if (e.key === "Escape" && !this.regpick.hidden) {
        e.preventDefault(); this.regpick.hidden = true; input.value = L()[input.dataset.id === "i-holes" ? "holes" : "clip"] || ""; return true;
      }
      return false;
    };
    this.regpick.addEventListener("pointerdown", e => e.preventDefault());   // keep focus in the field
    this.regpick.onclick = e => { const b = e.target.closest("button[data-v]"); if (!b) return;
      const input = $(this.regpick.dataset.for); setReg(input, b.dataset.v); this.regpick.hidden = true; input.blur(); };
    $("i-mask").onchange = e => { this.snap(); L().mask.by = e.target.value; this.refresh(); };
    $("i-mmode").onclick = e => { const b = e.target.closest("button"); if (!b) return; this.snap();
      if (b.dataset.id === "i-minv") L().mask.invert = !L().mask.invert; else L().mask.mode = b.dataset.m; this.refresh(); };
    $("i-action").onclick = e => { const b = e.target.closest("button"); if (!b) return; this.snap(); L().action = b.dataset.a; this.refresh(); };
    $("i-bg").onclick = async () => { const q = L(); if (q.kind !== "image") return; this.snap();
      if (q.remove_bg) { q.remove_bg = false; q._cutTok = null; if (q.imgOrig) q.img = q.imgOrig; this.refresh(); } else await this.cutout(q, false); };
    $("i-prompt").oninput = e => { this.once(); L().prompt = e.target.value; };
    $("i-den").oninput = e => { this.once(); L().denoise = +e.target.value; $("i-den-v").textContent = L().denoise.toFixed(2); };
    $("i-col").oninput = e => { L().color = e.target.value; };
    $("g-react").addEventListener("change", e => { if (e.target.dataset.id !== "r-by") return; this.snap(); const q = L();
      if (e.target.value === "off") delete q.light_react; else q.light_react = { by: e.target.value, amount: +(q.light_react?.amount ?? 1) };
      this.renderInspector(); this.renderLayers(); this.draw(); });
    $("g-react").addEventListener("input", e => { if (e.target.dataset.id !== "r-amt") return; this.once(); L().light_react.amount = +e.target.value;
      $("r-amt-v").textContent = Math.round(e.target.value * 100) + "%"; this.draw(); });
    $("g-fx").addEventListener("input", e => { const k = e.target.dataset.fx; if (!k) return; this.once(); const q = L();
      q.fx = { ...fxOf(q), [k]: +e.target.value }; const u = FX_UI.find(x => x[0] === k)?.[5] || "";
      const v = $("g-fx").querySelector(`[data-fxv="${k}"]`); if (v) v.textContent = +(+e.target.value).toFixed(2) + u; this.draw(); });
    const MV = e => { const i = e.target.dataset.mi; return i === undefined ? null : L().motion?.[+i]; };
    const motionChanged = () => { this.applyTime(this.tl.time); this.draw(); this.drawTimeline(); this.scheduleDraft(); };
    $("g-motion").addEventListener("input", e => { const b = MV(e), k = e.target.dataset.mk; if (!b || !e.target.hasAttribute("data-num")) return;
      this.once(); const v = parseFloat(String(e.target.value).replace(",", "."));
      if (k === "t_end" && !Number.isFinite(v)) b.t_end = -1; else if (Number.isFinite(v)) b[k] = k === "seed" ? Math.round(v) : v;
      motionChanged(); });
    $("g-motion").addEventListener("change", e => { const q = L();
      if (e.target.dataset.id === "m-add") { const pr = M_PRESETS[+e.target.value]; if (!pr) return; this.snap();
        const b = { id: this.newId("m", q.motion || []), on: true, t_start: 0, t_end: -1, fade: 0, ...JSON.parse(JSON.stringify(pr[2])) };
        if (b.path && b.type !== "loop" && b.type !== "stagger" && b.type !== "repeat" && !this.motionPaths(q).includes(b.path)) b.path = "";
        if (!b.path) b.path = b.type === "loop" ? (this.loopPaths(q)[0] || "") : this.motionPaths(q).find(p => p !== "position" && p !== "scale") || this.motionPaths(q)[0];
        if (String(b.path).startsWith("fx.") && !q.fx) q.fx = { ...fxOf(q) };
        (q.motion = q.motion || []).push(b); this.renderInspector(); motionChanged(); return; }
      const b = MV(e), k = e.target.dataset.mk; if (!b || e.target.hasAttribute("data-num")) return; this.snap();
      if (k === "on" || k === "orient") b[k] = e.target.checked; else b[k] = e.target.value;
      if (k === "type") {                            // a valid path, mode and the type's defaults
        const T = b.type, pr = (M_PRESETS.find(x => x[2].type === T && !x[2].path) || M_PRESETS.find(x => x[2].type === T))[2];
        for (const [dk, dv] of Object.entries(pr)) if (b[dk] === undefined && dk !== "path") b[dk] = dv;
        if (T === "repeat") b.path = "copies";
        else if (T === "stagger") { b.path = "regions"; if (!["sequence", "chase", "wave", "random"].includes(b.mode)) b.mode = "sequence"; }
        else if (T === "loop") { b.mode = ["cycle", "pingpong", "continue"].includes(b.mode) ? b.mode : "cycle"; b.path = this.loopPaths(q).includes(b.path) ? b.path : this.loopPaths(q)[0] || ""; }
        else { delete b.mode; if (!this.motionPaths(q).includes(b.path)) b.path = this.motionPaths(q).find(p => p !== "position" && p !== "scale") || this.motionPaths(q)[0]; }
        if (String(b.path).startsWith("fx.") && !q.fx) q.fx = { ...fxOf(q) };
      }
      if (k === "path" && String(b.path).startsWith("fx.") && !q.fx) q.fx = { ...fxOf(q) };
      if (k === "type" || k === "path" || k === "wave" || k === "trigger" || k === "mode" || k === "order" || k === "layout") this.renderInspector();
      motionChanged(); });
    $("g-motion").addEventListener("click", e => { const d = e.target.closest("[data-mdel]"); if (!d) return; this.snap();
      const q = L(); q.motion.splice(+d.dataset.mdel, 1); if (!q.motion.length) delete q.motion; this.renderInspector(); motionChanged(); });
    $("g-fx").addEventListener("click", e => { const q = L(), del = e.target.closest("[data-fxdel]");
      if (del) { const g = FX_GROUPS.find(x => x[0] === del.dataset.fxdel); this.snap();
        const f = fxOf(q); for (const k of g[2]) f[k] = FX_DEF[k]; q.fx = f;
        const a = this._fxAdded?.get(q); if (a) this._fxAdded.set(q, a.filter(n => n !== g[0]));
        if (q.motion) { q.motion = q.motion.filter(b => !g[2].some(k => b.path === "fx." + k)); if (!q.motion.length) delete q.motion; this.applyTime(this.tl.time); this.drawTimeline(); }
        this.renderInspector(); this.draw(); return; }
      if (!e.target.closest("[data-fxreset]")) return; this.snap(); delete q.fx; this._fxAdded?.delete(q); this.renderInspector(); this.draw(); });
    $("g-fx").addEventListener("change", e => { if (e.target.dataset.id !== "fx-add" || !e.target.value) return; const q = L(), name = e.target.value;
      (this._fxAdded ||= new WeakMap()).set(q, [...(this._fxAdded.get(q) || []), name]); this.renderInspector(); });
    $("sh-col").oninput = e => { this.once(); L().color = e.target.value; this.renderLayers(); this.draw(); };
    $("sh-f").oninput = e => { this.once(); const q = L(); q.shape = { ...q.shape, feather: +e.target.value }; $("sh-f-v").textContent = e.target.value + " px"; this.draw(); };
    $("i-size").oninput = e => { L().size = +e.target.value; $("i-size-v").textContent = L().size + " px"; };
    $("i-clear").onclick = () => { this.snap(); const q = L(); q.canvas.getContext("2d").clearRect(0, 0, this.PW, this.PH); q.dirty = true; q.ver = (q.ver || 0) + 1; this.refresh();
      (window.requestIdleCallback || (f => setTimeout(f, 50)))(() => { if (this.layers?.includes(q)) this.paintPix(q); }); };
    for (const k of ["brightness", "contrast", "saturate", "hue", "sepia"]) $("a-" + k).oninput = e => { this.once(); L().adjust[k] = +e.target.value;
      $("a-" + k + "-v").textContent = k === "hue" ? L().adjust[k] + "°" : (+L().adjust[k]).toFixed(2); this.draw(); };
    // projection mask slot
    this.pmFile = Object.assign(document.createElement("input"), { type: "file", accept: "image/*", hidden: true });
    document.body.appendChild(this.pmFile);
    this.pmFile.onchange = async e => { const f = e.target.files[0]; e.target.value = ""; if (!f) return;
      try { const name = await upload(f, (f.name || "mask.png").replace(/[^\w.\-]+/g, "_"), false); if (this.closed) return;
        this.snap(); Object.assign(this.pm, { source: "file:" + name, on: true }); this.loadPm(); } catch (err) { note("mask upload failed: " + err.message); } };
    const pmBox = $("pm");
    pmBox.addEventListener("click", e => {
      const b = e.target.closest("[data-pm],[data-pmv]"); if (!b || b.disabled) return;
      if (b.dataset.pm === "toggle") { if (e.target.closest("[data-pm=on]")) return; this.pmOpen = !this.pmOpen; this.saveLayout?.({ pm: this.pmOpen }); this.renderPm(); return; }
      if (b.dataset.pmv) { this.snap(); this.pm.view = b.dataset.pmv; this.renderPm(); this.draw(); return; }
      const k = b.dataset.pm;
      if (k === "on") { if (!this.pm.source) { this.setPmSource(this.views._silhouette ? "scene" : "file"); return; } this.snap(); this.pm.on = !this.pm.on; this.loadPm(); return; }
      if (k === "invert") { this.snap(); this.pm.invert = !this.pm.invert; this.loadPm(); return; }
      this.setPmSource(k);
    });
    pmBox.addEventListener("input", e => {
      const k = e.target.dataset.pmr; if (!k) return; this.once(); this.pm[k] = +e.target.value;
      const v = pmBox.querySelector(`[data-pmv-v="${k}"]`); if (v) v.textContent = e.target.value + " px";
      clearTimeout(this._pmTimer); this._pmTimer = setTimeout(() => this.loadPm(), 250);
    });
    // video panel
    const mp = $("g-media");
    mp.addEventListener("change", e => {
      const el = e.target.closest("[data-ms]"), V = L(); if (!el || !V?.media) return;
      const q = V.media.segments[+el.dataset.ms], k = el.dataset.mk; if (!q) return;
      this.snap();
      if (k === "fill") q.fill = el.value;
      else {
        const v = parseFloat(el.value), dur = V.vinfo?.duration || 0;
        if (k === "out" || k === "length") q[k] = Number.isFinite(v) && v > 0 ? v : -1;
        else if (Number.isFinite(v)) q[k] = k === "speed" ? Math.min(20, Math.max(0.05, v)) : k === "in" ? Math.min(Math.max(0, v), Math.max(0, dur - 0.01)) : Math.max(0, v);
      }
      this.segSel = +el.dataset.ms; this.applyTime(this.tl.time); this.renderInspector(); this.draw(); this.drawTimeline();
    });
    mp.addEventListener("input", e => { if (e.target.dataset.id !== "m-vol") return; this.once(); L().media.volume = +e.target.value; $("m-vol-v").textContent = Math.round(e.target.value * 100) + "%"; });
    mp.addEventListener("click", e => {
      const t = e.target, V = L(); if (!V?.media) return;
      if (t.closest("[data-id=m-split]")) { this.splitVideo(); return; }
      if (t.closest("[data-id=m-tl]")) { const end = Math.max(...this.segsOf(V).map(q => q.start + q.length));
        if (end > 0) { this.snap(); this.tl.duration = Math.min(3600, Math.ceil(end * 100) / 100); this.fillTl(); this.setTime(Math.min(this.tl.time, this.tl.duration)); this.scheduleDraft(); } return; }
      const fit = t.closest("[data-mfit]"); if (fit) { this.fitVideo(fit.dataset.mfit); return; }
      if (t.closest("[data-id=m-blend]")) { this.snap(); V.media.blend_frames = !V.media.blend_frames; this.renderInspector(); return; }
      const rev = t.closest("[data-mrev]");
      if (rev) { const q = V.media.segments[+rev.dataset.mrev]; if (q) { this.snap(); q.reverse = !q.reverse; this.applyTime(this.tl.time); this.renderInspector(); this.draw(); this.drawTimeline(); } return; }
      const del = t.closest("[data-mdel]");
      if (del) { if (V.media.segments.length < 2) { note("the last piece of a video: delete the layer instead"); return; }
        this.snap(); V.media.segments.splice(+del.dataset.mdel, 1); this.segSel = null; this.applyTime(this.tl.time); this.renderInspector(); this.draw(); this.drawTimeline(); return; }
      const row = t.closest("[data-msi]"); if (row && !t.closest("input,select")) { this.segSel = +row.dataset.msi; this.renderMediaPanel(); this.drawTimeline(); }
    });
    // timeline
    const fillTl = () => {
      $("tl-dur").value = this.tl.duration; $("tl-fps").value = this.tl.fps;
      $("tl-snap").setAttribute("aria-pressed", String(!!this.tl.snap));
      $("tl-ainfo").hidden = !this.tl.audio;
      if (this.tl.audio) {
        $("tl-aname").textContent = this.tl.audio.file.split("/").pop().replace(/_/g, " ").slice(0, 24) + (this.au ? "" : " (loading…)");
        $("tl-aoff").value = +(this.tl.audio.offset || 0).toFixed(3); $("tl-bpm").value = this.tl.beat?.bpm ?? "";
      }
    };
    this.fillTl = fillTl;
    this.afile = Object.assign(document.createElement("input"), { type: "file", accept: "audio/*", hidden: true });
    document.body.appendChild(this.afile);
    this.afile.onchange = e => { const f = e.target.files[0]; e.target.value = ""; if (f) this.importAudio(f); };
    $("tl-audio").onclick = () => this.afile.click();
    $("tl-arm").onclick = () => { this._audTok = null; this.stopAudio(); this.tl.audio = null; this.tl.beat = null; this.au = null; fillTl(); this.drawTimeline(); this.scheduleDraft(); };
    $("tl-aoff").onchange = e => { const v = parseFloat(e.target.value); if (Number.isFinite(v) && this.tl.audio) { this.tl.audio.offset = v; this.scheduleDraft(); } fillTl(); this.drawTimeline(); };
    const setBpm = v => { if (!(v >= 20 && v <= 400)) { fillTl(); return; } this.tl.beat = { bpm: Math.round(v * 10) / 10, phase: this.tl.beat?.phase || 0 }; fillTl(); this.drawTimeline(); this.scheduleDraft(); };
    $("tl-bpm").onchange = e => setBpm(parseFloat(e.target.value));
    $("tl-half").onclick = () => this.tl.beat && setBpm(this.tl.beat.bpm / 2);
    $("tl-dbl").onclick = () => this.tl.beat && setBpm(this.tl.beat.bpm * 2);
    $("tl-beat1").onclick = () => { if (!this.tl.beat) return;                    // the beat grid goes through the playhead
      const p = 60 / this.tl.beat.bpm, t = this.tl.time + (this.tl.audio?.offset || 0); this.tl.beat.phase = ((t % p) + p) % p; this.drawTimeline(); this.scheduleDraft(); };
    $("tl-h3").onclick = e => { e.stopPropagation(); if (this.clipPop && !this.clipPop.hidden) { this.clipPop.hidden = true; return; } this.openClips(); };
    $("tl-snap").onclick = () => { this.tl.snap = !this.tl.snap; fillTl(); this.scheduleDraft(); };
    this.addMarker = () => { const n = this.tl.markers.length + 1; this.snap(); this.segSel = null;
      this.tl.markers.push({ t: this.tl.time, name: `marker ${n}` }); this.tl.markers.sort((x, y) => x.t - y.t);
      this.tlMark = this.tl.markers.findIndex(m => m.t === this.tl.time); this.tlFocus = true; this.drawTimeline(); this.scheduleDraft(); };
    $("tl-mark").onclick = () => this.addMarker();
    $("tl-play").onclick = () => this.togglePlay();
    $("tl-rec").onclick = () => { this.autokey = !this.autokey; $("tl-rec").setAttribute("aria-pressed", String(this.autokey)); this.drawTimeline(); };
    $("tl-key").onclick = () => this.keyNow();
    $("tl-dur").onchange = e => { const v = Math.min(3600, parseFloat(e.target.value)); if (v > 0 && v !== this.tl.duration) { this.snap(); this.tl.duration = v; this.setTime(this.tl.time); this.scheduleDraft(); } fillTl(); };
    $("tl-fps").onchange = e => { const v = Math.min(120, parseInt(e.target.value, 10)); if (v > 0 && v !== this.tl.fps) { this.snap(); this.tl.fps = v; this.scheduleDraft(); } fillTl(); };
    const tlc = this.tlc; let td = null;
    const hitAt = e => {                               // a ◆ or a marker under the pointer
      const r = tlc.getBoundingClientRect(), g = this.tlGeom(), x = (e.clientX - r.left) * g.dpr, y = (e.clientY - r.top) * g.dpr;
      if (y < g.ruler) { const mi = this.tl.markers.findIndex(m => { const dx = x - this.tX(m.t); return dx > -3 * g.dpr && dx < 9 * g.dpr; }); if (mi >= 0) return { x, marker: mi }; }
      if (y >= 10 * g.dpr && y <= 19 * g.dpr) { const t = this.xT(x); const ci = (this.tl.clips || []).findIndex(c => t >= c.t_start && t <= c.t_end); if (ci >= 0) return { x, clip: ci }; }
      if (g.mrow && y >= g.mtop && y < g.top) {          // the video clip row: a segment, its left / right grip
        const tol = 6 * g.dpr, L = this.cur(), D = this.tl.duration, segs = this.segsOf(L);
        let best = null;                               // the nearest grip over all pieces
        for (const s of segs) {
          const a = this.tX(Math.min(D, s.start)), b = this.tX(Math.min(D, s.start + s.length));
          for (const [edge, ex] of [["l", a], ["r", b]]) {
            const d = Math.abs(x - ex); if (d > tol) continue;
            const side = (edge === "r") === (x < ex) ? 0 : 1;      // at a cut: left of it the left piece's end, right of it the right piece's start
            if (!best || d < best.d - 0.5 || (Math.abs(d - best.d) <= 0.5 && side < best.side)) best = { d, side, seg: s.i, edge };
          }
        }
        if (best) return { x, seg: best.seg, edge: best.edge };
        for (const s of [...segs].reverse()) {
          const a = this.tX(Math.min(D, s.start)), b = this.tX(Math.min(D, s.start + s.length));
          if (x >= a && x <= b) return { x, seg: s.i, edge: "" };
        }
        return { x, segRow: true };
      }
      const rows = this.tlRows(), i = Math.floor((y - g.top) / g.row);
      if (y < g.top || i < 0 || i >= rows.length) return { x };
      const j = rows[i].keys.findIndex(k => Math.abs(this.tX(k.t) - x) < 6 * g.dpr);
      return j >= 0 ? { x, p: rows[i].p, i: j } : { x };
    };
    tlc.addEventListener("pointerdown", e => {
      const fa = document.activeElement;             // a field being typed in commits before anything re-renders it
      if (fa && fa !== document.body && this.root.contains(fa) && fa.matches("input,select,textarea")) fa.blur();
      tlc.setPointerCapture(e.pointerId); this.tlFocus = true; if (this.playing) this.togglePlay();
      const h = hitAt(e), L = this.cur();
      if (h.clip !== undefined) { this.openClips(h.clip); return; }
      if (h.seg !== undefined) {
        this.segSel = h.seg; this.tlSel = null; this.tlMark = null;
        const s0 = this.segsOf(L).find(q => q.i === h.seg);
        td = { mode: "seg", L, raw: L.media.segments[h.seg], s0, edge: h.edge, t0: this.xT(h.x), alt: e.altKey, shift: e.shiftKey, before: this.state() };
        this.renderInspector(); this.drawTimeline(); return;
      }
      this.segSel = null;                            // any other click: no piece selected (Del acts on what was clicked)
      if (h.marker !== undefined) { this.tlMark = h.marker; this.tlSel = null; td = { mode: "marker", m: this.tl.markers[h.marker], before: this.state() }; this.setTime(td.m.t, true); return; }
      this.tlMark = null;
      if (h.p) {
        const k = L.anim[h.p][h.i];
        if (e.altKey) { this.snap(); k.e = { smooth: "linear", linear: "hold", hold: "smooth" }[k.e] || "linear"; this.setTime(this.tl.time); return; }
        this.tlSel = { p: h.p, i: h.i }; td = { mode: "key", k, keys: L.anim[h.p], before: this.state() }; this.setTime(k.t); return;
      }
      this.tlSel = null; td = { mode: "scrub" }; this.setTime(this.snapT(this.xT(h.x), e), true);
    });
    tlc.addEventListener("pointermove", e => {
      if (!td) return;
      const g = this.tlGeom(), x = (e.clientX - tlc.getBoundingClientRect().left) * g.dpr;
      const raw = this.xT(x), t = this.snapT(Math.round(raw * this.tl.fps) / this.tl.fps, e, td.k || td.m);
      if (td.mode === "scrub") { this.setTime(t, true); return; }
      if (td.mode === "seg") { const was = JSON.stringify(td.raw); this.dragSeg(td, raw, e); if (JSON.stringify(td.raw) !== was) td.before = this.commit(td.before); return; }
      if (td.mode === "marker") { if (t !== td.m.t) td.before = this.commit(td.before);
        td.m.t = t; this.tl.markers.sort((p, q) => p.t - q.t); this.tlMark = this.tl.markers.indexOf(td.m); this.setTime(t, true); return; }
      if (t !== td.k.t) td.before = this.commit(td.before);
      td.k.t = t; td.keys.sort((p, q) => p.t - q.t); this.tlSel.i = td.keys.indexOf(td.k);
      this.syncEval(); this.setTime(t, true);
    });
    tlc.addEventListener("dblclick", async e => {       // rename a marker
      const h = hitAt(e); if (h.marker === undefined) return;
      const m = this.tl.markers[h.marker];
      const name = await this.ask("marker", `At ${m.t.toFixed(2)} s:`, [["cancel", "cancel"], ["ok", "rename", "accent"]], m.name || "marker");
      if (name !== null && name !== m.name) { this.snap(); m.name = name; this.drawTimeline(); this.scheduleDraft(); }
    });
    const tlEnd = () => { if (td) { if (td.mode === "key") { const ks = td.keys, k = td.k;       // two keys on one frame: the moved one wins
        for (let i = ks.length - 1; i >= 0; i--) if (ks[i] !== k && Math.abs(ks[i].t - k.t) < 0.5 / this.tl.fps) ks.splice(i, 1);
        this.tlSel.i = ks.indexOf(k); }
      if (td.mode !== "scrub" && !td.before) this.scheduleDraft();       // before still set: it was only a click
      td = null; this.renderInspector(); this.renderLayers(); this.draw(); } };
    tlc.addEventListener("pointerup", tlEnd); tlc.addEventListener("pointercancel", tlEnd);
    this.stepFrame = (dir, toKey) => {
      if (this.playing) this.togglePlay();
      if (!toKey) { this.setTime(this.tl.time + dir / this.tl.fps); return; }
      const ts = [...new Set([...this.tlRows().flatMap(r => r.keys.map(k => k.t)), ...this.tl.markers.map(m => m.t)])].sort((p, q) => p - q);
      const t = dir > 0 ? ts.find(v => v > this.tl.time + 1e-6) : ts.reverse().find(v => v < this.tl.time - 1e-6);
      if (t !== undefined) this.setTime(t);
    };
    this.deleteKey = () => {
      const V = this.cur();
      if (V?.vinfo && this.segSel !== null && this.segSel !== undefined && V.media.segments[this.segSel]) {
        if (V.media.segments.length < 2) { note("the last piece of a video: delete the layer instead"); return true; }
        this.snap(); V.media.segments.splice(this.segSel, 1); this.segSel = null; this.applyTime(this.tl.time); this.renderInspector(); this.draw(); this.drawTimeline(); this.scheduleDraft(); return true;
      }
      if (this.tlMark !== null && this.tlMark !== undefined && this.tl.markers[this.tlMark]) {
        this.snap(); this.tl.markers.splice(this.tlMark, 1); this.tlMark = null; this.drawTimeline(); this.scheduleDraft(); return true; }
      const L = this.cur(), s = this.tlSel; if (!s || !L?.anim?.[s.p]?.[s.i]) return false;
      this.snap(); L.anim[s.p].splice(s.i, 1); if (!L.anim[s.p].length) delete L.anim[s.p];
      this.tlSel = null; this.setTime(this.tl.time); this.scheduleDraft(); return true;
    };
    // light panel (rebuilt on structure changes; sliders update in place so they keep the focus)
    const lp = $("g-light");
    const lampSum = (z) => z.type === "sun" ? `${Math.round(+z.azimuth || 0)}° / ${Math.round(+z.elevation || 0)}°` : `${(+z.x || 0).toFixed(1)} · ${(+z.height || 0).toFixed(1)} · ${(+z.distance || 0).toFixed(1)} m`;
    this.syncLamp = () => {                           // after dragging a lamp on the canvas
      const R = L()?.light, z = R?.lights[this.lsel]; if (!z) return;
      for (const k of ["x", "height"]) { const i = lp.querySelector(`[data-l="q.${k}"]`); if (i) { i.value = z[k]; lp.querySelector(`[data-v="q.${k}"]`).textContent = this.fmtL(z[k], " m"); } }
      const s = lp.querySelector(`[data-li="${this.lsel}"] small`); if (s) s.textContent = lampSum(z);
    };
    const setL = (el, final) => {
      const R = L().light, key = el.dataset.l, raw = el.type === "range" ? +el.value : el.value;
      if (key.startsWith("q.")) {
        const z = R.lights[this.lsel]; if (!z) return; z[key.slice(2)] = raw;
        if (key === "q.color") { const d = lp.querySelector(`[data-li="${this.lsel}"] .kkd-dot`); if (d) d.style.background = raw; }
        const s = lp.querySelector(`[data-li="${this.lsel}"] small`); if (s) s.textContent = lampSum(z);
      } else if (key.startsWith("g.")) {
        const g = R.glow[+el.closest("[data-g]").dataset.g]; if (!g) return;
        if (key === "g.src") { g.by = raw; this.renderLightPanel(); }
        else if (key === "g.by") { if (final) g.by = raw.trim(); }
        else g[key.slice(2)] = raw;
      } else if (key.startsWith("projector.")) {
        R.projector[key.slice(10)] = raw;
      } else {
        R[key] = raw;
        if (key === "environment") this.renderLightPanel();
      }
      const v = lp.querySelector(`[data-v="${key}"]`); if (v) v.textContent = this.fmtL(raw, { "q.power": R.lights[this.lsel]?.type === "sun" ? "" : " W", "q.x": " m", "q.height": " m", "q.distance": " m", "q.size": " m", "q.angle": "°", "q.azimuth": "°", "q.elevation": "°", env_rotation: "°", resolution_scale: "×" }[key] || "");
      this.renderLayers(); this.draw();
    };
    lp.addEventListener("input", e => { const el = e.target.closest("[data-l]"); if (!el || el.tagName === "SELECT" || el.dataset.l === "g.by" || el.dataset.l === "hdri_file") return; this.once(); setL(el, false); });
    lp.addEventListener("change", e => { const el = e.target.closest("[data-l]"); if (!el) return;
      if (el.tagName === "SELECT" || el.dataset.l === "g.by" || el.dataset.l === "hdri_file") { this.snap(); setL(el, true); } });
    lp.addEventListener("click", e => {
      const t = e.target, R = L()?.light; if (!R) return;
      const q = t.closest("[data-lq]"); if (q) { this.lq = q.dataset.lq; this.renderLightPanel(); this.draw(); return; }
      const add = t.closest("[data-ladd]");
      if (add) {
        this.snap(); const kind = add.dataset.ladd;
        if (kind === "glow") { const first = this.layers.find(z => z !== L() && (z.kind === "image" || z.kind === "paint")); R.glow.push({ id: this.newId("g", R.glow), by: first ? "layer:" + first.id : "", color: "#ffb060", strength: 20, on: true }); }
        else { R.lights.push({ id: this.newId("p", R.lights), type: kind, on: true, x: 0, height: this.facadeTop() * 0.4, ...LIGHT_DEF[kind] }); this.lsel = R.lights.length - 1; }
        this.renderLightPanel(); this.renderLayers(); this.draw(); return;
      }
      const on = t.closest("[data-lon]"); if (on) { this.snap(); const z = R.lights[+on.dataset.lon]; z.on = z.on === false; this.renderLightPanel(); this.renderLayers(); this.draw(); return; }
      const x = t.closest("[data-lx]");
      if (x) { this.snap();
        if (x.dataset.lx === "glow") R.glow.splice(+x.closest("[data-g]").dataset.g, 1);
        else { R.lights.splice(+x.closest("[data-li]").dataset.li, 1); this.lsel = Math.max(0, Math.min(this.lsel, R.lights.length - 1)); }
        this.renderLightPanel(); this.renderLayers(); this.draw(); return; }
      if (t.closest("[data-lpj]")) { this.snap(); R.projector.on = !R.projector.on;
        if (R.projector.on && R.view === "AgX" && !R.lights.some(z => z.on !== false)) R.view = "Standard";   // projector alone: true colours
        this.renderLightPanel(); this.renderLayers(); this.draw(); return; }
      const pm = t.closest("[data-lpm]"); if (pm) { this.snap(); R.projector.mode = pm.dataset.lpm; this.renderLightPanel(); this.draw(); return; }
      const vw = t.closest("[data-lview]"); if (vw) { this.snap(); R.view = vw.dataset.lview; this.renderLightPanel(); this.draw(); return; }
      const bgb = t.closest("[data-lbg]"); if (bgb) { this.snap(); R.background = bgb.dataset.lbg; this.renderLightPanel(); this.draw(); return; }
      const row = t.closest("[data-li]"); if (row) { this.lsel = +row.dataset.li; this.renderLightPanel(); this.draw(); }
    });
    // layer list: select, eye, drag to reorder
    const list = $("layers"); let ld = null;
    list.addEventListener("pointerdown", e => {
      this.tlFocus = false; this.tlSel = null;
      const eye = e.target.closest("[data-eye]");
      if (eye) { this.snap(); const q = this.layers[+eye.dataset.eye]; q.visible = !q.visible; this.refresh(); return; }
      const row = e.target.closest(".kkd-layer"); if (!row) return;
      if (this.sel !== +row.dataset.i) this.segSel = null;
      this.sel = +row.dataset.i; this.renderInspector(); this.draw();
      list.querySelectorAll(".kkd-layer").forEach(r => r.setAttribute("aria-selected", String(r === row)));
      ld = { from: this.sel, y0: e.clientY, row, active: false }; list.setPointerCapture(e.pointerId);
    });
    list.addEventListener("pointermove", e => {
      if (!ld) return; if (!ld.active && Math.abs(e.clientY - ld.y0) < 5) return;
      ld.active = true; ld.row.classList.add("dragging");
      const rows = [...list.querySelectorAll(".kkd-layer")]; let to = rows.length;
      for (let i = 0; i < rows.length; i++) { const r = rows[i].getBoundingClientRect(); if (e.clientY < r.top + r.height / 2) { to = i; break; } }
      ld.to = to; let line = list.querySelector(".kkd-drop"); if (!line) { line = document.createElement("div"); line.className = "kkd-drop"; list.appendChild(line); }
      const host = list.getBoundingClientRect(), ref = rows[Math.min(to, rows.length - 1)].getBoundingClientRect();
      line.style.top = ((to < rows.length ? ref.top : ref.bottom) - host.top + list.scrollTop - 1) + "px";     // the list scrolls on its own
      if (e.clientY < host.top + 24) list.scrollTop -= 10; else if (e.clientY > host.bottom - 24) list.scrollTop += 10;
    });
    list.addEventListener("pointerup", () => {
      if (ld && ld.active && ld.to !== undefined) { let to = ld.to; if (to > ld.from) to--; this.moveSel(to); } else if (ld) this.refresh();
      ld = null; list.querySelector(".kkd-drop")?.remove();
    });
    list.addEventListener("pointercancel", () => { if (ld) { ld = null; list.querySelector(".kkd-drop")?.remove(); this.renderLayers(); } });
    // canvas
    const cv = this.cv;
    const toM = e => { const r = cv.getBoundingClientRect(), d = cv.width / r.width; return [((e.clientX - r.left) * d - this.OX()) / this.S(), ((e.clientY - r.top) * d - this.OY()) / this.S()]; };
    this.toM = toM;
    const inside = (q, x, y) => { const a = -q.rotation * Math.PI / 180, cx = q.x + q.w / 2, cy = q.y + q.h / 2, dx = x - cx, dy = y - cy;
      const lx = dx * Math.cos(a) - dy * Math.sin(a), ly = dx * Math.sin(a) + dy * Math.cos(a); return Math.abs(lx) <= q.w / 2 && Math.abs(ly) <= q.h / 2; };
    let drag = null;
    const paint = (q, x0, y0, x1, y1) => {
      const g = q.canvas.getContext("2d"); g.save(); g.globalCompositeOperation = this.tool === "erase" ? "destination-out" : "source-over";
      g.strokeStyle = q.color; g.lineWidth = q.size * this.k; g.lineCap = "round"; g.lineJoin = "round";
      g.beginPath(); g.moveTo(x0 * this.k, y0 * this.k); g.lineTo(x1 * this.k, y1 * this.k); g.stroke(); g.restore(); q.dirty = true; q.ver = (q.ver || 0) + 1; this.draw();
    };
    const screen = e => { const r = cv.getBoundingClientRect(), d = cv.width / r.width; return [(e.clientX - r.left) * d, (e.clientY - r.top) * d]; };
    const lampAt = e => { const [sx, sy] = screen(e), tol = 14 * Math.min(devicePixelRatio || 1, 2);
      return this.lightSpots().filter(s => Math.hypot(sx - s.X, sy - s.Y) < tol).pop(); };
    cv.addEventListener("pointerdown", e => {
      const [x, y] = toM(e); cv.setPointerCapture(e.pointerId); this.tlFocus = false;
      if (this.tool === "hand" || this.space || e.button === 1) { drag = { mode: "pan", sx: e.clientX, sy: e.clientY, px: this.panX, py: this.panY }; cv.style.cursor = "grabbing"; return; }
      if (this.tool === "poly") { (this.polyDraft ||= []).push([x, y]); this.draw(); return; }
      if (this.tool === "move") {                    // lamps of the selected light layer: drag them over the facade
        const s = lampAt(e);
        if (s) { this.lsel = s.i; this.renderLightPanel(); drag = { mode: "lamp", q: s.q, x, y, lx: +s.q.x || 0, lh: +s.q.height || 0, pending: this.state(), sx: e.clientX, sy: e.clientY }; this.draw(); return; }
      }
      if (this.tool === "pick") { const r = this.regionAt(x, y), q = this.cur();
        if (r && q && q.kind !== "base") { this.snap(); q.clip = q.clip === r.name ? "" : r.name; this.refresh(); } return; }
      if (this.tool === "brush" || this.tool === "erase") { let q = this.cur(); if (!q || q.kind !== "paint") q = this.addLayer("paint");
        this.snap(); drag = { mode: "paint", q, last: [x, y] }; paint(q, x, y, x, y); return; }
      const q = this.cur(), tol = 10 / this.S();
      if (q && this.transformable(q) && q.visible) {
        const h = this.handles(q);
        const lazy = { pending: this.state(), sx: e.clientX, sy: e.clientY };
        if (Math.hypot(x - h.rot[0], y - h.rot[1]) < tol * 1.5) { drag = { mode: "rot", q, a0: Math.atan2(y - h.c[1], x - h.c[0]), r0: q.rotation, ...lazy }; return; }
        const SG = [[-1, -1], [1, -1], [1, 1], [-1, 1]], EG = [[0, -1], [1, 0], [0, 1], [-1, 0]];
        const ci = h.corners.findIndex(p => Math.hypot(x - p[0], y - p[1]) < tol * 1.5);
        if (ci >= 0) { drag = { mode: "corner", q, sg: SG[ci], anchor: h.corners[(ci + 2) % 4], c0: h.c, w0: q.w, h0: q.h, targets: this.snapTargets(q), ...lazy }; return; }
        const ei = h.edges.findIndex(p => Math.hypot(x - p[0], y - p[1]) < tol * 1.3);
        if (ei >= 0) { drag = { mode: "edge", q, sg: EG[ei], anchor: h.edges[(ei + 2) % 4], c0: h.c, w0: q.w, h0: q.h, targets: this.snapTargets(q), ...lazy }; return; }
      }
      const full = q => q.w >= this.W * 0.98 && q.h >= this.H * 0.98;
      let idx = this.layers.findIndex(q => q.visible && this.transformable(q) && !full(q) && inside(q, x, y));
      if (idx < 0) idx = this.layers.findIndex(q => q.visible && this.transformable(q) && inside(q, x, y));
      if (idx >= 0) { if (this.sel !== idx) { this.commitReg?.(); this.segSel = null; } this.sel = idx; let q2 = this.layers[idx];
        drag = { mode: "move", altCopy: e.altKey && q2.kind !== "base", q: q2, x, y, lx: q2.x, ly: q2.y, targets: this.snapTargets(q2), pending: this.state(), sx: e.clientX, sy: e.clientY }; this.refresh(); }
    });
    cv.addEventListener("pointermove", e => {
      const [x, y] = toM(e), r = cv.getBoundingClientRect();
      const reg = this.regionAt(x, y);
      this.$("hud").textContent = `x ${Math.round(x)} · y ${Math.round(y)} px${reg ? " · " + reg.name : ""} · ${Math.round(this.zoom * 100)}%`;
      if (this.tool === "pick" && reg?.id !== this.hoverRegion?.id) { this.hoverRegion = reg; this.draw(); }
      if (this.tool === "brush" || this.tool === "erase") { const q = this.cur(), sz = (q && q.kind === "paint" ? q.size : 60) * this.S() / (cv.width / r.width);
        Object.assign(this.$("brush").style, { left: (e.clientX - r.left) + "px", top: (e.clientY - r.top) + "px", width: sz + "px", height: sz + "px" }); }
      if (!drag) return;
      if (drag.mode === "pan") { const d = cv.width / r.width; this.panX = drag.px + (e.clientX - drag.sx) * d; this.panY = drag.py + (e.clientY - drag.sy) * d; this.draw(); return; }
      if (drag.mode === "paint") { paint(drag.q, drag.last[0], drag.last[1], x, y); drag.last = [x, y]; return; }
      if (drag.pending) {
        if (Math.hypot(e.clientX - drag.sx, e.clientY - drag.sy) < 3) return;
        if (drag.altCopy) { const c = this.duplicate(false); if (c) { drag.q = c; drag.pending = null; } drag.altCopy = false; }   // Alt drag: the copy moves (one undo step), a plain Alt click copies nothing
        if (drag.pending) drag.pending = this.commit(drag.pending);
      }
      if (drag.mode === "lamp") {
        const [ax, ah] = this.toFacade(drag.x, drag.y), [bx, bh] = this.toFacade(x, y);
        drag.q.x = +(drag.lx + bx - ax).toFixed(2); drag.q.height = +(drag.lh + bh - ah).toFixed(2);
        this.syncLamp(); this.draw(); return;
      }
      const q = drag.q;
      const snapping = this.snapOn && !(e.ctrlKey || e.metaKey);
      this.guides = [];
      if (drag.mode === "move") {
        q.x = drag.lx + (x - drag.x); q.y = drag.ly + (y - drag.y);
        if (snapping) {
          const [x0, y0, x1, y1] = this.aabb(q);
          const bx = this.nearest([x0, (x0 + x1) / 2, x1], drag.targets.tx), by = this.nearest([y0, (y0 + y1) / 2, y1], drag.targets.ty);
          if (bx) { q.x += bx[0]; this.guides.push({ x: bx[1] }); }
          if (by) { q.y += by[0]; this.guides.push({ y: by[1] }); }
        }
      }
      if (drag.mode === "corner" || drag.mode === "edge") {
        let px = x, py = y;
        if (snapping) {
          const bx = this.nearest([px], drag.targets.tx), by = this.nearest([py], drag.targets.ty);
          if (bx) { px += bx[0]; this.guides.push({ x: bx[1] }); }
          if (by) { py += by[0]; this.guides.push({ y: by[1] }); }
        }
        const a = q.rotation * Math.PI / 180, c = Math.cos(a), s_ = Math.sin(a), alt = e.altKey;
        const o = alt ? drag.c0 : drag.anchor, dx = px - o[0], dy = py - o[1];
        const lx = dx * c + dy * s_, ly = -dx * s_ + dy * c, mul = alt ? 2 : 1;
        let w = drag.w0, h = drag.h0;
        if (drag.mode === "corner") {
          w = Math.abs(lx) * mul; h = Math.abs(ly) * mul;
          if (!e.shiftKey) { const k = Math.max(w / drag.w0, h / drag.h0); w = drag.w0 * k; h = drag.h0 * k; }
        } else if (drag.sg[0]) { w = Math.abs(lx) * mul; if (e.shiftKey) h = drag.h0 * w / drag.w0; }
        else { h = Math.abs(ly) * mul; if (e.shiftKey) w = drag.w0 * h / drag.h0; }
        w = Math.max(w, 4); h = Math.max(h, 4);
        let cx = drag.c0[0], cy = drag.c0[1];
        if (!alt) {                                    // the opposite corner / edge stays where it was
          const hx = drag.sg[0] * w / 2, hy = drag.sg[1] * h / 2;
          cx = drag.anchor[0] + hx * c - hy * s_; cy = drag.anchor[1] + hx * s_ + hy * c;
          if (drag.mode === "edge") {                  // anchor is an edge midpoint: offset only along the moved axis
            const ex = drag.sg[0] ? drag.sg[0] * w / 2 : 0, ey = drag.sg[1] ? drag.sg[1] * h / 2 : 0;
            cx = drag.anchor[0] + ex * c - ey * s_; cy = drag.anchor[1] + ex * s_ + ey * c;
          }
        }
        q.w = w; q.h = h; q.x = cx - w / 2; q.y = cy - h / 2;
      }
      if (drag.mode === "rot") { let a = drag.r0 + (Math.atan2(y - (q.y + q.h / 2), x - (q.x + q.w / 2)) - drag.a0) * 180 / Math.PI; if (e.shiftKey) a = Math.round(a / 15) * 15; q.rotation = a; }
      this.draw();
    });
    const idlePix = q => (window.requestIdleCallback || (f => setTimeout(f, 50)))(() => { if (this.layers?.includes(q) && q.kind === "paint") this.paintPix(q); });
    const endDrag = () => { if (drag?.mode === "paint" && drag.q) idlePix(drag.q);
      if (drag?.mode === "paint" || drag?.mode === "lamp") this.renderLayers(); if (drag && !drag.pending) this.scheduleDraft(); drag = null; this.guides = []; this.setTool(this.tool); };
    cv.addEventListener("pointerup", endDrag);
    cv.addEventListener("pointercancel", endDrag);
    cv.addEventListener("lostpointercapture", () => { if (drag) endDrag(); });
    cv.addEventListener("dblclick", e => {             // light layer selected: a point light where you click
      if (this.tool === "poly") { this.finishPoly(); return; }
      const L = this.cur(); if (!L || L.kind !== "light" || !this.frame() || this.tool !== "move" || lampAt(e)) return;
      const [x, y] = toM(e); if (x < 0 || y < 0 || x > this.W || y > this.H) return;
      const [fx, fh] = this.toFacade(x, y);
      this.snap(); L.light.lights.push({ id: this.newId("p", L.light.lights), type: "point", on: true, x: +fx.toFixed(2), height: +fh.toFixed(2), ...LIGHT_DEF.point });
      this.lsel = L.light.lights.length - 1; this.renderLightPanel(); this.renderLayers(); this.draw();
    });
    cv.addEventListener("pointerleave", () => { if (!drag) this.$("brush").hidden = true; });
    cv.addEventListener("pointerenter", () => { this.$("brush").hidden = !(this.tool === "brush" || this.tool === "erase"); });
    cv.addEventListener("wheel", e => {
      e.preventDefault();
      if (e.ctrlKey || e.metaKey) { this.zoom = Math.max(0.2, Math.min(10, this.zoom * (e.deltaY > 0 ? 0.9 : 1.1))); this.draw(); return; }
      const q = this.cur(); if (!q || !this.transformable(q)) return; this.once();
      if (e.shiftKey) q.rotation += e.deltaY > 0 ? 2 : -2;
      else { const s = e.deltaY > 0 ? 0.95 : 1.05, cx = q.x + q.w / 2, cy = q.y + q.h / 2; q.w *= s; q.h *= s; q.x = cx - q.w / 2; q.y = cy - q.h / 2; }
      this.draw();
    }, { passive: false });
    // keyboard (captured, so ComfyUI's own shortcuts stay quiet while the window is open)
    this.onKey = e => {
      if (!this.root.isConnected) return;
      if (document.querySelector(".kkd-modal")) {    // a dialog of the window is open: its own keys only
        if (!e.target.closest?.(".kkd-modal")) e.stopPropagation();
        return;
      }
      if (!this.ready) { e.stopPropagation(); if (e.key === "Escape") this.close(); return; }     // still loading
      const typing = e.target.matches?.("textarea, select, input:not([type=range]):not([type=color]):not([type=checkbox]):not([type=file])");
      const k = e.key, ctrl = e.ctrlKey || e.metaKey;
      if (e.target.matches?.("input[type=range]") && (k.startsWith("Arrow") || ["Home", "End", "PageUp", "PageDown"].includes(k))) { e.stopPropagation(); return; }   // the slider moves
      if (typing && /^i-(clip|holes)$/.test(e.target.dataset?.id || "") && this.regKey?.(e.target, e)) { e.stopPropagation(); return; }
      if (this.tool === "poly" && (k === "Escape" || k === "Enter")) { e.stopPropagation(); e.preventDefault();
        if (k === "Enter") this.finishPoly(); else { this.polyDraft = null; this.setTool("move"); } return; }
      if (typing && k === "Escape") { e.stopPropagation(); e.preventDefault(); this.fillTl?.(); e.target.blur(); this.root.focus?.(); return; }   // leave the field, keep the window
      if (k === "Escape") { e.stopPropagation(); const open = [this.menu, this.filemenu, this.keypop, this.outpop, this.diffpop, this.exportpop, this.clipPop].filter(Boolean).find(p => !p.hidden); if (open) open.hidden = true; else this.close(); return; }
      if (typing) { e.stopPropagation(); if (ctrl && k === "Enter") { e.preventDefault(); this.apply(); } return; }
      e.stopPropagation();
      if ((k === " " || k === "Enter") && e.target.matches?.("input[type=checkbox],button,select")) return;   // the control's own key
      if (k === " ") { this.space = true; this.cv.style.cursor = "grab"; e.preventDefault(); return; }
      if (ctrl && k.toLowerCase() === "z") { e.preventDefault(); e.shiftKey ? this.doRedo() : this.doUndo(); return; }
      if (ctrl && k.toLowerCase() === "y") { e.preventDefault(); this.doRedo(); return; }
      if (ctrl && k === "Enter") { e.preventDefault(); this.apply(); return; }
      if (ctrl && k.toLowerCase() === "j") { e.preventDefault(); this.duplicate(); return; }
      if (ctrl && k.toLowerCase() === "k") { e.preventDefault(); this.splitVideo(); return; }
      if (ctrl && k.toLowerCase() === "s") { e.preventDefault(); this.saveAs(); return; }
      if (ctrl && k.toLowerCase() === "o") { e.preventDefault(); this.file.click(); return; }
      if (ctrl && e.altKey && k.toLowerCase() === "g") { e.preventDefault(); const q = this.cur(); if (q && q.kind !== "base") { this.snap(); q.mask.by = q.mask.by === "below" ? "" : "below"; this.refresh(); } return; }
      if (ctrl && (k === "]" || k === "}")) { e.preventDefault(); this.moveSel(e.shiftKey ? 0 : this.sel - 1); return; }
      if (ctrl && (k === "[" || k === "{")) { e.preventDefault(); this.moveSel(e.shiftKey ? this.layers.length - 1 : this.sel + 1); return; }
      if (ctrl) return;
      if (e.shiftKey && k.toLowerCase() === "n") { this.addLayer("paint"); return; }
      if (e.shiftKey && k.toLowerCase() === "a") { this.addLayer("adjust"); return; }
      if (e.shiftKey && k.toLowerCase() === "l") { this.addLayer("light"); return; }
      if (k.toLowerCase() === "d" && !e.shiftKey && !e.altKey) { this.diffuse(); return; }
      if (e.altKey && k.toLowerCase() === "i") { const q = this.cur(); if (q?.mask.by) { this.snap(); q.mask.invert = !q.mask.invert; this.refresh(); } return; }
      if (e.altKey && k.toLowerCase() === "h") { const q = this.cur(); if (q) { this.snap(); q.visible = !q.visible; this.refresh(); } return; }
      if (k.toLowerCase() === "s" && !e.altKey && !e.shiftKey) { this.toggleSnap(); return; }
      const t = { v: "move", r: "pick", b: "brush", e: "erase", h: "hand" }[k.toLowerCase()]; if (t && !e.altKey) { this.setTool(t); return; }
      if (/^[1-9]$/.test(k)) { const v = Object.keys(this.views).filter(n => !n.startsWith("_"))[+k - 1]; if (v) this.setView(v); return; }
      if (k === "0") { this.zoom = 1; this.panX = this.panY = 0; this.draw(); return; }
      if (k.toLowerCase() === "n" && !e.shiftKey) { this.names = !this.names; this.draw(); return; }
      if (k === "?") { this.$("keysbtn").click(); return; }
      if (k === "Delete" || k === "Backspace") { if (this.tlFocus) { if (!this.deleteKey()) note("select a key, marker or video piece to delete it"); return; } this.removeSel(); return; }
      if (k === "," || k === ".") { this.stepFrame(k === "." ? 1 : -1, false); return; }
      if (k === "<" || k === ">") { this.stepFrame(k === ">" ? 1 : -1, true); return; }
      if (k.toLowerCase() === "p" && !e.shiftKey) { if (!e.repeat) this.togglePlay(); return; }
      if (k.toLowerCase() === "m" && !e.shiftKey && !e.altKey) { this.addMarker(); return; }
      if (k.toLowerCase() === "k") { if (e.shiftKey) this.$("tl-rec").click(); else this.keyNow(); return; }
      if (k === "Home") { this.setTime(0); return; }
      if (k === "[" || k === "]") { const q = this.cur(); if (q?.kind === "paint") { q.size = Math.max(4, Math.min(600, Math.round(q.size * (k === "]" ? 1.2 : 1 / 1.2)))); this.renderInspector(); } return; }
      if (k.startsWith("Arrow")) { const q = this.cur(); if (!this.transformable(q)) return; e.preventDefault(); this.once();
        const d = e.shiftKey ? 20 : 2; if (k === "ArrowLeft") q.x -= d; if (k === "ArrowRight") q.x += d; if (k === "ArrowUp") q.y -= d; if (k === "ArrowDown") q.y += d; this.draw(); }
    };
    this.onKeyUp = e => { if (e.key === " ") { this.space = false; this.setTool(this.tool); } if (this.root.isConnected) e.stopPropagation(); };
    window.addEventListener("keydown", this.onKey, true); window.addEventListener("keyup", this.onKeyUp, true);
    this.onBlur = () => { if (this.space) { this.space = false; this.setTool(this.tool); } };   // Space released in another app
    window.addEventListener("blur", this.onBlur);
    // drop / paste images
    let depth = 0; const hasFiles = e => [...(e.dataTransfer?.types || [])].includes("Files");
    this.on_dragenter = e => { if (!hasFiles(e)) return; depth++; this.dropzone.hidden = false; };
    this.on_dragleave = e => { if (!hasFiles(e)) return; if (--depth <= 0) { depth = 0; this.dropzone.hidden = true; } };
    this.on_dragover = e => { if (hasFiles(e)) { e.preventDefault(); e.stopPropagation(); } };
    this.on_drop = e => { if (!hasFiles(e)) return; e.preventDefault(); e.stopPropagation(); depth = 0; this.dropzone.hidden = true;
      const at = e.target === this.cv ? toM(e) : null; [...e.dataTransfer.files].forEach((f, i) => this.importFile(f, at && [at[0] + i * 80, at[1] + i * 80])); };
    this.on_paste = e => {                              // never reaches the graph behind the window (it would paste nodes)
      if (!this.root.isConnected || e.target.matches?.("input,textarea")) return;
      e.preventDefault(); e.stopPropagation();
      const f = [...(e.clipboardData?.items || [])].find(i => i.type.startsWith("image/"));
      if (f) this.importFile(f.getAsFile());
    };
    for (const t of ["dragenter", "dragleave", "dragover", "drop", "paste"]) document.addEventListener(t, this[`on_${t}`], true);
    // hover helpers
    let timer = null;
    this.onTip = e => {
      const t = e.target.closest?.("[data-tip]"); clearTimeout(timer);
      const pop = [this.menu, this.filemenu, this.keypop, this.outpop, this.exportpop, this.diffpop, this.regpick].find(m => m && !m.hidden);
      if (!t || !(this.root.contains(t) || this.menu.contains(t)) || (pop && !pop.contains(t))) { this.tipEl.style.opacity = 0; return; }
      timer = setTimeout(() => {
        const [txt, key] = t.dataset.tip.split("|"); this.tipEl.innerHTML = `<b>${esc(txt)}</b>${key ? key.split(" / ").map(x => `<kbd>${esc(x)}</kbd>`).join("") : ""}`;
        const r = t.getBoundingClientRect(); this.tipEl.style.opacity = 1;
        this.tipEl.style.left = Math.min(innerWidth - this.tipEl.offsetWidth - 8, Math.max(8, r.left + r.width / 2 - this.tipEl.offsetWidth / 2)) + "px";
        this.tipEl.style.top = (r.bottom + 8 + this.tipEl.offsetHeight > innerHeight ? r.top - this.tipEl.offsetHeight - 8 : r.bottom + 8) + "px";
      }, 350);
    };
    document.addEventListener("pointerover", this.onTip, true);
  }
}

function openDirector(node) {
  const m = node.properties?.kuba_director_manifest;
  if (!m) { note("run the workflow once, then open the director - it needs the node's images"); return; }
  if (Director.current) Director.current.close();
  Director.current = new Director(node, m);
  Director.current.open();
}
window.kubakubDirector = () => Director.current;     // the open window (automated UI tests, the browser console)

function hideWidget(w) {
  if (!w) return;
  w.hidden = true; w.computeSize = () => [0, -4];
  if (w.element) w.element.style.display = "none";
  if (w.inputEl) w.inputEl.style.display = "none";
}

app.registerExtension({
  name: "kubakub.director",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== "KUBA_Director") return;
    const created = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = created?.apply(this, arguments);
      hideWidget(this.widgets?.find(w => w.name === "document"));
      const btn = this.addWidget("button", "open director", null, () => openDirector(this));
      btn.serialize = false;                     // a button has no value to save in the workflow
      this.properties = this.properties || {};
      return r;
    };
    const configured = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function () {
      const r = configured?.apply(this, arguments);
      hideWidget(this.widgets?.find(w => w.name === "document"));
      return r;
    };
    const executed = nodeType.prototype.onExecuted;
    nodeType.prototype.onExecuted = function (msg) {
      const r = executed?.apply(this, arguments);
      const m = msg?.kuba_director?.[0];
      if (m) { this.properties = this.properties || {}; this.properties.kuba_director_manifest = m; }
      return r;
    };
  },
});
