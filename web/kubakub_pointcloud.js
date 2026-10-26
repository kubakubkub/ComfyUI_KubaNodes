// kubakub point cloud on the node: the cloud live in WebGL2, from the frames, depth (and field / tracks) of the last
// run. Drag = look around, Shift-drag = move, wheel = nearer; 'start = view' / 'end = view' set the camera move,
// 'see the move' plays it. The look lives in the hidden 'look' widget (JSON, = kubakub/pointcloud.py look()).
// Every point is drawn from its index alone (gl_VertexID): grid, pcg hash, depth, motion - the same formulas as
// pointcloud.points(), so the render is what plays here. Plain ES module, no build step, no network.
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { delaunayEdges, mstEdges } from "./kubakub_track.js";
import { CAM_KEYS, basis, cameraAt, gridSize, gridUV, morphAt, project, rankPerm, trackEdges, trackWorld, LUMA } from "./kubakub_pointcloud_math.js";

const HERE = new URL(".", import.meta.url).href;
const ORANGE = "#f18a58", LAV = "#a187b7";
const PALETTE = ["#ffffff", ORANGE, LAV, "#39ff6a", "#e8b04a", "#ff3b30", "#6ae4ff"];
const MAX_POINT = 64, RING = 14;                 // = pointcloud.MAX_POINT, RING

// ------------------------------------------------------------------------------------------ shaders
const VS = `#version 300 es
precision highp float; precision highp int; precision highp usampler2D; precision highp sampler2D;
uniform sampler2D uColA, uDepA, uColB, uDepB, uField;
uniform usampler2D uPerm;
uniform int uGW, uGH, uColour, uHasField, uMorphOn, uFlip;
uniform float uJitter, uDepth, uFarCut, uSize, uOpacity, uT, uDrift, uBreathe, uWind, uPush, uScan, uS, uSwarm, uAspect, uOutAspect, uH, uMaxPoint;
uniform vec3 uTintA, uTintB, uCamC, uCamR, uCamU, uCamF;
out vec3 vCol; out float vA; out float vSize;
uint pcg(uint v) { uint s = v * 747796405u + 2891336453u; uint w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u; return (w >> 22u) ^ w; }
float rnd(uint i, uint k) { return float(pcg(i * 8u + k)) / 4294967295.0; }
vec2 gridUV(uint i) { uint gw = uint(uGW); float gx = float(i % gw), gy = float(i / gw);
  return vec2((gx + 0.5 + (rnd(i, 0u) - 0.5) * uJitter) / float(uGW), (gy + 0.5 + (rnd(i, 1u) - 0.5) * uJitter) / float(uGH)); }
float depthAt(sampler2D t, vec2 uv) { float d = texture(t, uv).r; return uFlip == 1 ? 1.0 - d : d; }
float distOf(float d) { return 1.0 + uDepth * 2.0 * (1.0 - d); }
vec3 placeAt(vec2 uv, float d) { float di = distOf(d); return vec3((uv.x - 0.5) * uAspect * di, (0.5 - uv.y) * di, -di); }
vec3 dirOf(uint i) { vec3 q = vec3(rnd(i, 5u) - 0.5, rnd(i, 6u) - 0.5, rnd(i, 7u) - 0.5); return q / (length(q) + 1e-6); }
void main() {
  uint i = uint(gl_VertexID);
  vec2 uv = gridUV(i);
  vec3 c = texture(uColA, uv).rgb; float d = depthAt(uDepA, uv);
  float dist = distOf(d); vec3 P = placeAt(uv, d);
  if (uMorphOn == 1 && uS > 0.0) {
    uint j = texelFetch(uPerm, ivec2(int(i % uint(uGW)), int(i / uint(uGW))), 0).r;
    vec2 uv2 = gridUV(j); vec3 c2 = texture(uColB, uv2).rgb; float d2 = depthAt(uDepB, uv2);
    float dist2 = distOf(d2); vec3 P2 = placeAt(uv2, d2);
    P = P + (P2 - P) * uS; c = c + (c2 - c) * uS; d = d + (d2 - d) * uS; dist = dist + (dist2 - dist) * uS;
    P += dirOf(i) * (uSwarm * sin(3.14159265 * uS) * 0.25 * dist);
  }
  float tt = uT, r2 = rnd(i, 2u), r3 = rnd(i, 3u), r4 = rnd(i, 4u), tau = 6.2831853;
  if (uDrift > 0.0) P += vec3(sin(0.8 * tt + tau * r2), sin(0.7 * tt + tau * r3), sin(0.6 * tt + tau * r4)) * (uDrift * 0.02 * dist);
  if (uBreathe > 0.0) P *= 1.0 + uBreathe * 0.05 * sin(1.6 * tt - d * 6.0);
  if (uWind > 0.0) { P.x += uWind * 0.06 * dist * (sin(1.3 * tt + uv.y * 9.0 + r2 * 3.0) + 0.5 * sin(2.3 * tt + r3 * tau));
                     P.y += uWind * 0.02 * dist * sin(1.7 * tt + uv.x * 7.0 + r4 * tau); }
  if (uHasField == 1 && uPush > 0.0) { float f = texture(uField, uv).r; P += dirOf(i) * (uPush * f * 0.3 * dist * (0.4 + r2)); }
  c = clamp(c, 0.0, 1.0);
  float lum = dot(c, vec3(${LUMA.join(", ")}));
  if (uColour == 1) c = vec3(lum); else if (uColour == 2) c = uTintA + (uTintB - uTintA) * lum; else if (uColour == 3) c = 1.0 - c;
  if (uScan > 0.0) { float sc = 1.0 - fract(tt * 0.25); float w = max(0.0, 1.0 - abs(d - sc) / 0.04) * uScan;
                     c = c + (1.0 - c) * min(w * 0.8, 1.0); P += dirOf(i) * (w * 0.02 * dist); }
  vec3 q = P - uCamC; float z = dot(q, uCamF);
  if (d < uFarCut || z <= 0.05) { gl_Position = vec4(2.0, 2.0, 2.0, 1.0); gl_PointSize = 0.0; vA = 0.0; vSize = 0.0; vCol = c; return; }
  gl_Position = vec4(dot(q, uCamR) / (z * 0.5 * uOutAspect), dot(q, uCamU) / (z * 0.5), clamp(z / 50.0, 0.0, 1.0) * 2.0 - 1.0, 1.0);
  float px = uSize * (uH / float(uGH)) * dist / z;
  vA = uOpacity * min(1.0, px * px); vSize = clamp(px, 1.0, uMaxPoint); gl_PointSize = vSize; vCol = c;
}`;
const FS = `#version 300 es
precision highp float;
in vec3 vCol; in float vA; in float vSize; uniform int uBlend; out vec4 o;
void main() {
  float a = vA;
  if (vSize >= 2.5) { float r = length(gl_PointCoord - 0.5) * 2.0; if (r > 1.0) discard; a *= 1.0 - smoothstep(0.6, 1.0, r); }
  if (uBlend == 0) o = vec4(vCol * a, 1.0); else if (uBlend == 1) o = vec4(1.0 - a * (1.0 - vCol), 1.0); else o = vec4(vCol, 1.0);
}`;

// ------------------------------------------------------------------------------------------ UI helpers
let styled_ = false;
function style() {
  if (styled_) return; styled_ = true;
  const s = document.createElement("style");
  s.textContent = `
@font-face{font-family:"Hanken Grotesk KKD";src:url("${HERE}fonts/HankenGrotesk-latin-var.woff2") format("woff2");font-weight:200 800;font-display:swap}
.kkp{display:flex;flex-direction:column;gap:5px;width:100%;height:100%;font-family:"Hanken Grotesk KKD",system-ui,sans-serif;font-size:11px;color:rgba(255,255,255,.85);text-transform:lowercase;box-sizing:border-box}
.kkp-stage{flex:1;min-height:320px;position:relative;background:#0b0b0c;border-radius:6px;overflow:hidden;cursor:grab}
.kkp-stage canvas{position:absolute;inset:0;width:100%;height:100%;display:block;touch-action:none;outline:none}
.kkp-stage .msg{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;text-align:center;padding:16px;color:rgba(255,255,255,.55);pointer-events:none}
.kkp-row{display:flex;flex-wrap:wrap;gap:3px;align-items:center}
.kkp-row>.k{color:rgba(255,255,255,.45);min-width:58px}
.kkp-row input[type=range]{flex:1;min-width:60px;max-width:170px;accent-color:${ORANGE}}
.kkp-row input[type=color]{width:22px;height:18px;border:none;padding:0;background:none;cursor:pointer}
.kkp-row .v{min-width:36px;color:rgba(255,255,255,.6);font-variant-numeric:tabular-nums}
.kkp-row .time{min-width:70px;color:rgba(255,255,255,.6);font-variant-numeric:tabular-nums;text-align:right}
.kkp button{font:inherit;text-transform:lowercase;border:1px solid rgba(255,255,255,.14);background:#1d1b20;color:rgba(255,255,255,.8);border-radius:999px;padding:1px 8px;cursor:pointer;line-height:16px}
.kkp button:hover{border-color:${ORANGE};color:#fff}
.kkp button[aria-pressed=true]{background:${LAV};border-color:${LAV};color:#141414}
.kkp button.act{border-color:${ORANGE};color:${ORANGE}}
.kkp button.sw{width:16px;height:16px;padding:0;border-radius:50%}
.kkp-hint{color:rgba(255,255,255,.5);min-height:14px}
.kkp-panel{display:flex;flex-direction:column;gap:3px;overflow:auto;max-height:34%}
.kkp-panel h4{margin:4px 0 0;font-size:11px;font-weight:700;color:#fff;letter-spacing:.04em}`;
  document.head.appendChild(s);
}
function hideWidget(w) { if (!w) return; w.hidden = true; w.computeSize = () => [0, -4]; if (w.element) w.element.style.display = "none"; if (w.inputEl) w.inputEl.style.display = "none"; }
const pad = (v, n) => String(v).padStart(n, "0");
const frameURL = (side, i) => api.apiURL(`/view?filename=${pad(i, 5)}.jpg&subfolder=${encodeURIComponent(side.subfolder)}&type=${encodeURIComponent(side.type || "temp")}`);
const viewURL = m => api.apiURL(`/view?filename=${encodeURIComponent(m.filename)}&subfolder=${encodeURIComponent(m.subfolder || "")}&type=${encodeURIComponent(m.type || "temp")}`);
const esc = s => String(s ?? "").replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
const hex01 = h => [1, 3, 5].map(i => parseInt(h.slice(i, i + 2), 16) / 255);
const stop = el => { for (const t of ["pointerdown", "pointermove", "pointerup", "wheel", "keydown"]) el.addEventListener(t, e => { if (!e.target.closest?.("canvas")) e.stopPropagation(); }, { passive: true }); };
const clone = o => JSON.parse(JSON.stringify(o));

let DEFAULTS = null;                                  // = pointcloud.DEFAULT_LOOK, from the node file
async function defaults() {
  if (DEFAULTS) return DEFAULTS;
  try { DEFAULTS = await (await api.fetchApi("/kubakub/pointcloud/defaults")).json(); } catch { DEFAULTS = null; }
  return DEFAULTS;
}
function merged(raw, def) {                           // the look on the widget over the defaults (pointcloud.look() clamps on the server)
  const out = clone(def || {}); let j = {};
  try { j = raw ? JSON.parse(raw) : {}; } catch { j = {}; }
  for (const [k, v] of Object.entries(j)) {
    if (v && typeof v === "object" && !Array.isArray(v) && out[k] && typeof out[k] === "object") {
      for (const [k2, v2] of Object.entries(v)) out[k][k2] = v2 && typeof v2 === "object" && !Array.isArray(v2) ? { ...(out[k][k2] || {}), ...v2 } : v2;
    } else out[k] = v;
  }
  return out;
}

// ------------------------------------------------------------------------------------------ images of the last run
class Frames {
  constructor(onLoad) { this.onLoad = onLoad; this.cache = new Map(); }
  img(side, i) {
    if (!side) return null; i = Math.min(Math.max(0, i), side.n - 1);
    const key = side.subfolder + "|" + i; let im = this.cache.get(key);
    if (!im) { im = new Image(); im.crossOrigin = "anonymous"; im.onload = () => this.onLoad(); im.src = frameURL(side, i); this.cache.set(key, im);
      if (this.cache.size > 600) this.cache.delete(this.cache.keys().next().value); }
    return im.complete && im.naturalWidth ? im : null;
  }
}

// ------------------------------------------------------------------------------------------ the GL view
class Cloud {
  constructor(canvas) {
    this.cv = canvas;
    const gl = this.gl = canvas.getContext("webgl2", { antialias: false, alpha: false, premultipliedAlpha: false, preserveDrawingBuffer: true });
    if (!gl) { this.err = "this browser has no WebGL2"; return; }
    const sh = (type, src) => { const s = gl.createShader(type); gl.shaderSource(s, src); gl.compileShader(s);
      if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s)); return s; };
    const p = this.prog = gl.createProgram();
    gl.attachShader(p, sh(gl.VERTEX_SHADER, VS)); gl.attachShader(p, sh(gl.FRAGMENT_SHADER, FS)); gl.linkProgram(p);
    if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p));
    this.vao = gl.createVertexArray();
    this.loc = {}; const n = gl.getProgramParameter(p, gl.ACTIVE_UNIFORMS);
    for (let k = 0; k < n; k++) { const u = gl.getActiveUniform(p, k); this.loc[u.name] = gl.getUniformLocation(p, u.name); }
    this.tex = {}; this.uploaded = {};
    for (const [name, unit] of [["uColA", 0], ["uDepA", 1], ["uColB", 2], ["uDepB", 3], ["uField", 4]]) { this.tex[name] = this.texture(); this.tex[name].unit = unit; this.blank(name); }
    this.permTex = gl.createTexture(); this.permKey = null;
    gl.bindTexture(gl.TEXTURE_2D, this.permTex); this.nearest();
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.R32UI, 1, 1, 0, gl.RED_INTEGER, gl.UNSIGNED_INT, new Uint32Array([0]));
    this.maxPoint = Math.min(MAX_POINT, gl.getParameter(gl.ALIASED_POINT_SIZE_RANGE)[1] || MAX_POINT);
  }
  nearest() { const gl = this.gl; for (const [k, v] of [[gl.TEXTURE_MIN_FILTER, gl.NEAREST], [gl.TEXTURE_MAG_FILTER, gl.NEAREST], [gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE], [gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE]]) gl.texParameteri(gl.TEXTURE_2D, k, v); }
  texture() { const gl = this.gl, t = gl.createTexture(); gl.bindTexture(gl.TEXTURE_2D, t);
    for (const [k, v] of [[gl.TEXTURE_MIN_FILTER, gl.LINEAR], [gl.TEXTURE_MAG_FILTER, gl.LINEAR], [gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE], [gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE]]) gl.texParameteri(gl.TEXTURE_2D, k, v);
    return t; }
  blank(name) { const gl = this.gl; gl.bindTexture(gl.TEXTURE_2D, this.tex[name]); gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 1, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, new Uint8Array([0, 0, 0, 255])); this.uploaded[name] = null; }
  put(name, img, key) {                              // an image into a texture (only when it is another one)
    if (!img || this.uploaded[name] === key) return; const gl = this.gl;
    gl.bindTexture(gl.TEXTURE_2D, this.tex[name]); gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, img); this.uploaded[name] = key;
  }
  perm(data, gw, gh, key) {
    if (this.permKey === key) return; const gl = this.gl;
    gl.bindTexture(gl.TEXTURE_2D, this.permTex); gl.pixelStorei(gl.UNPACK_ALIGNMENT, 4);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.R32UI, gw, gh, 0, gl.RED_INTEGER, gl.UNSIGNED_INT, data); this.permKey = key;
  }
  draw(u, view) {                                    // u = uniforms, view = [x, y, w, h] of the picture in the canvas
    const gl = this.gl, L = this.loc;
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, this.cv.width, this.cv.height);
    const bg = u.blend === 1 ? 1 : 0; gl.clearColor(bg, bg, bg, 1); gl.clearDepth(1); gl.disable(gl.SCISSOR_TEST);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    gl.viewport(view[0], this.cv.height - view[1] - view[3], view[2], view[3]);
    gl.useProgram(this.prog); gl.bindVertexArray(this.vao);
    for (const t of Object.values(this.tex)) { gl.activeTexture(gl.TEXTURE0 + t.unit); gl.bindTexture(gl.TEXTURE_2D, t); }
    gl.activeTexture(gl.TEXTURE5); gl.bindTexture(gl.TEXTURE_2D, this.permTex);
    for (const [name, t] of Object.entries(this.tex)) gl.uniform1i(L[name], t.unit);
    gl.uniform1i(L.uPerm, 5);
    for (const k of ["uGW", "uGH", "uColour", "uHasField", "uMorphOn", "uFlip"]) gl.uniform1i(L[k], u[k] | 0);
    gl.uniform1i(L.uBlend, u.blend);
    for (const k of ["uJitter", "uDepth", "uFarCut", "uSize", "uOpacity", "uT", "uDrift", "uBreathe", "uWind", "uPush", "uScan", "uS", "uSwarm", "uAspect", "uOutAspect", "uH"]) gl.uniform1f(L[k], u[k]);
    gl.uniform1f(L.uMaxPoint, this.maxPoint);
    for (const k of ["uTintA", "uTintB", "uCamC", "uCamR", "uCamU", "uCamF"]) gl.uniform3fv(L[k], u[k]);
    if (u.blend === 0) { gl.enable(gl.BLEND); gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_COLOR); gl.disable(gl.DEPTH_TEST); }
    else if (u.blend === 1) { gl.enable(gl.BLEND); gl.blendFunc(gl.ZERO, gl.SRC_COLOR); gl.disable(gl.DEPTH_TEST); }
    else { gl.disable(gl.BLEND); gl.enable(gl.DEPTH_TEST); gl.depthFunc(gl.LESS); }
    gl.drawArrays(gl.POINTS, 0, u.uGW * u.uGH);
    gl.bindVertexArray(null);
  }
}

// brightness of every point of a picture (for the morph): the picture at grid size, each point's pixel
function pointLum(img, gw, gh, jitter) {
  const c = document.createElement("canvas"); c.width = img.naturalWidth; c.height = img.naturalHeight;
  const g = c.getContext("2d", { willReadFrequently: true }); g.drawImage(img, 0, 0);
  const px = g.getImageData(0, 0, c.width, c.height).data, out = new Float32Array(gw * gh);
  for (let i = 0; i < out.length; i++) {
    const [u, v] = gridUV(i, gw, gh, jitter), x = Math.min(c.width - 1, Math.max(0, Math.floor(u * c.width))), y = Math.min(c.height - 1, Math.max(0, Math.floor(v * c.height))), o = (y * c.width + x) * 4;
    out[i] = (px[o] * LUMA[0] + px[o + 1] * LUMA[1] + px[o + 2] * LUMA[2]) / 255;
  }
  return out;
}

// ------------------------------------------------------------------------------------------ the node
function cloudUI(node) {
  style();
  const el = document.createElement("div"); el.className = "kkp";
  el.innerHTML = `<div class="kkp-stage"><canvas data-gl></canvas><canvas data-ov></canvas><div class="msg">run the node once (queue) - then the cloud plays here: drag to look around, shift-drag to move, wheel to go nearer</div></div>
<div class="kkp-row"><button data-play>play</button><input type="range" data-scrub min="0" max="0" step="1" value="0" style="max-width:none"><span class="time"></span></div>
<div class="kkp-row" data-camrow></div>
<div class="kkp-hint"></div>
<div class="kkp-panel"></div>
<div class="kkp-row"><span style="flex:1"></span><button data-render class="act" title="Render the cloud at full size, and whatever saves it after this node (Create Video, Save ...); nothing else in the workflow runs">render</button></div>`;
  stop(el);
  const stage = el.querySelector(".kkp-stage"), cvGL = el.querySelector("[data-gl]"), cvOv = el.querySelector("[data-ov]"), msg = el.querySelector(".msg");
  const hint = el.querySelector(".kkp-hint"), panel = el.querySelector(".kkp-panel"), camrow = el.querySelector("[data-camrow]");
  const scrub = el.querySelector("[data-scrub]"), timeEl = el.querySelector(".time"), playBtn = el.querySelector("[data-play]");
  const ov = cvOv.getContext("2d");
  const W_ = n => node.widgets?.find(w => w.name === n);
  const st = { meta: null, def: null, f: 0, playing: false, free: null, follow: true, tracks: null, perms: new Map(), view: [0, 0, 1, 1] };
  let cloud = null;
  try { cloud = new Cloud(cvGL); if (cloud.err) { msg.textContent = cloud.err; cloud = null; } } catch (e) { msg.textContent = "WebGL2: " + e.message; cloud = null; }
  const frames = new Frames(() => paint());
  const getLook = () => merged(W_("look")?.value || "", st.def);
  function setLook(fn) {
    const lk = getLook(); fn(lk); const w = W_("look"); if (w) { w.value = JSON.stringify(lk); w.callback?.(w.value); }
    node.setDirtyCanvas?.(true, true); buildPanel(); paint();
  }
  const setPath = (path, val) => setLook(lk => { const ks = path.split("."); let o = lk; for (const k of ks.slice(0, -1)) o = o[k] ||= {}; o[ks[ks.length - 1]] = val; });

  // ---- time and camera
  const nOut = () => st.meta?.n_out || 1, fps = () => st.meta?.fps || 24;
  const camNow = lk => st.free && !st.follow ? st.free : cameraAt(lk.camera, nOut() > 1 ? st.f / (nOut() - 1) : 0);
  function layout() {
    const d = Math.min(devicePixelRatio || 1, 2), w = Math.max(10, Math.round(stage.clientWidth * d)), h = Math.max(10, Math.round(stage.clientHeight * d));
    for (const c of [cvGL, cvOv]) if (c.width !== w || c.height !== h) { c.width = w; c.height = h; }
    const asp = st.meta ? st.meta.W / st.meta.H : 16 / 9;
    let vw = w, vh = Math.round(w / asp); if (vh > h) { vh = h; vw = Math.round(h * asp); }
    st.view = [Math.round((w - vw) / 2), Math.round((h - vh) / 2), vw, vh];
  }
  function paint() {
    if (!el.isConnected) return;
    layout(); ov.clearRect(0, 0, cvOv.width, cvOv.height);
    const m = st.meta; msg.style.display = m && cloud ? "none" : "flex";
    timeEl.textContent = `${(st.f / fps()).toFixed(2)} / ${(nOut() / fps()).toFixed(2)} s`;
    if (!m || !cloud) return;
    const lk = getLook(), t = st.f / fps(), [gw, gh] = gridSize(lk.density, m.W, m.H);
    let src = 0, k = 0, s = 0;
    if (m.mode === "video") src = st.f % m.n; else [k, s] = morphAt(t, m.T, m.K);
    const a = m.mode === "video" ? src : k;
    const ca = frames.img(m.colour, a), da = frames.img(m.depth, a);
    if (!ca || !da) { hint.textContent = "loading the frames…"; return; }
    cloud.put("uColA", ca, m.colour.subfolder + a); cloud.put("uDepA", da, m.depth.subfolder + a);
    let morphOn = 0;
    if (s > 0) {
      const cb = frames.img(m.colour, k + 1), db = frames.img(m.depth, k + 1);
      if (cb && db) {
        const key = `${m.colour.subfolder}|${k}|${gw}|${gh}|${lk.jitter}`;
        let perm = st.perms.get(key);
        if (!perm) { perm = rankPerm(pointLum(ca, gw, gh, lk.jitter), pointLum(cb, gw, gh, lk.jitter)); st.perms.set(key, perm); if (st.perms.size > 6) st.perms.delete(st.perms.keys().next().value); }
        cloud.put("uColB", cb, m.colour.subfolder + (k + 1)); cloud.put("uDepB", db, m.depth.subfolder + (k + 1)); cloud.perm(perm, gw, gh, key); morphOn = 1;
      }
    }
    let hasField = 0;
    if (m.field) { const fi = frames.img(m.field, Math.min(st.f, m.field.n - 1)); if (fi) { cloud.put("uField", fi, m.field.subfolder + Math.min(st.f, m.field.n - 1)); hasField = 1; } }
    if (m.mode === "video" && m.n > 1) for (let j = 1; j <= 8; j++) { frames.img(m.colour, (src + j) % m.n); frames.img(m.depth, (src + j) % m.n); }
    const cam = camNow(lk), B = basis(cam, lk.depth), vw = st.view[2], vh = st.view[3];
    const flip = m.depth_from === "node" && lk.near !== m.near ? 1 : 0;
    cloud.draw({ uGW: gw, uGH: gh, uColour: ["true", "mono", "tint", "negative"].indexOf(lk.colour), uHasField: hasField, uMorphOn: morphOn, uFlip: flip,
      blend: ["glow", "ink", "solid"].indexOf(lk.blend), uJitter: lk.jitter, uDepth: lk.depth, uFarCut: lk.far_cut, uSize: lk.size, uOpacity: lk.opacity,
      uT: t * lk.speed, uDrift: lk.drift, uBreathe: lk.breathe, uWind: lk.wind, uPush: lk.push, uScan: lk.scan, uS: morphOn ? s : 0, uSwarm: lk.swarm,
      uAspect: m.W / m.H, uOutAspect: vw / vh, uH: vh, uTintA: hex01(lk.tint[0]), uTintB: hex01(lk.tint[1]),
      uCamC: B[0], uCamR: B[1], uCamU: B[2], uCamF: B[3] }, st.view);
    if (st.tracks?.frames?.length) drawTracks(lk, B, m.mode === "pictures" ? st.f % st.tracks.frames.length : Math.min(st.f % m.n, st.tracks.frames.length - 1));
    hint.textContent = `${(gw * gh).toLocaleString()} points · ${m.mode === "video" ? `video, ${m.n} frames` : m.K > 1 ? `${m.K} pictures that morph` : "a picture"}`
      + (m.depth_from === "light" ? " · no depth connected: a relief from the brightness" : m.depth_from === "sample" ? " · the sample facade" : "")
      + (st.follow ? " · the camera move" : " · your view (start = view / end = view keep it)");
  }
  function drawTracks(lk, B, tf) {
    const pts = st.tracks.frames[tf] || [], m = st.meta, [vx, vy, vw, vh] = st.view, tl = lk.tracks, asp = m.W / m.H;
    if (!pts.length) return;
    const xy = pts.map(p => { const q = project(trackWorld(p, asp, lk.depth), B, vw, vh); return q[2] > 0.05 ? [vx + q[0], vy + q[1]] : null; });
    const unit = vh / 1080, lw = Math.max(1, Math.round(1.2 * unit));
    ov.save(); ov.globalAlpha = tl.opacity; ov.strokeStyle = tl.colour; ov.lineWidth = lw;
    ov.beginPath(); ov.rect(vx, vy, vw, vh); ov.clip();
    ov.beginPath();
    for (const [a, b] of trackEdges(pts, tl, asp, delaunayEdges, mstEdges)) if (xy[a] && xy[b]) { ov.moveTo(...xy[a]); ov.lineTo(...xy[b]); }
    ov.stroke();
    if (tl.marks) { ov.beginPath(); for (const q of xy) if (q) { ov.moveTo(q[0] + RING * tl.size * unit, q[1]); ov.arc(q[0], q[1], RING * tl.size * unit, 0, Math.PI * 2); } ov.stroke(); }
    ov.restore();
  }
  function play(on = !st.playing) {
    st.playing = on && nOut() > 1; playBtn.textContent = st.playing ? "pause" : "play"; playBtn.setAttribute("aria-pressed", st.playing);
    if (!st.playing) return;
    let prev = performance.now(), acc = 0;
    const step = now => { if (!st.playing || !el.isConnected) return; acc += (now - prev) / 1000 * fps(); prev = now;
      if (acc >= 1) { st.f = (st.f + Math.floor(acc)) % nOut(); acc %= 1; scrub.value = st.f; paint(); }
      requestAnimationFrame(step); };
    requestAnimationFrame(step);
  }
  playBtn.addEventListener("click", () => play());
  scrub.addEventListener("input", () => { st.f = +scrub.value; paint(); });

  // ---- looking around
  let drag = null;
  cvOv.addEventListener("pointerdown", e => { if (!st.meta) return; e.preventDefault(); e.stopPropagation(); cvOv.setPointerCapture(e.pointerId);
    const lk = getLook(); if (st.follow) { st.free = { ...camNow(lk) }; st.follow = false; buildCam(); }
    drag = { x: e.clientX, y: e.clientY, cam: { ...st.free }, pan: e.shiftKey || e.button === 1 || e.button === 2 }; stage.style.cursor = "grabbing"; });
  cvOv.addEventListener("pointermove", e => { if (!drag) return; e.stopPropagation(); const dx = e.clientX - drag.x, dy = e.clientY - drag.y, h = Math.max(50, stage.clientHeight);
    if (drag.pan) { st.free.x = drag.cam.x - dx / h; st.free.y = drag.cam.y + dy / h; }
    else { st.free.yaw = Math.max(-180, Math.min(180, drag.cam.yaw + dx * 0.25)); st.free.pitch = Math.max(-89, Math.min(89, drag.cam.pitch - dy * 0.25)); }
    paint(); });
  const end = e => { if (!drag) return; drag = null; stage.style.cursor = ""; try { cvOv.releasePointerCapture(e.pointerId); } catch { /* gone */ } };
  cvOv.addEventListener("pointerup", end); cvOv.addEventListener("pointercancel", end);
  cvOv.addEventListener("contextmenu", e => e.preventDefault());
  cvOv.addEventListener("wheel", e => { if (!st.meta) return; e.preventDefault(); e.stopPropagation();
    const lk = getLook(); if (st.follow) { st.free = { ...camNow(lk) }; st.follow = false; buildCam(); }
    st.free.dolly = Math.max(-3, Math.min(0.95, st.free.dolly - e.deltaY * 0.0008)); paint(); }, { passive: false });

  // ---- the controls
  const round = v => Math.round(v * 1000) / 1000;
  function buildCam() {
    const lk = getLook();
    camrow.innerHTML = `<span class="k">camera</span><button data-cam="follow" aria-pressed="${st.follow}" title="The camera goes from start to end over the render">see the move</button>`
      + `<button data-cam="start" title="The move starts from the view you see">start = view</button><button data-cam="end" title="The move ends at the view you see">end = view</button>`
      + `<button data-cam="reset" title="Back to the picture's own view">picture view</button>`
      + ["smooth", "linear", "still"].map(o => `<button data-ease="${o}" aria-pressed="${lk.camera.ease === o}">${o}</button>`).join("");
  }
  camrow.addEventListener("click", e => { const b = e.target.closest("button"); if (!b) return; e.stopPropagation(); const lk = getLook();
    if (b.dataset.ease) return setPath("camera.ease", b.dataset.ease);
    const c = b.dataset.cam;
    if (c === "follow") { st.follow = true; st.free = null; }
    else if (c === "reset") { st.free = { yaw: 0, pitch: 0, dolly: 0, x: 0, y: 0 }; st.follow = false; }
    else if (c === "start" || c === "end") { const v = {}; for (const k of CAM_KEYS) v[k] = round(camNow(lk)[k]); setPath(`camera.${c}`, v); st.follow = true; st.free = null; }
    buildCam(); paint(); });
  const pills = (path, label, opts, v) => `<div class="kkp-row"><span class="k">${label}</span>${opts.map(o => `<button data-set="${path}" data-val="${esc(o)}" aria-pressed="${String(v) === String(o)}">${esc(o)}</button>`).join("")}</div>`;
  const range = (path, label, min, max, step, v, tip = "") => `<div class="kkp-row" title="${esc(tip)}"><span class="k">${label}</span><input type="range" data-in="${path}" min="${min}" max="${max}" step="${step}" value="${v}"><span class="v">${v}</span></div>`;
  function buildPanel() {
    const lk = getLook(), m = st.meta;
    let h = pills("colour", "colour", ["true", "mono", "tint", "negative"], lk.colour);
    if (lk.colour === "tint") h += `<div class="kkp-row"><span class="k">tint</span><input type="color" data-in="tint.0" value="${lk.tint[0]}"> dark → bright <input type="color" data-in="tint.1" value="${lk.tint[1]}"></div>`;
    h += pills("blend", "blend", ["glow", "ink", "solid"], lk.blend);
    if (m?.depth_from === "node") h += pills("near", "near is", ["bright", "dark"], lk.near);
    h += `<h4>points</h4>` + range("density", "points", 256, 2048, 64, lk.density, "Points across the long side (2048 = up to 2 million)")
      + range("size", "size", 0.2, 6, 0.05, lk.size, "In grid cells: 1 = they just touch from the picture's view")
      + range("opacity", "opacity", 0.02, 1, 0.01, lk.opacity) + range("jitter", "jitter", 0, 1, 0.05, lk.jitter, "0 = an exact grid")
      + range("depth", "depth", 0, 4, 0.05, lk.depth, "How deep the picture goes") + range("far_cut", "far cut", 0, 1, 0.01, lk.far_cut, "Leaves out what is farther (the sky, the background)");
    h += `<h4>motion</h4>` + range("speed", "speed", 0, 4, 0.05, lk.speed) + range("drift", "drift", 0, 2, 0.01, lk.drift)
      + range("breathe", "breathe", 0, 2, 0.01, lk.breathe) + range("wind", "wind", 0, 2, 0.01, lk.wind)
      + range("scan", "scan", 0, 2, 0.01, lk.scan, "A bright layer that goes through the depth, near to far");
    if (m?.field) h += range("push", "push", 0, 3, 0.01, lk.push, "How far the field pushes the points out");
    if (m?.K > 1) h += range("swarm", "swarm", 0, 2, 0.01, lk.swarm, "How far the points fly out on their way to the next picture");
    if (st.tracks) {
      const tl = lk.tracks;
      h += `<h4>tracked things</h4><div class="kkp-row"><span class="k">rings</span><button data-tog="tracks.marks" aria-pressed="${!!tl.marks}">${tl.marks ? "on" : "off"}</button></div>`
        + pills("tracks.lines", "lines", ["off", "within", "network", "tree"], tl.lines)
        + (tl.lines === "within" ? range("tracks.distance", "closer than", 0.02, 1, 0.01, tl.distance) : "")
        + `<div class="kkp-row"><span class="k">colour</span>${PALETTE.map(c => `<button class="sw" data-set="tracks.colour" data-val="${c}" style="background:${c}" aria-pressed="${tl.colour === c}"></button>`).join("")}<input type="color" data-in="tracks.colour" value="${tl.colour}"></div>`
        + range("tracks.opacity", "opacity", 0, 1, 0.01, tl.opacity) + range("tracks.size", "ring size", 0.2, 5, 0.05, tl.size);
    }
    panel.innerHTML = h;
  }
  const valOf = (path, raw) => { const num = parseFloat(raw); return path === "density" ? Math.round(num) : Number.isFinite(num) && !String(raw).startsWith("#") ? num : raw; };
  panel.addEventListener("click", e => { const b = e.target.closest("button"); if (!b) return; e.stopPropagation();
    if (b.dataset.set) setPath(b.dataset.set, b.dataset.val);
    else if (b.dataset.tog) setLook(lk => { const ks = b.dataset.tog.split("."); let o = lk; for (const k of ks.slice(0, -1)) o = o[k]; o[ks[ks.length - 1]] = !o[ks[ks.length - 1]]; }); });
  panel.addEventListener("input", e => { const i = e.target; if (!i.dataset.in) return; const path = i.dataset.in;
    if (path.startsWith("tint.")) { setLook(lk => { lk.tint[+path.slice(5)] = i.value; }); return; }
    const v = valOf(path, i.value); const span = i.parentElement.querySelector(".v"); if (span) span.textContent = v;
    const lk = getLook(); const ks = path.split("."); let o = lk; for (const k of ks.slice(0, -1)) o = o[k]; o[ks[ks.length - 1]] = v;
    const w = W_("look"); if (w) w.value = JSON.stringify(lk); paint(); });
  panel.addEventListener("change", e => { if (e.target.dataset.in) { const w = W_("look"); w?.callback?.(w.value); node.setDirtyCanvas?.(true, true); } });

  el.querySelector("[data-render]").addEventListener("click", e => { e.stopPropagation();   // this node + the output nodes after it
    const g = node.graph, ids = new Set([String(node.id)]), seen = new Set([node.id]);
    const link = id => g.getLink?.(id) ?? g.links?.get?.(id) ?? g.links?.[id];
    const walk = n => { for (const o of n.outputs || []) for (const id of o.links || []) { const l = link(id), t = l && g.getNodeById(l.target_id);
      if (!t || seen.has(t.id)) continue; seen.add(t.id); if (t.constructor?.nodeData?.output_node) ids.add(String(t.id)); walk(t); } };
    walk(node); hint.textContent = `rendering${ids.size > 1 ? ` and ${ids.size - 1} node(s) after it` : ""}…`;
    app.queuePrompt(0, 1, [...ids]); });

  el.kkpLoad = async () => {
    st.def = await defaults();
    const m = node.properties?.kubakub_pointcloud || null;
    const fresh = JSON.stringify(m) !== JSON.stringify(st.meta);
    st.meta = m;
    if (fresh) {
      st.perms.clear(); st.tracks = null;
      if (m?.tracks) try { st.tracks = await (await fetch(viewURL(m.tracks))).json(); } catch { st.tracks = null; }
      scrub.max = Math.max(0, (m?.n_out || 1) - 1); st.f = Math.min(st.f, +scrub.max); scrub.value = st.f;
    }
    buildCam(); buildPanel(); paint();
  };
  new ResizeObserver(() => paint()).observe(stage);
  el.kkpLoad();
  return el;
}

app.registerExtension({
  name: "kubakub.pointcloud",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== "KUBA_PointCloud") return;
    const created = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = created?.apply(this, arguments);
      hideWidget(this.widgets?.find(w => w.name === "look"));
      const el = cloudUI(this);
      const w = this.addDOMWidget("cloud_view", "kubakub_pointcloud", el, { serialize: false, hideOnZoom: false, getMinHeight: () => 620 });
      w.serialize = false; this.kkpView = el;
      if ((this.size?.[1] || 0) < 900) this.setSize?.([Math.max(this.size?.[0] || 0, 520), 900]);
      return r;
    };
    const configured = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function () {
      const r = configured?.apply(this, arguments);
      hideWidget(this.widgets?.find(w => w.name === "look"));
      requestAnimationFrame(() => this.kkpView?.kkpLoad?.());
      return r;
    };
    const executed = nodeType.prototype.onExecuted;
    nodeType.prototype.onExecuted = function (msg) {
      const r = executed?.apply(this, arguments);
      const m = msg?.kubakub_pointcloud?.[0]; if (m) { (this.properties ||= {}).kubakub_pointcloud = m; this.kkpView?.kkpLoad?.(); }
      return r;
    };
  },
});
