// kubakub compare: two pictures or videos on the node, always on the same frame.
// Modes: swipe (drag the line), side by side, difference (what changed lights up; drag = brighter), blink, onion
// (drag = how much of B). Play / scrub / step; a still against a video holds. The corner button opens the big
// window: wheel = zoom at the mouse, drag = pan (swipe: drag the line), double click = fit, Space = play,
// , / . = a frame, 1-5 = the modes, Esc = close. The frames are the JPEGs the node wrote to the temp folder
// (nodes/utils/nodes_compare.py); mode and preview size live in the node's hidden widgets.
// The view is exported for the director's compare (one way of comparing, both places).
// Plain ES module, no build step, no network. Look: kubakub.art (Hanken Grotesk, #f18a58 actions, #a187b7 selection).
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const HERE = new URL(".", import.meta.url).href;
const ORANGE = "#f18a58", LAV = "#a187b7";
export const MODES = ["swipe", "side by side", "difference", "blink", "onion"];
const PREVIEWS = ["1280", "1920", "full"];
const KEEP = 360, AHEAD = 24;                        // frames kept per side, frames loaded ahead while playing

let styled = false;
export function compareStyle() {
  if (styled) return; styled = true;
  const s = document.createElement("style");
  s.textContent = `
@font-face{font-family:"Hanken Grotesk KKD";src:url("${HERE}fonts/HankenGrotesk-latin-var.woff2") format("woff2");font-weight:200 800;font-display:swap}
.kkq{display:flex;flex-direction:column;gap:5px;width:100%;height:100%;font-family:"Hanken Grotesk KKD",system-ui,sans-serif;font-size:11px;color:rgba(255,255,255,.85);text-transform:lowercase;box-sizing:border-box}
.kkq-stage{flex:1;min-height:0;position:relative;background:#101010;border-radius:6px;overflow:hidden}
.kkq canvas{position:absolute;inset:0;width:100%;height:100%;display:block;touch-action:none;outline:none;cursor:ew-resize}
.kkq-row{display:flex;flex-wrap:wrap;gap:3px;align-items:center}
.kkq-row input[type=range]{flex:1;min-width:60px;accent-color:${ORANGE}}
.kkq-row .kkq-t{min-width:64px;text-align:right;color:rgba(255,255,255,.6);font-variant-numeric:tabular-nums}
.kkq button{font:inherit;text-transform:lowercase;border:1px solid rgba(255,255,255,.14);background:#1d1b20;color:rgba(255,255,255,.8);border-radius:999px;padding:1px 8px;cursor:pointer;line-height:16px}
.kkq button:hover{border-color:${ORANGE};color:#fff}
.kkq button[aria-pressed=true]{background:${LAV};border-color:${LAV};color:#141414}
.kkq button.act{border-color:${ORANGE};color:${ORANGE}}
.kkq-big{position:fixed;inset:0;z-index:10000;background:rgba(8,8,10,.96);padding:14px;box-sizing:border-box}
.kkq-big .kkq{height:100%}
.kkq-hint{color:rgba(255,255,255,.45)}`;
  document.head.appendChild(s);
}

const frameURL = (side, i) => api.apiURL(`/view?filename=${String(i).padStart(5, "0")}.jpg&subfolder=${encodeURIComponent(side.subfolder)}&type=${encodeURIComponent(side.type || "temp")}`);

// The view. src() -> {a, b, fps} with a / b = {n, w, h, subfolder} (frames by URL) or {canvas} (live, the director).
// opts: {big, mode, setMode, preview, setPreview, labels, onClose}
export class CompareView {
  constructor(opts) {
    compareStyle();
    this.o = opts; this.t = 0; this.playing = false; this.split = 0.5; this.gain = 1; this.onion = 0.5;
    this.zoom = 1; this.panX = 0; this.panY = 0; this.cache = { a: new Map(), b: new Map() }; this.last = { a: null, b: null };
    const el = this.el = document.createElement("div");
    el.className = "kkq";
    el.innerHTML = `<div class="kkq-stage"><canvas tabindex="0"></canvas></div>
      <div class="kkq-row" data-row="modes"></div>
      <div class="kkq-row" data-row="time"><button data-play class="act">play</button><input type="range" min="0" max="0" step="1" value="0"><span class="kkq-t"></span></div>
      <div class="kkq-hint"></div>`;
    this.cv = el.querySelector("canvas"); this.ctx = this.cv.getContext("2d"); this.stage = el.querySelector(".kkq-stage");
    this.range = el.querySelector("input[type=range]"); this.tlabel = el.querySelector(".kkq-t"); this.hint = el.querySelector(".kkq-hint");
    this.tmp = document.createElement("canvas");
    new ResizeObserver(() => { const d = this.dpr(); this.cv.width = Math.max(10, this.stage.clientWidth * d); this.cv.height = Math.max(10, this.stage.clientHeight * d); this.draw(); }).observe(this.stage);
    this.bind(); this.renderModes();
  }
  dpr() { return Math.min(devicePixelRatio || 1, 2); }
  mode() { return this.o.mode?.() || "swipe"; }
  src() { return this.o.src() || {}; }
  length() { const s = this.src(); return Math.max(s.a?.n || (s.a ? 1 : 0), s.b?.n || (s.b ? 1 : 0), 1); }
  frame(side, tag) {                                // the picture of a side at this.t (a still holds), or the last one shown
    if (!side) return null;
    if (side.canvas) return side.canvas;
    const i = Math.min(this.t, side.n - 1), key = side.subfolder + "|" + i, c = this.cache[tag];
    let im = c.get(key);
    if (!im) { im = this.load(side, i, tag); }
    if (im.complete && im.naturalWidth) { this.last[tag] = im; return im; }
    return this.last[tag]?.kkqSub === side.subfolder ? this.last[tag] : null;
  }
  load(side, i, tag) {
    const c = this.cache[tag], key = side.subfolder + "|" + i;
    if (c.has(key)) return c.get(key);
    const im = new Image(); im.kkqSub = side.subfolder; im.decoding = "async";
    im.onload = () => this.draw(); im.src = frameURL(side, i); c.set(key, im);
    if (c.size > KEEP) c.delete(c.keys().next().value);
    return im;
  }
  ready() {                                          // both frames of this.t are there (playing waits instead of skipping)
    const s = this.src();
    return ["a", "b"].every(tag => { const d = s[tag]; if (!d || d.canvas) return true; const im = this.cache[tag].get(d.subfolder + "|" + Math.min(this.t, d.n - 1)); return im?.complete && im.naturalWidth; });
  }
  preload() {
    const s = this.src();
    for (const tag of ["a", "b"]) { const d = s[tag]; if (!d || d.canvas) continue; for (let k = 1; k <= AHEAD; k++) { const i = this.t + k; if (i < d.n) this.load(d, i, tag); } }
  }
  setT(t) { const n = this.length(); this.t = Math.max(0, Math.min(n - 1, Math.round(t))); this.preload(); this.draw(); }
  togglePlay(on = !this.playing) {
    if (this.length() < 2) on = false;
    this.playing = on; this.el.querySelector("[data-play]").textContent = on ? "pause" : "play";
    if (!on) return;
    let prev = performance.now(), acc = 0;
    const step = now => {
      if (!this.playing || !this.el.isConnected) { this.playing = false; return; }
      const fps = this.src().fps || 25; acc += (now - prev) / 1000 * fps; prev = now;
      if (acc >= 1 && this.ready()) { const n = this.length(); this.t = (this.t + Math.floor(acc)) % n; acc %= 1; this.preload(); this.draw(); }
      else if (acc > 3) acc = 1;                    // still loading: wait, do not run away
      requestAnimationFrame(step);
    };
    requestAnimationFrame(step);
  }
  fitRect() {                                        // where A sits on the canvas (zoom and pan applied)
    const s = this.src(), ref = s.a || s.b, w = ref?.w || ref?.canvas?.width || 16, h = ref?.h || ref?.canvas?.height || 9;
    const cw = this.cv.width, ch = this.cv.height, k = Math.min(cw / w, ch / h) * 0.98 * this.zoom;
    return { x: (cw - w * k) / 2 + this.panX, y: (ch - h * k) / 2 + this.panY, w: w * k, h: h * k, k };
  }
  draw() {
    if (this._raf) return;
    this._raf = requestAnimationFrame(() => { this._raf = 0; this.drawNow(); });
  }
  drawNow() {
    const ctx = this.ctx, D = this.dpr(), cw = this.cv.width, ch = this.cv.height, s = this.src(), mode = this.mode();
    ctx.setTransform(1, 0, 0, 1, 0, 0); ctx.globalCompositeOperation = "source-over"; ctx.globalAlpha = 1; ctx.filter = "none";
    ctx.fillStyle = "#101010"; ctx.fillRect(0, 0, cw, ch);
    this.range.max = String(this.length() - 1); this.range.value = String(this.t);
    this.el.querySelector("[data-row=time]").style.display = this.length() > 1 ? "" : "none";
    this.tlabel.textContent = `${this.t + 1} / ${this.length()}`;
    if (!s.a && !s.b) {
      ctx.fillStyle = "rgba(255,255,255,.5)"; ctx.font = `${13 * D}px "Hanken Grotesk KKD", sans-serif`; ctx.textAlign = "center";
      ctx.fillText(this.o.empty || "connect a and b, then run", cw / 2, ch / 2); ctx.textAlign = "left"; this.hint.textContent = ""; return;
    }
    const A = this.frame(s.a, "a"), B = this.frame(s.b, "b"), R = this.fitRect();
    ctx.imageSmoothingQuality = "high";
    const contain = (im, r) => {                     // a picture inside r, its own shape kept, centred
      const w = im.naturalWidth || im.width, h = im.naturalHeight || im.height, k = Math.min(r.w / w, r.h / h);
      return { x: r.x + (r.w - w * k) / 2, y: r.y + (r.h - h * k) / 2, w: w * k, h: h * k };
    };
    const put = (im, r) => { if (im) { const q = contain(im, r); ctx.drawImage(im, q.x, q.y, q.w, q.h); } };
    const label = (t, x, y, right) => { ctx.font = `700 ${12 * D}px "Hanken Grotesk KKD", sans-serif`; const w = ctx.measureText(t).width + 12 * D;
      const X = right ? x - w : x; ctx.fillStyle = "rgba(20,18,24,.8)"; ctx.fillRect(X, y, w, 20 * D); ctx.fillStyle = "#fff"; ctx.fillText(t, X + 6 * D, y + 14 * D); };
    const la = this.o.labels?.[0] || "a", lb = this.o.labels?.[1] || "b";
    if (mode === "side by side") {                    // each half fits its own picture; zoom and pan act on both alike
      const half = (x0, im) => { if (!im) return; const w = cw / 2 * 0.98 * this.zoom, h = ch * 0.98 * this.zoom;
        ctx.save(); ctx.beginPath(); ctx.rect(x0, 0, cw / 2, ch); ctx.clip();
        put(im, { x: x0 + (cw / 2 - w) / 2 + this.panX, y: (ch - h) / 2 + this.panY, w, h }); ctx.restore(); };
      half(0, A); half(cw / 2, B);
      ctx.fillStyle = "rgba(255,255,255,.25)"; ctx.fillRect(cw / 2 - D / 2, 0, D, ch);
      label(la, 8 * D, 8 * D); label(lb, cw / 2 + 8 * D, 8 * D);
    } else if (mode === "difference") {
      const t = this.tmp; if (t.width !== Math.round(R.w) || t.height !== Math.round(R.h)) { t.width = Math.max(1, Math.round(R.w)); t.height = Math.max(1, Math.round(R.h)); }
      const g = t.getContext("2d"); g.globalCompositeOperation = "source-over"; g.fillStyle = "#000"; g.fillRect(0, 0, t.width, t.height);
      const into = (im) => { const q = contain(im, { x: 0, y: 0, w: t.width, h: t.height }); g.drawImage(im, q.x, q.y, q.w, q.h); };
      if (A) into(A); g.globalCompositeOperation = "difference"; if (B) into(B);
      ctx.filter = this.gain > 1.01 ? `brightness(${this.gain})` : "none"; ctx.drawImage(t, R.x, R.y, R.w, R.h); ctx.filter = "none";
      label(`difference ×${Math.round(this.gain * 10) / 10}`, 8 * D, 8 * D);
    } else if (mode === "blink") {
      const showB = Math.floor(performance.now() / 500) % 2 === 1;
      put(showB ? B : A, R); label(showB ? lb : la, 8 * D, 8 * D);
      if (!this._blink) { this._blink = setInterval(() => { if (!this.el.isConnected || this.mode() !== "blink") { clearInterval(this._blink); this._blink = null; return; } this.draw(); }, 250); }
    } else if (mode === "onion") {
      put(A, R); ctx.globalAlpha = this.onion; put(B, R); ctx.globalAlpha = 1;
      label(`${lb} ${Math.round(this.onion * 100)} %`, 8 * D, 8 * D);
    } else {                                         // swipe
      const X = R.x + R.w * this.split;
      put(A, R);
      ctx.save(); ctx.beginPath(); ctx.rect(X, 0, cw - X, ch); ctx.clip(); put(B, R); ctx.restore();
      ctx.fillStyle = "#fff"; ctx.fillRect(X - D, R.y, 2 * D, R.h);
      ctx.fillStyle = ORANGE; ctx.beginPath(); ctx.arc(X, R.y + R.h / 2, 7 * D, 0, 7); ctx.fill();
      label(la, Math.max(R.x, 0) + 8 * D, Math.max(R.y, 0) + 8 * D); label(lb, Math.min(R.x + R.w, cw) - 8 * D, Math.max(R.y, 0) + 8 * D, true);
    }
    const warn = [];
    for (const [tag, d] of [["a", s.a], ["b", s.b]]) if (d?.cut) warn.push(`${tag}: first ${d.n} frames`);
    if (s.a && s.b && s.a.w && s.b.w && Math.abs(s.a.w / s.a.h - s.b.w / s.b.h) > 0.01) warn.push(`other shapes: a ${s.a.w}×${s.a.h}, b ${s.b.w}×${s.b.h} (each fitted, not stretched)`);
    this.hint.textContent = [{ swipe: "drag = the line", "side by side": "", difference: "drag sideways = brighter", blink: "", onion: "drag sideways = how much of b" }[mode],
      this.o.big ? "wheel = zoom · drag = pan · double click = fit · esc = close" : "", ...warn].filter(Boolean).join(" · ");
  }
  renderModes() {
    const m = this.mode(), p = this.o.preview?.();
    const row = this.el.querySelector("[data-row=modes]");
    row.innerHTML = MODES.map((n, i) => `<button data-mode="${n}" aria-pressed="${n === m}" title="${{ swipe: "a and b in their own colours, a line between them", "side by side": "a and b next to each other, own colours", difference: "not the colours: only what changed between a and b lights up (black = the same)", blink: "a and b in turn, own colours", onion: "b laid over a, own colours" }[n]}${this.o.big ? " · key " + (i + 1) : ""}">${n}</button>`).join("")
      + (p ? `<span style="flex:1"></span>${PREVIEWS.map(n => `<button data-preview="${n}" aria-pressed="${n === p}" title="the size the frames are shown at (next run)">${n}</button>`).join("")}` : "")
      + (this.o.big ? `<button data-close class="act">close</button>` : this.o.openBig ? `<button data-big class="act" title="the big window: zoom and pan">⤢</button>` : "");
  }
  setMode(n) { this.o.setMode?.(n); this.renderModes(); this.draw(); }
  bind() {
    const cv = this.cv;
    this.el.addEventListener("click", e => {
      const t = e.target.closest("button"); if (!t) return; e.stopPropagation();
      if (t.dataset.mode) this.setMode(t.dataset.mode);
      else if (t.dataset.preview) { this.o.setPreview?.(t.dataset.preview); this.renderModes(); }
      else if ("play" in t.dataset) this.togglePlay();
      else if ("big" in t.dataset) this.o.openBig?.(this);
      else if ("close" in t.dataset) this.o.onClose?.();
    });
    this.range.addEventListener("input", () => { this.togglePlay(false); this.setT(+this.range.value); });
    for (const t of ["pointerdown", "pointermove", "pointerup", "wheel"]) this.range.addEventListener(t, e => e.stopPropagation(), { passive: true });
    const xy = e => { const r = cv.getBoundingClientRect(); return [(e.clientX - r.left) * cv.width / (r.width || 1), (e.clientY - r.top) * cv.height / (r.height || 1)]; };
    let drag = null;
    cv.addEventListener("pointerdown", e => {
      if (e.button !== 0 && e.button !== 1) return;
      e.preventDefault(); e.stopPropagation(); cv.focus({ preventScroll: true });
      const [x, y] = xy(e), R = this.fitRect(), m = this.mode(), lineX = R.x + R.w * this.split;
      const kind = e.button === 1 || this.space ? "pan" : m === "swipe" ? (!this.o.big || Math.abs(x - lineX) < 14 * this.dpr() || this.zoom <= 1.001 ? "line" : "pan")
        : m === "difference" || m === "onion" ? "value" : this.o.big ? "pan" : "";
      drag = { kind, x, y, px: this.panX, py: this.panY, gain: this.gain, onion: this.onion };
      cv.setPointerCapture(e.pointerId); move(e);
    });
    const move = e => {
      if (!drag) return; e.stopPropagation();
      const [x, y] = xy(e), R = this.fitRect();
      if (drag.kind === "line") this.split = Math.max(0, Math.min(1, (x - R.x) / (R.w || 1)));
      else if (drag.kind === "pan") { this.panX = drag.px + x - drag.x; this.panY = drag.py + y - drag.y; }
      else if (drag.kind === "value") {
        const u = (x - drag.x) / Math.max(80, this.cv.width * 0.6);
        if (this.mode() === "difference") this.gain = Math.max(1, Math.min(16, drag.gain * Math.pow(16, u)));
        else this.onion = Math.max(0, Math.min(1, drag.onion + u));
      }
      this.draw();
    };
    cv.addEventListener("pointermove", move);
    const up = e => { if (!drag) return; drag = null; try { cv.releasePointerCapture(e.pointerId); } catch { /* released */ } };
    cv.addEventListener("pointerup", up); cv.addEventListener("pointercancel", up);
    cv.addEventListener("wheel", e => {
      e.stopPropagation();
      if (!this.o.big) return;
      e.preventDefault();
      const [x, y] = xy(e), R = this.fitRect(), f = e.deltaY > 0 ? 1 / 1.15 : 1.15, z = Math.max(1, Math.min(40, this.zoom * f)), k = z / this.zoom;
      const cx = this.cv.width / 2, cy = this.cv.height / 2;           // keep the point under the mouse
      this.panX = (x - cx) - ((x - cx) - this.panX) * k; this.panY = (y - cy) - ((y - cy) - this.panY) * k;
      this.zoom = z; if (z <= 1.0001) { this.panX = this.panY = 0; } void R; this.draw();
    }, { passive: false });
    cv.addEventListener("dblclick", e => { e.preventDefault(); e.stopPropagation(); this.zoom = 1; this.panX = this.panY = 0; this.split = 0.5; this.draw(); });
    cv.addEventListener("keydown", e => this.key(e));
  }
  key(e) {
    const k = e.key;
    if (k === " ") { this.togglePlay(); }
    else if (k === "," || k === ".") { this.togglePlay(false); this.setT(this.t + (k === "." ? 1 : -1)); }
    else if (/^[1-5]$/.test(k) && this.o.big) this.setMode(MODES[+k - 1]);   // on the node a stray number key must not switch the mode
    else if (k === "0") { this.zoom = 1; this.panX = this.panY = 0; this.draw(); }
    else if (k === "Escape" && this.o.big) this.o.onClose?.();
    else if (k === "Home") this.setT(0);
    else return false;
    e.preventDefault(); e.stopPropagation(); return true;
  }
}

function hideWidget(w) {
  if (!w) return;
  w.hidden = true; w.computeSize = () => [0, -4];
  if (w.element) w.element.style.display = "none";
  if (w.inputEl) w.inputEl.style.display = "none";
}

function openBig(opts, from) {                      // the big window over everything, on the same frame as the node
  const wrap = document.createElement("div"); wrap.className = "kkq-big";
  let view;
  const close = () => { view.togglePlay(false); wrap.remove(); window.removeEventListener("keydown", keys, true); from?.setT(view.t); from?.drawNow(); };
  view = new CompareView({ ...opts, big: true, openBig: null, onClose: close });
  view.t = from?.t || 0; view.split = from?.split ?? 0.5; view.gain = from?.gain || 1; view.onion = from?.onion ?? 0.5;
  const keys = e => { if (!wrap.isConnected) return; if (e.target.matches?.("input:not([type=range]),textarea")) return; view.key(e); e.stopPropagation(); };
  window.addEventListener("keydown", keys, true);
  wrap.appendChild(view.el); document.body.appendChild(wrap); view.cv.focus();
  return view;
}
export { openBig as openCompareWindow };

app.registerExtension({
  name: "kubakub.compare",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== "KUBA_Compare") return;
    const created = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = created?.apply(this, arguments);
      const node = this, W_ = n => node.widgets?.find(w => w.name === n);
      for (const n of ["mode", "preview"]) hideWidget(W_(n));
      const set = (n, v) => { const w = W_(n); if (!w || w.value === v) return; w.value = v; w.callback?.(v); node.graph?.setDirtyCanvas(true, true); };
      const title = i => { const up = node.getInputNode?.(i); return up ? String(up.title || up.type || "").toLowerCase().slice(0, 28) : ""; };
      const opts = {
        src: () => node.properties?.kubakub_compare || null,
        mode: () => W_("mode")?.value || "swipe", setMode: v => set("mode", v),
        preview: () => W_("preview")?.value || "1280", setPreview: v => set("preview", v),
        get labels() { const a = title(0), b = title(1); return [a ? "a · " + a : "a", b ? "b · " + b : "b"]; },
        empty: "connect a picture or video to a and b, then run",
      };
      opts.openBig = from => openBig(opts, from);
      const view = new CompareView(opts);
      const w = this.addDOMWidget("compare_view", "kubakub_compare", view.el, { serialize: false, hideOnZoom: false, getMinHeight: () => 320 });
      w.serialize = false;
      this.kkqView = view;
      if ((this.size?.[1] || 0) < 420) this.setSize?.([Math.max(this.size?.[0] || 0, 420), 420]);
      return r;
    };
    const configured = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function () {
      const r = configured?.apply(this, arguments);
      for (const n of ["mode", "preview"]) hideWidget(this.widgets?.find(w => w.name === n));
      requestAnimationFrame(() => { this.kkqView?.renderModes(); this.kkqView?.draw(); });
      return r;
    };
    const executed = nodeType.prototype.onExecuted;
    nodeType.prototype.onExecuted = function (msg) {
      const r = executed?.apply(this, arguments);
      const m = msg?.kubakub_compare?.[0];
      if (m) { (this.properties ||= {}).kubakub_compare = m; const v = this.kkqView; if (v) { v.t = Math.min(v.t, v.length() - 1); v.preload(); v.draw(); } }
      return r;
    };
  },
});
