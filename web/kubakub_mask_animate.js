// kubakub mask animate: the curve of the node drawn on the node itself, with the curve types as buttons.
// It only reads and sets the node's own widgets (curve, cycle, value_from, value_to, easing, duty, phase, seed,
// seconds), so the widgets stay the truth and a workflow without this file still runs.
// Drag left / right on the curve = faster / slower (cycle). The formulas are those of kubakub/maskfields.py shape().
// Plain ES module, no build step, no network. Look: kubakub.art (Hanken Grotesk, #f18a58 actions, #a187b7 selection).
import { app } from "../../scripts/app.js";

const HERE = new URL(".", import.meta.url).href;
const ORANGE = "#f18a58", LAV = "#a187b7";
const CURVES = ["ramp", "saw", "triangle", "sine", "square", "random", "noise",
  "sound level", "beats", "bars", "low hits", "mid hits", "high hits"];
const SOUND = { "beats": 0.5, "bars": 2, "low hits": 0.5, "mid hits": 1, "high hits": 0.5 };   // s between the example triggers (120 bpm)
const isSound = c => c in SOUND || c === "sound level";
const UNIT = { "move x": "widths", "move y": "heights", "rotate": "turns", "scale": "size" };

// ---- the curves (bit-identical to kubakub/director/motion.py hash01 / noise and kubakub/maskfields.py shape)
function hash01(i, seed) {
  let h = (Math.imul(i | 0, 374761393) + Math.imul(seed | 0, 668265263)) >>> 0;
  h = Math.imul(h ^ (h >>> 13), 1274126177) >>> 0;
  h = (h ^ (h >>> 16)) >>> 0;
  return h / 4294967296;
}
function vnoise(x, seed) {
  const i = Math.floor(x), f = x - i, a = hash01(i, seed) * 2 - 1, b = hash01(i + 1, seed) * 2 - 1;
  return a + (b - a) * f * f * (3 - 2 * f);
}
const noise = (x, seed) => vnoise(x, seed) * 0.72 + vnoise(x * 2.03 + 17.1, seed + 7) * 0.28;
function ease(u, kind) {
  u = Math.min(1, Math.max(0, u));
  if (kind === "ease in") return u * u;
  if (kind === "ease out") return 1 - (1 - u) ** 2;
  if (kind === "ease in out") return u * u * (3 - 2 * u);
  return u;
}
export function shape(kind, u, easing, duty, seed) {
  const whole = Math.floor(u), part = u - whole;
  if (kind === "ramp") return ease(u, easing);
  if (kind === "saw") return ease(part, easing);
  if (kind === "triangle") return ease(1 - Math.abs(2 * part - 1), easing);
  if (kind === "sine") return 0.5 - 0.5 * Math.cos(2 * Math.PI * u);
  if (kind === "square") return part < duty ? 1 : 0;
  if (kind === "random") return hash01(whole, seed);
  if (kind === "noise") return Math.min(1, Math.max(0, 0.5 + 0.65 * noise(u, seed)));
  return 0;
}
// 0..1 at t seconds; the sound curves are drawn on an example beat (the real sound decides when the node runs)
function valueAt(s, t) {
  if (s.curve === "sound level") {
    const since = t % 0.5;
    return Math.min(1, s.gain * Math.exp(-since / Math.max(0.01, s.release)));
  }
  if (s.curve in SOUND) {
    const gap = SOUND[s.curve] * s.nth, off = s.curve === "high hits" ? 0.25 : s.curve === "mid hits" ? 0.5 : 0;
    if (t < off) return 0;
    const k = Math.floor((t - off) / gap), run = ease((t - off - k * gap) / s.cycle, s.easing);
    if (s.steps <= 1) return run;
    const pos = (k + run) / s.steps, fr = pos - Math.floor(pos);
    return fr < 1e-9 ? 1 : fr;
  }
  return shape(s.curve, Math.max(0, t / s.cycle + s.phase), s.easing, s.duty, s.seed);
}

let styled = false;
function style() {
  if (styled) return;
  styled = true;
  const el = document.createElement("style");
  el.textContent = `
@font-face{font-family:"Hanken Grotesk KKD";src:url("${HERE}fonts/HankenGrotesk-latin-var.woff2") format("woff2");font-weight:200 800;font-display:swap}
.kkm{display:flex;flex-direction:column;gap:5px;width:100%;height:100%;font-family:"Hanken Grotesk KKD",system-ui,sans-serif;font-size:11px;color:rgba(255,255,255,.85);text-transform:lowercase;box-sizing:border-box}
.kkm canvas{display:block;width:100%;flex:1;min-height:0;background:#101010;border-radius:6px;cursor:ew-resize;touch-action:none}
.kkm-row{display:flex;flex-wrap:wrap;gap:4px}
.kkm button{font:inherit;text-transform:lowercase;border:0;border-radius:10em;padding:2px 9px;background:rgba(255,255,255,.12);color:rgba(255,255,255,.85);cursor:pointer}
.kkm button.on{background:${ORANGE};color:#141414;font-weight:700}
.kkm button.snd.on{background:${LAV}}
.kkm-hint{color:rgba(255,255,255,.5);min-height:14px}`;
  document.head.appendChild(el);
}

function build(node) {
  style();
  const el = document.createElement("div");
  el.className = "kkm";
  const cv = document.createElement("canvas");
  const row = document.createElement("div");
  row.className = "kkm-row";
  const hint = document.createElement("div");
  hint.className = "kkm-hint";
  el.append(cv, row, hint);
  const widget = name => node.widgets?.find(w => w.name === name);
  const val = (name, d) => { const w = widget(name); return w == null || w.value == null ? d : w.value; };
  const set = (name, v) => { const w = widget(name); if (w) { w.value = v; w.callback?.(v); node.graph?.setDirtyCanvas(true, true); } };
  const state = () => ({
    curve: val("curve", "ramp"), effect: val("effect", "reveal"), cycle: Math.max(0.01, +val("cycle", 2)),
    lo: +val("value_from", 0), hi: +val("value_to", 1), easing: val("easing", "linear"), duty: +val("duty", 0.5),
    phase: +val("phase", 0), seed: +val("seed", 1), seconds: +val("seconds", 0), nth: Math.max(1, +val("nth", 1)),
    steps: Math.max(1, +val("steps", 1)), gain: +val("gain", 1), release: +val("release", 0.15) });
  const buttons = CURVES.map(name => {
    const b = document.createElement("button");
    b.textContent = name;
    if (isSound(name)) b.classList.add("snd");
    b.title = isSound(name) ? "follows the sound connected to 'audio'" : "";
    b.addEventListener("click", () => { set("curve", name); draw(); });
    row.appendChild(b);
    return b;
  });

  const t0 = performance.now();
  let shown = "", last = 0;
  function draw() {
    const s = state(), dpr = window.devicePixelRatio || 1;
    const W = Math.max(40, cv.clientWidth), H = Math.max(40, cv.clientHeight);
    if (cv.width !== Math.round(W * dpr) || cv.height !== Math.round(H * dpr)) { cv.width = Math.round(W * dpr); cv.height = Math.round(H * dpr); }
    const g = cv.getContext("2d");
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, W, H);
    const span = s.seconds > 0 ? s.seconds : 4, padX = 6, top = 16, bot = H - 16;
    const X = t => padX + (t / span) * (W - 2 * padX), Y = v => bot - v * (bot - top);
    g.strokeStyle = "rgba(255,255,255,.08)";
    g.lineWidth = 1;
    const sound = isSound(s.curve);
    const grid = sound ? 0.5 : s.cycle;
    if (span / grid <= 64) for (let t = 0; t <= span + 1e-6; t += grid) { g.beginPath(); g.moveTo(X(t), top); g.lineTo(X(t), bot); g.stroke(); }
    g.strokeStyle = "rgba(255,255,255,.2)";
    g.beginPath(); g.moveTo(padX, top); g.lineTo(W - padX, top); g.moveTo(padX, bot); g.lineTo(W - padX, bot); g.stroke();
    g.strokeStyle = sound ? LAV : ORANGE;
    g.lineWidth = 1.6;
    g.beginPath();
    const n = Math.max(2, Math.round((W - 2 * padX) * 2));
    for (let i = 0; i <= n; i++) {
      const t = span * i / n, y = Y(valueAt(s, t));
      if (i) g.lineTo(X(t), y); else g.moveTo(X(t), y);
    }
    g.stroke();
    const now = ((performance.now() - t0) / 1000) % span;
    g.fillStyle = "#fff";
    g.beginPath(); g.arc(X(now), Y(valueAt(s, now)), 3.2, 0, 2 * Math.PI); g.fill();
    g.fillStyle = "rgba(255,255,255,.55)";
    g.font = '10px "Hanken Grotesk KKD",system-ui,sans-serif';
    const unit = UNIT[s.effect] ? " " + UNIT[s.effect] : "";
    g.textAlign = "left";
    g.fillText(`${s.hi}${unit}`, padX + 2, top - 4);
    g.fillText(`${s.lo}${unit}`, padX + 2, H - 4);
    g.textAlign = "right";
    g.fillText(`${Math.round(span * 100) / 100} s`, W - padX - 2, H - 4);
    const sk = s.curve + "|" + s.cycle;              // the buttons and the hint only when they change
    if (sk !== shown) {
      shown = sk;
      buttons.forEach(b => b.classList.toggle("on", b.textContent === s.curve));
      hint.textContent = sound ? "drawn on an example beat (120 bpm); your sound decides when the node runs"
        : `one cycle = ${Math.round(s.cycle * 100) / 100} s. drag left / right on the curve = faster / slower`;
    }
  }

  let drag = null;
  cv.addEventListener("pointerdown", e => { drag = { x: e.clientX, cycle: state().cycle }; cv.setPointerCapture(e.pointerId); e.preventDefault(); });
  cv.addEventListener("pointermove", e => {
    if (!drag) return;
    const c = drag.cycle * Math.exp((e.clientX - drag.x) / 120);
    set("cycle", Math.round(Math.min(3600, Math.max(0.01, c)) * 100) / 100);
  });
  const end = e => { if (drag) { drag = null; try { cv.releasePointerCapture(e.pointerId); } catch { /* already released */ } } };
  cv.addEventListener("pointerup", end);
  cv.addEventListener("pointercancel", end);

  let raf = 0;
  const tick = now => {                              // the moving dot needs no more than 20 pictures a second
    if (now - last >= 50 && el.isConnected && cv.clientWidth) { last = now; draw(); }
    raf = requestAnimationFrame(tick);
  };
  raf = requestAnimationFrame(tick);
  el.kkmStop = () => cancelAnimationFrame(raf);
  return el;
}

app.registerExtension({
  name: "kubakub.maskanimate",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== "KUBA_MaskAnimate") return;
    const created = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = created?.apply(this, arguments);
      const el = build(this);
      const w = this.addDOMWidget("curve_view", "kubakub_mask_curve", el, { serialize: false, hideOnZoom: false,
        getMinHeight: () => 190 });
      w.serialize = false;                               // nothing of its own to save: it shows the widgets
      this.kkmCurve = el;
      return r;
    };
    const executed = nodeType.prototype.onExecuted;
    nodeType.prototype.onExecuted = function (msg) {    // room for the playing masks under the curve, once
      const r = executed?.apply(this, arguments);
      if (msg?.images?.length && !this.kkmGrown) { this.kkmGrown = true; this.setSize([this.size[0], this.size[1] + 240]); }
      return r;
    };
    const removed = nodeType.prototype.onRemoved;
    nodeType.prototype.onRemoved = function () {
      this.kkmCurve?.kkmStop?.();
      return removed?.apply(this, arguments);
    };
  },
});
