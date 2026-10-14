// kubakub scene preview / scene walkthrough: where the audience stands, placed on a top view on the node itself.
// The plan edits the node's own text ("spots": one spot per line; "path": the keys of the walk, in order), so the
// text stays the truth and a workflow without this file still runs. Click the plan = a new spot, drag = move it along
// and away from the wall, wheel over it = eye height, double click or Delete = remove it. After a run the node sends
// the real top view (model, surroundings, projector) as the background; before the first run a plain grid stands in.
// The mapping (origin + x * ax + distance * ad) is the one of the light plan (web/kubakub_light_plan.js).
// Plain ES module, no build step, no network. Look: kubakub.art (Hanken Grotesk, #f18a58 actions, #a187b7 selection).
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const HERE = new URL(".", import.meta.url).href;
const ORANGE = "#f18a58", LAV = "#a187b7";
const SIZE = 768;
const FALLBACK = { origin: [SIZE / 2, SIZE * 0.3], ax: [9.6, 0], ad: [0, 9.6] };   // 80 m across
const NODES = {
  KUBA_ScenePreview: { text: "spots", line: false, word: "spot",
    tip: "click = a new audience spot. drag = move it. wheel over it = eye height. double click = remove" },
  KUBA_SceneWalkthrough: { text: "path", line: true, word: "key",
    tip: "click = the next point of the walk. drag = move it. wheel over it = eye height. double click = remove" },
};
const num = v => String(Math.round(v * 100) / 100);

// ---- the text <-> points. A line is 'x, distance[, eye[, more]]'; what follows the eye height is kept as it is.
function parse(text) {
  const pts = [];
  String(text || "").replace(/;/g, "\n").split("\n").forEach((raw, i) => {
    const v = raw.split("//")[0].trim().replace(/,/g, " ").split(/\s+/).filter(Boolean).map(Number);
    if (v.length >= 2 && v.every(Number.isFinite)) pts.push({ x: v[0], d: v[1], eye: v.length > 2 ? v[2] : 1.7, rest: v.slice(3), line: i });
  });
  return pts;
}
const lineOf = p => [num(p.x), num(p.d), num(p.eye), ...p.rest.map(num)].join(", ");
function write(text, pts) {                              // comments and empty lines stay where they are
  const lines = String(text || "").replace(/;/g, "\n").split("\n"), by = new Map(pts.filter(p => p.line != null).map(p => [p.line, p]));
  const out = [];
  lines.forEach((raw, i) => {
    const v = raw.split("//")[0].trim().replace(/,/g, " ").split(/\s+/).filter(Boolean).map(Number);
    const isPoint = v.length >= 2 && v.every(Number.isFinite);
    if (!isPoint) { out.push(raw); return; }
    const note = raw.includes("//") ? "   //" + raw.split("//").slice(1).join("//") : "";     // a note after a point stays (as in the light plan)
    if (by.has(i)) out.push(lineOf(by.get(i)) + note);  // a removed point's line is dropped
  });
  for (const p of pts) if (p.line == null) out.push(lineOf(p));
  return out.join("\n");
}

let styled = false;
function style() {
  if (styled) return;
  styled = true;
  const s = document.createElement("style");
  s.textContent = `
@font-face{font-family:"Hanken Grotesk KKD";src:url("${HERE}fonts/HankenGrotesk-latin-var.woff2") format("woff2");font-weight:200 800;font-display:swap}
.kks{display:flex;flex-direction:column;gap:5px;width:100%;height:100%;font-family:"Hanken Grotesk KKD",system-ui,sans-serif;font-size:11px;color:rgba(255,255,255,.85);text-transform:lowercase;box-sizing:border-box}
.kks-stage{flex:1;min-height:0;display:flex;align-items:center;justify-content:center;background:#101010;border-radius:6px}
.kks canvas{display:block;cursor:crosshair;touch-action:none}
.kks-hint{color:rgba(255,255,255,.5);min-height:14px}`;
  document.head.appendChild(s);
}

function build(node, cfg) {
  style();
  const text = node.widgets?.find(w => w.name === cfg.text);
  const el = document.createElement("div");
  el.className = "kks";
  el.innerHTML = `<div class="kks-stage"><canvas width="${SIZE}" height="${SIZE}" tabindex="0"></canvas></div><div class="kks-hint"></div>`;
  const cv = el.querySelector("canvas"), ctx = cv.getContext("2d"), hint = el.querySelector(".kks-hint"), stage = el.querySelector(".kks-stage");
  const st = { sel: -1, drag: -1, hover: -1, moved: false, list: null, bg: null, bgKey: "" };
  const meta = () => node.properties?.kubakub_spot_plan || FALLBACK;
  const points = () => st.list || parse(text?.value);
  const toPx = (x, d) => { const m = meta(); return [m.origin[0] + x * m.ax[0] + d * m.ad[0], m.origin[1] + x * m.ax[1] + d * m.ad[1]]; };
  const toM = (px, py) => {
    const m = meta(), u = px - m.origin[0], v = py - m.origin[1], det = m.ax[0] * m.ad[1] - m.ad[0] * m.ax[1] || 1;
    return [(u * m.ad[1] - v * m.ad[0]) / det, (v * m.ax[0] - u * m.ax[1]) / det];
  };
  const evPx = e => { const r = cv.getBoundingClientRect(); return [(e.clientX - r.left) * SIZE / (r.width || 1), (e.clientY - r.top) * SIZE / (r.height || 1)]; };
  const hit = (list, p) => {
    let best = -1, bd = 26;
    list.forEach((q, i) => { const a = toPx(q.x, q.d), d = Math.hypot(a[0] - p[0], a[1] - p[1]); if (d < bd) { bd = d; best = i; } });
    return best;
  };
  const commit = list => {
    if (!text) return;
    text.value = write(text.value, list);
    text.callback?.(text.value);
    if (text.inputEl) text.inputEl.value = text.value;
    node.graph?.setDirtyCanvas(true, true);
  };
  new ResizeObserver(() => {
    const side = Math.max(60, Math.floor(Math.min(stage.clientWidth, stage.clientHeight)));
    cv.style.width = cv.style.height = side + "px";
  }).observe(stage);

  function background() {
    const m = meta();
    if (!m.filename) return null;
    const key = `${m.filename}|${m.subfolder || ""}`;
    if (st.bgKey !== key) {
      st.bgKey = key;
      st.bg = null;
      const img = new Image();
      img.onload = () => { if (st.bgKey === key) { st.bg = img; draw(); } };
      img.onerror = () => { if (st.bgKey !== key) return; st.bg = null; st.bgKey = ""; delete node.properties.kubakub_spot_plan; draw(); };   // gone after a restart: the grid again, the next run loads it anew
      img.src = api.apiURL(`/view?filename=${encodeURIComponent(m.filename)}&subfolder=${encodeURIComponent(m.subfolder || "")}&type=${encodeURIComponent(m.type || "temp")}`);
    }
    return st.bg;
  }

  function draw() {
    const m = meta(), list = points(), bg = background();
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.fillStyle = "#101010";
    ctx.fillRect(0, 0, SIZE, SIZE);
    const k = Math.hypot(m.ax[0], m.ax[1]);
    if (bg) ctx.drawImage(bg, 0, 0, SIZE, SIZE);
    else {                                               // no run yet: a 10 m grid and the wall
      ctx.strokeStyle = "rgba(255,255,255,.07)";
      ctx.lineWidth = 1;
      for (let g = -60; g <= 60; g += 10) {
        let a = toPx(g, -30), b = toPx(g, 70);
        ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]); ctx.stroke();
        a = toPx(-60, g); b = toPx(60, g);
        ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]); ctx.stroke();
      }
      const a = toPx(-12, 0), b = toPx(12, 0);
      ctx.strokeStyle = ORANGE; ctx.lineWidth = 3;
      ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]); ctx.stroke();
      ctx.fillStyle = "rgba(255,255,255,.45)";
      ctx.font = '22px "Hanken Grotesk KKD", sans-serif';
      ctx.fillText("the wall (run the node once for the real plan)", a[0], a[1] - 10);
      ctx.fillText(`grid 10 m, ${num(SIZE / k)} m across`, 12, SIZE - 12);
    }
    const centre = toPx(0, 0);
    if (cfg.line && list.length > 1) {                   // the walk, in the order of the lines
      ctx.strokeStyle = LAV; ctx.lineWidth = 3; ctx.setLineDash([10, 8]);
      ctx.beginPath();
      list.forEach((p, i) => { const q = toPx(p.x, p.d); if (i) ctx.lineTo(q[0], q[1]); else ctx.moveTo(q[0], q[1]); });
      ctx.stroke();
      ctx.setLineDash([]);
    }
    list.forEach((p, i) => {
      const q = toPx(p.x, p.d), on = i === st.sel || i === st.drag;
      const look = p.rest.length >= 1 ? toPx(p.rest[0], 0) : centre;       // a walk key may look at another place on the wall
      ctx.strokeStyle = "rgba(161,135,183,.45)"; ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.moveTo(q[0], q[1]); ctx.lineTo(look[0], look[1]); ctx.stroke();
      ctx.fillStyle = on ? ORANGE : LAV;
      ctx.beginPath(); ctx.arc(q[0], q[1], on ? 15 : 13, 0, 7); ctx.fill();
      if (i === st.hover && !on) { ctx.strokeStyle = "rgba(255,255,255,.7)"; ctx.lineWidth = 3; ctx.beginPath(); ctx.arc(q[0], q[1], 20, 0, 7); ctx.stroke(); }
      ctx.fillStyle = "#141414";
      ctx.font = '700 17px "Hanken Grotesk KKD", sans-serif';
      ctx.textAlign = "center";
      ctx.fillText(String(i + 1), q[0], q[1] + 6);
      ctx.textAlign = "left";
      ctx.fillStyle = "rgba(255,255,255,.8)";
      ctx.font = '20px "Hanken Grotesk KKD", sans-serif';
      const say = `${num(p.d)} m away, eye ${num(p.eye)} m`, wide = ctx.measureText(say).width;
      ctx.fillText(say, q[0] + 22 + wide > SIZE - 6 ? q[0] - 22 - wide : q[0] + 22, q[1] + 7);       // near the right edge: on the left
    });
    const s = list[st.sel];
    hint.textContent = s ? `${cfg.word} ${st.sel + 1}: ${num(s.x)} m along the wall, ${num(s.d)} m away, eye ${num(s.eye)} m. wheel = eye height, delete = remove` : cfg.tip;
  }
  el.kksDraw = draw;

  cv.addEventListener("pointerdown", e => {
    if (e.button !== 0) return;
    const p = evPx(e), list = parse(text?.value);
    let i = hit(list, p);
    if (i < 0) {                                         // empty ground: a new point there (not behind the wall)
      const [x, d] = toM(p[0], p[1]);
      if (d < 0.5) return;
      list.push({ x: Math.round(x * 10) / 10, d: Math.round(d * 10) / 10, eye: list.length ? list[list.length - 1].eye : 1.7, rest: [], line: null });
      commit(list);
      i = parse(text?.value).length - 1;
    }
    st.sel = st.drag = i;
    st.list = parse(text?.value);
    st.moved = false;
    cv.setPointerCapture(e.pointerId);
    cv.focus({ preventScroll: true });
    e.preventDefault(); e.stopPropagation();
    draw();
  });
  cv.addEventListener("pointermove", e => {
    const p = evPx(e);
    if (st.drag < 0) { const h = hit(points(), p); if (h !== st.hover) { st.hover = h; draw(); } return; }
    const [x, d] = toM(p[0], p[1]), q = st.list[st.drag];
    q.x = Math.round(x * 10) / 10;
    q.d = Math.max(0.5, Math.round(d * 10) / 10);
    st.moved = true;
    draw();
  });
  const drop = e => {
    if (st.drag < 0) return;
    const list = st.list;
    st.drag = -1; st.list = null;
    try { cv.releasePointerCapture(e.pointerId); } catch { /* already released */ }
    if (st.moved) commit(list);
    draw();
  };
  cv.addEventListener("pointerup", drop);
  cv.addEventListener("pointercancel", drop);
  cv.addEventListener("pointerleave", () => { if (st.hover >= 0 && st.drag < 0) { st.hover = -1; draw(); } });
  const remove = i => {
    const list = parse(text?.value);
    if (i < 0 || i >= list.length || list.length < 2) return;      // one always stays: the node needs a place to stand
    list.splice(i, 1);
    st.sel = -1;
    commit(list);
    draw();
  };
  cv.addEventListener("dblclick", e => { remove(hit(parse(text?.value), evPx(e))); e.preventDefault(); e.stopPropagation(); });
  cv.addEventListener("keydown", e => { if (e.key === "Delete" || e.key === "Backspace") { remove(st.sel); e.preventDefault(); e.stopPropagation(); } });
  cv.addEventListener("wheel", e => {                    // over a point: its eye height, in steps of 10 cm
    const list = parse(text?.value), i = hit(list, evPx(e));
    if (i < 0) return;
    list[i].eye = Math.max(0.1, Math.round((list[i].eye + (e.deltaY < 0 ? 0.1 : -0.1)) * 10) / 10);
    st.sel = i;
    commit(list);
    draw();
    e.preventDefault(); e.stopPropagation();
  }, { passive: false });
  let seen = null;                                       // typed into the text: the plan follows
  const watch = setInterval(() => { if (!el.isConnected) return; if (text && text.value !== seen && st.drag < 0) { seen = text.value; draw(); } }, 300);
  el.kksStop = () => clearInterval(watch);
  requestAnimationFrame(draw);
  return el;
}

app.registerExtension({
  name: "kubakub.spotplan",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    const cfg = NODES[nodeData.name];
    if (!cfg) return;
    const created = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = created?.apply(this, arguments);
      const el = build(this, cfg);
      const w = this.addDOMWidget("spot_plan", "kubakub_spot_plan", el, { serialize: false, hideOnZoom: false, getMinHeight: () => 380 });
      w.serialize = false;                               // the points live in the text
      this.kksPlan = el;
      return r;
    };
    const configured = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function () {
      const r = configured?.apply(this, arguments);
      requestAnimationFrame(() => this.kksPlan?.kksDraw());
      return r;
    };
    const executed = nodeType.prototype.onExecuted;
    nodeType.prototype.onExecuted = function (msg) {
      const r = executed?.apply(this, arguments);
      const m = msg?.kubakub_spot_plan?.[0];
      if (m) { (this.properties ||= {}).kubakub_spot_plan = m; this.kksPlan?.kksDraw(); }
      return r;
    };
    const removed = nodeType.prototype.onRemoved;
    nodeType.prototype.onRemoved = function () { this.kksPlan?.kksStop?.(); return removed?.apply(this, arguments); };
  },
});
