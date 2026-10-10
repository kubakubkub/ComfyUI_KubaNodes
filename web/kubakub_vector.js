// kubakub mask field and kubakub shader: draw the direction as an arrow on the node, instead of typing angles and
// positions. It only reads and writes the node's own `vector` widget ("x0, y0, x1, y1", each 0 to 1 of the picture),
// so the widget stays the truth and a workflow without this file still runs.
// The picture under the arrow is what the node drew at its last run (run it once to see your regions).
// Plain ES module, no build step, no network. Look: kubakub.art (Hanken Grotesk, #f18a58 actions, #a187b7 selection).
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const ORANGE = "#f18a58", LAV = "#a187b7";
const NODES = { KUBA_MaskField: "direction: the move runs from the arrow's start to its end · radial: its start is the centre",
  KUBA_Shader: "the shader follows the arrow: along it, in its direction, or starting at its start" };

function parse(text) {
  const v = String(text || "").split(/[\s,;]+/).filter(Boolean).map(Number);
  return v.length === 4 && v.every(Number.isFinite) && Math.hypot(v[2] - v[0], v[3] - v[1]) > 1e-6 ? v : null;
}

function build(node, widget, hint) {
  const el = document.createElement("div");
  el.style.cssText = "display:flex;flex-direction:column;gap:4px;width:100%;height:100%;font:12px 'Hanken Grotesk',sans-serif;color:#ddd";
  const head = document.createElement("div");
  head.style.cssText = "display:flex;align-items:center;gap:6px;min-height:22px";
  const label = document.createElement("span");
  label.textContent = "draw the direction: drag on the picture";
  label.title = hint;
  label.style.cssText = "flex:1;opacity:.8;white-space:nowrap;overflow:hidden;text-overflow:ellipsis";
  const clear = document.createElement("button");
  clear.textContent = "clear";
  clear.title = "Remove the arrow";
  clear.style.cssText = `border:1px solid ${LAV};background:none;color:#ddd;border-radius:999px;padding:1px 10px;font:inherit;cursor:pointer`;
  head.append(label, clear);
  const cv = document.createElement("canvas");
  cv.style.cssText = "flex:1;width:100%;min-height:0;border-radius:4px;background:#161519;cursor:crosshair;touch-action:none";
  el.append(head, cv);

  const st = { img: null, drag: null };
  const rect = () => {                                 // where the picture sits in the canvas (fitted, centred)
    const W = cv.width, H = cv.height, a = st.img ? st.img.naturalWidth / st.img.naturalHeight : 16 / 9;
    const w = Math.min(W, H * a), h = w / a;
    return [(W - w) / 2, (H - h) / 2, w, h];
  };
  function draw() {
    const r = cv.getBoundingClientRect(), d = Math.min(2, window.devicePixelRatio || 1);
    const W = Math.max(2, Math.round(r.width * d)), H = Math.max(2, Math.round(r.height * d));
    if (cv.width !== W || cv.height !== H) { cv.width = W; cv.height = H; }
    const g = cv.getContext("2d"), [x, y, w, h] = rect();
    g.clearRect(0, 0, W, H);
    if (st.img) g.drawImage(st.img, x, y, w, h);
    else {
      g.fillStyle = "#222027"; g.fillRect(x, y, w, h); g.strokeStyle = "rgba(255,255,255,.08)"; g.lineWidth = 1;
      for (let i = 1; i < 8; i++) { g.beginPath(); g.moveTo(x + w * i / 8, y); g.lineTo(x + w * i / 8, y + h); g.stroke(); }
      for (let i = 1; i < 4; i++) { g.beginPath(); g.moveTo(x, y + h * i / 4); g.lineTo(x + w, y + h * i / 4); g.stroke(); }
      g.fillStyle = "rgba(255,255,255,.45)"; g.font = `${12 * d}px 'Hanken Grotesk', sans-serif`; g.textAlign = "center";
      g.fillText("run the node once to see the picture here", x + w / 2, y + h - 10 * d);
    }
    const v = st.drag || parse(widget.value);
    clear.style.visibility = parse(widget.value) ? "visible" : "hidden";
    if (!v) return;
    const x0 = x + v[0] * w, y0 = y + v[1] * h, x1 = x + v[2] * w, y1 = y + v[3] * h, a = Math.atan2(y1 - y0, x1 - x0), k = 9 * d;
    g.lineCap = "round";
    for (const [col, wd] of [["rgba(0,0,0,.6)", 5 * d], [ORANGE, 2.5 * d]]) {
      g.strokeStyle = col; g.fillStyle = col; g.lineWidth = wd;
      g.beginPath(); g.moveTo(x0, y0); g.lineTo(x1, y1); g.stroke();
      g.beginPath(); g.moveTo(x1, y1); g.lineTo(x1 - k * Math.cos(a - 0.45), y1 - k * Math.sin(a - 0.45)); g.lineTo(x1 - k * Math.cos(a + 0.45), y1 - k * Math.sin(a + 0.45)); g.closePath(); g.fill();
      g.beginPath(); g.arc(x0, y0, wd * 1.2, 0, 7); g.fill();
    }
  }
  const at = e => {
    const r = cv.getBoundingClientRect(), s = cv.width / Math.max(1, r.width), [x, y, w, h] = rect();
    return [((e.clientX - r.left) * s - x) / w, ((e.clientY - r.top) * s - y) / h];
  };
  const set = text => { widget.value = text; widget.callback?.(text); node.setDirtyCanvas(true, true); draw(); };
  cv.addEventListener("pointerdown", e => { if (e.button) return; e.preventDefault(); e.stopPropagation(); cv.setPointerCapture(e.pointerId);
    const p = at(e); st.drag = [p[0], p[1], p[0], p[1]]; draw(); });
  cv.addEventListener("pointermove", e => { if (!st.drag) return; e.stopPropagation(); const p = at(e); st.drag[2] = p[0]; st.drag[3] = p[1]; draw(); });
  const up = e => { if (!st.drag) return; e.stopPropagation(); const v = st.drag; st.drag = null;
    if (Math.hypot(v[2] - v[0], v[3] - v[1]) > 0.01) set(v.map(n => +n.toFixed(3)).join(", ")); else draw(); };
  cv.addEventListener("pointerup", up);
  cv.addEventListener("pointercancel", () => { st.drag = null; draw(); });
  for (const t of ["wheel", "dblclick", "contextmenu"]) cv.addEventListener(t, e => e.stopPropagation());
  clear.addEventListener("click", e => { e.stopPropagation(); set(""); });
  new ResizeObserver(draw).observe(cv);
  return { el, draw, picture(url) { const im = new Image(); im.onload = () => { st.img = im; draw(); }; im.src = url; } };
}

app.registerExtension({
  name: "kubakub.vector",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    const hint = NODES[nodeData.name];
    if (!hint) return;
    const created = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const out = created?.apply(this, arguments);
      const widget = this.widgets?.find(w => w.name === "vector");
      if (!widget || this.kkVec) return out;
      const view = this.kkVec = build(this, widget, hint);
      const w = this.addDOMWidget("vector_view", "kubakub_vector", view.el, { serialize: false, hideOnZoom: false, getMinHeight: () => 170 });
      w.serialize = false;                             // nothing of its own to save: it shows and sets `vector`
      const cb = widget.callback;
      widget.callback = function () { const r = cb?.apply(this, arguments); view.draw(); return r; };
      (widget.inputEl || widget.element)?.addEventListener?.("input", () => view.draw());
      this.setSize([this.size[0], this.size[1] + 180]);
      setTimeout(view.draw, 300);
      return out;
    };
    const configured = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function () { const r = configured?.apply(this, arguments); setTimeout(() => this.kkVec?.draw(), 300); return r; };
    const executed = nodeType.prototype.onExecuted;
    nodeType.prototype.onExecuted = function (msg) {    // what the node just drew goes under the arrow
      const r = executed?.apply(this, arguments);
      const im = msg?.images?.[0];
      if (im && this.kkVec) this.kkVec.picture(api.apiURL(`/view?filename=${encodeURIComponent(im.filename)}&type=${im.type || "temp"}&subfolder=${encodeURIComponent(im.subfolder || "")}&rand=${Date.now()}`));
      return r;
    };
  },
});
