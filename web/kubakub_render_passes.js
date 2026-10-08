// kubakub render passes: the layers inside a multilayer EXR as buttons on the node. Click one = that layer is a mask
// (or no longer one). It only reads the node's folder / render widgets and writes the exr_layers text, so the text
// stays the truth and a workflow without this file still runs. The list comes from the pack's own route
// /kubakub/render_passes/layers (it reads the EXR's header, nothing else); the rules are those of
// kubakub/render_passes.py pick_layers().
// Plain ES module, no build step, no network. Look: kubakub.art (Hanken Grotesk, #f18a58 actions, #a187b7 selection).
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const HERE = new URL(".", import.meta.url).href;
const ORANGE = "#f18a58", LAV = "#a187b7";
const FIXED = ["picture", "depth", "normal"];
const ALIAS = { picture: "picture", beauty: "picture", depth: "depth", normal: "normal", normals: "normal", mask: "mask", skip: "skip" };
const NONE = "# no layer as a mask";

// ---- the exr_layers text (as pick_layers reads it)
function glob(pat) {
  const re = pat.toLowerCase().replace(/[.+^${}()|\\]/g, "\\$&").replace(/\*/g, ".*").replace(/\?/g, ".");
  try { return new RegExp("^" + re + "$"); } catch { return /^$/; }
}
const hits = (pat, name) => { const re = glob(pat), n = name.toLowerCase(); return re.test(n) || re.test(n.slice(n.lastIndexOf(".") + 1)); };
function parse(text) {
  return String(text || "").split("\n").map(s => s.trim()).filter(s => s && !s.startsWith("#")).map(line => {
    const i = line.indexOf("=");
    return i < 0 ? { role: "", pat: line } : { role: ALIAS[line.slice(0, i).trim().toLowerCase()] || "?", pat: line.slice(i + 1).trim() };
  });
}
// does the text say which layers are masks (= render_passes._chooses_masks)? role and skip lines only correct the guess
const choosesMasks = text => String(text || "").split("\n").map(s => s.trim()).some(ln => {
  if (ln.toLowerCase() === NONE) return true;
  if (!ln || ln.startsWith("#")) return false;
  const i = ln.indexOf("=");
  return i < 0 || ALIAS[ln.slice(0, i).trim().toLowerCase()] === "mask";
});
export function uses(layers, text) {                  // -> { layer name: "picture" / "depth" / "normal" / "mask" / "" }
  const use = {}, crypto = n => n.toLowerCase().includes("crypto");
  for (const L of layers) use[L.name] = FIXED.includes(L.role) ? L.role : "";
  if (!choosesMasks(text)) for (const L of layers) if (L.role === "mask") use[L.name] = "mask";
  for (const { role, pat } of parse(text)) {
    const found = layers.filter(L => hits(pat, L.name)), wild = /[*?[]/.test(pat);
    if (!found.length) continue;
    if (FIXED.includes(role)) {
      for (const k in use) if (use[k] === role) use[k] = "";
      use[found[0].name] = role;
      continue;
    }
    for (const L of found) {
      if (role === "skip") use[L.name] = "";
      else if (wild && FIXED.includes(use[L.name])) continue;
      else if (crypto(L.name) || /multipart/.test(L.why || "")) continue;
      else if (role === "mask" || (!wild && (L.role === "skip" || L.role === "id")) || L.role === "mask") use[L.name] = "mask";
    }
  }
  return use;
}
export function toggled(layers, text, name) {         // the text after a click on the layer's button
  const use = uses(layers, text), same = s => s.trim().toLowerCase() === name.toLowerCase();
  const lines = String(text || "").split("\n").filter(s => s.trim() && s.trim() !== NONE);
  const named = s => { const i = s.indexOf("="); return i < 0 ? same(s) : same(s.slice(i + 1)) && ["mask", "skip"].includes(ALIAS[s.slice(0, i).trim().toLowerCase()]); };
  const blank = !String(text || "").trim(), want = use[name] !== "mask";   // empty text = every guessed mask: spell them out
  const out = (blank ? layers.filter(L => use[L.name] === "mask").map(L => L.name) : lines).filter(s => !named(s));
  if ((uses(layers, out.join("\n") || NONE)[name] === "mask") !== want) out.push(want ? name : "skip = " + name);   // a wildcard still takes it
  return out.join("\n") || NONE;
}

let styled = false;
function style() {
  if (styled) return;
  styled = true;
  const el = document.createElement("style");
  el.textContent = `
@font-face{font-family:"Hanken Grotesk KKD";src:url("${HERE}fonts/HankenGrotesk-latin-var.woff2") format("woff2");font-weight:200 800;font-display:swap}
.kkr{display:flex;flex-direction:column;gap:5px;width:100%;height:100%;font-family:"Hanken Grotesk KKD",system-ui,sans-serif;font-size:11px;color:rgba(255,255,255,.85);box-sizing:border-box}
.kkr-row{display:flex;flex-wrap:wrap;gap:4px;align-items:center}
.kkr-pills{flex:1;min-height:0;overflow-y:auto;align-content:flex-start}
.kkr button{font:inherit;border:0;border-radius:10em;padding:2px 9px;background:rgba(255,255,255,.12);color:rgba(255,255,255,.85);cursor:pointer}
.kkr button small{font-size:9px;opacity:.6;margin-left:5px;text-transform:lowercase}
.kkr button.on{background:${ORANGE};color:#141414;font-weight:700}
.kkr button.fix{background:${LAV};color:#141414;font-weight:700;cursor:default}
.kkr button.no{opacity:.45;cursor:default}
.kkr button.act{text-transform:lowercase;background:transparent;box-shadow:inset 0 0 0 1px rgba(255,255,255,.3)}
.kkr button.act:hover{box-shadow:inset 0 0 0 1px ${ORANGE};color:${ORANGE}}
.kkr-hint{color:rgba(255,255,255,.5);min-height:14px;text-transform:lowercase}`;
  document.head.appendChild(el);
}

function build(node) {
  style();
  const el = document.createElement("div");
  el.className = "kkr";
  const head = document.createElement("div");
  head.className = "kkr-row";
  const list = document.createElement("button");
  list.className = "act";
  list.textContent = "list layers";
  list.title = "read which layers are inside the EXR of the folder field";
  const hint = document.createElement("div");
  hint.className = "kkr-hint";
  head.append(list, hint);
  const pills = document.createElement("div");
  pills.className = "kkr-row kkr-pills";
  el.append(head, pills);
  const widget = name => node.widgets?.find(w => w.name === name);
  const val = name => String(widget(name)?.value ?? "");
  const set = (name, v) => { const w = widget(name); if (w) { w.value = v; w.callback?.(v); node.graph?.setDirtyCanvas(true, true); } };
  let layers = [], note = "", token = 0, asked = null, timer = 0;

  function render() {
    const use = uses(layers, val("exr_layers"));
    pills.replaceChildren(...layers.map(L => {
      const b = document.createElement("button"), u = use[L.name];
      const never = L.name.toLowerCase().includes("crypto") || /multipart/.test(L.why || "");
      b.append(L.name);
      const s = document.createElement("small");
      s.textContent = u && u !== L.role ? `${L.role} → ${u}` : L.role;
      b.append(s);
      b.className = FIXED.includes(u) ? "fix" : u === "mask" ? "on" : never ? "no" : "";
      b.title = `channels: ${L.channels.join(" ")}\n` + (FIXED.includes(u) ? `read as the ${u}. another layer for it: write '${u} = name' in exr_layers`
        : u === "mask" ? "read as a mask. click = leave it out"
        : never ? `not read: ${L.why}` : `not read${L.why ? " (" + L.why + ")" : ""}. click = read it as a mask`);
      if (!FIXED.includes(u) && !never) b.addEventListener("click", () => { set("exr_layers", toggled(layers, val("exr_layers"), L.name)); render(); });
      return b;
    }));
    const n = Object.values(use).filter(u => u === "mask").length;
    hint.textContent = layers.length ? `${note}. ${n} as masks. orange = mask, violet = picture / depth / normal` : note;
  }

  async function refresh(force) {
    const path = val("folder").trim(), key = path + "|" + val("render").trim();
    if (!force && key === asked) return;
    asked = key;
    const mine = ++token;
    if (!path) { layers = []; note = "no folder: the node runs on its sample passes (png files, no exr)"; render(); return; }
    note = "reading…"; hint.textContent = note;
    try {
      const r = await api.fetchApi("/kubakub/render_passes/layers", { method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ path, render: val("render") }) });
      const j = await r.json();
      if (mine !== token) return;
      layers = Array.isArray(j.layers) ? j.layers : [];
      note = j.note || (r.ok ? "" : "no answer (" + r.status + ")");
    } catch (e) {
      if (mine !== token) return;
      layers = []; note = "the layer list needs a restart of ComfyUI (type the names meanwhile)";
      console.warn("[kubakub render passes] layers", e);
    }
    render();
  }
  const soon = () => { clearTimeout(timer); timer = setTimeout(() => refresh(false), 400); };

  list.addEventListener("click", () => refresh(true));
  for (const name of ["folder", "render"]) {            // the folder changed: read its layers again
    const w = widget(name);
    if (!w) continue;
    const cb = w.callback;
    w.callback = function () { const r = cb?.apply(this, arguments); soon(); return r; };
  }
  const tw = widget("exr_layers");                      // typed by hand: the buttons follow
  if (tw) {
    const cb = tw.callback;
    tw.callback = function () { const r = cb?.apply(this, arguments); render(); return r; };
    tw.inputEl?.addEventListener?.("input", render);
  }
  el.kkrRefresh = soon;
  el.kkrStop = () => { clearTimeout(timer); token++; };
  soon();
  return el;
}

app.registerExtension({
  name: "kubakub.renderpasses",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== "KUBA_RenderPasses") return;
    const created = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = created?.apply(this, arguments);
      const el = build(this);
      const w = this.addDOMWidget("layer_buttons", "kubakub_exr_layers", el, { serialize: false, hideOnZoom: false,
        getMinHeight: () => 150 });
      w.serialize = false;                               // nothing of its own to save: exr_layers holds the choice
      this.kkrLayers = el;
      return r;
    };
    const configured = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function () {      // a loaded workflow: its folder is in the widget now
      const r = configured?.apply(this, arguments);
      this.kkrLayers?.kkrRefresh?.();
      return r;
    };
    const removed = nodeType.prototype.onRemoved;
    nodeType.prototype.onRemoved = function () {
      this.kkrLayers?.kkrStop?.();
      return removed?.apply(this, arguments);
    };
  },
});
