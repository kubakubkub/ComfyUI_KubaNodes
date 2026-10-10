// kubakub shader: pick a preset by its picture, and see at once whether pasted code can run.
// A "pictures" button right under `preset` opens a grid of all presets; a click sets the node's own `preset` widget,
// so the widget stays the truth and a workflow without this file still runs. A read-only line under it says what the
// node will draw (the preset, or your code and what it reads) or why pasted code cannot run, before you queue.
// The same grid opens from the shape fill of the kubakub director (pickShader is exported for it).
// The pictures are one sheet drawn by the node itself (/kubakub/shader/thumbs), so they show what a queue gives.
// Plain ES module, no build step, no network. Look: kubakub.art (Hanken Grotesk, #f18a58 actions, #a187b7 selection).
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const ORANGE = "#f18a58", LAV = "#a187b7";
const FONT = '"Hanken Grotesk KKD","Hanken Grotesk","Segoe UI",system-ui,sans-serif';
let listing = null;                                  // { presets, about, node_only, tile, columns }, asked once

export function shaderPresets() {
  return (listing ||= api.fetchApi("/kubakub/shader/presets").then(r => r.json()).catch(e => { listing = null; throw e; }));
}

// A grid of the presets next to `anchor`; onPick(name) on a click. Esc or a click outside closes it.
// skip: names to leave out (the director leaves out the presets only the node can draw).
export async function pickShader(anchor, current, onPick, skip = []) {
  document.querySelector(".kks-pick")?.remove();
  let info;
  try { info = await shaderPresets(); } catch (e) { console.warn("[kubakub shader] presets", e); return; }
  const [tw, th] = info.tile || [160, 90], cols = info.columns || 6, sheet = api.apiURL("/kubakub/shader/thumbs");
  const dark = matchMedia("(prefers-color-scheme: dark)").matches;
  const paper = dark ? "#141414" : "#ffffff", ink = dark ? "rgba(255,255,255,.88)" : "rgba(0,0,0,.85)", line = dark ? "rgba(255,255,255,.12)" : "rgba(0,0,0,.12)";
  const box = document.createElement("div");
  box.className = "kks-pick";
  box.style.cssText = `position:fixed;z-index:10050;background:${paper};color:${ink};border:1px solid ${line};border-radius:14px;
    box-shadow:0 18px 50px rgba(0,0,0,.35);padding:12px;box-sizing:border-box;width:min(92vw,${5 * tw + 4 * 10 + 44}px);max-height:80vh;overflow:auto;
    font:400 12px/1.35 ${FONT};text-transform:lowercase`;
  const head = document.createElement("div");
  head.textContent = "pick a shader · drawing the pictures (a few seconds the first time) …";
  head.style.cssText = "opacity:.6;margin:0 2px 8px";
  const grid = document.createElement("div");
  grid.style.cssText = `display:grid;grid-template-columns:repeat(auto-fill,${tw}px);gap:10px`;
  const pics = [];
  info.presets.forEach((name, i) => {
    if (skip.includes(name)) return;
    const b = document.createElement("button");
    b.title = info.about?.[name] || name;
    b.style.cssText = `all:unset;cursor:pointer;display:block;width:${tw}px;border-radius:10px;outline:2px solid ${name === current ? LAV : "transparent"};outline-offset:2px`;
    const pic = document.createElement("div");
    pic.style.cssText = `width:${tw}px;height:${th}px;border-radius:8px;background:${dark ? "#2c2c2e" : "#e9e6e3"};background-position:-${(i % cols) * tw}px -${Math.floor(i / cols) * th}px;background-repeat:no-repeat`;
    pics.push(pic);
    const label = document.createElement("div");
    label.textContent = name;
    label.style.cssText = `padding:4px 2px 0;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;color:${name === current ? LAV : "inherit"}`;
    b.append(pic, label);
    b.onmouseenter = () => { if (name !== current) b.style.outlineColor = ORANGE; };
    b.onmouseleave = () => { b.style.outlineColor = name === current ? LAV : "transparent"; };
    b.onclick = () => { close(); onPick(name); };
    grid.append(b);
  });
  const img = new Image();                              // the sheet is drawn by the node on its first request
  img.onload = () => { for (const q of pics) q.style.backgroundImage = `url("${sheet}")`; head.textContent = "pick a shader · hover for what it draws"; };
  img.onerror = () => { head.textContent = "pick a shader · no pictures here (the graphics context did not start), the names still work"; };
  img.src = sheet;
  box.append(head, grid);
  document.body.append(box);
  const r = anchor?.getBoundingClientRect?.() || { left: innerWidth / 2 - 200, bottom: innerHeight / 3, top: innerHeight / 3 };
  const bw = box.offsetWidth, bh = box.offsetHeight;
  box.style.left = Math.max(8, Math.min(innerWidth - bw - 8, r.left)) + "px";
  box.style.top = Math.max(8, r.bottom + 6 + bh > innerHeight ? Math.max(8, innerHeight - bh - 8) : r.bottom + 6) + "px";
  const away = e => { if (!box.contains(e.target)) close(); }, key = e => { if (e.key === "Escape") { e.stopPropagation(); close(); } };
  function close() { box.remove(); document.removeEventListener("pointerdown", away, true); document.removeEventListener("keydown", key, true); }
  setTimeout(() => { document.addEventListener("pointerdown", away, true); document.addEventListener("keydown", key, true); }, 0);
}

// A read-only text widget shows its label and no value, so the sentence is the label; all of it is in the tooltip.
const DRAWS_TIP = "What the node will draw: the preset, or your pasted code and what it reads; or why pasted code cannot run.";
function say(line, text) { line.value = ""; line.label = text || "draws"; line.tooltip = text ? text + "\n\n" + DRAWS_TIP : DRAWS_TIP; }

// what the node will draw, in a line: asked from the node itself, so it is what a queue does
async function check(node) {
  const w = name => node.widgets?.find(q => q.name === name), line = w("draws");
  if (!line) return;
  const code = String(w("code")?.value || ""), preset = String(w("preset")?.value || "");
  const mine = ++node._kksCheck;
  try {
    const r = await api.fetchApi("/kubakub/shader/translate", { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(code.trim() ? { code } : { preset }) });
    const j = await r.json();
    if (mine !== node._kksCheck) return;                // a newer check is on its way
    const reads = (j.channels || []).map(i => "image" + i).join(", ");
    const text = j.error ? "cannot run: " + j.error
      : code.trim() ? `your code (the preset is not used)${reads ? " · reads " + reads : ""}${(j.uses || []).includes("kPrev") ? " · builds on its last frame" : ""}`
        : `${preset}: ${j.about || ""}`;
    say(line, text);
  } catch (e) { if (mine === node._kksCheck) say(line, ""); }
  node.setDirtyCanvas(true, true);
}

app.registerExtension({
  name: "kubakub.shader",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== "KUBA_Shader") return;
    const created = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const out = created?.apply(this, arguments);
      const preset = this.widgets?.find(w => w.name === "preset");
      if (!preset || this.widgets.some(w => w.name === "pictures")) return out;
      const node = this;
      node._kksCheck = 0;
      const button = this.addWidget("button", "pictures", null, () => {
        const c = app.canvas, pos = c?.canvas ? c.canvas.getBoundingClientRect() : { left: 0, top: 0 };
        const at = c?.ds ? { left: pos.left + (node.pos[0] + node.size[0] + c.ds.offset[0]) * c.ds.scale + 8, bottom: pos.top + (node.pos[1] + c.ds.offset[1]) * c.ds.scale } : null;
        pickShader(at ? { getBoundingClientRect: () => ({ left: at.left, bottom: at.bottom, top: at.bottom }) } : null, preset.value, name => {
          preset.value = name; preset.callback?.(name);
          const controls = node.widgets.find(w => w.name === "controls");      // another shader has other controls
          if (controls && String(controls.value || "").trim()) { controls.value = ""; controls.callback?.(""); }
          check(node); node.setDirtyCanvas(true, true);
        });
      }, { serialize: false });
      button.label = "pick a preset by its picture";
      button.tooltip = "Shows every preset as a small picture; click one to use it. Used while 'code' is empty.";
      button.serialize = false;
      const line = this.addWidget("text", "draws", "", () => {}, { serialize: false });
      line.serialize = false; line.disabled = true;
      line.tooltip = DRAWS_TIP;
      // neither is saved (serialize = false), so saved workflows keep their values; they sit right under `preset`
      for (const w of [line, button]) { this.widgets.splice(this.widgets.indexOf(w), 1); this.widgets.splice(this.widgets.indexOf(preset) + 1, 0, w); }
      for (const name of ["preset", "code"]) { const w = this.widgets.find(q => q.name === name); if (!w) continue;
        const cb = w.callback; w.callback = function () { const r = cb?.apply(this, arguments); check(node); return r; };
        const el = w.inputEl || w.element; el?.addEventListener?.("input", () => { clearTimeout(node._kksT); node._kksT = setTimeout(() => check(node), 400); }); }
      setTimeout(() => check(node), 300);
      return out;
    };
    const configured = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function () { const r = configured?.apply(this, arguments); setTimeout(() => check(this), 300); return r; };
  },
});
