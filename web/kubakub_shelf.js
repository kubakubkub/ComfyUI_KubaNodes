// kubakub shelf: a strip at the edge of the screen that holds pictures, videos, sounds, prompts and colours while you
// work, and gives them back by dragging. It slides out when the pointer (or a drag) reaches the edge, or with Alt+S.
//   in:  drop files, node previews, renders from the timeline list, text or a colour on it; paste (Ctrl+V) while it is
//        open; right-click a node -> "put on the kubakub shelf" (its pictures, or its text)
//   out: drag onto the timeline (a clip), into a director window (a layer), onto a node (a Load Image / Load Video /
//        Load Audio file, a text field), onto empty canvas (a new loader or text node), or into any text field
// Files live in ComfyUI's input/kuba_director (the timeline's and the director's import folder) or stay where the
// render put them (output/); the list is a user file (user/default/kubakub/shelf.json). Nothing leaves this computer.
// Idea after Edge-Drop (a clipboard shelf for Windows); written new for ComfyUI.
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const HERE = new URL(".", import.meta.url).href;
export const SHELF_MIME = "application/x-kubakub-shelf";
const TIMELINE_MIME = "application/x-kubakub-master";   // = DRAG_MIME in kubakub_master.js: {source}
const FILE = "kubakub/shelf.json";
const MAX = 200;                                    // unpinned items above this go, oldest first
const EDGE_ID = "kubakub.shelf.edge";
const VIDEO_RE = /\.(mp4|mov|webm|mkv|avi|m4v|gif)$/i, IMAGE_RE = /\.(png|jpe?g|webp|bmp|tiff?)$/i, AUDIO_RE = /\.(wav|mp3|ogg|flac|m4a|aac|opus)$/i;
export const kindOf = name => (VIDEO_RE.test(name) ? "video" : IMAGE_RE.test(name) ? "image" : AUDIO_RE.test(name) ? "audio" : null);
const COLOUR_RE = /^\s*(#[0-9a-f]{3}|#[0-9a-f]{6}|#[0-9a-f]{8}|rgba?\([\d\s.,%]+\))\s*$/i;
export const isColour = t => COLOUR_RE.test(String(t || ""));
const uid = () => "s" + Date.now().toString(36) + Math.random().toString(36).slice(2, 6);

// ------------------------------------------------------------------------------------------------ model (no DOM)
// item: {id, kind: image|video|audio|text|colour, source?: 'file:<input/>' | 'output:<output/>', text?, name, pinned, t}
export function addItem(list, it, max = MAX) {
  const same = list.findIndex(o => o.kind === it.kind && (it.source ? o.source === it.source : o.text === it.text));
  if (same >= 0) { const [o] = list.splice(same, 1); list.unshift(o); o.t = Date.now(); return list; }   // again = to the top
  list.unshift(Object.assign({ id: uid(), pinned: false, t: Date.now() }, it));
  for (let i = list.length - 1; list.length > max && i >= 0; i--) if (!list[i].pinned) list.splice(i, 1);
  return list;
}
export function textItem(text) {
  const t = String(text || "").trim(); if (!t) return null;
  return isColour(t) ? { kind: "colour", text: t, name: t.toLowerCase() } : { kind: "text", text: t, name: t.split("\n")[0].slice(0, 60) };
}
export function sourceFromView(url) {               // /view?filename=&subfolder=&type= -> {source} or {temp: url}
  try {
    const u = new URL(url, location.href); if (!u.pathname.endsWith("/view")) return null;
    const f = u.searchParams.get("filename"), sub = u.searchParams.get("subfolder") || "", type = u.searchParams.get("type") || "output";
    if (!f || !kindOf(f)) return null;
    if (type === "temp") return { temp: u.href, name: f };          // previews are wiped on restart: copied in
    if (type !== "output" && type !== "input") return null;
    return { source: `${type === "output" ? "output:" : "file:"}${sub ? sub + "/" : ""}${f}`, name: f };
  } catch { return null; }
}
export function viewURL(source) {
  const [type, rest] = source.startsWith("output:") ? ["output", source.slice(7)] : ["input", source.replace(/^file:/, "")];
  const i = rest.lastIndexOf("/");
  return api.apiURL(`/view?${new URLSearchParams({ filename: rest.slice(i + 1), subfolder: i < 0 ? "" : rest.slice(0, i), type })}`);
}
// the value a core loader's file list takes for a source: 'sub/name.png' (input/) or 'sub/name.png [output]'
export function loaderValue(source) { return source.startsWith("output:") ? source.slice(7) + " [output]" : source.replace(/^file:/, ""); }

// ------------------------------------------------------------------------------------------------ the shelf
function h(tag, attrs = {}, ...kids) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k.startsWith("on")) e.addEventListener(k.slice(2), v); else if (k === "class") e.className = v; else if (v != null && v !== false) e.setAttribute(k, v === true ? "" : v);
  }
  for (const c of kids.flat()) if (c != null) e.append(c);
  return e;
}
const FILTERS = [["all", null], ["pictures", "image"], ["videos", "video"], ["sounds", "audio"], ["text", "text"], ["colours", "colour"]];

class Shelf {
  constructor() {
    this.items = []; this.filter = null; this.open = false; this.hold = false; this.timer = null; this.dragging = false;
    injectStyle();
    this.hot = h("div", { class: "kksh-hot", title: "kubakub shelf (Alt+S)" });
    this.list = h("div", { class: "kksh-list" });
    this.msg = h("div", { class: "kksh-msg" });
    this.pills = h("div", { class: "kksh-pills" }, FILTERS.map(([n, k]) => h("button", { class: "kksh-pill", "data-k": k ?? "", onclick: () => { this.filter = k; this.render(); } }, n)));
    this.pinBtn = h("button", { class: "kksh-pill", title: "keep the shelf open", onclick: () => { this.hold = !this.hold; this.render(); } }, "keep open");
    this.root = h("div", { class: "kksh-shelf", tabindex: "-1" },
      h("div", { class: "kksh-head" }, h("span", { class: "kksh-title" }, h("b", {}, "kubakub"), " shelf"), this.pinBtn),
      this.pills,
      h("div", { class: "kksh-add" },
        h("button", { class: "kksh-pill", onclick: () => this.addText() }, "+ text"),
        h("button", { class: "kksh-pill", title: "remove everything that is not pinned", onclick: () => { this.items = this.items.filter(i => i.pinned); this.save(); this.render(); } }, "clear")),
      this.list, this.msg,
      h("div", { class: "kksh-hint" }, "drop pictures, videos, sounds, text or colours here · drag them onto the timeline, a director, a node or the canvas"));
    document.body.append(this.hot, this.root);
    // open on the edge, and when a drag reaches it (a node preview, a file from the desktop)
    this.hot.addEventListener("pointerenter", () => { clearTimeout(this.timer); this.timer = setTimeout(() => this.show(true), 120); });
    this.hot.addEventListener("pointerleave", () => clearTimeout(this.timer));
    this.hot.addEventListener("dragenter", e => { e.preventDefault(); this.show(true); });
    this.hot.addEventListener("dragover", e => e.preventDefault());
    this.root.addEventListener("pointerleave", () => { if (!this.hold && !this.dragging) { clearTimeout(this.timer); this.timer = setTimeout(() => this.show(false), 450); } });
    this.root.addEventListener("pointerenter", () => clearTimeout(this.timer));
    this.root.addEventListener("dragover", e => { e.preventDefault(); e.stopPropagation(); clearTimeout(this.timer); if (!this.dragging) { e.dataTransfer.dropEffect = "copy"; this.root.classList.add("kksh-drop"); } });
    this.root.addEventListener("dragleave", e => { if (!this.root.contains(e.relatedTarget)) this.root.classList.remove("kksh-drop"); });
    this.root.addEventListener("drop", e => { e.preventDefault(); e.stopPropagation(); this.root.classList.remove("kksh-drop"); if (!this.dragging) this.takeDrop(e.dataTransfer); });
    this.root.addEventListener("paste", e => { if (e.target.matches?.("textarea,input")) return; e.preventDefault(); e.stopPropagation(); this.takeDrop(e.clipboardData); });
    this.root.addEventListener("keydown", e => { if (e.key === "Escape") this.show(false); e.stopPropagation(); });
    this.edge(app.extensionManager?.setting?.get?.(EDGE_ID) || "right");
    this.load();
  }
  edge(side) {
    this.side = side; this.hot.dataset.side = this.root.dataset.side = side;
    this.hot.hidden = side === "off"; if (side === "off") this.show(false);
  }
  toggle() { if (this.side !== "off") this.show(!this.open); }
  show(on) {
    this.open = on; this.root.classList.toggle("kksh-open", on);
    if (on) { this.render(); this.root.focus({ preventScroll: true }); } else { this.hold = false; this.status(""); }
  }
  status(t) { this.msg.textContent = t; }
  async load() {
    try { const r = await api.getUserData(FILE); if (r.ok) { const j = await r.json(); this.items = Array.isArray(j?.items) ? j.items.filter(i => i && i.kind) : []; } }
    catch { /* a first start: no file yet */ }
    this.render();
  }
  save() { api.storeUserData(FILE, { version: 1, items: this.items }, { overwrite: true, stringify: true, throwOnError: false }).catch(() => {}); }
  add(it) { if (!it) return; addItem(this.items, it); this.save(); this.render(); }

  // ---- in
  async takeDrop(dt) {
    if (!dt) return;
    try {
      const files = [...(dt.files || [])];
      if (files.length) { for (const f of files) await this.addFile(f); return; }
      const tl = dt.getData?.(TIMELINE_MIME);
      if (tl) { const p = JSON.parse(tl); if (p.source) this.add({ kind: kindOf(p.source) || "image", source: p.source, name: p.source.split("/").pop() }); else this.status("a director stays on the timeline"); return; }
      const url = (dt.getData("text/uri-list") || "").split("\n").find(l => l && !l.startsWith("#"))?.trim();
      const v = url && sourceFromView(url);
      if (v) { await this.addView(v); return; }
      const text = dt.getData("text/plain");
      if (text) { const v2 = sourceFromView(text.trim()); if (v2) await this.addView(v2); else this.add(textItem(text)); return; }
      this.status("nothing the shelf can hold");
    } catch (e) { this.status(e.message); }
  }
  async addView(v) {
    if (v.source) return this.add({ kind: kindOf(v.name), source: v.source, name: v.name });
    const blob = await (await fetch(v.temp)).blob();            // a temp preview: kept as a copy in input/
    return this.addFile(new File([blob], v.name, { type: blob.type }));
  }
  async addFile(f) {
    const name = f.name || (f.type === "image/png" ? "pasted.png" : "pasted");
    if (!kindOf(name)) { this.status(`${name}: not a picture, video or sound`); return; }
    this.status(`copying ${name}…`);
    const fd = new FormData(); fd.append("file", f, name);
    const r = await api.fetchApi("/kubakub/master/upload", { method: "POST", body: fd }), j = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(j.error || r.statusText);
    this.status(""); this.add({ kind: kindOf(j.name), source: "file:" + j.name, name: j.name.split("/").pop() });
  }
  addText() {
    const ta = h("textarea", { class: "kksh-edit", placeholder: "a prompt, a note or a colour (#f18a58) · Enter keeps it, Shift+Enter = new line", rows: "4" });
    const done = keep => { if (keep) this.add(textItem(ta.value)); ta.remove(); };
    ta.addEventListener("keydown", e => { e.stopPropagation(); if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); done(true); } if (e.key === "Escape") done(false); });
    ta.addEventListener("blur", () => done(!!ta.value.trim()));
    this.list.before(ta); ta.focus();
  }
  async fromNode(node) {                             // right-click -> put on the kubakub shelf
    let n = 0;
    for (const im of node.imgs || []) { const v = sourceFromView(im.src); if (v) { await this.addView(v); n++; } }
    const vid = node.videoContainer?.querySelector?.("video,source")?.src;            // core video previews
    if (vid) { const v = sourceFromView(vid); if (v) { await this.addView(v); n++; } }
    if (!n) for (const w of node.widgets || []) if (isTextWidget(w) && String(w.value || "").trim()) { this.add(textItem(w.value)); n++; }
    this.show(true); this.status(n ? "" : "this node has no picture or text to keep yet");
  }

  // ---- the list
  render() {
    for (const b of this.pills.children) b.setAttribute("aria-pressed", String((b.dataset.k || null) === this.filter));
    this.pinBtn.setAttribute("aria-pressed", String(this.hold));
    if (!this.open) return;
    const items = this.items.filter(i => !this.filter || i.kind === this.filter);
    this.list.replaceChildren(...(items.length ? items.map(i => this.card(i)) : [h("div", { class: "kksh-empty" }, this.items.length ? "nothing of this kind" : "empty: drop something here")]));
  }
  card(it) {
    let face;
    if (it.kind === "image") face = h("img", { src: viewURL(it.source), loading: "lazy", draggable: "false", alt: "" });
    else if (it.kind === "video") { face = h("video", { src: viewURL(it.source) + "#t=0.1", muted: true, preload: "metadata", draggable: "false" }); face.muted = true;
      face.addEventListener("pointerenter", () => face.play().catch(() => {})); face.addEventListener("pointerleave", () => { face.pause(); }); }
    else if (it.kind === "audio") face = h("div", { class: "kksh-sound" }, "♪");
    else if (it.kind === "colour") face = h("div", { class: "kksh-swatch", style: `background:${it.text}` });
    else face = h("div", { class: "kksh-text" }, it.text.slice(0, 220));
    const pin = h("button", { class: "kksh-icon", title: it.pinned ? "pinned: stays when the shelf is full or cleared" : "pin", "aria-pressed": String(!!it.pinned),
      onclick: e => { e.stopPropagation(); it.pinned = !it.pinned; this.save(); this.render(); } }, "●");
    const del = h("button", { class: "kksh-icon", title: "remove from the shelf (the file stays)", onclick: e => { e.stopPropagation(); this.items = this.items.filter(o => o !== it); this.save(); this.render(); } }, "×");
    const c = h("div", { class: "kksh-card", draggable: "true", "data-kind": it.kind,
      title: `${it.name}\ndrag it out${it.text ? " · click to copy" : ""}` }, face, h("div", { class: "kksh-foot" }, h("span", { class: "kksh-name" }, it.name), pin, del));
    if (it.text) c.addEventListener("click", () => navigator.clipboard?.writeText(it.text).then(() => this.status("copied"), () => {}));
    c.addEventListener("dragstart", e => {
      this.dragging = true; const dt = e.dataTransfer; dt.effectAllowed = "copy";
      dt.setData(SHELF_MIME, JSON.stringify(it));
      if (it.source) { const u = viewURL(it.source); dt.setData(TIMELINE_MIME, JSON.stringify({ source: it.source })); dt.setData("text/uri-list", u); dt.setData("text/plain", u); }
      else dt.setData("text/plain", it.text);
    });
    c.addEventListener("dragend", () => { this.dragging = false; if (!this.hold && !this.root.matches(":hover")) this.show(false); });
    return c;
  }
}

// ------------------------------------------------------------------------------------------------ out: the canvas
function isTextWidget(w) { return !!w && (w.type === "customtext" || w.options?.multiline || w.element?.tagName === "TEXTAREA" || w.inputEl?.tagName === "TEXTAREA"); }
const LOADERS = { image: ["LoadImage", "image"], video: ["LoadVideo", "file"], audio: ["LoadAudio", "audio"] };
function setWidget(node, w, value) {
  if (w.options?.values && Array.isArray(w.options.values) && !w.options.values.includes(value)) w.options.values.push(value);
  w.value = value; w.callback?.(value, app.canvas, node);
  node.setDirtyCanvas?.(true, true); app.graph?.setDirtyCanvas?.(true, true);
}
// a shelf item dropped on the graph canvas: onto a node that can take it, else a new node at the spot
export function dropOnCanvas(it, e) {
  const c = app.canvas, g = c?.graph; if (!c || !g) return false;
  const [x, y] = c.convertEventToCanvasOffset(e);
  const node = g.getNodeOnPos?.(x, y);
  if (node) {
    if (it.source) {
      const want = LOADERS[it.kind]?.[1];
      const w = node.widgets?.find(w => w.name === want) || node.widgets?.find(w => Array.isArray(w.options?.values) && w.options.values.some(v => typeof v === "string" && kindOf(v) === it.kind));
      if (w) { setWidget(node, w, loaderValue(it.source)); return true; }
    } else {
      const w = node.widgets?.find(isTextWidget) || node.widgets?.find(w => w.type === "text" || w.type === "string");
      if (w) { setWidget(node, w, it.text); return true; }
    }
  }
  const [type, wname] = it.source ? LOADERS[it.kind] || [] : ["PrimitiveStringMultiline", "value"];
  const n = type && globalThis.LiteGraph?.createNode?.(type); if (!n) return false;
  n.pos = [x, y]; g.add(n);
  const w = n.widgets?.find(w => w.name === wname); if (w) setWidget(n, w, it.source ? loaderValue(it.source) : it.text);
  return true;
}

function injectStyle() {
  if (document.getElementById("kksh-style")) return;
  const s = document.createElement("style"); s.id = "kksh-style";
  s.textContent = `
@font-face{font-family:"Hanken Grotesk KKD";src:url("${HERE}fonts/HankenGrotesk-latin-var.woff2") format("woff2");font-weight:200 800;font-display:swap}
.kksh-hot{position:fixed;top:0;bottom:0;width:6px;z-index:10002}
.kksh-hot[data-side=right]{right:0}.kksh-hot[data-side=left]{left:0}
.kksh-shelf{position:fixed;top:56px;bottom:56px;width:212px;z-index:10003;display:flex;flex-direction:column;gap:6px;padding:10px;box-sizing:border-box;
  background:rgba(28,28,30,.97);color:#eee;font:400 12px "Hanken Grotesk KKD",system-ui,sans-serif;border:1px solid #3a3a3e;
  box-shadow:0 8px 30px rgba(0,0,0,.45);transition:transform .18s ease,opacity .18s ease;opacity:0;pointer-events:none;outline:none}
.kksh-shelf[data-side=right]{right:0;border-radius:10px 0 0 10px;transform:translateX(100%)}
.kksh-shelf[data-side=left]{left:0;border-radius:0 10px 10px 0;transform:translateX(-100%)}
.kksh-shelf.kksh-open{transform:none;opacity:1;pointer-events:auto}
.kksh-shelf.kksh-drop{border-color:#f18a58;box-shadow:0 0 0 2px #f18a58 inset,0 8px 30px rgba(0,0,0,.45)}
.kksh-head{display:flex;align-items:center;justify-content:space-between}
.kksh-title{font-size:14px;color:#ddd}.kksh-title b{color:#f18a58;font-weight:600}
.kksh-pills,.kksh-add{display:flex;flex-wrap:wrap;gap:4px}
.kksh-pill{background:#2c2c2e;color:#ccc;border:1px solid #444;border-radius:999px;padding:2px 9px;font:inherit;cursor:pointer}
.kksh-pill:hover{border-color:#f18a58;color:#fff}.kksh-pill[aria-pressed=true]{background:#a187b7;border-color:#a187b7;color:#111}
.kksh-list{flex:1;overflow-y:auto;display:flex;flex-direction:column;gap:6px;min-height:40px;scrollbar-width:thin}
.kksh-card{background:#232325;border:1px solid #38383c;border-radius:8px;overflow:hidden;cursor:grab;flex:none}
.kksh-card:hover{border-color:#f18a58}
.kksh-card img,.kksh-card video{display:block;width:100%;max-height:120px;object-fit:contain;background:#000}
.kksh-swatch{height:34px}.kksh-sound{height:34px;display:flex;align-items:center;justify-content:center;color:#a187b7;font-size:18px}
.kksh-text{padding:6px 8px;white-space:pre-wrap;max-height:84px;overflow:hidden;color:#ddd;line-height:1.35}
.kksh-foot{display:flex;align-items:center;gap:4px;padding:3px 6px;border-top:1px solid #333}
.kksh-name{flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;color:rgba(255,255,255,.6)}
.kksh-icon{background:none;border:0;color:#777;cursor:pointer;padding:0 3px;font:inherit;font-size:13px}
.kksh-icon:hover{color:#f18a58}.kksh-icon[aria-pressed=true]{color:#a187b7}
.kksh-edit{width:100%;box-sizing:border-box;background:#151517;color:#eee;border:1px solid #f18a58;border-radius:6px;font:inherit;padding:6px;resize:vertical}
.kksh-empty{color:rgba(255,255,255,.4);padding:16px 4px;text-align:center}
.kksh-msg{min-height:14px;color:#f18a58}
.kksh-hint{color:rgba(255,255,255,.35);font-size:11px;line-height:1.3}`;
  document.head.append(s);
}

// ------------------------------------------------------------------------------------------------ register
let shelf = null;
const getShelf = () => (shelf = shelf || new Shelf());
export const shelfOpen = () => !!shelf?.open;

app.registerExtension({
  name: "kubakub.shelf",
  settings: [{ id: EDGE_ID, category: ["kubakub", "shelf", "edge"], name: "kubakub shelf: screen edge", type: "combo",
    options: ["right", "left", "off"], defaultValue: "right", tooltip: "where the shelf slides out (Alt+S opens it too); off hides it",
    onChange: v => shelf?.edge(v) }],
  commands: [{ id: "kubakub.shelf.toggle", label: "kubakub shelf: open / close", function: () => getShelf().toggle() }],
  getNodeMenuItems(node) {
    return [null, { content: "put on the kubakub shelf", callback: () => getShelf().fromNode(node).catch(e => getShelf().status(e.message)) }];
  },
  setup() {
    getShelf();
    // Alt+S: its own listener, added before any director window, so it works above the director too (the window
    // takes every other key while it is open); never while typing in a text field
    window.addEventListener("keydown", e => {
      if (!e.altKey || e.ctrlKey || e.metaKey || e.shiftKey || e.code !== "KeyS" || e.repeat) return;
      if (e.target.matches?.("textarea, input:not([type=range]):not([type=checkbox]):not([type=color])") && !e.target.closest?.(".kksh-shelf")) return;
      e.preventDefault(); e.stopPropagation(); getShelf().toggle();
    }, true);
    // a shelf item dropped on the graph canvas (the canvas element only: the director, the timeline and text fields
    // take their own drops)
    const onCanvas = e => e.target === app.canvasEl || e.target === app.canvas?.canvas;
    const carries = e => [...(e.dataTransfer?.types || [])].includes(SHELF_MIME);
    document.addEventListener("dragover", e => { if (carries(e) && onCanvas(e)) { e.preventDefault(); e.dataTransfer.dropEffect = "copy"; } }, true);
    document.addEventListener("drop", e => {
      if (!carries(e) || !onCanvas(e)) return;
      e.preventDefault(); e.stopPropagation();
      try { const it = JSON.parse(e.dataTransfer.getData(SHELF_MIME)); if (!dropOnCanvas(it, e)) getShelf().status("that node takes no " + it.kind); }
      catch (err) { console.error("[kubakub shelf]", err); }
    }, true);
  },
});
