// kubakub library: the searchable store behind the kubakub shelf (the shelf's "library" pill). Every picture, video
// and sound that reaches the shelf, a dropped folder, the runs of this ComfyUI session ("last generations") and any
// folder of this computer ("index a folder", read in place) is listed with what the file says about how it was made:
// the prompts, seed, steps, cfg, sampler, models and LoRAs. Search, filter, inspect, copy a prompt, drag it onto a
// node, open the file's workflow, put it on the shelf. The database is user/default/kubakub/library.db (routes in
// nodes/utils/nodes_library.py). Nothing leaves this computer. No extension of its own: the shelf builds it.
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { DRAG_MIME, sourceURL, historyFiles, outputSource, h } from "./kubakub_media.js";

const KINDS = [["all", ""], ["pictures", "image"], ["videos", "video"], ["sounds", "audio"]];
const SORTS = [["newest", "newest"], ["oldest", "oldest"], ["name", "name"]];
const PAGE = 120;
const json = async r => { const j = await r.json().catch(() => ({})); if (!r.ok) throw new Error(j.error || r.statusText); return j; };
const get = (path, q = {}) => api.fetchApi(path + "?" + new URLSearchParams(q)).then(json);
const post = (path, body) => api.fetchApi(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) }).then(json);
const fileURL = it => it.source ? sourceURL(it.source) : api.apiURL(`/kubakub/library/file?id=${it.id}`);
const thumbURL = it => api.apiURL(`/kubakub/library/thumb?id=${it.id}&v=${Math.round(it.mtime || 0)}`);
const short = n => String(n || "").split(/[\\/]/).pop();
const size = b => b > 1 << 30 ? (b / (1 << 30)).toFixed(1) + " GB" : b > 1 << 20 ? (b / (1 << 20)).toFixed(1) + " MB" : Math.max(1, Math.round(b / 1024)) + " kB";

// the library's own requests, for the shelf (no view needed)
export const library = {
  add: (body) => post("/kubakub/library/add", body),
  // this session's runs (ComfyUI's /history) -> into the library; -> {files, items, new}
  async lastGenerations() {
    const hist = await api.fetchApi("/history?max_items=10000").then(json);
    const files = historyFiles(hist);
    if (!files.length) return { files, items: [], new: 0 };
    const res = await post("/kubakub/library/add", { files, tags: ["this session"] });
    return { files, items: res.items || [], new: res.new || 0 };
  },
};

export class LibraryView {
  // shelf: the kubakub shelf (status(), add(), the drag flag and the shelf item format)
  constructor(shelf) {
    this.shelf = shelf; this.q = { q: "", kind: "", model: "", lora: "", tag: "", sort: "newest" };
    this.items = []; this.total = 0; this.facets = null; this.timer = null; this.follow = false; this.seq = 0;
    this.search = h("input", { class: "kkl-search", type: "search", placeholder: "search prompts, names, models, tags",
      title: "words from a prompt, a file name, a model or LoRA, a tag; several words = all of them" });
    this.search.addEventListener("input", () => { clearTimeout(this.timer); this.timer = setTimeout(() => { this.q.q = this.search.value; this.load(); }, 220); });
    this.search.addEventListener("keydown", e => e.stopPropagation());
    const pills = (list, key) => h("div", { class: "kksh-pills" }, list.map(([n, v]) => h("button", { class: "kksh-pill", "data-key": key, "data-v": v,
      onclick: () => { this.q[key] = v; this.load(); } }, n)));
    this.kindPills = pills(KINDS, "kind"); this.sortPills = pills(SORTS, "sort");
    this.pick = { model: this.select("model", "every model"), lora: this.select("lora", "every lora"), tag: this.select("tag", "every tag / folder") };
    this.followBtn = h("button", { class: "kksh-pill", "aria-pressed": "false", title: "every picture, video or sound a run saves from now on goes into the library and onto the shelf",
      onclick: () => this.setFollow(!this.follow) }, "keep adding new ones");
    this.folderIn = h("input", { class: "kkl-search", placeholder: "a folder of this computer · empty = the output folder", title: "read in place, nothing is copied; its name becomes a tag. Left empty: everything in ComfyUI's output folder" });
    this.folderIn.addEventListener("keydown", e => { e.stopPropagation(); if (e.key === "Enter") this.indexFolder(); });
    this.roots = h("div", { class: "kkl-roots" });
    this.folderRow = h("div", { class: "kkl-folder", hidden: true }, h("div", { class: "kkl-row" }, this.folderIn,
      h("button", { class: "kksh-pill", onclick: () => this.indexFolder() }, "index")), this.roots);
    this.grid = h("div", { class: "kkl-grid" });
    this.count = h("div", { class: "kkl-count" });
    this.more = h("button", { class: "kksh-pill", hidden: true, onclick: () => this.load(true) }, "more");
    this.browse = h("div", { class: "kkl-browse" },
      h("div", { class: "kkl-row" }, this.search),
      h("div", { class: "kkl-row" }, this.kindPills, h("span", { class: "kkl-sep" }), this.sortPills),
      h("div", { class: "kkl-row" }, this.pick.model, this.pick.lora, this.pick.tag),
      h("div", { class: "kkl-row" },
        h("button", { class: "kksh-pill", title: "everything the runs of this ComfyUI session saved (since it started), newest first", onclick: () => this.lastGenerations() }, "last generations"),
        this.followBtn,
        h("button", { class: "kksh-pill", title: "add a whole folder (an output folder, a project's renders) without copying it", onclick: () => { this.folderRow.hidden = !this.folderRow.hidden; if (!this.folderRow.hidden) this.folderIn.focus(); } }, "index a folder")),
      this.folderRow, this.count, this.grid, this.more);
    this.inspector = h("div", { class: "kkl-inspect", hidden: true });
    this.root = h("div", { class: "kkl" }, this.browse, this.inspector);
    this.onExecuted = ev => this.executed(ev.detail);
  }
  select(key, all) {
    const s = h("select", { class: "kkl-select", title: `only files made with this ${key === "tag" ? "tag (a dropped or indexed folder, \"this session\" ...)" : key}` });
    s.addEventListener("change", () => { this.q[key] = s.value; this.load(); });
    s.dataset.all = all; return s;
  }
  // ---- the list
  async open() { await this.loadFacets(); await this.load(); }
  async loadFacets() {
    try { this.facets = await get("/kubakub/library/facets"); } catch (e) { this.shelf.status(e.message); return; }
    for (const [key, list] of [["model", this.facets.models], ["lora", this.facets.loras], ["tag", this.facets.tags]]) {
      const s = this.pick[key], cur = this.q[key];
      s.replaceChildren(h("option", { value: "" }, s.dataset.all), ...list.slice(0, 300).map(o => h("option", { value: o.name }, `${short(o.name)} · ${o.count}`)));
      if (cur && !list.some(o => o.name === cur)) s.append(h("option", { value: cur }, short(cur)));
      s.value = cur; s.hidden = !list.length && !cur;
    }
    this.roots.replaceChildren(...this.facets.roots.map(r => h("div", { class: "kkl-rootline" }, h("span", { title: r }, r),
      h("button", { class: "kksh-icon", title: "index it again (new and changed files)", onclick: () => { this.folderIn.value = r; this.indexFolder(); } }, "↻"),
      h("button", { class: "kksh-icon", title: "take it out of the library (the files stay on disk)", onclick: () => this.forget(r) }, "×"))));
  }
  async load(more = false) {
    for (const b of this.root.querySelectorAll("[data-key]")) b.setAttribute("aria-pressed", String(this.q[b.dataset.key] === b.dataset.v));
    for (const k of ["model", "lora", "tag"]) if (this.pick[k].value !== this.q[k]) { this.pick[k].value = this.q[k]; }
    const seq = ++this.seq;
    try {
      const r = await get("/kubakub/library/search", { ...this.q, offset: more ? this.items.length : 0, limit: PAGE });
      if (seq !== this.seq) return;
      this.items = more ? this.items.concat(r.items) : r.items; this.total = r.total;
    } catch (e) { this.shelf.status(e.message); return; }
    const all = this.facets?.count ?? this.total;
    this.count.textContent = this.total === all ? `${all} in the library` : `${this.total} of ${all}`;
    if (!more) this.grid.replaceChildren();
    const start = more ? this.grid.children.length : 0;
    this.grid.append(...this.items.slice(start).map(it => this.card(it)));
    if (!this.items.length) this.grid.replaceChildren(h("div", { class: "kksh-empty" }, all ? "nothing matches" :
      "empty: drop files or a folder on the shelf, press last generations, or index a folder"));
    this.more.hidden = this.items.length >= this.total;
  }
  card(it) {
    let face;
    if (it.kind === "audio") face = h("div", { class: "kksh-sound" }, "♪");
    else { face = h("img", { src: thumbURL(it), loading: "lazy", draggable: "false", alt: "" }); face.onerror = () => face.replaceWith(h("div", { class: "kksh-sound" }, it.kind === "video" ? "▶" : "?")); }
    const c = h("div", { class: "kkl-card", draggable: it.source ? "true" : "false", "data-kind": it.kind,
      title: `${it.name}${it.positive ? "\n" + it.positive.slice(0, 200) : ""}\nclick: inspect${it.source ? " · drag it out" : ""}` },
      face, it.kind === "video" ? h("span", { class: "kkl-badge" }, "▶") : null, h("div", { class: "kkl-name" }, it.name));
    c.addEventListener("click", () => this.inspect(it.id));
    if (it.source) this.dragOut(c, () => ({ kind: it.kind, source: it.source, name: it.name }));
    return c;
  }
  // a file (with its source) or a prompt dragged out: the shelf's drag format, so the timeline, a director, a node,
  // the canvas and text boxes all take it
  dragOut(el, item) {
    el.addEventListener("dragstart", e => {
      const it = item(); this.shelf.dragging = true; const dt = e.dataTransfer; dt.effectAllowed = "copy";
      dt.setData(this.shelf.mime, JSON.stringify(it));
      if (it.source) { dt.setData(DRAG_MIME, JSON.stringify({ source: it.source })); dt.setData("text/plain", it.name); } else dt.setData("text/plain", it.text);
      e.stopPropagation();
    });
    el.addEventListener("dragend", () => { this.shelf.dragging = false; });
  }
  // ---- inspect
  async inspect(id) {
    let it; try { it = await get("/kubakub/library/item", { id }); } catch (e) { return this.shelf.status(e.message); }
    let big;
    if (it.kind === "image") big = h("img", { src: fileURL(it), alt: "" });
    else if (it.kind === "video") { big = h("video", { src: fileURL(it), controls: true, loop: true, preload: "metadata" }); big.muted = true; }
    else big = h("audio", { src: fileURL(it), controls: true });
    const prompt = (label, text) => {
      if (!text) return null;
      const b = h("div", { class: "kkl-prompt", draggable: "true", title: "click: copy · drag onto a node, the canvas or a text box" }, text);
      b.addEventListener("click", () => navigator.clipboard?.writeText(text).then(() => this.shelf.status("copied"), () => {}));
      this.dragOut(b, () => ({ kind: "text", text, name: text.split("\n")[0].slice(0, 60) }));
      return h("div", { class: "kkl-block" }, h("div", { class: "kkl-label" }, label,
        h("button", { class: "kksh-icon", title: "put this prompt on the shelf", onclick: () => this.shelf.keepText(text) }, "+ shelf")), b);
    };
    const chip = (text, key, value, title) => h("button", { class: "kkl-chip", title, onclick: () => { this.q = { q: "", kind: "", model: "", lora: "", tag: "", sort: this.q.sort, [key]: value }; this.search.value = ""; this.close(); this.load(); } }, text);
    const rows = [["seed", it.seed], ["steps", it.steps], ["cfg", it.cfg], ["sampler", it.sampler], ["scheduler", it.scheduler],
      ["denoise", it.denoise], ["size", it.w && it.h ? `${it.w} × ${it.h}` : null], ["length", it.duration ? `${it.duration} s` : null],
      ["file", `${size(it.size)} · ${new Date((it.mtime || 0) * 1000).toLocaleString()}`]].filter(([, v]) => v != null && v !== "");
    const newTag = h("input", { class: "kkl-search kkl-tagin", placeholder: "+ tag" });
    newTag.addEventListener("keydown", async e => { e.stopPropagation(); if (e.key === "Enter" && newTag.value.trim()) { await post("/kubakub/library/tag", { id: it.id, tag: newTag.value.trim(), on: true }).catch(() => {}); await this.loadFacets(); this.inspect(it.id); } });
    this.inspector.replaceChildren(...[
      h("div", { class: "kkl-row" }, h("button", { class: "kksh-pill", onclick: () => this.close() }, "‹ back"), h("span", { class: "kkl-title", title: it.name }, it.name)),
      h("div", { class: "kkl-big" }, big),
      h("div", { class: "kkl-row" },
        h("button", { class: "kksh-pill", title: it.has_workflow ? "open the workflow this file was made with (as when the file is dropped on ComfyUI)" : "this file carries no workflow", disabled: !it.has_workflow && it.source_app !== "a1111", onclick: () => this.loadWorkflow(it) }, "load workflow"),
        h("button", { class: "kksh-pill", title: it.source ? "keep it on the shelf strip" : "copies it into input/kuba_director, then keeps it on the shelf strip", onclick: () => this.toShelf(it) }, "to shelf"),
        h("button", { class: "kksh-pill", title: it.path, onclick: () => post("/kubakub/library/reveal", { id: it.id }).catch(e => this.shelf.status(e.message)) }, "show in folder")),
      h("div", { class: "kkl-path", title: "click: copy the path", onclick: () => navigator.clipboard?.writeText(it.path).then(() => this.shelf.status("path copied"), () => {}) }, it.path),
      prompt("prompt", it.positive), prompt("negative", it.negative),
      !it.positive && !it.negative ? h("div", { class: "kkl-none" }, it.has_workflow ? "no prompt text found in its graph" : "this file says nothing about how it was made") : null,
      h("div", { class: "kkl-table" }, rows.flatMap(([k, v]) => [h("span", { class: "kkl-k" }, k), h("span", { class: "kkl-v" }, String(v))])),
      it.models.length ? h("div", { class: "kkl-block" }, h("div", { class: "kkl-label" }, "models"), h("div", { class: "kkl-chips" }, it.models.map(m => chip(short(m), "model", m, "every file made with " + m)))) : null,
      it.loras.length ? h("div", { class: "kkl-block" }, h("div", { class: "kkl-label" }, "loras"), h("div", { class: "kkl-chips" }, it.loras.map(l => chip(`${short(l.name)}${l.strength != null ? " · " + l.strength : ""}`, "lora", l.name, "every file made with " + l.name)))) : null,
      h("div", { class: "kkl-block" }, h("div", { class: "kkl-label" }, "tags"), h("div", { class: "kkl-chips" },
        it.tags.map(t => h("span", { class: "kkl-chip" }, h("span", { onclick: () => { this.q = { q: "", kind: "", model: "", lora: "", tag: t, sort: this.q.sort }; this.search.value = ""; this.close(); this.load(); } }, t),
          h("button", { class: "kksh-icon", title: "remove this tag", onclick: async () => { await post("/kubakub/library/tag", { id: it.id, tag: t, on: false }).catch(() => {}); await this.loadFacets(); this.inspect(it.id); } }, "×"))), newTag))].filter(Boolean));
    this.browse.hidden = true; this.inspector.hidden = false; this.inspector.scrollTop = 0;
  }
  close() { this.inspector.querySelectorAll("video,audio").forEach(m => m.pause()); this.inspector.hidden = true; this.browse.hidden = false; }
  async loadWorkflow(it) {
    try {
      // the core way: the file itself handed to ComfyUI, as when it is dropped on the canvas (a ComfyUI workflow, an
      // API prompt or A1111 parameters)
      const r = await fetch(fileURL(it)); if (!r.ok) throw new Error("the file is not there any more");
      const blob = await r.blob();
      if (app.handleFile) await app.handleFile(new File([blob], it.name, { type: blob.type }));
      else { const wf = await get("/kubakub/library/workflow", { id: it.id }); if (wf.workflow) await app.loadGraphData(wf.workflow); else if (wf.prompt) app.loadApiJson?.(wf.prompt, it.name); }
      this.shelf.status("opened its workflow");
    } catch (e) { this.shelf.status(e.message); }
  }
  async toShelf(it) {
    try {
      const { source } = it.source ? it : await post("/kubakub/library/to_input", { id: it.id });
      this.shelf.add({ kind: it.kind, source, name: source.split("/").pop() }); this.shelf.status("on the shelf");
    } catch (e) { this.shelf.status(e.message); }
  }
  // ---- adding
  async lastGenerations() {
    this.shelf.status("reading this session's runs …");
    try {
      const r = await library.lastGenerations();
      if (!r.files.length) return this.shelf.status("this session has saved nothing yet (previews are not kept)");
      this.shelf.addMany(r.files.slice(0, 60).reverse().map(f => ({ kind: null, source: outputSource(f), name: f.filename })));
      this.shelf.status(`added ${r.new} from this session (${r.files.length} saved in all${r.files.length > 60 ? ", the newest 60 on the strip" : ""})`);
      this.q = { q: "", kind: "", model: "", lora: "", tag: "this session", sort: "newest" }; this.search.value = "";
      await this.loadFacets(); await this.load();
    } catch (e) { this.shelf.status(e.message); }
  }
  setFollow(on) {
    this.follow = on; this.followBtn.setAttribute("aria-pressed", String(on));
    api.removeEventListener("executed", this.onExecuted); if (on) api.addEventListener("executed", this.onExecuted);
    this.shelf.saveState?.();
  }
  async executed(detail) {
    const files = historyFiles({ x: { outputs: { n: detail?.output || {} } } });
    if (!files.length) return;
    try {
      await library.add({ files, tags: ["this session"] });
      this.shelf.addMany(files.slice().reverse().map(f => ({ kind: null, source: outputSource(f), name: f.filename })));
      if (!this.root.hidden && this.root.isConnected) { await this.loadFacets(); if (this.inspector.hidden) await this.load(); }
    } catch { /* the next run tries again */ }
  }
  async indexFolder() {
    const path = this.folderIn.value.trim();          // empty: ComfyUI's own output folder
    this.shelf.status("indexing " + (path || "the output folder") + " … (a big folder takes a while)");
    try {
      const r = await post("/kubakub/library/folder", { path });
      this.shelf.status(`${r.total} files in ${short(r.path)}${r.capped ? " (stopped at " + r.total + ": index its sub folders one by one)" : ""}`);
      this.folderIn.value = ""; this.q = { q: "", kind: "", model: "", lora: "", tag: r.tag || short(r.path), sort: "newest" }; this.search.value = "";
      await this.loadFacets(); await this.load();
    } catch (e) { this.shelf.status(e.message); }
  }
  async forget(path) {
    try { const r = await post("/kubakub/library/forget", { path }); this.shelf.status(`${r.removed} files out of the library (still on disk)`); if (this.q.tag === short(path)) this.q.tag = ""; await this.loadFacets(); await this.load(); }
    catch (e) { this.shelf.status(e.message); }
  }
  showTag(tag) { this.q = { q: "", kind: "", model: "", lora: "", tag, sort: "newest" }; this.search.value = ""; this.close(); return this.open(); }
}

export const LIBRARY_CSS = `
.kksh-shelf.kksh-wide{width:min(620px,92vw)}
.kkl{display:flex;flex-direction:column;flex:1;min-height:0}
.kkl-browse,.kkl-inspect{display:flex;flex-direction:column;gap:6px;flex:1;min-height:0;overflow-y:auto;scrollbar-width:thin}
.kkl-browse[hidden],.kkl-inspect[hidden],.kkl-folder[hidden]{display:none}
.kkl-row{display:flex;flex-wrap:wrap;gap:4px;align-items:center}
.kkl-sep{width:8px}
.kkl-search{flex:1;min-width:120px;background:#151517;color:#eee;border:1px solid #444;border-radius:999px;padding:4px 10px;font:inherit;outline:none}
.kkl-search:focus{border-color:#f18a58}
.kkl-select{flex:1;min-width:110px;max-width:190px;background:#2c2c2e;color:#ccc;border:1px solid #444;border-radius:999px;padding:2px 6px;font:inherit}
.kkl-select[hidden]{display:none}
.kkl-folder{display:flex;flex-direction:column;gap:4px;padding:6px;border:1px dashed #444;border-radius:8px}
.kkl-rootline{display:flex;gap:4px;align-items:center;color:rgba(255,255,255,.6)}.kkl-rootline span{flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.kkl-count{color:rgba(255,255,255,.45);font-size:11px}
.kkl-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(108px,1fr));gap:6px}
.kkl-card{position:relative;background:#232325;border:1px solid #38383c;border-radius:8px;overflow:hidden;cursor:pointer}
.kkl-card:hover{border-color:#f18a58}.kkl-card[draggable=true]{cursor:grab}
.kkl-card img{display:block;width:100%;height:96px;object-fit:cover;background:#000}
.kkl-card .kksh-sound{height:96px}
.kkl-badge{position:absolute;top:4px;left:4px;background:rgba(0,0,0,.6);color:#a187b7;border-radius:4px;padding:0 4px;font-size:10px}
.kkl-name{padding:2px 6px;color:rgba(255,255,255,.6);font-size:11px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.kkl-title{flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;color:#ddd}
.kkl-big img,.kkl-big video{display:block;width:100%;max-height:46vh;object-fit:contain;background:#000;border-radius:6px}.kkl-big audio{width:100%}
.kkl-path{color:rgba(255,255,255,.4);font-size:11px;word-break:break-all;cursor:copy}
.kkl-block{display:flex;flex-direction:column;gap:3px}
.kkl-label{display:flex;justify-content:space-between;color:#a187b7;font-size:11px}
.kkl-prompt{background:#151517;border:1px solid #38383c;border-radius:6px;padding:6px 8px;white-space:pre-wrap;line-height:1.35;color:#ddd;cursor:copy;max-height:220px;overflow-y:auto}
.kkl-prompt:hover{border-color:#f18a58}
.kkl-none{color:rgba(255,255,255,.4)}
.kkl-table{display:grid;grid-template-columns:auto 1fr;gap:2px 10px}.kkl-k{color:rgba(255,255,255,.45)}.kkl-v{color:#ddd;word-break:break-all}
.kkl-chips{display:flex;flex-wrap:wrap;gap:4px;align-items:center}
.kkl-chip{background:#2c2c2e;color:#ccc;border:1px solid #444;border-radius:999px;padding:1px 8px;font:inherit;cursor:pointer;display:inline-flex;gap:4px;align-items:center}
.kkl-chip:hover{border-color:#f18a58;color:#fff}
.kkl-tagin{flex:0 1 110px;min-width:80px;padding:1px 8px}
.kksh-card[data-kind=set] .kksh-setcount{position:absolute;right:6px;top:6px;background:rgba(0,0,0,.65);color:#f18a58;border-radius:4px;padding:0 5px}
.kksh-card[data-kind=set]{position:relative;cursor:pointer}`;
