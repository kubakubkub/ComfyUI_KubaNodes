// 'pick regions': a button under every 'which regions' field. It opens the list of what the connected regions hold
// (their groups, floors, sections, single names), each with the number of regions; a click writes the selector text
// into the field, so nobody has to read a report to know the names. In kubakub region plan the same list adds a
// section header to the rules. The text field stays the truth: typing still works, and a workflow without this file
// runs the same. The names come from the pack's own route /kubakub/regions/seen (kubakub/regions_seen.py): the
// tables of the regions nodes that ran in this session, from memory; nothing is saved into the workflow.
// Plain ES module, no build step, no network. Look: kubakub.art (Hanken Grotesk, #f18a58 actions, #a187b7 selection).
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const HERE = new URL(".", import.meta.url).href;
const ORANGE = "#f18a58", LAV = "#a187b7";
// node -> its 'which regions' widget
const FIELDS = { KUBA_Shader: "only", KUBA_RegionSampler: "only", KUBA_RegionVideoSampler: "only", KUBA_Versions: "only",
  KUBA_MaskField: "select", KUBA_RegionsMask: "select", KUBA_ScenePieces: "selection", KUBA_RegionsToVector: "select",
  KUBA_RegionPrompt: "which" };
const PLAN = { KUBA_RegionPlan: "rules" };            // here the choice becomes a [section] of the rules
const ALL = { KUBA_MaskField: "*", KUBA_RegionsMask: "*" };   // what 'every region' is written as ("" elsewhere)
const CARRIES = ["KUBA_REGIONS", "KUBA_PLAN", "*"];   // the cables regions travel on
const MAX_TERMS = 48, NAMES_SHOWN = 60;

// ---- a choice <-> the selector text (kubakub/plan.py: comma = or, space = and)
// a choice = { groups: [name], rows: { label: [tag] }, names: [region name] }: any of the groups, and in every row
// with a tag chosen any of its tags; single names are added on top
const safe = v => String(v).trim().replace(/[\s,\[\]]/g, "?").replace(/\/\//g, "/?");   // a space or comma would split the term, '//' is a comment
const nameTerm = n => /^[\w.\-]+$/.test(n) && !["default", "all", "then", "plan"].includes(n.toLowerCase()) ? n : "name:" + safe(n);
export function termsOf(pick) {                       // -> the alternatives, each a text of terms
  let alts = pick.groups.length ? pick.groups.map(g => ["group:" + safe(g)]) : [[]];
  for (const tags of Object.values(pick.rows)) if (tags.length) alts = alts.flatMap(a => tags.map(t => [...a, "tag:" + safe(t)]));
  return [...alts.filter(a => a.length).map(a => a.join(" ")), ...pick.names.map(nameTerm)];
}
export const textOf = pick => termsOf(pick).join(", ");
export function pickOf(text, data) {                  // the choice a text means, or null when it was typed by hand
  const pick = { groups: [], rows: {}, names: [] }, t = String(text || "").trim();
  if (!t || t === "*") return pick;
  const low = s => String(s).toLowerCase(), add = (arr, v) => { if (!arr.includes(v)) arr.push(v); };
  const groups = new Map(data.groups.map(([g]) => [low(safe(g)), g])), tags = new Map(), names = new Map();
  for (const row of data.rows) for (const [n] of row.tags) tags.set(low(safe(n)), [row.label, n]);
  for (const [n] of data.regions) names.set(low(nameTerm(n)), n);
  const alts = t.split(/[,\n]/).map(s => s.trim().split(/\s+/).filter(Boolean));
  for (const toks of alts) {
    if (!toks.length) return null;
    if (toks.length === 1 && names.has(low(toks[0]))) { add(pick.names, names.get(low(toks[0]))); continue; }
    for (const tok of toks) {
      const l = low(tok);
      if (l.startsWith("group:") && groups.has(l.slice(6))) add(pick.groups, groups.get(l.slice(6)));
      else if (l.startsWith("tag:") && tags.has(l.slice(4))) { const [label, n] = tags.get(l.slice(4)); add(pick.rows[label] ||= [], n); }
      else return null;
    }
  }
  const norm = list => list.map(a => a.map(low).sort().join(" ")).sort().join("\n");   // the order of terms is free
  return norm(termsOf(pick).map(s => s.split(" "))) === norm(alts) ? pick : null;
}
export function chosen(pick, data) {                  // the regions a choice takes
  const rows = Object.values(pick.rows).filter(a => a.length), any = pick.groups.length || rows.length;
  return data.regions.filter(([name, group, tags]) => pick.names.includes(name) ||
    (any && (!pick.groups.length || pick.groups.some(g => g.toLowerCase() === group.toLowerCase())) && rows.every(a => a.some(t => tags.includes(t)))));
}

// ---- which regions node feeds this node: the nodes up the regions / plan cables, the nearest first
function upstream(node) {
  const g = node.graph, seen = new Set([node.id]), out = [], queue = [node];
  const link = id => g?.getLink?.(id) ?? g?.links?.get?.(id) ?? g?.links?.[id];
  while (queue.length && out.length < 200) {
    const n = queue.shift();
    for (const inp of n.inputs || []) {
      if (inp.link == null || !CARRIES.includes(String(inp.type))) continue;
      const l = link(inp.link), src = l && g.getNodeById?.(l.origin_id);
      if (!src || seen.has(src.id)) continue;
      seen.add(src.id); out.push(String(src.id)); queue.push(src);
    }
  }
  return out;
}

let styled = false;
function style() {
  if (styled) return;
  styled = true;
  const el = document.createElement("style");
  el.textContent = `
@font-face{font-family:"Hanken Grotesk KKD";src:url("${HERE}fonts/HankenGrotesk-latin-var.woff2") format("woff2");font-weight:200 800;font-display:swap}
.kkw{position:fixed;z-index:10000;width:min(460px,calc(100vw - 16px));max-height:min(620px,calc(100vh - 16px));display:flex;flex-direction:column;gap:8px;padding:12px 14px;box-sizing:border-box;
  background:#1b1b1b;border:1px solid rgba(255,255,255,.14);border-radius:12px;box-shadow:0 12px 40px rgba(0,0,0,.6);
  font-family:"Hanken Grotesk KKD",system-ui,sans-serif;font-size:12px;color:rgba(255,255,255,.88)}
.kkw-head{display:flex;align-items:baseline;gap:8px}
.kkw-head b{font-weight:700;font-size:13px;text-transform:lowercase}
.kkw-head span{flex:1;color:rgba(255,255,255,.55);text-transform:lowercase}
.kkw-body{display:flex;flex-direction:column;gap:8px;overflow-y:auto;min-height:0}
.kkw-row{display:flex;flex-wrap:wrap;gap:4px;align-items:center}
.kkw-row>i{font-style:normal;flex:0 0 100%;color:rgba(255,255,255,.5);text-transform:lowercase;font-size:11px}
.kkw button{font:inherit;border:0;border-radius:10em;padding:3px 10px;background:rgba(255,255,255,.12);color:rgba(255,255,255,.88);cursor:pointer}
.kkw button small{font-size:10px;opacity:.6;margin-left:6px}
.kkw button:hover{background:rgba(255,255,255,.2)}
.kkw button.on{background:${LAV};color:#141414;font-weight:700}
.kkw button.act{text-transform:lowercase;background:transparent;box-shadow:inset 0 0 0 1px rgba(255,255,255,.3)}
.kkw button.act:hover{box-shadow:inset 0 0 0 1px ${ORANGE};color:${ORANGE}}
.kkw button.go{text-transform:lowercase;background:${ORANGE};color:#141414;font-weight:700}
.kkw button:disabled{opacity:.4;cursor:default}
.kkw input{font:inherit;flex:1;min-width:120px;background:rgba(255,255,255,.08);border:0;border-radius:10em;padding:4px 11px;color:inherit;outline:none}
.kkw input:focus{box-shadow:inset 0 0 0 1px ${LAV}}
.kkw-note{color:rgba(255,255,255,.55);line-height:1.45}
.kkw-text{font-family:ui-monospace,Consolas,monospace;font-size:11px;color:rgba(255,255,255,.75);background:rgba(255,255,255,.06);border-radius:8px;padding:6px 9px;word-break:break-word;max-height:66px;overflow-y:auto}`;
  document.head.appendChild(el);
}

const h = (tag, cls, text) => { const e = document.createElement(tag); if (cls) e.className = cls; if (text != null) e.textContent = text; return e; };
let open = null;                                       // the one list on screen
function shut() {
  if (!open) return;
  document.removeEventListener("keydown", open.key, true);
  document.removeEventListener("pointerdown", open.down, true);
  open.el.remove();
  open = null;
}

function place(el, node) {                             // beside the node, inside the window
  const c = app.canvas, box = c?.canvas?.getBoundingClientRect?.() || { left: 0, top: 0 };
  let x = innerWidth / 2 - 230, y = 80;
  if (c?.ds && node.pos) {
    x = box.left + (node.pos[0] + node.size[0] + c.ds.offset[0]) * c.ds.scale + 10;
    y = box.top + (node.pos[1] + c.ds.offset[1]) * c.ds.scale;
    if (x + el.offsetWidth > innerWidth - 8) x = box.left + (node.pos[0] + c.ds.offset[0]) * c.ds.scale - el.offsetWidth - 10;   // no room: on its left
  }
  el.style.left = Math.max(8, Math.min(x, innerWidth - el.offsetWidth - 8)) + "px";
  el.style.top = Math.max(8, Math.min(y, innerHeight - el.offsetHeight - 8)) + "px";
}

async function show(node, field, plan) {
  style();
  shut();
  const el = h("div", "kkw"), head = h("div", "kkw-head"), body = h("div", "kkw-body");
  const count = h("span", "", ""), done = h("button", "act", "close");
  head.append(h("b", "", plan ? "a section for which regions" : "which regions"), count, done);
  el.append(head, body);
  document.body.appendChild(el);
  open = { el, node, key: e => { if (e.key === "Escape") { e.stopPropagation(); shut(); } }, down: e => { if (!el.contains(e.target)) shut(); } };
  document.addEventListener("keydown", open.key, true);
  setTimeout(() => open?.el === el && document.addEventListener("pointerdown", open.down, true), 0);
  done.addEventListener("click", shut);
  const say = text => { body.replaceChildren(h("div", "kkw-note", text)); place(el, node); };
  place(el, node);

  const widget = node.widgets?.find(w => w.name === field);
  if (!plan && node.inputs?.find(i => i.name === field)?.link != null)
    return say("This field is fed by a cable, so the choice is made in the node the cable comes from (e.g. kubakub regions to mask).");
  const ids = upstream(node);
  if (!ids.length) return say("Connect regions (or a plan) to this node first: the list shows what those regions hold.");
  say("reading…");
  let data = null;
  try {
    const wf = String(app.rootGraph?.id ?? app.graph?.id ?? "");          // node 2 of another workflow is not this one
    const r = await api.fetchApi("/kubakub/regions/seen?nodes=" + encodeURIComponent(ids.join(",")) + "&workflow=" + encodeURIComponent(wf));
    data = await r.json();
  } catch (e) {
    console.warn("[kubakub which regions]", e);
    if (open?.el === el) say("The list needs a restart of ComfyUI (type the names meanwhile).");
    return;
  }
  if (open?.el !== el) return;
  if (!data?.node || !Array.isArray(data.regions))
    return say("The names are known once the regions node has run in this session. Queue the workflow once (it may stop at a later node, that is fine), then pick here.");

  const value = () => String(widget?.value ?? "");
  let pick = plan ? { groups: [], rows: {}, names: [] } : pickOf(value(), data);
  const typed = !plan && pick === null ? value().trim() : "";       // a text typed by hand: shown, replaced by the first click
  pick ||= { groups: [], rows: {}, names: [] };
  let filter = "";
  const write = () => {
    if (plan || !widget) return;
    const v = termsOf(pick).length ? textOf(pick) : (ALL[node.comfyClass || node.type] ?? "");
    widget.value = v;
    widget.callback?.(v);
    if (widget.inputEl && "value" in widget.inputEl) widget.inputEl.value = v;
    node.setDirtyCanvas?.(true, true);
  };
  const toggle = (arr, v) => { const i = arr.indexOf(v); if (i < 0) arr.push(v); else arr.splice(i, 1); };
  const chip = (label, n, on, click, tip) => {
    const b = h("button", on ? "on" : "", label);
    if (n != null) b.append(h("small", "", String(n)));
    if (tip) b.title = tip;
    b.addEventListener("click", () => { click(); write(); render(true); });
    return b;
  };
  const text = h("div", "kkw-text"), note = h("div", "kkw-note"), search = h("input");
  search.placeholder = "a single region: type part of its name";
  search.addEventListener("input", () => { filter = search.value.trim().toLowerCase(); render(true); });

  function render(touched) {
    const terms = termsOf(pick), hit = chosen(pick, data), rows = [];
    if (data.groups.length) {
      const row = h("div", "kkw-row");
      row.append(h("i", "", "groups"), ...data.groups.map(([g, n]) => chip(g, n, pick.groups.includes(g), () => toggle(pick.groups, g))));
      rows.push(row);
    }
    for (const r of data.rows) {
      const row = h("div", "kkw-row"), list = pick.rows[r.label] ||= [];
      row.append(h("i", "", r.label), ...r.tags.map(([t, n]) => chip(t, n, list.includes(t), () => toggle(list, t))));
      rows.push(row);
    }
    const names = h("div", "kkw-row"), all = data.regions.map(r => r[0]);
    const found = filter ? all.filter(n => n.toLowerCase().includes(filter)) : pick.names.slice();
    names.append(h("i", "", "single regions"), search,
      ...[...new Set([...pick.names, ...found])].slice(0, NAMES_SHOWN).map(n => chip(n, null, pick.names.includes(n), () => toggle(pick.names, n))));
    if (found.length > NAMES_SHOWN) names.append(h("span", "kkw-note", `${found.length - NAMES_SHOWN} more: type more of the name`));
    rows.push(names);
    for (const [label, n, example] of data.many || [])
      rows.push(h("div", "kkw-note", `${label}: ${n} names, too many to list. Type one into the field, e.g. tag:${example}`));

    const focused = document.activeElement === search;
    body.replaceChildren(...rows, plan ? planFoot(terms) : text, note);
    if (focused) search.focus();
    const total = data.count ?? data.regions.length;
    count.textContent = terms.length ? `${hit.length} of ${total} regions` : typed && !touched ? "typed by hand" : `all ${total} regions`;
    text.textContent = typed && !touched ? typed : terms.length ? textOf(pick) : (ALL[node.comfyClass || node.type] ?? "") || "(empty = every region)";
    note.textContent = terms.length > MAX_TERMS ? `That is ${terms.length} combinations in the text: choose fewer in one of the rows.`
      : terms.length && !hit.length ? "No region is all of that at once: within a row any choice counts, across rows all of them."
      : typed && !touched ? "This was typed by hand and is kept as it is. A click on a name replaces it."
      : "Within a row any choice counts (windows or columns); across rows all of them (windows on floor 1).";
    if (!touched) place(el, node);
  }

  function planFoot(terms) {                            // kubakub region plan: the choice as a [section]
    const foot = h("div", "kkw-row"), header = `[${terms.length ? textOf(pick) : "default"}]`;
    const add = h("button", "go", "add this section to the rules");
    add.disabled = terms.length > MAX_TERMS;
    add.addEventListener("click", () => {
      if (!widget) return;
      const box = widget.inputEl || widget.element, cur = value();
      const at = typeof box?.selectionStart === "number" && document.body.contains(box) ? box.selectionStart : cur.length;
      const next = cur.slice(at).search(/\n[ \t]*\[/);                                   // after the section the cursor is in,
      const end = next < 0 ? cur.length : at + next + 1;                                 // so no setting changes its owner
      const before = cur.slice(0, end), lead = !before || before.endsWith("\n\n") ? "" : before.endsWith("\n") ? "\n" : "\n\n";
      const v = before + lead + header + "\nprompt = \n" + (cur.slice(end) ? "\n" + cur.slice(end).replace(/^\n+/, "") : "");
      widget.value = v;
      widget.callback?.(v);
      if (box && "value" in box) { box.value = v; const p = (before + lead + header + "\nprompt = ").length; box.focus?.(); box.setSelectionRange?.(p, p); }
      node.setDirtyCanvas?.(true, true);
      shut();
    });
    const t = h("div", "kkw-text", header);
    t.style.flex = "1 1 100%";
    foot.append(t, add);
    return foot;
  }
  render(false);
}

function addButton(node, field, plan) {
  const target = node.widgets?.find(w => w.name === field);
  if (!target || node.widgets.some(w => w.name === "pick regions")) return;
  const b = node.addWidget("button", "pick regions", null, () => show(node, field, plan), { serialize: false });
  b.label = plan ? "add a section: pick its regions from a list" : "pick regions: groups, floors, sections";
  b.tooltip = plan ? "Opens the list of what your regions hold (groups, floors, sections, single names) and adds a [section] for the ones you choose to the rules, with an empty prompt to fill in."
    : "Opens the list of what the connected regions hold (groups, floors, sections, single names), each with its number of regions. A click writes the choice into 'which regions'. The names are known once the regions node has run.";
  b.serialize = false;                                 // nothing of its own to save: the text field holds the choice
  node.widgets.splice(node.widgets.indexOf(b), 1);
  node.widgets.splice(node.widgets.indexOf(target) + 1, 0, b);
}

app.registerExtension({
  name: "kubakub.which_regions",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    const field = FIELDS[nodeData.name] || PLAN[nodeData.name];
    if (!field) return;
    const created = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = created?.apply(this, arguments);
      addButton(this, field, !!PLAN[nodeData.name]);
      return r;
    };
    const removed = nodeType.prototype.onRemoved;
    nodeType.prototype.onRemoved = function () {
      if (open && this === open.node) shut();
      return removed?.apply(this, arguments);
    };
  },
});
