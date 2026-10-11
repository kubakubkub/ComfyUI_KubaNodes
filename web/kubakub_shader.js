// kubakub shader: pick a preset by its picture, and see at once whether pasted code can run.
// A "pictures" button right under `preset` opens a sheet of all presets, sorted by type (patterns, 3d scenes, picture
// looks ...), where a tile plays under the pointer; a click sets the node's own `preset` widget,
// and an "adjust" button opens the shader's controls as sliders, with its code and "save as my preset" (see adjustShader);
// so the widget stays the truth and a workflow without this file still runs. A read-only line under it says what the
// node will draw (the preset, or your code and what it reads) or why pasted code cannot run, before you queue.
// The same grid opens from the shape fill of the kubakub director (pickShader is exported for it).
// The pictures are one sheet drawn by the node itself (/kubakub/shader/thumbs), and the moving ones one loop per
// preset (/kubakub/shader/preview), so they show what a queue gives.
// Plain ES module, no build step, no network. Look: kubakub.art (Hanken Grotesk, #f18a58 actions, #a187b7 selection).
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const ORANGE = "#f18a58", LAV = "#a187b7";
const FONT = '"Hanken Grotesk KKD","Hanken Grotesk","Segoe UI",system-ui,sans-serif';
let listing = null;                                  // { presets, about, types, type, tags, node_only, tile, columns }, asked once
let known = null;                                    // the same, once it is here

export function shaderPresets() {
  return (listing ||= api.fetchApi("/kubakub/shader/presets").then(r => r.json()).then(j => (known = j)).catch(e => { listing = null; throw e; }));
}

// A sheet of the presets next to `anchor`, sorted by type; onPick(name) on a click. Esc or a click outside closes it.
// The pills at the top show one type, the box finds by name or by what a preset draws, and a tile plays under the
// pointer. skip: names to leave out (the director leaves out the presets only the node can draw).
export async function pickShader(anchor, current, onPick, skip = []) {
  document.querySelector(".kks-pick")?.remove();
  let info;
  try { info = await shaderPresets(); } catch (e) { console.warn("[kubakub shader] presets", e); return; }
  const [tw, th] = info.tile || [160, 90], cols = info.columns || 6, sheet = api.apiURL("/kubakub/shader/thumbs");
  const dark = matchMedia("(prefers-color-scheme: dark)").matches;
  const paper = dark ? "#141414" : "#ffffff", ink = dark ? "rgba(255,255,255,.88)" : "rgba(0,0,0,.85)", line = dark ? "rgba(255,255,255,.12)" : "rgba(0,0,0,.12)";
  const names = info.presets.filter(n => !skip.includes(n));
  const types = (info.types?.length ? info.types : [{ key: "", title: "presets", about: "" }])
    .map(t => ({ ...t, names: names.filter(n => (info.type?.[n] || "") === t.key || !info.types?.length) })).filter(t => t.names.length);
  let shown = "";                                      // "" = all types
  try { const was = localStorage.getItem("kubakub.shader.type") || ""; if (types.some(t => t.key === was)) shown = was; } catch (e) { /* no storage */ }
  if (current && shown && info.type?.[current] !== shown) shown = "";      // the one in use must be in sight

  const el = (tag, css, text) => { const q = document.createElement(tag); if (css) q.style.cssText = css; if (text != null) q.textContent = text; return q; };
  const box = el("div", `position:fixed;z-index:10050;background:${paper};color:${ink};border:1px solid ${line};border-radius:14px;
    box-shadow:0 18px 50px rgba(0,0,0,.35);box-sizing:border-box;width:min(94vw,${5 * tw + 4 * 10 + 44}px);max-height:86vh;display:flex;flex-direction:column;
    font:400 12px/1.35 ${FONT};text-transform:lowercase`);
  box.className = "kks-pick";
  const top = el("div", `display:flex;flex-wrap:wrap;gap:6px;align-items:center;padding:12px 12px 8px;border-bottom:1px solid ${line}`);
  const find = el("input", `all:unset;box-sizing:border-box;flex:1 1 120px;min-width:110px;padding:4px 10px;border:1px solid ${line};border-radius:999px;font:inherit;color:inherit;text-transform:none`);
  find.placeholder = "find: fire, vhs, depth …"; find.spellcheck = false;
  const body = el("div", "overflow:auto;padding:4px 12px 12px;flex:1 1 auto");
  const foot = el("div", `padding:8px 14px 10px;border-top:1px solid ${line};min-height:66px;box-sizing:border-box;flex:0 0 auto;text-transform:none`);      // three lines: the sheet keeps still while the pointer moves
  const say = (name) => {
    foot.textContent = "";
    if (!name) { foot.append(el("span", "opacity:.6", status)); return; }
    const type = types.find(t => t.key === (info.type?.[name] || ""));
    foot.append(el("b", `font-weight:600;color:${name === current ? LAV : "inherit"}`, name), el("span", "opacity:.55", type ? "  ·  " + type.title : ""),
      el("div", "opacity:.8;margin-top:2px", info.about?.[name] || ""));
    const tags = info.tags?.[name] || [];
    if (tags.length) foot.append(el("div", "opacity:.55;margin-top:2px", tags.map(t => t.startsWith("gives") || t === "node only" ? t : "reads " + t).join("  ·  ")));
  };
  let status = "drawing the pictures (a few seconds the first time) …";

  const pics = [], tiles = [], sections = [];
  let over = null;                                     // the tile under the pointer: it plays
  const still = pic => { pic.style.backgroundImage = pic._sheet ? `url("${sheet}")` : "none"; pic.style.backgroundSize = "auto"; pic.style.backgroundPosition = pic._at; };
  for (const t of types) {
    const sec = el("div", "margin-top:10px");
    const head = el("div", "display:flex;gap:8px;align-items:baseline;margin:0 2px 6px");
    head.append(el("b", "font-weight:600;font-size:13px", t.title), el("span", "opacity:.55;text-transform:none", t.about));
    const grid = el("div", `display:grid;grid-template-columns:repeat(auto-fill,${tw}px);gap:10px`);
    for (const name of t.names) {
      const i = info.presets.indexOf(name);
      const b = el("button", `all:unset;cursor:pointer;display:block;width:${tw}px;border-radius:10px;outline:2px solid ${name === current ? LAV : "transparent"};outline-offset:2px`);
      const pic = el("div", `position:relative;width:${tw}px;height:${th}px;border-radius:8px;overflow:hidden;background-color:${dark ? "#2c2c2e" : "#e9e6e3"};background-repeat:no-repeat`);
      pic._at = `-${(i % cols) * tw}px -${Math.floor(i / cols) * th}px`; pic.style.backgroundPosition = pic._at;
      const tags = (info.tags?.[name] || []).filter(q => q === "your picture" || q === "a clip" || q === "gives a depth");
      if (tags.length) { const row = el("div", "position:absolute;left:4px;bottom:4px;display:flex;gap:3px");      // what it reads and gives, on the picture
        row.append(...tags.map(q => el("span", "background:rgba(0,0,0,.62);color:#fff;border-radius:999px;padding:0 6px;font-size:10px;line-height:15px", q === "gives a depth" ? "depth" : q === "a clip" ? "clip" : "picture")));
        pic.append(row); }
      pics.push(pic);
      const label = el("div", `padding:4px 2px 0;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;color:${name === current ? LAV : "inherit"}`, name);
      b.append(pic, label);
      b.onmouseenter = () => {
        if (name !== current) b.style.outlineColor = ORANGE;
        say(name); over = pic;
        clearTimeout(pic._t);
        pic._t = setTimeout(() => {                    // a short rest on a tile: it plays one loop, drawn by the node
          if (over !== pic) return;
          const url = api.apiURL("/kubakub/shader/preview?name=" + encodeURIComponent(name)), im = new Image();
          im.onload = () => { if (over !== pic) return; pic.style.backgroundImage = `url("${url}")`; pic.style.backgroundSize = "cover"; pic.style.backgroundPosition = "center"; };
          im.src = url;
        }, 180);
      };
      b.onmouseleave = () => { b.style.outlineColor = name === current ? LAV : "transparent"; clearTimeout(pic._t); if (over === pic) over = null; still(pic); say(""); };
      b.onclick = () => { close(); onPick(name); };
      grid.append(b);
      tiles.push({ b, name, key: t.key, text: (name + " " + (info.about?.[name] || "") + " " + (info.tags?.[name] || []).join(" ")).toLowerCase() });
    }
    sec.append(head, grid); body.append(sec); sections.push({ sec, key: t.key });
  }

  const pills = [];
  const pill = (key, title, n, tip) => {
    const p = el("button", `all:unset;cursor:pointer;padding:3px 10px;border-radius:999px;border:1px solid ${line};white-space:nowrap`, `${title} ${n}`);
    p.title = tip; p.onclick = () => { shown = key; try { localStorage.setItem("kubakub.shader.type", key); } catch (e) { /* no storage */ } filter(); body.scrollTop = 0; };
    pills.push({ p, key }); top.append(p);
  };
  if (types.length > 1) { pill("", "all", names.length, "Every preset, sorted by type"); for (const t of types) pill(t.key, t.title, t.names.length, t.about); }
  top.append(find);
  const none = el("div", "opacity:.6;padding:18px 2px", "nothing found: try another word, or 'all'");
  body.append(none);
  function filter() {
    const q = find.value.trim().toLowerCase(), words = q ? q.split(/\s+/) : [];
    const key = q ? "" : shown;                        // a word looks through every type
    let any = false;
    for (const t of tiles) { const on = (!key || t.key === key) && words.every(w => t.text.includes(w)); t.b.style.display = on ? "block" : "none"; t.on = on; any ||= on; }
    for (const s of sections) s.sec.style.display = tiles.some(t => t.key === s.key && t.on) ? "block" : "none";
    for (const { p, key: k } of pills) { const on = !q && k === shown; p.style.background = on ? LAV : "transparent"; p.style.color = on ? "#fff" : "inherit"; p.style.borderColor = on ? LAV : line; }
    none.style.display = any ? "none" : "block";
  }
  find.oninput = filter;
  filter();

  const img = new Image();                              // the sheet is drawn by the node on its first request
  img.onload = () => { for (const q of pics) { q._sheet = true; if (q !== over) still(q); } status = "rest on a picture to see it move · click to use it"; if (!over) say(""); };
  img.onerror = () => { status = "no pictures here (the graphics context did not start), the names still work"; if (!over) say(""); };
  img.src = sheet;
  say("");
  box.append(top, body, foot);
  document.body.append(box);
  const r = anchor?.getBoundingClientRect?.() || { left: innerWidth / 2 - 200, bottom: innerHeight / 3, top: innerHeight / 3 };
  const bw = box.offsetWidth, bh = box.offsetHeight;
  box.style.left = Math.max(8, Math.min(innerWidth - bw - 8, r.left)) + "px";
  box.style.top = Math.max(8, r.bottom + 6 + bh > innerHeight ? Math.max(8, innerHeight - bh - 8) : r.bottom + 6) + "px";
  box.style.height = bh + "px";                         // one type or a search must not make the sheet jump
  tiles.find(t => t.name === current)?.b.scrollIntoView({ block: "center" });
  const away = e => { if (!box.contains(e.target)) close(); }, key = e => { if (e.key === "Escape") { e.stopPropagation(); close(); } else if (box.contains(e.target)) e.stopPropagation(); };
  function close() { box.remove(); document.removeEventListener("pointerdown", away, true); document.removeEventListener("keydown", key, true); }
  setTimeout(() => { document.addEventListener("pointerdown", away, true); document.addEventListener("keydown", key, true); }, 0);
}

// "adjust": the shader's controls as sliders next to a picture of it, for people who do not read code. The sliders
// write the node's own `controls` text (only what differs from the defaults), so the widget stays the truth. The
// picture is drawn by the node (/kubakub/shader/look): one frame while a slider moves, the loop when it rests.
// From here the preset's code can be put into `code` to change it, and the look saved as a preset of your own.
async function adjustShader(node, anchor) {
  document.querySelector(".kks-adjust")?.remove();
  const w = name => node.widgets?.find(q => q.name === name);
  const state = () => ({ preset: String(w("preset")?.value || ""), code: String(w("code")?.value || ""), controls: String(w("controls")?.value || "") });
  const post = (path, body) => api.fetchApi(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  const dark = matchMedia("(prefers-color-scheme: dark)").matches;
  const paper = dark ? "#141414" : "#ffffff", ink = dark ? "rgba(255,255,255,.88)" : "rgba(0,0,0,.85)", line = dark ? "rgba(255,255,255,.12)" : "rgba(0,0,0,.12)";
  const el = (tag, css, text) => { const q = document.createElement(tag); if (css) q.style.cssText = css; if (text != null) q.textContent = text; return q; };
  const pillCss = `all:unset;cursor:pointer;padding:4px 12px;border-radius:999px;border:1px solid ${line};white-space:nowrap`;
  const box = el("div", `position:fixed;z-index:10040;background:${paper};color:${ink};border:1px solid ${line};border-radius:14px;
    box-shadow:0 18px 50px rgba(0,0,0,.35);box-sizing:border-box;width:min(94vw,520px);max-height:90vh;display:flex;flex-direction:column;
    font:400 12px/1.35 ${FONT}`);
  box.className = "kks-adjust";
  const head = el("div", `display:flex;gap:8px;align-items:baseline;padding:12px 14px 8px;cursor:move;user-select:none`);
  const title = el("b", "font-weight:600;font-size:13px;text-transform:lowercase", "adjust"), sub = el("span", "opacity:.55;flex:1 1 auto;overflow:hidden;text-overflow:ellipsis;white-space:nowrap");
  const shut = el("button", "all:unset;cursor:pointer;opacity:.6;padding:0 4px;font-size:16px;line-height:1", "×");
  shut.title = "Close (Esc)";
  head.append(title, sub, shut);
  const pic = el("img", `display:block;width:calc(100% - 28px);margin:0 14px;aspect-ratio:16/9;border-radius:10px;background:${dark ? "#2c2c2e" : "#e9e6e3"};object-fit:cover`);
  pic.alt = "";
  const note = el("div", "padding:6px 14px 0;opacity:.6;min-height:17px");
  const rows = el("div", "overflow:auto;padding:6px 14px 4px;flex:1 1 auto");
  const foot = el("div", `display:flex;flex-wrap:wrap;gap:6px;align-items:center;padding:10px 14px 12px;border-top:1px solid ${line};text-transform:lowercase`);
  box.append(head, pic, note, rows, foot);

  let info = null, values = {}, seq = 0, timer = null, url = null, shown = "", said = "";      // said: what just happened, kept under the next picture
  const hex = name => "#" + ["r", "g", "b"].map(ch => Math.round(Math.min(1, Math.max(0, values[`${name}_${ch}`])) * 255).toString(16).padStart(2, "0")).join("");
  const parts = name => (info?.controls || []).filter(c => c.colour === name);
  const moved = c => Math.abs(values[c.name] - c.default) > 0.5 / 255;
  const text = () => (info?.controls || []).filter(c => c.colour ? c.name.endsWith("_r") && parts(c.colour).some(moved) : Math.abs(values[c.name] - c.default) > 1e-9)
    .map(c => c.colour ? `${c.colour} = ${hex(c.colour)}` : `${c.name.replace(/_$/, "")} = ${+values[c.name].toFixed(4)}`).join(", ");
  const write = () => { const c = w("controls"); if (!c) return; c.value = text(); c.callback?.(c.value); node.setDirtyCanvas(true, true); };
  async function draw(moving) {                        // the node draws it, so it is what a queue gives (nothing connected)
    const mine = ++seq;
    try {
      const r = await post("/kubakub/shader/look", { ...state(), moving });
      if (mine !== seq || !box.isConnected) return;
      if (!r.ok) { note.textContent = "no picture: " + (await r.text()); return; }
      const next = URL.createObjectURL(await r.blob());
      pic.src = next; if (url) URL.revokeObjectURL(url); url = next;
      note.textContent = moving ? (said || "as the node draws it with nothing connected") : "…";
      if (moving) said = "";
    } catch (e) { if (mine === seq) note.textContent = "no picture (the server did not answer)"; }
  }
  const changed = () => { write(); draw(false); clearTimeout(timer); timer = setTimeout(() => draw(true), 700); };

  function build() {
    rows.textContent = "";
    const cs = info?.controls || [];
    if (info?.error) { rows.append(el("div", "opacity:.7;padding:8px 0", "cannot run: " + info.error)); return; }
    if (!cs.length) rows.append(el("div", "opacity:.7;padding:8px 0", "this shader has no controls. A line like  uniform float glow; // 0.5 0 1  in its code makes one."));
    for (const c of cs) {
      if (c.colour) {                                  // a colour: one swatch for its red, green and blue
        if (!c.name.endsWith("_r")) continue;
        const row = el("label", "display:grid;grid-template-columns:96px 1fr 58px;gap:8px;align-items:center;padding:3px 0");
        const name = el("span", "overflow:hidden;text-overflow:ellipsis;white-space:nowrap;text-transform:lowercase", c.colour.replace(/_/g, " "));
        const sw = el("input", `all:unset;box-sizing:border-box;width:100%;height:22px;border:1px solid ${line};border-radius:6px;cursor:pointer;overflow:hidden`); sw.type = "color";
        const code = el("input", `all:unset;box-sizing:border-box;width:58px;padding:2px 4px;border:1px solid ${line};border-radius:6px;text-align:right;font:inherit;font-size:11px`); code.type = "text"; code.maxLength = 7; code.spellcheck = false;
        const put = () => { sw.value = hex(c.colour); code.value = hex(c.colour); name.style.color = parts(c.colour).some(moved) ? ORANGE : "inherit"; };
        const take = h => { ["r", "g", "b"].forEach((ch, i) => { values[`${c.colour}_${ch}`] = parseInt(h.slice(1 + 2 * i, 3 + 2 * i), 16) / 255; }); put(); changed(); };
        put();
        row.title = (c.about ? c.about + "\n" : "") + "Click the colour to pick another. Double-click the name to reset it.";
        sw.oninput = () => take(sw.value);
        code.onchange = () => { const h = code.value.trim().replace(/^#?/, "#"); if (/^#[0-9a-fA-F]{6}$/.test(h)) take(h.toLowerCase()); else put(); };
        name.ondblclick = e => { e.preventDefault(); for (const q of parts(c.colour)) values[q.name] = q.default; put(); changed(); };
        row.append(name, sw, code);
        rows.append(row);
        if (c.about) rows.append(el("div", "opacity:.5;margin:-3px 0 4px 104px", c.about));
        continue;
      }
      const row = el("label", "display:grid;grid-template-columns:96px 1fr 58px;gap:8px;align-items:center;padding:3px 0");
      const name = el("span", "overflow:hidden;text-overflow:ellipsis;white-space:nowrap;text-transform:lowercase", c.name.replace(/_$/, ""));
      const span = c.max - c.min, whole = /^(steps|count|rows|columns|cells|bars|rings|arms|tiles|mirrors|tones|palette|beams|stars|drops|blobs|forms|bays|lanes|panels|points|pixels|dots|stripes|lines|segments|sheets|passes|blocks|keys)$/.test(c.name) && span >= 4;
      const s = el("input", `width:100%;accent-color:${ORANGE};margin:0`); s.type = "range"; s.min = c.min; s.max = c.max; s.step = whole ? 1 : span > 0 ? Math.pow(10, Math.floor(Math.log10(span / 100))) : 0.01;      // round steps: 0.1 over a range of 10 to 99
      s.value = values[c.name];
      const n = el("input", `all:unset;box-sizing:border-box;width:58px;padding:2px 6px;border:1px solid ${line};border-radius:6px;text-align:right;font:inherit`); n.type = "text"; n.inputMode = "decimal";
      const put = () => { n.value = String(+values[c.name].toFixed(3)); name.style.color = Math.abs(values[c.name] - c.default) > 1e-9 ? ORANGE : "inherit"; };
      put();
      row.title = (c.about ? c.about + "\n" : "") + `${c.min} to ${c.max}, default ${c.default}. Double-click the name to reset it.`;
      s.oninput = () => { values[c.name] = +s.value; put(); changed(); };
      n.onchange = () => { const v = parseFloat(n.value.replace(",", ".")); values[c.name] = Number.isFinite(v) ? Math.min(c.max, Math.max(c.min, v)) : values[c.name]; s.value = values[c.name]; put(); changed(); };
      name.ondblclick = e => { e.preventDefault(); values[c.name] = c.default; s.value = c.default; put(); changed(); };
      row.append(name, s, n);
      rows.append(row);
      if (c.about) rows.append(el("div", "opacity:.5;margin:-3px 0 4px 104px", c.about));
    }
  }

  async function load() {                              // the controls of what the node will draw, and their values as typed
    const s = state();
    try { info = await (await post("/kubakub/shader/translate", s)).json(); } catch (e) { info = { error: "the server did not answer" }; }
    values = { ...(info.values || {}) };
    for (const c of info.controls || []) if (!Number.isFinite(values[c.name])) values[c.name] = c.default;
    shown = s.preset + "\n" + s.code;
    sub.textContent = s.code.trim() ? "your code" : s.preset + (info.about ? " · " + info.about : "");
    code.textContent = s.code.trim() ? "back to the preset" : "show the code";
    code.title = s.code.trim() ? "Empties 'code': the node draws the preset again. Your changes to the code are lost unless you saved them as a preset."
      : "Puts the preset's code into 'code', where you can change it: colours, shapes, anything. The sliders keep working.";
    del.style.display = info.mine && !s.code.trim() ? "inline-block" : "none";
    name.value = info.mine && !s.code.trim() ? s.preset : "";
    build(); draw(true);
  }

  const reset = el("button", pillCss, "reset"); reset.title = "Every control back to its default";
  reset.onclick = () => { for (const c of info?.controls || []) values[c.name] = c.default; build(); changed(); };
  const code = el("button", pillCss, "show the code");
  code.onclick = async () => {
    const c = w("code"); if (!c) return;
    c.value = String(c.value || "").trim() ? "" : (info?.text || ""); c.callback?.(c.value);
    node.setDirtyCanvas(true, true); await load();
  };
  const name = el("input", `all:unset;box-sizing:border-box;flex:1 1 110px;min-width:90px;padding:4px 10px;border:1px solid ${line};border-radius:999px;font:inherit`);
  name.placeholder = "a name for it"; name.spellcheck = false; name.maxLength = 40;
  const save = el("button", pillCss + `;background:${ORANGE};border-color:${ORANGE};color:#fff`, "save as my preset");
  save.title = "Saves the shader with the sliders as they stand as a preset of your own (a file in ComfyUI's user folder, kubakub/shaders). It then shows under 'my presets'.";
  const del = el("button", pillCss + ";opacity:.7", "delete"); del.title = "Removes this saved preset (the file). The built-in presets cannot be removed.";
  const pick = (n, list) => { const p = w("preset"); if (!p) return; if (list) p.options.values = list; p.value = n; p.callback?.(n); const c = w("code"); if (c && c.value) { c.value = ""; c.callback?.(""); }
    const k = w("controls"); if (k && k.value) { k.value = ""; k.callback?.(""); } node.setDirtyCanvas(true, true); };
  save.onclick = async () => {
    if (!name.value.trim()) { name.focus(); note.textContent = "give it a name first"; return; }
    try {
      const j = await (await post("/kubakub/shader/save", { ...state(), name: name.value })).json();
      if (j.error) { note.textContent = "not saved: " + j.error; return; }
      listing = known = null;                          // the list has one more
      const all = await shaderPresets();
      pick(j.name, all.presets);                       // the node now runs the saved preset: its defaults are the sliders
      said = `saved as '${j.name}' (under my presets)`; note.textContent = said;
      await load();
    } catch (e) { note.textContent = "not saved (the server did not answer)"; }
  };
  name.onkeydown = e => { if (e.key === "Enter") save.onclick(); };
  del.onclick = async () => {
    const s = state();
    if (del.dataset.sure !== s.preset) { del.dataset.sure = s.preset; del.textContent = "really delete?"; setTimeout(() => { del.dataset.sure = ""; del.textContent = "delete"; }, 3000); return; }
    try {
      const j = await (await post("/kubakub/shader/delete", { name: s.preset })).json();
      if (j.error) { note.textContent = "not deleted: " + j.error; return; }
      listing = known = null;
      const all = await shaderPresets();
      pick(all.presets.includes("warped noise") ? "warped noise" : all.presets[0], all.presets);
      del.dataset.sure = ""; del.textContent = "delete";
      said = `'${s.preset}' is deleted`; note.textContent = said;
      await load();
    } catch (e) { note.textContent = "not deleted (the server did not answer)"; }
  };
  foot.append(reset, code, name, save, del);

  document.body.append(box);
  const r = anchor?.getBoundingClientRect?.() || { left: innerWidth / 2 - 260, bottom: 80 };
  box.style.left = Math.max(8, Math.min(innerWidth - box.offsetWidth - 8, r.left)) + "px";
  box.style.top = Math.max(8, Math.min(innerHeight - 560, r.bottom + 6)) + "px";
  head.onpointerdown = e => {                          // it stays open next to the node: drag it by its head
    if (e.target === shut) return;
    const x0 = e.clientX - box.offsetLeft, y0 = e.clientY - box.offsetTop;
    const move = q => { box.style.left = Math.max(0, Math.min(innerWidth - 60, q.clientX - x0)) + "px"; box.style.top = Math.max(0, Math.min(innerHeight - 40, q.clientY - y0)) + "px"; };
    const up = () => { removeEventListener("pointermove", move); removeEventListener("pointerup", up); };
    addEventListener("pointermove", move); addEventListener("pointerup", up);
  };
  const key = e => { if (!box.isConnected) return; if (e.key === "Escape" && !document.querySelector(".kks-pick")) { e.stopPropagation(); close(); } else if (box.contains(e.target)) e.stopPropagation(); };
  const watch = setInterval(() => {                    // another preset was picked, or the code changed: other controls
    if (!box.isConnected || !node.graph) return close();
    const s = state();
    if (s.preset + "\n" + s.code !== shown && !box.contains(document.activeElement)) load();
  }, 500);
  function close() { clearInterval(watch); clearTimeout(timer); box.remove(); document.removeEventListener("keydown", key, true); if (url) URL.revokeObjectURL(url); }
  shut.onclick = close;
  document.addEventListener("keydown", key, true);
  await load();
}

// A read-only text widget shows its label and no value, so the sentence is the label; all of it is in the tooltip.
const DRAWS_TIP = "What the node will draw: the preset, or your pasted code and what it reads; or why pasted code cannot run.";
function say(line, text) { line.value = ""; line.label = text || "draws"; line.tooltip = text ? text + "\n\n" + DRAWS_TIP : DRAWS_TIP; }

function typeTitle(key) { const t = key && known?.types?.find?.(q => q.key === key); return t ? t.title + " · " : ""; }

// what the node will draw, in a line: asked from the node itself, so it is what a queue does
async function check(node) {
  const w = name => node.widgets?.find(q => q.name === name), line = w("draws");
  if (!line) return;
  const code = String(w("code")?.value || ""), preset = String(w("preset")?.value || "");
  const mine = ++node._kksCheck;
  try {
    if (!code.trim() && !known) await shaderPresets().catch(() => null);      // the names of the types
    const r = await api.fetchApi("/kubakub/shader/translate", { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(code.trim() ? { code } : { preset }) });
    const j = await r.json();
    if (mine !== node._kksCheck) return;                // a newer check is on its way
    const reads = (j.channels || []).map(i => "image" + i).join(", ");
    const text = j.error ? "cannot run: " + j.error
      : code.trim() ? `your code (the preset is not used)${reads ? " · reads " + reads : ""}${(j.uses || []).includes("kPrev") ? " · builds on its last frame" : ""}${(j.uses || []).includes("kDepth") ? " · gives a depth" : ""}`
        : `${typeTitle(j.type)}${preset}: ${j.about || ""}`;
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
      button.label = "browse the presets: pictures, by type";
      button.tooltip = "Shows every preset as a small picture, sorted by type (patterns, 3d scenes, on your regions, picture looks, screen and tape, picture with depth, over time). Rest on one to see it move, click to use it. Used while 'code' is empty.";
      button.serialize = false;
      const beside = () => { const c = app.canvas, pos = c?.canvas ? c.canvas.getBoundingClientRect() : { left: 0, top: 0 };
        const x = c?.ds ? pos.left + (node.pos[0] + node.size[0] + c.ds.offset[0]) * c.ds.scale + 8 : innerWidth / 2, y = c?.ds ? pos.top + (node.pos[1] + c.ds.offset[1]) * c.ds.scale : 80;
        return { getBoundingClientRect: () => ({ left: x, bottom: y, top: y }) }; };
      const adjust = this.addWidget("button", "adjust", null, () => adjustShader(node, beside()), { serialize: false });
      adjust.label = "adjust: sliders, the code, save as my preset";
      adjust.tooltip = "The shader's controls as sliders next to a picture of it. From there you can also put the preset's code into 'code' to change it, and save the result as a preset of your own.";
      adjust.serialize = false;
      const line = this.addWidget("text", "draws", "", () => {}, { serialize: false });
      line.serialize = false; line.disabled = true;
      line.tooltip = DRAWS_TIP;
      // neither is saved (serialize = false), so saved workflows keep their values; they sit right under `preset`
      for (const w of [line, adjust, button]) { this.widgets.splice(this.widgets.indexOf(w), 1); this.widgets.splice(this.widgets.indexOf(preset) + 1, 0, w); }
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
