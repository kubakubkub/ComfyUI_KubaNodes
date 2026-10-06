// kubakub light plan: the lamps of kubakub scene relight placed on a top view, on the node itself.
// The plan edits the node's "lights" text (one lamp per line, facade metres), so the text stays the truth and a
// workflow without this file still runs. After a run the node sends the real top view (model, surroundings,
// projector, audience) as the background; before the first run a plain grid stands in.
// Plain ES module, no build step, no network. Look: kubakub.art (Hanken Grotesk, #f18a58 actions, #a187b7 selection).
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const HERE = new URL(".", import.meta.url).href;
const ORANGE = "#f18a58", LAV = "#a187b7";
const SIZE = 768;                                       // the plan's own pixels (the picture the node sends)
const HEADER = "// type  x  height  distance  watts  #colour  size / angle   (metres from the facade's middle, the ground, the wall)";
const NEW = {                                           // new lamps: watts and colours as in the director's light layers,
                                                        // placed further out (a street, not a wall wash)
  point: { x: 0, height: 3, distance: 6, power: 400, color: "#ffb070", extra: 0.3 },
  area: { x: 0, height: 8, distance: 10, power: 1500, color: "#ffd2a0", extra: 4 },
  spot: { x: 0, height: 12, distance: 20, power: 3000, color: "#fff1dc", extra: 35 },
  sun: { azimuth: 30, elevation: 35, power: 3, color: "#fff4e0" } };
const FALLBACK = { origin: [SIZE / 2, SIZE * 0.36], ax: [9.6, 0], ad: [0, 9.6] };   // 80 m across

const num = v => String(Math.round(v * 100) / 100);

// ---- the "lights" text <-> lamps (the grammar of scene_view.parse_lights)
const textLines = text => String(text || "").replace(/;/g, "\n").split("\n");
function parseLine(raw) {
  const line = raw.split("//")[0].trim();
  if (!line) return null;
  const tok = line.replace(/,/g, " ").split(/\s+/);
  const type = tok[0].toLowerCase();
  if (!(type in NEW)) return null;
  const color = tok.slice(1).find(t => /^#[0-9a-f]{6}$/i.test(t)) || "#ffffff";
  const v = tok.slice(1).filter(t => !t.startsWith("#")).map(Number);
  if (v.some(Number.isNaN) || v.length < (type === "sun" ? 3 : 4)) return null;
  return type === "sun" ? { type, azimuth: v[0], elevation: v[1], power: v[2], color }
    : { type, x: v[0], height: v[1], distance: v[2], power: v[3], color, extra: v[4] };
}

// every lamp remembers the line it came from, so writing back touches only lamp lines
export function parseLights(text) {
  const out = [];
  textLines(text).forEach((raw, i) => { const L = parseLine(raw); if (L) { L.line = i; out.push(L); } });
  return out;
}

const lampText = L => L.type === "sun"
  ? `sun  ${num(L.azimuth)}  ${num(L.elevation)}  ${num(L.power)}  ${L.color}`
  : `${L.type.padEnd(5)}  ${num(L.x)}  ${num(L.height)}  ${num(L.distance)}  ${num(L.power)}  ${L.color}`
    + (L.extra != null && !Number.isNaN(L.extra) ? `  ${num(L.extra)}` : "");

// The text with these lamps: a lamp's own line is rewritten, a removed lamp's line goes, new lamps are added at the
// end, and every other line (comments, a line still being typed) stays as it is.
export function writeLights(lamps, text) {
  const lines = textLines(text), at = new Map(lamps.filter(L => L.line != null).map(L => [L.line, L]));
  const out = [];
  lines.forEach((raw, i) => {
    if (at.has(i)) {
      const note = raw.includes("//") ? "   //" + raw.split("//").slice(1).join("//") : "";
      out.push(lampText(at.get(i)) + note);
    } else if (!parseLine(raw)) out.push(raw);
  });
  while (out.length && !out[out.length - 1].trim()) out.pop();
  if (!out.some(l => l.trim().startsWith("//"))) out.unshift(HEADER);
  for (const L of lamps) if (L.line == null) out.push(lampText(L));
  return out.join("\n") + "\n";
}

let styled = false;
function style() {
  if (styled) return;
  styled = true;
  const s = document.createElement("style");
  s.textContent = `
@font-face{font-family:"Hanken Grotesk KKD";src:url("${HERE}fonts/HankenGrotesk-latin-var.woff2") format("woff2");font-weight:200 800;font-display:swap}
.kkl{display:flex;flex-direction:column;gap:6px;width:100%;height:100%;font-family:"Hanken Grotesk KKD",system-ui,sans-serif;font-size:12px;color:rgba(255,255,255,.85);text-transform:lowercase;box-sizing:border-box}
.kkl-stage{flex:1;min-height:0;display:flex;align-items:center;justify-content:center;background:#101010}
.kkl canvas{display:block;cursor:crosshair;touch-action:none}
.kkl-row{display:flex;flex-wrap:wrap;gap:5px;align-items:center}
.kkl-row[data-fields]{height:52px;align-content:flex-start;overflow:hidden}   /* always there: selecting a lamp must not move the plan under the pointer */
.kkl button{font:inherit;text-transform:lowercase;border:0;border-radius:10em;padding:3px 11px;background:${ORANGE};color:#141414;font-weight:700;cursor:pointer}
.kkl button.kkl-quiet{background:rgba(255,255,255,.12);color:rgba(255,255,255,.85);font-weight:400}
.kkl button:disabled{opacity:.3;cursor:default}
.kkl button.kkl-live{margin-left:auto;background:rgba(255,255,255,.12);color:rgba(255,255,255,.85);font-weight:400}
.kkl button.kkl-live.on{background:${LAV};color:#141414;font-weight:700}
.kkl label{display:flex;align-items:center;gap:3px;color:rgba(255,255,255,.6)}
.kkl input[type=number]{font:inherit;width:50px;background:rgba(255,255,255,.08);color:rgba(255,255,255,.9);border:1px solid transparent;border-radius:10em;padding:2px 8px}
.kkl input[type=number]:focus{outline:0;border-color:${LAV}}
.kkl input[type=color]{width:26px;height:20px;padding:0;border:0;background:none;cursor:pointer}
.kkl-name{color:${LAV};font-weight:700;min-width:54px}`;
  document.head.appendChild(s);
}

function buildPlan(node) {
  style();
  const text = node.widgets?.find(w => w.name === "lights");
  const el = document.createElement("div");
  el.className = "kkl";
  el.innerHTML = `
<div class="kkl-stage"><canvas width="${SIZE}" height="${SIZE}" title="Drag a lamp to move it along and away from the wall. Wheel over a lamp: its height. Drag a sun around the plan: where it shines from. Drag the violet ring: where the audience camera stands (camera = audience)."></canvas></div>
<div class="kkl-row">
  <button data-add="point" title="A bare lamp: light in all directions">+ point</button>
  <button data-add="area" title="A soft panel that faces the wall">+ area</button>
  <button data-add="spot" title="A cone aimed at the wall">+ spot</button>
  <button data-add="sun" title="Parallel light from far away: moon, sun">+ sun</button>
  <button class="kkl-quiet" data-del title="Remove the selected lamp (Delete)">remove</button>
  <button class="kkl-live" data-live title="Live: the picture renders again each time you drop a lamp or the camera, or change a value here. For quick looks lower resolution_scale and samples.">live</button>
</div>
<div class="kkl-row" data-fields></div>`;
  const cv = el.querySelector("canvas"), ctx = cv.getContext("2d"), fields = el.querySelector("[data-fields]");
  const st = { sel: -1, drag: -1, hover: -1, aud: false, turn: false, bg: null, bgKey: "" };
  const meta = () => node.properties?.kubakub_light_plan || FALLBACK;
  const lamps = () => dragging.list || parseLights(text?.value);
  // the audience camera: its ring is dragged like a lamp and writes audience_offset_m / audience_distance_m
  const widget = name => node.widgets?.find(w => w.name === name);
  const audience = () => widget("camera")?.value === "audience" && widget("audience_offset_m") && widget("audience_distance_m")
    ? { x: Number(widget("audience_offset_m").value) || 0, d: Number(widget("audience_distance_m").value) || 0 } : null;
  const setWidget = (name, v) => { const w = widget(name); if (w) { w.value = v; w.callback?.(v); } };
  // where the audience camera looks: at the frame's middle, turned by audience_turn_deg (+ = clockwise on the plan)
  const lookAngle = A => {
    const p = toPx(A.x, A.d), o = toPx(0, 0);
    return Math.atan2(o[1] - p[1], o[0] - p[0]) + (Number(widget("audience_turn_deg")?.value) || 0) * Math.PI / 180;
  };
  const lookHandle = A => { const p = toPx(A.x, A.d), a = lookAngle(A); return [p[0] + Math.cos(a) * 80, p[1] + Math.sin(a) * 80]; };
  // live: every finished change queues a run (a moment later, so quick changes become one run)
  let liveTimer = 0;
  const live = () => {
    if (!node.properties?.kubakub_live) return;
    clearTimeout(liveTimer);
    liveTimer = setTimeout(() => app.queuePrompt(0, 1), 250);
  };
  const commit = (list, keep) => {
    if (!text) return;
    text.value = writeLights(list, text.value);
    if (text.inputEl) text.inputEl.value = text.value;
    text.callback?.(text.value);
    node.graph?.setDirtyCanvas(true, true);
    draw();
    if (!keep) showFields();
    live();
  };
  // while a lamp is dragged the plan follows the pointer once per frame; the text is written when it is dropped
  const dragging = { list: null, frame: 0 };
  const dragDraw = () => { dragging.frame = 0; draw(); syncFields(); };

  // plan pixels <-> facade metres (x along the wall, d away from it); a sun sits on the rim, where it shines from
  const toPx = (x, d) => { const m = meta(); return [m.origin[0] + x * m.ax[0] + d * m.ad[0], m.origin[1] + x * m.ax[1] + d * m.ad[1]]; };
  const toM = (px, py) => {
    const m = meta(), u = px - m.origin[0], v = py - m.origin[1], det = m.ax[0] * m.ad[1] - m.ad[0] * m.ax[1] || 1;
    return [(u * m.ad[1] - v * m.ad[0]) / det, (v * m.ax[0] - u * m.ax[1]) / det];
  };
  const sunDir = L => {                                  // unit vector in plan pixels from the middle towards the sun
    const m = meta(), a = L.azimuth * Math.PI / 180, s = Math.sin(a), c = Math.cos(a);
    const v = [s * m.ax[0] + c * m.ad[0], s * m.ax[1] + c * m.ad[1]], n = Math.hypot(v[0], v[1]) || 1;
    return [v[0] / n, v[1] / n];
  };
  const lampPx = L => {
    if (L.type !== "sun") return toPx(L.x, L.distance);
    const d = sunDir(L);
    return [SIZE / 2 + d[0] * SIZE * 0.44, SIZE / 2 + d[1] * SIZE * 0.44];
  };
  const evPx = e => {                                    // page -> plan pixels (the graph may be zoomed)
    const r = cv.getBoundingClientRect();
    return [(e.clientX - r.left) * SIZE / (r.width || 1), (e.clientY - r.top) * SIZE / (r.height || 1)];
  };
  const stage = el.querySelector(".kkl-stage");
  new ResizeObserver(() => {                             // the plan stays square inside whatever room the node gives it
    const side = Math.max(60, Math.floor(Math.min(stage.clientWidth, stage.clientHeight)));
    cv.style.width = cv.style.height = side + "px";
  }).observe(stage);
  const hit = (list, p) => {
    let best = -1, bd = 24;
    list.forEach((L, i) => { const q = lampPx(L), d = Math.hypot(q[0] - p[0], q[1] - p[1]); if (d < bd) { bd = d; best = i; } });
    return best;
  };

  function background() {
    const m = meta();
    if (!m.filename) return null;
    const key = `${m.filename}|${m.subfolder || ""}`;
    if (st.bgKey !== key) {
      st.bgKey = key;
      st.bg = null;
      const img = new Image();
      img.onload = () => { if (st.bgKey === key) { st.bg = img; draw(); } };   // a slower, older picture must not win
      img.onerror = () => {                              // the temp picture is gone after a restart: the grid again,
        if (st.bgKey !== key) return;                    // and its own scale (the old one belonged to that picture)
        st.bg = null;
        delete node.properties.kubakub_light_plan;
        draw();
      };
      img.src = api.apiURL(`/view?filename=${encodeURIComponent(m.filename)}&subfolder=${encodeURIComponent(m.subfolder || "")}&type=${encodeURIComponent(m.type || "temp")}`);
    }
    return st.bg;
  }

  function draw() {
    const m = meta(), list = lamps(), bg = background();
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.fillStyle = "#101010";
    ctx.fillRect(0, 0, SIZE, SIZE);
    if (bg) ctx.drawImage(bg, 0, 0, SIZE, SIZE);
    else {                                               // no run yet: a 10 m grid and the wall
      const k = Math.hypot(m.ax[0], m.ax[1]);
      ctx.strokeStyle = "rgba(255,255,255,.07)";
      ctx.lineWidth = 1;
      for (let g = -60; g <= 60; g += 10) {
        let a = toPx(g, -30), b = toPx(g, 60);
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
    const A = audience(), ring = A ? toPx(A.x, A.d) : (m.camera === "previz" && m.viewer) || null;
    if (ring) {
      ctx.strokeStyle = ctx.fillStyle = LAV;
      if (A) {                                           // what it sees: the lens' angle of view, and the handle that turns it
        const a = lookAngle(A), half = Math.atan(18 / (Number(widget("lens_mm")?.value) || 24)), h = lookHandle(A);
        ctx.lineWidth = 1.5;
        ctx.globalAlpha = 0.55;
        for (const s of [-1, 1]) {
          ctx.beginPath(); ctx.moveTo(ring[0], ring[1]);
          ctx.lineTo(ring[0] + Math.cos(a + s * half) * 260, ring[1] + Math.sin(a + s * half) * 260); ctx.stroke();
        }
        ctx.globalAlpha = 1;
        ctx.beginPath(); ctx.moveTo(ring[0], ring[1]); ctx.lineTo(h[0], h[1]); ctx.stroke();
        ctx.beginPath(); ctx.arc(h[0], h[1], st.turn ? 9 : 7, 0, 7); ctx.fill();
      }
      ctx.lineWidth = st.aud ? 5 : 3;
      ctx.beginPath(); ctx.arc(ring[0], ring[1], 13, 0, 7); ctx.stroke();
      ctx.font = '600 22px "Hanken Grotesk KKD", sans-serif';
      ctx.fillText(A ? "audience" : "previz", ring[0] + 20, ring[1] + 7);
    }
    list.forEach((L, i) => {
      const p = lampPx(L), on = i === st.sel;
      ctx.fillStyle = ctx.strokeStyle = L.color;
      ctx.lineWidth = 2;
      if (L.type === "sun") {
        const d = sunDir(L);
        ctx.beginPath(); ctx.moveTo(p[0], p[1]); ctx.lineTo(p[0] - d[0] * 60, p[1] - d[1] * 60); ctx.stroke();
        ctx.beginPath(); ctx.arc(p[0], p[1], 12, 0, 7); ctx.fill();
      } else {
        if (L.type === "spot") {
          const w = toPx(L.x, 0);
          ctx.lineWidth = 1;
          ctx.beginPath(); ctx.moveTo(p[0], p[1]); ctx.lineTo(w[0], w[1]); ctx.stroke();
        }
        if (L.type === "area") ctx.fillRect(p[0] - 11, p[1] - 11, 22, 22);
        else { ctx.beginPath(); ctx.arc(p[0], p[1], 11, 0, 7); ctx.fill(); }
      }
      if (on || i === st.hover) {
        ctx.strokeStyle = on ? LAV : "rgba(255,255,255,.6)";
        ctx.lineWidth = 3;
        ctx.beginPath(); ctx.arc(p[0], p[1], 20, 0, 7); ctx.stroke();
      }
      ctx.fillStyle = "rgba(255,255,255,.92)";
      ctx.font = '600 24px "Hanken Grotesk KKD", sans-serif';
      ctx.fillText(L.type === "sun" ? `${i + 1} sun ${num(L.elevation)}°` : `${i + 1} ${L.type} ${num(L.height)} m`, p[0] + 26, p[1] + 8);
    });
  }

  function showFields() {
    const list = lamps(), L = list[st.sel];
    el.querySelector("[data-del]").disabled = !L;
    if (!L) { fields.innerHTML = ""; fields.dataset.lamp = ""; return; }
    const f = (key, label, tip, step) => `<label title="${tip}">${label}<input type="number" data-k="${key}" step="${step}" value="${num(L[key] ?? 0)}"></label>`;
    fields.innerHTML = `<span class="kkl-name">${st.sel + 1} ${L.type}</span>` + (L.type === "sun"
      ? f("azimuth", "from", "Where it shines from, in degrees: 0 = from the audience side, 90 = from the right", 5)
        + f("elevation", "height", "Degrees above the horizon", 1) + f("power", "strength", "1-5 is daylight, under 0.5 moonlight", 0.1)
      : f("x", "x", "Metres along the wall from its middle (+ = right)", 0.5) + f("height", "height", "Metres above the ground", 0.5)
        + f("distance", "out", "Metres from the wall towards the audience", 0.5) + f("power", "watts", "Brightness in watts", 50)
        + f("extra", L.type === "spot" ? "angle" : "size", L.type === "spot" ? "Cone angle in degrees" : "Size in metres: larger = softer shadows", L.type === "spot" ? 1 : 0.1))
      + `<label title="Colour of the light"><input type="color" data-k="color" value="${L.color}"></label>`;
    fields.dataset.lamp = `${st.sel}:${L.type}`;
    fields.querySelectorAll("input").forEach(inp => inp.addEventListener("input", () => {
      const now = lamps(), T = now[st.sel];
      if (!T) return;
      T[inp.dataset.k] = inp.type === "color" ? inp.value : (Number(inp.value) || 0);
      commit(now, true);
    }));
  }

  function syncFields() {                               // the numbers of the selected lamp, without rebuilding the row
    const L = lamps()[st.sel];
    if (!L || fields.dataset.lamp !== `${st.sel}:${L.type}`) return showFields();
    fields.querySelectorAll("input[type=number]").forEach(inp => {
      if (document.activeElement !== inp) inp.value = num(L[inp.dataset.k] ?? 0);
    });
  }

  // every pointer / key event stays in the plan: the graph behind must not pan, zoom or delete the node
  for (const t of ["pointerdown", "pointermove", "pointerup", "mousedown", "mouseup", "click", "dblclick", "contextmenu", "keydown", "keyup"])
    el.addEventListener(t, e => e.stopPropagation());
  cv.addEventListener("pointerdown", e => {
    const p = evPx(e), list = lamps(), i = hit(list, p), A = audience();
    st.turn = i < 0 && !!A && Math.hypot(lookHandle(A)[0] - p[0], lookHandle(A)[1] - p[1]) < 20;
    st.aud = i < 0 && !st.turn && !!A && Math.hypot(toPx(A.x, A.d)[0] - p[0], toPx(A.x, A.d)[1] - p[1]) < 24;
    st.sel = i;
    st.drag = i;
    if (i >= 0) dragging.list = list;
    if (i >= 0 || st.aud || st.turn) cv.setPointerCapture(e.pointerId);
    draw();
    showFields();
  });
  cv.addEventListener("pointermove", e => {
    const p = evPx(e), list = lamps();
    if (st.turn) {                                       // the camera turns towards the pointer
      const A = audience(), c = toPx(A.x, A.d), o = toPx(0, 0);
      let t = (Math.atan2(p[1] - c[1], p[0] - c[0]) - Math.atan2(o[1] - c[1], o[0] - c[0])) * 180 / Math.PI;
      t = ((t + 540) % 360) - 180;
      setWidget("audience_turn_deg", Math.abs(t) < 2 ? 0 : Math.round(t));        // near straight: back at the facade
      if (!dragging.frame) dragging.frame = requestAnimationFrame(dragDraw);
      return;
    }
    if (st.aud) {                                        // the audience camera follows the pointer, in half metres
      const q = toM(p[0], p[1]);
      setWidget("audience_offset_m", Math.round(q[0] * 2) / 2);
      setWidget("audience_distance_m", Math.max(0.5, Math.round(q[1] * 2) / 2));
      if (!dragging.frame) dragging.frame = requestAnimationFrame(dragDraw);
      return;
    }
    if (st.drag >= 0 && list[st.drag]) {
      const L = list[st.drag];
      if (L.type === "sun") {
        const m = meta(), q = toM(m.origin[0] + (p[0] - SIZE / 2), m.origin[1] + (p[1] - SIZE / 2));
        L.azimuth = Math.round(Math.atan2(q[0], q[1]) * 180 / Math.PI);
      } else {
        const q = toM(p[0], p[1]), snap = e.shiftKey ? 1 : 0.1;
        L.x = Math.round(q[0] / snap) * snap;
        L.distance = Math.round(q[1] / snap) * snap;
      }
      if (!dragging.frame) dragging.frame = requestAnimationFrame(dragDraw);
      return;
    }
    const h = hit(list, p);
    if (h !== st.hover) { st.hover = h; cv.style.cursor = h >= 0 ? "grab" : "crosshair"; draw(); }
  });
  const drop = () => {
    if (st.aud || st.turn) { st.aud = st.turn = false; node.graph?.setDirtyCanvas(true, true); draw(); live(); }
    st.drag = -1;
    const list = dragging.list;
    dragging.list = null;
    if (list) commit(list);
  };
  cv.addEventListener("pointerup", drop);
  cv.addEventListener("pointercancel", drop);
  cv.addEventListener("wheel", e => {                    // over a lamp: its height (a sun: its elevation)
    const list = lamps(), i = hit(list, evPx(e));
    if (i < 0) return;
    e.preventDefault();
    e.stopPropagation();
    const L = list[i], up = e.deltaY < 0 ? 1 : -1;
    if (L.type === "sun") L.elevation = Math.max(-90, Math.min(90, L.elevation + up * 5));
    else L.height = Math.round((L.height + up * (e.shiftKey ? 0.1 : 0.5)) * 10) / 10;
    st.sel = i;
    commit(list);
  }, { passive: false });
  el.querySelectorAll("[data-add]").forEach(b => b.addEventListener("click", () => {
    const list = lamps(), L = { type: b.dataset.add, ...NEW[b.dataset.add] };
    if (L.type !== "sun") L.x = ((list.length % 5) - 2) * 4;       // new lamps do not land on top of each other
    list.push(L);
    st.sel = list.length - 1;
    commit(list);
  }));
  const remove = () => {
    const list = lamps();
    if (!list[st.sel]) return;
    list.splice(st.sel, 1);
    st.sel = -1;
    commit(list);
  };
  el.querySelector("[data-del]").addEventListener("click", remove);
  const liveBtn = el.querySelector("[data-live]");
  const showLive = () => liveBtn.classList.toggle("on", !!node.properties?.kubakub_live);
  liveBtn.addEventListener("click", () => {
    node.properties = node.properties || {};
    node.properties.kubakub_live = !node.properties.kubakub_live;
    showLive();
    live();
  });
  el.tabIndex = 0;
  el.addEventListener("keydown", e => { if (e.key === "Delete" && e.target.tagName !== "INPUT") remove(); });
  text?.inputEl?.addEventListener("input", () => { draw(); showFields(); });     // typing in the text moves the lamps

  el.kklDraw = () => { draw(); showFields(); showLive(); };
  el.kklState = st;
  draw();
  showFields();
  return el;
}

app.registerExtension({
  name: "kubakub.lightplan",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== "KUBA_SceneRelight") return;
    const created = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = created?.apply(this, arguments);
      const el = buildPlan(this);
      const w = this.addDOMWidget("light_plan", "kubakub_light_plan", el, { serialize: false, hideOnZoom: false,
        getMinHeight: () => 500 });
      w.serialize = false;                               // the lamps live in the "lights" text
      this.kklPlan = el;
      if (this.size[1] < 1180) this.setSize([Math.max(this.size[0], 440), 1180]);
      return r;
    };
    const configured = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function () {
      const r = configured?.apply(this, arguments);
      requestAnimationFrame(() => this.kklPlan?.kklDraw());    // the saved lights text is in place now
      return r;
    };
    const executed = nodeType.prototype.onExecuted;
    nodeType.prototype.onExecuted = function (msg) {
      const r = executed?.apply(this, arguments);
      const m = msg?.kubakub_light_plan?.[0];
      if (m) {
        this.properties = this.properties || {};
        this.properties.kubakub_light_plan = m;          // the real plan: scale, origin and its picture
        this.kklPlan?.kklDraw();
      }
      return r;
    };
  },
});
