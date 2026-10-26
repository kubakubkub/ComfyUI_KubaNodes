// kubakub crop: the crop drawn on the picture, as in Photoshop, with no number to type.
// Drag = a new box, drag inside = move it (it snaps to the picture's edges), handles = size it (Shift keeps the
// ratio, Alt sizes from the centre), double click = all of the picture, arrows = 1 px (Shift = 10). Drag past the
// picture's edge to add canvas (filled as the 'fill' pill says). The pills set ratio, size and fill. The box lives
// in the node's own hidden widgets (box / ratio / size / model_safe / fill: 0-1 of the picture, = kubakub/crop.py),
// so a workflow without this file runs the same. The picture: the node feeding 'image' when it shows one (Load
// Image: at once, before a run), otherwise the picture of the last run.
// The box math below is exported for the director's layer crop (one engine, both places).
// Plain ES module, no build step, no network. Look: kubakub.art (Hanken Grotesk, #f18a58 actions, #a187b7 selection).
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const HERE = new URL(".", import.meta.url).href;
const ORANGE = "#f18a58", LAV = "#a187b7";
export const RATIOS = ["free", "original", "picture", "1:1", "4:5", "3:2", "16:9", "9:16", "21:9"];
const SWAP = { "4:5": "5:4", "5:4": "4:5", "3:2": "2:3", "2:3": "3:2", "16:9": "9:16", "9:16": "16:9" };
const SIZES = ["as cut", "1024", "1536", "2048", "3840"];
const FILLS = ["edge", "mirror", "black", "white", "grey"];
const REACH = 2;
const HIDDEN = ["box", "ratio", "size", "model_safe", "fill"];

// ---- the box (= kubakub/crop.py): [x, y, w, h] as 0-1 of the picture
export function parseBox(text) {
  const v = String(text ?? "").replace(/,/g, " ").trim().split(/\s+/).map(Number);
  if (v.length !== 4 || !v.every(Number.isFinite)) return [0, 0, 1, 1];
  const x0 = Math.min(Math.max(v[0], -REACH), 1), y0 = Math.min(Math.max(v[1], -REACH), 1);
  const x1 = Math.min(Math.max(v[0] + v[2], 0), 1 + REACH), y1 = Math.min(Math.max(v[1] + v[3], 0), 1 + REACH);
  return x1 - x0 <= 0 || y1 - y0 <= 0 ? [0, 0, 1, 1] : [x0, y0, x1 - x0, y1 - y0];
}
export const boxText = b => b.map(v => String(Math.round(v * 1e5) / 1e5)).join(" ");
export function ratioValue(name, W, H, of) {
  name = String(name || "free").toLowerCase();
  if (name === "original") return W / H;
  if (name === "picture") return of && of[1] ? of[0] / of[1] : W / H;
  const m = name.match(/^([\d.]+):([\d.]+)$/);
  return m && +m[1] > 0 && +m[2] > 0 ? +m[1] / +m[2] : null;
}
export function fitRatio(x, y, w, h, r) {          // the largest box of ratio r inside, same centre
  if (!r || w <= 0 || h <= 0) return [x, y, w, h];
  if (w / h > r) { const nw = h * r; return [x + (w - nw) / 2, y, nw, h]; }
  const nh = w / r; return [x, y + (h - nh) / 2, w, nh];
}
const pyRound = v => { const f = Math.floor(v), d = v - f; return d > 0.5 || (d === 0.5 && f % 2) ? f + 1 : f; };   // Python's round()
export function pixelBox(box, W, H, r, snap = 1) {   // = crop.pixel_box (the readout shows what the run will cut)
  let [x, y, w, h] = parseBox(boxText(box));
  const inside = x >= -1e-6 && y >= -1e-6 && x + w <= 1 + 1e-6 && y + h <= 1 + 1e-6;
  [x, y, w, h] = fitRatio(x * W, y * H, w * W, h * H, r);
  snap = Math.max(1, snap | 0);
  const cx = x + w / 2, cy = y + h / 2;
  let pw = Math.max(snap, Math.floor(pyRound(w) / snap) * snap), ph = Math.max(snap, Math.floor(pyRound(h) / snap) * snap);
  if (inside) { pw = Math.min(pw, Math.floor(W / snap) * snap || W); ph = Math.min(ph, Math.floor(H / snap) * snap || H); }
  if (r && snap > 1) {
    if (pw / ph > r * 1.0001 && pw > snap) pw = Math.max(snap, Math.floor(pyRound(ph * r) / snap) * snap);
    else if (pw / ph < r / 1.0001 && ph > snap) ph = Math.max(snap, Math.floor(pyRound(pw / r) / snap) * snap);
  }
  let px = pyRound(cx - pw / 2), py = pyRound(cy - ph / 2);
  if (inside) { px = Math.min(Math.max(px, 0), Math.max(0, W - pw)); py = Math.min(Math.max(py, 0), Math.max(0, H - ph)); }
  else { px = Math.min(Math.max(px, 1 - pw), W - 1); py = Math.min(Math.max(py, 1 - ph), H - 1); }
  return [px, py, pw, ph];
}
export function longSide(w, h, size, snap = 1) {
  const t = parseInt(size, 10); if (!Number.isFinite(t)) return [w, h];
  const k = t / Math.max(w, h); snap = Math.max(1, snap | 0);
  return [Math.max(snap, pyRound(w * k / snap) * snap), Math.max(snap, pyRound(h * k / snap) * snap)];
}
export function ratioName(w, h) {                    // 1920 x 1080 -> "16:9"; near a pill's ratio -> that pill
  const r = w / h;
  for (const n of ["1:1", "4:5", "5:4", "3:2", "2:3", "16:9", "9:16", "21:9", "4:3", "3:4"]) { const [a, b] = n.split(":").map(Number); if (Math.abs(r - a / b) < 0.006) return n; }
  return r >= 1 ? `${(Math.round(r * 100) / 100)}:1` : `1:${Math.round(100 / r) / 100}`;
}
// the box of a pixel drag (px of the picture): edges, ratio (or Shift), Alt = from the centre
export function dragBox(b0, handle, px, py, r, alt, minSize = 8) {
  let [x0, y0, x1, y1] = [b0[0], b0[1], b0[0] + b0[2], b0[1] + b0[3]];
  const cx = (x0 + x1) / 2, cy = (y0 + y1) / 2;
  if (handle.includes("w")) x0 = px; if (handle.includes("e")) x1 = px;
  if (handle.includes("n")) y0 = py; if (handle.includes("s")) y1 = py;
  if (alt) {
    if (handle.includes("w")) x1 = 2 * cx - x0; if (handle.includes("e")) x0 = 2 * cx - x1;
    if (handle.includes("n")) y1 = 2 * cy - y0; if (handle.includes("s")) y0 = 2 * cy - y1;
  }
  if (x1 < x0) { [x0, x1] = [x1, x0]; handle = handle.replace(/[we]/, c => c === "w" ? "e" : "w"); }
  if (y1 < y0) { [y0, y1] = [y1, y0]; handle = handle.replace(/[ns]/, c => c === "n" ? "s" : "n"); }
  let w = Math.max(minSize, x1 - x0), h = Math.max(minSize, y1 - y0);
  if (r) {
    const horiz = /[we]/.test(handle), vert = /[ns]/.test(handle);
    if (horiz && vert) { if (w / h > r) h = w / r; else w = h * r; }
    else if (horiz) h = w / r; else w = h * r;
    const ax = alt || !horiz ? (alt ? cx : (x0 + x1) / 2) : handle.includes("w") ? x1 : x0;
    const ay = alt || !vert ? (alt ? cy : (y0 + y1) / 2) : handle.includes("n") ? y1 : y0;
    x0 = alt || !horiz ? ax - w / 2 : handle.includes("w") ? ax - w : ax;
    y0 = alt || !vert ? ay - h / 2 : handle.includes("n") ? ay - h : ay;
  } else {
    if (x1 - x0 < minSize) { if (handle.includes("w")) x0 = x1 - minSize; }
    if (y1 - y0 < minSize) { if (handle.includes("n")) y0 = y1 - minSize; }
  }
  return [x0, y0, w, h];
}

let styled = false;
function style() {
  if (styled) return; styled = true;
  const s = document.createElement("style");
  s.textContent = `
@font-face{font-family:"Hanken Grotesk KKD";src:url("${HERE}fonts/HankenGrotesk-latin-var.woff2") format("woff2");font-weight:200 800;font-display:swap}
.kkc{display:flex;flex-direction:column;gap:5px;width:100%;height:100%;font-family:"Hanken Grotesk KKD",system-ui,sans-serif;font-size:11px;color:rgba(255,255,255,.85);text-transform:lowercase;box-sizing:border-box}
.kkc-stage{flex:1;min-height:0;position:relative;background:#101010;border-radius:6px;overflow:hidden}
.kkc canvas{position:absolute;inset:0;width:100%;height:100%;display:block;cursor:crosshair;touch-action:none;outline:none}
.kkc-row{display:flex;flex-wrap:wrap;gap:3px;align-items:center}
.kkc-row>span{color:rgba(255,255,255,.45);min-width:34px}
.kkc button{font:inherit;text-transform:lowercase;border:1px solid rgba(255,255,255,.14);background:#1d1b20;color:rgba(255,255,255,.8);border-radius:999px;padding:1px 8px;cursor:pointer;line-height:16px}
.kkc button:hover{border-color:${ORANGE};color:#fff}
.kkc button[aria-pressed=true]{background:${LAV};border-color:${LAV};color:#141414}
.kkc button.act{border-color:${ORANGE};color:${ORANGE}}
.kkc button:disabled{opacity:.35;cursor:default}
.kkc-hint{color:rgba(255,255,255,.5);min-height:14px}`;
  document.head.appendChild(s);
}

function hideWidget(w) {
  if (!w) return;
  w.hidden = true; w.computeSize = () => [0, -4];
  if (w.element) w.element.style.display = "none";
  if (w.inputEl) w.inputEl.style.display = "none";
}
const viewURL = m => api.apiURL(`/view?filename=${encodeURIComponent(m.filename)}&subfolder=${encodeURIComponent(m.subfolder || "")}&type=${encodeURIComponent(m.type || "temp")}`);

function build(node) {
  style();
  const W_ = n => node.widgets?.find(w => w.name === n);
  const el = document.createElement("div");
  el.className = "kkc";
  el.innerHTML = `<div class="kkc-stage"><canvas tabindex="0"></canvas></div>
    <div class="kkc-row" data-row="ratio"><span>ratio</span></div>
    <div class="kkc-row" data-row="size"><span>size</span></div>
    <div class="kkc-row" data-row="fill"><span>new area</span></div>
    <div class="kkc-hint"></div>`;
  const cv = el.querySelector("canvas"), ctx = cv.getContext("2d"), stage = el.querySelector(".kkc-stage"), hint = el.querySelector(".kkc-hint");
  const st = { img: null, key: "", drag: null, hover: "", view: null };

  const get = n => W_(n)?.value;
  const set = (n, v) => { const w = W_(n); if (!w || w.value === v) return; w.value = v; w.callback?.(v); node.graph?.setDirtyCanvas(true, true); };
  const meta = () => node.properties?.kubakub_crop || null;
  const box = () => st.drag?.box || parseBox(get("box"));
  const snap = () => (get("model_safe") === false ? 1 : 16);
  const linked = name => { const i = node.inputs?.findIndex(s => s.name === name); return i >= 0 && node.inputs[i].link != null ? i : -1; };
  const upImage = name => {                          // the picture the node feeding an input shows (Load Image: before a run)
    const i = linked(name); if (i < 0) return null;
    const up = node.getInputNode?.(i), im = up?.imgs?.[up.imageIndex ?? 0] || up?.imgs?.[0];
    return im?.src ? im : null;
  };
  function source() {                                // {url, W, H} of the picture now
    const up = upImage("image"), m = meta();
    if (up && (up.naturalWidth || 0) > 0) return { url: up.src, W: up.naturalWidth, H: up.naturalHeight };
    if (m) return { url: viewURL(m), W: m.W, H: m.H };
    return null;
  }
  function ratioOf() {
    const up = upImage("ratio_of"); if (up?.naturalWidth) return [up.naturalWidth, up.naturalHeight];
    return linked("ratio_of") >= 0 ? meta()?.ratio_of || null : null;
  }
  function picture() {
    const s = source(); if (!s) { st.img = null; st.key = ""; return null; }
    if (st.key !== s.url) {
      st.key = s.url; st.img = null;
      const im = new Image(); im.crossOrigin = "anonymous";
      im.onload = () => { if (st.key === s.url) { st.img = im; draw(); } };
      im.onerror = () => { if (st.key === s.url) { st.key = ""; if (meta() && s.url === viewURL(meta())) delete node.properties.kubakub_crop; draw(); } };
      im.src = s.url;
    }
    return st.img ? s : null;
  }
  const curRatio = (W, H) => ratioValue(get("ratio"), W, H, ratioOf());

  // ---- view: the picture and the box (+ room around it to drag past the edge), fitted to the stage
  function viewOf(W, H, b) {
    const m = 0.12, x0 = Math.min(-m * W, b[0] * W), y0 = Math.min(-m * H, b[1] * H);
    const x1 = Math.max(W * (1 + m), (b[0] + b[2]) * W), y1 = Math.max(H * (1 + m), (b[1] + b[3]) * H);
    return { x0, y0, x1, y1 };
  }
  const geo = (W, H) => {
    const v = st.view || viewOf(W, H, box()), cw = cv.width, ch = cv.height;
    const s = Math.min(cw / (v.x1 - v.x0), ch / (v.y1 - v.y0)), ox = (cw - (v.x1 - v.x0) * s) / 2 - v.x0 * s, oy = (ch - (v.y1 - v.y0) * s) / 2 - v.y0 * s;
    return { s, ox, oy, X: px => ox + px * s, Y: py => oy + py * s, PX: x => (x - ox) / s, PY: y => (y - oy) / s };
  };
  const dpr = () => Math.min(devicePixelRatio || 1, 2);
  new ResizeObserver(() => { cv.width = Math.max(10, stage.clientWidth * dpr()); cv.height = Math.max(10, stage.clientHeight * dpr()); draw(); }).observe(stage);
  const evXY = e => { const r = cv.getBoundingClientRect(); return [(e.clientX - r.left) * cv.width / (r.width || 1), (e.clientY - r.top) * cv.height / (r.height || 1)]; };

  function handleAt(g, b, W, H, x, y) {              // which handle / inside / "" under a canvas point
    const L = g.X(b[0] * W), T = g.Y(b[1] * H), R = g.X((b[0] + b[2]) * W), B = g.Y((b[1] + b[3]) * H), t = 9 * dpr();
    const nx = Math.abs(x - L) < t ? "w" : Math.abs(x - R) < t ? "e" : "", ny = Math.abs(y - T) < t ? "n" : Math.abs(y - B) < t ? "s" : "";
    const inX = x > L - t && x < R + t, inY = y > T - t && y < B + t;
    if (nx && ny) return ny + nx;
    if (nx && inY) return nx; if (ny && inX) return ny;
    return x > L && x < R && y > T && y < B ? "move" : "";
  }
  const CURSOR = { n: "ns-resize", s: "ns-resize", e: "ew-resize", w: "ew-resize", ne: "nesw-resize", sw: "nesw-resize", nw: "nwse-resize", se: "nwse-resize", move: "move", "": "crosshair" };

  function drawFill(g, img, W, H, b, fill) {         // the added canvas, as the run will fill it
    const L = g.X(b[0] * W), T = g.Y(b[1] * H), R = g.X((b[0] + b[2]) * W), B = g.Y((b[1] + b[3]) * H);
    const pL = g.X(0), pT = g.Y(0), pR = g.X(W), pB = g.Y(H), iw = img.naturalWidth, ih = img.naturalHeight;
    ctx.save(); ctx.beginPath(); ctx.rect(L, T, R - L, B - T); ctx.clip();
    if (fill === "black" || fill === "white" || fill === "grey") {
      ctx.fillStyle = { black: "#000", white: "#fff", grey: "#808080" }[fill]; ctx.fillRect(L, T, R - L, B - T);
    } else if (fill === "mirror") {
      const w = pR - pL, h = pB - pT;
      for (let j = -3; j <= 3; j++) for (let i = -3; i <= 3; i++) {
        if (!i && !j) continue;
        const x = pL + i * w, y = pT + j * h; if (x > R || x + w < L || y > B || y + h < T) continue;
        ctx.save(); ctx.translate(x + (i % 2 ? w : 0), y + (j % 2 ? h : 0)); ctx.scale(i % 2 ? -1 : 1, j % 2 ? -1 : 1); ctx.drawImage(img, 0, 0, w, h); ctx.restore();
      }
    } else {                                         // edge: the border pixels stretched outward
      const big = 1e5;
      ctx.drawImage(img, 0, 0, 1, ih, pL - big, pT, big, pB - pT); ctx.drawImage(img, iw - 1, 0, 1, ih, pR, pT, big, pB - pT);
      ctx.drawImage(img, 0, 0, iw, 1, pL, pT - big, pR - pL, big); ctx.drawImage(img, 0, ih - 1, iw, 1, pL, pB, pR - pL, big);
      ctx.drawImage(img, 0, 0, 1, 1, pL - big, pT - big, big, big); ctx.drawImage(img, iw - 1, 0, 1, 1, pR, pT - big, big, big);
      ctx.drawImage(img, 0, ih - 1, 1, 1, pL - big, pB, big, big); ctx.drawImage(img, iw - 1, ih - 1, 1, 1, pR, pB, big, big);
    }
    ctx.restore();
  }

  function draw() {
    const D = dpr();
    ctx.setTransform(1, 0, 0, 1, 0, 0); ctx.fillStyle = "#101010"; ctx.fillRect(0, 0, cv.width, cv.height);
    const s = picture(); renderPills(s);
    if (!s) {
      ctx.fillStyle = "rgba(255,255,255,.5)"; ctx.font = `${13 * D}px "Hanken Grotesk KKD", sans-serif`; ctx.textAlign = "center";
      ctx.fillText(linked("image") < 0 ? "connect a picture" : "pick a picture in the node before, or run once", cv.width / 2, cv.height / 2);
      ctx.textAlign = "left"; hint.textContent = ""; return;
    }
    const { W, H } = s, b = box(), g = geo(W, H), img = st.img;
    const L = g.X(b[0] * W), T = g.Y(b[1] * H), R = g.X((b[0] + b[2]) * W), B = g.Y((b[1] + b[3]) * H);
    ctx.imageSmoothingQuality = "high";
    ctx.drawImage(img, g.X(0), g.Y(0), W * g.s, H * g.s);
    const past = b[0] < 0 || b[1] < 0 || b[0] + b[2] > 1 || b[1] + b[3] > 1;
    if (past) drawFill(g, img, W, H, b, get("fill") || "edge");
    ctx.fillStyle = "rgba(10,10,12,.62)";             // outside the box: dimmed
    ctx.beginPath(); ctx.rect(0, 0, cv.width, cv.height); ctx.rect(L, T, R - L, B - T); ctx.fill("evenodd");
    ctx.strokeStyle = "rgba(255,255,255,.18)"; ctx.lineWidth = D; ctx.setLineDash([4 * D, 4 * D]);
    ctx.strokeRect(g.X(0), g.Y(0), W * g.s, H * g.s); ctx.setLineDash([]);   // where the picture ends
    const r = curRatio(W, H), [px, py, pw, ph] = pixelBox(b, W, H, r, snap());
    ctx.strokeStyle = "rgba(255,255,255,.28)"; ctx.lineWidth = D;   // thirds (as the run cuts it)
    const cl = g.X(px), ct = g.Y(py), cr = g.X(px + pw), cb = g.Y(py + ph);
    for (const f of [1 / 3, 2 / 3]) { ctx.beginPath(); ctx.moveTo(cl + (cr - cl) * f, ct); ctx.lineTo(cl + (cr - cl) * f, cb); ctx.moveTo(cl, ct + (cb - ct) * f); ctx.lineTo(cr, ct + (cb - ct) * f); ctx.stroke(); }
    ctx.strokeStyle = "#fff"; ctx.lineWidth = 1.5 * D; ctx.strokeRect(cl, ct, cr - cl, cb - ct);
    const hs = 7 * D;
    for (const [hx, hy, k] of [[cl, ct, "nw"], [cr, ct, "ne"], [cl, cb, "sw"], [cr, cb, "se"], [(cl + cr) / 2, ct, "n"], [(cl + cr) / 2, cb, "s"], [cl, (ct + cb) / 2, "w"], [cr, (ct + cb) / 2, "e"]]) {
      ctx.fillStyle = st.hover === k || st.drag?.handle === k ? ORANGE : "#fff"; ctx.fillRect(hx - hs / 2, hy - hs / 2, hs, hs);
    }
    const [ow, oh] = longSide(pw, ph, get("size") || "as cut", snap());
    const say = `${pw} × ${ph} · ${ratioName(pw, ph)}${ow !== pw || oh !== ph ? `  →  ${ow} × ${oh}` : ""}${past ? "  · new area: " + (get("fill") || "edge") : ""}`;
    ctx.font = `600 ${12 * D}px "Hanken Grotesk KKD", sans-serif`; const tw = ctx.measureText(say).width + 14 * D;
    ctx.fillStyle = "rgba(20,18,24,.85)"; ctx.fillRect(8 * D, cv.height - 30 * D, tw, 22 * D);
    ctx.fillStyle = "#fff"; ctx.fillText(say, 15 * D, cv.height - 15 * D);
    hint.textContent = st.drag ? "shift = keep the ratio · alt = from the centre" : "drag = new box · inside = move · handles = size · past the edge = more canvas · double click = all";
  }

  // ---- pills
  let pillKey = "";
  function renderPills(s) {
    const of = ratioOf(), ratio = get("ratio") || "free", size = get("size") || "as cut", fill = get("fill") || "edge", safe = get("model_safe") !== false;
    const key = [ratio, size, fill, safe, !!of].join("|"); if (key === pillKey) return; pillKey = key;
    const row = (n, html) => { const r = el.querySelector(`[data-row=${n}]`); r.innerHTML = r.firstElementChild.outerHTML + html; };
    const shown = RATIOS.filter(n => n !== "picture" || of);
    const named = shown.includes(ratio) ? ratio : SWAP[ratio] && shown.includes(SWAP[ratio]) ? SWAP[ratio] : ratio;
    row("ratio", shown.map(n => `<button data-ratio="${n}" aria-pressed="${n === named}" title="${n === "picture" ? "the shape of the picture on 'ratio_of'" : n === "original" ? "the shape of this picture" : ""}">${n === named && ratio !== n ? ratio : n}</button>`).join("")
      + `<button data-swap title="turn the box: wide <-> tall">⇄</button>`);
    row("size", SIZES.map(n => `<button data-size="${n}" aria-pressed="${n === size}">${n === "3840" ? "3840 (4k)" : n}</button>`).join("")
      + `<button data-safe aria-pressed="${safe}" title="width and height in steps of 16 px (what Flux and most models want)">16 px steps</button>`);
    row("fill", FILLS.map(n => `<button data-fill="${n}" aria-pressed="${n === fill}" title="fills the canvas added past the edge">${n}</button>`).join("")
      + `<button data-reset class="act" title="the whole picture">reset</button>`);
  }
  function reshape(ratio) {                          // a ratio picked: the box takes it, same centre and area (inside stays inside)
    const s = source(); if (!s) return;
    const { W, H } = s, r = ratioValue(ratio, W, H, ratioOf()); if (!r) return;
    let [x, y, w, h] = box().map((v, i) => v * (i % 2 ? H : W));
    const inside = x >= -0.5 && y >= -0.5 && x + w <= W + 0.5 && y + h <= H + 0.5, a = w * h, cx = x + w / 2, cy = y + h / 2;
    w = Math.sqrt(a * r); h = w / r;
    if (inside) { const k = Math.min(1, W / w, H / h); w *= k; h *= k; }
    x = cx - w / 2; y = cy - h / 2;
    if (inside) { x = Math.min(Math.max(x, 0), W - w); y = Math.min(Math.max(y, 0), H - h); }
    set("box", boxText([x / W, y / H, w / W, h / H]));
  }
  el.addEventListener("click", e => {
    const t = e.target.closest("button"); if (!t) return; e.stopPropagation();
    if (t.dataset.ratio) { set("ratio", t.dataset.ratio); reshape(t.dataset.ratio); }
    else if (t.dataset.size) set("size", t.dataset.size);
    else if (t.dataset.fill) set("fill", t.dataset.fill);
    else if ("safe" in t.dataset) set("model_safe", get("model_safe") === false);
    else if ("swap" in t.dataset) {
      const ratio = get("ratio") || "free", s = source();
      if (SWAP[ratio]) { set("ratio", SWAP[ratio]); reshape(SWAP[ratio]); }
      else if (s) { const { W, H } = s, [x, y, w, h] = box(), cx = (x + w / 2) * W, cy = (y + h / 2) * H, nw = h * H, nh = w * W;
        if (ratio !== "free") set("ratio", "free");
        set("box", boxText([(cx - nw / 2) / W, (cy - nh / 2) / H, nw / W, nh / H])); }
    } else if ("reset" in t.dataset) reset();
    st.view = null; draw();
  });
  function reset() {
    const s = source(); if (!s) { set("box", "0 0 1 1"); return; }
    const { W, H } = s, [x, y, w, h] = fitRatio(0, 0, W, H, curRatio(W, H));
    set("box", boxText([x / W, y / H, w / W, h / H]));
  }

  // ---- mouse and keys
  cv.addEventListener("pointerdown", e => {
    if (e.button !== 0) return;
    const s = source(); if (!s || !st.img) return;
    e.preventDefault(); e.stopPropagation(); cv.focus({ preventScroll: true });
    const { W, H } = s, b = box(); st.view = viewOf(W, H, b);
    const g = geo(W, H), [x, y] = evXY(e);
    const whole = b.every((v, i) => Math.abs(v - [0, 0, 1, 1][i]) < 1e-4);
    let h = handleAt(g, b, W, H, x, y);
    if (h === "move" && whole) h = "";               // the box is still all of the picture: a drag inside draws a new one
    const p = [g.PX(x), g.PY(y)], b0 = b.map((v, i) => v * (i % 2 ? H : W));
    st.drag = h ? { handle: h, start: p, b0, box: b } : { handle: "se", start: p, b0: [p[0], p[1], 0, 0], box: b, fresh: true };
    cv.setPointerCapture(e.pointerId);
  });
  cv.addEventListener("pointermove", e => {
    const s = source(); if (!s || !st.img) return;
    const { W, H } = s, g = geo(W, H), [x, y] = evXY(e);
    if (!st.drag) { const h = handleAt(g, box(), W, H, x, y); if (h !== st.hover) { st.hover = h; cv.style.cursor = CURSOR[h]; draw(); } return; }
    e.stopPropagation();
    const d = st.drag, px = Math.min(Math.max(g.PX(x), -REACH * W), (1 + REACH) * W), py = Math.min(Math.max(g.PY(y), -REACH * H), (1 + REACH) * H);
    const tol = 8 * dpr() / g.s, near = (v, ...to) => { for (const t of to) if (Math.abs(v - t) < tol) return t; return v; };
    let nb;
    if (d.handle === "move") {
      let nx = d.b0[0] + px - d.start[0], ny = d.b0[1] + py - d.start[1];
      const sx = near(nx, 0, W - d.b0[2]); nx = sx !== nx ? sx : near(nx + d.b0[2], W, 0) - d.b0[2];   // snap to the picture's edges
      const sy = near(ny, 0, H - d.b0[3]); ny = sy !== ny ? sy : near(ny + d.b0[3], H, 0) - d.b0[3];
      nb = [nx, ny, d.b0[2], d.b0[3]];
    } else {
      if (d.fresh && Math.hypot(px - d.start[0], py - d.start[1]) * g.s < 4 * dpr()) return;
      const fr = curRatio(W, H) || (e.shiftKey ? (d.b0[2] && d.b0[3] ? d.b0[2] / d.b0[3] : W / H) : null);
      nb = dragBox(d.b0, d.handle, near(px, 0, W), near(py, 0, H), fr, e.altKey && !d.fresh, Math.max(4, 8 / g.s));
    }
    d.box = [nb[0] / W, nb[1] / H, nb[2] / W, nb[3] / H]; d.moved = true;
    draw();
  });
  const drop = e => {
    const d = st.drag; if (!d) return; st.drag = null;
    try { cv.releasePointerCapture(e.pointerId); } catch { /* already released */ }
    if (d.moved) set("box", boxText(d.box));
    st.view = null; draw();
  };
  cv.addEventListener("pointerup", drop);
  cv.addEventListener("pointercancel", drop);
  cv.addEventListener("pointerleave", () => { if (!st.drag && st.hover) { st.hover = ""; draw(); } });
  cv.addEventListener("dblclick", e => { e.preventDefault(); e.stopPropagation(); reset(); st.view = null; draw(); });
  cv.addEventListener("keydown", e => {
    const s = source(); if (!s || !e.key.startsWith("Arrow")) return;
    e.preventDefault(); e.stopPropagation();
    const n = e.shiftKey ? 10 : 1, [x, y, w, h] = box();
    const dx = e.key === "ArrowLeft" ? -n : e.key === "ArrowRight" ? n : 0, dy = e.key === "ArrowUp" ? -n : e.key === "ArrowDown" ? n : 0;
    set("box", boxText([x + dx / s.W, y + dy / s.H, w, h])); draw();
  });
  cv.addEventListener("wheel", e => e.stopPropagation(), { passive: true });

  let seen = "";                                     // the node before shows another picture, a widget changed: redraw
  const watch = setInterval(() => {
    if (!el.isConnected || st.drag) return;
    const u = upImage("image")?.src || "", o = upImage("ratio_of")?.src || "";
    const k = [u, o, get("box"), get("ratio"), get("size"), get("fill"), get("model_safe"), linked("ratio_of")].join("|");
    if (k !== seen) { seen = k; pillKey = ""; draw(); }
  }, 300);
  el.kkcStop = () => clearInterval(watch);
  el.kkcDraw = () => { pillKey = ""; draw(); };
  return el;
}

app.registerExtension({
  name: "kubakub.crop",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== "KUBA_Crop") return;
    const created = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = created?.apply(this, arguments);
      for (const n of HIDDEN) hideWidget(this.widgets?.find(w => w.name === n));
      const el = build(this);
      const w = this.addDOMWidget("crop_view", "kubakub_crop", el, { serialize: false, hideOnZoom: false, getMinHeight: () => 360 });
      w.serialize = false;
      this.kkcView = el;
      if ((this.size?.[1] || 0) < 470) this.setSize?.([Math.max(this.size?.[0] || 0, 380), 470]);
      return r;
    };
    const configured = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function () {
      const r = configured?.apply(this, arguments);
      for (const n of HIDDEN) hideWidget(this.widgets?.find(w => w.name === n));
      requestAnimationFrame(() => this.kkcView?.kkcDraw());
      return r;
    };
    const executed = nodeType.prototype.onExecuted;
    nodeType.prototype.onExecuted = function (msg) {
      const r = executed?.apply(this, arguments);
      const m = msg?.kubakub_crop?.[0];
      if (m) { (this.properties ||= {}).kubakub_crop = m; this.kkcView?.kkcDraw(); }
      return r;
    };
    const removed = nodeType.prototype.onRemoved;
    nodeType.prototype.onRemoved = function () { this.kkcView?.kkcStop?.(); return removed?.apply(this, arguments); };
  },
});
