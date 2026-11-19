// Shared by the timeline (kubakub_master.js), the shelf (kubakub_shelf.js) and the director: what kind a file is,
// a source ('file:<input/>' | 'output:<output/>') as a /view link and back, the drag type every one of them reads,
// the upload into input/kuba_director, and the small element builder. No extension of its own.
import { api } from "../../scripts/api.js";

export const DRAG_MIME = "application/x-kubakub-master";       // {source} or {director}: timeline, shelf and director read it
const VIDEO_RE = /\.(mp4|mov|webm|mkv|avi|m4v|gif)$/i, IMAGE_RE = /\.(png|jpe?g|webp|bmp|tiff?)$/i, AUDIO_RE = /\.(wav|mp3|ogg|flac|m4a|aac|opus)$/i;
export const kindOf = name => (VIDEO_RE.test(name) ? "video" : IMAGE_RE.test(name) ? "image" : AUDIO_RE.test(name) ? "audio" : null);

export function sourceURL(source) {                  // 'file:<input/>' | 'output:<output/>' -> /view
  const [type, rest] = source.startsWith("output:") ? ["output", source.slice(7)] : ["input", source.replace(/^file:/, "")];
  const i = rest.lastIndexOf("/");
  return api.apiURL(`/view?${new URLSearchParams({ filename: rest.slice(i + 1), subfolder: i < 0 ? "" : rest.slice(0, i), type })}`);
}
export const fileName = source => source.replace(/^(file|output):/, "").split("/").pop();   // 'output:sub/a.png' -> 'a.png'
// a preview dragged out of a node of THIS ComfyUI (/view?filename=&subfolder=&type=) -> {source, name}; a temp preview
// -> {temp, name} (temp files are wiped on restart, so whoever keeps it copies it in); any other link -> null
export function sourceFromView(url) {
  try {
    const u = new URL(url, location.href); if (u.origin !== location.origin || !u.pathname.endsWith("/view")) return null;
    const f = u.searchParams.get("filename"), sub = u.searchParams.get("subfolder") || "", type = u.searchParams.get("type") || "output";
    if (!f || !kindOf(f)) return null;
    if (type === "temp") return { temp: u.href, name: f };
    if (type !== "output" && type !== "input") return null;
    return { source: `${type === "output" ? "output:" : "file:"}${sub ? sub + "/" : ""}${f}`, name: f };
  } catch { return null; }
}
// a file into input/kuba_director (streamed, any size; the same file again is reused) -> 'file:kuba_director/<name>';
// with a folder name: input/kuba_director/<folder>/<name> (a folder dropped on the shelf stays together)
export async function uploadSource(file, name = file.name, folder = "") {
  const fd = new FormData(); if (folder) fd.append("folder", folder); fd.append("file", file, name);
  const r = await api.fetchApi("/kubakub/master/upload", { method: "POST", body: fd }), j = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(j.error || r.statusText);
  return "file:" + j.name;
}

export function h(tag, attrs = {}, ...kids) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k.startsWith("on")) e.addEventListener(k.slice(2), v); else if (k === "class") e.className = v; else if (v != null && v !== false) e.setAttribute(k, v === true ? "" : v);
  }
  for (const c of kids.flat()) if (c != null) e.append(c);
  return e;
}

// ---- folders dropped from the desktop (drag and drop file system entries)
// What was dropped -> {files: [{entry, top}], skipped, capped}: every picture, video, sound and text (.txt / .md)
// below the dropped folders, `top` = the name of the dropped folder it came from ('' for a loose file). Read the
// entries (DataTransferItem.webkitGetAsEntry) inside the drop event: the list is gone after it.
export const TEXT_RE = /\.(txt|md)$/i;
export async function walkEntries(entries, max = 500) {
  const files = []; let skipped = 0, capped = false;
  const readAll = dir => new Promise(res => {             // readEntries hands out at most ~100 at a time
    const r = dir.createReader(), all = [];
    const next = () => r.readEntries(b => { if (!b.length) return res(all); all.push(...b); next(); }, () => res(all));
    next();
  });
  const walk = async (e, top) => {
    if (capped) return;
    if (e.isDirectory) {
      const kids = (await readAll(e)).sort((a, b) => a.name.localeCompare(b.name, undefined, { numeric: true }));
      for (const k of kids) { if (k.name.startsWith(".")) continue; await walk(k, top); }
    } else if (e.isFile) {
      if (!kindOf(e.name) && !TEXT_RE.test(e.name)) { skipped++; return; }
      if (files.length >= max) { capped = true; return; }
      files.push({ entry: e, top });
    }
  };
  for (const e of entries) await walk(e, e.isDirectory ? e.name : "");
  return { files, skipped, capped };
}
export const entryFile = e => new Promise((res, rej) => e.file(res, rej));
// run `fn` over `list`, `n` at a time; `tick(done)` after each
export async function inBatches(list, n, fn, tick) {
  const out = new Array(list.length); let i = 0, done = 0;
  const worker = async () => { while (i < list.length) { const k = i++; out[k] = await fn(list[k], k).catch(e => ({ error: e })); tick?.(++done); } };
  await Promise.all(Array.from({ length: Math.min(n, list.length) }, worker));
  return out;
}
// ComfyUI's /history -> the files the runs saved in output/, newest run first: [{filename, subfolder, type}]
export function historyFiles(history) {
  const runs = Object.values(history || {}).filter(r => r && r.outputs)
    .sort((a, b) => (b.prompt?.[0] ?? 0) - (a.prompt?.[0] ?? 0));
  const out = [], seen = new Set();
  for (const r of runs) for (const node of Object.values(r.outputs)) for (const list of Object.values(node || {})) {
    if (!Array.isArray(list)) continue;
    for (const f of list) {
      if (!f || typeof f.filename !== "string" || (f.type || "output") !== "output" || !kindOf(f.filename)) continue;
      const key = (f.subfolder || "") + "/" + f.filename; if (seen.has(key)) continue; seen.add(key);
      out.push({ filename: f.filename, subfolder: f.subfolder || "", type: "output" });
    }
  }
  return out;
}
export const outputSource = f => "output:" + (f.subfolder ? f.subfolder + "/" : "") + f.filename;
