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
// a file into input/kuba_director (streamed, any size; the same file again is reused) -> 'file:kuba_director/<name>'
export async function uploadSource(file, name = file.name) {
  const fd = new FormData(); fd.append("file", file, name);
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
