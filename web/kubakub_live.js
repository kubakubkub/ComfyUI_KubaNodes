// kubakub scene render with source = live blender / live houdini: a 'refresh from <app>' button that runs only this
// node (and what feeds it), and the open app (its kubakub live link, tools/blender or tools/houdini) sends the scene
// as it is now; the node's preview shows it. Run the workflow afterwards for the rest: the render comes from the cache.
// 'live camera' is a dropdown of the app's cameras (the names the app's link writes into its .alive file every 2 s,
// read through /kubakub/scene3d/live_cameras every few seconds while live) and what you look through; the choice is
// the 'camera' field, which stays the truth.
// The 'file' and 'camera' fields are hidden while the source is live. Only adds buttons and hides fields: a workflow
// without this file runs the same. Plain ES module; the run goes to this ComfyUI server only.
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const BUTTON = "refresh live";
const PICK = "live camera";
const LOOK = "what you look through";
const MISSING = "  (not in the scene now)";

function vueNodes() {
  try { return !!app.extensionManager?.setting?.get?.("Comfy.VueNodes.Enabled"); } catch { return false; }
}

const camText = node => (node.widgets?.find(w => w.name === "camera")?.value || "").trim();

// the app's camera names, asked every few seconds while the source is live (the app rewrites them every 2 s)
async function fetchCameras(node) {
  const source = node.widgets?.find(w => w.name === "source")?.value || "";
  if (!source.startsWith("live")) return;
  try {
    const j = await (await api.fetchApi("/kubakub/scene3d/live_cameras?source=" + encodeURIComponent(source))).json();
    node.kkLiveCams = Array.isArray(j.cameras) ? j.cameras : [];
    node.kkLinked = !!j.linked;
  } catch (e) { node.kkLiveCams = []; node.kkLinked = false; }
  tidy(node);
}

function cameraChoices(node) {
  const cur = camText(node), names = node.kkLiveCams || [];
  return [LOOK, ...names, ...(cur && !names.includes(cur) ? [cur + MISSING] : [])];
}

function setCamera(node, v) {
  const w = node.widgets?.find(x => x.name === "camera");
  if (!w) return;
  w.value = !v || v === LOOK ? "" : String(v).replace(MISSING, "");
  w.callback?.(w.value);
  tidy(node);
}

async function refresh(node) {
  try {
    const p = await app.graphToPrompt();
    const id = String(node.id);
    if (!p.output?.[id]) throw new Error("the scene render node is bypassed or muted");
    const r = await api.fetchApi("/prompt", { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ prompt: p.output, client_id: api.clientId ?? api.initialClientId ?? undefined,
        extra_data: { extra_pnginfo: { workflow: p.workflow } }, partial_execution_targets: [id] }) });
    const j = await r.json().catch(() => ({}));
    if (!r.ok || !j.prompt_id) throw new Error(j.error?.message || "queue refused (" + r.status + ")");
  } catch (e) {
    console.error("[kubakub live] refresh", e);
    app.extensionManager?.toast?.add({ severity: "error", summary: "refresh failed", detail: e.message, life: 6000 });
  }
}

function tidy(node) {
  const source = node.widgets?.find(w => w.name === "source")?.value || "file";
  const live = source.startsWith("live");
  let changed = false;
  for (const w of node.widgets || []) {
    const hide = w.name === "file" || w.name === "camera" ? live : w.name === BUTTON || w.name === PICK ? !live : null;
    if (hide !== null && !!w.hidden !== hide) { w.hidden = hide; changed = true; }
    if (w.name === BUTTON) w.label = `refresh from ${source.replace(/^live\s*/, "") || "3d app"}`;
    if (w.name === PICK) {
      const cur = camText(node);
      w.value = !cur ? LOOK : (node.kkLiveCams || []).includes(cur) || !node.kkLiveCams ? cur : cur + MISSING;
      w.label = node.kkLinked === false ? `camera (${source.replace(/^live\s*/, "")} not linked)` : "camera";
    }
  }
  if (changed && !vueNodes()) node.setSize([node.size[0], node.computeSize()[1]]);   // the Vue nodes size themselves
  node.setDirtyCanvas(true, true);
}

app.registerExtension({
  name: "kubakub.live",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== "KUBA_SceneRender") return;
    const created = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = created?.apply(this, arguments);
      this.properties = this.properties || {};
      this.properties.kk_unit_auto = 1;                  // a new node starts at auto already
      this.addWidget("combo", PICK, LOOK, v => setCamera(this, v), { values: () => cameraChoices(this), serialize: false })
        .serialize = false;                              // the 'camera' field holds the choice
      this.kkCamTimer = setInterval(() => { if (this.graph) fetchCameras(this); }, 3000);
      this.addWidget("button", BUTTON, null, () => refresh(this), { serialize: false }).serialize = false;
      const w = this.widgets?.find(x => x.name === "source"), cb = w?.callback;
      if (w) w.callback = (...a) => {                    // left a live app: its camera names mean nothing elsewhere,
        const out = cb?.apply(w, a);                     // start from what you look through
        if (w.value !== this.kkSource && String(this.kkSource || "").startsWith("live")) { this.kkLiveCams = null; setCamera(this, ""); }
        this.kkSource = w.value;
        tidy(this); fetchCameras(this); return out;
      };
      this.kkSource = w?.value;
      setTimeout(() => fetchCameras(this), 0);
      setTimeout(() => tidy(this), 0);
      return r;
    };
    const removed = nodeType.prototype.onRemoved;
    nodeType.prototype.onRemoved = function () { clearInterval(this.kkCamTimer); return removed?.apply(this, arguments); };
    const configure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function (info) {
      const r = configure?.apply(this, arguments);
      this.kkSource = this.widgets?.find(x => x.name === "source")?.value;   // the loaded source
      // unit_scale 0 = auto is the default since 2026-10: a node saved with the old default 1.0 becomes auto once
      // (auto keeps 1.0 for any model up to 300 m, so only files in cm / mm change); a 1.0 typed later stays
      const us = this.widgets?.find(x => x.name === "unit_scale");
      this.properties = this.properties || {};
      if (us && !info?.properties?.kk_unit_auto && us.value === 1) us.value = 0;
      this.properties.kk_unit_auto = 1;
      setTimeout(() => { tidy(this); fetchCameras(this); }, 0);
      return r;
    };
  },
});
