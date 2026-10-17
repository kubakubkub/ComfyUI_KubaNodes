// kubakub scene render with source = live blender / live houdini: a 'refresh from <app>' button that runs the
// workflow, and the open app (its kubakub live link, tools/blender or tools/houdini) sends the scene as it is now.
// Nothing changed in the app = the cached result, so only what the change touches runs again. The 'file' field is
// hidden while the source is live. Only adds the button and hides the field: a workflow without this file runs the
// same. Plain ES module, no network.
import { app } from "../../scripts/app.js";

const BUTTON = "refresh live";

function tidy(node) {
  const source = node.widgets?.find(w => w.name === "source")?.value || "file";
  const live = source.startsWith("live");
  let changed = false;
  for (const w of node.widgets || []) {
    const hide = w.name === "file" ? live : w.name === BUTTON ? !live : null;
    if (hide !== null && !!w.hidden !== hide) { w.hidden = hide; changed = true; }
    if (w.name === BUTTON) w.label = `refresh from ${source.replace(/^live\s*/, "") || "3d app"}`;
  }
  if (changed) node.setSize([node.size[0], node.computeSize()[1]]);
  node.setDirtyCanvas(true, true);
}

app.registerExtension({
  name: "kubakub.live",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== "KUBA_SceneRender") return;
    const created = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = created?.apply(this, arguments);
      this.addWidget("button", BUTTON, null, () => app.queuePrompt(0, 1), { serialize: false });
      const w = this.widgets?.find(x => x.name === "source"), cb = w?.callback;
      if (w) w.callback = (...a) => { const out = cb?.apply(w, a); tidy(this); return out; };
      setTimeout(() => tidy(this), 0);
      return r;
    };
    const configure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function () { const r = configure?.apply(this, arguments); setTimeout(() => tidy(this), 0); return r; };
  },
});
