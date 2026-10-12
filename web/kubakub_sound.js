// The 'sound' cable of kubakub sound: a node that listens hides its own sound settings while the cable is connected,
// because the cable brings them (tempo, gain, release, band, threshold, gap, where the sound starts). Pull the cable
// off and they are back. Only shows and hides the node's own widgets: a workflow without this file runs the same.
// Plain ES module, no build step, no network.
import { app } from "../../scripts/app.js";

const OWN = {
  KUBA_Shader: ["bpm", "gain", "release"],
  KUBA_MaskAnimate: ["listen", "gain", "release", "threshold", "gap", "bpm", "start"],
  KUBA_SoundMaskSwap: ["threshold", "gap", "bpm", "start"],
};

function tidy(node, names) {
  const on = node.inputs?.find(i => i.name === "sound")?.link != null;
  let changed = false;
  for (const w of node.widgets || []) if (names.includes(w.name) && !!w.hidden !== on) { w.hidden = on; changed = true; }
  if (changed) { node.setSize([node.size[0], Math.max(node.computeSize()[1], on ? 0 : node.size[1])]); node.setDirtyCanvas(true, true); }
}

app.registerExtension({
  name: "kubakub.sound",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name === "KUBA_Sound") {                // room for the picture of the sound under the settings, once
      const executed = nodeType.prototype.onExecuted;
      nodeType.prototype.onExecuted = function (msg) {
        const r = executed?.apply(this, arguments);
        if (msg?.images?.length && !this.properties?.kksGrown) {
          (this.properties ||= {}).kksGrown = true;
          this.setSize([Math.max(this.size[0], 440), this.size[1] + 170]);
        }
        return r;
      };
      return;
    }
    const names = OWN[nodeData.name];
    if (!names) return;
    for (const hook of ["onNodeCreated", "onConfigure", "onConnectionsChange"]) {
      const before = nodeType.prototype[hook];
      nodeType.prototype[hook] = function () { const r = before?.apply(this, arguments); setTimeout(() => tidy(this, names), 0); return r; };
    }
  },
});
