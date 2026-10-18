// Every kubakub node on the classic canvas: the fine-tuning inputs (advanced=True in the node's schema) fold away
// behind one 'more settings' row; the classic canvas otherwise shows all of them. The Vue nodes (Nodes 2.0) have
// their own 'Show advanced inputs' at the bottom of each node, so there this file stays out (one way per thing,
// and the Vue nodes do not redraw a widget that comes back from hidden). An advanced input still shows while its value differs from the
// default or a cable is connected to it, so nothing that changes the result is ever hidden. Open / closed is kept
// per node in the workflow. Only hides widgets: values, the saved workflow and a run are the same without this file.
// Plain ES module, no network.
import { app } from "../../scripts/app.js";

const TOGGLE = "kubakub more settings";

function advancedDefaults(nodeData) {
  const out = {};
  for (const group of ["required", "optional"]) {
    for (const [name, spec] of Object.entries(nodeData.input?.[group] || {})) {
      const opts = Array.isArray(spec) ? spec[1] || {} : {};
      if (opts.advanced) out[name] = opts.default;
    }
  }
  return out;
}

function vueNodes() {
  try { return !!app.extensionManager?.setting?.get?.("Comfy.VueNodes.Enabled"); } catch { return false; }
}

const same = (a, b) => (typeof a === "number" && typeof b === "number" ? Math.abs(a - b) < 1e-9
  : String(a ?? "") === String(b ?? ""));

function fold(node) {
  const adv = node.kkAdvanced;
  if (!adv) return;
  const vue = vueNodes();                            // switched to the Vue nodes: give back what this file hid
  const open = vue || !!node.properties?.kk_more_open;
  let folded = 0, changed = false;
  for (const w of node.widgets || []) {
    if (!(w.name in adv)) continue;
    const linked = node.inputs?.some(i => (i.widget?.name === w.name || i.name === w.name) && i.link != null);
    const hide = !open && !linked && same(w.value, adv[w.name]);
    if (hide) folded++;
    if (!!w.kkFolded !== hide) { w.kkFolded = hide; w.hidden = hide; changed = true; }
  }
  const t = node.widgets?.find(w => w.name === TOGGLE);
  if (t) {
    const any = !vue && Object.keys(adv).length > 0;
    t.label = open ? "fewer settings ▴" : `more settings (${folded}) ▾`;
    if (!!t.hidden !== !any) { t.hidden = !any; changed = true; }
  }
  if (changed && !vueNodes()) node.setSize([node.size[0], node.computeSize()[1]]);   // the Vue nodes size themselves
  node.setDirtyCanvas(true, true);
}

app.registerExtension({
  name: "kubakub.advanced",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (!String(nodeData.category || "").startsWith("kubakub")) return;
    const adv = advancedDefaults(nodeData);
    if (!Object.keys(adv).length) return;
    const created = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = created?.apply(this, arguments);
      this.kkAdvanced = adv;
      this.addWidget("button", TOGGLE, null, () => {
        this.properties = this.properties || {};
        this.properties.kk_more_open = !this.properties.kk_more_open;
        fold(this);
      }, { serialize: false }).serialize = false;
      const node = this;
      for (const w of this.widgets || []) {             // a value set back to its default folds away again
        if (!(w.name in adv)) continue;
        const cb = w.callback;
        w.callback = function () { const out = cb?.apply(this, arguments); setTimeout(() => fold(node), 0); return out; };
      }
      setTimeout(() => fold(this), 0);
      return r;
    };
    const configure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function () {      // a loaded workflow: its values are in now
      const r = configure?.apply(this, arguments);
      setTimeout(() => fold(this), 0);
      return r;
    };
    const conn = nodeType.prototype.onConnectionsChange;
    nodeType.prototype.onConnectionsChange = function () {
      const r = conn?.apply(this, arguments);
      setTimeout(() => fold(this), 0);
      return r;
    };
  },
});
