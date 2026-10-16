"""
Model free test of the model adapters (kubakub/adapters.py): which adapter a model gets, and what the Flux 3 one
does (the crop as reference on the positive and the negative, 50 steps cfg 3.5 for the base, 20 steps cfg 1 for the
step distilled file or a LoRA on the base, the Flux 2 grid). Model objects are stand-ins of the core classes, made
without weights. Does not start ComfyUI and stays on the CPU.

Run from the ComfyUI_windows_portable folder:
    python_embeded\\python.exe ComfyUI\\custom_nodes\\ComfyUI_KubaNodes\\tests\\test_adapters.py
"""

import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(HERE))))     # ComfyUI
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

sys.argv = sys.argv[:1] + ["--cpu"]
import comfy.options  # noqa: E402
comfy.options.enable_args_parsing()
import comfy.model_base  # noqa: E402

import _pack  # noqa: E402,F401
from kubakub import adapters  # noqa: E402

failures = []


def check(name, cond, detail=""):
    print(("ok   " if cond else "FAIL ") + name + (f"  ({detail})" if detail and not cond else ""))
    if not cond:
        failures.append(name)


class DM:                                   # the diffusion model: only what the adapters look at
    def __init__(self, flow_map=False):
        self.flow_map = flow_map


class Patcher:                              # a ModelPatcher stand-in around a core model class made without weights
    def __init__(self, klass, flow_map=False, patches=None):
        self.model = klass.__new__(klass)
        self.patches = patches or {}
        self._dm = DM(flow_map)

    def get_model_object(self, name):
        if name == "diffusion_model":
            return self._dm
        raise AttributeError(name)


class Clip:                                 # text -> one conditioning entry, the text kept to recognise it
    def tokenize(self, text):
        return text

    def encode_from_tokens_scheduled(self, tokens):
        return [[torch.zeros(1, 4, 8), {"text": tokens}]]


def make(klass, **kw):
    return adapters.make_adapter(Patcher(klass, **kw), Clip(), None)


# ---- which adapter --------------------------------------------------------------------------------------------
check("flux3 is in the family list", "flux3" in adapters.families())
check("a Flux 3 model gets the Flux 3 adapter", type(make(comfy.model_base.Flux3)) is adapters.Flux3Adapter)
check("a Flux 2 model still gets the Flux 2 adapter", type(make(comfy.model_base.Flux2)) is adapters.Flux2Adapter)
check("family 'flux3' can be chosen by name",
      type(adapters.make_adapter(Patcher(comfy.model_base.Flux3), Clip(), None, family="flux3")) is adapters.Flux3Adapter)

# ---- base / distilled defaults ----------------------------------------------------------------------------------
base = make(comfy.model_base.Flux3)
check("the base: 50 steps, cfg 3.5", (base.default_steps, base.default_cfg) == (50, 3.5),
      f"{base.default_steps} {base.default_cfg}")
sd = make(comfy.model_base.Flux3, flow_map=True)
check("the step distilled file (flow map): 20 steps, cfg 1", (sd.default_steps, sd.default_cfg) == (20, 1.0),
      f"{sd.default_steps} {sd.default_cfg}")
lora = make(comfy.model_base.Flux3, patches={"diffusion_model.x": [object()]})
check("the base with a LoRA (the distill LoRA): 20 steps, cfg 1", (lora.default_steps, lora.default_cfg) == (20, 1.0))
check("the Flux 3 grid is the Flux 2 one (16 px)", base.geometry.divisor == 16 and base.geometry.family == "flux2")

# ---- references -------------------------------------------------------------------------------------------------
lat = torch.ones(1, 128, 4, 4)
pos, neg = base.conds_for("a stained glass window", "", None, lat)
check("the crop is the reference of the positive", len(pos[0][1].get("reference_latents", [])) == 1)
check("... and of the negative too (core ReferenceMultiLatent; the base runs at cfg 3.5)",
      len(neg[0][1].get("reference_latents", [])) == 1)
check("the prompts stay what they were", pos[0][1]["text"] == "a stained glass window" and neg[0][1]["text"] == "")
p2, n2 = base.conds_for("a stained glass window", "")
check("without a crop there is no reference", "reference_latents" not in p2[0][1] and "reference_latents" not in n2[0][1])
p3 = base.add_reference(pos, lat * 2)
check("a second reference (a style picture) is added after the crop", len(p3[0][1]["reference_latents"]) == 2)
p4, n4 = base.add_extra(pos, neg, lat * 2)
check("Flux 3: a style picture goes on the positive and the negative",
      len(p4[0][1]["reference_latents"]) == 2 and len(n4[0][1]["reference_latents"]) == 2)
f2 = make(comfy.model_base.Flux2)
p5, n5 = f2.add_extra(*f2.conds_for("x", "", None, lat), lat * 2)
check("Flux 2: a style picture goes on the positive only, as before",
      len(p5[0][1]["reference_latents"]) == 2 and "reference_latents" not in n5[0][1])

print()
print("FAILED: " + ", ".join(failures) if failures else "all adapter tests passed")
sys.exit(1 if failures else 0)
