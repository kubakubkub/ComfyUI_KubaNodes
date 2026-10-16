"""
Model free test of the director's diffuse presets (nodes/director/nodes_director.py, diffusion_manifest): which
installed files a preset picks, against a made-up model folder. Does not start ComfyUI, loads no model, no GL.

Run from the ComfyUI_windows_portable folder:
    python_embeded\\python.exe ComfyUI\\custom_nodes\\ComfyUI_KubaNodes\\tests\\test_director_presets.py
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(HERE))))     # ComfyUI
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

sys.argv = sys.argv[:1] + ["--cpu"]
import comfy.options  # noqa: E402
comfy.options.enable_args_parsing()
import folder_paths  # noqa: E402

import _pack  # noqa: E402,F401
from kubapack.nodes.director import nodes_director as nd  # noqa: E402

failures = []


def check(name, cond, detail=""):
    print(("ok   " if cond else "FAIL ") + name + (f"  ({detail})" if detail and not cond else ""))
    if not cond:
        failures.append(name)


FILES = {
    "diffusion_models": ["flux3\\flux3_image_klein_base_int8_convrot.safetensors",
                         "flux3\\flux3_image_klein_sd_int8_convrot.safetensors",
                         "flux_tools\\flux2\\flux-2-klein-4b.safetensors",
                         "flux_tools\\flux2\\flux-2-klein-4b-nvfp4.safetensors"],
    "text_encoders": ["qwen3.5_9b_flux3_prompt_upsampler_w4a8.safetensors", "qwen3vl_4b_fp8_scaled.safetensors",
                      "qwen3vl_4b_int8_convrot.safetensors", "qwen_3_4b.safetensors", "qwen_3_4b_fp4_flux2.safetensors"],
    "vae": ["flux2-vae.safetensors", "qwen_image_lowgpu_vae.safetensors"],
    "loras": ["Flux\\flux3\\flux3_klein_sd_lora_rank_256_bf16.safetensors"],
}
keep = folder_paths.get_filename_list
folder_paths.get_filename_list = lambda kind: list(FILES.get(kind, []))
try:
    m = nd.diffusion_manifest()
    names = [p["name"] for p in m["presets"]]
    f3 = next((p for p in m["presets"] if p["name"] == "flux 3 klein"), None)
    check("flux 3 klein is offered when its files are there, first (the window's default)", names[:1] == ["flux 3 klein"],
          str(names))
    check("... the step distilled file, not the base", f3 and f3["unet"].endswith("flux3_image_klein_sd_int8_convrot.safetensors"))
    check("... with the Flux 3 encoder, not another qwen3vl 4b build nor the upsampler",
          f3 and f3["clip"] == "qwen3vl_4b_int8_convrot.safetensors", f3 and f3["clip"])
    check("... loaded as type flux3, 8 steps, flux2 vae", f3 and f3["type"] == "flux3" and f3["steps"] == 8
          and f3["vae"] == "flux2-vae.safetensors")
    check("the Klein 2 presets are still offered", "klein 4b nvfp4" in names and "klein 4b" in names)
    FILES["diffusion_models"] = FILES["diffusion_models"][2:]
    names2 = [p["name"] for p in nd.diffusion_manifest()["presets"]]
    check("without the Flux 3 file the window starts with Klein 2 as before", names2[:1] == ["klein 4b nvfp4"], str(names2))
finally:
    folder_paths.get_filename_list = keep

print()
print("FAILED: " + ", ".join(failures) if failures else "all director preset tests passed")
sys.exit(1 if failures else 0)
