"""
add_model_links.py

Writes the download link of every known model file into the loader nodes of the example workflows
(`properties.models`, the same field ComfyUI's own templates use), so that ComfyUI's "missing models" dialog can
offer the files when someone opens a workflow without them. The nodes themselves still download nothing.

The links come from MODEL_SETS in setup_kubakub.py plus EXTRA below. A loader whose file is not in the table is
left alone and listed, so nothing points at a file that was not checked.

    python tools/add_model_links.py            write the links
    python tools/add_model_links.py --check    only report (exit code 1 when a workflow would change)

Standard library only, no network.
"""

from __future__ import annotations

import glob
import importlib.util
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PACK = os.path.dirname(HERE)
HF = "https://huggingface.co/"

# Files the workflows name that are not part of a setup set: (file name, folder under ComfyUI/models, url)
EXTRA = [
    ("qwen3vl_8b_w4a8.safetensors", "text_encoders",
     HF + "Comfy-Org/Qwen-Image-2.1/resolve/main/text_encoders/qwen3vl_8b_w4a8.safetensors"),
    ("qwen3.5_9b_qwen_image_2.1_pe_i2i.int8_convrot.safetensors", "text_encoders",
     HF + "Comfy-Org/Qwen-Image-2.1/resolve/main/text_encoders/qwen3.5_9b_qwen_image_2.1_pe_i2i.int8_convrot.safetensors"),
    ("Qwen-Image-2.1-viggle-turbo-v0.2-5step-lora-r256.safetensors", "loras",
     HF + "Viggle/Qwen-Image-2.1-viggle-turbo/resolve/main/Qwen-Image-2.1-viggle-turbo-v0.2-5step-lora-r256.safetensors"),
    ("texture_fix_vae_for_qwen_image_2.1_bf16.safetensors", "vae",
     HF + "madebyollin/texture-fix-vae-for-qwen-image-2.1/resolve/main/texture_fix_vae_for_qwen_image_2.1_bf16.safetensors"),
    ("qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors", "text_encoders",
     HF + "Comfy-Org/MiniMax-H3/resolve/main/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"),
    ("minimax_h3_video_vae_int8_convrot.safetensors", "vae",
     HF + "Comfy-Org/MiniMax-H3/resolve/main/vae/minimax_h3_video_vae_int8_convrot.safetensors"),
    ("minimax_h3_audio_vae_fp32.safetensors", "vae",
     HF + "Comfy-Org/MiniMax-H3/resolve/main/vae/minimax_h3_audio_vae_fp32.safetensors"),
]
MODEL_ENDINGS = (".safetensors", ".sft", ".gguf", ".pth", ".ckpt")


def table() -> dict:
    spec = importlib.util.spec_from_file_location("setup_kubakub", os.path.join(HERE, "setup_kubakub.py"))
    setup = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(setup)
    files = [f for s in setup.MODEL_SETS for f in s["files"]] + EXTRA
    return {name: {"name": name, "url": url, "directory": folder} for name, folder, url in files}


def all_nodes(workflow: dict):
    yield from workflow.get("nodes", [])
    for sub in workflow.get("definitions", {}).get("subgraphs", []):
        yield from sub.get("nodes", [])


def link(workflow: dict, known: dict) -> tuple[int, list]:
    """Sets properties.models on the loaders; returns (changed nodes, file names without a link)."""
    changed, unknown = 0, []
    for node in all_nodes(workflow):
        values = node.get("widgets_values")
        if "Load" not in str(node.get("type", "")) or not isinstance(values, list):
            continue
        names = [os.path.basename(v.replace("\\", "/")) for v in values
                 if isinstance(v, str) and v.lower().endswith(MODEL_ENDINGS)]
        models = [known[n] for n in names if n in known]
        unknown += [n for n in names if n not in known]
        props = node.setdefault("properties", {})
        if models and props.get("models") != models:
            props["models"] = models
            changed += 1
    return changed, unknown


def main() -> int:
    check = "--check" in sys.argv
    known, would_change = table(), 0
    for path in sorted(glob.glob(os.path.join(PACK, "example_workflows", "*.json"))):
        with open(path, encoding="utf-8") as f:
            text = f.read()
        workflow = json.loads(text)
        changed, unknown = link(workflow, known)
        would_change += changed
        if changed and not check:
            with open(path, "w", encoding="utf-8", newline="\n") as f:
                json.dump(workflow, f, indent=1, ensure_ascii=text.isascii())    # the file's own layout: small diffs
        if changed or unknown:
            note = f"   no link for: {', '.join(sorted(set(unknown)))}" if unknown else ""
            print(f"{os.path.basename(path)}: {changed} loader(s) {'would change' if check else 'linked'}{note}")
    return 1 if check and would_change else 0


if __name__ == "__main__":
    sys.exit(main())
