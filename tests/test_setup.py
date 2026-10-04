"""
Model free test of the guided setup (tools/setup_kubakub.py): the model catalogue, the model search through
extra_model_paths.yaml and a full --check run. No network, nothing is installed.

Run from the ComfyUI_windows_portable folder:
    python_embeded\\python.exe ComfyUI\\custom_nodes\\ComfyUI_KubaNodes\\tests\\test_setup.py
"""

import contextlib
import io
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "tools"))
import setup_kubakub as st  # noqa: E402

failures = []


def check(name, cond, detail=""):
    print(("ok   " if cond else "FAIL ") + name + (f"  ({detail})" if detail else ""))
    if not cond:
        failures.append(name)


# catalogue: every url is a huggingface file url ending in the file name, into a known models folder
FOLDERS = {"diffusion_models", "text_encoders", "vae", "latent_upscale_models", "checkpoints", "loras"}
keys = [e["key"] for e in st.MODEL_SETS]
check("set keys unique", len(keys) == len(set(keys)), keys)
for e in st.MODEL_SETS:
    for name, folder, url in e["files"]:
        check(f"{e['key']}: {name} url", url.startswith("https://huggingface.co/") and "/resolve/main/" in url
              and url.rsplit("/", 1)[1] == name, url)
        check(f"{e['key']}: {name} folder", folder in FOLDERS, folder)
    check(f"{e['key']}: says what uses it", bool(e.get("used_by")))

# model search: ComfyUI/models plus extra_model_paths.yaml folders (relative to the yaml, several per key)
with tempfile.TemporaryDirectory() as tmp:
    comfy = os.path.join(tmp, "ComfyUI")
    os.makedirs(os.path.join(comfy, "models", "vae"))
    open(os.path.join(comfy, "models", "vae", "flux2-vae.safetensors"), "wb").close()
    other = os.path.join(tmp, "other")
    os.makedirs(os.path.join(other, "unet", "LTX"))
    os.makedirs(os.path.join(other, "clip"))
    open(os.path.join(other, "unet", "LTX", "Flux-2-Klein-4B.safetensors"), "wb").close()
    open(os.path.join(other, "clip", "qwen_3_4b.safetensors"), "wb").close()
    with open(os.path.join(comfy, "extra_model_paths.yaml"), "w", encoding="utf-8") as f:
        f.write("other:\n    base_path: ../other\n    diffusion_models: |\n        unet\n        diffusion_models\n"
                "    text_encoders: clip\n    is_default: false\n")
    roots = st.model_roots(comfy)
    have = st.model_index(roots)
    check("yaml folders found", os.path.normpath(os.path.join(comfy, "../other", "unet")) in
          [os.path.normpath(r) for r in roots], roots)
    check("files found in subfolders, any case", {"flux2-vae.safetensors", "flux-2-klein-4b.safetensors",
                                                  "qwen_3_4b.safetensors"} <= have, sorted(have))
    starter = next(e for e in st.MODEL_SETS if e["key"] == "starter")
    check("starter set complete", st.set_status(starter, have) == [])
    ltx = next(e for e in st.MODEL_SETS if e["key"] == "ltx")
    check("ltx set missing 5", len(st.set_status(ltx, have)) == 5)
    check("no yaml = only models", st.model_roots(tmp) == [os.path.join(tmp, "models")])

# the whole check run: no questions, no changes, no network
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    code = st.main(["--check"])
out = buf.getvalue()
check("--check exits 0", code == 0)
check("--check reports the sets", all(e["title"] in out for e in st.MODEL_SETS))
check("--check reports Blender", "Blender" in out)

print()
print("ALL OK" if not failures else f"{len(failures)} FAILED: {failures}")
sys.exit(1 if failures else 0)
