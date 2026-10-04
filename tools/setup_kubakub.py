"""
setup_kubakub.py

Guided setup of the kubakub nodes for things a plain `pip install -r requirements.txt` cannot do: it checks the
install, installs missing packages only after showing a pip dry run, finds Blender, and downloads model sets for
the example workflows into ComfyUI/models. Every step asks first; nothing is upgraded or reinstalled.

Optional, run by hand (see docs/MODELS.md). Not named install.py on purpose: ComfyUI Manager runs an
install.py by itself, and this script must never download gigabytes unless someone asks.

    python setup_kubakub.py            the guided setup
    python setup_kubakub.py --check    only report what is there and what is missing (no changes, no network)

Standard library only (yaml and torch are used when present).
"""

from __future__ import annotations

import argparse
import getpass
import importlib.util
import os
import re
import shutil
import subprocess
import sys
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
PACK = os.path.dirname(HERE)
COMFY = os.path.dirname(os.path.dirname(PACK))          # ComfyUI/custom_nodes/<pack> -> ComfyUI
MODELS = os.path.join(COMFY, "models")
BLENDER_LTS = "https://www.blender.org/download/lts/4-5/"
KJNODES = "https://github.com/kijai/ComfyUI-KJNodes"

# Model sets for the example workflows. URLs are the ones ComfyUI's own workflow templates use.
# (file name, folder under ComfyUI/models, url)
HF = "https://huggingface.co/"
MODEL_SETS = [
    {"key": "starter", "title": "Klein 4B (the starter workflows)",
     "used_by": "02 one prompt per region, 03 director, 05 sketch to render, clay_to_final",
     "files": [
         ("flux-2-klein-4b.safetensors", "diffusion_models",
          HF + "Comfy-Org/flux2-klein/resolve/main/split_files/diffusion_models/flux-2-klein-4b.safetensors"),
         ("qwen_3_4b.safetensors", "text_encoders",
          HF + "Comfy-Org/z_image_turbo/resolve/main/split_files/text_encoders/qwen_3_4b.safetensors"),
         ("flux2-vae.safetensors", "vae",
          HF + "Comfy-Org/flux2-dev/resolve/main/split_files/vae/flux2-vae.safetensors"),
     ]},
    {"key": "klein9b", "title": "Klein 9B fp8 (more detail, 16 GB VRAM is comfortable)",
     "used_by": "facade_regions_diffusion, frame_in_frame",
     "files": [
         ("flux-2-klein-9b-fp8.safetensors", "diffusion_models",
          HF + "black-forest-labs/FLUX.2-klein-9b-fp8/resolve/main/flux-2-klein-9b-fp8.safetensors"),
         ("qwen_3_8b_fp8mixed.safetensors", "text_encoders",
          HF + "Comfy-Org/flux2-klein-9B/resolve/main/split_files/text_encoders/qwen_3_8b_fp8mixed.safetensors"),
         ("flux2-vae.safetensors", "vae",
          HF + "Comfy-Org/flux2-dev/resolve/main/split_files/vae/flux2-vae.safetensors"),
     ],
     "note": "black-forest-labs repos can ask you to accept the model licence on huggingface.co first."},
    {"key": "qwen21", "title": "Qwen Image 2.1 (photographic windows, frame in frame)",
     "used_by": "facade_regions_diffusion",
     "files": [
         ("qwen_image_2.1_int8_convrot.safetensors", "diffusion_models",
          HF + "Comfy-Org/Qwen-Image-2.1/resolve/main/diffusion_models/qwen_image_2.1_int8_convrot.safetensors"),
         ("qwen3vl_8b_int8_convrot.safetensors", "text_encoders",
          HF + "Comfy-Org/Qwen-Image-2.1/resolve/main/text_encoders/qwen3vl_8b_int8_convrot.safetensors"),
         ("qwen_image_2.1_vae_bf16.safetensors", "vae",
          HF + "Comfy-Org/Qwen-Image-2.1/resolve/main/vae/qwen_image_2.1_vae_bf16.safetensors"),
     ],
     "note": "The workflows also use the Viggle turbo LoRA and the texture-fix VAE (see docs/MODELS.md); pick these "
             "files in the loaders where a workflow names another variant."},
    {"key": "ltx", "title": "LTX 2.5 (animate regions, director video)",
     "used_by": "04 animate the windows, director_video_ltx",
     "files": [
         ("ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors", "diffusion_models",
          HF + "Lightricks/LTX-2.5/resolve/main/diffusion_models/"
               "ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors"),
         ("gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors", "text_encoders",
          HF + "Lightricks/LTX-2.5/resolve/main/text_encoders/gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors"),
         ("ltx-2.5-video-vae-bf16.safetensors", "vae",
          HF + "Lightricks/LTX-2.5/resolve/main/vae/ltx-2.5-video-vae-bf16.safetensors"),
         ("ltx-2.5-audio-vae-bf16.safetensors", "vae",
          HF + "Lightricks/LTX-2.5/resolve/main/vae/ltx-2.5-audio-vae-bf16.safetensors"),
         ("ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors", "latent_upscale_models",
          HF + "Lightricks/LTX-2.5/resolve/main/latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors"),
     ]},
    {"key": "sam3", "title": "SAM 3.1 (regions from text or clicks)",
     "used_by": "regions_sam3_masks",
     "files": [
         ("sam3.1_multiplex_fp16.safetensors", "checkpoints",
          HF + "Comfy-Org/sam3.1/resolve/main/checkpoints/sam3.1_multiplex_fp16.safetensors"),
     ],
     "note": f"That workflow also uses the Points Editor from KJNodes ({KJNODES})."},
]
H3_NOTE = ("MiniMax H3 (director_h3_clips): open ComfyUI's own template 'video_minimax_h3_i2v' "
           "(Workflow > Browse templates); ComfyUI offers its models itself.")


# ---------------------------------------------------------------- helpers

def ask(question: str, default: bool = False) -> bool:
    hint = "Y/n" if default else "y/N"
    try:
        a = input(f"{question} [{hint}] ").strip().lower()
    except EOFError:
        return default
    return default if not a else a in ("y", "yes", "j", "ja", "t", "tak")


def load_file_module(name: str, path: str):
    """A pack file as a module without importing the pack (its __init__ needs ComfyUI)."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def has_module(mod: str) -> bool:
    try:
        return importlib.util.find_spec(mod) is not None
    except (ImportError, ValueError):
        return False


def comfyui_version() -> str:
    try:
        with open(os.path.join(COMFY, "comfyui_version.py"), encoding="utf-8") as f:
            m = re.search(r"""__version__\s*=\s*["']([^"']+)""", f.read())
        return m.group(1) if m else "?"
    except OSError:
        return "not found (is the pack inside ComfyUI/custom_nodes?)"


def gpu_info() -> tuple[str, tuple | None]:
    """-> (a readable line, compute capability or None). Imports torch, a few seconds."""
    try:
        import torch
    except Exception as e:  # noqa: BLE001
        return f"torch not importable ({e})", None
    line = f"torch {torch.__version__}, CUDA {torch.version.cuda}"
    if not torch.cuda.is_available():
        return line + ", no CUDA GPU visible", None
    cap = torch.cuda.get_device_capability(0)
    vram = torch.cuda.get_device_properties(0).total_memory / 2**30
    return line + f", {torch.cuda.get_device_name(0)} {vram:.0f} GB (sm_{cap[0]}{cap[1]})", cap


def model_roots(comfy: str = COMFY) -> list[str]:
    """ComfyUI/models plus every folder named in extra_model_paths.yaml (when yaml is installed)."""
    roots = [os.path.join(comfy, "models")]
    cfg = os.path.join(comfy, "extra_model_paths.yaml")
    if not os.path.isfile(cfg):
        return roots
    try:
        import yaml
        with open(cfg, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except Exception:  # noqa: BLE001  (no yaml, or a broken file: the main models folder still counts)
        return roots
    for section in data.values():
        if not isinstance(section, dict):
            continue
        base = os.path.expandvars(os.path.expanduser(str(section.get("base_path", "") or "")))
        if base and not os.path.isabs(base):
            base = os.path.join(os.path.dirname(cfg), base)
        for key, val in section.items():
            if key in ("base_path", "is_default") or not isinstance(val, str):
                continue
            for line in val.splitlines():
                line = line.strip()
                if line:
                    roots.append(os.path.join(base, line) if base else line)
    return roots


def model_index(roots) -> set[str]:
    """Lower-case file names of every model file below the roots."""
    names = set()
    for r in roots:
        if not os.path.isdir(r):
            continue
        for _dirpath, _dirs, files in os.walk(r):
            names.update(f.lower() for f in files)
    return names


def set_status(entry, have: set[str]) -> list[str]:
    """-> the file names of a set that are missing."""
    return [name for name, _d, _u in entry["files"] if name.lower() not in have]


# ---------------------------------------------------------------- pip

def pip_dry_run(pkgs) -> tuple[bool, str]:
    r = subprocess.run([sys.executable, "-m", "pip", "install", "--dry-run", *pkgs],
                       capture_output=True, text=True)
    out = r.stdout + r.stderr
    would = [ln for ln in out.splitlines() if ln.startswith("Would install")]
    return r.returncode == 0, (would[0] if would else out.strip()[-800:])


def pip_install(pkgs) -> bool:
    print(f"  pip install {' '.join(pkgs)}")
    return subprocess.run([sys.executable, "-m", "pip", "install", *pkgs]).returncode == 0


def offer_packages(pkgs, why: str, check_only: bool) -> None:
    if not pkgs:
        return
    print(f"\n  {why}: {', '.join(pkgs)}")
    if check_only:
        return
    ok, what = pip_dry_run(pkgs)
    print(f"  pip dry run: {what}")
    if not ok:
        print("  pip could not resolve this; nothing was changed.")
        return
    if ask("  Install?"):
        print("  done" if pip_install(pkgs) else "  pip reported an error (see above)")


# ---------------------------------------------------------------- downloads

def _request(url: str, token: str, method: str = "GET", start: int = 0):
    req = urllib.request.Request(url, method=method, headers={"User-Agent": "kubakub-setup"})
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    if start:
        req.add_header("Range", f"bytes={start}-")
    return urllib.request.urlopen(req, timeout=60)


def remote_size(url: str, token: str) -> int | None:
    with _request(url, token, "HEAD") as r:
        n = r.headers.get("Content-Length")
        return int(n) if n else None


def download(url: str, dest: str, token: str) -> None:
    """Resumable: bytes go to <dest>.part, renamed when complete."""
    part = dest + ".part"
    start = os.path.getsize(part) if os.path.exists(part) else 0
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with _request(url, token, start=start) as r:
        if start and r.status != 206:                      # server ignored the range: start over
            start = 0
        total = start + int(r.headers.get("Content-Length") or 0)
        done = start
        with open(part, "ab" if start else "wb") as f:
            while True:
                chunk = r.read(1 << 22)
                if not chunk:
                    break
                f.write(chunk)
                done += len(chunk)
                if total:
                    print(f"\r    {done / 2**30:6.2f} / {total / 2**30:.2f} GB", end="", flush=True)
    print()
    os.replace(part, dest)


def get_token(current: str) -> str:
    if current:
        return current
    print("  This file needs a Hugging Face login: accept the model's licence on its page, then paste a read token")
    print("  (huggingface.co/settings/tokens). It is used for this download only and not stored.")
    try:
        return getpass.getpass("  token (empty = skip): ").strip()
    except EOFError:
        return ""


def fetch_set(entry, have: set[str], token: str) -> str:
    missing = [(n, d, u) for n, d, u in entry["files"] if n.lower() not in have]
    sizes = []
    for name, _d, url in missing:
        try:
            sizes.append(remote_size(url, token))
        except urllib.error.HTTPError as e:
            if e.code in (401, 403):
                token = get_token(token)
                if not token:
                    print(f"  skipped {name} (needs a login)")
                    return token
                sizes.append(remote_size(url, token))
            else:
                print(f"  {name}: {e} (the link may have moved; download it by hand, see docs/MODELS.md)")
                return token
    total = sum(s or 0 for s in sizes)
    print(f"  {len(missing)} file(s), about {total / 2**30:.1f} GB, into {MODELS}")
    if not ask("  Download now?"):
        return token
    for name, folder, url in missing:
        print(f"  {name} -> models/{folder}")
        try:
            download(url, os.path.join(MODELS, folder, name), token)
            have.add(name.lower())
        except (urllib.error.URLError, OSError) as e:
            print(f"\n  failed: {e}. Run the setup again to resume.")
            return token
    return token


# ---------------------------------------------------------------- main

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="kubakub nodes setup")
    ap.add_argument("--check", action="store_true", help="only report, change nothing, no network")
    args = ap.parse_args(argv)
    check_only = args.check

    env = load_file_module("kub_env_standalone", os.path.join(PACK, "kub_env.py"))
    print("kubakub nodes setup" + (" (check only)" if check_only else ""))
    print(f"  pack     {PACK}")
    print(f"  python   {sys.executable} ({sys.version.split()[0]})")
    if "python_embeded" not in sys.executable.lower():
        print("           not the portable python_embeded: make sure this is the Python ComfyUI runs with")
    print(f"  ComfyUI  {comfyui_version()} (needs {'.'.join(map(str, env.MIN_COMFYUI))} or newer)")
    gpu, cap = gpu_info()
    print(f"  GPU      {gpu}")
    print(f"  tested   {env.TESTED}")

    # 1 packages (the lists live in kub_env.py, the startup check)
    req = [pip for mod, pip in env.REQUIRED if not has_module(mod)]
    if req:
        offer_packages(req, "REQUIRED, the pack does not load without", check_only)
    else:
        print("\n  required packages: all there")
    opt = [(pip, what) for mod, pip, what in env.OPTIONAL if not has_module(mod)]
    for pip, what in opt:
        offer_packages([pip], f"optional, only for {what}", check_only)
    if not opt:
        print("  optional packages: all there")

    # 2 ffmpeg
    if has_module("imageio_ffmpeg") or shutil.which("ffmpeg"):
        print("  ffmpeg: found")
    else:
        offer_packages(["imageio-ffmpeg"], "ffmpeg is missing (video layers, exports); this package brings one",
                       check_only)

    # 3 Blender
    bridge = load_file_module("kub_bridge_standalone", os.path.join(PACK, "kubakub", "scene3d", "bridge.py"))
    try:
        print(f"  Blender: {bridge.find_blender()}")
    except FileNotFoundError:
        print(f"\n  Blender 4.x not found (3D scene nodes, light layers, relight). Install Blender 4.5 LTS from")
        print(f"  {BLENDER_LTS}")
        print("  (a default install is found by itself), or give the path to blender.exe now.")
        if not check_only:
            try:
                path = input("  path to blender.exe (empty = later): ").strip().strip('"')
            except EOFError:
                path = ""
            if path and os.path.isfile(path):
                if ask("  Save it in kubakub.ini (blender = ...)?", True):
                    load_file_module("kub_settings_standalone", os.path.join(PACK, "settings.py")).save("blender", path)
                    print("  saved; restart ComfyUI to use it")
            elif path:
                print("  not a file, skipped")

    # 4 models
    have = model_index(model_roots())
    print("\n  model sets for the example workflows (01 needs none):")
    for i, entry in enumerate(MODEL_SETS, 1):
        missing = set_status(entry, have)
        state = "installed" if not missing else f"{len(entry['files']) - len(missing)}/{len(entry['files'])} files"
        print(f"   {i}. {entry['title']:58s} {state}")
        print(f"      for: {entry['used_by']}")
    print(f"   -  {H3_NOTE}")
    if cap is not None and cap < (10, 0):
        print("  This GPU is older than RTX 50: NVFP4 model files give no speed-up here, use the fp8 / int8 ones.")
    token = os.environ.get("HF_TOKEN", "")
    if not check_only:
        try:
            pick = input("  sets to download (e.g. 1 3, empty = none): ").split()
        except EOFError:
            pick = []
        for p in pick:
            if not p.isdigit() or not 1 <= int(p) <= len(MODEL_SETS):
                print(f"  '{p}' is not a set number")
                continue
            entry = MODEL_SETS[int(p) - 1]
            print(f"\n  {entry['title']}")
            if entry.get("note"):
                print(f"  note: {entry['note']}")
            if not set_status(entry, have):
                print("  already installed")
                continue
            token = fetch_set(entry, have, token)

    kj = os.path.isdir(os.path.join(COMFY, "custom_nodes", "ComfyUI-KJNodes")) or \
        os.path.isdir(os.path.join(COMFY, "custom_nodes", "comfyui-kjnodes"))
    if not kj:
        print(f"\n  KJNodes not found: only regions_sam3_masks needs it (Points Editor). ComfyUI Manager or {KJNODES}")
    print("\nDone. Restart ComfyUI; the console shows '[Kub] N nodes from M modules' and anything still missing.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
