# kubakub facade show (app)

A projection mapping look for a whole building from one picture. Every window, door and cornice of the facade is a
region with its own prompt; Flux 2 Klein repaints the facade region by region, then LTX 2.5 animates only the regions
you mark. Everything else stays pixel exact, so the video still lines up with the real building when it is
projected onto it.

It is one workflow, `example_workflows/09_facade_show.app.json`, saved for ComfyUI's **App Mode**: it opens as a
simple app with six fields and two results (the video and the still). The whole graph is behind it (App Mode off) and
every node of this pack can be added to it.

## Run it on your machine

1. **ComfyUI** 0.37 or newer (portable, desktop or a git install) and an NVIDIA GPU with 12 GB or more.
2. **This pack:** ComfyUI Manager → search *kubakub nodes* → install, or
   ```bash
   cd ComfyUI/custom_nodes
   git clone https://github.com/kubakubkub/ComfyUI_KubaNodes
   python -m pip install -r ComfyUI_KubaNodes/requirements.txt    # portable: python_embeded\python.exe -m pip ...
   ```
3. **Models** (sets 1 and 4 of [MODELS.md](MODELS.md)), about 30 GB:
   ```bash
   python ComfyUI/custom_nodes/ComfyUI_KubaNodes/tools/setup_kubakub.py     # asks before each download
   ```
   | file | folder |
   |---|---|
   | flux-2-klein-4b.safetensors | models/diffusion_models |
   | qwen_3_4b.safetensors | models/text_encoders |
   | flux2-vae.safetensors | models/vae |
   | ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors | models/diffusion_models |
   | gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors | models/text_encoders |
   | ltx-2.5-video-vae-conv-bf16.safetensors, ltx-2.5-audio-vae-bf16.safetensors | models/vae |
   | ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors | models/latent_upscale_models |

   Other variants (fp8, NVFP4, GGUF, Flux 3 Klein) work too: pick your file in the loader (App Mode off).
4. Restart ComfyUI, then **Workflow → Browse templates → ComfyUI_KubaNodes → 09 facade show** (or drag the
   .app.json onto the window) and press **Run**. With empty fields it runs on the built-in sample facade.

## The fields

| field | what it does |
|---|---|
| **facade_file** | your facade: the projection matrix, a photo or a render (PNG / JPG / TIF). Empty = the sample. |
| **masks_folder** | the folder of its masks: one PNG per region at the facade's size, as After Effects, Houdini, Blender or Illustrator export them. Subfolders become groups. Empty = the sample's. |
| **rules** | a prompt per region (see below) |
| **seed_offset** | another variation of the still |
| **seed** | another take of the motion |
| **frames** | video length (121 = about 5 s at 25 fps) |

## Rules

Regions are named by their mask files. A rule names them with wildcards and says what they become:

```ini
[default]
prompt = the building at night as smooth white plaster, deep carved shadows
denoise = 0.6                  ; how much may change: 0 = nothing, 1 = everything

[W_F*]                         ; every mask whose name starts with W_F (the windows)
prompt = rooms full of glowing violet and orange light behind the glass
denoise = 0.85
animate = on                   ; this region moves in the video
video_prompt = violet and orange light slowly flows through the rooms
t_start = 1                    ; seconds: when it starts / ends moving
motion = 0.7                   ; 0 calm ... 1 strong
```

`[group:Windows]` takes a whole subfolder; a region rule beats a group rule beats `[default]`. All keys:
[docs/generate.md](generate.md).

## Timing (RTX 5070 Ti laptop, 12 GB)

The sample (46 regions, 1920 x 1080): the still in a few minutes, the video (two moving region groups, 121 frames)
in about ten more. Queue again after changing one prompt: only that region and the clips it touches run again.

## Next steps in the same pack

The app is the short way. The pack has the long way too: regions from a 3D model, ID renders, Illustrator or a
cryptomatte; the **kubakub director** window (layers, lights, shaders, keyframes, sound, LTX or H3 clips on a
timeline); the **kubakub timeline** for the whole show; export for the projector. See the [README](../README.md).
