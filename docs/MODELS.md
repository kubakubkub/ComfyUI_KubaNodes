# Models for the example workflows

The nodes themselves need no models. The example workflows do. Every file below is a plain download: put it into
the folder in the table.

Optional helper: `python tools/setup_kubakub.py` (the portable build: `python_embeded\python.exe`) shows which sets
you already have and downloads the missing ones into `ComfyUI/models`, asking before each one and resuming broken
downloads. `--check` only reports.

| set | file | folder | used by |
|---|---|---|---|
| **1 Klein 4B** (start here) | [flux-2-klein-4b](https://huggingface.co/Comfy-Org/flux2-klein/blob/main/split_files/diffusion_models/flux-2-klein-4b.safetensors) | diffusion_models | 02, 03, 05, clay_to_final |
| | [qwen_3_4b](https://huggingface.co/Comfy-Org/z_image_turbo/blob/main/split_files/text_encoders/qwen_3_4b.safetensors) | text_encoders | |
| | [flux2-vae](https://huggingface.co/Comfy-Org/flux2-dev/blob/main/split_files/vae/flux2-vae.safetensors) | vae | also sets 2 |
| **2 Klein 9B fp8** | [flux-2-klein-9b-fp8](https://huggingface.co/black-forest-labs/FLUX.2-klein-9b-fp8) (accept the licence on the page) | diffusion_models | facade_regions_diffusion, frame_in_frame |
| | [qwen_3_8b_fp8mixed](https://huggingface.co/Comfy-Org/flux2-klein-9B/blob/main/split_files/text_encoders/qwen_3_8b_fp8mixed.safetensors) | text_encoders | |
| **3 Qwen Image 2.1** | [qwen_image_2.1_int8_convrot](https://huggingface.co/Comfy-Org/Qwen-Image-2.1/tree/main/diffusion_models) | diffusion_models | facade_regions_diffusion |
| | [qwen3vl_8b_int8_convrot](https://huggingface.co/Comfy-Org/Qwen-Image-2.1/tree/main/text_encoders) | text_encoders | |
| | [qwen_image_2.1_vae_bf16](https://huggingface.co/Comfy-Org/Qwen-Image-2.1/tree/main/vae) | vae | |
| | turbo LoRA: [Viggle/Qwen-Image-2.1-viggle-turbo](https://huggingface.co/Viggle/Qwen-Image-2.1-viggle-turbo) (by hand) | loras | schedule `qwen21_turbo_5` |
| | better decoder: [texture-fix VAE](https://huggingface.co/madebyollin/texture-fix-vae-for-qwen-image-2.1) (by hand) | vae | replaces the VAE above |
| **4 LTX 2.5** | [transformer int8 convrot, gemma4 text encoder, video + audio VAE, latent upscaler x2](https://huggingface.co/Lightricks/LTX-2.5) | diffusion_models, text_encoders, vae, latent_upscale_models | 04, director_video_ltx |
| **5 SAM 3.1** | [sam3.1_multiplex_fp16](https://huggingface.co/Comfy-Org/sam3.1/tree/main/checkpoints) | checkpoints | regions_sam3_masks (+ [KJNodes](https://github.com/kijai/ComfyUI-KJNodes) Points Editor) |
| **MiniMax H3** | open ComfyUI's template *video_minimax_h3_i2v* (Workflow > Browse templates), ComfyUI offers its models | | director_h3_clips |

**01 regions in 30 seconds** needs no model at all.

## Use the variant that suits your machine

The files above are one working choice, not a requirement. Every model comes in several variants (bf16, fp8, int8,
NVFP4, GGUF ...), and the right one depends on your GPU and VRAM: a 24-32 GB card will run bigger and more precise
files than a 12 GB laptop. The nodes work with any of them. When a workflow names a file you don't have, pick your
own variant in that loader.
