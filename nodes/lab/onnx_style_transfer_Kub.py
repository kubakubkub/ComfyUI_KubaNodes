import os
import numpy as np
import torch
from PIL import Image


class OnnxStyleTransfer:
    """Run a trained ONNX pix2pix model with tiled inference for full resolution."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "The image (or batch) to restyle; each image is done on its own."}),
                "model_path": ("STRING", {"default": "", "placeholder": "final_generator.onnx or a full path",
                                          "tooltip": "An .onnx file: a full path, or a name inside ComfyUI/models/onnx."}),
            },
            "optional": {
                "mode": (["single", "tiled"], {"default": "tiled",
                         "tooltip": "tiled runs the model on overlapping squares at full resolution. single squeezes "
                                    "the whole image into one square and scales it back (fast, softer)."}),
                "tile_size": ("INT", {"default": 512, "min": 256, "max": 1024, "step": 128,
                                      "tooltip": "Square size in pixels the model works at. Use the size it was "
                                                 "trained at (usually 512)."}),
                "overlap": ("INT", {"default": 64, "min": 0, "max": 256, "step": 16,
                                    "tooltip": "tiled only. Pixels neighbouring tiles share and blend across, "
                                               "so no seams show."}),
                "blend": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05,
                                    "tooltip": "Mix with the original: 1 = fully styled, 0 = the original image."}),
                "passes": ("INT", {"default": 1, "min": 1, "max": 5, "step": 1,
                                   "tooltip": "Run the model again on its own result this many times for a "
                                              "stronger style. 1 = once."}),
                "keep_loaded": ("BOOLEAN", {"default": False, "tooltip": "Keep the ONNX session (and its VRAM) "
                                            "between runs; off frees the GPU for the diffusion models."}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("styled_image",)
    OUTPUT_TOOLTIPS = ("The restyled image, same size as the input.",)
    DESCRIPTION = ("Restyles an image with your own trained pix2pix model (an .onnx file), at full resolution "
                   "by working in overlapping tiles. Runs on the GPU when onnxruntime has CUDA, else on the CPU.")
    FUNCTION = "run"
    CATEGORY = "kubakub/lab"

    def __init__(self):
        self.session = None
        self.loaded_model = None

    @staticmethod
    def _resolve(model_path):
        p = os.path.expandvars((model_path or "").strip().strip('"').strip("'"))
        if not p:
            raise ValueError("ONNX Style Transfer: set model_path (an .onnx file or a name in models/onnx)")
        if not os.path.isabs(p):
            import folder_paths
            cand = os.path.join(folder_paths.models_dir, "onnx", p)
            if os.path.isfile(cand):
                p = cand
        if not os.path.isfile(p):
            raise FileNotFoundError(f"ONNX model not found: {p}")
        return p

    def _load_model(self, model_path):
        try:
            import onnxruntime as ort
        except ImportError as e:
            raise ImportError("kubakub onnx style transfer needs the package onnxruntime. Install it and restart ComfyUI:\n"
                              "  python -m pip install onnxruntime\n"
                              "  (portable ComfyUI: python_embeded\\python.exe -m pip install onnxruntime)") from e
        model_path = self._resolve(model_path)
        if self.session is None or self.loaded_model != model_path:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            self.session = ort.InferenceSession(model_path, providers=providers)
            self.loaded_model = model_path
            print(f"[KubaNodes] ONNX loaded: {os.path.basename(model_path)}")
            print(f"[KubaNodes] Provider: {self.session.get_providers()[0]}")
        return self.session

    def _infer(self, session, arr_rgb_01, tile_size=512):
        """Run single inference. Input: HWC float32 0-1. Output: HWC float32 0-1."""
        h, w = arr_rgb_01.shape[:2]
        same = (h, w) == (tile_size, tile_size)      # tiles: no resize and no 8 bit round trip

        # Resize to tile_size
        if same:
            arr = arr_rgb_01.astype(np.float32)
        else:
            pil = Image.fromarray((np.clip(arr_rgb_01, 0, 1) * 255).astype(np.uint8))
            pil = pil.resize((tile_size, tile_size), Image.LANCZOS)
            arr = np.array(pil).astype(np.float32) / 255.0

        # Normalize [-1, 1], CHW, batch
        arr = (arr - 0.5) / 0.5
        arr = arr.transpose(2, 0, 1)[np.newaxis, ...]

        # Run
        input_name = session.get_inputs()[0].name
        out = session.run(None, {input_name: arr})[0]

        # Denormalize
        out = out[0].transpose(1, 2, 0)
        out = np.clip((out * 0.5 + 0.5), 0, 1).astype(np.float32)

        # Resize back
        if same:
            return out
        out_pil = Image.fromarray((out * 255).astype(np.uint8))
        out_pil = out_pil.resize((w, h), Image.LANCZOS)
        return np.array(out_pil).astype(np.float32) / 255.0

    def _infer_tiled(self, session, arr_rgb_01, tile_size=512, overlap=64):
        """Tiled inference with overlap blending. Full native quality."""
        h, w = arr_rgb_01.shape[:2]
        step = tile_size - overlap
        out = np.zeros((h, w, 3), dtype=np.float32)
        weight = np.zeros((h, w, 1), dtype=np.float32)

        # Create blend mask: 1 in center, fade at edges
        mask = np.ones((tile_size, tile_size, 1), dtype=np.float32)
        if overlap > 0:
            ramp = np.linspace(1.0 / overlap, 1, overlap)   # never 0: image border pixels keep a weight
            for i in range(overlap):
                mask[i, :, 0] *= ramp[i]
                mask[tile_size - 1 - i, :, 0] *= ramp[i]
                mask[:, i, 0] *= ramp[i]
                mask[:, tile_size - 1 - i, 0] *= ramp[i]

        tiles_done = 0
        total_tiles = ((h - overlap) // step + 1) * ((w - overlap) // step + 1)

        for y in range(0, h - overlap, step):
            for x in range(0, w - overlap, step):
                # Extract tile (handle edges)
                y2 = min(y + tile_size, h)
                x2 = min(x + tile_size, w)
                y1 = y2 - tile_size
                x1 = x2 - tile_size

                if y1 < 0: y1 = 0; y2 = min(tile_size, h)
                if x1 < 0: x1 = 0; x2 = min(tile_size, w)

                tile = arr_rgb_01[y1:y2, x1:x2]
                th, tw = tile.shape[:2]

                # Pad if needed
                if th < tile_size or tw < tile_size:
                    padded = np.zeros((tile_size, tile_size, 3), dtype=np.float32)
                    padded[:th, :tw] = tile
                    tile_result = self._infer(session, padded, tile_size)
                    tile_result = tile_result[:th, :tw]
                    tile_mask = mask[:th, :tw]
                else:
                    tile_result = self._infer(session, tile, tile_size)
                    tile_mask = mask

                out[y1:y2, x1:x2] += tile_result * tile_mask
                weight[y1:y2, x1:x2] += tile_mask

                tiles_done += 1
                if tiles_done % 4 == 0:
                    print(f"[KubaNodes] Tile {tiles_done}/{total_tiles}")

        # Normalize by weight
        weight = np.maximum(weight, 1e-6)
        out = out / weight

        print(f"[KubaNodes] Tiled inference done: {tiles_done} tiles at {tile_size}px")
        return np.clip(out, 0, 1)

    def run(self, image, model_path, mode="tiled", tile_size=512, overlap=64, blend=1.0, passes=1, keep_loaded=False):
        try:
            return self._run(image, model_path, mode, tile_size, overlap, blend, passes)
        finally:
            if not keep_loaded:
                self.session, self.loaded_model = None, None

    def _run(self, image, model_path, mode, tile_size, overlap, blend, passes):
        session = self._load_model(model_path)

        batch = image.shape[0]
        results = []

        for b in range(batch):
            img_np = image[b].cpu().numpy()  # [H, W, 3] float32 0-1
            h, w = img_np.shape[:2]

            if mode == "tiled" and (h > tile_size or w > tile_size):
                out_np = self._infer_tiled(session, img_np, tile_size, overlap)
            else:
                out_np = self._infer(session, img_np, tile_size)

            # Multi-pass refinement
            for p in range(1, passes):
                print(f"[KubaNodes] Pass {p+1}/{passes}")
                if mode == "tiled" and (h > tile_size or w > tile_size):
                    out_np = self._infer_tiled(session, out_np, tile_size, overlap)
                else:
                    out_np = self._infer(session, out_np, tile_size)

            # Blend with original
            if blend < 1.0:
                out_np = img_np * (1 - blend) + out_np * blend

            results.append(out_np)

        result = np.stack(results, axis=0)
        return (torch.from_numpy(result),)


NODE_CLASS_MAPPINGS = {
    "OnnxStyleTransfer": OnnxStyleTransfer,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "OnnxStyleTransfer": "kubakub onnx style transfer",
}
