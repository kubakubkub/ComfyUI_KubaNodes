import hashlib
import os
import re
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageOps

class LoadImagesFromDirWithNames:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "directory": ("STRING", {"default": "", "tooltip": "The folder to load .jpg, .png, .webp and .bmp "
                                                                   "images from (sorted by name)."}),
            },
            "optional": {
                "image_load_cap": ("INT", {"default": 1, "min": 0, "step": 1,
                                           "tooltip": "How many images to load per run. 0 = all of them; the "
                                                      "default 1 loads one image."}),
                "start_index": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "step": 1,
                                        "tooltip": "Which image to start at (0 = the first), counted after the "
                                                   "name filter and skipping."}),
                "load_always": ("BOOLEAN", {"default": False, "label_on": "enabled", "label_off": "disabled",
                                            "tooltip": "Reload on every Queue. Off reloads only when the settings "
                                                       "or the folder's files change."}),
                "include_subfolders": ("BOOLEAN", {"default": True, "label_on": "yes", "label_off": "no",
                                                   "tooltip": "Also load images from folders inside the folder."}),
                "subfolder_prefix": ("BOOLEAN", {"default": True, "label_on": "yes", "label_off": "no",
                                                 "tooltip": "Put the subfolder name in front of the filename output "
                                                            "(subfolder_name), so names from different folders "
                                                            "do not clash."}),
                "name_filter": ("STRING", {"default": "", "placeholder": "e.g. dormant or *.png",
                                           "tooltip": "Only load files whose name contains this text (any case); "
                                                      "* and ? work as wildcards. Empty = all files."}),
                "skip_existing_in": ("STRING", {"default": "", "placeholder": "output folder to check for existing .glb",
                                                "tooltip": "A folder of finished .glb files: images that already "
                                                           "have a .glb of the same name there are skipped. Empty "
                                                           "= skip nothing."}),
                "batch_fit": (["resize_to_first", "pad_to_first", "error"], {"default": "resize_to_first",
                              "tooltip": "image_batch / mask_batch: how images of other sizes join the first "
                                         "one's size. The list outputs are unchanged."}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "STRING", "STRING", "STRING", "INT", "INT", "IMAGE", "MASK")
    RETURN_NAMES = ("IMAGE", "MASK", "filename", "subfolder", "source_path", "total_count", "current_index",
                    "image_batch", "mask_batch")
    # the first outputs are lists (downstream nodes run once per image); image_batch / mask_batch are one
    # batch, for nodes that take several frames at once (walkthrough matrix, director layers ...)
    OUTPUT_IS_LIST = (True, True, True, True, True, False, False, False, False)
    OUTPUT_TOOLTIPS = ("one image per list item", "one mask per list item (1 = transparent)",
                       "each image's name without extension (with the subfolder in front when subfolder_prefix is on)",
                       "each image's subfolder inside the folder (empty for the folder itself)",
                       "each image's full file path",
                       "how many images the folder has after the filter and skipping",
                       "the index the loading started at",
                       "all loaded images as one batch", "all masks as one batch")
    DESCRIPTION = ("Loads the images of a folder, with their names, for batch jobs: each image runs through the "
                   "graph on its own (list outputs), or all together as one batch. Can filter by name and skip "
                   "images that already have a finished .glb.")
    FUNCTION = "load_images"
    CATEGORY = "kubakub/utils"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        if kwargs.get('load_always'):
            return float("NaN")
        # the settings and the folder's files (name, size, mtime): new or edited images trigger a rerun
        h = hashlib.sha256(repr(sorted((k, str(v)) for k, v in kwargs.items())).encode("utf-8", "replace"))
        directory = kwargs.get("directory", "")
        if os.path.isdir(directory):
            for path, _, _ in cls._gather_files(cls, directory, kwargs.get("include_subfolders", True)):
                try:
                    st = os.stat(path)
                    h.update(f"{path}|{st.st_size}|{st.st_mtime_ns}".encode("utf-8", "replace"))
                except OSError:
                    pass
        # skip_existing_in: a .glb written there moves the loader on to the next image, so its .glb files
        # (name, size, mtime) are part of the state; otherwise the next Queue is cached and does nothing
        skip = str(kwargs.get("skip_existing_in", "") or "").strip()
        if skip:
            h.update(b"|skip|")
            if os.path.isdir(skip):
                try:
                    names = sorted(f for f in os.listdir(skip) if f.lower().endswith(".glb"))
                except OSError:
                    names = []
                for f in names:
                    try:
                        st = os.stat(os.path.join(skip, f))
                        h.update(f"{f}|{st.st_size}|{st.st_mtime_ns}".encode("utf-8", "replace"))
                    except OSError:
                        pass
            else:
                h.update(b"missing")
        return h.hexdigest()

    def _gather_files(self, directory, include_subfolders):
        valid_extensions = ('.jpg', '.jpeg', '.png', '.webp', '.bmp')
        files = []
        if include_subfolders:
            for root, dirs, filenames in os.walk(directory):
                dirs.sort()
                for f in sorted(filenames):
                    if f.lower().endswith(valid_extensions):
                        rel = os.path.relpath(root, directory)
                        subfolder = rel if rel != '.' else ''
                        files.append((os.path.join(root, f), f, subfolder))
        else:
            for f in sorted(os.listdir(directory)):
                if f.lower().endswith(valid_extensions):
                    files.append((os.path.join(directory, f), f, ''))
        return files

    def _make_name(self, basename, subfolder, subfolder_prefix):
        name = os.path.splitext(basename)[0]
        if subfolder_prefix and subfolder:
            # Replace path separators for nested subfolders
            prefix = subfolder.replace(os.sep, '_').replace('/', '_')
            name = f"{prefix}_{name}"
        return name

    def load_images(self, directory, image_load_cap=1, start_index=0,
                    load_always=False, include_subfolders=True,
                    subfolder_prefix=True, name_filter="", skip_existing_in="", batch_fit="resize_to_first"):

        if not os.path.isdir(directory):
            raise FileNotFoundError(f"Directory '{directory}' cannot be found.")

        files = self._gather_files(directory, include_subfolders)

        if not files:
            raise FileNotFoundError(f"No images found in '{directory}'")

        # Filter by name pattern
        if name_filter.strip():
            pattern = name_filter.strip()
            # Support simple wildcard: *.png or dormant*
            pattern = pattern.replace('*', '.*').replace('?', '.')
            try:
                regex = re.compile(pattern, re.IGNORECASE)
                files = [f for f in files if regex.search(f[1])]
            except re.error:
                # Fallback: simple substring match
                files = [f for f in files if pattern.lower() in f[1].lower()]

        if not files:
            raise FileNotFoundError(f"No images match filter '{name_filter}' in '{directory}'")

        # Skip already processed: check if .glb exists in output folder
        if skip_existing_in.strip() and os.path.isdir(skip_existing_in.strip()):
            out_dir = skip_existing_in.strip()
            filtered = []
            for filepath, basename, subfolder in files:
                name = self._make_name(basename, subfolder, subfolder_prefix)
                glb_path = os.path.join(out_dir, f"{name}.glb")
                if not os.path.exists(glb_path):
                    filtered.append((filepath, basename, subfolder))
                else:
                    print(f"[KubaNodes] Skipping {name} — GLB already exists")
            files = filtered

        if not files:
            raise FileNotFoundError(f"All images already processed in '{skip_existing_in}'")

        total = len(files)
        actual_start = min(start_index, max(0, total - 1))
        files_to_load = files[actual_start:]

        images = []
        masks = []
        filenames = []
        subfolders = []
        source_paths = []
        limit = image_load_cap > 0
        count = 0

        for filepath, basename, subfolder in files_to_load:
            if limit and count >= image_load_cap:
                break

            i = Image.open(filepath)
            i = ImageOps.exif_transpose(i)
            image = i.convert("RGB")
            image = np.array(image).astype(np.float32) / 255.0
            image = torch.from_numpy(image)[None,]

            if 'A' in i.getbands():
                mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
                mask = 1. - torch.from_numpy(mask)
            else:
                mask = torch.zeros((i.size[1], i.size[0]), dtype=torch.float32)

            name = self._make_name(basename, subfolder, subfolder_prefix)

            images.append(image)
            masks.append(mask)
            filenames.append(name)
            subfolders.append(subfolder)
            source_paths.append(filepath)
            count += 1

        if not images:
            raise FileNotFoundError(f"No images to load after filtering")

        image_batch, mask_batch = self._batch(images, masks, batch_fit)
        return (images, masks, filenames, subfolders, source_paths, total, actual_start, image_batch, mask_batch)

    @staticmethod
    def _batch(images, masks, fit):
        h, w = images[0].shape[1:3]
        ims, mks = [], []
        for im, mk in zip(images, masks):
            if im.shape[1:3] != (h, w):
                if fit == "error":
                    raise ValueError(f"image_batch: sizes differ ({im.shape[2]}x{im.shape[1]} vs {w}x{h}); "
                                     "set batch_fit to resize_to_first or pad_to_first")
                if fit == "pad_to_first":
                    ih, iw = im.shape[1:3]
                    im = F.pad(im[:, :h, :w].permute(0, 3, 1, 2), (0, max(0, w - iw), 0, max(0, h - ih))).permute(0, 2, 3, 1)
                    mk = F.pad(mk[:h, :w][None], (0, max(0, w - iw), 0, max(0, h - ih)), value=1.0)[0]
                else:
                    im = F.interpolate(im.permute(0, 3, 1, 2), size=(h, w), mode="bilinear", align_corners=False).permute(0, 2, 3, 1)
                    mk = F.interpolate(mk[None, None], size=(h, w), mode="bilinear", align_corners=False)[0, 0]
            ims.append(im)
            mks.append(mk)
        return torch.cat(ims, 0), torch.stack(mks, 0)


NODE_CLASS_MAPPINGS = {
    "LoadImagesFromDirWithNames": LoadImagesFromDirWithNames,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadImagesFromDirWithNames": "kubakub load images from dir",
}