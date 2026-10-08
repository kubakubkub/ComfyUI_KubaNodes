"""
Model free test for the older Kub nodes fixed after the 2026-09-25 review: the batch image loader
(list outputs + image_batch / mask_batch, rerun on new files) and the ONNX style transfer tiling
(no black border, float tiles without an 8 bit round trip), with a fake identity ONNX session.

Run from the ComfyUI_windows_portable folder:
    python_embeded\\python.exe ComfyUI\\custom_nodes\\ComfyUI_KubaNodes\\tests\\test_legacy_nodes.py
"""

import contextlib
import io
import os
import shutil
import sys
import tempfile

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))     # embedded Python leaves the script folder off the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(HERE))))     # ComfyUI (folder_paths)
import folder_paths  # noqa: E402
import _pack  # noqa: E402,F401  (the pack as a package, see tests/_pack.py)
from kubapack.nodes.utils import batch_image_loader_Kub as bil  # noqa: E402
from kubapack.nodes.lab import onnx_style_transfer_Kub as ost  # noqa: E402

failures = []


def check(name, cond, detail=""):
    print(("ok   " if cond else "FAIL ") + name + (f"  ({detail})" if detail and not cond else ""))
    if not cond:
        failures.append(name)


tmp = tempfile.mkdtemp(prefix="kub_legacy_test_")
try:
    rng = np.random.default_rng(0)
    for i, (w, h) in enumerate(((64, 48), (80, 40), (64, 48))):
        a = (rng.random((h, w, 4)) * 255).astype(np.uint8)
        a[..., 3] = 255 if i != 1 else 128
        Image.fromarray(a, "RGBA").save(os.path.join(tmp, f"img_{i}.png"))
    node = bil.LoadImagesFromDirWithNames()
    out = node.load_images(tmp, image_load_cap=0, include_subfolders=False)
    ims, mks, names, _, _, total, start, batch, mbatch, loaded = out
    check("loaded_count is the last output: the images of this run", len(out) == 10 and loaded == 3 and total == 3
          and len(node.RETURN_TYPES) == len(node.RETURN_NAMES) == len(node.OUTPUT_IS_LIST) == len(node.OUTPUT_TOOLTIPS) == 10
          and node.RETURN_NAMES[:9] == ("IMAGE", "MASK", "filename", "subfolder", "source_path", "total_count",
                                        "current_index", "image_batch", "mask_batch"))
    check("list outputs stay lists (one item per image)", isinstance(ims, list) and len(ims) == 3 and names == ["img_0", "img_1", "img_2"])
    check("image_batch is one batch at the first image's size", tuple(batch.shape) == (3, 48, 64, 3) and tuple(mbatch.shape) == (3, 48, 64))
    check("resized mask keeps its value", abs(float(mbatch[1].mean()) - (1 - 128 / 255)) < 0.02)
    _, _, _, _, _, _, _, pb, pm, _ = node.load_images(tmp, image_load_cap=0, include_subfolders=False, batch_fit="pad_to_first")
    check("pad_to_first crops / pads into the first size", tuple(pb.shape) == (3, 48, 64, 3) and float(pm[1][:, :].max()) <= 1.0)
    try:
        node.load_images(tmp, image_load_cap=0, include_subfolders=False, batch_fit="error")
        check("batch_fit=error refuses different sizes", False)
    except ValueError:
        check("batch_fit=error refuses different sizes", True)
    kw = dict(directory=tmp, include_subfolders=False, load_always=False)
    h1 = bil.LoadImagesFromDirWithNames.IS_CHANGED(**kw)
    check("IS_CHANGED is stable while the folder is unchanged", h1 == bil.LoadImagesFromDirWithNames.IS_CHANGED(**kw))
    Image.fromarray(np.zeros((8, 8, 3), np.uint8)).save(os.path.join(tmp, "img_new.png"))
    check("IS_CHANGED changes when an image is added", h1 != bil.LoadImagesFromDirWithNames.IS_CHANGED(**kw))
    check("load_always still forces a run", bil.LoadImagesFromDirWithNames.IS_CHANGED(**dict(kw, load_always=True)) != h1)

    # skip_existing_in: each written .glb must change IS_CHANGED, so the next Queue loads the next image
    out_dir = os.path.join(tmp, "glb_out")
    os.makedirs(out_dir)
    kw = dict(directory=tmp, include_subfolders=False, load_always=False, skip_existing_in=out_dir,
              image_load_cap=1)
    s1 = bil.LoadImagesFromDirWithNames.IS_CHANGED(**kw)
    first = node.load_images(tmp, image_load_cap=1, include_subfolders=False, skip_existing_in=out_dir)[2]
    check("skip loop: stable while the output folder is unchanged", s1 == bil.LoadImagesFromDirWithNames.IS_CHANGED(**kw))
    with open(os.path.join(out_dir, f"{first[0]}.glb"), "wb") as f:
        f.write(b"glb")
    s2 = bil.LoadImagesFromDirWithNames.IS_CHANGED(**kw)
    second = node.load_images(tmp, image_load_cap=1, include_subfolders=False, skip_existing_in=out_dir)[2]
    check("skip loop: a new .glb changes IS_CHANGED", s2 != s1)
    check("skip loop: the next run loads the next image", first != second, f"{first} {second}")
    with open(os.path.join(out_dir, "notes.txt"), "w") as f:
        f.write("x")
    check("skip loop: other files in the output folder do not matter", s2 == bil.LoadImagesFromDirWithNames.IS_CHANGED(**kw))
    with open(os.path.join(out_dir, f"{first[0]}.glb"), "wb") as f:
        f.write(b"a longer glb")
    check("skip loop: a rewritten .glb changes IS_CHANGED", s2 != bil.LoadImagesFromDirWithNames.IS_CHANGED(**kw))
    check("skip loop: load_always still forces a run",
          bil.LoadImagesFromDirWithNames.IS_CHANGED(**dict(kw, load_always=True)) != bil.LoadImagesFromDirWithNames.IS_CHANGED(**dict(kw, load_always=True)))

    # natural order of numbered names, and the built-in sample for an empty folder field
    seq_dir = os.path.join(tmp, "seq")
    os.makedirs(os.path.join(seq_dir, "shot_10"))
    os.makedirs(os.path.join(seq_dir, "shot_9"))
    for n in (1, 2, 10, 11, 100):
        Image.fromarray(np.full((8, 8, 3), n, np.uint8)).save(os.path.join(seq_dir, f"frame_{n}.png"))
    Image.fromarray(np.zeros((8, 8, 3), np.uint8)).save(os.path.join(seq_dir, "shot_10", "a.png"))
    Image.fromarray(np.zeros((8, 8, 3), np.uint8)).save(os.path.join(seq_dir, "shot_9", "a.png"))
    got = node.load_images(seq_dir, image_load_cap=0, include_subfolders=False)
    check("numbered names load in number order (frame_2 before frame_10)",
          got[2] == ["frame_1", "frame_2", "frame_10", "frame_11", "frame_100"], str(got[2]))
    check("... and the batch has that order", [int(round(float(f[0, 0, 0]) * 255)) for f in got[7]] == [1, 2, 10, 11, 100])
    got = node.load_images(seq_dir, image_load_cap=0, include_subfolders=True, name_filter="a.png")[3]
    check("subfolders too (shot_9 before shot_10)", got == ["shot_9", "shot_10"], str(got))
    got = node.load_images(seq_dir, image_load_cap=2, start_index=3, include_subfolders=False)
    check("loaded_count follows image_load_cap, total_count the folder", got[9] == 2 and got[5] == 5 and got[2] == ["frame_11", "frame_100"])

    folder_paths.set_temp_directory(os.path.join(tmp, "temp"))
    out_txt = io.StringIO()
    with contextlib.redirect_stdout(out_txt):
        got = node.load_images("", image_load_cap=0)
    check("empty directory: the 12 built-in sample frames, in order", got[9] == 12 and got[5] == 12
          and got[2] == [f"sample_{i}" for i in range(1, 13)] and tuple(got[7].shape) == (12, 360, 640, 3), str(got[2]))
    check("... and the console says the sample is used", "built-in sample" in out_txt.getvalue() and "'directory'" in out_txt.getvalue(),
          out_txt.getvalue())
    check("... the frames differ (a light passes over the facade)", float((got[7][0] - got[7][11]).abs().mean()) > 0.02)
    check("IS_CHANGED works with an empty directory", bil.LoadImagesFromDirWithNames.IS_CHANGED(directory="", load_always=False)
          == bil.LoadImagesFromDirWithNames.IS_CHANGED(directory="", load_always=False))
    try:
        node.load_images(os.path.join(tmp, "not_there"))
        check("a folder that does not exist is still an error", False)
    except FileNotFoundError:
        check("a folder that does not exist is still an error", True)
finally:
    shutil.rmtree(tmp, ignore_errors=True)


class FakeSession:                      # identity "model": out = in, like a pix2pix that changed nothing
    class _In:
        name = "x"

    def get_inputs(self):
        return [self._In()]

    def run(self, _, feed):
        return [feed["x"]]


st = ost.OnnxStyleTransfer()
img = np.random.default_rng(1).random((300, 420, 3)).astype(np.float32)
out = st._infer_tiled(FakeSession(), img, tile_size=128, overlap=32)
check("tiled identity model reproduces the image", np.abs(out - img).max() < 1e-4, f"{np.abs(out - img).max()}")
check("no black border (edge rows / columns keep their values)", np.abs(out[0] - img[0]).max() < 1e-4 and np.abs(out[:, -1] - img[:, -1]).max() < 1e-4)
one = st._infer(FakeSession(), img[:128, :128], 128)
check("a tile at tile size is not round-tripped through 8 bit", np.abs(one - img[:128, :128]).max() < 1e-6)
try:
    ost.OnnxStyleTransfer._resolve("")
    check("empty model path gives a clear error", False)
except ValueError as e:
    check("empty model path gives a clear error", True)
    check("... that says where to put a model", "models/onnx" in str(e) and "model_path" in str(e), str(e))
try:
    ost.OnnxStyleTransfer._resolve("no_such_model.onnx")
    check("a missing model file says where to put it", False)
except FileNotFoundError as e:
    check("a missing model file says where to put it", "models/onnx" in str(e) and "no_such_model.onnx" in str(e), str(e))

# --- tooltips of the older nodes
from kubapack.nodes.lab import mosaic_illusion as mi  # noqa: E402
from kubapack.nodes.utils import timecode_filename_Kub as tcf  # noqa: E402
t = tcf.TimecodeFilenamePrefix
check("timecode filename prefix: a tooltip on every input and output",
      all(v[1].get("tooltip") for v in t.INPUT_TYPES()["required"].values()) and len(t.OUTPUT_TOOLTIPS) == len(t.RETURN_TYPES) == 3
      and t().build("clip", 12.0, 25.0, "_") == ("clip_00-12-00", "00-12-00", 300))
check("mosaic illusion: detail is marked adaptive only",
      mi.MosaicIllusion.INPUT_TYPES()["optional"]["detail"][1]["tooltip"].startswith("adaptive only"))

print()
if failures:
    print(f"{len(failures)} FAILED: {', '.join(failures)}")
    sys.exit(1)
print("all legacy node tests passed")
