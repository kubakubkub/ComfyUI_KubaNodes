"""
Model free round-trip test of the director's colour management (kubakub/director/colour.py, media.py):
ICC profiles on import (Adobe RGB, sRGB, CMYK - the Windows profiles), untagged HD video read as Rec.709,
10-bit video kept at 16 bits, proxies tagged Rec.709. Prints the measured errors.

Run from the ComfyUI_windows_portable folder:
    python_embeded\\python.exe ComfyUI\\custom_nodes\\ComfyUI_KubaNodes\\tests\\test_colour.py
"""

import io
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from kubakub.director import colour, media  # noqa: E402

failures = []


def check(name, cond, detail=""):
    print(("ok   " if cond else "FAIL ") + name + (f"  ({detail})" if detail else ""))
    if not cond:
        failures.append(name)


from PIL import Image, ImageCms  # noqa: E402

tmp = tempfile.mkdtemp(prefix="kuba_colour_")
PROF = r"C:\Windows\System32\spool\drivers\color"
adobe, cmyk = os.path.join(PROF, "AdobeRGB1998.icc"), os.path.join(PROF, "CoatedFOGRA39.icc")

# ---- images with ICC profiles
px = np.zeros((8, 8, 3), np.uint8); px[...] = (100, 150, 200)
if os.path.isfile(adobe):
    p = os.path.join(tmp, "adobe.png")
    Image.fromarray(px).save(p, icc_profile=open(adobe, "rb").read())
    img, note = colour.read_image(p)
    ref = np.asarray(ImageCms.profileToProfile(Image.fromarray(px), adobe, ImageCms.createProfile("sRGB"),
                                               renderingIntent=ImageCms.Intent.RELATIVE_COLORIMETRIC, outputMode="RGB"), np.float32) / 255
    moved = np.abs(img[4, 4, :3] * 255 - (100, 150, 200)).max()
    check("Adobe RGB PNG converted to sRGB (as LittleCMS does), not read raw", np.allclose(img[..., :3], ref, atol=1 / 255) and moved >= 3 and "Adobe RGB" in note,
          f"raw (100,150,200) -> {np.round(img[4, 4, :3] * 255).astype(int).tolist()}, {note}")
else:
    print("skip Adobe RGB test (profile not on this machine)")
p = os.path.join(tmp, "srgb.png")
Image.fromarray(px).save(p, icc_profile=ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes())
img, note = colour.read_image(p)
check("sRGB-tagged PNG untouched", np.allclose(img[4, 4, :3] * 255, (100, 150, 200), atol=0.01) and note == "", note)
p = os.path.join(tmp, "plain16.png")
import cv2  # noqa: E402
cv2.imwrite(p, np.full((4, 4, 3), 30000, np.uint16))
img, note = colour.read_image(p)
check("16-bit PNG without profile: precision kept", abs(img[0, 0, 0] - 30000 / 65535) < 1e-6)
if os.path.isfile(cmyk):
    p = os.path.join(tmp, "cmyk.jpg")
    Image.new("CMYK", (8, 8), (0, 0, 0, 0)).save(p, icc_profile=open(cmyk, "rb").read(), quality=100)
    img, note = colour.read_image(p)
    check("CMYK JPEG with FOGRA39: converted to RGB, paper white stays near white", img.shape[-1] == 4 and img[4, 4, :3].min() > 0.9 and "FOGRA39" in note,
          f"{np.round(img[4, 4, :3] * 255).astype(int).tolist()} {note}")

# ---- video: untagged HD read as Rec.709; 10-bit kept at 16 bits; proxies tagged
exe = media.ffmpeg_exe()
bars = np.zeros((720, 1280, 3), np.uint8)
cols = [(235, 235, 235), (235, 235, 16), (16, 235, 235), (16, 235, 16), (235, 16, 235), (235, 16, 16), (16, 16, 235), (128, 64, 32)]
for i, c in enumerate(cols):
    bars[:, i * 160:(i + 1) * 160] = c
un = os.path.join(tmp, "untagged709.mp4")          # encoded with the Rec.709 matrix, tags left out (common in the wild)
subprocess.run([exe, "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "1280x720", "-r", "25", "-i", "-",
                "-vf", "scale=out_color_matrix=bt709:out_range=tv", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "4", "-frames:v", "2",
                "-bsf:v", "h264_metadata=matrix_coefficients=2:colour_primaries=2:transfer_characteristics=2", un],   # tags: unspecified
               input=np.stack([bars, bars]).tobytes(), check=True)
ui = media.probe(un)
dec = media.decode(un, [0], info=ui)[0]
err709 = max(np.abs(dec[360, i * 160 + 80, :3].astype(int) - np.array(c)).max() for i, c in enumerate(cols))
old = media.decode(un, [0], info=dict(ui, matrix_tagged=True))[0]     # as before: ffmpeg's default (Rec.601)
err601 = max(np.abs(old[360, i * 160 + 80, :3].astype(int) - np.array(c)).max() for i, c in enumerate(cols))
check("untagged HD: probe sees no matrix tag", not ui["matrix_tagged"] and ui["yuv"], ui["pix_fmt"])
check("untagged HD decoded as Rec.709: bars within 3 levels", err709 <= 3, f"max error {err709} (with ffmpeg's old Rec.601 default: {err601})")
check("the Rec.601 default was really wrong for it", err601 > 10, str(err601))
grad = (np.tile(np.linspace(0, 1, 1024, dtype=np.float32), (64, 1))[..., None] * np.ones(3, np.float32) * 65535).astype(np.uint16)
pr = os.path.join(tmp, "grad10.mov")
subprocess.run([exe, "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb48le", "-s", "1024x64", "-r", "25", "-i", "-",
                "-c:v", "prores_ks", "-profile:v", "3", "-pix_fmt", "yuv422p10le", "-frames:v", "1", pr], input=grad.tobytes(), check=True)
pi = media.probe(pr)
fr = media.decode(pr, [0], info=pi)[0]
levels = len(np.unique(fr[32, :, 1]))
check("10-bit ProRes: probed 10 bit, decoded at 16 bits with more than 256 levels", pi["bits"] == 10 and fr.dtype == np.uint16 and levels > 256, f"{levels} levels")
f01 = media.mix({0: fr}, [(0, 1.0)])
check("mix reads 16-bit frames as 0..1", f01.max() <= 1.0001 and f01[32, 1000, 1] > 0.9)
px_ = media.make_proxy(un, os.path.join(tmp, "prox"), width=320)
pp = media.probe(px_)
check("proxy tagged Rec.709 (the browser never guesses)", pp["matrix_tagged"], pp["pix_fmt"])

shutil.rmtree(tmp, ignore_errors=True)
print()
if failures:
    print(f"{len(failures)} FAILED: {', '.join(failures)}")
    sys.exit(1)
print("all colour tests passed")
