"""
Model free test of kubakub align to source (kubakub/align.py): a restyled copy of the sample facade that was moved
and scaled by a known amount is put back to within a pixel; a picture that already sits right is left alone; a
picture with nothing in common, or one moved far more than a drift, is refused and returned unchanged.

Run from the ComfyUI_windows_portable folder:
    python_embeded\\python.exe ComfyUI\\custom_nodes\\ComfyUI_KubaNodes\\tests\\test_align.py
"""

import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from kubakub import align as al  # noqa: E402
from kubakub import sample_facade as sf  # noqa: E402

failures = []


def check(name, cond, detail=""):
    print(("ok   " if cond else "FAIL ") + name + (f"  ({detail})" if detail and not cond else ""))
    if not cond:
        failures.append(name)


W, H = 1280, 720
src = sf.render(W, H, style="sandstone")["image"]


def restyle(img, seed=3):
    """What an edit does: other colours, new texture and large painted forms; the building's edges mostly stay."""
    rng = np.random.default_rng(seed)
    g = img.mean(2, keepdims=True)
    out = np.clip(g * np.array([0.75, 0.45, 0.95]) + 0.12, 0, 1).astype(np.float32)        # violet
    blobs = cv2.GaussianBlur(rng.random((H // 16, W // 16)).astype(np.float32), (0, 0), 2)
    blobs = cv2.resize(blobs, (W, H), interpolation=cv2.INTER_CUBIC)
    form = (blobs > np.percentile(blobs, 72))[..., None]
    out = np.where(form, np.clip(out * 0.4 + np.array([0.9, 0.45, 0.1]) * 0.6, 0, 1), out)   # orange forms over a quarter
    out += rng.normal(0, 0.03, out.shape)
    return np.clip(out, 0, 1).astype(np.float32)


def drifted(img, scale_x, scale_y, dx, dy):
    """img as an edit returns it: scaled about its centre and moved. -> (picture, the 2x3 warp source -> picture)."""
    M = np.array([[scale_x, 0, dx + (1 - scale_x) * W / 2], [0, scale_y, dy + (1 - scale_y) * H / 2]])
    big = cv2.copyMakeBorder(img, 64, 64, 64, 64, cv2.BORDER_REFLECT)            # real pixels beyond the frame
    Mb = M.copy()
    Mb[:, 2] = M[:, 2] - M[:, :2] @ np.array([64, 64]) + 64                        # the same warp in the padded frame
    out = cv2.warpAffine(big, Mb, (W + 128, H + 128), flags=cv2.INTER_CUBIC)     # what was at p is now at M p
    return np.clip(out[64:-64, 64:-64], 0, 1), M


def corner_error(found, true):
    pts = np.array([[0, 0, 1], [W, 0, 1], [W, H, 1], [0, H, 1]], float)
    return float(np.linalg.norm(pts @ (found - true).T, axis=1).max())


edit = restyle(src)
pic, true = drifted(edit, 1.012, 1.008, 6.0, -4.0)
r = al.estimate(src, pic)
check("a restyled facade, scaled 1.2 % / 0.8 % and moved 6 / -4 px, is found", r["ok"] and r["why"] == "aligned", al.describe(r))
check("... to within a pixel at the corners", corner_error(r["matrix"], true) < 1.0, f"{corner_error(r['matrix'], true):.2f} px")
check("... and the report says how far it sat off", "px off at the corners" in al.describe(r) and r["corner_px"] > 5, al.describe(r))
fixed = al.apply(pic, r["matrix"])
inner = (slice(40, H - 40), slice(60, W - 60))
before = float(np.abs(pic[inner] - edit[inner]).mean())
after = float(np.abs(fixed[inner] - edit[inner]).mean())
check("putting it back gives the undrifted picture (error falls to a third or less)", after < before / 3, f"{before:.4f} -> {after:.4f}")
check("the result has the source's size and stays in 0..1", fixed.shape == src.shape and fixed.min() >= 0 and fixed.max() <= 1)

r0 = al.estimate(src, edit)
check("a picture that already sits right is left alone", r0["ok"] and np.allclose(r0["matrix"], np.eye(2, 3)), al.describe(r0))
check("... and comes back unchanged", np.array_equal(al.apply(edit, r0["matrix"]), edit))

pic_s, true_s = drifted(edit, 1.0, 1.0, -5.0, 3.0)
rs = al.estimate(src, pic_s, model="shift")
check("move only: a plain shift of -5 / 3 px", rs["ok"] and corner_error(rs["matrix"], true_s) < 0.6,
      f"{al.describe(rs)}; {corner_error(rs['matrix'], true_s):.2f} px")

small, _ = drifted(edit, 1.01, 1.01, 4.0, 2.0)
small = cv2.resize(small, (W // 2, H // 2), interpolation=cv2.INTER_AREA)
fixed_small, rr = al.align(src, small)
check("an edit at another size is fitted at the source's size", rr["ok"] and fixed_small.shape == src.shape, al.describe(rr))

noise = np.random.default_rng(0).random(src.shape).astype(np.float32)
rn = al.estimate(src, noise)
check("nothing in common: refused, the picture stays as it is", not rn["ok"] and np.allclose(rn["matrix"], np.eye(2, 3))
      and np.array_equal(al.apply(noise, rn["matrix"]), noise), al.describe(rn))
far, _ = drifted(edit, 1.0, 1.0, 140.0, 0.0)                    # more than a window bay: not a drift
rf = al.estimate(src, far)
check("moved by 140 px: not a drift, refused", not rf["ok"] and np.allclose(rf["matrix"], np.eye(2, 3)), al.describe(rf))
check("every refusal says why", all(len(x["why"]) > 10 for x in (rn, rf)))

u8 = (pic * 255).astype(np.uint8)
ru = al.estimate((src * 255).astype(np.uint8), u8)
check("8 bit pictures give the same fit", ru["ok"] and corner_error(ru["matrix"], true) < 1.0)

# ---- a clip: it drifts while it plays, one frame is rubbish, the windows are animated
N = 10
truth, clip = [], []
rng = np.random.default_rng(11)
for k in range(N):
    frame, M = drifted(edit, 1.0 + 0.001 * k, 1.0, 0.8 * k, -0.5 * k)          # 0 .. 7 px, growing
    truth.append(M)
    clip.append(frame)
clip[4] = rng.random(clip[4].shape).astype(np.float32)                         # a frame with nothing of the building in it
fits = [al.estimate(src, f) for f in clip]
off = al.drift(fits, W, H)
check("a clip: the drift per frame is measured and grows", off[1] < off[5] < off[9] and 8.0 < off[9] < 18.0 and off[4] == 0.0
      and not fits[4]["ok"], f"{[round(v, 1) for v in off]}")
track, filled = al.steady(fits, 5)
errs = [corner_error(track[k], truth[k]) for k in range(N)]
check("steady over time: every frame is put back to within 1.5 px, the rubbish frame takes its neighbours' correction",
      filled == 1 and max(errs) < 1.5 and errs[4] < 1.5 and not np.allclose(track[4], np.eye(2, 3)), f"{[round(e, 2) for e in errs]}")
jit = [al.estimate(src, drifted(edit, 1.0, 1.0, 3.0 + 0.4 * (-1) ** k, 2.0)[0]) for k in range(8)]
raw = np.array([r["matrix"][0, 2] for r in jit])
smooth = np.array([m[0, 2] for m in al.steady(jit, 7)[0]])
check("steady over time: a correction that trembles from frame to frame is smoothed", np.abs(np.diff(smooth)).max() < 0.5 * np.abs(np.diff(raw)).max(),
      f"raw {np.abs(np.diff(raw)).max():.2f} px, smoothed {np.abs(np.diff(smooth)).max():.2f} px")
check("steady over time: no frame fitted = nothing moved", all(np.allclose(m, np.eye(2, 3)) for m in al.steady([fits[4]] * 3, 5)[0]))

masks = sf.render(W, H, style="sandstone")["masks"]
windows = np.zeros((H, W), np.float32)
for name, (group, m) in masks.items():
    if "W_" in name or "window" in group.lower():
        windows = np.maximum(windows, (m > 0).astype(np.float32))
moving, Mw = drifted(edit, 1.0, 1.0, 5.0, 3.0)
lively = moving.copy()
stripes = (np.indices((H, W))[1] // 6 % 2).astype(np.float32)[..., None]
lively[windows > 0.5] = (stripes * np.array([1.0, 0.6, 0.2]))[windows > 0.5]      # hard stripes in every window: strong false edges
rs = al.estimate(src, lively, still=1.0 - windows)
no = al.estimate(src, lively)
print("   with the mask:", round(corner_error(rs["matrix"], Mw), 2), "px off the truth; without:", al.describe(no), round(corner_error(no["matrix"], Mw), 2))
check("a 'still' mask: the fit looks only where the clip is meant to stay on the source", 0.03 < float(windows.mean()) < 0.6 and rs["ok"]
      and corner_error(rs["matrix"], Mw) < 1.0, f"{al.describe(rs)}; windows cover {windows.mean():.2f}")
check("a 'still' mask that leaves nothing is refused with a reason", not al.estimate(src, lively, still=np.zeros((H, W), np.float32))["ok"]
      and "still" in al.estimate(src, lively, still=np.zeros((H, W), np.float32))["why"])

print()
print("all align tests passed" if not failures else f"{len(failures)} FAILED: " + ", ".join(failures))
sys.exit(1 if failures else 0)
