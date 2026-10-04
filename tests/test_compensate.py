"""
Model free test of kubakub brightness compensation (kubakub/scene3d/compensate.py) on synthetic brightness maps.

Run from the ComfyUI_windows_portable folder:
    python_embeded\\python.exe ComfyUI\\custom_nodes\\ComfyUI_KubaNodes\\tests\\test_compensate.py
"""

import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "kubakub", "scene3d"))
import compensate as cp  # noqa: E402

failures = []


def check(name, cond, detail=""):
    print(("ok   " if cond else "FAIL ") + name + (f"  ({detail})" if detail else ""))
    if not cond:
        failures.append(name)


H, W = 100, 200
fg = np.ones((H, W), bool)
fg[:, :10] = False                                   # sky on the left edge

g, info = cp.gain_map(np.ones((H, W), np.float32), fg)
check("even building -> gain 1", np.allclose(g, 1.0, atol=1e-5), f"{g.min()}..{g.max()}")

# left 40 % of the building half as bright (far away), the rest full
b = np.ones((H, W), np.float32)
b[:, :90] = 0.5
g, info = cp.gain_map(b, fg, "dim areas", 1.0, 1.0, 0.0)
check("dim areas: bright part darkened to 0.5", abs(float(np.median(g[:, 100:])) - 0.5) < 1e-4, str(info))
check("dim areas: dim part untouched", abs(float(np.median(g[:, 10:90])) - 1.0) < 1e-4)
check("light kept reported", 0.5 < info["kept"] < 0.8, f"{info['kept']:.3f}")

frame = np.full((H, W, 3), 0.8, np.float32)
out, clipped = cp.apply(frame, g)
seen = cp.srgb_to_linear(out)[..., 0] * b          # what the audience sees: content x projector light
check("after compensation both parts look the same", abs(float(np.median(seen[:, 20:80])) - float(np.median(seen[:, 120:]))) < 2e-3,
      f"{np.median(seen[:, 20:80]):.4f} vs {np.median(seen[:, 120:]):.4f}")
check("nothing clips without lift", clipped == 0.0)

g0, _ = cp.gain_map(b, fg, "dim areas", 0.0, 1.0, 0.0)
check("strength 0 -> unchanged", np.allclose(g0, 1.0))
gh, _ = cp.gain_map(b, fg, "dim areas", 0.5, 1.0, 0.0)
check("strength 0.5 -> halfway in log (0.707)", abs(float(np.median(gh[:, 100:])) - 0.5 ** 0.5) < 1e-3)

g2, info2 = cp.gain_map(b, fg, "average", 1.0, 2.0, 0.0)
check("average + lift 2: bright part stays, dim part lifted x2", abs(float(np.median(g2[:, 100:])) - 1) < 1e-4
      and abs(float(np.median(g2[:, 10:90])) - 2) < 1e-4, str(info2))
_, clipped2 = cp.apply(np.full((H, W, 3), 0.95, np.float32), g2)
check("lift clips bright content in the dim part", 0.3 < clipped2 < 0.5, f"{clipped2:.3f}")
g15, _ = cp.gain_map(b, fg, "average", 1.0, 1.5, 0.0)
check("lift capped at 1.5", abs(float(g15.max()) - 1.5) < 1e-4)

# grazing side faces (very dim) must not drag the target down
bz = np.ones((H, W), np.float32)
bz[:, 90:100] = 0.5
bz[:, 150:170] = 0.05                               # 11 % of the building: grazing
inc = np.zeros((H, W), np.float32)
inc[:, 150:170] = 80.0
gz, iz = cp.gain_map(bz, fg, "dim areas", 1.0, 1.0, 0.0, incidence=inc, grazing_deg=60.0)
gn, inn = cp.gain_map(bz, fg, "dim areas", 1.0, 1.0, 0.0)
check("grazing faces ignored for the target", abs(iz["target"] - 1.0) < 1e-4 or iz["target"] >= 0.5,
      f"with {iz['target']:.2f}, without {inn['target']:.2f}")
check("without the rule they drag it down", inn["target"] < 0.1, f"{inn['target']:.3f}")
check("grazing share reported", abs(iz["grazing"] - 20 / 190) < 1e-3, f"{iz['grazing']:.3f}")
check("grazing faces keep gain 1", abs(float(np.median(gz[:, 150:170])) - 1.0) < 1e-4)

# a gradient inside one region -> one gain per region
grad = np.tile(np.linspace(0.4, 1.0, W, dtype=np.float32), (H, 1))
lab = np.full((H, W), -1, np.int32)
lab[20:80, 30:170] = 3
gr, _ = cp.gain_map(grad, fg, "dim areas", 1.0, 1.0, 0.0, labels=lab)
check("per region: constant inside the region", float(np.ptp(gr[20:80, 30:170])) < 1e-6)
check("per region: pixels outside regions stay per pixel", float(np.ptp(gr[0:10, 10:])) > 0.1)
gs, _ = cp.gain_map(b, fg, "dim areas", 1.0, 1.0, 12.0)
check("smoothing: finite, soft step", np.isfinite(gs).all() and 0.5 < float(gs[50, 90]) < 1.0, f"{gs[50, 90]:.3f}")
check("outside the building filled from inside (no halo to 1)", abs(float(gs[50, 2]) - 1.0) < 0.05)

x = np.random.default_rng(0).random((64, 64, 3), dtype=np.float32)
lut, _ = cp.apply(x, np.full((64, 64), 0.6, np.float32))
exact = cp.linear_to_srgb(cp.srgb_to_linear(x) * 0.6)
check("lookup tables match the exact sRGB maths", float(np.abs(lut - exact).max()) < 2e-4, f"{np.abs(lut - exact).max():.2e}")
check("resize gain to the matrix", cp.resize_gain(g, 400, 200).shape == (200, 400))

big = np.random.default_rng(1).random((2160, 3840, 3), dtype=np.float32)
gb = np.full((2160, 3840), 0.7, np.float32)
t = time.perf_counter()
cp.apply(big, gb)
dt = time.perf_counter() - t
check("4k frame in under 2 s", dt < 2.0, f"{dt:.2f} s")

print("\n" + ("ALL OK" if not failures else f"{len(failures)} FAILED: {failures}"))
sys.exit(1 if failures else 0)
