"""
Model free test for fixed few-step schedules (kubakub/schedules.py).

Run from the ComfyUI_windows_portable folder:
    python_embeded\\python.exe ComfyUI\\custom_nodes\\ComfyUI_KubaNodes\\tests\\test_schedules.py
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from kubakub import schedules as sc  # noqa: E402

failures = []


def check(name, cond, detail=""):
    print(("ok   " if cond else "FAIL ") + name + (f"  ({detail})" if detail and not cond else ""))
    if not cond:
        failures.append(name)


mu = sc.qwen21_mu(1024, 1024)
check("mu at 1024x1024 matches core shift 0.69", abs(mu - 0.6935) < 0.001, f"{mu:.4f}")
check("mu at 256 tokens is the base shift", abs(sc.qwen21_mu(256, 256) - 0.5) < 1e-9)

s = sc.qwen21_turbo_sigmas(1.0, 1024, 1024)
check("full run: 5 steps + 0", len(s) == 6 and float(s[0]) == 1.0 and float(s[-1]) == 0.0, str(s))
check("decreasing", all(float(a) > float(b) for a, b in zip(s[:-1], s[1:])), str(s))
check("shift pushes sigmas up (mu > 0)", float(s[3]) > 0.5, str(s))
check("denoise 0.75 -> 3 steps", len(sc.qwen21_turbo_sigmas(0.75, 1024, 1024)) == 4)
check("denoise 0.55 -> 2 steps", len(sc.qwen21_turbo_sigmas(0.55, 1024, 1024)) == 3)
check("denoise 0.3 -> 1 step", len(sc.qwen21_turbo_sigmas(0.3, 1024, 1024)) == 2)
low = sc.qwen21_turbo_sigmas(0.2, 1024, 1024)
check("denoise below the last timestep starts at denoise (shifted)",
      len(low) == 2 and abs(float(low[0]) - sc.exp_shift(0.2, mu)) < 1e-6, str(low))
check("denoise 0 -> no steps", len(sc.qwen21_turbo_sigmas(0.0, 1024, 1024)) == 1)

card = [1.0, 14 / 15, 6 / 7, 10 / 13, 2 / 3, 6 / 11, 0.4, 2 / 9, 0.0]   # Pruna model card + final 0
p8 = [float(v) for v in sc.pruna_qwen21_8_sigmas(1.0)]
check("pruna full run = model card sigmas", len(p8) == 9 and all(abs(a - b) < 1e-6 for a, b in zip(p8, card)), str(p8))
p75 = sc.pruna_qwen21_8_sigmas(0.75)
check("pruna denoise 0.75 -> starts at 6/7, 6 steps", len(p75) == 7 and abs(float(p75[0]) - 6 / 7) < 1e-6, str(p75))
check("pruna 0.75 starts where viggle 0.75 does at 1 MP",
      abs(float(p75[0]) - float(sc.qwen21_turbo_sigmas(0.75, 1024, 1024)[0])) < 0.01)
p875 = sc.pruna_qwen21_8_sigmas(0.875)
check("pruna denoise 0.875 -> starts at 14/15, 7 steps", len(p875) == 8 and abs(float(p875[0]) - 14 / 15) < 1e-6)
plow = sc.pruna_qwen21_8_sigmas(0.1)
check("pruna below the last timestep starts at shifted denoise",
      len(plow) == 2 and abs(float(plow[0]) - sc.flow_shift(0.1, 2.0)) < 1e-6, str(plow))
check("pruna denoise 0 -> no steps", len(sc.pruna_qwen21_8_sigmas(0.0)) == 1)
check("pruna listed in SCHEDULES", "pruna_qwen21_8" in sc.SCHEDULES)

print()
if failures:
    print(f"{len(failures)} FAILED: {', '.join(failures)}")
    sys.exit(1)
print("all schedule tests passed")
