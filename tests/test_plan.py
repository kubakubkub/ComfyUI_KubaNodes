"""
Model free test for the Kuba Regions plan (kubakub/plan.py).

Resolves rule text against a small mask-folder region table and checks
selectors, priority, prompt composition, choices, ordering and error messages.
Does not start ComfyUI and loads no model.

Run from the ComfyUI_windows_portable folder:
    python_embeded\\python.exe ComfyUI\\custom_nodes\\ComfyUI_KubaNodes\\tests\\test_plan.py
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from kubakub import plan as rp  # noqa: E402

failures = []


def check(name, cond, detail=""):
    print(("ok   " if cond else "FAIL ") + name + (f"  ({detail})" if detail and not cond else ""))
    if not cond:
        failures.append(name)


def region(i, name, group, area, tags):
    return {"region_id": i, "name": name, "group_id": group, "bbox": [0, 0, 10, 10],
            "area": area, "tags": tags}


TABLE = {"width": 3840, "height": 2160, "regions": [
    region(0, "M_Cornices_Portals", "Groups", 600000, ["M_Facade_Except_Glass", "M_FLOOR_F3"]),
    region(1, "M_FLOOR_F3", "Groups", 620000, ["M_Facade_Except_Glass"]),
    region(2, "M_Pilasters_01", "M_Pilasters", 17000, ["M_Facade_Except_Glass", "M_FLOOR_F3"]),
    region(3, "M_Pilasters_02", "M_Pilasters", 18000, ["M_Facade_Except_Glass", "M_FLOOR_F3"]),
    region(4, "W_F3_C01", "Windows", 20000, ["M_FLOOR_F3", "M_Glass_All"]),
    region(5, "W_F1_C01", "Windows", 39000, ["M_FLOOR_F1", "M_Glass_All"]),
    region(6, "W_F1_C02", "Windows", 39100, ["M_FLOOR_F1", "M_Glass_All"]),
    region(7, "W_F1_C03", "Windows", 39200, ["M_FLOOR_F1", "M_Glass_All"]),
    region(8, "M_FLOOR_F1", "Groups", 1500000, ["M_Facade_Except_Glass"]),
]}

RULES = """
// test plan
[plan]
order = largest_first

[default]
prompt = weathered sandstone facade at night
denoise = 0.55

[group:M_Pilasters]
prompt += carved vines climbing the stone     // appended to the default

[group:windows]
prompt = dark glass, reflections of {name}
denoise = 0.7

[W_F1_*]
strategy = frame_in_frame
prompt = an underwater room with {jellyfish|a sunken piano|koi|a diver}

[tag:M_FLOOR_F3 group:Windows]
denoise = 0.9

[region:7]
denoise = 0.3

[region:0, M_FLOOR_F3]
strategy = keep

[tag:nothing_here]
denoise = 0.1
"""


def by_name(plan):
    return {e["name"]: e for e in plan["regions"]}


def test_resolve():
    p = rp.resolve(TABLE, RULES, seed=42)
    r = by_name(p)
    check("default prompt", r["M_FLOOR_F1"]["prompt"] == "weathered sandstone facade at night")
    check("default denoise", r["M_FLOOR_F1"]["denoise"] == 0.55)
    check("+= appends", r["M_Pilasters_01"]["prompt"] ==
          "weathered sandstone facade at night, carved vines climbing the stone", r["M_Pilasters_01"]["prompt"])
    check("group case insensitive + {name}", r["W_F3_C01"]["prompt"] == "dark glass, reflections of W_F3_C01")
    check("name wildcard beats group", r["W_F1_C01"]["strategy"] == "frame_in_frame")
    check("name wildcard prompt", r["W_F1_C01"]["prompt"].startswith("an underwater room with "))
    check("AND beats single group", r["W_F3_C01"]["denoise"] == 0.9)
    check("region beats name", r["W_F1_C03"]["denoise"] == 0.3)
    check("name rule keeps group denoise for others", r["W_F1_C02"]["denoise"] == 0.7)
    check("OR header: region:0 keep", r["M_Cornices_Portals"]["strategy"] == "keep")
    check("OR header: name keep", r["M_FLOOR_F3"]["strategy"] == "keep")
    check("rules listed", r["W_F1_C03"]["rules"][-1] == "region:7", str(r["W_F1_C03"]["rules"]))
    check("seed = plan seed + id", r["W_F1_C02"]["seed"] == 42 + 6)
    check("unmatched section noted", any("tag:nothing_here" in n for n in p["notes"]), str(p["notes"]))
    check("keep not in order", 0 not in p["order"] and 1 not in p["order"])
    areas = [by_id(p, i)["area"] for i in p["order"]]
    check("largest_first", areas == sorted(areas, reverse=True), str(areas))
    choices = {r[f"W_F1_C0{c}"]["prompt"] for c in (1, 2, 3)}
    check("choices vary across windows (seed 42)", len(choices) > 1, str(choices))
    p2 = rp.resolve(TABLE, RULES, seed=42)
    check("choices are repeatable", [e["prompt"] for e in p2["regions"]] == [e["prompt"] for e in p["regions"]])
    print(rp.report(p))


def by_id(p, i):
    return next(e for e in p["regions"] if e["region_id"] == i)


def test_negation_and_bare_settings():
    p = rp.resolve(TABLE, "prompt = stone\n[group:windows !W_F1_*]\nstrategy = keep\n")
    r = by_name(p)
    check("settings before any header = default", r["M_FLOOR_F1"]["prompt"] == "stone")
    check("negation excludes", r["W_F1_C01"]["strategy"] == "inpaint" and r["W_F3_C01"]["strategy"] == "keep")
    p = rp.resolve(TABLE, "[region:2-3]\ndenoise = 0.2\n[*]\nstrategy = skip\n[region:2-3]\nstrategy=inpaint")
    r = by_name(p)
    check("region range", r["M_Pilasters_01"]["denoise"] == 0.2 and r["M_Pilasters_02"]["denoise"] == 0.2)
    check("skip alias = keep", r["W_F1_C01"]["strategy"] == "keep")
    check("region beats later default", r["M_Pilasters_02"]["strategy"] == "inpaint")
    p = rp.resolve(TABLE, "[default]\nprompt = a, b\n[name:m_pilasters_01]\nprompt = {prompt}, c")
    check("{prompt} placeholder", by_name(p)["M_Pilasters_01"]["prompt"] == "a, b, c")


def expect_error(name, rules, fragment):
    try:
        rp.resolve(TABLE, rules)
        check(name, False, "no error")
    except rp.PlanError as e:
        check(name, fragment in str(e), str(e))


def test_errors():
    expect_error("unknown key with hint", "[default]\ndenoize = 0.5", "did you mean 'denoise'")
    expect_error("bad strategy", "[default]\nstrategy = paint", "must be one of")
    expect_error("bad number", "[default]\ndenoise = lots", "line 2")
    expect_error("missing region", "[region:99]\ndenoise = 0.5", "region 99 does not exist")
    expect_error("bad selector kind", "[floor:F1]\ndenoise = 0.5", "unknown selector kind")
    expect_error("unclosed header", "[default\ndenoise = 0.5", "missing its ']'")
    expect_error("+= on a number", "[default]\ndenoise += 0.5", "only works for prompt")
    expect_error("garbage line", "[default]\njust words", "expected 'key = value'")


if __name__ == "__main__":
    for t in (test_resolve, test_negation_and_bare_settings, test_errors):
        t()
    print()
    if failures:
        print(f"{len(failures)} FAILED: {', '.join(failures)}")
        sys.exit(1)
    print("all plan tests passed")
