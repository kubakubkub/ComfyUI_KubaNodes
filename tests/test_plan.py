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


def test_then():
    """[then]: the sections after it win over the ones before, whatever they name (the chain of kubakub region prompt)."""
    rules = "[W_F1_*]\nprompt = narrow\ndenoise = 0.2\n[then]\n[default]\nprompt = wide\n"
    p = by_name(rp.resolve(TABLE, rules))
    name = next(n for n in p if n.startswith("W_F1_"))
    check("[then]: a wider rule after it wins over a narrower one before it", p[name]["prompt"] == "wide")
    check("[then]: what the later rule does not set stays", p[name]["denoise"] == 0.2)
    q = by_name(rp.resolve(TABLE, rules.replace("[then]\n", "")))
    check("without [then] the narrower rule wins, as before", q[name]["prompt"] == "narrow")
    r = by_name(rp.resolve(TABLE, "[default]\nprompt = a\n[then]\nprompt = b\n"))
    check("[then]: a bare setting after it belongs to the later layer", r[name]["prompt"] == "b")


def test_reserved_and_comments():
    """'then' and 'plan' in a 'which regions' text are names; '//' is a comment only after a space."""
    t = {"regions": [{"region_id": 0, "name": "then", "group_id": "a", "tags": []},
                     {"region_id": 1, "name": "plan", "group_id": "a", "tags": []},
                     {"region_id": 2, "name": "wall", "group_id": "b", "tags": []}]}
    check("which regions: 'then' and 'plan' are region names there", rp.pick_ids(t, "then") == [0] and rp.pick_ids(t, "plan") == [1]
          and rp.selector("then, wall").match(t["regions"][2]) is not None and rp.selector("then").match(t["regions"][2]) is None)
    check("a header for the rules: a reserved word becomes a name selector",
          rp.header_of(" Then ") == "name:Then" and rp.header_of("group:a") == "group:a")
    p = rp.resolve(t, "[default]\nprompt = a 50//50 mix  // a note\n// a whole line\ndenoise = 0.4 // another\n")
    check("'//' inside a word is text, after a space it starts a comment",
          p["regions"][0]["prompt"] == "a 50//50 mix" and p["regions"][0]["denoise"] == 0.4, p["regions"][0]["prompt"])
    check("a URL in a prompt stays whole", rp.resolve(t, "[default]\nprompt = like http://x.y/z\n")["regions"][0]["prompt"] == "like http://x.y/z")


def expect_error(name, rules, fragment):
    try:
        rp.resolve(TABLE, rules)
        check(name, False, "no error")
    except rp.PlanError as e:
        check(name, fragment in str(e), str(e))


def test_one_reading():
    """Every 'which regions' input reads its text the same way (plan.pick_ids): lists by comma or new line, names
    and tags with spaces, and an unreadable part is an error (2026-10-18: the region sampler, versions, the video
    sampler and regions to svg had their own readings)."""
    t = {"regions": [{"region_id": 0, "name": "wall", "group_id": "walls", "tags": []},
                     {"region_id": 1, "name": "W_F1_01", "group_id": "Windows", "tags": ["front", "floor 1"]},
                     {"region_id": 2, "name": "W_F1_02", "group_id": "Windows", "tags": ["side"]},
                     {"region_id": 3, "name": "ground door", "group_id": "Doors", "tags": ["front"]}]}
    ids = lambda text: rp.pick_ids(t, text)  # noqa: E731
    check("empty, blank and * = all", ids("") == ids(" \n ") == ids("*") == [0, 1, 2, 3])
    check("new line = comma", ids("wall\nW_F1_02") == ids("wall, W_F1_02") == [0, 2])
    check("a name with a space inside a list", ids("wall, ground door") == [0, 3] and ids("Ground Door") == [3])
    check("a tag with a space", ids("tag:floor 1") == [1], str(ids("tag:floor 1")))
    check("space = and, ! = not, as in the rules", ids("group:Windows tag:front") == [1] and ids("tag:front !W_*") == [3])
    check("table order, each region once", ids("W_F1_02, wall, W_F1_*") == [0, 1, 2])
    check("matches nothing: empty (the nodes say so)", ids("nothing_here") == [])
    try:
        ids("floor:1")
        check("a part that is no selector and no name: an error", False)
    except rp.PlanError:
        check("a part that is no selector and no name: an error", True)
    try:
        rp.pick_or_say(t, "X_*", "kubakub node")
        check("pick_or_say: no match names the regions there are", False)
    except ValueError as e:
        check("pick_or_say: no match names the regions there are",
              str(e).startswith("kubakub node: which regions = 'X_*' matches no region (there are: wall, W_F1_01"), str(e))
    try:
        rp.pick_or_say(t, "X_*", "")
    except ValueError as e:
        check("pick_or_say without a node: the message itself", str(e).startswith("which regions = 'X_*'"), str(e))


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
    for t in (test_resolve, test_negation_and_bare_settings, test_then, test_reserved_and_comments, test_one_reading, test_errors):
        t()
    print()
    if failures:
        print(f"{len(failures)} FAILED: {', '.join(failures)}")
        sys.exit(1)
    print("all plan tests passed")
