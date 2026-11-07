"""
Model free test of the kubakub shader session (kubakub/shader_render.py Session, open_session, prepared): a shader
drawn frame after frame with its set-up kept must give the same numbers, bit for bit, as render() of each frame alone.
The GPU part is skipped with a note where ComfyUI's GL context does not start.

Run from the ComfyUI_windows_portable folder:
    python_embeded\\python.exe ComfyUI\\custom_nodes\\ComfyUI_KubaNodes\\tests\\test_shader_session.py
"""

import gc
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(HERE))))     # ComfyUI (folder_paths, comfy_api)
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)     # embedded Python leaves the script folder off the path
import _pack  # noqa: E402,F401  (the pack as a package, see tests/_pack.py)

from kubakub import shader_render as sr  # noqa: E402
from kubakub import shadertoy as st  # noqa: E402

failures = []


def check(name, cond, detail=""):
    print(("ok   " if cond else "FAIL ") + name + (f"  ({detail})" if detail and not cond else ""))
    if not cond:
        failures.append(name)


def same(a, b):
    a, b = np.asarray(a, np.float32), np.asarray(b, np.float32)
    return a.shape == b.shape and np.array_equal(a.view(np.uint32), b.view(np.uint32))


# prepared() without the card
rng = np.random.default_rng(5)
pic = rng.random((6, 5, 3), dtype=np.float32)
check("prepared: rgba, bottom row first, float32", sr.prepared(pic).dtype == np.float32
      and np.array_equal(sr.prepared(pic)[::-1], sr.rgba(pic)))
two = rng.random((4, 3, 2), dtype=np.float32)
check("prepared: field + mask as rgba() lays it out", np.array_equal(sr.prepared(two)[::-1], sr.rgba(two)))
grey = rng.random((4, 3), dtype=np.float32)
check("prepared: grey as rgba() lays it out", np.array_equal(sr.prepared(grey)[::-1], sr.rgba(grey)))
half = rng.random((4, 3, 4)).astype(np.float16)
check("prepared: half floats stay half floats", sr.prepared(half).dtype == np.float16
      and np.array_equal(sr.prepared(half)[::-1], half))

CODE = """uniform float gain; // 1 0 4
void mainImage(out vec4 o, vec2 f) {
    vec2 uv = f / iResolution.xy;
    vec4 a = texture(iChannel0, uv), b = texture(iChannel1, uv * 0.7 + 0.1);
    o = vec4(fract(a.rgb * gain + b.gba * 0.37 + iTime * 0.1 + float(iFrame) * 0.01)
             + iChannelResolution[1].x * 1e-4 + texture(iChannel2, uv).r * 0.3, 1.0);
}"""
PREV = """void mainImage(out vec4 o, vec2 f) { vec2 uv = f / iResolution.xy;
    o = vec4(kPrev(uv).rgb * 0.5 + texture(iChannel0, uv).rgb + iTime * 0.01, 1.0); }"""

try:
    W, H = 96, 64
    t = st.translate(CODE)
    still = rng.random((H, W, 3), dtype=np.float32)
    dyn = [(rng.random((H, W, 4), dtype=np.float32) * 3.0, rng.random((H, W, 4), dtype=np.float32)) for _ in range(3)]
    dyn[2][0][0, 0, 0] = 70000.0                         # a number too big for a half float
    dyn[1][0][:8, :8] = rng.random((8, 8, 4), dtype=np.float32) * 6e-5        # tiny numbers: the card rounds them
    times = sr.frame_times(3, 25.0)
    static = {"gain": 1.5, "kSeconds": 0.12}

    def one(i, source=t.source, chans=None):
        chans = chans or {0: dyn[i][0], 1: dyn[i][1], 2: still}
        out = []
        sr.render(source, W, H, [times[i]], static, {c: (lambda _, a=a, c=c: ((c, i), a)) for c, a in chans.items()},
                  lambda _, px: out.append(px.copy()))
        return out[0]

    s = sr.open_session(t.source, W, H)
    out = np.empty((H, W, 4), np.float32)
    ok = True
    for i in range(3):
        s.draw(times[i], static, {0: (None, sr.prepared(dyn[i][0])), 1: (None, dyn[i][1]),
                                  2: (("still", id(still)), still)}, out)
        ok = ok and same(out, one(i))
    check("session: every frame as render() of it alone, bit for bit", ok)
    kept = dyn[0][1].astype(np.float16)
    ok_half = same(s.draw(times[0], static, {0: (None, dyn[0][0]), 1: (None, kept), 2: (("still", id(still)), still)}),
                   one(0, chans={0: dyn[0][0], 1: kept.astype(np.float32), 2: still}))
    check("session: half floats uploaded as they are = their float32 through render()", ok_half)

    other = sr.open_session(st.translate(PREV).source, W, H)
    third = sr.open_session(t.source, W, H)
    for i in range(3):                                  # three sessions in turn: one gives its memory back each time
        a = other.draw(times[i], {}, {0: (None, dyn[i][0])})
        b = third.draw(times[i], static, {0: (None, dyn[i][0]), 1: (None, dyn[i][1]), 2: (("still", 1), still)})
        c = s.draw(times[i], static, {0: (None, dyn[i][0]), 1: (None, dyn[i][1]), 2: (("still", id(still)), still)})
        ok = ok and same(a, one(i, st.translate(PREV).source, {0: dyn[i][0]})) and same(b, one(i)) and same(c, one(i))
    check("session: sessions in turn (and made again after giving memory back) stay exact; kPrev reads black", ok)
    check("session: only the last two keep graphics memory", len(sr._SESSIONS) <= sr._KEEP_SESSIONS)

    sr.render(t.source, W, H, [times[0]], {"gain": 3.0}, None, lambda _, px: None)     # render() between draws
    check("session: a render() in between changes nothing", same(s.draw(times[1], static, {
        0: (None, dyn[1][0]), 1: (None, dyn[1][1]), 2: (("still", id(still)), still)}), one(1)))

    check("session: a frame's own picture (key None) is not kept, a kept one is",
          s._keys[0][1] is None and s._keys[2][1] is still)
    s.close()
    check("session: drawn again after close()", same(s.draw(times[0], static, {
        0: (None, dyn[0][0]), 1: (None, dyn[0][1]), 2: (("still", id(still)), still)}), one(0)))
    del s, other, third
    gc.collect()
    check("session: one that goes away leaves its objects to delete", len(sr._DEAD) >= 1)
    sr.still(t.source, 8, 8)
    check("session: they are deleted at the next use of the context", not sr._DEAD)

    s = sr.open_session(t.source, W, H)
    s.draw(times[0], static, {0: (None, dyn[0][0]), 1: (None, dyn[0][1]), 2: (("still", id(still)), still)})
    gone = sr.open_session(t.source, W, H)
    gone.draw(times[0], {}, {})
    del gone
    gc.collect()
    sr.release_sessions()
    check("release_sessions: live and dead sessions give their graphics memory back",
          not sr._SESSIONS and not sr._DEAD and not s._objs)
    check("release_sessions: a later draw makes it again, exact", same(s.draw(times[1], static, {
        0: (None, dyn[1][0]), 1: (None, dyn[1][1]), 2: (("still", id(still)), still)}), one(1)))
    s.close()
    for k in range(17):                                 # more shaders than are kept: the stamps go with the programs
        sr.still(st.translate(CODE.replace("0.37", f"0.{k + 40}")).source, 8, 8)
    check("program cache: a dropped shader's stamp goes with it", set(sr._STAMPS) == set(sr._PROGRAMS))
    check("session: a dropped shader is compiled again and stays exact", same(s.draw(times[2], static, {
        0: (None, dyn[2][0]), 1: (None, dyn[2][1]), 2: (("still", id(still)), still)}), one(2)))
except Exception as e:  # noqa: BLE001
    if "graphics context" in str(e):
        print("skip the GPU part: no GL context here")
    else:
        raise

print("\nALL PASSED" if not failures else f"\n{len(failures)} FAILED: {failures}")
sys.exit(1 if failures else 0)
