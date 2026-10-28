// Checks of the master timeline's clip model (web/kubakub_master.js), appended to it by tests/test_master.py.
// CASES (from Python's media.segments / source_time) is put in front of this file by the test.
let fails = 0;
const check = (name, ok, d) => { console.log((ok ? "ok   " : "FAIL ") + name + (ok ? "" : "  " + JSON.stringify(d))); if (!ok) fails++; };

// 1. the same seconds into the file as the director's Python model, at every probed time
const bad = [];
for (const c of CASES) {
  const s = segOf(c.seg, c.dur);
  c.t.forEach((t, i) => {
    const got = srcTime(s, t), want = c.want[i];
    if ((got === null) !== (want === null) || (got !== null && Math.abs(got - want) > 1e-9)) bad.push([c.name, t, got, want]);
  });
}
check(`segOf / srcTime = media.segments / source_time (${CASES.length} clips)`, !bad.length, bad.slice(0, 3));

// 2. a cut never changes the picture: left and right show what the whole clip showed
const env = { fps: 25, mediaDur: () => 8, mediaFps: () => 25, dirDur: () => 12 };
const shows = (items, t) => { for (const it of items) { const v = srcTime(segOf(it, 8), t); if (v !== null) return +v.toFixed(6); } return null; };
for (const [name, clip] of [["forward", { speed: 1.5 }], ["backwards", { reverse: true }], ["held at the end", { length: 9, fill: "hold" }],
  ["looped", { out: 3, length: 9, fill: "loop" }], ["trimmed in and out", { in: 1.2, out: 5.6, speed: 0.75 }]]) {
  const whole = { id: "a", type: "clip", media: "m", start: 2, in: 0, out: -1, speed: 1, length: -1, fill: "hold", reverse: false, props: {}, ...clip };
  const [a0, b0] = spanOf(whole, env), diffs = [];
  for (const cut of [a0 + 0.52, (a0 + b0) / 2, b0 - 0.6]) {
    const parts = splitItem(JSON.parse(JSON.stringify(whole)), Math.round(cut * 25) / 25, env);
    if (!parts) { diffs.push(["no cut", cut]); continue; }
    // looped: after the cut the loop starts again from the cut (as Ctrl K in the director), so only its first pass is the same
    const until = name === "looped" ? whole.start + segOf(whole, 8).natural : b0;
    for (let t = a0; t < until - 1e-9; t += 1 / 25) {
      const w = shows([whole], t), g = shows(parts, t);
      // a held last frame may be taken one frame earlier after the cut (as in the director)
      const tol = name === "held at the end" ? 1 / 25 + 1e-6 : 1e-3;
      if (w === null || g === null || Math.abs(w - g) > tol) { diffs.push([cut, +t.toFixed(3), w, g]); break; }
    }
    const [pa] = spanOf(parts[0], env), [, pb] = spanOf(parts[1], env);
    if (Math.abs(pa - a0) > 1e-6 || Math.abs(pb - b0) > 1e-3) diffs.push(["span", pa, pb, a0, b0]);
  }
  check(`cut keeps the picture: ${name}`, !diffs.length, diffs[0]);
}

// 3. a director block: cut and trims count in seconds of the director's own timeline
const d = { id: "d", type: "director", node: 7, start: 1, in: 0, length: -1, props: {} };
check("director block spans its timeline", JSON.stringify(spanOf(d, env)) === "[1,13]", spanOf(d, env));
const [dl, dr] = splitItem(JSON.parse(JSON.stringify(d)), 5, env);
check("director block cut", dl.length === 4 && dr.start === 5 && dr.in === 4 && dr.length === -1 && spanOf(dr, env)[1] === 13, [dl, dr]);
const t2 = JSON.parse(JSON.stringify(d)); trimLeftTo(t2, 3, env);
check("director block trim in", t2.start === 3 && t2.in === 2 && spanOf(t2, env)[1] === 13, t2);

// 3b. the start edge of a picture (no time inside it): start moves, the end stays
const penv = { ...env, mediaDur: () => 0 };
const pic = { id: "p", type: "clip", media: "m", start: 2, in: 0, out: -1, speed: 1, length: 2, fill: "hold", reverse: false, props: {} };
const p1 = JSON.parse(JSON.stringify(pic)); trimLeftTo(p1, 2.6, penv);
const p2 = JSON.parse(JSON.stringify(pic)); trimLeftTo(p2, 1, penv);
const p3 = JSON.parse(JSON.stringify(pic)); trimLeftTo(p3, 9, penv);
check("picture: trim in from the start", p1.start === 2.6 && p1.length === 1.4 && p1.in === 0, p1);
check("picture: the start pulled earlier makes it longer", p2.start === 1 && p2.length === 3, p2);
check("picture: never shorter than a frame", p3.start === 3.96 && p3.length === 0.04, p3);
const v1 = { id: "v", type: "clip", media: "m", start: 2, in: 0, out: -1, speed: 1, length: -1, fill: "hold", reverse: false, props: {} };
trimLeftTo(v1, 3, env);
check("video: trim in from the start (in follows)", v1.start === 3 && v1.in === 1 && spanOf(v1, env)[1] === 10, [v1, spanOf(v1, env)]);

// 4. overwrite: a piece dropped in the middle of another leaves both ends
const tr = { items: [{ id: "x", type: "clip", media: "m", start: 0, in: 0, out: -1, speed: 1, length: -1, fill: "hold", reverse: false, props: {} }] };
clearRange(tr, 2, 3, new Set(), env);
const shape = tr.items.map(it => spanOf(it, env).map(v => +v.toFixed(3)).join("-")).join(" ");
check("overwrite in the middle", shape === "0-2 3-8" && Math.abs(tr.items[1].in - 3) < 1e-9, shape);
const tr2 = { items: [{ ...tr.items[0], id: "y", start: 0, in: 0, out: -1 }] }; clearRange(tr2, -1, 9, new Set(), env);
check("overwrite of all of it removes it", tr2.items.length === 0, tr2.items);

// 5. small things
check("whole fps (as project settings)", wholeFps(29.97) === 30 && wholeFps(0) === 1 && wholeFps(999) === 120 && wholeFps("x") === 25);
const r = layoutRect({ fit: "fit", scale: 1, x: 0, y: 0 }, 1000, 1000, 1920, 1080);
check("fit: a square in 16:9 is as high as the frame", Math.abs(r.h - 1080) < 1e-9 && Math.abs(r.x - 420) < 1e-9, r);
const m = migrate({ fps: 23.976, tracks: [{ items: [{ type: "clip", media: "m" }, { type: "nonsense" }] }] });
check("migrate: whole fps, unknown pieces dropped, props filled", m.fps === 24 && m.tracks[0].items.length === 1 && m.tracks[0].items[0].props.fit === "fit", m);
// 5. keys: the director's interpolation; a cut, a trim in and a move keep every key where it is in time
const kbad = [];
for (const c of KEY_CASES) c.t.forEach((t, i) => { const g = keyValue(c.keys, t); if (Math.abs(g - c.want[i]) > 1e-9) kbad.push([t, g, c.want[i]]); });
check(`keyValue = key_value of the director (${KEY_CASES.length} key sets)`, !kbad.length, kbad.slice(0, 3));
const anim = () => ({ id: "k", type: "clip", media: "m", start: 2, in: 0, out: -1, speed: 1, length: -1, fill: "hold", reverse: false,
  props: { opacity: 1, x: 0, y: 0, scale: 1, fit: "fit" }, anim: { opacity: [{ t: 0, v: 0, e: "linear" }, { t: 3, v: 1, e: "smooth" }], x: [{ t: 1, v: -50, e: "hold" }, { t: 4, v: 80, e: "smooth" }] } });
const at = (items, t) => { for (const it of items) { const [a, b] = spanOf(it, env); if (t >= a - 1e-9 && t < b - 1e-9) return propsAt(it, t); } return null; };
const same = (p, q) => p && q && Math.abs(p.opacity - q.opacity) < 1e-6 && Math.abs(p.x - q.x) < 1e-6;
const kdiff = [], kwhole = anim();
const kparts = splitItem(anim(), 4.2, env);
for (let t = 2; t < 10 - 1e-9; t += 0.04) if (!same(at([kwhole], t), at(kparts, t))) { kdiff.push(["cut", +t.toFixed(2)]); break; }
const kt = anim(); trimLeftTo(kt, 3.5, env);
for (let t = 3.5; t < 10 - 1e-9; t += 0.04) if (!same(at([kwhole], t), at([kt], t))) { kdiff.push(["trim", +t.toFixed(2)]); break; }
check("keys stay in time through a cut and a trim in", !kdiff.length, kdiff[0]);
const k1 = anim(); k1.start = 5;
check("keys move with the piece", Math.abs(propsAt(k1, 6.5).opacity - 0.5) < 1e-9 && propsAt(k1, 5).x === -50, propsAt(k1, 6.5));
const k2 = anim(); delete k2.anim;
toggleKey(k2, "scale", 3, 25); setProp(k2, "scale", 2, 4, 25); setProp(k2, "opacity", 0.3, 4, 25);
check("a key button, then edits make keys (a value without keys stays plain)",
  k2.anim.scale.length === 2 && k2.anim.scale[0].t === 1 && k2.anim.scale[1].v === 2 && !k2.anim.opacity && k2.props.opacity === 0.3, k2);
setProp(k2, "scale", 3, 4.01, 25); toggleKey(k2, "scale", 3, 25); toggleKey(k2, "scale", 4, 25);
check("an edit within half a frame changes that key; removing the last key leaves its value",
  !k2.anim && k2.props.scale === 3, k2);
const k3 = anim(); easeAt(k3, 2, 25, "hold");
check("ease: every key at that time", k3.anim.opacity[0].e === "hold" && k3.anim.x[0].e === "hold", k3.anim);
check("key times, each once", JSON.stringify(keyTimes(anim())) === "[0,1,3,4]", keyTimes(anim()));
const mk = migrate({ tracks: [{ items: [{ type: "clip", media: "m", anim: { opacity: [{ t: 2, v: 1 }, { t: 1, v: "x" }, { t: 0, v: 0, e: "odd" }], fit: [{ t: 0, v: 1 }], x: [] } }] }] });
check("migrate: keys sorted and cleaned, only the four settings", JSON.stringify(mk.tracks[0].items[0].anim) === '{"opacity":[{"t":0,"v":0,"e":"smooth"},{"t":2,"v":1,"e":"smooth"}]}', mk.tracks[0].items[0].anim);
process.exit(fails ? 1 : 0);
