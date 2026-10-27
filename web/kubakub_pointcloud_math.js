// kubakub point cloud: the numbers the node's live view shares with the render (kubakub/pointcloud.py): the pcg hash,
// the grid, the morph timeline, the camera, the projection and how tracked things are joined. No imports, so
// tests/test_pointcloud.py runs it with node and compares it with the Python. The per-point motion is in the vertex
// shader of kubakub_pointcloud.js (the same formulas as pointcloud.points()).

export const CAM_KEYS = ["yaw", "pitch", "dolly", "x", "y"];
export const LUMA = [0.2126, 0.7152, 0.0722];

export function pcg(v) {
  const s = (Math.imul(v >>> 0, 747796405) + 2891336453) >>> 0;
  const w = Math.imul(((s >>> ((s >>> 28) + 4)) ^ s) >>> 0, 277803737) >>> 0;
  return ((w >>> 22) ^ w) >>> 0;
}
export const rnd = (i, k) => Math.fround(pcg(i * 8 + k) / 4294967295);

export function gridSize(density, W, H) {
  const d = Math.round(density);
  return W >= H ? [d, Math.max(2, Math.round(d * H / W))] : [Math.max(2, Math.round(d * W / H)), d];
}
export function gridUV(i, gw, gh, jitter) {
  const gx = i % gw, gy = Math.floor(i / gw);
  return [(gx + 0.5 + (rnd(i, 0) - 0.5) * jitter) / gw, (gy + 0.5 + (rnd(i, 1) - 0.5) * jitter) / gh];
}
export const distance = (d, depth) => 1 + depth * 2 * (1 - d);

export function morphAt(t, T, K) {                    // = pointcloud.morph_at
  if (K <= 1 || T <= 0) return [0, 0];
  const seg = T / K, k = Math.min(Math.floor(t / seg), K - 1);
  if (k >= K - 1) return [K - 1, 0];
  const m = 0.45 * seg, x = Math.min(Math.max((t - k * seg - (seg - m)) / m, 0), 1);
  return [k, x * x * (3 - 2 * x)];
}

export function ease(a, how) {
  a = Math.min(Math.max(+a, 0), 1);
  return how === "still" ? 0 : how === "linear" ? a : a * a * (3 - 2 * a);
}
export function cameraAt(cam, a, t = 0) {            // cam = look.camera; = pointcloud.camera_at (path, ease, shake)
  const out = {}, path = cam.path || [];
  if (path.length >= 2) {
    a = Math.min(Math.max(+a, 0), 1);
    let j = 0; while (j < path.length - 2 && path[j + 1][0] < a) j++;
    const p = path[j], q = path[j + 1], k = Math.min(Math.max((a - p[0]) / Math.max(q[0] - p[0], 1e-9), 0), 1);
    CAM_KEYS.forEach((key, n) => { out[key] = p[n + 1] + (q[n + 1] - p[n + 1]) * k; });
  } else {
    const k = ease(a, cam.ease);
    for (const key of CAM_KEYS) out[key] = cam.start[key] + (cam.end[key] - cam.start[key]) * k;
  }
  const sh = cam.shake || 0, S = Math.sin;
  if (sh > 0) {
    out.yaw += sh * 1.5 * (S(1.1 * t) + 0.5 * S(2.7 * t + 1.3) + 0.25 * S(6.1 * t + 0.7));
    out.pitch += sh * 1.0 * (S(0.9 * t + 2.1) + 0.5 * S(3.1 * t + 0.4) + 0.25 * S(5.3 * t + 2.9));
    out.x += sh * 0.01 * (S(1.7 * t + 0.3) + 0.5 * S(4.3 * t + 1.9));
    out.y += sh * 0.01 * (S(1.3 * t + 1.1) + 0.5 * S(3.7 * t + 0.8));
  }
  return out;
}
export const trailStride = share => Math.max(1, Math.floor(1 / share + 0.5));   // = pointcloud.trail_stride (half up)
export const TRAIL_STEPS = 12;
export function trailPoint(b, stride, n) {           // = pointcloud.trail_points: one point of every block, by a hash
  return stride <= 1 ? b : Math.min(b * stride + Math.floor(rnd(b, 3) * stride), n - 1);
}
// the flow, = pointcloud: steps over a life, the waves of the swirls (angle, frequency, amplitude, tempo), where it acts
export const FLOW_STEPS = 12, FINE = 1.5, MAX_TRACKED = 32;
export const WAVES = [[0.3, 1.0, 1.0, 0.31], [1.9, 1.7, 0.6, 0.23], [5.5, 2.3, 0.45, 0.11], [3.4, 2.9, 0.35, 0.17], [4.6, 4.3, 0.2, 0.13],
  [2.6, 6.1, 0.12, 0.29]];
export const WHERE = ["everywhere", "field", "tracked", "forces"];
// the forces, = pointcloud: drawn curves and region groups make one field over the picture (vx, vy, curl, near)
export const CURVE_KINDS = ["wind", "attract", "repel", "curl+", "curl-"], REGION_KINDS = ["off", "attract", "repel", "curl+", "curl-"];
export const MAX_CURVES = 8, MAX_CURVE_PTS = 64, FIELD_H = 270, CLOSED = 0.06, PULL = 0.12, CURL_SCALE = 2.5, CURL_SPEED = 0.2;
export const BASIS = [[-0.25, 1], [-1, 1], [-1, 1], [-2, 2], [-2, 2], [0, 1]];
const clampN = (v, lo, hi) => { const f = +v; return Number.isFinite(f) ? Math.min(Math.max(f, lo), hi) : null; };
export function curvesOf(lk) {                       // = pointcloud.curves_of: an older look's arrows are wind curves
  const out = [...(Array.isArray(lk.forces) ? lk.forces : []).filter(f => Array.isArray(f) && f.length === 4).map(f => ({ kind: "wind", pts: [f.slice(0, 2), f.slice(2)] })),
    ...(Array.isArray(lk.curves) ? lk.curves : []).filter(c => c && typeof c === "object")];
  const good = [];
  for (const c of out) {
    const pts = (Array.isArray(c.pts) ? c.pts : []).slice(0, MAX_CURVE_PTS).filter(p => Array.isArray(p) && p.length === 2)
      .map(p => [clampN(p[0], -0.5, 1.5), clampN(p[1], -0.5, 1.5)]).filter(p => p[0] !== null && p[1] !== null);
    let len = 0; for (let k = 1; k < pts.length; k++) len += Math.hypot(pts[k][0] - pts[k - 1][0], pts[k][1] - pts[k - 1][1]);
    if (CURVE_KINDS.includes(c.kind) && pts.length >= 2 && len > 1e-3) good.push({ kind: c.kind, pts });
  }
  return good.slice(-MAX_CURVES);
}
export const fieldSize = aspect => [Math.max(2, Math.min(4 * FIELD_H, Math.round(FIELD_H * aspect))), FIELD_H];
const sm = e => { e = Math.min(Math.max(e, 0), 1); return e * e * (3 - 2 * e); };
function inside(px, py, x, y) {
  let ins = false;
  for (let i = 0, j = px.length - 1; i < px.length; j = i++) {
    const yi = py[i], yj = py[j];
    if (yi !== yj && ((yi > y) !== (yj > y)) && x < (px[j] - px[i]) * (y - yi) / (yj - yi) + px[i]) ins = !ins;
  }
  return ins;
}
// reg = { names, default, q: [Uint8Array (fh * fw * 6) per group] } from the node's basis picture
export function regionKinds(lk, reg) {                // = pointcloud.region_kinds
  if (!reg?.names?.length) return {};
  const kinds = lk.regions && typeof lk.regions === "object" ? lk.regions : { [reg.default]: "attract" }, out = {};
  for (const n of reg.names) if (REGION_KINDS.includes(kinds[n]) && kinds[n] !== "off") out[n] = kinds[n];
  return out;
}
// = pointcloud.curve_field + region_field: Float32Array (fh * fw * 4) rgba = vx, vy, curl, near; null = no forces
const turn = (rx, ry, a) => { const n = Math.max(1, Math.hypot(rx, ry)); return [a * ry / n, -a * rx / n]; };   // = pointcloud._turn
export function forceField(lk, aspect, reg) {          // = pointcloud.force_field (at strength 1) + curly: a curl acts
  const curves = curvesOf(lk), kinds = regionKinds(lk, reg);
  if (!curves.length && !Object.keys(kinds).length) return null;
  const [fw, fh] = fieldSize(aspect), N = fw * fh, out = new Float64Array(N * 4), S = PULL, R = lk.reach;
  for (const c of curves) {
    const px = c.pts.map(p => p[0] * aspect), py = c.pts.map(p => p[1]), n = px.length;
    const loop = n >= 4 && Math.hypot(px[0] - px[n - 1], py[0] - py[n - 1]) < CLOSED;
    let cx = 0, cy = 0, rc = 0;
    if (loop) { cx = px.reduce((a, b) => a + b, 0) / n; cy = py.reduce((a, b) => a + b, 0) / n;
      rc = Math.max(px.reduce((a, x, k) => a + Math.hypot(x - cx, py[k] - cy), 0) / n, 1e-3); }
    const s = c.kind === "curl+" ? 1 : -1;
    for (let j = 0; j < fh; j++) for (let i = 0; i < fw; i++) {
      const X = (i + 0.5) / fw * aspect, Y = (j + 0.5) / fh;
      let dd = Infinity, qx = 0, qy = 0, tx = 0, ty = 0;
      for (let a = 0; a < n - 1; a++) {
        const abx = px[a + 1] - px[a], aby = py[a + 1] - py[a], ln2 = abx * abx + aby * aby;
        if (ln2 < 1e-12) continue;
        const k = Math.min(Math.max(((X - px[a]) * abx + (Y - py[a]) * aby) / ln2, 0), 1), sx = px[a] + k * abx, sy = py[a] + k * aby, d = Math.hypot(X - sx, Y - sy);
        if (d < dd) { const ln = Math.sqrt(ln2); dd = d; qx = sx; qy = sy; tx = abx / ln; ty = aby / ln; }
      }
      let w = sm(1 - dd / R);
      if (loop && inside(px, py, X, Y)) w = 1;
      const safe = Math.max(dd, 1e-9), ex = (X - qx) / safe, ey = (Y - qy) / safe, g = sm(dd / 0.01), o = (j * fw + i) * 4;
      let vx, vy;
      if (c.kind === "wind") { vx = tx * S * w; vy = ty * S * w; }
      else if (c.kind === "attract") { vx = -ex * S * w * g; vy = -ey * S * w * g; }
      else if (c.kind === "repel") { vx = ex * S * w; vy = ey * S * w; }
      else {
        if (loop) [vx, vy] = turn((X - cx) / rc, (Y - cy) / rc, s * S * w);
        else { vx = s * ey * S * w * g; vy = s * -ex * S * w * g; }
        out[o + 2] += s * w;
      }
      out[o] += vx; out[o + 1] += vy; out[o + 3] = Math.max(out[o + 3], w);
    }
  }
  for (const [name, kind] of Object.entries(kinds)) {
    const q = reg.q[reg.names.indexOf(name)], s = kind === "curl+" ? 1 : -1;
    const deq = (o, k) => BASIS[k][0] + q[o * 6 + k] / 255 * (BASIS[k][1] - BASIS[k][0]);
    for (let o = 0; o < N; o++) {
      const sd = deq(o, 0), nx = deq(o, 1), ny = deq(o, 2), rx = deq(o, 3), ry = deq(o, 4);
      const w = sd <= 0 ? 1 : sm(1 - sd / R), ins = Math.min(Math.max(0.5 - sd / 0.008, 0), 1);
      let vx, vy;
      if (kind === "attract") { vx = (1 - ins) * nx * S * w - ins * rx * 0.5 * S; vy = (1 - ins) * ny * S * w - ins * ry * 0.5 * S; }
      else if (kind === "repel") { vx = -(1 - ins) * nx * S * w + ins * rx * S; vy = -(1 - ins) * ny * S * w + ins * ry * S; }
      else { [vx, vy] = turn(rx, ry, s * S * w); out[o * 4 + 2] += s * w; }
      out[o * 4] += vx; out[o * 4 + 1] += vy; out[o * 4 + 3] = Math.max(out[o * 4 + 3], w);
    }
  }
  const curly = [...curves.map(c => c.kind), ...Object.values(kinds)].some(k => k.startsWith("curl"));
  return { data: Float32Array.from(out), fw, fh, curly };
}
export function basis(cam, depth) {                   // = pointcloud.basis -> [position, right, up, forward]
  const R = 1 + depth, ya = cam.yaw * Math.PI / 180, pa = cam.pitch * Math.PI / 180;
  const rot = (x, y, z) => { const y1 = y * Math.cos(pa) - z * Math.sin(pa), z1 = y * Math.sin(pa) + z * Math.cos(pa);
    return [x * Math.cos(ya) + z1 * Math.sin(ya), y1, -x * Math.sin(ya) + z1 * Math.cos(ya)]; };
  const right = rot(1, 0, 0), up = rot(0, 1, 0), fwd = rot(0, 0, -1), back = rot(0, 0, R * (1 - cam.dolly));
  const c = [0, 1, 2].map(i => [0, 0, -R][i] + back[i] + right[i] * cam.x * R + up[i] * cam.y * R);
  return [c, right, up, fwd];
}
export function project(p, B, W, H) {                 // a world point -> [sx, sy, z] in pixels of a W x H view
  const q = [p[0] - B[0][0], p[1] - B[0][1], p[2] - B[0][2]], dot = v => q[0] * v[0] + q[1] * v[1] + q[2] * v[2];
  const x = dot(B[1]), y = dot(B[2]), z = dot(B[3]), zs = Math.max(z, 1e-6);
  return [(x / (zs * 0.5 * (W / H)) + 1) * 0.5 * W, (1 - y / (zs * 0.5)) * 0.5 * H, z];
}
export function trackWorld(p, aspect, depth) {        // [id, u, v, d] -> its place in the cloud (no motion)
  const di = distance(p[3], depth);
  return [(p[1] - 0.5) * aspect * di, (0.5 - p[2]) * di, -di];
}
// which tracked things are joined: picture space 1000 high, like pointcloud.track_edges (network / tree come from
// kubakub_track.js, the same as kubakub track draw)
export function trackEdges(pts, tl, aspect, delaunayEdges, mstEdges) {
  const P = pts.map(p => [p[1] * 1000 * aspect, p[2] * 1000]);
  if (tl.lines === "network") return delaunayEdges(P);
  if (tl.lines === "tree") return mstEdges(P);
  if (tl.lines === "within") { const r = tl.distance * 1000 * aspect, out = [];
    for (let a = 0; a < P.length; a++) for (let b = a + 1; b < P.length; b++) if (Math.hypot(P[a][0] - P[b][0], P[a][1] - P[b][1]) < r) out.push([a, b]);
    return out; }
  return [];
}
// pictures: point i of picture a goes to the point of picture b with the same brightness rank (ties by index);
// lum = Float32Array of each point's brightness
export function rankPerm(lumA, lumB) {
  const n = lumA.length, ka = new Float64Array(n), kb = new Float64Array(n), S = 4194304;
  for (let i = 0; i < n; i++) { ka[i] = Math.round(lumA[i] * 65535) * S + i; kb[i] = Math.round(lumB[i] * 65535) * S + i; }
  ka.sort(); kb.sort();
  const perm = new Uint32Array(n);
  for (let k = 0; k < n; k++) perm[ka[k] % S] = kb[k] % S;
  return perm;
}
// the fine motion and the song's bands, = pointcloud: bands the depth hears, points per scan line, a blink's clock
export const BANDS = 8, ROW = 6, BLINK = 4.0;
