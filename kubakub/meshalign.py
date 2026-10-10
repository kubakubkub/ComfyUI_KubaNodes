"""
meshalign.py

Put a generated 3D model onto the real one. An image-to-3D model (Trellis, Hunyuan3D ...) comes back somewhere in
space, at some size, often turned; the real model of the building (the template) sits where the projector and the
camera expect it. This finds the move, the turn and the scale that lay the generated model onto the template, and,
when asked, a scale per axis, because a model made from one picture often misreads a length (mostly the depth).

numpy + scipy only, no ComfyUI imports (tests/test_meshalign.py). Works on points sampled from the two surfaces:
  1. both are brought to their own main axes; the 24 ways to match those axes (48 with mirrored ones) are each
     tried with a short fit, the best one is kept (a model lying on its side or facing backwards is found this way)
  2. the fit: nearest points, then the best turn + scale + move for them, again and again (the worst fifth of the
     pairs is left out each time, so a part that one model has and the other has not does not pull)
  3. per axis (optional): a scale and a move per axis of the template, on top

The result is one 4x4 matrix that takes the generated model's coordinates to the template's.

align_front() is the fit for a facade: an image-to-3D model of a facade picture comes back as a whole invented building
(back wall, sides, roof) with whatever stands in front of the wall, while the template is a relief a metre or two
deep. Laying surface on surface then presses the whole building into the relief. So only what is seen from the front
is fitted: the outline first, then the wall; the depth is scaled like width and height, so what stands in front of
the wall keeps its proportions. Measured on a Trellis 2 model of a 30 m facade (2026-10-10): the whole-model fit
squashed the depth 11 times and turned it round; the front fit left the wall 0.2 m off (median).
"""

from __future__ import annotations

import itertools
import math

import numpy as np

SCALES = ("per axis", "one scale", "keep the size")
FITS = ("auto", "the facade front", "the whole model")
FRONT_DENSITY = 10            # the front fit takes this many times the points: only the nearest point of each cell of the
#                               front view is used, and a cell with too few points lets the back wall show through
TEMPLATE_DENSITY = 5          # points on the template per point on the model. Paired points are never exact, and a sparse
#                               template pulls the fit off: on the sample building with a floor read a third too high, 1.4 %
#                               of the size off with as many template points as model points, 0.13 % with five times as many


def sample_pair(model, template, points=40000, seed=0) -> tuple:
    """(points on the model, points on the template) for align(): model and template as (vertices, faces)."""
    return (sample_surface(model[0], model[1], int(points), int(seed)),
            sample_surface(template[0], template[1], TEMPLATE_DENSITY * int(points), int(seed) + 1))


def sample_surface(vertices, faces, n=20000, seed=0) -> np.ndarray:
    """n points spread evenly over a triangle mesh ([V, 3], [F, 3]); a model without faces gives its vertices."""
    v = np.asarray(vertices, np.float64).reshape(-1, 3)
    if faces is None or len(faces) == 0:
        rng = np.random.default_rng(seed)
        return v if len(v) <= n else v[rng.choice(len(v), n, replace=False)]
    f = np.asarray(faces, np.int64).reshape(-1, 3)
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    if not np.isfinite(area).all() or area.sum() <= 0:
        return v
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(f), n, p=area / area.sum())
    r1, r2 = np.sqrt(rng.random(n))[:, None], rng.random(n)[:, None]
    return (1 - r1) * a[pick] + r1 * (1 - r2) * b[pick] + r1 * r2 * c[pick]


def _frame(p):
    """A point set's centre, its main axes as the columns of a rotation, and its size along them."""
    c = p.mean(0)
    w, v = np.linalg.eigh(np.cov((p - c).T))
    v = v[:, np.argsort(w)[::-1]]
    if np.linalg.det(v) < 0:
        v[:, 2] *= -1
    q = (p - c) @ v
    return c, v, np.maximum(q.max(0) - q.min(0), 1e-12)


def _axis_matches(mirror):
    """The ways to lay one set of axes onto another: 24 turns, and 24 more that mirror."""
    out = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1.0, -1.0), repeat=3):
            m = np.zeros((3, 3))
            for i, (j, s) in enumerate(zip(perm, signs)):
                m[i, j] = s
            d = np.linalg.det(m)
            if d > 0 or mirror:
                out.append(m)
    return out


def _similarity(src, dst, scale=True):
    """The turn R, the scale s and the move t with s * R @ src + t closest to dst (Umeyama 1991)."""
    cs, cd = src.mean(0), dst.mean(0)
    a, b = src - cs, dst - cd
    u, d, vt = np.linalg.svd(b.T @ a / len(src))
    sign = np.ones(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        sign[2] = -1
    r = u @ np.diag(sign) @ vt
    s = float((d * sign).sum() / max(a.var(0).sum(), 1e-18)) if scale else 1.0
    return r, s, cd - s * r @ cs


def _matrix(lin, t):
    m = np.eye(4)
    m[:3, :3], m[:3, 3] = lin, t
    return m


def align(source, template, scale="per axis", mirror=False, rounds=40, seed=0) -> dict:
    """
    source, template: points on the two surfaces ([N, 3] each, any units, any place).
    -> {matrix 4x4 (source coordinates -> template coordinates), turn 3x3, scales (3, along the template's x, y, z),
        mirrored, error (mean distance of the source's points to the template after the fit), error_95, size (the
        template's diagonal), before (the same mean distance before the fit), covered (the share of the template
        that has a source point near it)}
    scale   per axis = a scale for each of the template's axes; one scale; keep the size (turn and move only)
    mirror  also try mirrored fits (a generated model can come back mirrored); off = never mirror
    """
    from scipy.spatial import cKDTree
    src, dst = np.asarray(source, np.float64).reshape(-1, 3), np.asarray(template, np.float64).reshape(-1, 3)
    if len(src) < 16 or len(dst) < 16:
        raise ValueError("a model needs at least 16 points to be fitted")
    rng = np.random.default_rng(seed)
    few_s = src[rng.choice(len(src), min(len(src), 1500), replace=False)]
    few_d = dst[rng.choice(len(dst), min(len(dst), 6000), replace=False)]
    tree_few, tree = cKDTree(few_d), cKDTree(dst)
    cs, fs, es = _frame(src)
    cd, fd, ed = _frame(dst)
    size = float(np.linalg.norm(dst.max(0) - dst.min(0)))
    fixed = scale == "keep the size"

    def pairs(pts, tr, back, lin, t, keep=0.8):
        """Who belongs to whom, both ways: every model point with its nearest template point, and every template
        point (of `back`) with its nearest model point. One way alone lets the model shrink into a corner of the
        template, where every one of its points has a neighbour. The farthest fifth of each is left out."""
        moved = pts @ lin.T + t
        d, k = tr.query(moved, workers=-1)
        o = np.argpartition(d, max(8, int(len(d) * keep)) - 1)[:max(8, int(len(d) * keep))]
        d2, k2 = cKDTree(moved).query(back, workers=-1)
        o2 = np.argpartition(d2, max(8, int(len(d2) * keep)) - 1)[:max(8, int(len(d2) * keep))]
        return (np.concatenate([moved[o], moved[k2[o2]]]), np.concatenate([tr.data[k[o]], back[o2]]),
                float(d[o].mean() + d2[o2].mean()) / 2.0)

    def fit(pts, tr, back, lin, t, n, sized=not fixed):
        """n rounds of pair and move -> (lin, t)."""
        for _ in range(n):
            p, q, _e = pairs(pts, tr, back, lin, t)
            r, s, t2 = _similarity(p, q, scale=sized)
            lin, t = s * r @ lin, s * r @ t + t2
        return lin, t

    # 1. which of the template's axes is which of the model's: try them all, briefly
    best = None
    for m in _axis_matches(mirror):
        s0 = 1.0 if fixed else float(np.median(ed / (np.abs(m) @ es)))
        lin = s0 * fd @ m @ fs.T
        t = cd - lin @ cs
        lin, t = fit(few_s, tree_few, few_d[::2], lin, t, 6)
        err = pairs(few_s, tree_few, few_d[::2], lin, t)[2]      # two poses of a building that is nearly the same left
        if best is None or err < best[2]:                        # and right lie close: the error after the last move
            best = (lin, t, err)
    lin, t, _ = best
    before = _apart(src, few_d, tree)

    # 2. the fit itself, on all the points
    lin, t = fit(src, tree, few_d, lin, t, rounds)

    # 3. a scale and a move per axis of the template, the turn stays. Two starts, the closer one wins: from the fit
    # as it is, and from the two models' extents (a building with rows of the same windows fits its neighbour row
    # just as well, so nearest points alone cannot tell that it is a floor too high; the extents can. A model that
    # shows only a part has other extents, and then the first start wins).
    if scale == "per axis":
        def per_axis(lin, t, n=25):
            for _ in range(n):
                p, q, _e = pairs(src, tree, few_d, lin, t)
                pm, qm = p.mean(0), q.mean(0)
                # the ratio of the two spreads, not the slope of one on the other: pairs are never exact, and a
                # slope then comes out too small every round, which shrinks a thin axis (a facade's depth) to nothing
                k = np.clip(np.sqrt(((q - qm) ** 2).sum(0) / np.maximum(((p - pm) ** 2).sum(0), 1e-18)), 0.5, 2.0)
                lin, t = np.diag(k) @ lin, k * (t - pm) + qm
            return lin, t

        def by_pairs(lin, t):
            lin, t = per_axis(lin, t)
            for _ in range(3):                         # the turn was found while a length was still wrong: once more,
                lin, t = fit(src, tree, few_d, lin, t, 8, sized=False)      # turn and move only, then the lengths again
                lin, t = per_axis(lin, t, 8)
            return lin, t, pairs(src, tree, few_d, lin, t)[2]

        lo_d, hi_d = np.percentile(dst, [0.5, 99.5], axis=0)

        def by_extents(lin, t):
            for _ in range(5):
                lo_s, hi_s = np.percentile(src @ lin.T + t, [0.5, 99.5], axis=0)
                k = np.clip((hi_d - lo_d) / np.maximum(hi_s - lo_s, 1e-12), 0.5, 2.0)
                lin, t = np.diag(k) @ lin, k * (t - (lo_s + hi_s) / 2) + (lo_d + hi_d) / 2
                lin, t = fit(src, tree, few_d, lin, t, 8, sized=False)
            return lin, t, pairs(src, tree, few_d, lin, t)[2]

        lin, t, _ = min((by_pairs(lin, t), by_extents(lin, t)), key=lambda c: c[2])
    moved = src @ lin.T + t
    d = tree.query(moved)[0]
    u, _sv, vt = np.linalg.svd(lin)
    turn = u @ np.diag([1.0, 1.0, float(np.sign(np.linalg.det(u @ vt)))]) @ vt
    near = cKDTree(moved).query(dst)[0]
    return {"matrix": _matrix(lin, t), "turn": turn, "scales": np.linalg.norm(lin, axis=1),       # the stretch along the template's x, y, z
            "mirrored": bool(np.linalg.det(lin) < 0), "error": float(d.mean()), "error_95": float(np.percentile(d, 95)),
            "size": size, "before": before, "covered": float((near < 0.02 * size).mean())}


def _apart(src, few_d, tree) -> float:
    """How far two point sets lie apart, both ways. One way alone says 'close' for a small model that sits somewhere
    inside a big template."""
    from scipy.spatial import cKDTree
    few_s = src[::max(1, len(src) // 20000)]
    return float(tree.query(few_s)[0].mean() + cKDTree(few_s).query(few_d)[0].mean()) / 2.0


def looks_like_a_front(source, template) -> bool:
    """True when the template is a thin relief and the generated model is far deeper for its size: a whole building
    made from a picture of a facade. Then the front fit is the right one."""
    def thin(p):
        e = _frame(np.asarray(p, np.float64).reshape(-1, 3))[2]
        return float(e.min() / e.max())
    td, ts = thin(template), thin(source)
    return td < 0.25 and ts > 2.5 * td


def front_layer(p, cells=None) -> np.ndarray:
    """Of points [N, 3], the ones seen from +z: the nearest point of each cell of an x, y grid (about 8 points fall
    into a cell, so the back does not show through between them)."""
    cells = cells or int(np.clip(np.sqrt(len(p) / 8.0), 48, 400))
    lo, hi = p[:, :2].min(0), p[:, :2].max(0)
    step = max(float((hi - lo).max()) / cells, 1e-12)
    ij = np.floor((p[:, :2] - lo) / step).astype(np.int64)
    key = ij[:, 0] * (cells + 2) + ij[:, 1]
    o = np.lexsort((-p[:, 2], key))
    first = np.ones(len(o), bool)
    first[1:] = key[o][1:] != key[o][:-1]
    return p[o[first]]


def _wall_depth(front) -> float:
    """The depth most of a front view lies at: the wall."""
    h, e = np.histogram(front[:, 2], bins=120)
    k = int(np.argmax(h))
    return float((e[k] + e[k + 1]) / 2)


def _front_fit(fs, fd, scale, rounds):
    """The front view of the model (fs) onto the front view of the template (fd), both looking to +z, y up.
    -> (lin, t, mean distance of the pairs, the model's wall points after the fit)"""
    from scipy.spatial import cKDTree
    fixed, one = scale == "keep the size", scale != "per axis"
    # 1. the outline: the extents of the two front views
    lo_s, hi_s = np.percentile(fs[:, :2], [0.5, 99.5], axis=0)
    lo_d, hi_d = np.percentile(fd[:, :2], [0.5, 99.5], axis=0)
    k = (hi_d - lo_d) / np.maximum(hi_s - lo_s, 1e-12)
    if one:
        k[:] = np.sqrt(k[0] * k[1])
    if fixed:
        k[:] = 1.0
    kz = float(np.sqrt(k[0] * k[1]))
    lin = np.diag([k[0], k[1], kz])
    t = np.array([(lo_d[0] + hi_d[0]) / 2 - k[0] * (lo_s[0] + hi_s[0]) / 2,
                  (lo_d[1] + hi_d[1]) / 2 - k[1] * (lo_s[1] + hi_s[1]) / 2,
                  _wall_depth(fd) - kz * _wall_depth(fs)])
    # 2. the wall: the model's front points that lie in the template's depth, paired with the template's front
    zlo, zhi = float(fd[:, 2].min()), float(fd[:, 2].max())
    pad = max(0.25 * (zhi - zlo), 0.01 * float(np.linalg.norm(fd.max(0) - fd.min(0))))
    tree = cKDTree(fd)
    back = fd[np.random.default_rng(0).choice(len(fd), min(len(fd), 20000), replace=False)]

    def wall_of(lin, t):
        moved = fs @ lin.T + t
        return moved[(moved[:, 2] > zlo - pad) & (moved[:, 2] < zhi + pad)]

    err = float("inf")
    for i in range(rounds):
        wall = wall_of(lin, t)
        if len(wall) < 16:
            break
        d, j = tree.query(wall, workers=-1)
        o = np.argsort(d)[:max(8, int(len(d) * 0.8))]
        d2, j2 = cKDTree(wall).query(back, workers=-1)
        o2 = np.argsort(d2)[:max(8, int(len(d2) * 0.6))]     # where something covers the wall the template has no partner
        p, q = np.concatenate([wall[o], wall[j2[o2]]]), np.concatenate([fd[j[o]], back[o2]])
        err = float(d[o].mean() + d2[o2].mean()) / 2.0
        if i % 2 == 0 or fixed:                              # turn and move
            r, _s, t2 = _similarity(p, q, scale=False)
            lin, t = r @ lin, r @ t + t2
        else:                                                # the lengths in x and y; the depth follows their mean
            pm, qm = p.mean(0), q.mean(0)
            kk = np.clip(np.sqrt(((q - qm) ** 2).sum(0) / np.maximum(((p - pm) ** 2).sum(0), 1e-18)), 0.9, 1.1)
            if one:
                kk[:2] = np.sqrt(kk[0] * kk[1])
            kk[2] = np.sqrt(kk[0] * kk[1])
            lin, t = np.diag(kk) @ lin, kk * (t - pm) + qm
    return lin, t, err, wall_of(lin, t)


def align_front(source, template, scale="per axis", rounds=40, seed=0) -> dict:
    """
    The fit for a facade (see the top of this file). source: points on the generated model, upright (y up) and seen
    from +z, as image to 3D writes it. template: points on the real facade, y up, facing any way.
    -> the same as align(), and "front": {normal (the way the facade faces, in the template's coordinates), back (the
       depth of the template's back along it, a little behind), wall (the share of the model's front that is wall),
       ahead, behind (how far the model reaches in front of and behind the template)}
    scale   per axis = a scale for the width and one for the height; one scale; keep the size. The depth always
            takes the mean of the two.
    """
    from scipy.spatial import cKDTree
    src, dst = np.asarray(source, np.float64).reshape(-1, 3), np.asarray(template, np.float64).reshape(-1, 3)
    if len(src) < 16 or len(dst) < 16:
        raise ValueError("a model needs at least 16 points to be fitted")
    size = float(np.linalg.norm(dst.max(0) - dst.min(0)))
    w, v = np.linalg.eigh(np.cov((dst[:, [0, 2]] - dst[:, [0, 2]].mean(0)).T))
    n = v[:, int(np.argmin(w))]                               # the facade's thin direction on the ground
    if n[1] < 0 or (abs(n[1]) < 1e-9 and n[0] < 0):           # towards +z first (Blender's front view), then +x
        n = -n
    fs = front_layer(src)
    tries = []
    for side in (1.0, -1.0):                                  # the template's two faces: the closer fit is its front
        nx, nz = side * n
        turn = np.array([[nz, 0.0, -nx], [0.0, 1.0, 0.0], [nx, 0.0, nz]])
        fd = front_layer(dst @ turn.T)
        tries.append((_front_fit(fs, fd, scale, 10)[2], turn, fd))
    k = 1 if tries[1][0] < 0.95 * tries[0][0] else 0          # as good as equal (a flat template): the first face
    turn, fd = tries[k][1], tries[k][2]
    lin, t, _err, wall = _front_fit(fs, fd, scale, rounds)

    d = cKDTree(fd).query(wall)[0] if len(wall) else np.array([size])
    near = cKDTree(wall).query(fd)[0] if len(wall) else np.full(len(fd), size)
    z = (src @ lin.T + t)[:, 2]
    world, move = turn.T @ lin, turn.T @ t
    u, _sv, vt = np.linalg.svd(world)
    few_d = dst[np.random.default_rng(seed).choice(len(dst), min(len(dst), 6000), replace=False)]
    zlo, zhi = float(fd[:, 2].min()), float(fd[:, 2].max())
    return {"matrix": _matrix(world, move), "turn": u @ np.diag([1.0, 1.0, float(np.sign(np.linalg.det(u @ vt)))]) @ vt,
            "scales": np.linalg.norm(lin, axis=1), "mirrored": False, "error": float(d.mean()),
            "error_95": float(np.percentile(d, 95)), "size": size, "before": _apart(src, few_d, cKDTree(dst)),
            "covered": float((near < 0.02 * size).mean()),
            "front": {"normal": turn.T @ np.array([0.0, 0.0, 1.0]), "back": zlo - 0.1 * max(zhi - zlo, 0.01 * size),
                      "wall": len(wall) / max(len(fs), 1), "ahead": max(0.0, float(np.percentile(z, 99.9)) - zhi),
                      "behind": max(0.0, zlo - float(np.percentile(z, 0.1)))}}


def front_faces(vertices, faces, front) -> np.ndarray:
    """Which faces of an aligned model ([V, 3] in the template's coordinates, [F, 3]) are kept when the invented back
    is cut away: the ones not behind the template's back."""
    v, f = np.asarray(vertices, np.float64).reshape(-1, 3), np.asarray(faces, np.int64).reshape(-1, 3)
    depth = v @ np.asarray(front["normal"], np.float64)
    return depth[f].mean(1) >= float(front["back"])


def apply(matrix, points) -> np.ndarray:
    """Points [N, 3] through a 4x4 matrix."""
    p = np.asarray(points, np.float64).reshape(-1, 3)
    return p @ matrix[:3, :3].T + matrix[:3, 3]


def turn_degrees(turn) -> float:
    """The angle of a rotation matrix."""
    return math.degrees(math.acos(min(1.0, max(-1.0, (np.trace(turn) - 1.0) / 2.0))))


def describe(res, unit="units") -> list:
    """The fit in sentences: how far off it was, what was done, how close it is now, and what looks misread."""
    size = res["size"]
    s = res["scales"]
    front = res.get("front")
    lines = [f"before: on average {res['before']:.3g} {unit} off ({res['before'] / size * 100:.1f} % of the template's size)",
             f"after: on average {res['error']:.3g} {unit} off ({res['error'] / size * 100:.2f} %), 95 % of the {'wall' if front else 'surface'} within "
             f"{res['error_95']:.3g} {unit}; {res['covered'] * 100:.0f} % of the template is covered",
             f"turned {turn_degrees(res['turn']):.1f}°, scaled x {s[0]:.4g}, y {s[1]:.4g}, z {s[2]:.4g}"
             + (", mirrored" if res["mirrored"] else "")]
    if front:
        lines.insert(0, "fitted on the facade front (what is seen from in front of it): the outline, then the wall; "
                        "the depth is scaled like width and height")
        lines.append(f"{front['wall'] * 100:.0f} % of the model's front is wall; the model reaches {front['ahead']:.3g} {unit} "
                     f"in front of the template and {front['behind']:.3g} {unit} behind it")
    m = float(np.median(s))                           # two axes that agree are the size; the third is the misread one
    off = [(a, k / m) for a, k in zip("xyz", s) if abs(k / m - 1.0) > 0.03]
    if off:
        lines.append("the generated model had misread a length: " + ", ".join(
            f"{a} was {abs(1 / r - 1) * 100:.0f} % too {'long' if r < 1 else 'short'}" for a, r in off) + " (corrected)")
    if not front and float(np.max(s) / max(float(np.min(s)), 1e-12)) > 2.0:
        lines.append("a length changed by more than double is rarely a misreading: the generated model is probably a whole "
                     "building and the template only its facade. Set fit to 'the facade front'.")
    if res["error"] / size > 0.02:
        lines.append("this is not a close fit (more than 2 % of the size off on average): the two models differ in shape, "
                     "or the generated one shows only a part. Check it in the viewer before you use it.")
    if res["covered"] < 0.6:
        lines.append("less than 60 % of the template has the generated model near it: it may show one side only "
                     "(image to 3D sees the front), which is fine for a facade")
    return lines
