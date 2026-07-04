"""
arc_dsl.py — Milestone 1 of provable multi-step reasoning on ARC-AGI.

Mechanism B: program synthesis over a grid DSL.
  induce/search a program from the demonstration pairs
    -> VERIFY it reproduces EVERY demonstration output exactly
    -> apply to the test input -> serialized string answer
  No demo-consistent program found -> ABSTAIN (never guess → no confabulation).

This is Uchi-native multi-step reasoning: the "steps" are composed DSL ops, the
search is the deliberation, the verifier is the oracle. Each task is solved
independently by inducing a program from its OWN examples — generalization is
whether a demo-verified program predicts the held-out TEST output.

Milestone-1 DSL (extensible): geometric ops, induced global color-map, induced
tiling / scaling, crop-to-content, induced single-color replace. Compositions
up to depth 2, plus geometric∘colormap.

Usage:
    .venv/bin/python experiments/arc_dsl.py --split evaluation
    .venv/bin/python experiments/arc_dsl.py --split training --show 5
"""
from __future__ import annotations
import argparse, glob, json, os, itertools, time
import numpy as np

_DATA = os.path.join(os.path.dirname(__file__), "..",
                     "..", "..", "..", "..", "tmp", "claude-1000")  # placeholder; set via --data

# ── grid helpers ──────────────────────────────────────────────────────────────
def to_grid(g):
    return np.array(g, dtype=np.int8)

def eq(a, b):
    return a.shape == b.shape and bool(np.array_equal(a, b))

def serialize(g):
    return "\n".join(" ".join(str(int(c)) for c in row) for row in g)

# ── unary geometric ops (grid -> grid) ────────────────────────────────────────
GEO = {
    "identity": lambda g: g,
    "rot90":    lambda g: np.rot90(g, 1),
    "rot180":   lambda g: np.rot90(g, 2),
    "rot270":   lambda g: np.rot90(g, 3),
    "flip_h":   lambda g: np.fliplr(g),
    "flip_v":   lambda g: np.flipud(g),
    "transpose":lambda g: g.T,
}

# ── parametric inducers: (demos) -> callable|None (induced from examples) ──────
def induce_color_map(demos):
    """Global cell-wise color mapping consistent across all demos (same shape)."""
    mapping = {}
    for inp, out in demos:
        if inp.shape != out.shape:
            return None
        for a, b in zip(inp.flatten(), out.flatten()):
            a, b = int(a), int(b)
            if a in mapping and mapping[a] != b:
                return None
            mapping[a] = b
    return lambda g: np.vectorize(lambda x: mapping.get(int(x), int(x)))(g).astype(np.int8)

def induce_replace(demos):
    """Single color a->b (positions otherwise identical)."""
    a_set = set()
    for inp, out in demos:
        if inp.shape != out.shape:
            return None
        diff = inp != out
        if diff.any():
            a_set.update(int(x) for x in np.unique(inp[diff]))
    if len(a_set) != 1:
        return None
    return induce_color_map(demos)  # falls back to full map (still verified)

def induce_tile(demos):
    """output = input tiled ky×kx (plain repeat)."""
    inp, out = demos[0]
    ih, iw = inp.shape; oh, ow = out.shape
    if ih == 0 or iw == 0 or oh % ih or ow % iw:
        return None
    ky, kx = oh // ih, ow // iw
    if (ky, kx) == (1, 1):
        return None
    return lambda g: np.tile(g, (ky, kx)).astype(np.int8)

def induce_mirror_tile(demos):
    """output = 2x2 block of {g, flip_h(g); flip_v(g), rot180(g)} (common ARC pattern)."""
    def f(g):
        top = np.hstack([g, np.fliplr(g)])
        bot = np.hstack([np.flipud(g), np.rot90(g, 2)])
        return np.vstack([top, bot]).astype(np.int8)
    return f

def induce_scale(demos):
    """each cell -> ky×kx block."""
    inp, out = demos[0]
    ih, iw = inp.shape; oh, ow = out.shape
    if ih == 0 or iw == 0 or oh % ih or ow % iw:
        return None
    ky, kx = oh // ih, ow // iw
    if (ky, kx) == (1, 1):
        return None
    return lambda g: np.repeat(np.repeat(g, ky, axis=0), kx, axis=1).astype(np.int8)

def induce_crop_content(demos):
    """crop to bounding box of non-zero cells."""
    def f(g):
        nz = np.argwhere(g != 0)
        if nz.size == 0:
            return g
        r0, c0 = nz.min(0); r1, c1 = nz.max(0) + 1
        return g[r0:r1, c0:c1].astype(np.int8)
    return f

def induce_gravity(demos):
    """Non-zero cells fall to the bottom of each column."""
    def f(g):
        out = np.zeros_like(g); H, W = g.shape
        for j in range(W):
            col = [v for v in g[:, j] if v != 0]
            if col:
                out[H - len(col):, j] = col
        return out.astype(np.int8)
    return f

def induce_flood_fill(demos):
    """Fill background regions NOT connected to the border with an induced color."""
    from collections import deque
    inp0, out0 = demos[0]
    if inp0.shape != out0.shape:
        return None
    changed = inp0 != out0
    if not changed.any():
        return None
    frm = set(int(x) for x in np.unique(inp0[changed]))
    to = set(int(x) for x in np.unique(out0[changed]))
    if len(to) != 1 or len(frm) != 1:
        return None
    bg, C = frm.pop(), to.pop()
    def f(g):
        g = g.copy(); H, W = g.shape
        reach = np.zeros_like(g, dtype=bool); dq = deque()
        for i in range(H):
            for j in (0, W - 1):
                if g[i, j] == bg and not reach[i, j]:
                    reach[i, j] = True; dq.append((i, j))
        for j in range(W):
            for i in (0, H - 1):
                if g[i, j] == bg and not reach[i, j]:
                    reach[i, j] = True; dq.append((i, j))
        while dq:
            i, j = dq.popleft()
            for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ni, nj = i + di, j + dj
                if 0 <= ni < H and 0 <= nj < W and g[ni, nj] == bg and not reach[ni, nj]:
                    reach[ni, nj] = True; dq.append((ni, nj))
        g[(g == bg) & (~reach)] = C
        return g.astype(np.int8)
    return f

def induce_symmetry_complete(demos):
    """Fill a 'hole' color by completing a reflective symmetry."""
    inp0, out0 = demos[0]
    if inp0.shape != out0.shape:
        return None
    changed = inp0 != out0
    if not changed.any():
        return None
    hv = set(int(x) for x in np.unique(inp0[changed]))
    if len(hv) != 1:
        return None
    H = hv.pop()
    syms = [np.fliplr, np.flipud, lambda x: np.rot90(x, 2)]
    if inp0.shape[0] == inp0.shape[1]:
        syms.append(lambda x: x.T)
    def make(sym):
        def f(g):
            g = g.copy(); r = sym(g)
            m = (g == H) & (r != H); g[m] = r[m]
            return g.astype(np.int8)
        return f
    for sym in syms:
        f = make(sym)
        if all(eq(f(i), o) for i, o in demos):
            return f
    return None

def induce_split_logic(demos):
    """Split grid into two halves; cellwise boolean op → single induced color."""
    def split(g):
        H, W = g.shape; res = []
        if W % 2 == 0: res.append(("v", g[:, :W // 2], g[:, W // 2:]))
        if W % 2 == 1 and W > 1: res.append(("vs", g[:, :W // 2], g[:, W // 2 + 1:]))
        if H % 2 == 0: res.append(("h", g[:H // 2, :], g[H // 2:, :]))
        if H % 2 == 1 and H > 1: res.append(("hs", g[:H // 2, :], g[H // 2 + 1:, :]))
        return res
    ops = {"and": lambda a, b: a & b, "or": lambda a, b: a | b,
           "xor": lambda a, b: a ^ b, "nor": lambda a, b: ~(a | b),
           "nxor": lambda a, b: ~(a ^ b), "diff": lambda a, b: a & ~b}
    inp0, out0 = demos[0]
    for mode, A, B in split(inp0):
        if A.shape != B.shape or A.shape != out0.shape:
            continue
        a, b = (A != 0), (B != 0)
        for opn, fn in ops.items():
            r = fn(a, b)
            onv = set(int(x) for x in np.unique(out0[r])) if r.any() else set()
            offv = set(int(x) for x in np.unique(out0[~r])) if (~r).any() else set()
            if len(onv) == 1 and offv <= {0}:
                C = onv.pop()
                def make(mode, fn, C):
                    def f(g):
                        for m, A, B in split(g):
                            if m == mode and A.shape == B.shape:
                                r = fn(A != 0, B != 0)
                                o = np.zeros_like(A); o[r] = C
                                return o.astype(np.int8)
                        return g
                    return f
                prog = make(mode, fn, C)
                if all(eq(prog(i), o) for i, o in demos):
                    return prog
    return None

def induce_most_common_color(demos):
    """Output is a 1x1 grid of the most / least common non-zero color."""
    def freq(g, most):
        vals, cnts = np.unique(g[g != 0], return_counts=True)
        if len(vals) == 0:
            return None
        return int(vals[cnts.argmax() if most else cnts.argmin()])
    for most in (True, False):
        def f(g, most=most):
            c = freq(g, most)
            return np.array([[c if c is not None else 0]], dtype=np.int8)
        if all(eq(f(i), o) for i, o in demos):
            return f
    return None

INDUCERS = [induce_color_map, induce_replace, induce_tile, induce_mirror_tile,
            induce_scale, induce_flood_fill,
            induce_symmetry_complete, induce_split_logic, induce_most_common_color]

# ── object (connected-component) utilities + param-free ops ────────────────────
def _components(g, bg=0, diag=True):
    from collections import deque
    H, W = g.shape
    seen = np.zeros((H, W), bool); comps = []
    nb = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if diag:
        nb += [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    for i in range(H):
        for j in range(W):
            if g[i, j] != bg and not seen[i, j]:
                q = deque([(i, j)]); seen[i, j] = True; cells = [(i, j)]
                while q:
                    y, x = q.popleft()
                    for dy, dx in nb:
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < H and 0 <= nx < W and g[ny, nx] != bg and not seen[ny, nx]:
                            seen[ny, nx] = True; q.append((ny, nx)); cells.append((ny, nx))
                comps.append(cells)
    return comps

def _crop_cells(g, cells):
    ys = [c[0] for c in cells]; xs = [c[1] for c in cells]
    return g[min(ys):max(ys) + 1, min(xs):max(xs) + 1]

def op_crop_content(g):
    nz = np.argwhere(g != 0)
    if nz.size == 0:
        return g
    r0, c0 = nz.min(0); r1, c1 = nz.max(0) + 1
    return g[r0:r1, c0:c1].astype(np.int8)

def op_gravity(g):
    out = np.zeros_like(g); H, W = g.shape
    for j in range(W):
        col = [v for v in g[:, j] if v != 0]
        if col:
            out[H - len(col):, j] = col
    return out.astype(np.int8)

def _by_size(g, largest=True):
    comps = _components(g)
    return max(comps, key=len) if largest else min(comps, key=len) if comps else None

def op_crop_largest(g):
    c = _by_size(g, True)
    return _crop_cells(g, c).astype(np.int8) if c else g

def op_crop_smallest(g):
    c = _by_size(g, False)
    return _crop_cells(g, c).astype(np.int8) if c else g

def op_keep_largest(g):
    c = _by_size(g, True)
    if not c:
        return g
    m = np.zeros_like(g, bool)
    for y, x in c:
        m[y, x] = True
    return np.where(m, g, 0).astype(np.int8)

def op_denoise(g):
    out = g.copy()
    for c in _components(g):
        if len(c) == 1:
            y, x = c[0]; out[y, x] = 0
    return out.astype(np.int8)

def _shape_key(cells):
    ys = [c[0] for c in cells]; xs = [c[1] for c in cells]
    return frozenset((y - min(ys), x - min(xs)) for y, x in cells)

def op_crop_unique(g):
    comps = _components(g)
    if len(comps) < 2:
        return g
    from collections import Counter
    keys = [_shape_key(c) for c in comps]
    cnt = Counter(keys)
    uniq = [c for c, k in zip(comps, keys) if cnt[k] == 1]
    return _crop_cells(g, uniq[0]).astype(np.int8) if len(uniq) == 1 else g

PARAM_FREE = {**GEO, "gravity": op_gravity, "crop_content": op_crop_content,
              "crop_largest": op_crop_largest, "crop_smallest": op_crop_smallest,
              "keep_largest": op_keep_largest, "denoise": op_denoise,
              "crop_unique": op_crop_unique}

def compose(fns):
    def f(g):
        for fn in fns:
            g = fn(g)
        return g
    return f

# ── search ────────────────────────────────────────────────────────────────────
def verifies(prog, demos):
    try:
        return all(eq(prog(inp), out) for inp, out in demos)
    except Exception:
        return False

def induce_recolor_by_size(demos):
    """Recolor each object by its cell-count (induced size→color map)."""
    inp0, out0 = demos[0]
    if inp0.shape != out0.shape:
        return None
    size2col = {}
    for inp, out in demos:
        if inp.shape != out.shape:
            return None
        for cells in _components(inp):
            cols = set(int(out[y, x]) for y, x in cells)
            if len(cols) != 1:
                return None
            s, c = len(cells), cols.pop()
            if s in size2col and size2col[s] != c:
                return None
            size2col[s] = c
    def f(g):
        out = g.copy()
        for cells in _components(g):
            c = size2col.get(len(cells))
            if c is not None:
                for y, x in cells:
                    out[y, x] = c
        return out.astype(np.int8)
    return f

def _objects_meta(g):
    from collections import Counter
    metas = []
    for cells in _components(g):
        colors = [int(g[y, x]) for y, x in cells]
        metas.append({"cells": cells, "size": len(cells),
                      "color": Counter(colors).most_common(1)[0][0],
                      "shape": _shape_key(cells)})
    return metas

def _render_obj(g, obj, mode):
    if obj is None:
        return g
    if mode == "crop":
        return _crop_cells(g, obj["cells"]).astype(np.int8)
    out = np.zeros_like(g)
    for y, x in obj["cells"]:
        out[y, x] = g[y, x]
    return out.astype(np.int8)

def induce_object_select(demos):
    """Select THE object by a fixed criterion; output cropped or kept."""
    from collections import Counter
    def uniq(ms, key):
        c = Counter(m[key] for m in ms); u = [m for m in ms if c[m[key]] == 1]
        return u[0] if len(u) == 1 else None
    def common(ms, key):
        c = Counter(m[key] for m in ms); top = c.most_common(1)[0][0]
        cand = [m for m in ms if m[key] == top]
        return cand[0] if cand else None
    picks = [
        lambda ms: max(ms, key=lambda m: m["size"]),
        lambda ms: min(ms, key=lambda m: m["size"]),
        lambda ms: uniq(ms, "shape"), lambda ms: uniq(ms, "color"),
        lambda ms: common(ms, "shape"), lambda ms: common(ms, "color"),
    ]
    for pick in picks:
        for mode in ("crop", "keep"):
            def f(g, pick=pick, mode=mode):
                try:
                    ms = _objects_meta(g)
                    return _render_obj(g, pick(ms) if ms else None, mode)
                except Exception:
                    return g
            if all(eq(f(i), o) for i, o in demos):
                return f
    return None

def induce_recolor_by_rank(demos):
    """Objects recoloured by size RANK (generalises better than raw size)."""
    rank2col = {}
    for inp, out in demos:
        if inp.shape != out.shape:
            return None
        for r, m in enumerate(sorted(_objects_meta(inp), key=lambda m: -m["size"])):
            cols = set(int(out[y, x]) for y, x in m["cells"])
            if len(cols) != 1:
                return None
            c = cols.pop()
            if r in rank2col and rank2col[r] != c:
                return None
            rank2col[r] = c
    def f(g):
        out = g.copy()
        for r, m in enumerate(sorted(_objects_meta(g), key=lambda m: -m["size"])):
            if r in rank2col:
                for y, x in m["cells"]:
                    out[y, x] = rank2col[r]
        return out.astype(np.int8)
    return f

def induce_translate(demos):
    """Rigid translation inferred from the content bounding box."""
    def tl(g):
        nz = np.argwhere(g != 0)
        return None if nz.size == 0 else nz.min(0)
    inp0, out0 = demos[0]
    if inp0.shape != out0.shape:
        return None
    a, b = tl(inp0), tl(out0)
    if a is None or b is None:
        return None
    dy, dx = int(b[0] - a[0]), int(b[1] - a[1])
    if (dy, dx) == (0, 0):
        return None
    def f(g):
        h, w = g.shape; out = np.zeros_like(g)
        ys, xs = np.nonzero(g); ny, nx = ys + dy, xs + dx
        v = (ny >= 0) & (ny < h) & (nx >= 0) & (nx < w)
        out[ny[v], nx[v]] = g[ys[v], xs[v]]
        return out.astype(np.int8)
    return f

def induce_downscale(demos):
    """Block-collapse: output is input downscaled by an integer factor."""
    inp0, out0 = demos[0]
    ih, iw = inp0.shape; oh, ow = out0.shape
    if oh == 0 or ow == 0 or ih % oh or iw % ow:
        return None
    ky, kx = ih // oh, iw // ow
    if (ky, kx) == (1, 1):
        return None
    def f(g):
        h, w = g.shape
        return g[::ky, ::kx].astype(np.int8) if (h % ky == 0 and w % kx == 0) else g
    return f

def op_remove_separators(g):
    """Drop fully-constant rows/cols (grid separators)."""
    H, W = g.shape
    kr = [i for i in range(H) if len(set(g[i, :].tolist())) != 1]
    kc = [j for j in range(W) if len(set(g[:, j].tolist())) != 1]
    if (len(kr) == H and len(kc) == W) or not kr or not kc:
        return g
    return g[np.ix_(kr, kc)].astype(np.int8)

def op_trim_border(g):
    """Strip a uniform outer border repeatedly."""
    out = g
    while out.shape[0] > 2 and out.shape[1] > 2:
        b = np.concatenate([out[0, :], out[-1, :], out[:, 0], out[:, -1]])
        if len(set(b.tolist())) == 1:
            out = out[1:-1, 1:-1]
        else:
            break
    return out.astype(np.int8)

PARAM_FREE.update({"remove_sep": op_remove_separators, "trim_border": op_trim_border})

ALL_INDUCED = INDUCERS + [induce_recolor_by_size, induce_object_select,
                          induce_recolor_by_rank, induce_translate, induce_downscale]

# ── extended pure-function library + behaviour-BFS search ─────────────────────
def _grav_dir(d):
    def f(g):
        if d == "down":  return op_gravity(g)
        if d == "up":    return np.flipud(op_gravity(np.flipud(g))).astype(np.int8)
        if d == "left":  return op_gravity(g.T).T.astype(np.int8)
        return np.flipud(op_gravity(np.flipud(g.T))).T.astype(np.int8)  # right
    return f

def _keep_color(c):
    return lambda g: np.where(g == c, g, 0).astype(np.int8)
def _del_color(c):
    return lambda g: np.where(g == c, 0, g).astype(np.int8)

def _half(which):
    def f(g):
        H, W = g.shape
        if which == "top" and H > 1:   return g[:H // 2, :].astype(np.int8)
        if which == "bot" and H > 1:   return g[(H + 1) // 2:, :].astype(np.int8)
        if which == "left" and W > 1:  return g[:, :W // 2].astype(np.int8)
        if which == "right" and W > 1: return g[:, (W + 1) // 2:].astype(np.int8)
        return g
    return f

def _sym(axis):
    def f(g):
        r = np.fliplr(g) if axis == "h" else np.flipud(g)
        out = g.copy(); m = (out == 0) & (r != 0); out[m] = r[m]
        return out.astype(np.int8)
    return f

FUNCS = dict(PARAM_FREE)
FUNCS.update({f"grav_{d}": _grav_dir(d) for d in ("down", "up", "left", "right")})
FUNCS.update({f"keep{c}": _keep_color(c) for c in range(1, 10)})
FUNCS.update({f"del{c}": _del_color(c) for c in range(1, 10)})
FUNCS.update({
    "hcat_mir": lambda g: np.hstack([g, np.fliplr(g)]).astype(np.int8),
    "vcat_mir": lambda g: np.vstack([g, np.flipud(g)]).astype(np.int8),
    "hcat": lambda g: np.hstack([g, g]).astype(np.int8),
    "vcat": lambda g: np.vstack([g, g]).astype(np.int8),
    "top_half": _half("top"), "bot_half": _half("bot"),
    "left_half": _half("left"), "right_half": _half("right"),
    "sym_h": _sym("h"), "sym_v": _sym("v"),
})

def op_fill_bbox(g):
    from collections import Counter
    out = g.copy()
    for cells in _components(g):
        ys = [c[0] for c in cells]; xs = [c[1] for c in cells]
        col = Counter(int(g[y, x]) for y, x in cells).most_common(1)[0][0]
        out[min(ys):max(ys) + 1, min(xs):max(xs) + 1] = col
    return out.astype(np.int8)

def op_fill_holes(g):
    from collections import deque, Counter
    H, W = g.shape; out = g.copy()
    reach = np.zeros((H, W), bool); dq = deque()
    for i in range(H):
        for j in (0, W - 1):
            if g[i, j] == 0 and not reach[i, j]:
                reach[i, j] = True; dq.append((i, j))
    for j in range(W):
        for i in (0, H - 1):
            if g[i, j] == 0 and not reach[i, j]:
                reach[i, j] = True; dq.append((i, j))
    while dq:
        i, j = dq.popleft()
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ni, nj = i + di, j + dj
            if 0 <= ni < H and 0 <= nj < W and g[ni, nj] == 0 and not reach[ni, nj]:
                reach[ni, nj] = True; dq.append((ni, nj))
    nz = g[g != 0]
    if nz.size:
        out[(g == 0) & (~reach)] = Counter(int(x) for x in nz).most_common(1)[0][0]
    return out.astype(np.int8)

def _keep_freq(most):
    def f(g):
        from collections import Counter
        nz = g[g != 0]
        if nz.size == 0:
            return g
        cc = Counter(int(x) for x in nz)
        c = cc.most_common(1)[0][0] if most else cc.most_common()[-1][0]
        return np.where(g == c, g, 0).astype(np.int8)
    return f

def op_swap_top2(g):
    from collections import Counter
    nz = g[g != 0]
    cc = Counter(int(x) for x in nz).most_common() if nz.size else []
    if len(cc) < 2:
        return g
    a, b = cc[0][0], cc[1][0]; out = g.copy(); out[g == a] = b; out[g == b] = a
    return out.astype(np.int8)

def op_dedup_rows(g):
    rows = [g[0]]
    for i in range(1, g.shape[0]):
        if not np.array_equal(g[i], g[i - 1]):
            rows.append(g[i])
    return np.array(rows, dtype=np.int8)
def op_dedup_cols(g):
    return op_dedup_rows(g.T).T.astype(np.int8)

def _quad(which):
    def f(g):
        H, W = g.shape; h, w = H // 2, W // 2
        if h == 0 or w == 0:
            return g
        return {"tl": g[:h, :w], "tr": g[:h, W - w:],
                "bl": g[H - h:, :w], "br": g[H - h:, W - w:]}[which].astype(np.int8)
    return f

FUNCS.update({
    "fill_bbox": op_fill_bbox, "fill_holes": op_fill_holes,
    "keep_top": _keep_freq(True), "keep_bot": _keep_freq(False),
    "swap_top2": op_swap_top2, "dedup_rows": op_dedup_rows, "dedup_cols": op_dedup_cols,
    "quad_tl": _quad("tl"), "quad_tr": _quad("tr"),
    "quad_bl": _quad("bl"), "quad_br": _quad("br"),
    "scale2": lambda g: np.repeat(np.repeat(g, 2, 0), 2, 1).astype(np.int8),
    "anti_transpose": lambda g: np.rot90(g.T, 2).astype(np.int8),
})

def _overlay(axis):
    def f(g):
        H, W = g.shape
        if axis == "h":
            if W % 2:
                return g
            a, b = g[:, :W // 2], g[:, W // 2:]
        else:
            if H % 2:
                return g
            a, b = g[:H // 2, :], g[H // 2:, :]
        return np.where(a != 0, a, b).astype(np.int8)
    return f

def _connect(axis):
    def f(g):
        out = g.copy(); H, W = g.shape
        lines = range(H) if axis == "h" else range(W)
        for i in lines:
            vec = g[i, :] if axis == "h" else g[:, i]
            idx = [k for k in range(len(vec)) if vec[k] != 0]
            for a, b in zip(idx, idx[1:]):
                if vec[a] == vec[b]:
                    if axis == "h":
                        out[i, a:b + 1] = vec[a]
                    else:
                        out[a:b + 1, i] = vec[a]
        return out.astype(np.int8)
    return f

def _ray(d):
    def f(g):
        out = g.copy(); H, W = g.shape
        rng = range(H) if d in ("down", "up") else range(W)
        for j in (range(W) if d in ("down", "up") else range(H)):
            line = list(rng) if d in ("down", "right") else list(reversed(rng))
            cur = 0
            for i in line:
                y, x = (i, j) if d in ("down", "up") else (j, i)
                if g[y, x] != 0:
                    cur = g[y, x]
                elif cur:
                    out[y, x] = cur
        return out.astype(np.int8)
    return f

def _keep_objs(border):
    def f(g):
        H, W = g.shape; out = np.zeros_like(g)
        for cells in _components(g):
            touch = any(y in (0, H - 1) or x in (0, W - 1) for y, x in cells)
            if touch == border:
                for y, x in cells:
                    out[y, x] = g[y, x]
        return out.astype(np.int8)
    return f

def op_sym_rot(g):
    r = np.rot90(g, 2); out = g.copy(); m = (out == 0) & (r != 0); out[m] = r[m]
    return out.astype(np.int8)

FUNCS.update({
    "overlay_h": _overlay("h"), "overlay_v": _overlay("v"),
    "connect_h": _connect("h"), "connect_v": _connect("v"),
    "ray_down": _ray("down"), "ray_up": _ray("up"),
    "ray_left": _ray("left"), "ray_right": _ray("right"),
    "keep_border": _keep_objs(True), "keep_interior": _keep_objs(False),
    "sym_rot": op_sym_rot,
})

def _apply(f, g):
    try:
        r = f(g)
        if (isinstance(r, np.ndarray) and r.ndim == 2 and r.size > 0
                and r.shape[0] <= 40 and r.shape[1] <= 40):   # bound runaway growth
            return r
        return None
    except Exception:
        return None

def _key(grids):
    return tuple(g.tobytes() + str(g.shape).encode() for g in grids)

def bfs_search(demos, max_depth=3, cap=1500):
    """Behaviour-BFS: compose pure functions, dedup states by their tuple of
    outputs across ALL demos. A program is a solution when that tuple equals the
    demo outputs. Verified on every demo jointly by construction."""
    inputs = tuple(i for i, _ in demos)
    tgt = _key(tuple(o for _, o in demos))
    frontier = {_key(inputs): (inputs, [])}
    names = list(FUNCS)
    for _ in range(max_depth):
        nxt = {}
        for grids, prog in list(frontier.values()):
            for nm in names:
                f = FUNCS[nm]
                out = []
                ok = True
                for g in grids:
                    r = _apply(f, g)
                    if r is None:
                        ok = False; break
                    out.append(r)
                if not ok:
                    continue
                nk = _key(out)
                if nk == tgt:
                    return prog + [nm]
                if nk not in nxt and nk not in frontier:
                    nxt[nk] = (tuple(out), prog + [nm])
            if len(nxt) > cap * 6:
                break
        if len(nxt) > cap:
            nxt = dict(list(nxt.items())[:cap])
        frontier = nxt
        if not frontier:
            break
    return None


def _state_score(grids, targets):
    """Heuristic: how close the state's grids are to the demo targets (higher=better)."""
    s = 0.0
    for g, t in zip(grids, targets):
        if g.shape == t.shape:
            s += float((g == t).mean())
        else:
            # partial credit toward the right size; penalise being far off
            gh, gw = g.shape; th, tw = t.shape
            s += -0.3 - 0.2 * (abs(gh - th) + abs(gw - tw)) / (th + tw + 1)
    return s / max(len(grids), 1)

def beam_search(demos, max_depth=6, beam=400, time_budget=2.5):
    """Heuristic beam over the pure-function library — narrow frontier, goes deep.
    Scores partial states by similarity to demo targets; keeps top-`beam`.
    Abstains (returns None) if the per-task time budget is exceeded."""
    t0 = time.perf_counter()
    inputs = tuple(i for i, _ in demos)
    targets = tuple(o for _, o in demos)
    tgt = _key(targets)
    beamset = {_key(inputs): (inputs, [])}
    seen = set(beamset)
    names = list(FUNCS)
    for _ in range(max_depth):
        cand = {}
        for grids, prog in beamset.values():
            if time.perf_counter() - t0 > time_budget:
                return None
            for nm in names:
                f = FUNCS[nm]
                out = []
                ok = True
                for g in grids:
                    r = _apply(f, g)
                    if r is None:
                        ok = False; break
                    out.append(r)
                if not ok:
                    continue
                nk = _key(out)
                if nk == tgt:
                    return prog + [nm]
                if nk not in seen and nk not in cand:
                    cand[nk] = (tuple(out), prog + [nm])
        if not cand:
            break
        top = sorted(cand.items(),
                     key=lambda kv: _state_score(kv[1][0], targets),
                     reverse=True)[:beam]
        beamset = dict(top)
        seen.update(beamset.keys())
    return None


def loo_ok(ind, demos):
    """Leave-one-out: induce on all-but-one demo, must predict the held-out one.
    Rejects parameter-fits that memorise the demos instead of the rule."""
    if len(demos) < 2:
        return True
    for i in range(len(demos)):
        sub = demos[:i] + demos[i + 1:]
        try:
            p = ind(sub)
        except Exception:
            p = None
        if p is None or not verifies(p, [demos[i]]):
            return False
    return True

def search(demos, max_depth=3):
    """Compositional program search; shortest verified program wins, else None."""
    # 1. induced terminal solvers — must pass leave-one-out generalisation
    for ind in ALL_INDUCED:
        try:
            p = ind(demos)
        except Exception:
            p = None
        if p is not None and verifies(p, demos) and loo_ok(ind, demos):
            return p, ind.__name__
    # 2. heuristic beam search over the pure-function library (deep, narrow)
    prog_names = beam_search(demos, max_depth=6, beam=500)
    if prog_names is not None:
        fns = [FUNCS[n] for n in prog_names]
        return compose(fns), "bfs:" + "∘".join(reversed(prog_names))
    # 3. one structural op + induced color-map (color fix)
    for nm, op in FUNCS.items():
        try:
            demos2 = [(op(i), o) for i, o in demos]
            if all(a.shape == b.shape for a, b in demos2):
                cm = induce_color_map(demos2)
                if cm is not None and verifies(compose([op, cm]), demos):
                    return compose([op, cm]), f"colormap∘{nm}"
        except Exception:
            pass
    return None, None

# ── task solving ──────────────────────────────────────────────────────────────
def solve_task(task):
    demos = [(to_grid(p["input"]), to_grid(p["output"])) for p in task["train"]]
    prog, desc = search(demos)
    if prog is None:
        return None, None, "abstain"
    preds = []
    for t in task["test"]:
        preds.append(prog(to_grid(t["input"])))
    return preds, desc, "solved"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(
        "/tmp/claude-1000/-home-redleadr-workspace-uchi",
        "97bdddb2-ace8-4f72-8b3b-734acc5971c4/scratchpad/arc_agi/data"))
    ap.add_argument("--split", choices=["training", "evaluation"], default="evaluation")
    ap.add_argument("--show", type=int, default=0)
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.data, args.split, "*.json")))
    print(f"[*] {args.split}: {len(files)} tasks")

    from collections import Counter
    solved = attempted = correct = abstained = 0
    by_desc = Counter()
    shown = 0
    for f in files:
        task = json.load(open(f))
        preds, desc, status = solve_task(task)
        if status == "abstain":
            abstained += 1
            continue
        attempted += 1
        by_desc[desc] += 1
        # exact-match against held-out test outputs (all test inputs must match)
        ok = all(eq(p, to_grid(t["output"]))
                 for p, t in zip(preds, task["test"]) if "output" in t)
        if ok:
            correct += 1
            if shown < args.show:
                shown += 1
                print(f"\n  SOLVED {os.path.basename(f)}  via [{desc}]")
                print("  test input:\n" + serialize(to_grid(task['test'][0]['input'])))
                print("  predicted = truth:\n" + serialize(preds[0]))

    print("\n" + "─" * 56)
    print(f"  ARC-AGI DSL search — {args.split}")
    print(f"  tasks          : {len(files)}")
    print(f"  program found  : {attempted}  ({attempted/len(files)*100:.1f}%)")
    print(f"  abstained      : {abstained}")
    print(f"  SOLVED (test correct) : {correct}  ({correct/len(files)*100:.1f}%)")
    prec = correct / attempted * 100 if attempted else 0
    print(f"  precision when non-abstaining : {prec:.1f}%  (verified program → right test)")
    print(f"  solver breakdown : {dict(by_desc.most_common())}")
    print("─" * 56)


if __name__ == "__main__":
    main()
