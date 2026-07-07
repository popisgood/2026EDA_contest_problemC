"""SDS-style post-legalize compaction + soft-block shaping (opt-in: ELECTRO_COMPACT=1).

Runs AFTER the zero-overlap legalized layout.  Two moves, alternated:

  * COMPACTION (constraint-graph longest-path): left/down-pack blocks, preserving
    the relative order.  Closes POSITIONAL dead space (the gap the analytical soft
    equilibrium leaves) -- the bleed the Family-Two probe pinned down.  Overlap-free
    by construction: packing only ever DECREASES coordinates, so a block can never
    move into a right/upper neighbour, and preplaced blocks are pinned as anchors.

  * SHAPING (SDS spirit): widen a soft block into the free space to its right (or
    grow it up), keeping AREA exact and aspect <= AR_CAP.  Widening fills a horizontal
    gap and SHORTENS the block, which opens vertical slack for the next compaction to
    close.  Provably overlap-free: the block only ever grows into space that is empty
    within its own band, and its orthogonal extent shrinks.

Strictly additive / lowest-risk: returns the ORIGINAL layout unless the candidate is
a strict bbox-area improvement at (near) zero overlap.  Preserves preplaced positions,
fixed/preplaced dims, and MIB shapes (MIB members are never reshaped).
"""
from __future__ import annotations

import numpy as np


def _bbox_area(x, y, w, h):
    return (float((x + w).max()) - float(x.min())) * (float((y + h).max()) - float(y.min()))


def _overlap(x, y, w, h):
    n = len(x)
    t = 0.0
    for i in range(n):
        xi1, yi1 = x[i] + w[i], y[i] + h[i]
        for j in range(i + 1, n):
            ox = min(xi1, x[j] + w[j]) - max(x[i], x[j])
            oy = min(yi1, y[j] + h[j]) - max(y[i], y[j])
            if ox > 1e-7 and oy > 1e-7:
                t += ox * oy
    return t


def _compact_axis(pos, size, opos, osize, unit, locked_u, floor):
    """Longest-path pack toward `floor` along one axis, with RIGID UNITS (S1):
    blocks sharing a unit id (cluster members) translate together, so grouping
    abutment survives compaction; units containing a preplaced block are pinned.
    Pairs overlapping in the orthogonal axis are separation-constrained.  Packing
    only decreases coords (constraint <= original gap; unprocessed neighbours use
    their original position, an upper bound of their final one) -> overlap-free."""
    n = len(pos)
    E = 1e-7
    members = {}
    for i in range(n):
        members.setdefault(int(unit[i]), []).append(i)
    base = {u: min(pos[i] for i in m) for u, m in members.items()}
    off = pos - np.array([base[int(unit[i])] for i in range(n)])
    newbase = {}
    for u in sorted(members, key=lambda k: base[k]):
        if locked_u[u]:
            newbase[u] = base[u]
            continue
        lb = floor
        for i in members[u]:
            o0, o1 = opos[i], opos[i] + osize[i]
            for j in range(n):
                if int(unit[j]) == u:
                    continue  # rigid: no intra-unit constraints
                if opos[j] < o1 - E and o0 < opos[j] + osize[j] - E:
                    if pos[j] < pos[i] - E or (abs(pos[j] - pos[i]) <= E and j < i):
                        v = int(unit[j])
                        jpos = (newbase[v] + off[j]) if v in newbase else pos[j]
                        c = jpos + size[j] - off[i]
                        if c > lb:
                            lb = c
        newbase[u] = min(lb, base[u])  # packing never increases coords
    return np.array([newbase[int(unit[i])] + off[i] for i in range(n)])


def _compact(x, y, w, h, unit, locked_u, floor, rounds=2):
    x = x.copy()
    y = y.copy()
    for _ in range(rounds):
        x = _compact_axis(x, w, y, h, unit, locked_u, floor)
        y = _compact_axis(y, h, x, w, unit, locked_u, floor)
    return x, y


def _fill_right(x, y, w, h, reshapable, ar_cap):
    n = len(x)
    E = 1e-7
    bbr = float((x + w).max())
    w = w.copy()
    h = h.copy()
    for i in range(n):
        if not reshapable[i]:
            continue
        area = w[i] * h[i]
        y0, y1 = y[i], y[i] + h[i]
        lim = bbr
        for j in range(n):
            if j == i:
                continue
            if y[j] < y1 - E and y0 < y[j] + h[j] - E and x[j] >= x[i] + w[i] - E:
                lim = min(lim, x[j])
        new_w = min(lim - x[i], (area * ar_cap) ** 0.5)
        if new_w > w[i] + 1e-6:
            w[i] = new_w
            h[i] = area / new_w
    return w, h


def _fill_up(x, y, w, h, reshapable, ar_cap):
    n = len(x)
    E = 1e-7
    bbt = float((y + h).max())
    w = w.copy()
    h = h.copy()
    for i in range(n):
        if not reshapable[i]:
            continue
        area = w[i] * h[i]
        x0, x1 = x[i], x[i] + w[i]
        lim = bbt
        for j in range(n):
            if j == i:
                continue
            if x[j] < x1 - E and x0 < x[j] + w[j] - E and y[j] >= y[i] + h[i] - E:
                lim = min(lim, y[j])
        new_h = min(lim - y[i], (area * ar_cap) ** 0.5)
        if new_h > h[i] + 1e-6:
            h[i] = new_h
            w[i] = area / new_h
    return w, h


def compact_and_shape(x, y, w, h, is_soft, is_pre, mib_id,
                      ar_cap=4.0, floor=0.0, rounds=4, clust_id=None):
    """Return a compacted+shaped (x, y, w, h) -- or the input unchanged if the pass
    fails to strictly shrink the bbox at near-zero overlap.

    S1 (constraint-aware): pass `clust_id` to compact each grouping cluster as a
    rigid body (members keep their relative offsets -> V_grouping preserved);
    a unit containing a preplaced block is pinned entirely."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    w = np.asarray(w, float)
    h = np.asarray(h, float)
    n = len(x)
    pre = np.asarray(is_pre, bool)
    # rigid units: cluster id > 0 -> one unit per cluster; else singleton
    clust = (np.asarray(clust_id, int) if clust_id is not None
             else np.zeros(n, int))
    # never reshape MIB members (same-shape) NOR cluster members (reshaping
    # shrinks the orthogonal side and can detach them from the group)
    reshapable = (np.asarray(is_soft, bool) & (np.asarray(mib_id, int) == 0)
                  & (clust == 0))
    raw = np.where(clust > 0, clust, -(np.arange(n) + 2))
    _, unit = np.unique(raw, return_inverse=True)
    locked_u = np.zeros(int(unit.max()) + 1, dtype=bool)
    for i in range(n):
        if pre[i]:
            locked_u[unit[i]] = True

    a0 = _bbox_area(x, y, w, h)
    best = (x.copy(), y.copy(), w.copy(), h.copy())
    best_area = a0

    cx, cy, cw, ch = x.copy(), y.copy(), w.copy(), h.copy()
    for _ in range(rounds):
        cx, cy = _compact(cx, cy, cw, ch, unit, locked_u, floor)
        cw, ch = _fill_right(cx, cy, cw, ch, reshapable, ar_cap)
        cx, cy = _compact(cx, cy, cw, ch, unit, locked_u, floor)
        cw, ch = _fill_up(cx, cy, cw, ch, reshapable, ar_cap)
        cx, cy = _compact(cx, cy, cw, ch, unit, locked_u, floor)
        a = _bbox_area(cx, cy, cw, ch)
        if a < best_area - 1e-9 and _overlap(cx, cy, cw, ch) < 1e-6:
            best_area = a
            best = (cx.copy(), cy.copy(), cw.copy(), ch.copy())
    return best
