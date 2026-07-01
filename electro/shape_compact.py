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


def _compact_axis(pos, size, opos, osize, locked, floor):
    """Longest-path pack toward `floor` along one axis.  Pairs overlapping in the
    orthogonal axis are separation-constrained; order taken from current pos.  Packing
    only decreases coords (proven) so it never introduces overlap."""
    n = len(pos)
    order = sorted(range(n), key=lambda i: (pos[i], i))
    new = pos.copy()
    E = 1e-7
    for j in order:
        if locked[j]:
            continue  # preplaced: pinned anchor
        lb = floor
        o0, o1 = opos[j], opos[j] + osize[j]
        for i in range(n):
            if i == j:
                continue
            if opos[i] < o1 - E and o0 < opos[i] + osize[i] - E:        # orthogonal overlap
                if pos[i] < pos[j] - E or (abs(pos[i] - pos[j]) <= E and i < j):
                    c = new[i] + size[i]
                    if c > lb:
                        lb = c
        new[j] = lb
    return new


def _compact(x, y, w, h, locked, floor, rounds=2):
    x = x.copy()
    y = y.copy()
    for _ in range(rounds):
        x = _compact_axis(x, w, y, h, locked, floor)
        y = _compact_axis(y, h, x, w, locked, floor)
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
                      ar_cap=4.0, floor=0.0, rounds=4):
    """Return a compacted+shaped (x, y, w, h) -- or the input unchanged if the pass
    fails to strictly shrink the bbox at near-zero overlap."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    w = np.asarray(w, float)
    h = np.asarray(h, float)
    locked = np.asarray(is_pre, bool)
    reshapable = np.asarray(is_soft, bool) & (np.asarray(mib_id, int) == 0)

    a0 = _bbox_area(x, y, w, h)
    best = (x.copy(), y.copy(), w.copy(), h.copy())
    best_area = a0

    cx, cy, cw, ch = x.copy(), y.copy(), w.copy(), h.copy()
    for _ in range(rounds):
        cx, cy = _compact(cx, cy, cw, ch, locked, floor)
        cw, ch = _fill_right(cx, cy, cw, ch, reshapable, ar_cap)
        cx, cy = _compact(cx, cy, cw, ch, locked, floor)
        cw, ch = _fill_up(cx, cy, cw, ch, reshapable, ar_cap)
        cx, cy = _compact(cx, cy, cw, ch, locked, floor)
        a = _bbox_area(cx, cy, cw, ch)
        if a < best_area - 1e-9 and _overlap(cx, cy, cw, ch) < 1e-6:
            best_area = a
            best = (cx.copy(), cy.copy(), cw.copy(), ch.copy())
    return best
