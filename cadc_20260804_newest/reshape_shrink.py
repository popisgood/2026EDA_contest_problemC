"""Greedy edge-shrink + recompact: shrink whichever reshapeable blocks
currently sit at a bbox extreme (narrower along the offending axis, taller
along the other -- area held exact), then re-run the EXISTING longest-path
`legalize.compact_layout()` so the freed slack actually cascades through the
rest of the layout.  Repeat for a few rounds.

Why this replaces the earlier frozen-center continuous-optimization version
(shelved 2026-08-11): with every block's CENTER frozen, no block could ever
use a neighbor's freed slack -- there was no mechanism for anything to
SLIDE, so the bbox could only shrink by narrowing a block already sitting at
the extreme, and even then nothing downstream could follow it inward.
Confirmed empirically: bbox_area barely moved (some cases got slightly
worse).  This version is deliberately NOT a gradient/loss-weighted
optimization at all (the last two attempts at that -- an HPWL-pulling
L-BFGS polish, then a fixed-center Adam reshape -- both required delicate
weight tuning and both failed in different ways).  Instead it reuses
compact_layout()'s proven, deterministic, order-preserving compaction to do
the actual sliding; this module's only job is deciding WHICH blocks to
narrow before each recompact pass -- directly mirroring the manual
intuition: shrink the block currently against the wall so the wall itself
moves in, then let everything behind it follow.
"""
from __future__ import annotations

import numpy as np

from legalize import compact_layout, legalize


def reshape_shrink(x, y, w, h, is_reshapeable, is_pre, rounds=6, step=0.08,
                    ar_cap=4.0, floor=None):
    """Greedily narrow bbox-extreme reshapeable blocks and recompact, up to
    `rounds` times (stops early once a round narrows nothing).  `step` is the
    per-round log-aspect delta applied to each chosen block, capped in total
    at +-log(ar_cap) from its ORIGINAL aspect.  Returns (x, y, w, h); blocks
    with is_reshapeable[i] == False are never touched."""
    x = np.asarray(x, dtype=float).copy()
    y = np.asarray(y, dtype=float).copy()
    w = np.asarray(w, dtype=float).copy()
    h = np.asarray(h, dtype=float).copy()
    is_reshapeable = np.asarray(is_reshapeable, dtype=bool)
    is_pre = np.asarray(is_pre, dtype=bool)
    n = len(x)
    if n == 0 or not np.any(is_reshapeable):
        return x, y, w, h

    area = w * h
    la_total = np.zeros(n)          # cumulative log-aspect delta so far
    ar_delta_cap = float(np.log(ar_cap))
    tol = 1e-6

    for _ in range(int(rounds)):
        right = float((x + w).max())
        left = float(x.min())
        top = float((y + h).max())
        bot = float(y.min())
        changed = False
        for i in range(n):
            if not is_reshapeable[i]:
                continue
            at_right = (x[i] + w[i]) >= right - tol
            at_left = x[i] <= left + tol
            at_top = (y[i] + h[i]) >= top - tol
            at_bot = y[i] <= bot + tol
            if not (at_right or at_left or at_top or at_bot):
                continue
            # Narrowing (bigger la) helps a left/right-extreme block; getting
            # shorter (smaller la) helps a top/bottom-extreme one.  A corner
            # block can be both -- prefer whichever this round, alternating
            # naturally as la_total's cap is approached on one side.
            want_narrow = at_right or at_left
            want_short = at_top or at_bot
            if want_narrow and la_total[i] + step <= ar_delta_cap:
                la_total[i] += step
            elif want_short and la_total[i] - step >= -ar_delta_cap:
                la_total[i] -= step
            else:
                continue
            sqrt_a = np.sqrt(area[i])
            nw = sqrt_a * np.exp(0.5 * la_total[i])
            nh = sqrt_a * np.exp(-0.5 * la_total[i])
            cx_i = x[i] + 0.5 * w[i]
            cy_i = y[i] + 0.5 * h[i]
            w[i], h[i] = nw, nh
            x[i] = cx_i - 0.5 * nw
            y[i] = cy_i - 0.5 * nh
            changed = True
        if not changed:
            break
        # Reshaping a block at its old center can dent into a neighbor;
        # legalize() guarantees zero overlap robustly before compact_layout
        # pulls everything tight again (compact_layout assumes a near-legal
        # input, it does not itself resolve overlap).
        x, y = legalize(x, y, w, h, is_pre, floor=floor)
        x, y = compact_layout(x, y, w, h, is_pre, floor=floor, rounds=4)

    return x, y, w, h
