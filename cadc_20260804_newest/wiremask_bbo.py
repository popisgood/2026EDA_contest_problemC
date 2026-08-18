"""WireMask-BBO-style greedy constructive placement (Xue et al., NeurIPS
2023, "Macro Placement by Wire-Mask-Guided Black-Box Optimization",
arXiv:2306.16844 -- see LITERATURE_REVIEW.md Tier 1 #1).

Build a floorplan from scratch, ONE block at a time, in connectivity-degree
order (highest-degree first, so well-connected blocks get first pick of
good positions relative to whatever's already placed).  Each block is
inserted at whichever legal "skyline corner" of the already-placed blocks
minimizes its incremental wirelength to ALREADY-PLACED neighbors only
(unplaced neighbors contribute 0 to the cost -- this incremental,
partial-information scoring is the "wire-mask" idea).  The paper works on a
discretized grid mask; here it's adapted to continuous coordinates + a
corner-point candidate set (standard skyline / maximal-rectangles greedy
packing), since FloorSet blocks are real-valued rectangles, not grid cells,
and n <= 120 makes an explicit corner enumeration cheap.

Strictly additive: this produces ONE more raw (x,y,w,h) candidate, exactly
like slice_pack() -- it then goes through the SAME _finish() repair chain
(compact + grouping/boundary repair + remove_overlap) as every other
candidate source.  MIB shape-unification and soft grouping/boundary
alignment are deliberately NOT handled during construction here -- they're
left to the existing post-hoc repair mechanisms every other candidate
source already relies on.  This module's only job is contributing a
genuinely different construction order and greedy placement criterion than
the gradient-descent or guillotine-dissection paths.
"""
from __future__ import annotations

import numpy as np


def wiremask_place(area, is_fixed, is_pre, tp, eb, ep, pv,
                    aspects=(0.5, 0.7, 1.0, 1.4, 2.0)):
    """area, is_fixed, is_pre: [N] numpy arrays.  tp: [N,4] numpy array
    (x,y,w,h), -1 where free -- the contest's target_positions convention
    (preplaced: all four set; fixed-shape: cols 2,3 set, 0,1 free).
    eb: [K,3] (i,j,weight) b2b edges or None.  ep: [K,3] (pin_idx,
    block_idx,weight) p2b edges or None.  pv: [P,2] pin positions or None.
    Returns (x, y, w, h) numpy arrays covering all N blocks."""
    N = len(area)
    x = np.zeros(N)
    y = np.zeros(N)
    w = np.zeros(N)
    h = np.zeros(N)
    placed = np.zeros(N, dtype=bool)

    for i in range(N):
        if is_pre[i]:
            x[i], y[i], w[i], h[i] = tp[i]
            placed[i] = True

    # Per-block neighbor lists so the per-insertion cost pass only touches
    # this block's own edges, not the full edge arrays every time.
    b2b_nbrs = [[] for _ in range(N)]
    if eb is not None and len(eb):
        for a, b, wt in eb:
            a, b = int(a), int(b)
            if 0 <= a < N and 0 <= b < N:
                b2b_nbrs[a].append((b, float(wt)))
                b2b_nbrs[b].append((a, float(wt)))
    p2b_nbrs = [[] for _ in range(N)]
    if ep is not None and len(ep) and pv is not None and len(pv):
        for pin_i, bi, wt in ep:
            pin_i, bi = int(pin_i), int(bi)
            if 0 <= bi < N and 0 <= pin_i < len(pv):
                p2b_nbrs[bi].append((pin_i, float(wt)))

    degree = np.array([sum(wt for _, wt in b2b_nbrs[i]) + sum(wt for _, wt in p2b_nbrs[i])
                        for i in range(N)])
    order = sorted((i for i in range(N) if not is_pre[i]),
                    key=lambda i: (-degree[i], -area[i]))

    for i in order:
        placed_idx = np.nonzero(placed)[0]
        if len(placed_idx) == 0:
            corners = np.array([[0.0, 0.0]])
            px = py = pw = ph = np.empty(0)
        else:
            px, py, pw, ph = x[placed_idx], y[placed_idx], w[placed_idx], h[placed_idx]
            cx_c = np.concatenate([[0.0], px + pw, px])
            cy_c = np.concatenate([[0.0], py, py + ph])
            corners = np.unique(np.stack([cx_c, cy_c], axis=1), axis=0)

        if is_fixed[i]:
            cand_shapes = [(float(tp[i, 2]), float(tp[i, 3]))]
        else:
            sqrt_a = float(np.sqrt(max(area[i], 1e-12)))
            cand_shapes = [(sqrt_a * np.sqrt(a), sqrt_a / np.sqrt(a)) for a in aspects]

        best_cost = None
        best = None
        cx0_all, cy0_all = corners[:, 0], corners[:, 1]
        for cw, ch in cand_shapes:
            if len(placed_idx) == 0:
                legal_mask = np.ones(len(corners), dtype=bool)
            else:
                ox = (np.minimum(cx0_all[:, None] + cw, px[None, :] + pw[None, :])
                      - np.maximum(cx0_all[:, None], px[None, :]))
                oy = (np.minimum(cy0_all[:, None] + ch, py[None, :] + ph[None, :])
                      - np.maximum(cy0_all[:, None], py[None, :]))
                legal_mask = ~((ox > 1e-9) & (oy > 1e-9)).any(axis=1)
            if not legal_mask.any():
                continue
            ccx = cx0_all[legal_mask] + 0.5 * cw
            ccy = cy0_all[legal_mask] + 0.5 * ch
            cost = np.zeros(len(ccx))
            for j, wt in b2b_nbrs[i]:
                if placed[j]:
                    cost += wt * (np.abs(ccx - (x[j] + 0.5 * w[j]))
                                   + np.abs(ccy - (y[j] + 0.5 * h[j])))
            for pin_i, wt in p2b_nbrs[i]:
                cost += wt * (np.abs(ccx - pv[pin_i, 0]) + np.abs(ccy - pv[pin_i, 1]))
            k = int(np.argmin(cost))
            if best_cost is None or cost[k] < best_cost:
                best_cost = float(cost[k])
                best = (float(cx0_all[legal_mask][k]), float(cy0_all[legal_mask][k]), cw, ch)

        if best is None:
            # Should not normally trigger ((0,0) or a skyline extension is
            # always legal), but stack above the layout as a safety fallback
            # rather than ever leaving a block unplaced.
            top = float((y[placed] + h[placed]).max()) if placed.any() else 0.0
            cw, ch = cand_shapes[0]
            best = (0.0, top, cw, ch)

        x[i], y[i], w[i], h[i] = best
        placed[i] = True

    return x, y, w, h
