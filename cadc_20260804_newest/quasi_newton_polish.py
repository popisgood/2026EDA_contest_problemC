"""L-BFGS continuous polish for an already-legal candidate (post legalize+repair).

Why this is a SEPARATE module instead of an `ELECTRO_OPT=lbfgs` option inside
`analytical_place.place()`: that function's loss is an ANNEALED schedule (area
grows, overlap/grouping/boundary weights ramp over 300 iterations) -- the
objective changes every step.  L-BFGS builds its search direction from a
history of past gradients, which assumes the objective is FIXED; feeding it a
moving target defeats the whole point of the curvature estimate.  So instead
this runs AFTER a candidate already has legal, exact-area shapes: only (cx,cy)
are free, (w,h) are frozen, and the loss (HPWL + a soft overlap penalty) never
changes across iterations -- exactly the fixed-objective, few-iteration regime
L-BFGS is good at (literature: quasi-Newton floorplan refinement, "topology
fixed, only continuous coords adjusted", 3-5% HPWL gain near-zero extra cost).

Strictly additive, matching the rest of the portfolio: this only nudges
positions to squeeze out residual HPWL.  The soft overlap penalty here
DISCOURAGES but does not GUARANTEE zero overlap, so the caller must re-run
`legalize.remove_overlap()` on the output before scoring/accepting it -- same
"hard-feasibility net" pattern already used everywhere else in this pipeline.

2026-08-11 empirical finding (ELECTRO_QN_POLISH_DEBUG=1 on a real case):
the first version penalized overlap with relu(A)*relu(B) -- a product of two
ReLUs, which has a kink exactly at the zero-overlap boundary every candidate
here already sits on (it just came out of legalize+repair).  L-BFGS's
strong_wolfe line search assumes local smoothness; at that kink it found no
step satisfying the Wolfe conditions and returned step 0 on every seed
(moved_l1=0.0 across the board).  Setting lam_ov=0 to test the theory
confirmed it: blocks then moved by hundreds of units, but with nothing
pulling them back, V_rel exploded (0.088 -> 0.6+) as the boundary/grouping
alignment _finish() had built got wrecked.  Fix: (a) square the ReLUs so the
penalty is differentiable through the boundary, and (b) add a mild quadratic
anchor pulling positions back toward the pre-polish start, so this stays a
small local refinement instead of a re-placement.
"""
from __future__ import annotations

import numpy as np
import torch


def quasi_newton_polish(x, y, w, h, is_pre, eb, ep, pv, iters=40, lam_ov=1.0,
                         lam_anchor=0.01, lr=1.0):
    """Polish block centers via L-BFGS.  (w,h) are held fixed; preplaced blocks
    (is_pre) never move.  x,y,w,h are numpy arrays of absolute geometry; eb/ep/pv
    are the numpy edge/pin arrays from `electro_parallel._edges_np` (or None).
    Returns (new_x, new_y) numpy arrays -- (w,h) unchanged."""
    n = len(x)
    if n == 0:
        return x, y

    cx0 = torch.tensor(x + 0.5 * w, dtype=torch.float64)
    cy0 = torch.tensor(y + 0.5 * h, dtype=torch.float64)
    wt = torch.tensor(w, dtype=torch.float64)
    ht = torch.tensor(h, dtype=torch.float64)
    pre_mask = torch.tensor(is_pre, dtype=torch.bool)

    cx = cx0.clone().requires_grad_(True)
    cy = cy0.clone().requires_grad_(True)

    ia = ib = wb = None
    if eb is not None and len(eb):
        ia = torch.tensor(eb[:, 0], dtype=torch.long).clamp(0, n - 1)
        ib = torch.tensor(eb[:, 1], dtype=torch.long).clamp(0, n - 1)
        wb = torch.tensor(eb[:, 2], dtype=torch.float64)
    ebk = wp = tx = ty = None
    if ep is not None and len(ep) and pv is not None and len(pv):
        pvt = torch.tensor(pv, dtype=torch.float64)
        et = torch.tensor(ep[:, 0], dtype=torch.long).clamp(0, len(pv) - 1)
        ebk = torch.tensor(ep[:, 1], dtype=torch.long).clamp(0, n - 1)
        wp = torch.tensor(ep[:, 2], dtype=torch.float64)
        tx, ty = pvt[et, 0], pvt[et, 1]

    triu = torch.triu_indices(n, n, offset=1)
    ti, tj = triu[0], triu[1]

    opt = torch.optim.LBFGS([cx, cy], lr=lr, max_iter=iters,
                             history_size=10, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        ecx = torch.where(pre_mask, cx0, cx)
        ecy = torch.where(pre_mask, cy0, cy)
        loss = ecx.new_zeros(())
        if ia is not None:
            loss = loss + (wb * ((ecx[ia] - ecx[ib]).abs()
                                  + (ecy[ia] - ecy[ib]).abs())).sum()
        if ebk is not None:
            loss = loss + (wp * ((tx - ecx[ebk]).abs()
                                  + (ty - ecy[ebk]).abs())).sum()
        dx = (ecx[ti] - ecx[tj]).abs()
        dy = (ecy[ti] - ecy[tj]).abs()
        # Squared-ReLU overlap penalty: same zero-outside-the-box support as
        # relu(A)*relu(B), but differentiable through the boundary (see
        # module docstring for why the un-squared product form stalled
        # L-BFGS's line search at step 0 on every candidate tested).
        ox = torch.relu(0.5 * (wt[ti] + wt[tj]) - dx)
        oy = torch.relu(0.5 * (ht[ti] + ht[tj]) - dy)
        ov = (ox * ox + oy * oy).sum()
        loss = loss + lam_ov * ov
        # Anchor: keep this a small local refinement, not a re-placement --
        # without it (lam_ov=0 ablation) blocks drifted hundreds of units and
        # wrecked the boundary/grouping alignment _finish() had built.
        loss = loss + lam_anchor * ((ecx - cx0) ** 2 + (ecy - cy0) ** 2).sum()
        loss.backward()
        return loss

    opt.step(closure)

    with torch.no_grad():
        ecx = torch.where(pre_mask, cx0, cx)
        ecy = torch.where(pre_mask, cy0, cy)
        nx = (ecx - 0.5 * wt).numpy()
        ny = (ecy - 0.5 * ht).numpy()
    return nx, ny
