"""BADGE-style Dirichlet harmonic-extension initialization (§8.52, opt-in
ELECTRO_DIRICHLET_INIT=1).

Ported idea from two papers found via Connected Papers derivative-works
scan of the project's Zotero library, 2026-08-02 (see
8_Winning_Strategy_and_Roadmap.md §8.52 for the full writeup):

  * BADGE (Park & Paik, DATE'26): builds a weighted graph Laplacian,
    imposes Dirichlet constraints on FIXED vertices (their case: fixed
    macros), and solves for the minimum-quadratic-energy ("harmonic")
    extension into the free vertices -- a closed-form solve, not an
    iterative spectral filter. Reported to beat both random and GiFt
    initialization on HPWL and iteration count.

  * DPlanner / "Hierarchical Graph Learning-Based Floorplanning With
    Dirichlet Boundary Conditions" (Liu et al., IEEE TVLSI 2024, same
    authors as GiFt): the same Dirichlet-boundary framing, decomposing
    floorplanning into a convex QP (wirelength under location
    constraints) solved exactly, plus a combinatorial sub-problem.
    Reported 41-56% iteration/runtime reduction integrated with SOTA
    mixed-size placers.

Why this is a different bet than gsp_init.py (§8.50, rejected): that
module ran GiFt's spectral LOW-PASS FILTER (eigendecompose the
*normalized* Laplacian, filter a *random* signal) and found that adding
grouping/MIB/boundary as extra edges diluted the normalized degree and
actively hurt Vmib/Vbnd whenever more than one signal was mixed in. This
module instead solves the *exact* Dirichlet problem on the *unnormalized*
Laplacian: minimise sum_edges w_ij*(x_i-x_j)^2 subject to KNOWN values at
fixed nodes (preplaced blocks + pin/terminal positions). There is no
random component and no degree normalization to dilute -- grouping/MIB/
boundary edges just add more terms to the SAME energy functional being
exactly minimised, alongside wirelength.

BADGE's "boundary" means fixed macros; our contest also has a die-EDGE
alignment constraint (`bcode`) that isn't in their formulation. We extend
it the same way the rest of this project's soft constraints are modelled:
one virtual Dirichlet wall node per active boundary code, placed at the
real bounding-box extreme, connected to its required blocks by an
attractive edge -- this makes edge alignment part of the same harmonic
solve rather than a post-hoc repair.
"""
from __future__ import annotations

import numpy as np


def dirichlet_cluster_init(n, eb, ep, pv, is_pre, tp, clust_id, mib_id,
                            bcode=None, grp_weight=2.0, mib_weight=2.0,
                            bnd_weight=2.0, ridge=1e-3):
    """Return raw-scale (cx, cy) [n, 2] numpy array, or None if degenerate.

    n         : block count
    eb        : [n_edges, 3] (i, j, weight) b2b edges, or None
    ep        : [n_pin_edges, 3] (pin_idx, block_idx, weight) p2b edges, or None
    pv        : [n_pins, 2] raw-coord pin/terminal positions, or None
    is_pre    : [n] bool, preplaced blocks (position known & fixed -- Dirichlet)
    tp        : [n, 4] target_positions (x, y, w, h; -1 where unset)
    clust_id  : [n] int, 0 = no group
    mib_id    : [n] int, 0 = no group
    bcode     : [n] int bitmask (1=left,2=right,4=top,8=bottom), or None
    grp_weight/mib_weight/bnd_weight : extra attractive-edge weight for
        same-cluster, same-MIB, and boundary-wall pairs (0 = signal off).
    ridge     : small Tikhonov term (pulls isolated/zero-degree free nodes
        softly toward the known-anchor centroid instead of leaving them
        numerically undefined -- BADGE's "filler initialization" role).
    """
    if n < 1:
        return None
    n_pins = 0 if pv is None else len(pv)

    # --- known (Dirichlet) coordinate values + anchor bounding box, before
    # the wall nodes so they can be placed at the real extremes -------------
    known = []
    if pv is not None and n_pins:
        known.append(pv)
    pre_xy = tp[is_pre, :2] if np.any(is_pre) else None
    if pre_xy is not None and len(pre_xy):
        known.append(pre_xy)
    if known:
        k = np.concatenate(known, axis=0)
        lo, hi = k.min(axis=0), k.max(axis=0)
        center = (lo + hi) / 2.0
    else:
        lo, hi = np.zeros(2), np.ones(2)
        center = np.full(2, 0.5)

    bc = np.zeros(n, dtype=int) if bcode is None else np.asarray(bcode)
    wall_defs = [(1, np.array([lo[0], center[1]])),   # left
                 (2, np.array([hi[0], center[1]])),   # right
                 (4, np.array([center[0], hi[1]])),   # top
                 (8, np.array([center[0], lo[1]]))]   # bottom
    active_walls = [(bit, pos) for bit, pos in wall_defs
                    if bnd_weight > 0 and np.any((bc & bit) > 0)]
    n_walls = len(active_walls)

    # node layout: [0:n) blocks, [n:n+n_pins) pins, [..:..) walls
    total = n + n_pins + n_walls
    fixed_mask = np.zeros(total, dtype=bool)
    fixed_val = np.zeros((total, 2))
    if n_pins:
        fixed_mask[n:n + n_pins] = True
        fixed_val[n:n + n_pins] = pv
    for wi, (bit, pos) in enumerate(active_walls):
        idx = n + n_pins + wi
        fixed_mask[idx] = True
        fixed_val[idx] = pos
    if pre_xy is not None and len(pre_xy):
        pre_idx = np.where(is_pre)[0]
        fixed_mask[pre_idx] = True
        fixed_val[pre_idx] = pre_xy

    # --- weighted adjacency (unnormalized-Laplacian input) ------------------
    W = np.zeros((total, total))
    if eb is not None and len(eb):
        i = np.clip(eb[:, 0].astype(int), 0, n - 1)
        j = np.clip(eb[:, 1].astype(int), 0, n - 1)
        w = eb[:, 2].astype(np.float64)
        W[i, j] += w
        W[j, i] += w
    if ep is not None and pv is not None and len(ep) and n_pins:
        pi = np.clip(ep[:, 0].astype(int), 0, n_pins - 1) + n
        bi = np.clip(ep[:, 1].astype(int), 0, n - 1)
        w = ep[:, 2].astype(np.float64)
        W[bi, pi] += w
        W[pi, bi] += w

    def add_group_edges(ids, weight):
        if weight <= 0:
            return
        ids = np.asarray(ids)
        for g in np.unique(ids):
            if g <= 0:
                continue
            members = np.where(ids == g)[0]
            if len(members) < 2:
                continue
            for a in range(len(members)):
                for b in range(a + 1, len(members)):
                    i, j = int(members[a]), int(members[b])
                    W[i, j] += weight
                    W[j, i] += weight

    add_group_edges(clust_id, grp_weight)
    add_group_edges(mib_id, mib_weight)

    for wi, (bit, pos) in enumerate(active_walls):
        idx = n + n_pins + wi
        for i in np.where((bc & bit) > 0)[0]:
            W[int(i), idx] += bnd_weight
            W[idx, int(i)] += bnd_weight

    deg = W.sum(axis=1)
    L = np.diag(deg) - W   # unnormalized graph Laplacian

    free = ~fixed_mask
    if not np.any(free):
        return tp[:n, :2] if np.all(fixed_mask[:n]) else None

    L_UU = L[np.ix_(free, free)]
    L_UF = L[np.ix_(free, fixed_mask)]
    X_F = fixed_val[fixed_mask]

    # Ridge: (L_UU + ridge*I) x_U = -L_UF x_F + ridge*center -- the extra
    # ridge*center term makes the regularizer a soft pull toward the known-
    # anchor centroid (not toward the origin), so isolated/low-degree free
    # nodes still land somewhere geometrically sane.
    L_UU_reg = L_UU + ridge * np.eye(L_UU.shape[0])
    rhs = -(L_UF @ X_F) + ridge * center[None, :]

    try:
        X_U = np.linalg.solve(L_UU_reg, rhs)
    except np.linalg.LinAlgError:
        return None

    out = np.empty((total, 2))
    out[fixed_mask] = X_F
    out[free] = X_U
    if not np.all(np.isfinite(out[:n])):
        return None
    return out[:n]
