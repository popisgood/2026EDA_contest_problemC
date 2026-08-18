"""Structural (by-construction) grouping fix for slice_pack (§8.53, opt-in
ELECTRO_SLICE_CLUSTER_VIRTUALIZE=1).

Idea from Gemini's review of this project (2026-08-02, evaluated and largely
found to already exist or be redundant -- see 8_Winning_Strategy_and_Roadmap.md
§8.53 for the full writeup), the one genuinely new suggestion: pack each
grouping cluster into one virtual "super-block" BEFORE the guillotine
dissection ever runs, instead of relying on centroid-based sort order to make
members merely LIKELY to land in the same subtree (slice_pack.py's existing
soft mechanism -- see its module docstring). This mirrors the already-proven
"by-construction grouping" fix in the generative B*-tree route
(`ml/pack_tree.py::_collapse_clusters`, contributed -3.8% there, 2026-07-09),
adapted to slice_pack's guillotine-dissection topology instead of a B*-tree.

Design, and why it needed no changes to slice_pack.py itself:
  1. REDUCE: eligible clusters (all-soft, no boundary code, no MIB, no
     preplaced/fixed member -- see `_eligible_clusters`) are collapsed into
     ONE virtual entry with combined area, at the cluster's area-weighted
     centroid. The outer recursion now sees a smaller problem and treats
     this virtual entry exactly like any other flexible-shape soft leaf --
     no special-casing needed in `_place_leaf`, so it gets the SAME
     exact-area, region-flexible-aspect-ratio placement every other soft
     block gets.
  2. slice_pack() runs UNMODIFIED on the reduced problem.
  3. EXPAND: once a virtual entry lands at some (x0, y0, w, h) with
     w*h == combined_area (exact, by construction of step 1), its region is
     handed to a FRESH, independent call to slice_pack.py's own
     `_dissect()` -- the same guillotine-dissection primitive, just scoped
     to the cluster's real members and an externally-given target region
     instead of one derived from `_outline()`. A guillotine dissection of a
     rectangle always yields a fully-connected, exact-area tiling of that
     rectangle by construction (every leaf touches at least one sibling
     across the cut that created it), so V_grouping=0 for any cluster this
     module successfully expands -- no repair pass needed afterwards.
  4. If step 3 fails for a given cluster (rare -- soft blocks with exactly
     the right total area essentially always tile), that cluster is
     dropped from the WHOLE candidate (matching slice_pack.py's own "any
     step's failure returns None, caller falls back" discipline) rather
     than silently degrading -- strictly additive: a failing collapse
     costs one candidate, never produces a bad layout.
"""
from __future__ import annotations

import os

import numpy as np

import slice_pack as _sp

_L, _R, _T, _B = 1, 2, 4, 8


def _eligible_clusters(n, is_fixed, is_pre, clust_id, bcode, mib_id):
    """Cluster ids (>0) whose EVERY member is a plain soft block: not fixed,
    not preplaced, no boundary code, no MIB membership. Conservative by
    design (§8.53 v1) -- mixed/anchored clusters just fall back to the
    existing centroid-sort-based soft handling, unchanged."""
    groups = {}
    for i in range(n):
        cid = int(clust_id[i])
        if cid > 0:
            groups.setdefault(cid, []).append(i)
    out = {}
    for cid, members in groups.items():
        if len(members) < 2:
            continue
        if any(is_fixed[i] or is_pre[i] or int(bcode[i]) != 0 or
               (mib_id is not None and int(mib_id[i]) != 0)
               for i in members):
            continue
        out[cid] = members

    # §8.53 follow-up: collapsing a cluster replaces its N scattered members
    # with ONE concentrated point in the recursive-cut ORDERING, which can
    # shift where OTHER (possibly non-eligible) clusters' members land --
    # the same "shared decision path" coupling this project has hit before
    # (gradient budget in §8.48/49, normalized-Laplacian degree in §8.50).
    # Measured worse when a case has >=2 eligible clusters (more topology
    # disruption). ELECTRO_SLICE_CLUSTER_VIRT_MAX caps how many clusters get
    # collapsed per case (default: unlimited); when capped, keep the
    # LARGEST ones (biggest V_grouping win per collapse).
    max_n = int(os.environ.get("ELECTRO_SLICE_CLUSTER_VIRT_MAX", "0"))
    if max_n > 0 and len(out) > max_n:
        keep_ids = sorted(out, key=lambda cid: -len(out[cid]))[:max_n]
        out = {cid: out[cid] for cid in keep_ids}
    return out


def _reduce(x, y, w, h, areas, is_fixed, is_pre, clust_id, bcode, mib_id):
    """Build the smaller problem slice_pack() actually runs on. Returns
    (reduced arrays..., expand_info) where expand_info is None if there was
    nothing eligible to collapse (caller should just use slice_pack directly)."""
    n = len(x)
    clusters = _eligible_clusters(n, is_fixed, is_pre, clust_id, bcode, mib_id)
    if not clusters:
        return None

    absorbed = {i for members in clusters.values() for i in members}
    keep = [i for i in range(n) if i not in absorbed]
    cx = x + 0.5 * w
    cy = y + 0.5 * h

    r_x, r_y, r_w, r_h, r_a = [], [], [], [], []
    r_fixed, r_pre, r_bcode, r_clust, r_mib = [], [], [], [], []
    virtual_slot = {}   # reduced index -> cluster id
    orig_index = {}      # reduced index -> original index (for `keep` entries)

    def push(px, py, pw, ph, pa, pfixed, ppre, pb, pc, pm):
        r_x.append(px); r_y.append(py); r_w.append(pw); r_h.append(ph)
        r_a.append(pa); r_fixed.append(pfixed); r_pre.append(ppre)
        r_bcode.append(pb); r_clust.append(pc); r_mib.append(pm)

    for i in keep:
        j = len(r_x)
        orig_index[j] = i
        push(x[i], y[i], w[i], h[i], areas[i], bool(is_fixed[i]), bool(is_pre[i]),
             int(bcode[i]), 0, 0 if mib_id is None else int(mib_id[i]))

    for cid, members in clusters.items():
        total_a = float(sum(areas[i] for i in members))
        wsum = sum(areas[i] for i in members) or 1.0
        vcx = sum(cx[i] * areas[i] for i in members) / wsum
        vcy = sum(cy[i] * areas[i] for i in members) / wsum
        side = max(total_a, 1e-9) ** 0.5
        j = len(r_x)
        virtual_slot[j] = cid
        push(vcx - 0.5 * side, vcy - 0.5 * side, side, side, total_a,
             False, False, 0, 0, 0)

    expand_info = {
        "clusters": clusters,             # cid -> [original member indices]
        "virtual_slot": virtual_slot,      # reduced idx -> cid
        "orig_index": orig_index,          # reduced idx -> original idx (kept blocks)
        "areas": areas,
    }
    return (np.array(r_x), np.array(r_y), np.array(r_w), np.array(r_h),
           np.array(r_a), np.array(r_fixed, bool), np.array(r_pre, bool),
           np.array(r_clust, int), np.array(r_bcode, int),
           np.array(r_mib, int), expand_info)


def _expand_cluster(cid, members, region, areas, budget=3000):
    """Nested guillotine dissection of `region` among a cluster's real
    members -- the same primitive slice_pack.py's own outer recursion uses,
    just scoped smaller and given an externally-fixed target rectangle
    instead of one derived from _outline(). Returns {member_idx: (x,y,w,h)}
    or None on failure (region can't be exactly tiled -- essentially never
    happens for plain soft blocks whose combined area equals the region's,
    but the guard is kept per this project's "never produce an illegal
    layout" discipline)."""
    m = len(members)
    sub_x = np.zeros(m); sub_y = np.zeros(m)
    sub_w = np.zeros(m); sub_h = np.zeros(m)
    sub_a = np.array([areas[i] for i in members], dtype=float)
    sub_fixed = np.zeros(m, bool)
    sub_pre = np.zeros(m, bool)
    sub_bcode = np.zeros(m, int)
    sub_clust = np.zeros(m, int)

    ctx = _sp._Ctx(sub_x, sub_y, sub_w, sub_h, sub_a, sub_fixed, sub_pre,
                  sub_clust, sub_bcode)
    ctx.phx_hi = np.full(m, -np.inf); ctx.phx_lo = np.full(m, np.inf)
    ctx.phy_hi = np.full(m, -np.inf); ctx.phy_lo = np.full(m, np.inf)
    ctx.rig_w = np.zeros(m); ctx.rig_h = np.zeros(m)
    ctx.reg_w = np.zeros(m); ctx.reg_h = np.zeros(m)
    ctx.reg_x = np.zeros(m); ctx.reg_y = np.zeros(m)
    ctx.walls = False
    ctx.budget = [budget]

    out = [None] * m
    ok = _sp._dissect(region, list(range(m)), _L | _R | _T | _B, ctx, out)
    if not ok or any(o is None for o in out):
        return None
    return {members[k]: out[k] for k in range(m)}


def slice_pack_clustered(x, y, w, h, areas, is_fixed, is_pre, clust_id, bcode,
                         mib_id=None, **kwargs):
    """Drop-in wrapper around slice_pack.slice_pack(): collapses eligible
    grouping clusters into virtual super-blocks, runs the normal slice_pack
    on the reduced problem, then expands each virtual block back into its
    real members via a nested guillotine dissection. Falls back to plain
    slice_pack() unchanged if there is nothing eligible to collapse, or
    returns None (matching slice_pack's own contract) if any expansion
    fails -- never produces a partially-expanded/illegal layout."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    w = np.asarray(w, float); h = np.asarray(h, float)
    areas = np.asarray(areas, float)
    is_fixed = np.asarray(is_fixed, bool); is_pre = np.asarray(is_pre, bool)
    clust_id = np.asarray(clust_id, int); bcode = np.asarray(bcode, int)
    mib_arr = None if mib_id is None else np.asarray(mib_id, int)

    reduced = _reduce(x, y, w, h, areas, is_fixed, is_pre, clust_id, bcode, mib_arr)
    if reduced is None:
        return _sp.slice_pack(x, y, w, h, areas, is_fixed, is_pre, clust_id,
                              bcode, mib_id, **kwargs)

    (r_x, r_y, r_w, r_h, r_a, r_fixed, r_pre, r_clust, r_bcode, r_mib,
     info) = reduced
    res = _sp.slice_pack(r_x, r_y, r_w, r_h, r_a, r_fixed, r_pre, r_clust,
                         r_bcode, r_mib if mib_id is not None else None,
                         **kwargs)
    if res is None:
        return None

    return_pair = kwargs.get("return_pair", False)
    pairs = [res] if not return_pair else list(res)
    expanded_pairs = []
    for rx, ry, rw, rh in pairs:
        full_x = np.array(x, dtype=float, copy=True)
        full_y = np.array(y, dtype=float, copy=True)
        full_w = np.array(w, dtype=float, copy=True)
        full_h = np.array(h, dtype=float, copy=True)
        for j, i in info["orig_index"].items():
            full_x[i], full_y[i], full_w[i], full_h[i] = rx[j], ry[j], rw[j], rh[j]
        for j, cid in info["virtual_slot"].items():
            members = info["clusters"][cid]
            region = (float(rx[j]), float(ry[j]), float(rw[j]), float(rh[j]))
            placed = _expand_cluster(cid, members, region, areas)
            if placed is None:
                return None
            for i, (mx, my, mw, mh) in placed.items():
                full_x[i], full_y[i], full_w[i], full_h[i] = mx, my, mw, mh
        expanded_pairs.append((full_x, full_y, full_w, full_h))

    return tuple(expanded_pairs) if return_pair else expanded_pairs[0]
