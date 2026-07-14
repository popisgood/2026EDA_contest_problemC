"""Parallel multi-start worker, kept in its own importable module.

The contest harness loads electro_optimizer.py from a file path under the name
"optimizer_module", which is NOT importable by name -- so a worker function
defined there can't be pickled to a multiprocessing worker.  Defining the worker
here (electro_parallel is a normal module on sys.path, because electro_optimizer
inserts its own directory) makes it picklable, so the fork pool works.

Per-case inputs are stashed in the module global `WORK` *before* the pool is
created, so the fork inherits them and we never pickle the connectivity tensors.
"""
from __future__ import annotations

import os

import numpy as np

from analytical_place import place
from legalize import legalize, remove_overlap
from soft_repair import boundary_snap, grouping_repair

WORK = None   # per-case inputs, set by the parent before forking the pool


def run_start(seed, P):
    """One independent start: place + legalize + iterated min-displacement repair."""
    positions, _ = place(
        P["n"], P["area"], P["b2b"], P["p2b"], P["pins"], P["cons"], P["tp"],
        iters=P["iters"], lr=P["lr"], device=P["device"], seed=seed,
        init_centers=P["init"],
    )
    x = np.array([p[0] for p in positions], dtype=float)
    y = np.array([p[1] for p in positions], dtype=float)
    w = np.array([p[2] for p in positions], dtype=float)
    h = np.array([p[3] for p in positions], dtype=float)
    is_pre, clust_id, bcode = P["is_pre"], P["clust_id"], P["bcode"]
    # nonneg keeps the WHOLE chain (legalize + both repairs + final cleanup)
    # floored at 0, so blocks never drift far below the wall and get shoved back
    # -- the incremental floor that makes first-quadrant containment cheap rather
    # than exploding (a post-hoc floor-only-at-the-end shove cascades the legalizer).
    floor = 0.0 if P.get("nonneg", False) else None
    x, y = legalize(x, y, w, h, is_pre, floor=floor)
    for _ in range(P["rounds"]):
        x, y = grouping_repair(x, y, w, h, clust_id, is_pre, floor=floor)
        x, y = boundary_snap(x, y, w, h, bcode, is_pre, floor=floor)
    # final hard-feasibility net; nonneg also enforces the x=0/y=0 canvas walls
    x, y = remove_overlap(x, y, w, h, is_pre, nonneg=P.get("nonneg", False))
    return x, y, w, h


def m1_warmstart_variant(m1_pos, P):
    """Warm-start electro's analytical optimizer from M1's LEGAL rollout: M1 supplies
    the starting positions AND aspect ratios, then place() runs its gradient descent
    from there to clean up HPWL / grouping / boundary.  The continuous refinement is
    immune to M1's autoregressive exposure bias (it's smooth optimization, not
    step-by-step guessing), so it can fix the drift M1 alone can't.  Returned as an
    extra candidate; same legalize+repair tail as run_start."""
    import torch
    x0, y0, w0, h0 = (np.asarray(a, float) for a in m1_pos)
    init_centers = torch.tensor(np.stack([x0 + 0.5 * w0, y0 + 0.5 * h0], axis=1),
                                dtype=torch.float32)
    init_la = torch.tensor(np.log(np.clip(w0, 1e-6, None) / np.clip(h0, 1e-6, None)),
                           dtype=torch.float32)
    positions, _ = place(
        P["n"], P["area"], P["b2b"], P["p2b"], P["pins"], P["cons"], P["tp"],
        iters=P["iters"], lr=P["lr"], device=P["device"], seed=0,
        init_centers=init_centers, init_la=init_la)
    x = np.array([p[0] for p in positions], dtype=float)
    y = np.array([p[1] for p in positions], dtype=float)
    w = np.array([p[2] for p in positions], dtype=float)
    h = np.array([p[3] for p in positions], dtype=float)
    is_pre, clust_id, bcode = P["is_pre"], P["clust_id"], P["bcode"]
    floor = 0.0 if P.get("nonneg", False) else None
    x, y = legalize(x, y, w, h, is_pre, floor=floor)
    for _ in range(P.get("rounds", 3)):
        x, y = grouping_repair(x, y, w, h, clust_id, is_pre, floor=floor)
        x, y = boundary_snap(x, y, w, h, bcode, is_pre, floor=floor)
    x, y = remove_overlap(x, y, w, h, is_pre, nonneg=P.get("nonneg", False))
    return x, y, w, h


def compact_variant(start, P, aware=False):
    """SDS-style compaction + soft shaping of one legalized layout, returned as an
    ADDITIONAL candidate.  solve() ranks it against the un-compacted layout by the
    full cost proxy (incl. exp(2*V_rel)), so it is only ever chosen when it is NET
    better.  Overlap-free by construction.

    Two flavours, BOTH added as candidates (the ranking picks per case):
      * aware=False (plain): maximum compaction force, constraint-blind -- wins
        where grouping/boundary tolerate the squeeze (e.g. tid20/80).
      * aware=True (S1): clusters compact as RIGID bodies (grouping preserved,
        members excluded from reshaping), then the repair chain re-snaps boundary
        blocks onto the new tighter bbox -- rescues the cases where the plain
        squeeze blew up V_rel and got rejected (e.g. tid60/99)."""
    from shape_compact import compact_and_shape
    x, y, w, h = start
    nonneg = P.get("nonneg", False)
    floor = 0.0 if nonneg else float(min(x.min(), y.min()))
    ar = float(os.environ.get("ELECTRO_COMPACT_AR", "4.0"))
    cx, cy, cw, ch = compact_and_shape(
        x, y, w, h, P["is_soft"], P["is_pre"], P["mib_id"], ar_cap=ar, floor=floor,
        clust_id=P["clust_id"] if aware else None)
    if aware:
        for _ in range(2):
            cx, cy = grouping_repair(cx, cy, cw, ch, P["clust_id"], P["is_pre"],
                                     floor=floor)
            cx, cy = boundary_snap(cx, cy, cw, ch, P["bcode"], P["is_pre"],
                                   floor=floor)
    cx, cy = remove_overlap(cx, cy, cw, ch, P["is_pre"], nonneg=nonneg)
    return cx, cy, cw, ch


def pool_init(threads=1):
    """Give each worker its share of cores (cores/nproc threads).  Threads are set
    AFTER the fork, so the parent never holds a live OpenMP pool across fork."""
    try:
        import torch
        torch.set_num_threads(max(1, int(threads)))
    except Exception:
        pass


def seed_worker(seed):
    return run_start(seed, WORK)
