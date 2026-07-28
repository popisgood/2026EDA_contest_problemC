#!/usr/bin/env python3
"""Contest entry point: analytical placement + legalization + soft-constraint repair.

Per case:
    analytical global placement  (WL + spreading + grouping + boundary terms)
      -> legalize to exactly zero overlap
      -> grouping_repair  (abut isolated cluster members)
      -> boundary_snap    (slide boundary blocks onto bbox edges)
      -> (x, y, w, h)

Hard constraints by construction: no overlap, soft-block area exact, MIB same
shape, fixed dims locked, preplaced pinned.  Soft constraints reduced by the
analytical penalties + the two repair passes.
"""
from __future__ import annotations

import multiprocessing as mp
import os
import sys
import time
from typing import List, Optional, Tuple

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# --- SUBMISSION DEFAULT: first-quadrant containment ------------------------
# The contest harness imports this module and calls solve() with no env vars.
# We want the SUBMITTED behaviour to keep every block in the first quadrant
# (x, y >= 0, the contest's origin convention) rather than the negative-coord-
# drifted layout that scores marginally better locally.  These two knobs turn
# on the in-optimization lower-wall clamp (CLAMP) and the floor-aware
# legalize+repair chain (NONNEG), which together guarantee non-negative output.
# Both are still overridable: e.g. ELECTRO_CLAMP=0 ELECTRO_NONNEG=0 reproduces
# the lower-cost (but negative-coord) configuration.
os.environ.setdefault("ELECTRO_CLAMP", "1")
os.environ.setdefault("ELECTRO_NONNEG", "1")
# Portfolios (WIDESWAP and GROUPING_PUSHPAST):
# Both are defaulted to "1" (2026-07-16) because they are strictly additive portfolio
# variants (only adding candidate starts evaluated by the cost proxy) and have been
# validated to cooperatively reduce the Neutral RT score from 2.4822 to 2.4072.
os.environ.setdefault("ELECTRO_BOUNDARY_WIDESWAP", "1")
os.environ.setdefault("ELECTRO_GROUPING_PUSHPAST", "1")
# Iters portfolio: DEFAULTED OFF in the merged tree.  Measured on the full 100
# (real evaluator, neutral RT) it was the dominant runtime cost -- 3.41s -> 5.83s
# avg -- AND the source of most of the 17/100 cases where the graph branch scored
# WORSE than temp (tid6 +80.9%, tid47 +21.2% at 9.55s, tid60 +12.0% at 10.32s).
# The candidate ranking is a PROXY (exp(2*V_rel)*(hpwl/mean + area/mean)), not the
# true contest cost, so extra candidates are only "strictly additive" with respect
# to the proxy -- feeding it more candidates can and does make it pick a layout
# that is worse by the real cost.  Set to "adaptive" or "1" to re-enable.
os.environ.setdefault("ELECTRO_ITERS_PORTFOLIO", "off")
os.environ.setdefault("ELECTRO_ITERS_PORTFOLIO_VAL", "1200")
# SDS-style compaction + soft shaping, ported from the temp branch (0c665b0,
# 87ce991: full-100 2.966 -> 2.722 there).  Repair-only, no extra place() call.
os.environ.setdefault("ELECTRO_COMPACT", "1")
# Jacobi graph-layout initialization mode (2026-07-18):
#   - "replace"   : Jacobi is the only init.  Cheapest, but no fallback when
#                   Jacobi lands in a bad basin (measured full-100: one case
#                   scored 4.86 vs 2.27 under random init -- see "hedge" below).
#   - "portfolio" : Both Random and Jacobi run a FULL 600-iter start; ranking
#                   picks the best.  Safe but pays for two complete place() calls.
#   - "hedge"     : Jacobi runs full iters as primary; Random runs SHORT
#                   (ELECTRO_HEDGE_ITERS) as a cheap fallback basin.  Combined
#                   with ELECTRO_EXPAND_TOPK below, this is the merged tree's
#                   picked default -- see the serial full-100 benchmark this was
#                   chosen from (2026-07-28, real evaluator, neutral RT):
#                     friend graph default:        2.0196 @ 6.26s/case
#                     merged replace:               2.0368 @ 4.08s/case
#                     merged hedge-300 + top-1:     1.9683 @ 5.50s/case  <- picked
#                   hedge-300+top-1 is the only config that beat the graph
#                   branch on BOTH score (-2.5%) and per-case runtime (-12.1%)
#                   in that run; replace is faster still if a tighter RT median
#                   turns out to matter more than the extra ~4% score.
#   - "off"       : Runs Random-init only (no Jacobi).
os.environ.setdefault("ELECTRO_JACOBI_MODE", "hedge")
os.environ.setdefault("ELECTRO_HEDGE_ITERS", "300")
# Top-K candidate pruning (2026-07-28): the push-past/wide-swap/compact repair
# variants CASCADE (each re-runs on the previous stage's output), so their cost
# grows with the number of raw starts, not with place()'s iteration count.
# Expanding only from the best-ranked raw start is what let hedge (2 raw
# starts) get down to portfolio-like scores without paying portfolio's full
# cascade cost.  0 restores the old "expand every start" behaviour.
os.environ.setdefault("ELECTRO_EXPAND_TOPK", "1")


from iccad2026_evaluate import FloorplanOptimizer
from legalize import verify_overlap
from soft_repair import soft_violation_counts
import electro_parallel


def _edges_np(b2b, p2b, pins, n):
    """Pull the valid (non-padding) edges/pins out as numpy for HPWL scoring."""
    def valid(t):
        if t is None or t.numel() == 0:
            return None
        a = t.cpu().numpy()
        a = a[a[:, 0] != -1]
        return a if len(a) else None
    return valid(b2b), valid(p2b), valid(pins)


def _hpwl(cx, cy, eb, ep, pv):
    """Contest HPWL (center-to-center Manhattan, b2b + p2b) for seed ranking."""
    n = len(cx)
    wl = 0.0
    if eb is not None:
        i = np.clip(eb[:, 0].astype(int), 0, n - 1)
        j = np.clip(eb[:, 1].astype(int), 0, n - 1)
        wl += float((eb[:, 2] * (np.abs(cx[i] - cx[j]) + np.abs(cy[i] - cy[j]))).sum())
    if ep is not None and pv is not None and len(pv):
        pi = np.clip(ep[:, 0].astype(int), 0, len(pv) - 1)
        bi = np.clip(ep[:, 1].astype(int), 0, n - 1)
        wl += float((ep[:, 2] * (np.abs(pv[pi, 0] - cx[bi]) + np.abs(pv[pi, 1] - cy[bi]))).sum())
    return wl


class MyOptimizer(FloorplanOptimizer):
    def __init__(self, verbose: bool = False):
        super().__init__(verbose=verbose)
        self.iters = int(os.environ.get("ELECTRO_ITERS", "600"))
        # CPU by default.  This is a SMALL problem (n<=120) run for 600 sequential
        # iterations of tiny ops, so a GPU is ~6x SLOWER here (kernel-launch
        # overhead dominates) -- and on a laptop it would run on the display GPU
        # and freeze the screen.  GPU only pays off with seed-BATCHING (TODO);
        # opt in then with ELECTRO_DEVICE=cuda.
        self.device = os.environ.get("ELECTRO_DEVICE", "cpu")
        self.lr = float(os.environ.get("ELECTRO_LR", "0.02"))
        # Rounds of (grouping_repair -> boundary_snap).  Each repair is now
        # min-displacement, but the two fight over blocks that are both a cluster
        # member AND a boundary block; one round leaves boundary blocks freshly
        # snapped off their cluster.  Iterating lets them settle: full-100 score
        # 3.733 (1 round) -> 3.568 (2) -> 3.545 (3) -> 3.545 (4, saturated).
        self.repair_rounds = int(os.environ.get("ELECTRO_REPAIR_ROUNDS", "3"))
        # Multi-start: keep the best of N seeds.  More seeds -> lower quality score
        # (subset 1->2.54, 3->2.16, 8->2.07 with ML) but ~Nx runtime.  The contest
        # runtime penalty (R^0.3, UNCAPPED on the slow side) usually makes seeds=1
        # win the runtime-adjusted total unless the field's median runtime is very
        # high.  Default 1 (fast); raise it when runtime is cheap / median is high.
        # Default 8 (2026-07-28): multi-start diversity turned out to be worth more
        # than any single-start refinement we had.  Measured full-100: an 8-seed
        # parallel multi-start scored 1.9025 @ 3.84s/case where our best
        # single-start pipeline (Jacobi + hedge + repair cascade) got 1.9683 @
        # 5.50s -- better on BOTH axes.  The cases it wins are exactly the ones a
        # single clever init loses: tid6 1.89 vs our 2.46, tid60 1.85 vs our 2.17,
        # i.e. the bad-basin cases the hedge fallback was built for and only
        # partially fixed.  8 independent basins simply covers them.
        self.seeds = int(os.environ.get("ELECTRO_SEEDS", "8"))
        # Multi-start seeds in parallel fork processes.  ON by default now: with a
        # PERSISTENT pool (see _get_pool) the fork happens once per evaluation
        # rather than once per case, so N seeds cost ~1 seed of wall-clock on an
        # N-core box.  That is what makes seeds=8 affordable at all -- the
        # runtime penalty R^0.3 is uncapped on the slow side, so 8x sequential
        # would have been a losing trade.  Workers are single-threaded
        # (ELECTRO_WORKER_THREADS) since n<=120 gives intra-op parallelism nothing
        # to chew on and N workers x M threads just oversubscribes.
        self.parallel = os.environ.get("ELECTRO_PARALLEL", "1") == "1"
        self._pool = None
        # ML warm-start: use the trained FloorplanTransformer's predicted block
        # centers as the analytical placer's init (instead of random).  Lazily
        # loaded; falls back to random init if weights/model are unavailable.
        #
        # DEFAULTED OFF (2026-07-28).  Full-100 with everything else in this file
        # held fixed (8-seed persistent-pool multistart + compaction + wide-swap +
        # grouping push-past + top-K pruning + fixed-reference ranking):
        #     ELECTRO_ML_INIT=1 (ML):      1.7741 @ 4.68s/case
        #     ELECTRO_ML_INIT=0 (Jacobi):  1.7312 @ 4.53s/case  <- picked default
        #     50/50 split of the 8 seeds:  1.7783 @ 4.54s/case  (WORSE than either
        #       pure variant on 15/100 cases -- mixing candidate types within one
        #       ranking pool is not simply "average of both", tried and rejected)
        # Jacobi wins on both axes AND drops the ml/predict.py + floorplan_v2.pt
        # dependency from the submission entirely.  ELECTRO_ML_INIT=1 restores it.
        self.ml_init = os.environ.get("ELECTRO_ML_INIT", "0") == "1"
        self._predictor = None
        # Split multi-start knob kept for future A/B; NOT the picked default (see
        # above -- measured worse than committing all seeds to Jacobi).
        self.init_split = os.environ.get("ELECTRO_INIT_SPLIT", "off")

    def solve(
        self,
        block_count: int,
        area_targets: torch.Tensor,
        b2b_connectivity: torch.Tensor,
        p2b_connectivity: torch.Tensor,
        pins_pos: torch.Tensor,
        constraints: torch.Tensor,
        target_positions: Optional[torch.Tensor] = None,
    ) -> List[Tuple[float, float, float, float]]:
        if block_count == 0:
            return []
        t0 = time.time()

        cons = constraints[:block_count].cpu().numpy()
        is_pre = (cons[:, 1] != 0).astype(bool)
        is_fixed = (cons[:, 0] != 0).astype(bool)
        is_soft = ~(is_fixed | is_pre)
        mib_id = cons[:, 2].astype(int) if cons.shape[1] > 2 else np.zeros(block_count, int)
        clust_id = cons[:, 3].astype(int) if cons.shape[1] > 3 else np.zeros(block_count, int)
        bcode = cons[:, 4].astype(int) if cons.shape[1] > 4 else np.zeros(block_count, int)
        eb, ep, pv = _edges_np(b2b_connectivity, p2b_connectivity, pins_pos, block_count)

        # ML warm-start only helps WITH multi-start (jitter around the prediction);
        # a single pure-ML start is worse than a single random start, so for
        # seeds==1 we use random init.
        use_ml = self.ml_init and self.seeds > 1
        init_centers = self._ml_centers(
            block_count, area_targets, constraints, target_positions,
            b2b_connectivity, p2b_connectivity, pins_pos) if use_ml else None

        nseeds = max(1, self.seeds)
        P = {
            "n": block_count, "area": area_targets, "b2b": b2b_connectivity,
            "p2b": p2b_connectivity, "pins": pins_pos, "cons": constraints,
            "tp": target_positions, "iters": self.iters, "lr": self.lr,
            "device": self.device, "init": init_centers, "is_pre": is_pre,
            "clust_id": clust_id, "mib_id": mib_id, "bcode": bcode, "rounds": self.repair_rounds,
            "nonneg": os.environ.get("ELECTRO_NONNEG", "0") == "1",
            "is_soft": is_soft,
        }

        # Multi-start: each seed lands in a different basin; run them in parallel
        # processes (independent -> embarrassingly parallel) and keep the
        # lowest-cost-proxy result.  Each worker is single-threaded by default:
        # the parent runs the ML model (initialising the OpenMP pool), so forked
        # workers that spin up >1 thread can deadlock (libgomp fork hazard).  On a
        # 48-core box this is fine -- run MANY single-thread seeds in parallel at
        # the same wall-clock.  ELECTRO_WORKER_THREADS>1 opts into multi-thread
        # workers (only safe if the parent never touched OpenMP, e.g. ML_INIT=0).
        # The solver can't see the GT baseline, so we rank by
        # exp(2*V_rel)*(hpwl/mean + area/mean), mirroring contest cost.  CUDA can't
        # be forked, so on GPU we run seeds sequentially (fast anyway).
        # Determine Jacobi Mode: "replace", "portfolio", or "off" (2026-07-18)
        jacobi_mode = os.environ.get("ELECTRO_JACOBI_MODE", "replace").lower()
        old_init = os.environ.get("ELECTRO_INIT", "random")
        if jacobi_mode in ("replace", "hedge"):
            os.environ["ELECTRO_INIT"] = "jacobi"
        else:
            os.environ["ELECTRO_INIT"] = "random"

        # Split multi-start: first half of seeds "ml", rest "jacobi".  With an
        # odd nseeds the extra seed goes to jacobi (it was the slightly stronger
        # of the two in isolation, marginally).
        split_methods = None
        if self.init_split == "half":
            n_ml = nseeds // 2
            split_methods = ["ml"] * n_ml + ["jacobi"] * (nseeds - n_ml)

        starts = None
        needs_extension = False
        try:
            if self.parallel and nseeds > 1 and self.device == "cpu":
                try:
                    # PERSISTENT pool + per-case pickling of P, rather than the old
                    # "build a Pool inside a with-block every case" path.  Forking a
                    # parent that holds torch is expensive; doing it 100x (once per
                    # case) cost far more than the pickling it avoided.  P is small
                    # here (n<=120, a few hundred edges), so shipping it to workers
                    # each case is cheap.
                    pool = self._get_pool(nseeds)
                    if split_methods is not None:
                        res = pool.starmap(electro_parallel.run_start_dispatch,
                                           [(s, P, split_methods[s]) for s in range(nseeds)])
                    else:
                        res = pool.starmap(electro_parallel.run_start_diag,
                                           [(s, P) for s in range(nseeds)])
                    starts = [r[0] for r in res]
                    needs_extension = any(r[1] for r in res)
                except Exception as e:
                    sys.stderr.write(f"[electro] parallel failed ({e}); sequential\n")
                    self._close_pool()
                    starts = None

            if starts is None:
                starts = []
                for s in range(nseeds):
                    if split_methods is not None:
                        layout, needs_ext = electro_parallel.run_start_dispatch(s, P, split_methods[s])
                    else:
                        layout, needs_ext = electro_parallel.run_start_diag(s, P)
                    starts.append(layout)
                    if needs_ext:
                        needs_extension = True

            # Generate portfolio variants of the 600-iter starts first
            cands_600 = []
            cand_sources = []
            base_source = "jacobi" if jacobi_mode in ("replace", "hedge") else "random"
            for s in starts:
                cands_600.append(s)
                cand_sources.append(base_source)

            # "hedge": Jacobi runs at full iters as the primary track, plus a SHORT
            # random-init track as a fallback basin.  Full "portfolio" mode pays for
            # two complete place() calls to buy that fallback; the random track only
            # needs to be good enough to win the ranking on the cases where Jacobi
            # lands badly, so it runs at a fraction of the iterations.
            # Only worth it for a SINGLE start.  With nseeds>1 the multi-start
            # already supplies independent basins -- that is the same job the
            # hedge track was doing, done better -- so paying for an extra short
            # place() per seed on top would be redundant work, and it runs in the
            # parent (serial) where it costs full wall-clock instead of riding the
            # pool's parallelism.
            if jacobi_mode == "hedge" and nseeds == 1:
                hedge_iters = int(os.environ.get(
                    "ELECTRO_HEDGE_ITERS", str(max(1, self.iters // 2))))
                for s in range(nseeds):
                    c = electro_parallel.run_start_random_with_iters(s, P, hedge_iters)
                    cands_600.append(c)
                    cand_sources.append("random")

            # Jacobi graph-layout init portfolio: add Jacobi-initialized candidates
            # as EXTRA portfolio entries ONLY in "portfolio" mode.
            if jacobi_mode == "portfolio":
                # Run Jacobi starts and get their needs_extension flag
                jacobi_starts = []
                for s in range(nseeds):
                    layout, needs_ext = electro_parallel.run_start_jacobi_diag(s, P)
                    jacobi_starts.append(layout)
                    if needs_ext:
                        needs_extension = True
                for s in jacobi_starts:
                    cands_600.append(s)
                    cand_sources.append("jacobi")

            # Number of genuine independent starts, before the repair-only
            # expansions below multiply the candidate list.
            n_starts = len(cands_600)

            # Prune before expanding.
            #
            # The repair variants below are the real runtime driver, not place():
            # each one re-runs the grouping/boundary repair loop, and they CASCADE
            # (wide-swap expands the push-past outputs too), so the work grows
            # multiplicatively in the number of starts.  Measured: going from 1
            # start (replace) to 2 (hedge/portfolio) cost 3.97s -> 6.8-7.2s, far
            # more than the extra place() alone accounts for.
            #
            # A repair variant grown from a start that already loses the ranking
            # essentially never becomes the overall winner, so rank the raw starts
            # first and expand only the best.  ELECTRO_EXPAND_TOPK=0 expands from
            # every start (the old behaviour).
            topk = int(os.environ.get("ELECTRO_EXPAND_TOPK", "1"))

            def _proxy_of(c):
                x, y, w, h = c
                vb, vg, vm, nsoft = soft_violation_counts(x, y, w, h, bcode,
                                                          clust_id, mib_id)
                hp = _hpwl(x + 0.5 * w, y + 0.5 * h, eb, ep, pv)
                ar = (float((x + w).max() - x.min())
                      * float((y + h).max() - y.min()))
                return np.exp(2.0 * (vb + vg + vm) / nsoft), hp, ar

            if 0 < topk < n_starts:
                sc = [_proxy_of(c) for c in cands_600[:n_starts]]
                rh0 = sc[0][1] or 1.0
                ra0 = sc[0][2] or 1.0
                order = sorted(range(n_starts),
                               key=lambda i: sc[i][0] * (sc[i][1] / rh0
                                                         + sc[i][2] / ra0))
                expand_idx = order[:topk]
            else:
                expand_idx = list(range(n_starts))

            # Working pool the cascade expands from -- the pruned starts only.
            # Losing starts stay in cands_600 as candidates, they just don't seed
            # further variants.
            pool = [cands_600[i] for i in expand_idx]
            pool_src = [cand_sources[i] for i in expand_idx]

            def _add(c, src):
                cands_600.append(c)
                cand_sources.append(src)
                pool.append(c)
                pool_src.append(src)

            if os.environ.get("ELECTRO_BOUNDARY_PUSHPAST", "0") == "1":
                for c, s in list(zip(pool, pool_src)):
                    _add(electro_parallel.boundary_pushpast_variant(c, P), s)
            if os.environ.get("ELECTRO_GROUPING_PUSHPAST", "0") == "1":
                has_vg = False
                for (x, y, w, h) in pool:
                    _, vg, _, _ = soft_violation_counts(x, y, w, h, bcode,
                                                        clust_id, mib_id)
                    if vg > 0:
                        has_vg = True
                        break
                if has_vg:
                    for c, s in list(zip(pool, pool_src)):
                        _add(electro_parallel.grouping_pushpast_variant(c, P), s)
            if os.environ.get("ELECTRO_BOUNDARY_WIDESWAP", "0") == "1":
                for c, s in list(zip(pool, pool_src)):
                    _add(electro_parallel.boundary_wideswap_variant(c, P), s)
            # SDS-style compaction (ported from temp branch): applied to the pruned
            # STARTS only, not to the push-past/wide-swap expansions.
            if os.environ.get("ELECTRO_COMPACT", "0") == "1":
                for i in expand_idx:
                    for aware in (False, True):
                        c = electro_parallel.compact_variant(cands_600[i], P,
                                                             aware=aware)
                        cands_600.append(c)
                        cand_sources.append(cand_sources[i])

            # Evaluate the 600-iter candidates to determine best_600_score and best_source
            cands_eval_600 = []
            for i, (x, y, w, h) in enumerate(cands_600):
                vb, vg, vm, nsoft = soft_violation_counts(x, y, w, h, bcode, clust_id, mib_id)
                vrel = (vb + vg + vm) / nsoft
                hpwl = _hpwl(x + 0.5 * w, y + 0.5 * h, eb, ep, pv)
                area = (float((x + w).max() - x.min()) * float((y + h).max() - y.min()))
                cands_eval_600.append((x, y, w, h, vrel, hpwl, area, cand_sources[i]))

            mh_600 = np.mean([c[5] for c in cands_eval_600]) or 1.0
            ma_600 = np.mean([c[6] for c in cands_eval_600]) or 1.0
            best_idx = min(range(len(cands_eval_600)), key=lambda i: np.exp(2.0 * cands_eval_600[i][4]) * (cands_eval_600[i][5] / mh_600 + cands_eval_600[i][6] / ma_600))
            best_600 = cands_eval_600[best_idx]
            best_600_score = np.exp(2.0 * best_600[4]) * (best_600[5] / mh_600 + best_600[6] / ma_600)
            best_source = best_600[7]

            # Decide whether to run iters=1200
            portfolio_mode = os.environ.get("ELECTRO_ITERS_PORTFOLIO", "adaptive")
            best_score_thresh = float(os.environ.get("ELECTRO_ADAPTIVE_SCORE_THRESH", "2.0"))
            
            run_iters_portfolio = False
            if portfolio_mode == "1":
                run_iters_portfolio = True
            elif portfolio_mode == "adaptive":
                run_iters_portfolio = needs_extension and (best_600_score >= best_score_thresh)

            # Final candidates list starts with the 600-iter candidates
            final_starts = list(cands_600)

            if run_iters_portfolio:
                custom_iters = int(os.environ.get("ELECTRO_ITERS_PORTFOLIO_VAL", "1200"))
                
                # Run 1200-iter ONLY for the winning source!
                starts_1200 = []
                if best_source == "random":
                    starts_1200 = [electro_parallel.run_start_with_iters(s, P, custom_iters) for s in range(nseeds)]
                else: # best_source == "jacobi"
                    starts_1200 = [electro_parallel.run_start_jacobi_with_iters(s, P, custom_iters) for s in range(nseeds)]

                # Generate portfolio variants of the 1200-iter starts
                cands_1200 = list(starts_1200)
                if os.environ.get("ELECTRO_BOUNDARY_PUSHPAST", "0") == "1":
                    cands_1200 = cands_1200 + [electro_parallel.boundary_pushpast_variant(s, P) for s in starts_1200]
                if os.environ.get("ELECTRO_GROUPING_PUSHPAST", "0") == "1":
                    has_vg = False
                    for (x, y, w, h) in cands_1200:
                        _, vg, _, _ = soft_violation_counts(x, y, w, h, bcode, clust_id, mib_id)
                        if vg > 0:
                            has_vg = True
                            break
                    if has_vg:
                        cands_1200 = cands_1200 + [electro_parallel.grouping_pushpast_variant(s, P) for s in cands_1200]
                if os.environ.get("ELECTRO_BOUNDARY_WIDESWAP", "0") == "1":
                    cands_1200 = cands_1200 + [electro_parallel.boundary_wideswap_variant(s, P) for s in cands_1200]
                    
                final_starts = final_starts + cands_1200
        finally:
            os.environ["ELECTRO_INIT"] = old_init


        cands = []
        for (x, y, w, h) in final_starts:
            vb, vg, vm, nsoft = soft_violation_counts(x, y, w, h, bcode, clust_id, mib_id)
            vrel = (vb + vg + vm) / nsoft
            hpwl = _hpwl(x + 0.5 * w, y + 0.5 * h, eb, ep, pv)
            area = (float((x + w).max() - x.min()) * float((y + h).max() - y.min()))
            cands.append((x, y, w, h, vrel, hpwl, area))

        # Candidate ranking reference.
        #
        # The original normalisers were the MEAN hpwl/area over the candidate set,
        # which makes the ranking function itself depend on which candidates are
        # present: adding one candidate rescales the hpwl-vs-area trade-off for
        # every other candidate and can flip the winner among the pre-existing
        # ones.  That silently breaks the "strictly additive portfolio" contract
        # every variant in this file relies on -- a portfolio can only ever be
        # additive with respect to a FIXED ranking function.  Measured cost of the
        # bug on the full 100: enabling more portfolios moved 17 cases the WRONG
        # way (tid6 +80.9%, tid54 +52.2%, tid23 +36.1%).
        #
        # Fix: normalise by the FIRST candidate (the plain start, always present
        # and independent of what else got appended).  Same functional form, but
        # the reference no longer moves, so min() over a superset is genuinely
        # no worse than over a subset.  ELECTRO_RANK_REF=mean restores the old
        # behaviour for A/B.
        if os.environ.get("ELECTRO_RANK_REF", "fixed") == "mean":
            rh = np.mean([c[5] for c in cands]) or 1.0
            ra = np.mean([c[6] for c in cands]) or 1.0
        else:
            rh = cands[0][5] or 1.0
            ra = cands[0][6] or 1.0
        x, y, w, h, vrel, hpwl, area = min(
            cands, key=lambda c: np.exp(2.0 * c[4]) * (c[5] / rh + c[6] / ra))

        ov = verify_overlap(x, y, w, h)
        soft = ((cons[:, 0] == 0) & (cons[:, 1] == 0))
        at = area_targets[:block_count].cpu().numpy()
        drift = np.abs(w * h - at) / np.maximum(at, 1e-9)
        max_drift = float(drift[soft].max()) if soft.any() else 0.0
        dt = time.time() - t0
        sys.stderr.write(
            f"[electro] n={block_count} t={dt:.3f}s seeds={self.seeds} "
            f"resid_overlap={ov:.3g} max_area_drift={max_drift:.4f} "
            f"V_rel={vrel:.3f}\n"
        )
        return [(float(x[i]), float(y[i]), float(w[i]), float(h[i]))
                for i in range(block_count)]

    def _get_pool(self, nseeds):
        """One fork pool shared by the whole evaluation, created on first use.

        Deliberately lazy rather than built in __init__: fork is only safe while
        the parent is not inside an OpenMP parallel region (libgomp's fork hazard
        -- the child inherits a thread-pool state that never unlocks and hangs).
        On the first solve() call the parent has only imported torch, not run any
        tensor math, which is the safe moment; workers are pinned single-threaded
        afterwards so no further fork is ever needed.
        """
        if self._pool is None:
            nproc = max(1, min(nseeds, os.cpu_count() or 1))
            threads = int(os.environ.get("ELECTRO_WORKER_THREADS", "1"))
            ctx = mp.get_context("fork")
            self._pool = ctx.Pool(nproc, initializer=electro_parallel.pool_init,
                                  initargs=(threads,))
        return self._pool

    def _close_pool(self):
        """Drop the pool so a failure falls back to sequential cleanly, and the
        next case gets a fresh pool instead of repeating a wedged one."""
        if self._pool is not None:
            try:
                self._pool.terminate()
                self._pool.join()
            except Exception:
                pass
            self._pool = None

    def __del__(self):
        self._close_pool()

    def _ml_centers(self, block_count, area_targets, constraints, target_positions,
                    b2b, p2b, pins):
        """Predicted block centers [N,2] (raw coords) from the trained model, or
        None if the model/weights are unavailable or the case is too big."""
        if self._predictor is None:
            try:
                # Find the `ml/` package: explicit env override, then bundled next
                # to this file (submission layout), then the PARENT directory (the
                # dev-tree layout, where ml/ sits beside electro/).  All paths are
                # relative to __file__ -- no machine-specific absolute paths.
                here = os.path.dirname(os.path.abspath(__file__))
                ml_dir = None
                for d in (os.environ.get("ELECTRO_ML_DIR"), here,
                          os.path.dirname(here)):
                    if d and os.path.isdir(os.path.join(d, "ml")):
                        ml_dir = d
                        break
                if ml_dir is None:
                    raise FileNotFoundError("ml/ package not found")
                if ml_dir not in sys.path:
                    sys.path.insert(0, ml_dir)
                from ml.predict import Predictor
                wts = os.environ.get(
                    "ELECTRO_ML_WEIGHTS",
                    os.path.join(ml_dir, "ml", "weights", "floorplan_v2.pt"))
                self._predictor = Predictor(wts, device=self.device)
            except Exception as e:
                sys.stderr.write(f"[electro] ML init unavailable: {e}\n")
                self._predictor = False
        if not self._predictor:
            return None
        try:
            pred = self._predictor.predict(
                block_count, area_targets, constraints, target_positions,
                b2b, p2b, pins)
            if pred is None:
                return None
            return torch.tensor([[p[0], p[1]] for p in pred.positions],
                                dtype=torch.float32)
        except Exception as e:
            sys.stderr.write(f"[electro] ML predict failed: {e}\n")
            return None

