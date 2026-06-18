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

import os
import sys
import time
from typing import List, Optional, Tuple

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from iccad2026_evaluate import FloorplanOptimizer
from analytical_place import place
from legalize import legalize, remove_overlap, verify_overlap
from soft_repair import boundary_snap, grouping_repair, soft_violation_counts


class MyOptimizer(FloorplanOptimizer):
    def __init__(self, verbose: bool = False):
        super().__init__(verbose=verbose)
        self.iters = int(os.environ.get("ELECTRO_ITERS", "600"))
        self.device = os.environ.get("ELECTRO_DEVICE", "cpu")
        self.lr = float(os.environ.get("ELECTRO_LR", "0.02"))

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
        positions, _ = place(
            block_count, area_targets, b2b_connectivity, p2b_connectivity,
            pins_pos, constraints, target_positions,
            iters=self.iters, lr=self.lr, device=self.device,
        )

        x = np.array([p[0] for p in positions], dtype=float)
        y = np.array([p[1] for p in positions], dtype=float)
        w = np.array([p[2] for p in positions], dtype=float)
        h = np.array([p[3] for p in positions], dtype=float)

        cons = constraints[:block_count].cpu().numpy()
        is_pre = (cons[:, 1] != 0).astype(bool)
        mib_id = cons[:, 2].astype(int) if cons.shape[1] > 2 else np.zeros(block_count, int)
        clust_id = cons[:, 3].astype(int) if cons.shape[1] > 3 else np.zeros(block_count, int)
        bcode = cons[:, 4].astype(int) if cons.shape[1] > 4 else np.zeros(block_count, int)

        # legalize (zero overlap) -> grouping repair -> boundary snap
        x, y = legalize(x, y, w, h, is_pre)
        x, y = grouping_repair(x, y, w, h, clust_id, is_pre)
        x, y = boundary_snap(x, y, w, h, bcode, is_pre)
        x, y = remove_overlap(x, y, w, h, is_pre)   # final hard-feasibility net
        ov = verify_overlap(x, y, w, h)

        vb, vg, vm, nsoft = soft_violation_counts(x, y, w, h, bcode, clust_id, mib_id)
        soft = ((cons[:, 0] == 0) & (cons[:, 1] == 0))
        at = area_targets[:block_count].cpu().numpy()
        drift = np.abs(w * h - at) / np.maximum(at, 1e-9)
        max_drift = float(drift[soft].max()) if soft.any() else 0.0
        dt = time.time() - t0
        sys.stderr.write(
            f"[electro] n={block_count} t={dt:.3f}s "
            f"resid_overlap={ov:.3g} max_area_drift={max_drift:.4f} "
            f"V_bnd={vb} V_grp={vg} V_mib={vm} V_rel={(vb+vg+vm)/nsoft:.3f}\n"
        )
        return [(float(x[i]), float(y[i]), float(w[i]), float(h[i]))
                for i in range(block_count)]
