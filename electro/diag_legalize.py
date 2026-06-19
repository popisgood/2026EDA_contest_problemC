#!/usr/bin/env python3
"""Diagnostic: measure HPWL gap contributed by each pipeline stage.

For each test-id, run place() then legalize() then the repairs, and report the
contest HPWL (center-to-center, b2b+p2b) at each stage vs the GT baseline, so we
can see how much of hpwl_gap is global-placement vs legalization displacement.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/home/pop/IntelLabs_Floorset/FloorSet/iccad2026contest")
sys.path.insert(0, "/home/pop/IntelLabs_Floorset/FloorSet")
os.chdir("/home/pop/IntelLabs_Floorset/FloorSet/iccad2026contest")

import numpy as np
import torch

from litetestLoader import FloorplanDatasetLiteTest
from iccad2026_evaluate import (
    calculate_hpwl_b2b, calculate_hpwl_p2b, calculate_bbox_area, check_overlap,
)
from analytical_place import place
from legalize import legalize, remove_overlap, verify_overlap
from soft_repair import boundary_snap, grouping_repair, soft_violation_counts


def hpwl(pos, b2b, p2b, pins):
    return calculate_hpwl_b2b(pos, b2b) + calculate_hpwl_p2b(pos, p2b, pins)


def baseline_hpwl(labels, b2b, p2b, pins, n):
    polygons, metrics = labels
    positions = []
    for i in range(n):
        block = polygons[i]
        valid = block[block[:, 0] != -1]
        xmin, ymin = valid.min(dim=0).values
        xmax, ymax = valid.max(dim=0).values
        positions.append((float(xmin), float(ymin), float(xmax - xmin), float(ymax - ymin)))
    hb = calculate_hpwl_b2b(positions, b2b)
    hp = calculate_hpwl_p2b(positions, p2b, pins)
    area = calculate_bbox_area(positions)
    if metrics is not None and len(metrics) >= 8:
        if metrics[0] > 0:
            area = float(metrics[0])
        if metrics[-2] > 0:
            hb = float(metrics[-2])
        if metrics[-1] >= 0:
            hp = float(metrics[-1])
    return hb + hp, area


def to_pos(x, y, w, h):
    return [(float(x[i]), float(y[i]), float(w[i]), float(h[i])) for i in range(len(x))]


def cost(hg, ag, vr):
    import math
    return (1 + 0.5 * (max(0, hg) + max(0, ag))) * math.exp(2.0 * vr)


def main():
    ds = FloorplanDatasetLiteTest("../")

    ids = [int(a) for a in sys.argv[1:]] or [0, 30, 70, 99]
    print(f"{'tid':>4} {'n':>4} | {'r1':>8} | {'r2':>8} | {'r3':>8} | {'r4':>8}")
    totw = {k: 0.0 for k in "1234"}
    wsum = 0.0
    for idx in ids:
        sample = ds[idx]
        inputs, labels = sample['input'], sample['label']
        a, b2b, p2b, pins, cons = inputs
        n = int((a != -1).sum().item())
        bh, area_base = baseline_hpwl(labels, b2b, p2b, pins, n)

        # build opt_target_pos like the evaluator
        polygons = labels[0]
        tp = torch.full((n, 4), -1.0)
        nc = cons.shape[1] if cons.dim() > 1 else 0
        for i in range(n):
            isf = nc > 0 and cons[i, 0] != 0
            isp = nc > 1 and cons[i, 1] != 0
            if isf or isp:
                block = polygons[i]
                valid = block[block[:, 0] != -1]
                xmin, ymin = valid.min(dim=0).values
                xmax, ymax = valid.max(dim=0).values
                tw, th = float(xmax - xmin), float(ymax - ymin)
                if isp:
                    tp[i] = torch.tensor([float(xmin), float(ymin), tw, th])
                else:
                    tp[i, 2], tp[i, 3] = tw, th

        positions, _ = place(n, a, b2b, p2b, pins, cons, tp, iters=600, lr=0.02)
        x = np.array([p[0] for p in positions]); y = np.array([p[1] for p in positions])
        w = np.array([p[2] for p in positions]); h = np.array([p[3] for p in positions])
        h_place = hpwl(to_pos(x, y, w, h), b2b, p2b, pins)
        cx0, cy0 = x + 0.5 * w, y + 0.5 * h

        cons_np = cons[:n].cpu().numpy()
        is_pre = (cons_np[:, 1] != 0).astype(bool)
        mib = cons_np[:, 2].astype(int) if cons_np.shape[1] > 2 else np.zeros(n, int)
        clust = cons_np[:, 3].astype(int) if cons_np.shape[1] > 3 else np.zeros(n, int)
        bcode = cons_np[:, 4].astype(int) if cons_np.shape[1] > 4 else np.zeros(n, int)

        g = lambda v: (v - bh) / max(bh, 1e-6)
        ag = lambda xx, yy: (calculate_bbox_area(to_pos(xx, yy, w, h)) - area_base) / max(area_base, 1e-6)

        def vrel(xx, yy):
            vb, vg, vm, ns = soft_violation_counts(xx, yy, w, h, bcode, clust, mib)
            return (vb + vg + vm) / ns

        xl, yl = legalize(x.copy(), y.copy(), w, h, is_pre)

        def G(xx, yy):
            return grouping_repair(xx.copy(), yy.copy(), w, h, clust, is_pre)

        def Bd(xx, yy):
            return boundary_snap(xx.copy(), yy.copy(), w, h, bcode, is_pre)

        def score_state(xx, yy):
            xx, yy = remove_overlap(xx.copy(), yy.copy(), w, h, is_pre)
            return cost(g(hpwl(to_pos(xx, yy, w, h), b2b, p2b, pins)), ag(xx, yy), vrel(xx, yy))

        wt = np.exp(n / 12.0)
        wsum += wt
        cx_, cy_ = xl, yl
        cs = []
        for r in range(4):                 # r rounds of (grouping -> boundary)
            cx_, cy_ = Bd(*G(cx_, cy_))
            cs.append(score_state(cx_, cy_))
        for k, c in zip("1234", cs):
            totw[k] += c * wt
        print(f"{idx:>4} {n:>4} | " + " | ".join(f"{c:>8.3f}" for c in cs))

    print("-" * 50)
    print(f"{'WTD':>4} {'':>4} | " + " | ".join(f"{totw[k]/wsum:>8.4f}" for k in "1234"))


if __name__ == "__main__":
    main()
