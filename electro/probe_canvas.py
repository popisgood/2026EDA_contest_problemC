#!/usr/bin/env python3
"""One-off probe: dump GT die geometry vs our normalization for a few cases.

Run from the contest dir with the FloorSet venv python so the loaders import:
    cd ~/IntelLabs_Floorset/FloorSet/iccad2026contest
    /home/pop/IntelLabs_Floorset/FloorSet/venv/bin/python \
        /home/pop/2026_EDA_contest/electro/probe_canvas.py
"""
import math
import torch
from litetestLoader import FloorplanDatasetLiteTest

SUBSET = [0, 20, 40, 55, 60, 65, 70, 75, 80, 85, 90, 92, 95, 97, 99]

ds = FloorplanDatasetLiteTest("../")
print(f"loaded {len(ds)} cases")
print(f"{'tid':>4} {'n':>4} {'#pre':>4} {'GT_ur':>16} "
      f"{'GT_util':>7} {'GTasp':>6} {'pin_ll':>14} {'pin_ur':>16} {'pinasp':>6}")
for tid in SUBSET:
    s = ds[tid]
    area_t, b2b, p2b, pins, cons = s['input']
    polys, _ = s['label']
    n = int((area_t != -1).sum().item())
    pos = []
    for i in range(n):
        blk = polys[i]
        v = blk[blk[:, 0] != -1]
        if len(v):
            mn = v.min(dim=0).values
            mx = v.max(dim=0).values
            pos.append((float(mn[0]), float(mn[1]),
                        float(mx[0] - mn[0]), float(mx[1] - mn[1])))
        else:
            pos.append((0, 0, 1, 1))
    xs0 = min(p[0] for p in pos); ys0 = min(p[1] for p in pos)
    xs1 = max(p[0] + p[2] for p in pos); ys1 = max(p[1] + p[3] for p in pos)
    gt_bbox = (xs1 - xs0) * (ys1 - ys0)
    tot_area = float(sum(area_t[i] for i in range(n) if area_t[i] > 0))
    gt_util = tot_area / max(gt_bbox, 1e-9)
    S = math.sqrt(tot_area)
    L = math.sqrt(tot_area / 0.85)
    # pin terminal bbox (fixed external terminals) -- candidate die-aspect proxy
    pll = pur = None
    pasp = gasp = 0.0
    if pins is not None and pins.numel():
        pv = pins[pins[:, 0] != -1]
        if len(pv):
            pll = (float(pv[:, 0].min()), float(pv[:, 1].min()))
            pur = (float(pv[:, 0].max()), float(pv[:, 1].max()))
            pw = pur[0] - pll[0]; ph = pur[1] - pll[1]
            pasp = pw / max(ph, 1e-9)
    gasp = (xs1 - xs0) / max(ys1 - ys0, 1e-9)
    npre = int((cons[:n, 1] != 0).sum().item()) if cons.dim() > 1 else 0
    # are preplaced blocks inside the GT die box?
    pin = "-"
    if npre:
        idxs = [i for i in range(n) if cons[i, 1] != 0]
        pin = all(xs0 - 1e-3 <= pos[i][0] and pos[i][0] + pos[i][2] <= xs1 + 1e-3
                  and ys0 - 1e-3 <= pos[i][1] and pos[i][1] + pos[i][3] <= ys1 + 1e-3
                  for i in idxs)
    ps = f"({pll[0]:5.0f},{pll[1]:5.0f})" if pll else "       -"
    pe = f"({pur[0]:6.0f},{pur[1]:6.0f})" if pur else "        -"
    print(f"{tid:>4} {n:>4} {npre:>4} ({xs1:6.1f},{ys1:6.1f}) "
          f"{gt_util:7.3f} {gasp:6.2f} {ps:>14} {pe:>16} {pasp:6.2f}")
