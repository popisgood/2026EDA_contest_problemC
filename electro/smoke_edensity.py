#!/usr/bin/env python3
"""Smoke test for the eDensity term: run place() on a few cases with eDensity
on, print achieved bbox-util and min corner (negative-coord check). No legalize."""
import math
import os
import sys

import torch
from litetestLoader import FloorplanDatasetLiteTest

sys.path.insert(0, "/home/pop/2026_EDA_contest/electro")
import analytical_place as ap

ds = FloorplanDatasetLiteTest("../")
SUBSET = [0, 60, 99]


def build_tp(cons, polys, n):
    tp = torch.full((n, 4), -1.0)
    for i in range(n):
        blk = polys[i]
        v = blk[blk[:, 0] != -1]
        if len(v) == 0:
            continue
        mn = v.min(dim=0).values
        mx = v.max(dim=0).values
        x, y, w, h = float(mn[0]), float(mn[1]), float(mx[0] - mn[0]), float(mx[1] - mn[1])
        is_fixed = cons[i, 0] != 0
        is_pre = cons[i, 1] != 0
        if is_pre:
            tp[i] = torch.tensor([x, y, w, h])
        elif is_fixed:
            tp[i, 2] = w
            tp[i, 3] = h
    return tp


def run(tid, edw):
    s = ds[tid]
    area_t, b2b, p2b, pins, cons = s['input']
    polys, _ = s['label']
    n = int((area_t != -1).sum().item())
    tp = build_tp(cons, polys, n)
    os.environ["ELECTRO_EDENSITY"] = str(edw)
    out, diag = ap.place(n, area_t, b2b, p2b, pins, cons, tp,
                         iters=600, lr=0.02, seed=0, device="cpu")
    xs = [o[0] for o in out]
    ys = [o[1] for o in out]
    xr = [o[0] + o[2] for o in out]
    yr = [o[1] + o[3] for o in out]
    bbox = (max(xr) - min(xs)) * (max(yr) - min(ys))
    tot = float(sum(area_t[i] for i in range(n) if area_t[i] > 0))
    util = tot / bbox
    print(f"tid {tid:>3} ed={edw:>6} | minx={min(xs):8.2f} miny={min(ys):8.2f} "
          f"| bbox_util={util:5.3f} | hpwl={diag['hpwl']:.1f} "
          f"| ov%={diag['overlap_pct']:.1f}")


for tid in SUBSET:
    run(tid, 0.0)
    for w in (1.0, 10.0, 100.0, 1000.0):
        run(tid, w)
    print()
