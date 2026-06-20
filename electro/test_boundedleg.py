#!/usr/bin/env python3
"""Prototype: can a fixed-OUTLINE legalizer remove overlap WITHOUT expanding past
the canvas?  Take the eDensity place() output (tight, confined, small overlaps)
and run a bounded push-apart that keeps every movable block inside [0,Wc]x[0,Hc].
Report util / residual overlap / convergence vs the existing (expanding) legalize.
"""
import math
import os
import sys

import numpy as np
import torch
from litetestLoader import FloorplanDatasetLiteTest

sys.path.insert(0, "/home/pop/2026_EDA_contest/electro")
import analytical_place as ap
from legalize import legalize, verify_overlap

ds = FloorplanDatasetLiteTest("../")


def build_tp(cons, polys, n):
    tp = torch.full((n, 4), -1.0)
    for i in range(n):
        v = polys[i][polys[i][:, 0] != -1]
        if len(v) == 0:
            continue
        mn = v.min(dim=0).values; mx = v.max(dim=0).values
        x, y, w, h = float(mn[0]), float(mn[1]), float(mx[0]-mn[0]), float(mx[1]-mn[1])
        if cons[i, 1] != 0:
            tp[i] = torch.tensor([x, y, w, h])
        elif cons[i, 0] != 0:
            tp[i, 2] = w; tp[i, 3] = h
    return tp


def bounded_cleanup(x, y, w, h, is_pre, Wc, Hc, max_iter=8000):
    """Push-apart overlap removal that keeps movable blocks inside [0,Wc]x[0,Hc].
    Excess push that would cross a wall is redirected to the partner block."""
    x = x.copy(); y = y.copy()
    mv = ~is_pre
    x[mv] = np.clip(x[mv], 0.0, np.maximum(0.0, Wc - w[mv]))
    y[mv] = np.clip(y[mv], 0.0, np.maximum(0.0, Hc - h[mv]))
    N = len(x)
    iu = np.triu_indices(N, 1)
    for _ in range(max_iter):
        cx = x + 0.5*w; cy = y + 0.5*h
        ox = 0.5*(w[:, None]+w[None, :]) - np.abs(cx[:, None]-cx[None, :])
        oy = 0.5*(h[:, None]+h[None, :]) - np.abs(cy[:, None]-cy[None, :])
        m = (ox[iu] > 1e-6) & (oy[iu] > 1e-6)
        I, J = iu[0][m], iu[1][m]
        if len(I) == 0:
            return x, y, True
        for k in range(len(I)):
            i, j = int(I[k]), int(J[k])
            oxk = 0.5*(w[i]+w[j]) - abs((x[i]+0.5*w[i])-(x[j]+0.5*w[j]))
            oyk = 0.5*(h[i]+h[j]) - abs((y[i]+0.5*h[i])-(y[j]+0.5*h[j]))
            if oxk <= 0 or oyk <= 0:
                continue
            if oxk <= oyk:
                _sep(x, w, i, j, oxk+1e-6, is_pre, Wc)
            else:
                _sep(y, h, i, j, oyk+1e-6, is_pre, Hc)
    return x, y, verify_overlap(x, y, w, h) < 1e-6


def _sep(c, sz, i, j, move, is_pre, wall):
    lo, hi = (i, j) if c[i] <= c[j] else (j, i)
    if is_pre[lo] and is_pre[hi]:
        return
    # try symmetric, then redirect at either wall
    nlo = c[lo] - 0.5*move
    nhi = c[hi] + 0.5*move
    if is_pre[lo]:
        nlo = c[lo]; nhi = c[hi] + move
    elif is_pre[hi]:
        nhi = c[hi]; nlo = c[lo] - move
    if nlo < 0:
        deficit = -nlo; nlo = 0.0; nhi += deficit
    top = wall - sz[hi]
    if nhi > top:
        excess = nhi - top; nhi = top; nlo -= excess
        if nlo < 0:
            nlo = 0.0  # box too tight locally; leave (may keep tiny overlap)
    if not is_pre[lo]:
        c[lo] = nlo
    if not is_pre[hi]:
        c[hi] = nhi


for tid in [0, 60, 99]:
    s = ds[tid]
    area_t, b2b, p2b, pins, cons = s['input']
    polys, _ = s['label']
    n = int((area_t != -1).sum().item())
    tp = build_tp(cons, polys, n)
    tot = float(sum(area_t[i] for i in range(n) if area_t[i] > 0))
    is_pre = (cons[:n, 1] != 0).numpy().astype(bool)
    for u in [0.85, 0.90]:
        os.environ["ELECTRO_EDENSITY"] = "2"
        os.environ["ELECTRO_EDENSITY_UTIL"] = str(u)
        os.environ["ELECTRO_OV1"] = "30"
        os.environ["ELECTRO_OV0"] = "2.0"
        out, _ = ap.place(n, area_t, b2b, p2b, pins, cons, tp,
                          iters=600, lr=0.02, seed=0, device="cpu")
        x = np.array([o[0] for o in out]); y = np.array([o[1] for o in out])
        w = np.array([o[2] for o in out]); h = np.array([o[3] for o in out])

        def util(x, y):
            bb = (max(x+w)-min(x))*(max(y+h)-min(y)); return tot/bb
        u_place = util(x, y); ov_place = verify_overlap(x, y, w, h)
        # canvas dims (recompute aspect from pins, same as place())
        pv = pins[pins[:, 0] != -1].float()
        asp = 1.0
        if len(pv) >= 2:
            pwx = float(pv[:,0].max()-pv[:,0].min()); pwy = float(pv[:,1].max()-pv[:,1].min())
            asp = min(max(pwx/max(pwy,1e-9), 0.25), 4.0)
        Hc = math.sqrt(tot/u/asp); Wc = asp*Hc
        # existing (expanding) legalizer
        xe, ye = legalize(x, y, w, h, is_pre)
        # bounded legalizer
        xb, yb, conv = bounded_cleanup(x, y, w, h, is_pre, Wc, Hc)
        print(f"tid {tid:>3} u={u} | place util={u_place:.3f} ov={ov_place:.2g} "
              f"minxy=({min(x):.1f},{min(y):.1f}) || "
              f"EXIST util={util(xe,ye):.3f} minxy=({min(xe):.1f},{min(ye):.1f}) || "
              f"BOUND util={util(xb,yb):.3f} ov={verify_overlap(xb,yb,w,h):.2g} "
              f"conv={conv} minxy=({min(xb):.1f},{min(yb):.1f})")
