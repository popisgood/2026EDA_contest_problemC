#!/usr/bin/env python3
"""Aggregate stats over the 100 FloorSet-Lite validation cases for the report."""
import numpy as np
from litetestLoader import FloorplanDatasetLiteTest
ds = FloorplanDatasetLiteTest("../")

rows = []
for tid in range(len(ds)):
    s = ds[tid]
    area_t, b2b, p2b, pins, cons = s['input']
    polys, _ = s['label']
    n = int((area_t != -1).sum().item())
    c = cons[:n]
    nfix = int((c[:, 0] != 0).sum())
    npre = int((c[:, 1] != 0).sum())
    nsoft = n - nfix - npre
    nmib = int(c[:, 2].max()) if c.shape[1] > 2 else 0
    nclu = int(c[:, 3].max()) if c.shape[1] > 3 else 0
    nbnd = int((c[:, 4] != 0).sum()) if c.shape[1] > 4 else 0
    nb2b = int((b2b[:, 0] != -1).sum()) if b2b.numel() else 0
    npin = int((pins[:, 0] != -1).sum()) if pins is not None and pins.numel() else 0
    np2b = int((p2b[:, 0] != -1).sum()) if p2b is not None and p2b.numel() else 0
    # GT bbox util
    pos = []
    for i in range(n):
        v = polys[i][polys[i][:, 0] != -1]
        if len(v):
            mn = v.min(0).values; mx = v.max(0).values
            pos.append((float(mn[0]), float(mn[1]), float(mx[0]-mn[0]), float(mx[1]-mn[1])))
    bb = (max(p[0]+p[2] for p in pos)-min(p[0] for p in pos))*(max(p[1]+p[3] for p in pos)-min(p[1] for p in pos))
    tot = float(sum(area_t[i] for i in range(n) if area_t[i] > 0))
    util = tot/bb
    rows.append((tid, n, nsoft, nfix, npre, nb2b, npin, np2b, nmib, nclu, nbnd, util))

A = np.array([r[1:] for r in rows], float)
labels = ["n","soft","fixed","pre","b2b","pins","p2b","mib_grp","clusters","bnd_blk","GT_util"]
print("=== overall (100 cases) ===")
for j, lab in enumerate(labels):
    col = A[:, j]
    print(f"{lab:>9}: min={col.min():.3g} max={col.max():.3g} mean={col.mean():.3g} median={np.median(col):.3g}")
print(f"\ntotal b2b edges across all cases: {int(A[:,4].sum())}")
print(f"cases with >=1 preplaced: {int((A[:,3]>=1).sum())}/100")
print(f"cases with >=1 fixed:     {int((A[:,2]>=1).sum())}/100")
print(f"cases with MIB groups:    {int((A[:,7]>=1).sum())}/100")
print(f"cases with clusters:      {int((A[:,8]>=1).sum())}/100")
print(f"cases with boundary blks: {int((A[:,9]>=1).sum())}/100")

# n buckets
print("\n=== block-count buckets ===")
import collections
buck = collections.Counter()
for r in rows:
    n = r[1]
    b = (n//20)*20
    buck[b]+=1
for b in sorted(buck):
    print(f"  n={b:>3}-{b+19:>3}: {buck[b]} cases")

# the 15-case screening subset detail
SUB = [0,20,40,55,60,65,70,75,80,85,90,92,95,97,99]
print("\n=== screening subset (15 cases) ===")
print(f"{'tid':>4}{'n':>5}{'soft':>5}{'fix':>4}{'pre':>4}{'b2b':>5}{'pins':>5}{'mib':>4}{'clu':>4}{'bnd':>4}{'util':>7}")
for r in rows:
    if r[0] in SUB:
        print(f"{r[0]:>4}{r[1]:>5}{r[2]:>5}{r[3]:>4}{r[4]:>4}{r[5]:>5}{r[6]:>5}{r[8]:>4}{r[9]:>4}{r[10]:>4}{r[11]:>7.3f}")
