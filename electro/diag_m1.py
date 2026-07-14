#!/usr/bin/env python3
"""Diagnose the M1 rollout IN ISOLATION (no ranking): does it complete? how does
its raw quality compare to GT and to electro's chosen candidate (from the last
validate_m1.sh run)?  Answers "why isn't M1 winning the ranking yet" directly,
instead of guessing from the final cost alone.
"""
from __future__ import annotations
import os, sys
import numpy as np
import torch

FLOORSET = "/home/pop/IntelLabs_Floorset/FloorSet"
EVALDIR = FLOORSET + "/iccad2026contest"
REPO = "/home/pop/2026_EDA_contest"
ELECTRO = REPO + "/electro"
for p in (FLOORSET, EVALDIR, ELECTRO, REPO):
    if p not in sys.path:
        sys.path.insert(0, p)

import iccad2026_evaluate as ev
from ml.m1_infer import M1Predictor
from soft_repair import soft_violation_counts
import electro_parallel

IDS = [int(x) for x in os.environ.get("IDS", "0,20,40,60,80,99").split(",")]
WEIGHTS = os.environ.get("WEIGHTS", REPO + "/ml/weights/m1_v1.pt")


def _edges(t):
    if t is None or t.numel() == 0:
        return None
    a = t.cpu().numpy()
    a = a[a[:, 0] != -1]
    return a if len(a) else None


def hpwl(cx, cy, eb, ep, pv):
    v = 0.0
    if eb is not None:
        i, j = eb[:, 0].astype(int), eb[:, 1].astype(int)
        v += float((eb[:, 2] * (np.abs(cx[i] - cx[j]) + np.abs(cy[i] - cy[j]))).sum())
    if ep is not None and pv is not None:
        pi, bi = ep[:, 0].astype(int), ep[:, 1].astype(int)
        v += float((ep[:, 2] * (np.abs(pv[pi, 0] - cx[bi]) + np.abs(pv[pi, 1] - cy[bi]))).sum())
    return v


def overlap_pct(x, y, w, h):
    n = len(x)
    tot = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            ox = min(x[i] + w[i], x[j] + w[j]) - max(x[i], x[j])
            oy = min(y[i] + h[i], y[j] + h[j]) - max(y[i], y[j])
            if ox > 1e-7 and oy > 1e-7:
                tot += ox * oy
    return 100.0 * tot / max((w * h).sum(), 1e-9)


def metrics(x, y, w, h, cons_np, eb, ep, pv, gt_area, gt_hpwl):
    ov = overlap_pct(x, y, w, h)
    area = float((x + w).max() - x.min()) * float((y + h).max() - y.min())
    cx, cy = x + 0.5 * w, y + 0.5 * h
    hp = hpwl(cx, cy, eb, ep, pv)
    bcode = cons_np[:, 4].astype(int) if cons_np.shape[1] > 4 else np.zeros(len(x), int)
    clust = cons_np[:, 3].astype(int) if cons_np.shape[1] > 3 else np.zeros(len(x), int)
    mib = cons_np[:, 2].astype(int) if cons_np.shape[1] > 2 else np.zeros(len(x), int)
    vb, vg, vm, nsoft = soft_violation_counts(x, y, w, h, bcode, clust, mib)
    vrel = (vb + vg + vm) / max(nsoft, 1)
    return ov, area / max(gt_area, 1e-9) - 1.0, hp / max(gt_hpwl, 1e-9) - 1.0, vrel


def main():
    pred = M1Predictor(WEIGHTS, device="cpu")
    evalr = ev.ContestEvaluator(data_path=FLOORSET, verbose=False)
    evalr._load_dataset()
    print(f"{'tid':>4} {'n':>4} | {'--- M1 raw ---':^28} | {'--- M1 + S1-aware repair ---':^28}")
    print(f"{'':>4} {'':>4} | {'ov%':>4} {'area*':>7} {'hpwl*':>7} {'Vrel':>6} | "
          f"{'ov%':>4} {'area*':>7} {'hpwl*':>7} {'Vrel':>6}")
    print("(*: ratio vs GT baseline, same formula the real evaluator uses)")
    import time
    for tid in IDS:
        sample = evalr.dataset[tid]
        inputs, labels = (sample["input"], sample["label"]) if isinstance(sample, dict) \
            else (sample[0], sample[1])
        area_t, b2b, p2b, pins, cons = inputs
        n = int((area_t != -1).sum().item())
        baseline, gt = evalr._extract_baseline(tid, labels, b2b, p2b, pins, n)

        opt_tp = torch.full((n, 4), -1.0)
        nc = cons.shape[1] if cons.dim() > 1 else 0
        for i in range(n):
            is_fixed = nc > 0 and cons[i, 0] != 0
            is_pre = nc > 1 and cons[i, 1] != 0
            if is_pre:
                opt_tp[i] = torch.tensor(gt[i])
            elif is_fixed:
                opt_tp[i, 2] = gt[i][2]
                opt_tp[i, 3] = gt[i][3]

        t0 = time.time()
        pos = pred.predict(n, area_t, cons, opt_tp, b2b, p2b, pins)
        dt = time.time() - t0
        if pos is None:
            print(f"{tid:>4} {n:>4} | FAIL (dead-end / None)                       | {dt:6.2f}")
            continue

        x = np.array([p[0] for p in pos]); y = np.array([p[1] for p in pos])
        w = np.array([p[2] for p in pos]); h = np.array([p[3] for p in pos])
        eb, ep, pv = _edges(b2b), _edges(p2b), _edges(pins)
        cons_np = cons[:n].cpu().numpy()
        gt_area = baseline["area_baseline"]
        gt_hpwl = baseline["hpwl_baseline"]

        ov0, ag0, hg0, vr0 = metrics(x, y, w, h, cons_np, eb, ep, pv, gt_area, gt_hpwl)

        # run EXACTLY the pipeline's own S1-aware repair (electro_parallel.compact_variant,
        # aware=True) on M1's raw output -- parity with the real code path.
        is_pre = (cons_np[:, 1] != 0)
        is_fixed = (cons_np[:, 0] != 0)
        is_soft = ~(is_fixed | is_pre)
        mib_id = cons_np[:, 2].astype(int) if cons_np.shape[1] > 2 else np.zeros(n, int)
        clust_id = cons_np[:, 3].astype(int) if cons_np.shape[1] > 3 else np.zeros(n, int)
        bcode = cons_np[:, 4].astype(int) if cons_np.shape[1] > 4 else np.zeros(n, int)
        P = {"is_soft": is_soft, "is_pre": is_pre, "mib_id": mib_id,
             "clust_id": clust_id, "bcode": bcode, "nonneg": True}
        rx, ry, rw, rh = electro_parallel.compact_variant((x, y, w, h), P, aware=True)
        ov1, ag1, hg1, vr1 = metrics(rx, ry, rw, rh, cons_np, eb, ep, pv, gt_area, gt_hpwl)

        print(f"{tid:>4} {n:>4} | {ov0:4.1f} {ag0:+7.3f} {hg0:+7.3f} {vr0:6.3f} | "
              f"{ov1:4.1f} {ag1:+7.3f} {hg1:+7.3f} {vr1:6.3f}")


if __name__ == "__main__":
    main()
