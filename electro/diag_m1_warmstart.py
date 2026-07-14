#!/usr/bin/env python3
"""Does warm-starting electro's optimizer from M1 beat electro's OWN random start?

Per case, no ranking -- three layouts through the SAME legalize+repair tail:
  electro(rand) : run_start with random init  (what warm-start must out-rank)
  M1 raw        : M1's rollout, untouched      (the seed)
  M1 warmstart  : place() seeded from M1 (pos+aspect) -> legalize+repair

Metrics vs GT baseline (same ratio-1 formulas the real evaluator uses) + a proxy
cost so you can see the net trade the ranking will make:
  proxy = (1 + 0.5*(max0 area* + max0 hpwl*)) * exp(2*Vrel)   [RT ignored, ~=1 local]

Run:  IDS=0,20,40,60,80,99 python electro/diag_m1_warmstart.py
"""
from __future__ import annotations
import math, os, sys
import numpy as np
import torch

FLOORSET = "/home/pop/IntelLabs_Floorset/FloorSet"
EVALDIR = FLOORSET + "/iccad2026contest"
REPO = "/home/pop/2026_EDA_contest"
ELECTRO = REPO + "/electro"
for p in (FLOORSET, EVALDIR, ELECTRO, REPO):
    if p not in sys.path:
        sys.path.insert(0, p)
os.environ.setdefault("ELECTRO_CLAMP", "1")
os.environ.setdefault("ELECTRO_NONNEG", "1")

import iccad2026_evaluate as ev
from ml.m1_infer import M1Predictor
import electro_parallel
from diag_m1 import _edges, hpwl, overlap_pct, metrics

IDS = [int(x) for x in os.environ.get("IDS", "0,20,40,60,80,99").split(",") if x != ""]
WEIGHTS = os.environ.get("WEIGHTS", REPO + "/ml/weights/m1_v1.pt")


def proxy(ov, ag, hg, vr):
    if ov > 1e-6:
        return float("inf")  # illegal -> infeasible in the real evaluator
    return (1 + 0.5 * (max(0, ag) + max(0, hg))) * math.exp(2 * vr)


def main():
    pred = M1Predictor(WEIGHTS, device="cpu")
    evalr = ev.ContestEvaluator(data_path=FLOORSET, verbose=False)
    evalr._load_dataset()
    print(f"{'tid':>4} {'n':>4} | {'electro(rand)':^22} | {'M1 raw':^22} | "
          f"{'M1 warmstart':^22}")
    print(f"{'':>4} {'':>4} | {'area* hpwl*  Vrel prox':>22} | "
          f"{'area* hpwl*  Vrel prox':>22} | {'area* hpwl*  Vrel prox':>22}")
    print("-" * 96)
    agg = {"rand": [], "ws": []}
    for tid in IDS:
        sample = evalr.dataset[tid]
        inputs, labels = (sample["input"], sample["label"]) if isinstance(sample, dict) \
            else (sample[0], sample[1])
        area_t, b2b, p2b, pins, cons = inputs
        n = int((area_t != -1).sum().item())
        baseline, gt = evalr._extract_baseline(tid, labels, b2b, p2b, pins, n)
        ga, gh = baseline["area_baseline"], baseline["hpwl_baseline"]
        eb, ep, pv = _edges(b2b), _edges(p2b), _edges(pins)
        cons_np = cons[:n].cpu().numpy()

        opt_tp = torch.full((n, 4), -1.0)
        nc = cons.shape[1] if cons.dim() > 1 else 0
        for i in range(n):
            if nc > 1 and cons[i, 1] != 0:
                opt_tp[i] = torch.tensor(gt[i])
            elif nc > 0 and cons[i, 0] != 0:
                opt_tp[i, 2] = gt[i][2]; opt_tp[i, 3] = gt[i][3]

        P = {
            "n": n, "area": area_t, "b2b": b2b, "p2b": p2b, "pins": pins,
            "cons": cons, "tp": opt_tp, "iters": 600, "lr": 0.02, "device": "cpu",
            "init": None, "rounds": 3, "nonneg": True,
            "is_pre": (cons_np[:, 1] != 0),
            "is_soft": ~((cons_np[:, 0] != 0) | (cons_np[:, 1] != 0)),
            "mib_id": cons_np[:, 2].astype(int) if cons_np.shape[1] > 2 else np.zeros(n, int),
            "clust_id": cons_np[:, 3].astype(int) if cons_np.shape[1] > 3 else np.zeros(n, int),
            "bcode": cons_np[:, 4].astype(int) if cons_np.shape[1] > 4 else np.zeros(n, int),
        }

        cells = []
        # electro's own random-init result
        rx, ry, rw, rh = electro_parallel.run_start(0, P)
        cells.append(metrics(rx, ry, rw, rh, cons_np, eb, ep, pv, ga, gh))
        # M1 raw
        pos = pred.predict(n, area_t, cons, opt_tp, b2b, p2b, pins)
        mx = np.array([p[0] for p in pos]); my = np.array([p[1] for p in pos])
        mw = np.array([p[2] for p in pos]); mh = np.array([p[3] for p in pos])
        cells.append(metrics(mx, my, mw, mh, cons_np, eb, ep, pv, ga, gh))
        # M1 warm-start
        wx, wy, ww, wh = electro_parallel.m1_warmstart_variant((mx, my, mw, mh), P)
        cells.append(metrics(wx, wy, ww, wh, cons_np, eb, ep, pv, ga, gh))

        segs = []
        for (ov, ag, hg, vr) in cells:
            segs.append(f"{ag:+5.2f} {hg:+5.2f} {vr:4.2f} {proxy(ov,ag,hg,vr):4.2f}")
        print(f"{tid:>4} {n:>4} | {segs[0]:>22} | {segs[1]:>22} | {segs[2]:>22}")
        agg["rand"].append(proxy(*cells[0]))
        agg["ws"].append(proxy(*cells[2]))

    print("-" * 96)
    print(f"mean proxy  electro(rand)={np.mean(agg['rand']):.3f}   "
          f"M1 warmstart={np.mean(agg['ws']):.3f}   "
          f"(warmstart wins a case when its proxy < electro's)")


if __name__ == "__main__":
    main()
