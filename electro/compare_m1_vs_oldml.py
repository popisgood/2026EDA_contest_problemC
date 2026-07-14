#!/usr/bin/env python3
"""Head-to-head: old coordinate-regression ML vs M1 (constructive imitation),
both RAW (no legalizer/repair), same cases, same metrics, against GT.

  agree  = pairwise left/right+above/below agreement vs GT (1.0=identical
           arrangement, ~0.25=random; translation/scale/mirror-invariant).
  ov%    = overlap as % of total block area (0 = legal).
  area*  = bbox area vs GT baseline, same ratio-1 formula the real evaluator uses.
  hpwl*  = HPWL vs GT baseline, same formula.

Run:  IDS=0,20,40,60,80,99 RENDER=20,60 python electro/compare_m1_vs_oldml.py
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
from ml.predict import Predictor as OldMLPredictor
from ml.m1_infer import M1Predictor

from diag_m1 import _edges, hpwl, overlap_pct
from diag_ml_vs_gt import agree_disp, cxy_lowerleft, cxy_centers

IDS = [int(x) for x in os.environ.get("IDS", "0,20,40,60,80,99").split(",") if x != ""]
OLDML_WEIGHTS = os.environ.get("OLDML_WEIGHTS", REPO + "/ml/weights/floorplan_v3.pt")
M1_WEIGHTS = os.environ.get("M1_WEIGHTS", REPO + "/ml/weights/m1_v1.pt")
RENDER = [int(x) for x in os.environ.get("RENDER", "").split(",") if x != ""]
OUTDIR = ELECTRO + "/diag_out"


def raw_metrics(x, y, w, h, eb, ep, pv, gt_area, gt_hpwl):
    ov = overlap_pct(x, y, w, h)
    area = float((x + w).max() - x.min()) * float((y + h).max() - y.min())
    cx, cy = x + 0.5 * w, y + 0.5 * h
    hp = hpwl(cx, cy, eb, ep, pv)
    return ov, area / max(gt_area, 1e-9) - 1.0, hp / max(gt_hpwl, 1e-9) - 1.0


def render(idx, layouts, n):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    os.makedirs(OUTDIR, exist_ok=True)
    cmap = plt.get_cmap("tab20")
    names = list(layouts.keys())
    fig, axes = plt.subplots(1, len(names), figsize=(5 * len(names), 5))
    for ax, name in zip(axes, names):
        arr = layouts[name]
        if arr is None:
            ax.set_title(f"{name}: n/a"); ax.axis("off"); continue
        for i, (x, y, w, h) in enumerate(arr):
            ax.add_patch(Rectangle((x, y), w, h, facecolor=cmap(i % 20),
                                   edgecolor="black", lw=0.4, alpha=0.75))
            ax.text(x + 0.5 * w, y + 0.5 * h, str(i), ha="center", va="center",
                    fontsize=5)
        xs = [p[0] for p in arr]; ys = [p[1] for p in arr]
        xe = [p[0] + p[2] for p in arr]; ye = [p[1] + p[3] for p in arr]
        ax.set_xlim(min(xs) - 1, max(xe) + 1); ax.set_ylim(min(ys) - 1, max(ye) + 1)
        ax.set_aspect("equal"); ax.set_title(f"{name} (n={n})")
    fig.suptitle(f"test_id {idx}: raw model output, no legalizer -- same block id = same colour")
    out = f"{OUTDIR}/m1_vs_oldml_{idx:03d}.png"
    fig.savefig(out, dpi=110, bbox_inches="tight"); plt.close(fig)
    return out


def main():
    evalr = ev.ContestEvaluator(data_path=FLOORSET, verbose=False)
    evalr._load_dataset()
    old = OldMLPredictor(OLDML_WEIGHTS, device="cpu")
    m1 = M1Predictor(M1_WEIGHTS, device="cpu")

    print(f"old ML weights = {OLDML_WEIGHTS}")
    print(f"M1  weights    = {M1_WEIGHTS}")
    print("agree = pairwise arrangement match vs GT (1.0=identical, ~0.25=random)")
    print(f"{'tid':>4} {'n':>4} | {'--- old ML (regression) ---':^28} | "
          f"{'--- M1 (constructive) ---':^28}")
    print(f"{'':>4} {'':>4} | {'agree':>5} {'ov%':>5} {'area*':>7} {'hpwl*':>7} | "
          f"{'agree':>5} {'ov%':>5} {'area*':>7} {'hpwl*':>7}")
    print("-" * 78)

    for tid in IDS:
        sample = evalr.dataset[tid]
        inputs, labels = (sample["input"], sample["label"]) if isinstance(sample, dict) \
            else (sample[0], sample[1])
        area_t, b2b, p2b, pins, cons = inputs
        n = int((area_t != -1).sum().item())
        baseline, gt = evalr._extract_baseline(tid, labels, b2b, p2b, pins, n)
        gt_area = baseline["area_baseline"]
        gt_hpwl = baseline["hpwl_baseline"]
        eb, ep, pv = _edges(b2b), _edges(p2b), _edges(pins)
        G = cxy_lowerleft(gt)

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

        # --- old ML (raw regression output, center-format) ---
        old_pred = old.predict(n, area_t, cons, opt_tp, b2b, p2b, pins)
        if old_pred is not None:
            old_pos = old_pred.positions  # (cx, cy, w, h)
            ox = np.array([p[0] - 0.5 * p[2] for p in old_pos])
            oy = np.array([p[1] - 0.5 * p[3] for p in old_pos])
            ow = np.array([p[2] for p in old_pos])
            oh = np.array([p[3] for p in old_pos])
            a_old, _ = agree_disp(cxy_centers(old_pos), G)
            ov_o, ag_o, hg_o = raw_metrics(ox, oy, ow, oh, eb, ep, pv, gt_area, gt_hpwl)
        else:
            a_old, ov_o, ag_o, hg_o = float("nan"), float("nan"), float("nan"), float("nan")

        # --- M1 (raw rollout output, lower-left xywh) ---
        m1_pos = m1.predict(n, area_t, cons, opt_tp, b2b, p2b, pins)
        if m1_pos is not None:
            mx = np.array([p[0] for p in m1_pos]); my = np.array([p[1] for p in m1_pos])
            mw = np.array([p[2] for p in m1_pos]); mh = np.array([p[3] for p in m1_pos])
            a_m1, _ = agree_disp(cxy_lowerleft(m1_pos), G)
            ov_m, ag_m, hg_m = raw_metrics(mx, my, mw, mh, eb, ep, pv, gt_area, gt_hpwl)
        else:
            a_m1, ov_m, ag_m, hg_m = float("nan"), float("nan"), float("nan"), float("nan")

        print(f"{tid:>4} {n:>4} | {a_old:5.2f} {ov_o:5.1f} {ag_o:+7.3f} {hg_o:+7.3f} | "
              f"{a_m1:5.2f} {ov_m:5.1f} {ag_m:+7.3f} {hg_m:+7.3f}")

        if tid in RENDER:
            old_xywh = [(p[0] - 0.5 * p[2], p[1] - 0.5 * p[3], p[2], p[3])
                       for p in old_pos] if old_pred is not None else None
            m1_xywh = [(p[0], p[1], p[2], p[3]) for p in m1_pos] if m1_pos is not None else None
            out = render(tid, {"GT": gt, "old ML": old_xywh, "M1": m1_xywh}, n)
            print(f"     -> rendered {out}")


if __name__ == "__main__":
    main()
