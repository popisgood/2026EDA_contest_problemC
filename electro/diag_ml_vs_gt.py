#!/usr/bin/env python3
"""Diagnostic: how close is each pipeline stage to the ground-truth layout?

For each test case we compare four layouts, block-for-block, against GT:
    GT            -- the contest's reference solution (labels polygons -> bbox)
    ML            -- raw FloorplanTransformer prediction (the "ML output")
    GLOBAL        -- analytical place() output, ML-initialised, BEFORE legalize
    FINAL         -- after legalize + grouping_repair + boundary_snap + remove_overlap

Two similarity metrics, both robust to the layout's translation / scale / mirror
(we take the best of the 4 axis flips, since a mirrored floorplan is equivalent):

  agree  = fraction of block PAIRS whose left/right AND above/below relationship
           matches GT.  Random ~0.25, identical = 1.0.  (Translation+scale free.)
  disp   = median per-block centroid distance after normalising each layout to a
           unit bounding box.  0 = identical arrangement; ~0.4+ = unrelated.

Plus the key attribution number:
  leg_move = median distance the legalizer+repair moved each block, as a fraction
             of the chip size.  Small => legalizer only fine-tunes (ML/global did
             the real work).  Large => legalizer is doing the heavy lifting.

Run (from anywhere, via the FloorSet venv):
  IDS=0,40,60,80,99 RENDER=0,99 python electro/diag_ml_vs_gt.py
"""
from __future__ import annotations
import os, sys
import numpy as np
import torch

FLOORSET = "/home/pop/IntelLabs_Floorset/FloorSet"
EVALDIR  = FLOORSET + "/iccad2026contest"
REPO     = "/home/pop/2026_EDA_contest"
ELECTRO  = REPO + "/electro"
for p in (FLOORSET, EVALDIR, ELECTRO, REPO):
    if p not in sys.path:
        sys.path.insert(0, p)

# Match the submitted config's geometry knobs so place()/legalize behave as deployed.
os.environ.setdefault("ELECTRO_CLAMP", "1")
os.environ.setdefault("ELECTRO_NONNEG", "1")

import iccad2026_evaluate as ev
from analytical_place import place
from legalize import legalize, remove_overlap
from soft_repair import boundary_snap, grouping_repair
from ml.predict import Predictor

IDS     = [int(x) for x in os.environ.get("IDS", "0,40,60,80,99").split(",") if x != ""]
ITERS   = int(os.environ.get("ITERS", "600"))
LR      = float(os.environ.get("LR", "0.02"))
DEVICE  = "cpu"
WEIGHTS = os.environ.get("WEIGHTS", REPO + "/ml/weights/floorplan_v3.pt")
RENDER  = [int(x) for x in os.environ.get("RENDER", "").split(",") if x != ""]
OUTDIR  = ELECTRO + "/diag_out"


def cxy_lowerleft(arr):
    return np.array([(x + 0.5 * w, y + 0.5 * h) for (x, y, w, h) in arr], float)


def cxy_centers(positions):
    return np.array([(p[0], p[1]) for p in positions], float)


def _norm(P):
    P = P - P.mean(0)
    span = max((P[:, 0].max() - P[:, 0].min()),
               (P[:, 1].max() - P[:, 1].min()), 1e-9)
    return P / span


def agree_disp(C, G):
    """Best-of-4-flip pairwise-agreement and normalised median displacement."""
    n = len(G)
    iu = np.triu_indices(n, 1)
    dgx = np.sign(G[iu[0], 0] - G[iu[1], 0])
    dgy = np.sign(G[iu[0], 1] - G[iu[1], 1])
    vx, vy = dgx != 0, dgy != 0
    vboth = vx & vy
    Gn = _norm(G)
    best_ab, best_disp = -1.0, 9.9
    for sx in (1, -1):
        for sy in (1, -1):
            Cf = C * np.array([sx, sy])
            dcx = np.sign(Cf[iu[0], 0] - Cf[iu[1], 0])
            dcy = np.sign(Cf[iu[0], 1] - Cf[iu[1], 1])
            mx, my = (dcx == dgx), (dcy == dgy)
            ab = (mx & my)[vboth].mean() if vboth.any() else 0.0
            if ab > best_ab:
                best_ab = ab
            Cn = _norm(Cf)
            d = float(np.median(np.linalg.norm(Cn - Gn, axis=1)))
            if d < best_disp:
                best_disp = d
    return best_ab, best_disp


def run_legalize(gpos, cons):
    is_pre = (cons[:, 1] != 0)
    clust_id = cons[:, 3].astype(int)
    bcode = cons[:, 4].astype(int)
    x = np.array([p[0] for p in gpos], float)
    y = np.array([p[1] for p in gpos], float)
    w = np.array([p[2] for p in gpos], float)
    h = np.array([p[3] for p in gpos], float)
    floor = 0.0 if os.environ.get("ELECTRO_NONNEG", "0") == "1" else None
    x, y = legalize(x, y, w, h, is_pre, floor=floor)
    for _ in range(3):
        x, y = grouping_repair(x, y, w, h, clust_id, is_pre, floor=floor)
        x, y = boundary_snap(x, y, w, h, bcode, is_pre, floor=floor)
    x, y = remove_overlap(x, y, w, h, is_pre,
                          nonneg=os.environ.get("ELECTRO_NONNEG", "0") == "1")
    return list(zip(x.tolist(), y.tolist(), w.tolist(), h.tolist()))


def render(idx, layouts, n):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    os.makedirs(OUTDIR, exist_ok=True)
    cmap = plt.get_cmap("tab20")
    names = ["GT", "ML", "GL(rand)", "GL(ml)", "FINAL"]
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
    fig.suptitle(f"test_id {idx}: same block id = same colour across panels")
    out = f"{OUTDIR}/case_{idx:03d}.png"
    fig.savefig(out, dpi=110, bbox_inches="tight"); plt.close(fig)
    return out


def main():
    evalr = ev.ContestEvaluator(data_path=FLOORSET, verbose=False)
    evalr._load_dataset()
    pred = Predictor(WEIGHTS, device=DEVICE)
    print(f"weights = {WEIGHTS}")
    print("pair-agreement vs GT (1.0=identical arrangement, ~0.25=random); "
          "GL=analytical global placement")
    print(f"{'id':>4} {'n':>4} | {'ML':>5} | {'GL(ml)':>7} | {'GL(rand)':>8} | "
          f"{'FINAL':>6} | leg_move")
    print("-" * 60)
    for idx in IDS:
        try:
            sample = evalr.dataset[idx]
            if isinstance(sample, dict):
                inputs, labels = sample["input"], sample["label"]
            else:
                inputs, labels = sample[0], sample[1]
            area_t, b2b, p2b, pins, constraints = inputs
            n = int((area_t != -1).sum().item())
            baseline, gt = evalr._extract_baseline(idx, labels, b2b, p2b, pins, n)

            # locked geometry tensor exactly as the evaluator builds it
            opt_tp = torch.full((n, 4), -1.0)
            nc = constraints.shape[1] if constraints.dim() > 1 else 0
            # GT positions double as the (x,y,w,h) source for fixed/preplaced
            for i in range(n):
                is_fixed = nc > 0 and constraints[i, 0] != 0
                is_pre = nc > 1 and constraints[i, 1] != 0
                if is_pre:
                    opt_tp[i] = torch.tensor(gt[i])
                elif is_fixed:
                    opt_tp[i, 2] = gt[i][2]; opt_tp[i, 3] = gt[i][3]

            ml_pred = pred.predict(n, area_t, constraints, opt_tp, b2b, p2b, pins)
            ml_pos = ml_pred.positions if ml_pred is not None else None
            init = (torch.tensor([[p[0], p[1]] for p in ml_pos], dtype=torch.float32)
                    if ml_pos is not None else None)

            gpos, _ = place(n, area_t, b2b, p2b, pins, constraints, opt_tp,
                            iters=ITERS, lr=LR, seed=0, device=DEVICE,
                            init_centers=init)
            # ablation: same analytical placer but RANDOM init (no ML warm-start)
            gpos_r, _ = place(n, area_t, b2b, p2b, pins, constraints, opt_tp,
                              iters=ITERS, lr=LR, seed=0, device=DEVICE,
                              init_centers=None)
            cons_np = constraints[:n].cpu().numpy()
            final = run_legalize(gpos, cons_np)

            G = cxy_lowerleft(gt)
            GL = cxy_lowerleft(gpos)
            GLR = cxy_lowerleft(gpos_r)
            F = cxy_lowerleft(final)
            M = cxy_centers(ml_pos) if ml_pos is not None else None

            a_ml, d_ml = agree_disp(M, G) if M is not None else (float("nan"),) * 2
            a_gl, d_gl = agree_disp(GL, G)
            a_glr, d_glr = agree_disp(GLR, G)
            a_f, d_f = agree_disp(F, G)

            # legalizer movement: GLOBAL -> FINAL in real pipeline coords
            gt_area = ((max(p[0] + p[2] for p in gt) - min(p[0] for p in gt)) *
                       (max(p[1] + p[3] for p in gt) - min(p[1] for p in gt)))
            scale = max(gt_area ** 0.5, 1e-9)
            leg_move = float(np.median(np.linalg.norm(F - GL, axis=1))) / scale

            mlcell = (f"{a_ml:.2f}" if M is not None else " off ")
            print(f"{idx:>4} {n:>4} | {mlcell:>5} | {a_gl:>7.2f} | {a_glr:>8.2f} | "
                  f"{a_f:>6.2f} | {leg_move:.3f}")

            if idx in RENDER:
                ml_xywh = ([(p[0] - 0.5 * p[2], p[1] - 0.5 * p[3], p[2], p[3])
                            for p in ml_pos] if ml_pos is not None else None)
                out = render(idx, {"GT": gt, "ML": ml_xywh, "GL(rand)": gpos_r,
                                   "GL(ml)": gpos, "FINAL": final}, n)
                print(f"     -> rendered {out}")
        except Exception as e:
            import traceback
            print(f"{idx:>4}  ERROR: {e}")
            traceback.print_exc()


if __name__ == "__main__":
    main()
