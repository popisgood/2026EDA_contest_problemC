#!/usr/bin/env python3
"""M3 probe: GT layout -> B*-tree -> deterministic contour repack -> compare vs GT.

Answers the M3 (discrete-representation ML) feasibility question cheaply: if an ML
model predicted the GT's B*-tree *perfectly*, how close would the deterministically
packed layout be to the GT?  The round-trip error (extractor + packer) is exactly
the fidelity ceiling an M3 pipeline would inherit, since training targets would be
produced by this same extractor.

Tree semantics (Chang et al. B*-tree):
  * preorder DFS; root packs at x=0
  * left child of j packs at x = x_j + w_j   (adjacent to j's right edge)
  * right child of j packs at x = x_j        (stacked above j)
  * y = contour = max top of already-placed blocks overlapping [x, x+w), else 0

Extraction: greedy nearest-slot attach.  Blocks sorted by GT (x, y); each block
attaches to the free slot (parent, L/R) whose packed position would best match its
GT position.  Always yields a valid binary tree.

Metrics vs GT: bbox-area ratio, contest-HPWL ratio (b2b+p2b, GT terminals),
pairwise-agreement, median per-block displacement / sqrt(GT area).

Run:  IDS=0,20,40,60,80,99 python probe_m3_tree.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

FLOORSET = "/home/pop/IntelLabs_Floorset/FloorSet"
EVALDIR = FLOORSET + "/iccad2026contest"
for p in (FLOORSET, EVALDIR):
    if p not in sys.path:
        sys.path.insert(0, p)

import iccad2026_evaluate as ev

sys.path.insert(0, "/home/pop/2026_EDA_contest/electro")
from shape_compact import compact_and_shape

IDS = [int(t) for t in os.environ.get(
    "IDS", "0,10,20,30,40,50,60,70,80,90,99").split(",") if t != ""]


# ---------------- B*-tree extraction + packing ----------------

def build_tree(x, y, w, h, order_key="xy"):
    """Greedy nearest-slot attach; returns (root, left_child, right_child)."""
    n = len(x)
    root = min(range(n), key=lambda i: (y[i], x[i]))
    keys = {"xy": lambda i: (x[i], y[i]), "yx": lambda i: (y[i], x[i]),
            "diag": lambda i: (x[i] + y[i], x[i])}
    order = sorted((i for i in range(n) if i != root), key=keys[order_key])
    L, R = {}, {}
    inserted = [root]
    for b in order:
        best = None
        for j in inserted:
            if j not in L:  # left slot: packs at x_j + w_j, beside j
                err = abs((x[j] + w[j]) - x[b]) + abs(y[j] - y[b])
                if best is None or err < best[0]:
                    best = (err, j, "L")
            if j not in R:  # right slot: packs at x_j, above j
                err = abs(x[j] - x[b]) + abs((y[j] + h[j]) - y[b])
                if best is None or err < best[0]:
                    best = (err, j, "R")
        _, j, slot = best
        (L if slot == "L" else R)[j] = b
        inserted.append(b)
    return root, L, R


def pack_tree(root, L, R, w, h):
    """Deterministic contour packing (exact B*-tree semantics)."""
    n = len(w)
    px = np.zeros(n)
    py = np.zeros(n)
    placed = []
    stack = [(root, None, None)]
    while stack:
        node, parent, slot = stack.pop()
        if parent is None:
            xx = 0.0
        elif slot == "L":
            xx = px[parent] + w[parent]
        else:
            xx = px[parent]
        yy = 0.0
        for j in placed:  # contour: max top among x-overlapping placed blocks
            if px[j] < xx + w[node] - 1e-9 and xx < px[j] + w[j] - 1e-9:
                yy = max(yy, py[j] + h[j])
        px[node], py[node] = xx, yy
        placed.append(node)
        # push right then left so left (beside) is processed first (preorder)
        if node in R:
            stack.append((R[node], node, "R"))
        if node in L:
            stack.append((L[node], node, "L"))
    return px, py


# ---------------- metrics ----------------

def hpwl(cx, cy, eb, ep, pv):
    v = 0.0
    if eb is not None:
        i = eb[:, 0].astype(int)
        j = eb[:, 1].astype(int)
        v += float((eb[:, 2] * (np.abs(cx[i] - cx[j]) + np.abs(cy[i] - cy[j]))).sum())
    if ep is not None and pv is not None:
        pi = ep[:, 0].astype(int)
        bi = ep[:, 1].astype(int)
        v += float((ep[:, 2] * (np.abs(pv[pi, 0] - cx[bi]) + np.abs(pv[pi, 1] - cy[bi]))).sum())
    return v


def agree(C, G):
    n = len(G)
    iu = np.triu_indices(n, 1)
    dgx = np.sign(G[iu[0], 0] - G[iu[1], 0])
    dgy = np.sign(G[iu[0], 1] - G[iu[1], 1])
    v = (dgx != 0) & (dgy != 0)
    best = -1.0
    for sx in (1, -1):
        for sy in (1, -1):
            Cf = C * np.array([sx, sy])
            dcx = np.sign(Cf[iu[0], 0] - Cf[iu[1], 0])
            dcy = np.sign(Cf[iu[0], 1] - Cf[iu[1], 1])
            m = ((dcx == dgx) & (dcy == dgy))[v].mean() if v.any() else 0.0
            best = max(best, float(m))
    return best


def valid_edges(t):
    if t is None or t.numel() == 0:
        return None
    a = t.cpu().numpy()
    a = a[a[:, 0] != -1]
    return a if len(a) else None


def main():
    evalr = ev.ContestEvaluator(data_path=FLOORSET, verbose=False)
    evalr._load_dataset()
    print("M3 probe: GT -> B*-tree -> contour repack, error vs GT")
    print(f"{'tid':>4} {'n':>4} {'pre':>3} | {'areaR':>6} {'hpwlR':>6} "
          f"{'agree':>5} {'med_disp':>8} | {'ov%':>5}")
    rows = []
    for tid in IDS:
        sample = evalr.dataset[tid]
        inputs, labels = (sample["input"], sample["label"]) if isinstance(sample, dict) \
            else (sample[0], sample[1])
        area_t, b2b, p2b, pins, cons = inputs
        n = int((area_t != -1).sum().item())
        _, gt = evalr._extract_baseline(tid, labels, b2b, p2b, pins, n)
        npre = int((cons[:n, 1] != 0).sum().item()) if cons.dim() > 1 else 0

        x = np.array([p[0] for p in gt])
        y = np.array([p[1] for p in gt])
        w = np.array([p[2] for p in gt])
        h = np.array([p[3] for p in gt])
        x -= x.min()
        y -= y.min()

        # best-of extractor variants: 3 insertion orders; keep min reconstructed area
        best = None
        for ok in ("xy", "yx", "diag"):
            root, L, R = build_tree(x, y, w, h, ok)
            qx, qy = pack_tree(root, L, R, w, h)
            a_ = (qx + w).max() * (qy + h).max()
            if best is None or a_ < best[0]:
                best = (a_, qx, qy)
        _, px, py = best

        # + our existing compaction+shaping pass on the reconstruction (combined
        # ceiling of tree-repack -> compact_and_shape)
        is_fixed = (cons[:n, 0] != 0).numpy().astype(bool)
        is_pre_m = (cons[:n, 1] != 0).numpy().astype(bool)
        is_soft = ~(is_fixed | is_pre_m)
        mib = cons[:n, 2].numpy().astype(int) if cons.shape[1] > 2 else np.zeros(n, int)
        sx, sy, sw, sh = compact_and_shape(px, py, w, h, is_soft,
                                           np.zeros(n, bool), mib, floor=0.0)
        sc_area = (sx + sw).max() * (sy + sh).max()

        # overlap sanity (packer should be overlap-free)
        ov = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                ox = min(px[i] + w[i], px[j] + w[j]) - max(px[i], px[j])
                oy = min(py[i] + h[i], py[j] + h[j]) - max(py[i], py[j])
                if ox > 1e-7 and oy > 1e-7:
                    ov += ox * oy
        ovp = 100.0 * ov / float((w * h).sum())

        gt_area = (x + w).max() * (y + h).max()
        rc_area = (px + w).max() * (py + h).max()

        eb = valid_edges(b2b)
        ep = valid_edges(p2b)
        pv = valid_edges(pins)
        # terminals live in the GT frame; keep both layouts in origin-anchored frames
        h_gt = hpwl(x + 0.5 * w, y + 0.5 * h, eb, ep, pv)
        h_rc = hpwl(px + 0.5 * w, py + 0.5 * h, eb, ep, pv)

        G = np.stack([x + 0.5 * w, y + 0.5 * h], 1)
        C = np.stack([px + 0.5 * w, py + 0.5 * h], 1)
        ag = agree(C, G)
        disp = float(np.median(np.linalg.norm(C - G, axis=1))) / max(gt_area, 1e-9) ** 0.5

        rows.append((rc_area / gt_area, h_rc / max(h_gt, 1e-9), ag, disp,
                     sc_area / gt_area))
        print(f"{tid:>4} {n:>4} {npre:>3} | {rc_area / gt_area:6.3f} "
              f"{h_rc / max(h_gt, 1e-9):6.3f} {ag:5.2f} {disp:8.3f} | {ovp:5.2f} "
              f"| +cmp {sc_area / gt_area:6.3f}")

    a = np.array(rows)
    print("-" * 72)
    print(f"{'avg':>4} {'':>4} {'':>3} | {a[:, 0].mean():6.3f} {a[:, 1].mean():6.3f} "
          f"{a[:, 2].mean():5.2f} {a[:, 3].mean():8.3f} |       | +cmp {a[:, 4].mean():6.3f}")
    print("\nareaR/hpwlR = reconstructed/GT (1.000 = perfect; areaR-1 ~ inherited area_gap)")
    print("+cmp = area ratio after our compact_and_shape pass on the reconstruction")


if __name__ == "__main__":
    main()
