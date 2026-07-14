"""M1 (constructive imitation) shared core: canonical order, grid, masks, tokens.

Everything here is used IDENTICALLY by training (ml/m1_train.py) and inference
(ml/m1_infer.py).  If you change any encoding here, retrain.

Design (see electro/NEXT_STEPS.md sec 5.1 + M3-probe lesson in 5.3):
  * The decoder places blocks one at a time on a GRID x GRID canvas with FREE
    (x, y) slots (MaskPlace-style), NOT left/bottom contour rules -- the M3 probe
    showed contour packing cannot reproduce GT's interlocked tilings.
  * Canonical placement order is computed from INPUTS ONLY (preplaced first as
    fixed context, then area desc, degree desc) so train and inference agree.
  * Position target = grid cell of the block's lower-left corner, canvas-normalized.
  * Soft blocks additionally get a log-aspect bin (area stays exact by construction).
"""
from __future__ import annotations

import math

import numpy as np

GRID = 32                      # canvas is GRID x GRID cells
N_ASPECT = 9                   # log-aspect bins
LOG_AR_MAX = math.log(4.0)     # aspect w/h in [1/4, 4] (validated by AR_CAP sweep)
TOKEN_DIM = 26                 # per-block token feature size
MAX_N = 128                    # pad every case to this many block tokens


# ---------------- order / bins / cells ----------------

def canonical_order(area, is_pre, deg):
    """Placement order for the decoder: non-preplaced blocks, area desc,
    degree desc, index asc.  Computed from inputs only."""
    idx = [i for i in range(len(area)) if not is_pre[i]]
    return sorted(idx, key=lambda i: (-area[i], -deg[i], i))


def aspect_to_bin(w, h):
    la = math.log(max(w, 1e-9) / max(h, 1e-9))
    la = max(-LOG_AR_MAX, min(LOG_AR_MAX, la))
    k = int(round((la + LOG_AR_MAX) / (2 * LOG_AR_MAX) * (N_ASPECT - 1)))
    return max(0, min(N_ASPECT - 1, k))


def bin_to_wh(k, block_area):
    la = -LOG_AR_MAX + k * (2 * LOG_AR_MAX) / (N_ASPECT - 1)
    s = math.sqrt(block_area)
    return s * math.exp(0.5 * la), s * math.exp(-0.5 * la)


def cell_of(x, y, Wc, Hc):
    gx = min(GRID - 1, max(0, int(x / Wc * GRID)))
    gy = min(GRID - 1, max(0, int(y / Hc * GRID)))
    return gy * GRID + gx


def cell_to_xy(cell, Wc, Hc):
    gy, gx = divmod(int(cell), GRID)
    return gx * Wc / GRID, gy * Hc / GRID


# ---------------- legality mask (inference) ----------------

def legality_mask(w, h, placed, Wc, Hc, slack=0.02):
    """Boolean [GRID*GRID]: cells where a (w x h) block's lower-left corner can
    go with zero overlap against `placed` [(x,y,w,h)] and inside the canvas
    (with a small relative slack so quantization never dead-ends).  Vectorized."""
    gx = np.arange(GRID) * (Wc / GRID)
    gy = np.arange(GRID) * (Hc / GRID)
    X = np.tile(gx, GRID)                    # [G*G] lower-left x per cell
    Y = np.repeat(gy, GRID)                  # [G*G] lower-left y per cell
    ok = (X + w <= Wc * (1 + slack) + 1e-9) & (Y + h <= Hc * (1 + slack) + 1e-9)
    if len(placed):
        P = np.asarray(placed, float)        # [p,4]
        ox = np.minimum(X[:, None] + w, P[None, :, 0] + P[None, :, 2]) \
            - np.maximum(X[:, None], P[None, :, 0])
        oy = np.minimum(Y[:, None] + h, P[None, :, 1] + P[None, :, 3]) \
            - np.maximum(Y[:, None], P[None, :, 1])
        ok &= ~((ox > 1e-9) & (oy > 1e-9)).any(axis=1)
    return ok


def snap(x, y, w, h, placed, Wc, Hc):
    """Snap a chosen lower-left corner to the nearest abutment (placed block edge
    or canvas wall) within one cell -- recovers the interlocked tiling that grid
    quantization loses.  Only snaps if the snapped position stays overlap-free."""
    cw, ch = Wc / GRID, Hc / GRID
    cand_x = [0.0] + [p[0] + p[2] for p in placed] + [p[0] - w for p in placed]
    cand_y = [0.0] + [p[1] + p[3] for p in placed] + [p[1] - h for p in placed]
    bx = min((c for c in cand_x if abs(c - x) <= cw and c >= -1e-9),
             key=lambda c: abs(c - x), default=x)
    by = min((c for c in cand_y if abs(c - y) <= ch and c >= -1e-9),
             key=lambda c: abs(c - y), default=y)
    for tx, ty in ((bx, by), (bx, y), (x, by)):
        good = True
        for p in placed:
            ox = min(tx + w, p[0] + p[2]) - max(tx, p[0])
            oy = min(ty + h, p[1] + p[3]) - max(ty, p[1])
            if ox > 1e-9 and oy > 1e-9:
                good = False
                break
        if good:
            return tx, ty
    return x, y


# ---------------- per-case static preparation ----------------

def prep_case(area, cons, b2b, p2b, pins, Wc, Hc, gt_xywh=None, tp=None):
    """Build the static per-case arrays shared by every step of the case.

    area [n], cons [n,>=5] (fixed, preplaced, mib, cluster, bcode),
    b2b/p2b [e,3] (already -1-filtered), pins [t,2].
    gt_xywh [n,4] (training; raw coords, origin-shifted) or None (inference).
    tp [n,4] target_positions (inference; -1 = unset) or None.
    Returns dict with: static [n,17], w/h per block (locked dims or sqrt-area
    stubs), is_pre, is_fixed, is_soft, mib, order, deg, nbr (b2b adjacency),
    and pre-placed geometry."""
    n = len(area)
    is_fixed = cons[:, 0] != 0
    is_pre = cons[:, 1] != 0
    is_soft = ~(is_fixed | is_pre)
    mib = cons[:, 2].astype(int) if cons.shape[1] > 2 else np.zeros(n, int)
    bcode = cons[:, 4].astype(int) if cons.shape[1] > 4 else np.zeros(n, int)

    deg = np.zeros(n)
    wsum_p2b = np.zeros(n)
    tpullx = np.full(n, 0.5)
    tpully = np.full(n, 0.5)
    if b2b is not None and len(b2b):
        for a, b, wgt in b2b:
            deg[int(a)] += 1
            deg[int(b)] += 1
    if p2b is not None and len(p2b) and pins is not None and len(pins):
        acc = np.zeros((n, 2))
        for pi, bi, wgt in p2b:
            pi, bi = int(pi), int(bi)
            if pi < len(pins):
                acc[bi] += wgt * pins[pi]
                wsum_p2b[bi] += wgt
            deg[bi] += 1
        nz = wsum_p2b > 0
        tpullx[nz] = acc[nz, 0] / wsum_p2b[nz] / Wc
        tpully[nz] = acc[nz, 1] / wsum_p2b[nz] / Hc

    # locked dims (fixed/preplaced) or sqrt-area square stub for soft
    w0 = np.sqrt(np.maximum(area, 1e-9))
    h0 = w0.copy()
    px = np.zeros(n)
    py = np.zeros(n)
    if gt_xywh is not None:                       # training: dims from GT
        lock = is_fixed | is_pre
        w0[lock] = gt_xywh[lock, 2]
        h0[lock] = gt_xywh[lock, 3]
        px[is_pre] = gt_xywh[is_pre, 0]
        py[is_pre] = gt_xywh[is_pre, 1]
    elif tp is not None:                          # inference: dims from tp
        for i in range(n):
            if tp[i, 2] > 0 and tp[i, 3] > 0:
                w0[i], h0[i] = tp[i, 2], tp[i, 3]
            if is_pre[i] and tp[i, 0] >= 0:
                px[i], py[i] = tp[i, 0], tp[i, 1]

    ca = Wc * Hc
    st = np.zeros((n, 17), dtype=np.float32)
    st[:, 0] = np.log1p(area / ca)
    st[:, 1] = np.sqrt(area / ca)
    st[:, 2] = is_fixed
    st[:, 3] = is_pre
    st[:, 4] = mib > 0
    st[:, 5] = (cons[:, 3] > 0) if cons.shape[1] > 3 else 0
    st[:, 6] = (bcode & 1) > 0
    st[:, 7] = (bcode & 2) > 0
    st[:, 8] = (bcode & 4) > 0
    st[:, 9] = (bcode & 8) > 0
    st[:, 10] = np.log1p(deg)
    st[:, 11] = np.log1p(wsum_p2b)
    st[:, 12] = w0 / Wc
    st[:, 13] = h0 / Hc
    st[:, 14] = tpullx
    st[:, 15] = tpully
    st[:, 16] = np.log1p(area / max(np.median(area), 1e-9))

    nbr = [[] for _ in range(n)]                  # b2b adjacency (j, weight)
    if b2b is not None and len(b2b):
        for a, b, wgt in b2b:
            a, b = int(a), int(b)
            nbr[a].append((b, float(wgt)))
            nbr[b].append((a, float(wgt)))

    return {
        "static": st, "w": w0, "h": h0, "px": px, "py": py,
        "is_pre": is_pre, "is_fixed": is_fixed, "is_soft": is_soft,
        "mib": mib, "deg": deg, "nbr": nbr,
        "order": canonical_order(area, is_pre, deg),
        "Wc": Wc, "Hc": Hc, "n": n,
    }


def step_tokens(case, placed_mask, placed_xy, cur):
    """Token tensor [MAX_N, TOKEN_DIM] for one decoding step.
    placed_mask [n] bool; placed_xy [n,2] lower-left (raw); cur = block index."""
    n = case["n"]
    Wc, Hc = case["Wc"], case["Hc"]
    t = np.zeros((MAX_N, TOKEN_DIM), dtype=np.float32)
    t[:n, :17] = case["static"]
    t[:n, 17] = placed_mask
    t[:n, 18] = np.where(placed_mask, placed_xy[:, 0] / Wc, 0.0)
    t[:n, 19] = np.where(placed_mask, placed_xy[:, 1] / Hc, 0.0)
    t[:n, 20] = np.where(placed_mask, case["w"] / Wc, 0.0)
    t[:n, 21] = np.where(placed_mask, case["h"] / Hc, 0.0)
    t[cur, 22] = 1.0
    # wire pull toward already-placed b2b neighbours of the current block
    acc = np.zeros(2)
    wsum = 0.0
    for j, wgt in case["nbr"][cur]:
        if placed_mask[j]:
            acc += wgt * (placed_xy[j] + 0.5 * np.array([case["w"][j], case["h"][j]]))
            wsum += wgt
    t[cur, 23] = math.log1p(wsum)
    t[cur, 24] = acc[0] / wsum / Wc if wsum > 0 else 0.5
    t[cur, 25] = acc[1] / wsum / Hc if wsum > 0 else 0.5
    pad = np.zeros(MAX_N, dtype=bool)
    pad[:n] = True
    return t, pad
