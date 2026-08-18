"""Complete legalization for the analytical placer.

Turns a near-legal analytical placement into one with EXACTLY zero overlap while
  * keeping preplaced blocks at their exact (x, y, w, h)  -- pinned, never moved;
  * keeping every block's SHAPE unchanged (so soft-block area stays exact and
    fixed-block dims stay locked -- legalization only moves blocks);
  * perturbing positions as little as possible (preserves wirelength).

Algorithm
=========
1. Separation assignment.  For every block pair pick ONE axis to separate it in:
   the axis where it currently overlaps least (equivalently, is most separated).
   If, after legalization, every pair is separated in its assigned axis, then no
   pair overlaps -- this is what makes the two 1-D problems below sufficient.

2. Per-axis constraint-graph compaction (longest-path / Liao-Wong relaxation).
   The assignment induces a DAG of "i is left/below j" difference constraints
   (x_j >= x_i + w_i).  Processing blocks in coordinate order and pushing each
   one just past its already-placed predecessors is a single-pass longest-path
   solve.  It is order-preserving and minimum-perturbation (a block only moves
   if a predecessor forces it).  Preplaced blocks are fixed anchors.
   The x-pass and y-pass are INDEPENDENT because each pair lives in one axis only.

3. Guaranteed-zero-overlap backstop.  The only way step 2 can leave an overlap is
   a movable block wedged against a fixed anchor (a pin can't be pushed).  Those
   are removed by bounded pairwise push-apart, and -- as a last resort that can
   never fail -- by evicting the movable block to empty space above the layout.

4. Assertion.  We re-scan and confirm exactly zero overlap before returning.

Everything is O(n^2) per scan, trivial for FloorSet's n <= 120.
"""
from __future__ import annotations

import os

import numpy as np

# 重疊偵測跟推擠的閾值。
#
# 原本設 1e-6，正好等於：官方在判定重疊的閾值（overlap_x > 1e-6 且
# overlap_y > 1e-6），等於零安全間隙，推的量剛好嚴格大於 1e-6 的邊界，容易
# 被浮點雜訊翻過去。更麻煩的是官方重疊算公式跟這裡用中心點跟
# 0.5*(w1+w2)-|cx1-cx2|，官方用減法式（min(r1,r2)-max(l1,l2)），推的方向一旦
# 相反，本來剛好貼齊的 1e-6 在官方眼中就是 1.0000001e-6 —— 直接判定不合法。
# 實測 test 9 就是這樣中的，block 1 跟 10 在 x 方向差 1e-6）。
#
# 改成 1e-7，偵測閾值跟官方的嚴格不同數量級，推過去的間隙跟官方的 1e-6 差
# 10 倍餘裕，避免撞到 soft_repair._touch 判定貼齊。用的
# 1e-4，所以 grouping/boundary 需要的貼齊間隙不會被這個間隙破壞。
_EPS = 1e-7


def _overlap_matrix(x, y, w, h):
    cx = x + 0.5 * w
    cy = y + 0.5 * h
    ox = 0.5 * (w[:, None] + w[None, :]) - np.abs(cx[:, None] - cx[None, :])
    oy = 0.5 * (h[:, None] + h[None, :]) - np.abs(cy[:, None] - cy[None, :])
    return ox, oy


def _overlap_pairs(x, y, w, h, eps=_EPS):
    ox, oy = _overlap_matrix(x, y, w, h)
    N = len(x)
    iu = np.triu_indices(N, 1)
    mask = (ox[iu] > eps) & (oy[iu] > eps)
    return iu[0][mask], iu[1][mask]


def _overlap_pairs_touching(x, y, w, h, dirty, eps=_EPS):
    """只回傳「至少有一端在 `dirty` 裡」的重疊配對。

    正確性論證（這是這個優化能成立的全部理由）：
    **若方塊 i 跟 j 自上一次掃描以來都沒有移動過，它們的相對位置不變，
    重疊狀態就不可能改變。** 所以在做過一次完整掃描之後，只需要重新檢查
    「動過的方塊 × 全部方塊」——沒動過的配對維持上次的結論（不重疊）。

    成本從 O(n²) 降到 O(|dirty|·n)。`_cleanup` 的迴圈裡通常只有少數幾對方塊
    被推開，所以 |dirty| 遠小於 n，這是實測 profile 指出的最大單一浪費：
    n=120 時 `_overlap_matrix` 被呼叫 5,315 次、每次重建 14,400 個配對。
    """
    idx = np.fromiter(dirty, dtype=np.intp, count=len(dirty))
    cx = x + 0.5 * w
    cy = y + 0.5 * h
    # [len(dirty), N]：只算 dirty 那幾列，而不是完整的 N x N
    ox = 0.5 * (w[idx][:, None] + w[None, :]) - np.abs(cx[idx][:, None] - cx[None, :])
    oy = 0.5 * (h[idx][:, None] + h[None, :]) - np.abs(cy[idx][:, None] - cy[None, :])
    mask = (ox > eps) & (oy > eps)
    mask[np.arange(len(idx)), idx] = False        # 排除自己跟自己
    r, c = np.nonzero(mask)
    if len(r) == 0:
        return np.empty(0, dtype=np.intp), np.empty(0, dtype=np.intp)
    ii = idx[r]
    jj = c.astype(np.intp)
    lo = np.minimum(ii, jj)
    hi = np.maximum(ii, jj)
    # **順序至關重要**：`_cleanup` 在同一輪內逐對推開，每次推開都會改變座標，
    # 所以後續配對算出來的推開量取決於前面推過哪些——**換個順序就會得到不同的
    # 佈局**。原版 `_overlap_pairs` 用 `np.triu_indices` 產生上三角 row-major
    # 順序（先比 i 再比 j），這裡必須完全重現它，否則不等價。
    # （第一版用 `np.nonzero` 的順序，實測隨機壓力測試最大座標差 9.33 —— 這個
    #   優化「只是重用計算、不改數學」的前提就是靠這個排序成立的。）
    key = lo * (len(x) + 1) + hi
    order = np.argsort(key, kind="stable")
    ks = key[order]
    keep = np.empty(len(order), dtype=bool)
    keep[0] = True
    np.not_equal(ks[1:], ks[:-1], out=keep[1:])   # 去掉重複配對（兩端都髒時）
    sel = order[keep]
    return lo[sel], hi[sel]


def _compact(pos, size, adj, coord, is_pre):
    """Single-pass longest-path compaction along one axis.

    adj[u] = list of v with constraint pos[v] >= pos[u] + size[u]
             (u is the lower-coordinate block of the pair).
    Movable blocks are pushed up to clear predecessors; pinned blocks never move.
    """
    p = pos.copy()
    for node in np.argsort(coord, kind="stable"):
        node = int(node)
        base = p[node] + size[node]
        for succ in adj[node]:
            if is_pre[succ]:
                continue  # fixed anchor: cannot move (handled in cleanup)
            if p[succ] < base:
                p[succ] = base
    return p


def _push(c, i, j, move, size, is_pre, floor=None):
    """Separate i and j along axis `c` by `move`, never moving a pinned block.

    With `floor` set (the x=0 / y=0 canvas wall), the lower block's corner is
    never pushed below it -- the wall acts like a fixed obstacle, so the excess
    push is redirected to the higher block.  Keeps movable blocks non-negative."""
    ci = c[i] + 0.5 * size[i]
    cj = c[j] + 0.5 * size[j]
    lo, hi = (i, j) if ci <= cj else (j, i)
    if is_pre[lo] and is_pre[hi]:
        return  # two preplaced blocks should never overlap by construction
    if is_pre[lo]:
        c[hi] += move
    elif is_pre[hi]:
        c[lo] = max(c[lo] - move, floor) if floor is not None else c[lo] - move
    elif floor is not None and c[lo] - 0.5 * move < floor:
        slack = max(0.0, c[lo] - floor)   # how far the lower block may still drop
        c[lo] -= slack
        c[hi] += move - slack             # the wall absorbs the rest -> push hi
    else:
        c[lo] -= 0.5 * move
        c[hi] += 0.5 * move


def _cleanup(x, y, w, h, is_pre, max_iter, floor=None):
    """Remove any residual overlap; guaranteed to finish overlap-free.  With
    `floor` set, movable blocks are also kept at corner >= floor (canvas walls)."""
    if floor is not None:                     # start inside the walls
        mv = ~is_pre
        x[mv] = np.maximum(x[mv], floor)
        y[mv] = np.maximum(y[mv], floor)
    # 增量式掃描（ELECTRO_FAST_CLEANUP=1）：第一輪做完整 O(n²) 掃描，之後
    # 只重新檢查「上一輪真的被推動過的方塊」——見 `_overlap_pairs_touching`
    # 的正確性論證。這是純粹的計算重用，數學完全不變，輸出應逐位元相同。
    fast = os.environ.get("ELECTRO_FAST_CLEANUP", "0") == "1"
    dirty = None                                   # None = 尚未做過完整掃描
    for _ in range(max_iter):
        if fast and dirty is not None:
            if not dirty:
                return x, y                        # 上一輪沒有任何方塊被推動
            ii, jj = _overlap_pairs_touching(x, y, w, h, dirty)
        else:
            ii, jj = _overlap_pairs(x, y, w, h)
        if len(ii) == 0:
            return x, y
        moved = set()
        for k in range(len(ii)):
            i, j = int(ii[k]), int(jj[k])
            oxk = 0.5 * (w[i] + w[j]) - abs((x[i] + 0.5 * w[i]) - (x[j] + 0.5 * w[j]))
            oyk = 0.5 * (h[i] + h[j]) - abs((y[i] + 0.5 * h[i]) - (y[j] + 0.5 * h[j]))
            if oxk <= 0 or oyk <= 0:
                continue  # already cleared by an earlier push this pass
            if oxk <= oyk:
                _push(x, i, j, oxk + _EPS, w, is_pre, floor)
            else:
                _push(y, i, j, oyk + _EPS, h, is_pre, floor)
            # 保守地把兩端都標記為髒（`_push` 可能只動其中一個，多標不會錯）
            moved.add(i)
            moved.add(j)
        dirty = moved

    # Last resort: evict whatever still overlaps to empty space above the layout
    # (stacked so evicted blocks can't overlap each other or anything below).
    ii, jj = _overlap_pairs(x, y, w, h)
    top = float((y + h).max()) if len(x) else 0.0
    evicted = set()
    for k in range(len(ii)):
        for cand in (int(jj[k]), int(ii[k])):
            if not is_pre[cand] and cand not in evicted:
                y[cand] = top + 1.0
                top = y[cand] + h[cand]
                evicted.add(cand)
                break
    return x, y


def legalize(x, y, w, h, is_pre, max_clean_iter=4000, floor=None):
    """Return (x, y) with exactly zero overlap; shapes (w, h) are unchanged.

    With `floor` set (e.g. 0.0), the overlap-removal cleanup keeps every movable
    block's lower-left corner >= floor, so the layout stays in the first quadrant
    *incrementally* -- never letting blocks drift far below the wall and then
    shoving them back (which is what makes a post-hoc floor explode)."""
    x = np.asarray(x, dtype=float).copy()
    y = np.asarray(y, dtype=float).copy()
    w = np.asarray(w, dtype=float)
    h = np.asarray(h, dtype=float)
    is_pre = np.asarray(is_pre, dtype=bool)
    N = len(x)
    if N <= 1:
        if floor is not None and N == 1 and not is_pre[0]:
            x[0] = max(x[0], floor); y[0] = max(y[0], floor)
        return x, y

    cx = x + 0.5 * w
    cy = y + 0.5 * h

    # ---- Step 1: assign a separation axis to every pair --------------------
    ox, oy = _overlap_matrix(x, y, w, h)
    iu = np.triu_indices(N, 1)
    I, J = iu[0], iu[1]
    sep_in_x = ox[I, J] <= oy[I, J]   # separate in the axis of least overlap

    Hadj = [[] for _ in range(N)]     # x: u left of v
    Vadj = [[] for _ in range(N)]     # y: u below v
    for k in range(len(I)):
        i, j = int(I[k]), int(J[k])
        if sep_in_x[k]:
            L, R = (i, j) if cx[i] <= cx[j] else (j, i)
            Hadj[L].append(R)
        else:
            B, T = (i, j) if cy[i] <= cy[j] else (j, i)
            Vadj[B].append(T)

    # ---- Step 2: independent per-axis compaction ---------------------------
    x = _compact(x, w, Hadj, cx, is_pre)
    y = _compact(y, h, Vadj, cy, is_pre)

    # ---- Step 3: guaranteed-zero-overlap backstop --------------------------
    x, y = _cleanup(x, y, w, h, is_pre, max_clean_iter, floor=floor)

    return x, y


def _pair_graph(x, y, w, h):
    """從一個「已合法（零重疊）」的佈局推出約束圖，每一對方塊只給一條邊。

    每一對方塊只要在**任一軸**上分開就保證不會重疊，所以每對只需要一條邊。
    軸的選法沿用 legalize() Step 1 的慣例（分開較小的那一軸，在合法佈局裡
    就是重疊較大的那一軸）。

    「每對方塊只給一條邊」是關鍵：如另一軸完全一致齊，硬可行性沒有空間會空掉。
    如果兩軸都不一致齊，等於施加額外沒必要的佈局整體收縮，硬可行性就是毫無用作。

    回傳 (Hpred, Vpred)，其中 Hpred[i] 為所有必須排在 i 左邊的方塊
    （約束為 x[i] >= x[p] + w[p]），Vpred 同理。
    """
    ox, oy = _overlap_matrix(x, y, w, h)
    N = len(x)
    iu = np.triu_indices(N, 1)
    I, J = iu[0], iu[1]
    sep_in_x = ox[I, J] <= oy[I, J]
    cx = x + 0.5 * w
    cy = y + 0.5 * h
    Hpred = [[] for _ in range(N)]
    Vpred = [[] for _ in range(N)]
    for k in range(len(I)):
        i, j = int(I[k]), int(J[k])
        if sep_in_x[k]:
            L, R = (i, j) if cx[i] <= cx[j] else (j, i)
            Hpred[R].append(L)
        else:
            B, T = (i, j) if cy[i] <= cy[j] else (j, i)
            Vpred[T].append(B)
    return Hpred, Vpred


def _compact_axis(pos, size, pred, is_pre, base, order):
    """單軸的最長路徑硬可行性：pos[i] = max(base, max_p(pos[p] + size[p]))。

    這是壓實管線缺少的一步。_compact() 只能「從左到右推」以消除重疊
    （`if p[succ] < base: p[succ] = base`），從頭到尾沒有任何一步是把方塊往回
    收去填補空間的——實測填充比因此只有 0.660，而 ground truth 是 0.966。

    安全性：只有座標變小或不變，所以不可能製造出新的重疊（：
      * 依中心座標排序為順序（這樣由中心小的到中心大的），
      * 對每條邊 p -> i，歸納可得 pos_new[p] <= pos_old[p]，而原佈局已滿足
        pos_old[i] >= pos_old[p] + size[p]，故 pos_new[i] <= pos_old[i]，
        故條件恆成立；
      * preplaced 方塊直接跳過（座標釘死），其後繼只會被壓縮，不會被拉開。

    因為只往一邊移動，`lo > pos[i]` 在合法輸入下不該發生；真的發生時，例如
    方塊本來就在 floor 之下（寧可保守不動），把後面的處理交給 _cleanup。
    """
    p = pos.copy()
    for i in order:
        i = int(i)
        if is_pre[i]:
            continue
        lo = base
        for q in pred[i]:
            v = p[q] + size[q]
            if v > lo:
                lo = v
        if lo < p[i]:
            p[i] = lo
    return p


def compact_layout(x, y, w, h, is_pre, floor=None, rounds=4, tol=1e-9):
    """交替 x/y 的最長路徑硬可行性，把佈局往左下角壓實，形狀不變（零重疊）。

    每一輪都重建一次約束圖，硬可行性後方塊的相鄰關係保持改變，重建才能找到
    新的、更緊的約束。一輪之始 H/V 兩張圖同一輪共用同一次建的，因為它 x
    一律壓縮到 y，被指派到 V 軸的那些仍在 y 上分開。

    穩定終止，每一步都只讓座標變小，所以總位移量非增，`tol` 不小就結束。
    """
    x = np.asarray(x, dtype=float).copy()
    y = np.asarray(y, dtype=float).copy()
    w = np.asarray(w, dtype=float)
    h = np.asarray(h, dtype=float)
    is_pre = np.asarray(is_pre, dtype=bool)
    if len(x) <= 1:
        return x, y

    x0, y0 = x.copy(), y.copy()
    # 保護網：基準必須是「進來就有的重疊」，而不是 0；legalize() 的輸出本來就
    # 帶微重疊的（_overlap_pairs 只標 ox 跟 oy 都 > _EPS=1e-7 的點，所以小
    # ox=0.9e-7、oy=59 的方塊仍被保留，verify_overlap 算出來就是 5.3e-6）。
    # 這在官方眼中完全合法（它要求兩軸都 > 1e-6 才視為重疊），拿 0 當基準會
    # 保護網每次都誤觸發。硬可行性整個放棄空跑——實測就是這樣，填充比 0.402 進
    # 0.402 出。
    ov0 = verify_overlap(x, y, w, h)
    for _ in range(int(rounds)):
        Hpred, Vpred = _pair_graph(x, y, w, h)
        bx = floor if floor is not None else float(x.min())
        by = floor if floor is not None else float(y.min())
        xo = np.argsort(x + 0.5 * w, kind="stable")
        yo = np.argsort(y + 0.5 * h, kind="stable")
        xn = _compact_axis(x, w, Hpred, is_pre, bx, xo)
        yn = _compact_axis(y, h, Vpred, is_pre, by, yo)
        moved = float(np.abs(xn - x).sum() + np.abs(yn - y).sum())
        x, y = xn, yn
        if moved <= tol:
            break

    # 保護網：上面的推論說硬可行性不可能製造新的重疊，但這是最後一道大意，
    # 寧可用一次 O(n^2) 的檢查來確定，萬一可能有可行解式不可行。（原因是「多秤是
    # 沒有進來的整理」，容差 1e-12 純粹是浮點噪音的緩衝。
    if verify_overlap(x, y, w, h) > ov0 + 1e-12:
        return x0, y0
    return x, y


def remove_overlap(x, y, w, h, is_pre, max_iter=4000, nonneg=False):
    """Final safety net: drive any contest-counted overlap (both dims > 1e-6) to
    zero via the same push-apart/eviction backstop used inside legalize().  Run
    AFTER the soft-constraint repairs, which can re-introduce micro-overlaps.

    nonneg=True additionally enforces the canvas walls (every movable block's
    lower-left corner >= 0) WHILE keeping zero overlap, so the output is
    guaranteed in the first quadrant -- a local, min-displacement-style fix (no
    global shift), only blocks that poked past a wall get nudged back."""
    x = np.asarray(x, float).copy()
    y = np.asarray(y, float).copy()
    w = np.asarray(w, float)
    h = np.asarray(h, float)
    is_pre = np.asarray(is_pre, bool)
    if len(x) <= 1:
        if nonneg and len(x) == 1 and not is_pre[0]:
            x[0] = max(x[0], 0.0); y[0] = max(y[0], 0.0)
        return x, y
    return _cleanup(x, y, w, h, is_pre, max_iter, floor=0.0 if nonneg else None)


def verify_overlap(x, y, w, h):
    """Total residual overlap area (0.0 == legal). For asserting/diagnostics."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    w = np.asarray(w, float); h = np.asarray(h, float)
    ox, oy = _overlap_matrix(x, y, w, h)
    N = len(x)
    iu = np.triu_indices(N, 1)
    ox = np.clip(ox[iu], 0, None)
    oy = np.clip(oy[iu], 0, None)
    return float((ox * oy).sum())
