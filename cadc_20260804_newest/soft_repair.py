"""Deterministic soft-constraint repair passes, run AFTER legalization.

Mirrors the C++ SA solver's boundary / grouping repair logic, but in numpy and
operating on an already-overlap-free placement.  Every move is guarded: a block
is only relocated if its destination cell is free, so zero overlap is preserved.

  * boundary_snap   : slide each boundary block so its required edge coincides
                      with the layout bbox extreme (V_boundary).
  * grouping_repair : abut isolated cluster members to a connected sibling so the
                      group forms one connected component (V_grouping).

Preplaced blocks are never moved.
"""
from __future__ import annotations

import numpy as np

_EPS = 1e-6


def _free(idx, nx, ny, x, y, w, h, ignore=None):
    """True if placing block idx at (nx,ny) overlaps no other block.

    這是最裡層的頻繁呼叫（grouping_repair 找候選位置時可能被呼叫一千次）。
    試過改成 numpy 一次比對所有方塊（換算後反而更慢，1.87s -> 2.24s），n<=120
    時這種置換的迴圈長度大於迴圈本身的成本，一有不合法就 return。維持原樣。
    """
    N = len(x)
    l2, r2, b2, t2 = nx, nx + w[idx], ny, ny + h[idx]
    for j in range(N):
        if j == idx or (ignore is not None and j in ignore):
            continue
        if (l2 < x[j] + w[j] - _EPS and x[j] < r2 - _EPS and
                b2 < y[j] + h[j] - _EPS and y[j] < t2 - _EPS):
            return False
    return True


def _slot_along_y(i, X, x, y, w, h, ymn, ymx, floor=None):
    """Fix block i's x to X; find the NEAREST free y in [ymn, ymx-h_i].

    Min-displacement: among the discrete candidate slots (stay, the two wall
    ends, and the tight-pack positions just above/below each column neighbour)
    pick the free one closest to the block's current y -- this keeps the
    boundary block on its wall while moving it as little as possible, so
    satisfying the boundary constraint costs the least possible wirelength."""
    R = X + w[i]
    cands = [y[i], ymn, ymx - h[i]]
    for j in range(len(x)):
        if j == i:
            continue
        if x[j] < R - _EPS and X < x[j] + w[j] - _EPS:    # shares the column
            cands.append(y[j] + h[j])                      # above j
            cands.append(y[j] - h[i])                      # below j
    lo = ymn if floor is None else max(ymn, floor)
    valid = sorted({c for c in cands if lo - _EPS <= c <= ymx - h[i] + _EPS},
                   key=lambda c: abs(c - y[i]))
    for yy in valid:
        if _free(i, X, yy, x, y, w, h):
            return yy
    return None


def _slot_along_x(i, Y, x, y, w, h, xmn, xmx, floor=None):
    """Fix block i's y to Y; find the NEAREST free x in [xmn, xmx-w_i]
    (min-displacement; mirror of _slot_along_y)."""
    T = Y + h[i]
    cands = [x[i], xmn, xmx - w[i]]
    for j in range(len(x)):
        if j == i:
            continue
        if y[j] < T - _EPS and Y < y[j] + h[j] - _EPS:    # shares the row
            cands.append(x[j] + w[j])
            cands.append(x[j] - w[i])
    lo = xmn if floor is None else max(xmn, floor)
    valid = sorted({c for c in cands if lo - _EPS <= c <= xmx - w[i] + _EPS},
                   key=lambda c: abs(c - x[i]))
    for xx in valid:
        if _free(i, xx, Y, x, y, w, h):
            return xx
    return None


def _pair_move_ok(i, j, nxi, nyi, nxj, nyj, x, y, w, h, floor, bb):
    """把 i 移到 (nxi,nyi)，同時把 j 移到 (nxj,nyj) 之後是否仍然合法。

    三個條件：兩塊都不低於 floor；兩塊都仍在目前 bbox 內（bbox 不變，其他方塊
    的重疊判定不會跟著移動）；移完之後 i 的新位置跟 j 的新位置彼此不重疊（這一
    項 `_free` 拿彼此 ignore 才不會誤判，必須另外檢查）。
    """
    if floor is not None and min(nxi, nyi, nxj, nyj) < floor - _EPS:
        return False
    xmn, xmx, ymn, ymx = bb
    for px, py, bw, bh in ((nxi, nyi, w[i], h[i]), (nxj, nyj, w[j], h[j])):
        if (px < xmn - _EPS or px + bw > xmx + _EPS or
                py < ymn - _EPS or py + bh > ymx + _EPS):
            return False
    if (nxi < nxj + w[j] - _EPS and nxj < nxi + w[i] - _EPS and
            nyi < nyj + h[j] - _EPS and nyj < nyi + h[i] - _EPS):
        return False
    return (_free(i, nxi, nyi, x, y, w, h, ignore={j}) and
            _free(j, nxj, nyj, x, y, w, h, ignore={i}))


def _on_wall(j, c, x, y, w, h, bb):
    """方塊 j 目前是否已經站在位碼 c 要求的那個面（或那兩面）上。"""
    xmn, xmx, ymn, ymx = bb
    if c & 1 and abs(x[j] - xmn) >= 1e-6:
        return False
    if c & 2 and abs(x[j] + w[j] - xmx) >= 1e-6:
        return False
    if c & 4 and abs(y[j] + h[j] - ymx) >= 1e-6:
        return False
    if c & 8 and abs(y[j] - ymn) >= 1e-6:
        return False
    return True


def boundary_snap(x, y, w, h, bcode, is_pre, passes=3, floor=None,
                  clust_id=None, max_swaps=24):
    """Slide each boundary block onto its required bbox edge, searching along the
    wall for a free slot (not just the exact current spot).  With `floor` set,
    movable blocks are kept at corner >= floor (first-quadrant containment).

    多跑幾輪找不到位置就試「交換」：把 i 跟目前正站在該面的其他方塊互調。
    grouping_repair 的經驗完全適用於這裡——legalize 硬可行性之後往往沒有空隙，
    只有「已經站別人」，所以直接貼牆的版本一遇到滿的就直接放棄。修好 grouping
    之後 boundary 反而變成最大宗的殘留違規來源（大案例 110/235，46.8%）就是這個
    原因。交換一樣要保證：bbox 不變，且 i 跟 j 的 boundary（兩者所屬群組的
    grouping）合計不得比之前更差。
    """
    x = np.asarray(x, float); y = np.asarray(y, float)
    w = np.asarray(w, float); h = np.asarray(h, float)
    bcode = np.asarray(bcode).astype(int); is_pre = np.asarray(is_pre, bool)
    cl = None if clust_id is None else np.asarray(clust_id).astype(int)
    N = len(x)
    for _ in range(passes):
        xmn = x.min(); xmx = (x + w).max(); ymn = y.min(); ymx = (y + h).max()
        bb = (xmn, xmx, ymn, ymx)
        swap_budget = max_swaps
        moved = False
        for i in range(N):
            c = int(bcode[i])
            if c == 0 or is_pre[i]:
                continue
            want_x = (c & 1) or (c & 2)
            want_y = (c & 4) or (c & 8)
            X = xmn if (c & 1) else (xmx - w[i]) if (c & 2) else x[i]
            Y = ymn if (c & 8) else (ymx - h[i]) if (c & 4) else y[i]

            placed = False
            if want_x and want_y:                      # corner: one exact spot
                if _free(i, X, Y, x, y, w, h) and (abs(X - x[i]) > _EPS or abs(Y - y[i]) > _EPS):
                    x[i], y[i] = X, Y; moved = True; placed = True
            elif want_x:                               # left/right wall: slide y
                yy = _slot_along_y(i, X, x, y, w, h, ymn, ymx, floor)
                if yy is not None and (abs(X - x[i]) > _EPS or abs(yy - y[i]) > _EPS):
                    x[i], y[i] = X, yy; moved = True; placed = True
            elif want_y:                               # top/bottom wall: slide x
                xx = _slot_along_x(i, Y, x, y, w, h, xmn, xmx, floor)
                if xx is not None and (abs(xx - x[i]) > _EPS or abs(Y - y[i]) > _EPS):
                    x[i], y[i] = xx, Y; moved = True; placed = True

            # -- 沒放進去就試交換：跟目前佔著該面的鄰居調換 -----------------
            if placed or swap_budget <= 0:
                continue
            if _on_wall(i, c, x, y, w, h, bb):
                continue                                # 本來就合格，不用調
            cands = [j for j in range(N)
                     if j != i and not is_pre[j] and _on_wall(j, c, x, y, w, h, bb)]
            cands.sort(key=lambda j: abs(x[j] - x[i]) + abs(y[j] - y[i]))
            for j in cands[:3]:
                if swap_budget <= 0:
                    break
                # i 去對面的位置（沿著它的方向座標沿用 j 的），j 回到 i 原本的位置
                nxi = xmn if (c & 1) else (xmx - w[i]) if (c & 2) else x[j]
                nyi = ymn if (c & 8) else (ymx - h[i]) if (c & 4) else y[j]
                nxj, nyj = x[i], y[i]
                if not _pair_move_ok(i, j, nxi, nyi, nxj, nyj, x, y, w, h, floor, bb):
                    continue
                swap_budget -= 1
                gi = int(cl[i]) if cl is not None else 0
                gj = int(cl[j]) if cl is not None else 0

                def _v():
                    v = ((0 if _boundary_ok(i, bcode, x, y, w, h, bb) else 1) +
                         (0 if _boundary_ok(j, bcode, x, y, w, h, bb) else 1))
                    if cl is not None:
                        if gi:
                            v += _group_v(gi, cl, x, y, w, h)
                        if gj and gj != gi:
                            v += _group_v(gj, cl, x, y, w, h)
                    return v

                before = _v()
                oxi, oyi, oxj, oyj = x[i], y[i], x[j], y[j]
                x[i], y[i], x[j], y[j] = nxi, nyi, nxj, nyj
                if _bbox_same(x, y, w, h, bb) and _v() < before:
                    moved = True
                    break
                x[i], y[i], x[j], y[j] = oxi, oyi, oxj, oyj   # 撤回
        if not moved:
            break
    return x, y


def mib_unify(x, y, w, h, mib_id, is_fixed, is_pre, areas, floor=None,
              max_shift=True):
    """事後 MIB 形狀統一（post-hoc，作用在最終幾何上，不動優化過程）。

    為什麼是事後做，而不是在 analytical_place 裡鎖死
    -------------------------------------------------
    先前試過在 `shapes()` 裡就把 MIB 群組的成員強制成錨點形狀，結果慘敗
    （單案 area_gap 暴增到 183.9%，cost 1.8485 -> 5.335）：那等於從第 0 輪就
    拿掉梯度優化的一整個自由度，整個佈局被扭曲。這裡改成**最終幾何算完之後**
    才動形狀，優化過程完全不受影響，而且每一次改形狀都有守衛，改不動就跳過。

    為什麼這是合法的（不會踩到 1% 面積硬約束）
    ------------------------------------------
    實測 100 個 MIB 群組**全部都是等面積**，且 56 個「混合剛性+軟方塊」的群組
    裡，剛性成員（fixed/preplaced）的 `w*h` 都精確等於該群組的共享目標面積。
    所以軟成員採用剛性成員的 `(w,h)` 時，面積 `w*h` 仍然精確命中自己的目標，
    硬約束毫髮無傷——真正的 V_mib 不可約下限是 **0**。

    做法
    ----
    * 有剛性錨點的群組：所有軟成員改用錨點的 `(w,h)`。
    * 沒有剛性錨點的群組：挑「最多成員能採用」的那個成員形狀當共同形狀。
    * 每次改形狀都先檢查改完不會跟任何方塊重疊、不超出目前 bbox；試幾個
      對齊方式（保留左下角、保留中心、往內縮回 bbox），第一個過關的就採用，
      全都不行就跳過該成員（部分統一仍然能減少「相異形狀數 - 1」）。

    這是嚴格加法式的修復：改不動就維持原狀，不可能比進來時更差。
    """
    x = np.asarray(x, float).copy(); y = np.asarray(y, float).copy()
    w = np.asarray(w, float).copy(); h = np.asarray(h, float).copy()
    mib_id = np.asarray(mib_id).astype(int)
    rigid = np.asarray(is_fixed, bool) | np.asarray(is_pre, bool)
    areas = np.asarray(areas, float)
    N = len(x)
    if N == 0 or mib_id.size == 0:
        return x, y, w, h
    bb = _bbox(x, y, w, h)
    xmn, xmx, ymn, ymx = bb

    def _fits(i, nx, ny, nw, nh):
        """改形狀後不重疊、不越界、不低於 floor。"""
        if floor is not None and (nx < floor - _EPS or ny < floor - _EPS):
            return False
        if (nx < xmn - _EPS or nx + nw > xmx + _EPS or
                ny < ymn - _EPS or ny + nh > ymx + _EPS):
            return False
        for j in range(N):
            if j == i:
                continue
            if (nx < x[j] + w[j] - _EPS and x[j] < nx + nw - _EPS and
                    ny < y[j] + h[j] - _EPS and y[j] < ny + nh - _EPS):
                return False
        return True

    def _try_resize(i, nw, nh):
        """幾種對齊方式依序試，成功就寫回並回傳 True。"""
        if abs(nw - w[i]) < 1e-9 and abs(nh - h[i]) < 1e-9:
            return True                      # 形狀已經一樣
        cands = [(x[i], y[i])]               # 保留左下角
        if max_shift:
            cx = x[i] + 0.5 * w[i]
            cy = y[i] + 0.5 * h[i]
            cands.append((cx - 0.5 * nw, cy - 0.5 * nh))          # 保留中心
            cands.append((min(x[i], xmx - nw), y[i]))              # 往左縮回界內
            cands.append((x[i], min(y[i], ymx - nh)))              # 往下縮回界內
            cands.append((min(x[i], xmx - nw), min(y[i], ymx - nh)))
        for nx, ny in cands:
            if _fits(i, nx, ny, nw, nh):
                x[i], y[i], w[i], h[i] = nx, ny, nw, nh
                return True
        return False

    for g in range(1, int(mib_id.max()) + 1):
        mem = np.where(mib_id == g)[0]
        if len(mem) < 2:
            continue
        a = float(areas[mem[0]])
        # 群組面積不一致時，形狀本來就不可能全部相同，不強求（也不會有動作）
        if not np.allclose(areas[mem], a, rtol=1e-6):
            continue
        anchors = [i for i in mem if rigid[i]]
        soft = [i for i in mem if not rigid[i]]
        if anchors:
            aw, ah = float(w[anchors[0]]), float(h[anchors[0]])
            # 錨點自己的面積必須就是群組面積，否則採用它會破壞面積硬約束
            if abs(aw * ah - a) > 1e-6 * max(1.0, a):
                continue
            targets = [(aw, ah)]
        else:
            # 沒有錨點：以「成員自己現有的形狀」為候選，挑最多人能採用的
            targets = sorted({(round(float(w[i]), 6), round(float(h[i]), 6))
                              for i in soft},
                             key=lambda s: -sum(1 for i in soft
                                                if abs(w[i] - s[0]) < 1e-9))
        for (tw_, th_) in targets[:3]:
            if abs(tw_ * th_ - a) > 1e-6 * max(1.0, a):
                continue
            for i in soft:
                _try_resize(i, tw_, th_)
            break
    return x, y, w, h


def _touch(i, j, x, y, w, h):
    """True if blocks i,j share an edge segment of positive length (abut)."""
    ox = min(x[i] + w[i], x[j] + w[j]) - max(x[i], x[j])   # x-overlap length
    oy = min(y[i] + h[i], y[j] + h[j]) - max(y[i], y[j])   # y-overlap length
    share_v = (abs(x[i] + w[i] - x[j]) < 1e-4 or abs(x[j] + w[j] - x[i]) < 1e-4) and oy > 1e-4
    share_h = (abs(y[i] + h[i] - y[j]) < 1e-4 or abs(y[j] + h[j] - y[i]) < 1e-4) and ox > 1e-4
    return share_v or share_h


def _components(members, x, y, w, h):
    """Connected components of `members` under the abut relation (union-find)."""
    parent = {m: m for m in members}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for ai in range(len(members)):
        for bi in range(ai + 1, len(members)):
            i, j = members[ai], members[bi]
            if _touch(i, j, x, y, w, h):
                parent[find(i)] = find(j)
    comps = {}
    for m in members:
        comps.setdefault(find(m), []).append(m)
    return list(comps.values())


def soft_violation_counts(x, y, w, h, bcode, clust_id, mib_id):
    """Compute (V_boundary, V_grouping, V_mib, n_soft) exactly as the evaluator."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    w = np.asarray(w, float); h = np.asarray(h, float)
    bcode = np.asarray(bcode).astype(int)
    clust_id = np.asarray(clust_id).astype(int)
    mib_id = np.asarray(mib_id).astype(int)
    N = len(x)

    # boundary
    xmn, xmx = x.min(), (x + w).max()
    ymn, ymx = y.min(), (y + h).max()
    vb = 0
    n_boundary = int((bcode != 0).sum())
    for i in range(N):
        c = int(bcode[i])
        if c == 0:
            continue
        ok = True
        if c & 1 and abs(x[i] - xmn) >= 1e-6: ok = False
        if c & 2 and abs(x[i] + w[i] - xmx) >= 1e-6: ok = False
        if c & 4 and abs(y[i] + h[i] - ymx) >= 1e-6: ok = False
        if c & 8 and abs(y[i] - ymn) >= 1e-6: ok = False
        if not ok:
            vb += 1

    # grouping
    vg = 0; n_grp = 0
    for g in range(1, (int(clust_id.max()) if clust_id.size else 0) + 1):
        mem = np.where(clust_id == g)[0].tolist()
        n_grp += max(0, len(mem) - 1)
        if len(mem) > 1:
            vg += len(_components(mem, x, y, w, h)) - 1

    # mib
    vm = 0; n_mib = 0
    for g in range(1, (int(mib_id.max()) if mib_id.size else 0) + 1):
        mem = np.where(mib_id == g)[0].tolist()
        n_mib += max(0, len(mem) - 1)
        shapes = {(round(float(w[i]), 4), round(float(h[i]), 4)) for i in mem}
        vm += len(shapes) - 1

    n_soft = max(1, n_boundary + n_grp + n_mib)
    return vb, vg, vm, n_soft


_TOUCH = 1e-3      # 共享邊界正長度才算 abut（官方閾值 1e-4，這裡的除裡）


def _bbox(x, y, w, h):
    return (float(x.min()), float((x + w).max()),
            float(y.min()), float((y + h).max()))


def _bbox_same(x, y, w, h, bb):
    """bbox 是否維持不變。

    移動過程已限制在 bbox 內，但如果被移動的方塊正好是「唯一定義那個延伸」的
    那顆，它退縮就會讓整個 bbox 跟著縮——等於所有原本貼在牆上的方塊瞬間
    全部變成 boundary 違規（實測就是這樣把 boundary 從 87 提高到 102）。
    """
    nb = _bbox(x, y, w, h)
    return all(abs(a - b) < 1e-9 for a, b in zip(nb, bb))


def _boundary_ok(i, bcode, x, y, w, h, bb):
    """方塊 i 是否滿足自己的 boundary 位碼（相對於目前的 bbox）。"""
    c = int(bcode[i])
    if c == 0:
        return True
    xmn, xmx, ymn, ymx = bb
    if c & 1 and abs(x[i] - xmn) >= 1e-6:
        return False
    if c & 2 and abs(x[i] + w[i] - xmx) >= 1e-6:
        return False
    if c & 4 and abs(y[i] + h[i] - ymx) >= 1e-6:
        return False
    if c & 8 and abs(y[i] - ymn) >= 1e-6:
        return False
    return True


def _group_v(g, clust_id, x, y, w, h):
    """單一 cluster 群組的 V_grouping（連通元件數 - 1）。"""
    mem = np.where(clust_id == g)[0].tolist()
    if len(mem) <= 1:
        return 0
    return len(_components(mem, x, y, w, h)) - 1


def _y_slots(i, nx, s, x, y, w, h, floor, bb):
    """i 的左上角 x 座標固定在 nx，貼著 s 的左側或右側（擇一）依序試合法的 y 座標。

    要滿足 abut，i 跟 s 的共享邊必須有正長度，也就是
        y[s] - h[i] < ny < y[s] + h[s]
    候選點包括：貼齊 s 的上緣、貼齊 s 的下緣、維持 i 原本的 y，以及沿著同一豎直
    帶裡其他方塊的上緣/下緣——合法佈局中，好位置幾乎都長這樣，只試候選點即能
    找到空隙。
    """
    lo, hi = y[s] - h[i], y[s] + h[s]
    R = nx + w[i]
    cands = [y[s], y[s] + h[s] - h[i], y[i]]
    for j in range(len(x)):
        if x[j] < R - _EPS and nx < x[j] + w[j] - _EPS:   # 跟 i 分同一條豎直帶
            cands.append(y[j] + h[j])
            cands.append(y[j] - h[i])
    ymn, ymx = bb[2], bb[3]
    out, seen = [], set()
    for c in cands:
        if not (lo + _TOUCH < c < hi - _TOUCH):
            continue
        if c < ymn - _EPS or c + h[i] > ymx + _EPS:      # 不得超過 bbox 最大值
            continue
        if floor is not None and c < floor - _EPS:
            continue
        k = round(c, 6)
        if k not in seen:
            seen.add(k)
            out.append(c)
    return out


def _x_slots(i, ny, s, x, y, w, h, floor, bb):
    """`_y_slots` 的水平版本，i 貼在 s 的上下，換成可用的 x 座標。"""
    lo, hi = x[s] - w[i], x[s] + w[s]
    T = ny + h[i]
    cands = [x[s], x[s] + w[s] - w[i], x[i]]
    for j in range(len(x)):
        if y[j] < T - _EPS and ny < y[j] + h[j] - _EPS:   # 跟 i 分同一條水平帶
            cands.append(x[j] + w[j])
            cands.append(x[j] - w[i])
    xmn, xmx = bb[0], bb[1]
    out, seen = [], set()
    for c in cands:
        if not (lo + _TOUCH < c < hi - _TOUCH):
            continue
        if c < xmn - _EPS or c + w[i] > xmx + _EPS:
            continue
        if floor is not None and c < floor - _EPS:
            continue
        k = round(c, 6)
        if k not in seen:
            seen.add(k)
            out.append(c)
    return out


def _swap_ok(i, j, x, y, w, h, floor, bb):
    """i 跟 j 交換位置之後仍然合法，且兩者都留在目前 bbox 內。"""
    xi, yi, xj, yj = x[i], y[i], x[j], y[j]
    if floor is not None and min(xi, yi, xj, yj) < floor - _EPS:
        return False
    xmn, xmx, ymn, ymx = bb
    if (xj < xmn - _EPS or xj + w[i] > xmx + _EPS or
            yj < ymn - _EPS or yj + h[i] > ymx + _EPS or
            xi < xmn - _EPS or xi + w[j] > xmx + _EPS or
            yi < ymn - _EPS or yi + h[j] > ymx + _EPS):
        return False
    # 交換之後 i 跟 j 彼此之間不能重疊（_free 拿彼此 ignore，另外檢查這一項）
    if (xj < xi + w[j] - _EPS and xi < xj + w[i] - _EPS and
            yj < yi + h[j] - _EPS and yi < yj + h[i] - _EPS):
        return False
    return (_free(i, xj, yj, x, y, w, h, ignore={j}) and
            _free(j, xi, yi, x, y, w, h, ignore={i}))


def grouping_repair(x, y, w, h, clust_id, is_pre, passes=2, floor=None,
                    bcode=None, max_slot_tests=64, max_swaps=24):
    """把落單的 cluster 成員接回群組主體，降低 V_grouping。

    在 legalize 硬可行性之後的佈局幾乎不留任何空隙，每一輪只試 4 個貼齊點，
    座標完全對齊。落貼合點，而不要求目標位置完全空——硬可行性後的佈局沒有
    空間，這些點幾乎必然已被佔用，於是直接放棄。實測大案例（n>=100）的失分
    82.5%，其中直接需要 61% 來自 grouping，就是這個原因。

    三點強化：

    1. **沿邊界滑動選點**：貼齊點被佔用時，改沿共享邊往一方向選位置
       （見 :func:`_y_slots` / :func:`_x_slots`），窗中的高交換降級。
    2. **交換移動**：那條沿邊都找不到空位置時，改試把 i 跟「已經貼著群組主體」的
       某個方塊 j 交換——硬可行性佈局沒有空隙，但不一定可以彼此的鄰居。
    3. **移動不得超出目前 bbox**：所以 area_gap 不會因為修 grouping 而
       惡化，boundary 也只需檢查被移動的方塊自己。

    交換移動只考慮「有影響的群組 + 對側方塊的 boundary」（依估計需要移動，嚴格
    要好才接受，需要旁記 ``bcode`` 才啟用，省略時只做前兩項）。
    ``max_slot_tests`` / ``max_swaps`` 是每次呼叫的最小預算，用來限制 runtime。
    """
    x = np.asarray(x, float); y = np.asarray(y, float)
    w = np.asarray(w, float); h = np.asarray(h, float)
    clust_id = np.asarray(clust_id).astype(int); is_pre = np.asarray(is_pre, bool)
    bc = None if bcode is None else np.asarray(bcode).astype(int)
    G = int(clust_id.max()) if clust_id.size else 0
    N = len(x)
    fl = -np.inf if floor is None else floor

    for _ in range(passes):
        any_move = False
        swap_budget = max_swaps
        for g in range(1, G + 1):
            members = np.where(clust_id == g)[0].tolist()
            if len(members) <= 1:
                continue
            comps = _components(members, x, y, w, h)
            if len(comps) <= 1:
                continue
            comps.sort(key=len, reverse=True)
            main = set(comps[0])           # 最大的連通元件，其餘要接回來
            member_set = set(members)
            bb = _bbox(x, y, w, h)
            frontier = None                # 延遲計算，每次群組只算一次

            for comp in comps[1:]:
                for i in comp:
                    if is_pre[i]:
                        continue

                    # -- 第 1 步：原本的 4 個貼齊點（便宜，先試）----------
                    # 最小位移：在所有可用空位裡找離 i 目前位置最近的那個
                    # （Manhattan 度量，粗略反映 HPWL 的影響應該最小）。
                    best, best_d = None, None
                    for s in main:
                        for nx, ny in ((x[s] - w[i], y[s]), (x[s] + w[s], y[s]),
                                       (x[s], y[s] - h[i]), (x[s], y[s] + h[s])):
                            if nx < fl or ny < fl:
                                continue
                            if _free(i, nx, ny, x, y, w, h):
                                d = abs(nx - x[i]) + abs(ny - y[i])
                                if best_d is None or d < best_d:
                                    best, best_d = (nx, ny), d

                    # -- 第 2 步：沿共享邊滑動（只在原本 4 點全被占用時才用）--
                    if best is None:
                        near = sorted(main, key=lambda s: abs(x[s] - x[i]) +
                                      abs(y[s] - y[i]))[:6]
                        cands = []
                        for s in near:
                            for nx in (x[s] - w[i], x[s] + w[s]):
                                if nx < fl:
                                    continue
                                for ny in _y_slots(i, nx, s, x, y, w, h, floor, bb):
                                    cands.append((abs(nx - x[i]) + abs(ny - y[i]),
                                                  nx, ny))
                            for ny in (y[s] - h[i], y[s] + h[s]):
                                if ny < fl:
                                    continue
                                for nx in _x_slots(i, ny, s, x, y, w, h, floor, bb):
                                    cands.append((abs(nx - x[i]) + abs(ny - y[i]),
                                                  nx, ny))
                        cands.sort()
                        for d, nx, ny in cands[:max_slot_tests]:
                            if _free(i, nx, ny, x, y, w, h):
                                best, best_d = (nx, ny), d
                                break          # 已依位移排序，第一個就是最近的

                    # 套用前面的合法移動（bbox 不能變，其他方塊的重疊都要跟著
                    # 移動）；並且 i 自己原本合格的 boundary 不能被弄壞。
                    if best is not None:
                        was_ok = (bc is None or bc[i] == 0 or
                                  _boundary_ok(i, bc, x, y, w, h, bb))
                        ox, oy = x[i], y[i]
                        x[i], y[i] = best
                        ok = _bbox_same(x, y, w, h, bb)
                        if ok and was_ok and bc is not None and bc[i] != 0:
                            ok = _boundary_ok(i, bc, x, y, w, h, bb)
                        if ok:
                            main.add(i)
                            any_move = True
                            continue
                        x[i], y[i] = ox, oy          # 撤回

                    # -- 第 3 步：交換移動（沒有空隙時唯一能走得通的路）--------
                    if bc is None or swap_budget <= 0:
                        continue
                    if frontier is None:     # 每個群組只算一次，O(N×|main|) 次
                        frontier = [        # _touch：放在 i 的迴圈裡沒必要每次算
                            j for j in range(N)
                            if j not in member_set and not is_pre[j]
                            and any(_touch(j, s, x, y, w, h) for s in main)]
                    frontier.sort(key=lambda j: abs(x[j] - x[i]) + abs(y[j] - y[i]))
                    gi = int(clust_id[i])
                    for j in frontier[:3]:
                        if swap_budget <= 0:
                            break
                        if not _swap_ok(i, j, x, y, w, h, floor, bb):
                            continue
                        swap_budget -= 1
                        gj = int(clust_id[j])
                        before = (_group_v(gi, clust_id, x, y, w, h)
                                  + (_group_v(gj, clust_id, x, y, w, h) if gj and gj != gi else 0)
                                  + (0 if _boundary_ok(i, bc, x, y, w, h, bb) else 1)
                                  + (0 if _boundary_ok(j, bc, x, y, w, h, bb) else 1))
                        (x[i], y[i]), (x[j], y[j]) = (x[j], y[j]), (x[i], y[i])
                        if not _bbox_same(x, y, w, h, bb):
                            (x[i], y[i]), (x[j], y[j]) = (x[j], y[j]), (x[i], y[i])
                            continue
                        after = (_group_v(gi, clust_id, x, y, w, h)
                                 + (_group_v(gj, clust_id, x, y, w, h) if gj and gj != gi else 0)
                                 + (0 if _boundary_ok(i, bc, x, y, w, h, bb) else 1)
                                 + (0 if _boundary_ok(j, bc, x, y, w, h, bb) else 1))
                        if after < before:
                            main.add(i)
                            any_move = True
                            break
                        (x[i], y[i]), (x[j], y[j]) = (x[j], y[j]), (x[i], y[i])  # 撤回
        if not any_move:
            break
    return x, y
