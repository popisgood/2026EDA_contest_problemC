"""面積比例迴圈切割式打包器（切割式 / guillotine dissection）。

為什麼需要這個
----------------
density_probe.py 量到的 ground truth 的包夾平均填充比是 **0.966**，而解析式佈局
只有 0.595（加上 compact_layout 後 0.677）。`area_gap` 幾乎全部來自留白，
跟多軸長寬比無關。

而這個缺口在結構上是可以補起來的：官方對軟方塊**只查面積 1% 容差，完全沒有
任何長寬比限制**（iccad2026_evaluate.py 只有 check_area_tolerance，整份文檔
grep `aspect` 零命中），而 86% 的面積屬於這種自由軟方塊（census.py：fixed 9.7%
+ preplaced 4.0% 是釘死的）。

一組面積固定、長寬比自由的方塊，用一個矩形依子樹面積比例迴圈切割，填充比
**恰好 100%**（Otten 1982 切割式佈局 / Stockmeyer 1983 形狀曲線）。GT 的 0.966
跟 1.000，缺口那 3.4% 遠好於還會被釘死的那 13.7%。

跟 WINNING_STRATEGY.md 記錄的「B*-tree/contour 密度天花板 3.3185」不衝突：
B*-tree 是「左下角」的輪廓表示法，白空間是結構性的，那個天花板對此完全成立。
切割式的天花板完全不同。

設計上的關鍵原則
----------------
這**不是**從零建構的搜尋器——它反映解析佈局結果的順序，「拓樸繼承」，每一對
左右相鄰的方塊由原始解析佈局裡的座標決定。長寬比是繼承來的而不是重新發明，本質上
是把「解析式佈局的排列」跟「面積比例切割式佈局的形狀」下游完全不同。

硬約束的處理分兩層——**只有**結構性的，實際都仍交給下游的修復：
  * grouping：以 cluster 用群心當排序主軸 -> 成員在區域中連續 -> 落入同一子樹
    -> 天然彼此相鄰。
  * boundary：只有當**方塊本身貼著切割邊**才能推出正確方向，用 `touch`
    位碼追蹤；如果原位置是錯的——子孫方向一旦颠倒多次，「靠左」就不等於貼齊。
    位置只負責讓方塊留在正確的子孫裡，真正貼齊由下游的 boundary_snap 完成
    （理由見 _place_leaf 的註解，多帶隨機性，這裡先留著再修較差）。
  * MIB：切割完成後由 _unify_mib 把同群成員統一成同一形狀（位置不變）。
    先試過「改成硬要求成員能夠落入同一子樹再切」，結果明顯更差，理由見下方
    對應結果的註解。

安全性：任何一步無法保證合法輸出，方塊卡不進去確實會失敗。直線無法同時滿足面積跟
preplaced 的位置，就整套回傳 None，呼叫端沿用原本的梯度下降路徑。結果不產出
不合法的解。
"""
from __future__ import annotations

import os
import sys

import numpy as np

_EPS = 1e-9
_BIG = 1 << 20

# ELECTRO_SLICE_DEBUG=1 打開，「為什麼切割失敗」寫到 stderr。切割是尚未見過的
# ——失敗就是回傳 None，呼叫端退回梯度下降路徑——沒有踩到就只能看到 nslice=0/1，
# 完全無得知是哪個方塊卡住。這是 preplaced 或方向夾死。
_DEBUG = os.environ.get("ELECTRO_SLICE_DEBUG", "0") == "1"
_FAIL: list = []

# walls 模式的回溯預算，見 slice_pack 裡的說明。這是實質耗費的直接反映，
# 所以刻意可調。
_WBUDGET = int(os.environ.get("ELECTRO_SLICE_BUDGET", "4000"))

# util 二分搜尋的步數。每一步就是一次完整的切割，所以這是實打實的牆鐘時間。
# 在 100 案實測（seeds=8）：
#     steps=6 -> 1.4972 @ 4.31s      steps=3 -> 1.5097 @ 3.53s
#     steps=2 -> 1.5048 @ 3.15s      （折損 2 步多長寬比）
# 預設 2，省時 27%，分數只差 0.5%。二分搜尋收斂得很快，多出來的步數只是在
# 小數點後第三位微調長寬比，不值得多花時間。
_STEPS = int(os.environ.get("ELECTRO_SLICE_STEPS", "2"))

# util 二分搜尋的上下界。原本寫死在 slice_pack() 的簽名裡（0.70 / 0.995），
# 暴露成環境變數以便掃描：util_hi 太低會白白浪費密度，太高則第一次 attempt
# 幾乎必定失敗（多花一次完整切割的時間）；util_lo 必須是「幾乎一定可行」的
# 寬鬆值，太高會讓整個 slice_pack 直接放棄、退回梯度式路徑。
_UTIL_LO = float(os.environ.get("ELECTRO_SLICE_UTIL_LO", "0.70"))
_UTIL_HI = float(os.environ.get("ELECTRO_SLICE_UTIL_HI", "0.995"))


def _note(kind, **kw):
    if _DEBUG and len(_FAIL) < 6:
        _FAIL.append(f"{kind} " + " ".join(f"{k}={v}" for k, v in kw.items()))

# bcode 位碼：1=左, 2=右, 4=上, 8=下，跟 soft_repair 一致
_L, _R, _T, _B = 1, 2, 4, 8


class _Ctx:
    """遞迴期間共用的唯讀資料。"""

    __slots__ = ("cx", "cy", "x", "y", "w", "h", "areas",
                 "is_fixed", "is_pre", "bcode", "clust", "kx", "ky",
                 "phx_hi", "phx_lo", "phy_hi", "phy_lo", "rig_w", "rig_h",
                 "reg_w", "reg_h", "reg_x", "reg_y", "budget", "walls")

    def __init__(self, x, y, w, h, areas, is_fixed, is_pre, clust, bcode):
        self.x, self.y, self.w, self.h = x, y, w, h
        self.areas = areas
        self.is_fixed, self.is_pre, self.bcode = is_fixed, is_pre, bcode
        self.clust = clust
        self.cx = x + 0.5 * w
        self.cy = y + 0.5 * h
        # 排序主軸：同 cluster 的成員共用群心（區域中盡量連續 -> 落入同一子樹
        # -> 天然彼此相鄰）。無 cluster（id<=0）的方塊用自己的座標。
        self.kx = self.cx.copy()
        self.ky = self.cy.copy()
        for c in np.unique(clust):
            if c <= 0:
                continue
            m = clust == c
            if m.sum() > 1:
                self.kx[m] = self.cx[m].mean()
                self.ky[m] = self.cy[m].mean()


def _outline(ctx, util, asp_scale=1.0):
    """外框：面積 = 總面積/util，長寬比沿用原始佈局，不必完整包住 preplaced。

    除以 util 的膨脹是必要的：fixed 跟 preplaced 無法變形，它們自己的填充不一定
    最理想，沒有膨脹的話反而有可能把周圍的填充空間全部擠壓自己的方塊。

    `asp_scale` 讓外層試不同的外框比例。原始佈局的比例只是起點——實測最好
    跟 ground truth 的外框比例差距 |log(ours/GT)| = 0.211，而切割本身相對於下游
    的解析式佈局幾乎免費，多試幾個比例讓外層的搜尋有更多機會贏過原本目標。
    """
    A = float(ctx.areas.sum()) / max(util, 1e-6)
    sw = float((ctx.x + ctx.w).max() - ctx.x.min())
    sh = float((ctx.y + ctx.h).max() - ctx.y.min())
    asp = (sw / sh) if sh > _EPS else 1.0
    asp = min(max(asp * asp_scale, 0.25), 4.0)
    H = (A / asp) ** 0.5
    W = asp * H
    if ctx.is_pre.any():
        W = max(W, float((ctx.x[ctx.is_pre] + ctx.w[ctx.is_pre]).max()))
        H = max(H, float((ctx.y[ctx.is_pre] + ctx.h[ctx.is_pre]).max()))
    return 0.0, 0.0, W, H


def _order(ctx, idx, vert, touch):
    """沿切割軸排序，並將「這一輪希望貼在這條切割邊」的推到最前/最後的位置。"""
    key = ctx.kx if vert else ctx.ky
    own = ctx.cx if vert else ctx.cy
    lo_bit, hi_bit = (_L, _R) if vert else (_B, _T)

    def sk(i):
        b = ctx.bcode[i]
        bias = 0
        if (b & lo_bit) and (touch & lo_bit):
            bias = -_BIG
        elif (b & hi_bit) and (touch & hi_bit):
            bias = _BIG
        return (bias, key[i], own[i], i)

    return sorted(idx, key=sk)


def _wall_groups(ctx, idx, vert, touch):
    """算出這一軸上，被排在最左或最右端方塊的數量。

    _order 已經把需要貼低端（左/下）的方塊排到最前，需要貼高端（右/上）的
    排到最後。回傳兩邊群組的大小，讓 _choose_cut 可以優先把分割線切在群組邊界
    上，整群才能維持一條完整的貼齊帶。位碼用的是 elif，所以每個方塊只貢獻自己
    的低端。
    """
    lo_bit, hi_bit = (_L, _R) if vert else (_B, _T)
    nlo = nhi = 0
    for i in idx:
        b = ctx.bcode[i]
        if (b & lo_bit) and (touch & lo_bit):
            nlo += 1
        elif (b & hi_bit) and (touch & hi_bit):
            nhi += 1
    return nlo, nhi


def _locked(ctx, idx, touch):
    """判斷「這一整批方塊都要求同一個位碼」的鎖定情況。

    這種情況一旦被切到不同的子集合，被切到內側的方塊就永遠貼不到了。外層以
    此調整切割方向的嘗試順序——這正是 boundary 違規的主要來源，遞迴每分一次
    都可能造成同一個方向丟給不同方塊。
    """
    lock = 0
    for bit in (_L, _R, _T, _B):
        if (touch & bit) and all(ctx.bcode[i] & bit for i in idx):
            lock |= bit
    return lock


def _cut_options(ctx, seq, span, cross, base, p_hi, p_lo, rig_a, rig_c,
                 kpref=(), limit=3):
    """在排好序的 seq 中找候選分割點 k 跟分割線位置 t（相對於 base）。

    合法要滿足的三個條件：
      1. 面積：t*cross >= A(seq[:k]) 且 (span-t)*cross >= A(seq[k:])，
         否則某一側的填充會裝不下自己的方塊；
      2. 剛性尺寸：fixed 跟 preplaced 的形狀釘死，**面積夠不代表塞得進去**——
         實測就是被這樣卡住的（例如 9.56x16.16，面積 154.6）裝不下一個
         10.53x10.53（面積 111.0）的固定方塊。所以分割線兩側的跨度必須各自
         大於等於該側剛性方塊沿切割軸的最大尺寸（而 cross 方向要一次滿足
         整個 seq，兩側共用同一個 cross）；
      3. preplaced：左側的 preplaced 必須整個在 t 左邊，右側的整個在右邊；
      4. 盡量貼近面積比例點——排出來的子樹形狀盡可能接近正方形，這也是不失
         失敗性的一個提示。

    回傳最多 `limit` 組 (k, t)，依偏好排序，找不到任何可行方案就回傳空 list
    （呼叫端會試另一個切割方向，或沿用解析式路徑）。

    回傳「多組」而不是第一組，是因為在這一層可行的分割點不代表子樹之後走得下
    去。kpref 一路帶進來立刻公告，案例 89 的整份多軸比例全部切割失敗，就是因為
    邊界帶的分割點在前層可行、子樹卻死掉，然而卻沒有進來面積平衡分割點的路。
    """
    m = len(seq)
    # cross 方向兩側共用，同一次解決；這個方向根本鬆一下就浪費費得多的成本
    for i in seq:
        if rig_c[i] > cross + 1e-9:
            _note("CUT/cross", m=m, span=f"{span:.1f}", cross=f"{cross:.1f}",
                  need=f"{rig_c[i]:.1f}", blk=i)
            return []
    a = np.array([ctx.areas[i] for i in seq], dtype=float)
    cum = np.concatenate(([0.0], np.cumsum(a)))
    tot = cum[-1]

    # 面積提早否決（可證明等價，不改變任何輸出）：
    # 下面每個候選 k 的可行條件是 t_lo <= t_hi，其中
    #     t_lo >= a1/cross,  t_hi <= span - a2/cross
    # 所以可行必然蘊含 a1/cross <= span - a2/cross，即 **tot <= span*cross**。
    # 這個條件跟 k 無關，因此區域面積裝不下全部方塊時，**所有** k 都會失敗。
    # 既有程式碼是逐個 k 試完才發現全滅；這裡提前判定，跳過後面所有 O(m) 的
    # 前綴/後綴陣列建構與整個候選迴圈。
    # 實機 profile：`_cut_options` 被呼叫 23,888 次、其中 44-46% 回傳空結果，
    # 這是其中一類可以零成本濾掉的。
    if tot > span * cross + _EPS:
        _note("CUT/area", m=m, span=f"{span:.1f}", cross=f"{cross:.1f}",
              reg_area=f"{span * cross:.1f}", blk_area=f"{tot:.1f}")
        return []

    rig_pre = np.zeros(m + 1)           # seq[:k] 中剛性方塊沿軸的最大尺寸
    for j, i in enumerate(seq):
        rig_pre[j + 1] = max(rig_pre[j], rig_a[i])
    rig_suf = np.zeros(m + 1)           # seq[k:] 同上
    for j in range(m - 1, -1, -1):
        rig_suf[j] = max(rig_suf[j + 1], rig_a[seq[j]])

    # preplaced 沿切割軸的邊緣位置，用 ±inf 當初始值，不產生任何直線
    pre_hi = np.full(m + 1, -np.inf)    # seq[:k] 中 preplaced 的最大（高端）
    for j, i in enumerate(seq):
        pre_hi[j + 1] = max(pre_hi[j], p_hi[i])
    pre_lo = np.full(m + 1, np.inf)     # seq[k:] 中 preplaced 的最小（低端）
    for j in range(m - 1, -1, -1):
        pre_lo[j] = min(pre_lo[j + 1], p_lo[seq[j]])

    kb = int(np.searchsorted(cum, 0.5 * tot))
    kb = min(max(kb, 1), m - 1)
    # kpref 先試：優先切在邊界帶的分割點上，整群才能維持一條完整的貼齊帶，
    # 帶頭的方塊在都貼著的。剩下才依面積平衡為中心點的偏好。
    order = list(kpref) + [k for d in range(m)
                           for k in ((kb - d, kb + d) if d else (kb,))]
    opts, seen = [], set()
    for k in order:
        if k < 1 or k > m - 1 or k in seen:
            continue
        seen.add(k)
        a1, a2 = cum[k], tot - cum[k]
        t_lo = max(a1 / cross, pre_hi[k] - base, rig_pre[k], 0.0)
        t_hi = min(span - a2 / cross, pre_lo[k] - base,
                   span - rig_suf[k], span)
        if t_lo <= t_hi + _EPS:
            t_prop = span * (a1 / tot) if tot > _EPS else 0.5 * span
            opts.append((k, min(max(t_prop, t_lo), t_hi)))
            if len(opts) >= limit:
                break
    if not opts:
        _note("CUT/none", m=m, span=f"{span:.1f}", cross=f"{cross:.1f}",
              reg_area=f"{span * cross:.1f}", blk_area=f"{tot:.1f}",
              npre=int(sum(1 for i in seq if ctx.is_pre[i])),
              nrig=int(sum(1 for i in seq if rig_a[i] > 0)))
    return opts


def _place_leaf(i, reg, ctx, out):
    x0, y0, W, H = reg
    if ctx.is_pre[i]:
        # 座標跟形狀都釘死。前面的條件已經保證進來的一定完整合法。
        out[i] = (ctx.x[i], ctx.y[i], ctx.w[i], ctx.h[i])
        return True
    if ctx.is_fixed[i]:
        w, h = float(ctx.w[i]), float(ctx.h[i])
        if w > W + 1e-9 or h > H + 1e-9:
            _note("LEAF/fixed", blk=i, reg=f"{W:.2f}x{H:.2f}", wh=f"{w:.2f}x{h:.2f}")
            return False
    else:
        a = float(ctx.areas[i])
        if a > W * H + 1e-9:
            _note("LEAF/soft", blk=i, reg_area=f"{W * H:.2f}", blk_area=f"{a:.2f}")
            return False
        # 沿用區域的長寬比，面積精確等於目標。因為 a <= W*H，
        # w = sqrt(a*W/H) <= W 且 h = a/w <= H，必然放得下。
        # 這樣做因為每一層都會分開，遞迴子樹結果接近正方形，跟拓樸長寬比不失敗。
        w = (a * W / H) ** 0.5
        h = a / w
    ctx.reg_w[i], ctx.reg_h[i] = W, H   # 供 _unify_mib 判斷共用形狀時要用得上
    ctx.reg_x[i], ctx.reg_y[i] = x0, y0  # 供 hpwl_polish 算區域內可動範圍
    # 不強制貼左下角，留下剩餘空間留在右上。
    #
    # 試過並且實驗**更差**的替代方案：依 bcode 把需要貼右（右上方塊都要貼在
    # 區域的右上）——直覺上該減少 boundary 違規，實測反而讓案例 89 從 2.1078
    # 提高到 2.8629，填充比 0.830 降到 0.720。原因是多軸帶著膨脹，util 越貼死
    # 就有 18% 除裡，貼齊到**多軸**的邊界比實際 bbox 最大化整個多軸。
    # 膨脹在裡都浪費時，跟連鎖也跟移動位置，其他需要貼同一條邊的方塊反而
    # 全部失敗。真正的 bbox 要等到全部固定完才知道，這裡無得知到底。
    # 也試過「貼到**實際 bbox** 的邊」（跟隨多軸），那個版本在幾何上不是正確
    # 的——bbox <= 多軸，方塊反外移動也在自己原本原始，bbox 也不會被最大——實測
    # 確實有移動方塊（20 顆違規的方塊中約 7 顆），但不管案例的分數跟 V_rel **完全
    # 沒變**。因為 soft_repair.boundary_snap 後面本來就在做這件事，而不是真的得到
    # 正確的 bbox。純粹是重複實作，已經移除。
    #
    # 真正的殘留原因不是「沒貼齊」，而是有些方塊**根本不在正確的邊**，遞迴每分一
    # 次都可能不一樣，帶左那一批就分到不同方塊，被排到外側的就無法貼到左的。要修好需要
    # 「重新排出一條完整的貼齊帶」帶著的整個平行於該切割的解法。
    #
    # 貼齊交給後面的 boundary_snap 處理——_order 的貼齊排列已經把該貼齊的方塊
    # 儘量排進正確的子樹裡，它要走的距離會比較短。
    out[i] = (x0, y0, w, h)
    return True


# 試過但實測明顯更差的 MIB 特例——留在這裡，避免有人再走一次：
#
#   讓 MIB 群組的成員共用排序群心（區域中連續 -> 落入同一子樹），並要求該子樹的
#   形狀沿同一方向切成 k 等份。理論上很漂亮：W x H 等分成 k 份就是 k 個
#   W/k x H，形狀完全相同，任意 k 都成立，MIB 需要結構性歸零。這個沒有一路修復。
#
#   實測（seeds=1，選案例）：
#       case 89  2.1078 -> 2.8516
#       case 96  2.4396 -> 2.7963
#       case 99  1.5181 -> 1.5181
#   兩案明顯變差、一案打平。原因是強迫這批成員一起用群心當排序主軸，等於要寫成
#   解析式佈局給的長寬比最佳排——但這個路線的貢獻本質是繼承這種原始的分割，
#   等份強硬讓大群的每一塊都變得細長。MIB 拿位僅需要的 8.5%~16%，卻犧牲遠遠
#   遠遠大於這麼小的需要面。
#
#   結論：MIB 交給後面的 soft_repair 跟 analytical_place 的共用 log-aspect 處理。


def _unify_mib(ctx, mib, out):
    """事後把同一 MIB 群組的成員統一成同一形狀（位置完全不動）。

    MIB 要求同群成員的 (w,h) 完全相同，但切割式裡每顆軟方塊沿用**自己所在區域**
    的長寬比，成員落在不同區域就形狀不同——實測 MIB 違規因此從 56 溶解到 171
    （但整體違規只佔 21.5%）。

    這跟先前失敗的做法本質差別，那次是**排序**（用群心當主軸好讓成員落入
    同一子樹再等分），等於要寫成解析式佈局給的長寬比最佳排，犧牲遠遠大於
    這麼小的需要面。這裡只改形狀，不改位置也不改排序。

    不可能製造重疊：每份切割互不交疊，方塊仍在自己原區的左上角，只要新形狀仍然
    塞得下**自己原有邊**就不會碰到別人。成員 i 能接受的共用寬度範圍因此是
    [a/H_i, W_i]，w* <= W_i 且 a/w* <= H_i。

    找**最大可統一子集**而不是要求全帶，官方的 V_mib 目標是「每群相同形狀數
    - 1」，所以一個 5 成員的群若 5 個形狀在一堆就是 4 個違規，統一其中 4 個
    只剩 1 個違規——部分統一同樣有貢獻。要求全帶可行是有跟無，而邊界帶會讓
    切割每份都變得細長。這種條件常常不成立，實測 MIB 違規因此溶解到 194）。
    「被最大公共重疊」是掃描線的計算幾何小問題，掃描所有端點掃描排序即可，
    對 n<=120 完全不是負擔。
    """
    for g in np.unique(mib):
        if g <= 0:
            continue
        mem = [i for i in range(len(mib)) if mib[i] == g
               and not (ctx.is_fixed[i] or ctx.is_pre[i])]   # 剛性方塊不能變形
        if len(mem) < 2 and not (
                os.environ.get("ELECTRO_MIB_ANCHOR", "0") == "1" and len(mem) >= 1):
            continue
        a = float(ctx.areas[mem[0]])

        # --- 剛性錨點優先（ELECTRO_MIB_ANCHOR=1）--------------------------
        # 原本的寫法把 fixed/preplaced 成員整個排除在群組外，只讓軟成員彼此
        # 統一。但 V_mib 數的是「整個 MIB 群組的相異形狀數 - 1」，剛性成員也
        # 算在內——軟成員統一到某個區域推導出來的寬度，幾乎不可能剛好等於剛性
        # 錨點的形狀，所以混合群組**永遠至少殘留 1 個違規**（實測 56 個混合
        # 群組 = 至少 56 個違規的地板）。
        #
        # 實測驗證過：100 個 MIB 群組全部等面積，且 56 個混合群組的剛性錨點
        # 其 w*h 都精確等於群組面積——所以軟成員直接採用錨點的 (w,h) 面積仍然
        # 精確、硬約束安全，V_mib 才可能真正歸零。
        if os.environ.get("ELECTRO_MIB_ANCHOR", "0") == "1":
            rig = [i for i in range(len(mib)) if mib[i] == g
                   and (ctx.is_fixed[i] or ctx.is_pre[i])]
            if rig:
                aw, ah = float(ctx.w[rig[0]]), float(ctx.h[rig[0]])
                # 錨點面積必須就是群組面積，否則採用它會破壞 1% 面積硬約束
                if abs(aw * ah - a) <= 1e-6 * max(1.0, a):
                    for i in mem:
                        # 只在該成員自己的區域裝得下錨點形狀時才改，
                        # 否則會跟隔壁區域的方塊重疊
                        if aw <= ctx.reg_w[i] + 1e-12 and ah <= ctx.reg_h[i] + 1e-12:
                            x0, y0, _w, _h = out[i]
                            out[i] = (x0, y0, aw, ah)
                    continue      # 這個群組已用錨點形狀處理完，不再走下面的邏輯
        if len(mem) < 2:
            continue
        if any(abs(ctx.areas[i] - a) > 1e-9 * max(1.0, a) for i in mem):
            continue          # 面積不同就不可能變形
        # 掃描線找最大重疊點。+1 事件放在 -1 之前（同座標視為重疊），
        # 這樣剛好貼齊兩端的成員也算重疊進去。
        ev = []
        for i in mem:
            lo, hi = a / ctx.reg_h[i], ctx.reg_w[i]
            if lo <= hi + 1e-12:
                ev.append((lo, 0))      # 0 = 進入
                ev.append((hi, 1))      # 1 = 離開
        if not ev:
            continue
        ev.sort()
        cur = best = 0
        ws = None
        for v, kind in ev:
            if kind == 0:
                cur += 1
                if cur > best:
                    best, ws = cur, v
            else:
                cur -= 1
        if ws is None or best < 2:
            continue          # 那些成員互不重疊就沒有意義
        hs = a / ws
        for i in mem:
            if a / ctx.reg_h[i] <= ws + 1e-12 <= ctx.reg_w[i] + 1e-12:
                x0, y0, _w, _h = out[i]
                out[i] = (x0, y0, ws, hs)


def hpwl_polish(out, reg_w, reg_h, reg_x, reg_y, eb, ep, pv, bcode, is_pre,
                rounds=3):
    """在**各自的區域內**滑動方塊以降低 HPWL（零風險的免費改善）。

    為什麼這是免費的
    ----------------
    `_place_leaf` 把每顆軟方塊固定在自己區域的左下角，但方塊其實比區域小：
    區域面積 ≈ a/util（util 二分搜尋通常收在 0.90-0.93），所以每軸大約有
    `sqrt(1/util) - 1` ≈ 5% 的餘裕。**這個位置自由度目前完全沒被使用。**

    因為每顆方塊只在自己的區域內移動，而切割產生的區域彼此不重疊，所以
    **不可能製造出新的重疊**——不需要任何 `_free` 檢查，也不需要事後修復。
    bbox 只會縮小或不變（方塊不會超出區域、區域不會超出外框），所以面積也
    不會變差。純粹是撿走原本浪費掉的自由度。

    做法：逐軸座標下降。固定其他方塊時，單一方塊的加權 L1 線長對自己座標是
    分段線性凸函數，最佳解是相連鄰居座標的**加權中位數**，再夾回自己的區域
    範圍。跑幾輪讓它收斂。

    保守處理：有 boundary 位碼的方塊、以及 preplaced 方塊完全不動——前者一動
    就可能離開它該貼的那面牆，後者座標本來就釘死。
    """
    n = len(out)
    if n == 0:
        return out
    xs = np.array([o[0] for o in out], float)
    ys = np.array([o[1] for o in out], float)
    ws = np.array([o[2] for o in out], float)
    hs = np.array([o[3] for o in out], float)
    cx = xs + 0.5 * ws
    cy = ys + 0.5 * hs
    # 可動範圍（中心點座標）
    lox = reg_x + 0.5 * ws
    hix = reg_x + reg_w - 0.5 * ws
    loy = reg_y + 0.5 * hs
    hiy = reg_y + reg_h - 0.5 * hs
    movable = ~(np.asarray(is_pre, bool)) & (np.asarray(bcode).astype(int) == 0)
    if not movable.any():
        return out

    # 建鄰接表：每顆方塊連到 (鄰居索引或固定端點, 權重)
    nb = [[] for _ in range(n)]      # (kind, ref, weight); kind 0=block 1=pin
    if eb is not None:
        for r in eb:
            i, j, wgt = int(r[0]), int(r[1]), float(r[2])
            if 0 <= i < n and 0 <= j < n and i != j:
                nb[i].append((0, j, wgt))
                nb[j].append((0, i, wgt))
    if ep is not None and pv is not None and len(pv):
        for r in ep:
            pi, bi, wgt = int(r[0]), int(r[1]), float(r[2])
            if 0 <= bi < n and 0 <= pi < len(pv):
                nb[bi].append((1, pi, wgt))

    def wmedian(vals, wts):
        """加權中位數：加權 L1 距離和的最小值點。"""
        order = np.argsort(vals)
        v = np.asarray(vals)[order]
        wq = np.asarray(wts)[order]
        c = np.cumsum(wq)
        half = 0.5 * c[-1]
        k = int(np.searchsorted(c, half))
        return float(v[min(k, len(v) - 1)])

    for _ in range(int(rounds)):
        moved = 0.0
        for i in range(n):
            if not movable[i] or not nb[i]:
                continue
            vx, vy, wt = [], [], []
            for kind, ref, wgt in nb[i]:
                if kind == 0:
                    vx.append(cx[ref]); vy.append(cy[ref])
                else:
                    vx.append(float(pv[ref][0])); vy.append(float(pv[ref][1]))
                wt.append(wgt)
            nx = min(max(wmedian(vx, wt), lox[i]), hix[i])
            ny = min(max(wmedian(vy, wt), loy[i]), hiy[i])
            moved += abs(nx - cx[i]) + abs(ny - cy[i])
            cx[i], cy[i] = nx, ny
        if moved < 1e-9:
            break

    return [(float(cx[i] - 0.5 * ws[i]), float(cy[i] - 0.5 * hs[i]),
             float(ws[i]), float(hs[i])) for i in range(n)]


def _dissect(reg, idx, touch, ctx, out, depth=0):
    if len(idx) == 1:
        return _place_leaf(idx[0], reg, ctx, out)
    x0, y0, W, H = reg
    if W <= _EPS or H <= _EPS:
        return False
    # 回溯預算：每個節點最多 2 個方向 x 3 個候選點，不設上限的話最壞情況是
    # 6^depth。用完就直接放棄整條分割（呼叫端沿用解析式路徑），寧可少一個候選
    # 也不要讓單一案例耗費過多時間。
    ctx.budget[0] -= 1
    if ctx.budget[0] <= 0:
        return False
    # 先試偏好順序（面積長邊，子樹保持接近正方形），失敗改試另一個方向——剛性
    # 方塊卡不下去往往只是方向不對，換一個方向就成立。
    #
    # m==2 額外多試一次反序，此時 k 只能是 1，排序給的分派是唯一的，但「哪一顆
    # 放左邊」本來就是可以選的，剛性方塊被排到錯的那一側時，它浪費掉的面積或
    # 超過剛性的膨脹——實測 89 該案例正是卡在這裡（m=2 nrig=1）分配面積 1254.8
    # 只是方塊總面積 1164.0 多 7.8%）。m>2 時 k 的選擇已提供足夠自由度。
    dirs = (True, False) if W >= H else (False, True)
    if ctx.walls:
        # 若整批方塊都要求同一面的「一分開就貼不到」，那條就是被鎖住的，直接
        # 貼死在整份一邊，周圍那批就貼不到。這裡改變方向的嘗試順序，
        # 讓平行於該面的分割優先試——邊界帶因此得以維持完整。左右被鎖就走
        # 水平切，上下被鎖就走豎直切（兩者都鎖，視角落，就維持長邊優先）。
        lock = _locked(ctx, idx, touch)
        lr, tb = bool(lock & (_L | _R)), bool(lock & (_T | _B))
        if lr and not tb:
            dirs = (False, True)
        elif tb and not lr:
            dirs = (True, False)

    for vert in dirs:
        base = _order(ctx, idx, vert, touch)
        # 優先切在邊界帶的分割點上（見 _wall_groups）
        # kpref：優先切在邊界帶的分割點，只在**前層兩層**施加。它是回溯成本
        # 的來源，邊界帶的分割點常在該層可行，子樹卻死掉，於是每一層都要試
        # 多個分割點。實測放任何實際迭代下去，案例 96 要 30000 次預算才能找到好解，
        # runtime 從 6.5s 到 35.8s——5.5 倍換在比賽裡等於不到 5^0.3 = 1.62 的
        # 成本，遠超那 24% 的品質收益。
        # 邊界帶本來就是靠前層前一半分出來的，帶著方向已被 _locked 降低。
        # 演算法全部平行於的，不需要 kpref 之外多試。
        kpref = ()
        if ctx.walls and depth < 2:
            nlo, nhi = _wall_groups(ctx, idx, vert, touch)
            kpref = tuple(k for k in (nlo, len(idx) - nhi) if 0 < k < len(idx))
        for seq in ([base, base[::-1]] if len(idx) == 2 else [base]):
            if _try_cut(reg, seq, vert, touch, ctx, out, kpref, depth):
                return True
            # 這一輪的子樹失敗了，清掉可能已寫進 out 的部分結果並試下一種分法，
            # 否則殘留值會讓上層的 `any(o is None)` 檢查誤判為成功。
            for i in idx:
                out[i] = None
    return False


def _try_cut(reg, seq, vert, touch, ctx, out, kpref=(), depth=0):
    """依指定方向跟排序切一刀並遞迴兩側，任何候選分割點都失敗就回傳 False。"""
    x0, y0, W, H = reg
    # 只有 walls 模式的**前層兩層**需要把分割點回溯，邊界帶的分割點常在該層可行。
    # 子樹失敗時，除非有必要維持「取第一個可行分割點」的原始行為，那條路徑已知
    # 穩定且快，不值得去回溯的代價——實測 limit=3 在整個案子的樹底下遇到的
    # 預算，那會把本來能解決的案例 89 也被拖到全套。
    lim = 3 if (ctx.walls and depth < 2) else 1
    if vert:
        opts = _cut_options(ctx, seq, W, H, x0, ctx.phx_hi, ctx.phx_lo,
                            ctx.rig_w, ctx.rig_h, kpref, lim)
    else:
        opts = _cut_options(ctx, seq, H, W, y0, ctx.phy_hi, ctx.phy_lo,
                            ctx.rig_h, ctx.rig_w, kpref, lim)
    for k, t in opts:
        if vert:
            r1, r2 = (x0, y0, t, H), (x0 + t, y0, W - t, H)
            t1, t2 = touch & ~_R, touch & ~_L
        else:
            r1, r2 = (x0, y0, W, t), (x0, y0 + t, W, H - t)
            t1, t2 = touch & ~_T, touch & ~_B
        if (_dissect(r1, seq[:k], t1, ctx, out, depth + 1)
                and _dissect(r2, seq[k:], t2, ctx, out, depth + 1)):
            return True
        for i in seq:      # 清掉失敗留下的殘留
            out[i] = None
    return False


def intra_region_alignment(out, reg_w, reg_h, reg_x, reg_y, bcode, clust, is_pre):
    """利用切割區域內部的 Slack (util ≈ 0.90 帶來的 ~10% 空間) 進行微對齊：
      1. 對於要求貼牆的方塊 (bcode > 0)：推進至區域外壁。
      2. 對於同屬一個 Cluster (id > 0) 的方塊：計算 Cluster 中心點 (Cx, Cy)，
         並在其區域可動範圍內將方塊座標朝 (Cx, Cy) 偏移，縮短成員間距離。
    """
    n = len(out)
    if n == 0:
        return out
    xs = np.array([o[0] for o in out], float)
    ys = np.array([o[1] for o in out], float)
    ws = np.array([o[2] for o in out], float)
    hs = np.array([o[3] for o in out], float)
    bcode = np.asarray(bcode, int)
    clust = np.asarray(clust, int)
    is_pre = np.asarray(is_pre, bool)

    cx = xs + 0.5 * ws
    cy = ys + 0.5 * hs
    clust_centers = {}
    for c in np.unique(clust):
        if c <= 0:
            continue
        m = clust == c
        if m.sum() > 1:
            clust_centers[c] = (cx[m].mean(), cy[m].mean())

    new_out = []
    for i in range(n):
        if is_pre[i]:
            new_out.append(out[i])
            continue
        w_i, h_i = ws[i], hs[i]
        rx, ry, rw, rh = reg_x[i], reg_y[i], reg_w[i], reg_h[i]
        min_x, max_x = rx, rx + max(0.0, rw - w_i)
        min_y, max_y = ry, ry + max(0.0, rh - h_i)

        x_i, y_i = xs[i], ys[i]
        b = bcode[i]
        c_id = clust[i]

        if b & _L:
            x_i = min_x
        elif b & _R:
            x_i = max_x

        if b & _B:
            y_i = min_y
        elif b & _T:
            y_i = max_y

        if c_id in clust_centers and not (b & (_L | _R)):
            ccx, _ = clust_centers[c_id]
            target_x = ccx - 0.5 * w_i
            x_i = min(max(target_x, min_x), max_x)

        if c_id in clust_centers and not (b & (_T | _B)):
            _, ccy = clust_centers[c_id]
            target_y = ccy - 0.5 * h_i
            y_i = min(max(target_y, min_y), max_y)

        new_out.append((x_i, y_i, w_i, h_i))
    return new_out


def slice_pack(x, y, w, h, areas, is_fixed, is_pre, clust_id, bcode, mib_id=None,
               util_lo=None, util_hi=None, steps=None, asp_scale=1.0,
               walls=False, nets=None, align_slice=False, return_pair=False):
    """回傳 (x, y, w, h)，任何一步無法保證合法就回傳 None。
    """
    x = np.asarray(x, float); y = np.asarray(y, float)
    w = np.asarray(w, float); h = np.asarray(h, float)
    areas = np.asarray(areas, float)
    is_fixed = np.asarray(is_fixed, bool); is_pre = np.asarray(is_pre, bool)
    n = len(x)
    if n == 0:
        return None
    util_lo = _UTIL_LO if util_lo is None else util_lo
    util_hi = _UTIL_HI if util_hi is None else util_hi
    mib = None if mib_id is None else np.asarray(mib_id, int)
    ctx = _Ctx(x, y, w, h, areas, is_fixed, is_pre,
               np.asarray(clust_id, int), np.asarray(bcode, int))
    ctx.phx_hi = np.where(is_pre, x + w, -np.inf)
    ctx.phx_lo = np.where(is_pre, x, np.inf)
    ctx.phy_hi = np.where(is_pre, y + h, -np.inf)
    ctx.phy_lo = np.where(is_pre, y, np.inf)
    rigid = is_fixed | is_pre
    ctx.rig_w = np.where(rigid, w, 0.0)
    ctx.rig_h = np.where(rigid, h, 0.0)
    ctx.reg_w = np.zeros(n)
    ctx.reg_h = np.zeros(n)
    ctx.reg_x = np.zeros(n)
    ctx.reg_y = np.zeros(n)

    do_align = bool(align_slice or os.environ.get("ELECTRO_SLICE_ALIGN", "0") == "1")

    def attempt(util):
        reg = _outline(ctx, util, asp_scale)
        out = [None] * n
        ctx.walls = walls
        ctx.budget = [_WBUDGET if walls else 200000]
        if _DEBUG:
            _FAIL.clear()
        ok = _dissect(reg, list(range(n)), _L | _R | _T | _B, ctx, out)
        if ok and not any(o is None for o in out):
            if mib is not None:
                _unify_mib(ctx, mib, out)
            if return_pair:
                out_base = [list(o) for o in out]
                out_al = intra_region_alignment(out_base, ctx.reg_w, ctx.reg_h, ctx.reg_x,
                                                ctx.reg_y, ctx.bcode, ctx.clust, ctx.is_pre)
                o1 = np.array(out_base, dtype=float)
                o2 = np.array(out_al, dtype=float)
                r1 = (o1[:, 0], o1[:, 1], o1[:, 2], o1[:, 3])
                r2 = (o2[:, 0], o2[:, 1], o2[:, 2], o2[:, 3])
                return r1, r2
            if do_align:
                out = intra_region_alignment(out, ctx.reg_w, ctx.reg_h, ctx.reg_x,
                                             ctx.reg_y, ctx.bcode, ctx.clust, ctx.is_pre)
            if os.environ.get("ELECTRO_HPWL_POLISH", "0") == "1":
                nb_ = nets or {}
                out = hpwl_polish(out, ctx.reg_w, ctx.reg_h, ctx.reg_x,
                                  ctx.reg_y, nb_.get("eb"), nb_.get("ep"),
                                  nb_.get("pv"), ctx.bcode, ctx.is_pre,
                                  rounds=int(os.environ.get(
                                      "ELECTRO_HPWL_POLISH_ROUNDS", "3")))
            o = np.array(out, dtype=float)
            return o[:, 0], o[:, 1], o[:, 2], o[:, 3]
        if _DEBUG:
            sys.stderr.write(
                f"[slice] n={n} util={util:.3f} 多軸={reg[2]:.1f}x{reg[3]:.1f}"
                f"（面積 {reg[2] * reg[3]:.1f}，方塊總面積 {areas.sum():.1f}，失敗）\n"
                + "".join(f"[slice]   {s}\n" for s in _FAIL))
        return None

    best = attempt(util_hi)
    if best is not None:
        return best
    best = attempt(util_lo)
    if best is None:
        return None
    lo, hi = util_lo, util_hi          # lo 已知可行，hi 已知不可行
    for _ in range(_STEPS if steps is None else int(steps)):
        mid = 0.5 * (lo + hi)
        r = attempt(mid)
        if r is None:
            hi = mid
        else:
            best, lo = r, mid
    if _DEBUG:
        sys.stderr.write(f"[slice] n={n} 二分收斂於 util={lo:.3f}\n")
    return best
