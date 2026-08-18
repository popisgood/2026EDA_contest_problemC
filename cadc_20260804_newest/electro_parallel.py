"""Parallel multi-start worker, kept in its own importable module.

The contest harness loads electro_optimizer.py from a file path under the name
"optimizer_module", which is NOT importable by name -- so a worker function
defined there can't be pickled to a multiprocessing worker.  Defining the worker
here (electro_parallel is a normal module on sys.path, because electro_optimizer
inserts its own directory) makes it picklable, so the fork pool works.

Per-case inputs are stashed in the module global `WORK` *before* the pool is
created, so the fork inherits them and we never pickle the connectivity tensors.
"""
from __future__ import annotations

import sys
import time

import numpy as np

import os

from analytical_place import place
from legalize import compact_layout, legalize, remove_overlap
from slice_pack import slice_pack
from cluster_virtualize import slice_pack_clustered
from soft_repair import (boundary_snap, grouping_repair, mib_unify,
                         soft_violation_counts)

WORK = None   # per-case inputs, set by the parent before forking the pool


def _edges_np(b2b, p2b, pins, n):
    """Pull the valid (non-padding) edges/pins out as numpy for HPWL scoring."""
    def valid(t):
        if t is None or t.numel() == 0:
            return None
        a = t.cpu().numpy()
        a = a[a[:, 0] != -1]
        return a if len(a) else None
    return valid(b2b), valid(p2b), valid(pins)


def _hpwl(cx, cy, eb, ep, pv):
    """Contest HPWL (center-to-center Manhattan, b2b + p2b) for candidate ranking.

    放在這裡，而不是 electro_optimizer，是因為 worker 也要用同樣公式選擇最好；
    electro_optimizer 會匯入 electro_parallel。兩邊共用避免邏輯循環。
    """
    n = len(cx)
    wl = 0.0
    if eb is not None:
        i = np.clip(eb[:, 0].astype(int), 0, n - 1)
        j = np.clip(eb[:, 1].astype(int), 0, n - 1)
        wl += float((eb[:, 2] * (np.abs(cx[i] - cx[j]) + np.abs(cy[i] - cy[j]))).sum())
    if ep is not None and pv is not None and len(pv):
        pi = np.clip(ep[:, 0].astype(int), 0, len(pv) - 1)
        bi = np.clip(ep[:, 1].astype(int), 0, n - 1)
        wl += float((ep[:, 2] * (np.abs(pv[pi, 0] - cx[bi]) + np.abs(pv[pi, 1] - cy[bi]))).sum())
    return wl


# 試過但實測更差的兩種「seed portfolio」的死路，記在這裡避免有人再走一次：
#
#   (a) 每個 seed 給不同組的旋鈕擴散係數（ext_wl / bb1 / area_grow / ov1 ...），
#       seeds=4 得到 2.1014，比純多起點的 2.0929 還差。原因是表裡每個值本來
#       跟隨都是預設最好，累累大於於多樣性的貢獻。
#   (b) 用「非負座標的第一象限鉗夾/牆」當作 seed 維度，seeds=4 得到 2.2825，
#       差更多。clamp=0 實驗跟隨機比較好（2.4442 vs 2.7097），但那是搭配
#       NONNEG=0（容許負座標，違反題目），一旦保留 NONNEG=1，收尾的
#       remove_overlap 要把超界的方塊推回來，就把前面解警告的那樣違反假設在
#       那裡崩潰。
#
# 結論：直接多起點（同一組，所有起）。實測 1->2.4295, 2->2.1905, 4->2.0929。


def _finish(x, y, w, h, P, compact=None):
    """共用的收尾：壓實 -> 軟約束修復 -> 硬可行性保證。

    `compact=None` 表示沿用 P 的設定，切割式路徑會明確傳 False，因為它的輸出
    已經是緊的，而壓實反而左右把破壞它拚 bcode 貼齊到底，一路的方塊。
    """
    is_pre, clust_id, bcode = P["is_pre"], P["clust_id"], P["bcode"]
    # nonneg keeps the WHOLE chain (legalize + both repairs + final cleanup)
    # floored at 0, so blocks never drift far below the wall and get shoved back
    # -- the incremental floor that makes first-quadrant containment cheap rather
    # than exploding (a post-hoc floor-only-at-the-end shove cascades the legalizer).
    floor = 0.0 if P.get("nonneg", False) else None
    # 硬可行性保底（壓實之後）：修復用的 bbox 是壓實後的緊 bbox，貼邊需要走的
    # 距離比較短，反過來先壓實再修復，成功率比較好的方塊得到一致一來。
    if P.get("compact", True) if compact is None else compact:
        x, y = compact_layout(x, y, w, h, is_pre, floor=floor,
                              rounds=P.get("compact_rounds", 4))
    for _ in range(P["rounds"]):
        x, y = grouping_repair(x, y, w, h, clust_id, is_pre, floor=floor,
                               bcode=bcode)
        # 帶 clust_id，boundary_snap 交換移動需要它，才能確認被交換方塊丟入
        # 的還是屬於自己的 cluster 群組裡才吻。
        x, y = boundary_snap(x, y, w, h, bcode, is_pre, floor=floor,
                             clust_id=clust_id)
    # MIB 事後形狀統一（opt-in）。放在這個位置的理由：軟約束修復已經把方塊移到
    # 最終位置，此時才知道每顆方塊周圍實際有多少空間可以改形狀；而且改完形狀後
    # 下面還有 remove_overlap 當最後保底，就算守衛有疏漏也不會產出不合法的解。
    if os.environ.get("ELECTRO_MIB_UNIFY", "0") == "1" and P.get("mib_id") is not None:
        x, y, w, h = mib_unify(x, y, w, h, P["mib_id"], P["is_fixed"], is_pre,
                               P["areas_np"], floor=floor)
    # MIB 統一會改形狀，可能把原本已經貼齊的 boundary 方塊推歪（這正是 §8.36
    # 判定「無條件套用 mib_unify 會退步」的機制）。electro_v10 的做法是在後面
    # 補一輪 boundary_snap 把位移修回來——本旗標把那一輪獨立包起來，方便單獨
    # 量出它的貢獻（見計畫 Task 5 的歸因實驗）。
    # passes=2 沿用 electro_v10 的設定（boundary_snap 自己的預設是 3）。
    if os.environ.get("ELECTRO_MIB_POST_SNAP", "0") == "1":
        x, y = boundary_snap(x, y, w, h, bcode, is_pre, passes=2, floor=floor,
                             clust_id=clust_id)
    # final hard-feasibility net; nonneg also enforces the x=0/y=0 canvas walls
    x, y = remove_overlap(x, y, w, h, is_pre, nonneg=P.get("nonneg", False))
    return x, y, w, h


def anchors(P):
    """提供給正規化的錨點。**必須跟候選池無關**（否則多加一個候選就會改變既有
    候選的相對排名，用候選池平均當基準，實測案例 96 因此在加入更好的候選後
    反而選出更差的解）。

      * aa = 方塊總面積 / 0.966。0.966 是實測 ground truth 的包夾平均填充的，
        所以這是 area_baseline 的良好估計。
      * ha = sqrt(aa) x 邊數，純粹的尺度常數。

    獨立於 worker 共用一份，避免不同案子產生歧異。
    """
    eb, ep = P.get("eb"), P.get("ep")
    aa = float(P["areas_np"].sum()) / 0.966 or 1.0
    ne = max(1, (0 if eb is None else len(eb)) + (0 if ep is None else len(ep)))
    return (aa ** 0.5) * ne or 1.0, aa


def score(x, y, w, h, P, ha, aa):
    """候選的代理成本，形狀是官方 Cost：exp(2*V_rel) * (hpwl/ha + area/aa)。

    在 worker 裡算，而不是在總控程算，因為總控程要串接跑 seeds x 候選數的結果
    （8 seeds x 3 候選 = 24 份），每份都含 O(n^2) 的 soft_violation_counts，
    真正拉高 seeds=8 的 wall time（3.87s）比純 seed（2.20s）多出來的部分——8 個
    worker 平行時串起來的時間都花在總控程的匯合段。
    """
    vb, vg, vm, ns = soft_violation_counts(x, y, w, h, P["bcode"],
                                           P["clust_id"], P.get("mib_id"))
    vrel = (vb + vg + vm) / max(ns, 1)
    hp = _hpwl(x + 0.5 * w, y + 0.5 * h, P.get("eb"), P.get("ep"), P.get("pv"))
    ar = float((x + w).max() - x.min()) * float((y + h).max() - y.min())
    return vrel, hp, ar, float(np.exp(2.0 * vrel) * (hp / ha + ar / aa))


def _prerank(raws, P, topk):
    """從原始候選結果裡挑出最有希望的前 K 個，排除直接不用、不跑修復的。

    用的是跟總控程最後排名同一形式的代理成本
    `exp(2*V_rel) * (hpwl/ha + area/aa)`，而且跟總控程一樣候選池無關：
      * aa = 方塊總面積 / 0.966，0.966 是通用的 ground truth 填充的，
      * ha = 方塊總面積開根號 x 邊數，純粹的尺度常數

    這是排代，這裡計算的重要是**修復前**的，修復會改變它。但排序足夠用。

    注意，這個過濾**不是**能省算力。原本本來以為修復很貴，實測 profile 才發現
    實際佔單個 seed 約 5%（n=120 是 0.066s，而 place() 是 2.5s），所以拒絕多除的
    修復佔一直約 2%。真正天花是因為它讓。分區候選維度。這個成本逼近零，而候選
    維度正是品質的品質主要來源。
    """
    if len(raws) <= topk:
        return raws
    ha, aa = anchors(P)
    scored = [(score(rx, ry, rw, rh, P, ha, aa)[3], (rx, ry, rw, rh))
              for (rx, ry, rw, rh) in raws]
    scored.sort(key=lambda s: s[0])
    return [s[1] for s in scored[:topk]]


def _v11_run_start(seed, P):
    """One independent start.  只保留「已試過」的候選數字，總控程只選最好的。

    兩條合法化路徑分別對同一份解析佈局：
      A. 梯度式（legalize）——保底，行為跟以前完全一樣；
      B. 切割式（slice_pack）——面積比例迴圈切割，填充的結構性較高很多。
    B 部分可能會回傳 None（改性方塊卡不進去等），那就只有 A。交給總控程整選，
    所以多路徑不可能改分數變差。

    這裡在這裡算完（而不是在總控程分開），因為算含 O(n^2) 的 soft_violation_counts，
    總控程要串接跑 seeds x 候選數的結果，那段串起時間真正是 seeds=8 的 wall time
    比單 seed 多出來的主因。本進來就跟著 worker 一起平行了。
    """
    t_in = time.time()
    positions, _ = place(
        P["n"], P["area"], P["b2b"], P["p2b"], P["pins"], P["cons"], P["tp"],
        iters=P["iters"], lr=P["lr"], device=P["device"], seed=seed,
        init_centers=P["init"],
    )
    t_place = time.time() - t_in
    x = np.array([p[0] for p in positions], dtype=float)
    y = np.array([p[1] for p in positions], dtype=float)
    w = np.array([p[2] for p in positions], dtype=float)
    h = np.array([p[3] for p in positions], dtype=float)
    is_pre = P["is_pre"]
    floor = 0.0 if P.get("nonneg", False) else None

    ha, aa = anchors(P)
    done = []

    t_m = time.time()
    xa, ya = legalize(x, y, w, h, is_pre, floor=floor)
    t_legal = time.time() - t_m
    t_m = time.time()
    done.append(_finish(xa, ya, w, h, P))
    t_finish = time.time() - t_m
    t_slice = 0.0

    if P.get("slice", False):
        # 用兩個候選維度的乘積，多案案例 x 的帶策略。兩者都是齊，實際大致。實測
        # 直接切割失敗——切割帶案例 89 的 V_rel 從 0.246 減少到 0.108，cost
        # 2.1816 -> 1.6572）；換案例的是每案比例都切割不出來，preplaced 被
        # 推到切正中間，兩側都塞不下）。所以兩個維度都要試。
        t_m = time.time()
        raws = []
        do_portfolio = os.environ.get("ELECTRO_SLICE_ALIGN_PORTFOLIO", "0") == "1"
        # §8.53（opt-in ELECTRO_SLICE_CLUSTER_VIRTUALIZE=1）：純軟方塊的乾淨
        # cluster 額外用「虛擬超級方塊」guillotine 切割一次，by-construction
        # 保證 V_group=0（見 cluster_virtualize.py）。**嚴格加法式**：跟一般
        # slice_pack 的結果一起丟進候選池，讓既有的 proxy 排名逐案挑，變差就
        # 不會被選中。
        do_cluster_virt = os.environ.get("ELECTRO_SLICE_CLUSTER_VIRTUALIZE", "0") == "1"
        for a in P.get("slice_aspects", (1.0,)):
            for wl in P.get("slice_walls", (False,)):
                res = slice_pack(x, y, w, h, P["areas_np"], P["is_fixed"], is_pre,
                                 P["clust_id"], P["bcode"], P.get("mib_id"),
                                 asp_scale=a, walls=wl, return_pair=do_portfolio,
                                 nets={"eb": P.get("eb"), "ep": P.get("ep"),
                                       "pv": P.get("pv")})
                if res is not None:
                    if do_portfolio:
                        r1, r2 = res
                        raws.append(r1)
                        raws.append(r2)
                    else:
                        raws.append(res)
                if do_cluster_virt:
                    resc = slice_pack_clustered(
                        x, y, w, h, P["areas_np"], P["is_fixed"], is_pre,
                        P["clust_id"], P["bcode"], P.get("mib_id"),
                        asp_scale=a, walls=wl, return_pair=do_portfolio,
                        nets={"eb": P.get("eb"), "ep": P.get("ep"),
                              "pv": P.get("pv")})
                    if resc is not None:
                        if do_portfolio:
                            r1, r2 = resc
                            raws.append(r1)
                            raws.append(r2)
                        else:
                            raws.append(resc)
        t_slice = time.time() - t_m
        t_m = time.time()
        for r in _prerank(raws, P, P.get("slice_topk", 2)):
            done.append(_finish(r[0], r[1], r[2], r[3], P,
                                compact=P.get("slice_compact", False)))
        t_finish += time.time() - t_m

    # --- place-compact：用已壓實佈局的中心點重跑一輪短版解析佈局 -----------
    # 移植自我們自己 electro_optimized/ 路線本季最大的單一突破（2.1230→1.9666，
    # −7.4%，見 8_Winning_Strategy_and_Roadmap.md §8.35）。核心想法：legalize +
    # 軟約束修復之後的佈局比原始解析輸出緊密得多，把它的中心點當**新的初始化**
    # 再跑一輪短的梯度下降，等於讓優化器在一個好得多的起點上重新決定「哪些方塊
    # 當鄰居」——重做的是**排列**這個維度。
    #
    # 待驗證的問題：slice_pack 也是在重做排列（離散切割 vs 連續梯度），依 §8.35
    # 的正交性判準，同維度的機制大概率互相蓋掉。這正是需要實測的。
    if os.environ.get("ELECTRO_PLACE_COMPACT", "0") == "1" and done:
        pc_iters = int(os.environ.get("ELECTRO_PLACE_COMPACT_ITERS", "400"))
        pc_src = list(done)          # 對目前每個候選都做一份（含 slice 候選）
        if os.environ.get("ELECTRO_PLACE_COMPACT_BEST", "1") == "1":
            # 預設只對「目前最好的」候選做，避免候選數與 runtime 爆炸
            pc_src = [min(pc_src, key=lambda c: score(c[0], c[1], c[2], c[3],
                                                      P, ha, aa)[3])]
        elif "ELECTRO_PLACE_COMPACT_TOPK" in os.environ:
            topk = int(os.environ["ELECTRO_PLACE_COMPACT_TOPK"])
            pc_src = sorted(pc_src, key=lambda c: score(c[0], c[1], c[2], c[3],
                                                        P, ha, aa)[3])[:topk]
        for (bx, by, bw, bh) in pc_src:
            try:
                import torch as _t
                ic = _t.tensor([[bx[i] + 0.5 * bw[i], by[i] + 0.5 * bh[i]]
                                for i in range(len(bx))], dtype=_t.float32)
                pos2, _ = place(
                    P["n"], P["area"], P["b2b"], P["p2b"], P["pins"],
                    P["cons"], P["tp"], iters=pc_iters, lr=P["lr"],
                    device=P["device"], seed=seed, init_centers=ic,
                )
                x2 = np.array([p[0] for p in pos2], dtype=float)
                y2 = np.array([p[1] for p in pos2], dtype=float)
                w2 = np.array([p[2] for p in pos2], dtype=float)
                h2 = np.array([p[3] for p in pos2], dtype=float)
                xa2, ya2 = legalize(x2, y2, w2, h2, is_pre, floor=floor)
                done.append(_finish(xa2, ya2, w2, h2, P))
            except Exception:
                pass          # 嚴格加法式：失敗就沒有這個候選，不影響既有結果

    # --- MIB 統一版本當「額外候選」（ELECTRO_MIB_PORTFOLIO=1）--------------
    # 無條件套用 MIB 統一實測會有副作用：改形狀會挪動方塊邊緣，把原本貼齊的
    # boundary 弄歪、把原本相鄰的 grouping 拆開（小案例實測 Vmib 9->5 但
    # Vbnd 5->9，淨分數反而變差）。這正是本專題 §8.13-8.14 學到的教訓——
    # 同一招「無條件套用退步、做成 portfolio 淨勝」。
    #
    # 所以改成：對每個已完成的候選，額外產生一份 MIB 統一版，兩份都丟進候選池
    # 讓 proxy 逐案挑。**嚴格加法式**：統一版比較差就不會被選中，不可能退步。
    # 成本極低（不需要重跑 place()，只是改形狀 + 一次 remove_overlap）。
    if (os.environ.get("ELECTRO_MIB_PORTFOLIO", "0") == "1"
            and P.get("mib_id") is not None):
        extra = []
        for (cx, cy, cw, ch) in done:
            ux, uy, uw, uh = mib_unify(cx, cy, cw, ch, P["mib_id"],
                                       P["is_fixed"], is_pre, P["areas_np"],
                                       floor=floor)
            ux, uy = remove_overlap(ux, uy, uw, uh, is_pre,
                                    nonneg=P.get("nonneg", False))
            extra.append((ux, uy, uw, uh))
        done.extend(extra)

    out = []
    for (cx, cy, cw, ch) in done:
        vrel, hp, ar, s = score(cx, cy, cw, ch, P, ha, aa)
        # 末尾幾筆是 worker 自己回報的階段耗時。總控程拿 total 跟 starmap 的 wall
        # time 比就能算出平行效率（實測兩者只差 0.04s，平行本身沒有明顯損失），
        # 各段耗時純粹是曾經是在裡好幾次，前以為是修復慢，後以為是 place()），
        # 所以改成不再猜的。
        out.append((cx, cy, cw, ch, vrel, hp, ar, s,
                    time.time() - t_in, t_place, t_legal, t_slice, t_finish))
    return out


# --- LP-displacement portfolio (merged in from electro_v14, 2026-08-02) ----
# electro_v14 was a thin wrapper that monkey-patched this run_start via
# cross-directory sys.path tricks (importing electro_v11's electro_parallel
# as the bare `electro_parallel` name, then swapping in its own run_start).
# That's fragile to depend on for a new experiment, so it's merged in here
# directly: _v11_run_start (above) is the untouched v11 base, legalize_lp
# comes from `lp_legalize.py` (a local copy of electro_optimized/legalize.py,
# not a cross-directory import), and run_start below is what
# electro_optimizer.py actually calls. Strictly additive per floorplan-guard
# point 7: a failed/absent LP candidate just leaves the v11 baseline
# untouched. Default ON here (ELECTRO_LP_DISPLACEMENT_PORTFOLIO=1,
# SEEDS=4, MIN_BLOCKS=0) because that combination -- "v14 LP all sizes /
# all 4 seeds" -- was the best reproducible full-100 result documented
# before this file existed (Neutral 1.4112, real proxy 1.0745); still
# overridable via the same env vars for an apples-to-apples ablation.
from lp_legalize import legalize_lp

os.environ.setdefault("ELECTRO_LP_DISPLACEMENT_PORTFOLIO", "1")
os.environ.setdefault("ELECTRO_LP_DISPLACEMENT_SEEDS", "4")
os.environ.setdefault("ELECTRO_LP_DISPLACEMENT_MIN_BLOCKS", "0")
os.environ.setdefault("ELECTRO_LP_DISPLACEMENT_TOPK", "1")

from lp_legalize import legalize_qinfer_reshape

os.environ.setdefault("ELECTRO_RESHAPE_PORTFOLIO", "0")
os.environ.setdefault("ELECTRO_RESHAPE_SEEDS", "4")
os.environ.setdefault("ELECTRO_RESHAPE_TOPK", "1")
os.environ.setdefault("ELECTRO_RESHAPE_LAM_DISP", "1.0")
os.environ.setdefault("ELECTRO_RESHAPE_SHAPE_RATIO", "4.0")

# --- Quasi-Newton continuous polish portfolio (2026-08-11, new) ------------
# Literature-backed idea (see LITERATURE_REVIEW.md Tier 1 #1): after topology
# and shapes are decided (legalize + repair done), a short L-BFGS pass on
# (cx,cy) only -- (w,h) frozen -- squeezes out residual HPWL that the
# annealed Adam schedule in analytical_place.place() can leave on the table.
# Default OFF so the existing baseline is reproduced byte-for-byte unless
# explicitly opted in; this is a brand-new, not-yet-validated mechanism and
# should be A/B tested against the pre-existing full-100 baseline before
# flipping the default, per the same discipline used for RESHAPE_PORTFOLIO
# above. Strictly additive: a failed polish leaves the candidate absent, the
# rest of the pool is untouched.
from quasi_newton_polish import quasi_newton_polish

os.environ.setdefault("ELECTRO_QN_POLISH_PORTFOLIO", "0")
os.environ.setdefault("ELECTRO_QN_POLISH_SEEDS", "4")
os.environ.setdefault("ELECTRO_QN_POLISH_TOPK", "1")
os.environ.setdefault("ELECTRO_QN_POLISH_ITERS", "40")
os.environ.setdefault("ELECTRO_QN_POLISH_LAM_OV", "1.0")
os.environ.setdefault("ELECTRO_QN_POLISH_LAM_ANCHOR", "0.01")

# --- Area-shrinking reshape portfolio (2026-08-11) --------------------------
# Fixed-center aspect-ratio adjustment to shrink the bbox -- see
# reshape_shrink.py for why this targets Area_gap (linear in the cost
# formula) instead of HPWL (via the shelved quasi-Newton polish above, which
# empirically blew up V_rel, the EXPONENTIAL term, for a small linear gain).
# Default OFF pending A/B validation, same discipline as the other portfolio
# stages.
from reshape_shrink import reshape_shrink

os.environ.setdefault("ELECTRO_RESHAPE_SHRINK_PORTFOLIO", "0")
os.environ.setdefault("ELECTRO_RESHAPE_SHRINK_SEEDS", "4")
os.environ.setdefault("ELECTRO_RESHAPE_SHRINK_TOPK", "1")
os.environ.setdefault("ELECTRO_RESHAPE_SHRINK_ROUNDS", "6")
os.environ.setdefault("ELECTRO_RESHAPE_SHRINK_STEP", "0.08")
os.environ.setdefault("ELECTRO_RESHAPE_SHRINK_AR_CAP", "4.0")

# --- WireMask-BBO constructive candidate (2026-08-11) -----------------------
# See wiremask_bbo.py and LITERATURE_REVIEW.md Tier 1 #1 (Xue et al., NeurIPS
# 2023).  Unlike every _apply_* stage above (which REFINES a candidate
# already in the pool -- and each ran into some way of disturbing the
# boundary/grouping alignment those candidates already had), this builds a
# BRAND NEW candidate from scratch via a different construction order and
# greedy criterion, exactly like slice_pack() already does -- strictly
# additive with no risk of disturbing anything already in the pool.
from wiremask_bbo import wiremask_place

os.environ.setdefault("ELECTRO_WIREMASK_PORTFOLIO", "0")
# 2026-08-11 full-100 finding: the RAW (unrefined) greedy candidate never won
# a single case (Total Score unchanged to 4 decimals vs baseline) while
# still costing +60% average runtime -- every OTHER candidate source gets
# 300 iterations of continuous optimization and this one got zero, so of
# course it never competed.  Fix: feed its centers as init_centers into a
# short place() refinement pass (same trick place-compact already uses).
# The construction itself is deterministic (no randomness), so it now only
# runs once per case (seed==0), not redundantly across all 4 seeds.
os.environ.setdefault("ELECTRO_WIREMASK_REFINE_ITERS", "200")


def _reshapeable_mask(is_pre, is_fixed, mib_id):
    """True for blocks safe to reshape: soft, non-fixed, non-preplaced,
    and not a member of a MIB group with more than one member."""
    mib_id = np.asarray(mib_id, dtype=int)
    is_pre = np.asarray(is_pre, dtype=bool)
    is_fixed = np.asarray(is_fixed, dtype=bool)
    gmax = int(mib_id.max()) if mib_id.size else 0
    group_size = np.zeros(gmax + 1, dtype=int)
    for g in range(1, gmax + 1):
        group_size[g] = int((mib_id == g).sum())
    in_big_mib_group = (mib_id > 0) & (group_size[mib_id] > 1)
    return (~is_pre) & (~is_fixed) & (~in_big_mib_group)


def _best_seed_candidates(candidates, P, limit):
    hpwl_anchor, area_anchor = anchors(P)
    return sorted(
        candidates,
        key=lambda candidate: score(
            candidate[0], candidate[1], candidate[2], candidate[3],
            P, hpwl_anchor, area_anchor)[3],
    )[:limit]


def _apply_lp_displacement(candidates, seed, P):
    """Strictly-additive LP-displacement candidate (electro_v14 lineage).
    A no-op (returns candidates unchanged) when ELECTRO_LP_DISPLACEMENT_
    PORTFOLIO is off or this seed/case is outside its budget -- always
    called unconditionally from run_start() so later stages are never
    silently skipped by an early return here (see run_start's docstring)."""
    if os.environ.get("ELECTRO_LP_DISPLACEMENT_PORTFOLIO", "0") != "1":
        return candidates
    seed_budget = max(0, int(os.environ.get("ELECTRO_LP_DISPLACEMENT_SEEDS", "4")))
    if seed >= seed_budget:
        return candidates
    min_blocks = max(0, int(os.environ.get("ELECTRO_LP_DISPLACEMENT_MIN_BLOCKS", "0")))
    if P["n"] < min_blocks:
        return candidates

    is_pre = P["is_pre"]
    floor = 0.0 if P.get("nonneg", False) else None
    topk = max(1, int(os.environ.get("ELECTRO_LP_DISPLACEMENT_TOPK", "1")))
    extras = []
    for candidate in _best_seed_candidates(candidates, P, topk):
        x, y, w, h = candidate[:4]
        started = time.time()
        try:
            lp_x, lp_y = legalize_lp(x, y, w, h, is_pre, floor=floor)
            fx, fy, fw, fh = _finish(lp_x, lp_y, w, h, P)
            vrel, hpwl, area, proxy = score(fx, fy, fw, fh, P, *anchors(P))
            extras.append((
                fx, fy, fw, fh, vrel, hpwl, area, proxy,
                candidate[8] + (time.time() - started),
                candidate[9], candidate[10], candidate[11],
                candidate[12] + (time.time() - started),
            ))
        except Exception:
            # Strict-additive discipline: a failed LP candidate is simply
            # absent; the input candidates remain.
            continue
    return candidates + extras


def _apply_reshape_qinfer(candidates, seed, P):
    """Strictly-additive overlap-resolving reshape candidate (adjusts
    aspect ratio + position jointly to resolve overlap with minimum
    displacement -- see lp_legalize.legalize_qinfer_reshape).  No-op unless
    ELECTRO_RESHAPE_PORTFOLIO=1 and this seed is within budget."""
    if os.environ.get("ELECTRO_RESHAPE_PORTFOLIO", "0") != "1":
        return candidates
    seed_budget = max(0, int(os.environ.get("ELECTRO_RESHAPE_SEEDS", "4")))
    if seed >= seed_budget:
        return candidates
    topk = max(1, int(os.environ.get("ELECTRO_RESHAPE_TOPK", "1")))
    lam_disp = float(os.environ.get("ELECTRO_RESHAPE_LAM_DISP", "1.0"))
    shape_ratio = float(os.environ.get("ELECTRO_RESHAPE_SHAPE_RATIO", "4.0"))
    is_reshapeable = _reshapeable_mask(P["is_pre"], P["is_fixed"], P.get("mib_id"))
    extras = []
    for candidate in _best_seed_candidates(candidates, P, topk):
        cx0, cy0, cw0, ch0 = candidate[:4]
        started = time.time()
        try:
            rx, ry, rw, rh = legalize_qinfer_reshape(
                cx0, cy0, cw0, ch0, P["is_pre"], is_reshapeable,
                floor=(0.0 if P.get("nonneg", False) else None),
                lam_disp=lam_disp, shape_lam_ratio=shape_ratio)
            fx, fy, fw, fh = _finish(rx, ry, rw, rh, P)
            vrel, hpwl, area, proxy = score(fx, fy, fw, fh, P, *anchors(P))
            extras.append((
                fx, fy, fw, fh, vrel, hpwl, area, proxy,
                candidate[8] + (time.time() - started),
                candidate[9], candidate[10], candidate[11],
                candidate[12] + (time.time() - started),
            ))
        except Exception:
            continue
    return candidates + extras


def _apply_qn_polish(candidates, seed, P):
    """Strictly-additive L-BFGS position polish (see quasi_newton_polish.py).
    SHELVED 2026-08-11: empirically this moves centers to cut HPWL at the
    cost of blowing up V_rel (the boundary/grouping alignment _finish() had
    built), which the EXPONENTIAL exp(2*V_rel) term in the cost formula
    punishes far more than the linear HPWL_gap it improves -- confirmed on a
    real case (V_rel 0.088 -> 0.5-0.65, proxy cost 5-27x worse).  Left wired
    in and default OFF in case a future anchor/smoothing tune makes it
    viable, but ELECTRO_RESHAPE_SHRINK_PORTFOLIO below is the currently
    recommended direction instead."""
    if os.environ.get("ELECTRO_QN_POLISH_PORTFOLIO", "0") != "1":
        return candidates
    qn_seed_budget = max(0, int(os.environ.get("ELECTRO_QN_POLISH_SEEDS", "4")))
    if seed >= qn_seed_budget:
        return candidates
    qn_topk = max(1, int(os.environ.get("ELECTRO_QN_POLISH_TOPK", "1")))
    qn_iters = int(os.environ.get("ELECTRO_QN_POLISH_ITERS", "40"))
    qn_lam_ov = float(os.environ.get("ELECTRO_QN_POLISH_LAM_OV", "1.0"))
    qn_lam_anchor = float(os.environ.get("ELECTRO_QN_POLISH_LAM_ANCHOR", "0.01"))
    is_pre = P["is_pre"]
    debug = os.environ.get("ELECTRO_QN_POLISH_DEBUG", "0") == "1"
    qn_extras = []
    for candidate in _best_seed_candidates(candidates, P, qn_topk):
        cx0, cy0, cw0, ch0 = candidate[:4]
        started = time.time()
        try:
            qx0, qy0 = quasi_newton_polish(
                cx0, cy0, cw0, ch0, is_pre, P.get("eb"), P.get("ep"), P.get("pv"),
                iters=qn_iters, lam_ov=qn_lam_ov, lam_anchor=qn_lam_anchor)
            # compact_layout() is order-preserving (only removes slack along
            # each axis), so it tends to snap positions right back to nearly
            # the same tightly-packed layout regardless of the polish's fine
            # HPWL-driven nudges -- exactly why slice_pack's own candidates
            # pass compact=False in _v11_run_start.  Same reasoning here:
            # skip compact, keep the repair rounds + final remove_overlap net.
            qx, qy, qw, qh = _finish(qx0, qy0, cw0, ch0, P, compact=False)
            vrel, hpwl, area, proxy = score(qx, qy, qw, qh, P, *anchors(P))
            if debug:
                moved = float(np.abs(qx0 - cx0).sum() + np.abs(qy0 - cy0).sum())
                sys.stderr.write(
                    f"[qn_polish] seed={seed} moved_l1={moved:.4f} "
                    f"vrel={vrel:.4f} proxy={proxy:.4f}\n")
            qn_extras.append((
                qx, qy, qw, qh, vrel, hpwl, area, proxy,
                candidate[8] + (time.time() - started),
                candidate[9], candidate[10], candidate[11],
                candidate[12] + (time.time() - started),
            ))
        except Exception as e:
            # Strict-additive discipline: a failed polish leaves the rest of
            # the pool untouched.
            if debug:
                sys.stderr.write(f"[qn_polish] seed={seed} FAILED: {e!r}\n")
            continue
    return candidates + qn_extras


def _apply_reshape_shrink(candidates, seed, P):
    """Strictly-additive area-shrinking reshape (see reshape_shrink.py):
    greedily narrows whichever reshapeable block currently sits at a bbox
    extreme, then recompacts via legalize.compact_layout() so the freed
    slack cascades -- unlike the shelved fixed-center version, positions
    here DO move (via the proven compaction machinery), which is what
    actually lets neighbors slide into freed space.  Targets Area_gap
    directly (linear in the cost formula).

    Two guards against the cascade breaking soft constraints the rest of
    the pipeline already satisfied:
      1. Blocks with a boundary code OR a cluster membership are excluded
         from having their OWN shape changed (on top of the existing
         non-fixed/non-preplaced/non-multi-MIB mask) -- reshaping a
         boundary-coded block risks reopening V_boundary; reshaping a
         clustered block risks pulling it away from its cluster-mates.
      2. compact_layout()'s cascade has NO notion of clustering, so even
         reshaping only non-clustered blocks can still drag an unrelated
         clustered block away from its cluster-mates as a side effect.
         Each round is checked via soft_violation_counts BEFORE it's kept:
         if boundary+grouping violations exceed the ORIGINAL (pre-reshape)
         level, that round is discarded and the loop stops, keeping the
         last accepted state.

    No-op unless ELECTRO_RESHAPE_SHRINK_PORTFOLIO=1 and this seed is within
    budget."""
    if os.environ.get("ELECTRO_RESHAPE_SHRINK_PORTFOLIO", "0") != "1":
        return candidates
    seed_budget = max(0, int(os.environ.get("ELECTRO_RESHAPE_SHRINK_SEEDS", "4")))
    if seed >= seed_budget:
        return candidates
    topk = max(1, int(os.environ.get("ELECTRO_RESHAPE_SHRINK_TOPK", "1")))
    rounds = int(os.environ.get("ELECTRO_RESHAPE_SHRINK_ROUNDS", "6"))
    step = float(os.environ.get("ELECTRO_RESHAPE_SHRINK_STEP", "0.08"))
    ar_cap = float(os.environ.get("ELECTRO_RESHAPE_SHRINK_AR_CAP", "4.0"))
    debug = os.environ.get("ELECTRO_RESHAPE_SHRINK_DEBUG", "0") == "1"
    base_mask = _reshapeable_mask(P["is_pre"], P["is_fixed"], P.get("mib_id"))
    bcode = np.asarray(P["bcode"])
    clust_id = np.asarray(P.get("clust_id"))
    is_reshapeable = base_mask & (bcode == 0) & (clust_id == 0)
    is_pre = P["is_pre"]
    floor = 0.0 if P.get("nonneg", False) else None
    extras = []
    for candidate in _best_seed_candidates(candidates, P, topk):
        cx0, cy0, cw0, ch0 = candidate[:4]
        started = time.time()
        try:
            pre_area = (float(np.max(cx0 + cw0) - np.min(cx0))
                        * float(np.max(cy0 + ch0) - np.min(cy0)))
            vb0, vg0, _vm0, _ns0 = soft_violation_counts(
                cx0, cy0, cw0, ch0, P["bcode"], P["clust_id"], P.get("mib_id"))
            best = (cx0, cy0, cw0, ch0)
            best_area = pre_area
            n_rounds_kept = 0
            for _r in range(rounds):
                nx, ny, nw, nh = reshape_shrink(
                    *best, is_reshapeable, is_pre,
                    rounds=1, step=step, ar_cap=ar_cap, floor=floor)
                cur_area = (float(np.max(nx + nw) - np.min(nx))
                            * float(np.max(ny + nh) - np.min(ny)))
                vb, vg, _vm, _ns = soft_violation_counts(
                    nx, ny, nw, nh, P["bcode"], P["clust_id"], P.get("mib_id"))
                # Reject the round if it dragged a boundary/cluster block
                # along (vb+vg worse) OR if the bbox didn't actually shrink --
                # area-exact reshaping trades width for height, so narrowing
                # several blocks stacked in the same column can shrink the
                # column's width a little while their SUMMED height growth
                # blows up the bbox far more (found empirically: bbox_area
                # 16186 -> 25267 on one seed before this check existed).
                if (vb + vg) > (vb0 + vg0) + 1e-9 or cur_area > best_area - 1e-9:
                    break
                best = (nx, ny, nw, nh)
                best_area = cur_area
                n_rounds_kept += 1
            rx, ry, rw, rh = best
            fx, fy, fw, fh = _finish(rx, ry, rw, rh, P, compact=False)
            vrel, hpwl, area, proxy = score(fx, fy, fw, fh, P, *anchors(P))
            if debug:
                sys.stderr.write(
                    f"[reshape_shrink] seed={seed} rounds_kept={n_rounds_kept}/{rounds} "
                    f"bbox_area {pre_area:.4f} -> {area:.4f} "
                    f"vrel={vrel:.4f} proxy={proxy:.4f}\n")
            extras.append((
                fx, fy, fw, fh, vrel, hpwl, area, proxy,
                candidate[8] + (time.time() - started),
                candidate[9], candidate[10], candidate[11],
                candidate[12] + (time.time() - started),
            ))
        except Exception as e:
            if debug:
                sys.stderr.write(f"[reshape_shrink] seed={seed} FAILED: {e!r}\n")
            continue
    return candidates + extras


def _apply_wiremask_bbo(candidates, seed, P):
    """Strictly-additive WireMask-BBO-style candidate (see wiremask_bbo.py):
    builds a fresh floorplan via greedy connectivity-ordered corner
    insertion, THEN refines it with a short place() gradient-descent pass --
    the candidate's own centers become init_centers for more Adam
    iterations, exactly the trick place-compact already uses elsewhere in
    this file.

    Refinement is not optional: a full-100 test of the RAW (unrefined)
    greedy candidate never won a single case (Total Score unchanged to 4
    decimals vs baseline) while still costing +60% average runtime --
    unsurprising, since every OTHER candidate source gets 300 iterations of
    continuous optimization and this one got zero.  The greedy construction
    is still useful as a STARTING POINT (a genuinely different block
    ordering / basin than the gradient path's own random or ML-warm-started
    init), just not as a final candidate on its own.

    The construction is deterministic (no randomness), so unlike every
    other portfolio stage there is no benefit to repeating it per seed --
    only seed==0 does the work; other seeds are a no-op, avoiding the 4x
    redundant cost the unrefined version paid.  No-op unless
    ELECTRO_WIREMASK_PORTFOLIO=1."""
    if os.environ.get("ELECTRO_WIREMASK_PORTFOLIO", "0") != "1":
        return candidates
    if seed != 0:
        return candidates
    debug = os.environ.get("ELECTRO_WIREMASK_DEBUG", "0") == "1"
    refine_iters = int(os.environ.get("ELECTRO_WIREMASK_REFINE_ITERS", "200"))
    started = time.time()
    try:
        tp = P.get("tp")
        n = P["n"]
        tp_np = (tp[:n].cpu().numpy() if tp is not None and tp.numel() > 0
                  else np.full((n, 4), -1.0))
        wx, wy, ww, wh = wiremask_place(
            P["areas_np"], P["is_fixed"], P["is_pre"], tp_np,
            P.get("eb"), P.get("ep"), P.get("pv"))
        t_construct = time.time() - started

        is_pre = P["is_pre"]
        floor = 0.0 if P.get("nonneg", False) else None
        t_refine = 0.0
        if refine_iters > 0:
            import torch as _t
            ic = _t.tensor([[wx[i] + 0.5 * ww[i], wy[i] + 0.5 * wh[i]]
                             for i in range(n)], dtype=_t.float32)
            t_r = time.time()
            positions, _ = place(
                n, P["area"], P["b2b"], P["p2b"], P["pins"], P["cons"], P["tp"],
                iters=refine_iters, lr=P["lr"], device=P["device"], seed=seed,
                init_centers=ic)
            t_refine = time.time() - t_r
            rx = np.array([p[0] for p in positions], dtype=float)
            ry = np.array([p[1] for p in positions], dtype=float)
            rw = np.array([p[2] for p in positions], dtype=float)
            rh = np.array([p[3] for p in positions], dtype=float)
            xa, ya = legalize(rx, ry, rw, rh, is_pre, floor=floor)
            fx, fy, fw, fh = _finish(xa, ya, rw, rh, P)
        else:
            fx, fy, fw, fh = _finish(wx, wy, ww, wh, P)

        vrel, hpwl, area, proxy = score(fx, fy, fw, fh, P, *anchors(P))
        dt = time.time() - started
        if debug:
            sys.stderr.write(
                f"[wiremask] seed={seed} t_construct={t_construct:.3f}s "
                f"t_refine={t_refine:.3f}s vrel={vrel:.4f} hpwl={hpwl:.4f} "
                f"area={area:.4f} proxy={proxy:.4f} t={dt:.3f}s\n")
        extra = (fx, fy, fw, fh, vrel, hpwl, area, proxy, dt, dt, 0.0, 0.0, dt)
        return candidates + [extra]
    except Exception as e:
        if debug:
            sys.stderr.write(f"[wiremask] seed={seed} FAILED: {e!r}\n")
        return candidates


def run_start(seed, P):
    """_v11_run_start plus a chain of strictly-additive portfolio stages.
    Each _apply_* stage is a no-op (returns its input unchanged) when its
    own env var is off, and every stage is called UNCONDITIONALLY so a later
    stage never gets silently skipped by an earlier stage's own gate -- the
    bug behind the QN polish never firing while ELECTRO_RESHAPE_PORTFOLIO sat
    at its default "0" (found + fixed 2026-08-11 via ELECTRO_QN_POLISH_DEBUG=1
    producing zero [qn_polish] lines on a real run, despite the polish flag
    being on)."""
    candidates = _v11_run_start(seed, P)
    candidates = _apply_lp_displacement(candidates, seed, P)
    candidates = _apply_reshape_qinfer(candidates, seed, P)
    candidates = _apply_qn_polish(candidates, seed, P)
    candidates = _apply_reshape_shrink(candidates, seed, P)
    candidates = _apply_wiremask_bbo(candidates, seed, P)
    return candidates


def pool_init(threads=1):
    """Give each worker its share of cores (cores/nproc threads).  Threads are set
    AFTER the fork, so the parent never holds a live OpenMP pool across fork."""
    try:
        import torch
        torch.set_num_threads(max(1, int(threads)))
    except Exception:
        pass


def seed_worker(seed):
    return run_start(seed, WORK)
