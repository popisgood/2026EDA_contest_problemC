#!/usr/bin/env python3
"""Contest entry point: analytical placement + legalization + soft-constraint repair.

Per case:
    analytical global placement  (WL + spreading + grouping + boundary terms)
      -> legalize to exactly zero overlap
      -> grouping_repair  (abut isolated cluster members)
      -> boundary_snap    (slide boundary blocks onto bbox edges)
      -> (x, y, w, h)

Hard constraints by construction: no overlap, soft-block area exact, MIB same
shape, fixed dims locked, preplaced pinned.  Soft constraints reduced by the
analytical penalties + the two repair passes.
"""
from __future__ import annotations

import os
import sys
import time
from typing import List, Optional, Tuple

# --- SUBMISSION DEFAULTS: Electro / Placement Configuration ---
# Setting default values for all ELECTRO_ environment variables used in op_wrapper and submodules
os.environ.setdefault("ELECTRO_CLAMP", "1")
os.environ.setdefault("ELECTRO_NONNEG", "1")
os.environ.setdefault("ELECTRO_SEEDS", "4")
os.environ.setdefault("ELECTRO_PARALLEL", "1")
os.environ.setdefault("ELECTRO_ML_INIT", "1")
os.environ.setdefault("ELECTRO_ITERS", "300")
os.environ.setdefault("ELECTRO_PLACE_COMPACT", "1")
os.environ.setdefault("ELECTRO_PLACE_COMPACT_TOPK", "3")
os.environ.setdefault("ELECTRO_MIB_PORTFOLIO", "1")
os.environ.setdefault("ELECTRO_FAST_CLEANUP", "1")
os.environ.setdefault("ELECTRO_SLICE_ALIGN_PORTFOLIO", "1")
os.environ.setdefault("ELECTRO_DEVICE", "cpu")
os.environ.setdefault("ELECTRO_LR", "0.02")
os.environ.setdefault("ELECTRO_REPAIR_ROUNDS", "3")

electro_vars = {k: os.environ[k] for k in sorted(os.environ.keys()) if k.startswith("ELECTRO_")}
print("[ALL ELECTRO VARS]", file=sys.stderr)
for _k, _v in electro_vars.items():
    print(f"  {_k} = {_v}", file=sys.stderr)

# --- 執行緒必須在 import torch/numpy **之前**就定死 ------------------------
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import torch

torch.set_num_threads(1)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import multiprocessing as mp

from iccad2026_evaluate import FloorplanOptimizer
from legalize import verify_overlap
import electro_parallel
from electro_parallel import _edges_np


class MyOptimizer(FloorplanOptimizer):
    def __init__(self, verbose: bool = False):
        super().__init__(verbose=verbose)
        self.iters = int(os.environ.get("ELECTRO_ITERS", "600"))
        # CPU by default.  This is a SMALL problem (n<=120) run for 600 sequential
        # iterations of tiny ops, so a GPU is ~6x SLOWER here (kernel-launch
        # overhead dominates) -- and on a laptop it would run on the display GPU
        # and freeze the screen.  GPU only pays off with seed-BATCHING (TODO);
        # opt in then with ELECTRO_DEVICE=cuda.
        self.device = os.environ.get("ELECTRO_DEVICE", "cpu")
        self.lr = float(os.environ.get("ELECTRO_LR", "0.02"))
        # Rounds of (grouping_repair -> boundary_snap).  Each repair is now
        # min-displacement, but the two fight over blocks that are both a cluster
        # member AND a boundary block; one round leaves boundary blocks freshly
        # snapped off their cluster.  Iterating lets them settle: full-100 score
        # 3.733 (1 round) -> 3.568 (2) -> 3.545 (3) -> 3.545 (4, saturated).
        self.repair_rounds = int(os.environ.get("ELECTRO_REPAIR_ROUNDS", "3"))
        # Multi-start: keep the best of N seeds，用平行度換品質。
        #
        # 在 100 案實測（純品質分數，RuntimeFactor=1.0）：
        #     seeds=1 -> 2.4295 (1.35s)   seeds=2 -> 2.1905 (3.51s)
        #     seeds=4 -> 2.0929 (8.6s)    seeds=8 -> 1.9702 (15.0s)
        # 全部 100/100 feasible。實測數字（boundary/grouping 違規、迭代數、
        # ext_wl：已經驗證預設值是局部最佳，多起點是目前唯一還能大幅推進的機制。
        #
        # 預設改成 8 是為了「品質分數 < 2.0」這個目標。**但要注意 runtime 換算**：
        # 比賽的 RuntimeFactor = (你的耗時 / 全場中位數)^0.3，慢那一側沒有上限。
        # 15s 相對於 1.35s 是 11 倍，換算成分數沒罰約 1.11^... 實際是 11^0.3 ≈ 2.03 倍
        # ——如果全場中位數在 1 秒左右，seeds=8 的「比賽分數」總體效果比 seeds=1 差。
        # 也就是說：
        #   * 只看本機品質分數（本次預設）-> seeds=8
        #   * 要看真實比賽總分 -> 務必確認全場中位數，適時調回 1~2
        self.seeds = int(os.environ.get("ELECTRO_SEEDS", "8"))
        # 多起點的 seed 平行化（fork worker）。**預設開啟**。
        #
        # 沿用預設關掉，因為狀態的實體是「每個 case 都新建一次 Pool」，執行程碼
        # 帶 torch（100 個 case x N 個 seed 就是數百次重複 fork），實測每循環間
        # 換得多（seeds=8 跟 20 秒間換算不上第一等）。現在改成「整份試測層用一
        # 個常駐 pool」（見 _get_pool），fork 只發生一次，per-case 只需 pickle
        # 幾個小張量（n<=120，數量級數百），成本可忽略。
        #
        # worker 不會限定執行緒（ELECTRO_WORKER_THREADS，預設 1），這個問題規模
        # 很小（intra-op 平行沒有東西可賺，反而是 N 個 worker x M 條執行緒互相搶
        # 而彼此打架），真正的平行度來自「seed 之間互相獨立」這件事。
        self.parallel = os.environ.get("ELECTRO_PARALLEL", "1") == "1"
        self._pool = None
        # ML warm-start: use the trained FloorplanTransformer's predicted block
        # centers as the analytical placer's init (instead of random).  Lazily
        # loaded; falls back to random init if weights/model are unavailable.
        self.ml_init = os.environ.get("ELECTRO_ML_INIT", "1") == "1"
        self._predictor = None

    def solve(
        self,
        block_count: int,
        area_targets: torch.Tensor,
        b2b_connectivity: torch.Tensor,
        p2b_connectivity: torch.Tensor,
        pins_pos: torch.Tensor,
        constraints: torch.Tensor,
        target_positions: Optional[torch.Tensor] = None,
    ) -> List[Tuple[float, float, float, float]]:
        if block_count == 0:
            return []
        t0 = time.time()

        cons = constraints[:block_count].cpu().numpy()
        is_pre = (cons[:, 1] != 0).astype(bool)
        mib_id = cons[:, 2].astype(int) if cons.shape[1] > 2 else np.zeros(block_count, int)
        clust_id = cons[:, 3].astype(int) if cons.shape[1] > 3 else np.zeros(block_count, int)
        bcode = cons[:, 4].astype(int) if cons.shape[1] > 4 else np.zeros(block_count, int)
        eb, ep, pv = _edges_np(b2b_connectivity, p2b_connectivity, pins_pos, block_count)

        # ML warm-start only helps WITH multi-start (jitter around the prediction);
        # a single pure-ML start is worse than a single random start, so for
        # seeds==1 we use random init.
        use_ml = self.ml_init and self.seeds > 1
        init_centers = self._ml_centers(
            block_count, area_targets, constraints, target_positions,
            b2b_connectivity, p2b_connectivity, pins_pos) if use_ml else None

        nseeds = max(1, self.seeds)
        if os.environ.get("ELECTRO_ADAPTIVE_SEEDS", "0") == "1":
            if block_count >= 96:
                nseeds = int(os.environ.get("ELECTRO_LARGE_SEEDS", "24"))
            elif block_count <= 45:
                nseeds = int(os.environ.get("ELECTRO_SMALL_SEEDS", "12"))
        P = {
            "n": block_count, "area": area_targets, "b2b": b2b_connectivity,
            "p2b": p2b_connectivity, "pins": pins_pos, "cons": constraints,
            "tp": target_positions, "iters": self.iters, "lr": self.lr,
            "device": self.device, "init": init_centers, "is_pre": is_pre,
            "clust_id": clust_id, "bcode": bcode, "rounds": self.repair_rounds,
            "nonneg": os.environ.get("ELECTRO_NONNEG", "0") == "1",
            # 最長路徑硬可行性（legalize 之後、軟約束修復之前）。實測填充比只有
            # 0.660 而 ground truth 是 0.966，缺口在整片留白——因為原本的管線
            # 只有從左到右推去消除重疊，沒有從右往左收去填空間。ELECTRO_COMPACT=0 可
            # 選擇關掉以做 A/B。
            "compact": os.environ.get("ELECTRO_COMPACT", "1") == "1",
            "compact_rounds": int(os.environ.get("ELECTRO_COMPACT_ROUNDS", "4")),
            # 切割式分割成功（slice_pack），每個 seed 額外產生一份候選，交給
            # 相同的修復鏈跟排名邏輯比較。這樣保底分數不可能退步。
            "slice": os.environ.get("ELECTRO_SLICE", "1") == "1",
            "areas_np": area_targets[:block_count].cpu().numpy().astype(float),
            "is_fixed": (cons[:, 0] != 0).astype(bool),
            # 切割式的輸出本來就是緊的，而硬可行性把所有方塊都往左下移——正好破壞
            # _place_leaf 依 bcode 貼齊到位的結果。預設不開硬可行性。
            "slice_compact": os.environ.get("ELECTRO_SLICE_COMPACT", "0") == "1",
            # 只給 slice_pack 的 _unify_mib 事後統一形狀用（不影響排序）。
            "mib_id": mib_id,
            # 給 worker 做排名用的連線資料（見 electro_parallel._prerank）。
            "eb": eb, "ep": ep, "pv": pv,
            # 每個 seed 最多再多幾份切割候選進入修復（前排名後只取最好的 K 份）。
            # 這**不是**能省算力——profile 顯示修復只佔單個 seed 約 5%。它的作用
            # 是讓「增加候選維度」的邊際成本接近零。
            "slice_topk": int(os.environ.get("ELECTRO_SLICE_TOPK", "2")),
            # 多種長寬比的候選外框（相對於種子佈局的比例）。最佳化的**選項不多**
            # 差距也大——實測案例 89 在 1.0 得到 1.9159，在 0.7 直接切割失敗；
            # 案例 96 反過來（在 0.7 得到 2.1816，填充比 0.880），在 1.4 失敗。
            # 切割相對於下游的解析式佈局幾乎免費，所以全部跑不選擇交給後面比較。
            # 預設兩個比例（第一個比例 1.45）在全 100 案只差 0.0044 分，卻要
            # 多付 5% 的時間（seeds=8）3 比例 1.5092@3.31s vs 2 比例 1.5048@3.15s）。
            "slice_aspects": tuple(
                float(v) for v in
                os.environ.get("ELECTRO_SLICE_ASPECTS",
                               "0.7,1.0").split(",") if v),
            # 邊界帶策略（開/關）。邊界戰略純粹只有作用在邊界的群組的邊界上，不
            # 「全都要求同一面的」的判斷方向走平行於該切割軸的做法。
            "slice_walls": tuple(
                v == "1" for v in
                os.environ.get("ELECTRO_SLICE_WALLS", "0,1").split(",") if v),
        }

        # Multi-start: each seed lands in a different basin; run them in parallel
        # processes (independent -> embarrassingly parallel) and keep the
        # lowest-cost-proxy result.  Each worker is single-threaded by default:
        # the parent runs the ML model (initialising the OpenMP pool), so forked
        # workers that spin up >1 thread can deadlock (libgomp fork hazard).  On a
        # 48-core box this is fine -- run MANY single-thread seeds in parallel at
        # the same wall-clock.  ELECTRO_WORKER_THREADS>1 opts into multi-thread
        # workers (only safe if the parent never touched OpenMP, e.g. ML_INIT=0).
        # The solver can't see the GT baseline, so we rank by
        # exp(2*V_rel)*(hpwl/mean + area/mean), mirroring contest cost.  CUDA can't
        # be forked, so on GPU we run seeds sequentially (fast anyway).
        starts = None
        pool_wall = 0.0
        if self.parallel and nseeds > 1 and self.device == "cpu":
            try:
                # 常駐 pool + 選擇 pickle 參數。整份試測層用一份 pool 並 fork
                # 繼承 WORK 來避免 pickle 大張量——真正的 pickle 只有一份小
                # fork 成本（實測前面循環換好幾份）。
                pool = self._get_pool(nseeds)
                tp0 = time.time()
                starts = pool.starmap(electro_parallel.run_start,
                                      [(s, P) for s in range(nseeds)])
                pool_wall = time.time() - tp0
            except Exception as e:
                sys.stderr.write(f"[electro] parallel failed ({e}); sequential\n")
                self._close_pool()
                starts = None
        if starts is None:
            starts = [electro_parallel.run_start(s, P) for s in range(nseeds)]

        # run_start 回來的候選帶著已算好的 (vrel, hpwl, area)。**最貴的部分在
        # worker 裡平行掉**（soft_violation_counts 是 O(n^2) 的 Python 迴圈，
        # seeds x 候選數份結果若在總控程串起，就是 seeds=8 的 wall time 比
        # 單 seed 多出來的主因），**最後的算術在這裡**，因為它只有 O(1)，而
        # 錨點必須是同一致的——worker 自己算自己那個 seed 的錨點不可比。
        # 索引 0 是梯度式解析式路徑，其餘是切割式分割（origin 用來診斷，切割式到底
        # 有沒有被選中）。
        origin = [("slice" if j else "push") for s in starts
                  for j, _ in enumerate(s)]
        cands = [c for s in starts for c in s]

        # 最後排名必須跟候選池「無關」，原本用候選池的平均
        # （mean over cands），那會讓**多加一份候選就改變既有候選的相對排名**
        # ——候選數超過 7 之後這個缺陷就會人現，案例 96 加入更好的候選之後，
        # 排名反而選出更差的解（2.1816 -> 2.3145）。
        #
        # 改用跟候選池無關的錨點：
        #   * 面積：方塊總面積 / 0.966。0.966 是 density_probe 量到的 ground
        #     truth 的包夾平均填充，所以這個值就是 area_baseline 的良好估計。
        #   * 線長：seed 0 的梯度式解析式候選（cands[0]）。它永遠存在、永遠是同一份，
        #     不隨其他候選增減而變。改用直尺度常數當錨點試過，因為改變了線長跟
        #     面積的相對比重而總分數變好從 1.4972 進到 1.5095。
        ha = cands[0][5] or 1.0
        aa = float(P["areas_np"].sum()) / 0.966 or 1.0
        # ELECTRO_PROXY_ABSHPWL=1：HPWL 錨點改用「方塊總面積開根號」這個純尺度
        # 常數，而不是 seed 0 的候選 HPWL。cands[0] 雖然永遠存在、不隨候選池增減
        # 而變（已經比候選池平均好很多），但它畢竟是「某一個候選的實測值」——
        # 換一組 seed 或改任何影響 seed 0 的設定，整個排名的相對權重就跟著漂移。
        # 我們自己路線 §8.35 驗證過完全 pool 無關的 proxy 在候選池豐富時有效。
        if os.environ.get("ELECTRO_PROXY_ABSHPWL", "0") == "1":
            ha = float(np.sqrt(P["areas_np"].sum())) or 1.0
        best = int(np.argmin([np.exp(2.0 * c[4]) * (c[5] / ha + c[6] / aa)
                              for c in cands]))
        x, y, w, h, vrel, hpwl, area = cands[best][:7]

        ov = verify_overlap(x, y, w, h)
        soft = ((cons[:, 0] == 0) & (cons[:, 1] == 0))
        at = area_targets[:block_count].cpu().numpy()
        drift = np.abs(w * h - at) / np.maximum(at, 1e-9)
        max_drift = float(drift[soft].max()) if soft.any() else 0.0
        dt = time.time() - t0
        sys.stderr.write(
            f"[electro] n={block_count} t={dt:.3f}s seeds={self.seeds} "
            f"resid_overlap={ov:.3g} max_area_drift={max_drift:.4f} "
            f"V_rel={vrel:.3f} pick={origin[best]} "
            f"nslice={sum(1 for o in origin if o == 'slice')}/{len(origin)} "
            f"util={float((w * h).sum()) / max(area, 1e-9):.3f} "
            # 平行效率：pool_wall 是 starmap 的實際 wall time，wmax 是最慢的
            # worker 自報的耗費。理想應該兩者相等（worker 完全平行，pool 沒有額外
            # 成本）；pool_wall 明顯大於 wmax 就表示時間花在 pickle，成本花在不是
            # 真正的計算上。wplace 是其中解析式佈局佔的部分。
            f"pool_wall={pool_wall:.3f} "
            f"wmax={max((c[8] for c in cands), default=0.0):.3f} "
            f"place={max((c[9] for c in cands), default=0.0):.3f} "
            f"legal={max((c[10] for c in cands), default=0.0):.3f} "
            f"slice={max((c[11] for c in cands), default=0.0):.3f} "
            f"finish={max((c[12] for c in cands), default=0.0):.3f}\n"
        )
        return [(float(x[i]), float(y[i]), float(w[i]), float(h[i]))
                for i in range(block_count)]

    def _get_pool(self, nseeds):
        """整份試測層共用 fork pool，只在第一次真正需要時建立。

        為什麼延遲到這裡而不是 __init__（fork 的當下，執行緒不能正確於 OpenMP
        的平行初始起，libgomp 的 fork 危害——子行程有繼承到一個永遠不能被解開
        的執行緒池而卡死）。第一次呼叫 solve() 時執行程才剛匯入，還沒
        跑任何 torch 計算，是最安全的 fork 時機，而 worker 不會設定執行
        緒，之後就依此不需要 fork。
        """
        if self._pool is None:
            nproc = max(1, min(nseeds, os.cpu_count() or 1))
            threads = int(os.environ.get("ELECTRO_WORKER_THREADS", "1"))
            ctx = mp.get_context("fork")
            self._pool = ctx.Pool(nproc, initializer=electro_parallel.pool_init,
                                  initargs=(threads,))
        return self._pool

    def _close_pool(self):
        """平行路徑出錯或收尾時關掉 pool，換得案例乾淨地退回循環執行。"""
        if self._pool is not None:
            try:
                self._pool.terminate()
                self._pool.join()
            except Exception:
                pass
            self._pool = None

    def __del__(self):
        self._close_pool()

    def _ml_centers(self, block_count, area_targets, constraints, target_positions,
                    b2b, p2b, pins):
        """Predicted block centers [N,2] (raw coords) from the trained model, or
        None if the model/weights are unavailable or the case is too big."""
        if self._predictor is None:
            try:
                # Find the `ml/` package: explicit env override, then bundled next
                # to this file (submission layout), then the PARENT directory (the
                # dev-tree layout, where ml/ sits beside electro/).  All paths are
                # relative to __file__ -- no machine-specific absolute paths.
                here = os.path.dirname(os.path.abspath(__file__))
                ml_dir = None
                for d in (os.environ.get("ELECTRO_ML_DIR"), here,
                          os.path.dirname(here)):
                    if d and os.path.isdir(os.path.join(d, "ml")):
                        ml_dir = d
                        break
                if ml_dir is None:
                    raise FileNotFoundError("ml/ package not found")
                if ml_dir not in sys.path:
                    sys.path.insert(0, ml_dir)
                from ml.predict import Predictor
                wts = os.environ.get(
                    "ELECTRO_ML_WEIGHTS",
                    os.path.join(ml_dir, "ml", "weights", "floorplan_v2.pt"))
                self._predictor = Predictor(wts, device=self.device)
            except Exception as e:
                sys.stderr.write(f"[electro] ML init unavailable: {e}\n")
                self._predictor = False
        if not self._predictor:
            return None
        try:
            pred = self._predictor.predict(
                block_count, area_targets, constraints, target_positions,
                b2b, p2b, pins)
            if pred is None:
                return None
            return torch.tensor([[p[0], p[1]] for p in pred.positions],
                                dtype=torch.float32)
        except Exception as e:
            sys.stderr.write(f"[electro] ML predict failed: {e}\n")
            return None
