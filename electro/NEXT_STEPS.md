# 下一步路線圖：分數再降、時間再壓（2026-07-07）

本文件整理 electro pipeline 的下一步方向。所有建議都建立在**本 repo 已實測驗證的事實**上，
並對照近年 EDA / 演算法 / AI 文獻（連結見文末）。閱讀前提：`EXPERIMENTS.md`（技術嘗試全記錄）、
`CLAUDE.md`（注意其中計分描述是 v9，已過時）。

---

## 0. 現況與已確立的事實

**現況**（真評測器、full-100，2026-07-15 更新）：

| 指標 | 數值 |
|---|---|
| Total Score（v10 加權） | **2.3757**（S1 壓縮 + M1 熱啟動 + 平行化，全部 ON） |
| feasibility | 100/100 |
| runtime | 中位數 **4.2 s**、加總 546 s（平行化前 7.5s/821s；最原始提交 ~3.1s 中位數） |

**本 sprint 分數進程**：2.9660(原始提交) → 2.7215(S1壓縮) → **2.3757**(M1熱啟動，−20% 累計)。
runtime 進程：~3.1s → 6.6s(熱啟動翻倍) → **4.2s**(平行化拉回大部分，仍未回到 3.1s，見 T7)。

**v10 計分要點**（決定所有優先序，勿用 v9 直覺）：
- `quality = 1 + 0.5·(max(0,hpwl_gap) + max(0,area_gap))` — **贏過 baseline 不加分**（clamp）。
- `× exp(2·V_rel)`（boundary/grouping/MIB 軟違規）`× max(0.7, RT^0.3)`。
- `RT = 我方時間 / 全體參賽者該 case 的中位數`：**快最多賺 30%（封頂），慢無上限被罰**。
- Total 以 `e^{n/12}` 加權：n=120 約 34%，n<111 合計仍約 28% — 小 case 不可棄。

**本輪 sprint 實測確立（都有腳本可重跑）**：
1. 排列（相對位置）已達 GT 的 0.74–0.88 一致度（`diag_ml_vs_gt.py`）→ **缺的不是擺哪，是密度**。
2. **area_gap 是主要失血**（util ~0.55–0.6 vs GT ~0.965）；legalizer 只微調（位移 ≤1.2%）。
3. 迴圈內軟塑形（`la` 梯度共優化）**是承重牆但已調滿**：關掉變差（0.63→0.93），放寬 AR_CAP 4→16 也變差（`validate_familyB.sh`）。
4. C++ B\*-tree repack **已否決**：它比 electro 更鬆（area_gap 0.90 vs 0.61）、cost 全輸（`validate_A.sh`）。
5. **壓縮+塑形 pass 有效且嚴格加法**（`shape_compact.py`，額外候選 + cost-aware 排名）：subset 2/6 case 大贏、0 退步；但 **4/6 被排名拒絕**，原因是壓縮破壞 grouping/boundary（V_rel 升）→ 這是下一個最大槓桿（見 S1）。
6. ML 座標回歸有 mode-averaging 天花板（輸出塌成一坨）；僅在 hard-basin case 以 multi-start jitter 形式有窄價值。

**已否決路線**（勿重試，理由見 EXPERIMENTS.md / memory）：
迭代式 RL（GoodFloorplan、HyperGCN+DRQN — 優化被 clamp 的 HPWL、不處理約束群）、
diffusion placer（分鐘級 → RT 罰爆）、C++ repack、AR_CAP>4、單獨的座標回歸 ML。

---

## 1. 分數改善（依投報率排序）

### S1 — 群組/邊界感知壓縮（最高優先，直接接續現有程式）

**問題**：現在的壓縮是「約束盲」的 —— tid40 強制壓縮可把 area_gap 0.96→0.62，但把 cluster
成員拉散、boundary block 拉離牆邊（V_rel 0.24→0.43），`exp(2·V_rel)` 吃掉全部收益，
排名只好拒絕。6 case 中 4 個因此拿不到壓縮紅利。

**做法**（`shape_compact.py` 內改，~100 行）：
- **cluster 當剛體**：同 cluster 的成員合成一個 super-block 一起壓（相對位置不變）→ V_grouping 不可能變差。這是 floorplanning 處理 alignment/abutment 約束的標準做法（constraint-graph 中把群組視為單節點；JigsawPlanner 同精神）。
- **boundary block 釘牆**：left-boundary block 只允許沿 y 壓、x 保持貼 x_min；top-boundary 反向同理。壓縮後 bbox 縮小時，重新 snap 到新邊界。
- MIB 已保護（不塑形），維持。

**預期**：救回被拒的 4/6 型 case。tid40 若 V_rel 保住，cost 2.74→~2.0。full-100 估 2.84 → **~2.6–2.7**。

**✅ 已完成（2026-07-07）**：`shape_compact.py` unit-aware 剛體壓縮 + `compact_variant(aware=)`
修復鏈；關鍵設計 = **plain 與 S1 兩種變體都進候選**，排名逐 case 從 {不壓縮, plain, S1} 擇優
（第一輪只用 S1 時 tid20/80 的舊收益會被弄丟——雙候選解掉）。Subset 採納率 2/6→**4/6**
（S1 救回 tid60 2.402→2.028、tid99 1.890→1.664；plain 保住 tid20/80），平均 2.370→2.194。
**Full-100：2.9660(OFF) → 2.7215(ON)，100/100 feasible** —— 比 plain-only 的 2.8414 再 −0.12，
落在預估區間。殘餘課題：tid40 型（兩變體都被拒）→ 部分剛體 / S2 SDS 最優塑形。

### S2 — SDS 完整版 slack 分配（升級塑形品質）

現在的塑形是貪婪的 fill-right/fill-up。[SDS（ISPD'12/TCAD'13）](https://dl.acm.org/doi/10.1145/2160916.2160956)
給的是**最佳**解法：固定拓樸與外框寬上界，全域分配 slack、只塑形 critical path 上的 soft block，
高度單調下降、收斂到最優。另一個模式值得試：**以 pin-bbox 寬度為固定 W₀**（GT die 外框的強代理），
最小化 H —— 比自由左下壓更貼近 GT 外形。

### S3 — place ↔ compact 迭代迴圈

把壓縮後的緊layout **回灌**當 `place()` 的 init（`init_centers`），target_util 調高再跑短一輪
（200–300 iters）→ 再壓縮。解析式全域重新優化 HPWL/V 項、壓縮收密度，2–3 輪。
這是 multilevel/multi-stage placement（mPL、ePlace 系）的標準策略，且所有零件都已存在。

### S4 — 小 case 精確/組合式打包（n ≤ 40）

文獻明確：[B&B 窮舉在 <30 個矩形有效](https://www.sciencedirect.com/science/article/abs/pii/S0305054806001985)；
[soft rectangle packing 有精確/近似演算法](https://www.researchgate.net/publication/264352300_Exact_and_approximation_algorithms_for_a_soft_rectangle_packing_problem)（面積固定、長寬比連續 —— 正是我們的 soft block）。
v10 下小 case 合計 ~28% 權重，而我們小 case cost 仍在 1.87–2.0。
做法：n≤30–40 時另跑一個 skyline/BLF+B&B 或 soft-packing 候選，與 electro 候選同台由現有排名選優
（**portfolio racing**，嚴格加法，零風險）。目標把小 case 壓向 ~1.2–1.5。

### S5 — 收尾連續精修（quasi-Newton / 直接優化真 cost 代理）

拓樸定案後，對 (x, y, w, h) 做一次連續精修（L-BFGS 或投影梯度），目標函數直接用
`0.5·area_gap_proxy + hpwl_proxy + 2·V_rel_proxy`（可微版），~150 行（CLAUDE.md 原 roadmap 第 3 點，Ji 2021）。
放在壓縮之後，把「幾乎貼齊」的邊精確貼齊。

### S6 — 逐 n 段參數掃描

`TARGET_UTIL / EXT_WL / ITERS / COMPACT_AR` 依 n ∈ {21–40, 41–70, 71–100, 101–120} 分段調
（現在全域一組）。用 `score_subset.sh` 紀律：subset 篩選 → full-100 確認。

### S7 — ML（短期：只做有據的；長期升級見第 5 節）

- 保留現況：ML init 僅配 multi-start（hard-basin 救援）。
- **不要**再投資座標回歸（mode-averaging 天花板已實測）；**不要**上多步 diffusion 全流程（分鐘級 → RT 罰）。
- 「一次輸出就接近答案」的架構升級路線（建構式自回歸 / few-step 生成式 / 離散表示預測），
  完整調查與建議見 **第 5 節 ML 2.0**。

**✅ M1 v1 已完成訓練並驗證（2026-07-14）**：100k case 訓練到 val near=0.758。**但單獨的 M1
rollout 從未贏過排名**（診斷出 exposure bias：teacher forcing 準確率漲了，rollout 品質沒跟著漲）。
**真正拿到分數的是「M1 熱啟動 electro 自己的梯度優化器」**（`place()` 新增 `init_la` 參數，
用 M1 的位置+長寬比當起點，接著跑真正的梯度下降去清 HPWL/V——不受 M1 的逐步累積誤差影響）。
Full-100：2.7215 → **2.3757**（−12.7%）。詳見 `ml/M1_README.md`。

**下一個 ML 槓桿（設計完成，未實作）：scheduled sampling** —— 治本修 exposure bias 本身
（訓練時偶爾餵模型自己的 rollout 當上下文，機率隨 epoch 漸增，不要一開始就 100%——見下方
「為什麼不要一開始就全部用自己的猜測」）。若做成，可能讓 M1 raw rollout 本身也更好，
甚至讓熱啟動可以用更少迭代收斂（跟 T8 疊加）。工程量：`ml/m1_infer.py` 加「吃記憶體模型物件」
建構式、`ml/m1_dataset.py` 加 rollout cache + 機率替換、`ml/m1_train.py` 加每 epoch 前建 cache
的迴圈。成本估計每 epoch 多 10–25 分鐘（重用現有 rollout 邏輯，不是全新架構）。

---

## 2. 運算時間（依投報率排序）

**策略框架**：快的收益封頂在 30%（RT ≤ 0.3 倍中位數即滿），慢的懲罰無上限。
我們無法看到對手中位數 → 合理目標是**穩壓在估計中位數之下**（例如 2–3 s/case），
而不是無限追快。**絕不可用 area_gap 換時間**（0.7 地板讓極端快沒有額外回報）。

### T1 — 早停（最便宜，先做）

`place()` 現在固定 600 iters。加 loss-plateau 偵測（例如連續 50 iters 相對改善 < 1e-4 → 停）。
小 case 遠早於 600 收斂 → 估計小 case 省 30–50% 時間、零品質損失。~15 行。

### T2 — iters 隨 n 排程

`ITERS = f(n)`（例如 300 + 2.5·n），配合 T1 當保險。與 S6 一起掃。

### T3 — 降低 per-iter 開銷（最大單一槓桿）

n≤120 的張量極小，600 iters 的時間主要是 **Python/dispatch 開銷**，不是運算。選項：
- **`torch.compile(mode="reduce-overhead")`**：官方文件明示[小張量、launch-overhead 主導時收益最大](https://docs.nvidia.com/dl-cuda-graph/torch-cuda-graph/best-practices.html)；
  關鍵是 **static shape** —— 把所有 case pad 到固定 N=128，一次編譯、100 case 重用（首次編譯成本攤提掉）。
  這正是 [DREAMPlace](https://research.nvidia.com/publication/2019-06_dreamplace-deep-learning-toolkit-enabled-gpu-acceleration-modern-vlsi-placement)
  「placement=訓練一個網路」的思路（GPU 上 40× vs RePlAce；我們是 CPU 小問題，合理預期 2–5×）。
  注意 requirements.txt 已含 torch，**零新依賴**。
- 退路：把迴圈熱路徑改 numpy 手寫梯度（消 autograd 開銷），或融合現有逐項 loss 成單一 kernel 式表達。

### T4 — 壓縮 pass 向量化

`shape_compact.py` 目前是 O(n²) Python 迴圈（+8% 牆鐘時間）。numpy broadcasting 改寫可到 <1%。
S1 動這個檔時順手做。

### T5 — GPU seed/track-batching（條件性，工程量大）

⚠️ **勿直接把 `place()` 的 `device` 改成 cuda**：程式碼本身有記錄「n≤120 的小張量，600 次序列
迭代在 GPU 上實測比 CPU 慢 6 倍」（kernel-launch 開銷主導，跟資料量無關的固定延遲，序列丟
幾千次 kernel 全部虧在這裡）。GPU 真正能贏，只有**把多個起點疊成同一個張量的 batch 維、一組
迴圈一次 kernel 呼叫處理全部**（DREAMPlace 式）——但這需要把 `place()` 內部所有損失項（線長、
重疊、bbox、群組、邊界⋯）都重寫成支援 batch 維，工程量、風險都遠大於 T7 的 CPU 多進程版。
**目前 T7 已經拿到 33–44% 的實測改善、風險低**，這條先擱置，除非 T7+T8 做完仍覺得不夠快。

### T6 — 先 profile 再動手

動 T3 前先量：600-iter 迴圈 / legalize / repair / 壓縮 各占多少。避免優化錯段（我們在 eDensity
時踩過「優化非瓶頸段」的坑）。

### ✅ T7 — 雙軌 CPU 平行化（已完成，2026-07-14）

`ELECTRO_PARALLEL_TRACKS=1`：隨機起點軌道、M1(+熱啟動)軌道各分一半核心（`os.cpu_count()//2`）
同時跑，常駐 `spawn` Pool（`fork` 在這裡會死鎖 —— contest harness 在呼叫 `solve()` 前已經跑過
torch，parent 不是「OpenMP 乾淨」的，跟舊有 multi-seed 那套的 fork 假設不成立；且 Pool 若每個
case 重建會被 spawn 的直譯器啟動成本吃光利益，必須常駐在 `self._track_pool` 上）。
Full-100 驗證：cost **逐 case 完全相同**（0 個差異，純加速、零分數風險），runtime 加總
821s→546s（**−33%**）、中位數 7.5s→4.2s（**−44%**）。詳見 `electro_parallel.py::track_random/track_m1`。

### T8 — 縮短熱啟動迭代數（下一個最現成的時間槓桿，未做）

T7 平行化後，時間卡在「M1 軌道」——它比隨機起點軌道多做 M1 rollout + 熱啟動 `place()`
（跟隨機起點一樣跑滿 600 次），是兩軌道中較重的那個，平行化的天花板被它卡住。但熱啟動的
起點是 M1 給的、已經合法/排得七七八八的排版，**理論上不需要跑滿 600 次**——砍到
150–250 次做「精修」應該就夠。這樣兩軌道更平衡，平行化天花板會更接近真正的單次 `place()`
時間（更接近原始的 ~3.1s 中位數）。跟 T7 疊加、互不衝突。

---

## 3. 建議執行順序（兩週節奏）

| 週 | 項目 | 驗收 | 狀態 |
|---|---|---|---|
| ~~W1~~ | ~~**S1** 群組/邊界感知壓縮~~ | full-100 < 2.7 | ✅ 完成，2.7215 |
| ~~—~~ | ~~**M1 訓練 + 熱啟動**~~ | full-100 再降 | ✅ 完成，**2.3757** |
| ~~—~~ | ~~**T7** 雙軌 CPU 平行化~~ | runtime −30% 以上、分數不變 | ✅ 完成，−33%/−44% |
| **下一步** | **T8** 縮短熱啟動迭代數 | 平行化天花板更接近 3.1s | 未做，最現成 |
| **下一步** | scheduled sampling | M1 raw 品質提升，可能連動縮短熱啟動 | 已設計，未做 |
| 之後 | S2 SDS 完整塑形 / S4 小 case portfolio | 救 tid40-類殘餘 case | 未做 |
| 之後 | T3 torch.compile / T6 profile | 平均 case < 3 s | 未做 |
| W2 起（平行） | **M1 建構式模仿訓練**（第 5 節；M3 probe 已做並出局） | M1 出首版權重後與 electro 同台當候選 |
| 之後 | S4 小 case portfolio、S5 收尾精修、S6 分段掃描 | 逐項 A/B |

紀律不變：每項先 subset（15 case）廉價驗證 → full-100 確認 → 記入 `EXPERIMENTS.md` → commit。
所有新 pass 一律做成**額外候選**（cost-aware 排名擇優），維持嚴格加法、100/100 feasible。

---

## 4. 文獻對照表

| 主題 | 文獻 | 我們的用法 |
|---|---|---|
| Slack 塑形 | [SDS, ISPD'12](https://dl.acm.org/doi/10.1145/2160916.2160956) / [TCAD'13](https://ieeexplore.ieee.org/document/6416107/) | S2：最優塑形取代貪婪 fill |
| 白縫消除 | [JigsawPlanner, ICCAD'24](https://dl.acm.org/db/conf/iccad/iccad2024.html) | S1：約束感知壓縮的參考 |
| 迴圈內塑形 | [ICCAD'23 靜電法長寬比](https://dl.acm.org/doi/10.1145/3676536.3676818)（同會議系列）/ [PeF, TCAD'22](https://arxiv.org/pdf/2210.03293) | 已實作且已調滿（家族二驗證） |
| 解析式引擎 | [ePlace](https://cseweb.ucsd.edu/~jlu/papers/eplace-dac14/paper.pdf) / [ePlace-MS](https://cseweb.ucsd.edu/~jlu/papers/eplace-ms-tcad14/paper.pdf) / RePlAce | 現有引擎地基 |
| GPU/開銷 | [DREAMPlace](https://research.nvidia.com/publication/2019-06_dreamplace-deep-learning-toolkit-enabled-gpu-acceleration-modern-vlsi-placement) / [torch.compile+CUDA graphs](https://docs.nvidia.com/dl-cuda-graph/torch-cuda-graph/best-practices.html) | T3/T5 |
| 精確打包 | [B&B strip packing](https://www.sciencedirect.com/science/article/abs/pii/S0305054806001985) / [soft rectangle exact](https://www.researchgate.net/publication/264352300_Exact_and_approximation_algorithms_for_a_soft_rectangle_packing_problem) | S4 小 case portfolio |
| 生成式（僅取想法） | [MacroDiff LBR](https://ieeexplore.ieee.org/iel8/11132383/11132091/11132593.pdf) / [chipdiffusion, ICML'25](https://arxiv.org/abs/2407.12282) | S7 不變量表示；全流程已否決 |
| 建構式（僅取想法） | [MdpoPlanner, ASP-DAC'26](https://arxiv.org/abs/2510.15897) 系 / MaskPlace | position/wire-mask 思想（若日後做建構式候選） |

---

## 5. ML 2.0：「一次輸出就接近答案」的架構調查（2026-07-07 補充）

**目標**：ML 前向一次（或極少步）就輸出接近 GT 的合法排版 —— legalizer 只做小幅收尾，
area_gap ≈ 0、V_rel ≈ 0，且推論時間毫秒~次秒級（RT 直奔 0.7 地板）。

### 5.0 為什麼現在的架構到不了這個目標（已實測）

現有 `FloorplanTransformer` 是**逐塊座標回歸**：同一個輸入特徵對應多種正確排列（平移/鏡像/
組合多解），回歸的數學最優解是「取平均」→ 輸出塌成中央一坨（`diag_ml_vs_gt.py` 實測
agreement 僅 0.5–0.75、嚴重重疊）。這是**架構性天花板**，調參/加資料救不了。
要「一次接近答案」，必須換成能表達**多峰分布**或**條件化生成**的架構。以下三條路線都做得到，
依「與目標的貼合度」排序。

### 5.1 路線 M1 — 建構式自回歸 decoder（監督模仿 + 合法性遮罩）⭐ 推薦

**架構**：HGNN/Transformer 編碼 netlist（現有特徵管線可重用）→ 自回歸 decoder 逐塊放置：
每步以「已放好的部分排版」為條件，輸出 (哪一塊, 放哪格, 長寬比 bin)；
position mask 遮掉會重疊/出框的格子（MaskPlace/MdpoPlanner 的表示法），
但**訓練用純監督模仿**——把 1M 筆 FloorSet GT 拆成「部分排版 → 下一塊+位置」的 teacher-forcing
樣本，**完全不需要 RL**（比 MdpoPlanner 的 PPO 簡單一個量級，訓練穩定快速）。

- **為什麼消滅 mode-averaging**：每步條件在部分排版上，條件分布近乎單峰；殘餘多峰由取樣處理。
- **合法 by construction**：mask 保證零重疊、在外框內 → legalizer 真的只剩微調。
- **推論速度**：n≤120 → 至多 120 次小 decoder forward，CPU 次秒、GPU 毫秒級。
- **佐證**：[ChiPFormer（ICML'23）](https://arxiv.org/abs/2306.14744)證明 offline 資料學出的
  placement policy 可遷移到 unseen 電路、比 RL 快 10×（32 電路實驗）；我們條件更好——有 1M 筆
  官方 GT 可直接模仿，連 offline-RL 都不用。[Google 專利線（Mirhoseini 系）](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/11100266)
  同樣是「逐塊生成」框架。
- **要補的功**：soft 長寬比離散化（每塊多預測一個 aspect bin）；grouping/boundary/MIB 作為
  decoder 條件特徵（boundary 塊直接 mask 到對應牆邊格）；放置順序用 GT 幾何序（例如
  左下→右上掃描序）當 canonical 順序。
- **工時**：~2–3 週（資料管線與評測 harness 都已存在）。**風險最低、貼合目標最準。**

**✅ v1 已實作（2026-07-07）**：`ml/m1_{common,model,dataset,train,infer}.py` +
`electro_optimizer._m1_candidate`（`ELECTRO_M1=1`，額外候選、嚴格加法）。管線端到端已
冒煙驗證（1M case 讀取、teacher forcing、rollout 零重疊、爛候選被排名正確拒絕）。
訓練/使用說明：**`ml/M1_README.md`**。剩餘工作 = 規模化訓練（GPU、10 萬 case 級）+
subset/full-100 驗證 + 視結果做 scheduled sampling / 更細格點。

### 5.2 路線 M2 — few-step 生成式（flow matching + consistency 蒸餾）

**架構**：條件式 flow matching 模型（條件=HGNN netlist embedding），一次對整張排版的
(cx, cy, la) 做生成；再用 consistency/流蒸餾壓到 **1–4 步**推論。

- **佐證**：[LayoutFlow（ECCV'24）](https://arxiv.org/abs/2403.18187)證明 flow matching 做
  layout 生成品質持平 diffusion、步數大減（路徑直、不繞噪音軌跡）；
  [Consistency Models（ICML'23）](https://arxiv.org/pdf/2303.01469)及其後續蒸餾工作
  （[綜覽](https://github.com/G-U-N/Awesome-Consistency-Models)）把多步生成壓到 1–4 步而品質可保。
  合法性用 [DiffPlace](https://arxiv.org/abs/2510.15897) 的 constrained-manifold / guided
  sampling 壓重疊（其 50 步版已達 0.00 重疊；蒸餾後需重驗）。
- **相對 M1 的優劣**：一次生成整張（不受逐步誤差累積影響）、天然多峰；但重疊「趨近 0」而非
  「結構性為 0」→ legalizer 負擔比 M1 大一點；且需要位置正規化（平移/鏡像 canonical 化，或
  MacroDiff 的「生成線長關係而非座標」不變量表示——這同時是我們回歸 ML 塌縮病的正解）。
- **工時**：~3–4 週（訓練 + 蒸餾兩階段）。**上限高、研究味濃，蒸餾是新增風險點。**

### 5.3 路線 M3 — 離散表示預測（netlist → B\*-tree/SP token 序列 → 確定性打包）

**架構**：把 GT 排版反推成 B\*-tree（任何緊湊左下排版都有對應樹；GT util 0.965 本身就是
緊湊排版）；訓練 seq2seq/pointer decoder 直接生成樹的 token 序列；確定性 packer µs 級重建排版。

- **優**：合法性與緊密度**免費**（packer 保證零重疊、樹保證左下緊湊）；推論=一次 decoder
  生成 + µs packing，最快。
- **劣/風險**：注意我們否決的是「C++ SA **搜**不到好拓樸」，不是表示法本身——但 ML 直接
  **預測** GT 拓樸是未驗證的賭注：樹 token 序列長（~2n）、一步錯全盤歪（誤差累積比 M1 的
  幾何 mask 更難擋）；soft 長寬比要另外配（樹只定拓樸）；grouping/boundary 也要另掛。

**⛔ PROBE 已做，M3 降級（2026-07-07，`probe_m3_tree.py`，11 case 橫跨 n=21–120）**：
把 GT 反推成 B\*-tree（貪婪最近槽位抽取，best-of-3 插入序）再用標準 contour packer 重建——
即「ML 完美預測出樹」的品質上限——結果：

| 指標（重建/GT） | 平均 | 範圍 |
|---|---|---|
| 面積比 areaR | **1.403**（= 繼承 area_gap ~0.40） | 1.12–1.69 |
| HPWL 比 | 1.212 | 1.04–1.65 |
| 排列一致度 | 0.93（保住了） | 0.87–0.98 |
| 重疊 | 0.00%（合法性確實免費） | — |
| 再接我們的 compact+shape | **1.282** | 1.06–1.54 |

**判決**：合法性免費成立，**緊密度免費不成立** —— GT 是互相咬合的緊密拼磚，**不是**
left/bottom contour packing 可重現的 admissible 排版；就算完美預測 + 接現有壓縮，天花板
仍是 area_gap ~0.28，比 electro+compaction 現況還差，而真實 ML 預測誤差還要疊上去。
（保留的可能性：更聰明的抽取器/邊打包邊塑形或許能壓低，但那是研究題，不是兩週工。）
**M3 出局，資源全數轉 M1** —— M1 直接預測幾何+mask，不受樹 admissibility 束縛。

### 5.4 對照表與建議

| | M1 建構式模仿 ⭐ | M2 few-step 生成式 | M3 離散表示 |
|---|---|---|---|
| 一次輸出接近 GT | ✅（逐步條件化） | ✅（多峰生成） | ✅（若預測準） |
| 零重疊保證 | **✅ by construction** | 趨近 0（guided） | ✅ by construction |
| 推論時間 | 次秒（CPU）/ms（GPU） | 1–4 步，~百 ms | **最快**（µs pack） |
| 用上 1M GT | ✅ 直接模仿 | ✅ 生成目標 | ✅（需先反推樹） |
| 約束群（grp/bnd/MIB） | mask+特徵，最自然 | guidance 項，較弱 | 另掛，最麻煩 |
| 主要風險 | 逐步誤差累積 | 蒸餾後品質/重疊回驗 | 序列誤差放大 |
| 工時 | 2–3 週 | 3–4 週 | 2–3 週(高不確定) |

**建議（probe 後更新）**：主線押 **M1**（最貼「輸出即答案+legalizer 微調+快」的目標、風險最低、
1M 資料直接可用）；M2 當第二押注（若 M1 的逐步誤差在大 n 放大，M2 的整張生成是對沖）；
**M3 已 probe、已出局**（完美預測的天花板 area_gap ~0.40，見 5.3）——這也給 M1 一個具體警示：
M1 的 position mask 打包**不能**用純 left/bottom contour 規則（會踩同一個 admissibility 坑），
要用「GT 幾何格點 + 自由 (x,y) 槽位」的 MaskPlace 式表示，讓 decoder 能重現互相咬合的拼磚。

**與 v10 計分的閉環**：任一路線成功 → area_gap≈0、V_rel≈0 ⇒ quality→1、violation→1；
推論毫秒級 ⇒ RT 大概率踩 0.7 地板 ⇒ **理論單 case cost → ~0.7**（現在 2.84）。
這是本報告所有項目中天花板最高的一項——但也最貴，故排程放 S1/T1 之後、與 W2 平行起跑。

**紀律提醒**（CLAUDE.md）：訓練只用 1M training set，公開 100 題只當 smoke-screen 驗證，
不可對它調參；逆向 FloorSet 生成器是取消資格條款，模仿學習用的是官方提供的 GT 標籤，合規。
