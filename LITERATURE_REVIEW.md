# 2025-2026 EDA Floorplanning 文獻回顧與架構建議

搜尋日期：2026-08-11。來源混合 IEEE Xplore / ACM DL / arXiv（真正的 IEEE Xplore
全文檢索需要機構訂閱，這邊用 web search 廣泛涵蓋，每篇都標了發表場合方便你自己去
IEEE Xplore 核實）。目標：找近期（多數 2024-2025，含少數 2026 早期 arXiv）跟
floorplanning / placement 相關、可能對我們的分數或 runtime 有幫助的方法。

**跟 `Papers/` 資料夾的關係**：你們已經讀過 PARSAC、FloorSet、B*-tree+Fast-SA、
一篇 GNN+RL floorplanning（GoodFloorplan）、一篇 Hybrid RL+GA。這份報告**不重複**
這些，只列新方向。

**重要提醒（先講在前面）**：CLAUDE.md 描述的架構是 C++ B*-tree + Fast-SA
（`src/`, `include/`）。但這幾週實際在用、也是這次分析的對象，是 Python 的
`electro/` / `cadc1083` 那條線——連續梯度下降 analytical placement + guillotine
dissection slice-pack + legalize + soft-repair + multi-seed portfolio，跟
CLAUDE.md 寫的完全是兩條不同路線。**這份文獻回顧是針對後者（實際在跑、也是目前
拿到 1.4670 分數的那條）分析的**，如果 B*-tree+SA 那條 C++ 路線還要繼續維護，
架構建議需要另外調整，先跟你確認一下要用哪條當主線。

---

## 目錄

1. [跟我們現有架構直接對得上的技術](#1-跟我們現有架構直接對得上的技術)
2. [GPU / 電磁場式加速 placement](#2-gpu--電磁場式加速-placement)
3. [Diffusion / Flow-matching 生成式佈局](#3-diffusion--flow-matching-生成式佈局)
4. [強化學習 macro placement（含重要的反面證據）](#4-強化學習-macro-placement含重要的反面證據)
5. [GNN 相關](#5-gnn-相關)
6. [Guillotine / 打包理論](#6-guillotine--打包理論)
7. [LLM 用於 EDA](#7-llm-用於-eda)
8. [Chiplet / 3D / 熱感知（目前賽制用不到，僅供參考）](#8-chiplet--3d-熱感知目前賽制用不到僅供參考)
9. [綜合建議：分級行動清單](#9-綜合建議分級行動清單)

---

## 1. 跟我們現有架構直接對得上的技術

這幾篇的技術**跟我們目前 pipeline 的既有設計哲學（strict-additive portfolio，
候選池永遠只加不減）完全吻合**，風險最低、最該優先看。

| 論文 | 場合 | 重點 |
|---|---|---|
| [Macro Placement by Wire-Mask-Guided Black-Box Optimization](https://arxiv.org/abs/2306.16844)（WireMask-BBO）| NeurIPS 2023 | 用「wire-mask」貪婪法評估目標函數，比 RL 更快更準；**關鍵**：可以直接拿「既有的佈局」當初始解去微調，實測最多再省 50% HPWL。這跟我們 `electro_parallel.py` 的候選池機制幾乎是同構的，可以當作另一種候選產生器加進 portfolio。 |
| [A Quasi-Newton-based Floorplanner for fixed-outline floorplanning](https://www.sciencedirect.com/science/article/abs/pii/S0305054821000174) | ScienceDirect | 直接對應 `EVALUATION.md` / `CLAUDE.md` 裡「優先度 #3：SA 收斂後接 Quasi-Newton 精修，topology 固定只調 `(w,h,x,y)`，3-5% 改善、runtime 幾乎不增加」的規劃。這篇證實這個做法在文獻裡是有效、驗證過的，不是我們自己拍腦袋想的。 |
| [Tight Approximation Algorithms for 2D Guillotine Strip Packing](https://arxiv.org/pdf/2202.05989)（ACM TALG 2025）| ACM Transactions on Algorithms 2025 | 理論上界分析，guillotine 切割的填充比極限跟切割策略選擇有關。跟我們 `slice_pack.py` 的 aspect/wall 搜尋（目前只試 0.7/1.0 兩種比例）直接相關，可能可以指導更聰明的切割候選選擇，不用盲目掃描。 |
| [CSF: Fixed-outline Floorplanning Based on the Conjugate Subgradient Algorithm Assisted by Q-Learning](https://arxiv.org/pdf/2504.03796) | arXiv 2025 | 用 subgradient（跟我們的梯度下降同一家族）加一個輕量 Q-learning 來選超參數/方向，不是整個換掉優化器，改動幅度小。 |

---

## 2. GPU / 電磁場式加速 placement

| 論文 | 場合 | 重點 |
|---|---|---|
| [DREAMPlace: Deep Learning Toolkit-Enabled GPU Acceleration for Modern VLSI Placement](https://research.nvidia.com/sites/default/files/pubs/2019-06_DREAMPlace:-Deep-Learning/54_1_Lin_DREAMPLACE.pdf) | DAC 2019（經典，仍是業界標準） | electrostatic-based (ePlace) + FFT + Nesterov's method，用深度學習框架的自動微分/GPU 算子做 placement，百萬級 cell 一分鐘內完成。**這正是你們 memory 裡記錄的「eDensity built+OFF」那個方向**——已經寫好但沒開。 |
| [Accelerating Electrostatics-based Global Placement with Enhanced FFT Computation](https://arxiv.org/pdf/2510.21547) | arXiv 2025 | 針對 DREAMPlace 的 FFT 計算做加速，5.82x FFT 加速、總 runtime 再快 33%、線長還改善 1%。如果要重新評估開啟 eDensity，這篇是最新的調參參考。 |
| [DG-RePlAce: A Dataflow-Driven GPU-Accelerated Analytical Global Placement Framework](https://arxiv.org/pdf/2404.13049) | arXiv 2024 | 針對 ML accelerator 這種規則化 dataflow 電路的 GPU 加速 placement，跟我們的 n≤120 小規模問題關聯度較低，列出來備查。 |

**評估**：我們的問題規模（n≤120）比 DREAMPlace 設計目標（百萬 cell）小三個數量級，GPU
+ FFT 的加速效益在這個規模下不明顯（甚至可能因為 kernel launch overhead 更慢——這點
`electro_optimizer.py` 自己的註解也提到過，GPU 在小問題上實測比 CPU 慢 6 倍）。**這條
不建議追**，除非未來 contest 規模擴大。

---

## 3. Diffusion / Flow-matching 生成式佈局

| 論文 | 場合 | 重點 |
|---|---|---|
| [Chip Placement with Diffusion Models](https://arxiv.org/abs/2407.12282)（ChipDiffusion）| ICML 2025 | 用 diffusion model 在合成資料上訓練，guided sampling 取代 RL，**一步同時擺完所有 macro**（不像 RL 是逐一序列擺放），可以 zero-shot 泛化到新電路。[官方 code](https://github.com/vint-1/chipdiffusion)。 |
| [FlowPlace: Flow Matching for Chip Placement](https://arxiv.org/html/2604.23658v1) | arXiv 2026 | Flow matching（比 diffusion 少步數、不用調 noise schedule）版本的同類方法。 |
| [Physics-Guided Geometric Diffusion for Macro Placement Generation](https://arxiv.org/html/2605.16451) | arXiv 2026 | 把物理約束（重疊、邊界）直接編碼進 diffusion 的幾何結構，理論上比純資料驅動的 diffusion 更容易滿足硬約束——這點對我們（硬約束一堆：fixed-shape、preplaced、MIB、grouping）特別有意義。 |
| [DiffPlace: A Conditional Diffusion Framework for Simultaneous VLSI Placement](https://arxiv.org/pdf/2510.15897) | arXiv 2025 | Conditional diffusion，可以把約束當 condition 餵進去。 |

**評估**：這是**真正的架構級改動**。好處是理論上可以一步產生一個高品質、接近可行的初始
解，取代我們現在「梯度下降 300 iters + legalize + repair」的多階段管線。但代價：
1. 需要用 FloorSet 的 1M 訓練集重新訓練一個 diffusion model（我們的 M1 transformer
   warm-start已經証實訓練 pipeline 不是問題，但 diffusion 訓練通常比 autoregressive
   transformer更貴）
2. 團隊只有 3 人、離 deadline 很近，這種規模的重寫風險高
3. 我們的 M1 warm-start 已經走過一次「autoregressive rollout 訓練到位但實際分數沒有
   反映」的坑（exposure bias），diffusion 不會有 exposure bias 問題（一步生成不是逐步
   累積誤差），這點理論上比 M1 更有優勢，但仍然是全新的訓練+驗證週期

**建議**：不建議現在啟動，但可以記錄成「如果 5/26 deadline 之後還有時間，這是 Phase 3
diffusion post-processing 的具體候選方案」，跟 `EVALUATION.md` Phase 3 的定位一致。

---

## 4. 強化學習 macro placement（含重要的反面證據）

| 論文 | 場合 | 重點 |
|---|---|---|
| [An Updated Assessment of Reinforcement Learning for Macro Placement](https://arxiv.org/pdf/2302.11014)（Cheng, Kahng, Kundu, Wang）| **IEEE TCAD 2025** | ⚠️ **這篇很重要，是反面證據**。這是針對 Google 那篇 *Nature* AlphaChip 論文的獨立覆現研究，結論是：在公平比較下，RL-based placement **並沒有穩定贏過強力的傳統方法（包括 SA/analytical）**，尤其在中小規模問題上優勢不明顯，訓練/推論成本卻高很多。 |
| [LSTM-Characterized Approach for Chip Floorplanning: HyperGCN + DRQN](https://dl.acm.org/doi/10.1109/TCAD.2024.3436013) | IEEE TCAD 44(2), 2025 | 用 HyperGCN 抓約束關係 + DRQN 做序列決策，處理傳統 RL 忽略模組間連動效應的問題。 |
| [RulePlanner: All-in-One RL for Unifying Design Rules in 3D Floorplanning](https://arxiv.org/pdf/2601.22476) | arXiv 2026 | 3D floorplanning，跟我們的 2D 賽制不直接相關。 |

**評估**：**不建議走 RL 這條路**。理由：
1. Cheng/Kahng 這篇權威覆現研究明確指出 RL 在這個問題規模上沒有穩定優勢
2. 我們現有的 multi-seed 梯度下降 + portfolio 方法，本質上已經是一種更便宜、更可控的
   「多起點搜尋」，跟 RL 想解決的問題（探索多個佈局可能性）殊途同歸，但不需要訓練一個
   policy network、不需要處理 RL 訓練不穩定的問題
3. 團隊已經在 M1 warm-start 上驗證過「訓練到位但 rollout 分數不會自動反映」的坑，RL
   的 sim-to-real gap 問題只會更嚴重

---

## 5. GNN 相關

| 論文 | 場合 | 重點 |
|---|---|---|
| [Why are Graph Neural Networks Effective for EDA Problems?](https://dl.acm.org/doi/10.1145/3508352.3561093) | ICCAD 2022（41st）| 理論分析 GNN 為何適合 EDA 問題（圖結構天然對應電路連接關係），適合當入門背景閱讀。 |
| [TransPlace: Transferable Circuit Global Placement via Graph Neural Network](https://arxiv.org/html/2501.05667v1) | arXiv 2025 | 重點是「可遷移」——一個訓練好的 GNN 可以跨電路使用，不用每個電路重新訓練，這點對我們有意義：我們的 M1/floorplan_v2 已經是跨案例通用的模型，這篇的遷移機制可以參考來改善泛化。 |
| [A Physical and Timing Aware Placement Optimization Framework Based on GNN](https://dl.acm.org/doi/10.1145/3676536.3676772) | ICCAD 2024（43rd）| Timing-aware，賽制沒有 timing 約束，關聯度低。 |

**評估**：對應 `CLAUDE.md` 優先度 #4「GNN move proposer」的規劃，但要注意：**這裡指的
是拿 GNN 去引導『搜尋方向/移動提案』，不是像 M1 那樣做 autoregressive 逐步構造**。這
個差異很關鍵——M1 失敗的根因是 autoregressive rollout 的 exposure bias（訓練時 teacher-
forcing 準，推論時累積誤差），而「GNN 提案 + 傳統接受準則做驗證」的架構，每一步都有
真實 cost function 把關，不會有同樣的累積誤差問題，風險比 M1 低。

---

## 6. Guillotine / 打包理論

| 論文 | 場合 | 重點 |
|---|---|---|
| [Tight Approximation Algorithms for Two Dimensional Guillotine Strip Packing](https://arxiv.org/pdf/2202.05989) | ACM TALG 2025 | 見上方第 1 節，guillotine 切割填充比的理論上界。 |
| [Approximation Schemes and Structural Barriers for the Two-Dimensional Knapsack Problem with Rotations](https://arxiv.org/pdf/2603.23970) | arXiv 2026 | 帶旋轉的 2D packing，我們的軟方塊本來就可以自由長寬比，理論上跟「旋轉」的自由度類似，可能有可借鏡的切割策略。 |

**評估**：直接關聯到 `slice_pack.py`。目前的 aspect 搜尋是寫死的 `0.7, 1.0` 兩個值
（見 `electro_optimizer.py` 裡 `ELECTRO_SLICE_ASPECTS` 預設），這些理論結果可能可以
指導一個更精準、更少嘗試次數就能找到高填充比切割的策略，減少目前「試兩個 aspect x
兩個 wall 模式」的暴力搜尋量，**同時省時間、同時可能還更準**——這是少數「加速又加分」
的雙贏方向，值得花時間深入讀一下。

---

## 7. LLM 用於 EDA

| 論文 | 場合 | 重點 |
|---|---|---|
| [Subitizing-Inspired Large Language Models for Floorplanning](https://arxiv.org/pdf/2504.12076) | arXiv 2025（Yuan Ze University）| 微調 LLM（GPT4o-mini）直接做 floorplanning，用「感數」（subitizing，人類不用數數就能瞬間判斷少量物件數量的能力）當靈感。 |
| [Large Language Models (LLMs) for Electronic Design Automation (EDA)](https://arxiv.org/pdf/2508.20030) | arXiv 2025，survey | 全面 EDA + LLM 的綜述。 |
| [See it to Place it: Evolving Macro Placements with Vision-Language Models](https://arxiv.org/html/2603.28733v1) | arXiv 2026 | 用 VLM 對圖片化的佈局做視覺推理來演化佈局解，屬於「LLM 演化既有 analytical placer 程式碼」的思路（EvoPlace/VeoPlace）。 |

**評估**：**不建議**。LLM 推論延遲（就算是小模型）對我們每案 3-4 秒的 budget 而言太
貴，而且比賽的 `RuntimeFactor` 是超過中位數就無上限懲罰，LLM inference 的延遲風險
太高、報酬不確定。

---

## 8. Chiplet / 3D / 熱感知（目前賽制用不到，僅供參考）

FloorSet-Lite 賽制沒有多晶片（multi-die）、3D、或熱感知（thermal）約束，這批論文
（RLPlanner、Floorplet、STAMP-2.5D、ThermoDSE 等）**跟目前題目無關**，僅列出以防
未來賽制擴充：

- [RLPlanner: RL-Based Floorplanning for Chiplets with Fast Thermal Analysis](https://ieeexplore.ieee.org/document/10546812/)（DATE 2024）
- [Floorplet: Performance-aware Floorplan Framework for Chiplet Integration](https://arxiv.org/pdf/2308.01672)（TCAD 2024）
- [Simultaneous Multi-die Floorplanning and Technology Assignment](https://arxiv.org/pdf/2502.10932)（arXiv 2025）

---

## 9. 綜合建議：分級行動清單

按「風險 vs 效益」分三級，配合團隊 3 人、5/26 前 alpha-test 的時間壓力：

### 🟢 Tier 1：低風險、跟現有 portfolio 設計哲學一致，建議優先做

1. **Quasi-Newton 精修 pass**——文獻證實有效（第 1 節），對應 `EVALUATION.md` 既有
   規劃 #3，topology 不變只調連續變數，改動範圍小。
2. **WireMask-BBO 當額外候選產生器**——加進 `electro_parallel.py` 的 portfolio（跟
   `slice_pack` 平行的另一種候選來源），strict-additive，不會讓分數變差。
3. **guillotine aspect 搜尋策略優化**——參考第 6 節的理論上界，把現在暴力掃描
   `0.7/1.0` 兩個值改成有理論依據的選擇，可能同時省時間跟加分。

### 🟡 Tier 2：中風險、值得評估但要先做小規模驗證

4. **GNN 移動提案（非 autoregressive 構造）**——對應 `CLAUDE.md` 優先度 #4，注意
   要避開 M1 犯過的 exposure bias 陷阱：GNN 只提案、真實 cost function 把關每一步。
5. **重新評估 eDensity FFT spreading**（memory 顯示目前是 built+OFF）——參考第 2
   節最新的 FFT 加速論文，看能不能用更好的參數讓它值得開啟。

### 🔴 Tier 3：高風險、需要大量訓練/驗證週期，不建議現階段啟動

6. **Diffusion-based 一步生成式佈局**——理論吸引力最大（第 3 節，尤其
   physics-guided 版本能天生滿足硬約束），但訓練+驗證成本高，建議留到 alpha-test
   之後的 Phase 3。
7. **RL-based placement（AlphaChip 路線）**——有權威覆現論文（第 4 節）指出在這個
   規模上不穩定贏過傳統方法，**不建議投入**。
8. **LLM-based floorplanning**——推論延遲風險對 RuntimeFactor 不利，**不建議**。

---

*搜尋方法論說明：以上使用一般 web search 工具查詢 IEEE Xplore / ACM DL / arXiv 混合
來源，非直接 IEEE Xplore API 全文檢索，論文清單以「近期、跟我們問題規模/約束相關」為
篩選標準，並非窮舉 2025 年所有 EDA/floorplanning 論文。有訂閱權限的話建議自己上
IEEE Xplore 針對第 1、6 節列的關鍵詞再查一輪，可能有更多還沒被一般搜尋引擎索引到的
新論文。*
