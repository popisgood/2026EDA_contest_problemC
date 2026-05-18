# Floorplanner 修改說明

針對 ICCAD 2026 Contest C FloorSet-Lite 之 B\*-tree + Fast-SA floorplanner，修正 Case 55 出現的「狹長外框 + 對角階梯堆疊 + 碎裂留白 + 低使用率」問題。

## 問題診斷

原始輸出的 bbox 約為 150×410，方塊呈對角階梯堆疊（下一塊的左下緊貼前一塊的右上），導致：

- 外框極度狹長，aspect ratio 接近 2.7
- 內部留白破碎，繞線通道狹窄
- Boundary 方塊散落，grouping cluster 未聚集
- MIB 方塊出現在外圍

經分析，根本原因有四：

1. **SA cost 無 aspect-ratio 梯度**：150×410（=61500）與 250×250（=62500）面積幾乎相同，cost function 無法區分。
2. **w_area 過弱**：原值 1.0 對比 w_group / w_mib / w_bound = 80，相對影響微弱。
3. **初始樹不平衡**：`make_initial` 僅平衡 `n_lc` / `n_rc` 數量，未考慮子樹的幾何延伸範圍，仍會產生高瘦的左鏈。
4. **FixBoundary tactic 2 強制接受**：`always_accept = true` 將 boundary 違規塊拖至遠離原點的 anchor，正是產生階梯的元凶。
5. **Contour packer 不會回填**：拓樸層級的 move 無法消除天然產生的階梯空洞，需要顯式 compaction。

## 修改內容

四項邏輯獨立的修改，可單獨還原其中任一項。

### 1. `cost.hpp` / `cost.cpp` — 新增 aspect 與 density 懲罰項

於 `Costs` struct 新增：

```cpp
double bbox_aspect_excess;   // max(W/H, H/W) - 1.0
double dead_fraction;        // 1 - sum(w_i*h_i) / (W*H)
```

於 `SAWeights` 新增權重並調升 `w_area`：

```cpp
double w_aspect       = 8.0;
double w_density      = 3.0;
double target_aspect  = 1.10;   // 允許 ±10% 偏差不罰
double w_area         = 5.0;    // 原為 1.0
```

`sa_cost()` 增加項：

```cpp
w_aspect * std::max(0.0, aspect_excess - (target_aspect - 1.0))
+ w_density * dead_fraction
```

讓 SA 對「方正度」與「內部死區」有直接梯度，不再只看 bbox 面積。

### 2. `parallel.cpp` — 幾何平衡的初始樹

重寫 `make_initial` 中的 `insert_under` lambda：

- 為每個節點追蹤 `ext_w[i]` / `ext_h[i]`（spine 延伸量）
- 兩個子節點皆空時，選擇當前 root extent 較小的軸（朝正方形目標靠近）
- 兩側皆有子節點時，遞迴進入「子樹較矮」的那一側，而非隨機
- 每次插入後回溯更新 extent

舊版只平衡子節點數量，仍會產生高瘦的左鏈；新版在初始狀態即同時平衡兩軸的幾何延伸。

### 3. `packer.cpp` — 新增 `compact_left_down` 後處理

於 `Packer::pack()` 結尾、overlap check 之前呼叫的靜態函式：

- 對每個非 preplaced 方塊，盡可能向左、向下滑動，直到觸碰其他方塊或 (0,0)
- 兩階段 × 三輪迭代：先以 y 排序壓縮 y、再以 x 排序壓縮 x
- 完成後重新計算 bbox

Contour packer 天生會留下無法靠拓樸 move 修補的階梯空洞，這步用幾何方式直接回填。

> **註**：第一版實作有保留 E_TOP / E_RIGHT / corner-pinned 方塊不壓縮，但在 80-block 測試中反而 regression。最終版本對所有非 preplaced 方塊一視同仁地壓縮，並讓 SA 的 `w_bound` 透過 FixBoundary moves 來驅動 constraint 滿足。

### 4. `moves.cpp` — 移除 FixBoundary 階梯元凶

`apply_fixb` tactic 2（將 violator 移為 edge anchor 的子節點）：

```cpp
// m.always_accept = true;   // 已移除
```

這個 always_accept 會無視 cost 強制接受 move，把 boundary 違規塊拉到遠處 anchor 之下，產生連鎖階梯。移除後讓 SA 用 cost 篩選。

Tactic 1（與邊上非 constrained 塊 swap）保留 `always_accept`，因為它直接滿足 constraint 且不會產生幾何副作用。

## 驗證結果

在三個合成 benchmark 上比較舊版（OLD）與新版（NEW），8 threads、5–8s budget、seed 42：

| Benchmark | Aspect 舊 → 新 | Utilization 舊 → 新 | Contest cost 舊 → 新 |
|---|---|---|---|
| 20 blocks | 3.99 → **1.02** | 71.8% → **84.8%** | 1.17 → **0.99** |
| 30 blocks | 2.66 → **1.09** | 50.6% → **68.9%** | 1.78 → **0.79** (−56%) |
| 80 blocks（仿 Case 55 規模） | 1.87 → **1.02** | 27.9% → **48.3%** | 3.71 → **1.89** (−49%) |

並驗證新舊兩版輸出皆無 overlap pair。視覺化確認新版：bbox 趨近正方形、緊密堆疊、preplaced 方塊位於 anchor、boundary 方塊靠邊角、grouping cluster 聚集、MIB 方塊內聚。

## 檔案清單

修改的檔案：

- `cost.hpp`
- `cost.cpp`
- `parallel.cpp`
- `packer.cpp`
- `moves.cpp`

未修改的檔案：`btree.{hpp,cpp}`、`moves.hpp`、`sa.{hpp,cpp}`、`parser.{hpp,cpp}`、`types.hpp`、`main.cpp`。

## 調參建議

四項修改邏輯獨立，任一項在 full validation set 上 regression 時可單獨還原。最可能需要微調的是 `cost.hpp` 內的 aspect 相關參數：

- 若某些 testcase 因 preplaced 寬塊而合理地需要非正方形外框 → 將 `target_aspect` 調高至 1.3–1.5，或將 `w_aspect` 降至 3–4
- 若某些 testcase 仍出現碎裂留白 → 將 `w_density` 從 3 提高至 5–8
- 若整體偏離 boundary constraint → 確認 `w_bound` 是否仍為 80，視情況提高
