# ICCAD 2026 Contest C — electro_v19 (目前 REAL Total Score 最佳版本)

## 這是什麼

這是我們 electro pipeline 目前驗證過、REAL Total Score（含 runtime 因子）
最好的版本。**沒有開任何實驗性 flag**——如果你看到別的版本或別人講的
`ELECTRO_SEEDS=8`、`ELECTRO_RESHAPE_PORTFOLIO=1` 之類的環境變數，那些是
還在驗證中或已知效果不明的東西，這個版本全部保持預設關閉/正確值。

已知成績（100 案驗證，同一個 WSL session 測的，公平比較）：
- Neutral Total Score: **1.3776**
- REAL Total Score（含 RT 因子）: **0.9801**
- 100/100 合法，99/100 快過中位數

## 檔案結構

```
electro_v19_for_friend/
├── electro_optimizer.py   <- 真正的比賽入口（FloorplanOptimizer 子類別）
├── electro_parallel.py    <- multi-seed 平行執行 + portfolio 排名
├── analytical_place.py    <- 電靜力式全域佈局（PyTorch/Adam）
├── legalize.py             <- constraint-graph 合法化
├── lp_legalize.py          <- LP/Adam 額外候選（含合法化階段長寬比彈性，預設關閉）
├── slice_pack.py           <- 切割式打包
├── soft_repair.py          <- grouping/boundary 軟約束修復
├── dirichlet_init.py       <- Dirichlet 調和延拓初始化
├── cluster_virtualize.py   <- 切割式候選的群組虛擬化
└── ml/
    ├── predict.py, model.py, data.py, __init__.py
    └── weights/floorplan_v2.pt   <- ML 暖啟動權重（~2.2MB）
```

## 在 WSL 上跑（這份是給 WSL 用的）

**為什麼一定要 WSL，不能用原生 Windows**：`ELECTRO_SEEDS>1`（預設是 4）
需要 `fork()`，原生 Windows 沒有這個系統呼叫，會靜默退化成序列執行（結果
一樣，但速度慢、且測出來的 runtime 沒有意義）。

```bash
# 1. 把資料夾放進 WSL 檔案系統內（不要留在 /mnt/c/... 跑，NTFS 掛載點
#    每次檔案存取都很慢，會拖慢 pool fork/pickle）
cp -r /mnt/c/Users/<你的帳號>/Downloads/electro_v19_for_friend ~/electro_v19_for_friend
cd ~/electro_v19_for_friend

# 2. 安裝依賴（如果還沒裝過 torch/numpy）
pip install -r requirements.txt

# 3. 這份資料夾**沒有附官方框架檔案**（iccad2026_evaluate.py 本身還依賴
#    litetestLoader.py 等其他官方同層檔案，只複製單一檔案會壞掉）。你需要
#    自己電腦上已經有完整的官方 iccad2026contest 資料夾（比賽官方發的那份），
#    把它加進 PYTHONPATH：
export PYTHONPATH=/path/to/ICCAD-C-FloorSet-official/iccad2026contest

# 4. 用官方評測框架跑（--evaluate 是官方模式，直接呼叫 electro_optimizer.py
#    裡的 MyOptimizer.solve()，對照驗證集打分數）
python3 -m iccad2026_evaluate --module electro_optimizer --evaluate \
    --val /path/to/ICCAD-C-FloorSet-official/LiteTensorDataTest
```

如果你已經有自己一套跑分/評測腳本（例如 `ml.case_report_electro` 那種），
直接把這份資料夾當 `electro_v19` 目錄接進去用就好，介面完全一樣
（`electro_optimizer.MyOptimizer`，`solve()` 簽名照官方框架）。

## 重要：不要調這些環境變數

`electro_optimizer.py` 開頭已經把所有正確的生產預設值寫死成
`os.environ.setdefault(...)`（`ELECTRO_SEEDS=4`、`ELECTRO_PARALLEL=1`、
`ELECTRO_ML_INIT=1` 等等）。**不要手動設成別的值**，除非你知道自己在
做什麼——尤其不要學到 `ELECTRO_SEEDS=8`，那個設定是為了本機忽略 runtime
的品質分數，會拖累真正比賽的 REAL Total Score。
