#!/usr/bin/env bash
# Definitive full-100 weighted Total Score: compaction OFF vs ON.
set -u
EVAL=/home/pop/IntelLabs_Floorset/FloorSet/iccad2026contest
PY=/home/pop/IntelLabs_Floorset/FloorSet/venv/bin/python
OPT=/home/pop/2026_EDA_contest/electro/electro_optimizer.py
cd "$EVAL"

echo "[$(date +%T)] OFF full-100 ..."
ELECTRO_COMPACT=0 "$PY" iccad2026_evaluate.py --evaluate "$OPT" >/dev/null 2>&1
cp electro_optimizer_results.json /tmp/off_results.json

echo "[$(date +%T)] ON  full-100 (ELECTRO_COMPACT=1) ..."
ELECTRO_COMPACT=1 "$PY" iccad2026_evaluate.py --evaluate "$OPT" >/dev/null 2>&1
cp electro_optimizer_results.json /tmp/on_results.json

echo "[$(date +%T)] done; weighted Total Score (v10: e^((n-max_n)/12)):"
"$PY" - <<'PY'
import json, math
for tag, fn in (("OFF","/tmp/off_results.json"), ("ON ","/tmp/on_results.json")):
    rs = json.load(open(fn))["test_results"]
    mx = max(r["block_count"] for r in rs)
    num = sum(r["cost"]*math.exp((r["block_count"]-mx)/12) for r in rs)
    den = sum(math.exp((r["block_count"]-mx)/12) for r in rs)
    feas = sum(int(r["is_feasible"]) for r in rs)
    print(f"  {tag}  Total={num/den:.4f}  feasible={feas}/{len(rs)}")
PY
