#!/usr/bin/env bash
# Definitive full-100 weighted Total Score: current best (M1 off) vs M1 warm-start.
set -u
EVAL=/home/pop/IntelLabs_Floorset/FloorSet/iccad2026contest
PY=/home/pop/IntelLabs_Floorset/FloorSet/venv/bin/python
OPT=/home/pop/2026_EDA_contest/electro/electro_optimizer.py
cd "$EVAL"

echo "[$(date +%T)] OFF full-100 (M1 off, current best) ..."
ELECTRO_M1=0 "$PY" iccad2026_evaluate.py --evaluate "$OPT" >/dev/null 2>&1
cp electro_optimizer_results.json /tmp/ws_off.json

echo "[$(date +%T)] ON  full-100 (M1 + warm-start) ..."
ELECTRO_M1=1 ELECTRO_M1_WARMSTART=1 "$PY" iccad2026_evaluate.py --evaluate "$OPT" >/dev/null 2>&1
cp electro_optimizer_results.json /tmp/ws_on.json

echo "[$(date +%T)] done; weighted Total Score (v10: e^((n-max_n)/12)):"
"$PY" - <<'PY'
import json, math
for tag, fn in (("OFF","/tmp/ws_off.json"), ("ON ","/tmp/ws_on.json")):
    rs = json.load(open(fn))["test_results"]
    mx = max(r["block_count"] for r in rs)
    num = sum(r["cost"]*math.exp((r["block_count"]-mx)/12) for r in rs)
    den = sum(math.exp((r["block_count"]-mx)/12) for r in rs)
    feas = sum(int(r["is_feasible"]) for r in rs)
    rt = sorted(r["runtime_seconds"] for r in rs)
    print(f"  {tag}  Total={num/den:.4f}  feasible={feas}/{len(rs)}  "
          f"rt med={rt[len(rt)//2]:.1f}s max={rt[-1]:.1f}s")
PY
