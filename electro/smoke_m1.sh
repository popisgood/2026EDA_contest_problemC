#!/usr/bin/env bash
# Smoke-test the M1 candidate path end-to-end through the real evaluator.
set -u
EVAL=/home/pop/IntelLabs_Floorset/FloorSet/iccad2026contest
PY=/home/pop/IntelLabs_Floorset/FloorSet/venv/bin/python
cd "$EVAL"
export ELECTRO_M1=1
export ELECTRO_M1_WEIGHTS="${ELECTRO_M1_WEIGHTS:-/home/pop/2026_EDA_contest/ml/weights/m1_smoke.pt}"
TID="${1:-0}"
"$PY" iccad2026_evaluate.py --evaluate /home/pop/2026_EDA_contest/electro/electro_optimizer.py \
    --test-id "$TID" 2>&1 | grep -iE "\[m1\]|\[electro\]|error|Trace" | head -8
"$PY" - <<'PYEOF'
import json
r = json.load(open("electro_optimizer_results.json"))["test_results"][0]
print("feasible", r["is_feasible"], "| cost", round(r["cost"], 3),
      "| area_gap", round(r["area_gap"], 3), "| t", round(r["runtime_seconds"], 2), "s")
PYEOF
