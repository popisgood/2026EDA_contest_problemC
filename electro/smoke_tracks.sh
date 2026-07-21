#!/usr/bin/env bash
# Smoke-test the two-track parallel path end-to-end through the real evaluator.
set -u
EVAL=/home/pop/IntelLabs_Floorset/FloorSet/iccad2026contest
PY=/home/pop/IntelLabs_Floorset/FloorSet/venv/bin/python
cd "$EVAL"
export ELECTRO_M1=1 ELECTRO_M1_WARMSTART=1 ELECTRO_PARALLEL_TRACKS=1
TID="${1:-80}"
"$PY" iccad2026_evaluate.py --evaluate /home/pop/2026_EDA_contest/electro/electro_optimizer.py \
    --test-id "$TID" 2>&1 | grep -iE "\[electro\]|track|error|Trace|deadlock" | head -12
"$PY" - <<'PYEOF'
import json
r = json.load(open("electro_optimizer_results.json"))["test_results"][0]
print("feasible", r["is_feasible"], "| cost", round(r["cost"], 3),
      "| t", round(r["runtime_seconds"], 2), "s")
PYEOF
