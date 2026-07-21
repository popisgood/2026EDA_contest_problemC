#!/usr/bin/env bash
# Smoke-test the persistent-pool parallel path across MULTIPLE cases in ONE
# process invocation (matches how the real full-100 run works: one optimizer
# instance, solve() called repeatedly -> the pool amortizes across cases).
set -u
EVAL=/home/pop/IntelLabs_Floorset/FloorSet/iccad2026contest
PY=/home/pop/IntelLabs_Floorset/FloorSet/venv/bin/python
cd "$EVAL"
export ELECTRO_M1=1 ELECTRO_M1_WARMSTART=1 ELECTRO_PARALLEL_TRACKS=1
TIDS="${TIDS:-0 20 40 60 80 99}"
"$PY" iccad2026_evaluate.py --evaluate /home/pop/2026_EDA_contest/electro/electro_optimizer.py \
    --test-id $TIDS 2>&1 | grep -iE "\[electro\]|error|Trace|deadlock"
"$PY" - <<'PYEOF'
import json
rs = json.load(open("electro_optimizer_results.json"))["test_results"]
for r in rs:
    print(f"tid={r['test_id']:>4} n={r['block_count']:>4} feasible={r['is_feasible']} "
          f"cost={r['cost']:.3f} t={r['runtime_seconds']:.2f}s")
PYEOF
