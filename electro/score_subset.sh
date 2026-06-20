#!/usr/bin/env bash
# Score the large-n-heavy screening subset (or any ids) with the real evaluator.
# Env knobs (ELECTRO_*) are inherited, so call like:
#   ELECTRO_EDENSITY=10 bash score_subset.sh
#   bash score_subset.sh 0 99            # custom ids
set -u
EVAL=/home/pop/IntelLabs_Floorset/FloorSet/iccad2026contest
PY=/home/pop/IntelLabs_Floorset/FloorSet/venv/bin/python
OPT=/home/pop/2026_EDA_contest/electro/electro_optimizer.py
SUB="${*:-0 20 40 55 60 65 70 75 80 85 90 92 95 97 99}"
cd "$EVAL" || exit 1
"$PY" iccad2026_evaluate.py --evaluate "$OPT" --test-id $SUB 2>/dev/null \
  | grep -iE "Total Score|Feasible|Infeasible|Mean cost|cases"
echo "--- per-case (feasible/cost/area_gap/hpwl_gap/V_rel) ---"
"$PY" - <<'PYEOF'
import json
d = json.load(open("electro_optimizer_results.json"))
rows = d["test_results"]
nfeas = sum(1 for r in rows if r["is_feasible"])
print(f"feasible {nfeas}/{len(rows)}")
for r in rows:
    print(f"  tid {r['test_id']:>3} n={r['block_count']:>3} "
          f"feas={int(r['is_feasible'])} cost={r['cost']:.3f} "
          f"area_gap={r['area_gap']:+.3f} hpwl_gap={r['hpwl_gap']:+.3f} "
          f"V_rel={r['violations_relative']:.3f}")
PYEOF
