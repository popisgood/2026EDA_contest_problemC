#!/usr/bin/env bash
# Sweep eDensity configs over the screening subset; print Total Score + feas + mean gaps.
set -u
EVAL=/home/pop/IntelLabs_Floorset/FloorSet/iccad2026contest
PY=/home/pop/IntelLabs_Floorset/FloorSet/venv/bin/python
OPT=/home/pop/2026_EDA_contest/electro/electro_optimizer.py
SUB="0 20 40 55 60 65 70 75 80 85 90 92 95 97 99"
cd "$EVAL" || exit 1
run() {
  local label="$1"; shift
  env "$@" "$PY" iccad2026_evaluate.py --evaluate "$OPT" --test-id $SUB >/dev/null 2>&1
  "$PY" - "$label" <<'PYEOF'
import json, sys
lab = sys.argv[1]
d = json.load(open("electro_optimizer_results.json"))
r = d["test_results"]
import math
mx = max(x["block_count"] for x in r)
wsum = sum(math.exp((x["block_count"]-mx)/12) for x in r)
score = sum(x["cost"]*math.exp((x["block_count"]-mx)/12) for x in r)/wsum
nf = sum(1 for x in r if x["is_feasible"])
ag = sum(x["area_gap"] for x in r)/len(r)
hg = sum(x["hpwl_gap"] for x in r)/len(r)
vr = sum(x["violations_relative"] for x in r)/len(r)
print(f"{lab:<28} score={score:.4f} feas={nf}/{len(r)} "
      f"mean_area_gap={ag:+.3f} mean_hpwl_gap={hg:+.3f} mean_Vrel={vr:.3f}")
PYEOF
}
OMP_NUM_THREADS=4
export OMP_NUM_THREADS
run "baseline(ed=off)"
for u in 0.70 0.78 0.86 0.94; do
  run "ed=2 util=$u" ELECTRO_EDENSITY=2 ELECTRO_EDENSITY_UTIL=$u
done
