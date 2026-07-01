#!/usr/bin/env bash
# First-quadrant via the floor-aware legalize+repair chain (nonneg threaded all
# the way through, not just the final shove).  Does incremental flooring avoid
# the explosion?
set -u
EVAL=/home/pop/IntelLabs_Floorset/FloorSet/iccad2026contest
PY=/home/pop/IntelLabs_Floorset/FloorSet/venv/bin/python
OPT=/home/pop/2026_EDA_contest/electro/electro_optimizer.py
SUB="0 20 40 55 60 65 70 75 80 85 90 92 95 97 99"
cd "$EVAL" || exit 1
export OMP_NUM_THREADS=4
run() {
  local label="$1"; shift
  env "$@" "$PY" iccad2026_evaluate.py --evaluate "$OPT" --test-id $SUB >/dev/null 2>&1
  "$PY" - "$label" <<'PYEOF'
import json, sys, math
lab = sys.argv[1]
r = json.load(open("electro_optimizer_results.json"))["test_results"]
mx = max(x["block_count"] for x in r)
W = lambda n: math.exp((n-mx)/12)
score = sum(x["cost"]*W(x["block_count"]) for x in r)/sum(W(x["block_count"]) for x in r)
nf = sum(1 for x in r if x["is_feasible"])
ag = sum(x["area_gap"] for x in r)/len(r); hg = sum(x["hpwl_gap"] for x in r)/len(r)
vr = sum(x["violations_relative"] for x in r)/len(r)
print(f"{lab:<24} score={score:.4f} feas={nf}/{len(r)} "
      f"area_gap={ag:+.3f} hpwl_gap={hg:+.3f} Vrel={vr:.3f}")
PYEOF
}
run "baseline(neg)"
run "NONNEG (floor-chain)"        ELECTRO_NONNEG=1
run "CLAMP+NONNEG (floor-chain)"  ELECTRO_CLAMP=1 ELECTRO_NONNEG=1
