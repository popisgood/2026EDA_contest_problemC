#!/usr/bin/env bash
# Validate M1 warm-start: current best pipeline (M1 off) vs M1 rollout + warm-start
# (seed electro's optimizer from M1's legal layout).  Both legs keep compaction at
# its default ON.  Strictly additive (extra candidates), so ON should be <= OFF on
# cost everywhere; the question is how much it wins on the hard (large-n) cases.
set -u
EVAL=/home/pop/IntelLabs_Floorset/FloorSet/iccad2026contest
PY=/home/pop/IntelLabs_Floorset/FloorSet/venv/bin/python
OPT=/home/pop/2026_EDA_contest/electro/electro_optimizer.py
TIDS="${TIDS:-0 20 40 60 80 99}"
cd "$EVAL"

run_cfg () {
  local label="$1"; shift
  echo "=== $label ==="
  echo " tid    n  F  area_gap hpwl_gap  Vrel    t(s)   cost"
  for t in $TIDS; do
    env "$@" "$PY" iccad2026_evaluate.py --evaluate "$OPT" --test-id "$t" >/dev/null 2>&1
    "$PY" - "$t" <<'PYEOF'
import json,sys
t=sys.argv[1]
try:
  r=json.load(open("electro_optimizer_results.json"))["test_results"][0]
  print(f"{t:>4} {r['block_count']:>4}  {int(r['is_feasible'])}  "
        f"{r['area_gap']:+.3f}  {r['hpwl_gap']:+.3f}  {r['violations_relative']:.3f}  "
        f"{r['runtime_seconds']:6.2f}  {r['cost']:.3f}")
except Exception as e:
  print(f"{t:>4}  ERR {e}")
PYEOF
  done
  echo
}

run_cfg "warmstart OFF (M1 off; current best pipeline)"            ELECTRO_M1=0
run_cfg "warmstart ON  (M1 + ELECTRO_M1_WARMSTART=1)" ELECTRO_M1=1 ELECTRO_M1_WARMSTART=1
