#!/usr/bin/env bash
# Cheap validation of Family Two (in-loop differentiable soft-block shaping).
# electro ALREADY co-optimizes log-aspect `la` with positions, so we (a) ablate it
# (FREEZE_SHAPE=1 -> square blocks) to measure its current contribution, and (b)
# loosen the aspect cap (AR_CAP 4->16) to probe headroom.  Same evaluator, same cases.
set -u
EVAL=/home/pop/IntelLabs_Floorset/FloorSet/iccad2026contest
PY=/home/pop/IntelLabs_Floorset/FloorSet/venv/bin/python
REPO=/home/pop/2026_EDA_contest
OPT="$REPO/electro/electro_optimizer.py"
TIDS="${TIDS:-0 40 80 99}"
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

run_cfg "shape ON  (default, AR_CAP=4)"
run_cfg "shape OFF (square blocks, FREEZE_SHAPE=1)" ELECTRO_FREEZE_SHAPE=1
run_cfg "shape PUSHED (AR_CAP=16)"                  ELECTRO_AR_CAP=16
