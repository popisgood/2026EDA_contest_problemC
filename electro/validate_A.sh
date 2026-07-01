#!/usr/bin/env bash
# Validate hypothesis A cheaply: does the existing C++ B*-tree solver give an
# inherently LOW area_gap (tight packing by construction) compared to electro's
# loose analytical layout?  Side-by-side on the same cases via the real evaluator.
set -u
EVAL=/home/pop/IntelLabs_Floorset/FloorSet/iccad2026contest
PY=/home/pop/IntelLabs_Floorset/FloorSet/venv/bin/python
REPO=/home/pop/2026_EDA_contest
TIDS="${TIDS:-0 20 40 60 80 99}"
cd "$EVAL"

run_one () {
  local optpath="$1" jsonname="$2" tid="$3"
  "$PY" iccad2026_evaluate.py --evaluate "$optpath" --test-id "$tid" >/dev/null 2>&1
  "$PY" - "$jsonname" "$tid" <<'PYEOF'
import json,sys
jn,tid=sys.argv[1],sys.argv[2]
try:
    r=json.load(open(jn))["test_results"][0]
    print(f"{tid:>4} {r['block_count']:>4}  {int(r['is_feasible'])}  "
          f"{r['area_gap']:+.3f}  {r['hpwl_gap']:+.3f}  {r['violations_relative']:.3f}  "
          f"{r['runtime_seconds']:6.2f}  {r['cost']:.3f}")
except Exception as e:
    print(f"{tid:>4}  ERR {e}")
PYEOF
}

echo "=== C++ B*-tree solver (my_optimizer.py) ==="
echo " tid    n  F  area_gap hpwl_gap  Vrel    t(s)   cost"
export FLOORPLANNER_BIN="$REPO/floorplanner"
export FLOORPLANNER_TIME="6+0.15*n"
export FLOORPLANNER_THREADS=4
for t in $TIDS; do run_one "$REPO/my_optimizer.py" my_optimizer_results.json "$t"; done

echo
echo "=== electro analytical (electro_optimizer.py, submitted config) ==="
echo " tid    n  F  area_gap hpwl_gap  Vrel    t(s)   cost"
unset FLOORPLANNER_BIN FLOORPLANNER_TIME FLOORPLANNER_THREADS
for t in $TIDS; do run_one "$REPO/electro/electro_optimizer.py" electro_optimizer_results.json "$t"; done
