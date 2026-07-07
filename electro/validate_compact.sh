#!/usr/bin/env bash
# Validate the SDS-style compaction+shaping pass: electro WITHOUT vs WITH it
# (ELECTRO_COMPACT). The compacted layout is an extra candidate, so the cost-aware
# ranking keeps it only when net better -> ON should be <= OFF on cost everywhere.
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

# NOTE: ELECTRO_COMPACT now defaults ON in electro_optimizer.py, so the baseline
# must force it off explicitly.
run_cfg "compaction OFF (ELECTRO_COMPACT=0)" ELECTRO_COMPACT=0
run_cfg "compaction ON  (ELECTRO_COMPACT=1)" ELECTRO_COMPACT=1
