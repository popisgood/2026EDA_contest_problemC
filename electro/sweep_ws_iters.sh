#!/usr/bin/env bash
# T8 sweep: how few warm-start iters can we use before cost degrades?
# Cost is independent of parallelization, so run sequential (tracks off) for a
# clean read.  One process per (iters, subset) so cost is what we compare.
set -u
EVAL=/home/pop/IntelLabs_Floorset/FloorSet/iccad2026contest
PY=/home/pop/IntelLabs_Floorset/FloorSet/venv/bin/python
OPT=/home/pop/2026_EDA_contest/electro/electro_optimizer.py
TIDS="${TIDS:-0 40 60 80 99}"
cd "$EVAL"

for IT in 600 300 200 150 100; do
  echo "=== ELECTRO_WARMSTART_ITERS=$IT ==="
  tot=0
  for t in $TIDS; do
    ELECTRO_M1=1 ELECTRO_M1_WARMSTART=1 ELECTRO_PARALLEL_TRACKS=0 \
      ELECTRO_WARMSTART_ITERS=$IT \
      "$PY" iccad2026_evaluate.py --evaluate "$OPT" --test-id "$t" >/dev/null 2>&1
    c=$("$PY" -c "import json;print(f\"{json.load(open('electro_optimizer_results.json'))['test_results'][0]['cost']:.3f}\")")
    printf "  tid=%s cost=%s\n" "$t" "$c"
  done
done
