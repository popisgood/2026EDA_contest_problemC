#!/usr/bin/env bash
set -e
PKG=/tmp/sub_verify/electro_submission
PY=/home/pop/IntelLabs_Floorset/FloorSet/venv/bin/python
echo "=== 1) syntax / bytecode compile (all 6 .py) ==="
"$PY" -m py_compile "$PKG"/electro_optimizer.py "$PKG"/analytical_place.py \
  "$PKG"/legalize.py "$PKG"/soft_repair.py "$PKG"/electro_parallel.py \
  "$PKG"/shape_compact.py \
  && echo "  compile OK (no syntax errors)"
echo "=== 2) external imports actually used ==="
grep -hoE '^(import|from) [a-zA-Z_][a-zA-Z0-9_.]*' "$PKG"/*.py \
  | awk '{print $2}' | cut -d. -f1 | sort -u \
  | grep -vE '^(__future__|os|sys|time|typing|multiprocessing|analytical_place|legalize|soft_repair|electro_parallel|shape_compact|iccad2026_evaluate)$' \
  | sed 's/^/  needs: /'
echo "=== 2b) shape_compact importable from package dir (regression check for the missing-6th-file bug) ==="
( cd "$PKG" && "$PY" -c "from shape_compact import compact_and_shape; print('  OK')" )
echo "=== 3) requirements.txt ==="; cat "$PKG"/requirements.txt
echo "=== 4) README run command line ==="; grep -nE "evaluate|electro_optimizer.py" "$PKG"/README.txt | head -6
echo "=== 5) class check: FloorplanOptimizer subclass present ==="
grep -nE "class .*FloorplanOptimizer" "$PKG"/electro_optimizer.py
echo "=== 6) submission default knobs ==="
grep -nE "setdefault\(\"ELECTRO_(CLAMP|NONNEG)" "$PKG"/electro_optimizer.py
