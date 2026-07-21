#!/usr/bin/env bash
set -u
cd /home/pop/IntelLabs_Floorset/FloorSet/iccad2026contest
PY=/home/pop/IntelLabs_Floorset/FloorSet/venv/bin/python

echo "----- v1 (no SS) -----"
WEIGHTS=/home/pop/2026_EDA_contest/ml/weights/m1_v1.pt "$PY" /home/pop/2026_EDA_contest/electro/diag_m1.py 2>&1 | grep -v Warning

echo "----- v2 (with SS) -----"
WEIGHTS=/home/pop/2026_EDA_contest/ml/weights/m1_v2_ss.pt "$PY" /home/pop/2026_EDA_contest/electro/diag_m1.py 2>&1 | grep -v Warning
