#!/usr/bin/env bash
set -e
SUB=/home/pop/2026_EDA_contest/submit/electro_submission
SRC=/home/pop/2026_EDA_contest/electro
cp "$SRC/electro_optimizer.py" "$SRC/analytical_place.py" "$SRC/legalize.py" \
   "$SRC/soft_repair.py" "$SRC/electro_parallel.py" "$SUB/"
rm -rf "$SUB/__pycache__"
echo "=== files in package ==="; ls -la "$SUB"
echo "=== re-audit absolute paths in package ==="
grep -rnE '/home/|/Users/|/mnt/|[A-Z]:\\|/tmp/' "$SUB"/*.py || echo "  NONE (clean)"
echo "=== repack tar.gz (exclude pycache) ==="
cd /home/pop/2026_EDA_contest/submit
rm -f electro_submission.tar.gz
tar --exclude='__pycache__' -czf electro_submission.tar.gz electro_submission
echo "=== tar contents ==="; tar -tzf electro_submission.tar.gz
echo "=== size ==="; du -h electro_submission.tar.gz
