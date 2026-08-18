#!/usr/bin/env bash
# End-to-end submission verification: extract the .tar.gz to a FRESH arbitrary
# location and run the optimizer through the official evaluator with the EXACT
# command format the contest uses (no ELECTRO_* env vars -> plain defaults).
set -e
TAR=/home/pop/2026_EDA_contest/submit/electro_submission.tar.gz
WORK=/tmp/sub_verify
EVAL=/home/pop/IntelLabs_Floorset/FloorSet/iccad2026contest
PY=/home/pop/IntelLabs_Floorset/FloorSet/venv/bin/python

# 1) extract to a clean, arbitrary dir (simulates the grader unpacking anywhere)
rm -rf "$WORK"; mkdir -p "$WORK"
tar -xzf "$TAR" -C "$WORK"
OPT="$WORK/electro_submission/electro_optimizer.py"
echo "extracted optimizer: $OPT"
echo "python: $("$PY" --version 2>&1)"

# 2) run with the contest's command format, plain (no env knobs)
cd "$EVAL"
"$PY" iccad2026_evaluate.py --evaluate "$OPT" --test-id 5 30 60 95 --save-solutions >/tmp/verify.log 2>&1 || { echo "RUN FAILED"; tail -30 /tmp/verify.log; exit 1; }

# 3) report feasibility, cost, and min-coords (first-quadrant check)
"$PY" - <<'PYEOF'
import json
r = json.load(open("electro_optimizer_results.json"))["test_results"]
s = json.load(open("electro_optimizer_solutions.json"))
sols = s.get("solutions") or s.get("test_results")
byid = {}
for ss in sols:
    pos = ss.get("positions") or ss.get("solution")
    if pos:
        byid[ss.get("test_id")] = (min(p[0] for p in pos), min(p[1] for p in pos),
                                    min(p[2] for p in pos), min(p[3] for p in pos))
print("%-6s %-4s %-6s %-8s %-10s %-10s" % ("tid","n","feas","cost","min(x,y)","min(w,h)"))
allok = True
for rr in r:
    t = rr["test_id"]; mn = byid.get(t, (None,)*4)
    fq = (mn[0] is not None and mn[0] >= -1e-9 and mn[1] >= -1e-9)
    pos_wh = (mn[2] is not None and mn[2] > 0 and mn[3] > 0)
    ok = rr["is_feasible"] and fq and pos_wh
    allok = allok and ok
    print("%-6s %-4s %-6s %-8.3f (%6.2f,%6.2f) (%5.2f,%5.2f) %s" % (
        t, rr["block_count"], rr["is_feasible"], rr["cost"], mn[0], mn[1], mn[2], mn[3],
        "OK" if ok else "**CHECK**"))
print("\nALL FEASIBLE + FIRST-QUADRANT + POSITIVE DIMS:", "YES" if allok else "NO")
PYEOF
echo "--- runtime per case (from log) ---"
grep -aE "\[electro\] n=" /tmp/verify.log | sed 's/resid.*//'
