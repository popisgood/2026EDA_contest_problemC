#!/usr/bin/env bash
# Definitive full-100: sequential warm-start vs parallel-tracks warm-start.
# Both should give the IDENTICAL Total Score (pure perf refactor); the number
# that matters here is the runtime delta.
set -u
EVAL=/home/pop/IntelLabs_Floorset/FloorSet/iccad2026contest
PY=/home/pop/IntelLabs_Floorset/FloorSet/venv/bin/python
OPT=/home/pop/2026_EDA_contest/electro/electro_optimizer.py
cd "$EVAL"

echo "[$(date +%T)] sequential warm-start (no track parallelism) ..."
ELECTRO_M1=1 ELECTRO_M1_WARMSTART=1 ELECTRO_PARALLEL_TRACKS=0 \
    "$PY" iccad2026_evaluate.py --evaluate "$OPT" >/dev/null 2>&1
cp electro_optimizer_results.json /tmp/tracks_seq.json

echo "[$(date +%T)] parallel tracks ..."
ELECTRO_M1=1 ELECTRO_M1_WARMSTART=1 ELECTRO_PARALLEL_TRACKS=1 \
    "$PY" iccad2026_evaluate.py --evaluate "$OPT" >/dev/null 2>&1
cp electro_optimizer_results.json /tmp/tracks_par.json

echo "[$(date +%T)] done; weighted Total Score + runtime (v10: e^((n-max_n)/12)):"
"$PY" - <<'PY'
import json, math
for tag, fn in (("SEQ", "/tmp/tracks_seq.json"), ("PAR", "/tmp/tracks_par.json")):
    rs = json.load(open(fn))["test_results"]
    mx = max(r["block_count"] for r in rs)
    num = sum(r["cost"]*math.exp((r["block_count"]-mx)/12) for r in rs)
    den = sum(math.exp((r["block_count"]-mx)/12) for r in rs)
    feas = sum(int(r["is_feasible"]) for r in rs)
    rts = [r["runtime_seconds"] for r in rs]
    rt = sorted(rts)
    print(f"  {tag}  Total={num/den:.4f}  feasible={feas}/{len(rs)}  "
          f"rt sum={sum(rts):.0f}s med={rt[len(rt)//2]:.1f}s max={rt[-1]:.1f}s")

# per-case cost diff sanity check (should be ALL zero)
seq = {r["test_id"]: r["cost"] for r in json.load(open("/tmp/tracks_seq.json"))["test_results"]}
par = {r["test_id"]: r["cost"] for r in json.load(open("/tmp/tracks_par.json"))["test_results"]}
diffs = [(t, seq[t], par[t]) for t in seq if abs(seq[t]-par[t]) > 1e-9]
print(f"  cost differences: {len(diffs)} cases" + (f" -> {diffs[:5]}" if diffs else " (none, identical)"))
PY
