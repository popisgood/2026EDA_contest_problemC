ICCAD 2026 CAD Contest - Problem C (The FloorSet Challenge, FloorSet-Lite)
=========================================================================
Submission: analytical / electrostatic floorplacement (source-code form)

----------------------------------------------------------------------
1. WHAT THIS IS
----------------------------------------------------------------------
A continuous, gradient-based global placer (the ePlace / DREAMPlace
paradigm) specialised for the small FloorSet instances (n <= 120),
followed by exact legalisation and soft-constraint repair.

Entry point : electro_optimizer.py
Optimizer   : class MyOptimizer(FloorplanOptimizer) -- implements solve().

The contest harness imports electro_optimizer.py and calls solve() once
per test case, exactly as in the provided evaluation script:

    python iccad2026_evaluate.py --evaluate electro_optimizer.py

All SIX .py files must sit in the SAME directory (electro_optimizer.py
inserts its own directory on sys.path and imports the other five).

----------------------------------------------------------------------
2. FILES
----------------------------------------------------------------------
electro_optimizer.py   Entry point (FloorplanOptimizer subclass, multi-seed
                       driver, candidate-portfolio ranking).
analytical_place.py    Continuous global placement (PyTorch / Adam), incl.
                       Jacobi graph-layout warm-start and MIB shape loss.
legalize.py            Constraint-graph compaction + push-apart legalisation
                       (exact zero overlap; shapes unchanged).
soft_repair.py         Boundary / grouping soft-constraint repair passes
                       (incl. push-past / wide-swap variants).
electro_parallel.py    Per-seed worker (place -> legalize -> repair) and the
                       hedge / portfolio candidate-generation helpers.
shape_compact.py       SDS-style post-legalize compaction + soft reshaping,
                       used by electro_parallel.compact_variant.  REQUIRED --
                       ELECTRO_COMPACT defaults to 1, and the call site has no
                       try/except, so a package missing this file will crash
                       every case with ModuleNotFoundError the moment
                       compaction is reached.
requirements.txt       Python dependencies (torch, numpy).
README.txt             This file.

iccad2026_evaluate.py is NOT bundled -- it is provided by the contest
environment and imported at evaluation time.

----------------------------------------------------------------------
3. DEPENDENCIES
----------------------------------------------------------------------
Python 3.10+ , PyTorch >= 2.0 , NumPy >= 1.24  (see requirements.txt).
No internet access required at run time. CPU-only by default; a GPU is
NOT needed (this is a small problem and runs on CPU).

    pip install -r requirements.txt

----------------------------------------------------------------------
4. DEFAULT CONFIGURATION (as submitted, 2026-07-28)
----------------------------------------------------------------------
The submitted default GUARANTEES every block lands in the first quadrant
(x, y >= 0, the contest's (0,0)=lower-left origin convention):

    ELECTRO_CLAMP  = 1   (in-optimization lower-wall clamp)
    ELECTRO_NONNEG = 1   (floor-aware legalize + repair chain)

Candidate-portfolio defaults (all set at the top of electro_optimizer.py):

    ELECTRO_COMPACT           = 1        SDS-style post-legalize compaction
    ELECTRO_BOUNDARY_WIDESWAP = 1        boundary repair wide-swap variant
    ELECTRO_GROUPING_PUSHPAST = 1        grouping repair push-past variant
    ELECTRO_JACOBI_MODE       = hedge    Jacobi warm-start (full iters) as
                                         primary + short random-init fallback
    ELECTRO_HEDGE_ITERS       = 300      fallback track's iteration budget
    ELECTRO_EXPAND_TOPK       = 1        only expand repair-variant cascades
                                         from the single best-ranked start
    ELECTRO_ITERS_PORTFOLIO   = off      adaptive 1200-iter extension DISABLED
                                         (measured: costs +84% runtime for
                                         -5.8% score, and was the main source
                                         of per-case regressions -- see below)

Full-100 validation-set comparison (local evaluator, RuntimeFactor=1,
neutral RT, all configs 100/100 feasible), serial/uncontended measurement:

    config                        Total Score   avg runtime/case
    this submission (hedge+topk)     1.9683         5.50 s
    prior temp default               2.7215         3.41 s
    prior temp + M1 warm-start       2.3757         7.17 s

This config was picked because it is the only one tested that beat a
teammate's independently-developed variant on BOTH score (-2.5%) AND
runtime (-12.1%) in the same serial benchmark; ELECTRO_JACOBI_MODE=replace
scores slightly worse (2.0368) but runs faster still (4.08 s/case) if a
tight official RT median ends up mattering more than the extra ~4% score --
see ELECTRO_JACOBI_MODE's docstring comment in electro_optimizer.py for the
full trade-off and how to switch.

All behaviour is overridable via environment variables (optional):
    ELECTRO_CLAMP=0 ELECTRO_NONNEG=0   -> allows negative coordinates.
    ELECTRO_JACOBI_MODE=replace        -> faster, single-track (see above).
    ELECTRO_JACOBI_MODE=portfolio      -> two full 600-iter tracks, safest.
    ELECTRO_SEEDS=N                    -> N multi-start seeds (default 1).
    ELECTRO_ITERS=K                    -> placement iterations (default 600).

No code change is needed to run -- the defaults are production-ready.
