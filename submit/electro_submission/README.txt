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
electro_optimizer.py   Entry point (FloorplanOptimizer subclass): 8-seed
                       persistent-pool multi-start driver, candidate ranking.
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

Note: an ML warm-start path exists (ELECTRO_ML_INIT=1) but is OFF by default
and its ml/ package is NOT bundled here -- see section 4.

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

Multi-start + candidate-portfolio defaults (all set in electro_optimizer.py):

    ELECTRO_SEEDS             = 8        independent starts per case
    ELECTRO_PARALLEL          = 1        run the 8 seeds in a PERSISTENT fork
                                         pool (built once for the whole eval,
                                         not once per case) -- on an 8+ core
                                         machine this makes 8 seeds cost about
                                         one seed's wall-clock, not 8x
    ELECTRO_ML_INIT           = 0        ML warm-start OFF (see below)
    ELECTRO_JACOBI_MODE       = hedge    each seed's own place() call uses
                                         Jacobi graph-layout warm-start (the
                                         hedge fallback track only activates
                                         when ELECTRO_SEEDS==1 -- with 8 seeds
                                         already independent, it would be
                                         redundant serial work)
    ELECTRO_COMPACT           = 1        SDS-style post-legalize compaction
    ELECTRO_BOUNDARY_WIDESWAP = 1        boundary repair wide-swap variant
    ELECTRO_GROUPING_PUSHPAST = 1        grouping repair push-past variant
    ELECTRO_EXPAND_TOPK       = 1        only expand repair-variant cascades
                                         from the single best-ranked start
    ELECTRO_ITERS_PORTFOLIO   = off      adaptive 1200-iter extension DISABLED
                                         (measured: costs +84% runtime for
                                         -5.8% score on a single-seed config,
                                         and was the main source of per-case
                                         regressions -- see electro_optimizer.py)

ELECTRO_ML_INIT=1 restores a trained-model warm-start (ml/predict.py +
ml/weights/floorplan_v2.pt from the dev tree, NOT bundled in this package --
add ELECTRO_ML_DIR or copy ml/ alongside this directory to use it). It is
off by default because it measured WORSE than Jacobi at equal seed count
(see table below) and this package should run with zero external deps
beyond requirements.txt.

Full-100 validation-set comparison (local evaluator, RuntimeFactor=1,
neutral RT, all configs 100/100 feasible), serial/uncontended measurement,
with compaction/wide-swap/push-past/top-K-pruning held fixed across rows:

    config                              Total Score   avg runtime/case
    this submission (8-seed Jacobi)         1.7312         4.53 s
    8-seed ML warm-start instead             1.7741         4.68 s
    4-seed ML + 4-seed Jacobi split          1.7783         4.54 s  (worse
                                                than EITHER pure variant on
                                                15/100 cases -- mixing warm-
                                                start methods within one
                                                ranking pool is not simply
                                                "average of both"; rejected)
    prior single-seed hedge+top-K            1.9683         5.50 s
    teammate's 8-seed ML variant (no
      compaction/wide-swap/push-past)        1.9025         3.84 s
    prior temp default (no multi-start)      2.7215         3.41 s
    prior temp + M1 warm-start               2.3757         7.17 s

8-seed Jacobi is the only config here that is simultaneously the best score
AND has no ML weights to ship.  ELECTRO_JACOBI_MODE=replace or =portfolio,
ELECTRO_SEEDS=1, and ELECTRO_ML_INIT=1 are all still available as overrides
if a different point on the score/runtime/dependency trade-off is wanted --
see the docstring comments at the top of electro_optimizer.py.

All behaviour is overridable via environment variables (optional):
    ELECTRO_CLAMP=0 ELECTRO_NONNEG=0   -> allows negative coordinates.
    ELECTRO_SEEDS=N ELECTRO_PARALLEL=0 -> N seeds, sequential (no fork pool).
    ELECTRO_ML_INIT=1                  -> ML warm-start instead of Jacobi.
    ELECTRO_JACOBI_MODE=replace        -> single Jacobi start per seed with
                                         no random-init fallback (only
                                         relevant when ELECTRO_SEEDS=1).
    ELECTRO_ITERS=K                    -> placement iterations (default 600).

No code change is needed to run -- the defaults are production-ready.
