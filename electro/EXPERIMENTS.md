# Electro placement — technique experiments (score-reduction sprint)

Goal: push the full-100 Total Score down (lower = better) while staying 100% feasible.
Screening uses a fixed 15-case large-n-heavy subset (ids 0 20 40 55 60 65 70 75 80
85 90 92 95 97 99) scored by the real evaluator; winners confirmed on full 100.

Baseline at sprint start (min-displacement repairs + 3 iterate rounds):
  full-100 = 3.5450 (100% feas) ; subset = 3.2000

## Papers surveyed (recent EDA, last ~5y)
1. ePlace (Lu et al., DAC'14 / TODAES'15) — electrostatics + Nesterov + FFT density.
2. DREAMPlace (Lin et al., TCAD'21) — GPU analytical, WA/LSE wirelength.
3. DREAMPlace 4.0 (TCAD'23) — momentum net weighting + Lagrangian refine.
4. Weighted-Average wirelength model (Hsu/Chang, US8689164).
5. PeF: Poisson's-equation fixed-outline floorplanning (TCAD'22).
6. Modern Fixed-Outline FP with Rectilinear Soft Modules (ICCAD'24) — differentiable shaping.
7. Handling Orientation & Aspect Ratio in electrostatic fixed-outline FP (ICCAD'23).
8. CSF: conjugate-subgradient + Q-learning fixed-outline FP (2025).
9. Effective Fixed-Outline FP for Rectilinear Soft Modules (Lu/Huang) — module-area-growing.
10. GrandPlan (ISPD'26) / Floorplanning by Mixed-Variable Optimization (2024).

## Results (subset score; lower better)
| # | technique (source)                | knob                       | subset | verdict |
|---|-----------------------------------|----------------------------|--------|---------|
| - | baseline                          | -                          | 3.2000 | -       |
| T1| smooth/WA wirelength (#2,#4)       | WL_SMOOTH 0.02/0.05/0.1     | 3.44–3.66 | REJECT (worse, feasible) |
| T4| module-area-growing (#6,#9)       | AREA_GROW=0.1 GROW_END=0.7  | 2.6042 | ACCEPT (huge) |
| - | more iters                        | ITERS 900/1200             | 2.61/2.72 | REJECT |
| - | schedule retune                   | OV1/BB ramp                | 2.89–3.18 | REJECT |
| T5| fixed-outline containment (#5,#6) | LAM_OUT=2.0 UTIL=0.85      | 2.5373 | ACCEPT |
| T7| multi-start keep-best (proxy rank)| SEEDS 3/5/8/12             | 2.42/2.31/2.24/2.10 | ACCEPT |
| T2| Nesterov (naive SGD) (#1,#8)      | OPT=nesterov lr 0.04/0.1   | 3.48/2.87 | REJECT (Adam wins) |
| T8| ML warm-start init (FloorplanTfmr)| ML_INIT=1 (+ jitter)       | s1 2.78 / s3 2.16 / s5 2.11 / s8 2.07 | ACCEPT |

ML-init detail: pure prediction as init (seed 0, no jitter) is WORSE than random
(2.78 vs 2.54 at s1) -- the raw prediction sits in a mediocre basin.  But ML+jitter
multi-start beats random multi-start at every seed count, and ML s3 (2.16) already
beats random s8 (2.24): same quality, fewer seeds = less time.  Model trained on the
1M training split (not the 100 test cases) -> generalizes to hidden cases.

## Full-100 confirmations
- area-growing:        3.5450 -> 2.7450
- + fixed-outline:     2.7450 -> 2.6911
- + multi-start(3):    2.6911 -> 2.5407
- + ML-init + s8 + parallel: (pending)

## Runtime finding (IMPORTANT)
place() is dispatch-bound, not compute-bound: ~13-17s/seed on the WSL CPU box and
nearly INDEPENDENT of n (n=31 13.6s vs n=120 16.9s) -- it's 600 iters x ~15 tiny
torch ops x autograd.  A minimal 600-iter torch loop is already 3.5s on this box.
Consequences: extra threads barely help; the WSL CPU box is NOT representative of the
eval machine (native Linux + A100).

CPU fork-parallelism (electro_parallel.py) was a DEAD END: forked workers oversubscribe
the OpenMP runtime (N workers x M threads -> e.g. n=31 took 97s), and multi-thread
workers can deadlock across fork.  Left in as opt-in (ELECTRO_PARALLEL=1, single-thread
workers) but OFF by default.

Real speed levers: (1) seed-BATCHING -- run all seeds as one [seeds,N] tensor so the
per-iter dispatch is amortized; nearly free multi-start on CPU AND GPU (TODO, the right
fix); (2) GPU (auto-detected; A100 on eval box); (3) fewer iters (linear-ish).

## Generalization note (open vs hidden alpha cases)
Generalizable levers (no open-case info used): seed COUNT, the runtime proxy that
picks the best seed per case, ML init (trained on the distribution), min-disp repair.
Mild overfit risk only in the continuous constants (area_grow, lam_out, util, jitter)
-- but tuned on 15 cases and confirmed on full-100 incl. the other 85 untuned cases,
and the optima are broad plateaus.  Keep constants robust; don't razor-tune to open.

## Config knobs (electro_optimizer.py / analytical_place.py defaults)
ELECTRO_SEEDS=3  ELECTRO_PARALLEL=0  ELECTRO_ML_INIT=1  ELECTRO_REPAIR_ROUNDS=3
ELECTRO_AREA_GROW=0.1 ELECTRO_GROW_END=0.7  ELECTRO_LAM_OUT=2.0 ELECTRO_TARGET_UTIL=0.85
ELECTRO_ITERS=600 ELECTRO_LR=0.02 ELECTRO_OPT=adam ELECTRO_ML_JITTER=0.15
