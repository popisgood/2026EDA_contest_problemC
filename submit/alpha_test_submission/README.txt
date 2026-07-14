================================================================================
 ICCAD 2026 CAD Contest - Problem C (FloorSet)  --  Alpha Test Submission
================================================================================

Team        : CADC1083   (registration no.: 1196)
Entry point : my_optimizer.py
Platform    : Linux x86-64  (C++ solver is statically linked; no system libs)
Packaging   : delivered as a single .tar.gz; extract it, then run as in Sec. 1.


--------------------------------------------------------------------------------
0. TARGET EVALUATION ENVIRONMENT (per official guidelines, 2026-06-16)
--------------------------------------------------------------------------------
This submission was prepared for, and is compatible with:

    OS      : Debian GNU/Linux 13
    Python  : 3.13.x
    PyTorch : 2.12.0   (already installed on the eval machine)
    GLIBC   : 2.41      (the bundled solver is STATIC, so this does not matter)
    CPU     : Intel Xeon, up to 48 cores      Internet: NONE at run time

No internet is required at evaluation time: the only run-time Python dependency
is PyTorch, which the evaluation environment already provides.  (See Sec. 5.)


--------------------------------------------------------------------------------
1. HOW TO RUN
--------------------------------------------------------------------------------
This submission plugs into the official FloorSet evaluation framework.  After
extracting the archive, from the directory that contains the contest's
iccad2026_evaluate.py and the dataset, run:

    # evaluate on all cases
    python iccad2026_evaluate.py --evaluate my_optimizer.py

    # or a single case (debugging)
    python iccad2026_evaluate.py --evaluate my_optimizer.py --test-id 0

The framework loads the MyOptimizer class from my_optimizer.py and calls its
solve() once per test case.  solve() returns one (x, y, w, h) tuple per block.

On the official environment NOTHING needs to be installed (PyTorch is present).
If running on a fresh machine WITH internet, you may first do:

    pip install -r requirements.txt

No environment variables need to be set: all production defaults are baked in
(see Sec. 4).  The optimizer auto-detects the bundled solver binary and ML
model from this folder regardless of the working directory.


--------------------------------------------------------------------------------
2. WHAT IS INCLUDED
--------------------------------------------------------------------------------
    my_optimizer.py        Entry point (ML-augmented optimizer; the class the
                           framework loads).
    floorplan_base.py      Baseline optimizer + tensor<->text I/O helpers that
                           my_optimizer.py builds on.
    floorplanner           The C++ solver, STATICALLY LINKED x86-64 Linux ELF
                           (no runtime library / GLIBC dependency).
    ml/                    ML warm-start package:
        __init__.py  data.py  model.py  predict.py
        weights/floorplan_v2.pt    trained Graph-Transformer model (~2 MB)
    requirements.txt       Run-time Python dependency list (torch, numpy).
    README.txt             This file.
    src/  include/  Makefile    C++ source, provided so the binary can be
                           rebuilt if ever required (see Sec. 3).

This is a SOURCE-code submission (an allowed fallback in the guidelines): the
optimizer is a Python module the framework imports, plus a prebuilt static C++
solver binary.  PyInstaller is therefore not used -- the framework loads our
optimizer by importing my_optimizer.py, not by executing a standalone binary.


--------------------------------------------------------------------------------
3. HOW TO (RE)BUILD THE SOLVER BINARY  (optional - a prebuilt binary is shipped)
--------------------------------------------------------------------------------
The included ./floorplanner is already a portable, statically linked Linux
binary, so rebuilding is normally unnecessary.  To rebuild from source:

    make static       # produces a statically linked ./floorplanner (recommended)
    # or
    make              # dynamically linked build

Requirements to rebuild: g++ with C++17 support and pthreads (eval env has
GCC/G++ 14.2.0).


--------------------------------------------------------------------------------
4. ALGORITHM (one-paragraph summary)
--------------------------------------------------------------------------------
A Graph-Transformer (ml/) predicts an initial position for every block; these
seed a B*-tree on half of the parallel Fast-Simulated-Annealing search chains
(the other half use a constraint-aware heuristic start), and the best feasible
result is returned.  A contour packer legalizes each candidate and applies
deterministic repair passes (preplaced-anchor pre-seeding, boundary repair,
grouping repair) so hard/soft constraints are satisfied.  If a case still comes
back infeasible, the wrapper automatically retries it with a larger time budget.

Baked-in production defaults (overridable via env vars, but NOT required):
    per-case time budget : 1 + 0.05 * n  seconds   (n = block count)
    threads              : one per PHYSICAL CPU core (auto-detected; up to 48
                           on the evaluation machine) -- parallel best-of-N SA
                           chains.  Physical (not logical/HT) count avoids the
                           hyperthread oversubscription that slows this workload.
    ML model             : ml/weights/floorplan_v2.pt   (inference on CPU)
    feasibility escalate : on


--------------------------------------------------------------------------------
5. NOTES ON ROBUSTNESS / COMPATIBILITY
--------------------------------------------------------------------------------
- Static solver: ./floorplanner is fully statically linked (ldd reports "not a
  dynamic executable"), so it runs on Debian 13 / GLIBC 2.41 with no libraries.
- CPU-only at run time (no GPU required); torch is used only for the ~2 MB model
  and runs inference on CPU.
- Single run-time dependency: only `torch` is imported at solve() time, so the
  no-internet evaluation environment (which ships torch 2.12.0) needs nothing
  installed.
- Graceful ML fallback: if torch or the model file are unavailable, or the model
  fails to load/run for ANY reason, the optimizer silently disables the ML warm
  start and runs the pure C++ solver, which still returns a feasible result.  No
  test case can become infeasible due to an ML error.
================================================================================
