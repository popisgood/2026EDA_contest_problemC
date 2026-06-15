================================================================================
 ICCAD 2026 CAD Contest - Problem C (FloorSet)  --  Alpha Test Submission
================================================================================

Team        : CADC1083   (registration no.: 1196 )
Entry point : my_optimizer.py
Platform    : Linux x86-64  (binary is statically linked; no system libs needed)


--------------------------------------------------------------------------------
1. HOW TO RUN
--------------------------------------------------------------------------------
This submission plugs into the official FloorSet evaluation framework. From the
directory that contains the contest's iccad2026_evaluate.py and the dataset, run:

    # one-time: install Python dependencies
    pip install -r requirements.txt

    # evaluate on all 100 cases
    python iccad2026_evaluate.py --evaluate my_optimizer.py

    # or a single case (debugging)
    python iccad2026_evaluate.py --evaluate my_optimizer.py --test-id 0

The framework loads the MyOptimizer class from my_optimizer.py and calls its
solve() once per test case. solve() returns one (x, y, w, h) tuple per block.

No environment variables need to be set: all production defaults are baked in
(see section 4). The optimizer auto-detects the bundled solver binary and ML
model from this folder.


--------------------------------------------------------------------------------
2. WHAT IS INCLUDED
--------------------------------------------------------------------------------
    my_optimizer.py        Entry point (ML-augmented optimizer; the class the
                           framework loads).
    floorplan_base.py      Baseline optimizer + tensor<->text I/O helpers that
                           my_optimizer.py builds on.
    floorplanner           The C++ solver, STATICALLY LINKED x86-64 Linux ELF
                           (no runtime library dependencies).
    ml/                    ML warm-start package:
        __init__.py  data.py  model.py  predict.py
        weights/floorplan_v2.pt    trained Graph-Transformer model (~2 MB)
    requirements.txt       Python dependencies (torch, numpy, shapely, ...).
    README.txt             This file.
    src/  include/  Makefile    C++ source, provided so the binary can be
                           rebuilt if ever required (see section 3).


--------------------------------------------------------------------------------
3. HOW TO (RE)BUILD THE SOLVER BINARY  (optional - a prebuilt binary is shipped)
--------------------------------------------------------------------------------
The included ./floorplanner is already a portable, statically linked Linux
binary, so rebuilding is normally unnecessary. To rebuild from source:

    make static       # produces a statically linked ./floorplanner (recommended)
    # or
    make              # dynamically linked build

Requirements to rebuild: g++ with C++17 support and pthreads.


--------------------------------------------------------------------------------
4. ALGORITHM (one-paragraph summary)
--------------------------------------------------------------------------------
A Graph-Transformer (ml/) predicts an initial position for every block; these
seed a B*-tree on half of 8 parallel Fast-Simulated-Annealing search chains
(the other half use a constraint-aware heuristic start), and the best feasible
result is returned. A contour packer legalizes each candidate and applies
deterministic repair passes (preplaced-anchor pre-seeding, boundary repair,
grouping repair) so hard/soft constraints are satisfied. If a case still comes
back infeasible, the wrapper automatically retries it with a larger time budget.

Baked-in production defaults (overridable via env vars, but NOT required):
    per-case time budget : 1 + 0.05 * n  seconds   (n = block count)
    threads              : one per PHYSICAL CPU core (auto-detected; 48 on the
                           evaluation machine) -- parallel best-of-N SA chains.
                           Physical (not logical/HT) count avoids the
                           hyperthread oversubscription that slows this workload.
    ML model             : ml/weights/floorplan_v2.pt   (inference on CPU)
    feasibility escalate : on


--------------------------------------------------------------------------------
5. NOTES
--------------------------------------------------------------------------------
- Developed under WSL/Linux; binary built and tested on x86-64 Linux.
- CPU-only at run time (no GPU required); torch is used only for the ~2 MB model.
- If torch / the model file are unavailable, the optimizer automatically falls
  back to the pure C++ solver (still feasible), so it degrades gracefully.
================================================================================
