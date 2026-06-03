"""ML floorplanner for ICCAD 2026 FloorSet Challenge.

A supervised graph-Transformer predicts (cx, cy, w, h) per block from
constraint inputs; predictions are converted into a B*-tree by sorted
insertion, then handed off to the C++ SA solver for refinement.

Modules:
    data    -- FloorSet Lite/Prime dataset loader (training-time)
    model   -- Transformer-based position predictor
    train   -- Supervised training loop (MSE on positions + soft penalties)
    predict -- Inference helper used by my_optimizer_ml.py at solve-time
"""
