"""M1 trainer: pure supervised imitation (teacher forcing), no RL.

Usage (from the repo root, inside a python env with torch):

  # smoke test (CPU, minutes):
  python -m ml.m1_train --data-root ~/IntelLabs_Floorset/FloorSet/floorset_train_data \
      --max-cases 300 --epochs 2 --out ml/weights/m1_smoke.pt

  # real run (GPU recommended; scale max-cases/epochs to budget):
  python -m ml.m1_train --data-root ~/IntelLabs_Floorset/FloorSet/floorset_train_data \
      --max-cases 100000 --epochs 3 --bs 256 --device cuda --out ml/weights/m1_v1.pt

Metrics: pos_loss (CE over 1024 cells), pos_acc (exact cell), pos_near (within
1 cell in x and y -- the snap pass absorbs one-cell error), asp_acc.
Validation on a held-out slice of cases (--val-cases, taken before the train pick).
"""
from __future__ import annotations

import argparse
import math
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .m1_common import GRID
from .m1_dataset import M1Steps
from .m1_model import M1Net


def near_acc(logits, target):
    pred = logits.argmax(-1)
    px, py = pred % GRID, pred // GRID
    tx, ty = target % GRID, target // GRID
    return (((px - tx).abs() <= 1) & ((py - ty).abs() <= 1)).float().mean().item()


def run_epoch(model, loader, opt, device, train=True):
    model.train(train)
    tot = {"n": 0, "pos": 0.0, "acc": 0.0, "near": 0.0, "asp_n": 0, "asp_acc": 0.0}
    for batch in loader:
        tokens = batch["tokens"].to(device)
        pad = batch["pad"].to(device)
        cur = batch["cur"].to(device)
        cell = batch["cell"].to(device)
        asp = batch["asp"].to(device)
        with torch.set_grad_enabled(train):
            pos_logits, asp_logits = model(tokens, pad, cur)
            loss_pos = F.cross_entropy(pos_logits, cell)
            has_asp = asp >= 0
            loss_asp = (F.cross_entropy(asp_logits[has_asp], asp[has_asp])
                        if has_asp.any() else pos_logits.new_zeros(()))
            loss = loss_pos + 0.3 * loss_asp
            if train:
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
        b = cell.shape[0]
        tot["n"] += b
        tot["pos"] += loss_pos.item() * b
        tot["acc"] += (pos_logits.argmax(-1) == cell).float().mean().item() * b
        tot["near"] += near_acc(pos_logits, cell) * b
        if has_asp.any():
            tot["asp_n"] += int(has_asp.sum())
            tot["asp_acc"] += (asp_logits[has_asp].argmax(-1) == asp[has_asp]) \
                .float().sum().item()
    n = max(tot["n"], 1)
    return {"pos": tot["pos"] / n, "acc": tot["acc"] / n, "near": tot["near"] / n,
            "asp_acc": tot["asp_acc"] / max(tot["asp_n"], 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--max-cases", type=int, default=20000)
    ap.add_argument("--val-cases", type=int, default=300)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--d-model", type=int, default=192)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="ml/weights/m1_v1.pt")
    ap.add_argument("--resume", default=None)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    # val = first --val-cases cases; train picks from the rest (skip_cases)
    val_ds = M1Steps(args.data_root, max_cases=args.val_cases, seed=args.seed)
    train_ds = M1Steps(args.data_root, max_cases=args.max_cases, seed=args.seed,
                       skip_cases=args.val_cases)
    print(f"[m1] train steps={len(train_ds)}  val steps={len(val_ds)}")

    # NOTE shuffle=False keeps case-file locality (the .th files hold 112 cases;
    # random access would thrash the one-file cache).  Within-batch correlation
    # is acceptable for imitation CE; the case PICK is already randomized.
    tl = DataLoader(train_ds, batch_size=args.bs, shuffle=False, num_workers=0)
    vl = DataLoader(val_ds, batch_size=args.bs, shuffle=False, num_workers=0)

    model = M1Net(args.d_model, args.layers, args.heads).to(args.device)
    if args.resume:
        ck = torch.load(args.resume, map_location=args.device, weights_only=False)
        model.load_state_dict(ck["model_state"])
        print(f"[m1] resumed from {args.resume}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    nparam = sum(p.numel() for p in model.parameters())
    print(f"[m1] params={nparam/1e6:.2f}M  device={args.device}")

    best = math.inf
    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        tr = run_epoch(model, tl, opt, args.device, train=True)
        va = run_epoch(model, vl, opt, args.device, train=False)
        print(f"[m1] ep{ep}  train pos={tr['pos']:.3f} acc={tr['acc']:.3f} "
              f"near={tr['near']:.3f} asp={tr['asp_acc']:.3f} | "
              f"val pos={va['pos']:.3f} acc={va['acc']:.3f} near={va['near']:.3f} "
              f"asp={va['asp_acc']:.3f} | {time.time()-t0:.0f}s")
        if va["pos"] < best:
            best = va["pos"]
            torch.save({"model_state": model.state_dict(),
                        "config": {"d_model": args.d_model, "layers": args.layers,
                                   "heads": args.heads}}, args.out)
            print(f"[m1] saved -> {args.out} (val pos {best:.3f})")


if __name__ == "__main__":
    main()
