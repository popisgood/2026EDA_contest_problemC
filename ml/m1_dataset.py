"""M1 teacher-forcing dataset: (case, step) -> one placement decision sample.

Wraps ml/data.py's FloorSetLiteDataset (handles both the 1M-train .th format and
the 100-case test format).  For a case with placement order [b0, b1, ...], step t
is: tokens where preplaced + b0..b(t-1) sit at their GT positions, current = bt;
labels = bt's GT grid cell + (if soft) GT log-aspect bin.

Case indices are kept SORTED so consecutive samples hit the dataset's one-file
cache (the .th files hold 112 cases each); per-case static features are cached.
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from .data import FloorSetLiteDataset
from .m1_common import MAX_N, aspect_to_bin, cell_of, prep_case, step_tokens


def _np_edges(t):
    if t is None or t.numel() == 0:
        return None
    a = t.cpu().numpy()
    a = a[a[:, 0] != -1]
    return a if len(a) else None


class M1Steps(Dataset):
    def __init__(self, root: str, max_cases: int = 20000, seed: int = 0,
                 skip_cases: int = 0):
        self.ds = FloorSetLiteDataset(root)
        rng = np.random.default_rng(seed)
        total = len(self.ds)
        lo = min(skip_cases, total)
        pool = np.arange(lo, total)
        pick = pool if len(pool) <= max_cases else np.sort(
            rng.choice(pool, size=max_cases, replace=False))
        self.case_ids = [int(c) for c in pick]

        # enumerate (case, step); step count needs n per case -> read lazily by
        # assuming steps and fixing up in __getitem__ is messy; instead probe n
        # from the dataset's block_counts cache when available, else load.
        # Use the block-count cache ONLY if it already exists -- building it
        # scans all ~1M cases (many minutes).  Otherwise read n per picked case
        # directly: files are ~1.5 MB and sorted picks keep the cache warm.
        counts = None
        from pathlib import Path
        if (Path(root) / f".block_counts_{total}.pt").exists():
            counts = self.ds.block_counts(cache=True, verbose=False)
        self.samples = []
        for c in self.case_ids:
            n = int(counts[c]) if counts is not None else None
            if n is None:
                fi, ci = self.ds.index[c]
                self.ds._load_file(fi)
                blocks = self.ds._unpack_case(ci)[0]
                n = int(blocks.shape[0])
            n = min(n, MAX_N)
            # number of decoding steps = non-preplaced count; we don't know
            # preplaced count without loading, so enumerate up to n and let
            # __getitem__ clamp (steps beyond the order length wrap around).
            for t in range(n):
                self.samples.append((c, t))

        self._cache_cid = -1
        self._cache = None

    def __len__(self):
        return len(self.samples)

    def _case(self, cid):
        if cid == self._cache_cid:
            return self._cache
        fi, ci = self.ds.index[cid]
        self.ds._load_file(fi)
        blocks, b2b, p2b, pins, geom = self.ds._unpack_case(ci)
        n = min(int(blocks.shape[0]), MAX_N)
        blocks = blocks[:n]
        # GT rects from polygons: [n,4] = (x, y, w, h), origin-shifted
        g = geom[:n]
        xmin = g[:, :, 0].min(dim=1).values
        ymin = g[:, :, 1].min(dim=1).values
        xmax = g[:, :, 0].max(dim=1).values
        ymax = g[:, :, 1].max(dim=1).values
        gx = (xmin - xmin.min()).numpy().astype(float)
        gy = (ymin - ymin.min()).numpy().astype(float)
        gw = (xmax - xmin).numpy().astype(float)
        gh = (ymax - ymin).numpy().astype(float)
        gt = np.stack([gx, gy, gw, gh], 1)
        Wc = float((gx + gw).max())
        Hc = float((gy + gh).max())

        area = blocks[:, 0].numpy().astype(float)
        cons = blocks[:, 1:6].numpy()
        cons5 = np.zeros((n, 5))
        cons5[:, :cons.shape[1]] = cons          # (fixed, pre, mib, cluster, bcode)
        pins_np = pins.cpu().numpy() if pins is not None and pins.numel() else None
        if pins_np is not None:
            pins_np = pins_np[pins_np[:, 0] != -1] if len(pins_np) else None
        case = prep_case(area, cons5, _np_edges(b2b), _np_edges(p2b),
                         pins_np, Wc, Hc, gt_xywh=gt)
        case["gt"] = gt
        case["cells"] = np.array(
            [cell_of(gt[i, 0], gt[i, 1], Wc, Hc) for i in range(n)], dtype=np.int64)
        case["bins"] = np.array(
            [aspect_to_bin(gt[i, 2], gt[i, 3]) for i in range(n)], dtype=np.int64)
        self._cache_cid = cid
        self._cache = case
        return case

    def __getitem__(self, idx):
        cid, t = self.samples[idx]
        case = self._case(cid)
        order = case["order"]
        t = t % max(len(order), 1)               # clamp preplaced-count mismatch
        cur = order[t]

        placed = case["is_pre"].copy()
        for j in order[:t]:
            placed[j] = True
        xy = case["gt"][:, :2]                   # teacher forcing: GT positions
        tokens, pad = step_tokens(case, placed, xy, cur)

        asp = int(case["bins"][cur]) if case["is_soft"][cur] else -100  # ignore
        return {
            "tokens": torch.from_numpy(tokens),
            "pad": torch.from_numpy(pad),
            "cur": torch.tensor(cur, dtype=torch.long),
            "cell": torch.tensor(int(case["cells"][cur]), dtype=torch.long),
            "asp": torch.tensor(asp, dtype=torch.long),
        }
