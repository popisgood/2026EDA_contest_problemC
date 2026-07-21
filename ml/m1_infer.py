"""M1 inference: autoregressive rollout with exact legality masking + snapping.

M1Predictor.predict() takes the same solve()-time tensors the contest passes and
returns [(x, y, w, h)] with ZERO overlap by construction (mask) -- or None if
weights are unavailable (caller falls back, strictly-additive contract).

Rollout per case (n <= 120 forwards of a small transformer; sub-second on CPU):
  1. canvas: area/util with pin-bbox aspect (same proxy eDensity uses), expanded
     to cover any preplaced block;
  2. preplaced placed first at their exact target positions (locked context);
  3. per step: aspect bin (soft; MIB groups share the first member's bin)
     -> exact-geometry legality mask over the grid -> masked argmax -> snap to
     the nearest abutment (recovers interlocked tiling lost to quantization);
  4. dead-end safety: if the mask is empty, grow the canvas 10% and retry.
"""
from __future__ import annotations

import os

import numpy as np
import torch

from .m1_common import (GRID, bin_to_wh, cell_to_xy, legality_mask, prep_case,
                        snap, step_tokens)
from .m1_model import M1Net


def _np_edges(t):
    if t is None or t.numel() == 0:
        return None
    a = t.cpu().numpy()
    a = a[a[:, 0] != -1]
    return a if len(a) else None


@torch.no_grad()
def rollout_layout(model, case, area, device="cpu", deadend="none"):
    """Autoregressive masked rollout over a PREPPED case dict (from prep_case).
    Returns (x, y, w, h) numpy arrays [n].  Shared by inference (M1Predictor.predict)
    and by scheduled-sampling cache building in training (feed the model its own
    rollout back as context).  Does NOT mutate the caller's case dict.

    deadend: if a block has no legal cell even after growing the canvas --
      "none" -> return None (inference; caller falls back to electro)
      "stub" -> place it at the origin and keep going (training cache; never abort)."""
    case = dict(case)          # shallow copy: rebind w/h/Wc/Hc locally only
    n = case["n"]
    Wc, Hc = case["Wc"], case["Hc"]
    w = np.asarray(case["w"], float).copy()
    h = np.asarray(case["h"], float).copy()
    x = np.asarray(case["px"], float).copy()
    y = np.asarray(case["py"], float).copy()
    placed_mask = np.asarray(case["is_pre"], bool).copy()
    placed_list = [(x[i], y[i], w[i], h[i]) for i in range(n) if placed_mask[i]]
    mib_bin = {}
    for cur in case["order"]:
        case["w"], case["h"] = w, h            # tokens see current dims
        tokens, pad = step_tokens(case, placed_mask, np.stack([x, y], 1), cur)
        tt = torch.from_numpy(tokens)[None].to(device)
        pp = torch.from_numpy(pad)[None].to(device)
        ci = torch.tensor([cur], device=device)
        pos_logits, asp_logits = model(tt, pp, ci)
        pos_logits = pos_logits[0].detach().cpu().numpy()
        if case["is_soft"][cur]:               # choose shape first
            g = int(case["mib"][cur])
            if g > 0 and g in mib_bin:
                k = mib_bin[g]
            else:
                k = int(asp_logits[0].argmax())
                if g > 0:
                    mib_bin[g] = k
            w[cur], h[cur] = bin_to_wh(k, area[cur])
        mask = legality_mask(w[cur], h[cur], placed_list, Wc, Hc)
        tries = 0
        while not mask.any() and tries < 4:    # dead-end: grow canvas
            Wc *= 1.10
            Hc *= 1.10
            case["Wc"], case["Hc"] = Wc, Hc
            mask = legality_mask(w[cur], h[cur], placed_list, Wc, Hc)
            tries += 1
        if not mask.any():
            if deadend == "none":
                return None
            bx, by = 0.0, 0.0
        else:
            pos_logits[~mask] = -1e30
            bx, by = cell_to_xy(int(pos_logits.argmax()), Wc, Hc)
            bx, by = snap(bx, by, w[cur], h[cur], placed_list, Wc, Hc)
        x[cur], y[cur] = bx, by
        placed_mask[cur] = True
        placed_list.append((bx, by, w[cur], h[cur]))
    return x, y, w, h


class M1Predictor:
    def __init__(self, weights_path: str, device: str = "cpu"):
        self.device = device
        self.model = None
        if os.path.exists(weights_path):
            ck = torch.load(weights_path, map_location=device, weights_only=False)
            cfg = ck.get("config", {})
            self.model = M1Net(cfg.get("d_model", 192), cfg.get("layers", 4),
                               cfg.get("heads", 4)).to(device)
            self.model.load_state_dict(ck["model_state"])
            self.model.eval()
            print(f"[m1] loaded {weights_path}")
        else:
            print(f"[m1] weights not found at {weights_path} -- M1 disabled")

    def available(self):
        return self.model is not None

    @torch.no_grad()
    def predict(self, block_count, area_targets, constraints, target_positions,
                b2b, p2b, pins, util: float = 0.96):
        if not self.available() or block_count == 0:
            return None
        n = int(block_count)
        area = area_targets[:n].cpu().numpy().astype(float).clip(min=1e-9)
        cons = constraints[:n].cpu().numpy()
        cons5 = np.zeros((n, 5))
        cons5[:, :cons.shape[1]] = cons
        tp = (target_positions[:n].cpu().numpy().astype(float)
              if target_positions is not None else np.full((n, 4), -1.0))
        pv = _np_edges(pins)

        # canvas: total area / util, aspect from pin bbox; cover preplaced extents
        tot = float(area.sum())
        aspect = 1.0
        if pv is not None and len(pv) >= 2:
            pw = pv[:, 0].max() - pv[:, 0].min()
            ph = pv[:, 1].max() - pv[:, 1].min()
            aspect = min(max(pw / max(ph, 1e-6), 0.25), 4.0)
        Hc = (tot / util / aspect) ** 0.5
        Wc = aspect * Hc
        pre = (cons5[:, 1] != 0)
        for i in range(n):                      # canvas must contain preplaced
            if pre[i] and tp[i, 0] >= 0:
                Wc = max(Wc, tp[i, 0] + tp[i, 2])
                Hc = max(Hc, tp[i, 1] + tp[i, 3])

        case = prep_case(area, cons5, _np_edges(b2b), _np_edges(p2b), pv,
                         Wc, Hc, tp=tp)
        res = rollout_layout(self.model, case, area, device=self.device,
                             deadend="none")
        if res is None:
            return None                          # dead-end: caller falls back
        x, y, w, h = res
        return [(float(x[i]), float(y[i]), float(w[i]), float(h[i]))
                for i in range(n)]
