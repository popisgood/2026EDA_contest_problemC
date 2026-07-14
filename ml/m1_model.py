"""M1 network: transformer over block tokens -> position-cell + aspect-bin heads.

One forward = one placement decision (which cell for the CURRENT block, and its
aspect bin if soft).  The partial layout is encoded in the tokens themselves
(placed flag + normalized geometry), so no recurrent state is needed and every
step is an independent, batchable forward -- simple to train (teacher forcing)
and fast at inference (n small forwards, static shape MAX_N=128).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .m1_common import GRID, N_ASPECT, TOKEN_DIM


class M1Net(nn.Module):
    def __init__(self, d_model: int = 192, n_layers: int = 4, n_heads: int = 4,
                 dropout: float = 0.1):
        super().__init__()
        self.inp = nn.Linear(TOKEN_DIM, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model, n_heads, dim_feedforward=2 * d_model, dropout=dropout,
            batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, n_layers)
        self.pos_head = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, GRID * GRID))
        self.asp_head = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, N_ASPECT))

    def forward(self, tokens, pad_mask, cur_idx):
        """tokens [B, MAX_N, TOKEN_DIM]; pad_mask [B, MAX_N] True=valid;
        cur_idx [B] index of the block being placed.
        Returns (pos_logits [B, GRID*GRID], asp_logits [B, N_ASPECT])."""
        h = self.enc(self.inp(tokens), src_key_padding_mask=~pad_mask)
        hc = h[torch.arange(h.shape[0], device=h.device), cur_idx]
        return self.pos_head(hc), self.asp_head(hc)
