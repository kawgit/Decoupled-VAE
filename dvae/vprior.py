"""A prior over a V-way tree's codes (train_inner_vway.py), and a discriminator of the same shape
(train_vprior.py).

A code is a token in 0..V-1 at each depth. CodeNet reads, at every depth, the token taken at the depth before (none
at the root), the tree's own shares p there (what the tree's networks predict for that node's cells, log) and the
depth's position; causal attention lets depth k see its own inputs and everything before, never its own token. Its
head gives V logits per depth. As the prior they are p(token k | the code before it); as the discriminator, each
token's trust logit: how likely that token, after that code, came from real codes rather than the prior. The head
starts at zero, so an untrained prior is uniform and an untrained discriminator trusts everything one half.
"""

import torch
import torch.nn as nn
from torch import Tensor

from dvae.deltatree import Block


class CodeNet(nn.Module):
    def __init__(self, depths, V=8, width=256, layers=4, heads=4, dropout=0.0):
        super().__init__()
        self.config = dict(depths=depths, V=V, width=width, layers=layers, heads=heads, dropout=dropout)
        self.V = V
        self.share_in = nn.Linear(V, width)
        self.token_in = nn.Embedding(V + 1, width)  # (the token before, or V at the root)
        self.position = nn.Parameter(torch.randn(depths, width) * 0.02)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(Block(width, heads, dropout) for _ in range(layers))
        self.out_norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, V)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, tokens: Tensor, log_shares: Tensor) -> Tensor:
        """Logits [B, K, V] for codes tokens [B, K] (long) with the tree's log shares [B, K, V] at each depth."""
        B, K = tokens.shape
        before = torch.cat([tokens.new_full((B, 1), self.V), tokens[:, :-1]], 1)
        h = self.share_in(log_shares.to(self.share_in.weight.dtype)) + self.token_in(before) + self.position[:K]
        h = self.drop(h)
        for blk in self.blocks:
            h, _ = blk(h, causal=True)
        return self.head(self.out_norm(h))


__all__ = ["CodeNet"]
