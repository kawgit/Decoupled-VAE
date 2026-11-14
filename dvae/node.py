"""A node of a code tree, read as one module for every network that reads nodes (the V-way tree's encoder in
dvae/vway.py, the prior and discriminator in dvae/vprior.py): NodeEncoder reads the node's current mean (the sum of
every delta taken so far, as one token: a linear map of the raw vector, so the tree's own mean, a constant, is in its
bias) and its last `window` deltas (each as its direction, log length and whether it exists: a shallow node has
fewer; the most recent last), with self-attention among them. Every layer's normalized tokens are the node's memory,
and the mean token's last state its summary. QueryReader runs a task's query tokens over a node: they read its memory
and each other at every layer (Block with mem), the summary added to their input embeddings.
"""

import torch
import torch.nn as nn
from torch import Tensor

from dvae.deltatree import EPS, Block


def directions(x: Tensor) -> tuple[Tensor, Tensor]:
    """Vectors x [..., D] as their directions and log lengths [..., 1]."""
    n = x.norm(dim=-1, keepdim=True).clamp_min(EPS)
    return x / n, n.log()


class NodeEncoder(nn.Module):
    """A node's memory (each layer's normalized tokens [B, 1 + window, W], the mean's first) and summary [B, W]."""

    def __init__(self, depths, D=512, window=16, width=128, layers=4, heads=4, dropout=0.0):
        super().__init__()
        self.window = window
        self.mean_in = nn.Linear(D, width)
        self.delta_in = nn.Linear(D + 2, width)  # (direction, log length, exists)
        self.position = nn.Parameter(torch.randn(window + 1, width) * 0.02)  # (the mean's first)
        self.depth = nn.Parameter(torch.randn(depths + 1, width) * 0.02)  # (+ 1: a discriminator's child nodes)
        self.blocks = nn.ModuleList(Block(width, heads, dropout) for _ in range(layers))
        self.out_norm = nn.LayerNorm(width)

    def forward(self, win: Tensor, valid: Tensor, mean: Tensor, d: Tensor) -> tuple[list[Tensor], Tensor]:
        u, lg = directions(win)
        e = valid.float()[..., None]
        deltas = self.delta_in(torch.cat([u, lg * e, e], -1))
        m = torch.cat([(self.mean_in(mean) + self.depth[d])[:, None], deltas], 1) + self.position
        mems = []
        for blk in self.blocks:
            m, x = blk(m)
            mems.append(x)
        return mems, self.out_norm(m[:, 0])


class QueryReader(nn.Module):
    """Queries h [B, Q, W] reading a node's memory at every layer (blocks of their own), the node's summary added
    to their input; their final normalized states [B, Q, W]."""

    def __init__(self, width=128, layers=4, heads=4, dropout=0.0):
        super().__init__()
        self.summary_in = nn.Linear(width, width)
        self.blocks = nn.ModuleList(Block(width, heads, dropout) for _ in range(layers))
        self.out_norm = nn.LayerNorm(width)

    def forward(self, h: Tensor, mems: list[Tensor], summary: Tensor) -> Tensor:
        h = h + self.summary_in(summary)[:, None]
        for blk, mem in zip(self.blocks, mems):
            h, _ = blk(h, mem)
        return self.out_norm(h)


__all__ = ["NodeEncoder", "QueryReader", "directions"]
