"""A node of a code tree, read as one module for every network that reads nodes (the V-way tree's encoder in
dvae/vway.py, the prior and discriminator in dvae/vprior.py): NodeEncoder reads the node's current mean (the sum of
every delta taken so far, as one token: a linear map of the raw vector, so the tree's own mean, a constant, is in its
bias) and its last `window` deltas (each as its direction, log length and whether it exists: a shallow node has
fewer; the most recent last), with self-attention among them. Every layer's normalized tokens are the node's memory,
and the mean token's last state its summary. QueryReader runs a task's query tokens over a node: they read its memory
and each other at every layer (Block with mem), the summary added to their input embeddings.

GeoEncoder reads a node's vectors (its mean, recent deltas and whatever a task adds: centroids, candidates) as tokens
in one self-attention, with each pair's geometry computed in the vectors' own dimension (the cosine between them and
both log lengths) and turned into an attention bias per head and layer: a token's embedding is a fixed linear map of
its vector to the width, which cannot keep the inner products that decide a code's tokens at every depth (each depth
partitions along its own directions), and the biases keep them.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
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

    def forward(self, win: Tensor, valid: Tensor, mean: Tensor, d: Tensor,
                extra: Tensor | None = None) -> tuple[list[Tensor], Tensor]:
        """extra [B, E, W]: more tokens for the memory, already embedded (after the window's)."""
        u, lg = directions(win)
        e = valid.float()[..., None]
        deltas = self.delta_in(torch.cat([u, lg * e, e], -1))
        m = torch.cat([(self.mean_in(mean) + self.depth[d])[:, None], deltas], 1) + self.position
        if extra is not None:
            m = torch.cat([m, extra], 1)
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


class GeoLayer(nn.Module):
    """Pre-norm self-attention with an additive bias [B, heads, T, T], and a feed-forward."""

    def __init__(self, width, heads):
        super().__init__()
        self.heads = heads
        self.ln_a, self.ln_f = nn.LayerNorm(width), nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width)
        self.out = nn.Linear(width, width)
        self.ff = nn.Sequential(nn.Linear(width, 4 * width), nn.GELU(), nn.Linear(4 * width, width))

    def forward(self, h: Tensor, bias: Tensor) -> Tensor:
        B, T, W = h.shape
        q, k, v = self.qkv(self.ln_a(h)).view(B, T, 3, self.heads, W // self.heads).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=bias.to(q.dtype))
        h = h + self.out(a.transpose(1, 2).reshape(B, T, W))
        return h + self.ff(self.ln_f(h))


class GeoEncoder(nn.Module):
    """Tokens [B, T, W] (normalized) for a node's vectors S [B, T, D] (those that do not exist, e.g. deltas before
    the root, marked by exists [B, T]: no key attends to them), each of the kinds kind [T] (long: which embedding
    marks its role), at depths d [B]."""

    def __init__(self, depths, kinds, D=512, width=128, layers=4, heads=4, pair_hidden=64):
        super().__init__()
        self.layers_n, self.heads = layers, heads
        self.vec_in = nn.Linear(D + 1, width)  # (direction, log length)
        self.kind = nn.Parameter(torch.randn(kinds, width) * 0.02)
        self.depth = nn.Parameter(torch.randn(depths + 1, width) * 0.02)  # (+ 1: a discriminator's child nodes)
        self.pair = nn.Sequential(nn.Linear(5, pair_hidden), nn.GELU(), nn.Linear(pair_hidden, layers * heads))
        self.blocks = nn.ModuleList(GeoLayer(width, heads) for _ in range(layers))
        self.out_norm = nn.LayerNorm(width)

    def forward(self, S: Tensor, exists: Tensor, kind: Tensor, d: Tensor) -> Tensor:
        B, T, _ = S.shape
        S = S.float()
        e = exists.float()
        n = S.norm(dim=-1).clamp_min(EPS)
        u = S / n[..., None] * e[..., None]
        ln = n.log() * e
        with torch.autocast(S.device.type, enabled=False):  # (the geometry in float32, in the vectors' dimension)
            cos = u @ u.transpose(1, 2)
        li, lj = ln[:, :, None].expand(B, T, T), ln[:, None, :].expand(B, T, T)
        both = e[:, :, None] * e[:, None, :]
        bias = self.pair(torch.stack([cos, li, lj, li - lj, both], -1)).permute(0, 3, 1, 2)  # [B, L * H, T, T]
        bias = bias + (1 - e)[:, None, None, :] * -1e4  # (keys that do not exist)
        h = self.vec_in(torch.cat([u, ln[..., None]], -1)) + self.kind[kind] + self.depth[d][:, None]
        for i, blk in enumerate(self.blocks):
            h = blk(h, bias[:, i * self.heads:(i + 1) * self.heads])
        return self.out_norm(h)


__all__ = ["GeoEncoder", "GeoLayer", "NodeEncoder", "QueryReader", "directions"]
