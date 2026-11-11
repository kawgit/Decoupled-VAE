"""The decoder's network (train_decoder.py): TokenDecoder, p(x | leaf), reading a leaf's history through History.

A node's history is one token per child taken on its path: the child's delta (the offset of its centroid from
its parent's predicted mean) and its token. A node's predicted mean is the training mean plus those deltas.
"""

import torch
import torch.nn as nn
from torch import Tensor

from dvae.deltatree import D, EPS, Block, ImageHead, MeanImage


class History(nn.Module):
    """A node's memory: its history, one token per child taken (its delta's direction and log length, and its
    token), with self-attention among them. A query reads it and itself at every layer (Block
    with mem), so a node's memory is computed once for all its queries. With siblings, each history token also
    reads every child's offset at that depth (direction and log length, plus a slot embedding, summed over the
    children), what dvae/deltatree.py's side and m tell about the other side, for any V. With node_mean, the
    memory also holds a token for the node's predicted mean image (the training mean, the buffer mean, plus the
    deltas)."""

    mean: Tensor  # [D] the training mean, with node_mean

    def __init__(self, depth, V, width, layers, heads, node_mean=False, siblings=False, pixels=D):
        super().__init__()
        self.width, self.D = width, pixels
        self.delta_in = nn.Linear(pixels + 1, width)
        self.siblings = siblings
        if siblings:
            self.kid_in = nn.Linear(pixels + 1, width)
            self.kid_slot = nn.Parameter(torch.randn(V, width) * 0.02)
        self.token_in = nn.Embedding(V, width)
        self.position = nn.Parameter(torch.randn(max(1, depth), width) * 0.02)
        self.blocks = nn.ModuleList(Block(width, heads) for _ in range(layers))
        self.mean_in = MeanImage(width) if node_mean else None
        self.extra = int(node_mean)
        if node_mean:
            assert pixels == D, "node_mean reads 28 x 28 images"
            self.mean_token = nn.Parameter(torch.randn(width) * 0.02)
            self.register_buffer("mean", torch.zeros(D))

    def memory(self, deltas, tokens, kids=None):
        """Each layer's normalized memory tokens [B, extra + k, W] for histories deltas [B, k, D] and tokens
        [B, k], and with siblings every child's offset kids [B, k, V, D]."""
        B, k = tokens.shape
        length = deltas.norm(dim=-1, keepdim=True).clamp_min(EPS)
        h = self.delta_in(torch.cat([deltas / length, length.log()], -1))
        h = h + self.token_in(tokens) + self.position[:k]
        if self.siblings and k > 0:
            assert kids is not None, "a siblings history needs every child's offset"
            size = kids.norm(dim=-1, keepdim=True).clamp_min(EPS)
            h = h + (self.kid_in(torch.cat([kids / size, size.log()], -1)) + self.kid_slot).sum(2)
        if self.mean_in is not None:
            h = torch.cat([(self.mean_in(self.mean + deltas.sum(1)) + self.mean_token)[:, None], h], 1)
        if h.shape[1] == 0:
            return [deltas.new_zeros(B, 0, self.width) for _ in self.blocks]
        mems = []
        for blk in self.blocks:
            h, x = blk(h)
            mems.append(x)
        return mems


class TokenDecoder(History):
    """p(x | leaf) for a tree's codes, with History's memory (each child's delta and token). 256 logits per
    pixel, the sum of a path reading the history with a learned query and a few convolutions over the leaf's
    predicted mean image."""

    def __init__(self, depth, V=2, width=256, layers=4, heads=4, channels=64, shape=(1, 28, 28)):
        shape = tuple(shape)
        C, H, W = shape
        super().__init__(depth, V, width, layers, heads, pixels=C * H * W)
        self.config = dict(depth=depth, V=V, width=width, layers=layers, heads=heads, channels=channels,
                           shape=shape)
        self.shape = shape
        self.query_token = nn.Parameter(torch.randn(1, 1, width) * 0.02)
        self.out_norm = nn.LayerNorm(width)
        self.levels_head = ImageHead(width, 256, shape=shape)
        self.mean_path = nn.Sequential(nn.Conv2d(C, channels, 3, padding=1), nn.GELU(),
                                       nn.Conv2d(channels, channels, 3, padding=1), nn.GELU(),
                                       nn.Conv2d(channels, channels, 3, padding=1), nn.GELU(),
                                       nn.Conv2d(channels, 256 * C, 1))

    def forward(self, deltas, tokens, mu):
        """Logits [B, 256, D] for leaves with histories deltas [B, k, D], tokens [B, k] and predicted means
        mu [B, D]."""
        q = self.query_token.expand(len(mu), 1, -1)
        for blk, mem in zip(self.blocks, self.memory(deltas, tokens)):
            q, _ = blk(q, mem)
        logits = self.levels_head(self.out_norm(q))[:, 0]
        # (the convolution's channels are level-major, as ImageHead's: channel l * C + c)
        return logits + self.mean_path(mu.view(-1, *self.shape)).reshape(len(mu), 256, -1)
