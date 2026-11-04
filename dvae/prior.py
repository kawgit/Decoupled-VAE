"""A second stage for a delta tree: a prior over its codes, p(token k is rare | tokens before it).

The tree (stage 1) stays frozen, so every image has a fixed code, and at every node on its path the tree's
split has fixed answers: the rare side's mean delta r and the ratio m. The prior reads a code's splits in order,
one token per depth: that depth's split (r's direction and log length, and m) plus the side taken at the depth
before (none at the root). Causal attention lets token k see its own split and everything earlier, but not its
own side, which its head predicts as a logit. So one pass over a code trains every depth, the same as asking a
decoder with a side head at each node in turn.

With images=True, each token also shows the two means it chooses between, as pictures: the node's predicted mean
(the training mean plus the deltas before it) plus r (the rare side's) and minus m r (the common side's), as one
two-channel 28x28 image read by a small convolutional net. That adds no information (both follow from the code
so far) but puts it in a form where strokes are local, as a person comparing the two would see them.
"""

import torch
import torch.nn as nn
from torch import Tensor

from dvae.deltatree import EPS, Block


class SideMeans(nn.Module):
    """Features [..., W] of the two side means [..., 2, D], read as a two-channel 28x28 image."""

    def __init__(self, width, channels=32):
        super().__init__()
        c = channels
        self.net = nn.Sequential(nn.Conv2d(2, c, 3, stride=2, padding=1), nn.GELU(),  # 14 x 14
                                 nn.Conv2d(c, 2 * c, 3, stride=2, padding=1), nn.GELU(),  # 7 x 7
                                 nn.Conv2d(2 * c, 2 * c, 3, padding=1), nn.GELU(),
                                 nn.Flatten(), nn.Linear(2 * c * 49, width))

    def forward(self, images):
        lead = images.shape[:-2]
        return self.net(images.reshape(-1, 2, 28, 28)).reshape(*lead, -1)


class SidePrior(nn.Module):
    mean: Tensor  # [D] the tree's training mean, with images

    def __init__(self, depths, width=256, layers=4, heads=4, dropout=0.0, D=784, images=False):
        super().__init__()
        self.config = dict(depths=depths, width=width, layers=layers, heads=heads, dropout=dropout, D=D,
                           images=images)
        self.split_in = nn.Linear(D + 2, width)
        self.means_in = SideMeans(width) if images else None
        if images:
            self.register_buffer("mean", torch.zeros(D))
        self.side_in = nn.Embedding(3, width)  # the side before: common, rare, or none (the root)
        self.position = nn.Parameter(torch.randn(depths, width) * 0.02)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(Block(width, heads, dropout) for _ in range(layers))
        self.out_norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, 1)

    def forward(self, r: Tensor, m: Tensor, rare: Tensor) -> Tensor:
        """Logits [B, k] that each token is rare, for codes' splits r [B, k, D], m [B, k] and sides rare [B, k]
        (bool; only those before each token are read, so the last may be anything)."""
        B, k = m.shape
        length = r.norm(dim=-1, keepdim=True).clamp_min(EPS)
        before = torch.cat([rare.new_full((B, 1), 2, dtype=torch.long), rare[:, :-1].long()], 1)
        h = self.split_in(torch.cat([r / length, length.log(), m[..., None]], -1)) + self.side_in(before)
        if self.means_in is not None:
            common = -m[..., None] * r
            delta = torch.where(rare[..., None], r, common)
            mu = self.mean + torch.cumsum(delta, 1) - delta  # each node's predicted mean, from the deltas before it
            h = h + self.means_in(torch.stack([mu + r, mu + common], 2))
        h = self.drop(h + self.position[:k])
        for blk in self.blocks:
            h, _ = blk(h, causal=True)
        return self.head(self.out_norm(h))[..., 0]


def band_edges(depths, bands):
    """bands + 1 depth indices cutting 0..depths into bands of (nearly) equal width."""
    return [round(depths * i / bands) for i in range(bands + 1)]


class BandedPrior(nn.Module):
    """Several snapshots of one SidePrior, each answering for its own band of depths (edges [bands + 1]) and
    snapshot[i] being band i's. Being causal, a band's snapshot only reads the codes up to the band's end."""

    def __init__(self, priors: list[SidePrior], edges: list[int]):
        super().__init__()
        self.priors, self.edges = nn.ModuleList(priors), edges
        self.config = priors[0].config

    def forward(self, r: Tensor, m: Tensor, rare: Tensor) -> Tensor:
        k = m.shape[1]
        logits = m.new_zeros(m.shape)
        for prior, lo, hi in zip(self.priors, self.edges, self.edges[1:]):
            if lo < k:
                hi = min(hi, k)
                logits[:, lo:hi] = prior(r[:, :hi], m[:, :hi], rare[:, :hi])[:, lo:hi]
        return logits


def load_prior(ckpt, device):
    """A trained prior from its checkpoint (train_prior.py's), in eval mode: a SidePrior, or with per-band early
    stopping a BandedPrior of each band's kept snapshot."""
    def snapshot(state):
        prior = SidePrior(**ckpt["config"]).to(device)
        prior.load_state_dict(state)
        return prior

    if "bands" not in ckpt:
        return snapshot(ckpt["state_dict"]).eval()
    priors = [snapshot(ckpt["states"][epoch]) for epoch in ckpt["band_epochs"]]
    return BandedPrior(priors, ckpt["bands"]).eval()
