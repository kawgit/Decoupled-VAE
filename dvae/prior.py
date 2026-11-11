"""A second stage for a delta tree: a prior over its codes, p(token k | tokens before it).

The tree (stage 1) stays frozen, so every image has a fixed code, and at every node on its path the tree's
split has fixed children: each child's centroid, an offset from the node's predicted mean (r for the rare side,
-m r for the common one). The prior reads a code's nodes in order, one token per depth: that node's children plus
the token taken at the depth before (none at the root). Causal attention lets token k see its own node's children
and everything earlier, but not its own token, which its head predicts as V logits. So one pass over a code
trains every depth, the same as asking a network with a head at each node in turn.

With images=True, each token also shows the children's centroids it chooses between, as pictures: the node's
predicted mean (the training mean plus the offsets of the children taken before it) plus each child's offset, as
one V-channel image read by a small convolutional net. That adds no information (they follow from the code so
far) but puts it in a form where strokes are local, as a person comparing them would see them.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from dvae.codes import LazyRows
from dvae.deltatree import EPS, Block


class SideMeans(nn.Module):
    """Features [..., W] of the count side means [..., count, D] (two by default), read as a count-channel 28x28
    image."""

    def __init__(self, width, channels=32, count=2, shape=(1, 28, 28)):
        super().__init__()
        c = channels
        self.count, self.shape = count, tuple(shape)
        C, H, W = self.shape
        self.net = nn.Sequential(nn.Conv2d(count * C, c, 3, stride=2, padding=1), nn.GELU(),  # H / 2
                                 nn.Conv2d(c, 2 * c, 3, stride=2, padding=1), nn.GELU(),  # H / 4
                                 nn.Conv2d(2 * c, 2 * c, 3, padding=1), nn.GELU(),
                                 nn.Flatten(), nn.Linear(2 * c * ((H + 3) // 4) * ((W + 3) // 4), width))

    def forward(self, images):
        lead = images.shape[:-2]
        return self.net(images.reshape(-1, self.count * self.shape[0], *self.shape[1:])).reshape(*lead, -1)


def band_edges(depths, bands):
    """bands + 1 depth indices cutting 0..depths into bands of (nearly) equal width."""
    return [round(depths * i / bands) for i in range(bands + 1)]


class TokenPrior(nn.Module):
    """A prior over a tree's codes (dvae/codes.py's TreeCodes): p(token k | tokens before it), V-way. Each token
    reads its node's V children (each centroid offset's direction and log length plus a slot embedding, summed
    over the children) and the token before (none at the root); causal attention
    lets it see everything earlier but not its own token. It does not read the encoder's implied shares (they are
    train_prior.py's baseline); its head starts at zero, so an untrained prior is uniform. With images=True, each
    token also shows its children's centroids as images (the node's predicted mean plus each offset), one
    V-channel 28x28 image read by a small convolutional net."""

    mean: Tensor  # [D] the tree's training mean, with images

    def __init__(self, depths, V=2, width=256, layers=4, heads=4, dropout=0.0, D=784, images=False,
                 shape=(1, 28, 28)):
        super().__init__()
        shape = tuple(shape)
        assert D == shape[0] * shape[1] * shape[2], (D, shape)
        self.config = dict(depths=depths, V=V, width=width, layers=layers, heads=heads, dropout=dropout, D=D,
                           images=images, shape=shape)
        self.child_in = nn.Linear(D + 1, width)
        self.slot = nn.Parameter(torch.randn(V, width) * 0.02)
        self.means_in = SideMeans(width, count=V, shape=shape) if images else None
        if images:
            self.register_buffer("mean", torch.zeros(D))
        self.token_in = nn.Embedding(V + 1, width)  # the token before, or none (V, the root)
        self.position = nn.Parameter(torch.randn(depths, width) * 0.02)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(Block(width, heads, dropout) for _ in range(layers))
        self.out_norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, V)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        self.V = V

    def forward(self, offsets: Tensor, tokens: Tensor) -> Tensor:
        """Logits [B, k, V] for each token of codes with children offsets [B, k, V, D] and tokens [B, k] (only
        those before each token are read, so the last may be anything)."""
        B, k = tokens.shape
        length = offsets.norm(dim=-1, keepdim=True).clamp_min(EPS)
        h = (self.child_in(torch.cat([offsets / length, length.log()], -1)) + self.slot).sum(2)
        before = torch.cat([tokens.new_full((B, 1), self.V), tokens[:, :-1]], 1)
        h = h + self.token_in(before)
        if self.means_in is not None:
            taken = offsets.gather(2, tokens[:, :, None, None].expand(B, k, 1, offsets.shape[-1]))[:, :, 0]
            mu = self.mean + torch.cumsum(taken, 1) - taken  # each node's predicted mean, from the tokens before it
            h = h + self.means_in(mu[:, :, None] + offsets)
        h = self.drop(h + self.position[:k])
        for blk in self.blocks:
            h, _ = blk(h, causal=True)
        return self.head(self.out_norm(h))


class BandedTokenPrior(nn.Module):
    """Several snapshots of one TokenPrior, each answering for its own band of depths (edges [bands + 1]) and
    snapshot[i] being band i's. Being causal, a band's snapshot only reads the codes up to the band's end."""

    def __init__(self, priors: list[TokenPrior], edges: list[int]):
        super().__init__()
        self.priors, self.edges = nn.ModuleList(priors), edges
        self.config = priors[0].config

    def forward(self, offsets: Tensor, tokens: Tensor) -> Tensor:
        k = tokens.shape[1]
        logits = offsets.new_zeros(offsets.shape[:3])
        for prior, lo, hi in zip(self.priors, self.edges, self.edges[1:]):
            if lo < k:
                hi = min(hi, k)
                logits[:, lo:hi] = prior(offsets[:, :hi], tokens[:, :hi])[:, lo:hi]
        return logits


def load_token_prior(ckpt, device):
    """train_prior.py's checkpoint as a TokenPrior, or with per-band early stopping a BandedTokenPrior of each
    band's kept snapshot, in eval mode."""
    def snapshot(state):
        prior = TokenPrior(**ckpt["config"]).to(device)
        prior.load_state_dict(state)
        return prior

    if "bands" not in ckpt:
        return snapshot(ckpt["state_dict"]).eval()
    priors = [snapshot(ckpt["states"][epoch]) for epoch in ckpt["band_epochs"]]
    return BandedTokenPrior(priors, ckpt["bands"]).eval()


@torch.no_grad()
def code_token_bits(prior: TokenPrior | BandedTokenPrior, offsets: "Tensor | LazyRows", tokens: Tensor, device,
                    batch=256) -> Tensor:
    """The bits [N, K] each token of N codes costs under the prior (inputs as TokenPrior's, on any device), on
    the CPU."""
    prior.eval()
    out = []
    for i in range(0, len(tokens), batch):
        tb = tokens[i:i + batch].to(device)
        logits = prior(offsets[i:i + batch].to(device).float(), tb)
        out.append(F.cross_entropy(logits.flatten(0, 1).float(), tb.flatten(), reduction="none")
                   .view(tb.shape).cpu() / math.log(2))
    return torch.cat(out)
