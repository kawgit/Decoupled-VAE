"""A second stage for a delta tree: a prior over its codes, p(token k is rare | tokens before it).

The tree (stage 1) stays frozen, so every image has a fixed code, and at every node on its path the tree's
split has fixed answers: the rare side's mean delta r and the ratio m. The prior reads a code's splits in order,
one token per depth: that depth's split (r's direction and log length, and m) plus the side taken at the depth
before (none at the root). Causal attention lets token k see its own split and everything earlier, but not its
own side, which its head predicts as a logit. So one pass over a code trains every depth, the same as asking a
network with a side head at each node in turn.

With images=True, each token also shows the two means it chooses between, as pictures: the node's predicted mean
(the training mean plus the deltas before it) plus r (the rare side's) and minus m r (the common side's), as one
two-channel 28x28 image read by a small convolutional net. That adds no information (both follow from the code
so far) but puts it in a form where strokes are local, as a person comparing the two would see them.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from dvae.deltatree import EPS, Block


class SideMeans(nn.Module):
    """Features [..., W] of the count side means [..., count, D] (two by default), read as a count-channel 28x28
    image."""

    def __init__(self, width, channels=32, count=2):
        super().__init__()
        c = channels
        self.count = count
        self.net = nn.Sequential(nn.Conv2d(count, c, 3, stride=2, padding=1), nn.GELU(),  # 14 x 14
                                 nn.Conv2d(c, 2 * c, 3, stride=2, padding=1), nn.GELU(),  # 7 x 7
                                 nn.Conv2d(2 * c, 2 * c, 3, padding=1), nn.GELU(),
                                 nn.Flatten(), nn.Linear(2 * c * 49, width))

    def forward(self, images):
        lead = images.shape[:-2]
        return self.net(images.reshape(-1, self.count, 28, 28)).reshape(*lead, -1)


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
            delta = torch.where(rare[..., None], r, common)  # dvae/codes.py's deltas
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


@torch.no_grad()
def token_bits(prior: SidePrior | BandedPrior, rare: Tensor, m: Tensor, r: Tensor, device, batch=256) -> Tensor:
    """The bits [N, K] each token of N codes (rare [N, K], m [N, K], r [N, K, D], on any device) costs under
    the prior, on the CPU."""
    prior.eval()
    out = []
    for i in range(0, len(rare), batch):
        rb = rare[i:i + batch].to(device)
        logits = prior(r[i:i + batch].to(device).float(), m[i:i + batch].to(device).float(), rb)
        out.append(F.binary_cross_entropy_with_logits(logits, rb.float(), reduction="none").cpu() / math.log(2))
    return torch.cat(out)


class TokenPrior(nn.Module):
    """A prior over a refining tree's codes (dvae/refine.py, dvae/codes.py's TokenCodes): p(token k | tokens
    before it), V-way. Each token reads its node's V children (each centroid offset's direction and log length
    plus a slot embedding, summed over the children) and the token before (none at the root); causal attention
    lets it see everything earlier but not its own token. It does not read the encoder's implied shares (they are
    train_prior.py's baseline); its head starts at zero, so an untrained prior is uniform. With images=True, each
    token also shows its children's centroids as images (the node's predicted mean plus each offset), one
    V-channel 28x28 image read by a small convolutional net."""

    mean: Tensor  # [D] the tree's training mean, with images

    def __init__(self, depths, V=2, width=256, layers=4, heads=4, dropout=0.0, D=784, images=False):
        super().__init__()
        self.config = dict(depths=depths, V=V, width=width, layers=layers, heads=heads, dropout=dropout, D=D,
                           images=images)
        self.child_in = nn.Linear(D + 1, width)
        self.slot = nn.Parameter(torch.randn(V, width) * 0.02)
        self.means_in = SideMeans(width, count=V) if images else None
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
    """BandedPrior for TokenPriors: snapshot i answers for band i's depths (edges [bands + 1])."""

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
def code_token_bits(prior: TokenPrior | BandedTokenPrior, offsets: Tensor, tokens: Tensor, device,
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
