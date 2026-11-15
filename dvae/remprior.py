"""A prior over a V-way tree's codes (train_inner_vway.py) that models what the next token is about: the image's
remainder at the node. RemPrior reads the node's current mean (the reconstruction so far, as the latents' own
C x H x W picture) and the last delta taken, with a convolutional net conditioned on the depth (FiLM), and predicts
the remainder's distribution there: a Gaussian with a mean and a variance per latent number and, with rank r > 0, r
more directions of shared variation (a low-rank-plus-diagonal covariance, so that a projection's variance can differ
from the sum of its numbers' variances). It is trained on the remainders themselves (their negative log likelihood:
every latent number of every (image, depth) pair supervises it), in units of each depth's rms remainder, so that every
depth's targets are of one size. Its output starts at zero: a Gaussian of each depth's rms at the node's mean, the
isotropic baseline.

A token is the cell of the remainder among the partition centroids q (the last Lloyd step's queries: the nearest),
so the prior's probability of each token is the probability that a remainder drawn from the predicted Gaussian falls
in that cell: forward estimates it from `samples` draws, each assigned its nearest centroid (counts plus one half,
normalized). Sampling a code from the prior is drawing a token from that, at every depth (VTree.sample).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from dvae.vprior import NodeInputs


class FiLMBlock(nn.Module):
    """A residual block of two 3x3 convolutions, the first's output scaled and shifted by the conditioning."""

    def __init__(self, channels, cond):
        super().__init__()
        self.n1, self.n2 = nn.GroupNorm(8, channels), nn.GroupNorm(8, channels)
        self.c1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.c2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.film = nn.Linear(cond, 2 * channels)
        nn.init.zeros_(self.c2.weight)
        assert self.c2.bias is not None
        nn.init.zeros_(self.c2.bias)

    def forward(self, h: Tensor, e: Tensor) -> Tensor:
        scale, shift = self.film(e)[:, :, None, None].chunk(2, 1)
        y = self.c1(F.silu(self.n1(h)))
        y = self.n2(y) * (1 + scale) + shift
        return h + self.c2(F.silu(y))


class RemPrior(nn.Module):
    scale: Tensor  # [depths + 1] each depth's rms remainder per latent number (a buffer)

    def __init__(self, depths, V=8, shape=(2, 16, 16), channels=64, blocks=6, rank=0, samples=256):
        super().__init__()
        shape = tuple(shape)
        self.config = dict(kind="rem_prior", depths=depths, V=V, shape=shape, channels=channels, blocks=blocks,
                           rank=rank, samples=samples)
        self.V, self.shape, self.rank, self.samples = V, shape, rank, samples
        self.window = 1  # (CandStore.inputs' window: the last delta is all it reads)
        C = shape[0]
        self.D = C * shape[1] * shape[2]
        self.register_buffer("scale", torch.ones(depths + 1))  # each depth's rms remainder (set_scales)
        self.depth = nn.Embedding(depths + 1, 128)
        self.stem = nn.Conv2d(2 * C, channels, 3, padding=1)
        self.blocks = nn.ModuleList(FiLMBlock(channels, 128) for _ in range(blocks))
        self.out_norm = nn.GroupNorm(8, channels)
        self.out = nn.Conv2d(channels, C * (2 + rank), 3, padding=1)
        nn.init.zeros_(self.out.weight)
        assert self.out.bias is not None
        nn.init.zeros_(self.out.bias)

    def set_scales(self, rms: Tensor):
        """Each depth's rms remainder [depths] (per latent number), from training data."""
        self.scale[:len(rms)] = rms.to(self.scale)
        self.scale[len(rms):] = rms[-1]

    def dist(self, x: NodeInputs) -> tuple[Tensor, Tensor, Tensor | None]:
        """The remainder's predicted mean [B, D], log variance per number [B, D] and shared directions [B, D, r]
        (None with rank 0), in the latents' units."""
        B = len(x.d)
        C, H, W = self.shape
        s = self.scale[x.d]
        last = x.win[:, -1] / s[:, None]
        h = self.stem(torch.cat([x.mean, last], 1).view(B, 2 * C, H, W))
        e = self.depth(x.d)
        for blk in self.blocks:
            h = blk(h, e)
        o = self.out(F.silu(self.out_norm(h))).float().view(B, 2 + self.rank, self.D)
        mu = o[:, 0] * s[:, None]
        logvar = o[:, 1].clamp(-12, 6) + 2 * s.log()[:, None]
        L = o[:, 2:].transpose(1, 2) * s[:, None, None] if self.rank else None
        return mu, logvar, L

    def nll(self, x: NodeInputs, R: Tensor) -> Tensor:
        """The remainders R [B, D]'s negative log likelihood, nats per latent number [B]."""
        mu, logvar, L = self.dist(x)
        if L is None:
            return 0.5 * (logvar + (R - mu).square() / logvar.exp() + math.log(2 * math.pi)).mean(-1)
        with torch.autocast(mu.device.type, enabled=False):  # (its Cholesky factorization: float32 only)
            g = torch.distributions.LowRankMultivariateNormal(mu.float(), L.float(), logvar.float().exp())
            return -g.log_prob(R.float()) / self.D

    @torch.no_grad()
    def token_logprobs(self, x: NodeInputs, samples: int | None = None, chunk: int = 128) -> Tensor:
        """Each token's log probability [B, V]: the share of remainders drawn from the predicted Gaussian that fall
        in its cell (nearest of the inputs' parts), counts plus one half."""
        assert x.parts is not None, "RemPrior's tokens are cells of the partition centroids: a store that keeps them"
        S = samples or self.samples
        mu, logvar, L = self.dist(x)
        sd = (0.5 * logvar).exp()
        q = x.parts.float()
        qq = q.square().sum(-1)
        out = []
        for i in range(0, len(mu), chunk):
            j = slice(i, i + chunk)
            Rs = mu[j, None] + sd[j, None] * torch.randn(mu[j].shape[0], S, self.D, device=mu.device)
            if L is not None:
                Rs = Rs + torch.einsum("bdr,bsr->bsd", L[j], torch.randn(mu[j].shape[0], S, self.rank,
                                                                           device=mu.device))
            cell = (qq[j, None] - 2 * torch.einsum("bsd,bvd->bsv", Rs, q[j])).argmin(-1)
            counts = F.one_hot(cell, self.V).sum(1).float()
            out.append(((counts + 0.5) / (S + 0.5 * self.V)).log())
        return torch.cat(out)

    def forward(self, x: NodeInputs) -> Tensor:
        """Log probabilities [B, V] (as logits: a prior's interface, CodeQA.samples')."""
        return self.token_logprobs(x)


__all__ = ["FiLMBlock", "RemPrior"]
