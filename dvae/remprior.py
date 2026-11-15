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

    tau0: Tensor  # [depths + 1] each depth's soft-assignment temperature scale (a buffer, set_taus)

    def __init__(self, depths, V=8, shape=(2, 16, 16), channels=64, blocks=6, rank=0, samples=256, history=1,
                 soft=False):
        """history: the last deltas read beside the mean. soft: token probabilities from a soft assignment of the
        drawn remainders to cells (each a softmax of minus its squared distances over a learned temperature per
        depth), differentiable, so that the tokens' cross-entropy trains the net (token_logprobs_soft); otherwise
        the hard nearest-centroid counts."""
        super().__init__()
        shape = tuple(shape)
        self.config = dict(kind="rem_prior", depths=depths, V=V, shape=shape, channels=channels, blocks=blocks,
                           rank=rank, samples=samples, history=history, soft=soft)
        self.soft = soft
        self.V, self.shape, self.rank, self.samples = V, shape, rank, samples
        self.window = history  # (CandStore.inputs' window)
        C = shape[0]
        self.D = C * shape[1] * shape[2]
        self.register_buffer("scale", torch.ones(depths + 1))  # each depth's rms remainder (set_scales)
        self.depth = nn.Embedding(depths + 1, 128)
        self.stem = nn.Conv2d((1 + history) * C, channels, 3, padding=1)
        # (calibration of the sampled token probabilities: a temperature and a bias per token at each depth, fit
        # by cross-entropy on training tokens once the remainder model is trained; at first the identity)
        self.temp = nn.Parameter(torch.ones(depths + 1))
        self.bias = nn.Parameter(torch.zeros(depths + 1, V))
        self.register_buffer("tau0", torch.ones(depths + 1))
        self.log_tau = nn.Parameter(torch.zeros(depths + 1))
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
        recent = (x.win / s[:, None, None]).flatten(1)
        h = self.stem(torch.cat([x.mean, recent], 1).view(B, (1 + self.window) * C, H, W))
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

    def set_taus(self, gap: Tensor):
        """Each depth's typical gap [depths] between a remainder's nearest and second nearest centroid (squared
        distances): the soft assignment's temperature starts at it."""
        self.tau0[:len(gap)] = gap.to(self.tau0)
        self.tau0[len(gap):] = gap[-1]

    def token_logprobs_soft(self, x: NodeInputs, samples: int = 16) -> Tensor:
        """Each token's log probability [B, V], differentiable: the mean over remainders drawn from the predicted
        Gaussian of a softmax of minus their squared distances to the cells' centroids over the depth's
        temperature."""
        assert x.parts is not None, "RemPrior's tokens are cells of the partition centroids: a store that keeps them"
        mu, logvar, L = self.dist(x)
        B = len(mu)
        Rs = mu[:, None] + (0.5 * logvar).exp()[:, None] * torch.randn(B, samples, self.D, device=mu.device)
        if L is not None:
            Rs = Rs + torch.einsum("bdr,bsr->bsd", L, torch.randn(B, samples, self.rank, device=mu.device))
        q = x.parts.float()
        d2 = q.square().sum(-1)[:, None] - 2 * torch.einsum("bsd,bvd->bsv", Rs, q)  # (|R|^2 left out: per draw)
        tau = (self.tau0[x.d] * self.log_tau[x.d].exp())[:, None, None]
        return (-d2 / tau).log_softmax(-1).logsumexp(1) - math.log(samples)

    def calibrate(self, raw: Tensor, d: Tensor) -> Tensor:
        """Sampled log probabilities raw [B, V] (token_logprobs') at depths d [B], calibrated: log softmax of
        temp[d] raw + bias[d]."""
        return (self.temp[d, None] * raw + self.bias[d]).log_softmax(-1)

    @torch.no_grad()
    def token_logprobs(self, x: NodeInputs, samples: int | None = None, chunk: int = 128) -> Tensor:
        """Each token's log probability [B, V], uncalibrated: the share of remainders drawn from the predicted
        Gaussian that fall in its cell (nearest of the inputs' parts), counts plus one half."""
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
        """Calibrated log probabilities [B, V] (as logits: a prior's interface, CodeQA.samples')."""
        if self.soft:
            with torch.no_grad():
                return self.calibrate(self.token_logprobs_soft(x, self.samples), x.d)
        return self.calibrate(self.token_logprobs(x), x.d)


__all__ = ["FiLMBlock", "RemPrior"]
