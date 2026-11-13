"""A prior over a V-way tree's codes (train_inner_vway.py), and a discriminator of the same shape
(train_vprior.py).

A code is a token in 0..V-1 at each depth. CodeNet reads, at every depth, the token taken at the depth before (none
at the root), the tree's own shares p there (what the tree's networks predict for that node's cells, log) and the
depth's position; causal attention lets depth k see its own inputs and everything before, never its own token. Its
head gives V logits per depth. As the prior they are p(token k | the code before it); as the discriminator, each
token's trust logit: how likely that token, after that code, came from real codes rather than the prior. The head
starts at zero, so an untrained prior is uniform and an untrained discriminator trusts everything one half.

CodeQA decodes codes sampled from a prior into images (the tree replayed with the sampled tokens), for the run's plot.
"""

import torch
import torch.nn as nn
from torch import Tensor

from dvae.data import load_images
from dvae.deltatree import Block
from dvae.outer import load_outer
from dvae.vreplay import VTree


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


class CodeQA:
    """Images for judging a prior by eye: a few test images, their reconstructions by the tree's first K depths, and
    codes sampled from a prior and from the tree's own shares, each decoded by the tree (dvae/vreplay.py) and the
    outer decoder of the latents the tree coded."""

    def __init__(self, tree_run: str, device: torch.device, K: int, n: int = 10, fp16: bool = False,
                 data_root: str = "data", seed: int = 0):
        self.tree = VTree(tree_run, device, K, fp16=fp16)
        self.K, self.n, self.device, self.seed = self.tree.K, n, device, seed
        self.model, self.lat = load_outer(self.tree.args["outer"], device)
        test = self.lat["test"]
        idx = torch.randperm(len(test), generator=torch.Generator().manual_seed(seed))[:n]
        Z = test[idx].float().flatten(1).to(device)
        self.real = (load_images(self.lat["dataset"], "test", data_root)[idx].float() + 0.5) / 256
        g, _ = self.tree.code(Z, self.K)
        self.recon = self.decode(Z - g.R)

    @torch.no_grad()
    def decode(self, z: Tensor) -> Tensor:
        """Scaled latents [n, D] -> images [n, C, H, W] in [0, 1] on the host."""
        C, h, w = self.lat["shape"]
        mean = self.lat["mean"].to(z.device)[None, :, None, None]
        std = self.lat["std"].to(z.device)[None, :, None, None]
        self.model.decoder.eval()
        return self.model.decoder(z.float().view(-1, C, h, w) / self.lat["scale"] * std + mean).clamp(0, 1).cpu()

    @torch.no_grad()
    def samples(self, prior: CodeNet) -> dict:
        """n codes sampled from prior (each token from its distribution given the code before it and the tree's
        shares there) and n from the tree's shares alone, decoded; with the test images and their reconstructions
        (the same seed every call, so that calls compare)."""
        was = prior.training
        prior.eval()
        gen = torch.Generator(self.device).manual_seed(self.seed + 1)
        V = prior.V
        log_shares = torch.zeros(self.n, self.K, V, device=self.device)

        def by_prior(d, tokens, shares):
            log_shares[:, d] = shares.float().clamp_min(1e-6).log()
            full = torch.cat([tokens, tokens.new_zeros(self.n, self.K - d)], 1)  # (causal: the rest is unread)
            logits = prior(full, log_shares)[:, d].float()
            return torch.multinomial(logits.softmax(-1), 1, generator=gen)[:, 0]

        def by_shares(d, tokens, shares):
            return torch.multinomial(shares.float().clamp_min(0) + 1e-12, 1, generator=gen)[:, 0]

        _, z_prior = self.tree.sample(self.n, by_prior, self.K)
        _, z_shares = self.tree.sample(self.n, by_shares, self.K)
        prior.train(was)
        return dict(real=self.real, recon=self.recon, prior=self.decode(z_prior), shares=self.decode(z_shares),
                    K=self.K)


__all__ = ["CodeNet", "CodeQA"]
