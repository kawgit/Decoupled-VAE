"""A prior over a V-way tree's codes (train_inner_vway.py), and a discriminator of the same shape (train_vprior.py),
built as the tree's own networks are (dvae/vway.py's VStepNet).

At depth d of an image's code, CandNet reads the image's delta history as memory: the last `window` deltas the code
took (each as its direction, log length, and whether it exists: a shallow depth has fewer) and one early token, the
sum of every delta before the window. The tree's V candidates there (the cells' predicted means, the deltas the code
could take next) are its queries: each one's direction and log length, an embedding of its slot and of the depth;
they attend to the memory and to each other at every layer, and each gives one logit. As the prior, the softmax over
the V logits is p(token | history); as the discriminator, each logit is that candidate's trust logit: how likely
taking it after this history is the data's choice rather than the prior's. Token ids are never read: what a token
means is its candidate. The head starts at zero, so an untrained prior is uniform and an untrained discriminator
trusts everything one half. One network serves every depth.

CandStore holds replayed codes (dvae/vreplay.py's VTree) of n images at every depth: the V candidates in 8 bits (a
scale per image and depth), the token taken, the tree's shares, and the running sum of the deltas taken (for the
early token). CodeQA decodes codes sampled from a prior into images, for the run's plot.
"""

import torch
import torch.nn as nn
from torch import Tensor

from dvae.data import load_images
from dvae.deltatree import EPS, Block
from dvae.outer import load_outer
from dvae.vreplay import VTree


class CandStore:
    def __init__(self, n: int, K: int, V: int, D: int, device: torch.device):
        self.n, self.K, self.V, self.D = n, K, V, D
        self.cand = torch.zeros(n, K, V, D, dtype=torch.int8, device=device)
        self.scale = torch.zeros(n, K, device=device)
        self.tok = torch.zeros(n, K, dtype=torch.uint8, device=device)
        self.share = torch.zeros(n, K, V, dtype=torch.float16, device=device)
        self.cum = torch.zeros(n, K + 1, D, dtype=torch.float16, device=device)  # [:, j]: the deltas before j

    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in (self.cand, self.scale, self.tok, self.share, self.cum))

    def writer(self, rows: Tensor):
        """A visit(d, cands [m, V, D], tokens [m], shares [m, V]) for VTree.code and VTree.sample that stores rows'
        depths in order (its running sum of the deltas taken kept in float32)."""
        running = torch.zeros(len(rows), self.D, device=self.cand.device)

        def visit(d, cands, tokens, shares):
            cands = cands.float()
            s = cands.abs().amax((1, 2)).clamp_min(1e-12) / 127
            q = (cands / s[:, None, None]).round_().to(torch.int8)
            self.cand[rows, d], self.scale[rows, d] = q, s
            self.tok[rows, d] = tokens.to(torch.uint8)
            self.share[rows, d] = shares.half()
            running.add_(q[torch.arange(len(rows), device=q.device), tokens.long()].float() * s[:, None])
            self.cum[rows, d + 1] = running.half()
        return visit

    def inputs(self, i: Tensor, d: Tensor, window: int, cands: Tensor | None = None) -> tuple:
        """CandNet's inputs for images i [B] at depths d [B] (long): their candidates (or cands [B, V, D]), the
        window's deltas [B, window, D] (depths d - window .. d - 1; zeros before the root) and which exist, the
        early token [B, D] (the deltas before d - window, summed) and whether it exists, and d."""
        if cands is None:
            cands = self.cand[i, d].float() * self.scale[i, d][:, None, None]
        j = d[:, None] - window + torch.arange(window, device=d.device)
        valid = j >= 0
        j = j.clamp_min(0)
        ii = i[:, None].expand_as(j)
        win = self.cand[ii, j, self.tok[ii, j].long()].float() * (self.scale[ii, j] * valid)[..., None]
        early = self.cum[i, (d - window).clamp_min(0)].float()
        return cands, win, valid, early, d > window, d


class CandNet(nn.Module):
    def __init__(self, depths, V=8, D=512, window=16, width=128, layers=4, heads=4, dropout=0.0):
        super().__init__()
        self.config = dict(depths=depths, V=V, D=D, window=window, width=width, layers=layers, heads=heads,
                           dropout=dropout)
        self.V, self.window = V, window
        self.delta_in = nn.Linear(D + 2, width)  # (direction, log length, exists)
        self.position = nn.Parameter(torch.randn(window + 1, width) * 0.02)  # (the early token first)
        self.centroid_in = nn.Linear(D + 1, width)
        self.slot = nn.Parameter(torch.randn(V, width) * 0.02)
        self.depth = nn.Parameter(torch.randn(depths, width) * 0.02)
        self.blocks = nn.ModuleList(Block(width, heads, dropout) for _ in range(layers))
        self.out_norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, cands: Tensor, win: Tensor, valid: Tensor, early: Tensor, early_valid: Tensor,
                d: Tensor) -> Tensor:
        """Logits [B, V] (CandStore.inputs' tuple as arguments)."""
        x = torch.cat([early[:, None], win], 1)
        exists = torch.cat([early_valid[:, None], valid], 1).float()
        n = x.norm(dim=-1, keepdim=True).clamp_min(EPS)
        m = self.delta_in(torch.cat([x / n, n.log() * exists[..., None], exists[..., None]], -1)) + self.position
        L = cands.norm(dim=-1, keepdim=True).clamp_min(EPS)
        h = self.centroid_in(torch.cat([cands / L, L.log()], -1)) + self.slot + self.depth[d][:, None]
        for blk in self.blocks:
            m, mem = blk(m)
            h, _ = blk(h, mem)
        return self.head(self.out_norm(h))[..., 0]


class CodeQA:
    """Images for judging a prior by eye: a few test images, their reconstructions by the tree's first K depths, and
    codes sampled from a prior and from the tree's shares, each decoded by the tree (VTree.sample) and the outer
    decoder of the latents the tree coded."""

    def __init__(self, tree: VTree, n: int = 10, data_root: str = "data", seed: int = 0):
        self.tree, self.K, self.n, self.device, self.seed = tree, tree.K, n, tree.device, seed
        self.model, self.lat = load_outer(tree.args["outer"], tree.device)
        test = self.lat["test"]
        idx = torch.randperm(len(test), generator=torch.Generator().manual_seed(seed))[:n]
        Z = test[idx].float().flatten(1).to(tree.device)
        self.real = (load_images(self.lat["dataset"], "test", data_root)[idx].float() + 0.5) / 256
        g, _ = tree.code(Z, self.K)
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
    def samples(self, prior: CandNet, amp) -> dict:
        """n codes sampled from prior and n from the tree's shares alone, decoded; with the test images and their
        reconstructions (the same seed every call, so that calls compare)."""
        was = prior.training
        prior.eval()
        gen = torch.Generator(self.device).manual_seed(self.seed + 1)
        V, D = prior.V, len(self.tree.mean)
        store = CandStore(self.n, self.K, V, D, self.device)
        rows = torch.arange(self.n, device=self.device)
        put = store.writer(rows)

        def by_prior(d, cands, shares):
            dd = torch.full((self.n,), d, dtype=torch.long, device=self.device)
            with amp():
                logits = prior(*store.inputs(rows, dd, prior.window, cands.float())).float()
            token = torch.multinomial(logits.softmax(-1), 1, generator=gen)[:, 0]
            put(d, cands, token, shares)
            return token

        def by_shares(d, cands, shares):
            return torch.multinomial(shares.float().clamp_min(0) + 1e-12, 1, generator=gen)[:, 0]

        _, z_prior = self.tree.sample(self.n, by_prior, self.K)
        _, z_shares = self.tree.sample(self.n, by_shares, self.K)
        prior.train(was)
        return dict(real=self.real, recon=self.recon, prior=self.decode(z_prior), shares=self.decode(z_shares),
                    K=self.K)


__all__ = ["CandNet", "CandStore", "CodeQA"]
