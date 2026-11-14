"""A prior over a V-way tree's codes (train_inner_vway.py), and a discriminator (train_vprior.py), both reading the
node a code is at as the tree's own networks read theirs.

NodeEncoder reads a node: its current mean (the tree's mean plus every delta taken so far: what the code's image
looks like there, as one token, a linear map of the raw vector) and the last `window` deltas taken (each as its
direction, log length and whether it exists: a shallow node has fewer), with self-attention among them; every layer's
normalized tokens are kept as memory for queries to read, and the mean token's last state is the node's summary.
Task heads are query tokens that read a node's memory (Block with mem: the node's tokens and each other, in one
attention) at every layer, each query's input embedding plus a projection of the node's summary (so that what a query
means at this node is in it from the first layer).

NodePrior's queries are the tree's V candidates at the node (the cells' predicted means: the deltas the code could
take next; each's direction, log length and slot), each giving one logit: the softmax over them is
p(token | node). NodeDisc reads a node one deeper: the child a candidate leads to (the mean plus the candidate, the
window shifted by it), one learned query giving its trust logit: how likely the data's code reaches that child rather
than the prior's. Its forward takes the parent's inputs and scores all V children (as V nodes in one batch), so it
returns [B, V] as the prior does. Heads start at zero: an untrained prior is uniform, an untrained discriminator trusts
everything one half. One network of each serves every depth (a depth embedding on the mean token).

CandNet is the earlier design (the deltas before the window summed into one token instead of the mean; the
discriminator a prior-shaped network), kept for its runs' saved networks; make_net builds either from a saved config.

CandStore holds replayed codes (dvae/vreplay.py's VTree) of n images at every depth: the V candidates in 8 bits (a
scale per image and depth), the token taken, the tree's shares, and the running sum of the deltas taken. CodeQA
decodes codes sampled from a prior into images, for the run's plot.
"""

from typing import NamedTuple

import torch
import torch.nn as nn
from torch import Tensor

from dvae.data import load_images
from dvae.deltatree import EPS, Block
from dvae.node import NodeEncoder, QueryReader, directions
from dvae.outer import load_outer
from dvae.vreplay import VTree


class NodeInputs(NamedTuple):
    """A batch of B nodes (image i at depth d) as the networks read them."""
    cands: Tensor  # [B, V, D] the tree's candidates there
    win: Tensor  # [B, window, D] the last deltas taken (depths d - window .. d - 1; zeros before the root)
    valid: Tensor  # [B, window] which of them exist
    early: Tensor  # [B, D] the deltas before the window, summed (CandNet's)
    early_valid: Tensor  # [B] whether there are any
    mean: Tensor  # [B, D] the node's current mean: the tree's mean plus every delta taken
    d: Tensor  # [B] (long) the depth


class CandStore:
    def __init__(self, n: int, K: int, V: int, D: int, device: torch.device, base: Tensor | None = None):
        """base [D]: the tree's mean (every code's start)."""
        self.n, self.K, self.V, self.D = n, K, V, D
        self.base = base.float().to(device) if base is not None else torch.zeros(D, device=device)
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

    def inputs(self, i: Tensor, d: Tensor, window: int, cands: Tensor | None = None) -> NodeInputs:
        """The nodes of images i [B] at depths d [B] (long), with their stored candidates or cands [B, V, D]."""
        if cands is None:
            cands = self.cand[i, d].float() * self.scale[i, d][:, None, None]
        j = d[:, None] - window + torch.arange(window, device=d.device)
        valid = j >= 0
        j = j.clamp_min(0)
        ii = i[:, None].expand_as(j)
        win = self.cand[ii, j, self.tok[ii, j].long()].float() * (self.scale[ii, j] * valid)[..., None]
        early = self.cum[i, (d - window).clamp_min(0)].float()
        mean = self.base + self.cum[i, d].float()
        return NodeInputs(cands, win, valid, early, d > window, mean, d)


class NodePrior(nn.Module):
    def __init__(self, depths, V=8, D=512, window=16, width=128, layers=4, heads=4, dropout=0.0):
        super().__init__()
        self.config = dict(kind="node_prior", depths=depths, V=V, D=D, window=window, width=width, layers=layers,
                           heads=heads, dropout=dropout)
        self.V, self.window = V, window
        self.node = NodeEncoder(depths, D, window, width, layers, heads, dropout)
        self.reader = QueryReader(width, layers, heads, dropout)
        self.centroid_in = nn.Linear(D + 1, width)
        self.slot = nn.Parameter(torch.randn(V, width) * 0.02)
        self.head = nn.Linear(width, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: NodeInputs) -> Tensor:
        """Logits [B, V]."""
        mems, summary = self.node(x.win, x.valid, x.mean, x.d)
        h = self.centroid_in(torch.cat(directions(x.cands), -1)) + self.slot
        return self.head(self.reader(h, mems, summary))[..., 0]


class NodeDisc(nn.Module):
    def __init__(self, depths, V=8, D=512, window=16, width=128, layers=4, heads=4, dropout=0.0):
        super().__init__()
        self.config = dict(kind="node_disc", depths=depths, V=V, D=D, window=window, width=width, layers=layers,
                           heads=heads, dropout=dropout)
        self.V, self.window = V, window
        self.node = NodeEncoder(depths, D, window, width, layers, heads, dropout)
        self.reader = QueryReader(width, layers, heads, dropout)
        self.query = nn.Parameter(torch.randn(1, 1, width) * 0.02)
        self.head = nn.Linear(width, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: NodeInputs) -> Tensor:
        """Trust logits [B, V]: each candidate's child node's."""
        B, V, D = x.cands.shape
        c = x.cands.reshape(B * V, 1, D)
        win = torch.cat([x.win.repeat_interleave(V, 0)[:, 1:], c], 1)
        valid = torch.cat([x.valid.repeat_interleave(V, 0)[:, 1:], x.valid.new_ones(B * V, 1)], 1)
        mean = x.mean.repeat_interleave(V, 0) + c[:, 0]
        mems, summary = self.node(win, valid, mean, x.d.repeat_interleave(V, 0) + 1)
        return self.head(self.reader(self.query.expand(B * V, -1, -1), mems, summary))[:, 0, 0].view(B, V)


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

    def forward(self, x: NodeInputs) -> Tensor:
        """Logits [B, V]."""
        seq = torch.cat([x.early[:, None], x.win], 1)
        exists = torch.cat([x.early_valid[:, None], x.valid], 1).float()
        n = seq.norm(dim=-1, keepdim=True).clamp_min(EPS)
        m = self.delta_in(torch.cat([seq / n, n.log() * exists[..., None], exists[..., None]], -1)) + self.position
        L = x.cands.norm(dim=-1, keepdim=True).clamp_min(EPS)
        h = self.centroid_in(torch.cat([x.cands / L, L.log()], -1)) + self.slot + self.depth[x.d][:, None]
        for blk in self.blocks:
            m, mem = blk(m)
            h, _ = blk(h, mem)
        return self.head(self.out_norm(h))[..., 0]


PriorNet = NodePrior | NodeDisc | CandNet


def make_net(config: dict) -> PriorNet:
    """A network from its config (a saved one's, or a new one's with kind node_prior, node_disc or cand)."""
    config = dict(config)
    kind = config.pop("kind", "cand")
    return {"node_prior": NodePrior, "node_disc": NodeDisc, "cand": CandNet}[kind](**config)


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
    def samples(self, prior: PriorNet, amp) -> dict:
        """n codes sampled from prior and n from the tree's shares alone, decoded; with the test images and their
        reconstructions (the same seed every call, so that calls compare)."""
        was = prior.training
        prior.eval()
        gen = torch.Generator(self.device).manual_seed(self.seed + 1)
        V, D = prior.V, len(self.tree.mean)
        store = CandStore(self.n, self.K, V, D, self.device, self.tree.mean)
        rows = torch.arange(self.n, device=self.device)
        put = store.writer(rows)

        def by_prior(d, cands, shares):
            dd = torch.full((self.n,), d, dtype=torch.long, device=self.device)
            with amp():
                logits = prior(store.inputs(rows, dd, prior.window, cands.float())).float()
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


__all__ = ["CandNet", "CandStore", "CodeQA", "NodeDisc", "NodeInputs", "NodePrior", "PriorNet", "make_net"]
